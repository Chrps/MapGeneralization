import numpy as np
import torch
from torch import nn
# from torch.nn import functional as F
import dgl
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import sys
sys.path.append('..')
import src.graph_utils as graph_utils
import src.models as models


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels, features = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.LongTensor(batched_graph.number_of_nodes(), 1)
    batched_labels = torch.cat(labels, out=batched_labels)
    batched_features = torch.Tensor(batched_graph.number_of_nodes(), 1)
    batched_features = torch.cat(features, out=batched_features)
    return batched_graph, batched_labels, batched_features

def compute_class_weights(labels):
    n_doors = torch.sum(labels)
    n_other = len(labels) - n_doors
    other_weight = torch.true_divide(n_doors,len(labels))
    door_weight = 1.0-other_weight
    return torch.FloatTensor([other_weight, door_weight])

class LightningNodeClassifier(pl.LightningModule):

    def __init__(self, hparams):
      super().__init__()
      self.hparams = hparams
      network, config = models.get_model_and_config(self.hparams.network)
      config['extra_args'][0] = self.hparams.size_hidden
      config['extra_args'][1] = self.hparams.n_layers
      self.model = network(self.hparams.n_features,
                      self.hparams.n_classes,
                      *config['extra_args'])
      #self.lr = config['lr']
      self.lr = self.hparams.learning_rate
      self.weight_decay = config['weight_decay']

    def forward(self, g, f):
        return self.model(g, f)

    def prepare_data(self):
        self.data_train = graph_utils.group_labels_features(self.hparams.data_path,
                                                            self.hparams.train_list,
                                                            windowing=self.hparams.windowing)
        self.data_val = graph_utils.group_labels_features(self.hparams.data_path,
                                                          self.hparams.val_list,
                                                          windowing=self.hparams.windowing)
        self.data_test = graph_utils.group_labels_features(self.hparams.data_path,
                                                           self.hparams.test_list,
                                                           windowing=self.hparams.windowing)
    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.n_workers,
                          collate_fn=collate)
    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.n_workers,
                          collate_fn=collate)
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.n_workers,
                          collate_fn=collate)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        graphs, labels, features = batch
        output = self.forward(graphs, features)
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels))
        loss = criterion(output, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['batch_loss'] for x in outputs]).mean().detach().numpy()
        tensorboard_logs = {'train_loss': loss}
        return {'train_loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        graphs, labels, features = batch
        output = self.forward(graphs, features)
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels))
        loss = criterion(output, labels)
        if self.hparams.gpus:
            labels = labels.cpu()
            pred = output.argmax(dim=1, keepdim=True).cpu().view_as(labels)
        else:
            pred = output.argmax(dim=1, keepdim=True).view_as(labels)
        overall_acc = balanced_accuracy_score(y_true=labels, y_pred=pred)
        #
        class_accs = []
        for cl in range(self.hparams.n_classes):
            class_labels = labels[labels==cl]
            class_pred = pred[labels==cl]
            class_accs.append(accuracy_score(class_labels, class_pred))
        #batch_val_correct = pred.eq(labels).sum().item()/self.hparams.batch_size
        return {'loss': loss, 'overall_acc': overall_acc, 'non_door_acc': class_accs[0], 'door_acc': class_accs[1]}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([x['loss'] for x in outputs])).item()
        overall_acc = np.mean(np.stack([x['overall_acc'] for x in outputs]))
        non_door_acc = np.mean(np.stack([x['non_door_acc'] for x in outputs]))
        door_acc = np.mean(np.stack([x['door_acc'] for x in outputs]))
        tensorboard_logs = {'val_loss': loss,
                            'val_overall_acc': overall_acc,
                            'val_non_door_acc': non_door_acc,
                            'val_door_acc': door_acc}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        graphs, labels, features = batch
        output = self.forward(graphs, features)
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels))
        loss = criterion(output, labels)
        if self.hparams.gpus:
            labels = labels.cpu()
            pred = output.argmax(dim=1, keepdim=True).cpu().view_as(labels)
        else:
            pred = output.argmax(dim=1, keepdim=True).view_as(labels)
        overall_acc = balanced_accuracy_score(y_true=labels, y_pred=pred)
        class_accs = []
        for cl in range(self.hparams.n_classes):
            class_labels = labels[labels==cl]
            class_pred = pred[labels==cl]
             # TODO with bathc size 5 either class_labels, class_pred results in nan because
            class_accs.append(accuracy_score(class_labels, class_pred))
        return {'loss': loss, 'overall_acc': overall_acc, 'non_door_acc': class_accs[0], 'door_acc': class_accs[1]}

    def test_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([x['loss'] for x in outputs])).item()
        overall_acc = np.mean(np.stack([x['overall_acc'] for x in outputs]))
        non_door_acc = np.mean(np.stack([x['non_door_acc'] for x in outputs]))
        door_acc = np.mean(np.stack([x['door_acc'] for x in outputs]))
        tensorboard_logs = {'test_loss': loss,
                            'test_overall_acc': overall_acc,
                            'test_non_door_acc': non_door_acc,
                            'test_door_acc': door_acc}
        return {'test_loss': loss, 'log': tensorboard_logs}


        '''
        epoch_test_correct = np.stack([x['test_correct'] for x in outputs]).mean()

        ground_truths = []
        predictions = []
        for batch in outputs:
            for t,p in zip(batch['labels'],batch['pred']):
                ground_truths.append(t.item())
                predictions.append(p.item())

        cm = confusion_matrix(ground_truths, predictions)
        tp_and_fn = cm.sum(1)
        tp_and_fp = cm.sum(0)
        tp = cm.diagonal()

        precision = tp / tp_and_fp
        recall = tp / tp_and_fn

        print("precision {}".format(precision))
        print("recall {}".format(recall))

        tensorboard_logs = {'epoch_test_correct': epoch_test_correct,
                            'avg precision': precision.mean(),
                            'avg recall': recall.mean()}
        return {'log': tensorboard_logs}
        '''
