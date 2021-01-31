import torch
import numpy as np
import argparse
from sklearn.metrics import balanced_accuracy_score
import src.graph_utils as graph_utils
import src.models as models
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data/Public')
parser.add_argument('--evaluate-path', type=str, default='test_list.txt')
parser.add_argument('--model-path', type=str, default='gat_20-09-29_15-55-58')
args = parser.parse_args()


def load_model_txt(model_path_):
    model_txt = 'trained_models/' + model_path_ + '/predict_info.txt'
    data = [line.rstrip() for line in open(model_txt)]

    # network train on ()
    net = data[0]

    # Number of features per node
    n_features = int(data[1])

    # Number of classes
    n_classes = int(data[2])

    return net, n_features, n_classes


def evaluate(model, graphs, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graphs, features)
        logits = logits
        _, indices = torch.max(logits, dim=1)

        labels0_idx = np.where(labels.numpy() == 0)[0]
        labels1_idx = np.where(labels.numpy() == 1)[0]
        indices0 = torch.LongTensor(np.take(indices.numpy(), labels0_idx))
        indices1 = torch.LongTensor(np.take(indices.numpy(), labels1_idx))
        labels0 = torch.LongTensor(np.take(labels.numpy(), labels0_idx))
        labels1 = torch.LongTensor(np.take(labels.numpy(), labels1_idx))
        # For class 0 and class 1
        correct0 = torch.sum(indices0 == labels0)
        correct1 = torch.sum(indices1 == labels1)

        class0_acc = correct0.item() * 1.0 / len(labels0)
        class1_acc = correct1.item() * 1.0 / len(labels1)

        pred = indices.numpy()
        labels = labels.numpy()
        overall_acc = balanced_accuracy_score(y_true=labels, y_pred=pred)

        return overall_acc, class0_acc, class1_acc


if __name__ == '__main__':
    data_path = args.data_path
    evaluate_path = args.evaluate_path
    model_path = args.model_path

    net, n_features, n_classes = load_model_txt(model_path)
    trained_net, config = models.get_model_and_config(net)
    model = trained_net(n_features,
                        n_classes,
                        *config['extra_args'])
    model_state = 'trained_models/' + model_path + '/model.pth'
    model.load_state_dict(torch.load(model_state))

    print(evaluate_path)
    graphs, labels, features = graph_utils.batch_graphs(data_path, evaluate_path)
    overall_acc, class0_acc, class1_acc = evaluate(model, graphs, features, labels)
    print('Overall Accuracy: %.4f' % overall_acc)
    print('Door Accuracy: %.4f' % class1_acc)
    print('Non-door Accuracy: %.4f' % class0_acc)
