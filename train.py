import numpy as np
import torch
import time
import src.graph_utils as graph_utils
import src.models as models
from src.configs import *
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--desired_net', type=str, default='tagcn') # available models are gcn, gat, graphsage, gin, appnp, tagcn, sgc, agnn
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--train-path', type=str, default='data/train_file_list.txt')
parser.add_argument('--valid-path', type=str, default='data/valid_file_list.txt')
parser.add_argument('--num-classes', type=int, default=2)
parser.add_argument('--model_name', type=str, default='test_model')
args = parser.parse_args()


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

        labels0_idx = np.where(labels.numpy() == 0)[0]
        labels1_idx = np.where(labels.numpy() == 1)[0]
        indices0 = torch.LongTensor(np.take(indices.numpy(), labels0_idx))
        indices1 = torch.LongTensor(np.take(indices.numpy(), labels1_idx))
        labels0 = torch.LongTensor(np.take(labels.numpy(), labels0_idx))
        labels1 = torch.LongTensor(np.take(labels.numpy(), labels1_idx))
        correct0 = torch.sum(indices0 == labels0)
        correct1 = torch.sum(indices1 == labels1)

        return correct.item() * 1.0 / len(labels), correct0.item() * 1.0 / len(
            labels0), correct1.item() * 1.0 / len(labels1)


def plot_loss_and_acc(n_epochs, losses, acc_list, acc0_list, acc1_list):
    plt.axis([0, n_epochs, 0, 1])
    plt.plot(losses, 'b', label="loss")
    plt.plot(acc_list, 'r', label="acc all")
    plt.plot(acc0_list, 'g', label="acc Non-Door")
    plt.plot(acc1_list, color='orange', label="acc Door")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.0001)
    plt.clf()


def train(desired_net, num_epochs, train_path, valid_path, num_classes, model_name):
    # Load your training data in the form of a batched graph (essentially a giant graph)
    train_g, train_labels, train_features = graph_utils.batch_graphs(train_path, 'graph_annotations')
    train_mask = torch.BoolTensor(np.ones(train_g.number_of_nodes()))  # Mask tells which nodes are used for training (so all)
    valid_g, valid_labels, valid_features = graph_utils.batch_graphs(valid_path, 'graph_annotations')
    valid_mask = torch.BoolTensor(np.ones(valid_g.number_of_nodes()))

    # Print how many door vs non-door instances there are
    non_door_instances = 0
    door_instances = 0
    for label in train_labels:
        if label == 0:
            non_door_instances += 1
        if label == 1:
            door_instances += 1
    print("Non Door Instances: ", non_door_instances)
    print("Door Instances: ", door_instances)

    # create user specified model
    n_features = (train_features.size())[1]
    net, config = models.get_model_and_config(desired_net)
    model = net(n_features,
                num_classes,
                *config['extra_args'])
    print(model)

    # Define weights, loss and optimizer
    weights = [0.05, 0.95]
    weights = torch.FloatTensor(weights)
    loss_fcn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # initialize graph
    dur = []
    losses = []
    overall_acc_list = []
    non_door_acc_list = []
    door_acc_list = []

    print('\n --- BEGIN TRAINING ---')
    for epoch in range(num_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(train_g, train_features)
        loss = loss_fcn(logits[train_mask], train_labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        overall_acc, non_door_acc, door_acc = evaluate(model, valid_g, valid_features, valid_labels, valid_mask)
        print("Epoch {:05d} | Loss {:.4f} | Door Acc {:.4f} | Non-Door Acc {:.4f} | Total Acc {:.4f} |" 
              "Time(s) {:.4f}".format(epoch, loss.item(), door_acc, non_door_acc, overall_acc, np.mean(dur)))

        # Plot loss and accuracies
        losses.append(loss.item())
        overall_acc_list.append(overall_acc)
        non_door_acc_list.append(non_door_acc)
        door_acc_list.append(door_acc)
        plot_loss_and_acc(num_epochs, losses, overall_acc_list, non_door_acc_list, door_acc_list)

    # Saved the model
    model_dir = 'models/' + model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + '/' + model_name + '.pth'
    torch.save(model.state_dict(), model_path)
    # Save .txt file with model cfgs to load for predictions
    with open('models/' + model_name + '/' + model_name + '.txt', "w+") as model_txt:
        model_txt.write(desired_net + '\n')
        model_txt.write(str(n_features) + '\n')
        model_txt.write(str(num_classes))



if __name__ == '__main__':
    desired_net = args.desired_net
    num_epochs = args.num_epochs
    train_path = args.train_path
    valid_path = args.valid_path
    num_classes = args.num_classes
    model_name = args.model_name

    train(desired_net, num_epochs, train_path, valid_path, num_classes, model_name)