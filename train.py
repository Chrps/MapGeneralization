import numpy as np
import torch
import time
import src.graph_utils as graph_utils
import src.models as models
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import DataLoader
import dgl
import datetime
import time
from sklearn.metrics import balanced_accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--desired_net', type=str, default='gat')  # gcn, tagcn, graphsage, appnp, agnn, gin, gat, chebnet
parser.add_argument('--num-epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--train-path', type=str, default='data/train_file_list_Canterbury_and_AU.txt')
parser.add_argument('--valid-path', type=str, default='data/valid_file_list_Canterbury_and_AU.txt')
parser.add_argument('--num-classes', type=int, default=2)
parser.add_argument('--model_name', type=str, default='test_model')
parser.add_argument('--windowing', type=str, default=False)
args = parser.parse_args()


def evaluate(model, g, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits
        labels = labels
        _, indices = torch.max(logits, dim=1)
        #correct = torch.sum(indices == labels)

        labels0_idx = np.where(labels.numpy() == 0)[0]
        labels1_idx = np.where(labels.numpy() == 1)[0]
        indices0 = torch.LongTensor(np.take(indices.numpy(), labels0_idx))
        indices1 = torch.LongTensor(np.take(indices.numpy(), labels1_idx))
        labels0 = torch.LongTensor(np.take(labels.numpy(), labels0_idx))
        labels1 = torch.LongTensor(np.take(labels.numpy(), labels1_idx))
        # For class 0 and class 1
        correct0 = torch.sum(indices0 == labels0)
        correct1 = torch.sum(indices1 == labels1)

        #correct.item() * 1.0 / len(labels)
        class0_acc = correct0.item() * 1.0 / len(labels0)
        class1_acc = correct1.item() * 1.0 / len(labels1)

        pred = indices.numpy()
        labels = labels.numpy()
        #f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
        overall_acc = balanced_accuracy_score(y_true=labels, y_pred=pred)

        return overall_acc, class0_acc, class1_acc

def moving_average(a, n=10) :
    a_padded = np.pad(a, (n // 2, n - 1 - n // 2), mode='edge')
    a_smooth = np.convolve(a_padded, np.ones((n,)) / n, mode='valid')
    return a_smooth

def plot_loss_and_acc(n_epochs, epoch_list, losses, overall_acc_list, acc0_list, acc1_list, model_name):
    plt.axis([0, n_epochs, 0, 1])
    plt.plot(losses, 'b', alpha=0.3)
    plt.plot(epoch_list, overall_acc_list, 'r', label="Overall Accuracy")
    #plt.plot(epoch_list, acc0_list, 'g', alpha=0.3)
    #plt.plot(epoch_list, acc1_list, color='orange', alpha=0.3)

    avg_losses = list(moving_average(np.array(losses), n=20))
    #avg_overall_acc_list = list(moving_average(np.array(overall_acc_list)))
    #avg_acc0_list = list(moving_average(np.array(acc0_list)))
    #avg_acc1_list = list(moving_average(np.array(acc1_list)))

    #plt.plot(epoch_list, avg_overall_acc_list, 'r', label="Overall Accuracy")
    plt.plot(avg_losses, 'b', label="Loss")
    #plt.plot(epoch_list, avg_acc0_list, 'g', label="acc Non-Door")
    #plt.plot(epoch_list, avg_acc1_list, color='orange', label="acc Door")
    plt.xlabel('Epochs')

    plt.legend()
    plt.show(block=False)
    plt.pause(0.0001)
    if epoch_list[-1] % 100 == 0 or epoch_list[-1] == n_epochs:
        figure_name = model_name + '_accuracy_loss.png'
        # Where to save image
        directory = 'models/' + model_name + '/'
        plt.savefig(directory + figure_name)
    plt.clf()


def update_weights(labels):
    labels0_idx = np.where(labels.numpy() == 0)[0]
    labels1_idx = np.where(labels.numpy() == 1)[0]

    door_instances = float(len(labels1_idx))
    non_door_instances = float(len(labels0_idx))
    non_door_weight = float("{:.4f}".format(non_door_instances/(door_instances+non_door_instances)))  # specifying to 4 decimal places
    door_weight = 1.0-non_door_weight
    weights = [door_weight, non_door_weight]
    weights = torch.FloatTensor(weights)
    return weights


def save_model(model, model_name, epoch, desired_net, n_features, num_classes, start_time):
    # Saved the model
    model_dir = 'models/' + model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + '/' + model_name + '.pth'
    torch.save(model.state_dict(), model_path)
    # %% Log Total Time
    end_time = time.time()
    total_time = end_time - start_time
    total_time = float(total_time)
    day = total_time // (24 * 3600)
    total_time = total_time % (24 * 3600)
    hour = total_time // 3600
    total_time %= 3600
    minutes = total_time // 60
    total_time %= 60
    seconds = total_time
    # Save .txt file with model cfgs to load for predictions
    with open('models/' + model_name + '/' + model_name + '.txt', "w+") as model_txt:
        model_txt.write(desired_net + '\n')
        model_txt.write(str(n_features) + '\n')
        model_txt.write(str(num_classes))
    with open('models/' + model_name + '/meta.txt', "w+") as model_txt:
        model_txt.write(str(datetime.datetime.now()) + '\n')
        model_txt.write('Total Training Time (d:h:m:s):  %d:%d:%d:%d' % (day, hour, minutes, seconds) + '\n')
        model_txt.write('Model Type: ' + desired_net + '\n')
        model_txt.write('Number of Features: ' + str(n_features) + '\n')
        model_txt.write('Number of Classes: ' + str(num_classes) + '\n')
        model_txt.write('Saved at Epoch: ' + str(epoch) + '\n')


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


def train(desired_net, num_epochs, train_path, valid_path, num_classes, model_name, windowing, batch_size):

    # Retrieve dataset and prepare it for DataLoader
    trainset = graph_utils.group_graphs_labels_features(train_path, r"D:\University Stuff\OneDrive - Aalborg Universitet\P10 - Master's Thesis\data\graph_annotations", windowing=windowing)
    data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate)

    # Load the validation data
    valid_g, valid_labels, valid_features = graph_utils.batch_graphs(valid_path, r"D:\University Stuff\OneDrive - Aalborg Universitet\P10 - Master's Thesis\data\graph_annotations", windowing=windowing)

    # create user specified model
    n_features = trainset[0][2].shape[1]  # number of features is same throughout, so just get shape of first graph
    net, config = models.get_model_and_config(desired_net)
    model = net(n_features,
                num_classes,
                *config['extra_args'])
    print(n_features)

    print(model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # initialize graph
    dur = []
    losses = []
    overall_acc_list = []
    non_door_acc_list = []
    door_acc_list = []
    weights_list = []
    epoch_list = []
    best_acc_score = 0.0

    print('\n --- BEGIN TRAINING ---')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        for iter, (bg, labels, features) in enumerate(data_loader):
            # forward
            if epoch == 0:
                weights = update_weights(labels)
                weights_list.append(weights)
            logits = model(bg, features)
            # We need to update weights according to door and non-door instances in current batch
            loss_fcn = torch.nn.CrossEntropyLoss(weight=weights_list[iter])
            loss = loss_fcn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        # Evaluate after every 5th epoch
        if epoch % 5 == 0 or epoch == num_epochs:
            # Get Time
            dur.append(time.time() - t0)

            # Evaluate model
            overall_acc, non_door_acc, door_acc = evaluate(model, valid_g, valid_features, valid_labels)

            if epoch > 30 and best_acc_score < overall_acc:
                best_acc_score = overall_acc
                save_model(model, model_name, epoch, desired_net, n_features, num_classes, start_time)
            print("Epoch {:05d} | Loss {:.4f} | Door Acc {:.4f} | Non-Door Acc {:.4f} | Overall Acc {:.4f} |"
                  "Time(s) {:.4f}".format(epoch, loss.item(), door_acc, non_door_acc, overall_acc, np.mean(dur)))
            overall_acc_list.append(overall_acc)
            non_door_acc_list.append(non_door_acc)
            door_acc_list.append(door_acc)
            epoch_list.append(epoch)
            plot_loss_and_acc(num_epochs, epoch_list, losses, overall_acc_list, non_door_acc_list, door_acc_list, model_name)



if __name__ == '__main__':
    desired_net = args.desired_net
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    train_path = args.train_path
    valid_path = args.valid_path
    num_classes = args.num_classes
    model_name = args.model_name
    windowing = args.windowing

    train(desired_net, num_epochs, train_path, valid_path, num_classes, model_name, windowing, batch_size)
