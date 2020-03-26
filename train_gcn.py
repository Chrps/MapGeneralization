import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import time
import numpy as np
import os
import torch

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(4, 16, F.relu)
        self.gcn2 = GCN(16, 2, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def normalize_positions(positions):
    norm_pos = []
    min_x = min(positions)[0]
    min_y = min(positions)[1]
    max_x = max(positions)[0]
    max_y = max(positions)[1]
    if min_x < 0:
        max_x = max_x - min_x
    if min_y < 0:
        max_y = max_y - min_y
    for position in positions:
        temp_pos = []
        x_pos = 0
        y_pos = 0
        if min_x < 0:
            x_pos = position[0] - min_x
        if min_y < 0:
            y_pos = position[1] - min_y
        x_pos = position[0] / max_x
        y_pos = position[1] / max_y
        temp_pos = tuple([x_pos, y_pos])
        norm_pos.append(temp_pos)
    return norm_pos

def batch_graphs(data_list, folder):
    data_path = 'data/'
    data_files = [os.path.join(data_path, folder, line.rstrip()) for line in open(data_list)]

    all_labels = []
    all_norm_pos = []
    all_norm_deg = []
    all_norm_identity = []
    list_of_graphs = []

    for file in data_files:
        # Read each file as networkx graph and retrieve positions + labels
        nxg = nx.read_gpickle(file)
        positions = nx.get_node_attributes(nxg, 'pos')
        positions = list(positions.values())
        label_dict = nx.get_node_attributes(nxg, 'label')
        labels = list(label_dict.values())
        all_labels.extend(labels)

        # Define DGL graph from netx graph
        g = DGLGraph()
        g.from_networkx(nxg)
        g.readonly()
        list_of_graphs.append(g)  # Append graph to list to be batched

        # % Define some features for the graph
        # Normalized positions
        all_norm_pos.extend(normalize_positions(positions))

        # Normalized node degree (number of edges connected to node)
        norm_deg = 1. / g.in_degrees().float().unsqueeze(1)
        all_norm_deg.append(norm_deg)

        # Normalized unique identity for each node
        identity = np.linspace(0, 1, num=nxg.number_of_nodes())
        identity = np.reshape(identity, (nxg.number_of_nodes(), 1))
        all_norm_identity.extend(identity)

    # Batch the graphs
    batched_graph = dgl.batch(list_of_graphs)

    # Convert everything to tensor to be used by pytorch/DGL
    all_labels = torch.LongTensor(all_labels)
    all_norm_pos = torch.FloatTensor(all_norm_pos)
    all_norm_identity = torch.FloatTensor(all_norm_identity)

    # Normalized node degrees is a list of tensors so concatenate into one tensor
    conc_all_norm_deg = torch.Tensor(batched_graph.number_of_nodes(), 1)
    torch.cat(all_norm_deg, out=conc_all_norm_deg)

    # Define the features a one large tensor
    features = torch.cat((conc_all_norm_deg, all_norm_pos, all_norm_identity), 1)

    return batched_graph, all_labels, features


def main():
    CHKPT_PATH = 'checkpoint/model_gcn.pth'
    NUM_EPOCHS = 1000

    # Load your training data in the form of a batched graph (essentially a giant graph)
    g, labels, features = batch_graphs('data/train_file_list.txt', 'graph_annotations')
    train_mask = torch.BoolTensor(np.ones(g.number_of_nodes())) # Mask tells which nodes are used for training (so all)

    # Print how many door vs non-door instances there are
    non_door_instances = 0
    door_instances = 0
    for label in labels:
        if label == 0:
            non_door_instances += 1
        if label == 1:
            door_instances += 1
    print("Non Door Instances: ", non_door_instances)
    print("Door Instances: ", door_instances)

    # Create your network/model
    net = Net()
    print(net)

    # Define loss weights for each class (door vs non-door instances, gets printed in beginning of run)
    weights = [0.075, 1.0]
    weights = torch.FloatTensor(weights)

    # Define optimizer with learning rate
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)

    # Train for specified epochs
    for epoch in range(NUM_EPOCHS):

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask], weight=weights)

        print("Epoch[{}]: loss {}".format(epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO
        # Incorporate some test accuracy here because loss is a bit user defined from weights (not 100% reliable)
        #acc = evaluate(net, g, features, labels, test_mask)
        #print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            #epoch, loss.item(), acc, np.mean(dur)))

    # Save model when done training
    torch.save(net.state_dict(), CHKPT_PATH)

if __name__ == '__main__':
    main()