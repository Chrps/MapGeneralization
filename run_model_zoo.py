import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph, batch
from dgl.data import register_data_args, load_data
from models import *
from configs import *
import networkx as nx
import matplotlib.pyplot as plt
import os

CHKPT_PATH = 'checkpoint/model_gcn_mod2.pth'
TEST_PATH = 'data/graphs/MSP1-HoM-MA-XX+5-ET_cloudconvert_sub.gpickle'


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
    all_norm_ids = []

    for file in data_files:
        # Read each file as networkx graph and retrieve positions + labels
        nxg = nx.read_gpickle(file)

        ids = nx.get_node_attributes(nxg, 'id')
        ids = list(ids.values())
        for idx, id in enumerate(ids):
            if id == 0:
                ids[idx] = 0
            else:
                ids[idx] = 1. / id
        all_norm_ids.extend(ids)

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
    batched_graph = batch(list_of_graphs)

    # Convert everything to tensor to be used by pytorch/DGL
    all_labels = torch.LongTensor(all_labels)
    #all_norm_pos = torch.FloatTensor(all_norm_pos)
    all_norm_pos1 = torch.FloatTensor(list(zip(*all_norm_pos))[0])
    all_norm_pos2 = torch.FloatTensor(list(zip(*all_norm_pos))[1])
    all_norm_pos1 = all_norm_pos1.view(all_norm_pos1.size()[0], 1)
    all_norm_pos2 = all_norm_pos2.view(all_norm_pos2.size()[0], 1)
    all_norm_identity = torch.FloatTensor(all_norm_identity)
    all_norm_ids = torch.FloatTensor(all_norm_ids)
    all_norm_ids = all_norm_ids.view(all_norm_ids.size()[0], 1)
    all_norm_identity = all_norm_identity.view(all_norm_identity.size()[0],1)

    # Normalized node degrees is a list of tensors so concatenate into one tensor
    conc_all_norm_deg = torch.Tensor(batched_graph.number_of_nodes(), 1)
    torch.cat(all_norm_deg, out=conc_all_norm_deg)
    conc_all_norm_deg = conc_all_norm_deg.view(conc_all_norm_deg.size()[0],1)
    print(all_norm_ids.size())
    print(all_norm_pos1.size())
    print(all_norm_pos2.size())
    print(conc_all_norm_deg.size())
    print(all_norm_identity.size())
    # Define the features a one large tensor
    features = torch.cat((conc_all_norm_deg, all_norm_pos1, all_norm_pos2, all_norm_identity, all_norm_ids), 1)

    return batched_graph, all_labels, features


def get_features(graph, positions, ids):
    # % Define some features for the graph

    for idx, id in enumerate(ids):
        if id == 0:
            ids[idx] = 0
        else:
            ids[idx] = 1. / id

    # Normalized positions
    norm_pos = normalize_positions(positions)

    # Normalized node degree (number of edges connected to node)
    norm_deg = 1. / graph.in_degrees().float().unsqueeze(1)

    # Normalized unique identity for each node
    norm_identity = np.linspace(0, 1, num=graph.number_of_nodes())
    norm_identity = np.reshape(norm_identity, (graph.number_of_nodes(), 1))

    # Convert everything to tensor to be used by pytorch/DGL
    norm_pos1 = torch.FloatTensor(list(zip(*norm_pos))[0])
    norm_pos2 = torch.FloatTensor(list(zip(*norm_pos))[1])
    norm_identity = torch.FloatTensor(norm_identity)
    norm_ids = torch.FloatTensor(ids)
    norm_ids = norm_ids.view(norm_ids.size()[0], 1)
    norm_identity = norm_identity.view(norm_identity.size()[0], 1)
    norm_pos1 = norm_pos1.view(norm_pos1.size()[0], 1)
    norm_pos2 = norm_pos2.view(norm_pos2.size()[0], 1)
    norm_deg = norm_deg.view(norm_deg.size()[0], 1)

    # Define the features as one large tensor
    features = torch.cat((norm_deg, norm_pos1, norm_pos2, norm_identity, norm_ids), 1)

    return features


def get_model_and_config(name):
    name = name.lower()
    if name == 'gcn':
        return GCN, GCN_CONFIG
    elif name == 'gcn_mod':
        return GCN_MOD, GCN_CONFIG
    elif name == 'gat':
        return GAT, GAT_CONFIG
    elif name == 'graphsage':
        return GraphSAGE, GRAPHSAGE_CONFIG
    elif name == 'appnp':
        return APPNP, APPNP_CONFIG
    elif name == 'tagcn':
        return TAGCN, TAGCN_CONFIG
    elif name == 'agnn':
        return AGNN, AGNN_CONFIG
    elif name == 'sgc':
        return SGC, SGC_CONFIG
    elif name == 'gin':
        return GIN, GIN_CONFIG
    elif name == 'chebnet':
        return ChebNet, CHEBNET_CONFIG


def draw(results, ax, nx_G, positions):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'

    colors = []
    for v in range(len(nx_G)):
        cls = results[v]
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Results')
    nx.draw_networkx(nx_G.to_undirected(), positions, node_color=colors,
            with_labels=False, node_size=5, ax=ax)


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

        return correct.item() * 1.0 / len(labels), correct0.item() * 1.0 / len(labels0), correct1.item() * 1.0 / len(labels1)


def train(net, gpu, n_epochs, n_classes, n_features, self_loop):
    # load and preprocess dataset
    g, labels, features = batch_graphs('data/train_file_list.txt', 'graph_annotations')
    train_mask = torch.BoolTensor(np.ones(g.number_of_nodes()))
    val_mask = torch.BoolTensor(np.ones(g.number_of_nodes()))
    test_mask = torch.BoolTensor(np.ones(g.number_of_nodes()))
    '''data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)'''
    #in_feats = features.shape[1]
    #n_classes = data.num_labels
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    if gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    #g = data.graph
    # add self loop
    if self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    GNN, config = get_model_and_config(net)
    model = GNN(n_features,
                n_classes,
                *config['extra_args'])

    if cuda:
        model.cuda()

    print(model)

    weights = [0.075, 0.925]
    weights = torch.FloatTensor(weights)
    loss_fcn = torch.nn.CrossEntropyLoss(weight=weights)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # initialize graph
    dur = []
    losses = []
    acc_list = []
    acc0_list = []
    acc1_list = []
    for epoch in range(n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc, acc0, acc1 = evaluate(model, g, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "Accuracy Class0 {:.4f} | Accuracy Class1 {:.4f} |"
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, acc0, acc1, n_edges / np.mean(dur) / 1000))
        losses.append(loss.item())
        acc_list.append(acc)
        acc0_list.append(acc0)
        acc1_list.append(acc1)
        plt.axis([0, n_epochs, 0, 1])
        plt.plot(losses, 'b', label="loss")
        plt.plot(acc_list, 'r', label="acc all")
        plt.plot(acc0_list, 'g', label="acc class 0")
        plt.plot(acc1_list, color='orange', label="acc class 1")
        plt.legend()
        plt.show(block=False)
        plt.pause(0.0001)
        plt.clf()

    print()
    acc, acc0, acc1 = evaluate(model, g, features, labels, test_mask)
    print("Test Accuracy: All: {:.4f} | Class 0: {:.4f}| Class 1: {:.4f}".format(acc, acc0, acc1))
    torch.save(model.state_dict(), CHKPT_PATH)


def inference(net, n_classes, n_features, VISUALIZE=True):
    GNN, config = get_model_and_config(net)
    model = GNN(n_features,
                n_classes,
                *config['extra_args'])
    model.load_state_dict(torch.load(CHKPT_PATH))
    model.eval()

    # Load Test data
    nxg = nx.read_gpickle(TEST_PATH)
    positions = nx.get_node_attributes(nxg, 'pos')
    positions = list(positions.values())

    ids = nx.get_node_attributes(nxg, 'id')
    ids = list(ids.values())

    # Define DGL graph from netx graph
    g = DGLGraph()
    g.from_networkx(nxg)
    g.readonly()

    test_mask = torch.BoolTensor(np.ones(g.number_of_nodes()))

    # Get the features
    features = get_features(g, positions, ids)

    # Get labels
    label_dict = nx.get_node_attributes(nxg, 'label')
    labels = list(label_dict.values())
    labels = torch.LongTensor(labels)

    with th.no_grad():
        logits = model(g, features)
        _, predictions = th.max(logits, dim=1)
        predictions = predictions.numpy()

    acc, acc0, acc1 = evaluate(model, g, features, labels, test_mask)
    print("Test Accuracy: All: {:.4f} | Class 0: {:.4f}| Class 1: {:.4f}".format(acc, acc0, acc1))

    if VISUALIZE:
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        draw(predictions, ax, nxg, positions)  # draw the results
        plt.show()


if __name__ == '__main__':

    net = 'gcn_mod'  # available models are gcn, gat, graphsage, gin, appnp, tagcn, sgc, agnn
    gpu = -1  # gpu device, -1 = no gpu
    n_epochs = 350
    n_classes = 2
    n_features = 5
    self_loop = False  # Self-loop creates an edge from each node to itself
    #train(net, gpu, n_epochs, n_classes, n_features, self_loop)

    inference(net, n_classes, n_features)
