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
from sliding_window import SlidingWindow
sliding_window = SlidingWindow(window_x_size=4, window_y_size=4)
CHKPT_PATH = 'checkpoint/model_gcn_mod2.pth'
TEST_PATH = 'data/graph_annotations/public_toilet_w_annotations.gpickle'

def batch_graphs(data_list, folder):
    data_path = 'data/'
    data_files = [os.path.join(data_path, folder, line.rstrip()) for line in open(data_list)]

    all_labels = []
    all_norm_pos = []
    all_norm_deg = []
    all_norm_identity = []
    list_of_graphs = []
    all_norm_angles = []

    for file in data_files:
        # Read each file as networkx graph and retrieve positions + labels
        nxg = nx.read_gpickle(file)
        nxgs = sliding_window.perform_windowing(nxg)
        for nxg in nxgs:
            positions = nx.get_node_attributes(nxg, 'pos')
            positions = list(positions.values())
            angles = nx.get_node_attributes(nxg, 'node_angle')
            angles = list(angles.values())
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
            #all_norm_pos.extend(normalize_positions(positions))
            all_norm_pos.extend(positions)

            angles = np.reshape(angles, (nxg.number_of_nodes(), 1))
            all_norm_angles.extend(angles)

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
    all_norm_pos = torch.FloatTensor(all_norm_pos)
    all_norm_angles = torch.FloatTensor(all_norm_angles)
    all_norm_identity = torch.FloatTensor(all_norm_identity)

    # Normalized node degrees is a list of tensors so concatenate into one tensor
    conc_all_norm_deg = torch.Tensor(batched_graph.number_of_nodes(), 1)
    torch.cat(all_norm_deg, out=conc_all_norm_deg)

    # Define the features a one large tensor
    features = torch.cat((conc_all_norm_deg, all_norm_pos, all_norm_identity, all_norm_angles), 1)

    return batched_graph, all_labels, features

def get_features(graph, positions, angles):
    # % Define some features for the graph
    # Normalized positions
    #norm_pos = normalize_positions(positions)
    norm_pos = positions

    angles = np.reshape(angles, (graph.number_of_nodes(), 1))
    angles = torch.FloatTensor(angles)

    # Normalized node degree (number of edges connected to node)
    norm_deg = 1. / graph.in_degrees().float().unsqueeze(1)

    # Normalized unique identity for each node
    norm_identity = np.linspace(0, 1, num=graph.number_of_nodes())
    norm_identity = np.reshape(norm_identity, (graph.number_of_nodes(), 1))

    # Convert everything to tensor to be used by pytorch/DGL
    norm_pos = torch.FloatTensor(norm_pos)
    norm_identity = torch.FloatTensor(norm_identity)

    # Define the features as one large tensor
    features = torch.cat((norm_deg, norm_pos, norm_identity, angles), 1)

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
            with_labels=False, node_size=25, ax=ax)

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(model, gpu, n_epochs, n_classes, n_features, self_loop):
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
    in_feats = n_features
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
    GNN, config = get_model_and_config(model)
    model = GNN(5,
                2,
                *config['extra_args'])

    if cuda:
        model.cuda()

    print(model)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # initialize graph
    dur = []
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

        acc = evaluate(model, g, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, g, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    torch.save(model.state_dict(), CHKPT_PATH)
    #torch.save(model, CHKPT_PATH)


def inference(model_new, VISUALIZE=True):
    GNN, config = get_model_and_config(model_new)
    model_new = GNN(5,
                2,
                *config['extra_args'])
    model_new.load_state_dict(torch.load(CHKPT_PATH))
    model_new.eval()
    for param in model_new.parameters():
        print(param.data)

    # Load Test data
    nxg = nx.read_gpickle(TEST_PATH)
    org_positions = nx.get_node_attributes(nxg, 'pos')
    org_positions = list(org_positions.values())
    nxgs = sliding_window.perform_windowing(nxg)
    predictions_list = [[] for i in range(nxg.number_of_nodes())]
    for nxg_sub in nxgs:
        positions = nx.get_node_attributes(nxg_sub, 'pos')
        positions = list(positions.values())
        angles = nx.get_node_attributes(nxg_sub, 'node_angle')
        angles = list(angles.values())
        # Define DGL graph from netx graph
        g = DGLGraph()
        g.from_networkx(nxg_sub)
        g.readonly()

        # Get the features
        features = get_features(g, positions, angles)

        with th.no_grad():
            logits = model_new(g, features)
            _, predictions = th.max(logits, dim=1)
            predictions = predictions.numpy()
        for node in nxg_sub.nodes:
            predictions_list[int(nxg_sub.nodes[node]['org_node_name'])].append(predictions[node])

    all_preditions = []
    for prediction in predictions_list:
        if (sum(prediction) / len(prediction)) >= 0.75:
            all_preditions.append(1)
        else:
            all_preditions.append(0)
    all_predictions = np.array(all_preditions)

    if VISUALIZE:
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        draw(all_predictions, ax, nxg, org_positions)  # draw the results
        plt.show()


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser(description='Node classification on citation networks.')
    register_data_args(parser)
    parser.add_argument("--model", type=str, default='gcn',
                        help='model to use, available models are gcn, gat, graphsage, gin,'
                             'appnp, tagcn, sgc, agnn')
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    args = parser.parse_args()
    print(args)
    main(args)'''

    model = 'gcn_mod'  # available models are gcn, gat, graphsage, gin, appnp, tagcn, sgc, agnn
    gpu = -1  # gpu device, -1 = no gpu
    n_epochs = 50
    n_classes = 2
    n_features = 5
    self_loop = False  # Self-loop creates an edge from each node to itself
    #train(model, gpu, n_epochs, n_classes, n_features, self_loop)

    inference(model)
