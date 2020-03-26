import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def get_features(graph, positions):
    # % Define some features for the graph
    # Normalized positions
    norm_pos = normalize_positions(positions)

    # Normalized node degree (number of edges connected to node)
    norm_deg = 1. / graph.in_degrees().float().unsqueeze(1)

    # Normalized unique identity for each node
    norm_identity = np.linspace(0, 1, num=graph.number_of_nodes())
    norm_identity = np.reshape(norm_identity, (graph.number_of_nodes(), 1))

    # Convert everything to tensor to be used by pytorch/DGL
    norm_pos = torch.FloatTensor(norm_pos)
    norm_identity = torch.FloatTensor(norm_identity)

    # Define the features as one large tensor
    features = torch.cat((norm_deg, norm_pos, norm_identity), 1)

    return features


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


def main():
    TEST_PATH = 'data/graph_annotations/xray_room_w_annotations.gpickle'
    CHKPT_PATH = 'checkpoint/model_gcn.pth'
    VISUALIZE = True

    # Load Test data
    nxg = nx.read_gpickle(TEST_PATH)
    positions = nx.get_node_attributes(nxg, 'pos')
    positions = list(positions.values())
    # Define DGL graph from netx graph
    g = DGLGraph()
    g.from_networkx(nxg)
    g.readonly()

    # Get the features
    features = get_features(g, positions)

    # Load the model
    net = Net()
    net.load_state_dict(torch.load(CHKPT_PATH))
    print(net)

    net.eval()
    with th.no_grad():
        logits = net(g, features)
        _, predictions = th.max(logits, dim=1)
        predictions = predictions.numpy()
    if VISUALIZE:
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        draw(predictions, ax, nxg, positions)  # draw the results

        plt.show()

if __name__ == '__main__':
    main()