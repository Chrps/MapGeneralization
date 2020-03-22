import dgl
import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from sampler import GraphSampler
import warnings
warnings.filterwarnings('ignore')

TEST_PATH = 'data/graphs/public_toilet2.gpickle'
CHKPT_PATH = 'checkpoint/model.pth'
INPUT_SIZE = 3273
VISUALIZE = True

# Define the message and reduce function
# NOTE: We ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)

# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h

def draw(results, ax, nx_G, positions):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'

    colors = []
    for v in range(INPUT_SIZE):
        cls = results[v]
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Results')
    nx.draw_networkx(nx_G.to_undirected(), positions, node_color=colors,
            with_labels=False, node_size=25, ax=ax)

def main():
    # Load Test data
    nxg = nx.read_gpickle(TEST_PATH)

    graph_sampler = GraphSampler()
    #nxg_sub = graph_sampler.up_sampler(nxg, INPUT_SIZE)
    nxg_sub = graph_sampler.down_sampler(nxg, INPUT_SIZE)

    positions = nx.get_node_attributes(nxg_sub, 'pos')
    G = dgl.DGLGraph()
    G.from_networkx(nxg_sub, node_attrs=['pos'])

    # Load the model
    model = GCN(INPUT_SIZE, 5, 2)
    model.load_state_dict(torch.load(CHKPT_PATH))

    # Print model's state_dict(for double checking)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Run test data through loaded model
    inputs_test = torch.eye(INPUT_SIZE)
    output = model(G, inputs_test)
    output = output.detach()

    # Saving the output labels in results
    pos = {}
    results = []
    for v in range(INPUT_SIZE):
        pos[v] = output[v].numpy()
        results.append(pos[v].argmax())

    if VISUALIZE:
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        draw(results, ax, nxg_sub, positions)  # draw the results

        plt.show()

if __name__ == "__main__":
    main()