import dgl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

INPUT_SIZE = 3273
NR_EPOCHS = 60
TRAIN_PATH = 'data/graphs/test_graph.gpickle'
CHKPT_PATH = 'checkpoint/model.pth'
VISUALIZE = False

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

def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    return g

def draw(i, all_logits, ax, nx_G, positions):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(INPUT_SIZE):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), positions, node_color=colors,
            with_labels=False, node_size=25, ax=ax)



def main():
    #G = build_karate_club_graph()
    nxg = nx.read_gpickle(TRAIN_PATH)

    #label_dict = nx.get_node_attributes(nxg, 'label')
    #labels = list(label_dict.values())

    labels = list(np.load('data/graph_annotations/labels.npy'))
    nodes = list(nxg.nodes)
    positions = nx.get_node_attributes(nxg, 'pos')

    G = dgl.DGLGraph()
    G.from_networkx(nxg, node_attrs=['pos'])
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    # Step 2: Assign features to nodes or edges
    G.ndata['feat'] = torch.eye(INPUT_SIZE)

    # Step 3: Define a Graph Convolutional Network (GCN)
    # The first layer transforms input features of size of 34 to a hidden size of 5.
    # The second layer transforms the hidden layer and produces output features of
    # size 2, corresponding to the two groups of the karate club.
    net = GCN(INPUT_SIZE, 5, 2)

    # Step 4: Data preparation and initialization
    inputs = torch.eye(INPUT_SIZE)
    labeled_nodes = torch.tensor(nodes)  # Get all nodes
    labels = torch.tensor(labels, dtype=torch.int64)  # and their correspondng labels

    # Step 5: Train
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    all_logits = []
    for epoch in range(NR_EPOCHS):
        logits = net(G, inputs)
        # we save the logits for visualization later
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    
    # Save the model
    torch.save(net.state_dict(), CHKPT_PATH)

    if VISUALIZE:
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        #draw(10, all_logits, ax, nxg, positions)  # draw the prediction of the first epoch
        #plt.close()
        #ani = animation.FuncAnimation(fig, draw(all_logits=all_logits, ax=ax, nx_G=nxg, positions=positions), frames=len(all_logits), interval=200)
        ani = animation.FuncAnimation(fig, draw, fargs=(all_logits, ax, nxg, positions), frames=len(all_logits), interval=200)
        plt.show()


if __name__ == "__main__":
    main()