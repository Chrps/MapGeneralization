import dgl
from dgl.data import MiniGCDataset
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

def build_graph():
    g_nx = nx.Graph()
    # add nodes into the graph; nodes are labeled from 0~
    list_of_nodes = [(50, 50), (100, 50), (100, 100), (50, 100),
                     (0, 0), (200, 0), (200, 200), (0, 200)]

    for node_idx, node in enumerate(list_of_nodes):
        g_nx.add_node(node_idx, pos=node)

    # all edges as a list of tuples
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4)]
    g_nx.add_edges_from(edge_list)
    pos = nx.get_node_attributes(g_nx, 'pos')
    #convert nx graph to DGL graph
    g_dgl = dgl.DGLGraph(g_nx)

    return g_dgl, pos

def main():
    # %% Visualize the Graph
    G, pos = build_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    nx.draw(G.to_networkx(), pos, with_labels=True)
    plt.show()

    # %% Assign Features to Nodes and/or Edges

if __name__ == "__main__":
    main()