import dgl
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
matplotlib.use('TKAgg')
import random
import math

def generate_circle_graph():
    g_nx = nx.Graph()
    nr_nodes = random.randint(10, 20)
    radius = random.randint(10, 50)
    step_size = 360/nr_nodes
    degree = 0
    for node_idx in range(nr_nodes):
        # Calculate node on outer circle
        x_noise = random.uniform(5, 8)
        y_noise = random.uniform(5, 8)
        x = radius * math.cos(math.radians(degree)) + x_noise
        y = radius * math.sin(math.radians(degree)) + y_noise
        degree += step_size
        g_nx.add_node(node_idx, pos=(x, y))

        # Calculate node on inner circle
        radius_reduction = random.uniform(1.5, 1.9)
        x_noise = random.uniform(5, 8)
        y_noise = random.uniform(5, 8)
        x = (radius / radius_reduction) * math.cos(math.radians(degree)) + x_noise
        y = (radius / radius_reduction) * math.sin(math.radians(degree)) + y_noise
        g_nx.add_node(node_idx + nr_nodes, pos=(x, y))

        # Connect outer and inner node
        g_nx.add_edge(node_idx, node_idx + nr_nodes)
        if node_idx != 0:
            # Connect nodes in outer circle
            g_nx.add_edge(node_idx - 1, node_idx)
            # Connect nodes in inner circle
            g_nx.add_edge(node_idx + nr_nodes - 1, node_idx + nr_nodes)

    # Connect first and last node in outer circle
    g_nx.add_edge(0, nr_nodes - 1)
    # Connect first and last node in inner circle
    g_nx.add_edge(nr_nodes, nr_nodes + nr_nodes-1)

    pos = nx.get_node_attributes(g_nx, 'pos')
    g_dgl = dgl.DGLGraph(g_nx)

    return g_dgl, pos
'''
def generate_room_graph():
    g_nx = nx.Graph()
    #Create inner square
    nr_nodes = random.randint(10, 20)

    for node_idx in range(nr_nodes):
'''



def main():
    G, pos = generate_circle_graph()

    #label = graph_idx
    fig, ax = plt.subplots()
    nx.draw(G.to_networkx(), pos, with_labels=True, ax=ax)
    #ax.set_title('Class: {:d}'.format(label))
    plt.show()


if __name__ == "__main__":
    main()

