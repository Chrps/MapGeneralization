import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import networkx as nx
import os

GRAPH_PATH = "../data/graphs/MSP1-HoM-MA-XX+5-ET_cloudconvert.gpickle"


# TODO: CAN BE MADE MUCH BETTER
class SelectFromCollection(object):
    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':

    G = nx.read_gpickle(GRAPH_PATH)
    graph_data = nx.get_node_attributes(G, 'pos')
    data = np.array(tuple(graph_data.values()))
    labels = np.zeros(len(data))
    instances = np.zeros(len(data))
    edgelist = list(G.edges())
    edge_pos = np.asarray([(graph_data[e[0]], graph_data[e[1]]) for e in edgelist])
    edge_collection = LineCollection(edge_pos)
    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(None)

    fig, ax = plt.subplots()

    pts = ax.scatter(data[:, 0], data[:, 1], s=80, c = 'b')
    ax.add_collection(edge_collection)
    selector = SelectFromCollection(ax, pts)
    fc = pts.get_facecolors()
    mode = 'SELECT'
    instance = 1

    fig.suptitle("Mode: " + mode + " | Instance: " + str(instance), x=0.5, y=0.05)
    ann_list = []

    def accept(event):
        global mode
        global instance
        global ann_list
        if event.key == "n":
            mode = 'SELECT'

        if event.key == "m":
            mode = 'DESELECT'

        if event.key == ",":
            instance += 1

        if event.key == ".":
            instance -= 1

        if event.key == "b":
            file_name = os.path.basename(GRAPH_PATH)
            file_name = os.path.splitext(file_name)[0]
            print(selector.ind)
            sub_G = G.subgraph(selector.ind)
            sub_G = nx.relabel.convert_node_labels_to_integers(sub_G)
            nx.write_gpickle(sub_G, '../data/graphs/' + file_name + '_sub.gpickle')

        if event.key == "enter":


            for idx in range(len(selector.fc)):
                class_label = labels[idx]
                if class_label == 1:
                    selector.fc[idx] = [1, 0, 0, 1]
                else:
                    selector.fc[idx] = [0, 0, 1, 1]

            #pts = ax.scatter(data[:, 0], data[:, 1], s=80, c='r')

            #ax.scatter(data[:, 0], data[:, 1], s=80, color=labels)
            #ax.set_title("")
        fig.suptitle("Mode: " + mode + " | Instance: " + str(instance), x=0.5, y=0.05)
        fig.canvas.draw()


    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Mode: 'n'=SELECT, 'm'=DESELECT \n Instance: ','=INCREMENT, '.'=DECREMENT")

    plt.show()
