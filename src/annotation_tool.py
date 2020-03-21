import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

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
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    #np.random.seed(19680801)

    #data = np.random.rand(10000, 2)

    import networkx as nx
    G = nx.read_gpickle("../data/graphs/test_graph.gpickle")
    graph_data = nx.get_node_attributes(G, 'pos')
    data = np.array(tuple(graph_data.values()))
    print(data)
    labels = np.zeros(len(data))
    instances = np.zeros(len(data))

    while True:

        #subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
        #fig, ax = plt.subplots(subplot_kw=subplot_kw)
        fig, ax = plt.subplots()

        pts = ax.scatter(data[:, 0], data[:, 1], s=80, c = 'b')
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
                np.save('../data/graph_annotations/labels_walls_and_doors.npy', labels)

            if event.key == "enter":
                print(mode)
                print("Selected points:")
                #print(selector.xys[selector.ind])
                print(selector.ind)
                if mode == 'SELECT':
                    class_label = 1
                    np.put(instances, selector.ind, instance, mode='clip')
                elif mode == 'DESELECT':
                    class_label = 0
                    np.put(instances, selector.ind, 0, mode='clip')
                np.put(labels, selector.ind, class_label, mode='clip')

                print(instances)
                for idx in range(len(selector.fc)):
                    class_label = labels[idx]
                    if class_label == 1:
                        selector.fc[idx] = [1, 0, 0, 1]
                    else:
                        selector.fc[idx] = [0, 0, 1, 1]

                unique = np.unique(instances)
                for a in ann_list:
                    a.remove()
                ann_list = []
                for an_instance in unique:
                    if an_instance != 0:
                        # NOT SURE WHY IT MUST BE "!=" and not "="
                        instance_indices = np.where(instances == an_instance)
                        x_coor = np.mean(data[instance_indices, 0])
                        y_coor = np.mean(data[instance_indices, 1])
                        ann = ax.annotate(str(int(an_instance)),
                                    xy=(x_coor, y_coor))
                        ann_list.append(ann)



                #pts = ax.scatter(data[:, 0], data[:, 1], s=80, c='r')

                #ax.scatter(data[:, 0], data[:, 1], s=80, color=labels)
                #ax.set_title("")
            fig.suptitle("Mode: " + mode + " | Instance: " + str(instance), x=0.5, y=0.05)
            fig.canvas.draw()


        fig.canvas.mpl_connect("key_press_event", accept)
        ax.set_title("Mode: 'n'=SELECT, 'm'=DESELECT \n Instance: ','=INCREMENT, '.'=DECREMENT")

        plt.show()
