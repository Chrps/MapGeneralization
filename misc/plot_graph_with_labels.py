import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect
import matplotlib
matplotlib.use('TKAgg')
import os

GRAPH_PATH = r"D:\University Stuff\OneDrive - Aalborg Universitet\P10 - Master's Thesis\data\graph_annotations\AU\A1322PE-0_w_annotations.gpickle"
nxg = nx.read_gpickle(GRAPH_PATH)
label_dict = nx.get_node_attributes(nxg, 'label')
labels = list(label_dict.values())

# create empty list for node colors
node_color = []
# for each node in the graph
for lab in labels:
    if lab == 0.0:
        node_color.append('blue')
    elif lab == 1.0:
        node_color.append('red')

pos = nx.get_node_attributes(nxg, 'pos')

w, h = figaspect(5 / 3)
fig, ax = plt.subplots(figsize=(w, h))
nx.draw(nxg, pos, node_color=node_color, node_size=20, ax=ax)
#nx.draw_networkx_labels(new_g, pos, lab, ax=ax)
plt.show()



