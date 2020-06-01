import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect
import matplotlib
from math import sqrt
matplotlib.use('TKAgg')
import os

GRAPH_TO_SCALE_PATH = r"C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\graph_annotations\RTX\512D_w_annotations.gpickle"
OUTPUT_FOLDER = r"C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\scaled_graph_annotations\RTX"

def refreshGraph(graph, node_color, fig):
    print("here")
    #plt.clf()
    pos = nx.get_node_attributes(graph, 'pos')
    w, h = figaspect(5 / 3)
    fig, ax = plt.subplots(figsize=(w, h))
    nx.draw(graph, pos, node_color=node_color, node_size=20, ax=ax)
    fig.canvas.draw()
    #fig.canvas.flush_events()

def onclick(event, graph, node_color, fig):
    print(event.xdata, event.ydata)
    '''
    (x, y) = (event.xdata, event.ydata
    pos = nx.get_node_attributes(graph, 'pos')
    # Moving everything to origin
    positions = list(pos.values())
    x_pos = []
    y_pos = []
    for coord in positions:
        x_pos.append(coord[0])
        y_pos.append(coord[1])

    min_x = min(x_pos)
    max_x = max(x_pos)
    min_y = min(y_pos)
    max_y = max(y_pos)

    normalizedX = (x-min_x)/(max_x-min_x)
    normalizedy = (y - min_y) / (max_y - min_y)

    for i in range(graph.number_of_nodes()):
        node_pos = (graph.nodes[i]['pos'])
        normalized_nodeX = (node_pos[0] - min_x) / (max_x - min_x)
        normalized_nodeY = (node_pos[1] - min_y) / (max_y - min_y)
        #distance = pow(normalizedX - normalized_nodeX, 2) + pow(normalizedy - normalized_nodeY, 2)
        #distance = sqrt((x-float(node_pos[0]))**2 + (y-float(node_pos[1]))**2)
        distance = sqrt((normalizedX - normalized_nodeX) ** 2 + (normalizedy - normalized_nodeY) ** 2)
        #print(distance)
        if distance < 1000:
            node_color[i] = 'green'
            refreshGraph(graph, node_color, fig)
            break
    '''
# Plot Original Graph to Get Scale Factor
nxg = nx.read_gpickle(GRAPH_TO_SCALE_PATH)

label_dict = nx.get_node_attributes(nxg, 'label')
labels = list(label_dict.values())
values = list(set(labels))
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
fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, graph=nxg, node_color=node_color, fig=fig))
plt.show()

# After Calculating scale factor, prompt user for input
print("Please Input Scale Factor: ")
SCALE_FACTOR = float(input())

# Moving everything to origin
positions = list(pos.values())
x_pos = []
y_pos = []
for coord in positions:
    x_pos.append(coord[0])
    y_pos.append(coord[1])

min_x = min(x_pos)
max_x = max(x_pos)
min_y = min(y_pos)
max_y = max(y_pos)
if min_x < 0:  # Negative
    x_origin = abs(min_x)
else:  # Positive
    x_origin = -min_x
if min_y < 0:  # Negative
    y_origin = abs(min_y)
else:  # Positive
    y_origin = -min_y

# Rescale the Graph
copy_graph = nxg.copy()

for i in range(copy_graph.number_of_nodes()):
    orig_pos = (copy_graph.nodes[i]['pos'])
    new_pos = ((orig_pos[0] + x_origin) * SCALE_FACTOR, (orig_pos[1] + y_origin) * SCALE_FACTOR)
    copy_graph.nodes[i]['pos'] = new_pos

# Plot the new scaled graph to double check
scale_label_dict = nx.get_node_attributes(copy_graph, 'label')
scale_labels = list(label_dict.values())
scale_values = list(set(labels))
# create empty list for node colors
scale_node_color = []
# for each node in the graph
for lab in scale_labels:
    if lab == 0.0:
        scale_node_color.append('blue')
    elif lab == 1.0:
        scale_node_color.append('red')

scale_pos = nx.get_node_attributes(copy_graph, 'pos')
w, h = figaspect(5 / 3)
fig, ax = plt.subplots(figsize=(w, h))
nx.draw(copy_graph, scale_pos, node_color=scale_node_color, node_size=20, ax=ax)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Save the Results
file_name = os.path.basename(GRAPH_TO_SCALE_PATH)
file_name = os.path.splitext(file_name)[0]
np.save(OUTPUT_FOLDER + '/' + file_name + '.npy', labels)
#nx.set_node_attributes(copy_graph, labels, 'label')
#for node in copy_graph.nodes:
    #copy_graph.nodes[node]['label'] = labels[node]
nx.write_gpickle(copy_graph, OUTPUT_FOLDER + '/' + file_name + '.gpickle')