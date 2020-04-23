import os
import networkx as nx
import matplotlib.pyplot as plt

GRAPH_FILE = r"C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\graph_annotations\Canterbury\M S1 Newton Ground Floor_w_annotations.gpickle"
WIN_X = 1*1000
WIN_Y = 1*1000
STRIDE_X = WIN_X
STRIDE_Y = WIN_Y


def sample_window(positions, tl, br):
    nodes = []
    for i, position in enumerate(positions):
        if tl[0] <= position[0] <= br[0] and tl[1] >= position[1] >= br[1]:
            nodes.append(i)
    return nodes


'''with open(GRAPHS_FILE) as f:
    graph_paths = [line.rstrip() for line in f]

print(len(graph_paths))
for path in graph_paths:'''
file_name = os.path.basename(GRAPH_FILE)
file_name = os.path.splitext(file_name)[0]
if not os.path.exists('data/split_graphs/' + file_name):
    os.makedirs('data/split_graphs/' + file_name)
nxg = nx.read_gpickle(GRAPH_FILE)
positions = nx.get_node_attributes(nxg, 'pos')
positions = list(map(list, positions.values()))
max_pos = (max(positions, key=lambda x: x[0])[0], max(positions, key=lambda x: x[1])[1])
min_pos = (min(positions, key=lambda x: x[0])[0], min(positions, key=lambda x: x[1])[1])
original_nr_nodes = nxg.number_of_nodes()
fig = plt.figure(dpi=150)
ax = fig.subplots()

win_pos = [min_pos[0]+WIN_X, max_pos[1]-WIN_Y]
graph_file_idx = 0
while win_pos[1] > min_pos[1]+STRIDE_Y:
    while win_pos[0] < max_pos[0]-STRIDE_X:
        #print(win_pos)
        tl = (win_pos[0]-WIN_X, win_pos[1]+WIN_Y)
        br = (win_pos[0]+WIN_X, win_pos[1]-WIN_Y)
        nodes = sample_window(positions, tl, br)

        # Show sliding window
        tr = (win_pos[0]+WIN_X, win_pos[1]+WIN_Y)
        bl = (win_pos[0]-WIN_X, win_pos[1]-WIN_Y)
        nxg.add_node(original_nr_nodes + 1, pos=tl, color='r')
        nxg.add_node(original_nr_nodes + 2, pos=br, color='r')
        nxg.add_node(original_nr_nodes + 3, pos=tr, color='r')
        nxg.add_node(original_nr_nodes + 4, pos=bl, color='r')
        nxg.add_edge(original_nr_nodes + 1, original_nr_nodes + 3)
        nxg.add_edge(original_nr_nodes + 3, original_nr_nodes + 2)
        nxg.add_edge(original_nr_nodes + 2, original_nr_nodes + 4)
        nxg.add_edge(original_nr_nodes + 4, original_nr_nodes + 1)
        positions_tmp = nx.get_node_attributes(nxg, 'pos')
        nx.draw_networkx(nxg.to_undirected(), positions_tmp,
                         with_labels=False, node_size=25, ax=ax)

        sub_nxg = nxg.subgraph(nodes)
        sub_nxg = nx.relabel.convert_node_labels_to_integers(sub_nxg)
        sub_positions = nx.get_node_attributes(sub_nxg, 'pos')

        #nx.draw_networkx(sub_nxg.to_undirected(), sub_positions,
        #                 with_labels=False, node_size=25, ax=ax)
        plt.show(block=False)
        plt.pause(1)
        ax.cla()
        win_pos[0] += STRIDE_X
        #if sub_nxg.number_of_nodes() > 10:
        #    nx.write_gpickle(sub_nxg, 'data/split_graphs/' + file_name + '/' + file_name + str(graph_file_idx) + '.gpickle')
        #graph_file_idx += 1

    win_pos[1] -= STRIDE_Y
    win_pos[0] = min_pos[0]+WIN_X

