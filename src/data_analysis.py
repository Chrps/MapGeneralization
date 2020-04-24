import os
import networkx as nx
import numpy as np

def get_labels(nxg_):
    # Get labels from netx graph
    label_dict = nx.get_node_attributes(nxg_, 'label')
    return list(label_dict.values())

folder = r'C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\graph_annotations'
data_list = '../data/valid_file_list_Canterbury_and_AU.txt'
data_files = [os.path.join(folder, line.rstrip()) for line in open(data_list)]

nr_nodes_list = []
nr_edges_list = []
nr_door_nodes_list = []
nr_non_door_nodes_list = []
for file in data_files:
    nxg = nx.read_gpickle(file)
    nr_nodes_list.append(nxg.number_of_nodes())
    nr_edges_list.append(nxg.number_of_edges())
    labels = np.asarray(get_labels(nxg))
    labels0_idx = np.where(labels == 0)[0]
    labels1_idx = np.where(labels == 1)[0]
    nr_non_door_nodes_list.append(len(labels0_idx))
    nr_door_nodes_list.append(len(labels1_idx))

total_nr_graphs = len(data_files)
total_nr_nodes = sum(nr_nodes_list)
total_nr_edges = sum(nr_edges_list)
total_nr_door_nodes = sum(nr_door_nodes_list)
total_nr_non_door_nodes = sum(nr_non_door_nodes_list)
std_of_nodes = np.std(np.asarray(nr_nodes_list), axis=0)
std_of_edges = np.std(np.asarray(nr_edges_list), axis=0)
std_of_door_nodes = np.std(np.asarray(nr_door_nodes_list), axis=0)
std_of_non_door_nodes = np.std(np.asarray(nr_non_door_nodes_list), axis=0)

print("DATASET ANALYSIS:")
print("Nr. of graphs: %d" % total_nr_graphs)
print("Nr. of nodes: %d" % total_nr_nodes)
print("Nr. of edges: %d" % total_nr_edges)
print("Nr. of door nodes: %d" % total_nr_door_nodes)
print("Nr. of non-door nodes: %d\n" % total_nr_non_door_nodes)
print("Percentage of door nodes: %.2f%%" % float(total_nr_door_nodes/total_nr_nodes*100))
print("Percentage of non-door nodes: %.2f%%\n" % float(total_nr_non_door_nodes/total_nr_nodes*100))
print("Std. of nodes: %.2f" % std_of_nodes)
print("Std. of edges: %.2f" % std_of_edges)
print("Std. of door nodes: %.2f" % std_of_door_nodes)
print("Std. of non-door nodes: %.2f" % std_of_non_door_nodes)
