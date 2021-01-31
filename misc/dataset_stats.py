import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect
import matplotlib
matplotlib.use('TKAgg')
import os
import argparse


def count_nodes(GRAPH_PATH):
    nxg = nx.read_gpickle(GRAPH_PATH)
    label_dict = nx.get_node_attributes(nxg, 'label')
    labels = label_dict.values()

    n_nodes = nxg.number_of_nodes()
    n_door, n_non_door = 0, 0

    # for each node in the graph
    for lab in labels:
        if lab == 0.0:
            n_non_door += 1
        elif lab == 1.0:
            n_door += 1

    if n_nodes != n_door + n_non_door:
        print("mismatch")

    return n_nodes, n_door, n_non_door

def count_doors(bbox_path):
    with open(bbox_path,"r") as boxes_file:
        bb_lines = boxes_file.readlines()
    return len(bb_lines)

if __name__ == "__main__":
    """

    Command:
        -p path/to/.gpickle
    """
    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-l", "--list", type=str,
    #                default='../data/Public/test_list_reduced.txt', help="path to gpickles")
    #args = vars(ap.parse_args())

    lists = ['../data/Public/train_list_reduced.txt',
             '../data/Public/valid_list_reduced.txt',
             '../data/Public/test_list_reduced.txt']

    total_nodes, total_door_nodes, total_doors = 0, 0, 0

    for list_path in lists:
        print(list_path)
        with open(list_path) as f:
            list = f.read().splitlines()
            for gpickle_path in list:
                gpickle_path = os.path.join('../data/Public/',gpickle_path)
                n_nodes, n_door_nodes, n_non_door_nodes = count_nodes(gpickle_path)
                total_nodes += n_nodes
                total_door_nodes += n_door_nodes

                bbox_path = gpickle_path.replace('/anno/','/bboxes/').replace('w_annotations.gpickle','boxes_image_format.txt')
                n_doors = count_doors(bbox_path)
                total_doors += n_doors

    print("total_nodes {}, total_door_nodes {}, total_non_door_nodes {}".format(total_nodes, total_door_nodes, total_nodes-total_door_nodes))
    print("total_doors {}".format(total_doors))
