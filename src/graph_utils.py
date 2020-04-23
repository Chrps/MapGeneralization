import os
import networkx as nx
import dgl
from dgl import DGLGraph
import torch
import numpy as np
from src.deepwalk.deepwalk import DeepWalk
from src.sliding_window_class import SlidingWindow
sliding_window = SlidingWindow()

def norm_ang(angle):
    old_max = 180
    old_min = -180
    new_max = 1
    new_min = 0
    old_range = old_max - old_min
    new_range = new_max - new_min
    out_angle = (((angle - old_min) * new_range) / old_range) + new_min
    return out_angle

def calculate_angles(nxg):
    angles = []
    nx.set_node_attributes(nxg, angles, 'angle')
    nx.set_edge_attributes(nxg, angles, 'angle')

    # Calculate angle of each edge
    for edge in nxg.edges:
        pos1 = nxg.nodes[edge[0]]['pos']
        pos2 = nxg.nodes[edge[1]]['pos']
        deltaY = pos2[1] - pos1[1]
        deltaX = pos2[0] - pos1[0]
        angleInDegrees = np.degrees(np.arctan2(deltaY, deltaX))
        norm_angle = norm_ang(angleInDegrees)
        nxg.edges[edge]['angle'] = norm_angle

    # Calculate mean angle for all edges going to node.
    for node in nxg.nodes:
        angle_sum = 0
        for edge in nxg.edges(node):
            angle_sum += nxg.edges[edge]['angle']
        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['angle'] = 0
        else:
            nxg.nodes[node]['angle'] = angle_sum/len(nxg.edges(node))


def group_graphs_labels_features(data_list, folder, windowing=False):
    data_path = 'data/'
    data_files = [os.path.join(data_path, folder, line.rstrip()) for line in open(data_list)]

    # Initialize empty list
    dataset = []

    for idx, file in enumerate(data_files):
        graph = []
        nxg = nx.read_gpickle(file)


        # Get the annotated labels
        labels = get_labels(nxg)
        # Get the feature from the file
        features = chris_get_features(nxg)

        dgl_g = DGLGraph()
        dgl_g.from_networkx(nxg)
        dgl_g.readonly()

        # Append the information for batching
        graph.append(dgl_g)
        graph.append(labels)
        graph.append(features)
        dataset.append(graph)

    return dataset



def batch_graphs(data_list, folder, windowing=False):
    data_path = 'data/'
    data_files = [os.path.join(data_path, folder, line.rstrip()) for line in open(data_list)]

    all_graphs = []
    all_labels = []
    all_features = []

    for file in data_files:
        # Convert the gpickle file to a dgl graph for batching
        #dgl_g = convert_gpickle_to_dgl_graph(file)
        nxg = nx.read_gpickle(file)

        if windowing:
            nxg_list = sliding_window.perform_windowing(nxg)
            for nxg in nxg_list:
                # Get the annotated labels
                labels = get_labels(nxg)
                # Get the feature from the file
                features = chris_get_features(nxg)

                dgl_g = DGLGraph()
                dgl_g.from_networkx(nxg)
                dgl_g.readonly()

                # Append the information for batching
                all_graphs.append(dgl_g)
                all_labels.append(labels)
                all_features.append(features)
        else:
            # Get the annotated labels
            labels = get_labels(nxg)
            # Get the feature from the file
            features = chris_get_features(nxg)

            dgl_g = DGLGraph()
            dgl_g.from_networkx(nxg)
            dgl_g.readonly()

            # Append the information for batching
            all_graphs.append(dgl_g)
            all_labels.append(labels)
            all_features.append(features)

    # Batch the graphs
    batched_graph = dgl.batch(all_graphs)

    # all_labels is a list of tensors, so concetenate into one tensor
    conc_labels = torch.LongTensor(batched_graph.number_of_nodes(), 1)
    torch.cat(all_labels, out=conc_labels)

    # all_features is a list of tensors, so concetenate into one tensor
    conc_features = torch.Tensor(batched_graph.number_of_nodes(), 1)
    torch.cat(all_features, out=conc_features)


    return batched_graph, conc_labels, conc_features


def get_labels(nxg):
    # Get labels from netx graph
    label_dict = nx.get_node_attributes(nxg, 'label')
    labels = list(label_dict.values())
    labels = torch.LongTensor(labels)

    return labels

def convert_gpickle_to_dgl_graph(file):
    nxg = nx.read_gpickle(file)
    # Define DGL graph from netx graph
    dgl_g = DGLGraph()
    dgl_g.from_networkx(nxg)
    dgl_g.readonly()

    return dgl_g


def get_features(file):
    # Read file as networkx graph and retrieve positions + labels
    nxg = nx.read_gpickle(file)

    # % % Define some features for the graph
    # % Normalized positions
    positions = nx.get_node_attributes(nxg, 'pos')
    positions = list(positions.values())
    norm_positions = normalize_positions(positions)
    norm_positions = torch.FloatTensor(norm_positions)

    # % Normalized node degree (number of edges connected to node)
    dgl_g = convert_gpickle_to_dgl_graph(file)
    norm_degrees = 1. / dgl_g.in_degrees().float().unsqueeze(1)

    # % Normalized unique identity (entity type [ARC/CRCLE/LINE])
    id_dict = nx.get_node_attributes(nxg, 'id')
    ids = list(id_dict.values())
    for idx, id in enumerate(ids):
        if id == 4:  # CIRCLE
            ids[idx] = 0
        if id == 3:  # ARC
            ids[idx] = 4
        if id == 0:
            ids[idx] = 1
        if id == 1:
            ids[idx] = 2
        if id == 2:
            ids[idx] = 3

    # norm_ids = [float(i)/float(max(ids)) for i in ids]
    norm_ids = [float(i) / 4.0 for i in ids]
    norm_ids = torch.FloatTensor(np.asarray(norm_ids).reshape(-1, 1))

    # % DeepWalk
    dpwlk = DeepWalk(number_walks=4, walk_length=5, representation_size=2)
    # Create embedding file for given file
    dpwlk.create_embeddings(nxg, file)
    # Read the embedding file for given file
    embedding_feat = dpwlk.read_embeddings(file)

    # % Combine all features into one tensor
    #features = torch.cat((norm_positions, norm_deg, norm_ids, embedding_feat), 1)
    features = torch.cat((norm_degrees, norm_ids, embedding_feat), 1)

    return features

def chris_get_features(nxg):

    # % % Define some features for the graph
    # % Normalized positions
    positions = nx.get_node_attributes(nxg, 'pos')
    positions = list(positions.values())
    norm_positions = normalize_positions(positions)
    norm_positions = torch.FloatTensor(norm_positions)

    # % Normalized node degree (number of edges connected to node)
    dgl_g = DGLGraph()
    dgl_g.from_networkx(nxg)
    dgl_g.readonly()
    norm_degrees = 1. / dgl_g.in_degrees().float().unsqueeze(1)

    # % Normalized unique identity (entity type [ARC/CRCLE/LINE])
    id_dict = nx.get_node_attributes(nxg, 'id')
    ids = list(id_dict.values())
    for idx, id in enumerate(ids):
        if id == 0:
            ids[idx] = 0
        else:
            ids[idx] = 1. / id

    norm_identity = np.linspace(0, 1, num=nxg.number_of_nodes())
    norm_identity = np.reshape(norm_identity, (nxg.number_of_nodes(), 1))
    norm_identity = torch.FloatTensor(norm_identity)

    # norm_ids = [float(i)/float(max(ids)) for i in ids]
    #norm_ids = [float(i) / 4.0 for i in ids]
    norm_ids = torch.FloatTensor(np.asarray(ids).reshape(-1, 1))

    # Angles:
    calculate_angles(nxg)
    angles = nx.get_node_attributes(nxg, 'angle')
    angles = list(angles.values())
    angles = torch.FloatTensor(np.asarray(angles).reshape(-1, 1))

    # % DeepWalk
    #dpwlk = DeepWalk(number_walks=4, walk_length=5, representation_size=2)
    # Create embedding file for given file
    #dpwlk.create_embeddings(nxg, file)
    # Read the embedding file for given file
    #embedding_feat = dpwlk.read_embeddings(file)

    # % Combine all features into one tensor
    #features = torch.cat((norm_positions, norm_deg, norm_ids, embedding_feat), 1)
    features = torch.cat((norm_degrees, norm_identity, angles, norm_ids), 1)

    return features

def normalize_positions(positions):
    norm_pos = []
    min_x = min(positions)[0]
    min_y = min(positions)[1]
    max_x = max(positions)[0]
    max_y = max(positions)[1]
    if min_x < 0:
        max_x = max_x - min_x
    if min_y < 0:
        max_y = max_y - min_y
    for position in positions:
        temp_pos = []
        x_pos = 0
        y_pos = 0
        if min_x < 0:
            x_pos = position[0] - min_x
        if min_y < 0:
            y_pos = position[1] - min_y
        x_pos = position[0] / max_x
        y_pos = position[1] / max_y
        temp_pos = tuple([x_pos, y_pos])
        norm_pos.append(temp_pos)
    return norm_pos