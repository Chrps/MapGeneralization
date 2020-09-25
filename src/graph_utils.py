import os
# from builtins import range

import networkx as nx
import dgl
import torch
import numpy as np
# from src.deepwalk.deepwalk import DeepWalk
from src.sliding_window_class import SlidingWindow
sliding_window = SlidingWindow()

def extract_max_difference_angle(nxg):  
    # Maximum difference angle requires that angles have been computed in advance.
    max_diff_angle = []
    tmp_prev_edge_angle = []
    prev_edge_angle = []
    nx.set_node_attributes(nxg, max_diff_angle, 'max_diff_angle')
    nx.set_edge_attributes(nxg, max_diff_angle, 'max_diff_angle')
    for node_ndx, node in enumerate(nxg.nodes):
        max_diff_angle = 0.0
        if node == 0:
            prev_edge_angle = 0.0
        for edge_idx, edge in enumerate(nxg.edges(node)):
            if np.abs( np.asarray(nxg.edges[edge]['angle'])-prev_edge_angle ) > np.abs(max_diff_angle) :
                max_diff_angle = nxg.edges[edge]['angle']-prev_edge_angle
                tmp_prev_edge_angle = nxg.edges[edge]['angle']
        prev_edge_angle = tmp_prev_edge_angle
        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['max_diff_angle'] = 0
        else:
            nxg.nodes[node]['max_diff_angle'] = max_diff_angle

def extract_min_difference_angle(nxg):  
    # Minimum difference angle requires that angles have been computed in advance.
    min_diff_angle = []
    tmp_prev_edge_angle = []
    prev_edge_angle = []
    nx.set_node_attributes(nxg, min_diff_angle, 'min_diff_angle')
    nx.set_edge_attributes(nxg, min_diff_angle, 'min_diff_angle')
    for node_ndx, node in enumerate(nxg.nodes):
        if node == 0:
            prev_edge_angle = 0.0
        for edge_idx, edge in enumerate(nxg.edges(node)):
            test_angle = nxg.edges[edge]['angle']
            
            if edge[0]-edge[1] > 0: # if minimum is in the opposite direction the subtract -0.5 (negative of angle normalization value)
                test_angle = test_angle - 0.5
            
            if edge_idx==0: # initialization assumes that first difference is smallest
                min_diff_angle = test_angle-prev_edge_angle
                tmp_prev_edge_angle = test_angle

            if np.abs( test_angle-prev_edge_angle ) < np.abs(min_diff_angle) : 
                min_diff_angle = nxg.edges[edge]['angle']-prev_edge_angle
                tmp_prev_edge_angle = nxg.edges[edge]['angle']
        prev_edge_angle = tmp_prev_edge_angle
        
        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['min_diff_angle'] = 0
        else:
            nxg.nodes[node]['min_diff_angle'] = min_diff_angle

def extract_max_loglength_ratio_absolute(nxg):
    # maximum log-length ratio (Absolute)
    max_length_log_ratio = []
    prev_edge_length = []
    tmp_prev_edge_length = []

    nx.set_node_attributes(nxg, max_length_log_ratio, 'max_length_log_ratio')
    nx.set_edge_attributes(nxg, max_length_log_ratio, 'max_length_log_ratio')
    for node_ndx, node in enumerate(nxg.nodes):
        max_length_log_ratio = 0.0
        if node == 0:
            prev_edge_length = 0.0
        for edge_idx, edge in enumerate(nxg.edges(node)):
            if edge_idx==0: # initialization assumes that first difference is largest
                max_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
            if np.abs(nxg.edges[edge]['length_log']-prev_edge_length) > np.abs(max_length_log_ratio) : 
                max_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
        prev_edge_length = tmp_prev_edge_length

        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['max_length_log_ratio'] = 0
        else:
            nxg.nodes[node]['max_length_log_ratio'] = max_length_log_ratio


def extract_min_loglength_ratio_absolute(nxg):
    # Minimum Length Ratio (Absolute)
    prev_edge_length = []
    min_length_log_ratio = 1000000.0
    tmp_prev_edge_length = 0.0
    tmp_min = []
    skipped = False

    nx.set_node_attributes(nxg, min_length_log_ratio, 'min_length_log_ratio')
    nx.set_edge_attributes(nxg, min_length_log_ratio, 'min_length_log_ratio')
    for node_ndx, node in enumerate(nxg.nodes):
        min_length_log_ratio = 0.0
        if node == 0:
            prev_edge_length = 0.0
            nxg.nodes[node]['min_length_log_ratio'] = 0
            # For node[0] run through full loop and find minimum length except the first edge ehich is zero
            for edge_idx, edge in enumerate(nxg.edges(node)):
                if (edge_idx==0): # first value at position zero is always 0. We dont want this value
                    tmp_min = 0.0
                if (edge_idx>0 and tmp_min>nxg.edges[edge]['length_log']): # compare to find minimum.
                    tmp_min = nxg.edges[edge]['length_log']
                    prev_edge_length = tmp_min
            continue
        
        for edge_idx, edge in enumerate(nxg.edges(node)):
            if edge[0]-edge[1] > 0: # if opposite node-direction then skip comparison in this iteration of loop
                skipped = True
                continue
            if (edge_idx==0 or skipped==True): # initialization assumes that first ratio is smallest (same length)
                skipped = False
                min_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
            if np.abs(nxg.edges[edge]['length_log']-prev_edge_length) < np.abs(min_length_log_ratio) : 
                min_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']

        prev_edge_length = tmp_prev_edge_length

        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['min_length_log_ratio'] = 0
        else:
            nxg.nodes[node]['min_length_log_ratio'] = min_length_log_ratio

def norm_ang(angle, new_max=0.5, new_min=-0.5):
    old_max = 180
    old_min = -180
    #new_max = 1
    #new_min = -1 #0
    old_range = old_max - old_min
    new_range = new_max - new_min
    out_angle = (((angle - old_min) * new_range) / old_range) + new_min
    if (np.abs(out_angle)>new_max-0.00000001):
        out_angle = 0.0
    return out_angle

def calculate_angles_and_length(nxg):
    angles = []
    lengths_euql = []
    # max_diff_angles = []
    #min_diff_angles = []
    #min_length_log_ratio = []
    #max_length_log_ratio = []

    nx.set_node_attributes(nxg, angles, 'angle')
    nx.set_edge_attributes(nxg, angles, 'angle')
    nx.set_node_attributes(nxg, lengths_euql, 'length_euql')
    nx.set_edge_attributes(nxg, lengths_euql, 'length_euql')
    nx.set_node_attributes(nxg, lengths_euql, 'length_log')
    nx.set_edge_attributes(nxg, lengths_euql, 'length_log')
    # nx.set_node_attributes(nxg, max_diff_angles, 'max_diff_angle')
    # nx.set_edge_attributes(nxg, max_diff_angles, 'max_diff_angle')
    # nx.set_node_attributes(nxg, min_diff_angles, 'min_diff_angle')
    # nx.set_edge_attributes(nxg, min_diff_angles, 'min_diff_angle')
    # nx.set_node_attributes(nxg, min_length_log_ratio, 'min_length_log_ratio')
    # nx.set_edge_attributes(nxg, min_length_log_ratio, 'min_length_log_ratio')
    # nx.set_node_attributes(nxg, max_length_log_ratio, 'max_length_log_ratio')
    # nx.set_edge_attributes(nxg, max_length_log_ratio, 'max_length_log_ratio')

    # Calculate angle and distance of each edge
    for edge in nxg.edges:
        pos1 = nxg.nodes[edge[0]]['pos']
        pos2 = nxg.nodes[edge[1]]['pos']
        deltaY = pos2[1] - pos1[1]
        deltaX = pos2[0] - pos1[0]
        angleInDegrees = np.degrees(np.arctan2(deltaY, deltaX))
        norm_angle = norm_ang(angleInDegrees,0.5,-0.5)
        nxg.edges[edge]['angle'] = norm_angle

        length_euql = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
        if length_euql == 0:
            nxg.edges[edge]['length_euql'] = 0
        else:
            nxg.edges[edge]['length_euql']  = 1.0/length_euql
        #    nxg.edges[edge]['length_euql'] = np.abs(np.log(norm_length_euql))/10.0
        
        length_euql=length_euql+1.0
        nxg.edges[edge]['length_log'] = np.abs(np.log(length_euql))/10.0 # for 20: 1e9 is scaled to 1.

    # Mean angle of all edges going to node.
    for node in nxg.nodes:
        angle_sum = 0
        length_euql_sum = 0
        angle_max = 0
        angle_min = 0
        length_euql_max = 0
        length_euql_min = 0
        length_log_max = 0
        length_log_min = 0
        tmp_prev_edge_angle = 0

        for edge_idx, edge in enumerate(nxg.edges(node)):
            if edge_idx == 0:
                angle_max = nxg.edges[edge]['angle']
                angle_min = nxg.edges[edge]['angle']
                length_euql_max = nxg.edges[edge]['length_euql']
                length_euql_min = nxg.edges[edge]['length_euql']
                length_log_max = nxg.edges[edge]['length_log']
                length_log_min = nxg.edges[edge]['length_log']
            else:
                if np.abs(nxg.edges[edge]['angle']) > np.abs(angle_max):
                    angle_max = nxg.edges[edge]['angle']
                if np.abs(nxg.edges[edge]['angle']) < np.abs(angle_min):
                    angle_min = nxg.edges[edge]['angle']
                if nxg.edges[edge]['length_euql'] > length_euql_max:
                    length_euql_max = nxg.edges[edge]['length_euql']
                if nxg.edges[edge]['length_euql'] < length_euql_min:
                    length_euql_min = nxg.edges[edge]['length_euql']
                if nxg.edges[edge]['length_log'] > length_log_max:
                    length_log_max = nxg.edges[edge]['length_log']
                if nxg.edges[edge]['length_log'] < length_log_min:
                    length_log_min = nxg.edges[edge]['length_log']

            angle_sum += nxg.edges[edge]['angle']
            length_euql_sum += nxg.edges[edge]['length_euql']
        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['angle'] = 0
            nxg.nodes[node]['length_euql'] = 0
            nxg.nodes[node]['max_angle'] = 0
            nxg.nodes[node]['min_angle'] = 0
            nxg.nodes[node]['max_length_euql'] = 0
            nxg.nodes[node]['min_length_euql'] = 0
            nxg.nodes[node]['max_length_log'] = 0
            nxg.nodes[node]['min_length_log'] = 0
        else:
            nxg.nodes[node]['angle'] = angle_sum / len(nxg.edges(node))
            nxg.nodes[node]['length_euql'] = length_euql_sum / len(nxg.edges(node))
            nxg.nodes[node]['max_angle'] = angle_max
            nxg.nodes[node]['min_angle'] = angle_min
            nxg.nodes[node]['max_length_euql'] = length_euql_max
            nxg.nodes[node]['min_length_euql'] = length_euql_min
            nxg.nodes[node]['max_length_log'] = length_log_max
            nxg.nodes[node]['min_length_log'] = length_log_min

    # # Maximum difference angle
    # for node_ndx, node in enumerate(nxg.nodes):
    #     max_diff_angle = 0.0
    #     if node == 0:
    #         prev_edge_angle = 0.0
    #     #else:  
    #     for edge_idx, edge in enumerate(nxg.edges(node)):
    #         if np.abs( nxg.edges[edge]['angle']-prev_edge_angle ) > np.abs(max_diff_angle) :
    #             max_diff_angle = nxg.edges[edge]['angle']-prev_edge_angle
    #             tmp_prev_edge_angle = nxg.edges[edge]['angle']
    #     prev_edge_angle = tmp_prev_edge_angle
    #     if len(nxg.edges(node)) == 0:
    #         nxg.nodes[node]['max_diff_angle'] = 0
    #     else:
    #         nxg.nodes[node]['max_diff_angle'] = max_diff_angle

    # # Minimum difference angle
    # for node_ndx, node in enumerate(nxg.nodes):
    #     if node == 0:
    #         prev_edge_angle = 0.0
    #     for edge_idx, edge in enumerate(nxg.edges(node)):
    #         test_angle = nxg.edges[edge]['angle']
            
    #         if edge[0]-edge[1] > 0: # if minimum is in the opposite direction the subtract -0.5 (negative of angle normalization value)
    #             test_angle = test_angle - 0.5
            
    #         if edge_idx==0: # initialization assumes that first difference is smallest
    #             min_diff_angle = test_angle-prev_edge_angle
    #             tmp_prev_edge_angle = test_angle

    #         if np.abs( test_angle-prev_edge_angle ) < np.abs(min_diff_angle) : 
    #             min_diff_angle = nxg.edges[edge]['angle']-prev_edge_angle
    #             tmp_prev_edge_angle = nxg.edges[edge]['angle']
    #     prev_edge_angle = tmp_prev_edge_angle
        
    #     if len(nxg.edges(node)) == 0:
    #         nxg.nodes[node]['min_diff_angle'] = 0
    #     else:
    #         nxg.nodes[node]['min_diff_angle'] = min_diff_angle

    '''
    # minimum log-length ratio (Growing)
    for node_ndx, node in enumerate(nxg.nodes):
        min_length_log_ratio = 0.0
        if node == 0:
            prev_edge_length = 0.0
        #selse:
        for edge_idx, edge in enumerate(nxg.edges(node)):
            if edge_idx==0: # initialization assumes that first difference is largest
                min_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
            if nxg.edges[edge]['length_log']-prev_edge_length < min_length_log_ratio :
                min_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
        prev_edge_length = tmp_prev_edge_length
        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['min_length_log_ratio'] = 0
        else:
            nxg.nodes[node]['min_length_log_ratio'] = min_length_log_ratio   
 
    # maximum log-length ratio (Growing)
    for node_ndx, node in enumerate(nxg.nodes):
        max_length_log_ratio = 0.0
        if node == 0:
            prev_edge_length = 0.0
        #else:
        for edge_idx, edge in enumerate(nxg.edges(node)):
            if edge_idx==0: # initialization assumes that first difference is largest
                max_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
            if nxg.edges[edge]['length_log']-prev_edge_length > max_length_log_ratio : # uden > np.abs(diff_angle) : ??
                max_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
                tmp_prev_edge_length = nxg.edges[edge]['length_log']
        prev_edge_length = tmp_prev_edge_length
        if len(nxg.edges(node)) == 0:
            nxg.nodes[node]['max_length_log_ratio'] = 0
        else:
            nxg.nodes[node]['max_length_log_ratio'] = max_length_log_ratio

    '''

    # # maximum log-length ratio (Absolute)
    # for node_ndx, node in enumerate(nxg.nodes):
    #     max_length_log_ratio = 0.0
    #     if node == 0:
    #         prev_edge_length = 0.0
    #     #else:
    #     for edge_idx, edge in enumerate(nxg.edges(node)):
    #         if edge_idx==0: # initialization assumes that first difference is largest
    #             max_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
    #             tmp_prev_edge_length = nxg.edges[edge]['length_log']
    #         if np.abs(nxg.edges[edge]['length_log']-prev_edge_length) > np.abs(max_length_log_ratio) : 
    #             max_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
    #             tmp_prev_edge_length = nxg.edges[edge]['length_log']
    #     prev_edge_length = tmp_prev_edge_length

    #     if len(nxg.edges(node)) == 0:
    #         nxg.nodes[node]['max_length_log_ratio'] = 0
    #     else:
    #         nxg.nodes[node]['max_length_log_ratio'] = max_length_log_ratio

    # # Minimum Length Ratio (Absolute)
    # for node_ndx, node in enumerate(nxg.nodes):
    #     min_length_log_ratio = 0.0
    #     if node == 0:
    #         prev_edge_length = 0.0
    #         nxg.nodes[node]['min_length_log_ratio'] = 0
    #         # For node[0] run through full loop and find minimum length except the first edge ehich is zero
    #         for edge_idx, edge in enumerate(nxg.edges(node)):
    #             if (edge_idx==0): # first value at position zero is always 0. We dont want this value
    #                 tmp_min = 1000000.0                
    #             if (edge_idx>0 and tmp_min>nxg.edges[edge]['length_log']): # compare to find minimum.
    #                 tmp_min = nxg.edges[edge]['length_log']
    #                 prev_edge_length = tmp_min
    #         continue
        
    #     for edge_idx, edge in enumerate(nxg.edges(node)):
    #         if edge[0]-edge[1] > 0: # if opposite node-direction then skip comparison in this iteration of loop
    #             skipped = True
    #             continue
    #         if (edge_idx==0 or skipped==True): # initialization assumes that first ratio is smallest (same length)
    #             skipped = False
    #             min_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
    #             tmp_prev_edge_length = nxg.edges[edge]['length_log']
    #         if np.abs(nxg.edges[edge]['length_log']-prev_edge_length) < np.abs(min_length_log_ratio) : 
    #             min_length_log_ratio = nxg.edges[edge]['length_log']-prev_edge_length
    #             tmp_prev_edge_length = nxg.edges[edge]['length_log']

    #     prev_edge_length = tmp_prev_edge_length

    #     if len(nxg.edges(node)) == 0:
    #         nxg.nodes[node]['min_length_log_ratio'] = 0
    #     else:
    #         nxg.nodes[node]['min_length_log_ratio'] = min_length_log_ratio
    


def group_labels_features(data_root, data_list, windowing=False):
    data_path = 'data/' # m: comment out
    data_files = [line.rstrip() for line in open(os.path.join(data_root, data_list))]

    # Initialize empty list
    dataset = []

    #print("loading {} files".format(len(data_files))) # m
    for idx, file in enumerate(data_files):
        graph = []
        nxg = nx.read_gpickle(os.path.join(data_root, file))


        # Get the annotated labels
        labels = get_labels(nxg)
        # Get the feature from the file
        features = extract_features(nxg)

        dgl_g  = dgl.from_networkx(nxg)

        # Append the information for batching
        graph.append(dgl_g)
        graph.append(labels)
        graph.append(features)
        dataset.append(graph)

    return dataset


def batch_graphs(data_root, data_list, windowing=False):
    data_files = [line.rstrip() for line in open(os.path.join(data_root, data_list))]

    all_graphs = []
    all_labels = []
    all_features = []

    for file in data_files:
        # Convert the gpickle file to a dgl graph for batching
        nxg = nx.read_gpickle(os.path.join(data_root, file))

        if windowing:
            nxg_list = sliding_window.perform_windowing(nxg)
            for nxg in nxg_list:
                # Get the annotated labels
                labels = get_labels(nxg)
                # Get the feature from the file
                features = extract_features(nxg)

                dgl_g  = dgl.from_networkx(nxg)

                # Append the information for batching
                all_graphs.append(dgl_g)
                all_labels.append(labels)
                all_features.append(features)
        else:
            # Get the annotated labels
            labels = get_labels(nxg)
            # Get the feature from the file
            features = extract_features(nxg)

            dgl_g  = dgl.from_networkx(nxg)

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
    dgl_g  = dgl.from_networkx(nxg)   

    return dgl_g


# def get_features(file):
#     # Read file as networkx graph and retrieve positions + labels
#     nxg = nx.read_gpickle(file)

#     # % % Define some features for the graph
#     # % Normalized positions
#     positions = nx.get_node_attributes(nxg, 'pos')
#     positions = list(positions.values())
#     norm_positions = normalize_positions(positions)
#     norm_positions = torch.FloatTensor(norm_positions)

#     # % Normalized node degree (number of edges connected to node)
#     dgl_g = convert_gpickle_to_dgl_graph(file)
#     norm_degrees = 1. / dgl_g.in_degrees().float().unsqueeze(1)

#     # % Normalized unique identity (entity type [ARC/CRCLE/LINE])
#     id_dict = nx.get_node_attributes(nxg, 'id')
#     ids = list(id_dict.values())
#     for idx, id in enumerate(ids):
#         if id == 4:  # CIRCLE
#             ids[idx] = 0
#         if id == 3:  # ARC
#             ids[idx] = 4
#         if id == 0:
#             ids[idx] = 1
#         if id == 1:
#             ids[idx] = 2
#         if id == 2:
#             ids[idx] = 3

#     # norm_ids = [float(i)/float(max(ids)) for i in ids]
#     norm_ids = [float(i) / 4.0 for i in ids]
#     norm_ids = torch.FloatTensor(np.asarray(norm_ids).reshape(-1, 1))

#     # % DeepWalk
#     dpwlk = DeepWalk(number_walks=4, walk_length=5, representation_size=2)
#     # Create embedding file for given file
#     dpwlk.create_embeddings(nxg, file)
#     # Read the embedding file for given file
#     embedding_feat = dpwlk.read_embeddings(file)

#     # % Combine all features into one tensor
#     #  Compare length of feature vectors before cat
#     if norm_degrees.shape[0] != norm_ids.shape[0] or norm_ids.shape[0] != embedding_feat.shape[0]:
#         print("mismatch in feature vectors: norm_degrees {}, norm_ids {}, embedding_feat {}".format(norm_degrees.shape, norm_ids.shape, embedding_feat.shape))
#     #features = torch.cat((norm_positions, norm_deg, norm_ids, embedding_feat), 1)
#     features = torch.cat((norm_degrees, norm_ids, embedding_feat), 1)

#     return features

def extract_features(nxg):

    # % Extract features for the graph
    # % Normalized positions
    positions = nx.get_node_attributes(nxg, 'pos')
    positions = list(positions.values())
    norm_positions = normalize_positions(positions)
    norm_positions = torch.FloatTensor(norm_positions)

    # % Normalized node degree (number of edges connected to node)
    dgl_g  = dgl.from_networkx(nxg)

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

    norm_ids = torch.FloatTensor(np.asarray(ids).reshape(-1, 1))

    for node in nxg.nodes:            
        nxg.nodes[node]['norm_ids'] = norm_ids[node]
        nxg.nodes[node]['norm_degrees'] = norm_degrees[node]

    # Angles and length features:
    calculate_angles_and_length(nxg)

    extract_max_difference_angle(nxg)
    extract_min_difference_angle(nxg)
    extract_max_loglength_ratio_absolute(nxg)
    extract_min_loglength_ratio_absolute(nxg)
    
    angles = nx.get_node_attributes(nxg, 'angle')
    angles = list(angles.values())
    angles = torch.FloatTensor(np.asarray(angles).reshape(-1, 1))

    lengths = nx.get_node_attributes(nxg, 'length')
    lengths = list(lengths.values())
    lengths = torch.FloatTensor(np.asarray(lengths).reshape(-1, 1))

    min_length = nx.get_node_attributes(nxg, 'min_length_euql')
    min_length = list(min_length.values())
    min_length = torch.FloatTensor(np.asarray(min_length).reshape(-1, 1))

    max_length = nx.get_node_attributes(nxg, 'max_length_euql')
    max_length = list(max_length.values())
    max_length = torch.FloatTensor(np.asarray(max_length).reshape(-1, 1))

    min_length_log = nx.get_node_attributes(nxg, 'min_length_log')
    min_length_log = list(min_length_log.values())
    min_length_log = torch.FloatTensor(np.asarray(min_length_log).reshape(-1, 1))

    max_length_log = nx.get_node_attributes(nxg, 'max_length_log')
    max_length_log = list(max_length_log.values())
    max_length_log = torch.FloatTensor(np.asarray(max_length_log).reshape(-1, 1))

    min_length_log_ratio = nx.get_node_attributes(nxg, 'min_length_log_ratio')
    min_length_log_ratio = list(min_length_log_ratio.values())
    min_length_log_ratio = torch.FloatTensor(np.asarray(min_length_log_ratio).reshape(-1, 1))

    max_length_log_ratio = nx.get_node_attributes(nxg, 'max_length_log_ratio')
    max_length_log_ratio = list(max_length_log_ratio.values())
    #print("   MAX   ", max_length_log_ratio)
    max_length_log_ratio = torch.FloatTensor(np.asarray(max_length_log_ratio).reshape(-1, 1))

    max_angle = nx.get_node_attributes(nxg, 'max_angle')
    max_angle = list(max_angle.values())
    max_angle = torch.FloatTensor(np.asarray(max_angle).reshape(-1, 1))

    min_angle = nx.get_node_attributes(nxg, 'min_angle')
    min_angle = list(min_angle.values())
    min_angle = torch.FloatTensor(np.asarray(min_angle).reshape(-1, 1))

    max_diff_angle = nx.get_node_attributes(nxg, 'max_diff_angle')
    max_diff_angle = list(max_diff_angle.values())
    max_diff_angle = torch.FloatTensor(np.asarray(max_diff_angle).reshape(-1, 1))

    min_diff_angle = nx.get_node_attributes(nxg, 'min_diff_angle')
    min_diff_angle = list(min_diff_angle.values())
    min_diff_angle = torch.FloatTensor(np.asarray(min_diff_angle).reshape(-1, 1))

    norm_ids = nx.get_node_attributes(nxg, 'norm_ids')
    norm_ids = list(norm_ids.values())
    norm_ids = torch.FloatTensor(np.asarray(norm_ids).reshape(-1, 1))

    norm_degrees = nx.get_node_attributes(nxg, 'norm_degrees')
    norm_degrees = list(norm_degrees.values())
    norm_degrees = torch.FloatTensor(np.asarray(norm_degrees).reshape(-1, 1))

    # % DeepWalk
    #dpwlk = DeepWalk(number_walks=4, walk_length=5, representation_size=2)
    # Create embedding file for given file
    #dpwlk.create_embeddings(nxg, file)
    # Read the embedding file for given file
    #embedding_feat = dpwlk.read_embeddings(file)

    # % Combine all features into one tensor
    #features = torch.cat((norm_positions, norm_deg, norm_ids, embedding_feat), 1)
    features = torch.cat((norm_degrees, min_diff_angle, max_diff_angle, max_length_log_ratio, min_length_log_ratio), 1)
    # norm_degrees, max_diff_angle, min_angle, max_length, min_length
    # features = torch.cat((norm_degrees, angles, lengths), 1)

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
