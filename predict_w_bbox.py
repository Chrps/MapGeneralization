import torch
import src.graph_utils as graph_utils
import src.models as models
import matplotlib.pyplot as plt
import os
import networkx as nx
import argparse
# import math
import numpy as np
from itertools import count
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data/public')
parser.add_argument('--predict-path', type=str, default='test_list.txt')
parser.add_argument('--model_name', type=str, default='gat_20-05-22_19-04-14_final')

args = parser.parse_args()


def load_model_txt(model_name):
    model_txt = 'trained_models/' + model_name + '/predict_info.txt'
    data = [line.rstrip() for line in open(model_txt)]

    # network train on ()
    net = data[0]

    # Number of features per node
    n_features = int(data[1])

    # Number of classes
    n_classes = int(data[2])

    return net, n_features, n_classes


def draw(results, ax, nx_G, positions):
    cls1color = 'r'
    cls2color = 'b'

    colors = []
    for v in range(len(nx_G)):
        cls = results[v]
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    nx.draw_networkx(nx_G.to_undirected(), positions, node_color=colors,
            with_labels=False, node_size=10, ax=ax)


def draw_inst(nx_G, ax, positions):

    groups = set(nx.get_node_attributes(nx_G, 'instance').values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = nx_G.nodes()
    colors = [mapping[nx_G.nodes[n]['instance']] for n in nodes]

    ax.axis('off')
    nx.draw_networkx(nx_G.to_undirected(), positions, node_color=colors,
                     with_labels=False, node_size=10, ax=ax, cmap=plt.jet())



def post_processing(nxg_, predictions_):

    # Graph morphology closing
    predictions_alt = []
    for node in nxg_.nodes:
        nr_non_door_nodes = 0
        nr_door_nodes = 0

        # Get 2-order proximity neighbors
        all_neighbors = []
        neighbors = list(nxg_.neighbors(node))
        for neighbor in neighbors:
            neighbors2 = list(nxg_.neighbors(neighbor))
            all_neighbors.append(neighbors2)
        all_neighbors.append(list(neighbors))
        all_neighbors = [item for sublist in all_neighbors for item in sublist]
        all_neighbors = set(all_neighbors)
        if node in all_neighbors:
            all_neighbors.remove(node)

        for neighbor in all_neighbors:
            neighbor_class = predictions_[neighbor]
            if neighbor_class == 0:
                nr_non_door_nodes += 1
            if neighbor_class == 1:
                nr_door_nodes += 1

        # If the number of door nodes in the 2-order proximity is higher than
        # the number of non-door nodes the current node is set to be a door node
        if nr_door_nodes >= nr_non_door_nodes:
            predictions_alt.append(1)
        else:
            predictions_alt.append(predictions_[node])

    return predictions_alt


def instancing(nxg_, predictions, instance=1):
    door_indices = []
    for idx, prediction in enumerate(predictions):
        if prediction == instance:
            door_indices.append(idx)
    sub_nxg = nxg_.subgraph(door_indices)
    return sub_nxg


def reject_outliers_IQR(dataIn, lower_factor=2.0, higher_factor=6.0):
    q25, q75 = np.percentile(dataIn, 25), np.percentile(dataIn, 75)
    iqr = q75 - q25
    cut_off_upper = iqr * higher_factor
    cut_off_lower = iqr * lower_factor
    lower, upper = q25 - cut_off_lower, q75 + cut_off_upper
    inliers = []
    for idx, data in enumerate(dataIn):
        if (data > lower) and (data < upper):
            inliers.append(idx)
    return inliers


def reject_outliers_hardcoded(areas, lengths, heights, ratios):
    inliers = []
    for idx, data in enumerate(zip(areas, lengths, heights, ratios)):
        area, length, height, ratio = data
        if ratio > 0.3 and length < 3000 and height < 3000:
            inliers.append(idx)
    return inliers


def bounding_box_params(points):
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    width = top_right_x - bot_left_x
    height = top_right_y - bot_left_y

    width_height_list = [width, height]
    max_box = max(width_height_list)
    min_box = min(width_height_list)
    ratio = min_box/max_box

    return width * height, height, width, ratio


def bounding_box_trbl(points):
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    return [top_right_x, top_right_y, bot_left_x, bot_left_y]

def determine_bboxes(graphs):
    bboxes = []
    list_gen_bbox_graphs = []
    for graph in graphs:
        tmp_positions = nx.get_node_attributes(graph, 'pos')
        tmp_positions = np.array(list(tmp_positions.values()))
        bbox = bounding_box_trbl(tmp_positions)
        bboxes.append(bbox)
        pos0 = [bbox[2], bbox[1]] # tl
        pos1 = [bbox[0], bbox[1]] # tr
        pos2 = [bbox[0], bbox[3]] # br
        pos3 = [bbox[2], bbox[3]] # bl
        gen_door_graph = nx.Graph()
        gen_door_graph.add_node(0, pos=pos0)
        gen_door_graph.add_node(1, pos=pos1)
        gen_door_graph.add_node(2, pos=pos2)
        gen_door_graph.add_node(3, pos=pos3)
        gen_door_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        list_gen_bbox_graphs.append(gen_door_graph.copy())
        gen_door_graph.clear()
    return bboxes, list_gen_bbox_graphs


def predict(data_path, predict_path, model_name):
    # Read the parameters of the trained model
    net, n_features, n_classes = load_model_txt(model_name)

    # Load the trained model
    trained_net, config = models.get_model_and_config(net)
    model = trained_net(n_features,
                        n_classes,
                        *config['extra_args'])
    model_path = 'trained_models/' + model_name + '/model.pth'
    model.load_state_dict(torch.load(model_path))
    print(model)

    # Get the list of files for prediction
    pred_files = [os.path.join(data_path, line.rstrip()) for line in open(os.path.join(data_path, predict_path))]
    for file in pred_files:
        # Convert the gpickle file to a dgl graph
        dgl_g = graph_utils.convert_gpickle_to_dgl_graph(file)
        # Get the features from the given graph
        nxg = nx.read_gpickle(file)
        features = graph_utils.extract_features(nxg)

        model.eval()
        with torch.no_grad():
            logits = model(dgl_g, features)
            _, predictions = torch.max(logits, dim=1)
            predictions = predictions.numpy()

        # Get positions
        nxg = nx.read_gpickle(file)
        positions = nx.get_node_attributes(nxg, 'pos')
        positions = list(positions.values())

        # Plot graph
        ''''fig2 = plt.figure(dpi=150)
        fig2.clf()
        ax = fig2.subplots()
        inst_predictions = [0] * nxg.number_of_nodes()
        draw(inst_predictions, ax, nxg, positions)'''

        # Plot graph with predictions
        fig1 = plt.figure(dpi=150)
        fig1.clf()
        ax = fig1.subplots()
        draw(predictions, ax, nxg, positions)

        # Get labels
        labels = nx.get_node_attributes(nxg, 'label')
        labels = np.array(list(labels.values()))

        # Plot annotated graph
        '''fig2 = plt.figure(dpi=150)
        fig2.clf()
        ax = fig2.subplots()
        draw(labels, ax, nxg, positions)'''

        # Perform graph morphology closing
        predictions_alt = predictions
        # predictions_alt = post_processing(nxg, predictions)

        # Extract door nodes
        sub_nxg = instancing(nxg, predictions_alt)

        # Separate disjoint graphs (instancing)
        disjoint_sub_graphs = []
        for c in nx.connected_components(sub_nxg):
            disjoint_sub_graphs.append(sub_nxg.subgraph(c))

        clustered_disjoint_sub_graphs = []
        for graph in disjoint_sub_graphs:
            sub_positions = nx.get_node_attributes(graph, 'pos')
            sub_positions = np.array(list(sub_positions.values()))
            clustering = DBSCAN(eps=1100, min_samples=1).fit(sub_positions)
            cluster_labels = clustering.labels_
            graph_keys = list(graph._NODE_OK.nodes)
            for cluster_label in list(set(cluster_labels)):
                indices = []
                for idx, label in enumerate(cluster_labels):
                    if label == cluster_label:
                        indices.append(graph_keys[idx])
                sub_graph = graph.subgraph(indices)
                clustered_disjoint_sub_graphs.append(sub_graph)

        # Remove graphs not meeting conditions
        min_nr_nodes = 8
        selected_graphs = []
        area_list = []
        width_list = []
        height_list = []
        ratio_list = []

        for disjoint_sub_graph in clustered_disjoint_sub_graphs:
            if disjoint_sub_graph.number_of_nodes() > min_nr_nodes:
                selected_graphs.append(disjoint_sub_graph)
                tmp_positions = nx.get_node_attributes(disjoint_sub_graph, 'pos')
                tmp_positions = np.array(list(tmp_positions.values()))
                area, width, height, ratio = bounding_box_params(tmp_positions)
                area_list.append(area)
                width_list.append(width)
                height_list.append(height)
                ratio_list.append(ratio)

        seleted_graphs_joined = nx.Graph()

        for idx, graph in enumerate(selected_graphs):
            nx.set_node_attributes(graph, [], 'instance')
            for node in graph.nodes:
                graph.nodes[node]['instance'] = idx
            seleted_graphs_joined = nx.compose(seleted_graphs_joined, graph)

        inliers = reject_outliers_hardcoded(area_list, width_list, height_list, ratio_list)
        selected_graphs = [selected_graphs[i] for i in inliers]

        print('Numer of doors: %d' % len(selected_graphs))

        seleted_graphs_joined = nx.Graph()

        for idx, graph in enumerate(selected_graphs):
            nx.set_node_attributes(graph, [], 'instance')
            for node in graph.nodes:
                graph.nodes[node]['instance'] = idx
            seleted_graphs_joined = nx.compose(seleted_graphs_joined, graph)

        # Determine bbox
        list_bboxes, list_gen_bboxes = determine_bboxes(selected_graphs)

        # Plot graph with generalized doors
        pos = nx.get_node_attributes(nxg, 'pos')
        fig5 = plt.figure(dpi=150)
        fig5.clf()
        ax = fig5.subplots()
        nx.draw(seleted_graphs_joined, pos, with_labels=False, node_size=10, ax=ax, node_color='b')

        nxg_copy = nxg.copy()

        bbox_and_org_graph = nx.Graph()
        bbox_and_org_graph = nx.compose(bbox_and_org_graph, nxg_copy)
        bbox_graph = nx.Graph()
        for g_idx, g in enumerate(list_gen_bboxes):
            gen_pos = nx.get_node_attributes(g, 'pos')
            nx.draw(g, gen_pos, with_labels=False, node_color='g', node_size=30, ax=ax)
            nx.draw_networkx_edges(g, gen_pos, width=2, alpha=0.8, edge_color='g')
            g = nx.convert_node_labels_to_integers(g, first_label=4 * g_idx)
            bbox_graph = nx.compose(bbox_graph, g)
            bbox_graph = nx.convert_node_labels_to_integers(bbox_graph,
                                                        first_label=bbox_and_org_graph.number_of_nodes())
        bbox_and_org_graph = nx.compose(bbox_and_org_graph, bbox_graph)

        fig6 = plt.figure(dpi=150)
        fig6.clf()
        ax = fig6.subplots()
        bbox_and_org_graph_pos = nx.get_node_attributes(bbox_and_org_graph, 'pos')
        nx.draw(bbox_and_org_graph, bbox_and_org_graph_pos, with_labels=False, node_size=10, ax=ax, node_color='r')

        fig7 = plt.figure(dpi=150)
        fig7.clf()
        ax = fig7.subplots()
        bbox_graph_pos = nx.get_node_attributes(bbox_graph, 'pos')
        nx.draw(bbox_graph, bbox_graph_pos, with_labels=False, node_size=10, ax=ax,
                node_color='g')

        # Save res graph
        base = os.path.basename(file)
        file_name = os.path.splitext(base)[0]
        # nx.write_gpickle(door_graph, 'C:/Users/Chrips/Aalborg Universitet/Frederik Myrup Thiesson - data/door_graphs/' + file_name + '_door_graph.gpickle')
        # nx.write_gpickle(door_generalized_graph, 'C:/Users/Chrips/Aalborg Universitet/Frederik Myrup Thiesson - data/door_generalized_graphs/' + file_name + '_door_generalized_graph.gpickle')

        # Plot graph with instances
        fig4 = plt.figure(dpi=150)
        fig4.clf()
        ax = fig4.subplots()
        # ax.axis('equal')
        draw_inst(seleted_graphs_joined, ax, positions)

        plt.show()


if __name__ == '__main__':
    data_path = args.data_path
    predict_path = args.predict_path
    model_name = args.model_name

    predict(data_path, predict_path, model_name)
