import torch
import src.graph_utils as graph_utils
import src.models as models
import matplotlib.pyplot as plt
import os
import networkx as nx
import argparse
import math
import numpy as np
from itertools import count
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data/Public')
parser.add_argument('--predict-path', type=str, default='test_list.txt')
parser.add_argument('--model_name', type=str, default='gat_20-09-29_15-55-58')

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
                     with_labels=False, node_size=10, ax=ax, cmap=plt.cm.jet)



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


def find_furthest_node(G, source_node):
    furthest_node = None
    max_furthest_node_length = 0
    for idx, node in enumerate(G.nodes):
        max_path_length = 0
        try:
            paths = nx.all_shortest_paths(G, source=source_node, target=node)
            for path in map(nx.utils.pairwise, paths):
                path_length = 0
                for edge in path:
                    pos1 = G._node[edge[0]]['pos']
                    pos2 = G._node[edge[1]]['pos']
                    path_length += np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                    if path_length > max_path_length:
                        max_path_length = path_length
            if max_path_length > max_furthest_node_length:
                max_furthest_node_length = max_path_length
                furthest_node = node
        except:
            pass
    return furthest_node


def rotate_line_segment(a, b, angle):
    midpoint = [
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2
    ]

    a_mid = [
        a[0] - midpoint[0],
        a[1] - midpoint[1]
    ]
    b_mid = [
        b[0] - midpoint[0],
        b[1] - midpoint[1]
    ]

    a_rotated = [
        np.cos(angle) * a_mid[0] - np.sin(angle) * a_mid[1],
        np.sin(angle) * a_mid[0] + np.cos(angle) * a_mid[1]
    ]
    b_rotated = [
        np.cos(angle) * b_mid[0] - np.sin(angle) * b_mid[1],
        np.sin(angle) * b_mid[0] + np.cos(angle) * b_mid[1]
    ]

    a_rotated[0] = a_rotated[0] + midpoint[0]
    a_rotated[1] = a_rotated[1] + midpoint[1]
    b_rotated[0] = b_rotated[0] + midpoint[0]
    b_rotated[1] = b_rotated[1] + midpoint[1]

    return [a_rotated, b_rotated]


def find_longest_edge_center(G):
    max_edge_length = 0
    max_edge_node_1 = 0
    max_edge_node_2 = 0
    for edge in list(G.edges):
        node_1 = edge[0]
        node_2 = edge[1]
        pos1 = G._node[edge[0]]['pos']
        pos2 = G._node[edge[1]]['pos']
        edge_length = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
        if edge_length > max_edge_length:
            max_edge_length = edge_length
            max_edge_node_1 = node_1
            max_edge_node_2 = node_2
    max_pos1 = G._node[max_edge_node_1]['pos']
    max_pos2 = G._node[max_edge_node_2]['pos']
    center_point = [(max_pos1[0] + max_pos2[0])/2.0,
                    (max_pos1[1] + max_pos2[1])/2.0]
    g_copy = G.copy()
    g_copy.remove_edge(max_edge_node_1, max_edge_node_2)
    g_copy.add_node(0, pos=center_point)
    g_copy.add_edges_from([(0, max_edge_node_1), (0, max_edge_node_2)])
    furthest_from_center = find_furthest_node(g_copy, 0)
    return 0, furthest_from_center, g_copy


def generalize_doors_new(disjoint_sub_graphs):
    list_gen_door_graphs = []
    for subgraph in disjoint_sub_graphs:
        # Get degree to see if cyclical
        all_degrees = []
        for node in subgraph.nodes:
            neighbors = [n for n in subgraph.neighbors(node)]
            try:
                neighbors.remove(node)
            except ValueError:
                pass
            degree = len(neighbors)
            all_degrees.append(degree)

        bCyclical = all(x == 2 for x in all_degrees)
        if bCyclical == True:
            center_node, furthest_from_center, new_graph = find_longest_edge_center(subgraph)
            furthest_node_1 = center_node
            furthest_node_2 = furthest_from_center
            subgraph = new_graph
        else:
            source_node = list(subgraph.nodes)[0]
            furthest_node_1 = find_furthest_node(subgraph, source_node)
            furthest_node_2 = find_furthest_node(subgraph, furthest_node_1)
        orig_pos_1 = subgraph.nodes[furthest_node_1]['pos']
        orig_pos_2 = subgraph.nodes[furthest_node_2]['pos']
        x1 = orig_pos_1[0]
        y1 = orig_pos_1[1]
        x2 = orig_pos_2[0]
        y2 = orig_pos_2[1]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        angle_degrees = np.degrees(angle) + 180
        alt_angle = None
        if 0 < angle_degrees < 20:
            alt_angle = -angle
        elif 70 < angle_degrees < 90:
            alt_angle = np.radians(90) - angle
        elif 90 < angle_degrees < 110:
            alt_angle = np.radians(90) - angle
        elif 160 < angle_degrees < 180:
            alt_angle = np.radians(180) - angle
        elif 180 < angle_degrees < 200:
            alt_angle = np.radians(180) - angle
        elif 250 < angle_degrees < 270:
            alt_angle = np.radians(270) - angle
        elif 270 < angle_degrees < 290:
            alt_angle = np.radians(270) - angle
        elif 340 < angle_degrees < 360:
            alt_angle = np.radians(360) - angle
        if alt_angle is not None:
            rot_pos_1, rot_pos_2 = rotate_line_segment(orig_pos_1, orig_pos_2, alt_angle)
            x1 = rot_pos_1[0]
            y1 = rot_pos_1[1]
            x2 = rot_pos_2[0]
            y2 = rot_pos_2[1]
            dx = x2 - x1
            dy = y2 - y1

        dx_1 = x2 - x1
        dy_1 = y2 - y1
        regular_vec1 = normalize([dx_1, dy_1]) * 100
        x1 = x1 - regular_vec1[0]
        y1 = y1 - regular_vec1[1]

        dx_2 = x1 - x2
        dy_2 = y1 - y2
        regular_vec2 = normalize([dx_2, dy_2]) * 100
        x2 = x2 - regular_vec2[0]
        y2 = y2 - regular_vec2[1]

        norm1 = [-dy, dx]
        norm2 = [dy, -dx]
        mag1 = math.sqrt(norm1[0]**2 + norm1[1]**2)
        mag2 = math.sqrt(norm2[0] ** 2 + norm2[1] ** 2)
        scale_factor = 100
        pos0 = [scale_factor*(norm1[0]/mag1) + x1, scale_factor*(norm1[1]/mag1) + y1]
        pos1 = [scale_factor*(norm2[0]/mag2) + x1, scale_factor*(norm2[1]/mag2) + y1]
        pos2 = [scale_factor*(norm1[0]/mag1) + x2, scale_factor*(norm1[1]/mag1) + y2]
        pos3 = [scale_factor*(norm2[0]/mag2) + x2, scale_factor*(norm2[1]/mag2) + y2]
        gen_door_graph = nx.Graph()
        gen_door_graph.add_node(0, pos=pos0)
        gen_door_graph.add_node(1, pos=pos1)
        gen_door_graph.add_node(2, pos=pos2)
        gen_door_graph.add_node(3, pos=pos3)
        gen_door_graph.add_edges_from([(0, 1), (1, 3), (3, 2), (2, 0)])
        list_gen_door_graphs.append(gen_door_graph.copy())
        gen_door_graph.clear()
    return list_gen_door_graphs


def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def normalize(a):
    a = np.array(a)
    return a/np.linalg.norm(a)


def generalize_doors(disjoint_sub_graphs):
    list_gen_door_graphs = []
    bIsSubgraphDoor = True
    for subgraph in disjoint_sub_graphs:
        # % Pick a node (hopefully middle)
        nodes_list = list(subgraph.nodes)
        try:
            node_number = nodes_list[5]
            bIsSubgraphDoor = True
        except IndexError:
            bIsSubgraphDoor = False

        if bIsSubgraphDoor:

            # % Get list of connecting edges to the particular node
            edge_neighbor_list = list(nx.dfs_edges(subgraph, source=node_number))

            # % Now sort the list of edges from left and right neighbors
            right_neighbors = []
            left_neighbors = []
            # First edge always includes the node in question
            right_neighbors.append(edge_neighbor_list[0])
            edge_neighbor_list.remove(edge_neighbor_list[0])

            for edge in edge_neighbor_list:
                if edge[0] == node_number:
                    break
                right_neighbors.append(edge)

            remaining_edges = [x for x in edge_neighbor_list if x not in right_neighbors]
            left_neighbors = remaining_edges

            # % Now you have the two endpoints of the subgraph
            try:
                left_most_node = left_neighbors[-1][1]
                right_most_node = right_neighbors[-1][1]
            except IndexError:
                # Then we have accidently picked not the middle node
                if len(right_neighbors) != 0:
                    left_most_node = node_number
                    right_most_node = right_neighbors[-1][1]
                else:
                    left_most_node = left_neighbors[-1][1]
                    right_most_node = node_number

            # % Generate new graph for generalized door
            gen_door_graph = nx.Graph()

            orig_pos_1 = subgraph.nodes[left_most_node]['pos']
            orig_pos_2 = subgraph.nodes[right_most_node]['pos']
            x1 = orig_pos_1[0]
            y1 = orig_pos_1[1]
            x2 = orig_pos_2[0]
            y2 = orig_pos_2[1]


            norm_perp_1 = perpendicular(normalize([x1, y1]))
            norm_perp_2 = perpendicular(normalize([x2, y2]))
            gen_door_graph.add_node(0, pos=norm_perp_1)
            gen_door_graph.add_node(1, pos=-norm_perp_1)
            gen_door_graph.add_node(2, pos=norm_perp_2)
            gen_door_graph.add_node(3, pos=-norm_perp_2)
            gen_door_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

            # % Append new graph to be plotted later on
            list_gen_door_graphs.append(gen_door_graph.copy())
            gen_door_graph.clear()

    return list_gen_door_graphs


def predict(data_path, predict_path, model_name, show=True):
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
        features = graph_utils.chris_get_features(nxg)

        model.eval()
        with torch.no_grad():
            logits = model(dgl_g, features)
            _, predictions = torch.max(logits, dim=1)
            predictions = predictions.numpy()

        # Get positions
        nxg = nx.read_gpickle(file)
        positions = nx.get_node_attributes(nxg, 'pos')
        positions = list(positions.values())

        if show:
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

        if show:
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

        # Generalizing
        list_gen_doors = generalize_doors_new(selected_graphs)

        # Get a graph where door instance have been deleted
        nondoor_graph = nxg.copy()
        door_nodes_list = seleted_graphs_joined.nodes
        nondoor_graph.remove_nodes_from(door_nodes_list)

        if show:
            # Plot graph with generalized doors
            pos = nx.get_node_attributes(nxg, 'pos')
            fig5 = plt.figure(dpi=150)
            fig5.clf()
            ax = fig5.subplots()
            nx.draw(nondoor_graph, pos, with_labels=False, node_size=10, ax=ax, node_color='b')

            door_generalized_graph = nx.Graph()
            nondoor_graph = nx.convert_node_labels_to_integers(nondoor_graph)
            door_generalized_graph = nx.compose(door_generalized_graph, nondoor_graph)
            door_graph = nx.Graph()

            for g_idx, g in enumerate(list_gen_doors):
                gen_pos = nx.get_node_attributes(g, 'pos')
                nx.draw(g, gen_pos, with_labels=False, node_color='g', node_size=30, ax=ax)
                nx.draw_networkx_edges(g, gen_pos, width=2, alpha=0.8, edge_color='g')
                g = nx.convert_node_labels_to_integers(g, first_label=4*g_idx)
                door_graph = nx.compose(door_graph, g)

            door_graph = nx.convert_node_labels_to_integers(door_graph, first_label=door_generalized_graph.number_of_nodes())
            door_generalized_graph = nx.compose(door_generalized_graph, door_graph)

            # Save res graph
            base = os.path.basename(file)
            file_name = os.path.splitext(base)[0]
            #nx.write_gpickle(door_graph, 'C:/Users/Chrips/Aalborg Universitet/Frederik Myrup Thiesson - data/door_graphs/' + file_name + '_door_graph.gpickle')
            #nx.write_gpickle(door_generalized_graph, 'C:/Users/Chrips/Aalborg Universitet/Frederik Myrup Thiesson - data/door_generalized_graphs/' + file_name + '_door_generalized_graph.gpickle')

            fig6 = plt.figure(dpi=150)
            fig6.clf()
            ax = fig6.subplots()
            door_generalized_graph_pos = nx.get_node_attributes(door_generalized_graph, 'pos')
            nx.draw(door_generalized_graph, door_generalized_graph_pos, with_labels=False, node_size=10, ax=ax, node_color='r')

            #door_graph = nx.convert_node_labels_to_integers(door_graph)
            fig7 = plt.figure(dpi=150)
            fig7.clf()
            ax = fig7.subplots()
            door_graph_pos = nx.get_node_attributes(door_graph, 'pos')
            nx.draw(door_graph, door_graph_pos, with_labels=False, node_size=10, ax=ax,
                    node_color='g')

            # Plot graph with instances
            fig4 = plt.figure(dpi=150)
            fig4.clf()
            ax = fig4.subplots()
            #ax.axis('equal')
            draw_inst(seleted_graphs_joined, ax, positions)

            plt.show()


if __name__ == '__main__':
    data_path = args.data_path
    predict_path = args.predict_path
    model_name = args.model_name

    predict(data_path, predict_path, model_name)
