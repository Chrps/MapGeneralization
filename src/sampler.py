import random
import networkx as nx
import numpy as np
from numba import jit
import time
import math
import sys

# TODO should not be a class
class GraphSampler:

    @staticmethod
    @jit
    def incremental_farthest_search(points, k):
        # TODO: convert to numpy
        remaining_points = points[:]
        solution_set = []
        solution_idxs = []
        solution_set.append(remaining_points.pop(random.randint(0, len(remaining_points) - 1)))
        for _ in range(k - 1):
            distances = [abs(p - solution_set[0]) for p in remaining_points]
            for i, p in enumerate(remaining_points):
                for j, s in enumerate(solution_set):
                    distances[i] = min(distances[i], abs(p - s))
            solution_set.append(remaining_points.pop(distances.index(max(distances))))
        for idx in solution_set:
            solution_idxs.append(points.index(idx))
        return solution_idxs

    @staticmethod
    def subgraph_w_retained_connections(nxg, selected_points):
        all_points = list(range(0, nxg.number_of_nodes()))
        remove_list = np.setdiff1d(all_points, selected_points, True).tolist()
        for point in remove_list:
            neighbors = nxg.neighbors(point)
            for neighbor in neighbors:
                for own_neighbor in neighbors:
                    if neighbor not in nxg.neighbors(own_neighbor):
                        nxg.add_edge(neighbor, own_neighbor)
            nxg.remove_node(point)
        return nxg

    def down_sampler(self, nxg, output_size, nr_furthest_search):
        # Get node positions
        positions = nx.get_node_attributes(nxg, 'pos')

        # Convert node positions to complex number format for faster computation time
        complex_positions = []
        for position in list(positions.values()):
            complex_positions.append(complex(*position))
        # Select n points using farthest search
        # TODO: Use Voronoi method instead
        start_time = time.time()
        selected_points = self.incremental_farthest_search(complex_positions, nr_furthest_search)
        print(time.time() - start_time)
        # Select the rest of the points using random sampling
        full_selected_points = list(range(0, nxg.number_of_nodes()))
        for idx in selected_points:
            full_selected_points.remove(idx)

        full_selected_points = random.sample(full_selected_points, output_size - nr_furthest_search)
        full_selected_points = selected_points + full_selected_points

        nxg_sub = self.subgraph_w_retained_connections(nxg, list(full_selected_points))

        return nxg_sub

    @staticmethod
    def up_sampler(nxg, needed_size, labels=False):
        original_nr_nodes = nxg.number_of_nodes()
        for i in range(needed_size - original_nr_nodes):
            edge_lengths = []
            edge_list = list(nxg.edges)
            # Calculate all edge lengths
            # TODO: Can properly be optimized so this is not needed for each loop
            for edge in edge_list:
                pos1 = nxg._node[edge[0]]['pos']
                pos2 = nxg._node[edge[1]]['pos']
                length = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                edge_lengths.append(length)

            # Determine longest edge
            max_index = np.asscalar(np.argmax(edge_lengths))
            edge = edge_list[max_index]

            # Interpolate its middle point
            pos1 = nxg._node[edge[0]]['pos']
            pos2 = nxg._node[edge[1]]['pos']
            mid_pos = ((pos2[0] + pos1[0]) / 2, (pos2[1] + pos1[1]) / 2)

            # Add middle point as node
            if labels:
                label1 = nxg._node[edge[0]]['label']
                label2 = nxg._node[edge[1]]['label']
                label = 0
                if label1 == label2 == 1:
                    label = 1
                nxg.add_node(original_nr_nodes + i + 1, pos=tuple(mid_pos), label=label)
            else:
                nxg.add_node(original_nr_nodes + i + 1, pos=tuple(mid_pos))


            # Create edge between new node and old nodes
            nxg.add_edge(original_nr_nodes + i + 1, edge[0])
            nxg.add_edge(original_nr_nodes + i + 1, edge[1])

            # Remove old edge
            nxg.remove_edge(edge[0], edge[1])
        return nxg
