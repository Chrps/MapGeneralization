'''
Utility file for extracting the data (nodes and edges) from .dxf AutoCAD Files
'''

import ezdxf
from ezdxf.groupby import groupby
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect
import matplotlib
matplotlib.use('TKAgg')
import math
import networkx as nx

class Node:
    '''Class for keeping track of nodes in network'''
    def __init__(self, id, coordinate):
        self.id = id
        self.coordinate = coordinate
        self.connected_to = []

class DxfReader:
    """
    Class representing...
    """
    def __init__(self, dxf_file_path):
        extension = os.path.splitext(dxf_file_path)[1]
        if extension != '.dxf':
            raise Exception('File given is not of DXF type (.dxf)')
        else:
            self.dxf_path = dxf_file_path
            print('Loading ' + dxf_file_path + '. May take a while!')
            self.doc = ezdxf.readfile(self.dxf_path)
            self.msp = self.doc.modelspace()
            self.line_data = []
            self.point_data = []
            self.unrecognized_types = []
            self.arc_data = []

    def is_hidden(self, entity):
        layer_name = entity.dxf.layer
        layer = self.doc.layers.get(layer_name)
        if layer.is_off() or layer.is_frozen() or not entity.is_alive or not layer.dxf.plot == 1 \
                or not entity.is_alive or entity.dxf.invisible == 1:
        #if layer.is_on() and layer.dxf.plot == 1 and layer.is_alive and not layer.is_frozen():
            return True
        else:
            return False

    def entity2line(self, e):
        line_out = []
        if e.dxftype() is 'LINE':
            point1 = e.dxf.start
            point2 = e.dxf.end
            if not self.is_hidden(e):
                # Neglect the z-value as expected to be 0
                line = [point1[:2], point2[:2]]
                line_out.append(line)
        elif e.dxftype() is 'LWPOLYLINE':
            if not self.is_hidden(e):
                print('Linetype:', e.dxf.linetype)
                lwpolyline = []
                with e.points() as points:
                    # points is a list of points with the format = (x, y, [start_width, [end_width, [bulge]]])
                    for point in points:
                        # I only want the x,y coordinates
                        lwpolyline.append(point[:2])
                    if e.closed:
                        lwpolyline.append(lwpolyline[0])
                line_out.append(lwpolyline)
        elif e.dxftype() is 'POLYLINE':
            # POLYLINE points are saved as a list of vertices
            if not self.is_hidden(e):
                polyline = []
                vertices = e.vertices
                for vertex in vertices:
                    point = vertex.dxf.location[:2]
                    polyline.append(point)
                if e.is_closed:
                    polyline.append(polyline[0])
                line_out.append(polyline)
        elif e.dxftype() is 'HATCH':
            # TODO extract pattern!
            if not self.is_hidden(e):
                for path in e.paths:
                    if path.PATH_TYPE is 'PolylinePath':
                        polyline = []
                        for vertex in path.vertices:
                            point = list(vertex[:2])
                            polyline.append(point)
                        line_out.append(polyline)
                    elif path.PATH_TYPE is 'EdgePath':
                        for edge in path.edges:
                            if edge.EDGE_TYPE is 'LineEdge':
                                point1 = edge.start
                                point2 = edge.end
                                line = [point1[:2], point2[:2]]
                                line_out.append(line)
                            else:
                                print("unrecognized edge type: " + str(edge.EDGE_TYPE))
                    else:
                        print("unrecognized path type: " + str(path.PATH_TYPE))
        elif e.dxftype() is 'ARC':
            if not self.is_hidden(e):
                #print('ARC:', e.dxf.linetype)
                center = list(e.dxf.center.vec2)
                start_angle = e.dxf.start_angle
                end_angle = e.dxf.end_angle
                radius = e.dxf.radius
                delta_angle = end_angle - start_angle
                step = start_angle
                for i in range(int(delta_angle) + 1):
                    point = [center[0] + (radius * math.cos(math.radians(step))),
                             center[1] + (radius * math.sin(math.radians(step)))]
                    if i == 0:
                        prev_point = point
                    else:
                        curr_point = point
                        line = [prev_point, curr_point]
                        prev_point = curr_point
                        line_out.append(line)
                    step += 1
        elif e.dxftype() is 'CIRCLE':
            if not self.is_hidden(e):
                #print('Circle:', e.dxf.linetype)
                center = list(e.dxf.center.vec2)
                radius = e.dxf.radius
                step = 0
                for i in range(360 + 1):
                    point = [center[0] + (radius * math.cos(math.radians(step))),
                             center[1] + (radius * math.sin(math.radians(step)))]
                    if i == 0:
                        prev_point = point
                    else:
                        curr_point = point
                        line = [prev_point, curr_point]
                        prev_point = curr_point
                        line_out.append(line)
                    step += 1
        elif e.dxftype() is 'POINT':
            if not self.is_hidden(e):
                # TODO finish and plot
                self.point_data.append(list(e.dxf.location[:2]))
        elif e.dxftype() is 'INSERT':
            if not self.is_hidden(e):
                block = self.doc.blocks[e.dxf.name]
                if block.is_alive:
                    for b_e in block:
                        layer_name = b_e.dxf.layer
                        layer = self.doc.layers.get(layer_name)
                        #print("Layer", layer.dxf.linetype)
                        lines = self.entity2line(b_e)
                        for line in lines:
                            line_out.append(line)

        else:
            if not self.unrecognized_types:
                self.unrecognized_types.append([e.dxftype(), 1])
            else:
                if e.dxftype() in list(zip(*self.unrecognized_types))[0]:
                    self.unrecognized_types[list(zip(*self.unrecognized_types))[0].index(e.dxftype())][1] += 1
                else:
                    self.unrecognized_types.append([e.dxftype(), 1])
        return line_out

    def extract_data(self):
        for e in self.msp:
            lines = self.entity2line(e)
            if lines:
                if e.dxftype() is 'INSERT':
                    rotation = e.dxf.rotation
                    x_scale = e.dxf.xscale
                    y_scale = e.dxf.yscale
                    position = e.dxf.insert[:2]
                    for line in lines:
                        t_line = []
                        for point in line:
                            l_point = list(point)
                            # Scale
                            l_point[0] = l_point[0] * x_scale
                            l_point[1] = l_point[1] * y_scale
                            # Rotate
                            temp_point_x = l_point[0]
                            temp_point_y = l_point[1]
                            l_point[0] = temp_point_x * math.cos(math.radians(rotation)) - \
                                         temp_point_y * math.sin(math.radians(rotation))
                            l_point[1] = temp_point_y * math.cos(math.radians(rotation)) + \
                                         temp_point_x * math.sin(math.radians(rotation))
                            # Translate
                            l_point[0] += position[0]
                            l_point[1] += position[1]
                            t_line.append(tuple(l_point))
                        self.line_data.append(t_line)
                else:
                    for line in lines:
                        self.line_data.append(line)
                        pass
        for unrecognized_type in self.unrecognized_types:
            print("Type " + str(unrecognized_type[0]) + " not recognized. Nr. of instances: "
                  + str(unrecognized_type[1]))
        return self.line_data

    def plot_data(self):
        w, h = figaspect(5 / 3)
        lc = LineCollection(self.line_data, linewidths=1)
        fig, ax = plt.subplots(figsize=(w, h))
        ax.add_collection(lc)
        ax.autoscale()
        plt.show()

    def convert_data_to_graph(self, visualize=True):
        # Create network graph
        G = nx.Graph()
        list_of_nodes = []
        current_id = 0

        for entity in self.line_data:
            connectivity_list = []
            for point in entity:
                point = tuple([int(_) for _ in point])
                # If we are beginning then list is empty (i.e. no nodes yet)
                if not list_of_nodes:
                    list_of_nodes.append(Node(current_id, point))
                    connectivity_list.append(current_id)
                    current_id += 1
                else:
                    bFoundSameNode = False
                    for node in list_of_nodes:
                        # Check if the point in question is the same as a node
                        if point == node.coordinate:
                            connectivity_list.append(node.id)
                            bFoundSameNode = True
                            break
                    if not bFoundSameNode:
                        list_of_nodes.append(Node(current_id, point))
                        connectivity_list.append(current_id)
                        current_id += 1
            for con_idx, connection in enumerate(connectivity_list[:-1]):
                for node in list_of_nodes:
                    if node.id == connection:
                        node.connected_to.append(connectivity_list[con_idx + 1])

        for node in list_of_nodes:
            G.add_node(node.id, pos=node.coordinate)
        for node in list_of_nodes:
            if not node.connected_to:
                pass
            else:
                for node_connection in node.connected_to:
                    '''
                    p0 = np.asarray(list_of_nodes[node.id].coordinate)
                    p1 = np.asarray(list_of_nodes[node_connection].coordinate)
                    weight = np.linalg.norm(p0 - p1)
                    G.add_edge(node.id, node_connection, weight=weight)
                    '''
                    G.add_edge(node.id, node_connection)
        if visualize:
            pos = nx.get_node_attributes(G, 'pos')
            w, h = figaspect(5 / 3)
            fig, ax = plt.subplots(figsize=(w, h))
            nx.draw(G, pos, node_size=20, ax=ax)
            plt.show()

        return G
