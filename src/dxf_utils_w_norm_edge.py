'''
Utility file for extracting the data (nodes and edges) from .dxf AutoCAD Files
'''

import ezdxf
import os
import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect
# from matplotlib import colors as mcolors
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
    def __init__(self, dxf_file_path, max_length):
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
            self.max_length = max_length

    def is_hidden(self, entity):
        layer_name = entity.dxf.layer
        layer = self.doc.layers.get(layer_name)
        if layer.is_on() and not layer.is_frozen() :
            return False
        else:
            return True

    def split(self, start, end, segments):
        x_delta = (end[0] - start[0]) / float(segments)
        y_delta = (end[1] - start[1]) / float(segments)
        points = []
        for i in range(1, segments):
            points.append([start[0] + i * x_delta, start[1] + i * y_delta])
        return [start] + points + [end]

    def entity2line(self, e):
        points_out = []
        prev_point = []

        if not self.is_hidden(e):
            if e.dxftype() is 'LINE':
                # TODO if line is smaller than threshold then what?
                point1 = e.dxf.start
                point2 = e.dxf.end
                length = math.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
                if length > self.max_length:
                    nr_segments = math.ceil(length/self.max_length)
                    points = self.split(point1, point2, nr_segments)
                    for point in points:
                        self.point_data.append(point[:2])
                else:
                    points_out.append(list(point1[:2]))
                    points_out.append(list(point2[:2]))

            elif e.dxftype() is 'LWPOLYLINE':
                lwpolyline = []
                with e.points() as points:
                    # points is a list of points with the format = (x, y, [start_width, [end_width, [bulge]]])
                    for point in points:
                        # I only want the x,y coordinates
                        lwpolyline.append(point[:2])
                    if e.closed:
                        lwpolyline.append(lwpolyline[0])

                    for i,point in enumerate(lwpolyline):
                        if i == 0:
                            prev = point
                        else:
                            curr = point
                            length = math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
                            if length > self.max_length:
                                nr_segments = math.ceil(length / self.max_length)
                                points = self.split(prev, curr, nr_segments)
                                for point_ in points:
                                    self.point_data.append(point_[:2])
                            elif length < self.max_length:
                                self.point_data.append(point[:2])
                                # TODO add condition if length is too small
                                pass
                            prev = curr

                #line_out.append(lwpolyline)
            elif e.dxftype() is 'POLYLINE':
                # POLYLINE points are saved as a list of vertices
                polyline = []
                vertices = e.vertices
                for vertex in vertices:
                    point = vertex.dxf.location[:2]
                    polyline.append(point)
                if e.is_closed:
                    polyline.append(polyline[0])
                #line_out.append(polyline)
            elif e.dxftype() is 'HATCH':
                # TODO extract pattern!
                for path in e.paths:
                    if path.PATH_TYPE is 'PolylinePath':
                        polyline = []
                        for vertex in path.vertices:
                            point = list(vertex[:2])
                            polyline.append(point)
                        #line_out.append(polyline)
                    elif path.PATH_TYPE is 'EdgePath':
                        for edge in path.edges:
                            if edge.EDGE_TYPE is 'LineEdge':
                                point1 = edge.start
                                point2 = edge.end
                                line = [point1[:2], point2[:2]]
                                #line_out.append(line)
                            else:
                                print("unrecognized edge type: " + str(edge.EDGE_TYPE))
                    else:
                        print("unrecognized path type: " + str(path.PATH_TYPE))
            elif e.dxftype() is 'ARC':
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
                        #line_out.append(line)
                    step += 1
            elif e.dxftype() is 'CIRCLE':
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
                        #line_out.append(line)
                    step += 1
            elif e.dxftype() is 'ELLIPSE':
                # TODO make more generalized, assumes major_axis is aligned with either the x or y axis.
                center = list(e.dxf.center.vec2)
                major_axis = e.dxf.major_axis
                print(major_axis)
                major_radius = max(major_axis)
                idx_major_radius = tuple(major_axis).index(major_radius)
                print(idx_major_radius)
                ratio = e.dxf.ratio
                minor_radius = major_radius * ratio
                if idx_major_radius == 0:
                    x_radius = major_radius
                    y_radius = minor_radius
                else:
                    x_radius = major_radius
                    y_radius = minor_radius
                start_param = e.dxf.start_param
                end_param = e.dxf.end_param
                delta_param = end_param - start_param
                step = start_param
                for i in range(int(math.degrees(delta_param)) + 1):
                    if idx_major_radius == 0:
                        point = [center[0] + (x_radius * math.cos(step)),
                                 center[1] + (y_radius * math.sin(step))]
                    else:
                        point = [center[0] + (y_radius * math.cos(step)),
                                 center[1] + (x_radius * math.sin(step))]
                    if i == 0:
                        prev_point = point
                    else:
                        curr_point = point
                        line = [prev_point, curr_point]
                        prev_point = curr_point
                        #line_out.append(line)
                    step += math.radians(1)
            elif e.dxftype() is 'POINT':
                # TODO finish and plot
                #self.point_data.append(list(e.dxf.location[:2]))
                pass
            elif e.dxftype() is 'INSERT':
                if e.is_alive:
                    block = self.doc.blocks[e.dxf.name]
                    for b_e in block:
                        if not self.is_hidden(b_e) and b_e.is_alive:
                            lines = self.entity2line(b_e)
                            for line in lines:
                                #line_out.append(line)
                                pass
            else:
                if not self.unrecognized_types:
                    self.unrecognized_types.append([e.dxftype(), 1])
                else:
                    if e.dxftype() in list(zip(*self.unrecognized_types))[0]:
                        self.unrecognized_types[list(zip(*self.unrecognized_types))[0].index(e.dxftype())][1] += 1
                    else:
                        self.unrecognized_types.append([e.dxftype(), 1])
        return points_out

    def extract_data(self):
        # Would be cool to preallocate memory to the data list
        for e in self.msp:
            points = self.entity2line(e)
            #self.point_data.append(points)
            '''if lines:
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
                        #self.line_data.append(t_line)
                else:
                    for line in lines:
                        self.line_data.append(line)'''
        for unrecognized_type in self.unrecognized_types:
            print("Type " + str(unrecognized_type[0]) + " not recognized. Nr. of instances: "
                  + str(unrecognized_type[1]))
        return self.line_data

    def plot_data(self):
        w, h = figaspect(5 / 3)
        #colors = [mcolors.to_rgba(c)
        #          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        print(self.point_data)
        tpoint_data = list(zip(*self.point_data))
        #lc = LineCollection(self.line_data, linewidths=1, colors=colors)
        print(len(tpoint_data[0]))
        print(len(tpoint_data[1]))
        plt.scatter(tpoint_data[0], tpoint_data[1], s=1)
        #fig, ax = plt.subplots(figsize=(w, h))
        #ax.add_collection(lc)
        #ax.autoscale()
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
