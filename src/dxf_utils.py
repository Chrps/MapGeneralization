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
import open3d as o3d
import numpy as np
import os
import math
from math import isclose

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
        self.dxf_file_path = dxf_file_path
        extension = os.path.splitext(dxf_file_path)[1]
        if extension != '.dxf':
            raise Exception('File given is not of DXF type (.dxf)')
        else:
            self.dxf_path = dxf_file_path
            print('Loading ' + dxf_file_path + '. May take a while!')
            self.doc = ezdxf.readfile(self.dxf_path)
            self.msp = self.doc.modelspace()
            self.line_data = []
            self.id_data = []
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

    @staticmethod
    def arc(center, start_angle, end_angle, radius, line_out, id_out):
        delta_angle = 180.0 - abs(abs(start_angle - end_angle) - 180.0)

        end_angle = start_angle + delta_angle

        downsample_factor = delta_angle / 10.0

        np_angles = np.linspace(start_angle, end_angle, num=round(downsample_factor))

        for i, angle in enumerate(np_angles):
            point = [center[0] + (radius * math.cos(math.radians(angle))),
                     center[1] + (radius * math.sin(math.radians(angle)))]
            if i == 0:
                prev_point = point
            else:
                curr_point = point
                line = [prev_point, curr_point]
                prev_point = curr_point
                line_out.append(line)
                id_out.append(3)

    @staticmethod
    def arc_for_polylines(center, start_angle, end_angle, radius, start_point):
        delta_angle = 180 - abs(abs(start_angle - end_angle) - 180)

        end_angle = start_angle + delta_angle

        downsample_factor = delta_angle / 10.0

        np_angles = np.linspace(start_angle, end_angle, num=round(downsample_factor))

        all_points = []

        for angle in np_angles:
            point = [center[0] + (radius * math.cos(math.radians(angle))),
                     center[1] + (radius * math.sin(math.radians(angle)))]

            all_points.append(tuple(point))

        x1 = all_points[0][0]
        x2 = start_point[0]
        y1 = all_points[0][1]
        y2 = start_point[1]
        n = 5

        x_same = isclose(x1, x2, abs_tol=10**-n)
        y_same = isclose(y1, y2, abs_tol=10 ** -n)

        if x_same and y_same:
            return all_points
        else:
            return all_points[::-1]


    def entity2line(self, e):
        line_out = []
        id_out = []
        if e.dxftype() is 'LINE':
            point1 = e.dxf.start
            point2 = e.dxf.end
            if not self.is_hidden(e):
                # Neglect the z-value as expected to be 0
                line = [point1[:2], point2[:2]]
                line_out.append(line)
                id_out.append(0)
        elif e.dxftype() is 'LWPOLYLINE':
            if not self.is_hidden(e):
                # TODO: different line types?
                if e.dxf.handle == '1e5':
                    print("here")
                lwpolyline = []
                with e.points() as points:
                    # points is a list of points with the format = (x, y, [start_width, [end_width, [bulge]]])
                    for idx, point in enumerate(points):
                        if point[4] != 0.0:  # i.e. we have a bulge value
                            try:
                                arc_params = ezdxf.math.bulge_to_arc(start_point=point, end_point=points[idx+1], bulge=point[4])
                            except IndexError:
                                if e.closed:
                                    arc_params = ezdxf.math.bulge_to_arc(start_point=point, end_point=points[0], bulge=point[4])
                            center = arc_params[0]
                            start_angle = math.degrees(arc_params[1])
                            end_angle = math.degrees(arc_params[2])
                            radius = arc_params[3]
                            arc_points = self.arc_for_polylines(center, start_angle, end_angle, radius, point[:2])
                            for arc_point in arc_points:
                                lwpolyline.append(arc_point)
                        else:
                            lwpolyline.append(point[:2])
                    if e.closed:
                        lwpolyline.append(lwpolyline[0])
                line_out.append(lwpolyline)
                id_out.append(1)
        elif e.dxftype() is 'POLYLINE':
            # POLYLINE points are saved as a list of vertices
            if not self.is_hidden(e):
                polyline = []
                vertices = e.vertices
                for idx, vertex in enumerate(vertices):
                    bulge = vertex.dxf.bulge
                    if bulge != 0:
                        try:
                            arc_params = ezdxf.math.bulge_to_arc(start_point=vertex.dxf.location[:2], end_point=vertices[idx + 1].dxf.location[:2],
                                                                 bulge=bulge)
                        except IndexError:
                            if e.is_closed:
                                arc_params = ezdxf.math.bulge_to_arc(start_point=vertex.dxf.location[:2],
                                                                     end_point=vertices[0].dxf.location[:2],
                                                                     bulge=bulge)
                        center = arc_params[0]
                        start_angle = math.degrees(arc_params[1])
                        end_angle = math.degrees(arc_params[2])
                        radius = arc_params[3]
                        arc_points = self.arc_for_polylines(center, start_angle, end_angle, radius, vertex.dxf.location[:2])
                        for arc_point in arc_points:
                            polyline.append(arc_point)
                    else:
                        point = vertex.dxf.location[:2]
                        polyline.append(point)
                if e.is_closed:
                    polyline.append(polyline[0])
                line_out.append(polyline)
                id_out.append(2)
        elif e.dxftype() is 'HATCH':
            # TODO extract pattern!
            # TODO EllipseEdge
            # TODO SplineEdge
            if not self.is_hidden(e):
                for path in e.paths:
                    if path.PATH_TYPE is 'PolylinePath':
                        polyline = []
                        for vertex in path.vertices:
                            point = list(vertex[:2])
                            polyline.append(point)
                        line_out.append(polyline)
                        id_out.append(2)
                    elif path.PATH_TYPE is 'EdgePath':
                        for edge in path.edges:
                            if edge.EDGE_TYPE is 'LineEdge':
                                point1 = edge.start
                                point2 = edge.end
                                line = [point1[:2], point2[:2]]
                                line_out.append(line)
                                id_out.append(0)
                            elif edge.EDGE_TYPE is 'ArcEdge':
                                center = list(edge.center)
                                start_angle = edge.start_angle
                                end_angle = edge.end_angle
                                radius = edge.radius
                                self.arc(center, start_angle, end_angle, radius, line_out, id_out)
                            else:
                                print("unrecognized edge type: " + str(edge.EDGE_TYPE))
                    else:
                        print("unrecognized path type: " + str(path.PATH_TYPE))
        elif e.dxftype() is 'ARC':
            if not self.is_hidden(e):
                center = list(e.dxf.center.vec2)
                start_angle = e.dxf.start_angle
                end_angle = e.dxf.end_angle
                radius = e.dxf.radius
                self.arc(center, start_angle, end_angle, radius, line_out, id_out)
        elif e.dxftype() is 'CIRCLE':
            if not self.is_hidden(e):
                #print('Circle:', e.dxf.linetype)
                center = list(e.dxf.center.vec2)
                radius = e.dxf.radius
                step = 0
                downsample_factor = 10
                for i in range(360 + 1):
                    if i % downsample_factor == 0 or i == 0 or i == 360:
                        point = [center[0] + (radius * math.cos(math.radians(step))),
                                 center[1] + (radius * math.sin(math.radians(step)))]
                        if i == 0:
                            prev_point = point
                        else:
                            curr_point = point
                            line = [prev_point, curr_point]
                            prev_point = curr_point
                            line_out.append(line)
                            id_out.append(4)
                        step += downsample_factor
        elif e.dxftype() is 'ELLIPSE':
            if not self.is_hidden(e):
                center = list(e.dxf.center.vec2)
                major_axis = list(e.dxf.major_axis)
                ratio = e.dxf.ratio
                start_param = e.dxf.start_param
                end_param = e.dxf.end_param
                start_angle = start_param
                end_angle = end_param
                # Calculate major axis angle to x-axis
                delta_y = major_axis[1]
                delta_x = major_axis[0]
                angle2x_axis = np.arctan2(delta_y, delta_x)

                major_radius = np.sqrt(delta_y ** 2 + delta_x ** 2)
                minor_radius = major_radius*ratio

                delta_angle = np.degrees(end_angle) - np.degrees(start_angle)
                if delta_angle < 0:
                    delta_angle += 360
                step = np.degrees(start_angle)
                downsample_factor = 10
                for i in range(int(delta_angle) + 1):
                    if i % downsample_factor == 0 or i == 0 or i == int(delta_angle):
                        # Calculate point of ellipse
                        x = major_radius * np.cos(np.radians(step))
                        y = minor_radius * np.sin(np.radians(step))
                        s = np.sin((np.pi)-angle2x_axis)
                        c = np.cos((np.pi)-angle2x_axis)
                        x_rot = x * c - y * s
                        y_rot = x * s + y * c
                        x = x_rot + center[0]
                        y = y_rot + center[1]
                        point = [x, y]
                        if i == 0:
                            prev_point = point
                        else:
                            curr_point = point
                            line = [prev_point, curr_point]
                            prev_point = curr_point
                            line_out.append(line)
                            id_out.append(5)
                        step += downsample_factor
        elif e.dxftype() is 'POINT':
            if not self.is_hidden(e):
                # TODO finish and plot
                self.point_data.append(list(e.dxf.location[:2]))
        elif e.dxftype() is 'INSERT':
            if not self.is_hidden(e):
                block = self.doc.blocks[e.dxf.name]
                if block.is_alive:
                    list_e_types = []
                    b_name = block.name
                    insert = e.dxf.insert
                    if e.dxf.handle == "46E8D":
                        print("here")
                    for b_e in block:
                        list_e_types.extend([b_e.dxftype()])
                        layer_name = b_e.dxf.layer
                        lines, ids = self.entity2line(b_e)
                        if b_e.dxftype() is 'INSERT':
                            ins_location = b_e.dxf.insert
                            ins_lines = []
                            for idx_l, line in enumerate(lines):
                                tmp_line = []
                                for point in line:
                                    x = float(point[0]) + float(ins_location[0])
                                    y = float(point[1]) + float(ins_location[1])
                                    tmp_line.append((x, y))
                                ins_lines.extend([tmp_line])
                            for idx, ins_line in enumerate(ins_lines):
                                id = ids[idx]
                                line_out.append(ins_line)
                                id_out.append(id)
                        else:
                            for idx, ins_line in enumerate(lines):
                                id = ids[idx]
                                line_out.append(ins_line)
                                id_out.append(id)

        else:
            if not self.unrecognized_types:
                self.unrecognized_types.append([e.dxftype(), 1])
            else:
                if e.dxftype() in list(zip(*self.unrecognized_types))[0]:
                    self.unrecognized_types[list(zip(*self.unrecognized_types))[0].index(e.dxftype())][1] += 1
                else:
                    self.unrecognized_types.append([e.dxftype(), 1])
        return line_out, id_out

    def extract_data(self):
        for e in self.msp:
            lines, ids = self.entity2line(e)
            if lines:
                if e.dxftype() is 'INSERT':
                    block = self.doc.blocks[e.dxf.name]
                    b_name = block.name
                    rotation = e.dxf.rotation
                    x_scale = e.dxf.xscale
                    y_scale = e.dxf.yscale
                    if e.dxf.handle == '46F11':
                        print("here")
                    position_ = e.dxf.insert[:2]
                    ext_vect = e.dxf.extrusion
                    if ext_vect[2] == -1:
                        x = -(position_[0])
                        y = position_[1]
                    else:
                        x = position_[0]
                        y = position_[1]
                    position = (x, y)
                    base_point = block.block.dxf.base_point
                    new_origin = (base_point[0], base_point[1])
                    for idx, line in enumerate(lines):
                        id = ids[idx]
                        t_line = []
                        for point in line:
                            l_point = list(point)
                            # Scale
                            l_point[0] = (l_point[0] - new_origin[0]) * x_scale
                            l_point[1] = (l_point[1] - new_origin[1]) * y_scale
                            # Rotate
                            l_point = rotate(new_origin, l_point, math.radians(rotation))

                            # Translate
                            l_point[0] += position[0]
                            l_point[1] += position[1]
                            t_line.append(tuple(l_point))
                        self.line_data.append(t_line)
                        self.id_data.append(id)

                else:
                    for idx, line in enumerate(lines):
                        id = ids[idx]
                        self.line_data.append(line)
                        self.id_data.append(id)
                        pass
        for unrecognized_type in self.unrecognized_types:
            print("Type " + str(unrecognized_type[0]) + " not recognized. Nr. of instances: "
                  + str(unrecognized_type[1]))
        return self.line_data, self.id_data

    def plot_data(self):
        w, h = figaspect(5 / 3)
        lc = LineCollection(self.line_data, linewidths=1)
        fig, ax = plt.subplots(figsize=(w, h))
        ax.add_collection(lc)
        ax.autoscale()
        plt.show()

    def convert_data_to_graph(self, out_folder, visualize=True, save=True):
        # Create network graph
        self.G = nx.Graph()
        list_of_nodes = []
        list_of_ids = []
        current_id = 0

        for idx, entity in enumerate(self.line_data):
            entity_id = self.id_data[idx]
            connectivity_list = []
            for point in entity:
                point = tuple([int(_) for _ in point])
                # If we are beginning then list is empty (i.e. no nodes yet)
                if not list_of_nodes:
                    list_of_nodes.append(Node(current_id, point))
                    list_of_ids.append(entity_id)
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
                        list_of_ids.append(entity_id)
                        connectivity_list.append(current_id)
                        current_id += 1
            for con_idx, connection in enumerate(connectivity_list[:-1]):
                for node in list_of_nodes:
                    if node.id == connection:
                        node.connected_to.append(connectivity_list[con_idx + 1])

        for idx, node in enumerate(list_of_nodes):
            id = list_of_ids[idx]
            self.G.add_node(node.id, pos=node.coordinate, id=id)
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
                    self.G.add_edge(node.id, node_connection)
        if visualize:
            pos = nx.get_node_attributes(self.G, 'pos')
            ids = nx.get_node_attributes(self.G, 'id')
            myset = set(pos.values())
            w, h = figaspect(5 / 3)
            fig, ax = plt.subplots(figsize=(w, h))
            nx.draw(self.G, pos, node_size=20, ax=ax)
            #nx.draw_networkx_labels(self.G, pos, ids, ax=ax)
            plt.show()

        if save:
            file_name = os.path.basename(self.dxf_file_path)
            file_name = os.path.splitext(file_name)[0]
            nx.write_gpickle(self.G, out_folder + "\\" + file_name + ".gpickle")
            #nx.write_gexf(self.G, "data/graphs/test_graph.gexf")
            #nx.read_gexf("data/graphds/test_graph.gexf")

        return self.G

    def load_graph(self, gexf_file_path, visualize=True):
        file_name = os.path.basename(self.dxf_file_path)
        file_name = os.path.splitext(file_name)[0]
        self.G = nx.read_gpickle("data/graphs/" + file_name + ".gpickle")
        if visualize:
            pos = nx.get_node_attributes(self.G, 'pos')
            ids = nx.get_node_attributes(self.G, 'id')
            w, h = figaspect(5 / 3)
            fig, ax = plt.subplots(figsize=(w, h))
            nx.draw(self.G, pos, node_size=20, ax=ax)
            nx.draw_networkx_labels(self.G, pos, ids, ax=ax)
            plt.show()


    def graph2pcd(self):
        xyz = np.empty((0, 3), int)
        for idx in range(len(self.G.nodes)):
            x, y = self.G._node[idx]['pos']
            xyz = np.append(xyz, np.array([[x, y, 0]]), axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud('data/pcd/test_pointcloud.pcd', pcd)


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    rotated_point = [qx, qy]
    return rotated_point