'''
Utility file for extracting the data (nodes and edges) from .dxf AutoCAD Files
'''

import ezdxf
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect


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
            self.doc = ezdxf.readfile(self.dxf_path)
            self.msp = self.doc.modelspace()
            self.data = []
            self.unrecognized_types = []

    def is_hidden(self, entity):
        layer_name = entity.dxf.layer
        layer = self.doc.layers.get(layer_name)
        if layer.is_on() and not layer.is_frozen():
            return False
        else:
            return True

    def extract_data(self):
        # Would be cool to preallocate memory to the data list

        for e in self.msp:
            if e.dxftype() == 'LINE':
                point1 = e.dxf.start
                point2 = e.dxf.end
                if not self.is_hidden(e):
                    # Neglect the z-value as expected to be 0
                    line = [point1[:2], point2[:2]]
                    self.data.append(line)
            elif e.dxftype() == 'LWPOLYLINE':
                if not self.is_hidden(e):
                    lwpolyline = []
                    with e.points() as points:
                        # points is a list of points with the format = (x, y, [start_width, [end_width, [bulge]]])
                        for point in points:
                            # I only want the x,y coordinates
                            lwpolyline.append(point[:2])
                    self.data.append(lwpolyline)
            elif e.dxftype() == 'POLYLINE':
                # POLYLINE points are saved as a list of vertices
                if not self.is_hidden(e):
                    polyline = []
                    vertices = e.vertices
                    for vertex in vertices:
                        point = vertex.dxf.location[:2]
                        polyline.append(point)
                    self.data.append(polyline)
            else:
                if not self. unrecognized_types:
                    self.unrecognized_types.append([e.dxftype(), 1])
                else:
                    if e.dxftype() in list(zip(*self.unrecognized_types))[0]:
                        self.unrecognized_types[list(zip(*self.unrecognized_types))[0].index(e.dxftype())][1] += 1
                    else:
                        self.unrecognized_types.append([e.dxftype(), 1])

        for unrecognized_type in self.unrecognized_types:
            print("Type " + str(unrecognized_type[0]) + " not recognized. Nr. of instances: "
                  + str(unrecognized_type[1]))
        return self.data

    def plot_data(self):
        w, h = figaspect(5 / 3)
        lc = LineCollection(self.data, linewidths=2)
        fig, ax = plt.subplots(figsize=(w, h))
        ax.add_collection(lc)
        ax.autoscale()
        plt.show()
