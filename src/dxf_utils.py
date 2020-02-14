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

    def extract_data(self):
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()

        # Would be cool to preallocate memory to the data list
        self.data = []

        for e in msp:
            if e.dxftype() == 'LINE':
                point1 = e.dxf.start
                point2 = e.dxf.end
                # Some layers are frozen which means they are hidden, so neglect those
                layer_name = e.dxf.layer
                layer = doc.layers.get(layer_name)
                if layer.is_frozen() == False:
                    # Neglect the z-value as expected to be 0
                    line = [point1[:2], point2[:2]]
                    self.data.append(line)
            elif e.dxftype() == 'LWPOLYLINE':
                # Initialize list with size equal to number of points within 'LWPOLYLINE' layout
                lwpolyline = []
                layer_name = e.dxf.layer
                layer = doc.layers.get(layer_name)
                if layer.is_frozen() == False:
                    with e.points() as points:
                        # points is a list of points with the format = (x, y, [start_width, [end_width, [bulge]]])
                        for point in points:
                            # I only want the x,y coordinates
                            lwpolyline.append(point[:2])
                    self.data.append(lwpolyline)
            elif e.dxftype() == 'POLYLINE':
                # POLYLINE points are saved as a list of vertices
                layer_name = e.dxf.layer
                layer = doc.layers.get(layer_name)
                if layer.is_frozen() == False:
                    polyline = []
                    vertices = e.vertices
                    for vertex in vertices:
                        point = vertex.dxf.location[:2]
                        polyline.append(point)
                    self.data.append(polyline)

        return self.data

    def plot_data(self):
        w, h = figaspect(5 / 3)
        lc = LineCollection(self.data, linewidths=2)
        fig, ax = plt.subplots(figsize=(w, h))
        ax.add_collection(lc)
        ax.autoscale()
        plt.show()
