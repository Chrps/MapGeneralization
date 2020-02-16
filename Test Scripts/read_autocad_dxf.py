'''
This is a test script for reading autocad files (DWG files with the .dwg extension)
'''

import ezdxf
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import figaspect

def get_axis_limits(list_of_lists):
    min_x = min(list_of_lists[0], key=lambda t: t[0])[0]
    min_y = min(list_of_lists[0], key=lambda t: t[1])[1]
    max_x = max(list_of_lists[0], key=lambda t: t[0])[0]
    max_y = max(list_of_lists[0], key=lambda t: t[1])[1]
    for list in list_of_lists:
        new_min_x = min(list, key=lambda t: t[0])[0]
        new_min_y = min(list, key=lambda t: t[1])[1]
        new_max_x = max(list, key=lambda t: t[0])[0]
        new_max_y = max(list, key=lambda t: t[1])[1]
        if new_min_x < min_x:
            min_x = new_min_x
        if new_min_y < min_y:
            min_y = new_min_y
        if new_max_x > max_x:
            max_x = new_max_x
        if new_max_y > max_y:
            max_y = new_max_y
    x_limits = (min_x, max_x)
    y_limits = (min_y, max_y)
    return x_limits, y_limits

def main():
    # Changing the working directory to the project, not the script
    directory_of_script = os.path.dirname(os.getcwd())
    dxf_file_path = 'data/dxf_files/MSP1-HoM-MA-XX+4-ET.dxf'
    dxf_file_path = directory_of_script + r'\\' + dxf_file_path

    doc = ezdxf.readfile(dxf_file_path)

    msp = doc.modelspace()

    # Would be cool to preallocate memory to these lists
    list_of_lines = []
    list_of_lwpolylines = []
    list_of_raw_lwpolylines = []
    list_of_polylines = []
    counter = 0
    counter_in = 0
    for e in msp:
        if e.dxftype() == 'LINE':
            point1 = e.dxf.start
            point2 = e.dxf.end
            #Some layers are frozen which means they are hidden, so neglect those
            layer_name = e.dxf.layer
            layer = doc.layers.get(layer_name)
            if layer.is_frozen() == False:
                # Neglect the z-value as expected to be 0
                line = [point1[:2], point2[:2]]
                list_of_lines.append(line)
        elif e.dxftype() == 'LWPOLYLINE':
            # Initialize list with size equal to number of points within 'LWPOLYLINE' layout
            lwpolyline = []
            raw_lwpolyline = []
            layer_name = e.dxf.layer
            layer = doc.layers.get(layer_name)
            if layer.is_frozen() == False:
                with e.points() as points:
                    # points is a list of points with the format = (x, y, [start_width, [end_width, [bulge]]])
                    for point in points:
                        # I only want the x,y coordinates
                        lwpolyline.append(point[:2])
                        raw_lwpolyline.append(point)
                list_of_lwpolylines.append(lwpolyline)
                list_of_raw_lwpolylines.append(raw_lwpolyline)
        elif e.dxftype() == 'POLYLINE':
            counter += 1
            # POLYLINE points are saved as a list of vertices
            layer_name = e.dxf.layer
            layer = doc.layers.get(layer_name)
            if layer.is_frozen() == False:
                counter_in += 1
                polyline = []
                vertices = e.vertices
                for vertex in vertices:
                    point = vertex.dxf.location[:2]
                    polyline.append(point)
                list_of_polylines.append(polyline)
    print("POLYLINE out: ", counter)
    print("POLYLINE in: ", counter)
    # We need to set the plot limits.
    w, h = figaspect(5 / 3)
    plt.figure(1)
    fig, ax = plt.subplots(figsize=(w, h))
    x_limit, y_limit = get_axis_limits(list_of_lines)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    line_segments = LineCollection(list_of_lines,
                                   linestyle='solid')
    ax.add_collection(line_segments)
    ax.set_title('Only LINES')

    # We need to set the plot limits.
    plt.figure(2)
    fig, ax = plt.subplots(figsize=(w, h))
    x_limit, y_limit = get_axis_limits(list_of_lines)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    lwpolylines_segments = LineCollection(list_of_lwpolylines,
                                   linestyle='solid')
    ax.add_collection(lwpolylines_segments)
    ax.set_title('Only LWPOLYLINE')

    # We need to set the plot limits.
    plt.figure(3)
    fig, ax = plt.subplots(figsize=(w, h))
    x_limit, y_limit = get_axis_limits(list_of_lines)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    polylines_segments = LineCollection(list_of_polylines,
                                   linestyle='solid')
    ax.add_collection(polylines_segments)
    ax.set_title('Only POLYLINE')


    plt.show()
    print("done")

if __name__ == "__main__":
    main()