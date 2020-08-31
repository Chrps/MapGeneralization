import os
import json
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect


def scale_graph(data_path_):
    scale_json_file = "scale_factors.json"
    scale_json_path = os.path.join(data_path_, scale_json_file)

    with open(scale_json_path) as json_file:
        data = json.load(json_file)
        for graph_rel_path, scale_factor in data.items():
            graph_rel_path = graph_rel_path.replace("Public/", "Public/re_prod/")
            print('Scaling: ' + graph_rel_path)
            graph_abs_path = os.path.join(data_path_, graph_rel_path)
            Path(Path(graph_abs_path.replace("graphs", "graphs_scaled")).parent).mkdir(parents=True, exist_ok=True)

            nxg = nx.read_gpickle(graph_abs_path)
    
            # Moving everything to origin
            pos = nx.get_node_attributes(nxg, 'pos')
            positions = list(pos.values())
            x_pos = []
            y_pos = []
            for coord in positions:
                x_pos.append(coord[0])
                y_pos.append(coord[1])
    
            min_x = min(x_pos)
            max_x = max(x_pos)
            min_y = min(y_pos)
            max_y = max(y_pos)
            if min_x < 0:  # Negative
                x_origin = abs(min_x)
            else:  # Positive
                x_origin = -min_x
            if min_y < 0:  # Negative
                y_origin = abs(min_y)
            else:  # Positive
                y_origin = -min_y
    
            # Rescale the Graph
            copy_graph = nxg.copy()
    
            for i in range(copy_graph.number_of_nodes()):
                orig_pos = (copy_graph.nodes[i]['pos'])
                new_pos = ((orig_pos[0] + x_origin) * scale_factor, (orig_pos[1] + y_origin) * scale_factor)
                copy_graph.nodes[i]['pos'] = new_pos
    
            Path(Path(graph_abs_path.replace("graphs", "graphs_scaled")).parent).mkdir(parents=True, exist_ok=True)
            output_file = graph_abs_path.replace("graphs", "graphs_scaled").replace("\\", "/").replace(".gpickle", "_scaled.gpickle")

            '''scale_pos = nx.get_node_attributes(copy_graph, 'pos')
            w, h = figaspect(5 / 3)
            fig, ax = plt.subplots(figsize=(w, h))
            nx.draw(copy_graph, scale_pos, node_size=20, ax=ax)
            plt.show()'''
            nx.write_gpickle(copy_graph, output_file)


#data_path = r"C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\data_for_paper"
#scale_graph(data_path)