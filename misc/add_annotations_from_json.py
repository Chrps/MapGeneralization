import os
import json
import networkx as nx
from pathlib import Path


def add_annotations(data_path):
    pub_data_dir = data_path + '\Public'
    # Get all DXF files in path
    annotation_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pub_data_dir) for f in filenames if
                 os.path.splitext(f)[1] == '.json']

    for annotation_file in annotation_files:
        p = Path(annotation_file)
        graph_path = os.path.join(pub_data_dir, os.path.join('re_prod', os.path.join(p.parts[-3],
                                  os.path.join('graphs_scaled', p.parts[-1].replace('.json', '_scaled.gpickle')))))
        with open(annotation_file) as fp:
            annotation_dict = json.load(fp)

            nxg = nx.read_gpickle(graph_path)
            labels = []
            nx.set_node_attributes(nxg, labels, 'label')

            for key, value in annotation_dict.items():
                nxg._node[int(key)]['label'] = value

            Path(Path(graph_path.replace('graphs_scaled', 'anno')).parent).mkdir(parents=True, exist_ok=True)
            output_file = graph_path.replace("graphs_scaled", "anno").replace("\\", "/")\
                                    .replace("_scaled.gpickle", "_w_annotations.gpickle")
            nx.write_gpickle(nxg, output_file)


#data_path = r'C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\data_for_paper'
#add_annotations(data_path)
