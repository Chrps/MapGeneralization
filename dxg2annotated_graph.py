from src.dxf2graph import dxf2graph
from misc.scale_graphs_from_json import scale_graph
from misc.add_annotations_from_json import add_annotations

data_path = r'C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\data_for_paper'

dxf2graph(data_path)
scale_graph(data_path)
add_annotations(data_path)
