import os
from pathlib import Path
from src.dxf_utils import DxfReader


def dxf2graph(data_path_):
    pub_data_dir = data_path_ + '\Public'
    # Get all DXF files in path
    dxf_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pub_data_dir) for f in filenames if
                 os.path.splitext(f)[1] == '.dxf']

    # Get all subdirectories in path
    sub_dirs = [dI for dI in os.listdir(pub_data_dir) if os.path.isdir(os.path.join(pub_data_dir, dI))]

    # Create subdirectories in directory for preoduced data
    for sub_dir in sub_dirs:
        graph_dir = os.path.join(os.path.join(pub_data_dir, os.path.join('re_prod', sub_dir)), 'graphs')
        Path(graph_dir).mkdir(parents=True, exist_ok=True)

    # Convert DXF files to graphs
    for dxf_file in dxf_files:
        print('Converting: ' + dxf_file)
        dxf_obj = DxfReader(dxf_file_path=dxf_file)
        dxf_obj.extract_data()

        sub_dir_name = Path(dxf_file).parts[-3]
        out_dir = os.path.join(os.path.join(pub_data_dir, os.path.join('re_prod', sub_dir_name)), 'graphs')

        dxf_obj.convert_data_to_graph(out_dir, visualize=False)


#data_path = r'C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\data_for_paper'
#dxf2graph(data_path)
