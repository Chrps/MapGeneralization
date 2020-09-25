from src.dxf_utils import DxfReader
import os
import argparse


def dxf2graph(dxf_path, graph_dir):
    dxf_obj = DxfReader(dxf_file_path=dxf_path)
    dxf_obj.extract_data()
    dxf_obj.convert_data_to_graph(graph_dir, visualize=True)
    print("done")
    # dxf_obj.load_graph("data/graphs/test_graph.gpickle")


if __name__ == "__main__":
    """
    Convert dxf files.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dxf", type=str,
                    default='data/Public/Musikkenshus/dxf', help="directory with dxf files")

    args = vars(ap.parse_args())

    # make sure output dir exists
    graph_dir = args['dxf'].replace(args['dxf'].split('/')[-1], 'graphs')
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
        print("created new dir {} for saving graphs".format(graph_dir))
    else:
        print("saving graphs to {}".format(graph_dir))

    dxf_files = sorted([os.path.join(args['dxf'], f) for f in os.listdir(args['dxf']) if 'dxf' in f])

    for dxf_path in dxf_files[:]:
        print(dxf_path)
        filename = os.path.basename(dxf_path)
        graph_path = os.path.join(graph_dir, filename.replace('dxf', 'gpickle'))

        dxf2graph(dxf_path, graph_dir)
