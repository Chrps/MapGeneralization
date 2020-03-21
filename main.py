from src.dxf_utils import DxfReader


def main():
    file_path = "data/dxf_files/xray_room.dxf"

    dxf_obj = DxfReader(dxf_file_path=file_path)
    #dxf_obj.extract_data()
    #dxf_obj.convert_data_to_graph()
    dxf_obj.load_graph("data/graphs/test_graph.gpickle")
    #dxf_obj.graph2pcd()


if __name__ == "__main__":
    main()
