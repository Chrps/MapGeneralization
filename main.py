from src.dxf_utils import DxfReader

def main():
    file_path = r"D:\University Stuff\OneDrive - Aalborg Universitet\P10 - Master's Thesis\data\MapsPeople-dxf\AU\N¥1\1320\L1320_H1_EK_K13.03_B07.01_N01_ZEtage_K.dxf"
    out_folder = r"D:\University Stuff\OneDrive - Aalborg Universitet\P10 - Master's Thesis\data\graphs\AU"

    dxf_obj = DxfReader(dxf_file_path=file_path)
    dxf_obj.extract_data()
    dxf_obj.convert_data_to_graph(out_folder, visualize=True)
    print("done")
    #dxf_obj.load_graph("data/graphs/test_graph.gpickle")


if __name__ == "__main__":
    main()
