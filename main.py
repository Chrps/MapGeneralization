from src.dxf_utils import DxfReader
import glob


def main():
    in_folder = r"C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\data_for_paper\Public\Musikkenshus\dxf\*.dxf"
    out_folder = r"C:\Users\Chrips\Aalborg Universitet\Frederik Myrup Thiesson - data\data_for_paper\Public\Musikkenshus\graphs"

    file_paths = glob.glob(in_folder)

    for file_path in file_paths:
        dxf_obj = DxfReader(dxf_file_path=file_path)
        dxf_obj.extract_data()
        dxf_obj.convert_data_to_graph(out_folder, visualize=False)
        print("done")
        #dxf_obj.load_graph("data/graphs/test_graph.gpickle")


if __name__ == "__main__":
    main()
