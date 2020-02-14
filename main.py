from src.dxf_utils import DxfReader

def main():
    file_path = "data/dxf_files/MSP1-HoM-MA-XX+4-ET.dxf"

    dxf_obj = DxfReader(dxf_file_path=file_path)
    data = dxf_obj.extract_data()
    dxf_obj.plot_data()

if __name__ == "__main__":
    main()