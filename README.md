# MapGeneralization
Make generalized maps from technical drawings 

## Pipeline
**DWG(optional) -> DXF:** Can be done online at https://anyconv.com/dwg-to-dxf-converter/
**DXF -> Line data -> Graph:** Use DxfReader.extract_data() to create line data and DxfReader.convert_data_to_graph() to convert it to a graph
**Graph node classifiation:** Use train.py
