# MapGeneralization
Generalizing Doors in TechnicalDrawings using Graph Neutral Networks

## Pipeline
**DXF -> Line data -> Graph:** Use DxfReader.extract_data() to create line data and DxfReader.convert_data_to_graph() to convert it to a graph - see main.py  
**Annotate graphs:** Use annotation_tool.py to annotate the graphs    
**Train graph node classifiation:** Use train.py to train a model. Configure model using config.py  
**Predict graph node classes, instance and generalize:** Use predict_w_generalize.py  
**Just predict graph node classes:** Use predict.py. However this file is deprecated!! Change predict_w_generalize.py to just predict instead.  

## Note
**Be sure to change the paths in the files!**
