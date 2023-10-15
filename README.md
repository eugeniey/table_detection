### Deterministic method

Folder `table_detection_deterministic_method/get_lines.py` contains a script for table detection using a deterministic approach. This was to compare to ML approachs for table detection. While being efficient, ML methods had a higher rate and higher quality of table detection. By using HoughLines which detects lines. 

`table_detection_deterministic_method/v1_detectron_tablebank_combine_cv2.ipynb` I tried combining the deterministic method and ML methods by exploring ways to combine the predictions. This was an attempt to try and see if I could improve the ML methods.

### Metric

File `metric_tablebank_my_implementation.py` is script of my implementation of the metric defined by the Table Bank author. They did not link any implemnetation of the metric they use. Hence I implemented my own and made comparison using their model. 

File `CascadeTabNet.ipynb` is an Jupyter notebook using the CascadeTabNet model with the different datasets set from TableBank and the measure the quality of each.

`get_iou.py` is a script to define de IoU of a prediction box and ground truth box. This is tool use to define metric or just to investigate issues.

`metric_tool_data` contains the predictions and ground truth in the right format for the Metric Tool.

### Training

File `detectron_training_publaynet.ipynb` is a Jupyter notebook fine tuning the Detectron model with the PubLayNet data. It also contains testing with PubLayNet and TableBank.

`detectron_mymodel_tablebank_publaynet.ipynb` is a Jupyter notebook fine tuning the Detectron model with different combinaison of training data with PubLayNet, Latex TableBank, Word TableBank.

`tablebank_with_transformation.ipynb` apply transformation on training set, and fine tune the detectron model and measure the impact of these transformation on training.

### Dataset format

File `change_annotation_format.ipynb` is a Jupyter notebook to standarize the datasets format since we are using multiple datasets. These datasets all have different format. We need to defined format for each type of model, since each model will take data differently.

File `get_coco_format.py` is a script to prepare data in COCO format to be able to use them with Detectron.

`get_models_and_set.py` defines different methods to get the datasets of PubLayNet, TableBank. There is also functions to get different type of subsets from them, for example, subset of test/training data of images with only 1 table, 2 tables, 3 tables. 

### Labeling errors

File `find_labeling_errors.ipynb` is a Jupyter notebook to investigate the labeling errors in datasets TableBank and PubLayNet.

### Testing

`tablebank_trained_model_metrictool.ipynb` test all the possible combinaisons of training and testing data with the clean functions. In this notebook, we store all the data using the different type of model, with the different training data to use using the Metric Tool.

### Anything

`study_tablebank.ipynb` is just a notebook that contains different type of invetigation on the datasets. For example, investigating the kind of labeling errors made on images with very low precision. It also contains the code to plot the Precision/Recall curve.


