# Table Detection

This repo contains some Jupyter Notebooks and scripts used for my Master's thesis to study the field of table dection in documents. I mostly studied the TableBank dataset and the PubLayNet dataset.


### Deterministic method

We created a deterministic algorithm to detect tables in documents. This was to compare deterministic methods to ML approachs for table detection. While being efficient, ML methods had better performances and higher quality of table detections.

`table_detection_deterministic_method/get_lines.py` contains a script for table detection using a deterministic approach I created. Algorithm uses HoughLines which detects lines.

`table_detection_deterministic_method/detectron_tablebank_combine_cv2.ipynb` I tried combining the deterministic method and ML methods by exploring ways to combine the predictions. This was an attempt to try and see if I could improve the ML methods.


### Metric

The field of object detection uses a lot of metrics, which creates issues. We studied those different metrics in the field of table detections.

`utils/metric_tablebank_my_implementation.py` is a script of my implementation of the metric defined by the TableBank authors. They did not link any implemnetation of the metric they use. Hence, I implemented my own and made comparison using their model. 

`utils/get_iou.py` is a script to define de IoU of a prediction box and ground truth box. This is tool use to define metric or just to investigate the datasets.


#### Metric tool

We used [[Open-Source Visual Interface for Object Detection Metrics]](#1) which defines different type of metrics for object detection. 

`metric_tool_data` contains the predictions and ground truth in the right format for the Metric Tool.

`metric_tool/tablebank_trained_model_metrictool.ipynb` test all the possible combinaisons of training and testing data. In this notebook, we store all the data using the different type of model, with the different training data to use using the Metric Tool.


### CascadeTabNet

We used [[CascadeTabNet]](#3) which is an automatic table recognition method for interpretation of tabular data in document images.

`cascadetabnet_and_transformations/CascadeTabNet.ipynb` is an Jupyter notebook using the CascadeTabNet model with the different datasets set from TableBank and the measure the quality of each.

`cascadetabnet_and_transformations/tablebank_with_transformation.ipynb` is an Jupyter notebook using the transformation defined by CascadeTabNet authors to apply data augmentation to the training data. The goal was to study the effect of this data augmentation.


### Training with detectron

We used [[Detectron]](#2) to fine-tune some pretrained Detectron's models with TableBank and PubLayNet datasets.

`detectron_trainings/detectron_training_publaynet.ipynb` is a Jupyter notebook fine tuning the Detectron model with the PubLayNet data. It also contains testing with PubLayNet and TableBank.

`detectron_trainings/detectron_mymodel_tablebank_publaynet.ipynb` is a Jupyter notebook fine tuning the Detectron model with different combinaison of training data with PubLayNet, Latex TableBank, Word TableBank.

`detectron_trainings/tablebank_with_transformation.ipynb` apply transformation on training set, and fine tune the detectron model and measure the impact of these transformation on training.


### Dataset format

Object detections models all uses different types of format for datasets.

`detectron_trainings/change_annotation_format.ipynb` is a Jupyter notebook to standarize the datasets format since we are using multiple datasets. These datasets all have different format. We need to defined format for each type of model, since each model will take data differently.

`utils/get_coco_format.py` is a script to prepare data in COCO format to be able to use them with Detectron.

`utils/get_models_and_set.py` defines different methods to get the datasets of PubLayNet, TableBank. There is also functions to get different type of subsets from them, for example, subset of test/training data of images with only 1 table, 2 tables, 3 tables. 


### Labeling errors

`study_tablebank/find_labeling_errors.ipynb` is a Jupyter notebook to investigate the labeling errors in datasets TableBank and PubLayNet.

`study_tablebank/study_tablebank.ipynb` is just a notebook that contains different type of invetigation on the datasets. For example, investigating the kind of labeling errors made on images with very low precision. It also contains the code to plot the Precision/Recall curve.



## References
<a id="1">[Open-Source Visual Interface for Object Detection Metrics]</a> 
Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B. (2021). 
Open-Source Visual Interface for Object Detection Metrics.
(https://github.com/rafaelpadilla/review_object_detection_metrics)


<a id="2">[Detectron]</a> 
Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick (2019). 
Detectron2.
(https://github.com/facebookresearch/detectron2)


<a id="3">[CascadeTabNet]</a> 
Devashish Prasad and Ayan Gadpal and Kshitij Kapadni and Manish Visave and Kavita Sultanpure (2020). 
CascadeTabNet.
(https://github.com/DevashishPrasad/CascadeTabNet)