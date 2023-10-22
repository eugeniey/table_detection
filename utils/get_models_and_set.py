import torch
import torchvision

#%matplotlib notebook
import glob
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import Image
import time
import shutil
from os import path

from shapely.geometry import Polygon

# importing required libraries
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
import json
import os
from collections import Counter
import cv2
#import cv2
import numpy as np
from metric_tablebank import metric_table_bank_union

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer


path_data = "/data/rali5/Tmp/yockelle/TableBank/TableBank/Detection/"


def get_1k_test_set(type_data):
    path_data = "/data/rali5/Tmp/yockelle/TableBank/TableBank/"
    data = None
    if type_data == "word":
        f = open(path_data + "test_data_word_1k.json",)
        data = json.load(f)
    elif type_data == "latex":
        f = open(path_data + "test_data_latex_1k.json",)
        data = json.load(f)
    elif type_data == "publaynet":
        f = open(path_data + "test_data_publaynet_1k.json",)
        data = json.load(f)
    else: 
        print("TYPE DATA WAS NOT WRITTEN PROPERLY")

    return data



def get_model(type_, size_model, tablebank_model = True):

    cfg = get_cfg()

    if tablebank_model:
        cfg.merge_from_file("/data/rali5/Tmp/yockelle/TableBank/TableBank/output/"+type_+"/"+size_model+"/config_"+type_+"_"+size_model+".yaml")
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/data/rali5/Tmp/yockelle/TableBank/TableBank/output/"+ type_+"/"+size_model+"/model_final_"+type_+"_"+size_model+".pth")
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.TEST.EVAL_PERIOD = 1000
        CUDA_LAUNCH_BLOCKING = 1
        predictor = DefaultPredictor(cfg)

    # we take my model
    else:
        print(cfg.OUTPUT_DIR + "/" + type_+ "/", "model_final.pth")

        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        #cfg.DATASETS.TRAIN = (train_set_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR + "/" + type_+ "/", "model_final.pth")
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.TEST.EVAL_PERIOD = 1000
        CUDA_LAUNCH_BLOCKING = 1
        predictor = DefaultPredictor(cfg)

    return predictor



def evaluate_models(train_type_name, test_type_name, size_model, threshold=0.975, take_the_table_bank_model = True, take_only_1k = "true"):

    sum_numerator = 0
    sum_numerator_old = 0
    sum_denominator_precision = 0
    sum_denominator_recall = 0

    predictor = get_model(train_type_name, size_model, tablebank_model = take_the_table_bank_model)

    test_data = []
    # Take the WORD data
    if "word" in test_type_name:
        if take_only_1k == "true":
            test_data.extend(get_1k_test_set("word"))
        elif take_only_1k == "random":
            word = get_test_data_word()
            test_data.extend(random.sample(word, 1000))
        else:
            word = get_test_data_word()
            test_data.extend(word)
    # Take the LATEX data
    if "latex" in test_type_name:
        if take_only_1k == "true":
            test_data.extend(get_1k_test_set("latex"))
        elif take_only_1k == "random":
            latex = get_test_data_latex()
            test_data.extend(random.sample(latex, 1000))
        else:
            latex = get_test_data_latex()
            test_data.extend(latex)
    # Take the PUBLAYNET data
    if "publaynet" in test_type_name:
        if take_only_1k == "true":
            test_data.extend(get_1k_test_set("publaynet"))
        elif take_only_1k == "random":
            publaynet = get_test_data_publaynet()
            test_data.extend(random.sample(publaynet, 1000))
        else:
            publaynet = get_test_data_publaynet()
            test_data.extend(publaynet)

    print(test_data[0]["file_name"])
    print(len(test_data))
    for i in range(len(test_data)):
        image_dict = test_data[i]
        try: 
            image_name = "/data/rali5/Tmp/yockelle/TableBank/" + image_dict["file_name"]  
            im = cv2.imread(image_name)
            does_shape_work = im.shape
        except:
            image_name = image_dict["file_name"]
            im = cv2.imread(image_name)

        outputs = predictor(im)

        predictions_detectron = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores_detectron = outputs["instances"].scores.cpu().numpy()
        predictions_with_threshold = []
        for j, p in enumerate(predictions_detectron):
            if scores_detectron[j]>threshold:
                predictions_with_threshold.append(p)
        predictions_with_threshold = np.array(predictions_with_threshold)

        # Get the ground truth
        bbox_ground_truth = []
        truth = image_dict["annotations"]
        for t in truth:
            x_min = t["bbox"][0]
            y_min = t["bbox"][1]
            w = t["bbox"][2]
            h = t["bbox"][3]
            x_max = x_min + w
            y_max = y_min + h
            bbox_ground_truth.append([x_min, y_min, x_max, y_max])

        # get metrics Detectron
        metric_result = metric_table_bank_union(bbox_ground_truth, predictions_with_threshold)

        if metric_result != None:
            numerator, denominator_precision, denominator_recall, old_way_area_union = metric_result

            sum_numerator += numerator
            sum_numerator_old += old_way_area_union
            sum_denominator_precision += denominator_precision
            sum_denominator_recall += denominator_recall

    precision = sum_numerator/sum_denominator_precision
    recall = sum_numerator/sum_denominator_recall
    f1 = (2 * precision * recall) / (precision + recall)
    #print("Precision ", precision)
    #print("Recall ", recall)
    #print("F1 ", f1)

    return precision, recall, f1

#====================================================================================

def get_valid_data():
    latex = pickle.load( open(path_data + "annotations/COCO_tablebank_latex_val.pickle", "rb" ) )
    word = pickle.load( open(path_data + "annotations/COCO_tablebank_word_val.pickle", "rb" ) )
    return latex + word
    
# TRAINING DATA
def get_train_data_latex():
    latex = pickle.load( open(path_data + "annotations/COCO_tablebank_latex_train.pickle", "rb" ) )
    return latex

def get_train_data_word():
    word = pickle.load( open(path_data + "annotations/COCO_tablebank_word_train.pickle", "rb" ) )
    return word

def get_train_data_word_latex():
    latex = get_train_data_latex()
    word = get_train_data_word()
    total_train = latex + word
    random.shuffle(total_train)
    return total_train

def get_train_data_latex_pln():
    latex = get_train_data_latex()
    pln = get_train_data_publaynet()
    total_train = latex + pln
    random.shuffle(total_train)
    return total_train

def get_train_data_word_pln():
    word = get_train_data_word()
    pln = get_train_data_publaynet()
    total_train = word + pln
    random.shuffle(total_train)
    return total_train

def get_train_data_pln_latex_word():
    latex = get_train_data_latex()
    word = get_train_data_word()
    publaynet = get_train_data_publaynet()
    total_train = latex + word + publaynet
    random.shuffle(total_train)
    return total_train

#====================================================================================

# TEST DATA
def get_test_data_latex():
    latex = pickle.load( open(path_data + "annotations/COCO_tablebank_latex_test.pkl", "rb" ) )
    return latex

def get_test_data_word():
    word = pickle.load( open(path_data + "annotations/COCO_tablebank_word_test.pkl", "rb" ) )
    return word

def get_test_data_word_latex():
    latex = get_test_data_latex()
    word = get_test_data_word()
    total_test = latex + word
    random.shuffle(total_test)
    return total_test

def get_test_data_publaynet():
    test_publaynet = pickle.load( open("../PubLayNet/COCO_test_set_table_only.pkl", "rb" ) )
    cleaned_test_publaynet = []
    for image in test_publaynet:
        annotation_array = []
        for annot in image["annotations"]:
            x_min = annot["bbox"][0]
            y_min = annot["bbox"][1]
            w = annot["bbox"][2]
            h = annot["bbox"][3]
            x_max = x_min + w
            y_max = y_min + h
            area_truth = (x_max - x_min) * (y_max - y_min)
            if area_truth>0:
                annot["category_id"] = 0
                annotation_array.append(annot)
        if len(annotation_array)>0:
            image['categories'] = [{'supercategory': '', 'id': 0, 'name': 'table'}]
            image["annotations"] = annotation_array
            image["file_name"] = "/data/rali5/Tmp/yockelle/PubLayNet/" + image["file_name"]
            cleaned_test_publaynet.append(image)
    return cleaned_test_publaynet

def get_train_data_publaynet():
    train_publaynet = pickle.load( open("../PubLayNet/COCO_train_set_table_only.pkl", "rb" ) )
    for image in train_publaynet:
        annotation_array = []
        for annot in image["annotations"]:
            annot["category_id"] = 0
            annotation_array.append(annot)
        image['categories'] = [{'supercategory': '', 'id': 0, 'name': 'table'}]
        image["annotations"] = annotation_array
        image["file_name"] = "/data/rali5/Tmp/yockelle/PubLayNet/" + image["file_name"]
    return train_publaynet

#====================================================================================

def get_train_data_pln_and_empty():
    no_table = pickle.load( open("../PubLayNet/COCO_no_table_examples.pkl", "rb" ) )
    for image in no_table:
        image['categories'] = [{'supercategory': '', 'id': 0, 'name': 'table'}]
        image["annotations"] = []
        image["file_name"] = "/data/rali5/Tmp/yockelle/PubLayNet/" + image["file_name"]

    train = get_train_data_publaynet()
    total_test = no_table + train
    random.shuffle(total_test)

    return total_test


def get_train_data_and_word_pln_empty():

    word = get_train_data_word()

    no_table = pickle.load( open("../PubLayNet/COCO_no_table_examples.pkl", "rb" ) )
    for image in no_table:
        image['categories'] = [{'supercategory': '', 'id': 0, 'name': 'table'}]
        image["annotations"] = []
        image["file_name"] = "/data/rali5/Tmp/yockelle/PubLayNet/" + image["file_name"]

    train = get_train_data_publaynet()
    total_test = no_table + train + word
    random.shuffle(total_test)

    return total_test

def get_train_data_and_latex_pln_empty():

    latex = get_train_data_latex()

    no_table = pickle.load( open("../PubLayNet/COCO_no_table_examples.pkl", "rb" ) )
    for image in no_table:
        image['categories'] = [{'supercategory': '', 'id': 0, 'name': 'table'}]
        image["annotations"] = []
        image["file_name"] = "/data/rali5/Tmp/yockelle/PubLayNet/" + image["file_name"]

    train = get_train_data_publaynet()
    total_test = no_table + train + latex
    random.shuffle(total_test)

    return total_test

def get_train_data_word_and_latex_pln_empty():

    latex = get_train_data_both()

    no_table = pickle.load( open("../PubLayNet/COCO_no_table_examples.pkl", "rb" ) )
    for image in no_table:
        image['categories'] = [{'supercategory': '', 'id': 0, 'name': 'table'}]
        image["annotations"] = []
        image["file_name"] = "/data/rali5/Tmp/yockelle/PubLayNet/" + image["file_name"]

    train = get_train_data_publaynet()
    total_test = no_table + train + latex
    random.shuffle(total_test)

    return total_test

#====================================================================================

def get_test_data_word_specify_table(nb_table):
    new_set =[]
    set_ = get_test_data_word()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) == nb_table:
            new_set.append(set_[i])
    return new_set

def get_test_data_latex_specify_table(nb_table):
    new_set =[]
    set_ = get_test_data_latex()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) == nb_table:
            new_set.append(set_[i])
    return new_set

def get_test_data_publaynet_specify_table(nb_table):
    new_set =[]
    set_ = get_test_data_publaynet()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) == nb_table:
            new_set.append(set_[i])
    return new_set

#====================================================================================

def get_train_data_word_specify_table(nb_table):
    new_set =[]
    set_ = get_train_data_word()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) == nb_table:
            new_set.append(set_[i])
    return new_set

def get_train_data_latex_specify_table(nb_table):
    new_set =[]
    set_ = get_train_data_latex()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) == nb_table:
            new_set.append(set_[i])
    return new_set

def get_train_data_publaynet_specify_table(nb_table):
    new_set =[]
    set_ = get_train_data_publaynet()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) == nb_table:
            new_set.append(set_[i])
    return new_set

#====================================================================================


def get_test_data_word_two_andmore():
    new_set =[]
    set_ = get_test_data_word()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) >= 2:
            new_set.append(set_[i])
    return new_set

def get_test_data_latex_two_andmore():
    new_set =[]
    set_ = get_test_data_latex()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) >= 2:
            new_set.append(set_[i])
    return new_set

def get_test_data_publaynet_two_andmore():
    new_set =[]
    set_ = get_test_data_publaynet()
    for i in range(len(set_)):
        if len(set_[i]["annotations"]) >= 2:
            new_set.append(set_[i])
    return new_set

