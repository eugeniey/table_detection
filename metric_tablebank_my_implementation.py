from shapely.geometry import Polygon
from get_iou import get_max_iou, get_iou, get_overlap
import numpy as np

def metric_table_bank_union(ground_truths, predictions):

    sum_denominator_precision = 0
    sum_denominator_recall = 0
    total_area_union = 0
    old_way_area_union = 0
    
    # Put truths in Polygon
    polygon_ground_truth = []
    for i in range(len(ground_truths)):
        truth = ground_truths[i]
        x_min_truth = truth[0]
        y_min_truth = truth[1]
        x_max_truth = truth[2]
        y_max_truth = truth[3]
        #x_max_truth = x_min_truth + w
        #y_max_truth = y_min_truth + h
        # area
        area_truth = (x_max_truth - x_min_truth) * (y_max_truth - y_min_truth)
        if area_truth < 0:
            print(truth)
        sum_denominator_recall += area_truth
        # poly
        poly = Polygon([(x_min_truth, y_min_truth), (x_min_truth, y_max_truth), (x_max_truth, y_max_truth), (x_max_truth, y_min_truth)])
        polygon_ground_truth.append(poly)

    # Put predictions in Polygon
    polygon_predictions = []
    for i in range(len(predictions)):
        pred = predictions[i]
        x_min_pred = pred[0]
        y_min_pred = pred[1]
        x_max_pred = pred[2]
        y_max_pred = pred[3]
        #x_max_pred = x_min_truth + w
        #y_max_pred = y_min_truth + h
        # area
        area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
        sum_denominator_precision += area_pred
        # poly
        poly = Polygon([(x_min_pred, y_min_pred), (x_min_pred, y_max_pred), (x_max_pred, y_max_pred), (x_max_pred, y_min_pred)])
        polygon_predictions.append(poly)

    
    # Go into truths one by one 
    for t, truth in enumerate(polygon_ground_truth):

        # if they intersect
        intersect_box = []

        # See how many preds matches 
        for pred in polygon_predictions:
            if truth.intersects(pred) > 0:
                intersect = list(truth.intersection(pred).bounds)
                if len(intersect)>=4:
                    x_min = intersect[0]
                    y_min = intersect[1]
                    x_max = intersect[2]
                    y_max = intersect[3]
                    poly = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
                    intersect_box.append(poly)

        # go trough all the intersections
        area_union_of_predictions_and_ground_truth = 0
        
        # If multiple intersection
        if len(intersect_box) > 1:
            first = intersect_box[0]
            for k in range(len(intersect_box)):
                inter = intersect_box[k]
                if k == 0:
                    continue
                if k == 1:
                    uni = first.union(inter)
                if k > 1:
                    uni = uni.union(inter)
            area_union_of_predictions_and_ground_truth = uni.area
            
        # If one intersection
        elif len(intersect_box) == 1:
            intersection_polygon = intersect_box[0]
            area_union_of_predictions_and_ground_truth = intersection_polygon.area

        # If no preds
        else:
            area_union_of_predictions_and_ground_truth = 0

        ########
        # Get the max iou
        truth = ground_truths[t]
        result = get_max_iou(predictions, truth)

        if result is not None:

            iou_max = result[1]
            index = result[2]
            associated_pred = predictions[index]

            x_min = associated_pred[0]
            y_min = associated_pred[1]
            x_max = associated_pred[2]
            y_max = associated_pred[3]
            poly_pred = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

            x_min = truth[0]
            y_min = truth[1]
            x_max = truth[2]
            y_max = truth[3]
            poly_truth = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

            if iou_max>0.4:
                overlap = poly_truth.intersection(poly_pred).area
                old_way_area_union += overlap
        ########


        # for a ground truth
        total_area_union += area_union_of_predictions_and_ground_truth

    sum_numerator = total_area_union  
                
    return sum_numerator, sum_denominator_precision, sum_denominator_recall, old_way_area_union





"""
def metric_table_bank_union(ground_truths, predictions):

    sum_denominator_precision = 0
    sum_denominator_recall = 0
    total_area_union = 0
    old_way_area_union = 0
    
    # Put truths in Polygon
    polygon_ground_truth = []
    for i in range(len(ground_truths)):
        truth = ground_truths[i]
        x_min_truth = truth[0]
        y_min_truth = truth[1]
        x_max_truth = truth[2]
        y_max_truth = truth[3]
        # area
        area_truth = (x_max_truth - x_min_truth) * (y_max_truth - y_min_truth)
        sum_denominator_recall += area_truth
        # poly
        poly = Polygon([(x_min_truth, y_min_truth), (x_min_truth, y_max_truth), (x_max_truth, y_max_truth), (x_max_truth, y_min_truth)])
        polygon_ground_truth.append(poly)

    # Put predictions in Polygon
    polygon_predictions = []
    for i in range(len(predictions)):
        pred = predictions[i]
        x_min_pred = pred[0]
        y_min_pred = pred[1]
        x_max_pred = pred[2]
        y_max_pred = pred[3]
        # area
        area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
        sum_denominator_precision += area_pred
        # poly
        poly = Polygon([(x_min_pred, y_min_pred), (x_min_pred, y_max_pred), (x_max_pred, y_max_pred), (x_max_pred, y_min_pred)])
        polygon_predictions.append(poly)

    
    # Go into truths one by one 
    for t, truth in enumerate(polygon_ground_truth):

        # if they intersect
        intersect_box = []

        # See how many preds matches 
        for pred in polygon_predictions:
            if truth.intersects(pred) > 0:
                intersect = list(truth.intersection(pred).bounds)
                x_min = intersect[0]
                y_min = intersect[1]
                x_max = intersect[2]
                y_max = intersect[3]
                poly = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
                intersect_box.append(poly)

        # go trough all the intersections
        area_union_of_predictions_and_ground_truth = 0
        
        # If multiple intersection
        if len(intersect_box) > 1:
            first = intersect_box[0]
            for k in range(len(intersect_box)):
                inter = intersect_box[k]
                if k == 0:
                    continue
                if k == 1:
                    uni = first.union(inter)
                if k > 1:
                    uni = uni.union(inter)
            area_union_of_predictions_and_ground_truth = uni.area
            
        # If one intersection
        elif len(intersect_box) == 1:
            intersection_polygon = intersect_box[0]
            area_union_of_predictions_and_ground_truth = intersection_polygon.area

        # If no preds
        else:
            area_union_of_predictions_and_ground_truth = 0


        ########
        # Get the max iou
        truth = ground_truths[t]
        result = get_max_iou(np.array(predictions), np.array(truth))

        if result is not None:

            iou_max = result[1]
            index = result[2]
            associated_pred = predictions[index]

            x_min = associated_pred[0]
            y_min = associated_pred[1]
            x_max = associated_pred[2]
            y_max = associated_pred[3]
            poly_pred = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

            x_min = truth[0]
            y_min = truth[1]
            x_max = truth[2]
            y_max = truth[3]
            poly_truth = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

            if iou_max>0.4:
                overlap = poly_truth.intersection(poly_pred).area
                old_way_area_union += overlap
                #if overlap> area_union_of_predictions_and_ground_truth:
                #    print(iter_)
        ########


        # for a ground truth
        total_area_union += area_union_of_predictions_and_ground_truth

    sum_numerator = total_area_union  
                
    return sum_numerator, sum_denominator_precision, sum_denominator_recall, old_way_area_union
"""


"""
def measure_metric_table_bank(groud_truths, predictions):

  count_prediction = len(predictions)
  count_good_prediction = 0

  sum_numerator = 0
  sum_denominator_precision = 0
  sum_denominator_recall = 0
  
  for pred in predictions:
    x_min_pred = pred[0]
    y_min_pred = pred[1]
    x_max_pred = pred[2]
    y_max_pred = pred[3]
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    sum_denominator_precision += area_pred

  ## Measure IOU
  for j in range(len(groud_truths)):
    truth = groud_truths[j]

    # Get the max iou
    result = get_max_iou(predictions, truth)

    # Get the sum of the truth
    x_min_truth = truth[0]
    y_min_truth = truth[1]
    x_max_truth = truth[2]
    y_max_truth = truth[3]
    area_truth = (x_max_truth - x_min_truth) * (y_max_truth - y_min_truth)
    sum_denominator_recall += area_truth

    if result is not None:

      iou_max = result[1]
      index = result[2]
      associated_pred = predictions[index]

      if iou_max>0.4:
        # Get the sum of the truth
        x_min = associated_pred[0]
        y_min = associated_pred[1]
        x_max = associated_pred[2]
        y_max = associated_pred[3]
        poly_pred = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

        x_min = truth[0]
        y_min = truth[1]
        x_max = truth[2]
        y_max = truth[3]
        poly_truth = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

        if iou_max>0.4:
            #overlap = get_overlap(associated_pred, truth)
            overlap = poly_truth.intersection(poly_pred).area
            sum_numerator += overlap
          
      
  if count_prediction > 0:
    precision = sum_numerator/sum_denominator_precision
    recall = sum_numerator/sum_denominator_recall

    return precision, recall, sum_numerator, sum_denominator_precision, sum_denominator_recall

  else:
    return  0, 0, 0, sum_denominator_precision, sum_denominator_recall
"""