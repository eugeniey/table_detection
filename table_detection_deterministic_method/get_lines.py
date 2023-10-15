# importing required libraries
import cv2
import numpy as np


def get_HoughLines(image_name):
    """
    Return the horizontal and vertical lines given by HoughLines after going through a little cleaning 
    """
    img = cv2.imread(image_name)

    vertical_lines = []
    horizontal_lines = []
  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines_hough = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

    # Remove useless dimension
    lines_regular = lines_hough[:,0,:]

    # Separate into vertical and horizontal lines
    for line in lines_regular:
        line = list(line)
        # if vertical lines, x1 == x2
        if line[0] == line[2]:
            vertical_lines.append(line)
        # if horizontal lines, y1 == y2
        if line[1] == line[3]:
            horizontal_lines.append(line)

    vertical_lines   = np.array(vertical_lines)
    horizontal_lines = np.array(horizontal_lines)
    horizontal_lines_cleaned = []
    vertical_lines_cleaned   = []
    
    # Bad way to look if the current line was removed
    lines_to_remove = [[1,1,1,1]]
    
    # Sort on y1 and remove useless lines
    # for horizontal lines
    if len(horizontal_lines)>0:    
        # Sort
        horizontal_lines = horizontal_lines[np.lexsort((horizontal_lines[:,1],horizontal_lines[:,1]))]
        horizontal_lines_cleaned = horizontal_lines.tolist()
        
        # Remove useless lines
        for line in horizontal_lines:
            line = list(line)
            x1, y1, x2, y2 = line

            # if the line was removed, let's skip it
            if line in lines_to_remove:
                continue
            # Do I want to remove the line from the cleaned horizontal lines
            # remove lines that are smaller than current and almost at the same position
            lines_to_remove = [lin for lin in horizontal_lines_cleaned if (list(line)!=list(lin)) \
                               and (abs(y1 - lin[1])<6) \
                               and (abs(x2-x1)>abs(lin[2]-lin[0]) or abs(x2-x1)==abs(lin[2]-lin[0]))]

            # Remove 
            for line_to_remove in lines_to_remove:
                horizontal_lines_cleaned.remove(line_to_remove)

    # Bad way to look if the current line was removed
    lines_to_remove = [[1,1,1,1]]
    
    # Sort on x1 and remove useless lines
    # for vertical lines
    if len(vertical_lines)>0:
        # Sort
        vertical_lines[:, [3, 1]] = vertical_lines[:, [1, 3]]
        vertical_lines = vertical_lines[np.lexsort((vertical_lines[:,0],vertical_lines[:,0]))]
        vertical_lines_cleaned = vertical_lines.tolist()
        
        # Remove useless lines
        for line in vertical_lines:
            line = list(line)
            x1, y1, x2, y2 = line
            
            # if the line was removed, let's skip it
            if line in lines_to_remove:
                continue
            # Do I want to remove the line from the cleaned horizontal lines
            # remove lines that are smaller than current and almost at the same position
            lines_to_remove = [lin for lin in vertical_lines_cleaned if (list(line)!=list(lin)) \
                               and (abs(x1 - lin[0])<6) \
                               and (abs(y2-y1)>abs(lin[3]-lin[1]) or abs(y2-y1)==abs(lin[3]-lin[1]))]
            # Remove
            for line_to_remove in lines_to_remove:
                vertical_lines_cleaned.remove(line_to_remove)
                
    return horizontal_lines_cleaned, vertical_lines_cleaned


def does_add_vertical_line(dictionnary, vertical_lines, line, index):
    """
    After adding a new horizontal line to the tab,
    Look if we need to add vertical lines in that table

    dictionnary:    dict of detected tables 
    vertical_lines: arrays of all the vertical lines 
    line:           current horizontal lines added
    index:          index of current table

    return dictionnary and vertical_lines
    """
    x1, y1, x2, y2 = line

    # If we have vertical lines close to the added horizontal line
    match = [lin for lin in vertical_lines if \
             (abs(y1 - lin[1])<20 or abs(y1 - lin[3])<20 or (lin[1]<y1 and lin[3]>y1)) \
             and (abs(x1 - lin[0])<20 or abs(x2 - lin[0])<20 or (lin[0]>x1 and lin[0]<x2))]
    
    for verti_line in match:
        # Remove the added vertical lines from the pool of vertical lines
        vertical_lines.remove(verti_line)
        x1_v,y1_v,x2_v,y2_v = verti_line

        if "vertical" in dictionnary[index]:
            dictionnary[index]["vertical"].append([x1_v,y1_v,x2_v,y2_v])
        else:
            dictionnary[index]["vertical"] = [[x1_v,y1_v,x2_v,y2_v]]

    return dictionnary, vertical_lines


def get_relevant_lines(image_name, return_bounding_box=False):
    """
    Return dictionnary that contains all relevant lines
    Key of the dictionnary are integer that represents tables
    Example:
    {0: {"horizontal":[[xmin, xmax, ymin, ymax],[xmin, xmax, ymin, ymax]]
        "vertical":[[xmin, xmax, ymin, ymax],[xmin, xmax, ymin, ymax]]}
    1: {"horizontal":[[xmin, xmax, ymin, ymax],[xmin, xmax, ymin, ymax]]
        "vertical":[[xmin, xmax, ymin, ymax],[xmin, xmax, ymin, ymax]]}  
    ETC  
    }
    where the keys 0 and 1 represents a table, and the value is a dictionnary that has 2 keys "horizontal" and "vertical" where the values are a list of list of the lines

    if return_bounding_box is True, we only return a list of the bounding boxes of the tables
    """
    
    # Get lines of image
    horizontal_lines, vertical_lines = get_HoughLines(image_name)

    # Dictionnary that will contain the detected tables
    count_of_tab = 0
    dict_of_tables = {}

    # Loop through horizontal lines
    # and create table 
    for i, line in enumerate(horizontal_lines):
        
        x1, y1, x2, y2 = line
        found_match = 0
        is_there_close_vertical = 0

        # at least 70
        if (x2 - x1)>70:
            
            # if first iteration
            # create new table
            if len(dict_of_tables) == 0:
                dict_horizontal = {}
                dict_horizontal["horizontal"] = [[x1,y1,x2,y2]]
                dict_of_tables[count_of_tab] = dict_horizontal
                dict_of_tables, vertical_lines = does_add_vertical_line(dict_of_tables, vertical_lines, line, count_of_tab)
            
            # if not first iteration
            else:
                # Loop through predicted tab to match current line
                for key in dict_of_tables.copy():
                    ############################
                    # TAKE CURRENT INFORMATION
                    # if current is close to a vertical line
                    is_there_close_vertical = 0
                    
                    # take the last horizontal line of the predicted tab
                    last_horizontal_lines_of_predicted_tab = dict_of_tables[key]["horizontal"][-1]
                    x1_preced, y1_preced, x2_preced, y2_preced = last_horizontal_lines_of_predicted_tab
                    
                    # take the vertical info if exists
                    if "vertical" in dict_of_tables[key]:
                        vertical_lines_of_predicted_tab = dict_of_tables[key]["vertical"]
                        # if current is close to a vertical line
                        matched = [lin for lin in vertical_lines_of_predicted_tab if \
                            (abs(y1 - lin[3])<15 or abs(y1 - lin[1])<15 or (lin[1]<y1 and lin[3]>y1)) \
                            and ((abs(x1 - lin[0])<15 or abs(x2 - lin[0])<15) or (lin[0]>x1 and lin[0]<x2)) ]
                        
                        if len(matched)>0:
                            is_there_close_vertical = 1 
                            
                    ############################
                    # FIND WHERE TO MATCH

                    # if current is close to the last horizontal line
                    if abs(y1_preced - y1)<80 and (abs(x1_preced-x1)<60 or abs(x2_preced-x2)<60):
                        dict_of_tables[key]["horizontal"].append([x1,y1,x2,y2])
                        dict_of_tables, vertical_lines = does_add_vertical_line(dict_of_tables, vertical_lines, line, count_of_tab)
                        found_match = 1
                        
                    # if current is close to vertical line of current tab
                    elif is_there_close_vertical:
                        dict_of_tables[key]["horizontal"].append([x1,y1,x2,y2])
                        dict_of_tables, vertical_lines = does_add_vertical_line(dict_of_tables, vertical_lines, line, count_of_tab)
                        found_match = 1
                    
                    # if it's the last line
                    # and almost same size as previous line
                    elif i+1 == len(horizontal_lines) and (abs(x1_preced-x1)<20 or abs(x2_preced-x2)<20):
                        the_key = len(dict_of_tables)-1
                        dict_of_tables[the_key]["horizontal"].append([x1,y1,x2,y2])
                        found_match = 1
                        
                    # if we found a match, stop going through existing tables
                    if found_match:
                        break
                        
                # if current doesn't match with any table
                # create new tab
                if not found_match:
                    count_of_tab += 1
                    dict_horizontal = {}
                    dict_horizontal["horizontal"] = [[x1,y1,x2,y2]]
                    dict_of_tables[count_of_tab] = dict_horizontal
                    dict_of_tables, vertical_lines = does_add_vertical_line(dict_of_tables, vertical_lines, line, count_of_tab)

    # To get bounding boxes
    if return_bounding_box:
        predictions = []
        for key in dict_of_tables.copy():
            hori_and_verti = dict_of_tables[key]
            lines_hori = []
            line_verti = []
            
            sorted_x = []
            sorted_y = []
            
            if "horizontal" in hori_and_verti:
                lines_hori = hori_and_verti["horizontal"]
                if len(lines_hori)<2:
                    continue
                for line_h in lines_hori:
                    x1_f, y1_f, x2_f, y2_f = line_h
                    sorted_x.append(x1_f)
                    sorted_x.append(x2_f)
                    sorted_y.append(y1_f)
                    sorted_y.append(y2_f)

            if "vertical" in hori_and_verti:
                line_verti = hori_and_verti["vertical"]
                for line_v in line_verti:
                    x1_v, y1_v, x2_v, y2_v = line_v
                    sorted_x.append(x1_v)
                    sorted_x.append(x2_v)
                    sorted_y.append(y1_v)
                    sorted_y.append(y2_v)

            sorted_x.sort()
            sorted_y.sort()
            x_min = sorted_x[0]
            y_min = sorted_y[0]
            x_max = sorted_x[-1]
            y_max = sorted_y[-1]

            if x_max-x_min>10 and y_max-y_min>10:
                predictions.append([x_min, y_min, x_max, y_max])
    
        return predictions
    
    # If we just want all the lines
    else:
        return dict_of_tables