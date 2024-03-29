# Note: Image name will be stored as "Dilation_OriginalName" to avoid confict

import cv2
import glob
import numpy as np 

# DEFINE THE PATH
PATH_TO_DEST = "/data/rali5/Tmp/yockelle/TableBank/TableBank/Detection/dilated_images/"
PATH_TO_ORIGIAL_IMAGES = "/data/rali5/Tmp/yockelle/TableBank/TableBank/Detection/images/"

# if the source directory have other files than images, use extenstion of image 
# to get the files ( for example *.png )
img_files = glob.glob(PATH_TO_ORIGIAL_IMAGES+"*.*")
total = len(img_files)

# 2x2 Static kernal
kernal = np.ones((2,2),np.uint8)

for count,i in enumerate(img_files):
  try:
    image_name = i.split("/")[-1]
    if count%1000==0:
      print("Progress : ",count,"/",total)
    img = cv2.imread(i,0)
    _, mask = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    dst = cv2.dilate(mask,kernal,iterations = 1)
    dst = cv2.bitwise_not(dst)
    cv2.imwrite(PATH_TO_DEST+"/Dilation_"+image_name,dst)
  except:
    continue
