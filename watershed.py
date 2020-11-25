# pip install --upgrade opencv-contrib-python

import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from PIL import Image
import time

print(cv2.__version__)

img = cv2.imread('C:/Users/kimth/Desktop/1/input/9.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('a', thresh)
cv2.waitKey(0)

# noise removal
kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kernel, iterations = 3)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)

cv2.imshow('a', opening)
cv2.waitKey(0)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

cv2.imshow('a', sure_bg)
cv2.waitKey(0)


# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.02*dist_transform.max(),255,0)

cv2.imshow('a', sure_fg)
cv2.waitKey(0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.imshow('a', sure_fg)
cv2.waitKey(0)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imshow('a', img)
cv2.waitKey(0)

cv2.imwrite("asd.jpg", img)

gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    for m in range(len(Train_files)):
        gray = cv2.imread(Train_files[m], cv2.IMREAD_GRAYSCALE)
        for thres in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255]:
        #for thres in range(a):
            ret, gray_th = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY_INV)
            col = thres/5
            b = cv2.mean(gray_th, mask=None)
            ws.cell(row=1 + m, column=col+3).value = b[0]/255
        print("현재", m+1, "번째", "time :", time.time() - start)
        wb.save("result_VA.xlsx")
    break