# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:08:53 2021

@author: ashwi
"""

import cv2
import imutils
import glob
import os

def draw_color_mask(img, borders, color=(0, 0, 0)):
   
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):

    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:       
        for radius in gaussian_blur_radius_list:   
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    gray = draw_color_mask(gray, black_mask)
    
    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):

    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []

    for c in cnts:
        
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)
        
    return score, res_cnts, thresh

# Function to create folders
def create_folder(folder):  
    os.mkdir(folder)
    return folder


Input_Images = [cv2.imread(file) for file in glob.glob(r'G:\Tasks\Kopernikus\c23\*.png')] # Path to dataset

Duplicate_folder = create_folder(r'G:\Tasks\Kopernikus\c23\Duplicate_Images')  # Ducplicate_folder, to save the duplicate images in dataset 
Unique_folder = create_folder(r'G:\Tasks\Kopernikus\c23\Unique_Images')        # Unique_folder, to save the unique images in dataset

Outpt_compare_frame_func = []   


for i in range(len(Input_Images)):
    temp_image = Input_Images[i]
    preprocess_func_result = preprocess_image_change_detection(temp_image,(11,11),black_mask=(5, 10, 5, 0))
    Outpt_compare_frame_func.append(preprocess_func_result)  # Output_compare_frames_func list consists of the output images from the previous functiuons


for i in range(len(Outpt_compare_frame_func)-1):
    previous_frame = (Outpt_compare_frame_func[i])
    next_frame = (Outpt_compare_frame_func[i+1])
    compare_func_result = compare_frames_change_detection(previous_frame,next_frame,300)
    score, res_cnts, thresh = compare_func_result
    #print(i,i+1,score)
    if score == 0:
        #print(i,i+1,score)        
        path = r"G:\Tasks\kopernikus\c23\Duplicate_Images\\"+str(i)+".png"
        cv2.imwrite(path,Outpt_compare_frame_func[i])

    else:
        #print(i,i+1,score)
        path = r"G:\Tasks\Kopernikus\c23\Unique_Images\\"+str(i)+".png"
        cv2.imwrite(path,Outpt_compare_frame_func[i])  
      

path1,dirs1,file1 = next(os.walk(Duplicate_folder))
path2,dirs2,file2 = next(os.walk(Unique_folder))
print("The dataset has",len(file2),"Unique Images and",len(file1),"Duplicate Images")  



