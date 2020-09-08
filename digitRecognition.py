# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:02:07 2020

@author: PJ
"""

import cv2
import numpy as np
import tensorflow as tf
import math
from tensorflow_core.python.keras.layers.image_preprocessing import ResizeMethod

WINDOW_NAME = "Digit Recognition"
drawing = False 
background = np.ones((512,512), dtype = "uint8") * 255 #Create a white background

model = tf.keras.models.load_model('digit_model.h5')

def mouse_draw(event, x, y, flags, param):
    global previous_x, previous_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        previous_x, previous_y = x, y
    elif event ==  cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(background,(previous_x,previous_y),(x,y), (0,0,0), 10)
            previous_x = x
            previous_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(background,(previous_x,previous_y),(x,y), (0,0,0), 10)
        previous_x = x
        previous_y = y
    return x,y

def reshape_image(image):
    org_size = 22
    img_size = 28
    rows,cols = image.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    
    image = cv2.resize(image, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply apdding 
    image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')
    return image

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_draw)

while(True):
    cv2.imshow(WINDOW_NAME, background)
    key = cv2.waitKey(1)& 0xFF
    if key == 27: #Escape KEY
        break
    if key == ord('c'):
        background = np.ones((512,512), dtype = "uint8") * 255
    if key == ord('r'):
        ret, thresh = cv2.threshold(background, 0, 255, cv2.THRESH_BINARY_INV)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Hierarchy = [Next, Previous, First_Child, Parent]
        for counter, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            rec_area = w*h
            #if (hierarchy[0][counter][3] != -1 and rec_area > 100):
            roi = background[y:y+h, x:x+w]
            roi = cv2.bitwise_not(roi)
            roi = reshape_image(roi)
            test_image = roi.reshape(-1,28,28,1)
            predictions = model.predict(test_image)
            label = np.argmax(predictions[0])
            cv2.putText(background,str(label),(x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),1,cv2.LINE_AA)
            cv2.imshow("Roi:"+str(counter), roi)
            cv2.rectangle(background,(x,y),(x+w,y+h),(0,0,0),2)
            #cv2.imshow("Thresh", background)
    
cv2.destroyAllWindows()