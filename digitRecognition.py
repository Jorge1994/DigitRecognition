# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:02:07 2020

@author: PJ
"""

import cv2
import numpy as np

WINDOW_NAME = "Digit Recognition"
drawing = False 
background = np.ones((512,512), dtype = "uint8") * 255 #Create a white background

def mouse_draw(event, x, y, flags, param):
    global previous_x, previous_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        previous_x, previous_y = x, y
    elif event ==  cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(background,(previous_x,previous_y),(x,y), (0,0,0), 5)
            previous_x = x
            previous_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(background,(previous_x,previous_y),(x,y), (0,0,0), 5)
        previous_x = x
        previous_y = y
    return x,y

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
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for counter, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            #TODO: remove small rectangles
            roi = background[y:y+h, x:x+w]
            roi = cv2.bitwise_not(roi)
            cv2.imshow("Roi:"+str(counter), roi)
            cv2.rectangle(background,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow("Thresh", background)
    
cv2.destroyAllWindows()