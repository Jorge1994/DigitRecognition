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
        print("click")
    elif event ==  cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(background,(previous_x,previous_y),(x,y), (0,0,0), 5)
            previous_x = x
            previous_y = y
        print("mouse move")
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(background,(previous_x,previous_y),(x,y), (0,0,0), 5)
        previous_x = x
        previous_y = y
        print("release")
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
    
cv2.destroyAllWindows()