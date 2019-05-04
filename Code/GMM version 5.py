# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:32:32 2019

@author: andre
"""

#%% --------------------------------- IMPORT ---------------------------------

import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np
import time

#%% ----------------------------- INITIALIZATION -----------------------------

source = './webcam/web12/'
frame_rate = 5
take_freq = 2
threshold = 70
backgroundRatio = 0.7

originals = []
backgrounds = []

originals,backgrounds, foregrounds = backgroundSubtraction(source,frame_rate,take_freq,threshold,backgroundRatio)

#%% ------------------------- BACKGROUND SUBTRACTION -------------------------

####################################################################################
#
# originals,backgrounds backgroundSubtraction (path_webcam, frame_rate, take_freq, threshold, ratio)
#                         (eg: ./Webcam/web7)
#
# params: path_webcam ->  path of webcam folder
# params: frame_rate  ->  how many frames take
# params: take_freq   ->  how often takes frame for originals and backgrounds 
# params: threshold   ->  threshold for mog2
# params: ratio       ->  background ratio for mog2
#
# return: originals   ->  frame of original video, used for measure the accuracy
# return: backgrounds ->  background detected
#
####################################################################################

def backgroundSubtraction (source,frame_rate,take_freq,threshold,ratio):

    videoList = os.listdir(source)

    mog = cv.createBackgroundSubtractorMOG2()    
    mog.setVarThreshold(threshold)
    #mog.setHistory(5)
    mog.setBackgroundRatio(ratio)
    
    videoPaths = []
    
    original_frames = []
    bg_frames = []
    fg_frames = []
    means = []

    for video in videoList:
        pathVideo = '{}/{}'.format(source, video)
        videoPaths.append(pathVideo)
        
    index = 0
    videoPaths.sort()
    num_video = len(videoPaths)

    
    takeFlag = False;
    
    print('\n')
    print('-------------------------------------------------')
    print('|\t\t\t\t\t\t|')
    print('|\t  START BACKGROUND SUBTRACTION\t\t|')
    print('|\t\t\t\t\t\t|')
    print('-------------------------------------------------')
    print('|\t\t\t\t\t\t|')
    start_time = time.time()
    
#    num_video = 100
    #for all video
    while (index < num_video):
        
        cap = cv.VideoCapture(videoPaths[index])
        
        print('|  '+str(index+1)+'-  load video:   '+videoPaths[index]+'\t|')
        
        index += 1
        
        end = False
        frameCount = 0;
        
        
        if(index%take_freq==0):
            takeFlag = True;
            
        while(end == False):
            ret, frame = cap.read()
            if(ret == False):
                end = True
            else:
                if(frameCount%frame_rate==0):
                    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#                    frames.append(gray)
                    
                    fg = mog.apply(gray)
                    bg = mog.getBackgroundImage()
                    
                    cv.imshow('Original video', gray) 
                    cv.imshow('Background detected', bg)
                    cv.imshow('Foreground detected', fg)                     
                    
                    k = cv.waitKey(30)
    
                    # Press ESC to terminate
                    # Press S to save background (useless, is done in automatic way)
                    
                    if(takeFlag == True):
                        takeFlag = False
                        original_frames.append(gray)
                        fg_frames.append(fg)
                        bg_frames.append(bg)
                    
                    if(k == 27):
                        break
                    
                    if(k == 115):
                        bg_frames.append(bg)
                        fg_frames.append(fg)
                        original_frames.append(frame)
                    
            frameCount += 1
            
        cap.release()
        
        mog.setBackgroundRatio(ratio)        
        means.append(dynamicRatioEvaluation(gray))
        
        endVideo_time = time.time()
        print("|\t Spent time up to now: {:.2f} sec".format(endVideo_time - start_time))                            
        
                
        
        if(k == 27):
            break      

    cv.destroyAllWindows()
    
    print('|\t\t\t\t\t\t|')
    print('-------------------------------------------------')
    print('|\t\t\t\t\t\t|')
    print('|\t  END BACKGROUND SUBTRACTION\t\t|')
    print('|\t\t\t\t\t\t|')
    print('-------------------------------------------------')
    
    return original_frames, bg_frames, fg_frames

#
# Evaluation of sunset time and sunrise time, in order to change the background
# ratio dinamically
#
# It's work taking into account some row of the sky, so it depends from every 
# camera view
#
# Web12 --> 7 row of the upper pixel 
#
#   Average > 180 --> morning/afternoon
#           < 180 --> afternoon to night (sunset)  
#
def dynamicRatioEvaluation(frame):
    points = []
    
    for i in range(1,100):
        row = int(np.random.uniform(180-40, 180+40))
        column = int(np.random.uniform(320-40, 320+40))
        
        points.append(frame[row][column])
    
    mean = int(np.mean(points))
    print("Mean: " + str(mean))
    
    return mean
#%%
showImage(backgrounds[10])
dynamicRatioEvaluation(backgrounds[10])
showImage(backgrounds[15])
dynamicRatioEvaluation(backgrounds[15])
showImage(backgrounds[22])
dynamicRatioEvaluation(backgrounds[22])
showImage(backgrounds[23])
dynamicRatioEvaluation(backgrounds[23])
showImage(backgrounds[33])
dynamicRatioEvaluation(backgrounds[33])
showImage(backgrounds[44])
dynamicRatioEvaluation(backgrounds[44])
showImage(backgrounds[51])
dynamicRatioEvaluation(backgrounds[51])

showImage(backgrounds[14])
dynamicRatioEvaluation(backgrounds[14])

#%%
def showImage(frame):
    
    while(True):
        cv.imshow("image", frame)
        
        k = cv.waitKey(30)
        if(k ==27):
            break
        
    cv.destroyAllWindows()


#%%
means = []
for frame in backgrounds:
    means.append(dynamicRatioEvaluation(frame))

plt.hist(means)

#%%

hist = cv.calcHist(means,[0],None, [256], [0,256])
plt.plot(hist)
#showImage(hist)
#%%
list_image = backgrounds
i = 0
cycle = True

while(cycle):
    
    cv.imshow('Backgroung', backgrounds[i])
    cv.imshow('Foregroung', foregrounds[i])
    cv.imshow('Original', originals[i])    
    cv.moveWindow('Backgroung', 700,10)
    cv.moveWindow('Foregroung', 0,380)
    cv.moveWindow('Original', 0,5)

    print("Frame {}".format(i))    
    
    k = cv.waitKey(30)
    
    while(k != 97 and k != 115 and k != 27):
                k = cv.waitKey(30) & 0xff
        
                # key = s
                if(k == 115 and i < len(list_image)  - 1):
                        i = i + 1   # go to next frame
                        
                # key = a
                elif(k == 97 and i > 0):
                        i = i - 1   # go to previous frame
                        
                # key = ESC
                elif(k == 27):
                    print('Closing cv windows')
                    cv.destroyAllWindows()
                    cycle = False
                    break
    
cv.destroyAllWindows()
    
    