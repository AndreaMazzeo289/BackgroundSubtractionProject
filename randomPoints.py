# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:31:52 2019

@author: daniele
"""
#==============================================================================
#%% IMPORT
#==============================================================================
import numpy as np

#%%

def computePixelIntensityAverage(frame):
    points_value = []
    points_coord = []
    
    for i in range(1,100):
        
        row_left_up = int(np.random.uniform(30, 110))
        column_left_up= int(np.random.uniform(30, 110))
        points_coord.append([row_left_up, column_left_up])
        
        row_left_down = int(np.random.uniform(250, 330))
        column_left_down = int(np.random.uniform(30, 110))
        points_coord.append([row_left_down, column_left_down])        
        
        row_center = int(np.random.uniform(180-40, 180+40))
        column_center = int(np.random.uniform(320-40, 320+40))
        points_coord.append([row_center, column_center])
        
        row_right_up = int(np.random.uniform(30, 110))
        column_right_up= int(np.random.uniform(530, 610))
        points_coord.append([row_right_up, column_right_up])        
        
        row_right_down = int(np.random.uniform(250, 330))
        column_right_down= int(np.random.uniform(530, 610))
        points_coord.append([row_right_down, column_right_down])                       
        
        points_value.append(frame[row_center][column_center])
        points_value.append(frame[row_left_down][column_left_down])
        points_value.append(frame[row_left_up][column_left_up])
        points_value.append(frame[row_right_up][column_right_up])
        points_value.append(frame[row_right_down][column_right_down])
    
    
    drawPoints(frame, points_coord) 
    avg_pixel = int(np.mean(points_value))
    
    return avg_pixel

#==============================================================================
# ---------------------SQUARES FUNCTIONS--------------------------------------
#==============================================================================
def drawPoints(frame, points):
    
    temp = np.copy(frame)
    
    for point in points:
        x = point[1]
        y = point[0]
        
        temp[y][x] = 255
        
#    showImage(temp)

def drawAllRectangles(frame):
    # Upper left
    temp = drawRectangle(frame, 30, 110, 30, 110)
    # Lower left 
    temp = drawRectangle(temp, 30, 110, 250, 330)
    # Upper right
    temp = drawRectangle(temp, 530, 610, 30, 110)
    # Lower right
    temp = drawRectangle(temp, 530, 610, 250, 330)
    # Center
    temp = drawRectangle(temp, 280, 360, 140, 220)
    
    return temp

def drawRectangle(frame, xA, xB, yA, yB):

    temp = np.copy(frame)
#    for i in range(yA, yB):   
    for j in range(xA, xB):
        temp[yA][j] = 255
        temp[yB][j] = 255
        
    for j in range(yA, yB):
        temp[j][xA] = 255
        temp[j][xB] = 255
        
    return temp;
#==============================================================================
# 
#==============================================================================