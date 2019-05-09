# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:14:21 2019

@author: daniele
"""
#%% IMPORTS

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import tarfile
import pickle

#%%
#==============================================================================
#--------------------------GENERATE DAY/NIGHT GAUSSIANS------------------------
#   This function generate the two gaussian realtives to the trading of pixel intensity
#   from the day to the night
#   
#   PARAMETERS:
#
# - Avg_pixel_day       -->   average of random pixel values belonging to the daytime
# - Avg_pixel_night     -->   average of random pixel values belonging to the nighttime 
#
#   RETURN:
#   
# - Gauss_day       -->   gaussian distribution of the pixel intensity of the day
# - Gauss_night     -->   gaussian distribution of the pixel intensity of the night
#
#==============================================================================
def generateDayNightGaussians(avg_pixel_day, avg_pixel_night):
    
    mu_day = int(np.mean(avg_pixel_day))
    sigma_day = np.std(avg_pixel_day)
    gauss_day = norm(mu_day, sigma_day)    

    mu_night = int(np.mean(avg_pixel_night))
    sigma_night = np.std(avg_pixel_night)
    gauss_night = norm(mu_night, sigma_night)

    return gauss_day, gauss_night
    
#==============================================================================
#-------------------------PLOT DAY/NIGHT GAUSSIANS-----------------------------
#   Plot the gaussian distributions of the pixel intensity belonging to day and
#   night
#    
#   PARAMETERS:
#        
# - Gauss_day       -->   gaussian distribution of the pixel intensity of the day
# - Gauss_night     -->   gaussian distribution of the pixel intensity of the night
#
#==============================================================================
def plotDayNightGaussian(gauss_day, gauss_night):
   
    x_axis = np.linspace(0,255,255)     #255 is the max luminosity    
    plt.xlim(10,150)
    
    y_day = gauss_day.pdf(x_axis)
    y_night = gauss_night.pdf(x_axis)              # get the norm.pdf for x interval 

    plt.plot(x_axis, y_day, color = 'orange')
    plt.plot(x_axis, y_night, color = 'b')        
    plt.show()

#==============================================================================
#-----------------------------GET PROBABILITY----------------------------------
#
#   Given a pixel intensity and a gaussian distribution, you get the probability
#   of that itensity
#
#   PARAMETERS:
#
#   - pixel_intesity   -->   pixel intensity in the range [0,255]
#   - gauss_distr      -->   distrubution used to evaluate the probability
#   - color            -->   color for the plotted distribution
#
#==============================================================================
def getProbability(pixel_intensity, gauss_distr, color = 'r'):
    
    x_axis = np.linspace(0,255,255)
    y_axis = gauss_distr.pdf(x_axis)
    plt.plot(x_axis, y_axis, color = color)
    
    prob = y_axis[pixel_intensity]
    print("Probability\t--> \t{}\nPixel intensity --> \t{}\n".format(prob, pixel_intensity))


def getPixelIntensity(probability, gauss_distr, color = 'r'):
    
    pixel_intensity = gauss_distr.pdf(probability)
    
    print("Probability\t--> \t{}\nPixel intensity --> \t{}\n".format(probability, pixel_intensity))

#==============================================================================
#--------------------------SPLIT AVERAGE PIXELS--------------------------------
#   Splitting of the pixel intensity average into average belonging to the daytime
#   and to the nighttime
#
#   PARAMETERS:
#
#   - avg_pixels   -->   pixel intesity average of all frame
#
#   RETURN:
#
#   - Gauss_day       -->   gaussian distribution of the pixel intensity of the day
#   - Gauss_night     -->   gaussian distribution of the pixel intensity of the night
#   - Y_day           -->   y_axis of gaussian distribution belonging to the day
#   - Y_night           -->   y_axis of gaussian distribution belonging to the night
#
#==============================================================================
def splitAvgPixels(avg_pixels):
    day = []
    night = []    
    
    # the mid point splits avg pixel in avg pixel of day from night
    for avg_pixel in avg_pixels:
        mid_point = int((max(avg_pixels) - min(avg_pixels))/2)
        split = mid_point + min(avg_pixels)
        
        if(avg_pixel > split):        
            day.append(avg_pixel)
        elif(avg_pixel < split):
            night.append(avg_pixel)
            
    gauss_day, gauss_night = generateDayNightGaussians(day, night)
    
    x_axis = np.linspace(0,255,255)     #255 is the max luminosity    
#    plt.xlim(10,150)
    
    y_day = gauss_day.pdf(x_axis)
    y_night = gauss_night.pdf(x_axis)              # get the norm.pdf for x interval 

#    plt.plot(x_axis, y_day, color = 'y')
#    plt.plot(x_axis, y_night, color = 'b')        
#    plt.show()
    
    return gauss_day, gauss_night, y_day, y_night

#==============================================================================
#----------------------------OVERALL GAUSSIANS---------------------------------
#
#   Given the pixel intensity average, it splits the average in pixel intensity
#   average belonging to day and night, then merge all gaussian to find the
#   resulting one.
#
#   PARAMETERS:
#
#   - avg_pixels   -->   pixel intesity average of all frame
#   
#   RESULT:

#   - Gauss_day       -->   gaussian distribution of the pixel intensity of the day
#   - Gauss_night     -->   gaussian distribution of the pixel intensity of the night
#   - Y_day     -->   y_axis of gaussian distribution belonging to the overall_day
#   - Y_night   -->   y_axis of gaussian distribution belonging to the overall_night
#
#==============================================================================
def overallGaussians(avg_pixels):
    gauss_day, gauss_night, y_day, y_night = splitAvgPixels(avg_pixels)    
    
    return gauss_day, gauss_night, y_day, y_night


def plotOverallGaussian(y_overall_day, y_overall_night):
    
    x_axis = np.linspace(0,255,255)
    plt.xlim(10, 150)
    plt.plot(x_axis, y_overall_day, color = 'orange')
    plt.plot(x_axis, y_overall_night, color = 'b')
    plt.show()


#==============================================================================
#--------------------------INIT GAUSSIAN---------------------------------------
#   Load the initialization pixel intensity, based on 5 samples taken from 
#   five different webcam --> web12, web10, web7, web3, web2
#==============================================================================
def initGaussian():
    filepath = "./Dataset/means.spydata"
    avg_init_pixel = loadSpydata(filepath)
    return avg_init_pixel

#==============================================================================
# ---------------------------LOAD SPYDATA--------------------------------------
#==============================================================================
def loadSpydata(filepath):
    tar = tarfile.open(filepath, 'r')
    tar.extractall()
    extracted_files = tar.getnames()
    for f in extracted_files:
        if f.endswith('.pickle'):
             with open(f, 'rb') as fdesc:
                 data = pickle.loads(fdesc.read())
    
    avg_pixel_init = []
    for item in data:    
        avg_pixel_init = np.concatenate((avg_pixel_init, data[item]))
        
    return avg_pixel_init

