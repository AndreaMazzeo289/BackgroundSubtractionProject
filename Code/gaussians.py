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

DAY = 'day'
NIGHT = 'night'
MAX = 'max'
MIN = 'min'
X = np.linspace(0,255,255)
NORMAL_THRESHOLD = 70
NIGHT_THRESHOLD = 140
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

    return {DAY:gauss_day, NIGHT:gauss_night}
    
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
def plotDayNightGaussian(gaussians, pixel_intensity):
   
    x_axis = np.linspace(0,255,255)     #255 is the max luminosity    
    plt.xlim(10,150)
    
    y_day = gaussians[DAY].pdf(x_axis)
    y_night = gaussians[NIGHT].pdf(x_axis)              # get the norm.pdf for x interval 

    idx = np.argwhere(np.diff(np.sign(y_day - y_night)) != 0)
    idx = idx[0][0]

    plt.title('Day/Night Gaussians')
    plt.plot(x_axis[idx:255], y_day[idx:255], color = 'orange')
    plt.plot(x_axis[0:idx], y_night[0:idx], color = 'b') 
    plt.fill_between(x_axis[idx:255], y_day[idx:255], color = 'orange')
    plt.fill_between(x_axis[0:idx], y_night[0:idx], color = 'b')    
           
    # case: pixel intensity belongs to day time
    if(pixel_intensity >= idx):
      plt.scatter(pixel_intensity, y_day[pixel_intensity])
    
    # case: pixel intensity belongs to night time
    else:
      plt.scatter(pixel_intensity, y_night[pixel_intensity])
    
    
    plt.grid()
    plt.show()   
# =============================================================================
# -------------------------PLOT GAUSSIAN---------------------------------------
# =============================================================================
def plotGaussian(gauss_distr):
    x_axis = np.linspace(0,255,1000)
    y_axis = gauss_distr.pdf(x_axis)
    
#    plt.xlim(10,150)
    plt.plot(x_axis, y_axis, color = 'r')
    plt.fill_between(x_axis, 0, y_axis, color = '#b80303')
# =============================================================================
# ---------------------------GAUSSIAN COMPARISON-------------------------------
#    
#    this function shows the comparison between the old gaussians and the new 
#    ones, so the learning rate as well                 
#                     
#    PARAMS:
#    - old_gaussians     --> gaussians to be updated
#    - old_learning_rate --> learning rate to be updated
#    - new_gaussians     --> updated gaussians
#    - new_learning_rate --> updated learning rate
#    - avg_pixel  --> new pixel average computed    
#                 
# =============================================================================
def GaussianComparison(old_gaussians, new_gaussians,
                           old_learning_rate, new_learning_rate, avg_pixel):
     
     x_axis = np.linspace(0,255,255) 
     
     old_y_day = old_gaussians[DAY].pdf(x_axis)
     old_y_night = old_gaussians[NIGHT].pdf(x_axis)        
     new_y_day = new_gaussians[DAY].pdf(x_axis)
     new_y_night = new_gaussians[NIGHT].pdf(x_axis)    
     
     old_learning_idx = np.argwhere(np.diff(np.sign(old_learning_rate[DAY] - 
                                                    old_learning_rate[NIGHT])) != 0)
     new_learning_idx = np.argwhere(np.diff(np.sign(new_learning_rate[DAY] - 
                                                    new_learning_rate[NIGHT])) != 0)
     
     old_learning_idx = old_learning_idx[0][0]
     new_learning_idx = new_learning_idx[0][0]
     
     old_gauss_idx = np.argwhere(np.diff(np.sign(old_y_day - old_y_night)) != 0)
     new_gauss_idx = np.argwhere(np.diff(np.sign(new_y_day - new_y_night)) != 0)
     
     old_gauss_idx = old_gauss_idx[0][0]
     new_gauss_idx = new_gauss_idx[0][0]
     
     plt.title('Red --> new Gaussians\n Yellow --> old gaussians')
     plt.xlim(10,150)
     # NEW AND OLD GAUSSIANS 
     plt.plot(x_axis[new_gauss_idx:255],
              new_y_day[new_gauss_idx:255], color = '#b80303')
     plt.plot(x_axis[0:new_gauss_idx],
              new_y_night[0:new_gauss_idx], color = '#b80303')
     plt.fill_between(x_axis[new_gauss_idx:255], 0,
                      new_y_day[new_gauss_idx:255], color = 'r', linewidth = 3)
     plt.fill_between(x_axis[0:new_gauss_idx+1], 0,
                      new_y_night[0:new_gauss_idx+1], color = 'r', linewidth = 3)    
     
     plt.plot(x_axis[old_gauss_idx:255],
              old_y_day[old_gauss_idx:255], color = 'y', linewidth = 2)
     plt.plot(x_axis[0:old_gauss_idx+1],
              old_y_night[0:old_gauss_idx+1], color = 'y', linewidth = 2)
     
     plotPoint(new_y_day, new_y_night, avg_pixel)     
     plt.show()
 
     plt.xlim(10,150)
     plt.title('Blue --> new Learning Rate\nGreen --> old Learning Rate')
     # NEW AND OLD LEARNING RATE
     plt.plot(x_axis[new_learning_idx:255],
              new_learning_rate[DAY][new_learning_idx:255], color = 'b', linewidth = 2)
     plt.plot(x_axis[0:new_learning_idx],
              new_learning_rate[NIGHT][0:new_learning_idx], color = 'b', linewidth = 2)
     plt.fill_between(x_axis[new_learning_idx:255], 0,
                      new_learning_rate[DAY][new_learning_idx:255], color = 'b')
     plt.fill_between(x_axis[0:new_learning_idx+1], 0,
                      new_learning_rate[NIGHT][0:new_learning_idx+1], color = 'b')    
     
     plt.plot(x_axis[old_learning_idx:255],
              old_learning_rate[DAY][old_learning_idx:255], color = 'y', linewidth = 3)
     plt.plot(x_axis[0:old_learning_idx],
              old_learning_rate[NIGHT][0:old_learning_idx], color = 'y', linewidth = 3)        
 
     plotPoint(new_learning_rate[DAY], new_learning_rate[NIGHT], avg_pixel)
     plt.show()
     

# =============================================================================
# -----------------------------PLOT POINTS-------------------------------------
#     Point Plotting on the distribution of the gaussians. It check if the
#     point belongs to the day or night gaussian
#     
# =============================================================================
def plotPoint(day_part, night_part, point):
    idx = np.argwhere(np.diff(np.sign(day_part - night_part)) != 0)
    idx = idx[0][0]
    
    # day time case
    if(point >= idx):
        plt.scatter(point, day_part[point], c='g', marker='o', s=150)
        
    #night time case
    else:
        plt.scatter(point, night_part[point], c='g', marker='o', s=150)
        
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
    dayPixels = []
    nightPixels = []    
    
    # the mid point splits avg pixel in avg pixel of day from night
    for avg_pixel in avg_pixels:
        mid_point = int((max(avg_pixels) - min(avg_pixels))/2)
        split = mid_point + min(avg_pixels)
        
        if(avg_pixel > split):        
            dayPixels.append(avg_pixel)
        elif(avg_pixel < split):
            nightPixels.append(avg_pixel)
            
#    gauss_day, gauss_night = generateDayNightGaussians(day, night)
        
    gaussians = generateDayNightGaussians(dayPixels, nightPixels)
    
    x_axis = np.linspace(0,255,255)     #255 is the max luminosity    
    y_day = gaussians[DAY].pdf(x_axis)
    y_night = gaussians[NIGHT].pdf(x_axis)
   
    y_gaussians = {DAY:y_day, NIGHT:y_night}

    return gaussians, y_gaussians
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
    gaussians, y_gaussians = splitAvgPixels(avg_pixels)    
    
#    return gauss_day, gauss_night, y_day, y_night

    return gaussians, y_gaussians

def plotOverallGaussian(y_gaussians):
    
    x_axis = np.linspace(0,255,255)
    plt.xlim(10, 150)
    plt.plot(x_axis, y_gaussians[DAY], color = 'orange')
    plt.plot(x_axis, y_gaussians[NIGHT],color = 'b')
    plt.show()

# =============================================================================
# --------------------------GENERATE THRESHOLD---------------------------------
#    
#    Generate the threshold function in the range [70, 140], with a sigmoid 
#    function, generated through the gaussians of day and nught
#    
#    PARAMS:
#    - gaussians  --> gaussians of day and night
#    
#    RESULT:
#    - threshold  --> threshold function
#    
# =============================================================================
def generateThreshold(gaussians):
  
    mean_day = int(gaussians[DAY].mean())
    mean_night = int(gaussians[NIGHT].mean())
    
    mean = mean_night + (mean_day - mean_night)/2
    std = 10
#    print('night mean : {}'.format(mean_night))
#    print('day mean : {}'.format(mean_day))
#    print('mean : {}'.format(mean))
    x_axis = np.linspace(0,255,255)
    sigmoid = 1 - norm(mean, std).cdf(x_axis)
    threshold = 70*sigmoid + NORMAL_THRESHOLD

    return threshold
# =============================================================================
# ------------------------------PLOT THRESHOLD---------------------------------
# =============================================================================
def plotThreshold(threshold):
    x_axis = np.linspace(0,255,255)
    plt.xlim(10,150)
    plt.title('Threshold')
    plt.plot(x_axis, threshold)
    plt.grid()
    plt.show() 
# =============================================================================
# ---------------------GENERATE LEARNING RATE TRADING--------------------------
#    
#    Generate the learning rate function, starting from the day and night
#    gaussians: the function to be created will be the concatenation of 
#    the density function of the two gaussians. Learning rate function has 
#    a gaussian trade
#    
#    PARAMS:
#    - gauss_distr --> gaussians of day and night
#    
#    RETURN:
#    - learning_rate --> learning rate function
#    
# =============================================================================
def generateLearningRateTrading(gauss_distr):
  
    MEAN_DAY = gauss_distr[DAY].mean()
    STD_DAY = gauss_distr[DAY].std()
    OFFSET_DAY = -2

    MEAN_NIGHT = gauss_distr[NIGHT].mean()+10
    STD_NIGHT = gauss_distr[NIGHT].std()
    OFFSET_NIGHT = -2
    
    RANGE = 0.85
    MIN_VALUE = 0.1
    
    gauss_distr_shifted = {DAY: norm(MEAN_DAY + OFFSET_DAY, STD_DAY),
                           NIGHT: norm(MEAN_NIGHT + OFFSET_NIGHT, STD_NIGHT)}
    
    density_shifted_DAY = gauss_distr_shifted[DAY].cdf(X)
    density_shifted_NIGHT = gauss_distr_shifted[NIGHT].cdf(X)
    densities_shifted = {DAY: density_shifted_DAY, NIGHT: density_shifted_NIGHT}
  
    learning_rate_DAY = MIN_VALUE + (1 - densities_shifted[DAY]) * RANGE
    learning_rate_NIGHT = MIN_VALUE + (densities_shifted[NIGHT]) * RANGE
    learning_rate = {DAY: learning_rate_DAY, NIGHT: learning_rate_NIGHT}

    return learning_rate
# =============================================================================
# --------------------------PLOT LEARNING RATE---------------------------------
# =============================================================================
def plotLearningRate(learningRate, pixel_intensity = 0):
  
    x_axis = np.linspace(0,255,255)
    
    idx = np.argwhere(np.diff(np.sign(learningRate[DAY] - learningRate[NIGHT])) != 0)
    idx = idx[0][0]
    
    plt.xlim(10,150)
    plt.plot(x_axis[idx:255], learningRate[DAY][idx:255], c='g')
    plt.plot(x_axis[0:idx], learningRate[NIGHT][0:idx], c='g')
    
    # case: pixel intensity belongs to day time
    if(pixel_intensity >= idx):
      plt.scatter(pixel_intensity, learningRate[DAY][pixel_intensity])
    
    # case: pixel intensity belongs to night time
    else:
      plt.scatter(pixel_intensity, learningRate[NIGHT][pixel_intensity])
    
    plt.grid()
    plt.show()
# =============================================================================
# ----------------COMPARISON LEANRING RATE GAUSSIANS---------------------------
#
#    It plots the trading of learning rate w.r.t. the gaussians. The learning
#    rate function is normalized in order to be visible with the gaussians trading
#    
#    PARAMS:
#    - gaussians       --> gaussians to be compared with the learning rate trading
#    - learning rate   --> learning rate function to be compare with gaussians
#    - pixel_intensity --> pixel intesity to be showed in the two functions
#    
# =============================================================================
def comparisonLearningRateGaussians(gaussians, learningRate, pixel_intensity = 0):
    
    x_axis = np.linspace(0,255,255)

    idx_learning_rate = np.argwhere(np.diff(np.sign(learningRate[DAY] - learningRate[NIGHT])) != 0)
    idx_learning_rate = idx_learning_rate[0][0]
    
    gaussian_day = gaussians[DAY].pdf(x_axis)
    gaussian_night = gaussians[NIGHT].pdf(x_axis)      

    idx_gauss = np.argwhere(np.diff(np.sign(gaussian_day - gaussian_night)) != 0)
    idx_gauss = idx_gauss[0][0]
   
    plt.xlim(10,150)
         
    plotLearningRate(learningRate, pixel_intensity)
    
    plotDayNightGaussian(gaussians, pixel_intensity)

# =============================================================================
# --------------------COMPARISON THRESHOLD GAUSSIANS---------------------------
#
#    It plots the trading of learning rate w.r.t. the gaussians. The learning
#    rate function is normalized in order to be visible with the gaussians trading
#    
#    PARAMS:
#    - gaussians       --> gaussians to be compared with the learning rate trading
#    - threshold       --> threshold function to be compare with gaussians
#    
# =============================================================================
def comparisonThresholdGaussians(gaussians, threshold):
    
    x_axis = np.linspace(0,255,255)
  
    mean_day = int(gaussians[DAY].mean())
    ripartition = gaussians[DAY].pdf(x_axis)

    h = ripartition[mean_day]
    
    plt.xlim(10,150)
    
    y_day = gaussians[DAY].pdf(x_axis)
    y_night = gaussians[NIGHT].pdf(x_axis) 

    idx = np.argwhere(np.diff(np.sign(y_day - y_night)) != 0)
    idx = idx[0][0]

    plt.title('Day/Night Gaussians --> Orange/Blue\nThreshold --> Black')
    plt.plot(x_axis[idx:255], y_day[idx:255], color = 'orange')
    plt.plot(x_axis[0:idx], y_night[0:idx], color = 'b') 
    plt.fill_between(x_axis[idx:255], y_day[idx:255], color = 'orange')
    plt.fill_between(x_axis[0:idx], y_night[0:idx], color = 'b')  
    
    plt.plot(x_axis, -0.032 + threshold*0.0005, c='black')
           
    plt.grid()
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

