# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:36:47 2019

@author: andre
"""

from fpdf import FPDF
from PIL import Image
import numpy as np
import glob
from PyPDF2 import PdfFileWriter, PdfFileReader
import os

numWeb = '3'

webcamInfo = []
webcamRow = ['./Webcam/web2', 'A08', 'km. 5,6 - Milano Nord itinere Nord']
webcamInfo.append(webcamRow)
webcamRow = ['./Webcam/web3', 'R14', 'km. 2,0 - BO Casalecchio itinere Ovest']
webcamInfo.append(webcamRow)
webcamRow = ['./Webcam/web7', 'A04', 'km. 149,5 A4/A58 dir Ovest']
webcamInfo.append(webcamRow)
webcamRow = ['./Webcam/web10', 'A13', 'km. 33,0 Ferrara Sud shelter itinere Nord']
webcamInfo.append(webcamRow)
webcamRow = ['./Webcam/web12', 'A14', 'km. 16,0 Bologna Fiera itinere Sud']
webcamInfo.append(webcamRow)

backgroundTruth = []
im = Image.open('./GroundTruth/web'+numWeb+'/00.png')
backgroundTruth.append(np.array(im))
im.close()
im = Image.open('./GroundTruth/web'+numWeb+'/01.png')
backgroundTruth.append(np.array(im))
im.close()
im = Image.open('./GroundTruth/web'+numWeb+'/02.png')
backgroundTruth.append(np.array(im))
im.close()
im = Image.open('./GroundTruth/web'+numWeb+'/03.png')
backgroundTruth.append(np.array(im))
im.close()

backgroundDetected = []
im = Image.open('./BackgroundDetected/web'+numWeb+'/00.png')
backgroundDetected.append(np.array(im))
im.close()
im = Image.open('./BackgroundDetected/web'+numWeb+'/01.png')
backgroundDetected.append(np.array(im))
im.close()
im = Image.open('./BackgroundDetected/web'+numWeb+'/02.png')
backgroundDetected.append(np.array(im))
im.close()
im = Image.open('./BackgroundDetected/web'+numWeb+'/03.png')
backgroundDetected.append(np.array(im))
im.close()

modelTruth = []
im = Image.open('./Models/web'+numWeb+'/Day/model.png')
modelTruth.append(np.array(im))
im = Image.open('./Models/web'+numWeb+'/Night/model.png')
modelTruth.append(np.array(im))

foregroundDetected = []
im = Image.open('./Models/web'+numWeb+'/Day/foreground.png')
foregroundDetected.append(np.array(im))
im = Image.open('./Models/web'+numWeb+'/Night/foreground.png')
foregroundDetected.append(np.array(im))

outputReport = 'report_web'+numWeb+'.pdf'
accuracyThreshold = 10
errorThreshold = 12

#%%

index = 0
numBG = len(backgroundDetected)
print('\n')
print('START ANALYSIS')

initializationPage()
print('initiliazation done')

DAY_FRAME = 1
NIGHT_FRAME = 0 

firstAnalysis(modelTruth[0], foregroundDetected[0],DAY_FRAME)
firstAnalysis(modelTruth[1], foregroundDetected[1],NIGHT_FRAME)

while(index < numBG):
    
    print('Processing image '+ str(index) + '...')
    secondAnalysis(backgroundTruth[index], backgroundDetected[index], index, accuracyThreshold, errorThreshold)
    index += 1
    print('Done')

print('ANALYSIS TERMINATED')

print('\nMerge all reports...')

paths = glob.glob('report_num*.pdf')
paths.sort()
mergeReport(outputReport, paths)

print('Done')

for report in paths:
    os.remove(report)
    
os.remove('error.png')

print('Output report ' + outputReport + ' is ready')

#%%
#-------------------------------------------------------------------------------------------------------------------------------------------#
#
#   INITIALIZATION PAGE DEFINITION
#
#-------------------------------------------------------------------------------------------------------------------------------------------#

####################################################################################
#
# initializationPage()
#
# params: void
#
# return: void
#
####################################################################################

def initializationPage():
    
    pdf = FPDF()
    pdf.set_font("Arial", size=10)
    pdf.add_page()
    
    pdf.set_draw_color(0, 0, 0)
    
    pdf.line(10, 10, 200, 10)
    pdf.line(10, 287, 200, 287)
    pdf.line(10, 10, 10, 287)
    pdf.line(200, 10, 200, 287)
    
    pdf.image("logo.png",79,33,55,55)
    pdf.cell(0,80,"",border=0,ln=2)
    
    text = 'Politecnico di Milano'
    pdf.cell(0, 5, txt="{}".format(text), ln=2, border = 0, align="C")
    text = 'AA 2018 - 2019'
    pdf.cell(0, 5, txt="{}".format(text), ln=2, border = 0, align="C")
    
    pdf.cell(0,8,"",border=0,ln=2)
    text = 'Computer Science and Engineering'
    pdf.cell(0, 5, txt="{}".format(text), ln=2, border = 0, align="C")
    
    pdf.set_font("Arial", size=14)
    text = 'IMAGE ANALYSIS AND COMPUTER VISION PROJECT'
    pdf.cell(0, 15, txt="{}".format(text), ln=2, border = 0, align="C")
    
    pdf.set_font("Arial", "B", size=25)
    text = 'BACKGROUND SUBTRACTION'
    pdf.cell(0, 50, txt="{}".format(text), ln=2, border = 0, align="C")
    
    pdf.set_font("Arial", size=18)
    text = 'Results report:'
    pdf.cell(0, 20, txt="{}".format(text), ln=2, border = 0, align="C")
    
    if(numWeb == '2'):
        key = 0
    if(numWeb == '3'):
        key = 1
    if(numWeb == '7'):
        key = 2
    if(numWeb == '10'):
        key = 3
    if(numWeb == '12'):
        key = 4
        
    
    pdf.set_font("Arial", size=11)
    text = 'Webcam: ' + webcamInfo[key][1] + ' - ' + webcamInfo[key][2]
    pdf.cell(0, 7, txt="{}".format(text), ln=2, border = 0, align="C")
    
    pdf.cell(0,30,"",border=0,ln=2)
    
    pdf.set_font("Arial", size=10)
    text = 'Andrea Mazzeo - 895579'
    pdf.cell(0, 5, txt="{}".format(text), ln=2, border = 0, align="C")
    text = 'Daniele Moltisanti - 898977'
    pdf.cell(0, 5, txt="{}".format(text), ln=2, border = 0, align="C")
    
    name = 'report_num00.pdf'
    pdf.output(name)    

#%% ----------------------------- RESULT ANALYSIS -----------------------------

####################################################################################
#
# resultAnalysis (true, bg, index, accuracyThreshold, errorThreshold)
#
# params: true              ->  frame of original video
# params: bg                ->  frame of background detected
# params: index             ->  index of analyzed frame
# params: accuracyThreshold ->  threshold used to compute accuracy
# params: errorThreshold    ->  threshold used to compute error between two frames
#
# return: void
#
####################################################################################

def firstAnalysis(GTruth, FGDet, DN):
     
    pdf = FPDF()
    
    pdf.add_page()
    
    epw = pdf.w - 2*pdf.l_margin
    col_width = epw/4
    
    endRow = len(GTruth)
    endCol = len(GTruth[0])
    
    if(DN):
        path_GTruth = './Models/web'+numWeb+'/Day/model.png'
        path_FGDet = './Models/web'+numWeb+'/Day/foreground.png'
        name = 'report_num01.pdf'
        text = 'DAILY FRAME'
    else:
        path_GTruth = './Models/web'+numWeb+'/Night/model.png'
        path_FGDet = './Models/web'+numWeb+'/Night/foreground.png'
        name = 'report_num02.pdf'
        text = 'NIGHT FRAME'
    
    pdf.set_font("Arial", "B", size=14)
    
    pdf.cell(0, 24, txt="{}".format(text), ln=2, border=0, align="C")

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(1,endRow):
        for j in range(1,endCol):
            if(FGDet[i][j] == 255 and GTruth[i][j] == 255):
                TP += 1
            if(FGDet[i][j] == 255 and GTruth[i][j] == 0):
                FP += 1
            if(FGDet[i][j] == 0 and GTruth[i][j] == 255):
                FN += 1
            if(FGDet[i][j] == 0 and GTruth[i][j] == 0):
                TN += 1
    
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = (TP) / (TP+FP)
    
    accuracy = round(accuracy*100,2)
    precision = round(precision*100,2)
    
    pdf.set_font("Arial", size=12)
    
    text = 'Ground truth:'
    pdf.cell(0, 7, txt="{}".format(text), ln=2,border=0, align="L")
    pdf.cell(0, 78, "", ln=2,border=0)
    pdf.image(path_GTruth,40,42,130,72)
    
    text = 'Model identified:'
    pdf.cell(0, 7,  txt="{}".format(text), ln=2, border=0,align="L")    
    pdf.cell(0, 78, "", ln=2,border=0)
    pdf.image(path_FGDet,40,127,130,72)
    
    tableData = [['True positive','True negative','False positive','False negative'],
                 [TP,TN,FP,FN]
                ]
    
    th = pdf.font_size
    
    text = 'Results:'
    pdf.cell(0, 7, txt="{}".format(text), ln=2,border=0, align="L")
    
    for row in tableData:
        for data in row:
            pdf.cell(col_width, 2*th, str(data), border=1, align="C")
        pdf.ln(2*th)
        
    pdf.cell(0, 7, txt="", ln=2,border=0)
    
    pdf.set_font("Arial", "B", size=12)
    text = 'Precision: ' +str(precision)+ '%'
    pdf.cell(0, 7, txt="{}".format(text), ln=2,border=0, align="L")
    text = 'Accuracy: ' +str(accuracy)+ '%'
    pdf.cell(0, 7, txt="{}".format(text), ln=2,border=0, align="L")
    
    pdf.output(name)
    
#%% ----------------------------- RESULT ANALYSIS -----------------------------

####################################################################################
#
# resultAnalysis (true, bg, index, accuracyThreshold, errorThreshold)
#
# params: true              ->  frame of original video
# params: bg                ->  frame of background detected
# params: index             ->  index of analyzed frame
# params: accuracyThreshold ->  threshold used to compute accuracy
# params: errorThreshold    ->  threshold used to compute error between two frames
#
# return: void
#
####################################################################################

def secondAnalysis(BGTruth, BGDet, index, accuracyThreshold, errorThreshold):
     
    pdf = FPDF()
    
    pdf.add_page()
    
    endRow = len(BGTruth)
    endCol = len(BGTruth[0])
   
    equal=0
    total=0
    
    if(index==0):
        text = "MIDDAY BACKGROUND DETECTION"
    if(index==1):
        text = "SUNSET BACKGROUND DETECTION"
    if(index==2):
        text = "MIDNIGHT BACKGROUND DETECTION"
    if(index==3):
        text = "SUNRISE BACKGROUND DETECTION"
    
    pdf.set_font("Arial", "B", size=14)
    #text = 'BACKGROUND DETECTION NUMBER ' + str(index) + ' WITH THRESHOLD OF ' + str(accuracyThreshold)
    pdf.cell(0, 12, txt="{}".format(text), ln=2, border=0, align="C")
    
    indexString = '{0:02}'.format(index)

    path_BGTruth = './GroundTruth/web'+numWeb+'/'+indexString+'.png'
    path_BGDet = './GroundTruth/web'+numWeb+'/'+indexString+'.png'

    for i in range(1,endRow):
        for j in range(1,endCol):
            if(BGDet[i][j] in range(BGTruth[i][j]-accuracyThreshold,BGTruth[i][j]+accuracyThreshold)):
                equal += 1
            total += 1
    
    BGDet = BGDet.astype(np.int8)
    BGTruth = BGTruth.astype(np.int8)
    
    error = np.abs(BGTruth-BGDet)
    
    for i in range(0,endRow):
        for j in range(0,endCol):
            if(error[i][j] < errorThreshold):
                error[i][j] = 0
            else:
                error[i][j] = 255
                
    error = error.astype(np.uint8)
    
    image_error = Image.fromarray(error)
    image_error.save("error.png")
            
    accuracy = equal/total;
    accuracy = round(accuracy*100,2)
    
    pdf.set_font("Arial", size=12)
    text = 'Background detected with accuracy of '+ str(accuracy) + '% with threshold of '+str(accuracyThreshold)
    pdf.cell(0, 7, txt="{}".format(text), ln=2, border=0,align="L")
    
    text = 'Original frame:'
    pdf.cell(0, 7, txt="{}".format(text), ln=2,border=0, align="L")
    pdf.cell(0, 74, "", ln=2,border=0)
    pdf.image(path_BGTruth,40,37,130,72)
    
    text = 'Background detected:'
    pdf.cell(0, 7,  txt="{}".format(text), ln=2, border=0,align="L")    
    pdf.cell(0, 74, "", ln=2,border=0)
    pdf.image(path_BGDet,40,118,130,72)
    
    text = 'Error plotted with threshold of ' + str(errorThreshold) +':'
    pdf.cell(0, 7, txt="{}".format(text), ln=2,border=0, align="L")
    pdf.cell(0, 74, "", ln=2,border=0)
    pdf.image("error.png",40,199,130,72)
    
    indexString = '{0:02}'.format(index+3)
    name = 'report_num'+str(indexString)+'.pdf'
    pdf.output(name)
    
#%% ------------------------------ REPORT MERGER ------------------------------

####################################################################################
#
# mergeReport(output_path, input_paths):
#
# params: output_path  ->  path where store output pdf
# params: input_paths  ->  paths of input reports
#
# return: void
#
####################################################################################
 
def mergeReport(output_path, input_paths):
    pdf_writer = PdfFileWriter()
 
    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
 
    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)