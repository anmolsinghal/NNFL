#driver script for step2- Generation of detection of contours and generation of bounding boxes on the original image 
import numpy as np
import os
from createTraining import createTraining


for dirname in next(os.walk(os.getcwd()))[1]:
    for filename in os.listdir(dirname):
        if filename[-4:] == '.png' and dirname[0]=="0":
            createTraining(filename[:-4],dirname)
