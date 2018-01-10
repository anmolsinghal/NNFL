#Driver function for the segmentaiton part
import numpy as np
import os
from PIL import Image 
from ShowImg import show
from SaveImg import save
from PostProcess import binarize,clean 
from PreProcess import adjustContrast,preFuzzy
from FCM import fcmPlus
import time


def segment(filename,dirname):
    print("Processing "+dirname+'/'+filename+'...')
    print("\tStep0: Opening "+dirname+'/'+filename+'...')
    #Opening Image
    img = np.array(Image.open(os.path.join(os.getcwd(),dirname,filename)))
    shape=np.shape(img)

    #Preprocessing
    print("\tStep1: Preprocessing...")
    img=adjustContrast(img)
    img=preFuzzy(img)
    #FCM and Other Steps
    print("\tStep2: Fuzzy C-Means and Other Steps...")
    error=0.005
    maxiter=1000
    img=fcmPlus(img,c,error,maxiter,shape)
    #PostProcessing
    print("\tStep3: Postprocessing and Refining...")
    img=threshold(img)
    show(img,shape)
    img=clean(img)
    
    #Displaying segmented Image
    show(img,shape)
    #Saving segmented Image
    save(img, filename,dirname)
