#Preprocessing on the input grayscale image
from skimage import exposure
import numpy as np

#Function to adjust contrast
def adjustContrast(img):
    img = exposure.equalize_hist(img)
    return img

#Function to resize image array and conver image to 8-bit grayscale
def preFuzzy(img):
    imagearray = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imagearray.append([img[i][j]*255])
    return imagearray