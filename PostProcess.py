from skimage.filters import threshold_mean
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import time

#Converts the grayscale to binary matrix (image) using mean thresholding
def binarize(img):
    print("\t\tStep3.1: Binarizing Clustered Image...")
    thresh = threshold_mean(img)
    img = img > thresh
    return img
#Cleans the binarized image by filling holes (empty pixels) and outliers
def clean(img):
    print("\t\tStep3.1: Cleaning Image...")
    not_img_thresholded = np.invert(img)
    img=binary_fill_holes(np.invert(binary_fill_holes(img)))
    return img