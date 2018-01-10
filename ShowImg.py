#Used to display image from input matrix
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import numpy as np

def show(img):        
    print("\t\tLets Show")
    plt.imshow(np.array(img), cmap=cm.gray)