#Used to save image from input matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
import time

def save(img,filename,dirname):
    path = os.path.join(os.getcwd(),dirname+'_'+filename)
    print("Saving Image as"+path)
    time.sleep(5)
    plt.imsave(path, np.array(img), cmap=cm.gray)