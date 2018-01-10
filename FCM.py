#Contains FCM and functions related to FCM
import skfuzzy as fuzz
import numpy as np
import math
import time

def fcmPlus(img,error,maxiter,shape):
    print("\t\tStep2.1: Adaptively  Estimating C...")
    #Adaptive estimation of C as described in the paper
    c=int(np.mean(img)/25)
    #Main FCM step
    cntr, u, u0, d, jm, p, fpc =  fuzz.cluster.cmeans(np.transpose(np.array(img)-np.min(img)),
        c,2,error=error, maxiter=maxiter, init=None)
    result = np.transpose(u).reshape((shape[0],shape[1],c));
    sorted = np.argsort(cntr[:,0])
    c1 = sorted[-1]
    c2 = sorted[-2]
    resultarray = np.zeros((shape[0],shape[1]))
    print("\t\tStep2.3: Aggregating High Intensity Memberships...")
    #Aggregation of high intensity memberships as mentioned in the paper
    for i in range(shape[0]):
        for j in range(shape[1]):
            resultarray[i][j]=(result[i][j][c1]+math.exp(-1)*result[i][j][c2])
    return resultarray