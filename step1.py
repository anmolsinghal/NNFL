#Driver File for Step1
import os
from Segmentation import segment

#Directory Traversal
for dirname in next(os.walk(os.getcwd()))[1]:
    for filename in os.listdir(dirname):
        if filename[-4:] == '.png':
        	segment(filename,dirname)
        	