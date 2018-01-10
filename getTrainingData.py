import os
from PIL import Image
import numpy as np
def getImages():
	data = []
	for filename in os.listdir('trainimages'):
		data.append(np.array(Image.open(os.path.join('trainimages',filename)), dtype=np.float32).mean(2)[:32,:32])

	# print(np.shape(np.array(data)))
	return np.array(data)
def getLabels():
	labels = []
	for filename in os.listdir('trainimages'):
		labels.append(int(filename[0]))
	return np.array(labels)	
	
# getImages()