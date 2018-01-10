import cv2
import numpy as np 
import os
# read and scale down image

def createTraining(filename,dirname):
	#read the binary image
	img = cv2.pyrDown(cv2.imread(os.path.join(os.getcwd(),"binary",dirname+'_'+filename+".png"), cv2.IMREAD_UNCHANGED))
	#read the original image
	oimg = cv2.imread(os.path.join(os.getcwd(),dirname,filename+".png")) 
	#Generate testing path for lookup in neg list and pos list
	path = 'Train/'+dirname+'/'+filename+'.png'+'\n'
	#open neg list
	with open('neg.lst') as f:
	    neglist = f.readlines()
	#open pos list
	with open('pos.lst') as f1:
	    poslist = f1.readlines()
	#For negative examples use the Bounding boxes generated on binarized image
	if path in neglist:
	    # threshold image
	    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
	                127, 255, cv2.THRESH_BINARY)
	    threshed_img = cv2.bitwise_not(threshed_img)
	    # find contours and get the external one
	    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_LIST,
	                    cv2.CHAIN_APPROX_SIMPLE)
	    i=0
	    # with each contour, draw boundingRect
	    for c in contours:
	        # get the bounding rect
	        x, y, w, h = cv2.boundingRect(c)
	        # draw a rectangle to visualize the bounding rect
	        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	        i+=1
	        
	        #get the min area rect
	        rect = cv2.minAreaRect(c)
	        #Pruning of candidate bounding boxes
	        #Bounding Box should be upright horizontal or vertical
	        if rect[2]%90.0==0.0:
	        	#Ignore small trivial bounding boxes
	            if rect[1][1]>15 and rect[1][0]>rect[1][1]:
	                #Get the coordinates for the 4 corners of the Bounding box
	                box = cv2.boxPoints(rect)
	                #Get integer splicing coordinates for cropping
	                box = np.int0(box)
	                #Crop and resize image
	                img1=cv2.resize(oimg[box[1][1]:box[3][1],box[1][0]:box[3][0]],(32,32))
	                #Save the resized image (will be fed to the CNN)
	                cv2.imwrite('trainimages/'+'0_'+dirname+'_'+filename+'_'+str(i)+".png", img1)

	    
	    
	#For negative examples use the ground truth Bounding boxes from the dataset
	elif path in poslist:
		try:
			#Open annotations file
		    with open(os.path.join(os.getcwd(),'cleanannotations',dirname+'_'+filename+".txt")) as f:
		        annotations = f.readlines()
		    i =0 
		    #for every bounding box in the file
		    for annotation in annotations:
		        coordinates = annotation.split()
		        #Read the coordinates for the bounding box
		        coordinates=[int(j) for j in coordinates]
		        #Crop and resize image according to the box
		        img1=cv2.resize(oimg[coordinates[1]:coordinates[3],coordinates[1]:coordinates[3]],(128,128))
		        #Save the resized image (will be fed to the CNN)
		        cv2.imwrite('trainimages/'+'1_'+dirname+'_'+filename+'_'+str(i)+".png", img1)
		        i +=1
		except:
			print("No annotation")


