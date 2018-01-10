#Creates ground truth bounding box on the images in the dataset
from PIL import Image, ImageFont, ImageDraw
import os
import re

filename = '/media/anmol/More stuff/sem 3-1/NNFL/LSIFIR/Detection/Train/06/00023.png'
#Opening Image
source_img = Image.open(filename)
#initialize Draw object
draw = ImageDraw.Draw(source_img)
filename = '/media/anmol/More stuff/sem 3-1/NNFL/LSIFIR/Detection/Train/cleanannotations/06_00023.txt'
#Open Annotation corresponding to the image
with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content]
#Read Bounding box coordinates from annotations file
for x in content:
    pt = re.compile(r'\d+')
    nums = pt.findall(x)
    x1 = float(nums[0])
    x2= float(nums[1])
    x3 = float(nums[2])
    x4= float(nums[3])
    #Draw the bounding box using the draw object
    draw.rectangle(((x1, x2), (x3, x4)),outline="blue")
#Display Image
source_img.show()