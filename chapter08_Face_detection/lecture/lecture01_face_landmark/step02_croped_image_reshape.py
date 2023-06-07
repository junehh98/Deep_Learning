# -*- coding: utf-8 -*-
"""
croped image resize(100x100)
"""

from glob import glob # (*.jpg)
from PIL import Image # image file read
import numpy as np
from skimage import io


# 폴더 경로 
path = r'C:\ITWILL\6_DeepLearning\workspace\chapter08_Face_detection'
image_path = path + "/images_croped" # image path 


# 이미지 크기 규격화 함수 
def imgReshape() :     
    img_reshape = [] # image save 
    
    for file in glob(image_path + "/*.jpg") :  
        img = Image.open(file) # image read 
        
        # image 규격화 
        img = img.resize( (100, 100) )
        
        # PIL -> numpy
        img_data = np.array(img)    
        img_reshape.append(img_data)
    
    return np.array(img_reshape) # list -> numpy

# 함수 호출         
img_reshape = imgReshape()    
img_reshape.shape
    
size = img_reshape.shape[0]

for i in range(size) :
    img = img_reshape[i]
    io.imsave(image_path + '/croped_resize'+str(i+101)+'.jpg', img)
    
    