# -*- coding: utf-8 -*-
"""
real image + CNN basic
1. Convolution layer : image 특징 추출  
  -> Filter : 9x9 
  -> 특징맵 : 5개 
  -> strides = 2x2, padding='SAME'
2. Pooling layer : image 축소 
  -> ksize : 7x7
  -> strides : 4x4
  -> padding='SAME' 
"""

import tensorflow as tf # ver2.x

import numpy as np
import matplotlib.pyplot as plt # image print
from matplotlib.image import imread # image read

# 1. image load 
img = imread("C:/ITWILL/6_DeepLearning/data/images/parrots.png")
plt.imshow(img)
plt.show()

# 2. image shape & RGB 픽셀값 
print(img.shape) # (512, 768, 3) 
print(img)
 

# 3. input image 만들기   
Img = img.reshape(1, 512, 768, 3) # (size, h, w, c)


# 4. Filter 만들기  
Filter = tf.Variable(tf.random.normal([9,9,3,5])) #[h, w, c, map_size]


# 5. Convolution layer 
conv2d = tf.nn.conv2d(Img, Filter, 
                      strides=[1,2,2,1], padding='SAME')

# 합성곱(Convolution) 연산 결과 
conv2d_img = np.swapaxes(conv2d, 0, 3)
conv2d_img.shape # (5, 256, 384, 1)  

fig = plt.figure(figsize = (20, 6))  
for i, img in enumerate(conv2d_img) :
    fig.add_subplot(1, 5, i+1) 
    plt.imshow(img) 
plt.show()


# 6. Pool layer 
pool = tf.nn.max_pool(conv2d, ksize=[1,7,7,1], 
                      strides=[1,4,4,1], padding='SAME')

 
# 폴링(Pool) 연산 결과 
pool_img = np.swapaxes(pool, 0, 3)

fig = plt.figure(figsize = (20, 6))    
for i, img in enumerate(pool_img) :
    fig.add_subplot(1,5, i+1)
    plt.imshow(img) 
plt.show()
    




    