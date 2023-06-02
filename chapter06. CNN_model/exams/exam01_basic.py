'''
문1) 다음과 같은 조건으로 Convolution layer와 Max Pooling layer를 정의하고, 특징맵의 shape을 확인하시오.
  <조건1> input image : volcano.jpg 파일 대상    
  <조건2> Convolution layer 정의 
       -> Filter size : 6x6
       -> featuremap : 16개
       -> strides= 1x1, padding='SAME'  
  <조건3> Max Pooling layer 정의 
       -> ksize= 3x3, strides= 2x2, padding='SAME' 
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('C:/ITWILL/6_DeepLearning/data/images/volcano.jpg') # 이미지 읽어오기
plt.imshow(img)
plt.show()
print(img.shape)

img.shape # (405, 720, 3)


Img = img.reshape(1, 405, 720, 3)

Filter = tf.Variable(tf.random.normal([6,6,3,16]))


conv2d = tf.nn.conv2d(Img, Filter, 
                      strides=[1,1,1,1], padding='SAME')\
    
conv2d_img = np.swapaxes(conv2d, 0, 3)
conv2d_img.shape # (16, 405, 720, 1)


fig = plt.figure(figsize = (20, 6))  
for i, img in enumerate(conv2d_img) :
    fig.add_subplot(1, 16, i+1) 
    plt.imshow(img) 
plt.show()



pool = tf.nn.max_pool(conv2d, ksize=[1,3,3,1], 
                      strides=[1,2,2,1], padding='SAME')

 
# 폴링(Pool) 연산 결과 
pool_img = np.swapaxes(pool, 0, 3)
pool_img.shape # (16, 203, 360, 1)



fig = plt.figure(figsize = (20, 6))    
for i, img in enumerate(pool_img) :
    fig.add_subplot(1,16, i+1)
    plt.imshow(img) 
plt.show()
