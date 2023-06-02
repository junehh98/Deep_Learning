# -*- coding: utf-8 -*-
"""
MNIST dataset + CNN basic
 Convolution layer : 이미지 특징 추출
 Pooling layer : 이미지 픽셀 축소(다운 샘플링)
"""
import tensorflow as tf # ver2.x
from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 1. image read & input image 만들기  
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 


# 1) 자료형 변환 : int -> float
x_train = x_train.astype('float32') # type 일치 
x_test = x_test.astype('float32')

# 2)정규화 
x_train /= 255 # x_train = x_train / 255
x_test /= 255


# 3) input image 선정 : 첫번째 image 
img = x_train[0]
plt.imshow(img, cmap='gray') # 숫자 5, cmap ='gray' 흑백이미지로 출력
plt.show() 
img.shape # (28, 28)


# 4) input image 모양변경  
inputImg = img.reshape(1,28,28,1) # [size, h, w, c]


# 2. Filter 만들기 : image와 합성할 데이터 
Filter = tf.Variable(tf.random.normal([3,3,1,5]))#[h, w, c, feature map] 
'''
h=3, w=3 : 필터 가로,세로 크기
c=1 : image color 수 일치   
fmap=5 : image에서 추출될 특징 이미지 개수   
'''


# 3. Convolution layer : 이미지 특징 추출(특징맵)   
conv2d = tf.nn.conv2d(inputImg, Filter, strides=[1,1,1,1],
                      padding='SAME') # input_image vs Filter 선형결합

conv2d.shape # [1, 28, 28, 5] 1개의 이미지 28x28이 5개 생성
'''
strides=[1,1,1,1] : Filter 가로/세로 1칸씩 이동 
padding = 'SAME' : 원본이미지와 동일한 크기로 특징 이미지 추출
padding = 'VALID' : output = (input_image - filter) / S + 1

padding = 'SAME' = [1, 28, 28, 5] 
padding = 'VALID' = [1, 26, 26, 5] stride가 1일때 많이 사용
'''

# 합성곱(Convolution) 연산 결과  
conv2d_img = np.swapaxes(conv2d, 0, 3) 
conv2d_img.shape # (5, 28, 28, 1)


for i, img in enumerate(conv2d_img) : 
    plt.subplot(1, 5, i+1) # 1행 5열, 열index 
    plt.imshow(img, cmap='gray')  
plt.show()



# 4. Pool layer : 특징맵 픽셀축소(다운샘플링)  
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1],
                      padding='SAME')
'''
ksize=[1,2,2,1] : window 가로/세로 1칸씩 이동 
padding = 'SAME' : 원본이미지와 동일한 크기로 특징 이미지 추출 
output = (input_image - ksize) / S + 1
output = (28 - 2) / 2 + 1 = 14
'''  

# 폴링(Pool) 연산 결과 
pool_img = np.swapaxes(pool, 0, 3) 
pool_img.shape # (5, 14, 14, 1) 픽셀사이즈 28 -> 14


for i, img in enumerate(pool_img) :
    plt.subplot(1,5, i+1)
    plt.imshow(img, cmap='gray') 
plt.show()

    











