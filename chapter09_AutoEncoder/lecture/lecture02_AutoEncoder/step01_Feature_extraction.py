# -*- coding: utf-8 -*-
"""
step01_Feature_extraction.py

1. 특징 추출 오토인코더
 - 출력값을 입력값과 동일한 결과가 생성하도록 학습하는 비지도학습 모델
 - 모델 학습 데이터 : 입력과 출력 데이터 동일함 
   입력 : x -> 출력 : x
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.layers import Dense # DNN layer 구축 
import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() 
x_train.shape # (60000, 28, 28) 
y_train.shape # (60000,)


# 2. x 변수 전처리 
x_train = x_train / 255.
x_val = x_val / 255. # 255.


x_train = x_train.reshape(-1, 28*28) # (60000, 784)
x_val = x_val.reshape(-1, 28*28)


# 3. Functional API model 구축
###############################################
### Functional API : Input과 Model 클래스 이용 
###############################################
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

input_dim = (784,)  # 입력층 : 784(28x28)
encoding_dim = 64   # 인코딩층 : 64(8x8)
output_dim = 784    # 디코딩층 : 784(28x28)


# input layer
inputs = Input(shape = (input_dim)) 

# encoding layer  
encoded = Dense(units = encoding_dim, activation='relu')(inputs) 

# decoding layer  
outputs = Dense(units=output_dim)(encoded) 

# 4. autoencoder model 생성 & 학습  
autoencoder = Model(inputs, outputs) 

autoencoder.summary()
'''

 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 784)]             0         
                                                                 
 dense (Dense)               (None, 64)                50240     
                                                                 
 dense_1 (Dense)             (None, 784)               50960     
                                                                 
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
'''


# model compile 
autoencoder.compile(optimizer='adam', loss='mse')


# model 학습 
autoencoder.fit(x = x_train, y = x_train,  
               batch_size=256, 
               epochs=10)

# 학습된 가중치, 편향 확인
autoencoder.weights
autoencoder.layers
'''
[<keras.engine.input_layer.InputLayer at 0x1a7f8931580>, [0]
 <keras.layers.core.dense.Dense at 0x1a7f89314f0>, [1]
 <keras.layers.core.dense.Dense at 0x1a7f89352b0>] [2]
'''


# 5. 모델 재사용(model resue) 

# 1) encoder model 
inputs = Input(shape = (input_dim)) 
encoding_layer = autoencoder.layers[1]  
encoder = Model(inputs, encoding_layer(inputs)) 


# 2) decoder model 
encoding_inputs = Input(shape=(encoding_dim,)) # 64
decoding_layer = autoencoder.layers[-1] 
decoder = Model(encoding_inputs, decoding_layer(encoding_inputs))  


# encoder model 이미지 예측 
encoded_imgs = encoder.predict(x_val)
encoded_imgs.shape # (10000, 64)


# decoder model 이미지 예측 
decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs.shape # (10000, 784) -> 64개로 압축되었던 것을 784로 다시 복원


n = 10 # 이미지 개수
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # 원본 이미지 
    ax = plt.subplot(3, n, i) # 1~10
    plt.imshow(x_val[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # encoder 이미지 
    ax = plt.subplot(3, n, i + n) # 11~20
    plt.imshow(encoded_imgs[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    # decoder 이미지 
    ax = plt.subplot(3, n, i + n*2) # 21~30
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

        
plt.show()

