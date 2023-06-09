# -*- coding: utf-8 -*-
"""
2. 잡음 제거 오토인코더 
 - 입력에 포함된 잡음(noise)을 제거하는 비지도학습 모델  
 - 학습 데이터 
   입력 : x' = x + r(난수:잡음) -> 출력 : x
   사례1 : 입력 : x' = x + r(난수 잡음) -> 출력 : x
   사례2 : 입력 : x -> 출력 : x'
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 

tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


### 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) - 3d(size, h, w) -> 2d(size, h*w)
y_train.shape # (60000,)


### 2. x,y변수 전처리 
x_train = x_train / 255.
x_val = x_val / 255. # 255.

x_train.min(), x_train.max() # (0, 1.0)


x_train = x_train.reshape(-1, 28*28) # (60000, 784)
x_val = x_val.reshape(-1, 28*28)
x_train.shape 
x_val.shape 

# 1d -> 3d : CNN을 활용하므로 전체 4차원 텐서 변환 
x_train = x_train.reshape((-1,28,28,1))
x_val = x_val.reshape((-1,28,28,1))
print(x_train.shape)
print(x_val.shape)


### 3. 잡음(noise) 이미지 생성 
# - 가우지안 분포를 따르는 잡음을 이미지에 삽입
x_train_noise = x_train + 0.3*np.random.normal(size=x_train.shape)
x_val_noise = x_val + 0.3*np.random.normal(size=x_val.shape)


# 정상 이미지
plt.imshow(x_train[0])
plt.show()

# 손상 이미지
plt.imshow(x_train_noise[0])
plt.show()


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,UpSampling2D # CNN 추가 
from tensorflow.keras.models import Model



# 4. Functional API model 구축
###############################################
### Functional API : Input과 Model 클래스 이용 
###############################################
inputs = Input(shape=(28, 28, 1))  

# Conv layer1 
x = Conv2D(filters=16, kernel_size=(3, 3), 
           activation='relu', padding='same')(inputs)
x = MaxPooling2D(strides=(2, 2), padding='same')(x)


# Conv layer2
x = Conv2D(filters=8, kernel_size=(3, 3), 
           activation='relu', padding='same')(x)
encoded = MaxPooling2D(strides=(2, 2), padding='same')(x)


# Conv layer3    
x = Conv2D(filters=8, kernel_size=(3, 3), 
           activation='relu', padding='same')(encoded)
x = UpSampling2D(size=(2, 2))(x)

# Conv layer4  
x = Conv2D(filters=8, kernel_size=(3, 3), 
           activation='relu', padding='same')(x)
x = UpSampling2D(size=(2, 2))(x) # 이미지 확대


# output layer
outputs = Conv2D(filters=1, kernel_size=(3, 3), 
                 activation='sigmoid', padding='same')(x) # 비선형함수 



### 5. autoencoder model : encoder + decoder 
autoencoder = Model(inputs, outputs) 
autoencoder.summary()


# 6. model 학습환경 : output 활성함수='sigmoid'에 대한 이항분류  
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')



########################################################
# 사례1 :  model 학습 : 손상된 이미지를 깨끗한 이미지로 복원 
########################################################
'''
 입력 : 원본 이미지 + 잡음 -> 출력 : 원본 이미지 
'''     

# model 학습 
autoencoder.fit(x=x_train_noise, y=x_train, 
          epochs=3, # 반복학습 
          batch_size = 100, verbose=1) 


# 7. model reuse
 
# 1) encoder model
encoder = Model(inputs, encoded) # 인코더 모델 
encoder.summary()
'''
 Layer (type)                Output Shape              Param #   
=================================================================
 input_11 (InputLayer)       [(None, 28, 28, 1)]       0         
                                                                 
 conv2d_5 (Conv2D)           (None, 28, 28, 16)        160       
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 16)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 14, 14, 8)         1160      
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 8)          0         
 2D)                                                             
                                                                 
=================================================================
'''



# 2) decoder model : autoencoder model 대체 
    
# 1) encoder model 예측 이미지 
encoder_img_pred = encoder.predict(x_val_noise) # 손상된 이미지 
encoder_img_pred.shape # (10000, 7, 7, 8)

# 2) decoder model 예측 이미지  
decoder_img_pred = autoencoder.predict(x_val_noise) # 복원된 이미지
decoder_img_pred.shape # (10000, 28, 28, 1)


# 3) 예측 이미지 시각화 
# 잡음 이미지 -> 압축 이미지 -> 복원이미지
n = 10 # 이미지 개수   
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # 잡음(noise) 이미지 10개 
    ax = plt.subplot(3, n, i) # 1~10
    plt.imshow(x_val_noise[i].reshape(28, 28)) # 784픽셀 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # encoder model 예측 이미지 10개   
    ax = plt.subplot(3, n, i + n) # 11~20
    plt.imshow(encoder_img_pred[i,:,:,-1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    
    # decoder model 예측 이미지 10개 : 재구성된 데이터
    ax = plt.subplot(3, n, i + n*2) # 21~30
    plt.imshow(decoder_img_pred[i].reshape(28, 28)) # 784픽셀 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




########################################################
# 사례2 : 학습된 model에 손상이미지를 넣어서 정상 이미지로 복원 
########################################################


# model 학습 
autoencoder.fit(x=x_train, y=x_train, 
          epochs=3, # 반복학습 
          batch_size = 100, verbose=1) 


encoder = Model(inputs, encoded) # 인코더 모델 
encoder.summary()


encoder_img_pred = encoder.predict(x_val_noise) # 손상된 이미지 
encoder_img_pred.shape # (10000, 7, 7, 8)

# 2) decoder model 예측 이미지  
decoder_img_pred = autoencoder.predict(x_val_noise) # 복원된 이미지
decoder_img_pred.shape # (10000, 28, 28, 1)


# 3) 예측 이미지 시각화 
# 잡음 이미지 -> 압축 이미지 -> 복원이미지
n = 10 # 이미지 개수   
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # 잡음(noise) 이미지 10개 
    ax = plt.subplot(3, n, i) # 1~10
    plt.imshow(x_val_noise[i].reshape(28, 28)) # 784픽셀 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # encoder model 예측 이미지 10개   
    ax = plt.subplot(3, n, i + n) # 11~20
    plt.imshow(encoder_img_pred[i,:,:,-1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    
    # decoder model 예측 이미지 10개 : 재구성된 데이터
    ax = plt.subplot(3, n, i + n*2) # 21~30
    plt.imshow(decoder_img_pred[i].reshape(28, 28)) # 784픽셀 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()























