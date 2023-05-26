# -*- coding: utf-8 -*-
"""
step05_keras_mnist_flatten.py

Flatten layer : input data의 차원을 은닉층에 맞게 일치
  ex) 2차원 이미지(28, 28) -> 1차원(784) 
"""

from tensorflow.keras.datasets import mnist # mnist load 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding 
from tensorflow.keras import Sequential # keras model 생성 
from tensorflow.keras.layers import Dense, Flatten # DNN layer 구축 

################################
## keras 내부 w,b변수 seed 적용 
################################
import tensorflow as tf
import numpy as np 
import random as rd

tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset load 
(x_train, y_train), (x_val, y_val) = mnist.load_data() # (images, labels)

# images : X변수 
x_train.shape # (60000, 28, 28) - (size, h, w) : 2d 제공 
x_val.shape # (10000, 28, 28)

x_train[0] # 0~255
x_train.max() # 255

# labels : y변수 
y_train.shape # (60000,)
y_train[0] # 5


# 2. X,y변수 전처리 

# 1) X변수 : 정규화 
x_train = x_train / 255. # 정규화 
x_val = x_val / 255.



# 2) y변수 : one-hot encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# 3. keras layer & model 생성
model = Sequential()


# flatten layer [추가] 
model.add(Flatten(input_shape = (28,28)))  # 2d -> 1d(784)


# hidden layer1 : w[784, 128]
model.add(Dense(units=128, input_shape=(784,), activation='relu'))# 1층 

# hidden layer2 : w[128, 64]
model.add(Dense(units=64, activation='relu'))# 2층 

# hidden layer3 : w[64, 32]
model.add(Dense(units=32, activation='relu'))# 3층

# output layer : w[32, 10]
model.add(Dense(units=10, activation='softmax'))# 4층


#  model layer 확인 
model.summary()


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', # y : one-hot encoding
              metrics=['accuracy'])


# 5. model training : train(70) vs val(30)
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=10, # 반복학습 횟수 
          batch_size=100, # 1회 공급data 크기  
          verbose=1, # 출력여부 
          validation_data= (x_val, y_val)) # 검증셋


# 6. model evaluation : val dataset 
print('model evaluation')
model.evaluate(x=x_val, y=y_val)





