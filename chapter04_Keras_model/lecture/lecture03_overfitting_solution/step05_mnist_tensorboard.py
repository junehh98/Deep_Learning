# -*- coding: utf-8 -*-
"""
step05_keras_mnist_tensorboard.py

 Tensorboard : loss value, accuracy 시각화 
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense, Dropout # DNN layer 구축 

import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. mnist dataset laod 
(x_train, y_train), (x_val, y_val) = load_data() # (images, labels)

x_train.shape # (60000, 28, 28) - 3d(size, h, w) -> 2d(size, h*w)
y_train.shape # (60000,)

x_train[0] # first image pixel 
x_train[0].max() # 0~255

plt.imshow(x_train[0])
plt.show()

y_train[0] # first label - 10진수 : 5


# 2. x,y변수 전처리 

# 1) x변수 : 정규화 & reshape(3d -> 2d)
x_train = x_train / 255.
x_val = x_val / 255.

x_train[0]


# 3d -> 2d : [수정]
x_train = x_train.reshape(-1, 784)
# 28 * 28 = 784
x_train.shape # (60000, 784)

x_val = x_val.reshape(-1, 784)
x_val.shape # (10000, 784)


# 2) y변수 : one hot encoding 
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 
y_train.shape # (60000, 10)



# 3. keras model & layer 구축
model = Sequential()


input_dim = (784,) 

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_dim, activation = 'relu')) # 1층 
model.add(Dropout(rate = 0.3)) 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu')) # 2층 
model.add(Dropout(rate = 0.1)) 

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층
model.add(Dropout(rate = 0.1)) 

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 

# model layer 확인 
model.summary()


# 5. model compile : 학습과정 설정(다항분류기) - [수정]
from tensorflow.keras import optimizers

# optimizer='adam' -> default=0.001
model.compile(optimizer=optimizers.Adam(), # default=0.001
              loss = 'categorical_crossentropy', # y : one hot encoding 
              metrics=['accuracy'])


# [추가]
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime # 20210716-1540 

# 6. TensorBoard 시각화 
logdir ='c:\\graph\\' + datetime.now().strftime("%Y%m%d-%H%M%S")
callback = TensorBoard(log_dir=logdir)

model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=20, # 반복학습 
          batch_size = 100, #  mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val),
          callbacks = [callback]) # 검증셋 


# 7. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 8. model history 
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


import matplotlib.pyplot as plt 


# loss vs val_loss : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()














