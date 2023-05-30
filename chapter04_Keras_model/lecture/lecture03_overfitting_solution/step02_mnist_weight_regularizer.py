# -*- coding: utf-8 -*-
"""
step02_mnist_weight_regularizer

가중치규제(Weight regularizer(규제) 
  - 가중치가 너무 커지는 것을 방지하기 위해서 가중치를 감소시켜 훈련 데이터에 
     과적합이 발생하지 않도록 하는 기법 
  - 적용 예 : 네트워크 많은 layer 적용
"""

from tensorflow.keras.datasets.mnist import load_data # MNIST dataset 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense, Dropout # DNN layer 구축 
import matplotlib.pyplot as plt # images 

import tensorflow as tf
import numpy as np 
import random as rd 
import time # 실행 시간 측정 

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

y_train[0] # [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] <- 5

y_val[0] # [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.] <- 7


chktime = time.time() # 소요 시간 체크 

# 3. keras model 
model = Sequential()


from tensorflow.keras import regularizers # 가중치 규제 

# 4. DNN model layer 구축 

input_shape = (784,) 

# hidden layer1 : [784, 128] -> [input, output]
model.add(Dense(units=128, input_shape = input_shape, activation = 'relu', 
                kernel_regularizer=regularizers.l1(0.01))) # 1층 

# hidden layer2 : [128, 64] -> [input, output]
model.add(Dense(units=64, activation = 'relu',
          kernel_regularizer=regularizers.l1(0.01))) # 2층 

# hidden layer3 : [64, 32] -> [input, output]
model.add(Dense(units=32, activation = 'relu')) # 3층

# output layer : [32, 10] -> [input, output]
model.add(Dense(units=10, activation = 'softmax')) # 4층 

# model layer 확인 
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 128)               100480=(784*128)+128    
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256=(128*64)+64      
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080=(64*32)+32      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330=(32*10)+10       
=================================================================
Total params: 111,146
Trainable params: 111,146
Non-trainable params: 0
_________________________________________________________________
'''

# 5. model compile : 학습과정 설정(다항분류기)
model.compile(optimizer='adam', # default=0.001
              loss = 'categorical_crossentropy', # y : one hot encoding 
              metrics=['accuracy'])


# 6. model training : train(60,000) vs val(10,000) [수정]
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=15, # 반복학습 
          batch_size = 100, # 100*600 = 60000(1epoch)*10 = 600,000 -> mini batch
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 

chktime = time.time() - chktime  

print('실행 시간 : ', chktime)

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


# accuracy vs val_accuracy : : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

















