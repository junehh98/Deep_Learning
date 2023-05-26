# -*- coding: utf-8 -*-
"""
Keras : DNN model 생성을 위한 고수준 API
 
Keras 이항분류기 
 - X변수 : minmax_scale(0~1)
 - y변수 : one hot encoding(2진수 인코딩)
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. dataset load & 전처리 
X, y = load_iris(return_X_y=True)
X.shape
y.shape


# X변수 : 정규화
X = minmax_scale(X[:100]) # 100개 선택 


# y변수 : 10진수(label encoding)   
y = y[:100] # 100개 선택 
'''
10진수(label)  2진수(one-hot)
0           -->  1 0
1           -->  0 1
'''
y = to_categorical(y) # num_classes=2 
y.shape


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. keras layer & model 생성

model = Sequential()

# hidden layer1 : w[4, 8], b=8 
model.add(Dense(units=8, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2 : w[8, 4], b=4 
model.add(Dense(units=4, activation = 'relu')) # 2층 

# output layer : w[4, 2], b=2
model.add(Dense(units=2, activation = 'sigmoid')) # 3층 
'''
 10진수(label)   : output layer : units=1
 2진수(one hot)  : output layer : units=2
 label이 3개이상이면 sigmoid -> softmax
'''


# 4. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='adam', # 최적화 알고리즘(adam or sgd)
              loss = 'binary_crossentropy', # 손실함수(crossentropy)
              metrics=['accuracy']) # 평가방법


# 5. model training : train(70) vs val(30) 
model_fit= model.fit(x=x_train, y=y_train, 
          epochs=20,  
          verbose=1,  
          validation_data=(x_val, y_val)) 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)



import matplotlib.pyplot as plt
plt.plot(model_fit.history['loss'], 'y', label='train loss value')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss value')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()

# 2) epoch vs mae
plt.plot(model_fit.history['mae'], 'y', label='train mae')
plt.plot(model_fit.history['val_mae'], 'r', label='val mae')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.legend(loc='best')
plt.show()






 