# -*- coding: utf-8 -*-
"""
step02_keras_history.py

keras history 기능
 - model 학습과정과 검증과정의 손실(loss)을 기억하는 기능 
"""

# dataset 
from sklearn.datasets import load_iris # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# keras model 
import tensorflow as tf
from tensorflow.keras import Sequential # keara model 
from tensorflow.keras.layers import Dense # DNN layer 
import numpy as np 
import random as rd 

# tensorflow version
print(tf.__version__) # 2.3.0
# keras version 
print(tf.keras.__version__) # 2.4.0

## karas 내부 weight seed 적용 
tf.random.set_seed(123) # global seed 
np.random.seed(123) # numpy seed
rd.seed(123) # random seed 

# 1. dataset laod 
X, y = load_iris(return_X_y=True)

y # 0~2


# 2. 공급 data 생성 : 훈련셋, 검증셋 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


x_train.shape # (105, 4)
x_train.dtype # dtype('float64')

y_train.shape # (105,)


# 3. DNN model 생성 
model = Sequential() # keras model 
print(model) # Sequential object

# DNN model layer 구축 
'''
model.add(Dense(unit수, input_shape, activation)) : hidden1
model.add(Dense(unit수, activation)) : hidden2 ~ hiddenn
'''

##########################################
### hidden layer 2개 : unit=12, unit=6
##########################################

# hidden layer1 : unit=12 -> w[4, 12]
model.add(Dense(units=12, input_shape=(4,), activation='relu')) # 1층 

# hidden layer2 : unit=6 -> w[12, 6]
model.add(Dense(units=6, activation='relu')) # 2층

# output layer : unit=1 -> w[6, 1]
model.add(Dense(units=1)) # 3층 

# model layer 확인 
print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param(in*out)+b   
=================================================================
dense_6 (Dense)              (None, 12)                60=(4*12)+12        
_________________________________________________________________
dense_7 (Dense)              (None, 6)                 78=(12*6)+6        
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 7=(6*1)+1         
=================================================================
'''

# 4. model compile : 학습과정 설정 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# 5. model training : train(105) vs test(45)
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=100, # 반복학습 횟수 
          verbose=1,  # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋  


# 6. model evaluation : validation data 
loss_val, mae = model.evaluate(x_val, y_val)
print('loss value =', loss_val)
print('mae =', mae)


# 7. model history : epoch에 따른 model 평가  

# 1) epoch vs loss
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

