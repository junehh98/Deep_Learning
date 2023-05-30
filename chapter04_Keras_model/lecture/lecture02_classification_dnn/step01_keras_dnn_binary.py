# -*- coding: utf-8 -*-
"""
Keras : DNN model 생성을 위한 고수준 API
 
Keras 이항분류기 
 - X변수 : minmax_scale(0~1)
 - y변수 : one hot encoding(2진수 인코딩) 
   0 -> 1 0
   1 -> 0 1
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
X.shape # (150, 4)
y.shape 
y
# X변수 : 정규화
X = minmax_scale(X[:100]) # 100개 선택(0~1)
X.shape # (100, 4) 

# y변수 : 10진수(label encoding)   
y = y[:100] # 100개 선택 
y.shape # (100,)
y # 0 or 1


# one-hot encoding : 10진수 -> 2진수 
y = to_categorical(y) # 권장 
y
'''
10진수(lebel)  2진수(one-hot)
0    -> 1 0
1    -> 0 1
''' 


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. keras layer & model 생성

model = Sequential() # keras model 생성 

# hidden layer1 : w[4, 8], b=8 
model.add(Dense(units=8, input_shape =(4,), activation = 'relu')) # 1층 

# hidden layer2 : w[8, 4], b=4 
model.add(Dense(units=4, activation = 'relu')) # 2층 

# output layer : w[4, 2], b=2 
model.add(Dense(units=2, activation = 'sigmoid')) # 3층 
'''
y - 10진수(lebel) : output layer : units=1
y - 2진수(one-hot) : output layer : units=2
'''

# 4. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='adam', # 최적화 알고리즘(adam or sgd) 
              loss = 'binary_crossentropy', # 손실함수(crossentropy) 
              metrics=['accuracy']) # 평가방법 


# 5. model training : train(70) vs val(30) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=25,  # 반복학습 횟수 
          verbose=1,  # 출력 여부 
          validation_data=(x_val, y_val))  # 검증셋 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)










 