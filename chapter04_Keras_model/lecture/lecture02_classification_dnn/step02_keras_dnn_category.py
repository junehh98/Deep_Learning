# -*- coding: utf-8 -*-
"""
keras 다항분류기 
 - X변수 : minmax_scale(0~1)
 - y변수 : one hot encoding(2진수 인코딩)
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)
from sklearn.metrics import accuracy_score  # model 평가 

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 
from tensorflow.keras.models import load_model # model load 

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
y.shape # (150,)


# X변수 : 정규화(0~1)
X = minmax_scale(X) 

# y변수 : one hot encoding
y = to_categorical(y) 
y.shape # (150, 3) --> 3개의 범주 


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. keras layer & model 생성
model = Sequential()

# hidden layer1 : [4, 12] -> [input, output]
model.add(Dense(units=12, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2 : [12, 6] -> [input, output]
model.add(Dense(units=6, activation = 'relu')) # 2층 

# output layer : [6, 3] -> [input, output]
model.add(Dense(units=3, activation = 'softmax')) # 3층 : [수정]

model.summary()


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

'''
 이항분류 : loss = 'binary_crossentropy'
 다항분류 : loss = 'categorical_crossentropy'
'''

# 5. model training : train(70) vs val(30) 
model_fit= model.fit(x=x_train, y=y_train, 
          epochs=100,  # historty를 보고 최고의 정확도 찾기
          verbose=1,  
          validation_data=(x_val, y_val)) 


# 6. model evaluation : val dataset 
print('*-'*30)
print('model evaluation') # loss: 0.3747 - accuracy: 0.8667
model.evaluate(x=x_val, y=y_val)

