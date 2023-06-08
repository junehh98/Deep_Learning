# -*- coding: utf-8 -*-
"""
step02_Model_Reuse.py

모델 재사용(model reuse) : 학습 모델의 가중치와 편향을 재사용하여 new model 생성 
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)
from sklearn.metrics import accuracy_score # model 평가

from tensorflow.keras.utils import to_categorical # Y변수 : encoding 

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

# y변수 : one hot encoding(binary)
y = to_categorical(y)
y.shape 
'''
[1, 0] <- 0
[0, 1] <- 1
'''


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. Functional API model 구축
##################################################
### Functional API : Input과 Model 클래스 이용 
##################################################
from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.layers import Dense # DNN layer 구축
from tensorflow.keras.models import Model # DNN Model 생성 

input_dim = 4 # input data 차원
output_dim = 3 # output data 차원 

# 1) input layer 
inputs = Input(shape=(input_dim,))  

# 2) hidden1 : input[4] -> hidden1[12]
hidden1 = Dense(units=12, activation='relu')(inputs) # 1층 

# 3) hidden2 : hidden1[12] -> hidden2[6] 
hidden2 = Dense(units=6, activation='relu')(hidden1) # 2층 

# 4) output layer : hidden2[6] -> output[3] 
outputs = Dense(units=output_dim, activation='softmax')(hidden2) 


# 5) model 생성 : 구축된 layer을 바탕으로 model 생성 
model = Model(inputs, outputs) # (input_layer, output_layer)

model.summary()


# 4. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(70) vs val(30) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=200, # 반복학습
          verbose=1) # 출력여부 


# 6. model evaluation : val dataset 
print('\n')
print('model evaluation')
loss, accuracy = model.evaluate(x=x_val, y=y_val)
print('loss=', loss)
print('accuracy=',accuracy)




##############################################
### 학습된 model 재사용 
##############################################


### 1. model 계층(layer) 구성과 분리 

# model 계층(layer) 분리  
model.layers[0] # input layer
hidden1_layer = model.layers[1] # hidden1 layer object
hidden2_layer = model.layers[2] # hidden2 layer object
output_layer = model.layers[-1] # output layer object


### 2. 각 계층별 model 생성 : model reuse

# 1) 은닉층1 model : 입력층 -> 은닉층1  
inputs = Input(shape = (4,)) # 입력층 shape 
hidden1_model = Model(inputs, hidden1_layer(inputs)) 


# 2) 은닉층2 model : 은닉층1입력층 -> 은닉층2   
hidden1_inputs = Input(shape=(12,)) # 은닉층1입력층 shape 
hidden2_model = Model(hidden1_inputs, hidden2_layer(hidden1_inputs))  

 
# 3) 출력층 model : 은닉층2입력층 -> 출력층 
hidden2_inputs = Input(shape=(6,)) # 은닉층2입력층 shape 
output_model = Model(hidden2_inputs, output_layer(hidden2_inputs)) 


### 3. 각 계층에서 학습된 가중치와 편향 확인 
model.weights # 전체 모델 

hidden1_model.weights # 은닉층1 

hidden2_model.weights # 은닉층2

output_model.weights # 출력층 

y_pred1 = hidden1_model.predict(x_val)
y_pred1.shape # (45, 12) 4 -> 12

y_pred2 = hidden2_model.predict(y_pred1)
y_pred2.shape # (45, 6) 12 -> 6

y_pred = output_model.predict(y_pred2)
y_pred.shape # (45, 3) 6 -> 3

