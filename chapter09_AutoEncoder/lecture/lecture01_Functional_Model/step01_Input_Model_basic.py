# -*- coding: utf-8 -*-
"""
step01_Input_Model_basic.py

Functional API 모델 : Input과 Model 클래스를 이용하여 딥러닝 계층(layer) 설계 방법  
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
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
y.shape # (150,)
print(X)
print(y)

# X변수 : 정규화(0~1)
X = minmax_scale(X)

# y변수 : one hot encoding(binary)
y = to_categorical(y)
y.shape # (150, 3)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. Functional API model 구축
###############################################
### Functional API : Input과 Model 클래스 이용 
###############################################
from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.models import Model # DNN Model 생성 


input_dim = 4 # input
output_dim = 3 # output

# 1) input layer 
inputs = Input(shape=(input_dim,)) 

# 2) hidden1 layer 
hidden1 = Dense(units=12, activation='relu')(inputs) # 1층 

# 3) hidden2 layer  
hidden2 = Dense(units=6, activation='relu')(hidden1) # 2층 

# 4) output layer  
outputs = Dense(units=output_dim, activation = 'softmax')(hidden2) 


# 5) model 생성 
model = Model(inputs, outputs)



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
model.evaluate(x=x_val, y=y_val)
# loss: 0.1898 - accuracy: 0.9111


dir(model)
'''
 inputs
 layer
 outputs
 predict
 summary()
 weights : w, b = variables : w, b
'''
model.summary()
'''_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 4)]               0         
                                                                 
 dense_3 (Dense)             (None, 12)                60        
                                                                 
 dense_4 (Dense)             (None, 6)                 78        
                                                                 
 dense_5 (Dense)             (None, 3)                 21        
                                                                 
=================================================================
'''
model.layers 
model.layers[1] # 은닉층 1

# 각 계층의 학습된 가중치, 편향 확인
model.weights
























