# -*- coding: utf-8 -*-
"""
문) input data 2개를 입력받아서 정답을 예측하는 다중입력 모델을 구현하시오.[내용 채우기] 

  조건1> 첫번째 input data : iris 1~4번 칼럼 
  조건2> 두번째 input data : 난수 자료(random data) 
  조건3> output data : iris 5번 칼럼(꽃의종 2개)

  <model layer 구성>
  input1
          ->  DNN -> output       
  input2          
"""
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)
from tensorflow.keras.utils import to_categorical # Y변수 : encoding

from tensorflow.keras.layers import Concatenate # layer 묶음 
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


# 1-1. iris 입력자료 만들기  
X, y = load_iris(return_X_y=True)

# X변수 : 정규화
X = minmax_scale(X[:100]) # 100개 선택 
X.shape # (100, 4)

# y변수 
y = y[:100] # 100개 선택 
y = to_categorical(y) # one-hot encoding 
y.shape # (100, 2)


# iris dataset 공급 data 생성 : 훈련용:70, 검증용:30 
x_train_iris, x_val_iris, y_train_iris, y_val_iris = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 1-2. 난수(random) 입력자료 만들기  
x_train_r = tf.random.normal(shape=(70,)) 
x_val_r = tf.random.normal(shape=(30,)) 

x_train_r.shape #  # 70개
x_val_r.shape # # 30개



# 2. keras layer & model 생성
####################################################
## 2. Functional API 방식 : 개발자용(엔지니어)
####################################################
from tensorflow.keras.layers import Input, Dense # input layer
from tensorflow.keras.models import Model # DNN Model 생성 


input_dim = 4 # input data 차원
output_dim = 2 # output data 차원 


# 1) iris data input : [내용 채우기]   
input_iris = Input(shape=(input_dim,))  
hidden1 = Dense(units=12, activation='relu')(input_iris) # 1층 : 12개 노드 
output_iris = Dense(units=6, activation = 'relu')(hidden1) # 2층 : 6개 노드 


# 2) 일반 data input  
input_data = Input(shape=(1,))
output_data = Dense(units=6, activation = 'relu')(input_data)



# input layer 2개 묶음 : [내용 채우기]  
concatted= Concatenate()([output_iris, output_data])


# out layer 2개 -> DNN -> output 1개 
dnn = Dense(3,'relu')(concatted)
outputs = Dense(units=output_dim, activation='sigmoid')(dnn)


# model 생성 : [내용 채우기]  
model = Model(inputs=[input_iris, input_data], outputs = outputs) 


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])


# 5. model training : [내용 채우기] 
model_fit=model.fit(x=(x_train_iris, x_train_r), y = y_train_iris,
                    epochs=80,
                    validation_data=([x_val_iris, x_val_r], y_val_iris))


# 6. model evaluation : [내용 채우기]  
model.evaluate([x_val_iris, x_val_r], y_val_iris)


model.summary()







 







