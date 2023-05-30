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

# y변수 : one hot encoding [필수]
y = to_categorical(y) 
y.shape # (150, 3)
'''
0 -> 1 0 0 
1 -> 0 1 0
2 -> 0 0 1
'''

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. keras layer & model 생성
model = Sequential()

####################################################
## 1. Sequential API 방식 : 초보자용 
####################################################
# hidden layer1 : [4, 12] -> [input, output]
model.add(Dense(units=12, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2 : [12, 6] -> [input, output]
model.add(Dense(units=6, activation = 'relu')) # 2층 

# output layer : [6, 3] -> [input, output]
model.add(Dense(units=3, activation = 'softmax')) # 3층 : [수정]

model.summary() # Total params: 159


####################################################
## 2. Functional API 방식 : 개발자용(엔지니어)
####################################################
# 순방향(forward) 레이어 구축 : Input -> Hedden > Output

from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.models import Model # DNN Model 생성 

input_dim = 4 # input data 차원
output_dim = 3 # output data 차원 

# 1) input layer 
inputs = Input(shape=(input_dim,))  # Input(shape=(4,)) 

# 2) hidden layer1 : input[4] -> hidden1[8]
hidden1 = Dense(units=12, activation='relu')(inputs) # 1층 

# 3) hidden layer2 : hidden1[8] -> hidden2[4] 
hidden2 = Dense(units=6, activation='relu')(hidden1) # 2층 

# 4) output layer : hidden2[4] -> output[2] 
outputs = Dense(units=output_dim, activation = 'softmax')(hidden2)


## model 생성 : 구축된 layer을 바탕으로 model 생성 
model = Model(inputs, outputs) # 입력과 출력 연결 


# model layer 확인 
model.summary() # Total params: 159


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])
'''
이항분류 : loss = 'binary_crossentropy'
다항분류 : loss = 'categorical_crossentropy'
'''

# 5. model training : train(70) vs val(30) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=90,  # 반복학습 횟수 
          verbose=1,  # 출력 여부 
          validation_data=(x_val, y_val))  # 검증셋 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 7. model save & load : HDF5 파일 형식 
model.save('keras_model_iris.h5') # model 저장 

my_model = load_model('keras_model_iris.h5') # model 가져오기 

pred = my_model.predict(x_val) # 확률예측 
# [0.0752853 , 0.32583332, 0.59888136]

# 10진수 변환 
y_pred = tf.argmax(pred, axis=1)
y_true = tf.argmax(y_val, axis=1)

# 분류정확도 
acc = accuracy_score(y_true, y_pred)
print('accuracy =', acc)









