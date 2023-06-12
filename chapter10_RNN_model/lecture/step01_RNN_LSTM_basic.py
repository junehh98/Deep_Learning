# -*- coding: utf-8 -*-
"""
step01_RNN_LSTM_basic.py

RNN model 
 - 순환신경망 Many to One RNN 모델(PPT.8 참고)  
"""

import tensorflow as tf # seed value 
import numpy as np # ndarray
from tensorflow.keras import Sequential # model
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM # RNN layer 

import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# many-to-one : word(4개) -> 출력(1개)
X = [[[0.0], [0.1], [0.2], [0.3]], 
     [[0.1], [0.2], [0.3], [0.4]],
     [[0.2], [0.3], [0.4], [0.5]],
     [[0.3], [0.4], [0.5], [0.6]],
     [[0.4], [0.5], [0.6], [0.7]],
     [[0.5], [0.6], [0.7], [0.8]]] 

Y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 정답 

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

X.shape # (6, 4, 1) : RNN 3차원 입력(batch_size, time_steps, features)

model = Sequential() 

input_shape = (4, 1) # (timestep, feature)

# RNN layer 추가 
model.add(LSTM(units=30, input_shape=input_shape, 
                    return_state=False, # Many to One
                    activation='tanh'))
'''
SimpleRNN -> LSTM
'''
# DNN layer 추가 
model.add(Dense(units=1)) # 출력 : 회귀모델 

# model 학습환경 
model.compile(optimizer='adam', 
              loss='mse', metrics=['mae'])

# model training 
model.fit(X, Y, epochs=50, verbose=1)

# model prediction
y_pred = model.predict(X)
print(y_pred)
'''
   SimpleRNN
[[0.3498212 ] - 0.4
 [0.47630942] - 0.5
 [0.59820306] - 0.6
 [0.7142566 ] - 0.7
 [0.8235769 ] - 0.8
 [0.92562103]]- 0.9
    LSTM
[[0.4291078 ]
 [0.51832134]
 [0.6098001 ]
 [0.70309883]
 [0.7977449 ]
 [0.89324903]]   
'''











