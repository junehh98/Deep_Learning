# -*- coding: utf-8 -*-
"""
다항분류기 : 테스트 데이터 적용 
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score #  model 평가 
import numpy as np 

# 1. x, y 공급 data 
# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1]]) # [6, 2]

# [기타, 포유류, 조류] : [6, 3] 
y_data = np.array([ # one hot encoding 
    [1, 0, 0],  # 기타[0]
    [0, 1, 0],  # 포유류[1]
    [0, 0, 1],  # 조류[2]
    [1, 0, 0],  # 기타[0]
    [1, 0, 0],  # 기타[0]
    [0, 0, 1]   # 조류[2]
])


# 2. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # [관측치, 입력수] - [6, 2]
Y = tf.constant(y_data, tf.float32) # [관측치, 출력수] - [6, 3]


# 3. w, b변수 정의 : 초기값(난수) -> update 
w = tf.Variable(tf.random.normal(shape=[2, 3])) # [입력수, 출력수]
b = tf.Variable(tf.random.normal(shape=[3])) # [출력수]


# 4. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 


# 5. softmax 함수   
def softmax_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.softmax(model) 
    return y_pred 


# 6. 손실함수 : 손실값 반환 
def loss_fn() : # 인수 없음 
    y_pred = softmax_fn(X)
    # cross entropy : loss value 
    loss = -tf.reduce_mean(Y * tf.math.log(y_pred) + (1-Y) * tf.math.log(1-y_pred))
    return loss


# 7. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.1)


# 8. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
  
    




