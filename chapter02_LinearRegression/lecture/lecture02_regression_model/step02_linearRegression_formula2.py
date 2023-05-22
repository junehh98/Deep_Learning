# -*- coding: utf-8 -*-
"""
다중선형회귀방정식 작성 
  예) 독립변수 2개 
   y_pred = (X1 * a1 + X2 * a2) + b
   # 행렬곱
   y_pred = (X @ w) + b
"""

import tensorflow as tf 

# X, y 변수 정의 : 상수
X = [[1.0, 2.0], [1.2, 1.9]] # 독립변수(2, 2)
y = 2.5  # 종속변수


# w, b 변수 정의 : 변수  
tf.random.set_seed(1) # 난수 seed값 
w = tf.Variable(tf.random.normal([2, 1])) # 2개 난수 
b = tf.Variable(tf.random.normal([1])) # 1개 난수 

print('w:', w.numpy())
print('b:', b.numpy())


# 회귀모델 : y 예측치 반환
def linear_model(X) : 
    global w, b
    # y = (X @ w) + b
    y_pred = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return y_pred 

y_pred = linear_model(X)
print(y_pred.numpy())
'''
[[0.42327   ]  <- [1.0, 2.0]
 [0.37255418]] <- [1.2, 1.9]
'''



# model 오차 
def model_err(X, y) : # y=2.5
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 오차  
    return err 

err = model_err(X, y)
print(err.numpy())
'''
[[2.07673  ]  <--  2.5 - 0.42327
 [2.1274457]] <--  2.5 - 0.37255
'''



# 손실/비용 함수
def loss_fu(X, y) :
    err = model_err(X, y) # 오차 
    loss = tf.reduce_mean(tf.square(err)) # 손실함수   
    return loss

loss = loss_fu(X, y)
print('loss:',loss.numpy())






