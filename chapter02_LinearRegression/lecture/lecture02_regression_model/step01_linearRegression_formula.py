# -*- coding: utf-8 -*-
"""
단순선형회귀방정식(formula) 작성
"""

import tensorflow as tf  # ver 2.x

# X, y 변수 
X = tf.constant(6.5) # 독립변수(입력)
y = tf.constant(5.2) # 종속변수(출력/정답)

# w, b 변수 : 변수 정의(수정 가능)
w = tf.Variable(0.5) # 가중치(기울기)
b = tf.Variable(1.5) # 편향(절편)


# 회귀모델 : 입력(X) -> y예측치
def linear_model(X) : 
    global w, b
    # y = X*w + b (단순선형회귀)
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식  
    return y_pred # 4.75


# model 오차 : err = y - y_pred
def model_err(X, y) : 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # y - y예측치
    return err 



# 손실 함수(loss/cost) : 손실반환(MSE) -> 평균제곱오차
def loss_fn(X, y) :
    err = model_err(X, y) 
    loss = tf.reduce_mean(tf.square(err)) # 손실함수
    return loss

'''
 tf.square(err) -> tf.square(y - y_pred) 
   -> +로 기호변경, 오차 페널티
 tf.reduce_mean() : 각 관측치의 오차의 평균
'''


print('\n<<가중치, 편향 초기값>>')    
print('가중치(w) = %.3f, 편향(b) = %.3f'%(w, b))

print('model error =%.3f'%(model_err(X, y)))    
print('loss value = %.3f'%(loss_fn(X, y)))

'''
<<가중치, 편향 초기값>>
가중치(w) = 0.500, 편향(b) = 1.500
model error =0.450 : y - y_pred
loss value = 0.202 : MSE
'''



# 2차 : 가중치와 편향 수정(사용자)
dir(w) # assign(value) -> 값 수정
w.assign(0.6) # 가중치 0.5 -> 0.6 수정
b.assign(1.2) # 편향 1.5 -> 1.2 수정

print('\n<<가중치, 편향 수정>>')    
print('가중치(w) = %.3f, 편향(b) = %.3f'%(w, b))

print('model error =%.3f'%(model_err(X, y)))    
print('loss value = %.3f'%(loss_fn(X, y)))

























