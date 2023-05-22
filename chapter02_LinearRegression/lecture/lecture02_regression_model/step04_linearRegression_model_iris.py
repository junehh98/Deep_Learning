# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용  선형회귀모델 + csv file 
"""

import tensorflow as tf # 최적화 알고리즘 
import pandas as pd  # csv file 
from sklearn.preprocessing import minmax_scale # 정규화 
from sklearn.metrics import mean_squared_error # model 평가 

iris = pd.read_csv('C:/ITWILL/6_Tensorflow/data/iris.csv')
print(iris.info())

# 1. X, y data 생성
x_data = iris['Sepal.Length'] 
y_data = iris['Petal.Length']


# 2. X, y변수 만들기     
X = tf.constant(x_data, dtype=tf.float32) # dtype 지정 
y = tf.constant(y_data, dtype=tf.float32) # dtype 지정 


# 3. a,b 변수 정의 : 초기값 - 난수  
tf.random.set_seed(123)
w = tf.Variable(tf.random.normal([1])) # 가중치 
b = tf.Variable(tf.random.normal([1])) # 편향 


# 4. 회귀모델 
def linear_model(X) : # 입력 : X -> y예측치 
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식 
    return y_pred 


# 5. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss


# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체 
optimizer = tf.optimizers.Adam(learning_rate=0.5) # lr : 0.9 ~ 0.0001(e-04)

print(f'기울기(w) 초기값 = {w.numpy()}, 절편(b) 초기값 = {b.numpy()}')

# 7. 반복학습 : 100회
for step in range(100) :
    optimizer.minimize(loss=loss_fn, var_list=[w, b])#(손실값,update 대상변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())
    # a, b 변수 update 
    print(f'기울기(w) = {w.numpy()}, 절편(b) = {b.numpy()}')
    

