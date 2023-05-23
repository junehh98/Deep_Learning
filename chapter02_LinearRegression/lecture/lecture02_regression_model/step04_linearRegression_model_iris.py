# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용  선형회귀모델 + csv file 
"""

import tensorflow as tf # 최적화 알고리즘 
import pandas as pd  # csv file 
from sklearn.preprocessing import minmax_scale # 정규화 
from sklearn.metrics import mean_squared_error # model 평가 

iris = pd.read_csv('C:/ITWILL/6_DeepLearning/data/iris.csv')
print(iris.info())

# 1. X, y data 생성
x_data = iris['Sepal.Length'] 
y_data = iris['Petal.Length']

# x_data.dtype() # numpy.dtype[float64]
x_data.mean() # 5.843333333333334

# 라이브러리 사용 최대-최소 스케일링
x_data = minmax_scale(x_data)
x_data.mean() #  0.42870370370370364


# 라이브러리 사용 X 최대-최소 스케일링 
y_data.max() # 6.9
y_data = y_data / y_data.max()
y_data.mean() # 0.5446376811594204



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

y_pred = linear_model(X)
len(y_pred) # 150



# 5. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss



# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체 
optimizer = tf.optimizers.Adam(learning_rate=0.5) # lr : 0.9 ~ 0.0001(e-04)

print(f'기울기(w) 초기값 = {w.numpy()}, 절편(b) 초기값 = {b.numpy()}')



# 7. 반복학습 : 100회 --> 200회
for step in range(100) :
    optimizer.minimize(loss=loss_fn, var_list=[w, b])#(손실값,update 대상변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())
    # a, b 변수 update 
    print(f'가중치(w) = {w.numpy()}, 편향(b) = {b.numpy()}')

'''
----------스케일링 전-------------
step = 100 , loss value = 1.4621323
가중치(w) = [0.8448554], 편향(b) = [-1.039748]
 --> 학습율을 높이거나, 반복학습 횟수를 높일 필요가 있음
 
 ---------스케일링 후-------------
 step = 100 , loss value = 0.015607685
 가중치(w) = [0.9687343], 편향(b) = [0.12870643]
 
 --> 스케일링 전에 비해  loss가 줄어듬
'''

# 8. model 평가
y_pred = linear_model(X) # 최적화된 model에서 y예측치
y_pred = y_pred.numpy()


mse = mean_squared_error(y, y_pred) # (y_true, y_pred)
mse # 0.015607246























    

