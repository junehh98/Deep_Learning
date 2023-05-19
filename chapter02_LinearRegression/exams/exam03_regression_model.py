'''
문3) women.csv 데이터 파일을 이용하여 선형회귀모델 생성하시오.
     <조건1> x변수 : height,  y변수 : weight
     <조건2> learning_rate=0.5
     <조건3> 최적화함수 : Adam
     <조건4> 반복학습 : 200회
     <조건5> 학습과정 출력 : step, loss_value
     <조건6> 최적화 모델 검증 : MSE, 회귀선 시각화  
'''
import tensorflow as tf # ver2.x 

import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

women = pd.read_csv(r'C:\ITWILL\6_DeepLearning\data\women.csv')
print(women.info())
print(women)

# 1. x,y data 생성 
x_data = women['height']
y_data = women['weight']

# 정규화 
print(x_data.max()) # 72
print(y_data.max()) # 164

# 2. 정규화(0~1)
X = x_data / 72
Y = y_data / 164

# 자료형 변환 
X = tf.constant(X, dtype=tf.float32)
Y = tf.constant(Y, dtype=tf.float32)

# 3. w,b변수 정의 - 난수 이용 
w = tf.Variable(tf.random.uniform([1], 0.1, 1.0))
b = tf.Variable(tf.random.uniform([1], 0.1, 1.0))


# 4. 회귀모델 
def linear_model(X) : # 입력 X
    y_pred = tf.multiply(X, w) + b # y_pred = X * a + b
    return y_pred

# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 
def loss_fn() : #  인수 없음 
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.square(err)) # 오차제곱평균(MSE) 
    return loss

# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체  


# 7. 반복학습 : 200회 


# 8. 최적화된 model 검증
# 1) MSE 평가 


# 2) 회귀선    


    
  
    