# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용  다중선형회귀모델 + csv file 
 - y변수 : 1칼럼, X변수 : 2~4칼럼 
 - 딥러닝 최적화 알고리즘 : Adam 적용 
 - 반복학습(step) 적용 
"""

import tensorflow as tf  # 딥러닝 최적화 알고리즘
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import mean_squared_error # 평가 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



# 1. dataset load 
X, y = load_iris(return_X_y=True)

# X변수 정규화 
X_nor = minmax_scale(X)
type(X_nor) # numpy.ndarray


# y변수 : 1칼럼, X변수 : 2~4칼럼 
y_data = X_nor[:,0] # 1칼럼  - y변수 
x_data = X_nor[:,1:] # 2~4칼럼 - x변수 


# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=123)


x_train.dtype
x_train.shape # (105, 3)
# 가중치값도 수일치 시켜줘야함



# 3. w, b 변수 정의 : update 대상 
tf.random.set_seed(123) # w,b 난수 seed값 지정 
w = tf.Variable(tf.random.normal(shape=[3, 1], dtype='float64')) 
b = tf.Variable(tf.random.normal(shape=[1], dtype='float64')) 

w.dtype # tf.float64  --> x데이터와 자료형이 같아짐 



# 활성함수 : 항등함수(x -> x)
def activation(model) :
    return model 



# 4. 회귀모델 정의 : 행렬곱 이용 
def linear_model(X) : # X:입력 -> y 예측치 : 출력 
    model = tf.linalg.matmul(X, w) + b  # (x_train, w)
    y_pred = activation(model) # 활성함수
    return y_pred 

y_pred = linear_model(x_train)
'''
 InvalidArgumentError : 
     1.행렬곱 수 불일치 (x_train -> float64, w -> float32 타입)
     2.자료형 불일치
'''
y_pred.numpy()



# 5. 손실/비용 함수 정의 - MSE
def loss_fn() : # 인수 없음 
    y_pred = linear_model(x_train) # y 예측치
    err = tf.math.subtract(y_train, y_pred) # y - y_pred 
    loss = tf.reduce_mean(tf.square(err)) # MSE 
    return loss 



# 6. 최적화 객체 생성 
opt = tf.optimizers.Adam(learning_rate=0.01) # 학습률 : 0.1 -> 0.01
'''
1. learning_rate = 0.01, step=100
step = 100 , loss value = 0.5905510724168781
 - 비교적 안정적으로 최소점 수렴
 
2. learning_rate = 0.1, step=500
step = 500 , loss value = 0.05876381024096825
 - 비교적 빠른속도로 최소점 수렴, 모델 성능 개선
 
'''


print('초기값 : w =', w.numpy(), ", b =", b.numpy())



# 7. 반복학습
loss_value = [] # 각 stepp 단위 손실 저장 

for step in range(100) : # 100 -> 500
    opt.minimize(loss=loss_fn, var_list=[w, b]) 
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ', loss value =', loss_fn().numpy())
        print('가중치 : w =', w.numpy(), ", 편향 =", b.numpy())
        
    loss_value.append(loss_fn().numpy()) # 손실 저장 



# 8. 모델 성능 평가
y_pred = linear_model(x_test) # x_test로 실제 데이터 예측
y_pred = y_pred.numpy()


mse = mean_squared_error(y_test, y_pred)
print('mse :',mse)


# 9. loss 그래프 그리기 
len(loss_value) # 500

import matplotlib.pyplot as plt

plt.plot(loss_value, 'r--')
plt.ylabel('loss value')
plt.xlabel('step')
plt.show()











