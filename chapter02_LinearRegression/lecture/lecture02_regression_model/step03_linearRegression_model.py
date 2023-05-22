# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용  선형회귀모델 
"""

import tensorflow as tf  
import numpy as np  
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 1. X, y변수 생성 
X = np.array([1, 2, 3]) # 독립변수(입력) 
y = np.array([2, 4, 6]) # 종속변수(출력) 

# 2. w, b변수 정의(수정가능)
tf.random.set_seed(123)
# 가중치의 개수는 독립변수의 개수에 비례
w  = tf.Variable(tf.random.normal([1])) # 가중치 : 난수 
b  = tf.Variable(tf.random.normal([1])) # 편향 : 난수 



# 3. 회귀모델 : y예측치 반환
def linear_model(X) :  
    # y= (X * w) + b
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식 
    return y_pred 



# 4. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss

# 손실함수 호출
loss = loss_fn() #  손실(loss)
print('loss :', loss.numpy())


# 5. model 최적화 객체  
optimizer = tf.optimizers.Adam(learning_rate=0.1) # 딥러닝 최적화 알고리즘
# Defaluts : learnin_rate=0.01
dir(optimizer) # minimize -> 손실 최소
dir(tf.optimizers)
'''
 'Adadelta',
 'Adagrad',
 'Adam',
 'Adamax',
 'Ftrl',
 'Nadam',
 'Optimizer',
 'RMSprop',
 'SGD'
'''


# 6. 반복학습 : 100회
for step in range(100) : # 0~99
    # obj.minimize(loss=손실값, var_list=[가중치, 편향])
    optimizer.minimize(loss=loss_fn, var_list=[w, b])
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())

    # 조절변수 (w, b) 수정내용
    print('가중치(w) =', w.numpy(), '편향(b)=', b.numpy())

'''
  1차 : SGD(learning_rate=0.01)
  step = 100, loss value = 0.1620378
  
  2차 : SGD(learning_rate=0.1), 높은 학습률
  step = 100, loss value = 0.002018992
  
  
  학습율 0.1로 했을때 0.01이 100번 해야할 loss를 9번만에 찾아냄
   step = 9 , loss value = 0.16922037
   
   
   3차 : Adam(leaning_rate = 0.001) # Defaluts = 0.001
   step = 100, loss value = 31.849894
   
   4차 : Adam(leaning_rate = 0.1)
   step = 100 , loss value = 0.3030051
   
   # 최적의 가중치와 편향을 찾아야함
 
'''

# 7. model 최적화 test
X_test = [2.5] # 테스트용 독립변수

y_pred = linear_model(X_test) # w, b -> 수정(update)
print('y_pred :', y_pred.numpy())


# 전체 X변수
linear_model(X) # [1,2,3]
# 원래값 [1,2,3] -> [2,4,6] 2배 정도로 나옴

y_pred = linear_model(X)
print('y_pred :', y_pred.numpy())
print('y=', y)
# y= [2 4 6] ->  # y_pred : [2.7999618 4.1520033 5.504045 ]
# 실제값과 비슷하게 나옴 


# 산점도 vs 회귀선
import matplotlib.pyplot as plt

plt.plot(X, y, 'bo') # 산점도 생성
plt.plot(X, y_pred.numpy(), 'r-') # 회귀선
plt.show()











