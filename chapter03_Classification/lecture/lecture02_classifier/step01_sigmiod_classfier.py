# -*- coding: utf-8 -*-
"""
이항분류기 : 테스트 데이터 적용
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score #  model 평가 

# 1. x, y 공급 data 
# x변수 : [hours, video]
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # [6,2]

# y변수 : binary data (fail or pass)
y_data = [[0], [0], [0], [1], [1], [1]] # [6, 1] : 이항분류 (10진수)
'''
 10진수  2진수(one-hot-encoding)
 0  ->   1 0 0
 1  ->   0 1 0
 2  ->   0 0 1
'''



# 2. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # shape=(6, 2)
y = tf.constant(y_data, tf.float32) # shape=(6, 1)


# 3. w, b변수 정의 : 초기값(난수)  
w = tf.Variable(tf.random.normal(shape=[2, 1])) 
b = tf.Variable(tf.random.normal(shape=[1])) 



# 4. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 
    

# 5. sigmoid 함수  : 이항분류 활성함수 
def sigmoid_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.sigmoid(model) 
    return y_pred 

    
# 6. 손실함수 : cross entropy 이용 
def loss_fn() : # 인수 없음 
    y_pred = sigmoid_fn(X)
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss



# 7. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.5)


# 8. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
    
'''
step = 10 , loss val =  0.032901127
step = 20 , loss val =  0.03120943
step = 30 , loss val =  0.01883106
step = 40 , loss val =  0.016813932
step = 50 , loss val =  0.013855136
step = 60 , loss val =  0.0120356195
step = 70 , loss val =  0.010643013
step = 80 , loss val =  0.009447734
step = 90 , loss val =  0.008466918
step = 100 , loss val =  0.0076404433
'''

# 9. model 평가
y_pred = sigmoid_fn(X).numpy()
print(y_pred)
'''
[[0.00439674]
 [0.08868311]
 [0.11279511]
 [0.8778197 ]
 [0.9840596 ]
 [0.99518853]]
'''

# tf.cast(data, type) : 형변환 함수
y_pred = tf.cast(y_pred > 0.5, dtype=tf.int16)
print(y_pred)
'''
[[0]
 [0]
 [0]
 [1]
 [1]
 [1]]
'''
acc = accuracy_score(y_true=y, y_pred=y_pred )
print(acc)












