# -*- coding: utf-8 -*-
"""
다항분류기 : 실제 데이터(iris) 적용
"""

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import minmax_scale # x변수 전처리
from sklearn.preprocessing import OneHotEncoder # y변수 전처리
from sklearn.metrics import accuracy_score #  model 평가 

tf.random.set_seed(123) # seed 고정 - 동일 결과 

# 1. data load  
X, y = load_iris(return_X_y=True)


# 2. X, y변수 전처리 
x_data = minmax_scale(X) # x변수 : 정규화 

# y변수 : one-hot 인코딩 
y_data = OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()


# 3. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # [관측치, 입력수] - [150, 4]
y = tf.constant(y_data, tf.float32) # [관측치, 출력수] - [150, 3]


# 4. w, b변수 정의 : 초기값(난수) -> update 
w = tf.Variable(tf.random.normal(shape=[4, 3])) # [입력수, 출력수]
b = tf.Variable(tf.random.normal(shape=[3])) # [출력수]


# 5. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 


# 6. softmax 함수  : 다항분류 활성함수 
def softmax_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.softmax(model) # softmax + model
    return y_pred 


# 7. 손실함수 : 정답(Y) vs 예측치(y_pred) -> 손실값 반환 
def loss_fn() : # 인수 없음 
    y_pred = softmax_fn(X)
    # cross entropy : loss value 
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss


# 8. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.1)


# 9. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
  
    
# 10. 최적화된 model 검증 
y_pred = softmax_fn(X).numpy() # 예측치 반환 

y_pred = tf.argmax(y_pred, axis = 1) # 10진수 
y_true = tf.argmax(y, axis = 1) # 10진수 

acc = accuracy_score(y_true, y_pred)
print('accuracy =',acc) 



