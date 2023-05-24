# -*- coding: utf-8 -*-
"""
이항분류기 : 실제 데이터(iris) 적용 
"""

import tensorflow as tf
from sklearn.datasets import load_iris # dataset
from sklearn.preprocessing import minmax_scale # x변수 정규화 
from sklearn.preprocessing import OneHotEncoder # y변수 전처리
from sklearn.metrics import accuracy_score #  model 평가 


# 1. data load  
X, y = load_iris(return_X_y=True)

# x변수 선택 
X = X[:100]

# y변수 선택 
y = y[:100]  


# 2.  X, y 전처리 
x_data = minmax_scale(X) # x변수 정규화 : 0~1

# y변수 : one-hot 인코딩 
y_data = OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()


# 3. X, y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) 
y = tf.constant(y_data, tf.float32) 
X.shape # [100, 4]
y.shape # [100, 2]


# 4. w, b변수 정의 : 초기값(난수) 
w = tf.Variable(tf.random.normal(shape=[4, 2])) #[입력수, 출력수]
b = tf.Variable(tf.random.normal(shape=[2])) # 출력이 2개이기 때문


# 5. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b 
    return model 
    

# 6. sigmoid 함수   
def sigmoid_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.sigmoid(model) 
    return y_pred 
    

# 7. 손실함수  
def loss_fn() : # 인수 없음 
    y_pred = sigmoid_fn(X)
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
    


## 10. 최적화된 model 검증 
y_pred = tf.cast(sigmoid_fn(X).numpy() > 0.5, dtype=tf.float32)


# 분류정확도 
acc = accuracy_score(y, y_pred)
print('accuracy =',acc)
