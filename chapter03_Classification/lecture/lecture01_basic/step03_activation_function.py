'''
활성함수(activation function)
 - model의 결과를 출력 y로 활성화 시키는 비선형 함수 
 - 유형 : sigmoid, softmax 
'''

import tensorflow as tf
import numpy as np

### 1. sigmoid function : 이항분류
def sigmoid_fn(x) : # x : 입력변수 
    ex = tf.math.exp(-x)   
    y = 1 / (1 + ex)
    return y # y : 출력변수(예측치)    


for x in np.arange(-5.0, 6.0) :
    y = sigmoid_fn(x)  
    print(f"x : {x} -> y : {y.numpy()}")
    
    
    
### 2. softmax function : 다항분류
def softmax_fn(x) :    
    ex = tf.math.exp(x - x.max())
    print(ex)
    y = ex / sum(ex.numpy())
    return y

x_data = np.array([1.0, 2.0, 5.0])

y_data = softmax_fn(x_data)
y_data.numpy()
# array([0.01714783, 0.04661262, 0.93623955]) 3번째가 확률 제일 높음

y_pred = tf.argmax(y_data)
y_pred.numpy() # 2
