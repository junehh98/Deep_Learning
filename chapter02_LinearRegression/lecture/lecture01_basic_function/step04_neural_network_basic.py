# -*- coding: utf-8 -*-
"""
chapter02_2_LinearRegression 강의자료 : ppt.3 내용 

가중치(weight) : X변수 중요도 조절변수(기울기)
편향(bias) : 뉴런의 활성화 조절변수(절편)

y = X*w + b
"""

import tensorflow as tf # tf.matmul(x, y)
import numpy as np # np.dot(x, y)


# 1. 신경망 조절변수(W, b)  
def init_variable() :
    variable = {} # dict
    variable['W'] = np.array([[0.1], [0.3]]) # 가중치(기울기)
    variable['b'] = 0.1 # 편향(절편)      
    return variable
    


# 2. 활성함수 : 항등함수(회귀모델)
def activation(model) :
    return model 
 

    
# 3. 순방향(forward) 실행 
def forward(variable, X) :  #(조절변수(w,b), 입력) 
    W = variable['W']   
    b = variable['b']
    
    model = tf.linalg.matmul(X, W) + b # 망의총합 
    y = activation(model) # 활성함수 
    return y



# 프로그램 시작점 
if __name__ == '__main__' :  
    X = np.array([[1.0, 0.5]]) # X(1, 2) @ W(2, 1) = y[1,1]
    variable = init_variable() # W, b
    y = forward(variable, X)
    print('X =',X)
    print('W =', variable['W'])
    print('b =', variable['b'])
    print('y=', y.numpy()) 
 
'''
X = [[1.  0.5]]
W = [[0.1]
 [0.3]]
b = 0.1
y= [[0.35]]
'''


    

    
    










    
    
    






