# -*- coding: utf-8 -*-
"""
chapter02_2_LinearRegression 강의자료 : ppt.3 내용 

가중치(weight) : X 변수 중요도 조절변수
편향(bais) : 뉴런의 활성화 조절변수
"""

import numpy as np 


# 1. 신경망 조절변수(W, b)  
def init_variable() :
    variable = {} # dict
    variable['W'] = np.array([[0.1], [0.3]])   
    variable['b'] = 0.1       
    return variable
    


# 2. 활성함수 : 항등함수
def activation(model) :
    return model 
 

    
# 3. 순방향(forward) 실행 
def forward(variable, X) :   
    W = variable['W']   
    b = variable['b']
    
    model = np.dot(X, W) + b # 망의총합 
    y = activation(model) # 활성함수 
    return y



# 프로그램 시작점 
if __name__ == '__main__' :  
    pass










    
    
    






