# -*- coding: utf-8 -*-
"""
미분계수(differential coefficient) 구하기 
"""

import numpy as np
import matplotlib.pyplot as plt 


### 1단계 ###
# f(x) 함수 : 2차함수 
def fx(x) :
    y = 0.01*x**2 + 0.1*x #
    return y

# 독립변수 x 
x = np.arange(0.0, 20.0, 0.1)
len(x) # 100

# 종속변수 y
y = fx(x) 




### 2단계 ###
# f'(x) 도함수  
def diff_coefficient(f, x) :
    h = 0.0001 
    y = (f(x+h) - f(x)) / (h) 
    return y  



### 3단계 ###
# 미분계수 반환 
x_data = np.array([5.0, 10.0, 15.0])
values = diff_coefficient(fx, x_data) # array([0.2, 0.3, 0.4])



### 4단계 ###
# 직선의 방정식에 미분계수 적용 
for x, p in zip(x_data, values) :
    y2 = fx(x) + p*x * 0.01  
    plt.plot(x, y2, 'ro') 
plt.ylabel("y2")
plt.xlabel("x")
plt.show





