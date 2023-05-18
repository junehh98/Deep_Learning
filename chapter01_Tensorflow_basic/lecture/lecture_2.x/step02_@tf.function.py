# -*- coding: utf-8 -*-
"""
step02_@tf.function.py

@tf.function 함수 장식자
 - 함수내에서 python code 작성 지원 
 - tensorflow if, while문 python code 대체 
"""

import tensorflow as tf 

# if 처리 
@tf.function # 함수장식자  py code -> tf code
def if_fn(x) :   
    if x > 100 : # python code        
        y = tf.math.multiply(x, 10) # tensorflow code
    else :
        y = tf.math.add(x, 10) # tensorflow code
    return y

if_fn(5) # <tf.Tensor: shape=(), dtype=int32, numpy=15>


# while 처리 
@tf.function 
def while_fn(i) :    
    # python code
    while i < 100 : 
        i += 1
    return i 

print('i=', while_fn(0))
# i= tf.Tensor(100, shape=(), dtype=int32)















