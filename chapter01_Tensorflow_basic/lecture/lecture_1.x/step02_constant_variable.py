# -*- coding: utf-8 -*-
"""
step02_variable.py

Tensorflow 상수와 변수 정의 
"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver1.x -> ver2.x 마이그레이션 
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' 프로그램 정의 영역 '''
# 상수 정의 : 수정불가   
x = tf.constant([1.5, 2.5, 3.5]) # 1차원 상수 
print(x)
# Tensor("Const_1:0", shape=(3,), dtype=float32)



# 변수 정의 : 수정가능 
y = tf.Variable([1.0, 2.0, 3.0]) # 1차원 변수 
print(y)
# <tf.Variable 'Variable_1:0' shape=(3,) dtype=float32_ref>


''' 프로그램 실행 영역 '''
with tf.Session() as sess : # 세션 객체 생성     
    print('x =', sess.run(x)) # 상수 실행 
    
    sess.run(tf.global_variables_initializer()) # 변수초기화
    print('y=', sess.run(y)) # 변수 실행  
    
'''
x = [1.5 2.5 3.5]
y= [1. 2. 3.]
'''

























