# -*- coding: utf-8 -*-
"""
step02_variable.py

Tensorflow 상수와 변수 정의 
"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver1.x -> ver2.x 마이그레이션 
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' 프로그램 정의 영역 '''
# 상수 정의  
x = tf.constant([1.5, 2.5, 3.5]) # 1차원 상수 

# 변수 정의  
y = tf.Variable([1.0, 2.0, 3.0]) # 1차원 변수 


''' 프로그램 실행 영역 '''
with tf.Session() as sess : # 세션 객체 생성     
    print('x =', sess.run(x)) # 상수 실행 
    
    #sess.run(tf.global_variables_initializer()) 
    print('y=', sess.run(y)) # 변수 실행  
    
    



