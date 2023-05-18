# -*- coding: utf-8 -*-
"""
step01_constant_variable.py

1. Tensorflow 상수와 변수 
2. 즉시 실행(eager execution) 모드
 - session 사용 없이 자동으로 컴파일 
 - python 처럼 즉시 실행하는 모드 제공(python 코드 사용 권장)
 - API 정리 : tf.global_variables_initializer() 삭제됨 
"""


import tensorflow as tf # ver 2.x
print(tf.__version__) 


# 즉시 실행 모드 
tf.executing_eagerly() # default 활성화 


# 상수 정의 : 수정불가
x = tf.constant(value = [1.5, 2.5, 3.5]) # 1차원   
print('x =', x) 
# x = tf.Tensor([1.5 2.5 3.5], shape=(3,), dtype=float32) 텐서값도 같이 출력
print(x.numpy()) # [1.5 2.5 3.5]


# 변수 정의 : 수정가능  
y = tf.Variable([1.0, 2.0, 3.0]) # 1차원  
print('y =', y)
# y = <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>
print(y.numpy()) # [1. 2. 3.]


dir(y)
y.assign(value=[1.5, 2.5, 3.5])


# 식 정의 : 상수 or 변수 참조 
mul = tf.math.multiply(x, y) # x * y 
print('mul =', mul) 
# mul = tf.Tensor([ 2.25  6.25 12.25], shape=(3,), dtype=float32)

dir(tf.math)
'''
 tf.math.add()
 tf.math.subtract() - 뺄셈
 tf.math.divide() - 나누기
 tf.math.exp() - 지수함수
 tf.math.log() - 자연 로그
 tf.math.sqrt() - 루트
 tf.math.squqre() - 제곱근
'''