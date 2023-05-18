# -*- coding: utf-8 -*-
"""
step01_tensorflow_basic.py

- tensorflow ver1.x 작업 환경
- 프로그램 정의 영역과 실행 영역 
"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver2.x 환경에서 ver1.x 사용
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' 프로그램 정의 영역  : 모델 구성 '''

# 상수 정의 
x = tf.constant(10)  
y = tf.constant(20)  

# 식 정의 
z = tf.add(x, y) # z = x + y
print('z=', z)
# 텐서의 정보 z= Tensor("Add_7:0", shape=(), dtype=int32)


''' 프로그램 실행 영역 : 모델 실행 '''
sess = tf.Session() # 세션 생성 

# device에 할당 
print('x=', sess.run(x)) 
print('y=', sess.run(y))
print('z=', sess.run(z)) 

sess.close() # 세션 닫기 







