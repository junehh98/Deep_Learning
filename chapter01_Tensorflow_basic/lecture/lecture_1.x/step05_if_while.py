# -*- coding: utf-8 -*-
"""
step05_if_while.py

- tensorflow logic : if, while문 사용 
"""

import tensorflow.compat.v1 as tf # ver1.x 사용 
tf.disable_v2_behavior() # ver2.0 사용안함

# if
x = tf.constant(10) # x의 초기값 

def true_fn() :
    return tf.multiply(x , 10) # 침 : x * 10

def false_fn():
    return tf.add(x, 10) # 거짓 : x + 10
    
y = tf.cond(x > 100, true_fn, false_fn) # (조건식, True_fn, False_fn)

'''
if x > 100 :
    x*=10
else :
    x+=10
     
'''


# while
i = tf.constant(0) # i = 0 : 반복변수 

def cond(i) : # i = 반복변수 
    return tf.less(i, 100) # i < 100

def body(i) : # i = 반복변수 
    return tf.add(i, 1) # i += 1

loop = tf.while_loop(cond, body, (i,)) # (반복조건식, 반복함수, 반복변수)

'''
i = 0
while i < 100
    i += 1
'''

with tf.Session() as sess :
    print("if =", sess.run(y)) 
    print("loop =", sess.run(loop)) 
    
    
    
    
    
    
    
    
    
    
    



