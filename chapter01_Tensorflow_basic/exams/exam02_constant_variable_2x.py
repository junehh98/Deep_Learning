'''
문2) 다음 조건과 <출력 결과>를 참고하여 a와 b변수를 정의하고, 브로드캐스팅 연산을 
    수행한 결과를 출력하시오.
    <조건1> a변수 : constant() 이용 
    <조건2> b변수 : Variable() 이용
    <조건3> c변수 계산식 : c = a * b -> multiply()이용 
    <조건4> a, b, c변수 출력 : <출력 결과> 참고      

< 출력 결과 > 
a= [1. 2. 3.]
b= [[0.123]
 [0.234]
 [0.345]]
c= [[0.123      0.246      0.36900002]
 [0.234      0.468      0.702     ]
 [0.345      0.69       1.035     ]]
c 자료모양 : (3, 3)
'''

import tensorflow as tf # 2.x

a = tf.constant([1.0, 2.0, 3.0])
print(a.numpy())

b = tf.Variable([[0.123], [0.234], [0.345]])
print(b.numpy())

c = tf.math.multiply(a, b)
print(c.numpy())

