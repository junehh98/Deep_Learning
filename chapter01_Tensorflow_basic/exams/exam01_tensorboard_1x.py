'''
문1) 다음과 같은 상수와 사칙연산 함수를 이용하여 dataflow의 graph를 작성하여 
    tensorboard로 출력하시오.
    조건1> 상수와 식에 name 지정 
    조건2> 상수 정의 : x = 100, y = 50
    조건3> 식 정의 : result = ((x - 5) * y) / (y + 20)
        -> 사칙연산 함수 이용 : 식 작성  순서 
        1. sub = tf.subtract(x, 5)
        2. mul = tf.multiply(sub, y)
        3. add = tf.add(y, 20)
        4. result = tf.div(mul, add)
'''
import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

# tensorboard 초기화 
tf.reset_default_graph()


# 상수 정의 
x = tf.constant(100, name = 'x')
y = tf.constant(50, name = 'y')


# 식 정의 : result = ((x - 5) * y) / (y + 20)


# session 생성 : result 출력 & graph 모으기 & 로그파일 생성 
with tf.Session() as sess :
    pass




