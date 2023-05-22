'''
문) 다음과 같이 X, a 행렬을 상수로 정의하고 행렬곱으로 연산하시오.
    단계1 : X, a 행렬 
        X 행렬 : iris 2번~4번 칼럼으로 상수 정의 
        a 행렬 : [[0.2],[0.1],[0.3]] 값으로 상수 정의 
    단계2 : 행렬곱 : y 계산하기  
        y = X @ a
    단계3 : y 결과 출력
'''

import tensorflow as tf # 상수, 행렬곱 
import pandas as pd # csv file 


iris = pd.read_csv(r'C:\ITWILL\6_DeepLearning\data\iris.csv')
iris.shape # (150, 5)

#  단계1 : X, a 상수 정의 
X = tf.constant(value=iris.iloc[:, 1:-1], dtype="float32") # iris에서 2~4번
X.shape # [150, 3] 
X
a = tf.constant(value=[[0.2],[0.1],[0.3]]) # 기울기(가중치)
a.shape # [3, 1]

X.dtype # tf.float64 -> tf.float32
a.dtype # tf.float32

# 단계2 : 행렬곱 식 정의 
y = tf.linalg.matmul(X, a) # y = X1*a1 + X2*a2 + X3*a3  
'''
주의 : 자료형의 불일치 
InvalidArgumentError: cannot compute MatMul as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:MatMul]
'''

# 단계3 : 행렬곱 결과 출력 
print('y=', y.numpy())
y.shape # [150, 1]







