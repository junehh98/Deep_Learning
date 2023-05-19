'''
문) 다음과 같이 X, a 행렬을 상수로 정의하고 행렬곱으로 연산하시오.
    단계1 : X, a 행렬 
        X 행렬 : iris 2번~4번 칼럼으로 상수 정의 
        a 행렬 : [[0.2],[0.1],[0.3]] 값으로 상수 정의 
    단계2 : 행렬곱 : y 계산하기  
        y = X @ a
    단계3 : y 결과 출력
'''

import tensorflow as tf
import pandas as pd 


iris = pd.read_csv(r'C:\ITWILL\6_DeepLearning\data\iris.csv')

#  단계1 : X, a 상수 정의 


# 단계2 : 행렬곱 식 정의 


# 단계3 : 행렬곱 결과 출력 
