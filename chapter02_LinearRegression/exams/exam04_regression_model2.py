'''
문4) load_boston 데이터셋을 이용하여 다음과 같이 선형회귀모델 생성하시오.
     <조건1> x변수 : boston.data,  y변수 : boston.target
     <조건2> w변수, b변수 정의 : tf.random.normal() 이용 
     <조건3> learning_rate=0.5
     <조건4> 최적화함수 : Adam
     <조건5> 학습 횟수 1,000회
     <조건6> 학습과정과 MSE 출력 : <출력결과> 참고 
     
 <출력결과>
step = 100 , loss = 4.646641273041386
step = 200 , loss = 1.1614418341428459
step = 300 , loss = 0.40125618834653615
step = 400 , loss = 0.21101471610402903
step = 500 , loss = 0.13666187210671069
step = 600 , loss = 0.09779346604325287
step = 700 , loss = 0.07608768653282329
step = 800 , loss = 0.06372023833861612
step = 900 , loss = 0.0566559217407318
step = 1000 , loss = 0.05266675679250506
=============================================
MSE= 0.04122129293175945
'''

import tensorflow as tf # ver2.x
from sklearn.model_selection import train_test_split # datast splits
from sklearn.metrics import mean_squared_error # model 평가 
from sklearn.datasets import load_boston
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 

# 1. data loading
boston = load_boston()

# 변수 선택 
X = boston.data # x 
y = boston.target # y : 숫자 class(0~2)

print(X.shape) # (506, 13)
print(y.shape) # (506,)

# y변수 정규화 
y = minmax_scale(y)

# 2. train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)

# 3. w, b변수 정의 : tf.random.normal() 함수 이용 

# 4. 회귀모델 : 행렬곱 

# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 

# 6. 최적화 객체 

# 7. 반복학습 

# 8. 최적화된 model 평가





