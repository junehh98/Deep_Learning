# -*- coding: utf-8 -*-
"""
문3) 다음 digits 데이터셋을 이용하여 다항분류기를 작성하시오.
    <조건1> digits 데이터셋의 특성을 보고 전처리/공급data 생성  
    <조건2> 학습율과 반복학습 : <출력결과> 참고
    <조건3> epoch에 따른 loss 시각화 : Figure_exam03.png 파일 참고
   
 <출력결과>
step = 100 , loss = 0.20174500116134184
step = 200 , loss = 0.07932223449884587
step = 300 , loss = 0.048710212998260755
step = 400 , loss = 0.035595351846036176
step = 500 , loss = 0.028107977674384905
step = 600 , loss = 0.023223684237817892
step = 700 , loss = 0.01979145869469669
step = 800 , loss = 0.01725034877238523
step = 900 , loss = 0.01528246675917213
step = 1000 , loss = 0.013700554272071161
========================================
accuracy = 0.9537037037037037
"""

import tensorflow as tf # ver 2.0
from sklearn.preprocessing import minmax_scale, OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
 
'''
digits 데이터셋 : 숫자 필기체 이미지 -> 숫자 예측(0~9)

•타겟 변수 : y
 - 0 ~ 9 : 10진수 정수 
•특징 변수(64픽셀) : X 
 -0부터 9까지의 숫자를 손으로 쓴 이미지 데이터
'''

# 1. data load 
digits = load_digits() # dataset load

X = digits.data  # X변수 
y = digits.target # y변수 
print(X.shape) # (1797, 64) : 64=8x8
print(y.shape) # (1797,)


# 2. X, y변수 전처리 
X = None # X변수 정규화
y = None # one-hot encoding


# 3. w, b 변수 정의 : 초기값 난수 
w = None
b = None
 

# 4. digits dataset split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 5. 회귀방정식 
def linear_model(X) :
    y_pred = tf.matmul(X, w) + b  
    return y_pred

# 6. softmax 활성함수 적용 
def soft_fn(X):
    y_pred = linear_model(X)
    soft = tf.nn.softmax(y_pred)
    return soft

# 7. 손실 함수 정의 : 손실계산식 수정 
def loss_fn() : #  인수 없음 
    soft = soft_fn(x_train) # 훈련셋 -> 예측치 : 회귀방정식  
    loss = -tf.reduce_mean(y_train*tf.math.log(soft)+(1-y_train)*tf.math.log(1-soft))
    return loss


# 8. 최적화 객체 



# 9. 반복학습 
    
    
# 10. 최적화된 model 검증 


# 11. loss value vs epochs 시각화 






