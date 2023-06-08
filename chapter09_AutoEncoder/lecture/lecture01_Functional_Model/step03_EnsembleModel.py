# -*- coding: utf-8 -*-
"""
step03_ensembleModel_lec.py

앙상블 모델(ensemble model) : 여러개의 하위 모델을 이용하여 상위모델 학습 방법   
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. dataset load & 전처리 
X, y = load_iris(return_X_y=True)

X.shape # (150, 4)
y.shape # (150,)


# X변수 : 정규화(0~1)
X = minmax_scale(X)

# X, y변수 선택  
X = X[:, 1:4] # 2~4번 칼럼 
y = X[:, 0] # 1번 칼럼 

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. Functional API model 구축
##################################################
### Functional API : Input과 Model 클래스 이용 
##################################################

from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.layers import Dense # DNN layer 구축
from tensorflow.keras.models import Model # DNN Model 생성 
from tensorflow.keras.layers import average # model 평균 

# 하위 model 반환 함수 
def get_model() :
    inputs = Input(shape=(3,)) # 입력층 : 독립변수 3개
    x = Dense(units=12, activation='relu')(inputs) # 1층 
    x = Dense(units=6, activation='relu')(x) # 2층 
    outputs = Dense(units=1)(x) # 3층, 활성함수 x -> 회귀모델의 결과값 반환 
    return Model(inputs, outputs)


# 하위 model 생성
model1 = get_model()
model2 = get_model()
model3 = get_model()


# 입력층 : 하위 model 이용, 모델 reuse
inputs = Input(shape=(3,)) # 입력층 
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)

# 출력층 : 각 하위 model 예측값 평균  
outputs = average([y1, y2, y3]) 

# 상위 model 생성 : 구축된 layer을 바탕으로 model 생성 
model = Model(inputs, outputs) 

model.summary()

# 4. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='sgd', 
              loss = 'mse', metrics=['mae'])


# 5. model training : train(70) vs val(30) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=200, # 반복학습
          verbose=1) # 출력여부 



# 6. model evaluation : val dataset 
print('\n')
print('model evaluation')
loss, mae = model.evaluate(x=x_val, y=y_val)
print('loss=', loss)
print('mae=',mae)

