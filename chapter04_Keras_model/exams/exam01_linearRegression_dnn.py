# -*- coding: utf-8 -*-
"""
문1) boston 데이터셋을 이용하여 다음과 같이 Keras DNN model layer을 
    구축하고, model을 학습하고, 검증(evaluation)하시오. 
    <조건1> 4. DNN model layer 구축 
         1층(hidden layer1) : units = 64
         2층(hidden layer2) : units = 32
         3층(hidden layer3) : units = 16 
         4층(output layer) : units=1
    <조건2> 6. model training  : 훈련용 데이터셋 이용 
            epochs = 50
    <조건3> 7. model evaluation : 검증용 데이터셋 이용     
"""
from sklearn.datasets import load_boston  # dataset
from sklearn.model_selection import train_test_split # split
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
                                              
# keras model 관련 API
import tensorflow as tf # ver 2.x
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer


# 1. x,y data 생성 
X, y = boston = load_boston(return_X_y=True)
X.shape # (442, 10)
y.shape # (442,)

# y 정규화 
X = minmax_scale(X)
y = minmax_scale(y)

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size = 0.3, random_state=1)
x_train.shape 
y_train.shape 


# 3. keras model
model = Sequential() 
print(model) # object info


# 4. DNN model layer 구축 


# 5. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer = 'adam', 
         loss = 'mse', 
         metrics = ['mae'])


# model layer 확인 
model.summary()


# 6. model training 



# 7. model evaluation : test dataset
