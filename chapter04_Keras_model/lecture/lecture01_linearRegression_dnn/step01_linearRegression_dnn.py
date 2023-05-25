# -*- coding: utf-8 -*-
"""
step01_linearRegression_dnn

Keras : High Level API  
"""

# dataset 
from sklearn.datasets import load_iris # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# keras model 
import tensorflow as tf
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense # DNN layer 
import numpy as np 
import random 


## karas 내부 weight seed 적용 
tf.random.set_seed(123) # global seed 
np.random.seed(123) # numpy seed
random.seed(123) # random seed 


# 1. dataset laod 
X, y = load_iris(return_X_y=True)
X.shape # (150, 4)
y.shape # (150,)


# 2. 공급 data 생성 : 훈련셋, 검증셋 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)



# 3. DNN model 생성 
model = Sequential() 
dir(model) # model.add(units=노드수, activation='활성함수')



# 4. DNN model layer 구축 
# input_shape =(입력수,)

# hidden layer1 : w[4,12], b[12] 자동으로 가중치, 편향 생성
model.add(Dense(units=12, input_shape=(4,), activation='relu'))# 1층 

# hidden layer2 : w[12,6], b[6]
model.add(Dense(units=6, activation='relu'))# 2층

# output layer : w[6,1], b[1]
model.add(Dense(units=1))# 3층 



# 5. model compile : 학습과정 설정 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])



# 6. model training 
model.fit(x=x_train, y=y_train,  
          epochs=100,  # 반복학습횟수
          verbose=1,   # 콘솔 출력
          validation_data=(x_val, y_val))   
'''
 훈련셋 : loss: 0.0586 - mae: 0.1818 -검증셋 : val_loss: 0.0596 - val_mae: 0.1663
'''


# 7. model evaluate
score = model.evaluate(x_val, y_val, verbose=1) # MAE값 출력
print('val_loss :',score)
''' 
              training set         validation set          
val_loss : [0.05959518626332283, 0.16632400453090668]
'''

# mean_squared_error, r2_score
dir(model)

y_pred = model.predict(x_val)
y_true = y_val

r2 = r2_score(y_true, y_pred)
print(r2) # 0.9233289331093542


























