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
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 

# keras model 관련 API
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer


# 1. x,y data 생성 : keras datasests 이용 
(x_train, y_train), (x_val, y_val) = boston_housing.load_data()
x_train.shape # (404, 13)
x_val.shape # (102, 13)

# 2. X, y변수 정규화 
x_train = minmax_scale(x_train)
x_val = minmax_scale(x_val)

y_train = y_train / y_train.max()
y_val = y_val / y_val.max()

# 3. keras model
model = Sequential() 
print(model) # object info


# 4. DNN model layer 구축 
# hidden layer1 : w[13,64], b[64] 
model.add(Dense(units=64, input_shape=(13,), activation='relu'))# 1층 

# hidden layer2 : w[64,32], b[32]
model.add(Dense(units=32, activation='relu'))# 2층

# hidden layer2 : w[32,16], b[16]
model.add(Dense(units=16, activation='relu'))# 3층

# output layer : w[16,1], b[1] 
model.add(Dense(units=1))# 4층 


# 5. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer = 'adam', 
         loss = 'mse', 
         metrics = ['mae'])


# model layer 확인 
model.summary()


# 6. model training 
model.fit(x=x_train, y=y_train,  # 훈련셋 : 70%
          epochs=100,  # 반복학습횟수  
          verbose=1,  # 콘솔 출력 
          validation_data=(x_val, y_val))  # 검증셋 : 30% 


# 7. 모델 평가
score = model.evaluate(x_val, y_val, verbose=0)
print('val_loss:', score[0])
print('val_mae:', score[1])

