# -*- coding: utf-8 -*-
"""
문4) fashion_mnist 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
    
  조건1> keras layer
       L1 =  (28, 28) x 128
       L2 =  128 x 64
       L3 =  64 x 32
       L4 =  32 x 16
       L5 =  16 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'Adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 15, batch_size = 32   
  조건7> model evaluation : validation dataset
"""
from tensorflow.keras.utils import to_categorical # one hot
from tensorflow.keras.datasets import fashion_mnist # fashion
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense, Flatten # model layer
import matplotlib.pyplot as plt

# 1. MNIST dataset loading
(train_img, train_lab),(val_img, val_lab)=fashion_mnist.load_data() # (images, labels)
train_img.shape # (60000, 28, 28) 
train_lab.shape # (60000,) 
 


# 2. x, y변수 전처리 
# x변수 : 정규화(0~1)
train_img = train_img / 255.
val_img = val_img / 255.
train_img[0] # first image(0~1)
val_img[0] # first image(0~1)

# reshape(2d -> 1d)
x_train = train_img.reshape(-1, 784) # (60000, 28*28)
x_val = val_img.reshape(-1, 784) # (10000, 28*28)


# y변수 : one hot encoding 
train_lab = to_categorical(train_lab)
val_lab = to_categorical(val_lab)
val_lab.shape # (10000, 10)

# 입력 : 784개(28x28)
# 출력 : 10개 

# 훈련셋 : x_train, train_lab
# 검증셋 : x_val, val_lab

from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.models import Model # DNN Model 생성

# step04_keras_mnist_batch.py 참고 

# 3. keras model & layer 구축(Functional API 방식) 
input_dim = 784 # 입력차원 
output_dim = 10 # 출력차원 

# 1) input layer 
inputs = Input(shape=(input_dim,))   

# 2) hidden layer1 
hidden1 = Dense(units=128, activation='relu')(inputs) # 1층 

# 3) hidden layer2 
hidden2 = Dense(units=64, activation='relu')(hidden1) # 2층

# 3) hidden layer3 
hidden3 = Dense(units=32, activation='relu')(hidden2) # 3층 

# 4) hidden layer4  
hidden4 = Dense(units=16, activation='relu')(hidden3) # 4층 

# 5) output layer 
outputs = Dense(units=output_dim, activation = 'softmax')(hidden4) # 5층


## model 생성 : 구축된 layer을 바탕으로 model 생성 
model = Model(inputs, outputs) # 입력과 출력 연결 

# 4. model compile : 학습환경 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# 5. model training : 학습 
model.fit(x=x_train, y=train_lab, # 훈련셋 
          epochs=15, # 반복학습 횟수
          batch_size=32, # 1epoch=32*1875          
          verbose=1, # 출력여부 
          validation_data= (x_val, val_lab)) # 검증셋

# 6. model evaluation : validation dataset
print('model evaluation')
loss, acc = model.evaluate(x=x_val, y=val_lab)
print('loss =', loss)
print('accuracy =', acc)


