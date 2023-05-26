'''
문2) breast_cancer 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  30 x 64 : hidden1 layer
       L2 =  64 x 32 : hidden2 layer
       L3 =  32 x 2  : output layer 
  조건2> optimizer = 'adam',
  조건3> loss = 'binary_crossentropy'
  조건4> metrics = 'accuracy'
  조건5> epochs = 300 
'''

from sklearn.datasets import load_breast_cancer # data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale # 정규화 
from tensorflow.keras.utils import to_categorical # one hot encoding

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. breast_cancer data load
cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target
print(x_data.shape) # (569, 30) : matrix
print(y_data.shape) # (569,) : vector

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
y_one_hot = to_categorical(y_data)
y_one_hot.shape # (569, 2)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


#from tensorflow.keras.layers import Input # input layer
#from tensorflow.keras.models import Model # DNN Model 생성 
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # layer 

# 3. keras model & layer 구축(Functional API 방식) 
model = Sequential() # keras model 생성 

# hidden layer1 : w[30, 64], b=64 
model.add(Dense(units=64, input_shape =(30,), activation = 'relu')) # 1층 

# hidden layer2 : w[64, 32], b=32 
model.add(Dense(units=32, activation = 'relu')) # 2층 

# output layer : w[32, 2], b=2 
model.add(Dense(units=2, activation = 'sigmoid')) # 3층 

# 네트워크 확인 
model.summary() 
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param=W(IN*OUT)+B    
=================================================================
 dense_9 (Dense)             (None, 64)                1984=30*64+64      
                                                                 
 dense_10 (Dense)            (None, 32)                2080=64*32+32       
                                                                 
 dense_11 (Dense)            (None, 2)                 66=32*2+2        
                                                                 
=================================================================
Total params: 4,130
Trainable params: 4,130
'''
# 4. model compile : 학습환경 
model.compile(optimizer='adam', # 최적화 알고리즘(adam or sgd) 
              loss = 'binary_crossentropy', # 손실함수(crossentropy) 
              metrics=['accuracy']) # 평가방법 

# 5. model training : training dataset
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=80,  # 반복학습 횟수 
          verbose=1,  # 출력 여부 
          validation_data=(x_val, y_val))  # 검증셋 


# 6. model evaluation : validation dataset
print('='*30)
print('model evaluation')
loss, acc = model.evaluate(x=x_val, y=y_val)
print('loss :', loss)
print('acc :', acc)



# 7. model history 확인 : 반복학습 횟수 결정 
model_fit.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# 1) epoch vs loss
import matplotlib.pyplot as plt
plt.plot(model_fit.history['loss'], 'y', label='train loss value')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss value')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()

# 2) epoch vs accuracy
plt.plot(model_fit.history['accuracy'], 'y', label='train accuracy')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()








