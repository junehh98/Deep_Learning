# -*- coding: utf-8 -*-
"""
문5) fashion_mnist 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
    
  조건1> DNN layer(Functional API 방식)
       input_dim = (28, 28)
       output_dim = 10
       
       Flatten(input_shape = input_dim)       
       hidden1 = units=128, input_shape=(784,), activation='relu'
       Dropout(rate=0.3)
       hidden2 = units=64, activation='relu'
       Dropout(rate=0.1)
       hidden3 = units=32, activation='relu'
       Dropout(rate=0.1)
       hidden4 = units=16, activation='relu'
       output = units=output_dim, activation='softmax'       
  조건2> 학습환경 
         optimizer = 'Adam',
         loss = 'categorical_crossentropy'
         metrics = 'accuracy'
  조건3> 모델 학습 
         epochs = 30, batch_size = 32
         조기종료 : monitor='val_loss',patience=3
  조건4> history 기능으로 loss와 accuracy 시각화   
"""
from tensorflow.keras.utils import to_categorical # one hot
from tensorflow.keras.datasets import fashion_mnist # fashion
#from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense, Flatten, Dropout # model layer
from tensorflow.keras.callbacks import EarlyStopping # 조기종료 
from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.models import Model # DNN Model 생성

import matplotlib.pyplot as plt

# 1. MNIST dataset loading
(x_train, y_train),(x_val, y_val)=fashion_mnist.load_data() # (images, labels)
x_train.shape # (60000, 28, 28) 
y_train.shape # (60000,) 
x_val.shape # (10000, 28, 28) 
y_val.shape # (10000,)

x_train.max() # 255

# 2. x, y변수 전처리 
# x변수 : 정규화(0~1)
x_train = x_train / 255
x_val = x_val / 255


# y변수 : one hot encoding 
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_train[0] # [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]


# 3. layer 구축 & model 생성(Functional API 방식)  
input_dim = (28,28) # 입력차원 
output_dim = 10 # 출력차원 

# 1) input layer 
inputs = Input(shape = input_dim) # 입력층 
flatten = Flatten(input_shape = input_dim)(inputs)# 2d -> 1d

# 2) hidden layer1 
hidden1 = Dense(units=128, activation='relu')(flatten) # 1층 
dropout1 = Dropout(rate=0.3)(hidden1)

# 3) hidden layer2 
hidden2 = Dense(units=64, activation='relu')(dropout1) # 2층
dropout2 = Dropout(rate=0.1)(hidden2)

# 3) hidden layer3 
hidden3 = Dense(units=32, activation='relu')(dropout2) # 3층 
dropout3 = Dropout(rate=0.1)(hidden3)

# 4) hidden layer4  
hidden4 = Dense(units=16, activation='relu')(dropout3) # 4층 

# 5) output layer
outputs = Dense(units=output_dim, activation='softmax')(hidden4) # 5층 

# model 생성 
model = Model(inputs, outputs)

# model layer 확인 
model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 28, 28)] : 2d         0         
                                                                 
 flatten (Flatten)           (None, 784)   : 2d -> 1d      0         
                                                                 
 dense_40 (Dense)            (None, 128)               100480=784*128+128    
                                                                 
 dense_41 (Dense)            (None, 64)                8256=128*64+64      
                                                                 
 dense_42 (Dense)            (None, 32)                2080=64*32+32       
                                                                 
 dense_43 (Dense)            (None, 16)                528=32*16+16       
                                                                 
 dense_44 (Dense)            (None, 10)                170=16*10+10       
                                                                 
=================================================================
Total params: 111,514
'''

# 4. model compile : 학습환경 
model.compile(optimizer = 'Adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# 5. model training : 모델학습 
callback = EarlyStopping(monitor='val_loss',
                         patience=3)

model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
              epochs=30, # 반복학습 횟수
              batch_size=32, # 1epoch=32*1875          
              verbose=1, # 출력여부 
              validation_data= (x_val, y_val),# 검증셋
              callbacks = [callback]) # 조기종료

# 6. model history 시각화
# 1) loss vs val_loss 
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# 2) accuracy vs val_accuracy 
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

'''
Dropout 적용 전 조기종료 : Epoch 12
Epoch 12/30
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2361 - accuracy: 0.9111 - val_loss: 0.3519 - val_accuracy: 0.8773

Dropout 적용 후 조기종료 : Epoch 20
Epoch 20/30
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3073 - accuracy: 0.8873 - val_loss: 0.3410 - val_accuracy: 0.8794
''' 

