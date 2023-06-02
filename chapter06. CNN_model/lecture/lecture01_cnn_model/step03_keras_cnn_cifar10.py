# -*- coding: utf-8 -*-
"""
step03_keras_cnn_cifar10.py

CNN model 생성 
 1. image dataset load 
 2. image dataset 전처리 
 3. CNN model 생성 : layer 구축 + 학습환경 + 학습 
 4. CNN model 평가
 5. CMM model history 
"""

from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten # DNN layer 
import matplotlib.pyplot as plt 

# 1. image dataset load 
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # image : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # label : (50000, 1)


x_val.shape # image : (10000, 32, 32, 3)
y_val.shape # label : (10000, 1)



# 2. image dataset 전처리

# 1) image pixel 실수형 변환 
x_train = x_train.astype(dtype ='float32')  
x_val = x_val.astype(dtype ='float32')

# 2) image 정규화 : 0~1
x_train = x_train / 255
x_val = x_val / 255

'''
for i in x_train:
    plt.imshow(i)
    plt.show()
'''

# 3) label 전처리 : 10진수 -> one hot encoding(2진수) 
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)



# 3. CNN model & layer 구축 
input_shape = (32, 32, 3) # input images 

# 1) model 생성 
model = Sequential()


# 2) layer 구축 
# Conv layer1 : 기본값(strides=[1,1,1,1], padding='VALID')
model.add(Conv2D(filters=32, kernel_size=(5, 5),  
                 input_shape = input_shape, activation='relu')) 
'''
filter : kernel_size(5x5)

'''

model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2))) 
'''
window size = pool_size
strides : window 이동 칸
'''

# Conv layer2 
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu')) 
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2))) 

# Conv layer3  
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# MaxPool 없음 

# 전결합층 : Flatten layer : 3/2d -> 1d
model.add(Flatten()) 

# DNN1 : hidden layer 
model.add(Dense(units=64, activation='relu'))

# DNN2 : output layer  
model.add(Dense(units = 10, activation='softmax')) 
                  

model.summary()


# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(105) vs val(45) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=10, # 반복학습 
          batch_size = 100, # 1회 공급 image size 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. CNN model 평가 : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

 
# 7. CMM model history 
print(model_fit.history.keys()) # key 확인 
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# loss vs val_loss 
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy 
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()












