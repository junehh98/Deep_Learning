# -*- coding: utf-8 -*-
"""
 1. celeb5 이미지 분류기 : CNN 
 2. Image Generator : Model 공급할 이미지 생성
"""

from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution layer
from tensorflow.keras.layers import Dense, Flatten # DNN layer
import os

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)

# image resize
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) # input image 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 : [5,5,3,32]
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 : [3,3,32,64]
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


# Flatten layer : 3d -> 1d
model.add(Flatten()) # 전결합층 

# DNN hidden layer(Fully connected layer)
model.add(Dense(64, activation = 'relu'))

# DNN Output layer
model.add(Dense(5, activation = 'softmax')) # 5 classes 

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',  
              metrics = ['accuracy'])


# 2. image file preprocessing : ImageDataGenerator 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 경로 
base_dir = r"C:\ITWILL\6_DeepLearning\workspace\chapter08_Face_detection"

train_dir = os.path.join(base_dir, 'images_celeb5/train_celeb5') 
validation_dir = os.path.join(base_dir, 'images_celeb5/val_celeb5')


# 특정 폴더의 이미지 분류를 위한 학습 데이터셋 생성기
train_data = ImageDataGenerator(rescale=1./255) # 정규화 

# 특정 폴더의 이미지 분류를 위한 검증 데이터셋 생성기
validation_data = ImageDataGenerator(rescale=1./255) # 정규화 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150), # image reshape
        batch_size=20, # batch size
        class_mode='categorical') # categorical label

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='categorical')# categorical label


# 3. model training : ImageDataGenerator 이용 모델 훈련 
model_fit = model.fit_generator(
          generator = train_generator, 
          steps_per_epoch=50, # batch_size*50=1,000
          epochs=10, 
          validation_data=validation_generator,
          validation_steps=13) # 20*13

# model evaluation
print('model evaluation')
model.evaluate(validation_generator)


# 4. model history graph
import matplotlib.pyplot as plt
import os # os 환경설정 : 시각화 오류 해결 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
print(model_fit.history.keys())

loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']


## 과적합 시작점 확인  
epochs = range(1, len(acc) + 1)

# acc vs val_acc   
plt.plot(epochs, acc, 'b--', label='train acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('Training vs validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='best')
plt.show()

# loss vs val_loss 
plt.plot(epochs, loss, 'b--', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training vs validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
