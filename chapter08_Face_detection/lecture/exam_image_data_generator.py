# -*- coding: utf-8 -*-
"""
문) 다음과 같이 Celeb image의 분류기(classifier)를 생성하시오.  
   조건1> train image : train_celeb4
   조건2> validation image : val_celeb4
   조건3> image shape : 120 x 120
   조건4> Image Data Generator 이용 image 자료 생성 
   조건5> model layer 
         1. Convolution layer1 : kernel_size=(4, 4), 특징맵 16개
                                 pool_size=(2, 2)
         2. Convolution layer2 : kernel_size=(4, 4), 특징맵 32개
                                 pool_size=(2, 2)
         3. Flatten layer
         4. DNN hidden layer1 : 64 node
         5. DNN hidden layer2 : 32 node
         6. DNN output layer : 4 node
   조건6> 과적합 해결법 적용 
          Dropout과 EarlyStopping 이용 
"""
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D,Activation
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# images dir 
base_dir = r"C:\ITWILL\6_DeepLearning\workspace\chapter08_Face_detection"

# 훈련셋/검증셋 이미지 경로 
train_dir = os.path.join(base_dir, 'images_celeb4/train_celeb4')
val_dir = os.path.join(base_dir, 'images_celeb4/val_celeb4')

img_h = 120
img_w = 120
input_shape = (img_h, img_w, 3)



# 1. CNN Model layer 
model = Sequential()


model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(32,kernel_size=(4, 4), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


# Flatten layer : 3d -> 1d
model.add(Flatten())  
model.add(Dropout(rate=0.5)) 

# DNN hidden layer(Fully connected layer)
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

# DNN Output layer
model.add(Dense(4, activation = 'softmax'))  


model.summary()


model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])




# 2. image data 생성 : ImageDataGenerator 이용  
train_data = ImageDataGenerator(rescale=1./255) # 정규화 

validation_data = ImageDataGenerator(rescale=1./255) # 정규화 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(120,120), # image reshape
        batch_size=20, # batch size
        class_mode='categorical') 


validation_generator = validation_data.flow_from_directory(
        val_dir,
        target_size=(120,120),
        batch_size=20,
        class_mode='categorical')





# 3. model training : 이미지 제너레이터 객체 이용  
callback = EarlyStopping(monitor='val_loss', patience=5) 
# epoch=2 연속으로 검증 손실이 개선되지 않으면 조기종료 

model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=31, 
          epochs=20, 
          validation_data=validation_generator,
          validation_steps=10,
          callbacks = [callback])


model.evaluate(validation_generator)



# 4. model history graph

import matplotlib.pyplot as plt
 
loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']


epochs = range(1, len(acc) + 1) # epochs size 

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