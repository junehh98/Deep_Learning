# -*- coding: utf-8 -*-
"""
Cats vs Dogs image classifier 
 - image data generator 이용 : 학습 데이터셋 만들기 
"""
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution layer
from tensorflow.keras.layers import Dense, Flatten, Dropout # Affine layer
import os


# image resize
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) # (150, 150, 3)


# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 : kernel[3,3,3,32]
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer3 : maxpooling() 제외 
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten layer : 3d -> 1d
model.add(Flatten()) 
model.add(Dropout(0.5)) 

# DNN hidden layer(Fully connected layer)
model.add(Dense(256, activation = 'relu'))

# DNN Output layer
model.add(Dense(1, activation = 'sigmoid'))

# model training set  
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])





# 2. image file preprocessing : image 생성   
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dir setting
base_dir = "//Users/junehh98//Desktop//itwill//6_DeepLearning//data//cats_and_dogs"

train_dir = os.path.join(base_dir, 'train_dir') 
validation_dir = os.path.join(base_dir, 'validation_dir')


# 훈련셋 이미지 생성기 
train_data = ImageDataGenerator(rescale=1./255) # 정규화 

# 검증셋 이미지 생성기
validation_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150), # 규격화
        batch_size=20,  # 1회 모델 공급
        class_mode='binary') 
# Found 2000 images belonging to 2 classes.


validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')
# Found 1000 images belonging to 2 classes.


# 3. model training 
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=100,  
          epochs=10, 
          validation_data=validation_generator,
          validation_steps=50) 

# model evaluation
model.evaluate(validation_generator)


# 4. model history graph
import matplotlib.pyplot as plt
 
print(model_fit.history.keys())

loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']


## 3epoch 과적합 시작점 
epochs = range(1, len(acc) + 1) # range(1, 11)

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

