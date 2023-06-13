# -*- coding: utf-8 -*-
"""
전이학습(transfer_learning)
 - ImageNet 분류기 -> 고양이와 강아지 분류(chapter06 > lecture02 > step02) 
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model 

### 단계1 : Feature 추출 
#  - 사전에 학습된 ImageNet 분류 모델에서 가중치 추출 

# 1. VGG16 기본 model 생성 
 
# 1) input image
input_shape = (150, 150, 3) 

# 2) 기본 모델을 이용하여 객체 생성     
base_model = VGG16(weights='imagenet', # 기존 학습된 가중치 이용 
                   input_shape=input_shape,  # input image 
                   include_top = False) # 최상위 레이어(output) 사용안함

# CNN layer 확인 
base_model.summary()

# 3) new model 생성 : 필요한 레이어 선택 
inputs = base_model.layers[0].input # 첫번째 레이어 입력정보  
outputs = base_model.layers[-1].output # 마지막 레이어 출력정보

new_model = Model(inputs, outputs)
 

# 4) new model 가중치 학습여부 지정 
new_model.trainable = False # 모든 레이어의 가중치 학습 동결  


### 단계2 : 전이학습 모델 생성 
# - 학습된 모델의 가중치 이용 -> 입력 이미지 분류 

from tensorflow.keras import Sequential # keras model 
#from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution
from tensorflow.keras.layers import Dense, Flatten # Dropout,  layer
import os


# 1. CNN Model layer 
print('model create')
model = Sequential()


# 2. 전이학습 : 학습된 모델 적용  
model.add(new_model)


''' [기존 CNN 레이터 제외] 
# Convolution layer1 : kernel[3,3,3,32]
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 : kernel[3,3,32,64]
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer3 : kernel[5,5,64,128], maxpooling() 제외 
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten layer :4d -> 2d
model.add(Flatten()) 
# 드롭아웃 - 과적합 해결 
model.add(Dropout(0.5)) # fully connected 층 이전에 배치 
'''

# Flatten layer :3d -> 1d
model.add(Flatten()) 

# Affine layer(Fully connected layer1) : [n, 256]
model.add(Dense(256, activation = 'relu'))

# Output layer(Fully connected layer2) : [256, 1]
model.add(Dense(1, activation = 'sigmoid'))

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', # one hot encoding
              metrics = ['accuracy'])

# 2. image file preprocessing : 이미지 제너레이터 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("image preprocessing")

# dir setting
base_dir = "C:\\ITWILL\\6_DeepLearning\\data\\images\\cats_and_dogs"

train_dir = os.path.join(base_dir, 'train_dir')
validation_dir = os.path.join(base_dir, 'validation_dir')


# image 증식 - 과적합 해결
train_data = ImageDataGenerator(
        rescale=1./255, # 정규화 
        rotation_range = 40, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.2, # image 수평 이동 범위
        height_shift_range = 0.2, # image 수직 이용 범위  
        shear_range = 0.2, # image 전단 각도 범위
        zoom_range=0.2, # image 확대 범위
        horizontal_flip=True,) # image 수평 뒤집기 범위 

# 검증 데이터에는 증식 적용 안함 
validation_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=35, #[수정] batch size 올림
        class_mode='binary') # binary label
# Found 2000 images belonging to 2 classes.

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=35, # [수정] batch size 올림 
        class_mode='binary')
# Found 1000 images belonging to 2 classes.

# 3. model training : 배치 제너레이터 이용 모델 훈련 
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=58, #  58*35 = 1epoch 
          epochs=30, # [수정] 30 epochs()
          validation_data=validation_generator,
          validation_steps=29) #  29*35 = 1epoch

# model evaluation
model.evaluate(validation_generator)

# 4. model history graph
import matplotlib.pyplot as plt
 
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']

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

