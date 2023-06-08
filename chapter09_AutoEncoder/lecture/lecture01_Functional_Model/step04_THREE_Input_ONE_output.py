# -*- coding: utf-8 -*-
"""
step04_THREE_Input_ONE_output.py

사례1 : 이미지 데이터 2개를 입력하여 정답(label) 예측 모델 
 input image1(28x28)
                       ->   DNN  ->  output 1개 
 input image2(28x28)   

 사례2 : 이미지 데이터 2개 + 일반 데이터 1개 -> 정답(label) 예측 모델 
 input image1(28x28)
 input image2(28x28)      ->   DNN  ->  output 1개 
 input data     
"""
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,Dense,Dropout,Flatten
from tensorflow.keras.layers import Concatenate # layer 묶음 
from tensorflow.keras import Model


### 1. input data 생성 

# 1) minst data x, y변수 
from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
(x_train, y_train), (x_val, y_val) = load_data()
# 3d reshape 
x_train_img1 = x_train.reshape(-1, 28, 28, 1)
x_val_img1 = x_val.reshape(-1, 28, 28, 1) # [오류 수정]
x_train_img1.shape # (60000, 28, 28, 1)


# 2) fashion minst data x, y변수 
from tensorflow.keras.datasets.fashion_mnist import load_data
(x_train, y_train), (x_val, y_val) = load_data()
# 3d reshape 
x_train_img2 = x_train.reshape(-1, 28, 28, 1)
x_val_img2 = x_val.reshape(-1, 28, 28, 1)
x_train_img2.shape # (60000, 28, 28, 1)

# 3) input data 
y_train.shape # (60000,)
x_train_data = y_train
y_val.shape # (10000,)
y_val_data = y_val 


# 2. Functional API model 구축
##################################################
### Functional API : Input과 Model 클래스 이용 
##################################################

# 첫번째 이미지 입력 : Conv2개 + DNN2개  
input_image_1 = Input(shape=(28,28,1))
conv2d_1_l=Conv2D(32, (3,3), activation='relu', padding='same')(input_image_1)
maxpool_1_l=MaxPool2D((2,2))(conv2d_1_l)
dropout_1_l=Dropout(0.3)(maxpool_1_l)

conv2d_2_l=Conv2D(64, (3,3), activation='relu', padding='same')(dropout_1_l)
maxpool_2_l=MaxPool2D((2,2))(conv2d_2_l)
dropout_2_l=Dropout(0.1)(maxpool_2_l)

flatten_l=Flatten()(dropout_2_l)

dropout_3_l=Dropout(0.2)(flatten_l)
hidden_r=Dense(128,'relu')(dropout_3_l)
output_image1 = Dense(64,'relu')(hidden_r)

# 두번째 이미지 입력 : Conv2개 + DNN2개 
input_image_2=Input(shape=(28,28,1))
conv2d_1_r=Conv2D(32, (3,3), activation='relu', padding='same')(input_image_2)
maxpool_1_r=MaxPool2D((2,2))(conv2d_1_r)
dropout_1_r=Dropout(0.3)(maxpool_1_r)

conv2d_2_r=Conv2D(64, (3,3), activation='relu', padding='same')(dropout_1_r)
maxpool_2_r=MaxPool2D((2,2))(conv2d_2_r)
dropout_2_r=Dropout(0.1)(maxpool_2_r)

flatten_r=Flatten()(dropout_2_r)

dropout_3_r=Dropout(0.1)(flatten_r)
hidden_r=Dense(128,'relu')(dropout_3_r)
output_image2=Dense(64,'relu')(hidden_r)

# 세번째 입력층 
input_data = Input(shape = (1,))



# input layer 묶음 
concatted = Concatenate()([output_image1, output_image2, input_data])


# output layer  
dnn = Dense(32,'relu')(concatted)
final_output = Dense(1,'sigmoid')(dnn)


# model 생성 
model=Model(inputs=[input_image_1, input_image_2, input_data], outputs= final_output)
###########################################################################


# model 학습환경 
model.compile('adam','binary_crossentropy',['accuracy'])


# model 학습 
model_fit=model.fit(x=(x_train_img1, x_train_img2, x_train_data), y=y_train,
                    batch_size=32,
                    epochs=3,
                    validation_data=((x_val_img1, x_val_img2, y_val_data), y_val))









