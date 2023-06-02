'''
문2) 다음과 같은 조건으로 keras CNN model layer를 작성하시오.

1. Convolution1
    1) 합성곱층 
      - filters=64, kernel_size=5x5, padding='same'  
    2) 풀링층(max) 
     - pool_size= 2x2, strides= 2x2, padding='same'

2. Convolution2
    1) 합성곱층 
      - filters=128, kernel_size=5x5, padding='same'
    2) 풀링층
     - pool_size= 2x2, strides= 2x2, padding='same'
    
3. Flatten layer 

4. Affine layer(Fully connected)
    - 256 node, activation = 'relu'
    - Dropout : 0.5%

5. Output layer(Fully connected)
    - 10 node, activation = 'softmax'
    
--------------------------------------------------------
6. model training 
   - epochs=3, batch_size=100   

7. model evaluation
   - model.evaluate(x=x_val, y=y_val)                   
'''

from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation # Convolution layer
from tensorflow.keras.layers import Dense, Dropout, Flatten # Affine layer

# minst data read
(x_train, y_train), (x_val, y_val) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 

# image data reshape : [s, h, w, c]
x_train = x_train.reshape(60000, 28, 28, 1)
x_val = x_val.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(x_train[0]) # 0 ~ 255

# x_data : int -> float
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')


# x_data : 정규화 
x_train /= 255 # x_train = x_train / 255
x_val /= 255

# y_data : 10 -> 2(one-hot)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)



# model 생성 
model = Sequential()

# 1. CNN Model layer
input_shape = (28, 28, 1)

# Conv layer1
model.add(Conv2D(filters=64, kernel_size=(5, 5),  
                 input_shape = input_shape, activation='relu', padding='same')) 
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')) 

# Conv layer2
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same')) 
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')) 

# Conv layer2, pooling X
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

# 전결합층
model.add(Flatten()) 

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))

# DNN2 : output layer  
model.add(Dense(units = 10, activation='softmax')) 

model.summary()



# 2. model.compile
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])



# 3. model training
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=3, # 반복학습 
          batch_size = 100, 
          verbose=1,  
          validation_data=(x_val, y_val))



# 4. model evaluation
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)





print(model_fit.history.keys()) # key 확인 
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

import matplotlib.pyplot as plt 
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

