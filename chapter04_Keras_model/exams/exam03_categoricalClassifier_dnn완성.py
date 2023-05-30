'''
문3) digits 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  64 x 32
       L2 =  32 x 16
       L3 =  16 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 100 
  조건7> model save : keras_model_digits.h5
'''

import tensorflow as tf # ver 2.0
from sklearn.datasets import load_digits # dataset load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical # y변수 one hot
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense # model layer
from tensorflow.keras.models import load_model # saved model file -> loading


# 1. digits dataset loading
digits = load_digits()

x_data = digits.data
y_data = digits.target

print(x_data.shape) # (1797, 64) : matrix
print(y_data.shape) # (1797,) : vector

# image show
import matplotlib.pyplot as plt
plt.imshow(X=x_data[0].reshape(8,8))
plt.show()
# image 정답 
y_data[0] # 0

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
y_one_hot = to_categorical(y_data)
y_one_hot
'''
[1., 0., 0., ..., 0., 0., 0.],
'''
y_one_hot.shape # (1797, 10)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.models import Model # DNN Model 생성
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # layer 

# 3-1. keras model & layer 구축(Sequential API 방식) 
model = Sequential()

model.add(Dense(units = 32, input_shape = (64,), activation="relu")) # 1층
model.add(Dense(units = 16, activation="relu")) # 2층
model.add(Dense(units = 10, activation="softmax")) # 3층
model.summary() #Total params: 2,778

# 3-2. keras model & layer 구축(Functional API 방식) 
inputs = Input(shape=(64)) # Input 클래스 이용 

hidden1 = Dense(units = 32, activation="relu")(inputs)
hidden2 = Dense(units = 16, activation="relu")(hidden1)
outputs = Dense(units = 10, activation="softmax")(hidden2)

model = Model(inputs, outputs) # Model 클래스 이용 
model.summary() #Total params: 2,778

dir(model) # save

# 4. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

# 5. model training : training dataset
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=100,  # 반복학습 횟수 
          verbose=1,  # 출력 여부 
          validation_data=(x_val, y_val))  # 검증셋 


# 6. model evaluation : validation dataset
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)


# 7. model save : file save - HDF5 파일 형식 
model.save('keras_model_digits.h5')


my_model = load_model('keras_model_digits.h5')

# new data 
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_one_hot, test_size = 0.5)

y_pred = my_model.predict(x_test)

# 확률 -> 10진수 
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix

con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)
'''
[[ 96   0   0   0   0   0   0   0   0   0]
 [  0  82   0   0   0   0   0   0   0   0]
 [  0   1  84   0   0   0   0   0   1   0]
 [  0   0   1  93   0   0   0   0   0   0]
 [  0   0   0   0  96   0   0   0   0   0]
 [  0   0   1   0   0  88   0   1   0   0]
 [  0   0   0   0   0   0  83   0   0   0]
 [  0   0   0   0   0   0   0  86   1   0]
 [  0   0   0   0   0   0   1   0  79   0]
 [  0   0   0   1   0   1   0   0   0 103]]
'''

acc = accuracy_score(y_true, y_pred)
print(acc) # 0.9899888765294772








