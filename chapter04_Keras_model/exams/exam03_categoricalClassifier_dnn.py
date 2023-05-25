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

# 3. keras model & layer 구축(Functional API 방식) 


# 4. DNN model layer 구축 


# 5. model compile : 학습과정 설정(다항 분류기)


# 6. model training : training dataset


# 7. model evaluation : validation dataset


# 8. model save : file save - HDF5 파일 형식 
