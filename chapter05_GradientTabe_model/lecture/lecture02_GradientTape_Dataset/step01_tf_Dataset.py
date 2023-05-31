# -*- coding: utf-8 -*-
"""
step01_tf_Dataset.py

Dataset 클래스 
 -  사용할 데이터셋 slicing -> 메모리 로딩
 - mini batch 방식 : batch size 단위 데이터 공급 
"""

import tensorflow as tf
from tensorflow.python.data import Dataset

dir(Dataset)
'''
batch() : batch(size=100)
from_tensor_slices() : dataset slice
shuffle() : 셔플링
'''

############################################
# 형식1) from_tensor_slices(x, y).batch(n) 
#   - (x, y) : dataset  
#   - batch(n) : 1회 공급할 data size
############################################

# 1) x, y 변수 생성 
X = tf.random.normal([5, 2]) # 2차원
y = tf.random.uniform([5]) # 1차원 


# 2) Dataset 만들기 : batch size=2
train_ds = Dataset.from_tensor_slices( (X, y) ).batch(2) # slices 3개

# slices 단위로 넘김
cnt = 0
for X, y in train_ds: 
    cnt += 1
    print('slice num :', cnt)
    print("X={}, y={}".format(X.numpy(), y.numpy()))
'''
slice num : 1
X=[[-0.6373305  2.7589011]
 [ 1.026495   3.3563783]], y=[0.02379429 0.14300716]
slice num : 2
X=[[ 0.3897512   0.34321475]
 [ 0.45475012 -0.39804557]], y=[0.10024238 0.8821609 ]
slice num : 3
X=[[ 0.5830075  -0.58795774]], y=[0.4398737]
'''




###############################################################
# 형식2) from_tensor_slices(x, y).shuffle(buffer_size).batch(n) 
#     - shuffle() 함수 : 메모리 버퍼 크기 지정(서플링 데이터 크기)
#     - batch() : 1회 공급할 data size 지정  
################################################################

# Keras dataset 적용 
from tensorflow.keras.datasets.cifar10 import load_data # Cifar10
(X_train, y_train), (X_val, y_val) = load_data()

print(X_train.shape) # (50000, 32, 32, 3) : 4차원 
print(y_train.shape) # (50000, 1) : 2차원 


# 50,000 image 섞음 -> 100개 image 묶음
train_ds = Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(100) 

for X_train, y_train in train_ds : # 순서대로 1개 slice 넘김
    print("X={}, y={}".format(X_train.shape, y_train.shape))        

# 전체 image -> 서플링 없음 -> 1,000개 image 묶음(1000*10 = slices 10)
test_ds = Dataset.from_tensor_slices((X_val, y_val)).batch(1000) 

for X_val, y_val in test_ds : # 순서대로 1개 slice 넘김
    print("X={}, y={}".format(X_val.shape, y_val.shape)) 


