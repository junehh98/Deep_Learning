# -*- coding: utf-8 -*-
"""
 문) mnist 데이터셋을 대상으로 다음과 같이 train_ds, test_ds를 생성
    하고, slice 결과를 출력하시오.
     <조건1> train_ds : shuffle = 10,000, batch size = 32
     <조건2> test_ds : batch size = 1000
     <조건3> slice 결과 출력 : <출력결과> 참고 
     
 <출력결과>
 train_ds : 전체 slices = 1875
 test_ds : 전체 slices = 10    
"""

from tensorflow.python.data import Dataset # dataset 생성 
from tensorflow.keras.datasets.mnist import load_data # dataset

# 1. MNIST dataset load
(x_train, y_train), (x_val, y_val) = load_data()


# 2. train_ds : 60,000 image 섞음 -> 32개 image 묶음(100*500 = slices 500)  
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32) 
#<DatasetV1Adapter shapes: ((None, 32, 32, 3), (None, 1)), types: (tf.uint8, tf.uint8)>

# slices 500 -> slice 1(100 images) 
cnt = 0
for x_train, y_train in train_ds : # 순서대로 1개 slice 넘김
    #print("x={}, y={}".format(x_train.shape, y_train.shape))
    # x=(100, 32, 32, 3), y=(100, 1)
    cnt += 1
print("train_ds : 전체 slices =", cnt) # train_ds 전체 slices = 1875


# 3. test_ds : 전체 image -> 서플링 없음 -> 1,000개 image 묶음(1000*10 = slices 10)
test_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(1000) 

cnt = 0
for x_val, y_val in test_ds : 
    #print("x={}, y={}".format(x_val.shape, y_val.shape))
    cnt += 1
    
print("test_ds : 전체 slices =", cnt) # test_ds 전체 slices = 10
    
    
    
    
    
    