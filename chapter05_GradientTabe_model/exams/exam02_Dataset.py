# -*- coding: utf-8 -*-
"""
 문) mnist 데이터셋을 대상으로 다음과 같이 train_ds, test_ds를 생성
    하고, slice 결과를 출력하시오.
     <조건1> train_ds : shuffle = 10,000, batch size = 32
     <조건2> test_ds : batch size = 32
     <조건3> slice 결과 출력 : <출력결과> 참고 
     
 <출력결과>
 train_ds : 전체 slices = 1875
 test_ds : 전체 slices = 10    
"""

from tensorflow.python.data import Dataset # dataset 생성 
from tensorflow.keras.datasets.mnist import load_data # dataset

# 1. MNIST dataset load
(x_train, y_train), (x_val, y_val) = load_data()


