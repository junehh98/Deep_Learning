'''
KNN(K-Nearest Neighbor) 알고리즘
  - k개 최근접이웃 선정 : 학습과정 없음 
  - Euclidean 거리계산식
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 알려진 집단 
p1 = [1.2, 1.1] # A집단 
p2 = [1.0, 1.0] # A집단 
p3 = [1.8, 0.8] # B집단 
p4 = [2, 0.9]   # B집단 

x_data = np.array([p1, p2, p3, p4]) # 알려진 집단 
y_data = [1.6, 0.85] # 알려지지 않은 집단(분류대상)

label = np.array(['A','A','B','B']) # 알려진 집단 범주


plt.plot(x_data[:,0], x_data[:,1],'bo')
plt.plot(y_data[0], y_data[1], 'ro')
plt.show()         


# 집단 변수 정의 : 알려진 집단과 알려지지 않은 집단 
X = tf.constant(x_data, tf.float32) # 알려진 집단 
Y = tf.constant(y_data, tf.float32) # 알려지지 않은 집단(분류대상)


# Euclidean 거리계산식 
distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(X-Y),axis=1))
print(distance) 

# euclidean_distance = tf.norm(X - Y, axis=1)
# 결과값은 같음 


# 가장 가까운 거리 index 반환 
idx = tf.argmin(distance).numpy() 

print('분류 index :', idx) # 가장 거리가까운 색인  
print('k=1 분류결과', label[idx]) # k=1 분류결과 B 

# k=3인 경우
sorted_index = tf.argsort(distance).numpy()
sorted_index #  # array([2, 3, 0, 1])

result = label[sorted_index[:3]]
print(result) # ['B' 'B' 'A']

ser_arr = pd.Series(result) # series 객체로 타입 변경
cnt = ser_arr.value_counts()
print(cnt) # B선택
