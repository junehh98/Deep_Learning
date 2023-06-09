# -*- coding: utf-8 -*-
"""
step03_GrayShipUser.py

Autoencoder 모델 응용 사례 
 - 사용자의 영화평점에 대한 특징을 추출하여 영화취향이 
   독특한(예측이 어려운) 사용자(그레이십유저)를 선정한다. 
 - 즉 영화 평점을 기록한 movieLens 데이터셋을 대상으로 오토인코더 모델을 
   적용하여 loss가 가장 큰 상위 1%, 5% 그레이십유저 선정 
"""

import numpy as np
from tensorflow.keras.layers import Input # input layer 구축 
from tensorflow.keras.models import Model # DNN Model 생성  
from tensorflow.keras.layers import Dense # DNN layer 구축 
import pandas as pd # csv read 
pd.set_option('display.max_columns', 100) 
from sklearn.model_selection import train_test_split

import tensorflow as tf
import random as rd 
import pandas as pd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)

## 단계1. dataset 가져오기 : 평점예측한 전체 데이터셋(카페 다운로드)   
path = r'C:\ITWILL\6_DeepLearning\data'
ratings = pd.read_csv(path+'/ratings_pred_df.csv')
ratings.shape # (943, 1682)
'''
users : 943명
itmes : 1682개
'''

## 단계2. dataset split : model 훈련용과 평가용
train_data, test_data = train_test_split(ratings, test_size=0.3)
train_data.shape # (660, 1682)
test_data.shape # (283, 1682)


## 단계3. input, encoder, decoder layer 구축  
##################################################
### Functional API : Input과 Model 클래스 이용 
##################################################

# 계층 차원(dimension)
input_dim = (1682,) 
encoding_dim = 256 
decoding_dim = 1682 

# 1) input layer 
inputs = Input(shape = input_dim)  

# 2) encoding layer 
encoded = Dense(units = encoding_dim, activation='relu')(inputs)

# 3) decoding layer 
decoded = Dense(units = encoding_dim, activation='relu')(encoded)

# 4) output layer 
outputs = Dense(units = decoding_dim)(decoded) # 회귀모델 


## 단계4. Autoencoder model 생성 
autoencoder = Model(inputs, outputs) 


## 단계5. autoencoder model 학습 & 평가 
# 1) model 학습환경   
autoencoder.compile(optimizer='adam', loss='mse')

# 2) model 학습 : 훈련셋 학습, 테스트셋 평가  
autoencoder.fit(x=train_data, y=train_data, 
                batch_size=35, 
                epochs=20,
                validation_data=(test_data, test_data))





# 단계6. 모델 재사용(model reuse)
# 1) encoder model 
encoder = Model(inputs, encoded) 

# 2) decoder model 
encoded_inputs = Input(shape = (encoding_dim,)) # encoding_dim
decoded_layer = autoencoder.layers[-1] # 학습된 decoder 객체  
decoder = Model(encoded_inputs, decoded_layer(encoded_inputs))


## 단계7. 재구성결과와 실제데이터 손실 계산 
reconst = autoencoder.predict(ratings) # 재구성(reconstructions)

# 1) loss 계산 : mean(|재구성결과 - 실제데이터)
loss = tf.keras.losses.mae(reconst, ratings) # 평균절댓값오차 
loss = loss.numpy() 


# 2) loss 분포 : 히스토그램    
import matplotlib.pyplot as plt
plt.hist(loss, bins=50)
plt.xlabel("Real dataset loss")
plt.ylabel("No of examples")
plt.show()




## 단계8. 오토인코더 실험 결과 loss가 가장 큰 상위 1%, 5% 선정 
# 1) 손실 상위 임계값 선정 : 1%, 5%
loss_1percent = np.sort(loss)[::-1][int(0.01* len(loss))] # 내림차순 정렬 -> 상위 1% 
loss_5percent = np.sort(loss)[::-1][int(0.05* len(loss))] # 내림차순 정렬 -> 상위 5% 

print("상위 1% 손실 임계값 : ", format(loss_1percent, '.6f'))  
print("상위 5% 손실 임계값: ", format(loss_5percent, '.6f'))
# 상위 1% 손실 임계값 :  0.612895
# 상위 5% 손실 임계값:  0.548417
type(loss) # numpy.ndarray



# 2) 인원수 
print(len(loss[loss > loss_1percent])) # 9명
print(len(loss[loss > loss_5percent])) # 47명


# 3) 상위 1% 색인 추출
pred1 = loss > loss_1percent
idx1 = np.where(pred1)


# 4) 상위 5% tordls cncnf
pred2 = loss > loss_5percent
idx2 = np.where(pred2)


# 5) 상위 1%와 5% 사용자 선정
type(ratings) # pandas.core.frame.DataFrame
grayship1_user1 = ratings.iloc[idx1] # 9 rows x 1682
grayship1_user5 = ratings.iloc[idx2] # 47 rows x 1682


grayship1_user1 





















