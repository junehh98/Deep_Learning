# -*- coding: utf-8 -*-
"""
단변량 시계열 데이터(Univariate Time Series Data) 분석 : PPT.18 참고 

단변량모델 vs 다변량모델 : y변수 개수 기준 
단변량(univariate)모델 : y변수 1개 
 예) 독립변수 2개(품종, 뿌림방법) -> 종속변수 1개(상 or 하)
다변량(multivariate)모델 : y변수 2개 이상
 예) 독립변수 2개(품종, 뿌림방법) -> 종속변수 3개(상, 중, 하)
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# Matplotlib Setting
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
tf.random.set_seed(13) # Random Seed Setting
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


'''
에나 기후(jena climate) dataset 설명
독일 에나 연구소에서 제공하는 에나 기후(jena climate) 데이터셋으로
온도, 기압, 습도 등 14개의 날씨 관련 Columns를 가지고 있고, 2009~2016년 
간 매일 10분 마다 기록된 날씨 데이터셋
'''

# 1. csv file read : 에나 기후(jena climate) dataset [카페] 다운로드 
path = r'C:\ITWILL\6_DeepLearning\data'

df = pd.read_csv(path+'/jena_climate_2009_2016.csv')
df.info() # 420551
'''
RangeIndex: 420551 entries, 0 to 420550
Data columns (total 15 columns):
 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   Date Time        420551 non-null  object  : 날짜/시간 
 1   p (mbar)         420551 non-null  float64 : 대기압(밀리바 단위)
 2   T (degC)         420551 non-null  float64 : 온도(섭씨)
 3   Tpot (K)         420551 non-null  float64 : 온도(절대온도)
 4   Tdew (degC)      420551 non-null  float64 : 습도에 대한 온도
 5   rh (%)           420551 non-null  float64 : 상대 습도
 6   VPmax (mbar)     420551 non-null  float64 : 포화증기압
 7   VPact (mbar)     420551 non-null  float64 : 중기압 
 8   VPdef (mbar)     420551 non-null  float64 : 중기압부족 
 9   sh (g/kg)        420551 non-null  float64 : 습도 
 10  H2OC (mmol/mol)  420551 non-null  float64 : 수증기 농도 
 11  rho (g/m**3)     420551 non-null  float64 : 공기밀도 
 12  wv (m/s)         420551 non-null  float64 : 풍속 
 13  max. wv (m/s)    420551 non-null  float64 : 최대풍속
 14  wd (deg)         420551 non-null  float64 : 풍향 
''' 

df.head()
df.tail()

'''
1시간 관측치 : 6개 = 1*6(10분단위 기록)
1일 관측치 : 144개 = 6*24
5일 관측치 : 720개 = 144*5
'''
df.iloc[:144] # 1일 관측치 
'''
               Date Time  p (mbar)  T (degC)  ...  wv (m/s)  max. wv (m/s)  wd (deg)
0    01.01.2009 00:10:00    996.52     -8.02  ...      1.03           1.75     152.3
1    01.01.2009 00:20:00    996.57     -8.41  ...      0.72           1.50     136.1
2    01.01.2009 00:30:00    996.53     -8.51  ...      0.19           0.63     171.6
3    01.01.2009 00:40:00    996.51     -8.31  ...      0.34           0.50     198.0
4    01.01.2009 00:50:00    996.51     -8.27  ...      0.32           0.63     214.3
..                   ...       ...       ...  ...       ...            ...       ...
139  01.01.2009 23:20:00    999.81     -4.51  ...      0.44           0.88     198.6
140  01.01.2009 23:30:00    999.80     -4.33  ...      0.24           0.63     132.7
141  01.01.2009 23:40:00    999.71     -4.33  ...      0.77           1.13     206.4
142  01.01.2009 23:50:00    999.67     -4.58  ...      0.69           1.25     213.9
143  02.01.2009 00:00:00    999.59     -4.54  ...      0.41           0.88     155.0
'''

#######################################
# LSTM을 이용한 기상예측: 단변량
#######################################

### 1. 변수 선택 및 탐색  
uni_data = df['T (degC)'] # 온도 칼럼 
uni_data.index = df['Date Time'] # 날짜 칼럼으로 index 지정 


# Visualization the univariate : 표준화 필요성 확인 
uni_data.plot(subplots=True)
plt.show() # -20 ~ 40

# 시계열 자료 추출 
uni_data = uni_data.values # 값 추출 

# 표준화(Z-Normalization)   
uni_train_mean = uni_data.mean()
uni_train_std = uni_data.std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

# 표준화 여부 확인 
#plt.plot(uni_data)
#plt.show() # -3 ~ 3



### 2. 단변량 데이터 생성 : lstm모델 공급 자료에 적합한 형태 

# 1) 단변량 데이터 생성 함수 
def univariate_data(dataset, s_index, e_index, past_size) : 
    X = [] # x변수 
    y = [] # y변수 

    s_index = s_index + past_size
    if e_index is None: # val dataset 
        e_index = len(dataset) 
    
    for i in range(s_index, e_index): 
        indices = range(i-past_size, i) 
        X.append(np.reshape(dataset[indices], (past_size, 1))) # x변수(20, 1)  
        
        y.append(dataset[i]) # y변수(1,)  
        
    return np.array(X), np.array(y)


# 2) 단변량 데이터 생성 
TRAIN_SPLIT = 300000 # train vs val split 기준
past_data = 20 # x변수 : 과거 20개 자료[0~19, 1~20,...299979~299999] 

# 훈련셋 
X_train, y_train = univariate_data(uni_data,0,TRAIN_SPLIT,past_data)
# 검증셋 
X_val, y_val = univariate_data(uni_data,TRAIN_SPLIT, None, past_data)

# Check the Data
print(X_train.shape) # (299980, 20, 1) : (full_batch, time_steps, features)
print(y_train.shape) # (299980,) 
'''
X변수 : 과거 20개 자료 
y변수 : 21번째 정답 
'''
X_val.shape # (120531, 20, 1) 
y_val.shape # (120531,)


### 3. LSTM Model 학습 & 평가   
input_shape=(20, 1) # (time_steps : 1day~20day, features=독립변수)

model = Sequential()
model.add(LSTM(16, input_shape = input_shape)) 
model.add(Dense(1)) # 회귀함수(21day)
model.summary()


# 학습환경 
model.compile(optimizer='adam', loss='mse')


# 모델 학습 
model_history = model.fit(X_train, y_train, epochs=10, # trainset
          batch_size = 256,
          validation_data=(X_val, y_val))#, # valset 

          
# model evaluation 
print('='*30)
print('model evaluation')
model.evaluate(x=X_val, y=y_val)



### 4. Model 손실(Loss) 시각화 
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Single Step Training and validation loss')
plt.legend(loc='best')
plt.show()


### 5. model prediction 

# 테스트셋 선택 : 검증셋 중에서 5개 관측치 선택 
X_test = X_val[:5] # (5, 20, 1)
y_test = y_val[:5] # (5,)


# 예측치 : 5개 관측치 -> 5개 예측치 
y_pred = model.predict(X_test)  


# 6. 시계열 자료와 lstm모델 예측 시각화 

# 1) 시계열 자료와 예측치 시각화 함수  
def show_plot(dataset):
    # 레이블과 마커 준비 
    labels = ['History', 'True Future', 'Model Prediction'] 
    marker = ['.-', 'rx', 'go']
    
    # x축 : 시계열 자료(20개)   
    time_steps = list(range(-len(dataset[0]), 0)) # -20 ~ -1 
    
    for i, _ in enumerate(dataset):               
        if i==0: # i=0 일때 시계열 자료 시각화 
            plt.plot(time_steps, dataset[i].flatten(), # 1차원 평탄화 함수
                     marker[i], label=labels[i])            
        else: # i=1, 2 일때 실제 와 예측치 시각화(x=0)  
            plt.plot(0, dataset[i], marker[i], 
                     markersize=10, label=labels[i])

    
    plt.legend(loc="best")
    plt.title('Simple LSTM model')
    plt.xlim([time_steps[0], 10]) # x축범위 : -20 ~ 10 
    plt.xlabel('Time-Step') 
    plt.show()

    
# 2) 5개 관측치의 예측결과 시각화  
for X, y in zip(X_test, y_test) :
    X = X.reshape(1, 20, 1) 
    y_pred = model.predict(X) # y 예측치         
    
    show_plot([X[0], y, y_pred[0]]) # 함수 호출 


