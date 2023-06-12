# -*- coding: utf-8 -*-
"""
다변량 시계열 데이터(Multivariate Time Series Data) 분석 : PPT.27 참고 

다변량(multivariate) : y변수가 2개 이상인 경우
 - 주어진 과거의 자료로부터 주어진 구간을 예측하도록 학습한 모델
  예) 시간축에 따른 온도,기압,공기밀도 변화에서 지난 5일간의 데이터를 학습하여 다음날 연속 12시간 온도 예측 
"""


#from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Matplotlib Setting
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
tf.random.set_seed(13) # Random Seed Setting


# Dataset load 
path = r'C:\ITWILL\6_DeepLearning\data'
df = pd.read_csv(path+'/jena_climate_2009_2016.csv')
df.head()
df.iloc[:144] # 1일 
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
143  02.01.2009 00:00:00    999.59     -4.54  ...      0.41           0.88
'''


### 1. 변수 선택 & 탐색
df.info()
features = df[['p (mbar)','T (degC)','rho (g/m**3)']] # 3개 칼럼 선택
'''
 p (mbar)         420551 non-null  float64 : 대기압(밀리바 단위)
 T (degC)         420551 non-null  float64 : 온도(섭씨)
 rho (g/m**3)     420551 non-null  float64 : 공기밀도

x변수 : 기압, 온도, 밀도(5일 날짜) 
y변수 : 온도(다음날 0~12시간) 
'''


 
features.index = df['Date Time'] # 날짜시간 칼럼을 색인으로 지정 



# 3개 변수 시계열 시각화 
features.plot(subplots=True) 


# 표준화(Z-Normalization)
dataset = features.values # pandas -> numpy 변환 
data_mean = dataset.mean(axis=0) 
data_std = dataset.std(axis=0)
dataset = (dataset-data_mean)/data_std


### 2. 다변량 데이터 생성  

# 1) 다변량 데이터 생성 함수  
def multivariate_data(dataset, target, s_index, e_index, 
                      past_size, target_size, step):
    
    X = []; y = [] # x변수, y변수
    
    s_index = s_index + past_size # 720
    if e_index is None: # val dataset
        e_index = len(dataset) - target_size 
    
    # 5일 단위 시계열 데이터를 6개씩(STEP) 건너뛴 120개 구성 
    for i in range(s_index, e_index):
        indices = range(i-past_size, i, step) 
        X.append(dataset[indices]) # X변수 : [0, 720, 6]=120, [1,721,6]=120, ... 
        
        y.append(target[i:i+target_size]) # y변수 : 72개 

    return np.array(X), np.array(y)



'''
1시간 관측치 : 6개 = 1*6(10분 단위 기록)
1일 관측치 : 144 = 6*24
5일 관측치(past_data) : 720 = 144*5 -> x변수(5일 날씨)
12시간 관측치(target_data) : 72 = 144/2 -> y변수(다음날 0~12시간)

STEP = 6 일때 실제 x변수 
720/6 = 120 : 1시간 단위 건너뜀(1시간 이내 변경 없음 가정) 
'''
# 2) 다변량 데이터 생성 : Multi-Step Model에 적합한 형태 
past_data = 720 # 지난 5일 데이터
target_data = 72 # 예측할 데이터 : 다음날 연속 12시간 예측)
STEP = 6 # 샘플링 간격(1시간당 표본 추출)  

TRAIN_SPLIT = 300000 # train vs val split 기준

# 훈련셋 데이터 생성 
X_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_data,
                                               target_data, STEP)

X_train_multi.shape # (299280, 120, 3) : (5일 날짜 정보, 독립변수3개)
y_train_multi.shape # (299280, 72) : (72 : 0~12시간) -> 다변량 

# 검증셋 데이터 생성 
X_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_data,
                                             target_data, STEP)
X_val_multi.shape # (119759, 120, 3)
y_val_multi.shape # (119759, 72)

# model 공급 데이터 만들기 
BATCH_SIZE = 256 # 1회 공급데이터 크기  
BUFFER_SIZE = 10000

train_data_multi = tf.data.Dataset.from_tensor_slices((X_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((X_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()



X_train_multi.shape # (299280, 120, 3)
X_train_multi.shape[-2:] # (120, 3)

y_train_multi.shape # (299280, 72)
y_train_multi.shape[1] # 72

### 3. LSTM Model 생성(Single Step Model)
multi_model = Sequential()

# 입력 모양 
input_shape = X_train_multi.shape[-2:]  # (120, 3)
multi_model.add(LSTM(32, input_shape = input_shape)) ## (?, 120, 3)

multi_model.add(Dense(72)) # Multi Step : size=72

multi_model.compile(optimizer="rmsprop", loss='mse')

# Train the Model
multi_history = multi_model.fit(train_data_multi, 
                                epochs=10,
                                steps_per_epoch=200,
                                validation_data=val_data_multi,
                                validation_steps=50)


### 4. Model prediction 

# 1) 다변량 시계열 모델 예측 시각화 함수 
def multi_step_plot(past_data, future_data, pred): 
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(past_data), 0)) # 시계열 데이터 출력 구간
    num_out = len(future_data) # 72

    plt.plot(num_in, past_data, label='History')
    plt.plot(np.arange(num_out)/STEP, future_data,'bo',label='True Future')
    plt.plot(np.arange(num_out)/STEP, pred,'ro', label='Predicted')
    plt.legend(loc='best')
    plt.title('Multi-Step Prediction')
    plt.show()
    
    
# 2) 테스트셋 선택 : 검증셋 중에서 3개 관측치 선택   
for X, y in val_data_multi.take(3):
    y_pred = multi_model.predict(X)
    
    # 다변량 시계열 모델 예측 시각화 함수 호출  
    multi_step_plot(X[0, :, 1].numpy(), y[0].numpy(), y_pred[0])
    
    
    