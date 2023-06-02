'''
문1) bmi.csv 데이터셋을 이용하여 다음과 같이 sigmoid classifier의 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : 1,2번째 칼럼(height, weight) 
       -> y변수 : 3번째 칼럼(label)
   조건2> 딥러닝 최적화 알고리즘 : Adam
   조건3> learning rage = 0.01    
   조건4> 반복학습 : 1,000번, 100 step 단위로 loss 출력 
   조건5> 최적화 모델 테스트 :  분류정확도(Accuracy report) 출력 
   
 <출력결과>
step :  100 , loss val =  0.6281926
step :  200 , loss val =  0.5366587
step :  300 , loss val =  0.4676207
step :  400 , loss val =  0.4148042
step :  500 , loss val =  0.3735144
step :  600 , loss val =  0.3404998
step :  700 , loss val =  0.31354335
step :  800 , loss val =  0.29111955
step :  900 , loss val =  0.27215955
step :  1000 , loss val =  0.2558988
========================================
accuracy = 0.9743742550655542    
'''

import tensorflow as tf 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale # x변수 전처리 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # y변수 전처리 

import pandas as pd
 
# csv file load
bmi = pd.read_csv('/Users/junehh98/Desktop/itwill/6_DeepLearning/data/bmi.csv')
print(bmi.info())

bmi['label'].value_counts() 
'''
normal    7677 -> 0
fat       7425 -> 1
thin      4898 -> 이항분류를 위해서 제거 
'''


# subset 생성 : label에서 normal과 fat 추출(thib 제외) 
bmi = bmi[bmi.label.isin(['normal','fat'])]
print(bmi.head())


# x,y 변수 추출 
x_data = bmi.iloc[:,:2] # x변수(1,2칼럼)
y_data = bmi.iloc[:,2] # y변수(3칼럼)


# x_data 정규화
x_data = minmax_scale(x_data)

# y_data 인코딩(normal -> 0, fat -> 1)
y_data = LabelEncoder().fit_transform(y_data) # 문자형 -> 숫자형
# [1, 1, 1, ..., 1, 0, 0]
y_data = OneHotEncoder().fit_transform(y_data.reshape([-1, 1])).toarray()
'''
[[0., 1.],
'''
       
print(x_data.shape) # (15102, 2)
print(y_data.shape) # (15102, 2)

################# X,Y 데이터 전처리 완료 #####################




# 1. X,Y 변수 정의 : float32 -> float64 변환   
X = tf.constant(x_data, tf.float32) 
Y = tf.constant(y_data, tf.float32)
 

# 2. w,b 변수 정의 : 초기값(정규분포 난수 )
w = tf.Variable(tf.random.normal([2, 2]))# [입력수,출력수]
b = tf.Variable(tf.random.normal([2])) # [출력수] 


# 3. 회귀방정식 
def linear_model(X) : # train, test
    y_pred = tf.linalg.matmul(X, w) + b 
    return y_pred # 2차원 


# 4. sigmoid 활성함수 적용 
def sig_fn(X):
    y_pred = linear_model(X)
    sig = tf.nn.sigmoid(y_pred) 
    return sig

# 5. 손실 함수 정의 : 손실계산식 수정 
def loss_fn() : #  인수 없음 
    sig = sig_fn(X) 
    loss = -tf.reduce_mean(Y*tf.math.log(sig)+(1-Y)*tf.math.log(1-sig))
    return loss


# 6. 최적화 객체 : learning_rate= 0.01
opt = tf.optimizers.Adam(learning_rate=0.01)


# 7. 반복학습 : 반복학습 : 1,000번, 100 step 단위로 loss 출력 
for step in range(1000):
    opt.minimize(loss_fn, var_list=[w, b])
    
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())


# 8. 최적화된 model 테스트
y_pred = tf.cast(sig_fn(X).numpy() > 0.5, dtype=tf.float32)

acc = accuracy_score(Y, y_pred)
print(acc) # 0.9664282876440207


