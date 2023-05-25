'''
문2) bmi.csv 데이터셋을 이용하여 다음과 같이 softmax classifier 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : height, weight 칼럼 
       -> y변수 : label(3개 범주) 칼럼
    조건2> 딥러닝 최적화 알고리즘 : Adam
    조건3> learning rage : 0.01 or 0.05 선택(분류정확도 높은것) 
    조건4> 반복학습, step 단위로 loss : <출력결과> 참고 
    조건5> 분류정확도 출력
    조건6> 예측치와 정답 15개 출력   
    
  <출력 결과>
step = 100 , loss = 0.37230268
step = 200 , loss = 0.2721474
step = 300 , loss = 0.22131577
step = 400 , loss = 0.18909828
step = 500 , loss = 0.16659674
step = 600 , loss = 0.14988557
step = 700 , loss = 0.13691667
step = 800 , loss = 0.12651093
step = 900 , loss = 0.117940545
step = 1000 , loss = 0.11073165
========================================
accuracy = 0.98185
========================================
y_pred :  [2 2 1 1 1 1 2 0 2 0 1 0 1 2 0]
y_true :  [2 2 1 1 1 1 2 0 2 0 1 0 1 2 0]
'''

import tensorflow as tf # ver1.x
from sklearn.preprocessing import minmax_scale # 전처리
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # 전처리
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
 
bmi = pd.read_csv('C:/ITWILL/6_DeepLearning/data/bmi.csv')
print(bmi.info())

# x,y 변수 추출 
x_data = bmi.iloc[:,:2] # x변수(1,2칼럼)
y_data = bmi.iloc[:,2] # y변수(3칼럼)


# x_data 정규화
x_data = minmax_scale(x_data)

# y_data 레이블 인코딩 + one hot 인코딩
y_data = LabelEncoder().fit_transform(y_data) # 레이블 인코딩 
y_data = OneHotEncoder().fit_transform(y_data.reshape([-1, 1])).toarray()

print(x_data.shape) # (15102, 2)
print(y_data.shape) # (15102, 3)


################# X,Y 데이터 전처리 완료 #####################


# 1. X,Y변수 정의 : 공급형 변수 
X = tf.constant(x_data, tf.float32) 
y = tf.constant(y_data, tf.float32)

# 2. w,b 변수 정의 
w = tf.Variable(tf.random.normal([2, 3]))
b = tf.Variable(tf.zeros([3]))

# 3. 회귀방정식 
def linear_model(X) : # train, test
    y_pred = tf.matmul(X, w) + b  # 행렬곱 
    return y_pred

# 4. softmax 활성함수 적용 
def soft_fn(X):
    y_pred = linear_model(X)
    soft = tf.nn.softmax(y_pred)
    return soft

# 5. 손실 함수 정의 : 손실계산식 수정 
def loss_fn() : #  인수 없음 
    soft = soft_fn(X) # 훈련셋 -> 예측치 : 회귀방정식  
    loss = -tf.reduce_mean(y*tf.math.log(soft)+(1-y)*tf.math.log(1-soft))
    return loss


# 6. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.05)


# 7. 반복학습 
for step in range(1000):
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    if (step+1) % 100 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
'''
학습률(0.01) step = 1000 , loss val =  0.24172871
학습률(0.05) step = 1000 , loss val =  0.10076756 (더 좋음)
'''


# 8. 최적화된 model 검증 
y_pred = soft_fn(X).numpy()
print(y_pred)


y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.argmax(y, axis=1)


acc= accuracy_score(y_true, y_pred)
print(acc) # 0.98465













