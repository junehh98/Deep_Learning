# -*- coding: utf-8 -*-

'''
문) load_wine 데이터셋을 이용하여 GradientTape model을 구현하시오.
  조건1> train/test - 7:3비율 
  조건2> y 변수 : wine.target
  조건3> x 변수 : wine.data
  조건4> learning_rate=0.1
  조건5> optimizer = Adam
  조건6> epoch(step) = 300
  조건7> 모델 평가 : accuracy score
  
  
  <출력결과 예시>
초기 손실값 : 0.935543
------------------------------
Step = 020 -> loss = 0.242
Step = 040 -> loss = 0.126
Step = 060 -> loss = 0.091
Step = 080 -> loss = 0.073
Step = 100 -> loss = 0.061
Step = 120 -> loss = 0.053
Step = 140 -> loss = 0.046
Step = 160 -> loss = 0.041
Step = 180 -> loss = 0.036
Step = 200 -> loss = 0.033
Step = 220 -> loss = 0.030
Step = 240 -> loss = 0.027
Step = 260 -> loss = 0.025
Step = 280 -> loss = 0.023
Step = 300 -> loss = 0.021
------------------------------
분류정확도 : 0.9629629629629629 
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # y data -> one hot
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale # 정규화 
import tensorflow as tf 

# 1. data load
wine = load_wine()
print(wine) # "data", "target"

# 변수 선택  
X = wine.data  
X.mean() # 69.13366292091617

# X변수 정규화(0~1) 
x_data = minmax_scale(X)

# y변수 one-hot
y = wine.target
X.shape # (178, 13)

y_data = to_categorical(y)
print(y_data.shape) # (178, 3)


# train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=123)

x_train.shape # (124, 13)
y_train.shape # (124, 3)


# 2. Model 클래스 
class Model(tf.keras.Model): # keras Model class 상속 
  def __init__(self): # 생성자 
    super().__init__() 
    self.W = tf.Variable(tf.random.normal([13, 3])) #[입력수,출력수]
    self.B = tf.Variable(tf.random.normal([3])) # [출력수]
  def call(self, X): # 메서드 재정의 
    return tf.matmul(tf.cast(X, tf.float32), self.W) + self.B # 예측치  


# 3. 손실함수 : (예측치, 정답) -> 손실(loss) 
def loss_fn(model, X, y): 
    y_pred = tf.nn.softmax(model(X)) # 활성함수(model) 
    loss = -tf.reduce_mean(y*tf.math.log(y_pred) + (1-y)*tf.math.log(1-y_pred)) 
    return loss

# 4. 기울기 계산 함수 : 손실(loss) -> 기울기 반환  
@tf.function # 연산 속도 향상 
def gradient_fn(model, X, y) :  
    with tf.GradientTape() as tape:
        # 1. 손실 
        loss = loss_fn(model, X, y) 
        # 2. 미분계수        
        gradient = tape.gradient(loss, [model.W, model.B])
    return gradient # 미분계수 반환 
  

# 5. 모델 및 최적화 객체   
model = Model() # 모델 객체 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1) # 최적화 


# 초기 손실값 : 학습이전 손실 
loss = loss_fn(model, x_train, y_train)
print('초기 손실값 :', loss.numpy())
print('='*40)

# 6. 반복 학습(train) : Model 객체와 손실함수 이용
epochs = 300
for step in range(epochs) : 
    #  1.미분계수 반환 
    gradient = gradient_fn(model, x_train, y_train) # 훈련셋(train) 이용  

    # 2.최적화 객체 반영 -> model 학습(최적화)
    optimizer.apply_gradients(zip(gradient, [model.W, model.B])) 
    
    if (step+1) % 20 == 0 :
        print("step = %3d, loss = %.6f"%((step+1), 
                        loss_fn(model, x_train, y_train)))        

    
# 7. 모델 평가(test) : 분류정확도
y_pred_prob = tf.nn.softmax(model(x_test)) # 활성함수

# y예측치, y정답 : 10진수 변환 
y_pred = tf.argmax(y_pred_prob, 1) # 예측치  
y_true = tf.argmax(y_test, 1) # 정답  
   
print("="*40)   
acc = accuracy_score(y_true, y_pred)
print('분류정확도 =', acc)









  
