# -*- coding: utf-8 -*-
"""
step03_gradientTape_binary.py

GradientTape + Sigmoid
 - 비용함수 : Cross Entropy
 - 활성함수 : Sigmoid
"""

import tensorflow as tf # ver2.x
from sklearn.metrics import accuracy_score

# x,y 변수 생성 
X = tf.Variable([[1.,2.],[2,3],[3,1],[4,3],[5,3],[6,2]])# [6,2] 
y = tf.Variable([[0.], [0], [0], [1], [1], [1]])#[6,1]

# 1. model : 순방향 
class Model(tf.keras.Model) : # Model 클래스 상속
    # 1) 생성자 정의
    def __init__(self) :  
        super().__init__() # 부모생성자 호출 
        tf.random.set_seed(12)
        self.w = tf.Variable(tf.random.normal([2, 1])) # 가중치
        self.b = tf.Variable(tf.random.normal([1])) # 편향
    # 2) method 재정의     
    def call(self, X) :     
        return tf.matmul(X, self.w) + self.b # 행렬곱


# 2. 손실 함수 : 손실 반환   
def loss_fn(model, X, y): 
  y_pred = tf.nn.sigmoid(model(X)) # 활성함수 : 확률값 
  # cross entropy 이용 :  손실계산
  loss = -tf.reduce_mean(y*tf.math.log(y_pred) + (1-y)*tf.math.log(1-y_pred)) 
  return loss



# 3. 미분계수 자동 계산 : 역방향 
@tf.function # 연산 속도 향상 
def gradient_fn(model, X, y) :  
    with tf.GradientTape() as tape:
        # 1. 손실 계산 
        loss = loss_fn(model, X, y)  
        # 2. 미분계수       
        gradient = tape.gradient(loss, [model.w, model.b])
    return gradient # 미분계수 반환 


# 4. model 및 optimizer 생성 
model = Model() # 생성자 
optimizer = tf.optimizers.Adam(learning_rate=0.1) # 최적화 


# 5. model 학습 
for step in range(50) :
    # 1. 기울기 반환 
    gradient = gradient_fn(model, X, y) #  미분계수     
    # 2. 기울기와 조절변수를 최적화 알고리즘에 적용 
    optimizer.apply_gradients(zip(gradient, [model.w, model.b])) # model 최적화 
    
    # 모델의 최적화 확인 
    if (step+1) % 5 == 0 :
        print("step = %d, loss = %.6f"%((step+1), loss_fn(model, X, y)))        


# 6. 최적화된 model test
y_pred_prob = tf.sigmoid(model(X)) # 확률예측(0~1)  

# 확률 -> 0 or 1
y_pred= tf.cast(y_pred_prob > 0.5, tf.float32)      
y_pred = tf.squeeze(y_pred) # 예측치 - 1차원      
y_true = tf.squeeze(y) # 정답 - 1차원  

print("="*40)
acc = accuracy_score(y_true, y_pred)
print('분류정확도 =', acc)






