# -*- coding: utf-8 -*-
"""
step04_gradientTape_category.py

GradientTape + Softmax
 - 비용함수 : cross Entropy
 - 활성함수 : Softmax
"""

import tensorflow as tf # ver2.x
from sklearn.metrics import accuracy_score

# X, y 변수 정의 
# [털, 날개]
X = tf.Variable(
    [[0., 0.], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1]]) # [6, 2]

# [기타, 포유류, 조류] : [6, 3] -> one hot encoding
y = tf.Variable([
    [1., 0., 0.],  # 기타[0]
    [0, 1, 0],  # 포유류[1]
    [0, 0, 1],  # 조류[2]
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 1. model : 순방향 
class Model(tf.keras.Model) : 
    # 1) 생성자 정의
    def __init__(self) :  
        super().__init__() # 부모생성자 호출 
        tf.random.set_seed(12)
        self.w = tf.Variable(tf.random.normal([2, 3])) #가중치 초기화
        self.b = tf.Variable(tf.random.normal([3])) # 편향 초기화 
    # 2) method 재정의     
    def call(self, X) :     
        return tf.matmul(X, self.w) + self.b # 행렬곱


# 2. 손실 함수 : 손실값 반환 
def loss_fn(model, X, y):     
    #err = model(X) - y # 오차       
    y_pred = tf.nn.softmax(model(X)) # 활성함수 
    loss = -tf.reduce_mean(y*tf.math.log(y_pred) + (1-y)*tf.math.log(1-y_pred)) 
    return loss


# 3. 미분계수 자동 계산 : 역방향   
@tf.function # 연산 속도 향상 
def gradient_fn(model, X, y) :  
    with tf.GradientTape() as tape:
        # 1. 손실 
        loss = loss_fn(model, X, y) 
        # 2. 미분계수        
        gradient = tape.gradient(loss, [model.w, model.b])
    return gradient # 미분계수 반환 


# 4. 모델 및 최적화 객체   
model = Model() # 모델 객체 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1) # 최적화 


# 5. model 학습 : echops=50
for step in range(50) :
    #  1.미분계수 반환 
    gradient = gradient_fn(model, X, y)  

    # 2.최적화 객체 반영 -> model 학습(최적화)
    optimizer.apply_gradients(zip(gradient, [model.w, model.b])) 
    
    if (step+1) % 10 == 0 :
        print("step = %d, loss = %.6f"%((step+1), loss_fn(model, X, y)))        


# 6. 최적화된 model test
y_pred_prob = tf.nn.softmax(model(X)) # 활성함수
# [9.7153676e-01, 2.5946224e-02, 2.5170695e-03]

# y예측치, y정답 : 10진수 변환 
y_pred = tf.argmax(y_pred_prob, 1) # 예측치  
y_true = tf.argmax(y, 1) # 정답  
   
print("="*40)   
acc = accuracy_score(y_true, y_pred)
print('분류정확도 =', acc)

    
    
   