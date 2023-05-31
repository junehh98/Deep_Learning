# -*- coding: utf-8 -*-
"""
step02_gradientTape_model.py
  - tf.GradientTape + linear Regression model
  
딥러닝 순방향 & 역방향
 - 순방향(forword) : Model 클래스(X@w+b -> y예측치) -> 손실(loss)
 - 역방향(backword) : GradientTape 클래스(손실 vs 조절변수[w,b])  
"""

import tensorflow as tf

#  X, Y 변수 정의 
X = tf.Variable([1.0, 2.0, 3.0]) # x변수 : 입력(1) 
y = tf.Variable([2.0, 4.0, 6.0]) # y변수 : 정답(1) 


# 1. model : 순방향(y예측값 반환)
class Model(tf.keras.Model) : # Model 클래스 상속    
    def __init__(self) : # 생성자  
        super().__init__() # 부모생성자 호출 
        tf.random.set_seed(12) # seed값 
        self.w = tf.Variable(tf.random.normal([1])) # 가중치 초기화 
        self.b = tf.Variable(tf.random.normal([1])) # 편향 초기화          
    def call(self, X) : # self 메서드
        return X * self.w + self.b # y예측치 : 회귀방정식 
 
'''
model = Model() # 생성자
object.멤버변수
object.멤버메서드()
object() : call() 메서드 호출 
model.w.numpy() # [1.0524982]
model.b.numpy() # [-1.4789797]
model(X.numpy()) # y예측치 반환
'''

# 2. 손실 함수 : 손실 반환 
def loss_fn(model, X, y):  
    y_pred = model(X) # model.call(X) 
    err = y_pred - y # 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE : 손실계산 
    return loss


# 3. 미분계수(기울기) 자동 계산 : 역방향 
@tf.function # 함수장식자 : 연산속도 빠름  
def gradient_fn(model, X, y) :      
    with tf.GradientTape() as tape :       
        # 1. 손실 계산 
        loss = loss_fn(model, X, y) 
        # 2. 미분계수(손실, 조절변수) 
        gradient = tape.gradient(loss, [model.w, model.b])    
    
    return gradient # 미분계수 반환 

 
# 4. model & 최적화 알고리즘 객체  
model = Model() # model 객체 
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05) # 0.01-> 0.05

print('w init=',model.w.numpy(), 'b init=',model.b.numpy())


# 5. model 학습 
for step in range(10) :    
    # 1. 미분계수 반환
    gradient = gradient_fn(model, X, y) # 접선의기울기   
    
    # 2. 최적화 알고리즘 적용 : 모델 최적화(w, b 수정)    
    optimizer.apply_gradients(zip(gradient, [model.w, model.b])) # model 학습(최적화)
    print('w upate =',model.w.numpy(), 'b upate=',model.b.numpy())
    
    # 3. model 학습상태(loss) 출력 :      
    print("step = %d, loss = %.6f"%((step+1), loss_fn(model, X, y))) # 손실함수 호출 
'''
w init= [1.0524982] b init= [-1.4789797]
w upate = [1.7904618] b upate= [-1.1415813]
step = 1, loss = 2.464924
w upate = [2.1165626] b upate= [-0.98551553]
step = 2, loss = 0.575149
w upate = [2.25927] b upate= [-0.9102765]
step = 3, loss = 0.198271
w upate = [2.3203325] b upate= [-0.8711028]
step = 4, loss = 0.121510
w upate = [2.3450646] b upate= [-0.84805906]
step = 5, loss = 0.104322
w upate = [2.3536463] b upate= [-0.8322661]
step = 6, loss = 0.098996
w upate = [2.3550646] b upate= [-0.8197687]
step = 7, loss = 0.096068
w upate = [2.3533216] b upate= [-0.8088048]
step = 8, loss = 0.093661
w upate = [2.3501992] b upate= [-0.79858863]
step = 9, loss = 0.091401
w upate = [2.3464906] b upate= [-0.7887696]
step = 10, loss = 0.089213
'''
    



