# -*- coding: utf-8 -*-
"""
step01_gradientTape.py

미분계수(differential coefficient) 자동 계산   
 - 2차함수(u자 곡선)에서 접선의 기울기 계산
 - 딥러닝 모델에서 조절변수(가중치와 편향) 수정 기능
 - 미분계수 자동 계산 클래스 : tf.GradientTape    
"""

import tensorflow as tf


############################################
# 1. 미분계수 자동 계산 예(강의자료 ppt.8) 
############################################

# 단계1. 함수 f(x) : 2차함수(u자곡선 그래프) 
def f(x) :
    y = x**2+ x + 3  
    return y


# 단계2. 도함수 f'(x)  
def gradient(f, x) :    
    with tf.GradientTape() as tape:  
        # 도함수 : f'(x) = 2x + 1 + 0                               
        dy = tape.gradient(target=f(x), sources=x) # 도함수 f'(x)   
    return dy.numpy() 

'''
target = f(x), sources=x
딥러닝 : target = 손실(loss), sources=조절변수[w,b] 
'''

# 단계3. 도함수 f'(x)에 x값을 넣어서 미분계수 y값 반환
x = tf.Variable([2.0, 1.5]) # 독립변수 
  
dy = gradient(f, x)  
print('미분계수 =', dy)  # [5. 4.]
    

#############################################################
## 2. 미분계수 자동 계산 -> 조절변수[w, b] 수정 : model 최적화   
##############################################################
 
tf.random.set_seed(23)
w = tf.Variable(tf.random.normal([1])) # 가중치 
b = tf.Variable(tf.random.normal([1])) # 편형
X = [1.0] # 독립변수 
y = [2.0] # 종속변수 

# 조절변수 초기값 
print('='*45)
print('w init=',w.numpy(), 'b init=',b.numpy())
print('='*45)


# 1. 손실함수 
def loss_fn(X, y) :
    y_pred = X * w + b # y예측치 
    err = y_pred-y # 오차 
    loss = tf.reduce_mean(err**2) # MSE(손실)
    return loss


# 2. 미분계수(기울기) 자동 계산 
def gradient_fn(X, y) :
    with tf.GradientTape() as tape:
        # 1. 손실 
        loss = loss_fn(X, y)  
        # 2. 기울기 계산 : target = 손실(loss), sources=조절변수[w,b]
        gradient = tape.gradient(target=loss, sources=[w, b])
    return gradient # 기울기 


# 3. 최적화 알고리즘 
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) 


# 4. model 학습 
for step in range(5) : # epochs=5   
    # 1. 미분계수 반환
    gradient = gradient_fn(X, y) # w와 b 기울기   
    print('gradient step=', step+1, 'w=',gradient[0].numpy(), 'b=',gradient[1].numpy())

    # 2. 최적화 알고리즘 적용 : 조절변수(w, b) 수정 -> 최적화        
    optimizer.apply_gradients(zip(gradient, [w, b])) 
    print('update step=', step+1, 'w=', w.numpy(), 'b=',b.numpy())
    
    # 3. model 학습상태 : loss 출력 
    print('step =', step+1, 'loss = ', loss_fn(X, y).numpy())

'''
=============================================
w init= [0.7354863] b init= [0.28878012]
=============================================
gradient step= 1 w= [-1.951467] b= [-1.951467]
update step= 1 w= [0.755001] b= [0.3082948]
step = 1 loss =  0.8774147
gradient step= 2 w= [-1.8734083] b= [-1.8734083]
update step= 2 w= [0.7737351] b= [0.32702887]
step = 2 loss =  0.80862534
gradient step= 3 w= [-1.7984719] b= [-1.7984719]
update step= 3 w= [0.7917198] b= [0.3450136]
step = 3 loss =  0.7452292
gradient step= 4 w= [-1.7265332] b= [-1.7265332]
update step= 4 w= [0.8089851] b= [0.3622789]
step = 4 loss =  0.6868033
gradient step= 5 w= [-1.6574719] b= [-1.6574719]
update step= 5 w= [0.82555985] b= [0.37885362]
step = 5 loss =  0.632958
'''









