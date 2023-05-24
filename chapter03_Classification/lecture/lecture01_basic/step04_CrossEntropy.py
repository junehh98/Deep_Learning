"""
엔트로피(Entropy) 
 - 확률변수 p에 대한 불확실성의 측정 지수 
 - 값이 클 수록 일정한 방향성과 규칙성이 없는 무질서(chaos) 의미
 - Entropy = -sum(p * log(p))
"""

import numpy as np

# 1. 불확실성이 큰 경우(p1: 앞면, p2: 뒷면)
p1 = 0.5; p2 = 0.5

entropy = -(p1 * np.log2(p1) + p2 * np.log2(p2)) 
print('entropy =', entropy) # entropy = 1.0


# 2. 불확실성이 작은 경우(x1: 앞면, x2: 뒷면) 
p1 = 0.9; p2 = 0.1

entropy = -(p1 * np.log2(p1) + p2 * np.log2(p2)) # 공통부호 정리
print('entropy =', entropy) # entropy = 0.4689955935892812



'''
Cross Entropy : 딥러닝 분류모델에서 손실 계산
  - 두 확률변수 x와 y가 있을 때 x를 관찰한 후 y에 대한 불확실성 측정
  - Cross 의미 :  y=1, y=0 일때 서로 교차하여 손실 계산 
  - loss식 = -( y * log(y_pred) + (1-y) * log(1-y_pred))
  
  왼쪽 식 : y * log(y_pred) -> y=1일때 손실값 계산
  오른쪽 식 : (1-y) * log(1 - y_pred) -> y=0일때 손실값 계산
'''

import tensorflow as tf 

y_preds = [0.02, 0.98] # model 예측값[0수렴, 1수렴]

y = 1 # 정답(y)
for y_pred in y_preds:
    loss_val = -( y * tf.math.log(y_pred))
    print(loss_val.numpy())
'''
3.912023 -> y=1 vs y_pred = 0.02
0.020202687  y=1 vs y_pred = 0.98
'''

y = 0
for y_pred in y_preds:
    loss_val = -((1-y) * tf.math.log(1-y_pred))
    print(loss_val.numpy())
'''
0.020202687 : y=0 vs y_pred=0.02
3.912023    : y=0 vs y_pred =0.98
'''



# cross entropy : 손실(loss)
y = 1 # 정답(0 or 1)
for y_pred in y_preds:
    loss_val =-( y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    print(loss_val.numpy())















