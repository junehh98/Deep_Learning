'''
Hyper parameter : 사용자가 지정하는 파라미터
 - learning rate : model 학습율(0.9 ~ 0.0001)
 - iteration size : model 반복학습 횟수(epoch)
 - batch size : model 공급 데이터 크기  
'''

import matplotlib.pyplot as plt
import tensorflow as tf # ver2.0
from sklearn.datasets import load_iris


iris = load_iris() # 0-1에 근사한 변수 선택
X = iris.data
y_data = X[:, 2] # 꽃잎 길이(3)
x_data = X[:, 3] # 꽃잎 넓이(4)


# Hyper parameter
learning_rate = 0.001 # 학습율 
iter_size = 100 # 학습횟수 
'''
1차 테스트 : lr = 0.001, iter size = 100 -> 안정적인 형태   
'''

X = tf.constant(y_data, dtype=tf.float32) 
Y = tf.constant(x_data, dtype=tf.float32) 
X.dtype # tf.float32

tf.random.set_seed(123)
a = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))
a.dtype # tf.float32


# 4. 회귀모델 
def linear_model(X) : # 입력 X
    y_pred = tf.multiply(X, a) + b # y_pred = X * a + b
    return y_pred

# 5. 비용 함수 정의 
def loss_fn_l1() : # MAE : L1 loss function : Lasso 회귀  
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.abs(err)) 
    return loss

def loss_fn_l2() : # MSE : L2 loss function : Lidge 회귀  
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.square(err)) 
    return loss

# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체  
optimizer = tf.optimizers.SGD(lr = learning_rate) 


loss_l1_val = [] # L1 cost value
loss_l2_val = [] # L2 cost value


# 7. 반복학습 : 100회 
for step in range(iter_size) : 
    # 오차제곱평균 최적화 : 손실값 최소화 -> [a, b] 갱신(update)
    optimizer.minimize(loss_fn_l1, var_list=[a, b])#(손실값, 수정 대상)
    optimizer.minimize(loss_fn_l2, var_list=[a, b])#(손실값, 수정 대상)
    
    # loss value save
    loss_l1_val.append(loss_fn_l1().numpy())
    loss_l2_val.append(loss_fn_l2().numpy())
    
       
       
##### 최적화된 model(L1 vs L2) 비교 #####
''' loss values '''
print('loss values')
print('L1 =', loss_l1_val[-5:])
print('L2 =', loss_l2_val[-5:])

'''L1,L2 loss, learning rate, iteration '''
plt.plot(loss_l1_val, '-', label='loss L1')
plt.plot(loss_l2_val, '--', label='loss L2')
plt.title('L1 loss vs L2 loss')
plt.xlabel('Generation')
plt.ylabel('Loss values')
plt.legend(loc='best')
plt.show()

