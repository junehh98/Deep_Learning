'''
문2) 다음과 같이 다중선형회귀방정식으로 모델의 예측치와 오차를 이용하여 
     손실함수를 정의하고 결과를 출력하시오.

    <조건1> w변수[가중치] : Variable()이용 표준정규분포 난수 상수 4개
    <조건2> b변수[편향] : Variable()이용 표준정규분포 난수 상수 1개       
    <조건3> model 예측치 : pred_Y = (X * a) + b -> 행렬곱 함수 적용  
    <조건4> model 손실함수 출력 
        -> 손실함수는 python 함수로 정의 : 함수명 -> loss_fn(err)
    <조건5> 결과 출력 : << 출력 예시 >> 참고     

<< 출력 예시 >>    
w[가중치] =
[[-0.8777014 ]
 [-2.0691    ]
 [-0.47918326]
 [ 1.5532079 ]]
b[편향] = [1.4863125]
Y[정답] = 1.5
pred_Y[예측치] = [[0.7273823]]
loss function = 0.59693813 
'''

import tensorflow as tf 

# 1. X,Y 변수 정의 
X = tf.constant([[1.2, 2.2, 3.5, 4.2]]) # [1,4] - 입력수 : 4개 
y = tf.constant(1.5) # 출력수(정답) - 1개  


# 2. 가중치(w), 편향(b) 변수 정의 
tf.random.set_seed(1)
w = tf.Variable(tf.random.normal([4, 1])) # [내용 채우기] : 초기값 지정 
b = tf.Variable(tf.random.normal([1])) # [내용 채우기] : 초기값 지정 


# 3. model 예측치/오차/비용
y_pred = tf.matmul(X, w) + b # 예측치
err = tf.subtract(y, y_pred) # 오차

def linear_model(X) : 
    global w, b
    # y = (X @ w) + b
    y_pred = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return y_pred




def model_err(X, y):
    y_pred = linear_model(X)
    err = tf.math.subtract(y, y_pred)
    return err



# 4. 손실함수 정의 : MSE 반환 
def loss_fn(err) :
    err = model_err(X,y)
    loss = tf.reduce_mean(tf.square(err))    
    return loss


loss = loss_fn(err)

# 5. 결과 출력 : 출력 예시 참고 
print('w[가중치] =', w.numpy())
print('b[편향] =', b.numpy())
print('Y[정답] = ', y.numpy())
print('pred_Y[예측치] = ', y_pred.numpy())
print('loss function = ', loss.numpy())
'''
w[가중치] = [[-1.1012203 ]
 [ 1.5457517 ]
 [ 0.383644  ]
 [-0.87965786]]
b[편향] = [0.40308788]
Y[정답] =  1.5
pred_Y[예측치] =  [[0.1304687]]
loss function =  1.875616
'''