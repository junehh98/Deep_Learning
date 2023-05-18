'''
step05_transform.py


1. Tensor 모양변경  
 - tf.transpose : 전치행렬 
 - tf.reshape : 모양 변경 
'''

import tensorflow as tf

x = tf.random.normal([2, 3]) # 정규분포 난수 생성 
print(x)

xt = tf.transpose(x)
print(xt)

x_r = tf.reshape(tensor=x, shape=[1, 6]) # (tensor, shape)
print(x_r)


'''
2. squeeze
 - 차원의 size가 1인 경우 제거
'''

t = tf.zeros( (1,2,1,3) )
t.shape # [1, 2, 1, 3]

print(tf.squeeze(t)) # shape=(2, 3)

print(tf.squeeze(t).shape) # (2, 3)

print(tf.squeeze(t).get_shape()) # (2, 3)


'''
3. expand_dims
 - tensor에 축 단위로 차원을 추가하는 함수 
'''

const = tf.constant([1,2,3,4,5]) # 1차원 

print(const)
print(const.shape) # (5,)

d0 = tf.expand_dims(const, axis=0) # 행축 2차원 
print(d0) 
    
d1 = tf.expand_dims(const, axis=1) # 열축 2차원 
print(d1)
