'''
선형대수 연산 함수  
  단위행렬 -> tf.linalg.eye(dim) 
  대각행렬 -> tf.linalg.diag(x)  
  정방행렬의 행렬식 -> tf.linalg.det(x)
  정방행렬의 역행렬 -> tf.linalg.inv(x)
  두 텐서의 행렬곱 -> tf.linalg.matmul(x, y)
'''

import tensorflow as tf
import numpy as np
dir(tf.linalg)


# 정방행렬 데이터 생성 
x = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 
y = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 
x
'''
array([[0.79723736, 0.17178352],
       [0.61795433, 0.45396645]])
'''

eye = tf.linalg.eye(2) # 단위행렬
print(eye.numpy()) 
 
# 단위행렬 : one-hot-encoding(2진수)
'''
array([[1., 0.], - 'cat'
       [0., 1.]],- 'dog' dtype=float32)
'''

dia = tf.linalg.diag(x) # 대각행렬 
mat_deter = tf.linalg.det(x) # 정방행렬의 행렬식  
mat_inver = tf.linalg.inv(x) # 정방행렬의 역행렬
mat = tf.linalg.matmul(x, y) # 행렬곱 반환 
x.shape # (2, 2)
y.shape # (2, 2)
mat.shape # TensorShape([2, 2])


print(x)
print(dia.numpy()) 
print(mat_deter.numpy())
print(mat_inver.numpy())
print(mat.numpy())


## 행렬곱 
A = tf.constant([[1,2,3], [3,4,2], [3,2,5]]) # A행렬 
B = tf.constant([[15,3, 5], [3, 4, 2]]) # B행렬  

A.get_shape() # [3, 3]
B.get_shape() # [2, 3]

# 행렬곱 연산 
mat_mul = tf.linalg.matmul(a=A, b=B)
print(mat_mul.numpy())
'''
수 일치를 시키고 행렬곱 실행해야함
error : incompatible: In[0]: [3,3], In[1]: [2,3] [Op:MatMul]
'''

'''
 해결방안
 1. matmul (a=b, b=A) 
 2. A를 (3,2)형식으로 변경
'''
# 1번 해결방안
mat_mul2 = tf.linalg.matmul(a=B, b=A)
mat_mul2 
mat_mul2 .shape # TensorShape([2, 3])


# 2번 해결방안(전치행렬)
BB = tf.transpose(B)
BB.shape # mat_mul2 = tf.linalg.matmul(a=B, b=A)
mat_mul3 = tf.linalg.matmul(a=A, b=BB)
mat_mul3.shape # TensorShape([3, 2])
'''
[[36, 17],
 [67, 29],
 [76, 27]]
'''













