'''
step03_tensor_info.py

Tensor 정보 제공 함수 
 1. tensor shape
 2. tensor rank
 3. tensor size
 4. tensor reshape 
'''

import tensorflow as tf
print(tf.__version__) # 2.3.0

scala = tf.constant(1234) # 상수 
vector = tf.constant([1,2,3,4,5]) # 1차원 
matrix = tf.constant([ [1,2,3], [4,5,6] ]) # 2차원
cube = tf.constant([[ [1,2,3], [4,5,6], [7,8,9] ]]) # 3차원 

print(scala)
print(vector)
print(matrix)
print(cube)


# 1. tensor shape 
print('\ntensor shape')
print(scala.get_shape()) # () scalar.shape
print(vector.get_shape()) # (5,)
print(matrix.get_shape()) # (2, 3)
print(cube.get_shape()) # (1, 3, 3)



# 2. tensor rank
print('\ntensor rank')
print(tf.rank(scala)) 
print(tf.rank(vector)) 
print(tf.rank(matrix)) 
print(tf.rank(cube))



# 3. tensor size
print('\ntensor size')
print(tf.size(scala)) 
print(tf.size(vector)) 
print(tf.size(matrix)) 
print(tf.size(cube))
