'''
index 리턴 
  1. argmin/argmax
   - 최소/최대 값의 index 반환 
  2. argsort
   - 정렬 후 index 반환
'''
import tensorflow as tf # ver2.3

a = tf.constant([5,2,1,4,3], dtype=tf.int32) # 1차원 
b = tf.constant([4,5,2,3,1]) # 1차원 
c = tf.constant([[5,4,2], [3,2,4]]) # 2차원 



# 1. argmin/argmax : 최솟값/최댓값 색인(index)반환 
# 1차원 : argmin/argmax(input) 
print(tf.argmin(a).numpy()) # 2
print(tf.argmax(b).numpy()) # 1


# 2차원 : argmin/argmax(input, axis=0) 
print(tf.argmin(c, axis=0).numpy()) # 행축(열 단위)
print(tf.argmin(c, axis=1).numpy()) # 열축(행 단위)

print(tf.argmax(c, axis=0).numpy()) # 행축(열 단위)
print(tf.argmax(c, axis=1).numpy()) # 열축(행 단위)



# 2. argsort : 오름차순정렬 후 색인(index)반환 
# 형식) tf.argsort(values, direction='ASCENDING')
print(tf.argsort(a).numpy()) # [2 1 4 3 0]
print(tf.argsort(b).numpy()) # [4 2 3 0 1]


# 내림차순정렬 -> 색인 반환 
print(tf.argsort(a, direction='DESCENDING').numpy()) # [0 3 4 1 2]









