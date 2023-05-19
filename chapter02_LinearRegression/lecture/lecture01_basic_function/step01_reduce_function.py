import tensorflow as tf # ver2.x

##############################
## 차원축소 통계/수학 함수
##############################
'''
 tf.reduce_sum(input_tensor, axis) : 지정한 차원을 대상으로 원소들 덧셈
 tf.reduce_mean(input_tensor, axis) : 지정한 차원을 대상으로 원소들 평균
 tf.reduce_prod(input_tensor, axis) : 지정한 차원을 대상으로 원소들 곱셈
 tf.reduce_min(input_tensor, axis) : 지정한 차원을 대상으로 최솟값 계산
 tf.reduce_max(input_tensor, axis) : 지정한 차원을 대상으로 최댓값 계산

 tf.reduce_all(input_tensor) : tensor 원소가 전부 True -> True 반환
 tf.reduce_any(input_tensro) : tensor 원소가 하나라도 True -> True 반환  
'''
data = [[1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]
data
print(tf.reduce_sum(data, axis=0)) # 행축 합계:열 단위 합계 
print(tf.reduce_sum(data, axis=1)) # 열축 합계:행 단위 합계  
print(tf.reduce_sum(data)) # 모든값 합 -> 15.0


# 전체 data 연산 
print(tf.reduce_mean(data)) # 전체 data = 2.5  
print(tf.reduce_max(data)) # 3.5 
print(tf.reduce_min(data)) # 1.5

bool_data = [[True, True], [False, False]] 
print(tf.reduce_all(bool_data)) # False
print(tf.reduce_any(bool_data)) # True


