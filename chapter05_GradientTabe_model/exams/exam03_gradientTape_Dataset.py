# -*- coding: utf-8 -*-
"""
문3) cifar10 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하기 위해서
    단계2. Dataset 생성과 단계3. Model 클래스 내용을 채우시오.
    
  조건1> keras layer
       L1 =  (32, 32, 3) x 256
       L2 =  256 x 128
       L3 =  128 x 64
       L4 =  64 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'Adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 20, batch_size = 50 
  
<츨력 예시>  
Epoch 1, Train Loss: 2.19101, Train Acc: 0.25788, Test Loss: 2.13325, Test Acc: 0.31580
Epoch 2, Train Loss: 2.12259, Train Acc: 0.33052, Test Loss: 2.12327, Test Acc: 0.33180
Epoch 3, Train Loss: 2.10337, Train Acc: 0.35070, Test Loss: 2.10720, Test Acc: 0.34430
Epoch 4, Train Loss: 2.08615, Train Acc: 0.36664, Test Loss: 2.07554, Test Acc: 0.38020
Epoch 5, Train Loss: 2.08072, Train Acc: 0.37330, Test Loss: 2.07430, Test Acc: 0.37800
''' 생략 '''
"""


import tensorflow as tf
from tensorflow.python.data import Dataset # dataset 생성 
from tensorflow.keras.layers import Dense, Flatten # layer 구축 
from tensorflow.keras.datasets.cifar10 import load_data # Cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers,metrics#최적화,평가 

# 단계1. dataset load & preprocessing
print('data loading')
(x_train, y_train), (x_val, y_val) = load_data()

print(x_train.shape) # (50000, 32, 32, 3) : 4차원 
print(y_train.shape) # (50000, 1) : 2차원 

# x_data 전처리 : 0~1 정규화
x_train = x_train / 255.0
x_val = x_val / 255.0

# one-hot encoding 
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# 단계2. Dataset 생성 
# train_ds(images, labels) 
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(50) 

# test_ds(images, labels)
val_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(100)



# 단계3. 순방향 step : model layer 정의  
class Model(tf.keras.Model): # tf.keras Model 상속 
  def __init__(self): # 생성자 
    super().__init__()         
    tf.random.set_seed(34) # w, b 시드값 지정 
    self.f = Flatten() # 2d/3d -> 1d
    self.d1 = Dense(256, activation='relu') # hidden layer1
    self.d2 = Dense(128, activation='relu') # hidden layer2
    self.d3 = Dense(64, activation='relu') # hidden layer3
    self.d4 = Dense(10, activation='softmax') # output layer

  def call(self, X): # call 메서드 : self 메서드          
    x = self.f(X)      
    x = self.d1(x)    
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    return x # y_pred



# 단계4. loss function 
def loss_fn(X, y): # 입력, 정답 
    global model # model 객체 
    y_pred = model(X) # model.call(X) : 예측치       
    loss = -tf.reduce_mean(y*tf.math.log(y_pred) + (1-y)*tf.math.log(1-y_pred)) 
    return loss # 손실 반환

# 단계5. model & optimizer
model = Model()
optimizer = optimizers.Adam(0.001) # lr : 자동설정(lr=0.1)


# 단계6. model test : 1epoch -> train/test loss and accuracy 측정 
train_loss = metrics.Mean() # 전체 원소 -> 평균 객체 반환 
train_acc = metrics.CategoricalAccuracy() # 분류정확도 객체 반환
  

# 단계7. 역방향 step : tf.GradientTape 클래스 이용 
@tf.function # 연산 속도 향상 
def train_step(images, labels): # train step
    with tf.GradientTape() as tape:
        # 1) 순방향    
        preds = model(images, training=True) # 예측치  
        loss_value = loss_fn(images, labels) # 손실값 
        
        # 2) 역방향 
        grads = tape.gradient(loss_value, model.trainable_variables)
        # model optimizer
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # 3) 1epoch model training test : loss, acc save 
        train_loss(loss_value) # 1epoch loss 평균 : object(params) 
        train_acc(labels, preds) # 1epoch accuracy : object(params) 
  

epochs = 20

# 단계8. model training
for epoch in range(epochs) :
    # 초기화   
    train_loss.reset_states()
    train_acc.reset_states()  
    
    for X, y in train_ds: # 훈련셋 공급 
        train_step(X, y) # 순방향 -> 역방향 

    form = 'Epoch {0}, Train loss: {1:.6f}, Train acc: {2:.6f}'
    print(form.format(epoch+1, train_loss.result(), train_acc.result()))
 
#Epoch 20, Train loss: 0.198054, Train acc: 0.547680