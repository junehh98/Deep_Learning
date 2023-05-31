"""
step02_gradientTape_dnn_model.py

Tensorflow 2.x 전문가용 DNN model 구축 
 - tensorflow2.x 저수준 API 이용
 - Dataset 이용 : 학습에 필요한 데이터셋 생성(batch size 적용)
 - 순방향 step : Model 클래스 이용 model layer 구축
 - 역방향 step : GradientTape 클래스 이용 model 최적화 
"""

import tensorflow as tf
from tensorflow.python.data import Dataset # dataset 생성 
from tensorflow.keras.layers import Dense, Flatten # layer 구축 
from tensorflow.keras import datasets # MNIST dataset 
from tensorflow.keras import optimizers, metrics # 최적화, 평가 
from tensorflow.keras.utils import to_categorical # Y변수 : encoding


# 1. dataset load & preprocessing 
mnist = datasets.mnist
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train.shape # (60000, 28, 28)
y_train.shape # (60000,) 

# x변수 정규화 : 0 ~ 1
x_train, x_val = x_train / 255.0, x_val / 255.0

# y변수 : 10진수 사용(ont-hot encoding) 
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# 2. Dataset object 생성 : 32 image 묶음 공급 
# train_ds(images, labels) 
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32) 

# test_ds(images, labels)
val_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(100)


# 3. model 정의 : 순방향 
class Model(tf.keras.Model): 
  def __init__(self):
    super().__init__()         
    tf.random.set_seed(34) # w, b 시드값 지정 
    self.f = Flatten() 
    self.d1 = Dense(128, activation='relu') # hidden layer1
    self.d2 = Dense(64, activation='relu') # hidden layer2
    self.d3 = Dense(10, activation='softmax') # output layer

  def call(self, X):         
    x = self.f(X)      
    x = self.d1(x)    
    x = self.d2(x)
    x = self.d3(x)
    return x 


# 4. model & optimizer
model = Model()
optimizer = optimizers.Adam() # keras optimizer


# 5. model 평가방법 정의 : 1epoch -> loss and accuracy 측정  
train_loss = metrics.Mean() # 1epoch loss -> 평균  
train_acc = metrics.CategoricalAccuracy() # 1epoch accuracy
  

# 6. loss function
def loss_fn(X, y): # 입력, 정답 
    global model # model 객체 
    err = model(X) - y # 예측치  
    y_pred = tf.nn.softmax(err) # 활성함수 
    loss = -tf.reduce_mean(y*tf.math.log(y_pred) + (1-y)*tf.math.log(1-y_pred)) 
    return loss # 손실 반환 


# 7. 미분계수 자동 계산 : 역방향 
@tf.function # 함수 장식자 이용 : compile 처리속도 향상 
def model_train(X, y): 
    global model, optimizer # 최적화객체 
    
    with tf.GradientTape() as tape : # 매번 new object 생성    
        # 1) 손실값       
        loss = loss_fn(X, y) 
        # 2) 역방향 : 접선의 기울기 계산 
        gradient = tape.gradient(loss, model.trainable_variables)   
 
    # 3) 모델 최적화 
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    # 4) 1epoch 단위 손실과 분류정확도 저장
    train_loss(loss) # 1epoch loss 평균
    train_acc(y, model(X)) # 1epoch accuracy 


# 8. model 학습 
for epoch in range(10): # epochs = 10    
    train_loss.reset_states()
    train_acc.reset_states()  
    
    for X, y in train_ds: # 훈련셋 공급 
        model_train(X, y) # 순방향 -> 역방향 

    form = 'Epoch {0}, Train loss: {1:.6f}, Train acc: {2:.6f}'
    print(form.format(epoch+1, train_loss.result(), train_acc.result()))


  