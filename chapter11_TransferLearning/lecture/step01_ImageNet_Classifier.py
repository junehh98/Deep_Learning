# -*- coding: utf-8 -*-
"""
step01_ImageNet_Classifier.py

딥러닝 이미지넷 분류기 
 - ImageNet으로 학습된 이미지 분류기
"""

# 1. VGGNet(VGG16/VGG19) model 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19 

# 1) model load 
vgg16_model = VGG16(weights='imagenet') 
vgg19_model = VGG19(weights='imagenet') 

# 2) model layer 
vgg16_model.summary()


# 3) model test : 실제 image 적용 
from tensorflow.keras.preprocessing import image # image read 
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# 이미지 로드 
path = 'C:\\ITWILL\\6_DeepLearning\\data\\images\\'
img = image.load_img(path + 'umbrella.jpg', target_size=(224, 224))

X = image.img_to_array(img) # image 데이터 생성 
X = X.reshape(1, 224, 224, 3) # 모양 변경 

# image 전처리 
X = preprocess_input(X)

# image 예측치 
pred = vgg16_model.predict(X)
pred.shape # (1, 1000)

print('predicted :', decode_predictions(pred, top=3))


# 2. ResNet50 model 
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# 1) model load 
resnet50_model = ResNet50(weights='imagenet') 

# 2) model layer 
resnet50_model.summary()


# 이미지 로드 
img = image.load_img(path + 'umbrella.jpg', target_size=(224, 224))

X = image.img_to_array(img) # image 데이터 생성 
X = X.reshape(1, 224, 224, 3)

# image 전처리 
X = preprocess_input(X)

# image 예측치 
pred = resnet50_model.predict(X)
pred.shape # (1, 1000)


print('predicted :', decode_predictions(pred, top=3))


# 3. Inception_v3 model
from tensorflow.keras.applications.inception_v3 import InceptionV3 
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


# 1) model load 
inception_v3_model = InceptionV3(weights='imagenet') 

# 2) model layer 
inception_v3_model.summary()


# 이미지 로드 
img = image.load_img(path + 'Tank.jpeg', target_size=(299, 299))

X = image.img_to_array(img) # image 데이터 생성 
X = X.reshape(1, 299, 299, 3)

# image 전처리 
X = preprocess_input(X)

# image 예측치 
pred = inception_v3_model.predict(X)
pred.shape # (1, 1000)


print('predicted :', decode_predictions(pred, top=3))















