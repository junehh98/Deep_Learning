# -*- coding: utf-8 -*-
"""
tensorflow import test
"""

import tensorflow as tf  
import matplotlib.pyplot as plt
print(tf.__version__) # 2.10.0 : tensorflow버전 


mnist = tf.keras.datasets.mnist # dataset 로드 


train, test = mnist.load_data()
X_train, y_train = train

X_train.shape # (60000, 28, 28) 6만개의 이미지 28 x 28 픽셀
y_train.shape # (60000,)


plt.imshow(X_train[0]) # 28 x 28 픽셀
plt.show()