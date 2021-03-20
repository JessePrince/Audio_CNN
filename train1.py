#这里开始
#导入轮子-----------------------------------
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import cv2
#import matplotlib.pyplot as plt 
import os
from tensorflow.keras import layers
#-------------------------------------------

#读取图片------------------------------------
imgs = []
file_pathname = 'Audio_Spectrum'
lists = os.listdir(file_pathname)
lists.sort(key=lambda x:int(x[:-4]))
for filename in lists:
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        img = cv2.resize(img, (256,256),interpolation=cv2.INTER_CUBIC)
        img[img>=255]=255
        #img = img/255
        imgs.append(img)

#-------------------------------------------

#将其转换为四维数组-----------------------------
train_data=np.array(imgs)
train_data.shape
train_data=train_data
#--------------------------------------------

#读取标签--------------------------------------
labels = np.random.randn(3332)
labels = labels * 20
labels[labels<0]=0
lables1 = pd.DataFrame(labels)
lables1.to_csv('modeldata.csv')
#labels = np.array(pd.read_csv('C:\\Users\\LIGHT\\Desktop\\tesdata\\label.csv'))
#---------------------------------------------

#建立模型---------------------------------------
model = tf.keras.Sequential()

model.add(layers.Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)))
model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(layers.Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))

# model.add(layers.Conv2D(64,kernel_size=(2,2),activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
# #model.add(layers.Dropout(0.2))

# model.add(layers.Conv2D(64,kernel_size=(2,2),activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(256))
model.add(layers.Activation('relu'))

model.add(layers.Dense(128))
model.add(layers.Activation('relu'))

model.add(layers.Dense(1))

exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.01, decay_steps=10, decay_rate=0.96)
# 2. 送入优化器
optimizer = tf.keras.optimizers.Adam(exponential_decay)

model.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
model.summary()
#---------------------------------------------------------------------

train_x=train_data
train_y=labels

model.fit(train_x,train_y,batch_size=20,epochs=100,verbose=1,validation_split=0.2)
model.save('model3.h5')