from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Activation,Input, Conv2DTranspose,UpSampling2D
from keras.models import Model
import cv2
from reader import init
import os
import matplotlib.pyplot as plt


x_train,x_test,y_train,y_test = init()
inputs = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
   
def apply(x):
   x = Conv2D(filters = 32,kernel_size = [3,3],activation = 'elu',padding='same')(x)
   x = MaxPool2D(padding='same')(x)
   
   x = Conv2D(filters = 32,kernel_size = [3,3],activation = 'elu',padding='same')(x)
   
   x = MaxPool2D(padding='same')(x)
   
   
   x = Conv2DTranspose(filters = 1,kernel_size = [4,4],strides = (4,4),activation='relu')(x)
   return x

modelo = Model(inputs,apply(inputs))

modelo.compile('adam','mse')


modelo.fit(x_train,y_train,epochs = 400)


for i,imagem in enumerate(x_test):
    imagem_predicted = modelo.predict(imagem.reshape(1,imagem.shape[0],imagem.shape[1],imagem.shape[2]))    
    plt.imshow(imagem_predicted.reshape(imagem.shape[0],imagem.shape[1]),cmap='gray')
