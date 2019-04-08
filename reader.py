import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def init():

   x = []
   y = []
   endereco_treino = 'originais_com_filtro\\1.0_0.5_0.2'
   endereco_teste = 'US_padrao_ouro_original'

   for arquivo in os.listdir(endereco_treino):
      imagem = cv2.imread(endereco_treino+'\\'+arquivo)
      imagem = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
      imagem = cv2.resize(imagem,(100,100))
      imagem = imagem.reshape(imagem.shape[0],imagem.shape[1],1)
      x.append(imagem)

   for arquivo in os.listdir(endereco_teste):
      imagem = cv2.imread(endereco_teste+'\\'+arquivo)
      imagem = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
      imagem = cv2.resize(imagem,(100,100))
      imagem = imagem.reshape(imagem.shape[0],imagem.shape[1],1)
      y.append(imagem)
   
   x = np.array(x).reshape(-1,imagem.shape[0],imagem.shape[1],imagem.shape[2])
   y = np.array(y).reshape(-1,imagem.shape[0],imagem.shape[1],imagem.shape[2])
   x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)

   return x_train,x_test,y_train,y_test
