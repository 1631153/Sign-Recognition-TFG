#https://www.kaggle.com/code/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy/notebook
import numpy as np # linear algebra
import pandas as pd
import os
import tensorflow as tf
import keras
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

train_df=pd.read_csv('D:\\Descargas\\prueba 3 - Kaggle dataset\\sign_mnist_train.csv')
test_df=pd.read_csv('D:\\Descargas\\prueba 3 - Kaggle dataset\\sign_mnist_test.csv')
#train_df.info()
#test_df.info()

# Se usa para extraer las etiquetas de los datos de entrenamiento y prueba (Kaggle)
train_label=train_df['label']
test_label=test_df['label']

# Eliminar la columna de etiquetas de train y test para obtener solo los píxeles de las imágenes
trainset=train_df.drop(['label'],axis=1)
X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
X_test=test_df.drop(['label'],axis=1)

# Se usa para conseguir One-Hot Encoding
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)
X_test=X_test.values.reshape(-1,28,28,1)

# Aumentar datos para generar un buen modelo (Kaggle)
train_datagen = ImageDataGenerator(rescale = 1./255,rotation_range = 0,height_shift_range=0.2,width_shift_range=0.2,shear_range=0,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

X_test=X_test/255

# Crear el modelo (mix entre Kaggle y Detector de emociones)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=2,padding='same'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=1,activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2),2,padding='same'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(2, 2),strides=1,activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2),2,padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Dense(units=24, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_datagen.flow(X_train,y_train,batch_size=200),epochs = 15,validation_data=(X_test,y_test),shuffle=1)
(ls,acc)=model.evaluate(x=X_test,y=y_test)

# Guardar el modelo
#model.save("hand_model.h5")
#model.save_weights("hand_model.weights.h5")
print('MODEL ACCURACY = {}%'.format(acc*100))

