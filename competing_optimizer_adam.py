"""
created on Nov 11 2022

@author: Mohamed Fakhfakh
"""

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical


# Load data

# Exemple : COVID‑19 classification using CT images (COVID / NonCovid)
# https://www.kaggle.com/datasets/luisblanche/covidct

data=[]
labels=[]
batch_size = 64

for dir in ["/CT_NonCOVID/"]:
  for file in os.listdir(dir):
      print("time t",dir+file)
      image=cv2.imread(dir+file)
      image = cv2.resize(image, (200, 200))
      imag = img_to_array(image)
      data.append(imag)
      label = 0
      labels.append(label)

print("-----next-----")

for dir in ["/CT_COVID/"]:
    for file in os.listdir(dir):
      print("time t",dir+file)
      image=cv2.imread(dir+file)
      image = cv2.resize(image, (200, 200))
      imag = img_to_array(image)
      data.append(imag)
      label = 1
      labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=10)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)




# Model
input_shape=(200, 200, 3)
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.40))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(2, activation='sigmoid'))


model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=300,
          batch_size=64,
          validation_data=(x_test, y_test))
