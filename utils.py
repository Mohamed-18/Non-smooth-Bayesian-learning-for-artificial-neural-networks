import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from numpy import linalg as LA
from tensorflow.keras.utils import to_categorical



data=[]
labels=[]
batch_size = 64

# Exemple : COVID‑19 classification using CT images (COVID / NonCovid)
# https://www.kaggle.com/datasets/luisblanche/covidct

for dir in ["/home/irit/Bureau/Documnts Mohamed/datasets/COVID_Nom/CT_NonCOVID/"]:
  for file in os.listdir(dir):
      print("time t",dir+file)
      image=cv2.imread(dir+file)
      image = cv2.resize(image, (250, 250))
      imag = img_to_array(image)
      data.append(imag)
      label = 0
      labels.append(label)

print("-----next-----")

for dir in ["/home/irit/Bureau/Documnts Mohamed/datasets/COVID_Nom/CT_COVID/"]:
    for file in os.listdir(dir):
      print("time t",dir+file)
      image=cv2.imread(dir+file)
      image = cv2.resize(image, (250, 250))
      imag = img_to_array(image)
      data.append(imag)
      label = 1
      labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=10)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)



input_shape=(250, 250, 3)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))




def nsHMC_PlugPlay(xold, varp, epsilon, P0, X0, Leap):
    rdm = []
    fin = 0
    nb_param = np.random.randn(model.count_params())
    for s in range(len(P0)):
        st = tf.keras.backend.flatten(P0[s])
        f = len(st)
        fin = fin + f
        deb = fin - f
        rdm.append(tf.keras.backend.constant(nb_param[deb:fin], shape=P0[s].shape))
    pold = np.array(rdm) * varp
    pStar = pold - epsilon / 2 * (2 * np.array(xold) - X0 - P0)
    xStar = np.array(xold) + epsilon * pStar
    for jL in range(1, Leap - 1):
        xStar = xStar + epsilon * pStar
        pStar = pStar - epsilon / 2 * (2 * xStar - X0 - P0)
    return xStar, pStar


def Compute_K(xStar):
  s=0
  for i in range(len(xStar)):
    xstar=tf.keras.backend.flatten(xStar[i])
    s += LA.norm(xstar)**2 / LA.norm(xstar)**2
  return s


def Compute_E_theta(y_hat,y,W,alpha):
    somme=0
    for i in range(len(W)):
      step_grad=tf.keras.backend.flatten(W[i])
      somme+=(((alpha/2)*(LA.norm(y_hat-y)**2))+(LA.norm(step_grad,1)/LA.norm(step_grad,1)))
    return -somme


def SoftThresholding(ProxValue,x,alpha):
    tab=[]
    for i in range(len(x)):
      step_grad=tf.keras.backend.flatten(x[i])
      tab.append(tf.sign(np.array(step_grad))*tf.math.maximum(abs(np.array(step_grad))-alpha,0))
    for i in range(len(ProxValue)):
      ProxValue[i]=tf.constant(tab[i], shape=(ProxValue[i].shape))
    return ProxValue


def ComputeProx(alpha, grad, z):
    z2 = z - (alpha/2)*np.array(grad)
    Prox = SoftThresholding(grad, z2, 10e-06)
    return Prox


def Compute_y_hat(zStar, x_batch_train):
    model.set_weights(zStar)
    y_hat_star = model(x_batch_train)
    return y_hat_star


def accracy(y_pred, y_train):
    tab_pred = []
    for i in range(len(y_pred)):
        if (y_train[i][0] == 1):
            tab_pred.append(y_pred[i][0])
        if (y_train[i][1] == 1):
            tab_pred.append(y_pred[i][1])
    mean = 0
    for ni in range(len(tab_pred)):
        mean += tab_pred[ni]
    accuracy = mean / len(tab_pred)
    return accuracy


def prediction_test(zStar, teXX, teYY):
    model.set_weights(zStar)
    pred_test = accracy(model.predict(teXX), teYY)
    mse = tf.keras.losses.MeanSquaredError()
    losses_test = mse(teYY, model.predict(teXX)).numpy()
    return pred_test, losses_test

