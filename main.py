"""
@author: Mohamed Fakhfakh
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, GRU, Reshape, ZeroPadding2D, Add, AveragePooling2D, MaxPool2D,GlobalMaxPool2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from math import exp
from numpy import linalg as LA
from tensorflow.keras.utils import to_categorical
import time

data=[]
labels=[]
i=0


for dir in ["/home/dell/Bureau/Thèse INP/datasets/covid-xray/50/COVID/"]:
  for file in os.listdir(dir):
      print("time t",dir+file)
      image=cv2.imread(dir+file)
      image = cv2.resize(image, (200, 200))
      imag = img_to_array(image)
      data.append(imag)
      i=i+1
      label = 0
      labels.append(label)

print("-----next-----")
i1=0
for dir in ["/home/dell/Bureau/Thèse INP/datasets/covid-xray/50/Normal/"]:
    for file in os.listdir(dir):
      print("time t",dir+file)
      image=cv2.imread(dir+file)
      image = cv2.resize(image, (200, 200))
      imag = img_to_array(image)
      data.append(imag)
      i1=i1+1
      label = 1
      labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print(len(data))

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=20)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)



input_shape=(200, 200, 3)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
print(model.summary())



### ns-HMC ###

epochs = 1
loss_fn = tf.keras.losses.binary_crossentropy
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    # Iterate over the batches of the dataset.
    # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = model(x_train, training=True)  # Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y_train, logits)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    gradient = tape.gradient(loss_value, model.trainable_weights)


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

def Compute_K(xStar, varp):
  s=0
  for i in range(len(xStar)):
    xstar=tf.keras.backend.flatten(xStar[i])
    s += LA.norm(xstar)**2 / LA.norm(xstar)**2  #(2*varp**2)
  return s

def Compute_E_theta(y_hat,y,W,alpha):
    somme=0
    for i in range(len(W)):
      step_grad=tf.keras.backend.flatten(W[i])
      # somme+=( ( (alpha / 2) * (LA.norm(y_hat - y) ** 2)) + (LA.norm(step_grad, 1) / LA.norm(step_grad, 1)))
      somme+=(-(LA.norm(y_hat-y)**2 + (LA.norm(step_grad,1)/LA.norm(step_grad,1))*alpha/2))

    return somme

def SoftThresholding(ProxValue,x,alpha):
    tab=[]
    for i in range(len(x)):
      step_grad=tf.keras.backend.flatten(x[i])
      tab.append(tf.sign(np.array(step_grad))*tf.math.maximum(abs(np.array(step_grad))-alpha,0))
    for i in range(len(ProxValue)):
      ProxValue[i]=tf.constant(tab[i], shape=(ProxValue[i].shape))
    return ProxValue

def ComputeProx(alpha, grad, z):
    z2 = z - (alpha / 2) * np.array(grad)
    Prox = SoftThresholding(grad, z2, 1e-05)
    return Prox

def Compute_y_hat(zStar):
    model.set_weights(zStar)
    y_hat_star = model.predict(x_train)
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
    losses_test = LA.norm(model.predict(teXX) - teYY)**2 / len(teXX)
    return pred_test, losses_test

concat_z=[]
mean_weights=[]
w=[]
Leap = 10
varp = 1
epsilon = 0.1
Nl = model.count_params()

y = y_train
y_hat_old = y

zOld = concat_z
pOld = concat_z

acceptation = 0
Lambda = 1
Sigma2 = 1

for i in range(len(gradient)):
  z = tf.zeros(shape=gradient[i].shape)
  concat_z.append(np.array(z))
  w.append(np.array(z))
  mean_weights.append(np.array(z))

pr0 = ComputeProx(Lambda/Sigma2,gradient,concat_z)
print(pr0)

start_time = time.time()

Nmcmc = 500
mean_accuracy = []
losses = []
y_hat = []
compter = []
pred_test = []
loss_test = []
cpt = 0
start_time = time.time()
for ni in range(1, Nmcmc + 1):
    print(ni)
    zStar, pStar = nsHMC_PlugPlay(zOld, varp, epsilon, pr0, concat_z, Leap)
    y_hat_star = Compute_y_hat(zStar)
    y_hat.append(y_hat_star)
    K0 = Compute_K(pOld, varp)
    KStar = Compute_K(pStar, varp)
    #LikelihoodRatio = Compute_E_theta(y_hat_old, y, zOld, Lambda / Sigma2) - Compute_E_theta(y_hat_star, y, zStar, Lambda / Sigma2)
    LikelihoodRatio = Compute_E_theta(y_hat_star,y,zStar,Lambda/Sigma2) - Compute_E_theta(y_hat_old,y,zOld,Lambda/Sigma2)
    loss = LA.norm(y_hat_star - y_train) **2/ len(y_train)
    alpha = min(1, exp(K0 - KStar + LikelihoodRatio))
    acc = accracy(y_hat_star, y_train)
    predtest, losstest = prediction_test(zStar, x_test, y_test)
    u = np.random.uniform(0, 1)
    if u < alpha:
        acceptation += 1
        zOld = zStar
        pOld = pStar
        loss_old = loss
        y_hat_old = y_hat_star
        acc_old = acc
        pred_test_old = predtest
        loss_test_old = losstest

        print(acceptation)
        print("K0", K0)
        print("KStar", KStar)
        print("LikelihoodRatio", LikelihoodRatio)
        print("alpha", alpha)
        print("loss", loss_old)
        print("acc", acc_old)
        print("loss test", loss_test_old)
        print("acc test", pred_test_old)
        print("*****")

    losses.append(loss_old)
    mean_accuracy.append(np.round(acc_old, 2))
    loss_test.append(loss_test_old)
    pred_test.append(np.round(pred_test_old, 2))

    if ni > (Nmcmc / 2):
        cpt += 1
        mean_weights += zOld

# z = mean_weights / cpt
# model.set_weights(z)
temps_exec = time.time() - start_time
print("\n Nombre de secondes mis par le programme RMSprop: %.3f secondes" % temps_exec)