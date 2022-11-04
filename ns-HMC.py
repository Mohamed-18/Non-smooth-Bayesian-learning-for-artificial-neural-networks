"""
@author: Mohamed Fakhfakh
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from keras.utils.generic_utils import get_custom_objects
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from math import exp
from numpy import linalg as LA
from tensorflow.keras.utils import to_categorical


data = ...
labels = ...

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=10)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


model = Sequential()
...


### ns-HMC ###

epochs = 1
loss_fn = tf.keras.losses.binary_crossentropy
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    with tf.GradientTape() as tape:
        logits = model(x_train, training=True)
        loss_value = loss_fn(y_train, logits)
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
    s += LA.norm(xstar)**2 / (2*varp**2)
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
    mse = tf.keras.losses.MeanSquaredError()
    losses_test = mse(teYY, model.predict(teXX)).numpy()
    return pred_test, losses_test


concat_z=[]
Leap = 10
varp = 1
epsilon = 0.1
Nl = model.count_params() # Number of weights
y = y_train  # Ground truth
y_hat_old = y
zOld = concat_z
pOld = concat_z
acceptation = 0
Lambda = 1
Sigma2 = 1

for i in range(len(gradient)):
  z = tf.zeros(shape=gradient[i].shape)
  concat_z.append(np.array(z))

pr0 = ComputeProx(Lambda/Sigma2,gradient,concat_z)
Nmcmc = 300
for ni in range(1, Nmcmc + 1):
    zStar, pStar = nsHMC_PlugPlay(zOld, varp, epsilon, pr0, concat_z, Leap)

    y_hat_star = Compute_y_hat(zStar)

    K0 = Compute_K(pOld, varp)
    KStar = Compute_K(pStar, varp)
    LikelihoodRatio = Compute_E_theta(y_hat_star,y,zStar,Lambda/Sigma2) - Compute_E_theta(y_hat_old,y,zOld,Lambda/Sigma2)

    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_train,y_hat_star).numpy()
    acc = accracy(y_hat_star,y_train)
    predtest, losstest = prediction_test(zStar,x_test,y_test)
    alpha = min(1, exp(K0 - KStar + LikelihoodRatio))
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

