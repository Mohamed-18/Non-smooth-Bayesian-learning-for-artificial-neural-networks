from numpy import linalg as LA
import tensorflow as tf
import numpy as np



def utils_fn(model):

    def nsHMC_PlugPlay(xold, varp, epsilon, P0, X0, Leap):
        xStar = xold
        rdm = []
        fin = 0
        nb_param = np.random.randn(model.count_params()) * varp
        for s in range(len(P0)):
            st = tf.keras.backend.flatten(P0[s])
            f = len(st)
            fin = fin + f
            deb = fin - f
            rdm.append(tf.keras.backend.constant(nb_param[deb:fin], shape=P0[s].shape))
        pold = np.array(rdm)
        pStar = pold - epsilon/2 * (2 * np.array(xold) - X0 - P0)
        for jL in range(1, Leap):
            xStar = xStar + epsilon*pStar
            if (jL != Leap):
                pStar = pStar - epsilon/2 * (2*xStar - X0 - P0)

        pStar = pStar - epsilon/2 * (2*xStar - X0 - P0)
        pStar = -pStar
        return xStar, pStar
            # pStar = pold - epsilon / 2 * (2 * np.array(xold) - X0 - P0)
            # for jL in range(1, Leap - 1):
            #     xStar = xStar + epsilon * pStar
            #     pStar = pStar - epsilon / 2 * (2 * xStar - X0 - P0)
            # return xStar, pStar

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
          somme += (LA.norm(y_hat-y)**2 + (LA.norm(step_grad,1)))*alpha/2
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
        mae = tf.keras.losses.MeanSquaredLogarithmicError()
        losses_test = mae(teYY, model.predict(teXX)).numpy()
        return pred_test, losses_test

    return nsHMC_PlugPlay,Compute_K,Compute_E_theta,ComputeProx,Compute_y_hat,accracy,prediction_test


