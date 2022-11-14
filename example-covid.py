"""
@author: Mohamed Fakhfakh
"""


from math import exp
from numpy import linalg as LA
from tensorflow.keras.utils import to_categorical
from utils import *


Leap = 10
varp = 1
epsilon = 0.1

Nl = model.count_params()  # Number of weights
loss_fn = tf.keras.losses.MeanSquaredError()

Lambda = 1
Sigma2 = 1


epochs = 1
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        print("Step", step)
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)

        gradient = tape.gradient(loss_value, model.trainable_weights)

        # ns-HMC algorithm
        concat_z = []
        grads = []
        y = y_batch_train  # Ground truth
        y_hat_old = y_batch_train
        zOld = concat_z
        pOld = concat_z

        for i in range(len(gradient)):
            z = tf.zeros(shape=gradient[i].shape)
            concat_z.append(np.array(z))
            grads.append(np.array(z))

        acceptation = 0
        pr0 = ComputeProx(Lambda / Sigma2, gradient, concat_z)

        pred_test = []
        loss_test = []

        Nmcmc = 300
        for ni in range(1, Nmcmc + 1):
            print("ni", ni)
            zStar = 0
            pStar = 0
            zStar, pStar = nsHMC_PlugPlay(zOld, varp, epsilon, pr0, concat_z, Leap)

            y_hat_star = Compute_y_hat(zStar, x_batch_train)

            K0 = Compute_K(pOld)
            KStar = Compute_K(pStar)
            LikelihoodRatio = Compute_E_theta(y_hat_old, y_batch_train, zOld, Lambda / Sigma2) - Compute_E_theta(logits,
                                                                                                                 y_batch_train,
                                                                                                                 zStar,
                                                                                                                 Lambda / Sigma2)
            mse = tf.keras.losses.MeanSquaredError()
            accuracy_train = accracy(logits, y_batch_train)
            loss_train = mse(y_batch_train, logits).numpy()
            predtest, losstest = prediction_test(zStar, x_test, y_test)

            alpha = min(1, exp(K0 - KStar + LikelihoodRatio))

            u = np.random.uniform(0, 1)

            if u < alpha:
                acceptation += 1
                zOld = zStar
                pOld = pStar
                y_hat_old = y_hat_star
                loss_old = loss_train
                acc_old = np.array(accuracy_train)
                pred_test_old = predtest
                loss_test_old = losstest

                print("loss train", loss_old)
                print("accuracy train", acc_old)
                print("loss test", loss_test_old)
                print("accuracy test", pred_test_old)
                print("**********")

