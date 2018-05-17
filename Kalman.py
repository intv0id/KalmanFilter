#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, x_init, P_init, F, H, R, Q):
        """
            x is the state estimate
            P is the covariance estimate
            F is the state transition model which is applied to the previous x state
            H is the observation model which is useful 
                when dealing with multiple sensors, 
                when single sensors this is the identity
            R is the covariance of the observation noise
            Q is the covariance of the process noise
        """
        self.dim, _ = x_init.shape
        self.x, self.xp = x_init, x_init
        self.P, self.Pp = P_init, P_init
        self.F, self.H, self.R, self.Q = F, H, R, Q
    def predict(self):
        # Predicted (a priori) state estimate
        self.xp = self.F @ self.x
        # Predicted (a priori) estimate covariance
        self.Pp = self.F @ self.P @ self.F.transpose() + self.Q
    def update(self, z):
        I = np.eye(self.dim)
        # Innovation residual
        y = z - self.H @ self.xp
        # Innovation covariance
        S = self.R + self.H @ self.Pp @ self.H.transpose()
        # Optimal Kalman gain
        K = self.Pp @ self.H.transpose() @ np.linalg.inv(S)
        # Updated (a posteriori) state estimate
        self.x = self.xp + K @ y
        # Updated (a posteriori) estimate covariance
        self.P = (I - K @ self.H) @ self.Pp @ (I - K @ self.H).transpose() + K @ self.R @ K.transpose()
        # Measurement post-fit residual
        y = z - self.H @ self.x

    def update_cov(self, R, Q):
        self.R, self.Q = R, Q

class signal(object):
    def __init__(self, X, noise):
        self.real = X
        self.noise = noise
        self.noisy = self.real + self.noise

time = np.linspace(0, 10, 1000)
mainSignal = signal(np.cos(np.linspace(0, 10, 1000)), np.random.normal(0, .5, 1000))
derivSignal = signal(np.sin(np.linspace(0, 10, 1000)), np.random.normal(0, .5, 1000))
predicted = {"main":[], "deriv":[]}

kf = KalmanFilter(
    x_init = np.array([[0],[0]]),#Etat estime
    P_init = np.array([0, 0]),#Covariance estimee
    F = np.array([[1,1000/10],[0,1]]),#matrice de passage de l'etat i a i+1
    Q = np.array([[.0001,.01],[.01,.0001]]),#covariance vaximale souhaitee
    R = np.array([[10000,0],[0, 10000]]),#Covariance du bruit
    H = np.eye(2),
)

for i in range(1000):
    X = np.array([[mainSignal.noisy[i]], [derivSignal.noisy[i]]])
    kf.predict()
    kf.update(X)
    predicted["main"].append(kf.x[0, 0])
    predicted["deriv"].append(kf.x[1, 0])

plt.plot(time, mainSignal.noisy)
plt.plot(time, predicted["main"])
plt.show()

