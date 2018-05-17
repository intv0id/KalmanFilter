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

