from abc import ABC
import numpy as np


class Loss(ABC):
    def __init__(self):
        pass

    def fit(self, y, y_hat):
        pass


class MAE(Loss):
    """Mean Absolute Error"""
    def __init__(self):
        super(Loss, self).__init__()

    def fit(self, y, y_hat):
        return np.mean([abs(y_i - pred_i) for y_i, pred_i in zip(y, y_hat)])


class MSE(Loss):
    """Mean Squared Error"""
    def __init__(self):
        super(Loss, self).__init__()

    def fit(self, y, y_hat):
        return np.mean([(y_i - pred_i)**2 for y_i, pred_i in zip(y, y_hat)])


class RMSE(Loss):
    """Root Mean Squared Error"""
    def __init__(self):
        super(Loss, self).__init__()

    def fit(self, y, y_hat):
        return np.sqrt(np.mean([(y_i - pred_i)**2 for y_i, pred_i in zip(y, y_hat)]))

