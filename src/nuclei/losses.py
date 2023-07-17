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
        y = np.array(y)
        y_hat = np.array(y_hat)
        return np.mean(abs(y - y_hat))


class MSE(Loss):
    """Mean Squared Error"""

    def __init__(self):
        super(Loss, self).__init__()

    def fit(self, y, y_hat):
        y = np.array(y)
        y_hat = np.array(y_hat)
        return np.mean((y - y_hat) ** 2)


class RMSE(Loss):
    """Root Mean Squared Error"""

    def __init__(self):
        super(Loss, self).__init__()

    def fit(self, y, y_hat):
        y = np.array(y)
        y_hat = np.array(y_hat)
        return np.sqrt(np.mean((y - y_hat) ** 2))


class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def fit(self, y, y_hat):
        y = np.array(y)
        y_hat = np.array(y_hat)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
