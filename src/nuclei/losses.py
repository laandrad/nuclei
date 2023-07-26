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
        small_delta = 1e-10
        y = np.array(y, dtype=float)
        y_hat = np.array(y_hat, dtype=float)
        if y.any() == 0 or y.any() == np.nan:
            y += small_delta
        if y_hat.any() == 0 or y_hat.any() == np.nan:
            y_hat += small_delta
        if y.any() < 0:
            y *= -1
        if y_hat.any() < 0:
            y_hat *= -1
        return -np.nanmean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
