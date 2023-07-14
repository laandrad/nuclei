from abc import ABC


class Activation(ABC):
    def __init__(self):
        pass

    def fit(self, x):
        pass


class Linear(Activation):
    def __init__(self):
        super(Linear, self).__init__()

    def fit(self, x):
        return x


class Binary(Activation):
    def __init__(self):
        super(Binary, self).__init__()

    def fit(self, x):
        return 1 if x > 0 else 0


class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__()

    def fit(self, x):
        return x if x > 0 else 0
