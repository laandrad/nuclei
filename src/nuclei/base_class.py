import numpy as np


class Nucleus:
    def __init__(self, n: int = 5):
        self.n = n
        self.temp = 1
        self.nucleus = np.zeros((n, n))

    def train(self, x: np.array, y: int):
        pred = self.inference(x)
        if not int(pred) == y:
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
        else:
            self.temp *= 0.95

    def _jitter(self, a: np.array):
        jitter = np.random.random(size=self.n) * self.temp
        return a + jitter

    def inference(self, x: np.array):
        self._fill_diag(x)
        return self._inference()

    def _fill_diag(self, x: np.array):
        return np.fill_diagonal(self.nucleus, x)

    def _inference(self):
        return np.linalg.det(self.nucleus)

    def __repr__(self):
        return str(self.nucleus)

if __name__ == '__main__':
    a = Nucleus(3)
    print(a.inference(range(1, 4)))
    print(a)
