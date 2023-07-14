import numpy as np
from abc import ABC
from src.nuclei.activations import Activation, Binary


rng = np.random.default_rng(12345)


class Nucleus(ABC):
    def __init__(self, n, lr, activation):
        self.n = n
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.n, self.n))
        self.feat_ids = list(range(1, self.n, 2))
        self.biases_ids = list(range(0, self.n, 2))

    def project(self, stimulus):
        pass


class BaseNucleus(Nucleus):
    """Creates a nucleus object. A nucleus takes stimulus of size `input_size` and can project it through
    producing a scalar.
    Arguments:
        n: The input size.
        lr: The learning rate.
        activation: Whether to apply a transformation to output.
    """
    def __init__(self, n: int, lr: float = 1e-5, activation: Activation() = Binary()):
        super(BaseNucleus, self).__init__(n, lr, activation)
        assert n > 1, 'Size of stimulus needs to be > 1.'
        self.n = n * 2
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.n, self.n))
        self.feat_ids = list(range(1, self.n, 2))
        self.biases_ids = list(range(0, self.n, 2))

    def project(self, stimulus: np.array) -> float:
        """Projects the stimulus through the nucleus and returns a spike (a negative or positive signal)."""
        self._fill_diag(stimulus)
        return self._project(self.nucleus, self.activation)

    def _fill_diag(self, stimulus: np.array):
        """Passes the simulus values to the nucleus. Includes bias term as the beginning of stimulus."""
        for i, f in zip(self.feat_ids, stimulus):
            self.nucleus[i, i] = f

    @staticmethod
    def _project(nucleus: np.array, activation: Activation) -> float:
        """Converts the nucleus weights into a scalar."""
        det = get_determinant(nucleus)
        return activation.fit(det)

    def __repr__(self):
        return str(self.nucleus)


def get_determinant(a, slog=True):
    if slog:
        sign, logabsdet = np.linalg.slogdet(a)
        return sign * np.exp(logabsdet)
    else:
        return np.linalg.det(a)
