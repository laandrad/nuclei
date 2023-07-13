import numpy as np


rng = np.random.default_rng(12345)


class Nucleus:
    """Creates a nucleus object. A nucleus takes stimulus and can either project it or jitter.
    Arguments:
        n: The input size.
        lr: The learning rate.
        linear: Whether to apply nonlinear transformation to output.
    """
    def __init__(self, n: int, lr: float = 1e-5, linear=False):
        assert n > 1, 'Size of stimulus needs to be > 1.'
        self.n = n * 2
        self.lr = lr
        self.linear = linear
        self.nucleus = np.ones((self.n, self.n))
        self.feat_ids = list(range(1, self.n, 2))
        self.biases_ids = list(range(0, self.n, 2))

    def project(self, stimulus: np.array) -> float:
        """Projects the stimulus through the nucleus and returns a spike (a negative or positive signal)."""
        self._fill_diag(stimulus)
        return self._project(self.nucleus, self.linear)

    def _fill_diag(self, stimulus: np.array):
        """Passes the simulus values to the nucleus. Includes bias term as the beginning of stimulus."""
        for i, f in zip(self.feat_ids, stimulus):
            self.nucleus[i, i] = f

    @staticmethod
    def _project(nucleus, linear=False) -> float:
        """Converts the nucleus weights into a scalar."""
        if linear:
            return np.linalg.det(nucleus)
        return 2 if np.linalg.det(nucleus) > 0 else 0

    def __repr__(self):
        return str(self.nucleus)
