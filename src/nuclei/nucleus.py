import copy

import numpy as np
from abc import ABC
from typing import Union, List
from src.nuclei.activations import Activation, Binary


rng = np.random.default_rng(12345)


class Nucleus(ABC):
    """Creates a nucleus object. A nucleus takes stimulus of size `input_size` and can project it through
    producing a scalar.
    Arguments:
        m: The input size.
        lr: The learning rate.
        activation: Whether to apply a transformation to output.
    """
    def __init__(self, input_size, lr, activation):
        self.m = input_size
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.m, self.m))
        self.feat_ids = list(range(1, self.m, 2))
        self.biases_ids = list(range(0, self.m, 2))

    def project(self, stimulus):
        pass

    def __repr__(self):
        return str(self.nucleus)


class BaseNucleus1B(Nucleus):
    """A 2-D array with stimuli projected through the diagonal with only 1 bias."""
    def __init__(self, input_size: int, lr: float = 1e-5, activation: Activation() = Binary()):
        super(BaseNucleus1B, self).__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimulus needs to be > 1.'
        self.m = input_size + 1
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.m, self.m))

    def project(self, stimulus: np.array) -> float:
        """Projects the stimulus through the nucleus and returns a spike (a negative or positive signal)."""
        stimulus_in = self.nucleus[0, 0] + stimulus
        shadow = copy.deepcopy(self.nucleus)
        np.fill_diagonal(shadow, stimulus_in)
        return _project(shadow, self.activation)[0]


class BaseNucleus(Nucleus):
    """A 2-D array with stimuli projected through the diagonal with 1 bias per input feature."""
    def __init__(self, input_size: int, lr: float = 1e-5, activation: Activation() = Binary()):
        super(BaseNucleus, self).__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimulus needs to be > 1.'
        self.m = input_size * 2
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.m, self.m))
        self.feat_ids = list(range(1, self.m, 2))
        self.biases_ids = list(range(0, self.m, 2))

    def project(self, stimulus: np.array) -> float:
        """Projects the stimulus through the nucleus and returns a spike (a negative or positive signal)."""
        stimulus_in = list(zip([1] * self.m, stimulus))
        shadow = copy.deepcopy(self.nucleus)
        np.fill_diagonal(shadow, stimulus_in)
        return _project(shadow, self.activation)[0]


class LongNucleus(Nucleus):
    """A 3-D version that outputs a list of outcomes."""
    def __init__(self, input_size: int, output_size: Union[int, List[str]],
                 lr: float = 1e-5, activation: Activation() = Binary()):
        super(Nucleus, self).__init__()
        assert input_size > 1, 'Size of stimulus needs to be > 1.'
        self.m = input_size * 2
        self.output_size = output_size
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.output_size, self.m, self.m))
        self.feat_ids = list(range(1, self.m, 2))
        self.biases_ids = list(range(0, self.m, 2))

    def project(self, stimulus: np.array) -> List[float]:
        """Projects the stimulus through the nucleus and returns a spike (a negative or positive signal)."""
        shadow = copy.deepcopy(self.nucleus)
        stimulus_in = list(zip([1] * self.m, stimulus))
        stimulus_in = np.array(stimulus_in * self.output_size).reshape(self.output_size, self.m)
        r = np.arange(self.m)
        shadow.nucleus[:, r, r] = stimulus_in
        return _project(shadow, self.activation)


def _project(nucleus: np.array, activation: Activation) -> List[float]:
    """Converts the nucleus weights into a scalar."""
    determinants = get_determinant(nucleus)
    return [activation.fit(d) for d in determinants]


def get_determinant(a, slog=True):
    if slog:
        sign, logabsdet = np.linalg.slogdet(a)
        return sign * np.exp(logabsdet)
    else:
        return np.linalg.det(a)
