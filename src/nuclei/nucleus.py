import copy

import numpy as np
from abc import ABC
from itertools import product
from typing import Union, List, Tuple
from src.nuclei.activations import Activation, Sigmoid


rng = np.random.default_rng(12345)


class Nucleus(ABC):
    """Creates a nucleus object. A nucleus takes stimuli of size `input_size` and can project it through
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

    def jitter(self):
        pass

    def __repr__(self):
        return str(self.nucleus)


class BaseNucleus1B(Nucleus):
    """A 2-D array with stimuli projected through the diagonal with only 1 bias."""
    def __init__(self, input_size: int, lr: float = 1e-5, activation: Activation() = Sigmoid()):
        super().__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimuli needs to be > 1.'
        self.m = input_size + 1
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.m, self.m))
        self.feat_ids = list(range(1, self.m))
        self.biases_ids = [(0, 0)]

    def project(self, stimulus: np.array) -> float:
        """Projects the stimuli through the nucleus and returns a spike."""
        stimulus_in = np.ones(self.m)
        stimulus_in[self.feat_ids] = stimulus
        shadow = copy.deepcopy(self.nucleus)
        np.fill_diagonal(shadow, stimulus_in)
        return _project(shadow, self.activation)


class BaseNucleus(Nucleus):
    """A 2-D array with stimuli projected through the diagonal with 1 bias per input feature."""
    def __init__(self, input_size: int, lr: float = 1e-5, activation: Activation() = Sigmoid()):
        super().__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimuli needs to be > 1.'
        self.m = input_size * 2
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.m, self.m))
        self.feat_ids = list(range(1, self.m, 2))
        self.biases_ids = list(zip(range(0, self.m, 2), range(0, self.m, 2)))

    def project(self, stimulus: np.array) -> float:
        """Projects the stimuli through the nucleus and returns a spike."""
        stimulus_in = np.ones(self.m)
        stimulus_in[self.feat_ids] = stimulus
        shadow = copy.deepcopy(self.nucleus)
        np.fill_diagonal(shadow, stimulus_in)
        return _project(shadow, self.activation)

    def jitter(self):
        """Perturbs the nucleus' weights. Skips the stimuli indices."""
        jitter = rng.standard_normal() * self.lr
        all_i_j = product(range(self.m), range(self.m))
        list_i_j = [(i, j) for i, j in all_i_j if i != j] + self.biases_ids
        random_i_j = rng.choice(list_i_j, 3)
        for i, j in random_i_j:
            self.nucleus[i, j] += jitter


class LongNucleus(Nucleus):
    """A 3-D version with categorical outputs."""
    def __init__(self, input_size: int, output_size: Union[int, List[str]], lr: float = 1e-5,
                 activation: Activation() = Sigmoid()):
        super().__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimuli needs to be > 1.'
        self.m = input_size * 2
        self.l = output_size
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.l, self.m, self.m))
        self.feat_ids = list(range(1, self.m, 2))
        self.biases_ids = list(zip(range(self.l), range(0, self.m, 2), range(0, self.m, 2)))

    def project(self, stimulus: np.array) -> List[float]:
        """Projects the stimuli through the nucleus and returns a spike."""
        stimulus_in = np.ones(self.m)
        stimulus_in[self.feat_ids] = stimulus
        r = np.arange(self.m)
        shadow = copy.deepcopy(self.nucleus)
        shadow[:, r, r] = stimulus_in
        return _project(shadow, self.activation)

    def jitter(self):
        """Perturbs the nucleus' weights. Skips the stimuli indices."""
        jitter = rng.standard_normal() * self.lr
        all_idx = product(range(self.l), range(self.m), range(self.m))
        list_idx = [(l, i, j) for l, i, j in all_idx if i != j] + self.biases_ids
        random_idx = rng.choice(list_idx, 3)
        for l, i, j in random_idx:
            self.nucleus[l, i, j] += jitter


class LongNucleus1B(Nucleus):
    """A 3-D version with categorical outputs."""
    def __init__(self, input_size: int, output_size: Union[int, List[str]], lr: float = 1e-5,
                 activation: Activation() = Sigmoid()):
        super().__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimuli needs to be > 1.'
        self.m = input_size + 1
        self.l = output_size
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.l, self.m, self.m))
        self.feat_ids = list(range(1, self.m))
        self.biases_ids = [(0, 0, 0)]

    def project(self, stimulus: np.array) -> List[float]:
        """Projects the stimuli through the nucleus and returns a spike."""
        stimulus_in = np.ones(self.m)
        stimulus_in[self.feat_ids] = stimulus
        r = np.arange(self.m)
        shadow = copy.deepcopy(self.nucleus)
        shadow[:, r, r] = stimulus_in
        return _project(shadow, self.activation)


def _project(nucleus: np.array, activation: Activation) -> Union[float, List[float]]:
    """Converts the nucleus weights into a scalar."""
    determinant = get_determinant(nucleus)
    if isinstance(determinant, float):
        return activation.fit(determinant)
    else:
        return [activation.fit(d) for d in determinant]


def get_determinant(a, slog=True):
    if slog:
        sign, logabsdet = np.linalg.slogdet(a)
        return sign * np.exp(logabsdet)
    else:
        return np.linalg.det(a)
