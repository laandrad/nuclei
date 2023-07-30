import numpy as np
from itertools import product
from typing import Tuple
from src.nuclei.nucleus import Nucleus, _project
from src.nuclei.preprocess import Preprocess
from src.nuclei.activations import Activation, Sigmoid


rng = np.random.default_rng(12345)
process = Preprocess()


class Saccade:
    def __init__(self, width: int, time: int):
        self.width = width
        self.time = time
        self.stimulus = None

    def observe(self, image: np.array):
        stimulus = []
        for _ in range(self.time):
            stimulus.append(self.grab(image))
        self.stimulus = stimulus

    def grab(self, image: np.array) -> Tuple[np.array, Tuple[int, int]]:
        i = rng.choice(range(image.shape[0] - self.width))
        j = rng.choice(range(image.shape[1] - self.width))
        crop = image[i:i+self.width, j:j+self.width]
        return crop, (i, j)


class OpticalNucleus(Nucleus):
    def __init__(self, input_size: int, lr: float = 1e-5,
                 activation: Activation() = Sigmoid()):
        super().__init__(input_size, lr, activation)
        assert input_size > 1, 'Size of stimulus needs to be > 1.'
        self.m = input_size
        self.lr = lr
        self.activation = activation
        self.nucleus = np.ones((self.m, self.m))

    def project(self, saccade) -> float:
        """Multiply nucleus coords by cropped image to obtain template.
        Take determinant of template. Returns the mean of these determinants."""
        refs = []
        for crop, (x, y) in saccade.stimulus:
            ref = self.nucleus[x:x + saccade.width, y:y + saccade.width] * crop
            d = _project(ref, self.activation)
            refs.append(d)
        return np.array(refs).mean()

    def jitter(self):
        jitter = rng.standard_normal() * self.lr
        all_idx = list(product(range(self.m), range(self.m)))
        random_idx = rng.choice(all_idx, 3)
        for i, j in random_idx:
            self.nucleus[i, j] += jitter
