import copy

import numpy as np
from itertools import product
from typing import Tuple
from src.nuclei.nucleus import Nucleus, _project
from src.nuclei.preprocess import Preprocess
from src.nuclei.activations import Activation, Sigmoid, Binary, Relu

rng = np.random.default_rng(12345)
process = Preprocess()


class Saccade:
    def __init__(self, width: int, time: int):
        self.width = width
        self.time = time
        self.stimuli = None

    def observe(self, image: np.array):
        stimuli = []
        for _ in range(self.time):
            stimuli.append(self.grab(image))
        self.stimuli = stimuli

    def grab(self, image: np.array) -> Tuple:
        i = rng.choice(range(image.shape[0] - self.width))
        j = rng.choice(range(image.shape[1] - self.width))
        crop = image[i:i + self.width, j:j + self.width]
        return *crop.flatten(), i / image.shape[0], j / image.shape[1]


class OpticalNucleus(Nucleus):
    def __init__(self,
                 saccade_width: int,
                 saccade_time: int,
                 lr: float = 1e-5,
                 activation: Activation() = Binary()):
        super().__init__(saccade_width, lr, activation)
        assert saccade_width > 1, 'Size of stimuli needs to be > 1.'
        self.m = saccade_width + 2  # input size is independent of image size
        self.lr = lr
        self.activation = activation
        self.nucleus = np.zeros((self.m, self.m))
        self.feat_ids = list(product(range(1, self.m - 1), range(1, self.m - 1))) + \
                        [(0, 0), (self.m - 1, self.m - 1)]
        self.biases_ids = [i for i in product(range(self.m), range(self.m))
                           if i not in self.feat_ids]
        self.saccade = Saccade(saccade_width, saccade_time)

    def project(self, image) -> float:
        """Multiply nucleus coords by cropped image to obtain template.
        Take determinant of template. Returns the mean of these determinants."""
        self.saccade.observe(image)
        shadow = copy.deepcopy(self.nucleus)
        refs = []
        for stimulus in self.saccade.stimuli:
            for ids, val in zip(self.feat_ids, stimulus):
                shadow[ids] = val
            d = _project(shadow, Binary())
            refs.append(d)
        z = self.activation.fit(np.array(refs).mean())
        # print(f'\n\n{refs = }\n{z = }')
        return z

    def jitter(self):
        jitter = rng.standard_normal() * self.lr
        all_idx = self.biases_ids
        random_idx = rng.choice(all_idx, 3)
        for i, j in random_idx:
            self.nucleus[i, j] += jitter
