from typing import Union
from pprint import pprint
import numpy as np


rng = np.random.default_rng(12345)


class Nuclei:
    """
    When a nuclei projects a stimulus, it passes it through and returns a spike.
    When the nuclei is conditioned, it compares its spikes with an expected stimuli.
    If the spikes and expected stimulus are the same, the nuclei gets excited.
    Otherwise, the nuclei weights are perturbed via a random jitter.
    Arguments:
        n: the size of the stimulus
        h: the number of nuclei per layer. If `int` then all layers have equal size.
            if `list` then its `len` should be equal to depth.
        depth: the number of layers.
    """
    def __init__(self, n: int, h: Union[int, list], depth: int):
        assert h > 1, 'h parameter needs to be > 1.'
        assert depth > 1, 'depth parameter needs to be > 1.'
        self.n = n
        self.h = h
        if isinstance(h, int):
            self.h = [h] * depth
        self.h = [self.n] + self.h
        self.depth = depth - 1
        self.layers = None
        self.temperature = 1
        self._build()

    def condition(self, x_stimuli: np.array, y_stimuli: np.array, batch: int, steps: int):
        assert len(x_stimuli) == len(y_stimuli), 'x and y lengths should be the same.'
        temps = []
        vals = []
        for step in range(steps):
            spikes, excitation = self._condition(x_stimuli, y_stimuli, batch)
            # print(f'{step = }, {spikes = }, {excited = }, {self.temperature = }')
            if excitation < self.temperature:
                self._jitter()
            else:
                self.temperature *= 0.99
            temps.append(self.temperature)
            vals.append(self._validate(x_stimuli, y_stimuli)[1])
        return {'temps': temps, 'vals': vals}

    def _validate(self, x_stimuli: np.array, y_stimuli: np.array):
        spikes = []
        for stimulus in x_stimuli:
            spikes.append(self.project(stimulus))
        excitation = 1 - (y_stimuli - np.array(spikes)).mean()
        return spikes, excitation

    def _condition(self, x_stimuli: np.array, y_stimuli: np.array, batch):
        ids = rng.choice(range(len(x_stimuli)), size=batch)
        x = x_stimuli[ids]
        y = y_stimuli[ids]
        return self._validate(x, y)

    def _jitter(self):
        for layer in self.layers:
            for nucleus in self.layers[layer]:
                self.layers[layer][nucleus].jitter()

    def project(self, stimulus: np.array) -> float:
        layer_stimuli = stimulus.copy()
        for layer in self.layers:
            spikes = []
            for nucleus in self.layers[layer]:
                spike = self.layers[layer][nucleus].project(layer_stimuli)
                spikes.append(spike)
            layer_stimuli = spikes
        return spike

    def _build(self):
        layers = {}
        for j in range(self.depth):
            layer = {f'Nucleus_{i}': Nucleus(self.h[j]) for i in range(self.h[j])}
            layers[f'Layer_{j}'] = layer
        layers[f'Layer_{j + 1}'] = {'Final_Nucleus': Nucleus(self.h[-1])}
        self.layers = layers

    def __repr__(self):
        return str(pprint(self.layers))


class Nucleus:
    """Creates a nucleus object. A nucleus takes stimulus and can either project it or jitter."""
    def __init__(self, n: int):
        assert n > 1, 'Size of stimulus needs to be > 1.'
        self.n = n
        self.nucleus = np.zeros((n, n))

    def jitter(self, lr: float = 0.005):
        """Perturbs the nucleus' weights. Skips diagonal, which represents the stimulus."""
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self._jitter_i_j(i, j, lr)

    def _jitter_i_j(self, i: int, j: int, lr: float):
        """Perturbs the i, j nucleus' weight."""
        jitter = rng.standard_normal() * lr
        self.nucleus[i, j] += jitter

    def project(self, stimulus: np.array) -> float:
        """Projects the stimulus through the nucleus and returns a spike (a negative or positive signal)."""
        self._fill_diag(stimulus)
        return self._project()

    def _fill_diag(self, stimulus: np.array):
        """Passes the simulus values to the nucleus."""
        return np.fill_diagonal(self.nucleus, stimulus)

    def _project(self) -> float:
        """Converts the nucleus weights into a scalar."""
        return np.linalg.det(self.nucleus)
        # return np.sign(np.linalg.det(self.nucleus))

    def __repr__(self):
        return str(self.nucleus)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, 1, -1])
    a = Nuclei(n=2, h=3, depth=3)
    results = a.condition(X, Y, batch=3, steps=1000)
    plt.plot(results['temps'])
    plt.title('Temperature')
    plt.show()
    plt.plot(results['vals'])
    plt.title('Validation')
    plt.show()
    # # Train
    # for x, y in zip(X, Y):
    #     print('---')
    #     for epoch in range(50):
    #         a.process(x, y)
    #         # print(a)
    #         # time.sleep(2)
    # # Test
    # for x, y in zip(X, Y):
    #     pred = a.activate(x)
    #     print(f'{x = }, {y = }, {pred = }, {pred == y}')


