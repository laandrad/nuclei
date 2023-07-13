from typing import Union, List, Optional
from pprint import pprint
from pydantic import BaseModel
import numpy as np


rng = np.random.default_rng(12345)


class Nuclei:
    """
    When a nuclei projects a stimulus, it passes it through and returns a spike.
    When the nuclei is conditioned, it compares its spikes with an expected stimuli.
    If the spikes and expected stimulus are the same, the nuclei gets excited.
    Otherwise, the nuclei weights are perturbed via a random jitter.
    Arguments:
        input_size: the size of the stimulus
        num_layers: the number of nuclei per layer. If `int` then all layers have equal size.
            if `list` then its `len` should be equal to depth.
        depth: the number of layers.
    """
    def __init__(self, input_size: int,
                 nuc_per_layers: Union[int, List[int]],
                 depth: int,
                 temperature: float = 0.5,
                 lr: float = 1e-5):
        # assert nuc_per_layers > 1, 'nuc_per_layers parameter needs to be > 1.'
        # assert depth > 1, 'depth parameter needs to be > 1.'
        self.input_size = input_size
        if isinstance(nuc_per_layers, list):
            self.nuc_per_layers = [self.input_size] + nuc_per_layers
        else:
            self.nuc_per_layers = [self.input_size] + [nuc_per_layers]
        # if isinstance(nuc_per_layers, int):
        #     self.nuc_per_layers = [nuc_per_layers] * depth
        # self.nuc_per_layers = [self.input_size] + self.nuc_per_layers
        self.depth = depth
        self.temperature = temperature
        self.lr = lr
        self.layers: Optional[dict] = None
        self._build()

    def condition(self, x_stimuli: np.array, y_stimuli: np.array, batch: int, steps: int):
        assert len(x_stimuli) == len(y_stimuli), 'x and y should be the same length.'
        temps = []
        vals = []
        best_excitation = None
        for step in range(steps):
            old_architecture = self.layers.copy()
            self._jitter()
            _, excitation = self._test(x_stimuli, y_stimuli, batch)
            if not best_excitation:
                best_excitation = excitation
                continue
            if best_excitation < self.temperature:
                print(f'{best_excitation}, {self.temperature}, this should break')
                break
            elif excitation <= best_excitation:
                best_excitation = excitation
            else:
                self.layers = old_architecture
            vals.append(best_excitation)
        return {'temps': temps, 'vals': vals}

    def validate(self, x_stimuli: np.array, y_stimuli: np.array):
        spikes = []
        errors = []
        for stimulus, target in zip(x_stimuli, y_stimuli):
            spike = self.project(stimulus)
            error = (target - spike)**2
            spikes.append(spike)
            errors.append(error)
        excitation = np.sqrt(np.mean(errors))
        return spikes, excitation

    def project(self, stimulus: np.array) -> float:
        spike = None
        layer_stimuli = stimulus.copy()
        for layer in self.layers:
            spikes = []
            for nucleus in self.layers[layer]:
                spike = self.layers[layer][nucleus].project(layer_stimuli)
                spikes.append(spike)
            layer_stimuli = spikes
        return spike

    def _test(self, x_stimuli: np.array, y_stimuli: np.array, batch):
        ids = rng.choice(range(len(x_stimuli)), size=batch)
        x = x_stimuli[ids]
        y = y_stimuli[ids]
        return self.validate(x, y)

    def _jitter(self):
        for layer in self.layers:
            for nucleus in self.layers[layer]:
                self.layers[layer][nucleus].jitter()

    def _build(self):
        j = None
        layers = {}
        for j in range(self.depth):
            layer = {f'Nucleus_{i}': Nucleus(self.nuc_per_layers[j], self.lr, linear=True)
                     for i in range(self.nuc_per_layers[j])}
            layers[f'Layer_{j}'] = layer
        # if j:
        #     layers[f'Layer_{j + 1}'] = {'Final_Nucleus': Nucleus(self.nuc_per_layers[-1],
        #                                                          lr=self.lr,
        #                                                          linear=False)}
        self.layers = layers

    def __repr__(self):
        return str(pprint(self.layers))
