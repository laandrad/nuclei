import numpy as np
import copy
from abc import ABC
from itertools import product
from tqdm import tqdm
from typing import Tuple, List
from src.nuclei.nucleus import Nucleus
from src.nuclei.losses import Loss, MAE


rng = np.random.default_rng(12345)


class Anneal(ABC):
    def __init__(self, nucleus: Nucleus, temp: float, loss: Loss, steps, epochs):
        self.nucleus = nucleus
        self.loss = loss
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli):
        pass


class BaseAnneal(Anneal):
    def __init__(self, nucleus: Nucleus, temp: float, loss: Loss() = MAE(), steps=5, epochs=1):
        super(Anneal, self).__init__()
        self.nucleus = nucleus
        self.loss = loss
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli: dict, verbose=False) -> Tuple[Nucleus, List[float]]:
        a = generate_solution(self.nucleus)
        best = a
        losses = []

        for e in self.epochs:
            print('Epoch:', e + 1)
            for _ in tqdm(self.steps, desc=f'training in epoch {e + 1}'):
                b = generate_solution(best)
                best, cost = compare_energies(best, b, stimuli['x'], stimuli['y'], self.loss, self.temp, verbose)
                losses.append(cost)
                if cost == 0:
                    break
            self.temp *= 0.9
            print(f'Epoch training loss: {np.mean(losses[-len(self.steps):])}')

        return best, losses


def forward(nucleus, X):
    preds = []
    for x in X:
        preds.append(nucleus.project(x))
    return preds


def get_energy(nucleus, X, Y, loss: Loss):
    preds = forward(nucleus, X)
    return loss.fit(Y, preds)


def generate_solution(nucleus):
    """Perturbs the nucleus' weights. Skips diagonal, which represents the stimulus."""
    candidate = copy.deepcopy(nucleus)
    jitter = rng.standard_normal() * nucleus.lr
    all_i_j = product(range(nucleus.m), range(nucleus.m))
    list_i_j = [(i, j) for i, j in all_i_j if i != j] + [(0, 0)]
    random_i_j = rng.choice(list_i_j, 3)
    for i, j in random_i_j:
        candidate.nucleus[i, j] += jitter
    return candidate


def compare_energies(i: Nucleus, j: Nucleus, X: list, Y: list, loss: Loss, temp: float, verbose=False):
    e_i = get_energy(i, X, Y, loss)
    e_j = get_energy(j, X, Y, loss)
    prob = np.exp((e_j - e_i) / temp)
    r = rng.random()
    if verbose:
        print(f'{e_i = }, {e_j = }, {prob = }, {r = }')
    if e_j < e_i:
        return j, e_j
    else:
        if r > prob:
            return j, e_j
        else:
            return i, e_i
