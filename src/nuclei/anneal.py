import numpy as np
import copy
from itertools import product
from tqdm import tqdm
from typing import Tuple, List
from src.nuclei.nucleus import Nucleus
from src.nuclei.losses import Loss, MAE


rng = np.random.default_rng(12345)


class Anneal:
    def __init__(self, nucleus: Nucleus, temp: float, loss: Loss() = MAE(), steps=5, epochs=1):
        self.nucleus = nucleus
        self.loss = loss
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli) -> Tuple[Nucleus, List[float]]:
        a = generate_solution(self.nucleus)
        best = copy.deepcopy(a)
        losses = []

        for e in self.epochs:
            print('Epoch:', e + 1)
            for _ in tqdm(self.steps, desc=f'training in epoch {e + 1}'):
                b = generate_solution(best)
                best = compare_energies(best, b, stimuli['x'], stimuli['y'], self.loss, self.temp)
                cost = get_energy(best, stimuli['x'], stimuli['y'], self.loss)
                losses.append(cost)
                if cost == 0:
                    break
            self.temp *= 0.9
            print(f'Epoch training loss: {np.mean(losses)}')

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
    all_i_j = product(range(nucleus.n), range(nucleus.n))
    list_i_j = [(i, j) for i, j in all_i_j if i != j]
    random_i_j = rng.choice(list_i_j, 3)
    for i, j in random_i_j:
        candidate.nucleus[i, j] += jitter
    return candidate


def compare_energies(i, j, X, Y, loss, temp, verbose=False):
    e_i = get_energy(i, X, Y, loss)
    e_j = get_energy(j, X, Y, loss)
    prob = np.exp((e_j - e_i) / temp)
    r = rng.random()
    if verbose:
        print(f'{e_i = }, {e_j = }, {prob = }, {r = }')
    if e_j < e_i:
        return j
    else:
        if r > prob:
            return j
        else:
            return i
