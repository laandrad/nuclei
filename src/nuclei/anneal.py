import numpy as np
import copy
from itertools import product


rng = np.random.default_rng(12345)


class Anneal:
    def __init__(self, nucleus, temp, steps=5, epochs=1):
        self.nucleus = nucleus
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli):
        a = generate_solution(self.nucleus)
        best = copy.deepcopy(a)
        loss = []

        for e in self.epochs:
            for k in self.steps:
                b = generate_solution(best)
                best = compare_energies(best, b, stimuli['x'], stimuli['y'], absolute_error, self.temp)
                cost = fit(best, stimuli['x'], stimuli['y'], absolute_error)
                loss.append(cost)
                if cost == 0:
                    break
            self.temp *= 0.9

        return best, loss


def absolute_error(y, preds):
    return np.sum([abs(y_i - pred_i) for y_i, pred_i in zip(y, preds)])


def forward(nucleus, X):
    preds = []
    for x in X:
        preds.append(nucleus.project(x))
    return preds


def fit(nucleus, X, Y, loss):
    preds = forward(nucleus, X)
    return loss(Y, preds)


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
    e_i = fit(i, X, Y, loss)
    e_j = fit(j, X, Y, loss)
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