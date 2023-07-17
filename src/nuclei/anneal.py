import numpy as np
import copy
from abc import ABC
from itertools import product
from tqdm import tqdm
from typing import Tuple, List, Optional
from src.nuclei.nucleus import Nucleus, BaseNucleus, BaseNucleus1B, LongNucleus
from src.nuclei.losses import Loss, MAE


rng = np.random.default_rng(12345)


class Anneal(ABC):
    def __init__(self, nucleus: Nucleus, temp: float, loss_func: Loss, steps, epochs):
        self.nucleus = nucleus
        self.loss = loss_func
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli):
        pass


class BaseAnneal(Anneal):
    def __init__(self, nucleus: BaseNucleus, temp: float, loss_func: Loss() = MAE(), steps=5, epochs=1):
        super(Anneal, self).__init__()
        self.nucleus = nucleus
        self.loss = loss_func
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


class LongAnneal(Anneal):
    def __init__(self, nucleus: LongNucleus, temp: float, loss_func: Loss() = MAE(), steps=5, epochs=1):
        super(Anneal, self).__init__()
        self.nucleus = nucleus
        self.loss = loss_func
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli: dict, verbose=False,
                  validate=False, train_size=0.5) -> \
            Tuple[Nucleus, List[float], Optional[List[dict]]]:
        test = None
        if validate:
            stimuli, test = split_stimuli(stimuli, train_size)
        else:
            stimuli['y'] = dummy(stimuli['y'])
        a = generate_long_solution(self.nucleus)
        best = a
        losses = []
        val_losses = []
        cost = None
        for e in self.epochs:
            print('Epoch:', e + 1)
            for _ in tqdm(self.steps, desc=f'training in epoch {e + 1}'):
                b = generate_long_solution(best)
                best, cost = compare_energies(best, b, stimuli['x'], stimuli['y'], self.loss, self.temp, verbose)
                losses.append(cost)
                if cost == 0:
                    break
            self.temp *= 0.9
            if validate:
                preds = forward(best, test['x'])
                val = self.loss.fit(test['y'], preds)
                val_losses.append({'train': cost, 'val': val})
                print(f'Epoch {e + 1} validation error: {val}')
            print(f'Epoch training loss: {np.mean(losses[-len(self.steps):])}')

        return best, losses, val_losses


def forward(nucleus: Nucleus, X: np.array) -> np.array:
    preds = []
    for x in X:
        preds.append(nucleus.project(x))
    return preds


def calculate_loss(nucleus: Nucleus, X: np.array, Y: np.array,
                   loss_func: Loss) -> float:
    preds = forward(nucleus, X)
    return loss_func.fit(Y, preds)


def generate_solution(nucleus: Nucleus) -> Nucleus:
    """Perturbs the nucleus' weights. Skips the stimulus indices."""
    jitter = rng.standard_normal() * nucleus.lr
    all_i_j = product(range(nucleus.m), range(nucleus.m))
    list_i_j = [(i, j) for i, j in all_i_j if i != j] + nucleus.biases_ids
    random_i_j = rng.choice(list_i_j, 3)
    candidate = copy.deepcopy(nucleus)
    for i, j in random_i_j:
        candidate.nucleus[i, j] += jitter
    return candidate


def generate_long_solution(nucleus: LongNucleus) -> LongNucleus:
    """Perturbs the nucleus' weights. Skips the stimulus indices."""
    jitter = rng.standard_normal() * nucleus.lr
    all_idx = product(range(nucleus.l), range(nucleus.m), range(nucleus.m))
    list_idx = [(l, i, j) for l, i, j in all_idx if i != j] + nucleus.biases_ids
    random_idx = rng.choice(list_idx, 3)
    candidate = copy.deepcopy(nucleus)
    for l, i, j in random_idx:
        candidate.nucleus[l, i, j] += jitter
    return candidate


def compare_energies(i: Nucleus, j: Nucleus, X: np.array, Y: np.array,
                     loss_func: Loss, temp: float,
                     verbose=False) -> Tuple[Nucleus, float]:
    l_i = calculate_loss(i, X, Y, loss_func)
    l_j = calculate_loss(j, X, Y, loss_func)
    prob = np.exp((l_j - l_i) / temp)
    r = rng.random()
    if verbose:
        print(f'{l_i = }, {l_j = }, {prob = }, {r = }')
    if l_j < l_i:
        return j, l_j
    else:
        if r > prob:
            return j, l_j
        else:
            return i, l_i


def split_stimuli(stimuli: dict, train_size=0.5) -> Tuple[dict, dict]:
    n = len(stimuli['x'])
    train_idx = rng.choice(range(n), int(n * train_size))
    test_idx = [i for i in range(n) if i not in train_idx]
    train = {
        'x': stimuli['x'][train_idx],
        'y': dummy(stimuli['y'][train_idx])
    }
    test = {
        'x': stimuli['x'][test_idx],
        'y': dummy(stimuli['y'][test_idx])
    }
    return train, test


def dummy(y: np.array) -> List[List[int]]:
    """Vectorize categorical labels."""
    return [int2bin(num) for num in y]


def int2bin(num: int) -> List[int]:
    """Convert label number to a binary vector representation."""
    return [int(v) for v in list(f'{num:02b}')]
