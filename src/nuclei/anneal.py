import numpy as np
import copy
from tqdm import tqdm
from typing import Tuple, List, Optional
from src.nuclei.nucleus import Nucleus
from src.nuclei.losses import Loss
from src.nuclei.preprocess import Preprocess


rng = np.random.default_rng(12345)
process = Preprocess()


class Anneal:
    def __init__(self, nucleus: Nucleus, temp: float, loss_func: Loss, steps=5, epochs=1):
        self.nucleus = nucleus
        self.loss = loss_func
        self.temp = temp
        self.epochs = range(epochs)
        self.steps = range(steps)

    def stimulate(self, stimuli: dict, verbose=False,
                  validate=True, train_size=0.5) -> \
            Tuple[Nucleus, List[float], Optional[List[dict]]]:
        test = None
        if validate:
            stimuli, test = process.split_stimuli(stimuli, train_size)
        else:
            len_labels = len(np.unique(stimuli['y']))
            stimuli['y'] = process.binary_vec(stimuli['y'], len_labels)
        i = copy.deepcopy(self.nucleus)
        i.jitter()
        losses = []
        val_losses = []
        cost = None
        for e in self.epochs:
            print('Epoch:', e + 1)
            for _ in tqdm(self.steps, desc=f'training in epoch {e + 1}'):
                j = copy.deepcopy(i)
                j.jitter()
                i, cost = compare_energies(i, j, stimuli['x'], stimuli['y'], self.loss, self.temp, verbose)
                losses.append(cost)
                if cost == 0:
                    break
            self.temp *= 0.9
            if validate:
                preds = forward(i, test['x'])
                val = self.loss.fit(test['y'], preds)
                val_losses.append({'train': cost, 'val': val})
                print(f'Epoch {e + 1} validation loss: {val}')
            print(f'Epoch training loss: {np.mean(losses[-len(self.steps):])}')

        return i, losses, val_losses


def forward(nucleus: Nucleus, X: np.array) -> np.array:
    preds = []
    for x in X:
        preds.append(nucleus.project(x))
    return preds


def calculate_loss(nucleus: Nucleus, X: np.array, Y: np.array,
                   loss_func: Loss) -> float:
    logits = forward(nucleus, X)
    return loss_func.fit(Y, logits)


def compare_energies(i: Nucleus,
                     j: Nucleus,
                     X: np.array,
                     Y: np.array,
                     loss_func: Loss,
                     temp: float,
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
