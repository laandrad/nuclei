import numpy as np
from typing import Tuple, List


rng = np.random.default_rng(12345)


class Preprocess:
    def __init__(self):
        pass

    def split_stimuli(self, stimuli: dict, train_size=0.5) -> Tuple[dict, dict]:
        n = len(stimuli['x'])
        len_labels = len(np.unique(stimuli['y']))
        train_idx = rng.choice(range(n), int(n * train_size))
        test_idx = [i for i in range(n) if i not in train_idx]
        train = {
            # 'x': stimuli['x'][train_idx],
            'x': [stimuli['x'][i] for i in train_idx],
            'y': self.binary_vec(np.array(stimuli['y'])[train_idx], len_labels)
        }
        test = {
            # 'x': stimuli['x'][test_idx],
            'x': [stimuli['x'][i] for i in test_idx],
            'y': self.binary_vec(np.array(stimuli['y'])[test_idx], len_labels)
        }
        return train, test

    def binary_vec(self, y: np.array, len_labels: int) -> List[List[int]]:
        """Vectorize categorical labels."""
        max_n = len(f'{len_labels - 1:b}')
        vec = [self.int2bin(num, max_n) for num in y]
        return [self.str2vec(x) for x in vec]

    @staticmethod
    def one_hot(y: np.array, max_n: int) -> List[np.array]:
        vectors = []
        for v in y:
            vec = np.zeros(max_n)
            vec[v] = 1
            vectors.append(vec)
        return vectors

    @staticmethod
    def bin2int(bin_vec: List[int]) -> int:
        bin_str = ''.join([str(v) for v in bin_vec])
        return int(bin_str, 2)

    @staticmethod
    def int2bin(num: int, max_n: int) -> str:
        """Convert label number to a binary vector representation."""
        binary = ''.join([str(int(v)) for v in list(f'{num:b}')])
        return binary.zfill(max_n)

    @staticmethod
    def str2vec(x: str) -> List[int]:
        return [int(n) for n in x]
