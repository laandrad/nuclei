import numpy as np


rng = np.random.default_rng(12345)


class Nuclei:
    def __init__(self):
        pass


class Nucleus:
    """Creates a nucleus object. A nucleus takes stimuli and can either project it or process it.
    When the nucleus projects a stimuli, it passes it through and returns a spike.
    When the nucleus processes a stimuli, it compares the spike with an expected outcome.
    If spike and expected outcome are the same, the nucleus gets excited.
    Otherwise, the nucleus weights are perturbed via a random jitter."""
    def __init__(self, n: int):
        self.n = n
        self.temp = 1
        self.nucleus = np.zeros((n, n))

    def process(self, stimuli: np.array, expected_outcome: int) -> bool:
        """Processes a stimuli. It can either excite due to spike == outcome, or jitter.
        A jitter perturbs the nucleus weights by a little."""
        excite = False
        spike = self.project(stimuli)
        if not spike == expected_outcome:
            self._jitter()
        else:
            if rng.random() < self.temp:
                self._jitter()
            self.temp *= 0.95
            excite = True
        return excite

    def _jitter(self):
        """Perturbs the nucleus' weights. Skips diagonal, which represents the stimuli."""
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self._jitter_i_j(i, j)

    def _jitter_i_j(self, i: int, j: int, lr: float = 0.005):
        """Perturbs the i, j nucleus' weight."""
        jitter = rng.standard_normal() * lr
        self.nucleus[i, j] += jitter

    def project(self, stimuli: np.array) -> float:
        """Projects the stimuli through the nucleus and returns a spike (a negative or positive signal)."""
        self._fill_diag(stimuli)
        return self._project()

    def _fill_diag(self, stimuli: np.array):
        """Passes the simuli values to the nucleus."""
        return np.fill_diagonal(self.nucleus, stimuli)

    def _project(self) -> float:
        """Converts the nucleus weights into a scalar."""
        return np.sign(np.linalg.det(self.nucleus))

    def __repr__(self):
        return str(self.nucleus)


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, -1, 1, -1])
    a = Nucleus(2)
    # Train
    for x, y in zip(X, Y):
        print('---')
        for epoch in range(50):
            a.process(x, y)
            # print(a)
            # time.sleep(2)
    # Test
    for x, y in zip(X, Y):
        pred = a.activate(x)
        print(f'{x = }, {y = }, {pred = }, {pred == y}')


