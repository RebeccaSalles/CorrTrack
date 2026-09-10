"""Multi-grid index with voting (multi-table LSH style).

Several GridIndex instances, each with a random shift of the cell boundaries.
A key is a candidate if it co-occurs with the query in at least
`freq_threshold` grids. Several shifted grids catch the close pairs that a
single cell separates (boundary effect) -> better recall, in the spirit of v1's
`freq_threshold` vote.
"""

from collections import Counter

import numpy as np

from .base import SketchIndex
from .grid import GridIndex


class MultiGridIndex(SketchIndex):
    def __init__(self, n_grids, cell_size, freq_threshold, grid_dimension, n_vectors, seed):
        rng = np.random.default_rng(seed)
        dim = min(max(1, grid_dimension), n_vectors)
        self.grids = []
        for _ in range(max(1, n_grids)):
            # each grid indexes a small subset of the coordinates
            dims = sorted(rng.choice(n_vectors, size=dim, replace=False).tolist())
            offset = rng.uniform(0.0, cell_size, size=dim)
            self.grids.append(GridIndex(cell_size, dims=dims, offset=offset))
        # min number of grids in which the pair must co-occur (>=1 = union)
        self.freq_threshold = max(1, int(freq_threshold))

    def insert(self, key, vec):
        for grid in self.grids:
            grid.insert(key, vec)

    def query(self, vec):
        votes = Counter()
        for grid in self.grids:
            for key in set(grid.query(vec)):  # 1 vote max par grille
                votes[key] += 1
        return [key for key, count in votes.items() if count >= self.freq_threshold]

    def remove(self, key):
        for grid in self.grids:
            grid.remove(key)
