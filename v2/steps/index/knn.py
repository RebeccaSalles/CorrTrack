"""Index **k-Nearest-Neighbors** — `KNNIndex`.

Fundamental difference with the other indexes (tree/bptree/kdtree/quadtree/
octree/bst), which all perform a **radius query**: here the **k nearest
neighbours** are returned, with no distance threshold. The number of candidates
per query is therefore **fixed = k**, independent of the local density.

**Trade-off**:
  * +  predictable: exactly `k × n_series` candidates per window.
  * +  free of the radius choice (a very dense area -> explosion) and of a too
       small radius (collapsing recall).
  * −  distant pairs may be returned when the current window is isolated
       (but they are filtered by the Pearson validation anyway).

**Parameter:** `n_neighbors` (k). The larger k is, the more candidates and the
more recall; too small a k misses dense pairs.

**Implementation:** `numpy.argpartition` over the distance matrix (a single
batched BLAS call through `backend.cdist_batch` when the backend is
non-python), far faster than a full sort (`O(N)` vs `O(N log N)`).
"""

import numpy as np

from .base import SketchIndex


class KNNIndex(SketchIndex):
    def __init__(self, k, backend=None):
        self.k = max(1, int(k))
        self.backend = backend
        self.points = {}            # key -> vec

    def insert(self, key, vec):
        self.points[key] = np.asarray(vec, dtype=float).ravel()

    def remove(self, key):
        self.points.pop(key, None)

    def _knn(self, dists, keys):
        """Top-k indices through argpartition (O(N) instead of O(N log N))."""
        k = min(self.k, len(keys))
        if k <= 0:
            return []
        if k >= len(keys):
            return list(keys)
        idx = np.argpartition(dists, k - 1)[:k]
        return [keys[i] for i in idx]

    def query(self, vec):
        if not self.points:
            return []
        keys = list(self.points)
        P = np.stack(list(self.points.values()))
        if self.backend is not None:
            d = self.backend.distance_batch(vec, P)
        else:
            d = np.sqrt(((P - np.asarray(vec, dtype=float).ravel()) ** 2).sum(axis=1))
        return self._knn(d, keys)

    def query_batch(self, queries):
        if not self.points:
            return [[] for _ in queries]
        keys = list(self.points)
        P = np.stack(list(self.points.values()))
        Q = np.asarray(queries, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        if self.backend is not None:
            D = self.backend.cdist_batch(Q, P)
        else:
            D = np.sqrt(((Q[:, None] - P[None, :]) ** 2).sum(axis=2))
        k = min(self.k, D.shape[1])
        if k <= 0:
            return [[] for _ in queries]
        if k >= D.shape[1]:
            return [list(keys) for _ in range(D.shape[0])]
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        return [[keys[j] for j in idx[i]] for i in range(D.shape[0])]
