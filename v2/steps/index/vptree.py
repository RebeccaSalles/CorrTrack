"""VP-tree (Vantage-Point tree) — Yianilos 1993.

Each node picks a **pivot point** and partitions the others into:
  * `inside`  : at distance `< median_dist(pivot, ·)`,
  * `outside` : at distance `>= median_dist`.

Radius query `r`: pruning through the **triangle inequality** —
`|dist(q, pivot) − median| > r` lets a whole side be eliminated.

Advantages over the kd-tree in high dimension: no axis-aligned split (which
degenerates when n_vec > 20), works for any metric distance. Lazy rebuild like
kdtree/octree (median per axis → balanced tree).
"""

import numpy as np

from .base import SketchIndex


class _VPNode:
    __slots__ = ("idx", "thr", "inside", "outside")

    def __init__(self, idx, thr=0.0):
        self.idx = idx        # index of the pivot inside coords[]
        self.thr = thr        # radius (median dist) separating inside/outside
        self.inside = None    # sub-tree of the points at dist < thr
        self.outside = None   # sub-tree of the points at dist >= thr


class VPTreeIndex(SketchIndex):
    def __init__(self, radius, backend=None, seed=0):
        self.radius = float(radius)
        self.backend = backend
        self._rng = np.random.default_rng(int(seed))
        self.points = {}      # key -> vec
        self._dirty = True
        self._keys = []
        self._coords = None   # (N, D) float64
        self._root = None

    def insert(self, key, vec):
        self.points[key] = np.asarray(vec, dtype=float)
        self._dirty = True

    def remove(self, key):
        if self.points.pop(key, None) is not None:
            self._dirty = True

    def _build(self):
        self._keys = list(self.points)
        if not self._keys:
            self._coords, self._root = None, None
            self._dirty = False
            return
        self._coords = np.stack([self.points[k] for k in self._keys])
        self._root = self._make(list(range(len(self._keys))))
        self._dirty = False

    def _make(self, idxs):
        if not idxs:
            return None
        # randomly drawn pivot
        p = int(self._rng.integers(0, len(idxs)))
        pivot = idxs[p]
        rest = idxs[:p] + idxs[p + 1:]
        if not rest:
            return _VPNode(pivot, 0.0)
        # distances pivot ↔ rest
        d = np.linalg.norm(self._coords[rest] - self._coords[pivot], axis=1)
        median = float(np.median(d))
        in_mask = d < median
        inside = [rest[i] for i in range(len(rest)) if in_mask[i]]
        outside = [rest[i] for i in range(len(rest)) if not in_mask[i]]
        node = _VPNode(pivot, median)
        node.inside = self._make(inside)
        node.outside = self._make(outside)
        return node

    def query(self, vec):
        if self._dirty:
            self._build()
        if self._root is None:
            return []
        q = np.asarray(vec, dtype=float).ravel()
        r = self.radius
        out = []
        stack = [self._root]
        while stack:
            n = stack.pop()
            if n is None:
                continue
            d = float(np.linalg.norm(self._coords[n.idx] - q))
            if d <= r:
                out.append(self._keys[n.idx])
            # pruning through the triangle inequality
            if d - r <= n.thr:           # peut intersecter inside
                stack.append(n.inside)
            if d + r >= n.thr:           # peut intersecter outside
                stack.append(n.outside)
        return out

    def query_batch(self, queries):
        # non-python backend -> 1 cdist over every coord (skips the tree).
        if self.backend is not None and getattr(self.backend, "name", "python") != "python":
            if self._dirty:
                self._build()
            if self._root is None:
                return [[] for _ in queries]
            Q = np.asarray(queries, dtype=float)
            if Q.ndim == 1:
                Q = Q[None, :]
            D = self.backend.cdist_batch(Q, self._coords)
            return [[self._keys[j] for j, ok in enumerate(D[i] <= self.radius) if ok]
                    for i in range(D.shape[0])]
        return super().query_batch(queries)
