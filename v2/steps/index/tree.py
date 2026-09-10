"""Indexing backend: radius neighbourhood (PLACEHOLDER, linear scan).

To be replaced by a real tree (KD-tree / ball-tree). Keeps the interface so it
stays interchangeable with GridIndex. The distance computation goes through the
compute backend (python/cython/cuda swap).
"""

import numpy as np

from .base import SketchIndex


class TreeIndex(SketchIndex):
    def __init__(self, radius, backend):
        self.radius = radius
        self.backend = backend
        self.points = {}  # key -> vec

    def insert(self, key, vec):
        self.points[key] = np.asarray(vec, dtype=float)

    def query(self, vec):
        if not self.points:
            return []
        vec = np.asarray(vec, dtype=float)
        keys = list(self.points)
        P = np.stack([self.points[k] for k in keys])            # (N, D)
        d = self.backend.distance_batch(vec, P)                 # backend-batched
        return [k for k, ok in zip(keys, d <= self.radius) if ok]

    def query_batch(self, queries):
        if not self.points:
            return [[] for _ in queries]
        keys = list(self.points)
        P = np.stack([self.points[k] for k in keys])
        Q = np.asarray(queries, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        D = self.backend.cdist_batch(Q, P)
        return [[keys[j] for j, ok in enumerate(D[i] <= self.radius) if ok]
                for i in range(D.shape[0])]

    def remove(self, key):
        self.points.pop(key, None)
