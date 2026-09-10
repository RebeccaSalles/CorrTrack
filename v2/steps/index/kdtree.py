"""n-dimensional KD-tree index — the "generic tree" of the sketches.

Radius query (same results as TreeIndex's linear scan, but with per-axis
pruning -> ~O(log n + k) instead of O(n) per query). Insertions and deletions
(window eviction) mark the tree "dirty"; it is lazily rebuilt at the next query
(median per axis = balanced).
"""

import numpy as np

from .base import SketchIndex


class _Node:
    __slots__ = ("idx", "axis", "left", "right")

    def __init__(self, idx, axis):
        self.idx = idx
        self.axis = axis
        self.left = None
        self.right = None


class KDTreeIndex(SketchIndex):
    def __init__(self, radius, backend=None):
        self.radius = radius
        self.backend = backend
        self.points = {}      # key -> vec (np.ndarray)
        self._dirty = True
        self._keys = []
        self._coords = None
        self._root = None

    def insert(self, key, vec):
        self.points[key] = np.asarray(vec, dtype=float)
        self._dirty = True

    def remove(self, key):
        if self.points.pop(key, None) is not None:
            self._dirty = True

    def _build(self):
        self._keys = list(self.points)
        if self._keys:
            self._coords = np.array([self.points[k] for k in self._keys], dtype=float)
            self._root = self._make(list(range(len(self._keys))), 0)
        else:
            self._coords, self._root = None, None
        self._dirty = False

    def _make(self, idxs, depth):
        if not idxs:
            return None
        axis = depth % self._coords.shape[1]
        idxs.sort(key=lambda i: self._coords[i, axis])
        mid = len(idxs) // 2
        node = _Node(idxs[mid], axis)
        node.left = self._make(idxs[:mid], depth + 1)
        node.right = self._make(idxs[mid + 1:], depth + 1)
        return node

    def query(self, vec):
        if self._dirty:
            self._build()
        if self._root is None:
            return []
        # non-python backend -> batched linear scan (result identical to the
        # tree, faster on small N thanks to vectorization/GPU).
        if self.backend is not None and getattr(self.backend, "name", "python") != "python":
            d = self.backend.distance_batch(vec, self._coords)
            return [self._keys[i] for i, ok in enumerate(d <= self.radius) if ok]
        q = np.asarray(vec, dtype=float)
        r2 = self.radius * self.radius
        out = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            c = self._coords[node.idx]
            d = c - q
            if d @ d <= r2:
                out.append(self._keys[node.idx])
            diff = q[node.axis] - c[node.axis]
            if diff * diff <= r2:        # the sphere crosses the plane -> both sides
                stack.append(node.left)
                stack.append(node.right)
            elif diff < 0:               # query left of the plane -> left side only
                stack.append(node.left)
            else:
                stack.append(node.right)
        return out

    def query_batch(self, queries):
        # non-python backend -> 1 batched cdist (skips the tree, faster at this
        # scale). python -> sequential loop (the tree stays valid).
        if self.backend is None or getattr(self.backend, "name", "python") == "python":
            return super().query_batch(queries)
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
