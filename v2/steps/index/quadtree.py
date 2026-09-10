"""Quadtree index (2D) — recursive subdivision of the plane into 4 quadrants.

A quadtree is 2D by nature; since the sketches are high-dimensional, the first
`dims` sketch coordinates are indexed (2 by default, like `grid_dimension`).
For high dimension, prefer `kdtree` (generic n-dim tree).

Radius query over those 2 coordinates: returns the keys whose 2D projection is
at distance <= radius. The downstream Pearson validation guarantees precision;
the index only drives recall (like the grid).
"""

import numpy as np

from .base import SketchIndex

_CAP = 8  # capacity of a leaf before subdivision


class _QNode:
    __slots__ = ("cx", "cy", "hw", "items", "kids")

    def __init__(self, cx, cy, hw):
        self.cx, self.cy, self.hw = cx, cy, hw   # centre + half-width
        self.items = []                          # [(key, x, y)] if leaf
        self.kids = None                         # 4 child nodes once subdivided


class QuadTreeIndex(SketchIndex):
    def __init__(self, radius, backend=None, dims=2, n_vectors=0, seed=0):
        self.radius = radius
        self.backend = backend
        self.dims = dims
        self.points = {}      # key -> projected (x, y)
        self._dirty = True
        self._root = None
        # "sketch of sketch": when n_vectors > 0, 2D random projection of the
        # full sketch; otherwise fall back to truncation (sketch[:2]).
        if n_vectors:
            from ._projection import make_random_projection
            self._proj = make_random_projection(2, n_vectors, seed)
        else:
            self._proj = None

    def _to2(self, vec):
        v = np.asarray(vec, dtype=float).ravel()
        if self._proj is not None:
            p = self._proj @ v
            return float(p[0]), float(p[1])
        return float(v[0]), float(v[1]) if v.shape[0] > 1 else 0.0

    def insert(self, key, vec):
        self.points[key] = self._to2(vec)
        self._dirty = True

    def remove(self, key):
        if self.points.pop(key, None) is not None:
            self._dirty = True

    def _build(self):
        if not self.points:
            self._root = None
            self._dirty = False
            return
        xs = [p[0] for p in self.points.values()]
        ys = [p[1] for p in self.points.values()]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        hw = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 1e-9
        self._root = _QNode(cx, cy, hw)
        for key, (x, y) in self.points.items():
            self._insert_node(self._root, key, x, y)
        self._dirty = False

    def _insert_node(self, node, key, x, y):
        if node.kids is None:
            node.items.append((key, x, y))
            if len(node.items) > _CAP and node.hw > 1e-6:
                self._subdivide(node)
            return
        self._insert_node(self._child(node, x, y), key, x, y)

    def _subdivide(self, node):
        h = node.hw / 2
        node.kids = [
            _QNode(node.cx - h, node.cy - h, h), _QNode(node.cx + h, node.cy - h, h),
            _QNode(node.cx - h, node.cy + h, h), _QNode(node.cx + h, node.cy + h, h),
        ]
        items, node.items = node.items, []
        for key, x, y in items:
            self._insert_node(self._child(node, x, y), key, x, y)

    def _child(self, node, x, y):
        return node.kids[(1 if x >= node.cx else 0) + (2 if y >= node.cy else 0)]

    def query(self, vec):
        if self._dirty:
            self._build()
        if self._root is None:
            return []
        qx, qy = self._to2(vec)         # 2D projection (random or truncation)
        v = np.array([qx, qy])
        # non-python backend -> batched linear scan of the 2D points (result
        # identical to the quadtree, accelerated by numpy/cython/mps).
        if self.backend is not None and getattr(self.backend, "name", "python") != "python":
            keys = list(self.points)
            P = np.asarray([self.points[k] for k in keys], dtype=float)
            d = self.backend.distance_batch(v, P)
            return [k for k, ok in zip(keys, d <= self.radius) if ok]
        qx, qy = float(v[0]), float(v[1]) if v.shape[0] > 1 else 0.0
        r = self.radius
        r2 = r * r
        out = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            # pruning: the ball [qx,qy]±r must cross the node square
            if abs(qx - node.cx) > node.hw + r or abs(qy - node.cy) > node.hw + r:
                continue
            if node.kids is None:
                for key, x, y in node.items:
                    if (x - qx) ** 2 + (y - qy) ** 2 <= r2:
                        out.append(key)
            else:
                stack.extend(node.kids)
        return out

    def query_batch(self, queries):
        if not self.points:
            return [[] for _ in queries]
        keys = list(self.points)
        P = np.asarray([self.points[k] for k in keys], dtype=float)  # (N, 2)
        Q2 = np.asarray([self._to2(q) for q in queries], dtype=float)  # projection 2D
        if Q2.ndim == 1:
            Q2 = Q2[None, :]
        D = self.backend.cdist_batch(Q2, P) if self.backend is not None \
            else np.sqrt(((Q2[:, None] - P[None, :]) ** 2).sum(axis=2))
        return [[keys[j] for j, ok in enumerate(D[i] <= self.radius) if ok]
                for i in range(D.shape[0])]
