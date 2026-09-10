"""Annoy — Approximate Nearest Neighbors Oh Yeah (Spotify, Bernhardsson).

**Forest of N binary trees**; each tree splits on a **random hyperplane**
(true projection LSH). At query time:
  1. descend each tree towards the leaf containing the query,
  2. the union of the leaves over every tree = candidate set,
  3. refine through the full Euclidean distance ≤ `tree_radius`.

Forest = robustness: a single tree is random (it misses boundary neighbours),
N trees = OR vote (each neighbour has a `1 − p^N` chance of being caught, p =
probability of being missed by one tree). Typical Pareto significantly better
than `grid` at equal recall.

Parameters:
  * `n_trees` (10 by default, `n_grids`): larger = better recall.
  * `leaf_size` (32): below that size a leaf is a linear scan.
  * `tree_radius`: post-collection validation radius.
"""

import numpy as np

from .base import SketchIndex


class _Leaf:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids       # list of internal indices


class _Split:
    __slots__ = ("normal", "thr", "left", "right")

    def __init__(self, normal, thr):
        self.normal = normal   # hyperplan (D,)
        self.thr = thr         # offset scalaire
        self.left = None
        self.right = None


class AnnoyIndex(SketchIndex):
    def __init__(self, radius, backend=None, n_trees=10, leaf_size=32, seed=0):
        self.radius = float(radius)
        self.backend = backend
        self.n_trees = max(1, int(n_trees))
        self.leaf_size = max(4, int(leaf_size))
        self._rng = np.random.default_rng(int(seed))
        self.points = {}                              # key -> vec
        self._dirty = True
        self._keys = []
        self._coords = None
        self._roots = []                              # n_trees racines

    def insert(self, key, vec):
        self.points[key] = np.asarray(vec, dtype=float)
        self._dirty = True

    def remove(self, key):
        if self.points.pop(key, None) is not None:
            self._dirty = True

    def _build(self):
        self._keys = list(self.points)
        if not self._keys:
            self._coords, self._roots = None, []
            self._dirty = False
            return
        self._coords = np.stack([self.points[k] for k in self._keys])
        self._roots = [self._make_tree(list(range(len(self._keys))))
                       for _ in range(self.n_trees)]
        self._dirty = False

    def _make_tree(self, idxs):
        if len(idxs) <= self.leaf_size:
            return _Leaf(idxs)
        # random hyperplane = vector between two randomly drawn points
        # (Annoy style — not a generic N(0,1) vector).
        i, j = self._rng.choice(len(idxs), size=2, replace=False)
        a, b = self._coords[idxs[int(i)]], self._coords[idxs[int(j)]]
        normal = a - b
        nn = np.linalg.norm(normal)
        if nn < 1e-12:
            return _Leaf(idxs)                        # points identiques
        normal = normal / nn
        proj = self._coords[idxs] @ normal
        thr = float(np.median(proj))
        left_idxs = [idxs[k] for k in range(len(idxs)) if proj[k] < thr]
        right_idxs = [idxs[k] for k in range(len(idxs)) if proj[k] >= thr]
        if not left_idxs or not right_idxs:           # degenerate split
            return _Leaf(idxs)
        node = _Split(normal, thr)
        node.left = self._make_tree(left_idxs)
        node.right = self._make_tree(right_idxs)
        return node

    def _descend(self, root, q):
        """Walk the query down to its leaf (a single path)."""
        node = root
        while isinstance(node, _Split):
            node = node.left if (q @ node.normal) < node.thr else node.right
        return node.ids

    def query(self, vec):
        if self._dirty:
            self._build()
        if not self._roots:
            return []
        q = np.asarray(vec, dtype=float).ravel()
        # union of the leaves over every tree
        ids = set()
        for root in self._roots:
            ids.update(self._descend(root, q))
        if not ids:
            return []
        ids = list(ids)
        coords = self._coords[ids]
        d = (self.backend.distance_batch(q, coords) if self.backend is not None
             else np.linalg.norm(coords - q, axis=1))
        return [self._keys[ids[i]] for i, ok in enumerate(d <= self.radius) if ok]

    def query_batch(self, queries):
        # the descent stays sequential; ONLY the refinement is batched
        if self._dirty:
            self._build()
        if not self._roots:
            return [[] for _ in queries]
        out = []
        for q in queries:
            qa = np.asarray(q, dtype=float).ravel()
            ids = set()
            for root in self._roots:
                ids.update(self._descend(root, qa))
            if not ids:
                out.append([])
                continue
            ids = list(ids)
            coords = self._coords[ids]
            d = (self.backend.distance_batch(qa, coords) if self.backend is not None
                 else np.linalg.norm(coords - qa, axis=1))
            out.append([self._keys[ids[i]] for i, ok in enumerate(d <= self.radius) if ok])
        return out
