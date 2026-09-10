"""bst3D : variante 3D du `bst` (vrai arbre binaire AVL).

Sketch projected into 3D through a JL random projection (sketch-of-sketch). The
AVL is ordered by the 1st projected coord (for rotations/insertions), the
in-order range query filters on [v₀−τ, v₀+τ] in 1D, then refines through the 3D
distance. Candidate set identical to `octree`/`bptree3d`; AVL = guaranteed
O(log N) insertion/deletion (against O(N) for bptree).
"""

import numpy as np

from ._projection import make_random_projection
from .base import SketchIndex
from .bst import _Node, _insert, _remove, _range


class BST3DIndex(SketchIndex):
    def __init__(self, radius, backend=None, n_vectors=0, seed=0, proj_fn=None):
        self.radius = float(radius)
        self.backend = backend
        # configurable 3D projection (see bptree3d).
        self._proj_fn = proj_fn or (lambda v, _R=make_random_projection(
            3, max(1, int(n_vectors or 1)), seed):
            _R @ np.asarray(v, dtype=float).ravel())
        self._root = None
        self._key_loc = {}      # key -> (proj_0, seq)
        self._proj_full = {}    # seq -> proj_3d
        self._seq = 0

    def _proj(self, vec):
        return np.asarray(self._proj_fn(vec), dtype=float).ravel()[:3]

    def insert(self, key, vec):
        if key in self._key_loc:
            self.remove(key)
        p = self._proj(vec)
        val = float(p[0])
        seq = self._seq
        self._seq += 1
        self._key_loc[key] = (val, seq)
        self._proj_full[seq] = p
        # The 3D projection is stored in _Node's `vec` (retrieved later).
        self._root = _insert(self._root, val, seq, p, key)

    def remove(self, key):
        loc = self._key_loc.pop(key, None)
        if loc is None:
            return
        val, seq = loc
        self._proj_full.pop(seq, None)
        self._root = _remove(self._root, val, seq)

    def query(self, vec):
        if self._root is None:
            return []
        q = self._proj(vec)
        tau = self.radius
        r2 = tau * tau
        v0 = float(q[0])
        out_pairs = []
        _range(self._root, v0 - tau, v0 + tau, out_pairs)    # [(key, vec_3D), …]
        if not out_pairs:
            return []
        # 3D refinement over the returned block
        keys = [k for k, _ in out_pairs]
        P = np.stack([v for _, v in out_pairs])
        d = (self.backend.distance_batch(q, P) if self.backend is not None
             else np.sqrt(((P - q) ** 2).sum(axis=1)))
        return [k for k, ok in zip(keys, d <= tau) if ok]

    def query_batch(self, queries):
        if self._root is None:
            return [[] for _ in queries]
        # in-order traversal, once → every point
        items = []
        stack, n = [], self._root
        while stack or n:
            while n is not None:
                stack.append(n); n = n.l
            n = stack.pop()
            items.append((n.key, n.vec))
            n = n.r
        keys = [k for k, _ in items]
        P = np.stack([v for _, v in items])                  # (N, 3)
        Q = np.stack([self._proj(q) for q in queries])       # (Nq, 3)
        D = (self.backend.cdist_batch(Q, P) if self.backend is not None
             else np.sqrt(((Q[:, None] - P[None, :]) ** 2).sum(axis=2)))
        return [[keys[j] for j, ok in enumerate(D[i] <= self.radius) if ok]
                for i in range(D.shape[0])]
