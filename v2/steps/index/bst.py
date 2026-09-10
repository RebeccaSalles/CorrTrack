"""Index backed by a **real AVL binary search tree** (self-balancing).

The "honest tree" variant of `bptree` (which is a sorted list + bisect despite
its name). 1D key = `sketch[0]`. The AVL guarantees **O(log N) everywhere**
(insertion, deletion, range query) thanks to the rotations, where `bptree` pays
O(N) on insertion (array shift).

Candidate result IDENTICAL to `bptree`/`kdtree`/`tree` (same pairs within the
full Euclidean radius; the AVL only provides the 1D pre-filter, followed by the
full-vector refinement through `backend.distance_batch`).
"""

import numpy as np

from .base import MatrixStore, SketchIndex


class _Node:
    __slots__ = ("v", "seq", "vec", "key", "l", "r", "h")

    def __init__(self, v, seq, vec, key):
        self.v, self.seq, self.vec, self.key = v, seq, vec, key
        self.l = self.r = None
        self.h = 1


def _h(n): return n.h if n else 0
def _upd(n): n.h = 1 + max(_h(n.l), _h(n.r))
def _bal(n): return _h(n.l) - _h(n.r) if n else 0


def _rot_r(y):
    x = y.l; y.l = x.r; x.r = y
    _upd(y); _upd(x)
    return x


def _rot_l(x):
    y = x.r; x.r = y.l; y.l = x
    _upd(x); _upd(y)
    return y


def _rebalance(n):
    _upd(n)
    b = _bal(n)
    if b > 1 and _bal(n.l) >= 0: return _rot_r(n)
    if b > 1: n.l = _rot_l(n.l); return _rot_r(n)
    if b < -1 and _bal(n.r) <= 0: return _rot_l(n)
    if b < -1: n.r = _rot_r(n.r); return _rot_l(n)
    return n


def _insert(n, v, seq, vec, key):
    if n is None:
        return _Node(v, seq, vec, key)
    if (v, seq) < (n.v, n.seq):
        n.l = _insert(n.l, v, seq, vec, key)
    else:
        n.r = _insert(n.r, v, seq, vec, key)
    return _rebalance(n)


def _min(n):
    while n.l is not None:
        n = n.l
    return n


def _remove(n, v, seq):
    if n is None:
        return None
    k, nk = (v, seq), (n.v, n.seq)
    if k < nk:
        n.l = _remove(n.l, v, seq)
    elif k > nk:
        n.r = _remove(n.r, v, seq)
    else:
        if n.l is None: return n.r
        if n.r is None: return n.l
        succ = _min(n.r)
        n.v, n.seq, n.vec, n.key = succ.v, succ.seq, succ.vec, succ.key
        n.r = _remove(n.r, succ.v, succ.seq)
    return _rebalance(n)


def _range(n, lo, hi, out):
    """In-order traversal du sous-arbre intersectant [lo, hi]."""
    if n is None:
        return
    if n.v >= lo:
        _range(n.l, lo, hi, out)
    if lo <= n.v <= hi:
        out.append((n.key, n.vec))
    if n.v <= hi:
        _range(n.r, lo, hi, out)


class BSTIndex(SketchIndex):
    def __init__(self, radius, backend=None, key_fn=None):
        self.radius = float(radius)
        self.backend = backend
        # 1D key for the AVL comparison; see bptree.
        self.key_fn = key_fn or (lambda v: float(v[0]))
        self._root = None
        self._key_loc = {}     # key -> (value, seq) pour suppression
        self._seq = 0
        # matrix store for query_batch (avoids traversal + per-window restack).
        self._mstore = MatrixStore()

    def insert(self, key, vec):
        v = np.asarray(vec, dtype=float).ravel()
        if key in self._key_loc:
            self.remove(key)
        val = self.key_fn(v)
        seq = self._seq
        self._seq += 1
        self._key_loc[key] = (val, seq)
        self._root = _insert(self._root, val, seq, v, key)
        self._mstore.add(key, v)

    def remove(self, key):
        loc = self._key_loc.pop(key, None)
        if loc is None:
            return
        val, seq = loc
        self._root = _remove(self._root, val, seq)
        self._mstore.remove(key)

    def bulk_load(self, items):
        # mode batch : query_batch n'utilise QUE le MatrixStore -> on saute
        # l'insertion AVL (O(N log N)) et le _key_loc. query()/remove() KO ensuite.
        for key, vec in items:
            self._mstore.add(key, np.asarray(vec, dtype=float).ravel())

    def query(self, vec):
        if self._root is None:
            return []
        q = np.asarray(vec, dtype=float).ravel()
        tau = self.radius
        v0 = self.key_fn(q)
        out = []
        _range(self._root, v0 - tau, v0 + tau, out)        # range 1D O(log N + k)
        if not out:
            return []
        keys = [k for k, _ in out]
        P = np.stack([v for _, v in out])
        if self.backend is not None:
            d = self.backend.distance_batch(q, P)
        else:
            d = np.sqrt(((P - q) ** 2).sum(axis=1))
        return [k for k, ok in zip(keys, d <= tau) if ok]

    def query_batch(self, queries):
        # P and keys come from the matrix store (no traversal, no restack).
        if len(self._mstore) == 0:
            return [[] for _ in queries]
        keys = self._mstore.keys
        P = self._mstore.matrix
        Q = np.asarray(queries, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        D = self.backend.cdist_batch(Q, P) if self.backend is not None \
            else np.sqrt(((Q[:, None] - P[None, :]) ** 2).sum(axis=2))
        return self.candidates_from_dist(D, self.radius, keys)
