"""B+tree index (port of the v1 `BalancedIndex`, candidate_kernels.py).

A list **sorted by the 1st sketch coordinate** (1D key), queried through
`bisect` over the interval `[v0-tau, v0+tau]` (fast pre-filter = necessary
condition), then **refined with the full Euclidean distance** `||a-b|| <= tau`.
The candidate set is therefore identical to a radius query (like kdtree/tree),
but found through a sorted 1D index — this is the index used by v1
(`candidate_backend = "bptree"`). `tau = config.tree_radius`.
"""

import bisect

import numpy as np

from .base import MatrixStore, SketchIndex


class BPTreeIndex(SketchIndex):
    def __init__(self, radius, backend=None, key_fn=None):
        self.radius = float(radius)
        self.backend = backend
        # 1D key for the bisect: sketch[0] by default (v1 mode = truncate),
        # or injected by `make_index` per `index_key_mode` (median/sum/random).
        self.key_fn = key_fn or (lambda v: float(v[0]))
        self._entries = []     # sorted list of (value, seq); value = key_fn(sketch)
        self._meta = {}        # seq -> (key, vec)
        self._key_seq = {}     # key -> seq (for remove / re-insertion)
        self._seq = 0
        # matrix store for query_batch (avoids the per-window restack).
        self._mstore = MatrixStore()

    def insert(self, key, vec):
        v = np.asarray(vec, dtype=float).ravel()
        if key in self._key_seq:           # re-insertion -> replace
            self.remove(key)
        seq = self._seq
        self._seq += 1
        bisect.insort(self._entries, (self.key_fn(v), seq))
        self._meta[seq] = (key, v)
        self._key_seq[key] = seq
        self._mstore.add(key, v)

    def remove(self, key):
        seq = self._key_seq.pop(key, None)
        if seq is None:
            return
        val = self.key_fn(self._meta[seq][1])
        i = bisect.bisect_left(self._entries, (val, seq))
        if i < len(self._entries) and self._entries[i] == (val, seq):
            self._entries.pop(i)
        del self._meta[seq]
        self._mstore.remove(key)

    def bulk_load(self, items):
        # mode batch : query_batch n'utilise QUE le MatrixStore -> on saute le
        # bisect.insort (O(N²)) et le _meta/_key_seq. query()/remove() KO ensuite.
        for key, vec in items:
            self._mstore.add(key, np.asarray(vec, dtype=float).ravel())

    def query(self, vec):
        if not self._entries:
            return []
        q = np.asarray(vec, dtype=float).ravel()
        tau = self.radius
        v0 = self.key_fn(q)
        lo = bisect.bisect_left(self._entries, (v0 - tau, -1))
        hi = bisect.bisect_right(self._entries, (v0 + tau, self._seq))
        if lo == hi:
            return []
        # full-vector refinement, BATCHED through the backend (vectorized/cython/mps).
        cands = [self._meta[seq] for _val, seq in self._entries[lo:hi]]
        keys = [c[0] for c in cands]
        P = np.stack([c[1] for c in cands])
        if self.backend is not None:
            d = self.backend.distance_batch(q, P)
        else:
            d = np.sqrt(((P - q) ** 2).sum(axis=1))
        return [k for k, ok in zip(keys, d <= tau) if ok]

    def query_batch(self, queries):
        if len(self._mstore) == 0:
            return [[] for _ in queries]
        # 1 cdist over ALL the pairs (the queries give up the bisect, but gain
        # multi-threaded BLAS + 1 MPS transfer instead of N).
        # P and keys come from the matrix store (no per-window restack).
        keys = self._mstore.keys
        P = self._mstore.matrix
        Q = np.asarray(queries, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        D = self.backend.cdist_batch(Q, P) if self.backend is not None \
            else np.sqrt(((Q[:, None] - P[None, :]) ** 2).sum(axis=2))
        return self.candidates_from_dist(D, self.radius, keys)
