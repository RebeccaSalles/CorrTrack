"""bptree3D: 3D variant of `bptree` (sorted list + 1D bisect).

Projects the sketch into **3D** through a JL random projection
(sketch-of-sketch), then stores the triplets in a list sorted by the 1st coord.
The range query does:
  1. **bisect** on the 1st coord of the triplet → O(log N) 1D pre-filter.
  2. **refinement** through the 3D Euclidean distance `‖a−b‖ ≤ tree_radius`
     (on the projected coords, not on the full sketch).

Candidate set **identical to `octree`** (same radius query in projected 3D),
just implemented with a sorted list instead of a spatial subdivision. Faster to
insert (O(N) shift) than the octree for bursty inserts, and simpler to debug.
"""

import bisect

import numpy as np

from ._projection import make_random_projection
from .base import SketchIndex


class BPTree3DIndex(SketchIndex):
    def __init__(self, radius, backend=None, n_vectors=0, seed=0, proj_fn=None):
        self.radius = float(radius)
        self.backend = backend
        # configurable 3D projection (truncate/median/sum/random — see _keyfn).
        # Default = random JL (= octree behaviour).
        self._proj_fn = proj_fn or (lambda v, _R=make_random_projection(
            3, max(1, int(n_vectors or 1)), seed):
            _R @ np.asarray(v, dtype=float).ravel())
        self._entries = []     # list sorted by (proj[0], seq)
        self._meta = {}        # seq -> (key, proj_3d)
        self._key_seq = {}
        self._seq = 0

    def _proj(self, vec):
        return np.asarray(self._proj_fn(vec), dtype=float).ravel()[:3]

    def insert(self, key, vec):
        if key in self._key_seq:
            self.remove(key)
        p = self._proj(vec)
        seq = self._seq
        self._seq += 1
        bisect.insort(self._entries, (float(p[0]), seq))
        self._meta[seq] = (key, p)
        self._key_seq[key] = seq

    def remove(self, key):
        seq = self._key_seq.pop(key, None)
        if seq is None:
            return
        p0 = float(self._meta[seq][1][0])
        i = bisect.bisect_left(self._entries, (p0, seq))
        if i < len(self._entries) and self._entries[i] == (p0, seq):
            self._entries.pop(i)
        del self._meta[seq]

    def query(self, vec):
        if not self._entries:
            return []
        q = self._proj(vec)
        tau = self.radius
        r2 = tau * tau
        lo = bisect.bisect_left(self._entries, (float(q[0]) - tau, -1))
        hi = bisect.bisect_right(self._entries, (float(q[0]) + tau, self._seq))
        out = []
        for _v, seq in self._entries[lo:hi]:
            key, p = self._meta[seq]
            d = q - p
            if float(d @ d) <= r2:
                out.append(key)
        return out

    def query_batch(self, queries):
        if not self._entries:
            return [[] for _ in queries]
        keys = []
        P = []
        for _v, seq in self._entries:
            k, p = self._meta[seq]
            keys.append(k)
            P.append(p)
        P = np.stack(P)                                       # (N, 3)
        Q = np.stack([self._proj(q) for q in queries])        # (Nq, 3)
        D = (self.backend.cdist_batch(Q, P) if self.backend is not None
             else np.sqrt(((Q[:, None] - P[None, :]) ** 2).sum(axis=2)))
        return [[keys[j] for j, ok in enumerate(D[i] <= self.radius) if ok]
                for i in range(D.shape[0])]
