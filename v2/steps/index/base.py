"""Common interface of the sketch indexing structures."""

from abc import ABC, abstractmethod


class MatrixStore:
    """(keys, vectors) store backed by an ndarray, for `query_batch`.

    Maintains a `P` (n, D) matrix with amortized growth (capacity doubling) +
    O(1) deletion through swap-with-last. Avoids the `np.stack` of ALL the
    vectors at each window (hot spot of `select` as measured by the profiler:
    ~0.4 s of restack on an index of a few thousand entries).

    `keys[j]` is aligned with row `j` of `matrix`.
    """

    def __init__(self):
        import numpy as np
        self._np = np
        self._P = None          # ndarray (cap, D), or None while empty
        self._n = 0             # rows in use
        self._cap = 0
        self._keys = []         # key per row (aligned)
        self._pos = {}          # key -> row

    def __len__(self):
        return self._n

    def add(self, key, vec):
        np = self._np
        v = np.asarray(vec, dtype=float).ravel()
        if key in self._pos:        # re-insertion -> replace
            self.remove(key)
        if self._n == self._cap:    # amortized growth
            cap = 8 if self._cap == 0 else self._cap * 2
            P = np.empty((cap, v.shape[0]), dtype=float)
            if self._P is not None:
                P[:self._n] = self._P[:self._n]
            self._P, self._cap = P, cap
        self._P[self._n] = v
        self._pos[key] = self._n
        self._keys.append(key)
        self._n += 1

    def remove(self, key):
        pos = self._pos.pop(key, None)
        if pos is None:
            return
        last = self._n - 1
        if pos != last:                         # swap-with-last
            self._P[pos] = self._P[last]
            moved = self._keys[last]
            self._keys[pos] = moved
            self._pos[moved] = pos
        self._keys.pop()
        self._n -= 1

    @property
    def keys(self):
        return self._keys

    @property
    def matrix(self):
        """(n, D) view of the rows in use (no copy)."""
        return None if self._P is None else self._P[:self._n]


class SketchIndex(ABC):
    @abstractmethod
    def insert(self, key, vec):
        """Index the sketch `vec` under the identifier `key`."""

    @abstractmethod
    def query(self, vec):
        """Return the list of candidate keys neighbouring `vec`."""

    @abstractmethod
    def remove(self, key):
        """Remove `key` from the index (eviction of the old windows)."""

    def query_batch(self, queries):
        """Return a list of candidate lists (1 per query).

        Default = sequential loop over `self.query`. The radius indexes (tree/
        bptree/kdtree/quadtree/octree/bst) override it with a single
        `backend.cdist_batch` call → 1 BLAS matmul instead of N passes (and 1
        GPU transfer instead of N for MPS).
        """
        return [self.query(q) for q in queries]

    def bulk_load(self, items):
        """Bulk-load a list of `(key, vec)` to prepare `query_batch`.

        Default = `insert()` one by one (valid for every index). The
        `MatrixStore` indexes (bptree/bst) override it to fill ONLY the matrix
        store — the search structure (sorted list / tree) is only useful for a
        unit `query()`, which is unused in window-parallel batch mode, and its
        insertion is expensive (bptree `insort` O(N²)). After `bulk_load`, only
        `query_batch` is guaranteed; `query`/`remove` are not.
        """
        for key, vec in items:
            self.insert(key, vec)

    @staticmethod
    def candidates_from_dist(D, radius, keys):
        """Group a distance matrix `D` (n_q, n_p) into candidate key lists
        (1 per query), `||q-p|| <= radius`.

        Vectorized through `np.nonzero`: only walks the RETAINED entries (at the
        C level) instead of scanning the whole boolean matrix in Python — the
        real hot spot of `select` when the index is large (see profiling). The
        candidate order per query (increasing j) is preserved.
        """
        import numpy as np
        mask = np.asarray(D) <= radius
        out = [[] for _ in range(mask.shape[0])]
        ii, jj = np.nonzero(mask)
        for i, j in zip(ii.tolist(), jj.tolist()):
            out[i].append(keys[j])
        return out
