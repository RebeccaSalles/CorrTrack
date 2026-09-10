"""Vectorized (numpy) backend — batched computation, no Python loops.

`correlate` computes the Pearson correlation of every pair at once (stacked
matrices). All the windows share the same length (window_size), so the stacking
is valid.
"""

import numpy as np

from .base import Backend


class VectorizedBackend(Backend):
    name = "vectorized"

    def project(self, matrix, values):
        # errstate: on numpy 2.2.6/py3.14 arm64, matmul emits SPURIOUS warnings
        # ("divide by zero", "overflow"…) although the result is correct
        # (checked == python, finite values). Every category is silenced.
        with np.errstate(all="ignore"):
            return matrix @ values

    def znormalize(self, vec):
        std = vec.std()
        if not np.isfinite(std) or std <= 0.0:
            return np.zeros_like(vec)
        return (vec - vec.mean()) / std

    def pearson(self, x, y):
        return self.correlate([(x, y)])[0]

    def correlate(self, pairs):
        if not pairs:
            return []
        X = np.stack([np.asarray(x, dtype=float) for x, _ in pairs])
        Y = np.stack([np.asarray(y, dtype=float) for _, y in pairs])
        Xc = X - X.mean(axis=1, keepdims=True)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        # NB: np.einsum is used (rather than .sum(axis=1)) for the row-wise
        # reduction — on Python 3.14 / numpy 2.2.6, `.sum(axis=1)` returned
        # wrong values in function scope; einsum is correct and just as fast.
        num = np.einsum("ij,ij->i", Xc, Yc)
        sxx = np.einsum("ij,ij->i", Xc, Xc)
        syy = np.einsum("ij,ij->i", Yc, Yc)
        den = np.sqrt(sxx * syy)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(den > 0.0, num / den, np.nan)
        return corr.tolist()

    def cosine(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return float(u @ v / (nu * nv))

    def distance(self, u, v):
        return float(np.linalg.norm(np.asarray(u, dtype=float) - np.asarray(v, dtype=float)))
