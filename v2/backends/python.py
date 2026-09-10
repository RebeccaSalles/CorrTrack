"""Sequential Python backend — the reference implementation.

Explicit loops (no vectorization), numpy used only as a container. Serves as
the reference and as the default behaviour.
"""

import math

import numpy as np

from .base import Backend


class PythonBackend(Backend):
    name = "python"

    def project(self, matrix, values):
        n_vectors, window_size = matrix.shape
        out = np.empty(n_vectors, dtype=float)
        for i in range(n_vectors):
            acc = 0.0
            row = matrix[i]
            for j in range(window_size):
                acc += row[j] * values[j]
            out[i] = acc
        return out

    def znormalize(self, vec):
        mean = vec.mean()
        std = vec.std()
        if not np.isfinite(std) or std <= 0.0:
            return np.zeros_like(vec)
        return (vec - mean) / std

    def pearson(self, x, y):
        n = len(x)
        if n == 0:
            return float("nan")
        mean_x = x.mean()
        mean_y = y.mean()
        cov = sxx = syy = 0.0
        for i in range(n):
            dx = x[i] - mean_x
            dy = y[i] - mean_y
            cov += dx * dy
            sxx += dx * dx
            syy += dy * dy
        denom = math.sqrt(sxx * syy)
        if denom == 0.0:
            return float("nan")
        return cov / denom

    def correlate(self, pairs):
        return [self.pearson(x, y) for x, y in pairs]

    def cosine(self, u, v):
        dot = nu = nv = 0.0
        for a, b in zip(u, v):
            dot += a * b
            nu += a * a
            nv += b * b
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return dot / math.sqrt(nu * nv)

    def distance(self, u, v):
        acc = 0.0
        for a, b in zip(u, v):
            acc += (a - b) ** 2
        return math.sqrt(acc)
