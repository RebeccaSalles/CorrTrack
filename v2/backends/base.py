"""Interface of the compute backends.

All the "hot" numeric logic goes through these primitives. To change technology
(numpy vectorized, parallel, Cython, CUDA), writing a subclass is enough: the
rest of the pipeline does not change.

`correlate(pairs)` is the batch primitive (the hot spot = correlating many
pairs): this is where vectorization and parallelization pay off.
"""

from abc import ABC, abstractmethod


class Backend(ABC):
    name = "abstract"

    def __init__(self, max_workers=0, cores="auto"):
        self.max_workers = max_workers  # used by the parallel backends
        self.cores = cores              # macOS P/E hint (perf|eco|auto), parallel backends

    @abstractmethod
    def project(self, matrix, values):
        """Project a window onto the random vectors -> raw sketch."""

    @abstractmethod
    def znormalize(self, vec):
        """Centre and scale a vector (z-score)."""

    @abstractmethod
    def pearson(self, x, y):
        """Pearson correlation between two raw windows (float)."""

    @abstractmethod
    def correlate(self, pairs):
        """Batch: list of (x, y) -> list of Pearson correlations."""

    @abstractmethod
    def cosine(self, u, v):
        """Cosine similarity between two sketches (float)."""

    @abstractmethod
    def distance(self, u, v):
        """Distance euclidienne entre deux sketches (float)."""

    def cosine_batch(self, pairs):
        """Batch: list of (u, v) -> list of cosine similarities.

        The counterpart of `correlate` for the sketch pre-filter: a single
        numpy batch instead of one `cosine` call per pair (the Python loop is
        the hot spot of `validate_sketches` when candidates are numerous;
        measured at ~9 s on a dense grid). Every sketch of a window has the same
        length -> stacking is valid. Zero norm -> 0.0 (consistent with
        `cosine`).
        """
        import numpy as np
        if not pairs:
            return []
        U = np.stack([np.asarray(u, dtype=float).ravel() for u, _ in pairs])
        V = np.stack([np.asarray(v, dtype=float).ravel() for _, v in pairs])
        with np.errstate(all="ignore"):
            num = np.einsum("ij,ij->i", U, V)
            nu = np.sqrt(np.einsum("ij,ij->i", U, U))
            nv = np.sqrt(np.einsum("ij,ij->i", V, V))
            den = nu * nv
            cos = np.where(den > 0.0, num / den, 0.0)
        return cos.tolist()

    def cosine_rows(self, U, V):
        """Row-wise cosine of two ALREADY aligned (n, d) matrices (gather).

        Array-native access: the sketches are passed already gathered through
        fancy indexing (`SK[ia]`, `SK[ib]`) instead of a list of pairs +
        `np.stack` (≈7× faster on large batches, see the sharded mode). Zero
        norm -> 0.0.
        """
        import numpy as np
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if U.size == 0:
            return np.empty(0)
        with np.errstate(all="ignore"):
            num = np.einsum("ij,ij->i", U, V)
            nu = np.sqrt(np.einsum("ij,ij->i", U, U))
            nv = np.sqrt(np.einsum("ij,ij->i", V, V))
            den = nu * nv
            return np.where(den > 0.0, num / den, 0.0)

    def correlate_rows(self, X, Y):
        """Row-wise Pearson of two ALREADY aligned (n, L) matrices (gather).

        Centring through einsum (and NOT `.mean(axis=1)`): on numpy 2.2.6 /
        py3.14, the `.mean`/`.sum(axis=)` reductions return WRONG values in
        function scope (same cause as the vectorized.correlate workaround).
        einsum is correct. Zero denominator -> nan.
        """
        import numpy as np
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.size == 0:
            return np.empty(0)
        n = X.shape[1]
        with np.errstate(all="ignore"):
            Xc = X - (np.einsum("ij->i", X) / n)[:, None]
            Yc = Y - (np.einsum("ij->i", Y) / n)[:, None]
            num = np.einsum("ij,ij->i", Xc, Yc)
            sxx = np.einsum("ij,ij->i", Xc, Xc)
            syy = np.einsum("ij,ij->i", Yc, Yc)
            den = np.sqrt(sxx * syy)
            return np.where(den > 0.0, num / den, np.nan)

    def distance_batch(self, q, points):
        """Euclidean distances from `q` (1D) to each row of `points` (2D).

        Default numpy implementation (used by python/vectorized/cython/
        parallel/coreml). MPS overrides it to compute on the GPU.
        """
        import numpy as np
        qv = np.asarray(q, dtype=float).ravel()
        P = np.asarray(points, dtype=float)
        if P.ndim == 1:
            P = P[None, :]
        return np.sqrt(((P - qv) ** 2).sum(axis=1))

    def median(self, values, axis=None):
        """Median of an array (optionally along an axis).

        Used by the median sketches (`steps/sketch_median.py`). Default numpy
        implementation (C/MKL introsort, already very fast). MPS overrides it
        with `torch.median` on the GPU.
        """
        import numpy as np
        return np.median(np.asarray(values, dtype=float), axis=axis)

    def fft_magnitude(self, values):
        """FFT magnitude of a real signal: `|rfft(values)|` (size `n//2+1`).

        Used by the FFT sketches (`steps/sketch_fft.py`). Default numpy
        implementation (MKL/Accelerate under the hood, already very fast). MPS
        overrides it with `torch.fft.rfft` on the GPU — a clear gain mostly on
        windows ≥ 1024, where the host↔device overhead is amortized.
        """
        import numpy as np
        v = np.asarray(values, dtype=float)
        return np.abs(np.fft.rfft(v))

    def cdist_batch(self, Q, P):
        """Euclidean distance matrix — `Q` (N_q, D) × `P` (N_p, D) → (N_q, N_p).

        The `||q-p||² = ||q||² + ||p||² − 2·q·p` trick → a single matmul
        (multi-threaded BLAS). Far more efficient than the `distance_batch`
        loop when there are several queries (1 call for a whole window instead
        of N). MPS overrides it with `torch.cdist` on the GPU (a single
        host↔device transfer).
        """
        import numpy as np
        Q = np.asarray(Q, dtype=float)
        P = np.asarray(P, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        if P.ndim == 1:
            P = P[None, :]
        qq = (Q * Q).sum(axis=1, keepdims=True)            # (N_q, 1)
        pp = (P * P).sum(axis=1)                            # (N_p,)
        # errstate: on numpy 2.2.6/py3.14 arm64, the matmul emits spurious
        # warnings (divide/overflow/invalid) although the result is correct.
        with np.errstate(all="ignore"):
            return np.sqrt(np.maximum(qq + pp - 2.0 * Q @ P.T, 0.0))
