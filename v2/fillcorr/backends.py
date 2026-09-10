"""FilCorr compute backends.

They all expose the same API:

    backend = get_backend(name, max_workers=…)
    bands   = backend.band_fft(block, lb, ub)            # (n_series, m) -> (n_series, B) complex
    corrs   = backend.correlate_pairs(WX, WY)            # (N, B) × (N, B) -> (N,)

5 implementations:
- `python`    : O(N·B) pure-Python reference, naive (time-domain FFT).
- `vectorized`: numpy batched (batched FFT + einsum) — the fast default.
- `parallel`  : ProcessPool + `vectorized` (split into chunks).
- `cython`    : compiled kernel for the Parseval correlation; falls back to `vectorized`.
- `mps`       : `torch.fft` on Apple Silicon; CPU fallback.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

from .algo import (
    full_band_fft_batch,
    naive_filtered_pearson,
    parseval_corr_batch,
)

# --------------------------------------------------------------------- base


class FilCorrBackend:
    name = "abstract"

    def __init__(self, max_workers: int = 0, cores: str = "auto"):
        self.max_workers = max_workers
        self.cores = cores

    def band_fft(self, block: np.ndarray, lb: int, ub: int) -> np.ndarray:
        return full_band_fft_batch(block, lb, ub)

    def correlate_pairs(self, WX: np.ndarray, WY: np.ndarray) -> np.ndarray:
        return parseval_corr_batch(WX, WY)


# --------------------------------------------------------------------- python


class PythonFilCorrBackend(FilCorrBackend):
    """Reference: no batching, window-by-window computation.

    Uses `naive_filtered_pearson` (time-domain filtering through IFFT) — gives
    THE SAME result as the vectorized/Parseval backends up to floating-point
    precision. Serves as a cross-check.
    """

    name = "python"

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers=max_workers, cores=cores)
        self._raw_cache = None  # (window_size, lb, ub) -> nothing: the inter-API stays simple

    def band_fft(self, block, lb, ub):
        # The raw windows are stored in the "FFT" slot to stay API-compatible;
        # the correlation redoes the time-domain filtering to stay naive.
        return _RawWindows(np.asarray(block, dtype=float), lb, ub)

    def correlate_pairs(self, WX, WY):
        assert isinstance(WX, _RawWindows) and isinstance(WY, _RawWindows)
        lb, ub = WX.lb, WX.ub
        out = np.empty(len(WX.raw), dtype=float)
        for i in range(len(WX.raw)):
            out[i] = naive_filtered_pearson(WX.raw[i], WY.raw[i], lb, ub)
        return out


class _RawWindows:
    """Container used by the `python` backend to short-circuit the FFT step."""

    __slots__ = ("raw", "lb", "ub")

    def __init__(self, raw, lb, ub):
        self.raw, self.lb, self.ub = raw, lb, ub

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, key):
        return _RawWindows(self.raw[key], self.lb, self.ub)


# ---------------------------------------------------------------- vectorized


class VectorizedFilCorrBackend(FilCorrBackend):
    """numpy batched: batched FFT + einsum. Fast default implementation."""

    name = "vectorized"


# ------------------------------------------------------------------ parallel


def _correlate_chunk(args):
    WX, WY = args
    return parseval_corr_batch(WX, WY)


class ParallelFilCorrBackend(FilCorrBackend):
    """`vectorized` spread over N processes for the pairwise correlation.

    The band FFT is done *locally* (already very fast, numpy-vectorized); only
    the `correlate_pairs` call is parallelized (that is the hot path when there
    are many pairs).
    """

    name = "parallel"
    MIN_BATCH = 256  # below that, the pool overhead exceeds the gain

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers=max_workers, cores=cores)
        self._pool: Optional[ProcessPoolExecutor] = None

    def _workers(self):
        if self.max_workers > 0:
            return self.max_workers
        return max(1, (os.cpu_count() or 1) - 1)

    def correlate_pairs(self, WX, WY):
        n = len(WX)
        w = self._workers()
        if n < self.MIN_BATCH or w <= 1:
            return parseval_corr_batch(WX, WY)
        chunk = math.ceil(n / w)
        chunks = [(WX[i:i + chunk], WY[i:i + chunk]) for i in range(0, n, chunk)]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=w)
        parts = list(self._pool.map(_correlate_chunk, chunks))
        return np.concatenate(parts)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


# -------------------------------------------------------------------- cython


class CythonFilCorrBackend(FilCorrBackend):
    """Cython backend — compiled kernel for the Parseval correlation.

    When Cython is not installed OR the compilation fails, we fall back to
    `parseval_corr_batch` (numpy) → the code stays 100% functional.
    """

    name = "cython"

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers=max_workers, cores=cores)
        self._kernel = _try_load_cython_kernel()

    def correlate_pairs(self, WX, WY):
        if self._kernel is None:
            return parseval_corr_batch(WX, WY)
        WX = np.ascontiguousarray(WX, dtype=np.complex128)
        WY = np.ascontiguousarray(WY, dtype=np.complex128)
        return self._kernel.parseval_corr_batch(WX, WY)


def _try_load_cython_kernel():
    """Try to import (and if needed compile) the Cython module."""
    try:
        from . import _kernels  # type: ignore
        return _kernels
    except ImportError:
        pass
    try:
        import pyximport  # noqa: F401
        import Cython  # noqa: F401
    except ImportError:
        return None
    try:
        import pyximport
        pyximport.install(language_level=3,
                          setup_args={"include_dirs": [np.get_include()]})
        from . import _kernels  # type: ignore
        return _kernels
    except Exception:
        return None


# ----------------------------------------------------------------------- mps


class MpsFilCorrBackend(FilCorrBackend):
    """PyTorch + Apple Silicon GPU. Fallback CPU si torch/mps absent.

    Useful for large blocks or very long windows: the GPU FFT and the einsum
    product happen on the device, with a single host↔device transfer.
    """

    name = "mps"

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers=max_workers, cores=cores)
        self._torch = None
        self._device = None
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._torch = torch
                self._device = torch.device("mps")
        except ImportError:
            pass

    def band_fft(self, block, lb, ub):
        if self._torch is None:
            return full_band_fft_batch(block, lb, ub)
        t = self._torch.as_tensor(np.asarray(block, dtype=np.float32),
                                  device=self._device)
        W = self._torch.fft.fft(t, dim=-1)
        return W[:, lb:ub].cpu().numpy().astype(np.complex128)

    def correlate_pairs(self, WX, WY):
        if self._torch is None:
            return parseval_corr_batch(WX, WY)
        t_x = self._torch.as_tensor(np.asarray(WX, dtype=np.complex64),
                                    device=self._device)
        t_y = self._torch.as_tensor(np.asarray(WY, dtype=np.complex64),
                                    device=self._device)
        sxy = (t_x.conj() * t_y).sum(dim=-1).real
        sxx = (t_x.conj() * t_x).sum(dim=-1).real
        syy = (t_y.conj() * t_y).sum(dim=-1).real
        denom = (sxx * syy).sqrt()
        out = (sxy / denom).cpu().numpy().astype(float)
        out[~np.isfinite(out)] = float("nan")
        return out


# --------------------------------------------------------------------- registry


BACKENDS = {
    "python": PythonFilCorrBackend,
    "vectorized": VectorizedFilCorrBackend,
    "parallel": ParallelFilCorrBackend,
    "cython": CythonFilCorrBackend,
    "mps": MpsFilCorrBackend,
}


def get_backend(name: str, **kwargs) -> FilCorrBackend:
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown filcorr backend {name!r}. Choices: {sorted(BACKENDS)}")
    return cls(**kwargs)


def available():
    return sorted(BACKENDS)
