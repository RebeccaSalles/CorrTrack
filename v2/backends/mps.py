"""Backend Apple Silicon — GPU via Metal Performance Shaders (PyTorch `mps`).

The hot spot (`correlate`: Pearson over every pair) and the projection run on
the Apple Silicon GPU in unified memory (torch device="mps"). **Transparent**
fallback to the numpy vectorized backend when PyTorch is not installed or the
mps device is unavailable (a single warning).

  pip install torch   (official wheels with mps support on macOS arm64)

NB: the Neural Engine (ANE) is NOT targeted — it is only reachable through
CoreML and does not suit these kernels; mps targets the GPU, the right
accelerator here.
"""

import warnings

from .vectorized import VectorizedBackend

_TORCH = None
_TRIED = False


def _torch_mps():
    """The torch module when mps is usable, otherwise None (numpy fallback)."""
    global _TORCH, _TRIED
    if _TRIED:
        return _TORCH
    _TRIED = True
    try:
        import torch
        if torch.backends.mps.is_available():
            _TORCH = torch
        else:
            raise RuntimeError("device mps indisponible")
    except Exception as exc:
        warnings.warn(
            f"MpsBackend: torch/mps unavailable ({exc}) — falling back to the numpy "
            "vectorized backend. `pip install torch` for the Apple Silicon GPU.",
            RuntimeWarning, stacklevel=2,
        )
        _TORCH = None
    return _TORCH


class MpsBackend(VectorizedBackend):
    name = "mps"

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers, cores)
        self._torch = _torch_mps()
        self._dev = "mps" if self._torch is not None else None

    def distance_batch(self, q, points):
        if self._torch is None:
            return super().distance_batch(q, points)
        t = self._torch
        qt = t.as_tensor(q, dtype=t.float32, device=self._dev)
        Pt = t.as_tensor(points, dtype=t.float32, device=self._dev)
        return ((Pt - qt) ** 2).sum(dim=1).sqrt().cpu().numpy()

    def fft_magnitude(self, values):
        """`|torch.fft.rfft(values)|` on the MPS device (numpy fallback without torch)."""
        if self._torch is None:
            return super().fft_magnitude(values)
        t = self._torch
        vt = t.as_tensor(values, dtype=t.float32, device=self._dev)
        return t.abs(t.fft.rfft(vt)).cpu().numpy().astype(float)

    def median(self, values, axis=None):
        """`torch.median(values, dim=axis)` on GPU (numpy fallback without torch).

        On MPS the median computation stays fast; the net gain depends on the
        window size (the host↔device overhead dominates on small ones).
        """
        if self._torch is None:
            return super().median(values, axis=axis)
        t = self._torch
        vt = t.as_tensor(values, dtype=t.float32, device=self._dev)
        if axis is None:
            return float(t.median(vt).cpu().item())
        # torch.median(dim=...) returns (values, indices); only values is kept.
        return t.median(vt, dim=int(axis)).values.cpu().numpy().astype(float)

    def cdist_batch(self, Q, P):
        if self._torch is None:
            return super().cdist_batch(Q, P)
        t = self._torch
        Qt = t.as_tensor(Q, dtype=t.float32, device=self._dev)
        Pt = t.as_tensor(P, dtype=t.float32, device=self._dev)
        if Qt.ndim == 1: Qt = Qt.unsqueeze(0)
        if Pt.ndim == 1: Pt = Pt.unsqueeze(0)
        return t.cdist(Qt, Pt).cpu().numpy()                  # un seul transfert

    def project(self, matrix, values):
        if self._torch is None:
            return super().project(matrix, values)
        t = self._torch
        R = t.as_tensor(matrix, dtype=t.float32, device=self._dev)
        v = t.as_tensor(values, dtype=t.float32, device=self._dev)
        return (R @ v).cpu().numpy()

    def correlate(self, pairs):
        if not pairs:
            return []
        if self._torch is None:
            return super().correlate(pairs)
        t = self._torch
        import numpy as np
        X = t.as_tensor(np.stack([np.asarray(x, float) for x, _ in pairs]),
                        dtype=t.float32, device=self._dev)
        Y = t.as_tensor(np.stack([np.asarray(y, float) for _, y in pairs]),
                        dtype=t.float32, device=self._dev)
        Xc = X - X.mean(dim=1, keepdim=True)
        Yc = Y - Y.mean(dim=1, keepdim=True)
        num = (Xc * Yc).sum(dim=1)
        den = t.sqrt((Xc * Xc).sum(dim=1) * (Yc * Yc).sum(dim=1))
        corr = t.where(den > 0, num / den, t.full_like(num, float("nan")))
        return corr.cpu().numpy().tolist()

    # cosine / distance: inherited from VectorizedBackend (outside the hot path).
