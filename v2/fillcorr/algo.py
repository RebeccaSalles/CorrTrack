"""Mathematical core of FilCorr.

The algorithm (Zhong, Souza, Mueen — ICDM 2020) computes a Pearson correlation
over *filtered* windows (band-pass `[fs, ft]`) through Parseval:

    corr = (Σ|Wx|² + Σ|Wy|² − Σ|Wx − Wy|²) / (2·√Σ|Wx|²·√Σ|Wy|²)

This identity holds when the DC component is zeroed (the filter always cuts
`k=0`) and only the positive half of the spectrum is used (real signal). The FFT
coefficients outside the band `[LB, UB]` contribute neither to the numerator nor
to the denominator → only the `B = UB − LB + 1` useful coefficients are
maintained/carried around.

With a sampling step of 1 (sliding step=1), a coefficient updates in O(1);
altogether, updating a window costs O(B) instead of O(m log m) for a full FFT.
Both variants are exposed: `full_band_fft` (recompute) and `incremental_update`
(one-sample slide).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FilCorrParams:
    """Parameters specific to FilCorr.

    - `fs`, `ft`: bounds of the pass-band (in Hz when `sampling_rate>0`,
      otherwise as a fraction of the equivalent Nyquist frequency).
    - `sampling_rate`: `f` (Hz). When 0, fs/ft are read directly as fractions
      ∈ [0, 0.5].
    - `remove_dc`: always `True` in FilCorr (the Parseval derivation assumes
      DC=0). Exposed for debugging.
    """
    fs: float = 0.0
    ft: float = 0.5
    sampling_rate: float = 1.0
    remove_dc: bool = True


def band_indices(window_size: int, params: FilCorrParams) -> Tuple[int, int]:
    """Return `(LB, UB)`, the start (inclusive) and end (exclusive) indices of
    the band within the positive half of the FFT spectrum, as in the paper:
        LB = ⌊m · fs / f⌋,  UB = ⌊m · ft / f⌋
    with an `LB ≥ 1` constraint when `remove_dc` (the Parseval derivation
    assumes DC=0).
    """
    m = int(window_size)
    f = float(params.sampling_rate) if params.sampling_rate > 0 else 1.0
    lb = int(np.floor(m * float(params.fs) / f))
    ub = int(np.floor(m * float(params.ft) / f))
    if params.remove_dc:
        lb = max(lb, 1)
    # guard: at least 1 coefficient, and never beyond Nyquist
    ub = max(ub, lb + 1)
    ub = min(ub, m // 2 + 1)
    return lb, ub


# ------------------------------------------------------------------------- FFT


def full_band_fft(window: np.ndarray, lb: int, ub: int) -> np.ndarray:
    """Full FFT + extraction of the band coefficients `[lb, ub)`.

    Returns a complex vector of size `B = ub − lb`.
    """
    W = np.fft.fft(np.asarray(window, dtype=float))
    return W[lb:ub]


def full_band_fft_batch(block: np.ndarray, lb: int, ub: int) -> np.ndarray:
    """Band FFT for a `(n_series, m)` block → complex `(n_series, B)`.

    Vectorized: a single `np.fft.fft` call over axis `-1`.
    """
    W = np.fft.fft(np.asarray(block, dtype=float), axis=-1)
    return W[:, lb:ub]


def incremental_update(W_band: np.ndarray, m: int, lb: int, ub: int,
                       drop: float, append: float) -> np.ndarray:
    """Update the band coefficients when the window slides by 1 sample.

    `W_band`: (B,) complex — coefficients at time `t`, within the band.
    `drop`  : old `w[0]` (sample leaving on the left).
    `append`: new `w[m-1]` (sample entering on the right).

    Identity (eq. 12 of the paper, derived sample by sample):
        W_{t+1}[k] = e^{i·2π·k/m} · (W_t[k] − drop + append · e^{−i·2π·m·k/m})
                  = e^{i·2π·k/m} · (W_t[k] − drop + append)    (since e^{−i·2π·k} = 1)
    Cost O(B), independent of m.
    """
    k = np.arange(lb, ub, dtype=float)
    twiddle = np.exp(1j * 2.0 * np.pi * k / float(m))
    return twiddle * (W_band - drop + append)


# ---------------------------------------------------------------- correlation


def parseval_corr(Wx: np.ndarray, Wy: np.ndarray) -> float:
    """Pearson correlation over filtered windows through Parseval (real spectrum).

    `Wx`, `Wy`: (B,) band coefficients (without DC). Returns `corr ∈ [-1, 1]`
    or `nan` when one of the filtered windows is zero.
    """
    sx = float(np.vdot(Wx, Wx).real)            # Σ|Wx|²
    sy = float(np.vdot(Wy, Wy).real)            # Σ|Wy|²
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    sxy = float(np.vdot(Wx, Wy).real)           # Σ Re(Wx · conj(Wy))
    return sxy / (np.sqrt(sx) * np.sqrt(sy))


def parseval_corr_batch(WX: np.ndarray, WY: np.ndarray) -> np.ndarray:
    """Batch version: `WX`, `WY`: (N, B) → (N,) correlations.

    Computes a single row-wise product with einsum, then normalizes.
    """
    WX = np.asarray(WX)
    WY = np.asarray(WY)
    sxy = np.einsum("ij,ij->i", WX.conj(), WY).real
    sxx = np.einsum("ij,ij->i", WX.conj(), WX).real
    syy = np.einsum("ij,ij->i", WY.conj(), WY).real
    denom = np.sqrt(sxx * syy)
    out = np.full(sxy.shape, np.nan, dtype=float)
    ok = denom > 0
    out[ok] = sxy[ok] / denom[ok]
    return out


def naive_filtered_pearson(wx: np.ndarray, wy: np.ndarray,
                           lb: int, ub: int) -> float:
    """NAIVE variant (algorithm 1 of the paper) — reference for the tests.

    Filters in the time domain (FFT → zero outside the band → IFFT) then applies
    the classic Pearson. Slower (O(m log m + m)) but the result is IDENTICAL to
    `parseval_corr` up to floating-point precision.
    """
    m = len(wx)
    Wx = np.fft.fft(np.asarray(wx, dtype=float))
    Wy = np.fft.fft(np.asarray(wy, dtype=float))
    Wxf = np.zeros_like(Wx)
    Wyf = np.zeros_like(Wy)
    Wxf[lb:ub] = Wx[lb:ub]
    Wyf[lb:ub] = Wy[lb:ub]
    # restore the negative half (real signal: conjugate symmetric)
    Wxf[m - ub + 1:m - lb + 1] = np.conj(Wxf[lb:ub])[::-1]
    Wyf[m - ub + 1:m - lb + 1] = np.conj(Wyf[lb:ub])[::-1]
    xf = np.fft.ifft(Wxf).real
    yf = np.fft.ifft(Wyf).real
    # classic Pearson
    xc = xf - xf.mean()
    yc = yf - yf.mean()
    sx = np.sqrt((xc * xc).sum())
    sy = np.sqrt((yc * yc).sum())
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float((xc * yc).sum() / (sx * sy))


# -------------------------------------------------- lagged correlation (helper)


def best_lag_corr(Wx_seq: List[np.ndarray], Wy_seq: List[np.ndarray],
                  neg_corr: bool = False) -> Tuple[float, int]:
    """Give the (corr, lag) maximizing |corr| (or corr when `neg_corr=False`)
    between a sequence of band FFT windows of X and that of Y.

    `Wx_seq[i]` = window X at time `t - i`, same for Y. The lag is the integer
    offset (i_y − i_x), positive when Y precedes X.
    Only used when several lags are to be explored for a same (X, Y) couple at a
    given instant. The v2 pipeline already enumerates the pairs (X@t, Y@t'), so
    the runner can do without it.
    """
    best = -np.inf if not neg_corr else 0.0
    best_lag = 0
    for ix, Wx in enumerate(Wx_seq):
        for iy, Wy in enumerate(Wy_seq):
            c = parseval_corr(Wx, Wy)
            key = abs(c) if neg_corr else c
            if key > best:
                best, best_lag = key, iy - ix
    # `best` is |corr| when neg_corr; restore the correct sign otherwise:
    if neg_corr:
        # recompute the signed corr at the lag found
        for ix, Wx in enumerate(Wx_seq):
            iy = ix + best_lag
            if 0 <= iy < len(Wy_seq):
                c = parseval_corr(Wx, Wy_seq[iy])
                if abs(c) == best:
                    return c, best_lag
        return float("nan"), 0
    return best, best_lag
