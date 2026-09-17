"""Dataset characteristics recorded next to every N-way run (user's request, 2026-09-17), so a
result can be read against the data it was measured on: size, correlation density at the
protocol's threshold, how "cooperative" the series are (low-frequency energy share, the axis
StatStream / FilCorr / BRAID bet on), smoothness, and degeneracy.

    profile = profile_dataset(test_data, W, step, bf_record=record_of_the_bruteforce_arm)

`test_data` is the runner's (1 + m, T) array (row 0 is the index). Statistics over windows are
computed on a deterministic sample of at most `max_windows` window starts.
"""
from __future__ import annotations

import numpy as np


def profile_dataset(test_data, W, step, bf_record=None, n_coeffs=16, max_windows=64, meta=None):
    X = np.asarray(test_data[1:], dtype=np.float64)
    m, T = X.shape
    starts = np.arange(0, T - W + 1, step)
    if starts.size > max_windows:
        starts = starts[np.linspace(0, starts.size - 1, max_windows).astype(int)]
    lowf, const, ac1, spread = [], 0, [], []
    for s0 in starts:
        win = X[:, s0:s0 + W]
        z = win - win.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(z, axis=1)
        ok = nrm > 1e-12 * max(1.0, np.abs(win).max())
        const += int((~ok).sum())
        if ok.any():
            zn = z[ok] / nrm[ok, None]
            F = np.abs(np.fft.rfft(zn, axis=1)) ** 2
            tot = F[:, 1:].sum(axis=1)
            lowf.extend((F[:, 1:n_coeffs + 1].sum(axis=1) / np.maximum(tot, 1e-300)).tolist())
            ac1.extend((np.sum(zn[:, 1:] * zn[:, :-1], axis=1)).tolist())
            spread.extend((win[ok].std(axis=1) / np.maximum(np.abs(win[ok].mean(axis=1)), 1e-12)).tolist())
    prof = dict(
        m=int(m), n_obs=int(T), n_windows=int(len(range(0, T - W + 1, step))), W=int(W), step=int(step),
        # cooperativeness: share of window energy in the first n_coeffs DFT coefficients (StatStream's digest);
        # near 1 = smooth / random-walk-like (grid methods prune well), near 2*n/W = white noise
        low_frequency_energy_share_mean=float(np.mean(lowf)) if lowf else None,
        low_frequency_energy_share_p10=float(np.percentile(lowf, 10)) if lowf else None,
        white_noise_reference=float(2.0 * n_coeffs / W),
        lag1_autocorr_mean=float(np.mean(ac1)) if ac1 else None,
        constant_window_fraction=float(const / max(len(starts) * m, 1)),
        coefficient_of_variation_median=float(np.median(spread)) if spread else None,
        nan_fraction_source=(meta or {}).get("nan_fraction"), regime_source=(meta or {}).get("regime"),
    )
    if bf_record is not None:
        pairs = bf_record.get("total_candidates") or bf_record.get("tested") or 0
        prof.update(pair_windows=int(pairs), correlated_pair_windows=int(bf_record.get("correlated") or 0),
                    density_at_threshold=(float(bf_record["correlated"]) / pairs) if pairs else None)
    return prof
