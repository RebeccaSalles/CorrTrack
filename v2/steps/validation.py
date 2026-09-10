"""Step 5 — exact validation on raw data (Pearson via backend).

Correlation computed in batches (`backend.correlate`) -> benefits from the
backend's vectorization / parallelization. Handles negative correlation and
returns the signed lag.
"""

import numpy as np


def validate(pairs, store, corr_threshold, backend, neg_corr=False):
    """Return the correlated pairs: list of (keyA, keyB, corr, signed_lag).

    signed lag = t_A - t_B (with A, B in the canonical order of the pair):
    negative if A precedes B, positive otherwise.
    """
    if not pairs:
        return []
    xy = [(store[a]["raw"], store[b]["raw"]) for a, b in pairs]
    corrs = backend.correlate(xy)

    out = []
    for (a, b), corr in zip(pairs, corrs):
        if corr is None or not np.isfinite(corr):
            continue
        ok = abs(corr) >= corr_threshold if neg_corr else corr >= corr_threshold
        if ok:
            out.append((a, b, corr, int(a[1] - b[1])))
    return out
