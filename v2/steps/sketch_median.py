"""Step 3 — MEDIAN sketches (simple alternatives to the random projection).

Two methods, both non-linear and robust to outliers:

  * **MedianBlocksSketcher** (`config.sketch_method = "median_blocks"`) — median
    PAA. The window is split into `n_vectors` CONTIGUOUS blocks and the median
    of each block is kept. Compresses the window while keeping the local
    central value:

        sketch[i] = median( window[i·B : (i+1)·B] )   with B = window_size // n_vectors

    Two correlated windows (same coarse shape) will have close sketches. Robust
    to noise (median ≠ mean). Very fast (a single `np.median(axis=1)` over a
    reshaped matrix).

  * **MedianPhaseSketcher** (`config.sketch_method = "median_phase"`) — phase
    median / periodic folding. The window is split into
    `n_blocks = window_size // n_vectors` regular blocks, then the median is
    taken at the same position-within-block across every block:

        sketch[j] = median( window[j], window[j+B], window[j+2B], … )

    where **B = n_vectors** (block size) and j ∈ [0, n_vectors).
    Detects the **periodicity** at step `n_vectors`: if the window is periodic
    with period `n_vectors`, the aggregated values sit at the same "phase" of
    the cycle → robust median of the cycle shape. For hourly series with a
    daily cycle, set `n_vectors = 24`.

Both require `window_size % n_vectors == 0` (explicit error otherwise). Both
are z-normalized through `backend.znormalize`. No parameter is specific to
these sketches (`n_vectors` and `window_size` do everything).
**Incompatible with `basic_window > 0`** (no per-sub-window linear structure —
the median is not additive).
"""

import numpy as np


def _check_divisible(window_size, n_vectors, name):
    if window_size % n_vectors != 0:
        raise ValueError(
            f"{name}: window_size ({window_size}) must be a MULTIPLE of "
            f"n_vectors ({n_vectors}) to split into regular blocks. "
            f"Adjust window_size or n_vectors (e.g. 168/24, 168/14, 96/16).")


class MedianBlocksSketcher:
    """`sketch[i] = median(window[i·B : (i+1)·B])`, B = window_size // n_vectors."""

    def __init__(self, n_vectors, window_size, backend):
        _check_divisible(window_size, n_vectors, "median_blocks")
        self.n_vectors = n_vectors
        self.block_size = window_size // n_vectors
        self.backend = backend

    def sketch(self, sid, start_index, values):
        v = np.asarray(values, dtype=float)
        # reshape (n_vectors, block_size) → median over axis=1 → n_vectors-D vector
        # Delegated to the backend: numpy by default (cython/vectorized/parallel
        # inherit it), torch.median on MPS.
        med = self.backend.median(v.reshape(self.n_vectors, self.block_size), axis=1)
        return self.backend.znormalize(med)


class MedianPhaseSketcher:
    """`sketch[j] = median(window[j], window[j+n_vectors], window[j+2·n_vectors], …)`.

    Folds the window into `n_blocks = window_size // n_vectors` layers of size
    `n_vectors`, then takes the median at each phase position. The natural
    period is `n_vectors` (set `n_vectors=24` for a daily cycle on hourly
    data).
    """

    def __init__(self, n_vectors, window_size, backend):
        _check_divisible(window_size, n_vectors, "median_phase")
        self.n_vectors = n_vectors
        self.n_blocks = window_size // n_vectors
        self.backend = backend

    def sketch(self, sid, start_index, values):
        v = np.asarray(values, dtype=float)
        # reshape (n_blocks, n_vectors) → median over axis=0 → n_vectors-D vector
        # i.e. each column (= one "phase") is taken and aggregated.
        # Delegated to the backend (same as MedianBlocksSketcher).
        med = self.backend.median(v.reshape(self.n_blocks, self.n_vectors), axis=0)
        return self.backend.znormalize(med)


def make_median_sketcher(config, backend):
    """Build the median sketcher according to `config.sketch_method`.

    Returns `None` when `config.sketch_method` is not a median variant.
    """
    method = getattr(config, "sketch_method", "random_projection")
    if method == "median_blocks":
        return MedianBlocksSketcher(config.n_vectors, config.window_size, backend)
    if method == "median_phase":
        return MedianPhaseSketcher(config.n_vectors, config.window_size, backend)
    return None
