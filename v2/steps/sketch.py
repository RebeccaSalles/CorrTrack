"""Step 3 — sketch of a window.

Two implementations, common interface `sketch(sid, start_index, values)`:

  * Sketcher            : recomputes the projection over the whole window (simple).
  * IncrementalSketcher : splits the window into `basic_window` sub-windows and
                          caches the projection of each one; when the window
                          slides, only the new sub-windows are computed.

The heavy computation (projection) is delegated to the backend, so it is
compatible with the python/vectorized/parallel swap.

`make_sketcher(config, backend)` chooses the implementation: incremental if
`config.basic_window > 0`, otherwise full recomputation.
"""

import numpy as np


class Sketcher:
    def __init__(self, n_vectors, window_size, seed, backend):
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((n_vectors, window_size))  # projection matrix
        self.backend = backend

    def sketch(self, sid, start_index, values):
        raw = self.backend.project(self.R, values)
        return self.backend.znormalize(raw)


class IncrementalSketcher:
    def __init__(self, n_vectors, window_size, basic_window, seed, backend):
        if window_size % basic_window != 0:
            raise ValueError("window_size must be a multiple of basic_window")
        self.backend = backend
        self.basic_window = basic_window
        self.n_basic = window_size // basic_window
        rng = np.random.default_rng(seed)
        # a single projection matrix, reused for each sub-window
        self.W = rng.standard_normal((n_vectors, basic_window))
        # per-position sign toggle: preserves the ordering info of the sub-windows
        tog = np.random.default_rng(seed + 1).integers(0, 2, size=self.n_basic) * 2 - 1
        self.toggle = tog.astype(float)
        self.cache = {}  # sid -> {basic_start: dot_vector}

    def sketch(self, sid, start_index, values):
        bw = self.basic_window
        series_cache = self.cache.setdefault(sid, {})

        raw = None
        for p in range(self.n_basic):
            bstart = int(start_index + p * bw)
            dot = series_cache.get(bstart)
            if dot is None:  # new sub-window -> project it only once
                dot = self.backend.project(self.W, values[p * bw:(p + 1) * bw])
                series_cache[bstart] = dot
            contrib = self.toggle[p] * dot
            raw = contrib if raw is None else raw + contrib

        # purge sub-windows that have left the current window
        for k in [k for k in series_cache if k < start_index]:
            del series_cache[k]

        return self.backend.znormalize(raw)


def make_sketcher(config, backend):
    """Dispatch between the sketch methods.

    The `config.sketch_method` parameter picks:
      * `random_projection` (default, v1 port) → `Sketcher` (or
        `IncrementalSketcher` when `basic_window > 0`) — Gaussian matrix;
      * `fft_lowpass` / `fft_topk` → `sketch_fft.make_fft_sketcher` — FFT
        magnitude, lag-invariant (∼Parseval). See `v2/steps/sketch_fft.py`.

    The incremental mode (basic_window > 0) is only compatible with
    random_projection — the FFT has no per-sub-window linear structure that is
    easy to cache.
    """
    method = getattr(config, "sketch_method", "random_projection")
    if method != "random_projection":
        if config.basic_window and config.basic_window > 0:
            raise ValueError(
                f"sketch_method={method!r} incompatible with basic_window>0 "
                "(the incremental sketch is only defined for random_projection).")
        if method.startswith("fft_"):
            from .sketch_fft import make_fft_sketcher
            s = make_fft_sketcher(config, backend)
        elif method.startswith("median_"):
            from .sketch_median import make_median_sketcher
            s = make_median_sketcher(config, backend)
        else:
            s = None
        if s is None:
            raise ValueError(f"unknown sketch_method: {method!r}")
        return s
    if config.basic_window and config.basic_window > 0:
        return IncrementalSketcher(
            config.n_vectors, config.window_size, config.basic_window,
            config.seed, backend,
        )
    return Sketcher(config.n_vectors, config.window_size, config.seed, backend)
