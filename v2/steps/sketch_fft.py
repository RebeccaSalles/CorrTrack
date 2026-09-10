"""Step 3 — FOURIER sketch (alternative to the random projection).

Instead of projecting the window through a random Gaussian matrix, the **FFT**
of the window is computed and its `n_vectors` most informative components are
retained. Two variants:

  * **FFTLowpassSketcher** (`config.sketch_method = "fft_lowpass"`): keeps the
    **first `n_vectors` frequencies** (DC + low harmonics). Suits periodic or
    smooth signals; the high-frequency energy is discarded.

  * **FFTTopKSketcher** (`config.sketch_method = "fft_topk"`): keeps the
    `n_vectors` **largest magnitudes** regardless of their frequency
    (per-window adaptive selection). More robust on signals with a
    non-canonical sparse spectrum.

In both cases the **magnitude `|FFT[k]|`** is taken (not the complex part) →
the sketch becomes nearly **invariant to time shifts** (a shift t multiplies by
a phase factor, which vanishes in the modulus). This is a DIFFERENT property
from the random projection (which is lag-sensitive) and can capture the
correlation at an unknown lag better — at the price of FALSE POSITIVES on
uncorrelated periodic signals (the downstream Pearson validation removes them).

API identical to `Sketcher`: `sketch(sid, start_index, values)` returns an
`n_vectors`-D vector, z-normalized through `backend.znormalize`.

**FFT backend.** The FFT goes through `backend.fft_magnitude(values)` (numpy by
default, MPS override through `torch.fft.rfft`). To benefit from the GPU
acceleration, use `--sketch-backend mps`. The output has size
`window_size//2 + 1`, so `n_vectors <= window_size // 2 + 1` is required.
"""

import numpy as np


class _FFTBase:
    """Shared computation: rfft + component selection."""

    def __init__(self, n_vectors, window_size, backend):
        max_freqs = window_size // 2 + 1
        if n_vectors > max_freqs:
            raise ValueError(
                f"n_vectors={n_vectors} > window_size//2+1={max_freqs}; "
                "either increase window_size or reduce n_vectors.")
        self.n_vectors = n_vectors
        self.window_size = window_size
        self.backend = backend

    def _fft_mag(self, values):
        """rfft -> magnitudes (taille window_size//2 + 1).

        Delegated to the backend: `Backend.fft_magnitude(values)`. numpy by
        default (MKL/Accelerate), overridden by MPS to use `torch.fft.rfft` on
        the GPU.
        """
        return self.backend.fft_magnitude(values)


class FFTLowpassSketcher(_FFTBase):
    """Sketch = first `n_vectors` magnitudes (DC + low frequencies)."""

    def sketch(self, sid, start_index, values):
        mag = self._fft_mag(values)[:self.n_vectors]
        return self.backend.znormalize(mag)


class FFTTopKSketcher(_FFTBase):
    """Sketch = `n_vectors` largest magnitudes (adaptive top-k).

    To keep a dimension comparable across windows, the SORTED values
    (descending) are kept, not the indices. Two windows with the same peaks in
    a different frequency order will have close sketches.
    """

    def sketch(self, sid, start_index, values):
        mag = self._fft_mag(values)
        # top-k through argpartition (O(N)), then sorted for stability (O(k log k))
        k = self.n_vectors
        if k >= len(mag):
            top = np.sort(mag)[::-1]
        else:
            idx = np.argpartition(mag, -k)[-k:]
            top = np.sort(mag[idx])[::-1]
        return self.backend.znormalize(top)


def make_fft_sketcher(config, backend):
    """Build the FFT sketcher according to `config.sketch_method`.

    Returns `None` when `config.sketch_method` is not one of the FFT variants
    (letting the main dispatcher pick the random projection).
    """
    method = getattr(config, "sketch_method", "random_projection")
    if method == "fft_lowpass":
        return FFTLowpassSketcher(config.n_vectors, config.window_size, backend)
    if method == "fft_topk":
        return FFTTopKSketcher(config.n_vectors, config.window_size, backend)
    return None
