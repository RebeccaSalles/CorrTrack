"""Cython backend — compiled kernels for the hot primitives.

The kernels live in `_cython_kernels.pyx`. They are loaded:
  1. from the already compiled extension (`_cython_kernels*.so`), otherwise
  2. compiled on the fly through pyximport (requires Cython installed).
When Cython is unavailable / the compilation fails, the backend **falls back
transparently** to the pure-Python loops of `PythonBackend` (a warning is
emitted once). `--backend cython` therefore always works.

To enable the acceleration:  pip install cython  (the C compilers are then
enough; pyximport compiles at the first import).
"""

import warnings

from .python import PythonBackend

_KERNELS = None
_TRIED = False


def _load_kernels():
    """Compiled kernel module, or None when unavailable (pure-Python fallback)."""
    global _KERNELS, _TRIED
    if _TRIED:
        return _KERNELS
    _TRIED = True
    try:  # 1) already compiled extension
        from . import _cython_kernels as kernels
    except Exception:
        try:  # 2) on-the-fly compilation
            import numpy
            import pyximport
            pyximport.install(setup_args={"include_dirs": numpy.get_include()},
                              language_level=3)
            from . import _cython_kernels as kernels
        except Exception:
            kernels = None
    if kernels is None:
        warnings.warn(
            "CythonBackend: kernels not compiled (Cython missing?) — "
            "falling back to the python backend. `pip install cython` to speed up.",
            RuntimeWarning, stacklevel=2,
        )
    _KERNELS = kernels
    return _KERNELS


class CythonBackend(PythonBackend):
    name = "cython"

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers, cores)
        self._k = _load_kernels()

    def project(self, matrix, values):
        if self._k is not None:
            return self._k.project(matrix, values)
        return super().project(matrix, values)

    def pearson(self, x, y):
        if self._k is not None:
            return self._k.pearson(x, y)
        return super().pearson(x, y)

    def correlate(self, pairs):
        if self._k is not None:
            return self._k.correlate(pairs)
        return super().correlate(pairs)

    # cosine / distance: inherited from PythonBackend (outside the hot path).
