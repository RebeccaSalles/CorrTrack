"""Registry of the compute backends — the single swap point.

To change technology: `--backend vectorized` (CLI) or `CORRTRACK_BACKEND=cuda`.
Adding a backend = writing the class then registering it in BACKENDS.
"""

from .base import Backend
from .coreml import CoremlBackend
from .cuda import CudaBackend
from .cython import CythonBackend
from .mps import MpsBackend
from .parallel import (
    CythonParallelBackend,
    ParallelBackend,
    VectorizedParallelBackend,
)
from .python import PythonBackend
from .vectorized import VectorizedBackend

BACKENDS = {
    PythonBackend.name: PythonBackend,
    VectorizedBackend.name: VectorizedBackend,
    ParallelBackend.name: ParallelBackend,
    VectorizedParallelBackend.name: VectorizedParallelBackend,
    CythonBackend.name: CythonBackend,
    CythonParallelBackend.name: CythonParallelBackend,
    MpsBackend.name: MpsBackend,
    CoremlBackend.name: CoremlBackend,
    CudaBackend.name: CudaBackend,
}

# `sharded_<base>` aliases: window-parallel STATIC PARTITION execution
# (see `core/_static_shard.py`). `workers` threads, each in charge of a slice
# of windows fixed up front, do their whole share (sketch then select+validate)
# — NO per-window re-dispatch, NO nested pool.
# The compute backend used by each worker is `<base>` (hence
# "vectorize/cython over sharding"); the prefix is consumed by the dispatcher
# in `core/pipeline.py`. No new backend class — this is a MARKER. Relies on the
# FULL index in RAM (memory guard on the pipeline side; otherwise it falls back
# to serial streaming).
SHARDED_BASES = ("python", "vectorized", "cython", "mps")
SHARDED_ALIASES = {f"sharded_{b}": b for b in SHARDED_BASES}


def base_of(name):
    """Return the effective compute backend name (resolves sharded_* aliases)."""
    return SHARDED_ALIASES.get(name, name)


def is_sharded(name):
    return name in SHARDED_ALIASES


def get_backend(name, **kwargs):
    """Instantiate the requested backend (kwargs forwarded to the constructor).

    The `sharded_<base>` aliases (sharded_python / sharded_vectorized /
    sharded_cython / sharded_mps) resolve to the standard `<base>` backend —
    the parallelization (static partition of the windows over N threads) lives
    in the pipeline, not in the backend. The gain grows with the number of
    series (the parallelized select+validate share then dominates the serial
    sketch). NB: `sharded_python` stays GIL-bound on the validation (pure
    Python loops); `sharded_vectorized`/`sharded_cython` are the useful
    variants (numpy/Cython kernels releasing the GIL per worker).
    """
    resolved = base_of(name)
    try:
        cls = BACKENDS[resolved]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}. "
                         f"Choices: {list(BACKENDS) + list(SHARDED_ALIASES)}")
    return cls(**kwargs)


def available():
    return list(BACKENDS) + list(SHARDED_ALIASES)
