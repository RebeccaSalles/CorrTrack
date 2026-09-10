"""Parallel execution: spreads `correlate` (the hot spot) over processes.

A composable wrapper over a **base kernel** (`python` | `vectorized` |
`cython`): the pairs are split into chunks computed on a process pool, each
chunk using the base kernel's `correlate`; every other primitive is delegated
to the base kernel.

Hence the available combinations:
  * `parallel`            = python kernel   + processes
  * `vectorized_parallel` = numpy kernel    + processes
  * `cython_parallel`     = Cython kernels  + processes

(`cython` and `vectorized` are alternative kernels: two of them are never
combined; to mix techniques, use the PER-PHASE backend selection.)
"""

import atexit
import os
from concurrent.futures import ProcessPoolExecutor

from .base import Backend

# below this number of pairs we stay sequential: the serialization/dispatch
# overhead towards the processes exceeds the gain.
MIN_PARALLEL_BATCH = 10_000


def _chunk_worker(args):
    """Rebuild the base kernel in the worker and correlate a chunk of pairs."""
    base_name, chunk = args
    from . import get_backend
    return get_backend(base_name).correlate(chunk)


def _worker_init(cores):
    """Worker initializer: applies the QoS (P/E cores) to the worker process."""
    from ..core.qos import set_qos
    set_qos(cores)


class ParallelBackend(Backend):
    name = "parallel"
    base_name = "python"   # kernel executed inside the workers

    def __init__(self, max_workers=0, cores="auto"):
        super().__init__(max_workers, cores)
        from . import get_backend
        self._base = get_backend(self.base_name)
        self._pool = None              # PERSISTENT pool (created on first use)

    def _get_pool(self):
        if self._pool is None:
            workers = self.max_workers or (os.cpu_count() or 1)
            self._pool = ProcessPoolExecutor(max_workers=workers,
                                             initializer=_worker_init,
                                             initargs=(self.cores,))
            atexit.register(self._pool.shutdown, wait=False)
        return self._pool

    # non-hot primitives -> delegated to the base kernel
    def project(self, matrix, values):
        return self._base.project(matrix, values)

    def znormalize(self, vec):
        return self._base.znormalize(vec)

    def pearson(self, x, y):
        return self._base.pearson(x, y)

    def cosine(self, u, v):
        return self._base.cosine(u, v)

    def distance(self, u, v):
        return self._base.distance(u, v)

    def correlate(self, pairs):
        if not pairs:
            return []
        workers = self.max_workers or (os.cpu_count() or 1)
        # small batch -> sequential (the process overhead would not be amortized)
        if workers <= 1 or len(pairs) < MIN_PARALLEL_BATCH:
            return self._base.correlate(pairs)

        size = max(1, (len(pairs) + workers - 1) // workers)
        chunks = [(self.base_name, pairs[i:i + size])
                  for i in range(0, len(pairs), size)]
        try:
            out = []
            for part in self._get_pool().map(_chunk_worker, chunks):  # pool reused
                out.extend(part)
            return out
        except Exception:
            return self._base.correlate(pairs)  # sequential fallback


class VectorizedParallelBackend(ParallelBackend):
    name = "vectorized_parallel"
    base_name = "vectorized"


class CythonParallelBackend(ParallelBackend):
    name = "cython_parallel"
    base_name = "cython"
