"""Per-series sharded sketch+insert phase (used by the `sharded_*` backends).

Enabled by `pipeline.run` when `config.backend.startswith("sharded_")`.
Partitions the `series_id` over N shards (through `ShardedIndex.shard_idx_for`)
and launches N thread workers (one per shard) doing, in order:

    for each series of the slice:
        vec  = local_sketcher.sketch(...)
        local_shard.insert(key, vec)

Every worker runs in parallel; a barrier (`as_completed`) waits for completion
before `select_candidates` is called — the requested "resync before select".

Why threads and not processes:
- `vectorized` (numpy bulk), `cython` (`nogil` kernels) and `mps` (PyTorch C++)
  release the GIL → real thread parallelism without the pickling overhead.
- `sharded_python` is pointless (the GIL serializes) — not exposed.

Limits:
- The timing of the local insert is included in the `sketch` phase (a single
  span per worker, to match the "sketch+insert together" semantics). The
  sequential `index` phase has no equivalent here.
"""

from concurrent.futures import ThreadPoolExecutor

from ..steps.index._sharded import ShardedIndex
from ..steps.sketch import make_sketcher


class ShardedPhase:
    """Callable matching the signature of `_sketch_and_insert_serial`.

    Owns the persistent pool, the per-worker sketchers (thread-safe cache) and
    the pre-computed partition of the series.
    """

    def __init__(self, config, sketch_be, n_shards):
        self.n = n_shards
        # one sketcher per worker → `IncrementalSketcher` cache not shared.
        # They all use the same seed → projection identical to the serial mode.
        self.sketchers = [make_sketcher(config, sketch_be) for _ in range(n_shards)]
        self.pool = ThreadPoolExecutor(max_workers=n_shards,
                                       thread_name_prefix="shard")
        self._partition = None
        self._ids_ref = None

    def _partition_for(self, ids, index):
        if self._ids_ref is ids and self._partition is not None:
            return self._partition
        parts = [[] for _ in range(self.n)]
        # routing consistent with `ShardedIndex.shard_idx_for((sid, t))` (which
        # relies on key[0] = sid). Same Python hash → same value within the same
        # process (threads share PYTHONHASHSEED).
        if isinstance(index, ShardedIndex):
            for i, sid in enumerate(ids):
                parts[index.shard_idx_for((sid, 0))].append((i, sid))
        else:
            for i, sid in enumerate(ids):
                parts[hash(sid) % self.n].append((i, sid))
        self._partition = parts
        self._ids_ref = ids
        return parts

    def __call__(self, ids, win, config, store, sketcher, index, sketches_out,
                 mode, limit, timings):
        # bf has no sketch phase — delegate to the serial path (defensive case,
        # since bf does not normally use sharded).
        from .pipeline import _sketch_and_insert_serial, CORRTRACK_STEPS
        if mode != "corrtrack":
            return _sketch_and_insert_serial(
                ids, win, config, store, sketcher, index, sketches_out,
                mode, limit, timings,
            )

        partition = self._partition_for(ids, index)
        std_thr = config.std_threshold
        win_start_time = win.start_time
        win_start_index = win.start_index
        block = win.block

        def _worker(shard_idx):
            local_sketcher = self.sketchers[shard_idx]
            local_shard = (index.shards[shard_idx]
                           if isinstance(index, ShardedIndex) else index)
            out = []
            for i, sid in partition[shard_idx]:
                if std_thr > 0.0 and float(block[i].std()) < std_thr:
                    continue
                key = (sid, win_start_time)
                vec = local_sketcher.sketch(sid, win_start_index, block[i])
                local_shard.insert(key, vec)
                out.append((key, vec, block[i]))
            return out

        # sketch + insert (per shard) under a SINGLE "sketch" span — the serial
        # `index` phase is absorbed here (the workers already inserted into
        # their local shard).
        with timings.track("sketch"):
            futures = [self.pool.submit(_worker, s) for s in range(self.n)]
            per_shard = [f.result() for f in futures]   # BARRIER (resync)

        # post-barrier: merge into the shared store (on the main thread, hence
        # safe without a lock)
        current_keys = []
        for shard_results in per_shard:
            for key, vec, raw in shard_results:
                store[key] = {"time": win_start_time, "raw": raw, "sketch": vec}
                sketches_out.append((key, vec))
                current_keys.append(key)
        return current_keys


def make(config, sketch_be, n_shards):
    """Build the sharded `sketch_insert_fn` (callable)."""
    return ShardedPhase(config, sketch_be, n_shards)
