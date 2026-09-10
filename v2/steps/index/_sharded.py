"""Sharded index — wrapper spreading keys and queries over N child indexes.

Only used when `config.backend.startswith("sharded_")`. Each shard is a
standard `SketchIndex` (any of the 13); insertions/deletions are routed by
`hash(key) % N`; `query_batch` fans out in parallel over a thread pool and
unions the candidates per query.

ANN limit: for the indexes whose recall depends on the global connectivity of
the graph / forest (`hnsw`, `annoy`), sharding degrades recall. Detected
through the `_NO_SHARD` set → a single global shard (the parallel sketch keeps
its gain, the index stays sequential).
"""

import os
from concurrent.futures import ThreadPoolExecutor

from .base import SketchIndex

_NO_SHARD = {"hnsw", "annoy"}


def make_sharded_index(config, candidate_be, cores, make_one):
    """Build a `ShardedIndex` (or a plain index when the index belongs to the
    `_NO_SHARD` set).

    `make_one()` is a zero-arg factory building a fresh `SketchIndex`
    (typically `lambda: make_index(config, candidate_be)` injected by the
    pipeline to avoid the circular import).
    """
    if config.index_backend in _NO_SHARD or cores <= 1:
        return make_one()
    shards = [make_one() for _ in range(cores)]
    return ShardedIndex(shards)


class ShardedIndex(SketchIndex):
    """`SketchIndex` composite : N shards, routage par `hash(key) % N`."""

    def __init__(self, shards):
        self.shards = list(shards)
        self.n = len(self.shards)
        self._pool = None

    def shard_idx_for(self, key):
        """Shard index for `key`. Routed by sid (key[0]) when key is a
        (sid, time) tuple — invariant: every window of a given sid stays on the
        same shard, which keeps insert/remove consistent."""
        routing = key[0] if isinstance(key, tuple) else key
        return hash(routing) % self.n

    def _shard_for(self, key):
        return self.shards[self.shard_idx_for(key)]

    def _get_pool(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.n)
        return self._pool

    def insert(self, key, vec):
        self._shard_for(key).insert(key, vec)

    def remove(self, key):
        self._shard_for(key).remove(key)

    def query(self, vec):
        # union of the candidates over every shard
        out = []
        for shard in self.shards:
            out.extend(shard.query(vec))
        return out

    def query_batch(self, queries):
        if not queries:
            return []
        pool = self._get_pool()
        # 1 future per shard → each shard runs its own loop / cdist
        futures = [pool.submit(shard.query_batch, queries) for shard in self.shards]
        per_shard = [f.result() for f in futures]
        # per-query union, deduplicated while preserving order
        out = []
        for q_idx in range(len(queries)):
            seen = set()
            merged = []
            for shard_results in per_shard:
                for cand in shard_results[q_idx]:
                    if cand not in seen:
                        seen.add(cand)
                        merged.append(cand)
            out.append(merged)
        return out

    def __len__(self):
        return sum(len(s) if hasattr(s, "__len__") else 0 for s in self.shards)
