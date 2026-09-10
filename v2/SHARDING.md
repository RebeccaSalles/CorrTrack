# Static-partition sharding — `sharded_<base>`

Parallel execution mode for corrtrack v2: `workers` threads, each in charge of a
**data bundle fixed up front** that does its share of the job — no per-window
re-dispatch, no nested pool. Enabled through the backend alias `sharded_<base>`
(`sharded_vectorized`, `sharded_cython`, `sharded_python`, `sharded_mps`) with
`workers > 1`.

```
python -m v2.corrtrack data.csv --backend sharded_vectorized --workers 8
# (pipeline JSON) "params": {"backend": "sharded_vectorized", "workers": 8}
```

Relies on the **FULL index in RAM** (no eviction) → memory guard on the pipeline
side (`CORRTRACK_STATIC_SHARD_MAX_GB`, default 4); beyond that, it falls back to
serial streaming. The result (`correlated.csv`) is **byte-identical** to serial
streaming (`causal` filter: the latest window owns the pair).

Implementation: `v2/core/_static_shard.py` (dispatched from `v2/core/pipeline.py`).

---

## Two blocks, each partitioned into per-thread bundles

### Block 1 — sketch + index
- **sketch**: parallel over a **bundle of windows** (one sketcher per worker,
  cache not shared). Scales well.
- **index**: serial merge (store + array-native `SK`/`RAW` matrices +
  `bulk_load` O(N)). Every key `(sid, t)` gets an **integer id** = its row in
  `SK`/`RAW`.

> **Id space**: store and index are keyed by integer id, NOT by `(sid, t)`.
> `select_candidates` therefore returns id pairs directly → NO key→id lookup
> over the millions of candidates (that used to be the bottleneck). Only the
> ~survivors are converted back to keys at assembly time (`_record`, in
> canonical key order so the output stays identical to the serial one).

### Block 2 — select + validate (`config.shard_mode`)

Two splitting strategies, selectable. **The goal is the fastest possible
validate** → `split` by default.

#### `split` (default) — TWO sub-phases, each with its own bundle
- **select**: parallel over a **bundle of WINDOWS** → gathers every candidate
  pair into global id arrays `(IA, IB)`.
- **validate**: parallel over a **balanced bundle of PAIRS** (`workers` equal
  slices of `IA/IB`) → cosine (sketch filter) + **array-native** Pearson on
  large batches, GIL released. Splitting **by pair count** (rather than by
  window) removes straggler threads when a few windows concentrate the
  candidates → **better validate speedup**.

The measured `validate` span contains ONLY the parallel compute (workers return
arrays; result-tuple assembly and the **vectorized** owner-window attachment via
`searchsorted` happen AFTERWARDS, off the hot path).

**Memory (memory-safe)**: select converts each window into an id array right
away (no millions of live Python tuples), and validate processes each bundle
**in chunks** of `_VALIDATE_CHUNK` pairs (default 250,000, env
`CORRTRACK_SHARD_VALIDATE_CHUNK`) to bound the `SK[ia]`/`RAW[ia]` gather —
without it, a bundle of millions of pairs materializes `(N×n_vec)` and
`(N×window)` matrices → **OOM** (e.g. 25M pairs, n_vectors=32, window=168 → tens
of GB). Remaining trade-off vs `fused`: global `IA/IB` plus the survivors held
in RAM.

> On **dense** configurations (many candidates), validate becomes
> *memory-bandwidth bound* (little gain, sometimes a regression, beyond ~2-4
> workers) and `select` stays *GIL-bound* (candidate generation in pure Python,
> does not scale). The sharded speedup is then moderate; it is decisive when
> candidates are few (validate dominates and parallelizes well).

#### `fused` (alternative) — select+validate FUSED per window bundle
Each worker, FOR EACH WINDOW of its slice: select THEN validate (array-native
cosine+Pearson), accumulating only the correlations. **Memory bounded to a
single window** (no global materialization of the pairs) → safe on very dense
data (ASOS, large `n_lags`: ~10⁸ pairs). The wall time is split synthetically
between select/validate (readable columns), so validate is *interleaved* there,
not isolated.

```
# fused alternative:
--backend sharded_vectorized --workers 8        # + env CORRTRACK_SHARD_MODE=fused
# (pipeline JSON) "params": {"backend": "sharded_vectorized", "workers": 8, "shard_mode": "fused"}
```

| | `split` (default) | `fused` (alternative) |
|---|---|---|
| select | parallel window bundle → global pairs | fused with validate |
| validate | **balanced parallel PAIR bundle** (isolated, scales) | fused, parallel window bundle |
| memory | + all candidate pairs | bounded to one window |
| measured validate | pure compute (≈ the fastest) | synthetic (interleaved) |
| when | default; moderate candidate counts | dense data (huge #pairs) |

---

## Measuring the per-step speedup

The **per-step** metrics (see `v2/REALTIME.md` §2) give, for `sketch` / `index` /
`select` / `validate`: time per computation plus input/output throughput. Under
static sharding, each step is a **fused span** (`calls=1`, hence
`min=max=median`) but the effective input/output throughputs are correct.
Comparing `phase_validate_time_total` (or `phase_validate_in_overall`) between
two runs = validate speedup.

Example (FR air-temperature, 25 series × 1 year, bptree, `sharded_vectorized`):

```
                 sketch  index  select  validate  runtime   (identical corr)
serial vector.    0.22   0.07    3.55    12.60     26.84s
split  w8         0.22   0.01    0.28     0.26      2.04s
fused  w8         0.22   0.01    0.32     0.14*     1.92s     (*synthetic validate)
```

Validate: **12.6 s → 0.26 s** (≈ 48×) — isolated and parallel in `split`.
