"""Benchmark harness: compares CorrTrack to brute-force and emits the stats row
(~70 columns, `;` separator) in the v1 harness format (OPTIM_RESULT_COLUMNS).

For a given config, runs BOTH modes on the same CSV:
  * bf       -> ground truth (every pair validated exactly);
  * corrtrack-> prediction (sketch + index + validation).
It then maps timings/counters, computes speedup/waste and the quality metrics
(`core.metrics`), and writes/appends a CSV row.

Columns with no v2 equivalent (disk artifacts, MPI nodes, etc.) are left empty
or at 0; the useful numeric columns are computed faithfully.
"""

import csv
import itertools
import math
import os
from dataclasses import asdict

from . import metrics
from . import pipeline
from .config import Config

COLUMNS = [
    "dataset_id", "alg", "n_ts", "n_w", "total_w", "mem_w", "nodes", "exec_mode",
    "parallel_sketch", "parallel_candidates", "parallel_validation",
    "workers_total", "workers_sketch", "workers_candidates", "workers_validation",
    "window_size", "window_step", "basic_window", "n_lags", "seed", "seed_toggle",
    "preprocess", "sketch_norm", "candidate_backend", "corr_threshold",
    "grid_max", "cell_stretch", "grid_cell", "n_vectors", "grid_n_coords",
    "grid_vote_min", "grid_n_tables", "query_radius", "n_neighbors",
    "key_mode", "key_proj_dim", "quadtree_proj_dim",
    "annoy_n_trees", "annoy_leaf_size", "hnsw_m", "hnsw_ef_c", "hnsw_ef_s",
    "cand_time_bf", "val_time_bf", "monit_time_bf", "runtime_bf", "artifact_time_bf",
    "sk_time", "cand_time", "val_time", "monit_time", "runtime", "artifact_time",
    "speedup", "speedup_ceil", "rel_speedup_eff",
    "corr_w_bf", "corr_w", "tested_w_bf", "tested_w", "cand_w_bf", "cand_w",
    "corr_prop", "waste_val_bf", "waste_val", "rel_waste_red",
    "precision_pos", "recall_pos", "f1_pos", "precision_neg", "recall_neg", "f1_neg",
    "precision", "recall", "specificity", "recall_min", "f1", "aucroc", "pr_auc",
    "status", "error",
]

NAN = float("nan")


def _t(timings, key):
    """Cumulated time of a phase (0.0 when absent)."""
    return timings.get(key, 0.0)


# parameters the brute-force result (and timing) depends on: lets the bf be
# shared across every config of a same group during a sweep.
_BF_KEYS = ("window_size", "window_step", "n_lags", "corr_threshold", "neg_corr",
            "workers")


def _bf_key(config):
    return tuple(getattr(config, k) for k in _BF_KEYS) + (config.backend_for("validate"),)


def build_record(config, dataset_id, reference, run, alg="corrtrack",
                 metrics_override=None):
    """Assemble a stats row out of TWO already computed results.

    `reference` plays the brute-force role (ground truth / baseline) and `run`
    the evaluated config: speedup = runtime_ref/runtime_run, corr_w_bf = ref,
    precision/recall = run compared to `reference`.

    `metrics_override`: an already computed block of metrics (e.g. by
    `metrics.compare_streaming`, which reads the pairs from disk). When it is
    provided, `compute_metrics` is NOT called — `reference`/`run` then do not
    need to hold the full `correlated` list (memory frugality on dense datasets
    where it weighs tens of GB). Otherwise, the historical behaviour: metrics
    computed from `reference["correlated"]` / `run["correlated"]` in memory.
    """
    record = {c: "" for c in COLUMNS}
    record.update(_config_columns(config, dataset_id))
    record["alg"] = alg
    record.update(_run_columns(reference, run))
    record.update(_derived_columns(record))
    if metrics_override is not None:
        record.update(metrics_override)
    elif reference.get("correlated") is not None and run.get("correlated") is not None:
        # historical in-memory path (sweep: small lists, already in RAM)
        record.update(metrics.compute_metrics(
            reference["correlated"], run["correlated"],
            total_pairs_bf=reference["n_tested"]))
    # otherwise: neither override nor in-memory lists (e.g. watch.py on a run
    # whose correlated.csv is not loaded) → quality columns left empty.
    record["status"] = "success"
    record["_config"] = asdict(config)  # full config (for the optimizer; not a CSV column)
    return record


def compare(config, path, dataset_id="dataset", bf=None):
    """Run (as needed) bf + corrtrack and return the stats-row dict.

    `bf`: pre-computed brute-force result to reuse (shared across a sweep).
    When `config.sweep_validate` is False, the corrtrack run stops at
    `validate_sketches` (the post-cosine sketch set serves as the prediction).
    """
    record = {c: "" for c in COLUMNS}
    record.update(_config_columns(config, dataset_id))
    try:
        if bf is None:
            bf = pipeline.run(config, path, mode="bf", step="monitor")
        if config.sweep_validate:
            ct = pipeline.run(config, path, mode="corrtrack", step="monitor")
        else:
            ct = pipeline.run(config, path, mode="corrtrack", step="validate_sketches")
            ct = _synthesize_ct_from_prevalidated(ct, bf)
        record = build_record(config, dataset_id, bf, ct, alg="corrtrack")
    except Exception as exc:  # partial row kept (config untouched)
        record["status"] = "error"
        record["error"] = str(exc)
    return record


def _synthesize_ct_from_prevalidated(ct, bf):
    """Adapt a `validate_sketches` result (truncated corrtrack) so build_record
    can consume it as if it were a full `monitor` run.

    Assumptions: without validate, neither the sign nor the exact corr is known;
    (a, b, NaN, signed_lag) is synthesized so compute_metrics matches per pair.
    n_series/n_windows: taken from the bf (same data).
    """
    prevalidated = ct.get("candidates", [])
    synthetic_correlated = [(a, b, float("nan"), int(a[1] - b[1])) for a, b in prevalidated]
    ct = dict(ct)
    ct["correlated"] = synthetic_correlated
    ct["n_correlated"] = len(prevalidated)
    ct["n_candidates"] = ct.get("n_candidates", len(prevalidated))  # raw (before the cosine filter)
    ct["n_tested"] = len(prevalidated)
    ct["n_series"] = bf.get("n_series", 0)
    ct["n_windows"] = bf.get("n_windows", 0)
    return ct


def build_configs(axes):
    """Cartesian product of the axes {field -> [values]} -> list of Config.

    Empty `axes` -> a single config (defaults). A single axis with one value ->
    one config; several values -> one config per combination.
    """
    names = list(axes)
    if not names:
        return [Config.build()]
    return [Config.build(**dict(zip(names, combo)))
            for combo in itertools.product(*(axes[n] for n in names))]


# Parallel sweep executor (env override `CORRTRACK_SWEEP_EXEC`):
#   "bundle" (DEFAULT): RUN BUNDLING — the grid is partitioned into `workers`
#       contiguous bundles, ONE PROCESS per bundle, each running its partition
#       SEQUENTIALLY (bf shared locally). Real parallelism (processes bypass the
#       GIL — mandatory because the sweep backend is `python`, pure Python ⇒
#       threads would not scale) with a minimal spawn overhead (N startups
#       instead of one per config).
#   "thread": same bundling but over THREADS (reserve it for a sweep backend
#       that releases the GIL — vectorized/cython/mps).
#   "pool"  : former task-per-config pool (ProcessPoolExecutor, 1 task/config).
_SWEEP_EXEC = os.environ.get("CORRTRACK_SWEEP_EXEC", "bundle").strip().lower()


def _partition(items, n):
    """`n` balanced CONTIGUOUS bundles of `items` (sizes within ±1)."""
    n = max(1, min(int(n), len(items)))
    k, m = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        sz = k + (1 if j < m else 0)
        out.append(items[i:i + sz])
        i += sz
    return [b for b in out if b]


def _run_bundle(configs, path, dataset_id, bf_cache):
    """Run a BUNDLE of configs → stats rows (a module-level function is
    picklable, and it is the unit submitted to each process/thread). `bf_cache`
    (pre-computed, shared bf results) is shared in memory for the threads, or
    pickled (the bundle's subset) for the processes."""
    return [compare(c, path, dataset_id, bf=bf_cache[_bf_key(c)]) for c in configs]


def _build_bf_cache(configs, path):
    """Pre-compute the bf ONCE per `_bf_key` group (shared across the whole sweep)."""
    bf_cache = {}
    for c in configs:
        key = _bf_key(c)
        if key not in bf_cache:
            bf_cache[key] = pipeline.run(c, path, mode="bf", step="monitor")
    return bf_cache


def _sweep_bundles(configs, path, dataset_id, workers, progress, total, use_threads,
                   bf_cache):
    """DEFAULT parallel strategy: run bundling (see `_SWEEP_EXEC`).

    Partitions the grid into `workers` contiguous bundles and launches one
    worker (process by default, thread when `use_threads`) per bundle. The bf is
    pre-computed ONCE (shared in RAM for the threads; pickled per bundle for the
    processes). `progress` is called on each bundle completion."""
    bundles = _partition(configs, workers)
    rows = []
    if use_threads:
        import threading
        lock = threading.Lock()

        def work(bundle):
            res = _run_bundle(bundle, path, dataset_id, bf_cache)  # shared bf_cache
            with lock:
                rows.extend(res)
                done, snap = len(rows), list(rows)
            if progress is not None:
                progress(done, total, snap)

        threads = [threading.Thread(target=work, args=(b,), name=f"sweep{i}",
                                    daemon=True) for i, b in enumerate(bundles)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return rows

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    # `spawn`: a pristine interpreter per worker (minimal RSS). A single spawn
    # per BUNDLE (≠ per config). Each bundle only receives ITS bf (minimal pickle).
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(bundles), mp_context=ctx) as ex:
        futs = []
        for b in bundles:
            bf_sub = {k: bf_cache[k] for k in {_bf_key(c) for c in b}}
            futs.append(ex.submit(_run_bundle, b, path, dataset_id, bf_sub))
        for fut in as_completed(futs):
            rows.extend(fut.result())
            if progress is not None:
                progress(len(rows), total, rows)
    return rows


def _sweep_pool(configs, path, dataset_id, workers, progress, total, bf_cache):
    """Former executor: task-per-config pool (`CORRTRACK_SWEEP_EXEC=pool`)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    import sys
    rows = []
    ctx = mp.get_context("spawn")
    pp_kwargs = {"max_workers": int(workers), "mp_context": ctx}
    if sys.version_info >= (3, 11):
        pp_kwargs["max_tasks_per_child"] = 8
    with ProcessPoolExecutor(**pp_kwargs) as ex:
        futs = [ex.submit(compare, c, path, dataset_id, bf=bf_cache[_bf_key(c)])
                for c in configs]
        for fut in as_completed(futs):
            rows.append(fut.result())
            if progress is not None:
                progress(len(rows), total, rows)
    return rows


def sweep(axes, path, dataset_id="dataset", progress=None, workers=0, configs=None):
    """Sweep every combination; the brute-force is shared per group of the
    parameters affecting it. Returns the list of stats rows.

    `progress(done, total, rows)`: progress callback.
    `workers > 1`: DEFAULT parallel strategy = **run bundling** (the grid is cut
    into `workers` partitions, one per worker; see `_SWEEP_EXEC` /
    `_sweep_bundles`). `configs` (optional): an already built list of Config
    (used by Successive Halving to resubmit subsets)."""
    if configs is None:
        configs = build_configs(axes)
    total = len(configs)
    bf_cache = _build_bf_cache(configs, path)   # bf pre-computed ONCE (shared)

    if workers and workers > 1 and total > 1:
        if _SWEEP_EXEC == "pool":
            return _sweep_pool(configs, path, dataset_id, int(workers), progress,
                               total, bf_cache)
        return _sweep_bundles(configs, path, dataset_id, int(workers), progress,
                              total, use_threads=(_SWEEP_EXEC == "thread"),
                              bf_cache=bf_cache)

    rows = []
    for config in configs:
        rows.append(compare(config, path, dataset_id, bf=bf_cache[_bf_key(config)]))
        if progress is not None:
            progress(len(rows), total, rows)
    return rows


def _config_columns(config, dataset_id):
    par = {p: config.backend_for(p) == "parallel"
           for p in ("sketch", "candidate", "validate")}
    w = config.workers
    return {
        "dataset_id": dataset_id, "alg": "corrtrack",
        "nodes": 0, "exec_mode": "sequential",
        "parallel_sketch": par["sketch"], "parallel_candidates": par["candidate"],
        "parallel_validation": par["validate"],
        "workers_total": w,
        "workers_sketch": w if par["sketch"] else 1,
        "workers_candidates": w if par["candidate"] else 1,
        "workers_validation": w if par["validate"] else 1,
        "window_size": config.window_size, "window_step": config.window_step,
        "basic_window": config.basic_window, "n_lags": config.n_lags,
        "seed": config.seed, "preprocess": config.neg_corr,
        "sketch_norm": "znormalize", "candidate_backend": config.index_backend,
        "corr_threshold": config.corr_threshold,
        "n_vectors": config.n_vectors,
        # grid
        "grid_cell": config.grid_cell, "grid_n_tables": config.grid_n_tables,
        "grid_n_coords": config.grid_n_coords, "grid_vote_min": config.grid_vote_min,
        # universel
        "query_radius": config.query_radius, "n_neighbors": config.n_neighbors,
        # quadtree
        "quadtree_proj_dim": config.quadtree_proj_dim,
        # bptree/bst (+3D)
        "key_mode": config.key_mode, "key_proj_dim": config.key_proj_dim,
        # hnsw
        "hnsw_m": config.hnsw_m, "hnsw_ef_c": config.hnsw_ef_c,
        "hnsw_ef_s": config.hnsw_ef_s,
        # annoy
        "annoy_n_trees": config.annoy_n_trees,
        "annoy_leaf_size": config.annoy_leaf_size,
    }


def _cand_time(t):
    """Candidate generation time, whatever the mode:
    bf -> 'candidates' phase (enumeration); corrtrack -> index + select + cosine filter."""
    return (_t(t, "candidates") + _t(t, "index") + _t(t, "select")
            + _t(t, "validate_sketches"))


def _run_columns(bf, ct):
    n_ts, n_w = ct["n_series"], ct["n_windows"]
    bt, tt = bf["timings"], ct["timings"]
    return {
        "n_ts": n_ts, "n_w": n_w, "total_w": n_ts * n_w,
        "sk_time_bf": _t(bt, "sketch"),  # 0 when the baseline is bf (no sketch phase)
        "cand_time_bf": _cand_time(bt), "val_time_bf": _t(bt, "validate"),
        "monit_time_bf": _t(bt, "monitor"), "runtime_bf": bf["runtime"],
        "artifact_time_bf": 0.0,
        "sk_time": _t(tt, "sketch"), "cand_time": _cand_time(tt),
        "val_time": _t(tt, "validate"), "monit_time": _t(tt, "monitor"),
        "runtime": ct["runtime"], "artifact_time": 0.0,
        "corr_w_bf": bf["n_correlated"], "corr_w": ct["n_correlated"],
        "tested_w_bf": bf["n_tested"], "tested_w": ct["n_tested"],
        "cand_w_bf": bf["n_candidates"], "cand_w": ct["n_candidates"],
    }


def _derived_columns(r):
    runtime, runtime_bf = r["runtime"], r["runtime_bf"]
    speedup = math.inf if runtime <= 0 else runtime_bf / runtime
    corr_prop = metrics._safe_div(r["corr_w_bf"], r["cand_w_bf"])
    # theoretical ceiling: validate only the actually correlated proportion
    num = r["cand_time_bf"] + r["val_time_bf"] + r["monit_time_bf"]
    denom = (r["sk_time"] + r["cand_time"]
             + (r["val_time_bf"] * corr_prop if corr_prop == corr_prop else NAN)
             + r["monit_time_bf"])
    speedup_ceil = metrics._safe_div(num, denom)
    waste_val_bf = metrics._safe_div(r["cand_w_bf"], r["corr_w_bf"])
    waste_val = metrics._safe_div(r["cand_w"], r["corr_w"])
    return {
        "speedup": speedup, "speedup_ceil": speedup_ceil,
        "rel_speedup_eff": metrics._safe_div(speedup, speedup_ceil),
        "corr_prop": corr_prop,
        "waste_val_bf": waste_val_bf, "waste_val": waste_val,
        "rel_waste_red": metrics._safe_div(waste_val_bf, waste_val),
    }


def _fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(v)
    return str(v)


def write_rows(path, rows, append=True, columns=None):
    """Write stats rows into a `;`-separated CSV.

    append=True (default, sweeps): appends at the end (header when the file is
    new). append=False (e.g. a pipeline's comparison.csv): rewrites from
    scratch.
    columns: list of columns (COLUMNS by default; the pipeline adds opt_time
             and the per-phase speedups to it).
    """
    cols = columns or COLUMNS
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = (not append) or not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if write_header:
            w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])
