"""Chained pipeline runner (JSON-configured).

Runs an ordered list of executions described in a JSON file, each into its own
sub-folder under `results/<pipeline name>/<run name>/`, then writes a comparison
(benchmark ~70-column format) measuring speedup, correlations found and quality
of every run against a designated **baseline** run (treated as the reference, in
the role of brute-force ground truth).

    python -m v2.pipeline pipeline.json

JSON schema:
    {
      "name": "asos_backends",          # pipeline name -> results/<name>/
      "dataset": "fr-air_temperature.csv",
      "baseline": 0,                     # index of the reference run (default 0)
      "clean": true,                     # wipe results/<name>/ first (optional)
      "output": "results",               # root output dir (optional)
      "runs": [
        {"name": "python",     "mode": "bf", "params": {"n_series": 5, "n_years": 1}},
        {"name": "vectorized", "mode": "bf", "params": {"n_series": 5, "n_years": 1,
                                                        "backend": "vectorized"}}
      ]
    }

`name` is optional per run (defaults to the auto slug). `mode` is "bf" or
"corrtrack". `params` are Config overrides. The comparison is `<name>/comparison.csv`.
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from dataclasses import fields

from .core.config import Config

_CONFIG_FIELDS = {f.name for f in fields(Config)}


# --- ANSI colors for the console table (speedup / recall / prec / spec) -----
# Disabled when stdout is not a TTY (clean logs) or when NO_COLOR is set.
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _ansi(s, code):
    """Wrap `s` (already padded string) with an ANSI code. Width preserved."""
    return f"\x1b[{code}m{s}\x1b[0m" if (_USE_COLOR and code) else s


def _safe_float(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _color_speedup(v):
    """vert vif ≥ 5× | vert 1-5× | jaune 0.8-1× | rouge < 0.8×"""
    x = _safe_float(v)
    if x is None:           return ""
    if x >= 5:              return "1;32"
    if x >= 1:              return "32"
    if x >= 0.8:            return "33"
    return "31"


def _color_score(v):
    """recall/prec/spec : vert vif ≥ 0.95 | vert ≥ 0.7 | jaune ≥ 0.5 | rouge"""
    x = _safe_float(v)
    if x is None:           return ""
    if x >= 0.95:           return "1;32"
    if x >= 0.7:            return "32"
    if x >= 0.5:            return "33"
    return "31"


# --- compat caches d'avant le rename de mai 2026 -----------------------------
# Mappage non-ambigu (s'applique partout)
_LEGACY_RENAMES = {
    "tree_radius": "query_radius",
    "cell_size": "grid_cell",
    "freq_threshold": "grid_vote_min",
    "index_key_mode": "key_mode",
    "index_key_dim": "key_proj_dim",
}
# Ambiguous mapping (depends on index_backend)
#   n_grids        → grid_n_tables  (grid)  | annoy_n_trees (annoy)
#   grid_dimension → grid_n_coords  (grid)  | quadtree_proj_dim (quadtree)


def _migrate_legacy_params(params):
    """Reconcile a params dict possibly written with OLD names (before the
    May 2026 rename) → current names.

    Never crashes: any unknown field is left as-is and will be ignored by
    `Config.build` (which only iterates over `fields(Config)`).
    """
    if not isinstance(params, dict):
        return params
    out = dict(params)
    ib = out.get("index_backend") or out.get("index-backend")
    # ambiguous
    if "n_grids" in out:
        v = out.pop("n_grids")
        if ib == "annoy":
            out.setdefault("annoy_n_trees", v)
        else:
            out.setdefault("grid_n_tables", v)
    if "grid_dimension" in out:
        v = out.pop("grid_dimension")
        if ib == "quadtree":
            out.setdefault("quadtree_proj_dim", v)
        else:
            out.setdefault("grid_n_coords", v)
    # unambiguous
    for old, new in _LEGACY_RENAMES.items():
        if old in out:
            out.setdefault(new, out.pop(old))
    return out


_PSEUDO_FIELDS = {"optimize_backend"}  # meta keys (non-Config) known to the pipeline


def _norm_params(params):
    """Normalize JSON param keys (- -> _) and warn on keys that aren't Config fields."""
    out = {}
    for k, v in params.items():
        key = k.replace("-", "_")
        if key in _PSEUDO_FIELDS:
            out[key] = v
            continue
        if key not in _CONFIG_FIELDS:
            print(f"[pipeline] WARNING: unknown param '{k}' ignored "
                  f"(not a Config field; valid e.g. index_backend, grid_cell, n_vectors)")
            continue
        out[key] = v
    return out

# fields persisted per run so a result can be reused without re-executing
_RESULT_KEYS = ("n_series", "n_windows", "n_candidates", "n_tested",
                "n_correlated", "n_episodes", "n_anomalies", "runtime")
# "ephemeral" fields that do not affect the result (excluded from the signature)
_EPHEMERAL = {"output", "clean", "profile", "log_level", "log_file", "log_every"}

# params that IDENTIFY what is computed (data + thresholds + algorithm),
# independently of the setting chosen by the optimizer (n_vectors, grid_cell,
# query_radius…). Used for the AUTO UPGRADE of an old result.json (without
# cache_sig): when those keys match the declared run, the old result is valid →
# it is kept and stamped (see load_result). The optim keys need not match.
_IDENTITY_KEYS = {
    "n_series", "n_years", "obs_mode", "window_size", "window_step", "n_lags",
    "basic_window", "corr_threshold", "neg_corr", "std_threshold",
    "index_backend", "sketch_method", "seed",
}


def _atomic_write_json(path, obj):
    """Atomic write (tmp + os.replace): a kill during the write cannot leave a
    corrupted JSON behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _params_signature(config):
    """Signature of a run's params, to invalidate the cache when they change.

    asdict of the Config (dataclass fields) + non-dataclass attrs (e.g. the
    filcorr_* injected by the fillcorr hook) — minus the purely ephemeral fields
    that do not change the result (`output`, `log_*`, `clean`, `profile`).
    """
    import dataclasses
    d = dataclasses.asdict(config)
    for k, v in vars(config).items():
        if k not in d and isinstance(v, (int, float, str, bool, list, tuple)):
            d[k] = v
    return {k: v for k, v in d.items() if k not in _EPHEMERAL}


def _declared_signature(declared_cfg, run, opt_settings):
    """Cache signature of a run, based on its DECLARED params (before optim)
    plus its `optimize` settings — NEVER on the params chosen by the optimizer.

    WHY: for an `optimize: true` run, `save_result` used to store the signature
    of the POST-optim params (n_vectors/grid_cell/… picked by the sweep), while
    on reload the pipeline only knows the PRE-optim params (the declared
    defaults). The two never matched → the run was deemed "incomplete" and
    re-optimized + re-run every time (broken cache for EVERY optimized run).

    This signature is reproduced identically at save, at load and at migration
    time (same inputs: pre-optim cfg + run block + opt_settings from the JSON),
    so an already finished run is recognized as cached.
    """
    sig = dict(_params_signature(declared_cfg))
    if run.get("optimize") or run.get("use_optimized"):
        opt = opt_settings or {}
        # optim settings that change the result → any change invalidates the cache
        sig["__optimize__"] = json.dumps(
            {k: opt.get(k) for k in (
                "grid", "train_ratio", "target_recall", "target_precision",
                "sweep_validate", "successive_halving", "smart_grid",
                "auto_derive", "n_series_opt", "n_years_opt")},
            sort_keys=True, default=str)
    return sig


def save_result(out_dir, res, opt_time=0.0, config=None, cache_sig=None):
    """Persist a run's result (scalars + fine timings + opt_time + params) for reuse.

    ATOMIC write (tmp + os.replace): a kill during the write cannot leave a
    corrupted result.json — either the old one is intact, or the new one is
    complete. Without it, a truncated JSON passed the `os.path.exists()`
    pre-check then crashed json.load() on the next rerun (or worse: let the run
    be considered "cached and finished" although it had been interrupted).
    """
    if not out_dir:
        return
    keep = {k: res[k] for k in _RESULT_KEYS if k in res}
    keep["timings"] = res.get("timings", {})
    # real-time metrics (per-phase throughput + CPU/mem/energy resources)
    if res.get("throughput") is not None:
        keep["throughput"] = res["throughput"]
    if res.get("resources") is not None:
        keep["resources"] = res["resources"]
    if res.get("phase_metrics") is not None:
        keep["phase_metrics"] = res["phase_metrics"]
    keep["opt_time"] = opt_time
    if config is not None:
        # `params` = EFFECTIVE params (post-optim) — used to DISPLAY the params
        # chosen by the optim in the comparison table.
        keep["params"] = _params_signature(config)
    if cache_sig is not None:
        # `cache_sig` = DECLARED signature (pre-optim) — used to INVALIDATE the
        # cache (see _declared_signature). Kept separate from `params` because
        # the two differ for an optimized run.
        keep["cache_sig"] = cache_sig
    _atomic_write_json(os.path.join(out_dir, "result.json"), keep)


def _slim_result(res):
    """Extract from a run result only the SCALARS + timings the comparison
    (`benchmark.build_record`) needs. This allows the full `res` to be released
    afterwards (the `correlated`/`candidates`/`episodes`/`anomalies` lists, which
    weigh tens of GB on dense datasets) without losing anything for the
    comparison row — the quality metrics being streamed from disk."""
    keep = {k: res.get(k) for k in _RESULT_KEYS}
    keep["timings"] = res.get("timings", {})
    if res.get("throughput") is not None:
        keep["throughput"] = res["throughput"]
    if res.get("resources") is not None:
        keep["resources"] = res["resources"]
    if res.get("phase_metrics") is not None:
        keep["phase_metrics"] = res["phase_metrics"]
    return keep


def load_result(out_dir, config=None, declared_sig=None):
    """Reload a previously completed run, or None if anything looks incomplete.

    POLICY (since 2026-05-28): only `result.json` (written ATOMICALLY at the
    very end by `save_result`) marks a run as finished. Any other combination of
    files (correlated.csv alone, summary.csv without result.json, a corrupted
    result.json, a null runtime, an inconsistent params signature…) → None is
    returned and the main loop takes the "incomplete run" branch, which DELETES
    the folder and re-runs from scratch.

    Why: a kill (OOM, SIGTERM, Ctrl-C) can leave partial CSVs behind, which
    passed the former `exists()` pre-check and were then considered "cached" →
    the failed run was never re-run.
    """
    rj = os.path.join(out_dir, "result.json")
    cc = os.path.join(out_dir, "correlated.csv")
    if not os.path.exists(rj) or not os.path.exists(cc):
        return None
    try:
        with open(rj) as f:
            res = json.load(f)
    except (json.JSONDecodeError, OSError):
        # truncated result.json (kill during the write on older code, before the
        # atomic fix) or unreadable → the run is considered incomplete.
        return None
    # sanity: a finished run necessarily has the runtime key and a runtime > 0.
    if "runtime" not in res or not _safe_float(res.get("runtime")):
        return None
    # signature invalidation: the DECLARED signature (pre-optim + optimize
    # settings) is compared — the only one reproducible on reload.
    # `declared_sig` is computed by _declared_signature() on the pipeline side.
    if declared_sig is not None:
        if "cache_sig" in res:
            if res["cache_sig"] != declared_sig:
                return None          # declared params / optimize grid changed
        else:
            # OLD result.json (from before the cache_sig fix): auto upgrade.
            # It is kept IFF its identity (data/thresholds/algorithm) matches the
            # declared run — the keys set by the optim need not match. It is then
            # stamped with cache_sig so it is recognized directly afterwards.
            # Avoids recomputing already finished runs.
            legacy = res.get("params") or {}
            if any(k in declared_sig and legacy.get(k) != declared_sig.get(k)
                   for k in _IDENTITY_KEYS):
                return None
            res["cache_sig"] = declared_sig
            _atomic_write_json(rj, res)
    elif config is not None and "params" in res:
        # legacy path (the caller provides no declared_sig): former behaviour
        if res["params"] != _params_signature(config):
            return None
    # MEMORY: the `correlated` list is NOT loaded into RAM (on dense ASOS it
    # weighs tens of GB → OOM when baseline and run coexist). Only the PATH of
    # correlated.csv is exposed; the quality comparison is streamed from disk
    # (see metrics.compare_streaming).
    res["correlated_path"] = cc
    return res


def load_jsonc(path):
    """Parse a JSON config tolerant of `//` and `/* */` comments + trailing commas.

    Lets you comment out runs/params (e.g. `// "backend": "mps",`). Comment
    stripping is string-aware (it won't touch `//` inside string values).
    """
    text = open(path).read()
    out, i, n = [], 0, len(text)
    in_str = esc = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":      # // line comment
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":      # /* block comment */
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(c)
            i += 1
    cleaned = re.sub(r",(\s*[}\]])", r"\1", "".join(out))        # trailing commas
    return json.loads(cleaned)

from .core import benchmark, metrics, optimize
from .core.cli import clean_output_dir, run_slug
from .core.config import Config
from .core.log import get_logger, reset as reset_logger
from .core.pipeline import run as run_pipeline
from .utils.illustrate import illustrate_run

# Registers the "filcorr" mode: importing the sub-package installs the hooks
# (dispatch in core.pipeline.run + absorption of the filcorr_* params in
# Config.build / _norm_params). See v2/fillcorr/__init__.py.
from . import fillcorr as _filcorr_register  # noqa: F401


def _precompute_optims(runs, base, base_params, opt_settings, dataset, dataset_id,
                       incremental):
    """PRE-PASS: compute IN PARALLEL the optim of every run that needs one.

    Each run keeps ITS OWN optim (nothing is shared); since the optims are
    independent tasks, they are spread over a process pool → the TOTAL optim time
    is divided by the number of concurrent jobs. Each optim runs a SEQUENTIAL
    sweep there (`sweep_workers=1`): the parallelism lives at the RUN level, not
    at the grid level, so there is no oversubscription.

    Returns {run_idx: (best_params|None, opt_time)} for the runs concerned
    (enabled, NOT cached, corrtrack + grid). Parallelism level:
    `CORRTRACK_OPTIM_JOBS` (default min(#jobs, cores−2)). The main loop reads
    those results back.
    """
    if not opt_settings.get("grid"):
        return {}
    jobs = []  # (run_idx, run_explicit, opt_dir, tag)
    for idx, run in enumerate(runs):
        if run.get("enabled", True) is False or run.get("skip") is True:
            continue
        if run.get("mode", "corrtrack") != "corrtrack":
            continue
        if not (run.get("optimize") or run.get("use_optimized")):
            continue
        run_explicit = _norm_params(run.get("params", {}))
        cfg = Config.build(**{**base_params, **run_explicit})
        run_name = run.get("name") or run_slug(cfg, run.get("mode", "corrtrack"))
        run_dir = os.path.join(base, run_name)
        declared_sig = _declared_signature(cfg, run, opt_settings)
        rerun = run.get("rerun", not incremental)
        if not rerun and load_result(run_dir, declared_sig=declared_sig) is not None:
            continue  # already cached → no optim to redo
        jobs.append((idx, run_explicit, os.path.join(run_dir, "optimize"), run_name))
    if not jobs:
        return {}

    n = len(jobs)
    budget = max(1, (os.cpu_count() or 2) - 2)
    try:
        budget = int(os.environ.get("CORRTRACK_OPTIM_JOBS", budget))
    except ValueError:
        pass
    budget = max(1, budget)

    # TWO levels of parallelism (mutually exclusive, so process pools are NOT
    # nested — fragile under spawn):
    #   • "runs" (default): N optims IN PARALLEL, each grid SEQUENTIAL.
    #                       Optimal when there are many runs (≥ cores).
    #   • "grid"          : SEQUENTIAL runs, but the GRID of each optim is
    #                       parallelized over `budget` workers (the XX configs in
    #                       parallel). Useful with few runs / for a fast optim.
    # Selection: env CORRTRACK_OPTIM_GRID (1=grid, 0=runs). AUTO when unset:
    # grid as soon as there is only ONE run to optimize (otherwise runs).
    env_grid = os.environ.get("CORRTRACK_OPTIM_GRID")
    grid_mode = (env_grid.strip().lower() in ("1", "true", "yes", "on")
                 if env_grid is not None else (n == 1))

    out = {}
    if grid_mode:
        print(f"[pipeline] PARALLEL optim (GRID mode): {n} run(s) in SERIES, "
              f"grid of each optim over {budget} worker(s) (configs in parallel)")
        for idx, re_, od, tag in jobs:
            out[idx] = _optimize_run(opt_settings, base_params, re_, dataset,
                                     dataset_id, od, tag=tag, level="info",
                                     sweep_workers=budget)
        return out

    # FLAT POOL (multi-run DEFAULT): flattens every (run × config) into a single
    # queue of independent tasks over `budget` workers → ~100% occupancy, no tail
    # effect and no under-utilization. Also handles successive_halving (each SH
    # stage flattened across runs). Disable it with CORRTRACK_OPTIM_FLAT=0.
    flat_env = os.environ.get("CORRTRACK_OPTIM_FLAT", "1").strip().lower()
    if flat_env not in ("0", "false", "no", "off"):
        return _run_optims_flat(jobs, opt_settings, base_params, dataset,
                                dataset_id, budget)

    pool = max(1, min(budget, n))
    if pool == 1:
        # A single run to optimize → no parallelism at the RUN level, so the
        # GRID of that optim can be parallelized over the RUN's workers without
        # risking nested process pools. `sweep_workers=None` ⇒ _optimize_run uses
        # run_workers → a REAL measurement of the optim's parallel gain.
        print(f"[pipeline] optim (RUNS mode): {n} run in SERIES, grid parallel "
              f"over the run's workers")
        for idx, re_, od, tag in jobs:
            out[idx] = _optimize_run(opt_settings, base_params, re_, dataset,
                                     dataset_id, od, tag=tag, level="error",
                                     sweep_workers=None)
        return out
    print(f"[pipeline] PARALLEL optim (RUNS mode): {n} run(s) over {pool} "
          f"worker(s) — one optim per run (sequential grid)")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=pool, mp_context=ctx) as ex:
        futs = {ex.submit(_optimize_run, opt_settings, base_params, re_, dataset,
                          dataset_id, od, tag, "error", 1): idx
                for idx, re_, od, tag in jobs}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                out[idx] = fut.result()
            except Exception as exc:                # a failed optim does not break everything
                print(f"[pipeline] WARNING: optim of run #{idx} failed ({exc}) "
                      f"— static params used")
                out[idx] = (None, 0.0)
    return out


def _compute_kernel(backend):
    """COMPUTE backend (kernel) of a run backend, for the optim sweep.

    Strips the parallelization wrappers — useless/counter-productive on the small
    optim subset, and oversubscribing when N optims run in parallel:
        sharded_X      -> X         (sharded_vectorized -> vectorized)
        X_parallel     -> X         (cython_parallel    -> cython)
        parallel       -> python    (parallel = pure python pool)
        vectorized/cython/python    -> unchanged
    Keeps the KERNEL → a tie-break representative of the real execution.
    """
    b = (backend or "python")
    if b.startswith("sharded_"):
        b = b[len("sharded_"):]
    if b == "parallel":
        return "python"
    if b.endswith("_parallel"):
        b = b[:-len("_parallel")]
    return b or "python"


def _optimize_run(settings, base_params, run_explicit, dataset, dataset_id, opt_dir,
                  tag="", level="info", sweep_workers=None,
                  return_configs=False, prebuilt=None, prebuilt_opt_time=0.0,
                  prebuilt_workers=0):
    """Sweep the central grid on a SMALL subset (`train_ratio`, default 0.1),
    select the best config, write optimize_stats.csv + best_params.json into
    `opt_dir`. Returns (best_params dict, opt_time); best_params None if no winner.
    Logs a banner + the swept configs (recall/speedup/corr%) to the console and
    `opt_dir/optimize.log`.

    Precedence: base_params (swept if in grid) < grid (optimized) < run_explicit
    (STATIC — a param the run sets explicitly is fixed even if it's in the grid).
    The grid may include execution params (`backend`, `workers`, `cores`):
    since recall is identical across them, the selection picks the fastest.

    `workers` (sweep parallelism) is now inherited from the run itself
    (`run_explicit["workers"]` or `base_params["workers"]`) — there is no global
    `optimize.workers` any more.
    """
    # the run's workers = single source for the sweep parallelization
    run_workers = int(run_explicit.get("workers")
                      or base_params.get("workers") or 0)
    grid = {k: (v if isinstance(v, list) else [v])
            for k, v in _norm_params(settings.get("grid", {})).items()}

    auto = bool(settings.get("auto_derive"))
    smart = bool(settings.get("smart_grid")) or auto
    fixed = run_explicit  # user-set per-run params (never touched)
    here = lambda k: k in fixed or k in base_params or k in grid

    # AUTO-DERIVE: empirically invariant axes (on this run) + theoretical bounds.
    # Forces the grid (the run's explicit `params` still wins).
    if auto:
        for k, v in (("grid_n_coords", 1), ("sketch_threshold", 0.7),
                     ("grid_vote_min", 1)):
            if k not in fixed:
                grid[k] = [v]
        # theoretical LSH bound of the radius: √(2·D·(1−thr)) on z-normalized sketches
        nv = (fixed.get("n_vectors") or base_params.get("n_vectors")
              or grid.get("n_vectors", [16])[0])
        thr = (fixed.get("corr_threshold") or base_params.get("corr_threshold")
               or grid.get("corr_threshold", [0.8])[0])
        qr_max = (2.0 * float(nv) * (1.0 - float(thr))) ** 0.5
        if "query_radius" in grid:
            kept = [r for r in grid["query_radius"] if r <= qr_max]
            grid["query_radius"] = kept or grid["query_radius"]

    # SMART-GRID: sweep only the axes RELEVANT to the chosen index (grid_* has
    # no effect on kdtree/tree/bptree/bst; query_radius none on grid/knn)
    if smart:
        idx = (fixed.get("index_backend") or base_params.get("index_backend")
               or grid.get("index_backend", ["grid"])[0])
        per_index_axes = {
            "grid":     {"grid_cell", "grid_n_tables", "grid_vote_min", "grid_n_coords"},
            "quadtree": {"query_radius", "quadtree_proj_dim"},
            "octree":   {"query_radius"},
            "knn":      {"n_neighbors"},
            "bptree3d": {"query_radius", "key_mode"},
            "bst3d":    {"query_radius", "key_mode"},
            "hnsw":     {"query_radius", "n_neighbors", "hnsw_m", "hnsw_ef_s"},
            "vptree":   {"query_radius"},
            "annoy":    {"query_radius", "annoy_n_trees", "annoy_leaf_size"},
            "tree":     {"query_radius"},
            "kdtree":   {"query_radius"},
            "bptree":   {"query_radius", "key_mode", "key_proj_dim"},
            "bst":      {"query_radius", "key_mode", "key_proj_dim"},
        }
        always = {"n_vectors", "sketch_threshold", "backend", "workers",
                  "seed", "corr_threshold", "neg_corr", "std_threshold",
                  "window_size", "window_step", "n_lags", "n_series", "n_years",
                  "obs_mode", "train_ratio", "index_backend"}
        relevant = per_index_axes.get(idx, set()) | always
        grid = {k: v for k, v in grid.items() if k in relevant}

    # `optimize_backend` (per-run) only drives the optim — it is extracted BEFORE
    # the merge into axes so it does not leak into the swept configs.
    run_explicit_clean = {k: v for k, v in run_explicit.items()
                          if k != "optimize_backend"}
    axes = {}
    axes.update({k: [v] for k, v in base_params.items()})
    axes.update(grid)                                            # grid sweeps
    axes.update({k: [v] for k, v in run_explicit_clean.items()}) # static run params win
    # Sweep backend — precedence (highest to lowest priority):
    #   1. run.params.optimize_backend (per-run override)
    #   2. optimize.backend (global top-level override)
    #   3. ALIGNED ON THE RUN: the run's own backend, reduced to its compute
    #      KERNEL (sharded_*/*_parallel → vectorized/cython/python). The
    #      parallelization wrapper is stripped because the sweep runs on a SMALL
    #      subset (where sharding/forking brings nothing and would oversubscribe
    #      under parallel runs); the kernel is enough for a tie-break
    #      representative of the real execution.
    #      → cython run ⇒ cython, sharded_vectorized ⇒ vectorized, etc.
    #   4. "python" as a last resort.
    # The run's FINAL backend (cfg.backend) is NOT affected.
    opt_be = run_explicit.get("optimize_backend") or settings.get("backend")
    if not opt_be:
        run_be = (run_explicit.get("backend") or base_params.get("backend") or "python")
        opt_be = _compute_kernel(run_be)
    axes["backend"] = [opt_be]
    axes["train_ratio"] = [settings.get("train_ratio", 0.1)]  # small subset by default
    # sweep_validate=false : skip the Pearson validation phase during the sweep
    # (faster, but precision/val_time/speedup reflect candidate generation only).
    # Not propagated to the final run (default True restored from Config).
    axes["sweep_validate"] = [bool(settings.get("sweep_validate", True))]
    # The swept configs MUST NOT persist: without this, each sweep run inherits
    # the Config.output="results" default and writes
    # correlated/summary/episodes/anomalies.csv at the results/ root (pollution).
    # "" = no writing (see config.py). The FINAL run writes into its run_dir.
    axes["output"] = [""]
    # REDUCED target for the optim only (`n_series_opt` / `n_years_opt`): the
    # sweep runs on fewer series / years → far shorter. The FINAL run keeps the
    # n_series/n_years of `base_params` (unaffected). Combines with `train_ratio`
    # (time prefix) already applied above.
    if settings.get("n_series_opt"):
        axes["n_series"] = [int(settings["n_series_opt"])]
    if settings.get("n_years_opt"):
        axes["n_years"] = [int(settings["n_years_opt"])]
    configs = list(benchmark.build_configs(axes))
    n = len(configs)
    os.makedirs(opt_dir, exist_ok=True)

    # FLAT POOL: only the configs are BUILT here (tagged upstream); the sweep is
    # executed by the orchestrator in a single multi-run queue.
    if return_configs:
        return axes, grid, configs

    reset_logger()
    log = get_logger(level, "")  # console banner (visible in the [INFO] stream)
    bar = "-" * 64
    log.info(bar)
    workers_hint = run_workers
    extra_h = ""
    if opt_be:
        extra_h += f" backend={opt_be}"
    if workers_hint > 1:
        extra_h += f" workers={workers_hint}"
    if not axes["sweep_validate"][0]:
        extra_h += " sweep_validate=false"
    if settings.get("n_series_opt"):
        extra_h += f" n_series_opt={int(settings['n_series_opt'])}"
    if settings.get("n_years_opt"):
        extra_h += f" n_years_opt={int(settings['n_years_opt'])}"
    log.info("--- OPTIMIZE %s: sweeping %d configs (train_ratio=%s)%s…",
             tag, n, axes["train_ratio"][0], extra_h)
    log.info(bar)

    t0 = time.perf_counter()
    reset_logger()
    get_logger("error", "")  # silence the internal sweep (each config is a full run)

    # progress checkpoints every ~5 s (the sweep's own logger is silenced, so we
    # print to stderr in the [INFO] format, like the pipeline checkpoints).
    last = [t0]

    def _progress(done, total, rows_so_far):
        now = time.perf_counter()
        if now - last[0] < 5.0 and done < total:
            return
        last[0] = now
        best_r = max((float(r.get("recall") or 0) for r in rows_so_far
                      if r.get("status") == "success"), default=0.0)
        eta = (now - t0) / done * (total - done) if done else 0.0
        sys.stderr.write(f"[INFO]   optim {tag}: {done}/{total} configs "
                         f"({now - t0:.0f}s, best recall={best_r:.3f}, ETA {eta:.0f}s)\n")

    # FLAT POOL: rows already computed by the orchestrator (single multi-run
    # queue) → the sweep is skipped and we finalize directly (selection + write).
    if prebuilt is not None:
        rows = list(prebuilt)
        rows_for_select = rows
        opt_time = float(prebuilt_opt_time)
        workers = int(prebuilt_workers)
    else:
        # `sweep_workers` forces the sweep's INTERNAL parallelism (grid). In the
        # parallel pre-pass (optims of several runs at once) it is set to 1 so as
        # not to oversubscribe: the parallelism THEN lives at the RUN level, not
        # at the grid level.
        workers = run_workers if sweep_workers is None else int(sweep_workers)
        sh = settings.get("successive_halving")
        if sh:
            # Successive Halving: k stages with an increasing budget (train_ratio
            # × 2 at each stage), top 50% retained, last budget = final
            # train_ratio. Saves ~50% of the total time vs a full sweep on
            # everything.
            from dataclasses import replace
            n_stages = int(sh) if isinstance(sh, int) and sh > 1 else 3
            base_tr = axes["train_ratio"][0]
            sh_configs = benchmark.build_configs(axes)
            rows = []
            for s in range(n_stages):
                tr = base_tr * (0.5 ** (n_stages - 1 - s))
                stage_cfgs = [replace(c, train_ratio=tr) for c in sh_configs]
                sys.stderr.write(f"[INFO]   SH {tag} stage {s+1}/{n_stages}: "
                                 f"{len(stage_cfgs)} configs @ train_ratio={tr:.3f}\n")
                stage_rows = benchmark.sweep({}, dataset, dataset_id=dataset_id,
                                             progress=_progress, workers=workers,
                                             configs=stage_cfgs)
                rows.extend(stage_rows)
                if s < n_stages - 1:                # keep the top 50% by recall
                    def _r(rw):
                        try: return float(rw.get("recall") or 0)
                        except: return 0.0
                    ranked = sorted(zip(sh_configs, stage_rows), key=lambda x: -_r(x[1]))
                    keep = max(1, len(sh_configs) // 2)
                    sh_configs = [c for c, _ in ranked[:keep]]
            opt_time = time.perf_counter() - t0
            # NB: select_best only looks at the rows of the LAST stage (full budget)
            final_rows = rows[-len(stage_rows):]
            rows_for_select = final_rows
        else:
            rows = benchmark.sweep(axes, dataset, dataset_id=dataset_id,
                                   progress=_progress, workers=workers)
            opt_time = time.perf_counter() - t0
            rows_for_select = rows
    benchmark.write_rows(os.path.join(opt_dir, "optimize_stats.csv"), rows, append=False)
    best = optimize.select_best(
        rows_for_select, settings.get("target_recall", 0.95),
        settings.get("recall_fallback_near_ratio", 0.98),
        settings.get("speedup_near_ratio", 0.98),
        target_precision=settings.get("target_precision", 0.0))

    reset_logger()
    log = get_logger(level, os.path.join(opt_dir, "optimize.log"))  # results -> console + file

    def _g(r, k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return float("nan")
    gkeys = list(grid)
    for r in sorted(rows, key=lambda r: (-_g(r, "recall"), -_g(r, "speedup"))):
        gv = " ".join(f"{k}={r.get('_config', {}).get(k)}" for k in gkeys)
        cwp = 100 * _g(r, "corr_w") / _g(r, "corr_w_bf") if _g(r, "corr_w_bf") else float("nan")
        mark = " <= best" if r is best else ""
        log.info("  optim %s | recall=%.3f speedup=%6.2fx corr%%=%5.1f | %s%s",
                 tag, _g(r, "recall"), _g(r, "speedup"), cwp, gv, mark)
    if best is None:
        log.warning("  optimize %s: no successful config.", tag)
        return None, opt_time
    cfg_dict = best.get("_config", {})
    keys = set(base_params) | set(grid) | set(run_explicit)
    params = {k: cfg_dict[k] for k in keys if k in cfg_dict}
    # `n_series_opt`/`n_years_opt`: the reduced target serves the optim ONLY —
    # it must not leak into the best params (otherwise the final run would use the
    # subset). The final run falls back to base_params' n_series/n_years.
    for k in ("n_series", "n_years"):
        if settings.get(f"{k}_opt"):
            params.pop(k, None)
    # Reference serial time of the sweep = sum of the per-config runtimes
    # (1 thread). `opt_time` = actual wall clock of the sweep (== parallel
    # makespan when `workers`>1). `sweep_workers` = REAL parallelism of the grid
    # (1 = sequential grid). → viz can display the true serial↔parallel gain of
    # the optim.
    def _grt(r):
        try:
            return float(r.get("runtime") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    opt_serial = sum(_grt(r) for r in rows if r.get("status") == "success")
    with open(os.path.join(opt_dir, "best_params.json"), "w") as f:
        json.dump({"params": params, "recall": best.get("recall"),
                   "speedup": best.get("speedup"), "cand_w": best.get("cand_w"),
                   "opt_time": opt_time, "opt_serial": opt_serial,
                   "sweep_workers": int(workers)}, f, indent=2)
    log.info("--- OPTIMIZE %s best: recall=%.3f -> %s (%.1fs) ---",
             tag, _g(best, "recall"), params, opt_time)
    target = settings.get("target_recall", 0.0)
    if target and _g(best, "recall") < target:
        idx = cfg_dict.get("index_backend", "grid")
        lever = ("query_radius" if idx in ("kdtree", "tree", "bptree", "bst",
                                            "octree", "quadtree", "vptree",
                                            "annoy", "hnsw", "bptree3d", "bst3d")
                 else "n_neighbors" if idx == "knn"
                 else "grid_n_coords (→1), grid_cell, grid_n_tables")
        swept = set(grid)
        log.warning("  optimize %s: target_recall=%.2f NOT reached (best=%.3f). For "
                    "index=%s the recall lever is %s — add it to the optimize grid "
                    "(currently sweeping: %s).", tag, target, _g(best, "recall"), idx,
                    lever, ", ".join(sorted(swept)) or "nothing")
    return params, opt_time


def _eval_config_task(config, path, dataset_id, bf):
    """Atomic task of the flat pool: evaluate ONE config → a stats row.
    Module-level function (picklable) submitted to the ProcessPool."""
    return benchmark.compare(config, path, dataset_id, bf=bf)


def _run_optims_flat(jobs, opt_settings, base_params, dataset, dataset_id, budget):
    """Unified FLAT pool: flattens EVERY (run × config) into ONE queue of
    independent tasks over `budget` workers.

    Replaces the 2-level scheme (runs in parallel, each grid in series), which
    left workers idle at the end of the pool (tail effect) and under-used the
    cores when there were few runs. Here each config evaluation is one task (bf
    shared through `_bf_key`), all in the same queue → ~100% occupancy.
    Compatible with `successive_halving`: EACH SH stage is flattened across all
    the runs (top-50% pruning per run between stages). The final rows are grouped
    per run for the usual selection + write (`_optimize_run(prebuilt=…)`).
    Returns {run_idx: (params, opt_time)}.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dataclasses import replace
    import multiprocessing as mp

    def _rt(r):
        try:
            return float((r or {}).get("runtime") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _recall(r):
        try:
            return float((r or {}).get("recall") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # 1) build the configs of each run (the sweep is NOT executed here)
    built = {}          # idx -> {"re","od","tag","configs","base_tr"}
    for idx, re_, od, tag in jobs:
        try:
            axes, _grid, configs = _optimize_run(
                opt_settings, base_params, re_, dataset, dataset_id, od,
                tag=tag, return_configs=True)
        except Exception as exc:
            print(f"[pipeline] WARNING: build configs run #{idx} failed ({exc})")
            continue
        built[idx] = {"re": re_, "od": od, "tag": tag, "configs": configs,
                      "base_tr": axes["train_ratio"][0]}
    if not built:
        return {}

    # A WAVE = run a list of (key, config) through the single queue of `budget`
    # workers; returns {key: row} (key = task identifier provided by the caller →
    # order/grouping preserved despite as_completed).
    def _wave(tagged, label):
        if not tagged:
            return {}
        bf_cache = benchmark._build_bf_cache([c for _, c in tagged], dataset)
        ctx = mp.get_context("spawn")
        results, total = {}, len(tagged)
        t0, last, done = time.perf_counter(), time.perf_counter(), 0
        print(f"[pipeline] FLAT POOL optim{label}: {total} configs "
              f"({len(built)} run(s)) over {budget} worker(s) — single queue")
        with ProcessPoolExecutor(max_workers=max(1, int(budget)), mp_context=ctx) as ex:
            futs = {ex.submit(_eval_config_task, c, dataset, dataset_id,
                              bf_cache[benchmark._bf_key(c)]): key
                    for key, c in tagged}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    results[key] = fut.result()
                except Exception as exc:
                    print(f"[pipeline] WARNING: config {key} failed ({exc})")
                done += 1
                now = time.perf_counter()
                if now - last >= 5.0 or done == total:
                    last = now
                    eta = (now - t0) / done * (total - done) if done else 0.0
                    sys.stderr.write(f"[INFO]   FLAT POOL optim{label}: {done}/{total} "
                                     f"configs ({now - t0:.0f}s, ETA {eta:.0f}s)\n")
        wall = time.perf_counter() - t0
        serial = sum(_rt(r) for r in results.values())
        spd = serial / wall if wall > 0 else 0.0
        print(f"[pipeline] FLAT POOL optim{label} done: wall={wall:.1f}s, "
              f"serial≈{serial:.0f}s → {spd:.1f}× over {budget} worker(s)")
        return results

    # 2) execution: SH stage by stage (each flattened across runs), or one wave
    sh = opt_settings.get("successive_halving")
    final_rows = {idx: [] for idx in built}
    if sh:
        n_stages = int(sh) if isinstance(sh, int) and sh > 1 else 3
        surviving = {idx: list(b["configs"]) for idx, b in built.items()}
        for s in range(n_stages):
            tagged = []
            for idx, b in built.items():
                tr = b["base_tr"] * (0.5 ** (n_stages - 1 - s))
                tagged += [((idx, pos), replace(c, train_ratio=tr))
                           for pos, c in enumerate(surviving[idx])]
            res = _wave(tagged, f" SH {s + 1}/{n_stages}")
            # group per run, IN THE ORDER of the surviving configs (for the zip)
            stage = {idx: [res.get((idx, pos)) for pos in range(len(surviving[idx]))]
                     for idx in built}
            if s < n_stages - 1:                       # top-50% pruning per run
                for idx in built:
                    pairs = [(c, r) for c, r in zip(surviving[idx], stage[idx]) if r]
                    pairs.sort(key=lambda cr: -_recall(cr[1]))
                    surviving[idx] = [c for c, _ in pairs[:max(1, len(pairs) // 2)]]
            else:
                final_rows = {idx: [r for r in stage[idx] if r] for idx in built}
    else:
        tagged = [((idx, pos), c) for idx, b in built.items()
                  for pos, c in enumerate(b["configs"])]
        res = _wave(tagged, "")
        final_rows = {idx: [res[(idx, pos)] for pos in range(len(b["configs"]))
                            if (idx, pos) in res]
                      for idx, b in built.items()}

    # 3) finalize each run from its rows (selection + best_params.json).
    #    prebuilt_workers=1: the reported per-run cost stays the serial sum (the
    #    real gain of the flat pool is GLOBAL, logged above); viz keeps its
    #    per-run serial↔parallel projection.
    out = {}
    for idx, b in built.items():
        rows = final_rows.get(idx) or []
        try:
            out[idx] = _optimize_run(
                opt_settings, base_params, b["re"], dataset, dataset_id, b["od"],
                tag=b["tag"], level="error", prebuilt=rows,
                prebuilt_opt_time=sum(_rt(r) for r in rows), prebuilt_workers=1)
        except Exception as exc:
            print(f"[pipeline] WARNING: finalize run #{idx} failed ({exc})")
            out[idx] = (None, 0.0)
    return out


def _save_run_config(run_dir, run_spec, cfg, run_idx):
    """Write `<run_dir>/config.json` — snapshot of the run, to reproduce it.

    Contains (a) the run's JSON block as it was in the pipeline (name, mode,
    params, optimize, enabled, …) and (b) the effective config used by the
    execution (post-optim, post-merge with base_params) under
    `effective_params`. Saved BEFORE execution → present even if the run
    crashes.
    """
    from dataclasses import asdict
    snapshot = {
        "run_index": run_idx,
        "run": run_spec,                           # as-is in the pipeline JSON
        "effective_params": asdict(cfg),           # every resolved Config field
    }
    try:
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(snapshot, fh, indent=2, default=str)
    except OSError:
        pass  # do not fail the run when the folder is not writable


def _build_comparison_row(base_dir, run_name, mode, cfg, baseline, baseline_corr_path,
                          res, run_corr_path, opt_time, log=None):
    """Compute a run's comparison row against the baseline and write missed.csv.

    Extracted so the rows can be built while the main loop streams (instead of
    deferring until ALL the runs have stored their full `res` in memory — which
    blows up the RSS on 100+ runs).

    The quality metrics (precision/recall/f1/missed) are computed by STREAMING
    from both `correlated.csv` (see metrics.compare_streaming): neither the
    baseline nor the run keeps its `correlated` list in RAM.
    """
    missed_path = os.path.join(base_dir, run_name, "missed.csv")
    qual, missed_count = metrics.compare_streaming(
        baseline_corr_path, run_corr_path,
        total_pairs_bf=baseline.get("n_tested", 0),
        missed_path=missed_path, log=log)
    row = benchmark.build_record(cfg, run_name, baseline, res, alg=mode,
                                 metrics_override=qual)
    row["missed"] = missed_count
    row["opt_time"] = opt_time
    # per-phase speedup vs baseline (baseline_phase / run_phase)
    # sk_time_bf=0 when baseline=bf (no sketch) → undefined speedup (NaN, shown as "—")
    sk_bf = float(row.get("sk_time_bf", 0) or 0)
    row["sk_speedup"] = (metrics._safe_div(sk_bf, row["sk_time"])
                         if sk_bf > 0 else float("nan"))
    row["cand_speedup"] = metrics._safe_div(row["cand_time_bf"], row["cand_time"])
    row["val_speedup"] = metrics._safe_div(row["val_time_bf"], row["val_time"])
    row["monit_speedup"] = metrics._safe_div(row["monit_time_bf"], row["monit_time"])
    # proportion of the baseline's candidates / correlations (in %)
    cwp = metrics._safe_div(row["cand_w"], row["cand_w_bf"])
    cop = metrics._safe_div(row["corr_w"], row["corr_w_bf"])
    row["cand_w_pct"] = 100.0 * cwp if cwp == cwp else cwp
    row["corr_w_pct"] = 100.0 * cop if cop == cop else cop
    # REAL-TIME metrics (windows/s throughput + resources) — scalar summary in
    # the aggregated report. The full curve lives in throughput.csv.
    _nan = float("nan")
    _tp = (res.get("throughput") or {}) if isinstance(res, dict) else {}
    _rs = (res.get("resources") or {}) if isinstance(res, dict) else {}
    _win = _tp.get("windows") or {}
    row["tput_win_min"] = _win.get("min", _nan)
    row["tput_win_median"] = _win.get("median", _nan)
    row["tput_win_max"] = _win.get("max", _nan)
    row["tput_win_overall"] = _win.get("overall", _nan)
    row["cpu_pct_median"] = (_rs.get("cpu_pct") or {}).get("median", _nan)
    row["mem_peak_mb"] = _rs.get("mem_peak_mb", _nan)
    row["energy_cpu_seconds"] = _rs.get("energy_cpu_seconds", _nan)
    # number of configs swept during this run's optimization (if any)
    opt_csv = os.path.join(base_dir, run_name, "optimize", "optimize_stats.csv")
    row["opt_n"] = (sum(1 for _ in open(opt_csv)) - 1) if os.path.exists(opt_csv) else 0
    # total wall-time (one-shot cost = optim + run) — vs `runtime`, which is the
    # amortized cost once the optim is frozen (through pipeline-run-best.json).
    try:
        row["total_time"] = float(row["runtime"]) + float(row["opt_time"])
    except (TypeError, ValueError):
        row["total_time"] = row["runtime"]
    return row


TEMPLATE = {
    "name": "my_pipeline",
    "dataset": "data.csv",
    "baseline": 0,                     # index of the reference run (speedup/recall vs it)
    "clean": True,
    "output": "results",
    "params": {"n_series": 5, "n_years": 1},   # shared by all runs; each run overrides
    # central optimize settings; a run runs its own sweep with "optimize": true
    "optimize": {
        # recall levers: grid_n_coords=1 + bigger grid_cell + more grid_n_tables → higher recall
        "grid": {"n_vectors": [16, 32], "grid_n_coords": [1, 2],
                 "grid_cell": [0.5, 1.0], "grid_n_tables": [8, 16],
                 "sketch_threshold": [0.6, 0.8],
                 "query_radius": [2, 4, 6, 8], "n_neighbors": [32, 64, 128, 256]},
        "train_ratio": 0.2, "target_recall": 0.9, "target_precision": 0.0,
        # "sweep_validate": false,  # skip Pearson during sweep (faster, candidate-precision only)
        # REDUCED target for the optim only (final run = n_series/n_years of params):
        # "n_series_opt": 5, "n_years_opt": 1,   # far shorter sweep on large datasets
    },
    "runs": [
        {"name": "bf_python", "mode": "bf"},
        {"name": "bf_vectorized", "mode": "bf", "params": {"backend": "vectorized"}},
        {"name": "corrtrack_static", "mode": "corrtrack",
         "params": {"backend": "vectorized", "grid_cell": 0.5, "n_vectors": 16},
         # "illustrate": which steps to render (JPG) + windows
         "illustrate": {"sketch": True, "candidates": True, "validate": True, "windows": [0]}},
        {"name": "corrtrack_optimized", "mode": "corrtrack", "optimize": True,
         "params": {"backend": "vectorized"},
         "illustrate": {"candidates": True, "windows": [0]}},
    ],
}


def _recompile_cython_kernels(force=True):
    """Recompile the Cython kernels INPLACE when the pipeline starts → the
    LATEST version of the `.pyx` is always used (the loaders load the `.so`
    placed in the package first — `cython.py` and `fillcorr/backends.py` try
    `from . import _kernels` before pyximport). `force=True` forces the rebuild
    even when the `.pyx` has not changed ("latest version" guarantee,
    ~2 s/kernel).

    SILENT fallback when Cython/pyximport is missing or the compilation fails:
    the `cython` backend then falls back to the pure-Python loops (see
    cython.py).
    """
    try:
        import numpy
        from pyximport import pyxbuild
    except Exception as e:
        print(f"[pipeline] Cython/pyximport missing — recompilation skipped ({e}).")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for sub, fn in (("backends", "_cython_kernels.pyx"), ("fillcorr", "_kernels.pyx")):
        pyx = os.path.join(here, sub, fn)
        if not os.path.exists(pyx):
            continue
        try:
            pyxbuild.pyx_to_dll(pyx, force_rebuild=bool(force), inplace=True,
                                setup_args={"include_dirs": [numpy.get_include()]})
            print(f"[pipeline] Cython recompiled (inplace): v2/{sub}/{fn}")
        except Exception as e:
            print(f"[pipeline] WARNING: Cython recompilation of v2/{sub}/{fn} failed "
                  f"({e}) — pure-Python fallback.")


def main():
    p = argparse.ArgumentParser(description="CorrTrack v2 — chained pipeline runner (JSON).")
    p.add_argument("config", nargs="?", help="JSON pipeline configuration.")
    p.add_argument("--dataset", default=None, help="Override the dataset CSV path.")
    p.add_argument("--init", nargs="?", const="pipeline.json", default=None,
                   metavar="PATH",
                   help="Write a template JSON config to PATH (default ./pipeline.json) and exit.")
    p.add_argument("--no-cython-rebuild", action="store_true",
                   help="Do NOT recompile the Cython kernels at startup "
                        "(by default they are recompiled inplace so the latest "
                        "version of the .pyx is always used).")
    args = p.parse_args()

    if args.init is not None:
        if os.path.exists(args.init):
            print(f"[pipeline] {args.init} already exists — not overwritten.")
            return
        with open(args.init, "w") as f:
            json.dump(TEMPLATE, f, indent=2)
        print(f"[pipeline] template written to {args.init} — edit it then run: "
              f"python -m v2.pipeline {args.init}")
        return

    if not args.config:
        p.error("a JSON config is required (or use --init to generate a template).")

    # 1st step: recompile the Cython kernels so the latest .pyx version is always
    # used (unless --no-cython-rebuild, which skips the step entirely).
    if not args.no_cython_rebuild:
        _recompile_cython_kernels(force=True)

    spec = load_jsonc(args.config)

    name = spec.get("name", "pipeline")
    dataset = args.dataset or spec["dataset"]
    base = os.path.join(spec.get("output", "results"), name)
    clean_output_dir(base, spec.get("clean", False))

    base_params = _norm_params(spec.get("params", {}))  # shared by all runs (each run overrides)
    incremental = spec.get("incremental", False)  # reuse cached results by default
    runs = list(spec["runs"])
    # --- BASELINE = the SLOWEST brute-force present ------------------------------
    # bf is the REFERENCE for speedup/recall/precision. When several bf coexist
    # (python/cython/vectorized/mps…), the reference is the SLOWEST one (the most
    # "pure", typically python). Runtimes are unknown before execution → they are
    # ranked by backend (python = slowest → mps/cuda = fastest). The baseline is
    # forced to the front (#0): required to build the comparison row on the fly
    # and to free memory run by run (on 100+ dense ASOS runs, retaining every
    # `res` overflows 32 GB).
    # OPT-OUT: `"baseline": false` disables auto-adding a missing bf_python
    #          (brute-force bf is slow and not always needed).
    _BF_SLOWNESS = {"python": 4, "parallel": 3, "cython": 2,
                    "vectorized": 1, "vectorized_parallel": 1,
                    "mps": 0, "coreml": 0, "cuda": 0}
    def _bf_slowness(r):
        be = (r.get("params") or {}).get("backend") or "python"
        return _BF_SLOWNESS.get(be, 2)

    auto_bf = spec.get("baseline", 0) is not False
    bf_idx = [i for i, r in enumerate(runs) if r.get("mode") == "bf"]
    if not bf_idx and auto_bf:
        print("[pipeline] WARNING: no brute-force present — 'bf_python' prepended "
              "(reference for speedup/recall).")
        runs.insert(0, {"name": "bf_python", "mode": "bf"})
        bf_idx = [0]

    if bf_idx:
        # slowest bf; on equal slowness, the first one encountered (-i)
        baseline_pos = max(bf_idx, key=lambda i: (_bf_slowness(runs[i]), -i))
        if baseline_pos != 0:
            print(f"[pipeline] baseline = slowest bf (#{baseline_pos}: "
                  f"{runs[baseline_pos].get('name')}) — moved to the front.")
            runs.insert(0, runs.pop(baseline_pos))
    spec["baseline"] = 0

    # NB: the optim does NOT depend on a pipeline-level bf. The sweep computes
    # its OWN bf on the `train_ratio` subsample (see benchmark.sweep) — fast. The
    # pipeline-level bf (slow, full dataset) is ONLY the reference for the final
    # speedup/recall columns. With `"baseline": false` it is omitted: the optim
    # still runs, and speedup/recall become relative to run #0 (a red warning is
    # printed at the end as a reminder).

    # central optimize settings (grid + selection knobs); a run opts in per step
    # via "optimize": true (or "use_optimized": true) -> its own sweep is run.
    opt_settings = spec.get("optimize") or {}

    results_meta = []   # (run_name, mode, cfg, opt_time) — without res (freed)
    rows = []           # comparison rows built on the fly
    # PRE-PASS: the runs' optims (independent tasks) are computed IN PARALLEL —
    # each run keeps its OWN optim, but N optims run at once → total optim time
    # ÷N. Each optim runs a SEQUENTIAL sweep there (no nested parallelism). The
    # loop below reuses those results.
    opt_precomputed = _precompute_optims(runs, base, base_params, opt_settings,
                                         dataset, name, incremental)
    baseline_res = None # baseline SCALARS (reference for every other run)
    baseline_corr_path = None  # path of the baseline's correlated.csv (never in RAM)
    for run_idx, run in enumerate(runs):
        # skip if explicitly disabled (enabled=false ou skip=true)
        if run.get("enabled", True) is False or run.get("skip") is True:
            print(f"[pipeline]   {run.get('name', f'run#{run_idx}')}: skipped "
                  f"({'enabled=false' if run.get('enabled') is False else 'skip=true'})")
            continue
        mode = run.get("mode", "corrtrack")
        run_explicit = _norm_params(run.get("params", {}))
        params = {**base_params, **run_explicit}
        cfg = Config.build(**params)
        run_name = run.get("name") or run_slug(cfg, mode)
        run_dir = os.path.join(base, run_name)

        # DECLARED cache signature (pre-optim params + optimize settings):
        # computed BEFORE the optim, reused at load AND at save → an already
        # finished optimized run is recognized as cached (see
        # _declared_signature).
        declared_sig = _declared_signature(cfg, run, opt_settings)

        rerun = run.get("rerun", not incremental)
        # declared_sig passed → load_result invalidates the cache when the
        # declared params or the optimize settings changed (otherwise a run
        # modified under the same name would be silently reused).
        cached = None if rerun else load_result(run_dir, declared_sig=declared_sig)
        if cached is not None:
            # Rebuild the cfg from the params actually used (saved into
            # result.json by save_result). Without this, the displayed cfg is the
            # pre-optim one (defaults) although the cache holds post-optim
            # results → one would see e.g. `query_radius=0.5` (default) instead
            # of the `query_radius=4` chosen by the optim.
            #
            # `_migrate_legacy_params` maps the OLD names (tree_radius,
            # cell_size, n_grids, …) → new names (query_radius, grid_cell,
            # grid_n_tables/annoy_n_trees, …) so caches from before the May 2026
            # rename stay readable WITHOUT a rerun.
            display_cfg = cfg
            cached_params = cached.get("params")
            if isinstance(cached_params, dict) and cached_params:
                try:
                    migrated = _migrate_legacy_params(cached_params)
                    display_cfg = Config.build(**migrated)
                except Exception as e:
                    print(f"[pipeline]   {run_name}: cached params unreadable ({e}), "
                          f"falling back to pre-optim cfg for display")
            print(f"[pipeline]   {run_name}: reusing cached result (rerun=false)")
            opt_t = cached.get("opt_time", 0.0)
            cached_corr_path = cached.get("correlated_path") or \
                os.path.join(run_dir, "correlated.csv")
            if run_idx == 0:
                baseline_res = cached
                baseline_corr_path = cached_corr_path
            row = _build_comparison_row(base, run_name, mode, display_cfg,
                                        baseline_res, baseline_corr_path,
                                        cached, cached_corr_path, opt_t)
            rows.append(row)
            results_meta.append((run_name, mode, display_cfg, opt_t))
            if run_idx != 0:
                del cached
                gc.collect()
            continue

        # NO VALID CACHE while the folder exists: either a previous run was
        # interrupted (crash, Ctrl-C…) and must be cleaned, or it is simply the
        # `optimize/` the parallel PRE-PASS has just written (run NEVER executed
        # yet). The stale RUN content is deleted but `optimize/` is PRESERVED
        # (otherwise the freshly computed best_params.json/stats would be lost),
        # and "incomplete run" is only reported when partial run artifacts really
        # remained — not for a folder containing only `optimize/`.
        if os.path.isdir(run_dir):
            stale = [e for e in os.listdir(run_dir) if e != "optimize"]
            if stale:
                print(f"[pipeline]   {run_name}: incomplete run detected "
                      f"(no result.json) — cleaning up (optimize/ preserved)")
                for e in stale:
                    p = os.path.join(run_dir, e)
                    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) \
                        else os.remove(p)

        opt_time = 0.0
        if run.get("optimize") or run.get("use_optimized"):  # per-run optimization
            if mode != "corrtrack":
                print(f"[pipeline] WARNING: '{run_name}' optimize ignored — only meaningful "
                      f"for corrtrack (grid params don't affect bf).")
            elif opt_settings.get("grid"):
                # optim already computed by the parallel PRE-PASS (each run has
                # its own); fall back to a direct computation if missing (safety).
                if run_idx in opt_precomputed:
                    best, opt_time = opt_precomputed[run_idx]
                else:
                    best, opt_time = _optimize_run(
                        opt_settings, base_params, run_explicit, dataset, name,
                        os.path.join(run_dir, "optimize"), tag=run_name, level=cfg.log_level)
                if best:
                    # run_explicit applied LAST → guarantees the run's params
                    # (notably `backend` when `optimize.backend` forced a
                    # different engine for the sweep) are restored for the final
                    # execution. The sweep may e.g. run in `python` and the final
                    # run in `sharded_vectorized`. `optimize_backend` is stripped
                    # (an optim field, not a valid Config field).
                    run_clean = {k: v for k, v in run_explicit.items()
                                 if k != "optimize_backend"}
                    params = {**base_params, **best, **run_clean}
                    cfg = Config.build(**params)
            else:
                print(f"[pipeline] WARNING: '{run_name}' optimize requested but no "
                      f"top-level 'optimize.grid' — using static params.")
        cfg.output = run_dir
        os.makedirs(run_dir, exist_ok=True)
        # snapshot of the run in its folder — JSON block as-is + post-optim
        # effective params (useful to reproduce or to debug)
        _save_run_config(run_dir, run, cfg, run_idx)
        reset_logger()  # fresh logger -> each run logs into its own sub-folder
        bar = "=" * 64
        log = get_logger(cfg.log_level, os.path.join(run_dir, "run.log") if run_dir else "")
        log.info(bar)
        log.info("=== RUN %d/%d: %s  (mode=%s)", run_idx + 1, len(runs), run_name, mode)
        log.info(bar)
        res = run_pipeline(cfg, dataset, mode=mode, step="monitor")
        # config=cfg (post-optim) → displayed `params` = params chosen by the optim.
        # cache_sig=declared_sig (pre-optim) → recognized as cached on the next run.
        save_result(run_dir, res, opt_time, config=cfg, cache_sig=declared_sig)

        # CRUCIAL: the `correlated` list (and the run's other large arrays) is
        # now on disk (correlated.csv). It is extracted as SCALARS and `res` is
        # freed IMMEDIATELY — including for the baseline, which used to keep its
        # 66M pairs in RAM for the whole pipeline (→ OOM when the next run
        # accumulated its own). The quality comparison is then streamed from
        # disk.
        run_corr_path = os.path.join(run_dir, "correlated.csv")
        slim = _slim_result(res)
        del res
        gc.collect()

        il = run.get("illustrate")
        if il:
            spec_il = il if isinstance(il, dict) else {}
            windows = spec_il.get("windows", [spec_il.get("window", 0)])
            steps = [s for s in ("sketch", "candidates", "validate") if spec_il.get(s)]
            if not steps:                       # bare {"windows":…} or true -> candidates
                steps = ["candidates"]
            imgs = illustrate_run(cfg, dataset, os.path.join(run_dir, "illustrate"),
                                  steps=steps, windows=windows, mode=mode)
            print(f"[pipeline]   {run_name}: illustrate {steps} -> {len(imgs)} image(s)")

        # Baseline = run 0 (forced to the front above) → its SCALARS and the
        # PATH of its correlated.csv are kept (never the list itself).
        if run_idx == 0:
            baseline_res = slim
            baseline_corr_path = run_corr_path

        row = _build_comparison_row(base, run_name, mode, cfg,
                                    baseline_res, baseline_corr_path,
                                    slim, run_corr_path, opt_time, log=log)
        rows.append(row)
        results_meta.append((run_name, mode, cfg, opt_time))

    # comparison vs the baseline run (used as reference / ground truth).
    # POLICY: `bf_python` is ALWAYS the reference for EVERY metric. It was
    # forced to position 0 (see the start of main) — so rows[0] is the baseline
    # and the rows were built on the fly inside the loop (streaming, to free each
    # run's `res` as early as possible and avoid memory build-up on 100+ runs).
    bidx = 0
    extra = ["opt_time", "opt_n", "total_time",
             "sk_speedup", "cand_speedup", "val_speedup", "monit_speedup",
             "cand_w_pct", "corr_w_pct", "missed", "recall_relation",
             # real-time metrics (windows/s throughput + resources)
             "tput_win_min", "tput_win_median", "tput_win_max", "tput_win_overall",
             "cpu_pct_median", "mem_peak_mb", "energy_cpu_seconds"]
    comparison = os.path.join(base, "comparison.csv")
    benchmark.write_rows(comparison, rows, append=False,
                         columns=benchmark.COLUMNS + extra)

    def _f(v, fmt="{:.3f}"):
        try:
            x = float(v)
            return "nan" if x != x else fmt.format(x)
        except (TypeError, ValueError):
            return str(v)

    print(f"[pipeline] {name}: {len(runs)} runs -> {base}/ "
          f"(baseline={results_meta[bidx][0]}, comparison -> {comparison})")
    w = max([len(rn) for rn, *_ in results_meta] + [len("run")])  # largeur de nom dynamique
    head = (f"{'run':<{w}}{'runtime':>10}{'speedup':>9}{'opt_t':>8}{'total':>10}"
            f"{'sk_t':>7}{'sk_x':>7}{'cand_t':>8}{'cand_x':>7}"
            f"{'val_t':>9}{'val_x':>7}{'monit_t':>9}{'monit_x':>7}"
            f"{'cand_w':>11}{'cand%':>7}"
            f"{'tested_w':>10}{'corr_w':>9}{'corr%':>7}{'missed':>8}"
            f"{'recall':>8}{'rec_rel':>8}{'prec':>7}{'spec':>7}")
    print("[pipeline] " + head)
    def _fphase(v):
        """Format speedup phase : 'NN.NNx' ou '   —  ' (NaN = pas applicable)."""
        x = _safe_float(v)
        return "    —  " if x is None else f"{x:.2f}x"

    for row in rows:
        # Cells pre-formatted+padded, THEN colored (the width stays valid).
        spd   = _ansi(f"{_f(row['speedup'],'{:.2f}')+'x':>9}",  _color_speedup(row['speedup']))
        sk_x  = _ansi(f"{_fphase(row['sk_speedup']):>7}",        _color_speedup(row['sk_speedup']))
        cd_x  = _ansi(f"{_fphase(row['cand_speedup']):>7}",      _color_speedup(row['cand_speedup']))
        vl_x  = _ansi(f"{_fphase(row['val_speedup']):>7}",       _color_speedup(row['val_speedup']))
        mn_x  = _ansi(f"{_fphase(row['monit_speedup']):>7}",     _color_speedup(row['monit_speedup']))
        rec   = _ansi(f"{_f(row['recall']):>8}",                _color_score(row['recall']))
        rrel  = _ansi(f"{_f(row['recall_relation']):>8}",       _color_score(row['recall_relation']))
        prec  = _ansi(f"{_f(row['precision']):>7}",             _color_score(row['precision']))
        spc   = _ansi(f"{_f(row['specificity']):>7}",           _color_score(row['specificity']))
        print("[pipeline] " + (
            f"{row['dataset_id']:<{w}}{_f(row['runtime'],'{:.2f}')+'s':>10}"
            f"{spd}{_f(row['opt_time'],'{:.1f}')+'s':>8}"
            f"{_f(row['total_time'],'{:.2f}')+'s':>10}"
            f"{_f(row['sk_time'],'{:.2f}'):>7}{sk_x}"
            f"{_f(row['cand_time'],'{:.2f}'):>8}{cd_x}"
            f"{_f(row['val_time'],'{:.2f}'):>9}{vl_x}"
            f"{_f(row['monit_time'],'{:.3f}'):>9}{mn_x}"
            f"{_f(row['cand_w'],'{:.0f}'):>11}{_f(row['cand_w_pct'],'{:.1f}')+'%':>7}"
            f"{_f(row['tested_w'],'{:.0f}'):>10}"
            f"{_f(row['corr_w'],'{:.0f}'):>9}{_f(row['corr_w_pct'],'{:.1f}')+'%':>7}"
            f"{_f(row['missed'],'{:.0f}'):>8}"
            f"{rec}{rrel}{prec}{spc}"))

    print(f"[pipeline] parameters used per run (optim picks shown here):")
    # Columns: index-specific, to make clear what drives each run.
    # `key=qr/nn` depending on the index; `extra` = main amplification parameter.
    phead = (f"{'run':<{w}}{'index':>10}{'n_vec':>6}{'key':>8}{'extra':>14}"
             f"{'sk_thr':>7}{'corr':>6}  backend")
    print("[pipeline] " + phead)
    for run_name, mode, cfg, opt_time in results_meta:
        bs, bc, bv = (cfg.backend_for("sketch"), cfg.backend_for("candidate"),
                      cfg.backend_for("validate"))
        if mode == "bf":
            back = bv
            key_v, extra = "-", "-"
            cells = ["-", "-", "-", "-", "-", "-"]
        else:
            back = bv if bs == bc == bv else f"sk:{bs}/cand:{bc}/val:{bv}"
            ib = cfg.index_backend
            # primary parameter of the index (radius or k)
            if ib == "grid":
                key_v = f"gc={cfg.grid_cell}"
                extra = f"gnt={cfg.grid_n_tables} gnc={cfg.grid_n_coords}"
            elif ib == "knn":
                key_v = f"nn={cfg.n_neighbors}"; extra = "-"
            elif ib == "hnsw":
                key_v = f"qr={cfg.query_radius}"
                extra = f"M={cfg.hnsw_m} efs={cfg.hnsw_ef_s}"
            elif ib == "annoy":
                key_v = f"qr={cfg.query_radius}"
                extra = f"ant={cfg.annoy_n_trees}"
            elif ib in ("bptree", "bst", "bptree3d", "bst3d"):
                key_v = f"qr={cfg.query_radius}"
                extra = f"km={cfg.key_mode}"
            elif ib == "quadtree":
                key_v = f"qr={cfg.query_radius}"
                extra = f"qpd={cfg.quadtree_proj_dim}"
            else:                                          # tree, kdtree, octree, vptree
                key_v = f"qr={cfg.query_radius}"; extra = "-"
            cells = [ib, cfg.n_vectors, key_v, extra,
                     cfg.sketch_threshold, cfg.corr_threshold]
        print("[pipeline] " + (
            f"{run_name:<{w}}{str(cells[0]):>10}{str(cells[1]):>6}"
            f"{str(cells[2]):>8}{str(cells[3]):>14}"
            f"{str(cells[4]):>7}{str(cells[5]):>6}  {back}"))

    # auto-generate PDF report + PNG figures in <pipeline>/report/
    try:
        from .utils.report import generate as _gen_report
        pdf = _gen_report(base)
        if pdf:
            print(f"[pipeline] PDF report -> {pdf}")
    except Exception as e:
        print(f"[pipeline] report generation skipped ({e})")

    # auto-write a frozen '-best' config (optim picks baked as static params) so
    # the tuned setup can be re-run instantly without sweeping again.
    if args.config and any(r.get("optimize") or r.get("use_optimized")
                           for r in spec.get("runs", [])):
        from .utils.freeze import freeze
        best_spec, frozen, _ = freeze(spec, base, "-best")
        if frozen:
            best_path = os.path.splitext(args.config)[0] + "-best.json"
            with open(best_path, "w") as f:
                json.dump(best_spec, f, indent=2)
            print(f"[pipeline] frozen best-params config -> {best_path} ({frozen} optimized runs)")

    # FINAL WARNING (red): when no `bf_python` brute-force reference is present
    # (e.g. `"baseline": false`), the speedup/recall/prec columns are RELATIVE to
    # run #0, not to a true exact bf. Reminder at the end, just in case.
    has_bf_ref = any(r.get("name") == "bf_python" or
                     (r.get("mode") == "bf" and not (r.get("params") or {}).get("backend"))
                     for r in runs)
    if not has_bf_ref:
        print(_ansi(
            f"[pipeline] ⚠️  WARNING: no 'bf_python' (brute-force) in this pipeline — "
            f"speedup / recall / precision are RELATIVE to run #0 ({results_meta[0][0]}), "
            f"not to an exact reference.", "1;31"))


if __name__ == "__main__":
    main()
