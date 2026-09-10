"""Hyperparameter search: selecting the best config out of a sweep.

Faithful port of the v1 selection (`CorrTrack_optimize.get_optim_params`). The
sweep itself is `benchmark.sweep` (each config measured against brute-force on
the `train_ratio` prefix of the data). The selection then goes:

  0. keep only the `status == success` runs;
  1. feasible: `speedup > 1` AND `grid_vote_min < 1` (otherwise: all of them);
  2. recall: `recall >= target_recall`; otherwise fall back to
     `recall >= recall_fallback_near_ratio × best_recall` (otherwise: the max);
  3. near-best speedup: `speedup >= speedup_near_ratio × best_speedup`;
  4. tie-break: increasing `cand_w` (fewer candidates), then decreasing
     `speedup`.
"""

import math

# parameter columns reported in best_params (those defining a config)
PARAM_KEYS = ["n_vectors", "grid_cell", "grid_n_coords", "grid_vote_min",
              "query_radius", "n_neighbors", "key_mode", "key_proj_dim",
              "window_size", "window_step", "n_lags", "seed", "corr_threshold",
              "candidate_backend"]


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return float("nan")


def _clamp(x, lo=0.0, hi=1.0, default=0.98):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return max(lo, min(hi, x))


def select_best(rows, target_recall=0.95, recall_fallback_near_ratio=0.98,
                speedup_near_ratio=0.98, target_precision=0.0):
    """Return the record (dict) of the best config, or None when none succeeded."""
    configs = [r for r in rows if r.get("status") == "success"]
    if not configs:
        return None

    # 1. feasible
    feasible = [r for r in configs
                if _f(r, "speedup") > 1 and _f(r, "grid_vote_min") < 1] or configs

    # 1b. precision constraint (hard; 0.0 = disabled)
    if target_precision > 0:
        feasible = [r for r in feasible
                    if _f(r, "precision") >= target_precision] or feasible

    # 2. recall (with fallback)
    qualified = [r for r in feasible if _f(r, "recall") >= target_recall]
    if not qualified:
        recalls = [_f(r, "recall") for r in feasible if _f(r, "recall") == _f(r, "recall")]
        max_recall = max(recalls) if recalls else 0.0
        floor = max(max_recall * _clamp(recall_fallback_near_ratio), 0.0)
        qualified = ([r for r in feasible if _f(r, "recall") >= floor]
                     or [r for r in feasible if _f(r, "recall") == max_recall])

    # 3. near-best speedup band
    best_speedup = max(_f(r, "speedup") for r in qualified)
    near = ([r for r in qualified
             if _f(r, "speedup") >= best_speedup * _clamp(speedup_near_ratio)]
            or [r for r in qualified if _f(r, "speedup") == best_speedup])

    # 4. tie-break: fewer candidates, then more speedup (NaN cand_w last)
    def key(r):
        cw = _f(r, "cand_w")
        return (math.inf if cw != cw else cw, -_f(r, "speedup"))

    return sorted(near, key=key)[0]


def best_params(record):
    """Extract the parameters defining the winning config (for best_params.json)."""
    return {k: record[k] for k in PARAM_KEYS if k in record}
