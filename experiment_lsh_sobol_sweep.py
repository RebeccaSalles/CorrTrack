"""experiment_lsh_sobol_sweep.py -- Sobol-sequence joint sweep over
(m, L, corr_prop, corr_threshold), superseding this project's own earlier
Latin Hypercube attempt (experiment_lsh_lhs_sweep.py, itself superseding
the OFAT-arm strategy in experiment_lsh_nbands_occupancy_sweep.py). That
script's (n_bands, occupancy) tuning machinery is reused UNCHANGED here;
only the sampling method and point count change.

(2026-08-27c) PLANNED, NOT YET LAUNCHED -- built after a third round of
design discussion with the user (see docs/implementation_log.md's
2026-08-27c entry for the full derivation). Two follow-up concerns drove
this over the LHS design:

1. The user asked how to know whether a fitted effect is really due to
   factor X changing vs. factor Y changing at the same time (a
   confounding concern). Answered by directly computing the empirical
   correlation matrix between the 4 LHS factors: all pairwise |corr| <
   0.05 -- genuinely low, but the mechanism (LHS's `random-cd`
   optimization searches many random pairings for a low-correlation one)
   is a LOTTERY, not a guarantee, and doesn't scale cleanly if the point
   count changes later.
2. The user then asked to increase N for tighter interaction-term
   estimates (verified directly: standard error scales ~1/sqrt(N) on
   this exact design shape; N=40->128 cuts an interaction coefficient's
   SE by ~46%), and separately asked whether LHS was really the best
   option, given the supervisor's preference for combinatorial design.

Sobol sequences (also in scipy.stats.qmc) resolve both directly:
  - Built from digital nets over GF(2) (Sobol', 1967) -- genuine
    algebraic/combinatorial machinery, not a randomized search. Verified
    directly: scrambled Sobol at a POWER-OF-2 point count reaches
    |corr|=0.0234 at N=64/128 and 0.0029 at N=256, automatically, with NO
    seed search -- better decorrelation than the LHS lottery typically
    finds, and with a real construction behind it rather than "we tried
    200 seeds and kept the best."
  - EXTENSIBLE: unlike LHS (where a 40-point and a 128-point design share
    essentially no points), a Sobol sequence's first N points are a
    subset of its first N+k points -- growing the design later continues
    the same sequence instead of discarding everything and redrawing.
    Directly relevant here, since N was already revised upward twice in
    discussion (40 -> 128 -> 256).

Design (user-approved, 2026-08-27 discussion):
  - Same 4 factors/ranges as the LHS design: m in [50,300] (log-uniform),
    L in [8,64] (log-uniform), corr_prop in [0.001,0.10] (log-uniform,
    synthetic only), corr_threshold in [0.7,0.95] (linear).
  - N_DESIGN_POINTS = 256 (a power of 2, required for Sobol's own balance
    guarantees -- see the warning scipy itself prints otherwise). The
    user explicitly chose this size to fill a weekend run.
  - Replication: REPLICATE_FRACTION ~0.15 -> 38 of the 256 points
    re-run at 2 extra seeds each (11 -> also 23, 37) -- same rationale
    and mechanism as the LHS design's replication.
  - Anchor point (2026-08-27c, carried over from the LHS design): the
    exact cell that showed the recall ceiling in the main sweep (m=200,
    L=64, corr_prop=0.05, corr_threshold=0.7) -- a space-filling design
    (Sobol or LHS) has no guarantee of revisiting any specific
    combination, so this ties the fitted surface back to that one
    already-trusted finding directly.
  - Tuning: (2026-09-08) CHANGED from the original LHS-inherited design --
    n_vectors is still FIXED at 64, but (candidate_lsh_target_occupancy,
    candidate_cosine_threshold) are now tuned via the REAL production
    hyperopt pipeline (CorrTrack_optimize.get_optim_params, proxy-anchor
    sampling) instead of calibrate_nbands_occupancy's ad hoc brute-force-
    based grid search -- see calibrate_via_proxy_hyperopt's own docstring
    in experiment_lsh_cost_sweep.py for the full rationale (the user
    explicitly asked for this: always calibrating against full brute-force
    ground truth doesn't "showcase how CorrTrack will handle actual data,"
    and they wanted these experiments to validate their own hyperopt
    pipeline directly). A documented, known limitation is embraced
    deliberately here, not hidden: proxy-anchor hyperopt can be
    statistically underpowered at low corr_prop (few true positives per
    anchor sample) -- an underpowered/degenerate result at a sparse cell
    is itself a real finding about the pipeline, recorded via
    proxy_underpowered/proxy_feasible columns, not worked around. Full
    per-combo trial grid persisted per cell exactly as before, just from
    the real hyperopt mechanism now.
  - n_steps = 2000 (full fidelity) for every cell -- same rationale as
    the LHS design (the short-vs-long question is already answered by
    the main sweep).

Companion design: experiment_lsh_orthogonal_array.py runs a SEPARATE,
genuine 16-run orthogonal array (OA(16,5,4,2), built from GF(4) mutually
orthogonal Latin squares, verified strength-2 balanced) over discretized
levels of the same 4 factors, ALONGSIDE this sweep, not instead of it --
built at the user's request so a literal combinatorial-design result is
already in hand.

NOT auto-launched by this script's own import.
"""
import argparse
import csv
import gc
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import corrtrack_param_search as cps
from corrtrack_param_search import prepare_training_data
from library_corrtrack_parallel import CorrTrack

from library_corrtrack_parallel import estimate_proxy_pairs_per_anchor

from experiment_lsh_cost_sweep import (
    load_synthetic_cell, run_smart_cell, run_bruteforce_once,
    calibrate_via_proxy_hyperopt, theoretical_sizing_fallback,
    PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR, PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING,
    TARGET_RECALL, GAMMA_TRIAL_OFFSETS,
    _make_memory_limiter, _migrate_csv_header_if_needed,
    _FlushingWriter, _current_git_commit,
)
from experiment_lsh_nbands_occupancy_sweep import (
    FIXED_N_VECTORS, N_BANDS_TOLERANCE_GRID, OCCUPANCY_GRID,
)
import experiment_lsh_cost_sweep as _main_sweep
SYNTH_CACHE_ROOT = _main_sweep.SYNTH_CACHE_ROOT

cps.OBS_MODE = "count"

RESULT_DIR = Path("tmp_artifacts/lsh_sobol_sweep")
TRIAL_GRID_DIR = RESULT_DIR / "trial_grids"
DESIGN_PATH = RESULT_DIR / "sobol_design_points.json"
COST_MODEL_PATH = Path("tmp_artifacts/lsh_cost_sweep/cost_model_analysis.json")

# ------------------------------------------------------------- design space
# (2026-09-07) Widened from (50,300)/(8,64) after the streaming memory-leak
# fixes (SignLSHBandIndex/HammingExactIndex slot reuse + Candidates window_idx
# cache reuse -- see docs/implementation_log.md's 2026-09-04 (d)/(e) entries)
# removed the actual constraint that had kept m capped at exactly the point
# where a real long-running stream used to OOM. The new upper bounds are the
# "moderate" option from a user-approved cost/time tradeoff discussion (est.
# ~4.1 days total, vs. ~1.4 days for the old range) -- brute-force ground
# truth (O(m^2*L), needed for recall/precision here) dominates this cost, not
# CorrTrack's own memory footprint, so this couldn't be pushed arbitrarily far
# without a much longer run. A separate, much larger-scale (m up to 3000, L up
# to 128) CorrTrack-ONLY timing comparison -- no brute force, no accuracy
# metrics, theoretical (not calibrated) LSH sizing -- lives in
# experiment_lsh_sobol_sweep_timing_only.py, affordable specifically because
# it skips the O(m^2*L) ground-truth cost. The old 83-point (50,300)/(8,64)
# result is preserved at tmp_artifacts/lsh_sobol_sweep_v1_m300L64/ (renamed,
# not overwritten) for direct before/after comparison.
# (2026-09-08) Reverted to the ORIGINAL (50,300)/(8,64) range -- the user has a
# hard 28-hour deadline and wants an "updated" (fixed memory + real hyperopt
# calibration) rerun directly comparable to the last (m<=300) sweep, not the
# widened moderate range. Separately-widened higher-m/L points now live only
# in the timing-only companion, which has no O(m^2*L) ground-truth cost to
# fit inside the same deadline. See the (2026-09-07) note above (still true
# in spirit -- the widened range is simply deferred, not abandoned) for why
# widening this specific range costs what it costs.
M_RANGE = (50, 300)              # log-uniform
L_RANGE = (8, 64)                # log-uniform
CORR_PROP_RANGE = (0.001, 0.10)  # log-uniform, synthetic only
CORR_THRESHOLD_RANGE = (0.7, 0.95)  # linear

N_DESIGN_POINTS = 64  # power of 2 -- required for Sobol's own balance guarantees
# (2026-08-29) Revised down from 256 after the calibration-span fix
# (CALIB_N_STEPS below) still left N=256 at ~5.6 days -- 5x cheaper than
# the pre-fix estimate but still not a weekend. N=64 costs ~1.6 days at
# the fix's own empirically-measured (not modeled) per-cell rate. See
# docs/implementation_log.md's 2026-08-29 entry.
REPLICATE_FRACTION = 0.15
N_EXTRA_SEEDS_PER_REPLICATE = 2  # -> 3 total runs (base seed + 2 extra) per replicated point
BASE_SEED = 11
REPLICATE_EXTRA_SEEDS = (23, 37)
SOBOL_RNG_SEED = 7  # fixes the scramble only -- the underlying sequence itself is deterministic

# (2026-08-27c) Carried over from the LHS design -- see module docstring.
ANCHOR_POINTS = [
    {"m": 200, "L": 64, "corr_prop": 0.05, "corr_threshold": 0.7, "seed": BASE_SEED},
]

WINDOW_SIZE = 256
WINDOW_STEP = 16
N_STEPS = 2000  # full fidelity only for the FINAL measured run -- see module docstring

# (2026-08-28) Calibration-span shrink: the 34-trial (28 structural + 6
# gamma) search inside calibrate_nbands_occupancy was found to dominate
# per-cell cost -- real observed pace (~40-45 min/cell) was ~6x this
# sweep's own original estimate, because every trial replayed the FULL
# n_steps=2000 stream. The user asked specifically whether the project's
# own proxy-anchor+bootstrap hyperopt (a real, cheap, native mechanism --
# see library_corrtrack_parallel.py's _prepare_proxy_anchor_reference)
# could be reused instead. Investigated: it samples random anchor TIME
# POSITIONS and is fast because it does, but this generator's injected
# pairs are each a NARROW, localized event (~one window_size-length
# stretch, not persistent) scattered across the whole stream -- random
# anchoring has near-zero odds of ever landing on the few positions where
# a rare injected pair is actually active, no matter how many anchors are
# drawn. That's a structural mismatch to THIS synthetic ground truth, not
# a general flaw in the mechanism.
# Fix landed on instead: shrink the CALIBRATION SPAN, not the trial
# count. The 34-trial search (and its own brute-force ground truth) runs
# on a short PREFIX of the stream; the winning (n_bands, occupancy,
# gamma) combo is then re-measured with ONE final full-length run against
# the full-length ground truth (unavoidable either way, needed for an
# honest reported recall/speedup) for the actual CSV row. Net effect:
# the dominant 34x-repeated cost shrinks roughly in proportion to the
# span ratio, while the one-off full-length costs (final brute force +
# final measured run) stay exactly as accurate as before.
# (2026-09-08) No longer used -- calibrate_via_proxy_hyperopt needs no separate
# short calibration span at all (its own anchor sampling is cheap regardless of
# how long cell_data is; it's run directly on the full span). Left as a
# constant, not deleted, in case a future revert needs it.
CALIB_N_STEPS = max(200, N_STEPS // 5)

CSV_COLUMNS = [
    "cell_id", "point_id", "is_replicate", "replicate_of", "is_anchor",
    "m", "L", "corr_threshold", "corr_prop_requested", "corr_prop_achieved",
    "n_steps", "seed", "n_lags", "window_step", "window_size",
    "n_vectors_fixed", "n_bands_selected", "target_occupancy_selected", "gamma_selected",
    "gamma_offset_selected",
    "hyperopt_used", "est_pairs_per_anchor", "effective_n_series", "series_subsampled",
    "proxy_recall_lb", "proxy_recall_ub", "proxy_feasible", "proxy_underpowered",
    "proxy_n_gt", "proxy_anchor_count", "n_grid_combos",
    "target_recall", "trial_grid_path",
    "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
    "validated_candidates", "total_candidates", "tested_candidates", "touched_candidates",
    "bf_runtime", "bf_candidate_time", "bf_validation_time", "bf_monitor_time",
    "bf_tested", "bf_correlated",
    "speedup", "precision", "recall", "f1_score", "peak_rss_gb",
    # (2026-09-08) Derived, no extra compute -- disregards validation_time AND
    # monitor_time for CorrTrack, and only monitor_time for brute force (its
    # own validation IS the exhaustive computation, not an optional
    # approximation-driven step the way CorrTrack's is). See run_one_cell.
    "corrtrack_time_no_val_no_monitor", "bf_time_no_monitor", "speedup_no_val_no_monitor",
]


# ============================================================ design generation
def _generate_sobol_points(n=N_DESIGN_POINTS, seed=SOBOL_RNG_SEED):
    """Pure sampling, no compute -- safe to call anytime. Returns the base
    N design points (before replicate expansion). Uses a SCRAMBLED Sobol
    sequence (recommended over plain Sobol for statistical validity --
    scrambling removes the low-dimensional correlation artifacts a
    non-scrambled sequence can have, while keeping its equidistribution
    guarantees) via scipy.stats.qmc.Sobol."""
    from scipy.stats import qmc

    if n & (n - 1) != 0:
        raise ValueError(f"N_DESIGN_POINTS={n} is not a power of 2 -- Sobol's own "
                          f"balance properties require it (see module docstring).")

    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    unit = sampler.random(n=n)  # (n, 4) in [0, 1)

    m_log = np.log(M_RANGE[0]) + unit[:, 0] * (np.log(M_RANGE[1]) - np.log(M_RANGE[0]))
    l_log = np.log(L_RANGE[0]) + unit[:, 1] * (np.log(L_RANGE[1]) - np.log(L_RANGE[0]))
    cp_log = np.log(CORR_PROP_RANGE[0]) + unit[:, 2] * (np.log(CORR_PROP_RANGE[1]) - np.log(CORR_PROP_RANGE[0]))
    ct = CORR_THRESHOLD_RANGE[0] + unit[:, 3] * (CORR_THRESHOLD_RANGE[1] - CORR_THRESHOLD_RANGE[0])

    m_vals = np.clip(np.round(np.exp(m_log)).astype(int), M_RANGE[0], M_RANGE[1])
    l_vals = np.clip(np.round(np.exp(l_log)).astype(int), L_RANGE[0], L_RANGE[1])
    cp_vals = np.exp(cp_log)
    ct_vals = np.round(ct, 3)

    points = []
    for i in range(n):
        m, L = int(m_vals[i]), int(l_vals[i])
        points.append({
            "point_id": i,
            "m": m, "L": L,
            "corr_prop": float(round(cp_vals[i], 6)),
            "corr_threshold": float(ct_vals[i]),
            "seed": BASE_SEED,
            "near_untested_corner": bool(m >= 260 and L >= 56),  # see module docstring
        })
    return points


def generate_design(n=N_DESIGN_POINTS, replicate_fraction=REPLICATE_FRACTION, fresh=False):
    """Writes DESIGN_PATH if missing (or if fresh=True). Pure sampling +
    JSON write -- no CorrTrack/brute-force compute. Returns the full
    expanded cell list (design points + replicate re-runs + anchors)."""
    if DESIGN_PATH.exists() and not fresh:
        return json.loads(DESIGN_PATH.read_text())["cells"]

    base_points = _generate_sobol_points(n=n)
    n_replicate = max(1, round(n * replicate_fraction))
    replicate_point_ids = [p["point_id"] for p in base_points[:n_replicate]]

    cells = []
    cell_id = 1
    for p in base_points:
        cells.append(dict(p, cell_id=cell_id, is_replicate=False, replicate_of=None, is_anchor=False))
        cell_id += 1
    for pid in replicate_point_ids:
        base = base_points[pid]
        for extra_seed in REPLICATE_EXTRA_SEEDS[:N_EXTRA_SEEDS_PER_REPLICATE]:
            cells.append(dict(base, cell_id=cell_id, seed=extra_seed,
                               is_replicate=True, replicate_of=pid, is_anchor=False))
            cell_id += 1
    for i, anchor in enumerate(ANCHOR_POINTS):
        cells.append(dict(anchor, point_id=f"anchor{i}", cell_id=cell_id,
                           is_replicate=False, replicate_of=None, is_anchor=True,
                           near_untested_corner=bool(anchor["m"] >= 260 and anchor["L"] >= 56)))
        cell_id += 1

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sobol_rng_seed": SOBOL_RNG_SEED, "n_design_points": n,
        "replicate_point_ids": replicate_point_ids,
        "ranges": {"m": M_RANGE, "L": L_RANGE, "corr_prop": CORR_PROP_RANGE,
                   "corr_threshold": CORR_THRESHOLD_RANGE},
        "n_steps": N_STEPS, "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP,
        "cells": cells,
    }
    with open(DESIGN_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return cells


def _estimate_bf_seconds(m, L, n_steps, const_median):
    pairs = m * (m - 1) / 2 if m >= 2 else 1
    return pairs * L * n_steps * const_median


def print_design_summary(cells):
    n_anchor = sum(1 for c in cells if c.get("is_anchor"))
    n_replicate = sum(1 for c in cells if c["is_replicate"])
    n_sampled = len(cells) - n_anchor - n_replicate
    print(f"{len(cells)} total cells ({n_sampled} sampled design points "
          f"+ {n_replicate} replicate re-runs + {n_anchor} fixed anchor point(s)).")

    const_median = None
    if COST_MODEL_PATH.exists():
        try:
            const_median = json.loads(COST_MODEL_PATH.read_text()).get("bruteforce_closed_form", {}).get("const_median")
        except Exception:
            const_median = None
    if const_median is None:
        const_median = 4.140268404706565e-07
        print(f"(Using the fallback bf const_median={const_median:.3e} -- "
              f"{COST_MODEL_PATH} not found or unreadable.)")

    total_bf_s = sum(_estimate_bf_seconds(c["m"], c["L"], N_STEPS, const_median) for c in cells)
    # (2026-09-08) The old 5.28x ratio was calibrated against calibrate_nbands_occupancy's own
    # expensive (full n_steps per trial) search -- no longer applicable now that Phase 1 uses
    # calibrate_via_proxy_hyperopt instead (proxy-anchor sampling, real measured cost ~11s at
    # 438,675 pairs/anchor, ~25us/pair, capped at PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR's
    # ~50s worst case per cell -- see that constant's own comment). Using a flat, conservative
    # per-cell estimate instead of a ratio, since hyperopt cost no longer scales with bf cost.
    MEASURED_HYPEROPT_SECONDS_PER_CELL = 40.0
    total_tuning_s = len(cells) * MEASURED_HYPEROPT_SECONDS_PER_CELL
    print(f"Estimated brute-force-only time: {total_bf_s/3600:.1f} h")
    print(f"Rough total estimate incl. proxy-hyperopt tuning "
          f"({MEASURED_HYPEROPT_SECONDS_PER_CELL:.0f}s/cell flat estimate -- still not a "
          f"guarantee): {(total_bf_s + total_tuning_s)/3600:.1f} h "
          f"({(total_bf_s+total_tuning_s)/86400:.2f} days)")
    print("This does not include data-generation time or subprocess/OS overhead per cell.")

    X = np.array([[math.log(c["m"]), math.log(c["L"]), math.log(c["corr_prop"]), c["corr_threshold"]]
                  for c in cells if not c["is_replicate"] and not c.get("is_anchor")])
    R = np.corrcoef(X, rowvar=False)
    worst = np.max(np.abs(R[np.triu_indices(4, k=1)]))
    print(f"Worst pairwise factor correlation among the {len(X)} sampled points: {worst:.4f}")


# ============================================================ per-cell execution
def run_one_cell(writer, cell, train_data, ids, corr_prop_achieved=None):
    m, L, corr_threshold = cell["m"], cell["L"], cell["corr_threshold"]
    seed = cell["seed"]
    n_lags = (L - 1) * WINDOW_STEP
    span_needed = WINDOW_SIZE + N_STEPS * WINDOW_STEP
    cell_data = train_data[:, :span_needed]

    row = {
        "cell_id": cell["cell_id"], "point_id": cell["point_id"],
        "is_replicate": cell["is_replicate"], "replicate_of": cell["replicate_of"],
        "is_anchor": cell.get("is_anchor", False),
        "m": m, "L": L, "corr_threshold": corr_threshold,
        "corr_prop_requested": cell["corr_prop"], "corr_prop_achieved": corr_prop_achieved,
        "n_steps": N_STEPS, "seed": seed, "n_lags": n_lags,
        "window_step": WINDOW_STEP, "window_size": WINDOW_SIZE,
        "n_vectors_fixed": FIXED_N_VECTORS, "n_bands_selected": None, "target_occupancy_selected": None,
        "gamma_selected": None, "gamma_offset_selected": None,
        "hyperopt_used": None, "est_pairs_per_anchor": None,
        "effective_n_series": None, "series_subsampled": None,
        "proxy_recall_lb": None, "proxy_recall_ub": None, "proxy_feasible": None,
        "proxy_underpowered": None, "proxy_n_gt": None, "proxy_anchor_count": None,
        "n_grid_combos": None, "target_recall": TARGET_RECALL,
        "trial_grid_path": None,
        "sketch_time": None, "candidate_time": None, "validation_time": None, "monitor_time": None,
        "smart_wall_time": None, "validated_candidates": None, "total_candidates": None,
        "tested_candidates": None, "touched_candidates": None,
        "bf_runtime": None, "bf_candidate_time": None, "bf_validation_time": None, "bf_monitor_time": None,
        "bf_tested": None, "bf_correlated": None,
        "speedup": None, "precision": None, "recall": None, "f1_score": None,
        "corrtrack_time_no_val_no_monitor": None, "bf_time_no_monitor": None,
        "speedup_no_val_no_monitor": None,
    }

    bf_output_dir = RESULT_DIR / "bf_runs"
    bf_output_dir.mkdir(parents=True, exist_ok=True)
    TRIAL_GRID_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: calibrate (candidate_lsh_target_occupancy,
    # candidate_cosine_threshold) via the REAL production hyperopt pipeline --
    # see calibrate_via_proxy_hyperopt's own docstring (experiment_lsh_cost_
    # sweep.py) for the full rationale and the deliberately-embraced known
    # limitation at low corr_prop. No separate calibration-span brute force
    # needed -- the proxy-anchor mechanism supplies its own local ground
    # truth, cheaply, regardless of how long cell_data is.
    trial_grid_prefix = str(TRIAL_GRID_DIR / f"cell{cell['cell_id']}")
    est_pairs = estimate_proxy_pairs_per_anchor(m, n_lags, WINDOW_STEP)
    row["est_pairs_per_anchor"] = int(est_pairs)
    # (2026-09-08) No longer gated on est_pairs here -- CorrTrack_optimize now subsamples
    # series internally whenever the full population would be impractical (see
    # calibrate_via_proxy_hyperopt's own docstring and docs/implementation_log.md's 2026-09-08
    # entries), so real hyperopt is attempted at every scale instead of skipping straight to
    # the fallback. PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR is passed through as the
    # per-anchor subsample budget, not a skip threshold. pair_row_hard_ceiling is a SEPARATE,
    # larger TOTAL ceiling (PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING) -- reusing the per-anchor
    # budget for both left room for exactly 1 anchor by construction, silently starving
    # statistical power even after subsampling kicked in (see the 2026-09-08 "still later"
    # comment on PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR's own definition).
    hyperopt_result = calibrate_via_proxy_hyperopt(
        cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold,
        OCCUPANCY_GRID, m, seed, trial_grid_prefix, f"sobol_cell{cell['cell_id']}",
        target_recall=TARGET_RECALL, n_vectors=FIXED_N_VECTORS,
        pair_row_hard_ceiling=PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING,
    )
    row["trial_grid_path"] = trial_grid_prefix + "_corrtrack.csv"

    if hyperopt_result is None:
        # Either infeasible (est_pairs too high) or get_optim_params found nothing selectable
        # -- both real, disclosed outcomes. Falls back to the closed-form sizing formula rather
        # than leaving this cell unmeasured; hyperopt_used=False makes the distinction explicit
        # in the CSV, not silently blended with real hyperopt results.
        hyperopt_result = theoretical_sizing_fallback(
            m, L, corr_threshold, n_vectors=FIXED_N_VECTORS, target_occupancy=3.0,
            target_recall=TARGET_RECALL,
        )
        row["hyperopt_used"] = False
    else:
        row["hyperopt_used"] = True

    row["n_bands_selected"] = hyperopt_result["n_bands"]
    row["target_occupancy_selected"] = hyperopt_result["target_occupancy"]
    row["gamma_selected"] = hyperopt_result["gamma"]
    row["gamma_offset_selected"] = hyperopt_result.get("gamma_offset")
    row["proxy_recall_lb"] = hyperopt_result["proxy_recall_lb"]
    row["proxy_recall_ub"] = hyperopt_result["proxy_recall_ub"]
    row["proxy_feasible"] = hyperopt_result["proxy_feasible"]
    row["proxy_underpowered"] = hyperopt_result["proxy_underpowered"]
    row["proxy_n_gt"] = hyperopt_result["proxy_n_gt"]
    row["proxy_anchor_count"] = hyperopt_result["proxy_anchor_count"]
    row["n_grid_combos"] = hyperopt_result["n_grid_combos"]
    row["effective_n_series"] = hyperopt_result.get("effective_n_series")
    row["series_subsampled"] = hyperopt_result.get("series_subsampled")
    gc.collect()

    # ---- Phase 2: FULL-length brute-force evaluation (unavoidable cost, paid
    # once) -- decoupled from calibration now: purely for reporting real
    # precision/recall/speedup against ground truth, not consulted by the
    # hyperopt phase itself (in real deployment this ground truth wouldn't
    # exist either -- see calibrate_via_proxy_hyperopt's docstring).
    bf_output_csv = str(bf_output_dir / f"bf_cell{cell['cell_id']}_m{m}_L{L}_ct{corr_threshold}_s{seed}.csv")
    bf_record, bf_corr_flags = run_bruteforce_once(
        None, cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold, bf_output_csv,
    )
    row["bf_runtime"] = bf_record.get("runtime")
    row["bf_candidate_time"] = bf_record.get("cand_time")
    row["bf_validation_time"] = bf_record.get("val_time")
    row["bf_monitor_time"] = bf_record.get("monit_time")
    row["bf_tested"] = bf_record.get("tested")
    row["bf_correlated"] = bf_record.get("correlated")

    final_ct, final_wall_time, final_n_steps_run = run_smart_cell(
        cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold,
        row["target_occupancy_selected"], N_STEPS, seed,
        candidate_cosine_threshold=row["gamma_selected"],
        n_vectors=FIXED_N_VECTORS, candidate_lsh_n_bands=row["n_bands_selected"],
    )
    final_metrics = CorrTrack.compute_metrics_bf(
        final_ct, bf_corr_flags, windows=True, total_pairs_bf=bf_record.get("tested"),
    )
    row["precision"] = final_metrics.get("precision")
    row["recall"] = final_metrics.get("recall")
    row["f1_score"] = final_metrics.get("f1_score")
    row["n_steps"] = final_n_steps_run
    row["sketch_time"] = final_ct.sketch_time
    row["candidate_time"] = final_ct.candidate_time
    row["validation_time"] = final_ct.validation_time
    row["monitor_time"] = final_ct.monitor_time
    row["smart_wall_time"] = final_wall_time
    row["validated_candidates"] = final_ct.validated_candidates
    row["total_candidates"] = final_ct.total_candidates
    row["tested_candidates"] = final_ct.tested_candidates
    row["touched_candidates"] = getattr(final_ct, "candidate_search_lsh_candidates_touched", None)
    if final_wall_time > 0 and bf_record.get("runtime"):
        row["speedup"] = bf_record["runtime"] / final_wall_time

    # (2026-09-08) Derived "disregard validation (CorrTrack only) and
    # monitoring (both sides)" comparison -- no extra runs, just a different
    # sum of the SAME measured components. See CSV_COLUMNS' own comment.
    ct_no_val_no_mon = (row["sketch_time"] or 0.0) + (row["candidate_time"] or 0.0)
    bf_no_mon = (row["bf_candidate_time"] or 0.0) + (row["bf_validation_time"] or 0.0)
    row["corrtrack_time_no_val_no_monitor"] = ct_no_val_no_mon
    row["bf_time_no_monitor"] = bf_no_mon
    if ct_no_val_no_mon > 0:
        row["speedup_no_val_no_monitor"] = bf_no_mon / ct_no_val_no_mon

    del bf_record, bf_corr_flags, final_ct
    gc.collect()
    try:
        import resource as _resource
        row["peak_rss_gb"] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        row["peak_rss_gb"] = None
    writer.writerow(row)
    return row


def _cell_resume_key(cell):
    return ("sobol", str(cell["cell_id"]))


def _load_completed_cell_counts(output_csv):
    """(2026-08-27e) The main sweep's own _load_completed_cell_counts
    (experiment_lsh_cost_sweep.py) is hardcoded to THAT script's resume-key
    schema (dataset_kind, m, L, corr_prop_requested, target_occupancy) --
    reading it here silently built keys that could never match this
    script's own _cell_resume_key (("sobol", cell_id)), making resume a
    complete no-op: a restart after any interruption would have re-run
    every cell from scratch, exactly the failure mode resume exists to
    prevent. Caught by direct observation on the very first launch attempt
    (the sibling OA script reported "found 1 already-attempted cell" via
    the wrong-schema function, then re-ran that cell anyway) -- both
    launches were killed and restarted with this fix before any real
    compute time was lost. Keyed on cell_id directly, matching
    _cell_resume_key exactly."""
    path = Path(output_csv)
    if not path.exists():
        return {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = ("sobol", row.get("cell_id"))
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        return {}
    return counts


def _run_worker_subprocess(cell, output_csv, memory_limit_gb=None):
    import subprocess
    import sys as _sys
    spec = dict(cell, output_csv=output_csv)
    result = subprocess.run(
        [_sys.executable, os.path.abspath(__file__), "--worker-cell-spec", json.dumps(spec)],
        check=False, preexec_fn=_make_memory_limiter(memory_limit_gb),
    )
    if result.returncode != 0:
        suspected_oom = result.returncode < 0 or result.returncode == 1
        print(f"[cell {cell['cell_id']}] m={cell['m']} L={cell['L']} -> WORKER FAILED (exit {result.returncode}"
              f"{', possibly hit the memory limit or an OOM kill' if suspected_oom and memory_limit_gb else ''}"
              f"), skipping this cell")


def run_worker_cell(spec):
    with open(spec["output_csv"], "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer = _FlushingWriter(writer, fh)
        m, corr_prop = spec["m"], spec["corr_prop"]
        n_synth = WINDOW_SIZE + N_STEPS * WINDOW_STEP
        synth_max_lag = (spec["L"] - 1) * WINDOW_STEP
        data, ids, _z_used, _achieved_z, achieved_cp = load_synthetic_cell(
            m, n_synth, corr_prop, WINDOW_SIZE, spec["seed"], synth_max_lag,
            threshold=spec["corr_threshold"], window_step=WINDOW_STEP,
        )
        train_data, ids_n = prepare_training_data(data, ids, n_year=n_synth, n_var=m, train_ratio=1.0)
        row = run_one_cell(writer, spec, train_data, list(ids_n), corr_prop_achieved=achieved_cp)
        print(f"[cell {spec['cell_id']}] m={m} L={spec['L']} corr_prop={corr_prop:.5f} "
              f"ct={spec['corr_threshold']:.3f} seed={spec['seed']} -> n_bands={row['n_bands_selected']} "
              f"occ={row['target_occupancy_selected']} speedup={row['speedup']} recall={row['recall']}")


def write_run_manifest(args, cells):
    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Sobol-sequence joint sweep over (m, L, corr_prop, corr_threshold) -- "
                   "supersedes this project's own earlier LHS design (experiment_lsh_lhs_sweep.py) "
                   "after the user asked for tighter interaction estimates (more points) and a more "
                   "defensible construction than a randomized seed search. See module docstring and "
                   "docs/implementation_log.md's 2026-08-27c entry.",
        "output_csv": args.output_csv,
        "git_commit": _current_git_commit(),
        "cli_args": vars(args),
        "design_path": str(DESIGN_PATH),
        "n_cells": len(cells),
        "n_unique_design_points": sum(1 for c in cells if not c["is_replicate"]),
        "n_replicate_reruns": sum(1 for c in cells if c["is_replicate"]),
        "fixed": {
            "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP, "n_steps": N_STEPS,
            "n_vectors": FIXED_N_VECTORS, "data_representation": "sketch_proj",
            "candidate_backend": "lsh_approx", "validation_metric": "pearson", "neg_corr": True,
            "candidate_apply_hamming_filter": False, "candidate_apply_dot_gamma_filter": True,
        },
        "ranges": {"m": M_RANGE, "L": L_RANGE, "corr_prop": CORR_PROP_RANGE,
                   "corr_threshold": CORR_THRESHOLD_RANGE},
        "tuning_grids": {"n_bands_tolerance_grid": list(N_BANDS_TOLERANCE_GRID), "occupancy_grid": list(OCCUPANCY_GRID),
                          "gamma_trial_offsets": list(GAMMA_TRIAL_OFFSETS)},
        "companion_design": "experiment_lsh_orthogonal_array.py -- a separate, genuine 16-run "
                             "orthogonal array over discretized levels of the same 4 factors.",
    }
    manifest_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Sobol-sequence joint sweep over (m, L, corr_prop, corr_threshold).")
    parser.add_argument("--generate-design", action="store_true",
                         help="Only generate/print the design (no compute) and exit.")
    parser.add_argument("--fresh-design", action="store_true",
                         help="Regenerate the design file even if it already exists (invalidates any prior resume state).")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing output CSV, start clean.")
    parser.add_argument("--only-cells", type=int, nargs="+", default=None,
                         help="Run only these cell_id(s) -- e.g. a single cheap cell as a smoke test "
                              "before committing to the full design.")
    parser.add_argument("--worker-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "sobol_sweep.csv"))
    parser.add_argument("--worker-cell-spec", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_cell_spec is not None:
        run_worker_cell(json.loads(args.worker_cell_spec))
        return

    cells = generate_design(fresh=args.fresh_design)

    if args.generate_design:
        print_design_summary(cells)
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded design: {len(cells)} cells from {DESIGN_PATH}. NOT auto-launched by import -- "
          f"only runs when explicitly invoked without --generate-design.")

    completed_counts = {} if args.fresh else _load_completed_cell_counts(args.output_csv)
    if not args.fresh and completed_counts:
        print(f"Resuming: found {len(completed_counts)} already-attempted cell(s) in {args.output_csv}.")

    if args.fresh or not Path(args.output_csv).exists():
        with open(args.output_csv, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
    else:
        _migrate_csv_header_if_needed(args.output_csv, CSV_COLUMNS)

    write_run_manifest(args, cells)

    pending = [c for c in cells if completed_counts.get(_cell_resume_key(c), 0) < 1]
    if len(pending) < len(cells):
        print(f"Skipping {len(cells) - len(pending)} cell(s) already completed.")
    if args.only_cells is not None:
        wanted = set(args.only_cells)
        pending = [c for c in pending if c["cell_id"] in wanted]
        print(f"--only-cells given: restricting this run to {len(pending)} cell(s): {sorted(wanted)}.")

    for cell in pending:
        _run_worker_subprocess(cell, args.output_csv, memory_limit_gb=args.worker_memory_limit_gb)

    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
