"""(2026-08-27c) SUPERSEDED by experiment_lsh_sobol_sweep.py -- after the
user asked for tighter interaction-term estimates (more points) and a
more defensible, non-lottery-based construction, LHS was replaced by a
scrambled Sobol sequence (see that file's own module docstring for the
full reasoning: verified sqrt(N) power scaling, verified Sobol
decorrelation beats an LHS seed-search at power-of-2 N, and Sobol's
extensibility property, all directly relevant once N started being
revised upward). This file's own smoke-test bug fixes (threshold and
window_step forwarding, both real, both fixed in experiment_lsh_cost_
sweep.py's shared load_synthetic_cell) are NOT lost -- the Sobol script
reuses the same shared, now-fixed function. Left in place, unlaunched, as
a historical record of the debugging that found those two bugs and as a
simpler fallback if Sobol/scipy's qmc module is ever unavailable.

experiment_lsh_lhs_sweep.py -- Latin Hypercube joint sweep over
(m, L, corr_prop, corr_threshold), superseding the OFAT-arm strategy in
experiment_lsh_nbands_occupancy_sweep.py for these four factors (that
script's (n_bands, occupancy) tuning machinery is reused UNCHANGED here;
only the top-level workload design changes).

(2026-08-26c) PLANNED, NOT YET LAUNCHED -- built after two rounds of
design discussion with the user (see docs/implementation_log.md's
2026-08-26c entry for the full derivation). Two design decisions were
explicitly rejected before landing on this one:

1. A 2-level screen-then-prune plan (run a cheap 2^4 corners-only
   factorial at short n_steps + a coarse tuning grid, then only extend
   the factors/interactions it flagged to full resolution) was proposed
   and REJECTED by the user, for good reason: this project's own
   accuracy_summary already documents that short n_steps runs show a
   real statistical-power effect (recall artificially low/zero, not an
   algorithm failure) -- screening under that regime and pruning based on
   it risks throwing away real effects that only appear at full n_steps.
   Similarly, a coarse tuning grid could have silently missed the
   m=200/L=64 recall-ceiling finding (2026-08-26 (d) entry), which only
   showed up because the FULL fine grid was exhausted. Any screen run
   under cheaper conditions than the real experiment can only be trusted
   if the cheap and expensive regimes are homologous -- there was no
   reason to believe that here, so the two-phase design was dropped
   entirely, not just tuned.
2. A plain multi-level GRID (e.g. 4-6 discrete levels per factor, crossed
   fully or via a small OFAT core) was also rejected as not "fine"
   enough -- the user wants real curve resolution (many distinct values
   per axis), and a grid wastes runs by repeating the same handful of
   levels across every combination.

Both concerns are resolved by using a LATIN HYPERCUBE instead of either a
corners-only screen or a shared-level grid: every one of the N design
points gets a DISTINCT value on every axis (no repeated levels to waste
runs on), and EVERY point runs at the exact same full fidelity as the
final experiment (n_steps=2000, the full (n_bands x occupancy) tuning
grid) -- there is no cheap proxy regime, so there is no generalization
gap to worry about. Interactions and curvature are recovered afterward
via a proper multi-factor regression across all points, not guessed at
via pre-chosen corners.

Design (user-approved, 2026-08-26 discussion):
  - Factors: m in [50, 300] (log-uniform), L in [8, 64] (log-uniform),
    corr_prop in [0.001, 0.10] (log-uniform, synthetic only), corr_threshold
    in [0.7, 0.95] (linear). Range explicitly bounded to the box already
    exercised without a crash in this sweep family (m=300 confirmed only
    at L=32; L=64 confirmed only at m=200 -- the single corner near
    m=300,L=64 simultaneously is therefore genuinely UNTESTED territory
    within the approved box, not just an extrapolation; flagged per-point
    in the generated design, not excluded, and protected by the same
    per-cell subprocess isolation + memory cap + resume infra used
    throughout this sweep family so a failure there doesn't take down any
    other point).
  - N_DESIGN_POINTS = 40 (the "~35-45 points, ~1-2 days" budget the user
    chose over a ~20-25-point half-day option and a ~60+-point multi-day
    option).
  - Replication: REPLICATE_FRACTION ~0.15 -> 6 of the 40 points (the
    first 6 in generation order, a deterministic and arbitrary-as-any
    choice given the design is already randomized) are each re-run at 2
    EXTRA seeds (11 -> also 23, 37) for a real noise/error estimate, per
    the user's explicit ask last turn -- 40 + 6*2 = 52 total cells.
  - Anchor point(s) (2026-08-27c): ANCHOR_POINTS adds the exact cell that
    showed the recall ceiling in the main sweep (m=200, L=64, corr_prop=
    0.05, corr_threshold=0.7) on top of the 52 above -- an LHS design has
    no guarantee of revisiting any specific combination (raised by the
    user as a real limitation of the approach), so this ties the fitted
    surface back to that one already-trusted finding directly. 53 total
    cells as of this addition.
  - Tuning: n_vectors FIXED at 64 (matches experiment_lsh_nbands_
    occupancy_sweep.py's own finding that the main sweep's tuner never
    chose otherwise), (n_bands, occupancy) searched via THAT script's
    own calibrate_nbands_occupancy, reused unchanged (full grid: 7
    n_bands x 4 occupancy = 28 stage-1 trials + up to 6 gamma trials).
    Full stage-1 trial grid persisted per cell (not just the winner),
    same rationale/format as that script's own 2026-08-26b addition.
  - n_steps = 2000 (long/full-fidelity) for every cell -- no short-run
    duplicate this time. The short-vs-long fixed-cost/phase-share
    question this sweep family cares about was already characterized by
    the main sweep (experiment_lsh_cost_sweep.py); duplicating it across
    52 LHS points would double the cost for a question already answered.

Two-step workflow, deliberately separated so the concrete design can be
inspected and approved BEFORE any compute runs:
  1. `--generate-design` writes the 52-row point list to
     RESULT_DIR/lhs_design_points.json (pure sampling, no CorrTrack/
     brute-force compute -- cheap, safe to run anytime) and prints a
     summary table plus an analytic cost estimate (using the brute-force
     closed-form constant already fitted in
     tmp_artifacts/lsh_cost_sweep/cost_model_analysis.json) so the total
     wall-clock commitment is known before committing to it.
  2. `main()` (the real per-cell run) READS that persisted file rather
     than regenerating the design live -- removes any fragility from
     scipy version drift or code changes silently reshuffling which
     cell_id maps to which (m, L, corr_prop, corr_threshold, seed) mid-run.
     Refuses to run if the design file is missing (prints the exact
     `--generate-design` command needed).

NOT auto-launched by this script's own import, and per the user's
explicit instruction this entire script (including `--generate-design`)
has only been run in its cheap, compute-free design-generation mode as
part of building it -- the actual 52-cell sweep has NOT been started.
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

# Reuse tested building blocks directly -- see experiment_lsh_nbands_
# occupancy_sweep.py's own import comment for why _run_worker_subprocess
# is NOT imported (it hardcodes os.path.abspath(__file__), which binds to
# whichever module defines it).
from experiment_lsh_cost_sweep import (
    load_synthetic_cell, run_smart_cell, run_bruteforce_once,
    calibrate_gamma, TARGET_RECALL, GAMMA_TRIAL_OFFSETS,
    _make_memory_limiter, _migrate_csv_header_if_needed,
    _FlushingWriter, _current_git_commit,
)
from experiment_lsh_nbands_occupancy_sweep import (
    calibrate_nbands_occupancy, FIXED_N_VECTORS, N_BANDS_TOLERANCE_GRID, OCCUPANCY_GRID,
)
import experiment_lsh_cost_sweep as _main_sweep
SYNTH_CACHE_ROOT = _main_sweep.SYNTH_CACHE_ROOT

cps.OBS_MODE = "count"

RESULT_DIR = Path("tmp_artifacts/lsh_lhs_sweep")
TRIAL_GRID_DIR = RESULT_DIR / "trial_grids"
DESIGN_PATH = RESULT_DIR / "lhs_design_points.json"
COST_MODEL_PATH = Path("tmp_artifacts/lsh_cost_sweep/cost_model_analysis.json")

# ------------------------------------------------------------- design space
M_RANGE = (50, 300)             # log-uniform
L_RANGE = (8, 64)               # log-uniform
CORR_PROP_RANGE = (0.001, 0.10)  # log-uniform, synthetic only
CORR_THRESHOLD_RANGE = (0.7, 0.95)  # linear

N_DESIGN_POINTS = 40
REPLICATE_FRACTION = 0.15
N_EXTRA_SEEDS_PER_REPLICATE = 2  # -> 3 total runs (base seed + 2 extra) per replicated point
BASE_SEED = 11
REPLICATE_EXTRA_SEEDS = (23, 37)
LHS_RNG_SEED = 20260826  # fixed, for a reproducible design -- see DESIGN_PATH note above

# (2026-08-27c) User-approved anchor point: the exact cell that showed the
# recall ceiling in the main sweep (2026-08-26 (d) entry) -- an LHS design
# has no guarantee of revisiting any specific combination (that's the
# honest trade-off discussed with the user), so this is added on top of
# the 40 sampled points to directly tie this sweep's fitted surface back
# to that one already-trusted finding.
ANCHOR_POINTS = [
    {"m": 200, "L": 64, "corr_prop": 0.05, "corr_threshold": 0.7, "seed": BASE_SEED},
]

WINDOW_SIZE = 256
WINDOW_STEP = 16
N_STEPS = 2000  # full fidelity only -- see module docstring

CSV_COLUMNS = [
    "cell_id", "point_id", "is_replicate", "replicate_of", "is_anchor",
    "m", "L", "corr_threshold", "corr_prop_requested", "corr_prop_achieved",
    "n_steps", "seed", "n_lags", "window_step", "window_size",
    "n_vectors_fixed", "n_bands_selected", "target_occupancy_selected", "gamma_selected",
    "structural_met_target_recall", "n_structural_trials", "gamma_met_target_recall", "gamma_trials_tried",
    "target_recall", "trial_grid_path",
    "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
    "validated_candidates", "total_candidates", "tested_candidates",
    "bf_runtime", "bf_tested", "bf_correlated",
    "speedup", "precision", "recall", "f1_score", "peak_rss_gb",
]


# ============================================================ design generation
def _generate_lhs_points(n=N_DESIGN_POINTS, seed=LHS_RNG_SEED):
    """Pure sampling, no compute -- safe to call anytime. Returns the base
    N design points (before replicate expansion), each with a distinct
    value on every axis (m, L, corr_prop, corr_threshold)."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=4, seed=seed, optimization="random-cd")
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
            # (2026-08-26c) m and L rounded independently to integers can
            # collide across two distinct continuous LHS draws (a known,
            # accepted minor imperfection of integer-constrained LHS, not
            # fixed here) -- corr_prop/corr_threshold still differ per
            # point in that case, so no two rows are ever fully identical.
            "near_untested_corner": bool(m >= 260 and L >= 56),  # see module docstring
        })
    return points


def generate_design(n=N_DESIGN_POINTS, replicate_fraction=REPLICATE_FRACTION, fresh=False):
    """Writes DESIGN_PATH if missing (or if fresh=True). Pure sampling +
    JSON write -- no CorrTrack/brute-force compute. Returns the full
    expanded cell list (design points + replicate re-runs)."""
    if DESIGN_PATH.exists() and not fresh:
        return json.loads(DESIGN_PATH.read_text())["cells"]

    base_points = _generate_lhs_points(n=n)
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
        "lhs_rng_seed": LHS_RNG_SEED, "n_design_points": n,
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
    print(f"{'cell':>4} {'m':>4} {'L':>3} {'corr_prop':>10} {'corr_thr':>9} {'seed':>5} {'flag':>10}")
    for c in cells:
        if c.get("is_anchor"):
            flag = "ANCHOR"
        elif c["is_replicate"]:
            flag = "REPLICATE"
        elif c.get("near_untested_corner"):
            flag = "EDGE"
        else:
            flag = ""
        print(f"{c['cell_id']:>4} {c['m']:>4} {c['L']:>3} {c['corr_prop']:>10.5f} "
              f"{c['corr_threshold']:>9.3f} {c['seed']:>5} {flag:>10}")

    const_median = None
    if COST_MODEL_PATH.exists():
        try:
            const_median = json.loads(COST_MODEL_PATH.read_text()).get("bf_const_median") \
                or json.loads(COST_MODEL_PATH.read_text()).get("bruteforce_closed_form", {}).get("const_median")
        except Exception:
            const_median = None
    if const_median is None:
        # fall back to the value already reported in report_data.json (2026-08-26)
        const_median = 4.140268404706565e-07
        print(f"\n(Using the fallback bf const_median={const_median:.3e} -- "
              f"{COST_MODEL_PATH} not found or unreadable.)")

    total_bf_s = sum(_estimate_bf_seconds(c["m"], c["L"], N_STEPS, const_median) for c in cells)
    # Tuning-search cost isn't in closed form -- rough-scaled off the same
    # brute-force estimate using this sweep family's own observed ratio
    # (a tuned smart run's per-step cost is well under brute force's; ~34
    # trials at a MIX of n_bands means roughly 3-6x one smart run's cost).
    # This is an order-of-magnitude guide for planning, not a guarantee.
    total_tuning_s = total_bf_s * 0.5  # conservative: tuning search often cheaper in aggregate than one bf pass, given LSH's own speedup
    print(f"\nEstimated brute-force-only time: {total_bf_s/3600:.1f} h")
    print(f"Rough total estimate incl. tuning search (order-of-magnitude, NOT a guarantee): "
          f"{(total_bf_s + total_tuning_s)/3600:.1f} h")
    print("This does not include data-generation time or subprocess/OS overhead per cell.")


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
        "gamma_selected": None,
        "structural_met_target_recall": None, "n_structural_trials": 0,
        "gamma_met_target_recall": None, "gamma_trials_tried": 0, "target_recall": TARGET_RECALL,
        "trial_grid_path": None,
        "sketch_time": None, "candidate_time": None, "validation_time": None, "monitor_time": None,
        "smart_wall_time": None, "validated_candidates": None, "total_candidates": None, "tested_candidates": None,
        "bf_runtime": None, "bf_tested": None, "bf_correlated": None,
        "speedup": None, "precision": None, "recall": None, "f1_score": None,
    }

    bf_output_dir = RESULT_DIR / "bf_runs"
    bf_output_dir.mkdir(parents=True, exist_ok=True)
    bf_output_csv = str(bf_output_dir / f"bf_cell{cell['cell_id']}_m{m}_L{L}_ct{corr_threshold}_s{seed}.csv")
    bf_record, bf_corr_flags = run_bruteforce_once(
        None, cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold, bf_output_csv,
    )
    row["bf_runtime"] = bf_record.get("runtime")
    row["bf_tested"] = bf_record.get("tested")
    row["bf_correlated"] = bf_record.get("correlated")

    result = calibrate_nbands_occupancy(
        cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold,
        N_STEPS, seed, bf_corr_flags, bf_record.get("tested"),
    )
    best = result["gamma_trial"]
    row["n_bands_selected"] = result["n_bands"]
    row["target_occupancy_selected"] = result["occupancy"]
    row["structural_met_target_recall"] = result["structural_met_target"]
    row["n_structural_trials"] = result["n_structural_trials"]

    TRIAL_GRID_DIR.mkdir(parents=True, exist_ok=True)
    trial_grid_path = TRIAL_GRID_DIR / f"cell{cell['cell_id']}.json"
    with open(trial_grid_path, "w") as fh:
        json.dump({
            "cell_id": cell["cell_id"], "m": m, "L": L, "corr_threshold": corr_threshold,
            "corr_prop_requested": cell["corr_prop"], "n_vectors_fixed": FIXED_N_VECTORS,
            "target_recall": TARGET_RECALL, "stage1_trials": result["stage1_trials"],
        }, fh, indent=2, default=str)
    row["trial_grid_path"] = str(trial_grid_path)

    row["gamma_selected"] = best["gamma"]
    row["gamma_met_target_recall"] = result["gamma_met_target"]
    row["gamma_trials_tried"] = result["n_gamma_trials"]
    row["precision"] = best["metrics"].get("precision")
    row["recall"] = best["metrics"].get("recall")
    row["f1_score"] = best["metrics"].get("f1_score")
    row["n_steps"] = best["n_steps_run"]
    row["sketch_time"] = best["sketch_time"]
    row["candidate_time"] = best["candidate_time"]
    row["validation_time"] = best["validation_time"]
    row["monitor_time"] = best["monitor_time"]
    row["smart_wall_time"] = best["wall_time"]
    row["validated_candidates"] = best["validated_candidates"]
    row["total_candidates"] = best["total_candidates"]
    row["tested_candidates"] = best["tested_candidates"]
    if best["wall_time"] > 0 and bf_record.get("runtime"):
        row["speedup"] = bf_record["runtime"] / best["wall_time"]

    del bf_record, bf_corr_flags
    gc.collect()
    try:
        import resource as _resource
        row["peak_rss_gb"] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        row["peak_rss_gb"] = None
    writer.writerow(row)
    return row


def _cell_resume_key(cell):
    return ("lhs", str(cell["cell_id"]))


def _load_completed_cell_counts(output_csv):
    """(2026-08-27e) See experiment_lsh_sobol_sweep.py's identical function
    for the full story: the main sweep's own _load_completed_cell_counts is
    hardcoded to a different resume-key schema and never matches this
    script's ("lhs", cell_id) keys, making resume a silent no-op. Fixed
    here too even though this script is superseded, so its historical
    record stays technically correct rather than shipping a known bug."""
    path = Path(output_csv)
    if not path.exists():
        return {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = ("lhs", row.get("cell_id"))
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
        # (2026-08-27) load_synthetic_cell returns 5 values (data, ids, z,
        # achieved_z, achieved_corr_prop), not 2 -- caught by the smoke
        # test (ValueError: too many values to unpack). It already looks
        # up the generator's own meta.json for achieved_corr_prop, so the
        # manual stem_dir/meta.json re-lookup previously duplicated here
        # (and never even executed, since the unpack raised first) is
        # removed in favor of the value load_synthetic_cell already computes.
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
        "purpose": "Latin Hypercube joint sweep over (m, L, corr_prop, corr_threshold) -- "
                   "supersedes the OFAT-arm strategy for these 4 factors after the user rejected "
                   "both a screen-then-prune 2-level design (generalization risk from a cheap proxy "
                   "regime) and a shared-level grid (not enough distinct points per axis). See module "
                   "docstring and docs/implementation_log.md's 2026-08-26c entry.",
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
    }
    manifest_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Latin Hypercube joint sweep over (m, L, corr_prop, corr_threshold).")
    parser.add_argument("--generate-design", action="store_true",
                         help="Only generate/print the design (no compute) and exit.")
    parser.add_argument("--fresh-design", action="store_true",
                         help="Regenerate the design file even if it already exists (invalidates any prior resume state).")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing output CSV, start clean.")
    parser.add_argument("--only-cells", type=int, nargs="+", default=None,
                         help="Run only these cell_id(s) -- e.g. a single cheap cell as a smoke test "
                              "before committing to the full design. Does not affect resume bookkeeping "
                              "for the other cells.")
    parser.add_argument("--worker-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "lhs_sweep.csv"))
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
