"""experiment_lsh_sobol_sweep_timing_only.py -- large-scale (m, L) CorrTrack-
ONLY timing companion to experiment_lsh_sobol_sweep.py.

(2026-09-07) Built at the user's explicit request, right after the two real
streaming memory leaks were fixed (SignLSHBandIndex/HammingExactIndex slot
reuse + Candidates._window_idx cache reuse -- see docs/implementation_log.md's
2026-09-04 (d)/(e) entries): "rerun the sobol benchmark and since memory is in
check now, increase m and L swept". A cost/time tradeoff discussion followed
(brute-force ground truth, O(m^2*L), dominates total wall time regardless of
how efficient CorrTrack's own memory/candidate-search is) and the user chose
a moderate widening for the FULL, accuracy-validated rerun (see
experiment_lsh_sobol_sweep.py's own M_RANGE/L_RANGE, now (50,500)/(8,96)),
plus: "add higher m and L for running corrtrack only (although we will not
have accuracy metrics, we can compare the running time)". This script is
exactly that: push m and L much further (this session's own memory
benchmarks already validated m=1000-3000 directly), skip brute force
entirely (the actual cost driver at scale), and compare CorrTrack's own
timing/candidate-volume behavior across a much wider range than the
accuracy-validated sweep can afford.

What this script deliberately does NOT do, and why:
  - No brute-force ground truth -- it is the O(m^2*L) cost this script exists
    to avoid. Consequence: no precision/recall/f1/speedup columns at all;
    this script answers "how does CorrTrack's own cost scale", not "is it
    still accurate at this scale" (already established at moderate scale by
    the sibling sweep; assumed, not re-verified, here).
  - No empirical (n_bands, occupancy, gamma) calibration -- that calibration
    search itself needs brute-force ground truth to check achieved recall
    (see experiment_lsh_nbands_occupancy_sweep.calibrate_nbands_occupancy).
    Instead this uses the THEORETICAL sizing directly:
    candidate_kernels.compute_lsh_sizing(m, L, target_occupancy,
    corr_threshold, n_vectors, target_recall, n_bands_tolerance=1.0) -- the
    exact overlap-corrected, safety-margin-padded formula _finalize_sizing
    itself uses internally (2026-09-03/-04 work), at tolerance=1.0 (this
    session's own "AUTO" convention: trust the calibrated margin, no extra
    multiplier). target_occupancy is FIXED at 3.0 (the project's own
    calibrated default) rather than swept/tuned -- a real simplification,
    disclosed plainly, not hidden.
  - gamma (candidate_cosine_threshold) is left at CorrTrack's own default
    (None -> derived from corr_threshold) rather than calibrated -- same
    reasoning: gamma calibration exists to trade off precision against
    recall, neither of which this script measures. Using the same
    uncalibrated derivation at every cell keeps cells comparable to each
    other, which is what a timing-vs-scale comparison needs.
  - No replicate re-runs, no anchor point -- this is a cost scaling study,
    not a statistical inference design; a single seed per point is enough to
    see how wall-time/candidate-volume trend with (m, L).

Design: same scrambled-Sobol machinery as the sibling sweep (4 factors: m,
L, corr_prop, corr_threshold), same log-uniform corr_prop/linear
corr_threshold ranges, but m in [50, 3000] and L in [8, 128] -- reaching
past the accuracy-validated sweep's own (50,500)/(8,96) range, into the
scale this session's own memory-leak-fix benchmarks (docs/implementation_
log.md's 2026-09-04 (d) entry) directly validated as memory-safe.
N_DESIGN_POINTS=64, matching the sibling sweep. Cost estimate (from a power-
law fit -- log(wall_time) ~ 1.586*log(m) + 0.346*log(L) - 6.39, R^2=0.80 --
against experiment_lsh_sobol_sweep.py's own OLD (m<=300,L<=64) run's real
measured smart_wall_time): ~5.2 hours total for N=64 points at this wider
range -- affordable specifically because brute force is skipped.
"""
import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

from corrtrack_param_search import prepare_training_data
from library_corrtrack_parallel import CorrTrack, estimate_proxy_pairs_per_anchor
import candidate_kernels as _cand_kernels

from experiment_lsh_cost_sweep import (
    load_synthetic_cell, run_smart_cell, TARGET_RECALL,
    calibrate_via_proxy_hyperopt, theoretical_sizing_fallback,
    PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR, PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING,
    _make_memory_limiter, _migrate_csv_header_if_needed,
    _FlushingWriter, _current_git_commit,
)
from experiment_lsh_nbands_occupancy_sweep import FIXED_N_VECTORS, OCCUPANCY_GRID

RESULT_DIR = Path("tmp_artifacts/lsh_sobol_sweep_timing_only")
DESIGN_PATH = RESULT_DIR / "sobol_timing_design_points.json"

# ------------------------------------------------------------- design space
# (2026-09-10) M_RANGE lower bound raised from 50 to 300: this sweep exists ONLY to extend
# CorrTrack's own measured range PAST where the accuracy sweep + brute-force comparison already
# cover it (m up to 296). Sampling m<300 here just re-measures a range that already has full
# brute-force-compared coverage -- pure duplicated compute. The 2026-09-08 run of this file used
# (50, 3000) and wasted ~12 completed + 16 pending cells on m<300 before the overlap was caught
# (see docs/implementation_log.md's 2026-09-09 (n) entry); those redundant rows remain in that
# run's CSV, harmless. Any fresh regeneration of the design should use this corrected range.
M_RANGE = (300, 3000)            # log-uniform -- starts just above the accuracy sweep's own ceiling (296)
L_RANGE = (8, 128)                # log-uniform
CORR_PROP_RANGE = (0.001, 0.10)  # log-uniform, synthetic only -- same as sibling sweep
CORR_THRESHOLD_RANGE = (0.7, 0.95)  # linear -- same as sibling sweep

N_DESIGN_POINTS = 64  # power of 2 -- required for Sobol's own balance guarantees
BASE_SEED = 11
SOBOL_RNG_SEED = 13  # deliberately different from the sibling sweep's 7 -- an
                      # independent draw, not a re-derivation of the same points

FIXED_TARGET_OCCUPANCY = 3.0  # project default -- see module docstring, not tuned here
N_BANDS_TOLERANCE = 1.0       # theoretical minimum, this session's own "AUTO" convention

WINDOW_SIZE = 256
WINDOW_STEP = 16
N_STEPS = 2000  # matches the sibling sweep's full-fidelity run length

CSV_COLUMNS = [
    "cell_id", "point_id", "m", "L", "corr_threshold",
    "corr_prop_requested", "corr_prop_achieved",
    "n_steps", "seed", "n_lags", "window_step", "window_size",
    "n_vectors_fixed", "target_occupancy_used", "gamma_used", "n_bands_used",
    "hyperopt_used", "est_pairs_per_anchor", "effective_n_series", "series_subsampled",
    "proxy_recall_lb", "proxy_recall_ub", "proxy_feasible", "proxy_underpowered",
    "proxy_n_gt", "proxy_anchor_count", "trial_grid_path",
    "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
    "validated_candidates", "total_candidates", "tested_candidates", "touched_candidates",
    "candidate_rate_vs_m2L", "peak_rss_gb",
    # (2026-09-08) Derived, no extra compute -- see the sibling accuracy sweep's identical
    # column for the full rationale. No brute force exists in this script, so there's no
    # speedup to compute here, just the CorrTrack-side stripped time itself.
    "corrtrack_time_no_val_no_monitor",
]


# ============================================================ design generation
def _generate_sobol_points(n=N_DESIGN_POINTS, seed=SOBOL_RNG_SEED):
    """Pure sampling, no compute -- mirrors experiment_lsh_sobol_sweep.py's
    own _generate_sobol_points exactly, just with this script's own (wider)
    M_RANGE/L_RANGE and its own independent SOBOL_RNG_SEED."""
    from scipy.stats import qmc

    if n & (n - 1) != 0:
        raise ValueError(f"N_DESIGN_POINTS={n} is not a power of 2 -- Sobol's own "
                          f"balance properties require it.")

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
            "cell_id": i + 1, "point_id": i,
            "m": m, "L": L,
            "corr_prop": float(round(cp_vals[i], 6)),
            "corr_threshold": float(ct_vals[i]),
            "seed": BASE_SEED,
        })
    return points


def generate_design(n=N_DESIGN_POINTS, fresh=False):
    if DESIGN_PATH.exists() and not fresh:
        return json.loads(DESIGN_PATH.read_text())["cells"]

    cells = _generate_sobol_points(n=n)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sobol_rng_seed": SOBOL_RNG_SEED, "n_design_points": n,
        "ranges": {"m": M_RANGE, "L": L_RANGE, "corr_prop": CORR_PROP_RANGE,
                   "corr_threshold": CORR_THRESHOLD_RANGE},
        "n_steps": N_STEPS, "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP,
        "target_occupancy_fixed": FIXED_TARGET_OCCUPANCY, "n_bands_tolerance": N_BANDS_TOLERANCE,
        "cells": cells,
    }
    with open(DESIGN_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return cells


# ============================================================ per-cell execution
def run_one_cell(writer, cell, train_data, ids, corr_prop_achieved=None):
    m, L, corr_threshold = cell["m"], cell["L"], cell["corr_threshold"]
    seed = cell["seed"]
    n_lags = (L - 1) * WINDOW_STEP
    span_needed = WINDOW_SIZE + N_STEPS * WINDOW_STEP
    cell_data = train_data[:, :span_needed]

    # (2026-09-08) Calibration added at the user's explicit request ("Even the timing-only
    # sweep should have calibration... normally we will not know beforehand where the true
    # correlations are"). No longer gated on est_pairs_per_anchor -- CorrTrack_optimize now
    # subsamples series internally whenever the full population would be impractical (see
    # calibrate_via_proxy_hyperopt's docstring and docs/implementation_log.md's 2026-09-08
    # entries), so real hyperopt is attempted at every scale, including this sweep's largest
    # m. hyperopt_used still records the rare genuine-failure fallback (get_optim_params found
    # nothing selectable at all), disclosed, not hidden.
    trial_grid_prefix = str(RESULT_DIR / "hyperopt_trials" / f"cell{cell['cell_id']}")
    Path(trial_grid_prefix).parent.mkdir(parents=True, exist_ok=True)
    est_pairs = estimate_proxy_pairs_per_anchor(m, n_lags, WINDOW_STEP)
    # pair_row_hard_ceiling is the TOTAL ceiling, kept separate from the per-anchor subsample
    # budget (PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR) -- see that constant's own 2026-09-08
    # "still later" comment for why conflating the two silently caps anchor_count at 1.
    hyperopt_result = calibrate_via_proxy_hyperopt(
        cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold,
        OCCUPANCY_GRID, m, seed, trial_grid_prefix, f"sobol_timing_cell{cell['cell_id']}",
        target_recall=TARGET_RECALL, n_vectors=FIXED_N_VECTORS,
        pair_row_hard_ceiling=PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING,
    )
    trial_grid_path = trial_grid_prefix + "_corrtrack.csv"
    hyperopt_used = hyperopt_result is not None
    if hyperopt_result is None:
        hyperopt_result = theoretical_sizing_fallback(
            m, L, corr_threshold, n_vectors=FIXED_N_VECTORS, target_occupancy=FIXED_TARGET_OCCUPANCY,
            target_recall=TARGET_RECALL,
        )

    row = {
        "cell_id": cell["cell_id"], "point_id": cell["point_id"],
        "m": m, "L": L, "corr_threshold": corr_threshold,
        "corr_prop_requested": cell["corr_prop"], "corr_prop_achieved": corr_prop_achieved,
        "n_steps": N_STEPS, "seed": seed, "n_lags": n_lags,
        "window_step": WINDOW_STEP, "window_size": WINDOW_SIZE,
        "n_vectors_fixed": FIXED_N_VECTORS,
        "target_occupancy_used": hyperopt_result["target_occupancy"],
        "gamma_used": hyperopt_result["gamma"],
        "n_bands_used": hyperopt_result["n_bands"],
        "hyperopt_used": hyperopt_used, "est_pairs_per_anchor": int(est_pairs),
        "effective_n_series": hyperopt_result.get("effective_n_series"),
        "series_subsampled": hyperopt_result.get("series_subsampled"),
        "proxy_recall_lb": hyperopt_result["proxy_recall_lb"],
        "proxy_recall_ub": hyperopt_result["proxy_recall_ub"],
        "proxy_feasible": hyperopt_result["proxy_feasible"],
        "proxy_underpowered": hyperopt_result["proxy_underpowered"],
        "proxy_n_gt": hyperopt_result["proxy_n_gt"],
        "proxy_anchor_count": hyperopt_result["proxy_anchor_count"],
        "trial_grid_path": trial_grid_path,
        "sketch_time": None, "candidate_time": None, "validation_time": None, "monitor_time": None,
        "smart_wall_time": None, "validated_candidates": None, "total_candidates": None,
        "tested_candidates": None, "touched_candidates": None,
        "candidate_rate_vs_m2L": None, "peak_rss_gb": None,
        "corrtrack_time_no_val_no_monitor": None,
    }

    final_ct, final_wall_time, final_n_steps_run = run_smart_cell(
        cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold,
        row["target_occupancy_used"], N_STEPS, seed,
        candidate_cosine_threshold=row["gamma_used"],
        n_vectors=FIXED_N_VECTORS, candidate_lsh_n_bands=row["n_bands_used"] or 64,
        target_recall=TARGET_RECALL,
    )
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
    row["corrtrack_time_no_val_no_monitor"] = (row["sketch_time"] or 0.0) + (row["candidate_time"] or 0.0)
    total_pairs_possible = (m * (m - 1) / 2 if m >= 2 else 1) * L * final_n_steps_run
    if total_pairs_possible > 0 and final_ct.tested_candidates is not None:
        row["candidate_rate_vs_m2L"] = final_ct.tested_candidates / total_pairs_possible

    del final_ct
    gc.collect()
    try:
        import resource as _resource
        row["peak_rss_gb"] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        row["peak_rss_gb"] = None
    writer.writerow(row)
    return row


def _cell_resume_key(cell):
    return ("sobol_timing", str(cell["cell_id"]))


def _load_completed_cell_counts(output_csv):
    path = Path(output_csv)
    if not path.exists():
        return {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = ("sobol_timing", row.get("cell_id"))
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        return {}
    return counts


def _run_worker_subprocess(cell, output_csv, memory_limit_gb=None):
    import os
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
              f"ct={spec['corr_threshold']:.3f} seed={spec['seed']} -> n_bands={row['n_bands_used']} "
              f"hyperopt_used={row['hyperopt_used']} wall_time={row['smart_wall_time']:.1f}s "
              f"candidate_rate={row['candidate_rate_vs_m2L']} touched={row['touched_candidates']}")


def write_run_manifest(args, cells):
    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Large-scale (m,L) CorrTrack-ONLY timing companion to experiment_lsh_sobol_sweep.py "
                   "-- no brute force, no accuracy metrics, theoretical (not calibrated) LSH sizing. "
                   "Built after the streaming memory-leak fixes; see module docstring and "
                   "docs/implementation_log.md's 2026-09-07 entry.",
        "output_csv": args.output_csv,
        "git_commit": _current_git_commit(),
        "cli_args": vars(args),
        "design_path": str(DESIGN_PATH),
        "n_cells": len(cells),
        "fixed": {
            "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP, "n_steps": N_STEPS,
            "n_vectors": FIXED_N_VECTORS, "target_occupancy": FIXED_TARGET_OCCUPANCY,
            "n_bands_tolerance": N_BANDS_TOLERANCE, "data_representation": "sketch_proj",
            "candidate_backend": "lsh_approx", "validation_metric": "pearson", "neg_corr": True,
            "candidate_apply_hamming_filter": False, "candidate_apply_dot_gamma_filter": True,
            "gamma": "CorrTrack default (derived from corr_threshold, not calibrated)",
        },
        "ranges": {"m": M_RANGE, "L": L_RANGE, "corr_prop": CORR_PROP_RANGE,
                   "corr_threshold": CORR_THRESHOLD_RANGE},
        "sibling_accuracy_sweep": "experiment_lsh_sobol_sweep.py -- full brute-force ground truth "
                                   "+ empirical calibration, moderate (50,500)/(8,96) range.",
    }
    manifest_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Large-scale (m,L) CorrTrack-only timing sweep (no brute force).")
    parser.add_argument("--generate-design", action="store_true")
    parser.add_argument("--fresh-design", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--only-cells", type=int, nargs="+", default=None)
    parser.add_argument("--worker-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "sobol_timing_sweep.csv"))
    parser.add_argument("--worker-cell-spec", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_cell_spec is not None:
        run_worker_cell(json.loads(args.worker_cell_spec))
        return

    cells = generate_design(fresh=args.fresh_design)

    if args.generate_design:
        print(f"{len(cells)} cells. m range observed: "
              f"[{min(c['m'] for c in cells)}, {max(c['m'] for c in cells)}], "
              f"L range observed: [{min(c['L'] for c in cells)}, {max(c['L'] for c in cells)}]")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
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
