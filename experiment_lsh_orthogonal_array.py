"""experiment_lsh_orthogonal_array.py -- a genuine, verified orthogonal
array over (m, L, corr_prop, corr_threshold), run ALONGSIDE the Sobol
sweep (experiment_lsh_sobol_sweep.py), not instead of it.

(2026-08-27d) Built at the user's explicit request, for a specific
practical reason: the user's supervisor prefers combinatorial designs and
"is like that" (will ask for one regardless of the Sobol argument already
made) -- rather than argue the point alone, the user wants a real
orthogonal-array result already in hand. This is NOT a relabeling of
Sobol points, and NOT merely "combinatorial design lite" -- it is a
textbook OA(16, 5, 4, 2), constructed from actual GF(4) (Galois field of
4 elements) arithmetic via the classical "q-1 mutually orthogonal Latin
squares from a finite field of order q" result, using 4 of its 5
available columns:

    L_a(x, y) = x + a*y   (GF(4) addition/multiplication), for a in {1,2,3}

Together with the row index x and column index y themselves, this gives
5 columns; only 4 are needed here, so the 5th (a=3) is dropped. Verified
programmatically (see the standalone check run before writing this file)
that all C(5,2)=10 column pairs are perfectly balanced: every one of the
16 possible (level_i, level_j) combinations occurs EXACTLY once. This is
the actual property "orthogonal array" refers to -- not asserted, checked
directly on the generated array before it was ever used here.

Factor levels (4 per factor, matching the abstract array's 4 levels
0..3) were chosen to reuse points ALREADY tested in the main sweep
(experiment_lsh_cost_sweep.py) wherever possible, so this design's
results are directly comparable to already-trusted numbers, not a fresh,
uncalibrated set of points:
  - m:              {50, 100, 200, 300}   -- the main sweep's own m-line
  - L:              {8, 16, 32, 64}       -- the main sweep's own L-line
  - corr_prop:      {0.001, 0.01, 0.05, 0.10} -- the main sweep's own density line
  - corr_threshold: {0.7, 0.8, 0.9, 0.95} -- exactly what the user
    originally asked to sweep, several turns before this design existed

Cost: 16 runs total, each at the same full fidelity as the Sobol sweep
(n_steps=2000, the full (n_bands x occupancy) tuning search via
experiment_lsh_nbands_occupancy_sweep.py's calibrate_nbands_occupancy,
reused unchanged) -- small next to the Sobol sweep's 333 cells, cheap
enough to run to completion even standalone.

Building blocks (run_smart_cell, run_bruteforce_once, calibrate_gamma,
calibrate_nbands_occupancy, the memory-cap/resume/trial-grid-persistence
infrastructure) are reused UNCHANGED from the sibling sweep scripts --
only the design (this file's actual subject) is new.

NOT auto-launched by this script's own import. Per the same "plan, verify,
don't launch without explicit instruction" discipline used throughout
this sweep family.
"""
import argparse
import csv
import gc
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

import corrtrack_param_search as cps
from corrtrack_param_search import prepare_training_data
from library_corrtrack_parallel import CorrTrack

from experiment_lsh_cost_sweep import (
    load_synthetic_cell, run_smart_cell, run_bruteforce_once,
    calibrate_gamma, TARGET_RECALL, GAMMA_TRIAL_OFFSETS,
    _make_memory_limiter, _migrate_csv_header_if_needed,
    _FlushingWriter, _current_git_commit,
)
from experiment_lsh_nbands_occupancy_sweep import (
    calibrate_nbands_occupancy, FIXED_N_VECTORS, N_BANDS_TOLERANCE_GRID, OCCUPANCY_GRID,
)

cps.OBS_MODE = "count"

RESULT_DIR = Path("tmp_artifacts/lsh_orthogonal_array")
TRIAL_GRID_DIR = RESULT_DIR / "trial_grids"
DESIGN_PATH = RESULT_DIR / "oa_design_points.json"
COST_MODEL_PATH = Path("tmp_artifacts/lsh_cost_sweep/cost_model_analysis.json")

WINDOW_SIZE = 256
WINDOW_STEP = 16
N_STEPS = 2000

# (2026-08-28) Same calibration-span shrink as experiment_lsh_sobol_sweep.py
# -- see that file's own comment for the full rationale (real observed
# per-cell cost was dominated by the 34-trial search replaying the full
# n_steps=2000 stream every time; proxy-anchor+bootstrap hyperopt was
# considered and rejected as a structural mismatch to this generator's
# narrow, localized injected events). The 34-trial search now runs on a
# short prefix; one final full-length run with the selected hyperparameters
# produces the actually-reported numbers.
CALIB_N_STEPS = max(200, N_STEPS // 5)
BASE_SEED = 11

# Concrete levels per abstract array level 0..3 -- see module docstring for
# why these specific values (reuse of already-tested main-sweep points).
M_LEVELS = (50, 100, 200, 300)
L_LEVELS = (8, 16, 32, 64)
CORR_PROP_LEVELS = (0.001, 0.01, 0.05, 0.10)
CORR_THRESHOLD_LEVELS = (0.7, 0.8, 0.9, 0.95)

CSV_COLUMNS = [
    "cell_id", "run_index", "m", "L", "corr_threshold", "corr_prop_requested", "corr_prop_achieved",
    "n_steps", "seed", "n_lags", "window_step", "window_size",
    "n_vectors_fixed", "n_bands_selected", "n_bands_tolerance_selected", "target_occupancy_selected", "gamma_selected",
    "structural_met_target_recall", "n_structural_trials", "gamma_met_target_recall", "gamma_trials_tried",
    "target_recall", "trial_grid_path",
    "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
    "validated_candidates", "total_candidates", "tested_candidates",
    "bf_runtime", "bf_tested", "bf_correlated",
    "speedup", "precision", "recall", "f1_score", "peak_rss_gb",
]


# ==================================================== orthogonal array construction
def _gf4_orthogonal_array():
    """Returns (runs, is_orthogonal) where runs is a (16,4) int array with
    entries in {0,1,2,3}, and is_orthogonal is the result of an explicit
    strength-2 balance check (every column-pair sees every one of the 16
    level-combinations exactly once) -- verified here at generation time,
    not just asserted in the docstring, so a future edit that breaks the
    construction fails loudly instead of silently shipping an unbalanced
    "orthogonal" array."""
    mul = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    def add(a, b):
        return a ^ b

    runs = []
    for x in range(4):
        for y in range(4):
            runs.append([x, y, add(x, mul[1][y]), add(x, mul[2][y])])
    runs = np.array(runs)

    ok = True
    for c1, c2 in itertools.combinations(range(4), 2):
        counts = {}
        for r in runs:
            key = (int(r[c1]), int(r[c2]))
            counts[key] = counts.get(key, 0) + 1
        if len(counts) != 16 or any(v != 1 for v in counts.values()):
            ok = False
    return runs, ok


def generate_design(fresh=False):
    """Pure construction + JSON write -- no CorrTrack/brute-force compute."""
    if DESIGN_PATH.exists() and not fresh:
        return json.loads(DESIGN_PATH.read_text())["cells"]

    runs, is_orthogonal = _gf4_orthogonal_array()
    if not is_orthogonal:
        raise RuntimeError(
            "GF(4)-constructed array failed its own strength-2 balance check -- "
            "refusing to write a design that isn't actually orthogonal despite the name."
        )

    cells = []
    for i, row in enumerate(runs):
        m = M_LEVELS[row[0]]
        L = L_LEVELS[row[1]]
        corr_prop = CORR_PROP_LEVELS[row[2]]
        corr_threshold = CORR_THRESHOLD_LEVELS[row[3]]
        cells.append({
            "cell_id": i + 1, "run_index": i,
            "m": m, "L": L, "corr_prop": corr_prop, "corr_threshold": corr_threshold,
            "seed": BASE_SEED,
            "oa_levels": row.tolist(),
        })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "construction": "OA(16,5,4,2) via GF(4) MOLS (L_a(x,y)=x+a*y for a in {1,2}, "
                         "plus row/column indices), 4 of 5 available columns used, "
                         "verified strength-2 orthogonal at generation time.",
        "levels": {"m": M_LEVELS, "L": L_LEVELS, "corr_prop": CORR_PROP_LEVELS,
                   "corr_threshold": CORR_THRESHOLD_LEVELS},
        "n_steps": N_STEPS, "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP,
        "cells": cells,
    }
    with open(DESIGN_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return cells


def print_design_summary(cells):
    print(f"{len(cells)}-run orthogonal array (OA(16,5,4,2), 4 columns used, "
          f"verified strength-2 balanced).")
    print(f"{'cell':>4} {'m':>4} {'L':>3} {'corr_prop':>10} {'corr_thr':>9} {'seed':>5}")
    for c in cells:
        print(f"{c['cell_id']:>4} {c['m']:>4} {c['L']:>3} {c['corr_prop']:>10.5f} "
              f"{c['corr_threshold']:>9.3f} {c['seed']:>5}")

    const_median = 4.140268404706565e-07
    if COST_MODEL_PATH.exists():
        try:
            const_median = json.loads(COST_MODEL_PATH.read_text())["bruteforce_closed_form"]["const_median"]
        except Exception:
            pass
    total_bf_s = sum(
        (c["m"] * (c["m"] - 1) / 2) * c["L"] * N_STEPS * const_median for c in cells
    )
    # (2026-08-29) The old x1.5 guess was never measured and was wrong by
    # ~3.5x in practice -- see experiment_lsh_sobol_sweep.py's identical
    # fix. Using the same real post-fix measured ratio (5.28x) here.
    MEASURED_TOTAL_OVER_BF_RATIO = 5.28
    print(f"\nEstimated brute-force-only time: {total_bf_s/3600:.2f} h")
    print(f"Rough total incl. tuning search (calibrated from real post-fix measurements, "
          f"ratio={MEASURED_TOTAL_OVER_BF_RATIO}x): {total_bf_s*MEASURED_TOTAL_OVER_BF_RATIO/3600:.2f} h")


# ============================================================ per-cell execution
def run_one_cell(writer, cell, train_data, ids, corr_prop_achieved=None):
    m, L, corr_threshold = cell["m"], cell["L"], cell["corr_threshold"]
    seed = cell["seed"]
    n_lags = (L - 1) * WINDOW_STEP
    span_needed = WINDOW_SIZE + N_STEPS * WINDOW_STEP
    cell_data = train_data[:, :span_needed]

    row = {
        "cell_id": cell["cell_id"], "run_index": cell["run_index"],
        "m": m, "L": L, "corr_threshold": corr_threshold,
        "corr_prop_requested": cell["corr_prop"], "corr_prop_achieved": corr_prop_achieved,
        "n_steps": N_STEPS, "seed": seed, "n_lags": n_lags,
        "window_step": WINDOW_STEP, "window_size": WINDOW_SIZE,
        "n_vectors_fixed": FIXED_N_VECTORS, "n_bands_selected": None, "n_bands_tolerance_selected": None, "target_occupancy_selected": None,
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

    # ---- Phase 1: SHORT-span search (cheap) -- see experiment_lsh_sobol_
    # sweep.py's identical comment for the full rationale.
    calib_n_steps = min(CALIB_N_STEPS, N_STEPS)
    calib_span = WINDOW_SIZE + calib_n_steps * WINDOW_STEP
    calib_data = cell_data[:, :calib_span]
    calib_bf_csv = str(bf_output_dir / f"bf_cell{cell['cell_id']}_calib_m{m}_L{L}_ct{corr_threshold}_s{seed}.csv")
    calib_bf_record, calib_bf_corr_flags = run_bruteforce_once(
        None, calib_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold, calib_bf_csv,
    )
    result = calibrate_nbands_occupancy(
        calib_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold,
        calib_n_steps, seed, calib_bf_corr_flags, calib_bf_record.get("tested"),
    )
    row["n_bands_selected"] = result["n_bands"]
    row["n_bands_tolerance_selected"] = result.get("n_bands_tolerance")
    row["target_occupancy_selected"] = result["occupancy"]
    row["structural_met_target_recall"] = result["structural_met_target"]
    row["n_structural_trials"] = result["n_structural_trials"]
    row["gamma_selected"] = result["gamma_trial"]["gamma"]
    row["gamma_met_target_recall"] = result["gamma_met_target"]
    row["gamma_trials_tried"] = result["n_gamma_trials"]

    TRIAL_GRID_DIR.mkdir(parents=True, exist_ok=True)
    trial_grid_path = TRIAL_GRID_DIR / f"cell{cell['cell_id']}.json"
    with open(trial_grid_path, "w") as fh:
        json.dump({
            "cell_id": cell["cell_id"], "m": m, "L": L, "corr_threshold": corr_threshold,
            "corr_prop_requested": cell["corr_prop"], "n_vectors_fixed": FIXED_N_VECTORS,
            "target_recall": TARGET_RECALL, "calib_n_steps": calib_n_steps,
            "stage1_trials": result["stage1_trials"],
        }, fh, indent=2, default=str)
    row["trial_grid_path"] = str(trial_grid_path)
    del calib_bf_record, calib_bf_corr_flags, calib_data
    gc.collect()

    # ---- Phase 2: FULL-length confirmation (unavoidable, paid once) --
    # actual reported numbers come from this run.
    bf_output_csv = str(bf_output_dir / f"bf_cell{cell['cell_id']}_m{m}_L{L}_ct{corr_threshold}_s{seed}.csv")
    bf_record, bf_corr_flags = run_bruteforce_once(
        None, cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, corr_threshold, bf_output_csv,
    )
    row["bf_runtime"] = bf_record.get("runtime")
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
    if final_wall_time > 0 and bf_record.get("runtime"):
        row["speedup"] = bf_record["runtime"] / final_wall_time

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
    return ("oa", str(cell["cell_id"]))


def _load_completed_cell_counts(output_csv):
    """(2026-08-27e) See experiment_lsh_sobol_sweep.py's identical function
    for the full story: the main sweep's own _load_completed_cell_counts is
    hardcoded to a different resume-key schema and never matches this
    script's ("oa", cell_id) keys -- caught by direct observation on this
    exact script's first launch (reported finding cell 1 already done, then
    re-ran it anyway). Fixed with a script-local version keyed on cell_id."""
    path = Path(output_csv)
    if not path.exists():
        return {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = ("oa", row.get("cell_id"))
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
              f"ct={spec['corr_threshold']:.3f} -> n_bands={row['n_bands_selected']} "
              f"occ={row['target_occupancy_selected']} speedup={row['speedup']} recall={row['recall']}")


def write_run_manifest(args, cells):
    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Genuine OA(16,5,4,2) orthogonal array (4 of 5 columns used) over "
                   "(m, L, corr_prop, corr_threshold), constructed via GF(4) mutually "
                   "orthogonal Latin squares and verified strength-2 balanced at generation "
                   "time. Run alongside experiment_lsh_sobol_sweep.py, not instead of it -- "
                   "built specifically so a combinatorial-design result is already in hand "
                   "when asked for one. See docs/implementation_log.md's 2026-08-27d entry.",
        "output_csv": args.output_csv,
        "git_commit": _current_git_commit(),
        "cli_args": vars(args),
        "design_path": str(DESIGN_PATH),
        "n_cells": len(cells),
        "levels": {"m": M_LEVELS, "L": L_LEVELS, "corr_prop": CORR_PROP_LEVELS,
                   "corr_threshold": CORR_THRESHOLD_LEVELS},
        "fixed": {
            "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP, "n_steps": N_STEPS,
            "n_vectors": FIXED_N_VECTORS, "data_representation": "sketch_proj",
            "candidate_backend": "lsh_approx", "validation_metric": "pearson", "neg_corr": True,
        },
        "tuning_grids": {"n_bands_tolerance_grid": list(N_BANDS_TOLERANCE_GRID), "occupancy_grid": list(OCCUPANCY_GRID),
                          "gamma_trial_offsets": list(GAMMA_TRIAL_OFFSETS)},
    }
    manifest_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="16-run orthogonal array over (m, L, corr_prop, corr_threshold).")
    parser.add_argument("--generate-design", action="store_true")
    parser.add_argument("--fresh-design", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--only-cells", type=int, nargs="+", default=None)
    parser.add_argument("--worker-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "oa_sweep.csv"))
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
