"""(2026-08-26c) SUPERSEDED for the joint (m, L, corr_prop, corr_threshold)
question by experiment_lsh_lhs_sweep.py's Latin Hypercube design, after the
user wanted finer/more points than this file's OFAT arms give and was
unconvinced a cheap screen-then-prune step would generalize (see that
script's own module docstring and docs/implementation_log.md's 2026-08-26c
entry). This file's `calibrate_nbands_occupancy` (and FIXED_N_VECTORS/
N_BANDS_GRID/OCCUPANCY_GRID) is reused UNCHANGED by that script -- nothing
here was wrong, the OFAT top-level design around it was just replaced by
something denser. Left in place, still unlaunched, as a simpler fallback
if the LHS design turns out to be infeasible.

experiment_lsh_nbands_occupancy_sweep.py -- follow-up to
experiment_lsh_cost_sweep.py, scoped to answer ONE specific question left
open by that sweep (see docs/implementation_log.md's 2026-08-26 entries and
the "Reading the LSH Sweep" report): is the m^2.29 candidate-search scaling
exponent found there a real property of this algorithm, or an artifact of
a coarse tuning strategy -- n_bands jumping 64->128->256 (must-double
steps) while candidate_lsh_target_occupancy was held fixed at 3.0 the
whole time, never co-tuned with m?

(2026-08-26) User approved this scope, explicitly asked it be BUILT BUT
NOT LAUNCHED -- do not run this script as part of implementing it. Launch
only on explicit instruction.

Design (differs from experiment_lsh_cost_sweep.py's calibrate_hyperparams
in exactly the two ways needed to test the question above; everything
else -- gamma calibration, brute-force ground truth, memory cap, resume
logic -- is reused directly, not reimplemented):

  - m: {50, 75, 100, 150, 200, 300} -- 6 points (up from the main sweep's
    3: 50, 100, 200) to actually resolve whether the exponent holds, bends,
    or was noise. m=300 is included as an ATTEMPT -- it failed once
    already under a tighter memory cap; the other 5 points don't depend on
    it succeeding (resume logic + per-cell subprocess isolation mean a
    failure there just gets skipped, same as the main sweep).
  - L fixed at 32, corr_prop fixed at 0.05 -- matches the main sweep's own
    m-line reference exactly, so results are directly comparable to it.
  - n_vectors FIXED at 64, NOT searched. The main sweep's own tuner chose
    64 at every single point along its m-line (m=50,100,200) -- there is
    no evidence it needs to vary, so this dimension is dropped to free up
    budget for finer resolution on the two dimensions actually in
    question.
  - n_bands: finer grid {48, 64, 96, 128, 192, 256, 384} -- ~1.4-1.5x
    steps instead of the main sweep's must-double 2x steps. Tests whether
    a smaller increment than doubling is enough to hold recall at each m.
  - target_occupancy: SWEPT JOINTLY with m, {2.0, 3.0, 5.0, 10.0} -- part
    of the same stage-1 structural search as n_bands (not a separate,
    fixed-elsewhere axis like in the main sweep). Tests whether loosening/
    tightening occupancy can substitute for adding bands as m grows.
  - gamma: identical 2-stage approach as the main sweep -- reused directly
    via calibrate_gamma (stage 2, unchanged) after this script's own
    stage-1 structural search over (n_bands, target_occupancy) at gamma=0
    (mirrors calibrate_hyperparams' own rationale exactly, just searching
    a different pair of parameters).

Cost: 7 n_bands x 4 occupancy = 28 stage-1 trials + up to 6 gamma trials
= up to 34 trials per m point x 6 m points ~= 200 trials total, plus one
brute-force ground-truth run per m (reused across all 28 structural
combos at that m, same caching discipline as the main sweep).

(2026-08-26b) PLANNED, NOT YET LAUNCHED -- added per the user's explicit
request after reviewing the main sweep's report: a corr_threshold OFAT
extension, plus per-cell persistence of the full (n_bands, occupancy)
trial grid (previously computed and discarded -- see
docs/implementation_log.md's 2026-08-26b entry and the "Reading the LSH
Sweep" report's Question 7 for why the main sweep's own report could only
show the WINNING combo, not the full search surface).

  - corr_threshold: {0.7 (default/existing), 0.8, 0.9, 0.95}. NOT crossed
    with the full 6-point m-line (would multiply total cost ~4x, on top
    of a grid that already failed once at m=300 under a tight memory
    cap). Instead, OFAT at REFERENCE_M_FOR_THRESHOLD=200 (this sweep's own
    m=200 cell already covers threshold=0.7, so only 3 NEW cells are
    added: 0.8, 0.9, 0.95, each still running the FULL (n_bands,
    occupancy) structural search + gamma fine-tuning, since a tighter
    threshold changes both the recall target's difficulty and gamma's own
    anchor point -- reusing another threshold's winning combo would not
    be valid). This mirrors this script's own existing pattern (L and
    occupancy already vary only at fixed reference points elsewhere in
    the main sweep design) rather than introducing a new, differently-
    shaped grid.
  - Trial-grid persistence: calibrate_nbands_occupancy's stage1_trials
    (all 28 (n_bands, occupancy) -> (recall, tested_candidates) results,
    not just the winner) are now returned and written to a per-cell
    side JSON under RESULT_DIR/trial_grids/. Gamma's own trial list is
    NOT persisted in this pass -- calibrate_gamma is reused UNCHANGED
    from the main sweep (see the import comment below) and its trials
    list is smaller (<=6 points) and lower-priority than the (n_bands,
    occupancy) surface the user specifically flagged as the candidate-
    search hotspot; persisting it too is a natural, separately-scoped
    follow-up, not done here to keep this change minimal.

Extra cost from this addition: 3 new cells (threshold in {0.8,0.9,0.95})
x up to 34 trials each ~= 100 extra trials, plus 3 extra brute-force runs
at m=200 -- roughly the cost of one more m-point in the existing grid,
not a new axis multiplying the whole design.
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

# Reuse the main sweep's own tested building blocks directly rather than
# reimplementing brute-force invocation, gamma calibration, memory capping,
# or resume logic. Only _run_worker_subprocess is NOT reused (it hardcodes
# `os.path.abspath(__file__)` to re-exec the CURRENT script -- importing it
# would re-launch experiment_lsh_cost_sweep.py instead of this file, since
# __file__ binds to the module it's defined in, not the caller).
from experiment_lsh_cost_sweep import (
    load_synthetic_cell, run_smart_cell, run_bruteforce_once,
    calibrate_gamma, TARGET_RECALL, GAMMA_TRIAL_OFFSETS,
    _make_memory_limiter, _load_completed_cell_counts, _migrate_csv_header_if_needed,
    _FlushingWriter, _current_git_commit,
)
# (2026-08-26) load_synthetic_cell (imported above) reads/writes the
# synthetic data cache using experiment_lsh_cost_sweep's OWN module-level
# SYNTH_CACHE_ROOT internally (a closure over that module's global, not a
# parameter) -- defining a separate SYNTH_CACHE_ROOT here would silently
# look in the wrong directory for the achieved-corr_prop meta.json below
# (a real bug caught before ever running this script), and would also
# forfeit a real efficiency win: since this sweep uses the identical
# (corr_prop=0.05, window_size=256, seed=11, n_steps=2000) generator
# parameters as the main sweep's own m-line, m=50/100/200 may already be
# cached from that earlier run and can be reused instead of regenerated.
import experiment_lsh_cost_sweep as _main_sweep
SYNTH_CACHE_ROOT = _main_sweep.SYNTH_CACHE_ROOT

cps.OBS_MODE = "count"

RESULT_DIR = Path("tmp_artifacts/lsh_nbands_occupancy_sweep")

# Matches the main sweep's own reference line exactly, so results are
# directly comparable to it (same L, same density).
REFERENCE_L = 32
REFERENCE_CORR_PROP = 0.05
FIXED_N_VECTORS = 64

M_VALUES = (50, 75, 100, 150, 200, 300)
# (2026-08-30) Replaced the absolute-value grid with a tolerance-multiplier
# grid over the closed-form LSH-banding minimum (b_min), now computed
# directly inside CorrTrack/SignLSHBandIndex (candidate_lsh_n_bands_tolerance
# -- see candidate_kernels.pyx's SignLSHBandIndex._n_bands_tolerance and
# docs/implementation_log.md's 2026-08-30 entries for the derivation and
# real-data validation). tolerance=2.0 matched or beat the old empirical
# grid search's own reliability across 12 real cells at a median of 75% as
# many bands -- {1.0, 1.5, 2.0} brackets that finding directly, replacing
# the old N_BANDS_GRID=(48,64,96,128,192,256,384) (7 absolute values,
# median 2.74x more bands than theoretically necessary).
N_BANDS_TOLERANCE_GRID = (1.0, 1.5, 2.0)
OCCUPANCY_GRID = (2.0, 3.0, 5.0, 10.0)

# (2026-08-26b) PLANNED, NOT YET LAUNCHED -- see module docstring. OFAT
# extension at a fixed reference m, not crossed with the full m-line.
CORR_THRESHOLD_GRID = (0.7, 0.8, 0.9, 0.95)
REFERENCE_M_FOR_THRESHOLD = 200  # the m-line's own existing m=200 cell covers threshold=0.7

TRIAL_GRID_DIR = RESULT_DIR / "trial_grids"

CSV_COLUMNS = [
    "cell_id", "m", "L", "corr_threshold", "corr_prop_requested", "corr_prop_achieved",
    "n_vectors_fixed", "n_bands_selected", "target_occupancy_selected", "gamma_selected",
    "n_steps", "seed",
    "structural_met_target_recall", "n_structural_trials", "gamma_met_target_recall", "gamma_trials_tried",
    "target_recall", "trial_grid_path",
    "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
    "validated_candidates", "total_candidates", "tested_candidates",
    "bf_runtime", "bf_tested", "bf_correlated",
    "speedup", "precision", "recall", "f1_score", "peak_rss_gb",
]


def calibrate_nbands_occupancy(cell_data, ids, window_size, window_step, n_lags, corr_threshold,
                                n_steps, seed, bf_corr_flags, bf_tested):
    """Stage 1: joint (n_bands_tolerance, target_occupancy) structural
    search at gamma=0.0 (isolates the LSH retrieval stage's own recall
    ceiling, exactly as calibrate_hyperparams does for (n_vectors, n_bands)
    in the main sweep) -- n_vectors held at FIXED_N_VECTORS throughout, not
    searched. Keep combos reaching TARGET_RECALL; among those, pick the one
    touching fewest candidates (fastest). Stage 2: calibrate_gamma, reused
    unchanged, within the winning (tolerance, occupancy).

    (2026-08-30) n_bands is no longer picked from a discrete absolute-value
    grid -- each trial passes candidate_lsh_n_bands_tolerance directly to
    CorrTrack, which computes n_bands = ceil(tolerance * b_min) internally
    (candidate_kernels.pyx's SignLSHBandIndex, once observed_m is known).
    The actual EFFECTIVE n_bands used (read back via ct.grid_nodes[0].
    _lsh_index.n_bands, a real Cython property, not re-derived in Python)
    is recorded per trial so the persisted trial grid still shows real
    band counts, not just the tolerance that produced them."""
    stage1_trials = []
    for tolerance, occ in itertools.product(N_BANDS_TOLERANCE_GRID, OCCUPANCY_GRID):
        ct, wall_time, n_steps_run = run_smart_cell(
            cell_data, ids, window_size, window_step, n_lags, corr_threshold,
            occ, n_steps, seed, candidate_cosine_threshold=0.0,
            n_vectors=FIXED_N_VECTORS, candidate_lsh_n_bands=64,
            candidate_lsh_n_bands_tolerance=tolerance, target_recall=TARGET_RECALL,
        )
        metrics = CorrTrack.compute_metrics_bf(
            ct, bf_corr_flags, windows=True, total_pairs_bf=bf_tested,
        )
        effective_n_bands = ct.grid_nodes[0]._lsh_index.n_bands
        stage1_trials.append({
            "tolerance": tolerance, "occupancy": occ, "n_bands": int(effective_n_bands),
            "recall": metrics.get("recall"), "tested_candidates": ct.tested_candidates,
        })
        del ct
        gc.collect()

    meeting_target = [t for t in stage1_trials if t["recall"] is not None and t["recall"] >= TARGET_RECALL]
    if meeting_target:
        best_structural = min(meeting_target, key=lambda t: t["tested_candidates"])
        structural_met_target = True
    else:
        best_structural = max(stage1_trials, key=lambda t: (t["recall"] if t["recall"] is not None else -1.0))
        structural_met_target = False

    tolerance_sel = best_structural["tolerance"]
    occ_sel = best_structural["occupancy"]
    n_bands_sel = best_structural["n_bands"]

    best_gamma_trial, gamma_met_target, n_gamma_trials, gamma_default = calibrate_gamma(
        cell_data, ids, window_size, window_step, n_lags, corr_threshold,
        occ_sel, n_steps, seed, bf_corr_flags, bf_tested,
        n_vectors=FIXED_N_VECTORS, candidate_lsh_n_bands=64,
        candidate_lsh_n_bands_tolerance=tolerance_sel, target_recall=TARGET_RECALL,
    )

    return {
        "n_bands": n_bands_sel, "n_bands_tolerance": tolerance_sel, "occupancy": occ_sel,
        "structural_met_target": structural_met_target, "n_structural_trials": len(stage1_trials),
        "gamma_trial": best_gamma_trial, "gamma_met_target": gamma_met_target,
        "n_gamma_trials": n_gamma_trials, "gamma_default": gamma_default,
        # (2026-08-26b) full stage-1 grid, not just the winner -- see module
        # docstring's "Trial-grid persistence" note. Written to disk by the
        # caller (run_one_cell), not here, so this function stays a pure
        # calibration step with no I/O side effects.
        "stage1_trials": stage1_trials,
    }


def run_one_cell(writer, cell_id, m, window_step, window_size, corr_threshold, n_steps, seed,
                  corr_prop_requested, train_data, ids, corr_prop_achieved=None):
    n_lags = (REFERENCE_L - 1) * window_step
    span_needed = window_size + n_steps * window_step
    cell_data = train_data[:, :span_needed]

    row = {
        "cell_id": cell_id, "m": m, "L": REFERENCE_L, "corr_threshold": corr_threshold,
        "corr_prop_requested": corr_prop_requested, "corr_prop_achieved": corr_prop_achieved,
        "n_vectors_fixed": FIXED_N_VECTORS, "n_bands_selected": None, "target_occupancy_selected": None,
        "gamma_selected": None, "n_steps": n_steps, "seed": seed,
        "structural_met_target_recall": None, "n_structural_trials": 0,
        "gamma_met_target_recall": None, "gamma_trials_tried": 0, "target_recall": TARGET_RECALL,
        "trial_grid_path": None,
        "sketch_time": None, "candidate_time": None, "validation_time": None, "monitor_time": None,
        "smart_wall_time": None, "validated_candidates": None, "total_candidates": None, "tested_candidates": None,
        "bf_runtime": None, "bf_tested": None, "bf_correlated": None,
        "speedup": None, "precision": None, "recall": None, "f1_score": None,
    }

    bf_output_csv = str((RESULT_DIR / "bf_runs" / f"bf_m{m}_ct{corr_threshold}_n{n_steps}_s{seed}.csv"))
    Path(bf_output_csv).parent.mkdir(parents=True, exist_ok=True)
    bf_record, bf_corr_flags = run_bruteforce_once(
        None, cell_data, ids, window_size, window_step, n_lags, corr_threshold, bf_output_csv,
    )
    row["bf_runtime"] = bf_record.get("runtime")
    row["bf_tested"] = bf_record.get("tested")
    row["bf_correlated"] = bf_record.get("correlated")

    result = calibrate_nbands_occupancy(
        cell_data, ids, window_size, window_step, n_lags, corr_threshold,
        n_steps, seed, bf_corr_flags, bf_record.get("tested"),
    )
    best = result["gamma_trial"]
    row["n_bands_selected"] = result["n_bands"]
    row["target_occupancy_selected"] = result["occupancy"]
    row["structural_met_target_recall"] = result["structural_met_target"]
    row["n_structural_trials"] = result["n_structural_trials"]

    TRIAL_GRID_DIR.mkdir(parents=True, exist_ok=True)
    trial_grid_path = TRIAL_GRID_DIR / f"cell{cell_id}_m{m}_ct{corr_threshold}.json"
    with open(trial_grid_path, "w") as fh:
        json.dump({
            "cell_id": cell_id, "m": m, "L": REFERENCE_L, "corr_threshold": corr_threshold,
            "n_vectors_fixed": FIXED_N_VECTORS, "target_recall": TARGET_RECALL,
            "stage1_trials": result["stage1_trials"],  # full (n_bands, occupancy) -> (recall, tested_candidates) grid
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


def _cell_resume_key(spec):
    # (2026-08-26b) Includes corr_threshold now that multiple threshold
    # cells can share the same m (the REFERENCE_M_FOR_THRESHOLD OFAT
    # extension) -- keying on m alone would make the second threshold
    # cell at that m look "already completed" and get skipped on resume.
    return ("nbands_occ", str(spec["m"]), str(spec["corr_threshold"]))


def _run_worker_subprocess(spec, memory_limit_gb=None):
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, os.path.abspath(__file__), "--worker-cell-spec", json.dumps(spec)],
        check=False, preexec_fn=_make_memory_limiter(memory_limit_gb),
    )
    if result.returncode != 0:
        suspected_oom = result.returncode < 0 or result.returncode == 1
        print(f"[cell {spec['cell_id']}] m={spec['m']} ct={spec['corr_threshold']} -> WORKER FAILED (exit {result.returncode}"
              f"{', possibly hit the memory limit or an OOM kill' if suspected_oom and memory_limit_gb else ''}"
              f"), skipping this cell")


def run_worker_cell(spec):
    with open(spec["output_csv"], "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer = _FlushingWriter(writer, fh)
        m = spec["m"]
        # (2026-08-27) Fixed two real bugs, caught while smoke-testing the
        # sibling LHS sweep script (this script itself was never actually
        # run, per the "build but do not launch" instruction, so neither
        # had surfaced yet): (1) load_synthetic_cell returns 5 values, not
        # 2 -- this line would have crashed with "too many values to
        # unpack" the first time it ran; (2) `threshold` wasn't forwarded,
        # so injected pairs were always calibrated for a 0.7 bar regardless
        # of this script's own corr_threshold OFAT extension (0.7-0.95) --
        # would have silently produced near-zero recall at every threshold
        # above ~0.7.
        data, ids, _z_used, _achieved_z, achieved_cp = load_synthetic_cell(
            m, spec["n_synth"], REFERENCE_CORR_PROP, spec["window_size"], spec["seed"], spec["synth_max_lag"],
            threshold=spec["corr_threshold"], window_step=spec["window_step"],
        )
        train_data, ids_n = prepare_training_data(data, ids, n_year=spec["n_synth"], n_var=m, train_ratio=1.0)
        # achieved_cp already computed by load_synthetic_cell above -- the
        # manual re-derivation previously here duplicated that (and, worse,
        # would silently overwrite it with None whenever stem_dir/meta.json
        # lookup failed) -- removed in the same 2026-08-27 pass.
        row = run_one_cell(
            writer, spec["cell_id"], m, spec["window_step"], spec["window_size"],
            spec["corr_threshold"], spec["n_steps"], spec["seed"],
            REFERENCE_CORR_PROP, train_data, list(ids_n),
            corr_prop_achieved=achieved_cp,
        )
        print(f"[cell {spec['cell_id']}] m={m} ct={spec['corr_threshold']} -> n_bands={row['n_bands_selected']} "
              f"occupancy={row['target_occupancy_selected']} speedup={row['speedup']} recall={row['recall']}")


def write_run_manifest(args):
    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Follow-up to experiment_lsh_cost_sweep.py -- tests whether the m^2.29 "
                   "candidate-search exponent found there is real or a coarse-tuning artifact, "
                   "by searching n_bands at finer resolution and target_occupancy jointly with m "
                   "(both held fixed/coarse in the main sweep), with n_vectors fixed at 64 (what "
                   "the main sweep's own tuner always chose, freeing budget for the axes above).",
        "output_csv": args.output_csv,
        "git_commit": _current_git_commit(),
        "cli_args": vars(args),
        "fixed": {
            "L": REFERENCE_L, "corr_prop": REFERENCE_CORR_PROP, "n_vectors": FIXED_N_VECTORS,
            "data_representation": "sketch_proj", "candidate_backend": "lsh_approx",
            "validation_metric": "pearson", "neg_corr": True,
            "candidate_apply_hamming_filter": False, "candidate_apply_dot_gamma_filter": True,
        },
        "search_grids": {
            "m_values": list(M_VALUES), "n_bands_tolerance_grid": list(N_BANDS_TOLERANCE_GRID),
            "target_occupancy_grid": list(OCCUPANCY_GRID), "gamma_trial_offsets": list(GAMMA_TRIAL_OFFSETS),
            "corr_threshold_ofat_values": [] if args.skip_threshold_sweep else list(args.corr_threshold_values),
            "corr_threshold_reference_m": REFERENCE_M_FOR_THRESHOLD,
        },
        "trial_grid_dir": str(TRIAL_GRID_DIR),
        "comparison_baseline": "experiment_lsh_cost_sweep.py's own m-line at L=32, corr_prop=0.05, "
                                "occupancy=3.0 (fixed), n_bands in {64,128,256} (tuned) -- "
                                "found candidate_time ~ m^2.29.",
    }
    manifest_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Follow-up: finer n_bands + m-jointly-tuned occupancy sweep.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--m-values", type=int, nargs="+", default=list(M_VALUES))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-step", type=int, default=16)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--corr-threshold-values", type=float, nargs="+", default=list(CORR_THRESHOLD_GRID),
                         help="OFAT extension values, run only at REFERENCE_M_FOR_THRESHOLD "
                              f"(={REFERENCE_M_FOR_THRESHOLD}); --corr-threshold's own value is skipped "
                              "here since the main m-line already covers it there.")
    parser.add_argument("--skip-threshold-sweep", action="store_true",
                         help="Run only the m-line (n_bands x occupancy), skip the corr_threshold OFAT extension.")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--worker-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "lsh_nbands_occupancy_sweep.csv"))
    parser.add_argument("--worker-cell-spec", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_cell_spec is not None:
        run_worker_cell(json.loads(args.worker_cell_spec))
        return

    if args.smoke:
        args.m_values = [50, 100]
        args.n_steps = 150
        args.output_csv = str(RESULT_DIR / "lsh_nbands_occupancy_sweep_smoke.csv")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "bf_runs").mkdir(parents=True, exist_ok=True)

    n_synth = args.window_size + args.n_steps * args.window_step
    synth_max_lag = (REFERENCE_L - 1) * args.window_step

    n_extra_ct = 0 if args.skip_threshold_sweep or REFERENCE_M_FOR_THRESHOLD not in args.m_values else \
        len([ct for ct in args.corr_threshold_values if abs(ct - args.corr_threshold) >= 1e-9])
    print(f"Grid: {len(args.m_values)} m-values x ({len(N_BANDS_TOLERANCE_GRID)} n_bands_tolerance x {len(OCCUPANCY_GRID)} "
          f"occupancy = {len(N_BANDS_TOLERANCE_GRID) * len(OCCUPANCY_GRID)} structural trials + up to "
          f"{len(GAMMA_TRIAL_OFFSETS)} gamma trials) per m-value, plus {n_extra_ct} extra corr_threshold "
          f"OFAT cell(s) at m={REFERENCE_M_FOR_THRESHOLD} ({sorted(set(args.corr_threshold_values) - {args.corr_threshold})}), "
          f"each cell in its own subprocess (memory-capped at {args.worker_memory_limit_gb}GB, m=300 is an "
          f"attempt that may fail without affecting the other points). Each cell's full (n_bands, occupancy) "
          f"trial grid is now persisted to {TRIAL_GRID_DIR}/, not just the winner. NOT auto-launched by this "
          f"script's own import -- this only runs when explicitly invoked.")

    completed_counts = {} if args.fresh else _load_completed_cell_counts(args.output_csv)
    if not args.fresh and completed_counts:
        print(f"Resuming: found {len(completed_counts)} already-attempted m-value(s) in {args.output_csv}.")

    if args.fresh or not Path(args.output_csv).exists():
        with open(args.output_csv, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
    else:
        _migrate_csv_header_if_needed(args.output_csv, CSV_COLUMNS)

    write_run_manifest(args)

    shared = dict(
        window_size=args.window_size, window_step=args.window_step,
        n_steps=args.n_steps, seed=args.seed, output_csv=args.output_csv,
        n_synth=n_synth, synth_max_lag=synth_max_lag,
    )

    pending = [dict(shared, m=m, corr_threshold=args.corr_threshold) for m in sorted(args.m_values)]
    # (2026-08-26b) PLANNED, NOT YET LAUNCHED -- corr_threshold OFAT
    # extension at a fixed reference m (see module docstring). Skips
    # args.corr_threshold itself since the m-line's own
    # REFERENCE_M_FOR_THRESHOLD cell already covers it -- adding it again
    # here would just duplicate that cell under a different cell_id.
    if not args.skip_threshold_sweep and REFERENCE_M_FOR_THRESHOLD in args.m_values:
        for ct in args.corr_threshold_values:
            if abs(ct - args.corr_threshold) < 1e-9:
                continue
            pending.append(dict(shared, m=REFERENCE_M_FOR_THRESHOLD, corr_threshold=ct))

    if completed_counts:
        n_before = len(pending)
        pending = [spec for spec in pending if completed_counts.get(_cell_resume_key(spec), 0) < 1]
        skipped = n_before - len(pending)
        if skipped:
            print(f"Skipping {skipped} cell(s) already completed.")

    for cell_id, spec in enumerate(pending, start=1):
        spec["cell_id"] = cell_id
        _run_worker_subprocess(spec, memory_limit_gb=args.worker_memory_limit_gb)

    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
