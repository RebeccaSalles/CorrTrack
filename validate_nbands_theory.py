"""validate_nbands_theory.py -- tests whether the closed-form LSH-banding
n_bands lower bound (derived 2026-08-30, see docs/implementation_log.md)
actually holds up against real measurements, before it goes anywhere near
the production tuning path.

Formula under test (classical LSH banding S-curve, matches this project's
own SignLSHBandIndex sign-random-projection mechanism and its existing
band_width = ceil(log2(m*L/occupancy)) sizing exactly):

    p(s) = 1 - arccos(s)/pi                    # single-bit match prob at true similarity s
    r    = ceil(log2(m*L/occupancy))            # band_width, already used in production
    recall(s) = 1 - (1 - p(s)^r)^b              # b = n_bands
    b_min = ceil( ln(1-target_recall) / ln(1-p(corr_threshold)^r) )

Motivation: real data from the just-completed Sobol/OA sweeps showed the
EMPIRICAL grid-search-selected n_bands is ALWAYS above b_min -- median
2.74x, worst case 7.1x -- and that overshoot correlates with m, which
directly explains why the measured candidate-search exponent (~1.71-2.29
depending on which dataset) came out worse than what theory predicts for
this threshold range (~1.15-1.42). This script checks the other
direction: does FORCING n_bands down to (a multiple of) b_min actually
still hit target recall in practice, or does the idealized formula miss
something real (finite-sample noise, generator imperfections, the sign-
projection model not perfectly matching this project's actual sketch) that
the empirical search's overshoot was silently correcting for?

(2026-08-30) Built at the user's explicit request, held until the Sobol/
OA sweeps finished per their own instruction ("I cannot publish the
method with this weakness"). Reuses run_bruteforce_once, calibrate_gamma,
run_smart_cell, load_synthetic_cell, the memory-cap/resume infrastructure
-- all UNCHANGED -- from the existing sweep family. Only the n_bands
selection method (formula instead of grid search) and the specific test
cells (a stratified sample of 12 already-completed, real cells spanning
m=50..300, L=8..64, corr_threshold=0.7..0.93) are new.

Design per cell:
  1. Compute FULL-length brute-force ground truth ONCE (n_steps=2000,
     reused across every n_bands candidate for that cell -- same
     unavoidable cost as the production path).
  2. For n_bands in {b_min, ceil(1.5*b_min), ceil(2*b_min)} (deduplicated):
     run calibrate_gamma (the SAME cheap 6-trial gamma-only search the
     production path already uses) at that FIXED n_bands -- isolates the
     n_bands-selection-method comparison cleanly: both the theory-driven
     and grid-search-driven paths get identical gamma tuning, only
     n_bands differs.
  3. Record recall/precision/speedup/candidate_time per (cell, n_bands
     candidate), and compare against that cell's ALREADY-RECORDED
     empirical grid-search result (n_bands_selected, recall, speedup)
     from the just-completed sweep CSVs.

NOT auto-launched by this script's own import.
"""
import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corrtrack_param_search import prepare_training_data
from library_corrtrack_parallel import CorrTrack

from experiment_lsh_cost_sweep import (
    load_synthetic_cell, run_bruteforce_once, calibrate_gamma,
    _make_memory_limiter, _migrate_csv_header_if_needed, _FlushingWriter, _current_git_commit,
)
# (2026-08-30) NOT importing _load_completed_cell_counts from there -- it's
# hardcoded to the main sweep's own resume-key schema and would silently
# never match this script's ("validate_nbands", test_id) keys, the exact
# bug found and fixed on 2026-08-27e for the Sobol/OA scripts. This script
# defines its own _load_completed_test_counts below instead.
from experiment_lsh_nbands_occupancy_sweep import FIXED_N_VECTORS

RESULT_DIR = Path("tmp_artifacts/validate_nbands_theory")
WINDOW_SIZE = 256
WINDOW_STEP = 16
N_STEPS = 2000
TARGET_RECALL = 0.95
MULTIPLIERS = (1.0, 1.5, 2.0)

CSV_COLUMNS = [
    "test_id", "source_cell_ref", "m", "L", "corr_prop_requested", "corr_threshold",
    "occupancy_used", "multiplier", "n_bands_theory", "n_bands_empirical_baseline",
    "recall_empirical_baseline", "speedup_empirical_baseline",
    "gamma_selected", "gamma_met_target_recall", "n_gamma_trials",
    "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
    "validated_candidates", "total_candidates", "tested_candidates",
    "bf_runtime", "bf_tested", "bf_correlated",
    "speedup", "precision", "recall", "f1_score", "peak_rss_gb",
]


def compute_b_min(m, L, occupancy, corr_threshold, target_recall=TARGET_RECALL):
    """Closed-form minimum n_bands for target_recall at the given (m, L,
    occupancy, corr_threshold), per the S-curve derivation in the module
    docstring. Mirrors SignLSHBandIndex._finalize_sizing's own band_width
    formula exactly (candidate_kernels.pyx) so r matches what production
    actually uses."""
    r = max(3, min(24, math.ceil(math.log2(max(m * L / occupancy, 2)))))
    p = 1 - math.acos(min(max(corr_threshold, -1.0), 1.0)) / math.pi
    p_r = p ** r
    if p_r >= 1.0:
        return 3  # degenerate: single band already guarantees a match
    denom = math.log(1 - p_r)
    if denom == 0:
        return 24 * 100  # p_r == 0 -- no finite b_min exists at this r; flag as absurd downstream
    b = math.ceil(math.log(1 - target_recall) / denom)
    return max(3, b)


def select_test_cells(n=12):
    """Pulls a stratified-by-m sample of already-completed, non-replicate
    cells straight from the just-finished sweep CSVs -- so every theory
    prediction is checked against a REAL empirical baseline, not a fresh
    unvalidated point."""
    sobol_path = Path("tmp_artifacts/lsh_sobol_sweep/sobol_sweep.csv")
    oa_path = Path("tmp_artifacts/lsh_orthogonal_array/oa_sweep.csv")
    frames = []
    if sobol_path.exists():
        s = pd.read_csv(sobol_path)
        s = s[~s["is_replicate"]]
        s["source"] = "sobol"
        frames.append(s)
    if oa_path.exists():
        o = pd.read_csv(oa_path)
        o["source"] = "oa"
        frames.append(o)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("m").reset_index(drop=True)
    idx = sorted(set(min(i, len(combined) - 1) for i in range(0, len(combined), max(1, len(combined) // n))))[:n]
    return combined.iloc[idx].to_dict(orient="records")


def build_test_matrix(cells):
    """Expands each source cell into one test row per distinct n_bands
    candidate (b_min * each multiplier, deduplicated)."""
    tests = []
    test_id = 1
    for i, c in enumerate(cells):
        m, L = int(c["m"]), int(c["L"])
        occ = float(c["target_occupancy_selected"])
        ct = float(c["corr_threshold"])
        b_min = compute_b_min(m, L, occ, ct)
        seen = set()
        for mult in MULTIPLIERS:
            n_bands = max(3, math.ceil(b_min * mult))
            if n_bands in seen:
                continue
            seen.add(n_bands)
            tests.append({
                "test_id": test_id, "source_index": i, "m": m, "L": L,
                "corr_prop_requested": float(c["corr_prop_requested"]),
                "corr_threshold": ct, "occupancy_used": occ,
                "multiplier": mult, "n_bands_theory": n_bands, "b_min": b_min,
                "n_bands_empirical_baseline": int(c["n_bands_selected"]),
                "recall_empirical_baseline": float(c["recall"]),
                "speedup_empirical_baseline": float(c["speedup"]),
                "source": c["source"],
            })
            test_id += 1
    return tests


def run_one_test(writer, test, train_data, ids):
    m, L, ct, occ = test["m"], test["L"], test["corr_threshold"], test["occupancy_used"]
    n_bands = test["n_bands_theory"]
    seed = 11
    n_lags = (L - 1) * WINDOW_STEP
    span_needed = WINDOW_SIZE + N_STEPS * WINDOW_STEP
    cell_data = train_data[:, :span_needed]

    row = {k: None for k in CSV_COLUMNS}
    row.update({
        "test_id": test["test_id"], "source_cell_ref": f"{test['source']}#{test['source_index']}",
        "m": m, "L": L, "corr_prop_requested": test["corr_prop_requested"], "corr_threshold": ct,
        "occupancy_used": occ, "multiplier": test["multiplier"], "n_bands_theory": n_bands,
        "n_bands_empirical_baseline": test["n_bands_empirical_baseline"],
        "recall_empirical_baseline": test["recall_empirical_baseline"],
        "speedup_empirical_baseline": test["speedup_empirical_baseline"],
    })

    bf_output_dir = RESULT_DIR / "bf_runs"
    bf_output_dir.mkdir(parents=True, exist_ok=True)
    bf_output_csv = str(bf_output_dir / f"bf_test{test['test_id']}_m{m}_L{L}_ct{ct}.csv")
    bf_record, bf_corr_flags = run_bruteforce_once(
        None, cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, ct, bf_output_csv,
    )
    row["bf_runtime"] = bf_record.get("runtime")
    row["bf_tested"] = bf_record.get("tested")
    row["bf_correlated"] = bf_record.get("correlated")

    best, met_target, n_trials, _default_gamma = calibrate_gamma(
        cell_data, ids, WINDOW_SIZE, WINDOW_STEP, n_lags, ct, occ, N_STEPS, seed,
        bf_corr_flags, bf_record.get("tested"), n_vectors=FIXED_N_VECTORS,
        candidate_lsh_n_bands=n_bands,
    )
    row["gamma_selected"] = best["gamma"]
    row["gamma_met_target_recall"] = met_target
    row["n_gamma_trials"] = n_trials
    row["precision"] = best["metrics"].get("precision")
    row["recall"] = best["metrics"].get("recall")
    row["f1_score"] = best["metrics"].get("f1_score")
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


def _test_resume_key(test):
    return ("validate_nbands", str(test["test_id"]))


def _load_completed_test_counts(output_csv):
    path = Path(output_csv)
    if not path.exists():
        return {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = ("validate_nbands", row.get("test_id"))
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        return {}
    return counts


def _run_worker_subprocess(test, output_csv, memory_limit_gb=None):
    import subprocess
    import sys as _sys
    spec = dict(test, output_csv=output_csv)
    result = subprocess.run(
        [_sys.executable, os.path.abspath(__file__), "--worker-test-spec", json.dumps(spec)],
        check=False, preexec_fn=_make_memory_limiter(memory_limit_gb),
    )
    if result.returncode != 0:
        print(f"[test {test['test_id']}] m={test['m']} L={test['L']} n_bands={test['n_bands_theory']} "
              f"-> WORKER FAILED (exit {result.returncode}), skipping")


def run_worker_test(spec):
    with open(spec["output_csv"], "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer = _FlushingWriter(writer, fh)
        m, corr_prop = spec["m"], spec["corr_prop_requested"]
        n_synth = WINDOW_SIZE + N_STEPS * WINDOW_STEP
        synth_max_lag = (spec["L"] - 1) * WINDOW_STEP
        data, ids, _z, _az, _acp = load_synthetic_cell(
            m, n_synth, corr_prop, WINDOW_SIZE, 11, synth_max_lag,
            threshold=spec["corr_threshold"], window_step=WINDOW_STEP,
        )
        train_data, ids_n = prepare_training_data(data, ids, n_year=n_synth, n_var=m, train_ratio=1.0)
        row = run_one_test(writer, spec, train_data, list(ids_n))
        print(f"[test {spec['test_id']}] m={m} L={spec['L']} ct={spec['corr_threshold']:.3f} "
              f"n_bands_theory={spec['n_bands_theory']} (baseline was {spec['n_bands_empirical_baseline']}) "
              f"-> recall={row['recall']} (baseline {spec['recall_empirical_baseline']:.3f}) "
              f"speedup={row['speedup']} (baseline {spec['speedup_empirical_baseline']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Validate the closed-form n_bands lower bound against real data.")
    parser.add_argument("--n-cells", type=int, default=12)
    parser.add_argument("--print-only", action="store_true",
                         help="Build and print the test matrix (no compute) and exit.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--only-tests", type=int, nargs="+", default=None)
    parser.add_argument("--worker-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "validation_results.csv"))
    parser.add_argument("--worker-test-spec", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_test_spec is not None:
        run_worker_test(json.loads(args.worker_test_spec))
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells = select_test_cells(n=args.n_cells)
    tests = build_test_matrix(cells)
    print(f"{len(cells)} source cells -> {len(tests)} theory-vs-empirical test points "
          f"(multipliers {MULTIPLIERS} of the theoretical b_min, deduplicated per cell).")
    if args.print_only:
        for t in tests:
            print(f"  test {t['test_id']:>3}: {t['source']}#{t['source_index']} m={t['m']:>4} L={t['L']:>3} "
                  f"ct={t['corr_threshold']:.3f} -> n_bands_theory={t['n_bands_theory']:>4} "
                  f"(x{t['multiplier']}, b_min={t['b_min']}) vs. baseline={t['n_bands_empirical_baseline']}")
        return

    completed_counts = {} if args.fresh else _load_completed_test_counts(args.output_csv)
    if args.fresh or not Path(args.output_csv).exists():
        with open(args.output_csv, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
    else:
        _migrate_csv_header_if_needed(args.output_csv, CSV_COLUMNS)

    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": _current_git_commit(),
        "purpose": "Validates the closed-form LSH-banding n_bands lower bound against real "
                   "measurements on 12 already-completed sweep cells, before adopting it in "
                   "the production tuning path. See module docstring.",
        "multipliers": list(MULTIPLIERS), "target_recall": TARGET_RECALL, "n_steps": N_STEPS,
        "source_cells": [{"m": c["m"], "L": c["L"], "corr_threshold": float(c["corr_threshold"]),
                           "source": c["source"]} for c in cells],
    }
    with open(Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    pending = [t for t in tests if completed_counts.get(_test_resume_key(t), 0) < 1]
    if args.only_tests is not None:
        wanted = set(args.only_tests)
        pending = [t for t in pending if t["test_id"] in wanted]
        print(f"--only-tests given: restricting to {len(pending)} test(s): {sorted(wanted)}.")
    elif len(pending) < len(tests):
        print(f"Resuming: skipping {len(tests) - len(pending)} already-completed test(s).")

    for test in pending:
        _run_worker_subprocess(test, args.output_csv, memory_limit_gb=args.worker_memory_limit_gb)

    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
