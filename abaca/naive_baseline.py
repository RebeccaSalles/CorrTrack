"""The "naive" baseline the competitor papers measured against, versus this project's bruteforce
(user, 2026-09-18). Three all-pairs exact computations of the same quantity, same windows, same
threshold, on the same span, m swept:

  naive_python     per pair and per step, Pearson from scratch with numpy on the two windows
                   (the literal reading of the papers' "naive": O(m^2 W) scalar work, interpreted loop)
  naive_numpy      per step, one vectorized np.corrcoef over all m windows (the best a "naive"
                   implementation gets without incremental sums or compiled loops)
  bruteforce       this project's arm: all pairs enumerated, exact Pearson recomputed per pair-window
                   in the vectorized Cython kernel (the campaign's baseline)
  exact_stomp      this project's incremental five-sum arm (the stricter baseline)

Reports the wall time and the time ratio of each to bruteforce, and checks the correlated counts
agree. The ratios are what a paper that reports "Nx faster than naive" would have had to divide by.

    python abaca/naive_baseline.py --dataset-config experiment_dataset_sp500.py --window-size 60 --window-step 5 \
        --corr-threshold 0.9 --ms 25,50,100,200 --n-obs 800 --out naive.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO_DIR", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, REPO)
os.chdir(REPO)

import corrtrack_run_bruteforce as bfmod  # noqa: E402
from library_corrtrack_parallel import run_and_log_bruteforce  # noqa: E402


def naive_python(X, W, step, T, neg):
    """Per pair, per step: np.corrcoef on the two windows. Counts pair-windows at or above T."""
    m, N = X.shape
    count = 0
    t0 = time.perf_counter()
    for s0 in range(0, N - W + 1, step):
        win = X[:, s0:s0 + W]
        for i in range(m):
            xi = win[i]
            for j in range(i + 1, m):
                c = np.corrcoef(xi, win[j])[0, 1]
                if (abs(c) if neg else c) >= T:
                    count += 1
    return count, time.perf_counter() - t0


def naive_numpy(X, W, step, T, neg):
    """Per step: one vectorized corrcoef over all series."""
    m, N = X.shape
    count = 0
    t0 = time.perf_counter()
    iu = np.triu_indices(m, 1)
    for s0 in range(0, N - W + 1, step):
        C = np.corrcoef(X[:, s0:s0 + W])[iu]
        count += int(np.sum((np.abs(C) if neg else C) >= T))
    return count, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--window-size", type=int, default=60)
    ap.add_argument("--window-step", type=int, default=5)
    ap.add_argument("--corr-threshold", type=float, default=0.9)
    ap.add_argument("--neg-corr", action="store_true")
    ap.add_argument("--n-obs", type=int, default=None)
    ap.add_argument("--ms", default="25,50,100,200")
    ap.add_argument("--skip-python-above", type=int, default=400, help="the interpreted naive loop is skipped above this m (hours)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg_dataset = bfmod._load_dataset_config(Path(args.dataset_config))
    bfmod._apply_dataset_config(cfg_dataset)
    country, var, data, ids = next(bfmod.iter_datasets())
    n_obs = args.n_obs if args.n_obs is not None else bfmod.N_YEARS[0]
    rows = []
    for m in [int(v) for v in args.ms.split(",")]:
        test_data, ids_m = bfmod.prepare_test_data(data, ids, n_obs, m, bfmod.TRAIN_RATIO, tuning_mode="sampling")
        X = np.asarray(test_data[1:], dtype=np.float64)
        W, step, T, neg = args.window_size, args.window_step, args.corr_threshold, args.neg_corr
        # synchronous only: the naive loops above have no lag dimension
        base = dict(window_size=W, window_step=step, basic_window=step, n_lags=0, corr_threshold=T, neg_corr=neg, exec="sequential",
                    parallel_sketch=False, parallel_candidates=False, parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True,
                    artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged", save_only_required_artifacts=True,
                    save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")
        r = {"m": m, "n_obs": int(X.shape[1]), "W": W, "step": step, "T": T}
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("bruteforce", "exact_stomp"):
                t0 = time.perf_counter()
                rec, _, _ = run_and_log_bruteforce(f"naive_{m}", test_data, ids_m, dict(base, baseline_mode=mode), f"{tmp}/{mode}.csv", metadata={"nodes": 0},
                                                   recall_by_window=True, verbose=False, testing=False)
                r[mode] = {"wall": time.perf_counter() - t0, "correlated": int(rec["correlated"]), "runtime": rec.get("runtime")}
        cnt, wall = naive_numpy(X, W, step, T, neg); r["naive_numpy"] = {"wall": wall, "correlated": cnt}
        if m <= args.skip_python_above:
            cnt, wall = naive_python(X, W, step, T, neg); r["naive_python"] = {"wall": wall, "correlated": cnt}
        bfw = r["bruteforce"]["wall"]
        print(f"m={m:5d}: bruteforce {bfw:8.2f}s | exact_stomp {r['exact_stomp']['wall']:8.2f}s ({bfw / r['exact_stomp']['wall']:.2f}x) | "
              f"naive_numpy {r['naive_numpy']['wall']:8.2f}s ({r['naive_numpy']['wall'] / bfw:.1f}x slower) | "
              + (f"naive_python {r['naive_python']['wall']:8.2f}s ({r['naive_python']['wall'] / bfw:.0f}x slower)" if "naive_python" in r else "naive_python skipped")
              + f" | counts bf/stomp/numpy{'/py' if 'naive_python' in r else ''}: {r['bruteforce']['correlated']}/{r['exact_stomp']['correlated']}/{r['naive_numpy']['correlated']}"
              + (f"/{r['naive_python']['correlated']}" if "naive_python" in r else ""), flush=True)
        rows.append(r)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
