"""Diagnostic (2026-07-15, research thread B): how long does a pair, once
it enters the "persistent core" (true |corr| >= persistent_threshold),
stay there? This determines whether a cheap, RARELY-REFRESHED index over
just the persistent core (the idea raised this session: separate the
persistent >=0.95 spike from the diffuse bulk, index the former cheaply,
full-search only the latter) is a safe design, or whether persistent-set
membership churns too fast to trust an infrequently-refreshed structure.

Same-timestep (lag=0) pairwise Pearson correlation is tracked across many
consecutive real window-start steps (SketchCache's step cadence), and for
each pair that is ever seen persistent, the run-length (consecutive steps
staying persistent) is recorded.

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/diag_persistent_core_duration.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_fr_air_temperature_121_1.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --persistent-threshold 0.95 --n-steps 150
"""
import argparse
from pathlib import Path

import numpy as np

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize
from diag_gamma_reliability import _load_module, _resolve_cfg_value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--exec-param-config", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-step", type=int, default=16)
    parser.add_argument("--persistent-threshold", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    args = parser.parse_args()

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    cps._apply_dataset_config(dataset_cfg)
    country, var, data, ids = next(cps.iter_datasets())
    n_var = cps._get_cfg_attr(dataset_cfg, "N_VARS", "N_SERIES")
    n_year = cps._get_cfg_attr(dataset_cfg, "N_YEARS", "N_OBS")
    n_var = n_var[0] if isinstance(n_var, (list, tuple)) else n_var
    n_year = n_year[0] if isinstance(n_year, (list, tuple)) else n_year
    train_data, proxy_ids = cps.prepare_training_data(data, ids, n_year, n_var, args.train_ratio)
    n_series = train_data.shape[0] - 1
    values = np.asarray(train_data[1:1 + n_series, :], dtype=np.float64)
    length_data = train_data.shape[1]
    print(f"Dataset: {country}/{var} n_series={n_series} length={length_data}")

    max_start = length_data - args.window_size
    all_starts = list(range(0, max_start + 1, args.window_step))[:args.n_steps]

    tau = args.persistent_threshold
    run_lengths = []
    current_runs = {}  # pair -> current run length
    persistent_counts = []

    for step_id, start in enumerate(all_starts):
        window = values[:, start:start + args.window_size]
        valid = np.array([CorrTrack_optimize._proxy_window_is_valid(window[i]) for i in range(n_series)])
        idx = np.where(valid)[0]
        w = window[idx]
        centered = w - w.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        unit = centered / norm
        corr = unit @ unit.T
        n = len(idx)
        iu = np.triu_indices(n, k=1)
        persistent_mask = np.abs(corr[iu]) >= tau
        pairs_now = set()
        for k in np.where(persistent_mask)[0]:
            i, j = idx[iu[0][k]], idx[iu[1][k]]
            pairs_now.add((int(i), int(j)))
        persistent_counts.append(len(pairs_now))

        for p in list(current_runs.keys()):
            if p not in pairs_now:
                run_lengths.append(current_runs.pop(p))
        for p in pairs_now:
            current_runs[p] = current_runs.get(p, 0) + 1

        if step_id % 30 == 0:
            print(f"  step {step_id}/{len(all_starts)}, persistent_now={len(pairs_now)}", flush=True)

    for p, rl in current_runs.items():
        run_lengths.append(rl)

    run_lengths = np.array(run_lengths)
    print(f"\nTotal distinct persistent-episode instances: {len(run_lengths)}")
    print(f"Mean persistent-set size per step: {np.mean(persistent_counts):.1f} "
          f"(min={min(persistent_counts)}, max={max(persistent_counts)})")
    if len(run_lengths):
        print(f"Run-length (consecutive steps persistent) stats: "
              f"mean={run_lengths.mean():.2f}, median={np.median(run_lengths):.1f}, "
              f"p10={np.percentile(run_lengths, 10):.1f}, p90={np.percentile(run_lengths, 90):.1f}, "
              f"max={run_lengths.max()}")
        censored_at_end = sum(1 for p, rl in current_runs.items())
        frac_single_step = float((run_lengths == 1).mean())
        print(f"Fraction of episodes lasting only 1 step (pure noise crossings): {frac_single_step:.3f}")
        print(f"Episodes still ongoing at end of run (right-censored, run length is a lower bound): {censored_at_end}")


if __name__ == "__main__":
    main()
