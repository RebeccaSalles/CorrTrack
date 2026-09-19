"""Parallel scaling of CorrTrack (user, 2026-09-18): the tuned CorrTrack with exec="thread" and
parallel sketch / candidates / validation at max_workers in --threads, against its own sequential run,
and the plain bruteforce arm sequential vs parallel (parallel validation) at the same thread counts.
Same span, same truth; per variant: wall, stage times, speedup vs the sequential run of the same arm,
recall (must stay 1.0 for bruteforce and identical for CorrTrack), per-step latency ticks. Pin BLAS
threads (OMP/OPENBLAS/MKL_NUM_THREADS=1) outside so only the harness threads vary.

    python abaca/parallel_scaling.py --dataset-config experiment_dataset_sp500.py --window-size 60 --window-step 5 \
        --n-lags 20 --corr-threshold 0.9 --best-params <optim>/best_params_corrtrack.json --threads 1,2,4,8,16 --out scaling.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = os.environ.get("REPO_DIR", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, REPO)
os.chdir(REPO)

import corrtrack_run_bruteforce as bfmod  # noqa: E402
from library_corrtrack_parallel import CorrTrack, run_and_log_bruteforce, run_and_log_corrtrack  # noqa: E402
from abaca.resource_probe import run_isolated  # noqa: E402

STEP_KEYS = ("n_steps", "step_time_min", "step_time_q1", "step_time_median", "step_time_q3", "step_time_max",
             "step_time_whisker_lo", "step_time_whisker_hi", "step_time_outliers", "step_time_mean")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--window-size", type=int, default=168)
    ap.add_argument("--window-step", type=int, default=12)
    ap.add_argument("--basic-window", type=int, default=None)
    ap.add_argument("--n-lags", type=int, default=0)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--neg-corr", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--n-obs", type=int, default=None)
    ap.add_argument("--best-params", required=True, help="the cell's hyperopt best_params_corrtrack.json (required)")
    ap.add_argument("--threads", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=1, help="repeat every timing this many times, keep the median wall")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_dataset = bfmod._load_dataset_config(Path(args.dataset_config))
    bfmod._apply_dataset_config(cfg_dataset)
    country, var, data, ids = next(bfmod.iter_datasets())
    n_obs = args.n_obs if args.n_obs is not None else bfmod.N_YEARS[0]
    n_var = args.n_series if args.n_series is not None else bfmod.N_VARS[0]
    test_data, ids_n_var = bfmod.prepare_test_data(data, ids, n_obs, n_var, bfmod.TRAIN_RATIO, tuning_mode="sampling")
    label = f"{bfmod._dataset_slug(country, var)}_{len(ids_n_var)}_{test_data.shape[1]}"
    tuned = {k: v for k, v in json.load(open(args.best_params)).items() if not k.startswith("_")}
    tuned["preprocess"] = bool(args.preprocess)
    threads = [int(t) for t in args.threads.split(",")]

    def base_for(exec_mode, workers):
        par = exec_mode == "thread"
        return dict(window_size=args.window_size, window_step=args.window_step, basic_window=args.basic_window or args.window_step, n_lags=args.n_lags,
                    corr_threshold=args.corr_threshold, neg_corr=args.neg_corr, preprocess=bool(args.preprocess), exec=exec_mode,
                    parallel_sketch=par, parallel_candidates=par, parallel_validation=par, max_workers=int(workers), monitor=False,
                    track_min_dist=True, artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
                    save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")

    results = {"bruteforce": {}, "corrtrack": {}}
    with tempfile.TemporaryDirectory() as tmp:
        variants = [("sequential", 0)] + [("thread", t) for t in threads]
        bf_flags_ref = None
        for arm in ("bruteforce", "corrtrack"):
            for exec_mode, workers in variants:
                key = "seq" if exec_mode == "sequential" else f"thr{workers}"
                walls, rec, flags, reslist = [], None, None, []
                os.makedirs(f"{tmp}/{arm}_{key}", exist_ok=True)
                for _ in range(max(1, args.repeats)):
                    # (2026-09-19) forked child per run: the thread pool's memory and CPU time are the run's own (abaca/resource_probe.py)
                    if arm == "bruteforce":
                        (rec, _, flags), res = run_isolated(run_and_log_bruteforce, label, test_data, ids_n_var, dict(base_for(exec_mode, workers), baseline_mode="bruteforce"),
                                                            f"{tmp}/{arm}_{key}/{arm}.csv", metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False,
                                                            artifact_dir=f"{tmp}/{arm}_{key}")
                    else:
                        (rec, _, flags), res = run_isolated(run_and_log_corrtrack, label, test_data, ids_n_var, base_for(exec_mode, workers), tuned, f"{tmp}/{arm}_{key}/{arm}.csv",
                                                            metadata={"nodes": 0, "alg": arm}, recall_by_window=True, corr_val=True, monitor=False, verbose=False, testing=False,
                                                            artifact_dir=f"{tmp}/{arm}_{key}")
                    walls.append(res["wall_s"]); reslist.append(res)
                if arm == "bruteforce" and key == "seq":
                    bf_flags_ref = flags
                m = CorrTrack.compute_metrics_bf(flags, bf_flags_ref, windows=True, total_pairs_bf=None) if bf_flags_ref is not None else {}
                r = {"exec": exec_mode, "workers": workers, "wall_median": sorted(walls)[len(walls) // 2], "walls": walls, "recall_vs_seq_bf": m.get("recall"),
                     "correlated": rec["correlated"], "sk_time": rec.get("sk_time"), "cand_time": rec["cand_time"], "val_time": rec["val_time"],
                     "runtime": rec.get("runtime"), **{k: rec.get(k) for k in STEP_KEYS},
                     "resources": reslist[len(reslist) // 2], "cpu_total_s_median": sorted(x["cpu_user_s"] + x["cpu_sys_s"] for x in reslist)[len(reslist) // 2],
                     "peak_rss_delta_mb_max": max((x["peak_rss_delta_mb"] or 0) for x in reslist)}
                results[arm][key] = r
                seq = results[arm].get("seq")
                sp = (seq["wall_median"] / r["wall_median"]) if seq and r["wall_median"] else float("nan")
                print(f"{arm:10s} {key:6s} wall={r['wall_median']:8.2f}s speedup_vs_seq={sp:5.2f}x sk={r['sk_time'] or 0:.2f} cand={r['cand_time']:.2f} val={r['val_time']:.2f} "
                      f"recall={r['recall_vs_seq_bf']} step_med={1e3 * (r['step_time_median'] or 0):.3f}ms", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"dataset": label, "config": vars(args), "results": results}, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
