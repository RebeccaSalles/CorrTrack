"""CorrTrack ablation (user, 2026-09-17/18): all phases off, then on one at a time, so every phase
shows what it buys against the plain all-pairs computation. One fixed parameter set for the whole
ladder, always the cell's hyperopt best_params (required), the same span and the same bruteforce
ground truth; per variant: recall, precision, counters, sk/cand/val times, wall, and
the per-step latency boxplot ticks.

  bruteforce        all phases off: plain all-pairs Pearson (the reference)
  sketch_only       sketch computed, no candidate search: every pair goes through the shared validation
  sketch_hamming    sketch + sign-Hamming gate only (no index)
  sketch_dot        sketch + dot >= gamma gate only (no index)
  sketch_both       sketch + both gates (no index)
  lsh_hamming       sketch + LSH band index + Hamming gate only
  lsh_dot           sketch + LSH band index + dot gate only
  lsh_both          sketch + LSH band index + both gates (CorrTrack as run in the campaign)

    python abaca/ablation_corrtrack.py --dataset-config experiment_dataset_sp500.py --window-size 60 \
        --window-step 5 --n-lags 20 --corr-threshold 0.9 [--best-params <optim>/best_params_corrtrack.json] --out ablation.json
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


LADDER = {
    "sketch_only": {"candidate_backend": "all_pairs", "candidate_apply_dot_gamma_filter": False, "candidate_apply_hamming_filter": False},
    "sketch_hamming": {"candidate_backend": "all_pairs", "candidate_apply_dot_gamma_filter": False, "candidate_apply_hamming_filter": True},
    "sketch_dot": {"candidate_backend": "all_pairs", "candidate_apply_dot_gamma_filter": True, "candidate_apply_hamming_filter": False},
    "sketch_both": {"candidate_backend": "all_pairs", "candidate_apply_dot_gamma_filter": True, "candidate_apply_hamming_filter": True},
    "lsh_hamming": {"candidate_backend": "lsh_sign_dot", "candidate_apply_dot_gamma_filter": False, "candidate_apply_hamming_filter": True},
    "lsh_dot": {"candidate_backend": "lsh_sign_dot", "candidate_apply_dot_gamma_filter": True, "candidate_apply_hamming_filter": False},
    "lsh_both": {"candidate_backend": "lsh_sign_dot", "candidate_apply_dot_gamma_filter": True, "candidate_apply_hamming_filter": True},
}


def variants(tuned: dict):
    return dict(LADDER)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--window-size", type=int, default=168)
    ap.add_argument("--window-step", type=int, default=12)
    ap.add_argument("--basic-window", type=int, default=None)
    ap.add_argument("--n-lags", type=int, default=0)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--neg-corr", action="store_true")
    ap.add_argument("--preprocess", action="store_true", help="the cell's first-difference flag, applied to the truth and every rung")
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--n-obs", type=int, default=None)
    ap.add_argument("--best-params", default=None)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_dataset = bfmod._load_dataset_config(Path(args.dataset_config))
    bfmod._apply_dataset_config(cfg_dataset)
    country, var, data, ids = next(bfmod.iter_datasets())
    n_obs = args.n_obs if args.n_obs is not None else bfmod.N_YEARS[0]
    n_var = args.n_series if args.n_series is not None else bfmod.N_VARS[0]
    test_data, ids_n_var = bfmod.prepare_test_data(data, ids, n_obs, n_var, bfmod.TRAIN_RATIO, tuning_mode="sampling")
    label = f"{bfmod._dataset_slug(country, var)}_{len(ids_n_var)}_{test_data.shape[1]}"
    base = dict(window_size=args.window_size, window_step=args.window_step, basic_window=args.basic_window or args.window_step, n_lags=args.n_lags,
                corr_threshold=args.corr_threshold, neg_corr=args.neg_corr, preprocess=bool(args.preprocess), exec="sequential", parallel_sketch=False, parallel_candidates=False,
                parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True, artifact_mode="final", artifact_buffer_max_rows=250000,
                artifact_merge_mode="merged", save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
                validation_metric="pearson")
    # (2026-09-18, user) the ladder always runs with the TUNED parameters of the cell (CorrTrack's own
    # hyperopt output for this dataset / W / step / L / T), never with defaults: refuse otherwise
    if not (args.best_params and os.path.exists(args.best_params)):
        raise SystemExit("--best-params <optim>/best_params_corrtrack.json is required: the ablation uses the cell's tuned parameters")
    tuned = {k: v for k, v in json.load(open(args.best_params)).items() if not k.startswith("_")}
    tuned["preprocess"] = bool(args.preprocess)
    src = args.best_params
    todo = variants(tuned)
    if args.variants != "all":
        todo = {k: v for k, v in todo.items() if k in args.variants.split(",")}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        # (2026-09-19) every variant in its own forked child: phase times, peak / mean RSS, I/O, CPU, energy (abaca/resource_probe.py)
        os.makedirs(f"{tmp}/bf", exist_ok=True)
        (bf, _, bf_flags), bf_res = run_isolated(run_and_log_bruteforce, label, test_data, ids_n_var, dict(base, baseline_mode="bruteforce"), f"{tmp}/bf/bf.csv",
                                                 metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False, artifact_dir=f"{tmp}/bf")
        results["bruteforce"] = {"wall": bf_res["wall_s"], "correlated": bf["correlated"], "total_candidates": bf["total_candidates"],
                                 "cand_time": bf["cand_time"], "val_time": bf["val_time"], "runtime": bf.get("runtime"), **{k: bf.get(k) for k in STEP_KEYS}, "resources": bf_res}
        print(f"{'bruteforce':18s} wall={results['bruteforce']['wall']:.2f}s correlated={bf['correlated']} step med/q3={bf.get('step_time_median')}/{bf.get('step_time_q3')}", flush=True)
        for name, over in todo.items():
            rp = dict(tuned, **over)
            os.makedirs(f"{tmp}/{name}", exist_ok=True)
            try:
                (rec, _, flags), res = run_isolated(run_and_log_corrtrack, label, test_data, ids_n_var, base, rp, f"{tmp}/{name}/{name}.csv", metadata={"nodes": 0, "alg": name},
                                                    recall_by_window=True, corr_val=True, monitor=False, verbose=False, testing=False, artifact_dir=f"{tmp}/{name}")
            except Exception as exc:  # noqa: BLE001
                results[name] = {"status": "ERROR", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0]}", "overrides": over}
                print(f"{name:18s} ERROR {str(exc).splitlines()[0]}", flush=True)
                continue
            m = CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf["total_candidates"])
            U, P = int(bf["total_candidates"]), int(bf["correlated"]); fp = max(int(rec["total_candidates"]) - int(rec["correlated"]), 0)
            results[name] = {"status": "ok", "overrides": over, "wall": res["wall_s"], "resources": res, "runtime": rec.get("runtime"), "recall": m.get("recall"), "precision": m.get("precision"),
                             "candidate_precision": rec.get("candidate_precision"), "candidate_specificity": (1.0 - fp / (U - P)) if U > P else None,
                             "correlated": rec["correlated"], "total_candidates": rec["total_candidates"], "tested": rec["tested"],
                             "sk_time": rec.get("sk_time"), "cand_time": rec["cand_time"], "val_time": rec["val_time"], **{k: rec.get(k) for k in STEP_KEYS}}
            r = results[name]
            spec = f"{r['candidate_specificity']:.4f}" if r.get("candidate_specificity") is not None else "-"
            print(f"{name:18s} wall={r['wall']:.2f}s recall={r['recall']:.4f} cand={r['total_candidates']} cand_prec={r['candidate_precision']:.4f} cand_spec={spec} sk_t={r['sk_time']:.3f} cand_t={r['cand_time']:.3f} val_t={r['val_time']:.3f} "
                  f"step med/q3={r['step_time_median']:.5f}/{r['step_time_q3']:.5f}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"dataset": label, "config": vars(args), "tuned_source": src, "tuned": tuned, "variants": results}, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
