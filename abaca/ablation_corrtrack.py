"""CorrTrack ablation (user's request, 2026-09-17): show that each phase earns its place.

Starting from the tuned configuration (hyperopt best_params, or defaults), one component is
removed or replaced at a time and the arm is re-run on the same span against the same bruteforce
ground truth; recall, counters, time and the per-step latency quantiles are reported next to the
full system. Variants:

  full               the tuned CorrTrack
  no_gamma_filter    candidate_apply_dot_gamma_filter=False: every index hit goes to validation
  no_hamming_filter  candidate_apply_hamming_filter=False: no sign-Hamming pre-gate before the dot
  no_filters         both gates off: the raw LSH band retrieval alone
  hamming_exact      candidate_backend=lsh_hamming_exact: the exact packed-Hamming index instead of banded LSH
  no_hybrid / hybrid hybrid_validation toggled relative to the tuned value
  n_vectors_half / n_vectors_double   sketch width halved / doubled (sensitivity; hyperopt covers the rest)
  bruteforce         no candidate stage at all (reference; recall 1 by definition)

    python abaca/ablation_corrtrack.py --dataset-config experiment_dataset_sp500.py --window-size 60 \
        --window-step 5 --n-lags 20 --corr-threshold 0.9 --best-params <optim>/best_params_corrtrack.json --out ablation.json
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

STEP_KEYS = ("n_steps", "step_time_p50", "step_time_p90", "step_time_p99", "step_time_max", "step_time_mean")


def variants(tuned: dict):
    nv = int(tuned.get("n_vectors", 32))
    hyb = bool(tuned.get("hybrid_validation", False))
    return {
        "full": {},
        "no_gamma_filter": {"candidate_apply_dot_gamma_filter": False},
        "no_hamming_filter": {"candidate_apply_hamming_filter": False},
        "no_filters": {"candidate_apply_dot_gamma_filter": False, "candidate_apply_hamming_filter": False},
        "hamming_exact": {"candidate_backend": "lsh_hamming_exact"},
        ("hybrid" if not hyb else "no_hybrid"): {"hybrid_validation": not hyb},
        "n_vectors_half": {"n_vectors": max(4, nv // 2)},
        "n_vectors_double": {"n_vectors": nv * 2},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--window-size", type=int, default=168)
    ap.add_argument("--window-step", type=int, default=12)
    ap.add_argument("--basic-window", type=int, default=None)
    ap.add_argument("--n-lags", type=int, default=0)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--neg-corr", action="store_true")
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
                corr_threshold=args.corr_threshold, neg_corr=args.neg_corr, exec="sequential", parallel_sketch=False, parallel_candidates=False,
                parallel_validation=False, max_workers=0, monitor=True, track_min_dist=True, artifact_mode="final", artifact_buffer_max_rows=250000,
                artifact_merge_mode="merged", save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
                validation_metric="pearson")
    if args.best_params and os.path.exists(args.best_params):
        tuned = {k: v for k, v in json.load(open(args.best_params)).items() if not k.startswith("_")}
        src = args.best_params
    else:
        tuned = dict(n_vectors=32, seed=2468, seed_toggle=1357, preprocess=False)
        src = "UNTUNED defaults"
    todo = variants(tuned)
    if args.variants != "all":
        todo = {k: v for k, v in todo.items() if k in args.variants.split(",")}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        bf, _, bf_flags = run_and_log_bruteforce(label, test_data, ids_n_var, dict(base, baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                                                 metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
        results["bruteforce"] = {"wall": time.perf_counter() - t0, "correlated": bf["correlated"], "total_candidates": bf["total_candidates"],
                                 "cand_time": bf["cand_time"], "val_time": bf["val_time"], **{k: bf.get(k) for k in STEP_KEYS}}
        print(f"{'bruteforce':18s} wall={results['bruteforce']['wall']:.2f}s correlated={bf['correlated']} p50/p99 step={bf.get('step_time_p50')}/{bf.get('step_time_p99')}", flush=True)
        for name, over in todo.items():
            rp = dict(tuned, **over)
            t0 = time.perf_counter()
            try:
                rec, _, flags = run_and_log_corrtrack(label, test_data, ids_n_var, base, rp, f"{tmp}/{name}.csv", metadata={"nodes": 0, "alg": name},
                                                      recall_by_window=True, corr_val=True, monitor=True, verbose=False, testing=False)
            except Exception as exc:  # noqa: BLE001
                results[name] = {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}", "overrides": over}
                print(f"{name:18s} ERROR {exc}", flush=True)
                continue
            m = CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf["total_candidates"])
            results[name] = {"status": "ok", "overrides": over, "wall": time.perf_counter() - t0, "recall": m.get("recall"), "precision": m.get("precision"),
                             "correlated": rec["correlated"], "total_candidates": rec["total_candidates"], "tested": rec["tested"],
                             "sk_time": rec.get("sk_time"), "cand_time": rec["cand_time"], "val_time": rec["val_time"], **{k: rec.get(k) for k in STEP_KEYS}}
            r = results[name]
            print(f"{name:18s} wall={r['wall']:.2f}s recall={r['recall']:.4f} cand={r['total_candidates']} cand_t={r['cand_time']:.3f} val_t={r['val_time']:.3f} "
                  f"p50/p99 step={r['step_time_p50']:.5f}/{r['step_time_p99']:.5f}  {over}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"dataset": label, "config": vars(args), "tuned_source": src, "tuned": tuned, "variants": results}, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
