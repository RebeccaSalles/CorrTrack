"""Four-way comparison: bruteforce (plain) vs exact_stomp vs filcorr vs CorrTrack (main,
LSH-accelerated), on the real fr_air_temperature_121_1 dataset. Abaca version: no memory
capping/subprocess isolation needed (192GB node, vs the 7.8GB WSL box this was developed
against) -- single process, in-memory metrics via CorrTrack.compute_metrics_bf.
"""
import os
import sys
import time
from pathlib import Path

REPO = os.environ.get("REPO_DIR", os.path.expanduser("~/corrtrack_release_dev"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import corrtrack_run_bruteforce as bfmod
from library_corrtrack_parallel import (
    run_and_log_bruteforce,
    run_and_log_corrtrack,
    CorrTrack,
)

cfg_dataset = bfmod._load_dataset_config(Path("experiment_dataset_fr_air_temperature_121_1.py"))
bfmod._apply_dataset_config(cfg_dataset)

WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD = 168, 12, None, 168, 0.7
NEG_CORR = True

country, var, data, ids = next(bfmod.iter_datasets())
n_year, n_var = bfmod.N_YEARS[0], bfmod.N_VARS[0]
test_data, ids_n_var = bfmod.prepare_test_data(data, ids, n_year, n_var, bfmod.TRAIN_RATIO,
                                                tuning_mode="sampling")
print(f"dataset: n_series={len(ids_n_var)} n_obs={test_data.shape[1]}", flush=True)

base_config = dict(
    window_size=WINDOW_SIZE, window_step=WINDOW_STEP, basic_window=BASIC_WINDOW, n_lags=N_LAGS,
    corr_threshold=CORR_THRESHOLD, neg_corr=NEG_CORR, exec="sequential",
    parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
    max_workers=0, monitor=True, track_min_dist=True,
    artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
    save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
    validation_metric="pearson",
)

results = {}

for mode in ("bruteforce", "exact_stomp", "filcorr"):
    cfg = dict(base_config, baseline_mode=mode, filcorr_fs=0.0, filcorr_ft=0.5,
               filcorr_sampling_rate=1.0)
    t0 = time.perf_counter()
    record, _, corr_flags = run_and_log_bruteforce(
        "fr121", test_data, ids_n_var, cfg, f"/tmp/{mode}_run.csv",
        metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False,
    )
    wall = time.perf_counter() - t0
    results[mode] = dict(record)
    results[mode]["_wall"] = wall
    results[mode]["_flags"] = corr_flags
    print(f"{mode:12s} done: wall={wall:.2f}s correlated={record['correlated']} "
          f"cand={record['cand_time']:.3f} val={record['val_time']:.3f} "
          f"monit={record['monit_time']:.3f}", flush=True)

import json as _json

_best_params_path = os.path.join(
    "correlation", "asos_exp/tests", "fr_air_temperature_121_1",
    bfmod.config_folder(), "optim", "best_params_corrtrack.json",
)
if os.path.exists(_best_params_path):
    with open(_best_params_path) as _f:
        run_params = _json.load(_f)
    print(f"corrtrack: using HYPEROPT-TUNED params from {_best_params_path}", flush=True)
    print(f"  n_vectors={run_params.get('n_vectors')} "
          f"candidate_backend={run_params.get('candidate_backend')} "
          f"candidate_lsh_target_occupancy={run_params.get('candidate_lsh_target_occupancy')}",
          flush=True)
else:
    run_params = dict(n_vectors=32, seed=2468, seed_toggle=1357, preprocess=False)
    print(f"corrtrack: {_best_params_path} not found -- using UNTUNED defaults", flush=True)

_n_vectors_override = os.environ.get("CORRTRACK_N_VECTORS_OVERRIDE")
if _n_vectors_override:
    run_params = dict(run_params, n_vectors=int(_n_vectors_override))
    print(f"corrtrack: n_vectors OVERRIDDEN to {run_params['n_vectors']} "
          f"(everything else unchanged from above)", flush=True)

t0 = time.perf_counter()
record_ct, _, corr_flags_ct = run_and_log_corrtrack(
    "fr121", test_data, ids_n_var, base_config, run_params, "/tmp/corrtrack_run.csv",
    metadata={"nodes": 0, "alg": "corrtrack"}, recall_by_window=True, corr_val=True,
    monitor=True, verbose=False, testing=False,
)
wall = time.perf_counter() - t0
results["corrtrack"] = dict(record_ct)
results["corrtrack"]["_wall"] = wall
results["corrtrack"]["_flags"] = corr_flags_ct
print(f"corrtrack    done: wall={wall:.2f}s correlated={record_ct['correlated']} "
      f"sk={record_ct.get('sk_time')} cand={record_ct['cand_time']:.3f} "
      f"val={record_ct['val_time']:.3f} monit={record_ct['monit_time']:.3f}", flush=True)

bf_flags = results["bruteforce"]["_flags"]
bf_total_candidates = results["bruteforce"].get("total_candidates") or results["bruteforce"].get("tested")

for mode in ("exact_stomp", "filcorr", "corrtrack"):
    metrics = CorrTrack.compute_metrics_bf(
        results[mode]["_flags"], bf_flags, windows=True, total_pairs_bf=bf_total_candidates,
    )
    results[mode]["_precision"] = metrics.get("precision")
    results[mode]["_recall"] = metrics.get("recall")
    results[mode]["_f1"] = metrics.get("f1_score")
    print(f"{mode:12s} vs bruteforce: precision={metrics.get('precision'):.4f} "
          f"recall={metrics.get('recall'):.4f} f1={metrics.get('f1_score'):.4f}", flush=True)

print("\n=== SUMMARY ===")
header = (f"{'method':12s} {'wall_s':>8s} {'cand_s':>8s} {'val_s':>8s} {'monit_s':>8s} "
          f"{'correlated':>11s} {'precision':>9s} {'recall':>7s} {'speedup':>8s}")
print(header)
bf_wall = results["bruteforce"]["_wall"]
for mode in ("bruteforce", "exact_stomp", "filcorr", "corrtrack"):
    r = results[mode]
    prec = r.get("_precision")
    rec = r.get("_recall")
    prec_s = f"{prec:.4f}" if prec is not None else "   -   "
    rec_s = f"{rec:.4f}" if rec is not None else "  -  "
    speedup = bf_wall / r["_wall"] if r["_wall"] else float("nan")
    print(f"{mode:12s} {r['_wall']:8.2f} {r['cand_time']:8.3f} {r['val_time']:8.3f} "
          f"{r['monit_time']:8.3f} {r['correlated']:11d} {prec_s:>9s} {rec_s:>7s} {speedup:8.2f}x")
