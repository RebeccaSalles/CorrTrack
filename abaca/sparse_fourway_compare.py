"""Four-way comparison on SPARSE synthetic data (density-targeted generator), to see
CorrTrack in the regime it's actually built for -- the real fr_air_temperature dataset
turned out to be unusually DENSE (~17.6% of tested pairs correlated), which is a known
unfavorable regime for LSH-based pruning. Same m=121 series count as the real-data runs
for comparability; target_density is a real, controllable parameter here (~2%, well
within this project's own Sobol sweep range of 0.001-0.10, corr_prop=0.05 anchor).
"""
import os
import sys
import time

REPO = os.environ.get("REPO_DIR", os.path.expanduser("~/corrtrack_release_dev"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
from synth_corr_gen import make_density_targeted_dataset
from library_corrtrack_parallel import (
    run_and_log_bruteforce,
    run_and_log_corrtrack,
    CorrTrack,
)

WINDOW_SIZE, WINDOW_STEP, N_LAGS, CORR_THRESHOLD = 168, 12, 168, 0.7
TARGET_DENSITY = float(os.environ.get("SPARSE_TARGET_DENSITY", "0.02"))
M = 121
N_OBS = 6144

print(f"generating density-targeted dataset: m={M} n={N_OBS} "
      f"target_density={TARGET_DENSITY}", flush=True)
t0 = time.perf_counter()
gen = make_density_targeted_dataset(
    M, N_OBS, target_density=TARGET_DENSITY, corr_threshold=CORR_THRESHOLD,
    window_size=WINDOW_SIZE, window_step=WINDOW_STEP, n_lags=N_LAGS,
    n_eval_steps=400, base_proc={"type": "ar1", "phi": 0.6}, corr_sign="both",
    n_epochs=6, duty=1.0, lag_band=4, seed=7, verify_bf=False,
)
print(f"generated in {time.perf_counter()-t0:.1f}s -- "
      f"analytic_density={gen['analytic_density']:.5f} "
      f"gt_pairs={len(gen['gt_rows'])}", flush=True)

test_data = np.ascontiguousarray(gen["data"].T)   # (n, m+1) -> (m+1, n)
ids_n_var = list(gen["ids"])
print(f"dataset: n_series={len(ids_n_var)} n_obs={test_data.shape[1]}", flush=True)

base_config = dict(
    window_size=WINDOW_SIZE, window_step=WINDOW_STEP, basic_window=None, n_lags=N_LAGS,
    corr_threshold=CORR_THRESHOLD, neg_corr=True, exec="sequential",
    parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
    max_workers=0, monitor=True, track_min_dist=True,
    artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
    save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
    validation_metric="pearson",
)

results = {}

for mode in ("bruteforce", "bf_incremental", "filcorr"):
    cfg = dict(base_config, baseline_mode=mode, filcorr_fs=0.0, filcorr_ft=0.5,
               filcorr_sampling_rate=1.0)
    t0 = time.perf_counter()
    record, _, corr_flags = run_and_log_bruteforce(
        "sparse121", test_data, ids_n_var, cfg, f"/tmp/sparse_{mode}_run.csv",
        metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False,
    )
    wall = time.perf_counter() - t0
    results[mode] = dict(record)
    results[mode]["_wall"] = wall
    results[mode]["_flags"] = corr_flags
    print(f"{mode:12s} done: wall={wall:.2f}s correlated={record['correlated']} "
          f"cand={record['cand_time']:.3f} val={record['val_time']:.3f} "
          f"monit={record['monit_time']:.3f}", flush=True)

n_vectors = int(os.environ.get("CORRTRACK_N_VECTORS_OVERRIDE", "64"))
run_params = dict(n_vectors=n_vectors, seed=2468, seed_toggle=1357, preprocess=False,
                   candidate_backend="lsh_sign_dot", candidate_lsh_target_occupancy=3.0)
print(f"corrtrack: n_vectors={n_vectors} candidate_backend=lsh_sign_dot occupancy=3.0", flush=True)
t0 = time.perf_counter()
record_ct, _, corr_flags_ct = run_and_log_corrtrack(
    "sparse121", test_data, ids_n_var, base_config, run_params, "/tmp/sparse_corrtrack_run.csv",
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

for mode in ("bf_incremental", "filcorr", "corrtrack"):
    metrics = CorrTrack.compute_metrics_bf(
        results[mode]["_flags"], bf_flags, windows=True, total_pairs_bf=bf_total_candidates,
    )
    results[mode]["_precision"] = metrics.get("precision")
    results[mode]["_recall"] = metrics.get("recall")
    results[mode]["_f1"] = metrics.get("f1_score")
    print(f"{mode:12s} vs bruteforce: precision={metrics.get('precision'):.4f} "
          f"recall={metrics.get('recall'):.4f} f1={metrics.get('f1_score'):.4f}", flush=True)

print("\n=== SUMMARY (sparse data, target_density={:.4f}) ===".format(TARGET_DENSITY))
header = (f"{'method':12s} {'wall_s':>8s} {'cand_s':>8s} {'val_s':>8s} {'monit_s':>8s} "
          f"{'correlated':>11s} {'precision':>9s} {'recall':>7s} {'speedup':>8s}")
print(header)
bf_wall = results["bruteforce"]["_wall"]
for mode in ("bruteforce", "bf_incremental", "filcorr", "corrtrack"):
    r = results[mode]
    prec = r.get("_precision")
    rec = r.get("_recall")
    prec_s = f"{prec:.4f}" if prec is not None else "   -   "
    rec_s = f"{rec:.4f}" if rec is not None else "  -  "
    speedup = bf_wall / r["_wall"] if r["_wall"] else float("nan")
    print(f"{mode:12s} {r['_wall']:8.2f} {r['cand_time']:8.3f} {r['val_time']:8.3f} "
          f"{r['monit_time']:8.3f} {r['correlated']:11d} {prec_s:>9s} {rec_s:>7s} {speedup:8.2f}x")
