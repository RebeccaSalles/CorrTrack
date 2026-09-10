"""Cluster/factor-bucketing feasibility oracle (2026-07-13 investigation,
following the temporal-continuity dead end and InstinctIndex's real-data
negative result).

Central idea being tested: instead of comparing every pair (i,j) directly
(the O(m^2) axis every backend tried so far still pays), maintain a SMALL
number of cluster centroids (K_factors << m) in the EXISTING sketch space,
assign each series to its nearest centroid (O(m*K_factors), not O(m^2)), and
only generate candidate pairs within (or between nearby) clusters. This does
NOT change the incremental sketch algorithm at all -- clustering operates on
sketch vectors already produced by the existing pipeline, same "preprocessing
feeds the existing pipeline" contract as everything else in this project.

A quick version of this idea ("IVF-style k-means partitioning") was tried on
2026-07-10 on the adversarial-transient SYNTHETIC benchmark and found "real
signal, not yet a win" (best config: ~27% better touch-rate than baseline,
but recall_tp=0.928, short of the 0.95 target) -- never built into a real
backend. This script re-examines the idea on REAL data at real scale, and
adds the one dimension that was never tested: TEMPORAL STABILITY. Clusters
can't be free -- correlation structure can drift (nonstationarity is an
explicit requirement here), so re-clustering has a real cost, and clusters
held stale for many steps will progressively miss newly-forming
correlations. This measures that trade-off directly, not just a single
static snapshot's touch-rate/recall.

Measures, at a sequence of real steps:
  1. FRESH-clustering touch-rate/recall (best case: re-cluster every step)
     across several K (number of clusters).
  2. STALE-clustering touch-rate/recall (cluster ONCE at step 0, reuse
     unchanged for many subsequent steps) -- the gap between this and (1)
     is the real cost of nonstationarity for this idea.
  3. Cluster-assignment churn: what fraction of series change their nearest
     centroid between the stale (step-0) assignment and a fresh
     reclustering at the current step -- a direct proxy for how much
     "adaptation work" a real re-clustering policy would need to do.
  4. A widened variant (same-cluster OR one of the K_probe nearest
     centroids) to see if a cheap widening recovers missed recall.

All recall/touch-rate numbers use exact Pearson correlation as ground
truth (vectorized/BLAS, same identity as _fast_corr_and_dist), computed
independently of the clustering itself, restricted to lag=0 (synchronous)
pairs for this first pass -- extending to lagged pairs is a natural
follow-up, not done here to keep the oracle's first answer simple.

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/diag_cluster_bucketing_oracle.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_fr_air_temperature_121_1.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --corr-threshold 0.7 --neg-corr \\
        --n-vectors 64 --n-steps 80 --k-values 4,8,16,32 --stale-offsets 5,20,50 \\
        --result-folder tmp_artifacts/cluster_bucketing_diag
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack
from diag_gamma_reliability import _load_module, _resolve_cfg_value, SketchCache


def pearson_matrix(X, Y):
    """Vectorized (BLAS-backed) Pearson correlation between every row of X
    and every row of Y -- same identity as _fast_corr_and_dist, batched over
    all m^2 pairs via one matmul instead of per-pair loops."""
    w = X.shape[1]
    sum_x = X.sum(axis=1)
    sum_x2 = (X * X).sum(axis=1)
    sum_y = Y.sum(axis=1)
    sum_y2 = (Y * Y).sum(axis=1)
    sum_xy = X @ Y.T
    cov = w * sum_xy - np.outer(sum_x, sum_y)
    var_x = np.maximum(w * sum_x2 - sum_x * sum_x, 0.0)
    var_y = np.maximum(w * sum_y2 - sum_y * sum_y, 0.0)
    denom = np.sqrt(np.outer(var_x, var_y))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 1e-12, cov / np.where(denom > 1e-12, denom, 1.0), np.nan)
    return corr


def cluster_touch_and_recall(labels_i, labels_j, corr, corr_threshold, mask, k_probe_pairs=None):
    """labels_i/labels_j: which centroid each series is assigned to (may be
    the SAME assignment array for both sides, since we're testing lag=0
    same-population pairs). k_probe_pairs: optional set of (cluster_a,
    cluster_b) pairs ALSO considered "touched" (a widened bucketing probing
    nearby clusters too, not just the exact same one)."""
    same_cluster = labels_i[:, None] == labels_j[None, :]
    if k_probe_pairs:
        widened = np.zeros_like(same_cluster)
        for a, b in k_probe_pairs:
            widened |= (labels_i[:, None] == a) & (labels_j[None, :] == b)
            widened |= (labels_i[:, None] == b) & (labels_j[None, :] == a)
        touched = same_cluster | widened
    else:
        touched = same_cluster
    touched &= mask
    true_pos = mask & (np.abs(corr) >= corr_threshold - 1e-9) & np.isfinite(corr)
    n_true = int(true_pos.sum())
    n_found = int((touched & true_pos).sum())
    n_total = int(mask.sum())
    n_touched = int(touched.sum())
    recall = n_found / max(n_true, 1)
    touch_rate = n_touched / max(n_total, 1)
    return touch_rate, recall, n_true


def nearest_cluster_pairs(centroids, k_probe):
    """For each cluster, its k_probe nearest OTHER clusters (by centroid
    L2 distance) -- used for the widened-bucketing variant."""
    K = centroids.shape[0]
    dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    pairs = set()
    for a in range(K):
        nearest = np.argsort(dists[a])[:k_probe]
        for b in nearest:
            pairs.add((min(a, int(b)), max(a, int(b))))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--exec-param-config", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--corr-threshold", type=float, default=None)
    parser.add_argument("--neg-corr", dest="neg_corr", action="store_true", default=None)
    parser.add_argument("--no-neg-corr", dest="neg_corr", action="store_false")
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--n-vectors", type=int, default=64)
    parser.add_argument("--sketch-norm", type=str, default="mean_l2")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--seed-toggle", type=int, default=1357)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--k-values", type=str, default="4,8,16,32")
    parser.add_argument("--stale-offsets", type=str, default="5,20,50",
                         help="how many steps after the initial clustering to re-check it, unchanged")
    parser.add_argument("--k-probe", type=int, default=1, help="nearest-cluster widening for the widened variant")
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/cluster_bucketing_diag"))
    args = parser.parse_args()
    args.result_folder.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")

    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    corr_threshold = _resolve_cfg_value(args.corr_threshold, exec_cfg, "CORR_THRESHOLD", 0.7)
    neg_corr = _resolve_cfg_value(args.neg_corr, exec_cfg, "NEG_CORR", True)

    cps._apply_dataset_config(dataset_cfg)
    country, var, data, ids = next(cps.iter_datasets())
    n_var = cps._get_cfg_attr(dataset_cfg, "N_VARS", "N_SERIES")
    n_year = cps._get_cfg_attr(dataset_cfg, "N_YEARS", "N_OBS")
    n_var = n_var[0] if isinstance(n_var, (list, tuple)) else n_var
    n_year = n_year[0] if isinstance(n_year, (list, tuple)) else n_year
    train_data, proxy_ids = cps.prepare_training_data(data, ids, n_year, n_var, args.train_ratio)
    proxy_ids = [str(x) for x in proxy_ids]
    m = train_data.shape[0] - 1
    length_data = train_data.shape[1]
    raw_values = np.asarray(train_data[1:1 + m, :], dtype=np.float64)

    print(f"PARAMS: dataset={country}/{var} m={m} window={window_size}/{window_step} "
          f"corr_threshold={corr_threshold} n_vectors={args.n_vectors} n_steps={args.n_steps}")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, args.n_vectors,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )

    max_start = length_data - window_size
    all_starts = list(range(0, max_start + 1, window_step))[:args.n_steps]
    k_values = [int(x) for x in args.k_values.split(",") if x]
    stale_offsets = sorted(int(x) for x in args.stale_offsets.split(",") if x)

    def get_sketch_matrix(start):
        _raw, norm_by_sid = sketch_cache.get(start)
        V = np.zeros((m, args.n_vectors), dtype=np.float64)
        valid = np.zeros(m, dtype=bool)
        for s_idx, sid in enumerate(proxy_ids):
            v = norm_by_sid.get(sid)
            if v is not None:
                V[s_idx] = v
                valid[s_idx] = True
        return V, valid

    iu_mask_base = np.triu(np.ones((m, m), dtype=bool), k=1)

    # step 0: initial ("stale-origin") clustering per K, kept fixed
    V0, valid0 = get_sketch_matrix(all_starts[0])
    raw0 = raw_values[:, all_starts[0]:all_starts[0] + window_size]
    corr0 = pearson_matrix(raw0, raw0)
    mask0 = iu_mask_base & valid0[:, None] & valid0[None, :]

    stale_models = {}
    for K in k_values:
        km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(V0)
        stale_models[K] = km
        touch, recall, n_true = cluster_touch_and_recall(km.labels_, km.labels_, corr0, corr_threshold, mask0)
        print(f"[step 0, K={K}] fresh==stale here by definition: touch_rate={touch:.4f} recall={recall:.4f} "
              f"n_true_pairs={n_true}")

    rows = []
    for k, start in enumerate(all_starts):
        if k == 0:
            continue
        V, valid = get_sketch_matrix(start)
        raw = raw_values[:, start:start + window_size]
        corr = pearson_matrix(raw, raw)
        mask = iu_mask_base & valid[:, None] & valid[None, :]

        for K in k_values:
            # FRESH: re-cluster at this exact step (best case, ignoring re-clustering cost)
            km_fresh = KMeans(n_clusters=K, n_init=4, random_state=0).fit(V)
            fresh_touch, fresh_recall, n_true = cluster_touch_and_recall(
                km_fresh.labels_, km_fresh.labels_, corr, corr_threshold, mask)
            probe_pairs = nearest_cluster_pairs(km_fresh.cluster_centers_, args.k_probe)
            fresh_touch_w, fresh_recall_w, _ = cluster_touch_and_recall(
                km_fresh.labels_, km_fresh.labels_, corr, corr_threshold, mask, k_probe_pairs=probe_pairs)

            row = dict(step=k, start=start, K=K, n_true_pairs=n_true,
                       fresh_touch_rate=fresh_touch, fresh_recall=fresh_recall,
                       fresh_touch_rate_widened=fresh_touch_w, fresh_recall_widened=fresh_recall_w)

            if k in stale_offsets:
                km_stale = stale_models[K]
                stale_labels = km_stale.predict(V)  # assign CURRENT sketches to the OLD (step-0) centroids
                stale_touch, stale_recall, _ = cluster_touch_and_recall(
                    stale_labels, stale_labels, corr, corr_threshold, mask)
                churn = float(np.mean(stale_labels != km_fresh.labels_))
                row.update(stale_touch_rate=stale_touch, stale_recall=stale_recall, assignment_churn=churn)

            rows.append(row)

        if k % 10 == 0:
            print(f"  step {k}/{len(all_starts)-1}", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== Fresh-clustering (best case, re-cluster every step) summary, by K ===")
    summary_fresh = df.groupby("K")[["fresh_touch_rate", "fresh_recall", "fresh_touch_rate_widened", "fresh_recall_widened"]].mean()
    print(summary_fresh.to_string())

    print(f"\n=== Stale-clustering (cluster once at step 0, reuse unchanged) at offsets {stale_offsets} ===")
    stale_rows = df.dropna(subset=["stale_touch_rate"]) if "stale_touch_rate" in df.columns else df.iloc[0:0]
    if len(stale_rows):
        stale_rows = stale_rows.copy()
        # efficiency: recall gained per unit of touch-rate paid -- a stale
        # clustering that touches MORE pairs will show inflated raw recall
        # even if it has become less discriminating; this normalizes for that.
        stale_rows["stale_efficiency"] = stale_rows["stale_recall"] / stale_rows["stale_touch_rate"].clip(lower=1e-9)
        summary_stale = stale_rows.groupby(["K", "step"])[
            ["stale_touch_rate", "stale_recall", "stale_efficiency", "assignment_churn"]].mean()
        print(summary_stale.to_string())
        print("\n(compare stale_touch_rate/stale_recall against the FRESH WIDENED row for the same K above --")
        print(" that's the realistic 'always re-cluster' operating point stale clustering is competing against)")
    else:
        print("(no stale-offset rows collected -- check --stale-offsets against --n-steps)")

    out = args.result_folder / "cluster_bucketing_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
