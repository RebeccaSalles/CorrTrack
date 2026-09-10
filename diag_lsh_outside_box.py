"""Diagnostic (2026-07-10, candidate_selector thread, "think outside the
box" follow-up): cheap, Python-level checks for two structurally different
ideas proposed for lsh_sign_dot, before writing any real backend code:

  1. IVF-style coarse partitioning: cluster the alive population (k-means)
     and only search within a query's own cluster (+ optionally its
     nearest neighbor clusters). Does this preserve recall of true
     survivor pairs while meaningfully shrinking the searched population
     per query? Tested at several K and multi-probe depths.

  2. Same-series lag-window redundancy: are a series' own L concurrently-
     alive lagged copies as self-similar as hypothesized? If so, that is
     real, exploitable redundancy inflating alive_count without adding
     distinct information.

Uses the same real-data, bounded-window, SketchCache-based methodology as
diag_lsh_rescue_ideas.py (mean_l2 only -- see that script's docstring for
why "z" cannot be faithfully tested this way). Ground truth (is_tp) is
computed directly via _fast_corr_and_dist on raw window data, independent
of any candidate-search backend.

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/diag_lsh_outside_box.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 --neg-corr \\
        --n-vectors 64 --gamma 0.55 --n-steps 200 \\
        --result-folder tmp_artifacts/lsh_outside_box_diag
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize, _fast_corr_and_dist
from diag_gamma_reliability import _load_module, _resolve_cfg_value, SketchCache


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--exec-param-config", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--n-lags", type=int, default=None)
    parser.add_argument("--corr-threshold", type=float, default=None)
    parser.add_argument("--neg-corr", dest="neg_corr", action="store_true", default=None)
    parser.add_argument("--no-neg-corr", dest="neg_corr", action="store_false")
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--n-vectors", type=int, default=64)
    parser.add_argument("--sketch-norm", type=str, default="mean_l2")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--seed-toggle", type=int, default=1357)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--k-values", type=str, default="4,8,16,32")
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/lsh_outside_box_diag"))
    args = parser.parse_args()
    args.result_folder.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")

    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    n_lags = _resolve_cfg_value(args.n_lags, exec_cfg, "N_LAGS", 168)
    corr_threshold = _resolve_cfg_value(args.corr_threshold, exec_cfg, "CORR_THRESHOLD", 0.7)
    neg_corr = _resolve_cfg_value(args.neg_corr, exec_cfg, "NEG_CORR", True)

    print(f"PARAMS: window_size={window_size} window_step={window_step} n_lags={n_lags} "
          f"corr_threshold={corr_threshold} neg_corr={neg_corr} n_vectors={args.n_vectors} "
          f"gamma={args.gamma} n_steps={args.n_steps} k_values={k_values} sketch_norm={args.sketch_norm}")

    cps._apply_dataset_config(dataset_cfg)
    country, var, data, ids = next(cps.iter_datasets())
    n_var = cps._get_cfg_attr(dataset_cfg, "N_VARS", "N_SERIES")
    n_year = cps._get_cfg_attr(dataset_cfg, "N_YEARS", "N_OBS")
    n_var = n_var[0] if isinstance(n_var, (list, tuple)) else n_var
    n_year = n_year[0] if isinstance(n_year, (list, tuple)) else n_year
    train_data, proxy_ids = cps.prepare_training_data(data, ids, n_year, n_var, args.train_ratio)
    proxy_ids = [str(x) for x in proxy_ids]
    n_series = train_data.shape[0] - 1
    length_data = train_data.shape[1]
    print(f"Dataset loaded: n_series={n_series} length_data={length_data}")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, args.n_vectors,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )

    max_start = length_data - window_size
    all_starts = list(range(0, max_start + 1, window_step))[:args.n_steps]
    values = np.asarray(train_data[1:1 + n_series, :], dtype=np.float64)
    window_cache = {}
    valid_cache = {}

    def get_window(series_idx, start_idx):
        key = (series_idx, start_idx)
        cached = window_cache.get(key)
        if cached is None:
            cached = values[series_idx, start_idx:start_idx + window_size]
            window_cache[key] = cached
        return cached

    def is_valid(series_idx, start_idx):
        key = (series_idx, start_idx)
        if key not in valid_cache:
            valid_cache[key] = CorrTrack_optimize._proxy_window_is_valid(get_window(series_idx, start_idx))
        return valid_cache[key]

    # ---- Part 1: same-series lag-window redundancy (cheap, no clustering needed) ----
    print("\n=== Part 1: same-series concurrently-alive lag-window self-similarity ===")
    same_series_cos = []
    for cur_start in all_starts[:60]:
        min_lag_start = max(0, cur_start - n_lags)
        lag_starts = sorted(set(s for s in range(cur_start, min_lag_start - 1, -window_step) if 0 <= s <= max_start))
        if len(lag_starts) < 2:
            continue
        norm_by_start = {s: sketch_cache.get(s)[1] for s in lag_starts}
        for s_idx in range(n_series):
            vecs = []
            for s in lag_starts:
                if not is_valid(s_idx, s):
                    continue
                v = norm_by_start[s].get(proxy_ids[s_idx])
                if v is not None:
                    vecs.append(np.asarray(v))
            if len(vecs) < 2:
                continue
            mat = np.stack(vecs)
            sims = mat @ mat.T
            iu = np.triu_indices(len(vecs), k=1)
            same_series_cos.extend(sims[iu].tolist())
    same_series_cos = np.array(same_series_cos)
    print(f"n pairs (same series, different concurrently-alive lags): {len(same_series_cos)}")
    if len(same_series_cos):
        print(f"cosine similarity: mean={same_series_cos.mean():.4f} median={np.median(same_series_cos):.4f} "
              f"min={same_series_cos.min():.4f} p10={np.percentile(same_series_cos,10):.4f}")
        print(f"fraction with cosine >= 0.9: {np.mean(same_series_cos >= 0.9):.4f}")
        print(f"fraction with cosine >= 0.55 (gamma): {np.mean(same_series_cos >= args.gamma):.4f}")

    # ---- Part 2: IVF-style coarse clustering -- recall vs touch-fraction tradeoff ----
    print("\n=== Part 2: k-means coarse partitioning ===")
    n_baseline = 0
    n_tp = 0
    n_surv = 0
    stats = {}  # (K, probe) -> [retrieved_tp, retrieved_surv, touched_total]
    probes = (1, 2)
    for K in k_values:
        for p in probes:
            if p > K:
                continue
            stats[(K, p)] = [0, 0, 0]

    for step_id, cur_start in enumerate(all_starts):
        min_lag_start = max(0, cur_start - n_lags)
        lag_starts = sorted(set(s for s in range(cur_start, min_lag_start - 1, -window_step) if 0 <= s <= max_start))
        if not lag_starts or cur_start not in lag_starts:
            continue

        norm_by_start = {s: sketch_cache.get(s)[1] for s in lag_starts}
        keys_pool, vec_pool = [], []
        for s in lag_starts:
            for s_idx in range(n_series):
                if not is_valid(s_idx, s):
                    continue
                v = norm_by_start[s].get(proxy_ids[s_idx])
                if v is None:
                    continue
                keys_pool.append((s_idx, s))
                vec_pool.append(np.asarray(v, dtype=np.float64))
        if len(keys_pool) < max(k_values) * 2:
            continue
        vec_matrix = np.stack(vec_pool)
        key_index = {k: i for i, k in enumerate(keys_pool)}
        n_alive = len(keys_pool)

        cluster_labels_by_K = {}
        cluster_centers_by_K = {}
        for K in k_values:
            km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(vec_matrix)
            cluster_labels_by_K[K] = km.labels_
            cluster_centers_by_K[K] = km.cluster_centers_

        for s_current in range(n_series):
            q_key = (s_current, cur_start)
            if q_key not in key_index:
                continue
            qi = key_index[q_key]
            q_vec = vec_matrix[qi]

            # (2026-07-10) neg_corr=True means a negatively-correlated
            # candidate sits near the ANTIPODAL point (-q_vec) in cosine
            # space, not near q_vec itself -- k-means clusters by raw
            # Euclidean proximity, so without also probing -q_vec's
            # nearest clusters, every negatively-correlated true pair
            # would be structurally unreachable regardless of K/probe,
            # mirroring the "own key + complement" doubling
            # SignLSHBandIndex already does for the same reason.
            allowed_by_Kp = {}
            for K in k_values:
                labels = cluster_labels_by_K[K]
                centers = cluster_centers_by_K[K]
                q_label = labels[qi]
                pos_order = np.argsort(np.linalg.norm(centers - q_vec, axis=1))
                neg_order = np.argsort(np.linalg.norm(centers + q_vec, axis=1))
                for p in probes:
                    if (K, p) not in stats:
                        continue
                    allowed = set(pos_order[:p].tolist()) | set(neg_order[:p].tolist())
                    allowed.add(int(q_label))
                    allowed_by_Kp[(K, p)] = allowed
                    touched_mask = np.isin(labels, list(allowed))
                    touched_mask[qi] = False
                    stats[(K, p)][2] += int(touched_mask.sum())

            for other_key, oi in key_index.items():
                if other_key == q_key:
                    continue
                s_other, lag_start = other_key
                if lag_start == cur_start and s_other <= s_current:
                    continue
                x = get_window(s_current, cur_start)
                y = get_window(s_other, lag_start)
                corr, _dist = _fast_corr_and_dist(x, y)
                if not np.isfinite(corr):
                    continue
                n_baseline += 1
                is_tp = bool(abs(corr) >= corr_threshold)
                if is_tp:
                    n_tp += 1
                score = float(np.dot(q_vec, vec_matrix[oi]))
                is_surv = abs(score) >= args.gamma
                if is_surv:
                    n_surv += 1
                for K in k_values:
                    labels = cluster_labels_by_K[K]
                    for p in probes:
                        if (K, p) not in stats:
                            continue
                        if labels[oi] in allowed_by_Kp[(K, p)]:
                            if is_tp:
                                stats[(K, p)][0] += 1
                            if is_surv:
                                stats[(K, p)][1] += 1

        if step_id % 20 == 0:
            print(f"  progress: step {step_id}/{len(all_starts)}, {n_baseline} pairs so far", flush=True)

    print(f"\nBaseline: {n_baseline} pairs, {n_tp} true positives, {n_surv} sketch survivors.")
    rows = []
    for (K, p), (ret_tp, ret_surv, touched) in stats.items():
        rows.append(dict(
            K=K, probe=p, n_baseline=n_baseline, n_tp=n_tp, n_surv=n_surv,
            recall_tp=ret_tp / n_tp if n_tp else float("nan"),
            recall_surv=ret_surv / n_surv if n_surv else float("nan"),
            avg_touched_per_query=touched / n_baseline * (n_baseline / max(1, n_tp)) if False else touched,
            touch_rate=touched / (n_baseline) if n_baseline else float("nan"),
        ))
    df = pd.DataFrame(rows).sort_values(["K", "probe"])
    print("\n" + df.to_string(index=False))
    out_path = args.result_folder / "ivf_partition_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
