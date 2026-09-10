"""Phase 6 diagnostic (spike, not a real index yet): does a top-k dominant-
coordinate inverted index avoid most full sketch-dot checks while still
retrieving the pairs that matter?

Per the approved research plan's priority order, Phases 2-5 (reliability/
adaptive gamma, subsketch/multi-sketch refinement, calibration) were found
in Phase 1 to have weak, unstable signal at the real n_vectors=64 operating
point (see docs/implementation_log.md). This script tests Phase 6's core
idea directly, as a cheap offline SIMULATION (posting lists built and
queried in plain Python/numpy over real sketch vectors) rather than a full
Cython implementation -- the point is to learn whether the idea has any
teeth before investing in a real index, per this session's own "spike
first" recommendation.

For each anchor (same causal anchor/lag-start design as
diag_gamma_reliability.py, reusing its SketchCache/dataset-loading code
directly): build an inverted index over the historical windows using each
window's top-k largest-magnitude coordinates (keyed by (dim, sign)), then
for each query window look up postings for its own top-k coordinates,
accumulate votes, and see which candidates the index would flag for a full
dot-product computation under votes >= v_min.

Reports, per (k, v_min):
    n_full_dot_checks_baseline   (brute force -- one per candidate pair)
    n_full_dot_checks_topk       (pairs the index would compute a full dot for)
    reduction_in_full_dot_checks
    recall_of_exact_true_positives  (fraction of |exact_corr|>=tau pairs retrieved)
    recall_of_sketch_survivors      (fraction of abs(sketch_score)>=gamma pairs retrieved)

Usage:
    PYTHONPATH="$(pwd)" python3 corrtrack_release_dev/diag_topk_inverted_index.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 \\
        --neg-corr --train-ratio 0.33 --n-vectors 64 --gamma-baseline 0.55 \\
        --max-pair-rows 550000 \\
        --result-folder tmp_artifacts/topk_inverted_index_diag
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize, _fast_corr_and_dist
from diag_gamma_reliability import _load_module, _resolve_cfg_value, SketchCache


def build_postings(vectors_by_key, k):
    postings = defaultdict(list)
    topk_by_key = {}
    for key, v in vectors_by_key.items():
        idx = np.argpartition(-np.abs(v), min(k, len(v) - 1))[:k]
        topk_by_key[key] = idx
        for i in idx:
            sign = 1 if v[i] >= 0 else -1
            postings[(int(i), sign)].append(key)
    return postings, topk_by_key


def query_candidates(q_vec, q_topk_idx, postings, signed_abs):
    """Returns dict: candidate_key -> (votes, score_partial)."""
    votes = defaultdict(int)
    score_partial = defaultdict(float)
    for i in q_topk_idx:
        qi = q_vec[i]
        sign_q = 1 if qi >= 0 else -1
        for key in postings.get((int(i), sign_q), ()):
            votes[key] += 1
            score_partial[key] += abs(qi)  # |q_i| is a cheap per-coordinate contribution proxy
        if signed_abs:
            for key in postings.get((int(i), -sign_q), ()):
                votes[key] += 1
                score_partial[key] += abs(qi)
    return votes, score_partial


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
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--n-vectors", type=int, default=64)
    parser.add_argument("--sketch-norm", type=str, default="mean_l2")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--seed-toggle", type=int, default=1357)
    parser.add_argument("--gamma-baseline", type=float, default=0.55)
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32])
    parser.add_argument("--v-min-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-pair-rows", type=int, default=550000)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/topk_inverted_index_diag"))
    args = parser.parse_args()

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")

    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    n_lags = _resolve_cfg_value(args.n_lags, exec_cfg, "N_LAGS", 168)
    corr_threshold = _resolve_cfg_value(args.corr_threshold, exec_cfg, "CORR_THRESHOLD", 0.7)
    neg_corr = _resolve_cfg_value(args.neg_corr, exec_cfg, "NEG_CORR", True)
    train_ratio = _resolve_cfg_value(args.train_ratio, exec_cfg, "TRAIN_RATIO", 0.3)
    anchor_count_requested = int(getattr(exec_cfg, "OPTIM_PROXY_ANCHOR_COUNT", 128))
    random_seed = int(getattr(exec_cfg, "OPTIM_PROXY_RANDOM_SEED", 2468))
    max_pair_rows = int(args.max_pair_rows)

    print(f"PARAMS: n_vectors={args.n_vectors} k_values={args.k_values} v_min_values={args.v_min_values} "
          f"gamma_baseline={args.gamma_baseline} max_pair_rows={max_pair_rows}")

    cps._apply_dataset_config(dataset_cfg)
    country, var, data, ids = next(cps.iter_datasets())
    n_var = cps._get_cfg_attr(dataset_cfg, "N_VARS", "N_SERIES")
    n_year = cps._get_cfg_attr(dataset_cfg, "N_YEARS", "N_OBS")
    n_var = n_var[0] if isinstance(n_var, (list, tuple)) else n_var
    n_year = n_year[0] if isinstance(n_year, (list, tuple)) else n_year
    train_data, proxy_ids = cps.prepare_training_data(data, ids, n_year, n_var, train_ratio)
    proxy_ids = [str(x) for x in proxy_ids]
    n_series = train_data.shape[0] - 1
    length_data = train_data.shape[1]
    print(f"Dataset loaded: n_series={n_series} length_data={length_data} train_ratio={train_ratio}")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, args.n_vectors,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )

    max_start = length_data - window_size
    all_starts = list(range(0, max_start + 1, window_step))
    full_history_starts = [s for s in all_starts if s >= n_lags]
    candidate_anchors = full_history_starts or all_starts
    lag_count = max(1, int(n_lags // window_step) + 1)
    est_pairs_per_anchor = int(n_series * (n_series - 1) / 2) + int(n_series * n_series * max(lag_count - 1, 0))
    est_pairs_per_anchor = max(1, est_pairs_per_anchor)
    capped_anchor_count = max(1, int(max_pair_rows // est_pairs_per_anchor))
    anchor_count = min(anchor_count_requested, len(candidate_anchors), capped_anchor_count)
    anchor_count = max(1, anchor_count)
    rng = np.random.default_rng(random_seed)
    if anchor_count < len(candidate_anchors):
        anchor_starts = sorted(rng.choice(candidate_anchors, size=anchor_count, replace=False).astype(int).tolist())
    else:
        anchor_starts = sorted(int(x) for x in candidate_anchors)
    print(f"Sampling {len(anchor_starts)} anchors out of {len(candidate_anchors)} candidates "
          f"(~{est_pairs_per_anchor} pairs/anchor).")

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

    # Accumulators, per (k, v_min).
    settings = [(k, v) for k in args.k_values for v in args.v_min_values]
    n_baseline_pairs = 0
    n_true_pos = 0
    n_sketch_survivors = 0
    topk_pairs = {s: 0 for s in settings}
    topk_true_pos_recovered = {s: 0 for s in settings}
    topk_survivors_recovered = {s: 0 for s in settings}

    for anchor_id, anchor_start in enumerate(anchor_starts):
        min_lag_start = max(0, anchor_start - n_lags)
        lag_starts = list(range(anchor_start, min_lag_start - 1, -window_step))
        lag_starts = sorted(set(s for s in lag_starts if 0 <= s <= max_start))
        if not lag_starts or anchor_start not in lag_starts:
            continue

        # Build the historical-window vector pool for this anchor (all valid
        # (series, lag_start) windows across the lag window), keyed by
        # (series_idx, start) -- this is the "index" side.
        norm_by_start = {}
        pool_vectors = {}
        for s in lag_starts:
            _raw_by_sid, norm_by_sid = sketch_cache.get(s)
            norm_by_start[s] = norm_by_sid
            for s_idx in range(n_series):
                if not is_valid(s_idx, s):
                    continue
                v = norm_by_sid.get(proxy_ids[s_idx])
                if v is not None:
                    pool_vectors[(s_idx, s)] = v

        postings_by_k = {}
        topk_by_key_by_k = {}
        for k in args.k_values:
            postings, topk_by_key = build_postings(pool_vectors, k)
            postings_by_k[k] = postings
            topk_by_key_by_k[k] = topk_by_key

        for s_current in range(n_series):
            if not is_valid(s_current, anchor_start):
                continue
            q_key = (s_current, anchor_start)
            q_vec = pool_vectors.get(q_key)
            if q_vec is None:
                continue
            x = get_window(s_current, anchor_start)

            votes_scores_by_k = {}
            for k in args.k_values:
                q_topk_idx = topk_by_key_by_k[k].get(q_key)
                if q_topk_idx is None:
                    q_topk_idx = np.argpartition(-np.abs(q_vec), min(k, len(q_vec) - 1))[:k]
                votes, scores = query_candidates(q_vec, q_topk_idx, postings_by_k[k], signed_abs=neg_corr)
                votes_scores_by_k[k] = (votes, scores)

            for lag_start in lag_starts:
                for s_other in range(n_series):
                    if lag_start == anchor_start and s_other <= s_current:
                        continue
                    x_key = (s_other, lag_start)
                    x_vec = pool_vectors.get(x_key)
                    if x_vec is None:
                        continue
                    y = get_window(s_other, lag_start)
                    corr, _dist = _fast_corr_and_dist(x, y)
                    if not np.isfinite(corr):
                        continue
                    n_baseline_pairs += 1
                    is_tp = bool(abs(corr) >= corr_threshold)
                    if is_tp:
                        n_true_pos += 1
                    sketch_score = float(np.dot(q_vec, x_vec))
                    is_survivor = abs(sketch_score) >= args.gamma_baseline
                    if is_survivor:
                        n_sketch_survivors += 1

                    for k, v_min in settings:
                        votes, _scores = votes_scores_by_k[k]
                        retrieved = votes.get(x_key, 0) >= v_min
                        if retrieved:
                            topk_pairs[(k, v_min)] += 1
                            if is_tp:
                                topk_true_pos_recovered[(k, v_min)] += 1
                            if is_survivor:
                                topk_survivors_recovered[(k, v_min)] += 1

    print(f"\nBaseline: {n_baseline_pairs} pairs, {n_true_pos} exact true positives, "
          f"{n_sketch_survivors} sketch survivors (abs(score)>={args.gamma_baseline}).")

    rows = []
    for k, v_min in settings:
        n_topk = topk_pairs[(k, v_min)]
        tp_recovered = topk_true_pos_recovered[(k, v_min)]
        surv_recovered = topk_survivors_recovered[(k, v_min)]
        rows.append({
            "k": k,
            "v_min": v_min,
            "n_possible_pairs": n_baseline_pairs,
            "n_full_dot_checks_baseline": n_baseline_pairs,
            "n_full_dot_checks_topk": n_topk,
            "reduction_in_full_dot_checks": 1.0 - n_topk / n_baseline_pairs if n_baseline_pairs else float("nan"),
            "recall_of_exact_true_positives": tp_recovered / n_true_pos if n_true_pos else float("nan"),
            "recall_of_sketch_survivors": surv_recovered / n_sketch_survivors if n_sketch_survivors else float("nan"),
        })
    result = pd.DataFrame(rows).sort_values(["k", "v_min"])
    args.result_folder.mkdir(parents=True, exist_ok=True)
    out_path = args.result_folder / "topk_inverted_index_results.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    pd.set_option("display.width", 160)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
