"""Follow-up to diag_representation_options.py (2026-07-10). That first
pass found product_quantization and pca_adapted_hyperplanes inconclusive
(AUC 0.633/0.554 and 0.556/0.517, overall/hard-band) -- but both had
identified, specific implementation weaknesses, not necessarily fundamental
ones:

  - product_quantization used SYMMETRIC exact-code-match agreement
    (fraction of the pq_m subspaces where query and candidate quantize to
    the same centroid index) -- a much cruder signal than PQ's standard
    ASYMMETRIC distance-table formulation (keep the query's raw subvector,
    look up its distance to every centroid in each subspace once per
    query, sum the candidate's per-subspace centroid distances -- no
    quantization error on the query side at all).

  - pca_adapted_hyperplanes projected onto directions of maximum VARIANCE
    across the whole population (raw np.cov(warmup.T) eigendecomposition)
    -- those directions have no reason to separate true-correlated pairs
    from false ones. This tests LABEL-INFORMED directions instead: a
    Fisher-discriminant-style generalized eigenproblem on TRUE-PAIR
    difference vectors (v_i - v_j for true positives, which should have
    LOW variance along a good discriminative direction -- correlated
    series should look similar after sketching) vs NEGATIVE-PAIR
    difference vectors (should have HIGH variance along that same
    direction). Directions maximizing Sigma_neg / Sigma_tp variance ratio
    are the generalized eigenvectors of (Sigma_neg, Sigma_tp).

Both new features are compared side by side with the ORIGINAL crude
versions (kept, not removed, for honest before/after evidence) plus the
same dot_product / sign_alignment_frac baselines, same AUC methodology
(overall + hard-band) as diag_representation_options.py.

Usage: same CLI pattern as diag_representation_options.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh as scipy_eigh
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

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
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--lda-fit-steps", type=int, default=30)
    parser.add_argument("--lda-max-pairs-per-class", type=int, default=6000)
    parser.add_argument("--lda-ridge", type=float, default=1e-3)
    parser.add_argument("--pq-subvectors", type=int, default=8)
    parser.add_argument("--pq-k", type=int, default=16)
    parser.add_argument("--pca-n-components", type=int, default=16)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/representation_options_diag_v2"))
    args = parser.parse_args()
    args.result_folder.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")

    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    n_lags = _resolve_cfg_value(args.n_lags, exec_cfg, "N_LAGS", 168)
    corr_threshold = _resolve_cfg_value(args.corr_threshold, exec_cfg, "CORR_THRESHOLD", 0.7)
    neg_corr = _resolve_cfg_value(args.neg_corr, exec_cfg, "NEG_CORR", True)

    print(f"PARAMS: window_size={window_size} window_step={window_step} n_lags={n_lags} "
          f"corr_threshold={corr_threshold} neg_corr={neg_corr} n_vectors={args.n_vectors} "
          f"gamma={args.gamma} n_steps={args.n_steps} sketch_norm={args.sketch_norm} "
          f"lda_fit_steps={args.lda_fit_steps}")

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

    def gather_step_pool(cur_start):
        min_lag_start = max(0, cur_start - n_lags)
        lag_starts = sorted(set(s for s in range(cur_start, min_lag_start - 1, -window_step) if 0 <= s <= max_start))
        if not lag_starts or cur_start not in lag_starts:
            return None
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
        if len(keys_pool) < 10:
            return None
        vec_matrix = np.stack(vec_pool)
        key_index = {k: i for i, k in enumerate(keys_pool)}
        return vec_matrix, key_index

    # ---- Phase 0: warm-up sample for PQ codebook ----
    warmup_vecs = []
    for cur_start in all_starts[:max(20, args.n_steps // 4)]:
        _raw, norm_by_sid = sketch_cache.get(cur_start)
        for s_idx in range(n_series):
            if not is_valid(s_idx, cur_start):
                continue
            v = norm_by_sid.get(proxy_ids[s_idx])
            if v is not None:
                warmup_vecs.append(np.asarray(v, dtype=np.float64))
    warmup = np.stack(warmup_vecs)
    print(f"Warm-up sample: {warmup.shape[0]} vectors")

    n_vec = args.n_vectors
    pq_m = args.pq_subvectors
    pq_sub_dim = n_vec // pq_m
    assert pq_sub_dim * pq_m == n_vec, "n_vectors must be divisible by pq_subvectors"
    pq_codebooks = []
    for s in range(pq_m):
        sub = warmup[:, s * pq_sub_dim:(s + 1) * pq_sub_dim]
        km = KMeans(n_clusters=args.pq_k, n_init=3, random_state=s).fit(sub)
        pq_codebooks.append(km.cluster_centers_)
    print(f"PQ codebooks fit: {pq_m} subspaces x {args.pq_k} centroids x {pq_sub_dim} dims")

    # unsupervised PCA directions (kept for direct before/after comparison)
    cov = np.cov(warmup.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    pca_directions = eigvecs[:, order[:args.pca_n_components]].T
    print(f"Unsupervised PCA directions: top {args.pca_n_components} components, "
          f"explained variance ratio sum={eigvals[order[:args.pca_n_components]].sum() / eigvals.sum():.4f}")

    def pq_encode(mat):
        codes = np.empty((mat.shape[0], pq_m), dtype=np.int32)
        for s in range(pq_m):
            sub = mat[:, s * pq_sub_dim:(s + 1) * pq_sub_dim]
            d = ((sub[:, None, :] - pq_codebooks[s][None, :, :]) ** 2).sum(axis=2)
            codes[:, s] = np.argmin(d, axis=1)
        return codes

    def pq_dist_table(q_vec):
        # (pq_m, pq_k) squared distance from query's raw subvector to every
        # centroid in that subspace -- the query itself is NEVER quantized
        # (standard asymmetric distance computation / ADC).
        table = np.empty((pq_m, args.pq_k), dtype=np.float64)
        for s in range(pq_m):
            sub = q_vec[s * pq_sub_dim:(s + 1) * pq_sub_dim]
            table[s] = ((pq_codebooks[s] - sub[None, :]) ** 2).sum(axis=1)
        return table

    # ---- Phase 0.5: fit label-informed (LDA-style) directions on a
    # smaller pool of steps, using TRUE-PAIR and hard-NEGATIVE-PAIR
    # difference vectors. ----
    tp_diffs = []
    neg_diffs = []
    fit_starts = all_starts[:max(1, args.lda_fit_steps)]
    for cur_start in fit_starts:
        pool = gather_step_pool(cur_start)
        if pool is None:
            continue
        vec_matrix, key_index = pool
        for s_current in range(n_series):
            q_key = (s_current, cur_start)
            if q_key not in key_index:
                continue
            qi = key_index[q_key]
            q_vec = vec_matrix[qi]
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
                d = q_vec - vec_matrix[oi]
                if abs(corr) >= corr_threshold:
                    if len(tp_diffs) < args.lda_max_pairs_per_class:
                        tp_diffs.append(d)
                elif abs(corr) < corr_threshold * 0.5:
                    # "easy" negatives (clearly uncorrelated) -- deliberately
                    # not hard-band-restricted here so Sigma_neg reflects the
                    # bulk negative population the index must reject.
                    if len(neg_diffs) < args.lda_max_pairs_per_class:
                        neg_diffs.append(d)
        if len(tp_diffs) >= args.lda_max_pairs_per_class and len(neg_diffs) >= args.lda_max_pairs_per_class:
            break

    tp_diffs = np.array(tp_diffs)
    neg_diffs = np.array(neg_diffs)
    print(f"LDA fit pool: {len(tp_diffs)} true-positive diffs, {len(neg_diffs)} negative diffs")

    sigma_tp = np.cov(tp_diffs.T) + args.lda_ridge * np.eye(n_vec)
    sigma_neg = np.cov(neg_diffs.T) + args.lda_ridge * np.eye(n_vec)
    # generalized eigenproblem: Sigma_neg w = lambda Sigma_tp w -- directions
    # maximizing the ratio of negative-pair spread to true-pair spread
    # (true pairs should look ALIKE on a good direction, negatives should not).
    eigvals_lda, eigvecs_lda = scipy_eigh(sigma_neg, sigma_tp)
    order_lda = np.argsort(eigvals_lda)[::-1]
    lda_directions = eigvecs_lda[:, order_lda[:args.pca_n_components]].T
    print(f"LDA directions: top {args.pca_n_components} components, "
          f"top eigenvalue (neg/tp variance ratio)={eigvals_lda[order_lda[0]]:.4f}")

    # ---- Phase 1: exhaustive-within-bounded-window pairwise feature collection ----
    labels = []
    feat_dot = []
    feat_sign = []
    feat_pq_symmetric = []
    feat_pq_asymmetric = []
    feat_pca_unsupervised = []
    feat_pca_lda = []

    for step_id, cur_start in enumerate(all_starts):
        pool = gather_step_pool(cur_start)
        if pool is None:
            continue
        vec_matrix, key_index = pool

        sign_matrix = (vec_matrix >= 0)
        pq_codes = pq_encode(vec_matrix)
        pca_proj_unsup = vec_matrix @ pca_directions.T
        pca_sign_unsup = (pca_proj_unsup >= 0)
        pca_proj_lda = vec_matrix @ lda_directions.T
        pca_sign_lda = (pca_proj_lda >= 0)

        for s_current in range(n_series):
            q_key = (s_current, cur_start)
            if q_key not in key_index:
                continue
            qi = key_index[q_key]
            q_vec = vec_matrix[qi]

            q_sign = sign_matrix[qi]
            sign_agree = (sign_matrix == q_sign).mean(axis=1)
            feat_sign_batch = np.maximum(sign_agree, 1.0 - sign_agree)

            q_pq = pq_codes[qi]
            feat_pq_sym_batch = (pq_codes == q_pq[None, :]).mean(axis=1)

            dist_table = pq_dist_table(q_vec)  # (pq_m, pq_k)
            asym_dist = dist_table[np.arange(pq_m)[None, :], pq_codes].sum(axis=1)
            feat_pq_asym_batch = -asym_dist  # higher = more similar

            q_pca_sign_unsup = pca_sign_unsup[qi]
            pca_agree_unsup = (pca_sign_unsup == q_pca_sign_unsup[None, :]).mean(axis=1)
            feat_pca_unsup_batch = np.maximum(pca_agree_unsup, 1.0 - pca_agree_unsup)

            q_pca_sign_lda = pca_sign_lda[qi]
            pca_agree_lda = (pca_sign_lda == q_pca_sign_lda[None, :]).mean(axis=1)
            feat_pca_lda_batch = np.maximum(pca_agree_lda, 1.0 - pca_agree_lda)

            dot_batch = vec_matrix @ q_vec

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
                is_tp = 1 if abs(corr) >= corr_threshold else 0
                labels.append(is_tp)
                feat_dot.append(abs(float(dot_batch[oi])))
                feat_sign.append(float(feat_sign_batch[oi]))
                feat_pq_symmetric.append(float(feat_pq_sym_batch[oi]))
                feat_pq_asymmetric.append(float(feat_pq_asym_batch[oi]))
                feat_pca_unsupervised.append(float(feat_pca_unsup_batch[oi]))
                feat_pca_lda.append(float(feat_pca_lda_batch[oi]))

        if step_id % 10 == 0:
            print(f"  progress: step {step_id}/{len(all_starts)}, {len(labels)} pairs so far", flush=True)

    labels = np.array(labels)
    feat_dot = np.array(feat_dot)
    feat_sign = np.array(feat_sign)
    feat_pq_symmetric = np.array(feat_pq_symmetric)
    feat_pq_asymmetric = np.array(feat_pq_asymmetric)
    feat_pca_unsupervised = np.array(feat_pca_unsupervised)
    feat_pca_lda = np.array(feat_pca_lda)

    n_tp = int(labels.sum())
    print(f"\nTotal pairs: {len(labels)}, true positives: {n_tp} ({n_tp/len(labels)*100:.2f}%)")

    dot_hi = np.percentile(feat_dot[labels == 1], 90) if n_tp > 0 else 1.0
    hard_mask = (labels == 1) | ((labels == 0) & (feat_dot >= np.percentile(feat_dot, 80)) & (feat_dot <= dot_hi))
    print(f"Hard-band population: {hard_mask.sum()} pairs ({(labels[hard_mask]==1).sum()} true positives)")

    def auc_safe(y, score, mask=None):
        if mask is not None:
            y = y[mask]
            score = score[mask]
        if len(np.unique(y)) < 2:
            return float("nan")
        return roc_auc_score(y, score)

    rows = []
    for name, feat in [
        ("dot_product (baseline)", feat_dot),
        ("sign_alignment_frac (baseline)", feat_sign),
        ("product_quantization_symmetric (original)", feat_pq_symmetric),
        ("product_quantization_asymmetric (new)", feat_pq_asymmetric),
        ("pca_unsupervised_hyperplanes (original)", feat_pca_unsupervised),
        ("lda_adapted_hyperplanes (new)", feat_pca_lda),
    ]:
        rows.append(dict(
            representation=name,
            auc_overall=auc_safe(labels, feat),
            auc_hard_band=auc_safe(labels, feat, hard_mask),
        ))
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out_path = args.result_folder / "representation_options_v2_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
