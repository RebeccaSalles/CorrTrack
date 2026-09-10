"""Diagnostic (2026-07-10, candidate_selector thread): investigate four
alternative sketch REPRESENTATIONS for candidate retrieval, all derived
from the ALREADY sketch-then-normalize pipeline (no change to sketch
computation order -- see docs/implementation_log.md) -- before writing any
real backend code. Every prior indexing mechanism tried this session
(SignLSHBandIndex bands, IVFHammingIndex partition+multi-probe) hit the
same wall: near-gamma-boundary true pairs don't have strong enough
sign-bit locality to be captured without losing most of the population-
reduction benefit (the 2026-07-09 hard-band finding: even the exact dot
only reaches AUC~0.65 there). This diagnostic checks whether a RICHER
representation of the same (mean_l2-normalized) sketch does better,
specifically in that hard band, before investing in a real Cython
backend:

  1. Product Quantization (PQ): split the sketch into m subvectors,
     quantize each against a small k-means codebook (fit once on a
     warm-up sample), score pairs via a PQ-approximate distance.
  2. Winner-Take-All (WTA) hashing: for n_hashes random K-dim subsets,
     hash = argmax(|value|) within the subset; score pairs via hash
     agreement fraction.
  3. Ternary quantization: per-dimension {-1,0,+1} code (threshold on
     |value|), instead of binary sign; score via agreement among
     non-zero-on-both-sides dimensions.
  4. PCA-adapted (data-dependent) hyperplanes: instead of random
     dimension subsets/directions, project onto the top-K principal
     components of the sketch population itself; score via sign
     agreement on those directions.

Baselines recomputed on the SAME population for a fair comparison:
sign_alignment_frac (the feature behind SignLSHBandIndex) and the exact
dot product itself.

For every feature: overall AUC, and HARD-BAND AUC restricted to pairs
whose true correlation sits in [corr_threshold, corr_threshold+0.15] vs
true negatives in the same dot-score neighborhood -- i.e., can this
feature help decide near the decision boundary, not just far from it
(where everything already works).

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/diag_representation_options.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 --neg-corr \\
        --n-vectors 64 --gamma 0.55 --n-steps 100 \\
        --result-folder tmp_artifacts/representation_options_diag
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--pq-subvectors", type=int, default=8)
    parser.add_argument("--pq-k", type=int, default=16)
    parser.add_argument("--wta-n-hashes", type=int, default=32)
    parser.add_argument("--wta-k", type=int, default=8)
    parser.add_argument("--pca-n-components", type=int, default=16)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/representation_options_diag"))
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
          f"gamma={args.gamma} n_steps={args.n_steps} sketch_norm={args.sketch_norm}")

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

    # ---- Phase 0: warm-up sample for PQ codebook / PCA directions ----
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

    rng = np.random.default_rng(17)
    wta_subsets = [rng.choice(n_vec, size=args.wta_k, replace=False) for _ in range(args.wta_n_hashes)]

    cov = np.cov(warmup.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    pca_directions = eigvecs[:, order[:args.pca_n_components]].T  # (n_components, n_vec)
    print(f"PCA directions: top {args.pca_n_components} components, "
          f"explained variance ratio sum={eigvals[order[:args.pca_n_components]].sum() / eigvals.sum():.4f}")

    def pq_encode(mat):
        # mat: (N, n_vec) -> codes (N, pq_m) int
        codes = np.empty((mat.shape[0], pq_m), dtype=np.int32)
        for s in range(pq_m):
            sub = mat[:, s * pq_sub_dim:(s + 1) * pq_sub_dim]
            d = ((sub[:, None, :] - pq_codebooks[s][None, :, :]) ** 2).sum(axis=2)
            codes[:, s] = np.argmin(d, axis=1)
        return codes

    def wta_encode(mat):
        codes = np.empty((mat.shape[0], args.wta_n_hashes), dtype=np.int32)
        absmat = np.abs(mat)
        for h, subset in enumerate(wta_subsets):
            codes[:, h] = np.argmax(absmat[:, subset], axis=1)
        return codes

    def ternary_encode(mat, thresh):
        code = np.zeros(mat.shape, dtype=np.int8)
        code[mat > thresh] = 1
        code[mat < -thresh] = -1
        return code

    ternary_thresh = np.percentile(np.abs(warmup), 40)  # ~40% of dims -> 0 ("unreliable")
    print(f"Ternary threshold (40th pct of |value|): {ternary_thresh:.4f}")

    # ---- Phase 1: exhaustive-within-bounded-window pairwise feature collection ----
    labels = []
    feat_dot = []
    feat_sign = []
    feat_pq = []
    feat_wta = []
    feat_ternary = []
    feat_pca = []

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
        if len(keys_pool) < 10:
            continue
        vec_matrix = np.stack(vec_pool)
        key_index = {k: i for i, k in enumerate(keys_pool)}

        sign_matrix = (vec_matrix >= 0)
        pq_codes = pq_encode(vec_matrix)
        wta_codes = wta_encode(vec_matrix)
        ternary_matrix = ternary_encode(vec_matrix, ternary_thresh)
        pca_proj = vec_matrix @ pca_directions.T  # (N, n_components)
        pca_sign = (pca_proj >= 0)

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
            feat_pq_batch = (pq_codes == q_pq[None, :]).mean(axis=1)

            q_wta = wta_codes[qi]
            wta_agree = (wta_codes == q_wta[None, :]).mean(axis=1)
            feat_wta_batch = wta_agree

            q_tern = ternary_matrix[qi]
            both_nonzero = (ternary_matrix != 0) & (q_tern[None, :] != 0)
            agree = (ternary_matrix == q_tern[None, :]) & both_nonzero
            disagree = (ternary_matrix == -q_tern[None, :]) & both_nonzero
            denom = both_nonzero.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = (agree.sum(axis=1) - disagree.sum(axis=1)) / np.maximum(denom, 1)
            feat_ternary_batch = np.where(denom > 0, np.abs(raw), 0.0)

            q_pca_sign = pca_sign[qi]
            pca_agree = (pca_sign == q_pca_sign[None, :]).mean(axis=1)
            feat_pca_batch = np.maximum(pca_agree, 1.0 - pca_agree)

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
                feat_pq.append(float(feat_pq_batch[oi]))
                feat_wta.append(float(feat_wta_batch[oi]))
                feat_ternary.append(float(feat_ternary_batch[oi]))
                feat_pca.append(float(feat_pca_batch[oi]))

        if step_id % 10 == 0:
            print(f"  progress: step {step_id}/{len(all_starts)}, {len(labels)} pairs so far", flush=True)

    labels = np.array(labels)
    feat_dot = np.array(feat_dot)
    feat_sign = np.array(feat_sign)
    feat_pq = np.array(feat_pq)
    feat_wta = np.array(feat_wta)
    feat_ternary = np.array(feat_ternary)
    feat_pca = np.array(feat_pca)

    n_tp = int(labels.sum())
    print(f"\nTotal pairs: {len(labels)}, true positives: {n_tp} ({n_tp/len(labels)*100:.2f}%)")

    # hard band: true positives just above threshold, vs true negatives with
    # similar DOT score (the decision-relevant comparison, not the whole population)
    hard_lo, hard_hi = corr_threshold, min(1.0, corr_threshold + 0.15)
    # approximate true correlation via feat_dot's rank isn't available; use labels directly
    # plus restrict negatives to those with feat_dot in a comparable range to avoid
    # comparing "obviously easy" negatives.
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
        ("product_quantization", feat_pq),
        ("winner_take_all", feat_wta),
        ("ternary_quantization", feat_ternary),
        ("pca_adapted_hyperplanes", feat_pca),
    ]:
        rows.append(dict(
            representation=name,
            auc_overall=auc_safe(labels, feat),
            auc_hard_band=auc_safe(labels, feat, hard_mask),
        ))
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out_path = args.result_folder / "representation_options_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
