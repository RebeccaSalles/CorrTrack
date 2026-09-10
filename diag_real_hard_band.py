"""Focused follow-up to the real-data runs of diag_representation_options*.py
(2026-07-13). Those runs answered the overall-separability question decisively
(dot AUC 0.9988/0.9999 on fr_air_temperature/fr_wind_direction vs 0.978 on the
synthetic benchmark) but their hard-band AUC came back NaN -- the dot-percentile
hard-band mask found ZERO negatives overlapping the true-positive dot range.
That NaN is itself evidence of extreme separation, but the 2026-07-09 synthetic
"wall" (dot AUC ~0.65 near the boundary) was defined on TRUE-correlation bands,
not dot percentiles. This script computes the like-for-like number on real
data: AUC restricted to pairs whose true |corr| sits in a band around the
corr_threshold boundary, for the dot and sign_alignment_frac features, plus
the true-|corr| histogram and the candidate-reduction each feature achieves at
95% recall on the full population.

Usage: same CLI pattern as diag_representation_options.py (plus --step-stride
to subsample window starts for speed; the band AUC is a per-pair property, so
subsampling steps does not bias it, unlike anchor sampling).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--step-stride", type=int, default=8)
    parser.add_argument("--band-halfwidth", type=float, default=0.15,
                        help="hard band = true |corr| in [tau-hw, tau+hw]")
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/real_hard_band_diag"))
    args = parser.parse_args()
    args.result_folder.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")
    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    n_lags = _resolve_cfg_value(args.n_lags, exec_cfg, "N_LAGS", 168)
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
    n_series = train_data.shape[0] - 1
    length_data = train_data.shape[1]
    print(f"Dataset: {country}/{var}, n_series={n_series}, length={length_data}, "
          f"window={window_size}/{window_step}, n_lags={n_lags}, tau={corr_threshold}, "
          f"stride={args.step_stride}")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, args.n_vectors,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )
    max_start = length_data - window_size
    all_starts = list(range(0, max_start + 1, window_step))[::args.step_stride][:args.n_steps]
    values = np.asarray(train_data[1:1 + n_series, :], dtype=np.float64)
    window_cache, valid_cache = {}, {}

    def get_window(s_idx, start):
        key = (s_idx, start)
        if key not in window_cache:
            window_cache[key] = values[s_idx, start:start + window_size]
        return window_cache[key]

    def is_valid(s_idx, start):
        key = (s_idx, start)
        if key not in valid_cache:
            valid_cache[key] = CorrTrack_optimize._proxy_window_is_valid(get_window(s_idx, start))
        return valid_cache[key]

    corrs, dots, signs = [], [], []
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
        for s_current in range(n_series):
            q_key = (s_current, cur_start)
            if q_key not in key_index:
                continue
            qi = key_index[q_key]
            dot_batch = vec_matrix @ vec_matrix[qi]
            agree = (sign_matrix == sign_matrix[qi]).mean(axis=1)
            sign_batch = np.maximum(agree, 1.0 - agree)
            for other_key, oi in key_index.items():
                if other_key == q_key:
                    continue
                s_other, lag_start = other_key
                if lag_start == cur_start and s_other <= s_current:
                    continue
                corr, _ = _fast_corr_and_dist(get_window(s_current, cur_start), get_window(s_other, lag_start))
                if not np.isfinite(corr):
                    continue
                corrs.append(abs(float(corr)))
                dots.append(abs(float(dot_batch[oi])))
                signs.append(float(sign_batch[oi]))
        if step_id % 50 == 0:
            print(f"  step {step_id}/{len(all_starts)}, {len(corrs)} pairs", flush=True)

    corrs = np.array(corrs); dots = np.array(dots); signs = np.array(signs)
    labels = (corrs >= corr_threshold).astype(int)
    print(f"\nTotal pairs: {len(corrs)}, TP: {labels.sum()} ({labels.mean()*100:.2f}%)")

    hist, edges = np.histogram(corrs, bins=np.arange(0, 1.05, 0.05))
    print("true |corr| histogram (bin_lo, count):")
    for lo, c in zip(edges[:-1], hist):
        print(f"  {lo:.2f}: {c}")

    lo, hi = corr_threshold - args.band_halfwidth, min(1.0, corr_threshold + args.band_halfwidth)
    band = (corrs >= lo) & (corrs <= hi)
    print(f"\nhard band: true |corr| in [{lo:.2f}, {hi:.2f}] -> {band.sum()} pairs "
          f"({labels[band].sum()} TP, {(band.sum() - labels[band].sum())} neg)")

    def auc(y, s):
        return roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")

    def reduction_at_recall(y, s, target=0.95):
        # fraction of ALL pairs NOT touched when thresholding s to keep >= target of TPs
        tp_scores = np.sort(s[y == 1])
        k = int(np.floor((1 - target) * len(tp_scores)))
        thr = tp_scores[k] if len(tp_scores) else 0.0
        touched = (s >= thr).mean()
        got = (s[y == 1] >= thr).mean()
        return 1.0 - touched, got

    rows = []
    for name, feat in (("dot_product", dots), ("sign_alignment_frac", signs)):
        red, got = reduction_at_recall(labels, feat)
        rows.append(dict(feature=name,
                         auc_overall=auc(labels, feat),
                         auc_hard_band=auc(labels[band], feat[band]),
                         candidate_reduction_at_95pct_recall=red,
                         achieved_recall=got))
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = args.result_folder / "real_hard_band_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
