"""Phase 1 diagnostic (see docs/implementation_log.md, "candidate_selector
diagnostics: is gamma=0.55 globally necessary"): for a sample of real
anchor-vs-history pairs (same causal anchor-sampling design as the
proxy-anchor hyperopt path), compute BOTH the exact Pearson correlation and
the real production sketch dot-product score, plus per-window reliability
features -- none of which the existing hyperopt path retains (it thresholds
exact_corr into a bool and never surfaces a continuous sketch score).

Deliberately a separate, standalone script (not touching
library_corrtrack_parallel.py's hyperopt-critical proxy-anchor code path) --
it reuses the same building blocks (_fast_corr_and_dist,
CorrTrack_optimize._proxy_window_is_valid, corrtrack_param_search's dataset
loading helpers, and the real Sketches class for production-identical
sketch construction) without modifying any of them.

Answers: is gamma=0.55 needed globally, or only for a low-reliability
subset of pairs/windows? Output: a CSV with gamma_95/gamma_99 per
reliability group, plus a raw per-pair CSV for follow-up experiments.

Usage (mirrors the real experiment CLI's flag names where applicable):

    PYTHONPATH="$(pwd)" python3 corrtrack_release_dev/diag_gamma_reliability.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 \\
        --neg-corr --train-ratio 0.33 --n-vectors 16 \\
        --result-folder tmp_artifacts/gamma_reliability_diag
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize, Sketches
import library_corrtrack_parallel as lcp
from library_corrtrack_parallel import _fast_corr_and_dist


def _load_module(config_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _resolve_cfg_value(cli_value, cfg_module, attr, default):
    if cli_value is not None:
        return cli_value
    return getattr(cfg_module, attr, default)


# ---------------------------------------------------------------------------
# Monkeypatch: capture the raw pre-normalization sketch matrix.
#
# _sketches_from_scratch's Cython-accelerated path (_cy_build_sketch_matrix)
# computes projection + normalization in one call, never exposing the raw
# (pre-centering, pre-scaling) matrix to Python. Forcing that name to None
# makes the code fall back to the pure-Python path
# (raw_matrix = np.sum(series_dots, axis=1); norm_matrix =
# self._normalize_sketch_matrix(raw_matrix)), which we DO wrap, purely
# within this script's process -- library_corrtrack_parallel.py itself is
# never modified. Mathematically the two paths compute the same values
# (one is just a faster implementation of the other); a diagnostic script
# does not need the speed.
# ---------------------------------------------------------------------------
lcp._cy_build_sketch_matrix = None
lcp._cy_apply_orth_and_normalize = None

_orig_normalize_sketch_matrix = Sketches._normalize_sketch_matrix


def _patched_normalize_sketch_matrix(self, matrix):
    self._diag_raw_matrix = np.array(matrix, dtype=np.float64, copy=True)
    return _orig_normalize_sketch_matrix(self, matrix)


Sketches._normalize_sketch_matrix = _patched_normalize_sketch_matrix


# ---------------------------------------------------------------------------
# Sketch cache: real, production-identical sketches for a given window start.
# ---------------------------------------------------------------------------
class SketchCache:
    def __init__(self, train_data, proxy_ids, window_size, basic_window, window_step,
                 seed, seed_toggle, n_vectors, grid_dimension, preprocess, neg_corr, sketch_norm):
        self.train_data = train_data
        self.proxy_ids = proxy_ids
        self.window_size = window_size
        self.basic_window = basic_window
        self.window_step = window_step
        self.seed = seed
        self.seed_toggle = seed_toggle
        self.n_vectors = n_vectors
        self.grid_dimension = grid_dimension
        self.preprocess = preprocess
        self.neg_corr = neg_corr
        self.sketch_norm = sketch_norm
        self._cache = {}

    def get(self, start_idx):
        start_idx = int(start_idx)
        cached = self._cache.get(start_idx)
        if cached is not None:
            return cached
        sketcher = Sketches(
            self.window_size, self.basic_window, self.window_step,
            self.seed, self.seed_toggle, self.n_vectors, self.grid_dimension, [],
            self.preprocess, neg_corr=self.neg_corr, sketch_norm=self.sketch_norm,
            full_vector_candidates=True,
        )
        data_step = np.array(
            self.train_data[:, start_idx:start_idx + self.window_size],
            dtype=object, copy=True,
        )
        data_step[0, :] = np.arange(start_idx, start_idx + data_step.shape[1], dtype=np.int64)
        norm_sketches, _partitions = sketcher.run(
            data_step, self.proxy_ids, verbose=False, testing=False, distribute=False,
        )
        raw_matrix = getattr(sketcher, "_diag_raw_matrix", None)
        sketch_keys = list(sketcher._sketch_keys)
        if raw_matrix is None:
            raise RuntimeError(
                "raw sketch matrix was not captured -- monkeypatch did not fire "
                "(unexpected code path; check _sketches_from_scratch branching)"
            )
        # sketch_keys[i] = (series_id, start_time, window_size), aligned with
        # raw_matrix[i] (raw, pre-normalization) and norm_sketches[sketch_keys[i]]
        # (final mean_l2-normalized vector).
        raw_by_sid = {}
        norm_by_sid = {}
        for i, key in enumerate(sketch_keys):
            sid = key[0]
            raw_by_sid[sid] = raw_matrix[i]
            norm_by_sid[sid] = np.asarray(norm_sketches.get(key), dtype=np.float64)
        result = (raw_by_sid, norm_by_sid)
        self._cache[start_idx] = result
        return result


# ---------------------------------------------------------------------------
# Per-window reliability features (computed once per (series, window start)).
# ---------------------------------------------------------------------------
def _window_features(raw_v, norm_w):
    raw_v = np.asarray(raw_v, dtype=np.float64)
    norm_w = np.asarray(norm_w, dtype=np.float64)
    n = raw_v.size
    mean_v = float(np.mean(raw_v)) if n else 0.0
    centered = raw_v - mean_v
    centered_l2 = float(np.linalg.norm(centered))
    n_pos = int(np.sum(raw_v > 0))
    n_neg = int(np.sum(raw_v < 0))
    return {
        "raw_sketch_mean": mean_v,
        "raw_sketch_l2": float(np.linalg.norm(raw_v)),
        "centered_sketch_l2": centered_l2,
        "raw_sketch_l1": float(np.sum(np.abs(raw_v))),
        "normalized_sketch_l1": float(np.sum(np.abs(norm_w))) if norm_w.size else 0.0,
        "max_abs_raw": float(np.max(np.abs(raw_v))) if n else 0.0,
        "max_abs_norm": float(np.max(np.abs(norm_w))) if norm_w.size else 0.0,
        "sign_balance": float(n_pos - n_neg) / n if n else 0.0,
        "sketch_variance": float(np.var(raw_v)) if n else 0.0,
    }


def _pair_features(fq, fx, abs_score):
    centered_l2_min = min(fq["centered_sketch_l2"], fx["centered_sketch_l2"])
    centered_l2_max = max(fq["centered_sketch_l2"], fx["centered_sketch_l2"])
    return {
        "abs_sketch_score": abs_score,
        "centered_l2_min": centered_l2_min,
        "centered_l2_max": centered_l2_max,
        "centered_l2_product": fq["centered_sketch_l2"] * fx["centered_sketch_l2"],
        "l1_diff": abs(fq["normalized_sketch_l1"] - fx["normalized_sketch_l1"]),
        "max_abs_product": fq["max_abs_norm"] * fx["max_abs_norm"],
        "max_abs_norm_min": min(fq["max_abs_norm"], fx["max_abs_norm"]),
        "normalized_l1_min": min(fq["normalized_sketch_l1"], fx["normalized_sketch_l1"]),
    }


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
    parser.add_argument("--n-vectors", type=int, default=16,
                         help="Sketch dimensionality for this diagnostic run (not swept -- "
                              "the real grid sweeps [8,16,32,64]; pick one representative value "
                              "and re-run for others if needed). Default 16.")
    parser.add_argument("--sketch-norm", type=str, default="mean_l2")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--seed-toggle", type=int, default=1357)
    parser.add_argument("--gamma-baseline", type=float, default=0.55,
                         help="Reference gamma to report against (not a filter -- all pairs "
                              "in the sampled universe are scored regardless). Default 0.55.")
    parser.add_argument("--max-pair-rows", type=int, default=None,
                         help="Override OPTIM_PROXY_MAX_PAIR_ROWS for this one-off diagnostic "
                              "(not bound by production's memory/compute budget). With a "
                              "50-series dataset and n_lags/window_step giving ~11 lagged "
                              "windows, a single anchor alone can produce ~26k pairs -- the "
                              "production default (50000) caps the real anchor_count_requested "
                              "down to effectively 1 anchor. Raise this to get a broader, more "
                              "representative sample across many anchors.")
    parser.add_argument("--n-bins", type=int, default=3, help="Quantile bins per reliability feature. Default 3.")
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/gamma_reliability_diag"))
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
    max_pair_rows = int(args.max_pair_rows) if args.max_pair_rows is not None else int(getattr(exec_cfg, "OPTIM_PROXY_MAX_PAIR_ROWS", 50000))
    random_seed = int(getattr(exec_cfg, "OPTIM_PROXY_RANDOM_SEED", 2468))

    print(f"PARAMS: window_size={window_size} window_step={window_step} n_lags={n_lags} "
          f"corr_threshold={corr_threshold} neg_corr={neg_corr} train_ratio={train_ratio} "
          f"n_vectors={args.n_vectors} sketch_norm={args.sketch_norm} seed={args.seed} "
          f"seed_toggle={args.seed_toggle} anchor_count_requested={anchor_count_requested} "
          f"max_pair_rows={max_pair_rows} random_seed={random_seed} "
          f"gamma_baseline={args.gamma_baseline}")

    # --- Load the real dataset exactly as corrtrack_param_search does. ---
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
    print(f"Dataset loaded: n_series={n_series} length_data={length_data} (train prefix only, "
          f"train_ratio={train_ratio})")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    grid_dimension = args.n_vectors

    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, grid_dimension,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )

    # --- Anchor/lag-start sampling: same design as
    # CorrTrack_optimize._prepare_proxy_anchor_reference, but keeping the
    # raw exact_corr float (which that method discards) and computing a
    # real sketch dot-product score for every pair from the same windows. ---
    max_start = length_data - window_size
    if max_start < 0:
        raise ValueError("train prefix shorter than window_size -- reduce --train-ratio or check dataset")
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
    print(f"Sampling {len(anchor_starts)} anchors (requested {anchor_count_requested}, "
          f"capped by max_pair_rows to {capped_anchor_count}) out of {len(candidate_anchors)} candidates.")

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

    rows = []
    n_pairs_seen = 0
    max_rows_reached = False
    for anchor_id, anchor_start in enumerate(anchor_starts):
        min_lag_start = max(0, anchor_start - n_lags)
        lag_starts = list(range(anchor_start, min_lag_start - 1, -window_step))
        lag_starts = sorted(set(s for s in lag_starts if 0 <= s <= max_start))
        if not lag_starts or anchor_start not in lag_starts:
            continue

        raw_by_start = {}
        norm_by_start = {}
        for s in lag_starts:
            raw_by_sid, norm_by_sid = sketch_cache.get(s)
            raw_by_start[s] = raw_by_sid
            norm_by_start[s] = norm_by_sid
        window_feat_cache = {}

        def wfeat(sid_idx, start):
            key = (sid_idx, start)
            cached = window_feat_cache.get(key)
            if cached is not None:
                return cached
            sid = proxy_ids[sid_idx]
            raw_v = raw_by_start[start].get(sid)
            norm_w = norm_by_start[start].get(sid)
            if raw_v is None or norm_w is None:
                return None
            feat = _window_features(raw_v, norm_w)
            window_feat_cache[key] = feat
            return feat

        for s_current in range(n_series):
            if not is_valid(s_current, anchor_start):
                continue
            fq = wfeat(s_current, anchor_start)
            if fq is None:
                continue
            x = get_window(s_current, anchor_start)
            for lag_start in lag_starts:
                for s_other in range(n_series):
                    if lag_start == anchor_start and s_other <= s_current:
                        continue
                    if not is_valid(s_other, lag_start):
                        continue
                    fx = wfeat(s_other, lag_start)
                    if fx is None:
                        continue
                    y = get_window(s_other, lag_start)
                    corr, _dist = _fast_corr_and_dist(x, y)
                    if not np.isfinite(corr):
                        continue
                    norm_q = raw_by_start[anchor_start]
                    norm_w_q = norm_by_start[anchor_start].get(proxy_ids[s_current])
                    norm_w_x = norm_by_start[lag_start].get(proxy_ids[s_other])
                    if norm_w_q is None or norm_w_x is None:
                        continue
                    sketch_score = float(np.dot(norm_w_q, norm_w_x))
                    abs_sketch_score = abs(sketch_score)
                    pair_feat = _pair_features(fq, fx, abs_sketch_score)
                    is_true_pos = bool(abs(corr) >= corr_threshold)
                    rows.append({
                        "anchor_id": anchor_id,
                        "s_current": s_current,
                        "s_other": s_other,
                        "anchor_start": anchor_start,
                        "lag_start": lag_start,
                        "exact_corr": float(corr),
                        "is_true_pos": is_true_pos,
                        "sketch_score": sketch_score,
                        **pair_feat,
                    })
                    n_pairs_seen += 1
                    if n_pairs_seen >= max_pair_rows:
                        max_rows_reached = True
                        break
                if max_rows_reached:
                    break
            if max_rows_reached:
                break
        if max_rows_reached:
            print(f"max_pair_rows ({max_pair_rows}) reached at anchor {anchor_id}; stopping early.")
            break

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} pairs across {df['anchor_id'].nunique()} anchors; "
          f"{int(df['is_true_pos'].sum())} exact true positives (|corr| >= {corr_threshold}).")

    args.result_folder.mkdir(parents=True, exist_ok=True)
    raw_path = args.result_folder / "gamma_reliability_raw_pairs.csv"
    df.to_csv(raw_path, index=False)
    print(f"Wrote raw per-pair data: {raw_path} ({len(df)} rows)")

    # --- Group true positives by quantiles of each reliability feature and
    # compute gamma_95 / gamma_99 -- the core diagnostic question. ---
    group_features = ["centered_l2_min", "normalized_l1_min", "max_abs_norm_min"]
    true_pos = df[df["is_true_pos"]]
    summary_rows = []

    def _summarize(group_name, feature_name, subset):
        scores = subset["abs_sketch_score"].to_numpy()
        n_true_pos = len(subset)
        if n_true_pos == 0:
            gamma_95 = gamma_99 = mean_s = median_s = float("nan")
        else:
            gamma_95 = float(np.percentile(scores, 5))
            gamma_99 = float(np.percentile(scores, 1))
            mean_s = float(np.mean(scores))
            median_s = float(np.median(scores))
        summary_rows.append({
            "group_feature": feature_name,
            "group_name": group_name,
            "n_pairs": int((df[feature_name] if feature_name in df.columns else pd.Series(dtype=float)).notna().sum())
            if feature_name in df.columns else len(df),
            "n_true_pos": n_true_pos,
            "gamma_95": gamma_95,
            "gamma_99": gamma_99,
            "mean_abs_score_true_pos": mean_s,
            "median_abs_score_true_pos": median_s,
        })

    _summarize("global", "global", true_pos)

    for feat in group_features:
        try:
            bin_labels = [f"q{i+1}_of_{args.n_bins}" for i in range(args.n_bins)]
            df[f"_{feat}_bin"] = pd.qcut(df[feat], q=args.n_bins, labels=bin_labels, duplicates="drop")
        except ValueError as exc:
            print(f"Skipping grouping by {feat}: {exc}")
            continue
        for label in df[f"_{feat}_bin"].cat.categories:
            group_mask = df[f"_{feat}_bin"] == label
            group_true_pos = df[group_mask & df["is_true_pos"]]
            n_pairs_in_group = int(group_mask.sum())
            scores = group_true_pos["abs_sketch_score"].to_numpy()
            if len(scores) == 0:
                gamma_95 = gamma_99 = mean_s = median_s = float("nan")
            else:
                gamma_95 = float(np.percentile(scores, 5))
                gamma_99 = float(np.percentile(scores, 1))
                mean_s = float(np.mean(scores))
                median_s = float(np.median(scores))
            summary_rows.append({
                "group_feature": feat,
                "group_name": str(label),
                "n_pairs": n_pairs_in_group,
                "n_true_pos": len(scores),
                "gamma_95": gamma_95,
                "gamma_99": gamma_99,
                "mean_abs_score_true_pos": mean_s,
                "median_abs_score_true_pos": median_s,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.result_folder / "gamma_reliability_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
