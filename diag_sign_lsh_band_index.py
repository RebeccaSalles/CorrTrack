"""Sign-bit LSH band index diagnostic (follow-up to diag_pair_separability.py,
2026-07-09): sign_alignment_frac = max(agree_frac, 1-agree_frac) -- Hamming
similarity between two windows' full sign(w_hat) patterns -- scored AUC=0.92,
62.7% candidate reduction as a CONTINUOUS pairwise feature, the strongest
genuinely-cheap-to-compute signal found in that study. But it is NOT a
single-scalar-per-window index (sign(w_hat) is a k-bit pattern, not one
sortable number) -- real indexing needs LSH bit-sampling/banding: split the
k sign bits into B independent bands of b bits each, bucket windows by each
band's value, and retrieve a pair as a candidate if ANY band matches
(standard OR-of-bands LSH amplification). This is genuinely non-quadratic:
building the index is O(N*B) (N windows), querying is O(B) bucket lookups
plus however many collisions land in those buckets -- no full pairwise
enumeration.

This is a DIFFERENT (and expected-stronger) test than the existing
simhash_hamming_r* feature in diag_pair_separability.py, which banded on
only r=2-16 RANDOM PROJECTIONS' signs (a heavy compression of the real
64-dim sign pattern, and it scored badly: AUC~0.53-0.55). Here the bands are
drawn directly from the REAL per-coordinate signs of w_hat, which is what
sign_alignment_frac's strong AUC was actually measuring.

Because a pair might be a match via SIGN AGREEMENT (positive correlation)
OR SIGN OPPOSITION (negative correlation, since neg_corr=True), each band
lookup checks BOTH the query's own band key and its bitwise-complement
within that band -- same "signed/absolute" doubling already used for
InstinctIndex/TopKInvertedIndex's signed query mode.

Exhaustive design matches diag_pair_separability.py's Tier 1 (every
window-start step across the whole dataset, no anchor sampling, no
train-prefix restriction) -- reuses SketchCache from diag_gamma_reliability.py
for real production-identical sketch construction. Only aggregate counters
are accumulated per (b, B) setting, not per-pair rows, so memory stays
trivial regardless of dataset scale (no OOM risk like diag_pair_separability.py
hit).

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/diag_sign_lsh_band_index.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 --neg-corr \\
        --n-vectors 64 --gamma 0.55 \\
        --result-folder tmp_artifacts/sign_lsh_band_diag
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize, _fast_corr_and_dist
from diag_gamma_reliability import _load_module, _resolve_cfg_value, SketchCache


def build_band_specs(n_vectors, b_values, n_bands_values, seed):
    """For each b (band width), generate max(n_bands_values) independent
    random b-dim index subsets (with replacement across bands, without
    replacement within a band) -- shared across all (b, n_bands) settings so
    smaller n_bands settings are proper prefixes of larger ones."""
    rng = np.random.default_rng(seed)
    specs = {}
    max_bands = max(n_bands_values)
    for b in b_values:
        specs[b] = [rng.choice(n_vectors, size=min(b, n_vectors), replace=False) for _ in range(max_bands)]
    return specs


def band_keys(sign_bits, band_dim_sets):
    """sign_bits: bool array (n_vectors,). Returns one int key per band."""
    keys = []
    for dims in band_dim_sets:
        bits = sign_bits[dims]
        key = 0
        for bit in bits:
            key = (key << 1) | int(bit)
        keys.append(key)
    return keys


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
    parser.add_argument("--b-values", type=str, default="4,6,8,10,12,16", help="Band width (bits per band), comma-separated.")
    parser.add_argument("--n-bands-values", type=str, default="1,2,4,8,16,32", help="Number of independent bands, comma-separated.")
    parser.add_argument("--band-seed", type=int, default=4242)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/sign_lsh_band_diag"))
    args = parser.parse_args()

    b_values = [int(x) for x in args.b_values.split(",")]
    n_bands_values = [int(x) for x in args.n_bands_values.split(",")]

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")

    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    n_lags = _resolve_cfg_value(args.n_lags, exec_cfg, "N_LAGS", 168)
    corr_threshold = _resolve_cfg_value(args.corr_threshold, exec_cfg, "CORR_THRESHOLD", 0.7)
    neg_corr = _resolve_cfg_value(args.neg_corr, exec_cfg, "NEG_CORR", True)

    print(f"PARAMS: window_size={window_size} window_step={window_step} n_lags={n_lags} "
          f"corr_threshold={corr_threshold} neg_corr={neg_corr} n_vectors={args.n_vectors} "
          f"gamma={args.gamma} b_values={b_values} n_bands_values={n_bands_values}")

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
    all_starts = list(range(0, max_start + 1, window_step))
    lag_count = max(1, int(n_lags // window_step) + 1)
    est_pairs_per_step = int(n_series * (n_series - 1) / 2) + int(n_series * n_series * max(lag_count - 1, 0))
    print(f"Exhaustive sweep: {len(all_starts)} steps, ~{est_pairs_per_step} pairs/step, "
          f"~{len(all_starts) * est_pairs_per_step} total pairs.")

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

    band_specs = build_band_specs(args.n_vectors, b_values, n_bands_values, args.band_seed)
    settings = [(b, nb) for b in b_values for nb in n_bands_values]

    n_baseline = 0
    n_true_pos = 0
    n_survivors = 0
    retrieved = {s: 0 for s in settings}
    retrieved_tp = {s: 0 for s in settings}
    retrieved_surv = {s: 0 for s in settings}

    for step_id, cur_start in enumerate(all_starts):
        min_lag_start = max(0, cur_start - n_lags)
        lag_starts = sorted(set(s for s in range(cur_start, min_lag_start - 1, -window_step) if 0 <= s <= max_start))
        if not lag_starts or cur_start not in lag_starts:
            continue

        norm_by_start = {s: sketch_cache.get(s)[1] for s in lag_starts}
        sign_by_key = {}
        for s in lag_starts:
            for s_idx in range(n_series):
                if not is_valid(s_idx, s):
                    continue
                v = norm_by_start[s].get(proxy_ids[s_idx])
                if v is not None:
                    sign_by_key[(s_idx, s)] = np.asarray(v) >= 0

        # Build per-band posting dicts over the historical pool for this step.
        postings_by_setting = {}
        for b in b_values:
            for nb in n_bands_values:
                dim_sets = band_specs[b][:nb]
                posting = [defaultdict(list) for _ in range(nb)]
                for key, sbits in sign_by_key.items():
                    keys = band_keys(sbits, dim_sets)
                    for j, k in enumerate(keys):
                        posting[j][k].append(key)
                postings_by_setting[(b, nb)] = (posting, dim_sets)

        for s_current in range(n_series):
            if not is_valid(s_current, cur_start):
                continue
            q_key = (s_current, cur_start)
            q_sign = sign_by_key.get(q_key)
            if q_sign is None:
                continue
            x = get_window(s_current, cur_start)
            q_vec = norm_by_start[cur_start].get(proxy_ids[s_current])

            retrieved_sets = {}
            for (b, nb), (posting, dim_sets) in postings_by_setting.items():
                cand = set()
                q_keys = band_keys(q_sign, dim_sets)
                for j, k in enumerate(q_keys):
                    band_width = len(dim_sets[j])
                    complement = k ^ ((1 << band_width) - 1)
                    cand.update(posting[j].get(k, ()))
                    cand.update(posting[j].get(complement, ()))
                retrieved_sets[(b, nb)] = cand

            for lag_start in lag_starts:
                for s_other in range(n_series):
                    if lag_start == cur_start and s_other <= s_current:
                        continue
                    x_key = (s_other, lag_start)
                    if x_key not in sign_by_key:
                        continue
                    y = get_window(s_other, lag_start)
                    corr, _dist = _fast_corr_and_dist(x, y)
                    if not np.isfinite(corr):
                        continue
                    n_baseline += 1
                    is_tp = bool(abs(corr) >= corr_threshold)
                    if is_tp:
                        n_true_pos += 1
                    x_vec = norm_by_start[lag_start].get(proxy_ids[s_other])
                    is_surv = x_vec is not None and abs(float(np.dot(q_vec, x_vec))) >= args.gamma
                    if is_surv:
                        n_survivors += 1
                    for s in settings:
                        if x_key in retrieved_sets[s]:
                            retrieved[s] += 1
                            if is_tp:
                                retrieved_tp[s] += 1
                            if is_surv:
                                retrieved_surv[s] += 1

        if step_id % 50 == 0:
            print(f"  progress: step {step_id}/{len(all_starts)}, {n_baseline} pairs so far", flush=True)

    print(f"\nBaseline: {n_baseline} pairs, {n_true_pos} exact true positives, {n_survivors} sketch survivors.")

    rows = []
    for b, nb in settings:
        n_ret = retrieved[(b, nb)]
        rows.append(dict(
            band_width=b, n_bands=nb,
            n_baseline_pairs=n_baseline,
            n_retrieved=n_ret,
            candidate_rate=n_ret / n_baseline if n_baseline else float("nan"),
            candidate_reduction=1.0 - n_ret / n_baseline if n_baseline else float("nan"),
            recall_of_exact_true_positives=retrieved_tp[(b, nb)] / n_true_pos if n_true_pos else float("nan"),
            recall_of_sketch_survivors=retrieved_surv[(b, nb)] / n_survivors if n_survivors else float("nan"),
        ))
    result = pd.DataFrame(rows).sort_values(["band_width", "n_bands"])
    args.result_folder.mkdir(parents=True, exist_ok=True)
    out_path = args.result_folder / "sign_lsh_band_results.csv"
    result.to_csv(out_path, index=False)
    pd.set_option("display.width", 200)
    print(f"\nWrote {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
