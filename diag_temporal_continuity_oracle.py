"""Temporal-continuity feasibility oracle (2026-07-13 investigation, Stage A +
Stage D reframed along the temporal/lag axis instead of the spatial/
arrival-order axis the 2026-07-06 block-similarity work already tried and
found ineffective on real data).

Context: a proposal to give CorrTrack "temporal certificates" -- skip
pair-lag states whose cached previous sketch-dot score, plus a safe bound on
how much each endpoint's sketch could have moved since it was last computed,
cannot reach gamma -- without recomputing the dot from scratch. This script
measures, on REAL production data (not synthetic), whether that bound is
selective enough to be worth building a real backend around, per this
project's own established discipline (diagnose cheap before implementing
real; see diag_pair_separability.py, diag_sign_lsh_band_index.py, etc.).

Lag-state correspondence (answers Section 4.1 of the original proposal):
CorrTrack's lag is a RELATIVE offset between two windows' end times. As the
stream advances by one window_step, both a pair's "current" window and its
"historical, lag-L" window advance by exactly one window_step together --
so the pair-lag key (i, j, lag) persists identically across steps; only the
absolute window-start indices it refers to change. This means BOTH endpoints
of every (i, j, lag) cell move by one step's worth of sketch movement between
consecutive script iterations -- the two-sided bound applies uniformly,
including for lag=0 (synchronous) cells.

Two-sided norm bound (verified sound, standard Cauchy-Schwarz perturbation
bound for unit vectors -- see docs/implementation_log.md):
    d_ij' <= d_ij + delta_i + delta_j + delta_i*delta_j

Stage D reframed: instead of grouping pair-lag cells by arrival-order/
similarity (2026-07-06, real-data prune rate = 0 every time re-checked),
tile along (series-block x series-block x LAG-block) -- lag directly encodes
a temporal offset, making it the one axis where "nearby cells behave
similarly" is actually a temporal claim, not a spatial one.

(2026-07-13 follow-up) Also stratifies the Stage A skip rate by whether a
cell's PREVIOUS-step exact Pearson correlation already sat in the
"persistent" high-correlation regime (|corr| >= --persistent-corr-threshold,
default 0.95 -- matching the bimodal spike found in the same-day real-data
hard-band diagnostic). Hypothesis: pairs already known to be persistently,
strongly correlated might move less step-to-step than the general
population, since a real underlying relationship (shared weather system,
etc.) constrains how much the sketch can drift while the true correlation
stays high. Exact correlation is computed via a vectorized (BLAS-backed)
Pearson formula over raw window matrices, not the sketch dot -- an
independent ground truth, same identity CorrTrack's own `_fast_corr_and_dist`
uses, just batched over all m^2 pairs at once for speed.

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/diag_temporal_continuity_oracle.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_fr_air_temperature_7_1.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 --neg-corr \\
        --n-vectors 64 --gamma 0.55 --n-steps 200 --persistent-corr-threshold 0.95 \\
        --result-folder tmp_artifacts/temporal_continuity_diag
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize
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
    parser.add_argument("--n-steps", type=int, default=200,
                         help="number of consecutive window-start steps to walk through")
    parser.add_argument("--series-blocks", type=str, default="1,4,8,16",
                         help="comma-separated series-block sizes to sweep for the temporal-tile oracle")
    parser.add_argument("--lag-blocks", type=str, default="1,2,4",
                         help="comma-separated lag-block sizes to sweep for the temporal-tile oracle")
    parser.add_argument("--persistent-corr-threshold", type=float, default=0.95,
                         help="|exact Pearson corr| at/above this, on the PREVIOUS step, marks a cell "
                              "as 'persistent' for stratified skip-rate reporting")
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/temporal_continuity_diag"))
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

    n_lagged_windows = n_lags // window_step + 1
    print(f"PARAMS: dataset={country}/{var} n_series={n_series} length={length_data} "
          f"window={window_size}/{window_step} n_lags={n_lags} n_lagged_windows={n_lagged_windows} "
          f"corr_threshold={corr_threshold} gamma={args.gamma} n_vectors={args.n_vectors} "
          f"n_steps={args.n_steps}")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, args.n_vectors,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )
    raw_values = np.asarray(train_data[1:1 + n_series, :], dtype=np.float64)

    def raw_window_matrix(start):
        return raw_values[:, start:start + window_size]

    def pearson_matrix(X, Y):
        """Vectorized (BLAS-backed) Pearson correlation between every row of
        X and every row of Y -- same identity as _fast_corr_and_dist, batched
        over all m^2 pairs via one matmul instead of per-pair loops."""
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

    max_start = length_data - window_size
    all_starts = list(range(0, max_start + 1, window_step))[:args.n_steps]
    m = n_series
    L_max = n_lagged_windows - 1

    print(f"Walking {len(all_starts)} steps, m={m} series, L_max={L_max} lags...")

    # V_hist[k] holds the (m, n_vectors) normalized sketch matrix for step k,
    # aligned by proxy_ids order. delta_hist[k] holds ||V[k][i]-V[k-1][i]||
    # per series (movement ENDING at step k). Kept for the last L_max+2 steps
    # only (all we ever need to look back).
    keep = L_max + 2
    V_hist = {}
    delta_hist = {}
    D_prev = {lag: None for lag in range(L_max + 1)}  # D_prev[lag] = (m,m) dot matrix from the last time this lag was updated
    corr_prev = {lag: None for lag in range(L_max + 1)}  # exact Pearson, same cadence as D_prev

    total_cells = 0
    skipped_cells = 0
    false_skips = 0
    above_gamma_cells = 0
    delta_samples = []
    margin_samples = []  # gamma - d_ij (old score), for unresolved-rate-by-band reporting

    # persistent-correlation stratification (2026-07-13 follow-up)
    persistent_total = 0
    persistent_skipped = 0
    general_total = 0
    general_skipped = 0

    # temporal-tile oracle accounting
    series_blocks = [int(x) for x in args.series_blocks.split(",") if x and int(x) > 1]
    # (sb=1 is intentionally excluded -- it's identical to the per-cell oracle
    # already reported above, and at large m it's the single most expensive
    # combination to loop over in Python for zero additional information.)
    lag_blocks = [int(x) for x in args.lag_blocks.split(",") if x]
    tile_stats = {(sb, lb): {"total_tiles": 0, "skipped_tiles": 0, "cells_in_skipped_tiles": 0, "cells_total": 0}
                  for sb in series_blocks for lb in lag_blocks}

    for k, cur_start in enumerate(all_starts):
        _raw, norm_by_sid = sketch_cache.get(cur_start)
        V_curr = np.zeros((m, args.n_vectors), dtype=np.float64)
        valid_mask = np.zeros(m, dtype=bool)
        for s_idx, sid in enumerate(proxy_ids):
            v = norm_by_sid.get(sid)
            if v is not None:
                V_curr[s_idx] = v
                valid_mask[s_idx] = True
        V_hist[k] = V_curr

        if k - 1 in V_hist:
            delta = np.linalg.norm(V_curr - V_hist[k - 1], axis=1)
            delta[~valid_mask] = np.inf  # invalid windows can't offer a safe bound
            delta_hist[k] = delta
            delta_samples.extend(delta[np.isfinite(delta)].tolist())
        else:
            delta_hist[k] = np.full(m, np.inf)

        raw_curr = raw_window_matrix(cur_start)

        for lag in range(min(k, L_max) + 1):
            hist_k = k - lag
            if hist_k not in V_hist or hist_k not in delta_hist:
                continue
            V_hist_lag = V_hist[hist_k]
            delta_i = delta_hist[k]       # movement ending at step k (current side)
            delta_j = delta_hist[hist_k]  # movement ending at step k-lag (historical side)

            D_new = V_curr @ V_hist_lag.T  # ground truth, (m, m)
            raw_hist_lag = raw_window_matrix(all_starts[hist_k])
            corr_new = pearson_matrix(raw_curr, raw_hist_lag)  # independent exact ground truth

            if D_prev[lag] is not None:
                bound = D_prev[lag] + delta_i[:, None] + delta_j[None, :] + np.outer(delta_i, delta_j)
                safe_skip = bound < args.gamma
                actually_above = D_new >= args.gamma
                false = safe_skip & actually_above
                if lag == 0:
                    mask = np.zeros((m, m), dtype=bool)
                    mask[np.triu_indices(m, k=1)] = True
                else:
                    mask = np.ones((m, m), dtype=bool)
                mask &= valid_mask[:, None] & valid_mask[None, :]

                n_cells = int(mask.sum())
                total_cells += n_cells
                skipped_cells += int((safe_skip & mask).sum())
                false_skips += int((false & mask).sum())
                above_gamma_cells += int((actually_above & mask).sum())
                margin_samples.extend((args.gamma - D_prev[lag][mask]).tolist())

                # persistent-correlation stratification: was THIS cell already
                # strongly correlated (|corr| >= threshold) on the PREVIOUS
                # step, i.e. the last time it was actually resolved? Uses only
                # information that would have been available at decision time.
                cp = corr_prev[lag]
                if cp is not None:
                    persistent_mask = mask & np.isfinite(cp) & (np.abs(cp) >= args.persistent_corr_threshold)
                    general_mask = mask & ~persistent_mask
                    persistent_total += int(persistent_mask.sum())
                    persistent_skipped += int((safe_skip & persistent_mask).sum())
                    general_total += int(general_mask.sum())
                    general_skipped += int((safe_skip & general_mask).sum())

                # temporal-tile oracle: only meaningful once we have a real bound to test
                for sb in series_blocks:
                    for lb in lag_blocks:
                        if lag % lb != 0:
                            continue  # only evaluate each lag-block once, at its start
                        lag_lo, lag_hi = lag, min(lag + lb, L_max + 1)
                        # gather the max delta_j across every lag in this block (each needs
                        # its own historical step k-l, l in [lag_lo, lag_hi))
                        max_delta_j_block = delta_j.copy()
                        block_prev_max = D_prev[lag].copy()
                        have_full_block = True
                        for l2 in range(lag_lo + 1, lag_hi):
                            hk2 = k - l2
                            if hk2 not in delta_hist or D_prev.get(l2) is None:
                                have_full_block = False
                                break
                            max_delta_j_block = np.maximum(max_delta_j_block, delta_hist[hk2])
                            block_prev_max = np.maximum(block_prev_max, D_prev[l2])
                        if not have_full_block:
                            continue
                        for i0 in range(0, m, sb):
                            i1 = min(i0 + sb, m)
                            for j0 in range(0, m, sb):
                                j1 = min(j0 + sb, m)
                                sub_mask = mask[i0:i1, j0:j1]
                                n_sub = int(sub_mask.sum())
                                if n_sub == 0:
                                    continue
                                tile_di = float(np.max(delta_i[i0:i1]))
                                tile_dj = float(np.max(max_delta_j_block[j0:j1]))
                                tile_prev = float(np.max(block_prev_max[i0:i1, j0:j1]))
                                tile_bound = tile_prev + tile_di + tile_dj + tile_di * tile_dj
                                st = tile_stats[(sb, lb)]
                                st["total_tiles"] += 1
                                st["cells_total"] += n_sub
                                if tile_bound < args.gamma:
                                    st["skipped_tiles"] += 1
                                    st["cells_in_skipped_tiles"] += n_sub

            D_prev[lag] = D_new
            corr_prev[lag] = corr_new

        # prune history we no longer need
        oldest_needed = k - L_max - 1
        for old_k in list(V_hist.keys()):
            if old_k < oldest_needed:
                del V_hist[old_k]
                delta_hist.pop(old_k, None)

        if k % 20 == 0:
            print(f"  step {k}/{len(all_starts)}  total_cells={total_cells} skipped={skipped_cells} "
                  f"false_skips={false_skips}", flush=True)

    delta_samples = np.array(delta_samples)
    margin_samples = np.array(margin_samples)

    print(f"\n=== Stage A: pairwise temporal-certificate oracle ===")
    print(f"total pair-lag evaluations: {total_cells}")
    print(f"safely skipped (bound < gamma): {skipped_cells} ({100*skipped_cells/max(total_cells,1):.2f}%)")
    print(f"unresolved (had to check individually): {total_cells - skipped_cells}")
    print(f"actually above gamma (real candidates): {above_gamma_cells}")
    print(f"FALSE SKIPS (must be zero): {false_skips}")
    if delta_samples.size:
        print(f"delta_i distribution: mean={delta_samples.mean():.4f} median={np.median(delta_samples):.4f} "
              f"p90={np.percentile(delta_samples,90):.4f} p95={np.percentile(delta_samples,95):.4f} "
              f"p99={np.percentile(delta_samples,99):.4f}")
    if margin_samples.size:
        print(f"gamma-d_ij (old-score margin) distribution: mean={margin_samples.mean():.4f} "
              f"median={np.median(margin_samples):.4f} frac_negative(already_above_gamma)="
              f"{float(np.mean(margin_samples<0)):.4f}")

    temporal_skip_rate = skipped_cells / max(total_cells, 1)
    print(f"\nTEMPORAL SKIP RATE = {temporal_skip_rate*100:.2f}%")
    if temporal_skip_rate > 0.95:
        verdict = "excellent"
    elif temporal_skip_rate > 0.90:
        verdict = "very promising"
    elif temporal_skip_rate > 0.75:
        verdict = "possibly useful with cheap tiles"
    else:
        verdict = "likely insufficient"
    print(f"Decision-criteria verdict: {verdict}")

    print(f"\n=== Persistent-correlation stratification (threshold={args.persistent_corr_threshold}) ===")
    persistent_rate = persistent_skipped / max(persistent_total, 1)
    general_rate = general_skipped / max(general_total, 1)
    print(f"persistent stratum (prev |corr| >= {args.persistent_corr_threshold}): "
          f"{persistent_total} cells, skip rate = {persistent_rate*100:.2f}%")
    print(f"general stratum (everything else): {general_total} cells, skip rate = {general_rate*100:.2f}%")
    if persistent_total > 0:
        print(f"lift (persistent / general): {persistent_rate / max(general_rate, 1e-9):.2f}x")

    print(f"\n=== Stage D reframed: temporal-tile oracle (grouped by lag, the temporal axis) ===")
    rows = []
    for (sb, lb), st in tile_stats.items():
        if st["total_tiles"] == 0:
            continue
        tile_skip_rate = st["skipped_tiles"] / st["total_tiles"]
        cell_coverage = st["cells_in_skipped_tiles"] / max(st["cells_total"], 1)
        rows.append(dict(series_block=sb, lag_block=lb, total_tiles=st["total_tiles"],
                          skipped_tiles=st["skipped_tiles"], tile_skip_rate=tile_skip_rate,
                          cells_total=st["cells_total"], cells_in_skipped_tiles=st["cells_in_skipped_tiles"],
                          cell_coverage_by_skipped_tiles=cell_coverage))
    tile_df = pd.DataFrame(rows).sort_values(["lag_block", "series_block"])
    print(tile_df.to_string(index=False))

    out = args.result_folder / "temporal_continuity_summary.csv"
    summary = pd.DataFrame([dict(
        dataset=f"{country}/{var}", n_series=m, n_lagged_windows=n_lagged_windows,
        n_steps=len(all_starts), gamma=args.gamma, total_cells=total_cells,
        skipped_cells=skipped_cells, false_skips=false_skips, temporal_skip_rate=temporal_skip_rate,
        delta_mean=float(delta_samples.mean()) if delta_samples.size else None,
        delta_p95=float(np.percentile(delta_samples, 95)) if delta_samples.size else None,
        persistent_corr_threshold=args.persistent_corr_threshold,
        persistent_cells=persistent_total, persistent_skip_rate=persistent_rate,
        general_cells=general_total, general_skip_rate=general_rate,
    )])
    summary.to_csv(out, index=False)
    tile_out = args.result_folder / "temporal_tile_summary.csv"
    tile_df.to_csv(tile_out, index=False)
    print(f"\nSaved: {out}\nSaved: {tile_out}")


if __name__ == "__main__":
    main()
