"""Fair, PER-BACKEND-TUNED extension of bench_output_sensitivity_table.py
(2026-07-16). Human's ask: "I want to compare the backends fairly, so they
should be tuned to their best before comparing them" -- the untuned table
used ONE fixed hyperparameter setting per approximate backend (see
`backend_kwargs()` in bench_output_sensitivity_table.py) across every
(dataset, m, L) config, which is not a fair "best case" comparison.

Only `instinct`, `lsh_sign_dot`, `circular_grouped_lsh`, `bptree_mixed` have
a real approximation knob to tune -- `bptree`, `sorted_arrays_bs`,
`lsh_hamming_exact` are exact/deterministic (no threshold/approximation
parameter besides the shared `corr_threshold`/`gamma`, which stays fixed for
everyone). For each of the four tunable backends, this script runs a small
per-config sweep and keeps the variant that:

  1. meets `--target-recall` (default 0.95), with the LOWEST relative wall
     time (this backend's wall / sorted_arrays_bs's wall, same config), or
  2. if none clears the recall bar, the variant with the HIGHEST recall
     (reported honestly -- still may not meet target).

Every variant tried is also saved to `<result-folder>/tuning_sweep_detail.csv`
for transparency; the "winner per backend" rows go to the normal
`output_sensitivity_table.csv` schema (same columns as the untuned script),
plus a `tuned_params` column recording which hyperparameter value won.

(2026-07-16) `prefiltered_pairs` sources from `candidate_search_unique_pre_dot_pairs`
(candidates surviving every cheap filter + dedup, right before an internal
cosine pre-check) rather than the coarser `candidate_search_index_candidates` --
see bench_output_sensitivity_table.py's docstring for the full derivation.
`dot_checks` is renamed `dot_valid_pairs` throughout (same underlying value,
`corrtrack.total_candidates` -- the count of candidates valid for, and about
to receive, the final expensive exact Pearson computation).

Usage: same flags as bench_output_sensitivity_table.py, plus
--target-recall (default 0.95).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import corrtrack_param_search as cps
from bench_output_sensitivity_table import n_lagged_windows, run_one, true_dot_computations
from diag_gamma_reliability import _load_module

# ---- per-backend hyperparameter grids (only for backends with a real,
# fairness-relevant approximation knob; bptree/sorted_arrays_bs/
# lsh_hamming_exact are exact and untuned) ----
TUNING_GRID = {
    # (2026-07-17) Sweeps candidate_ann_ef (the best-first traversal budget,
    # "ef_search"), NOT candidate_instinct_entry_points anymore. A direct
    # diagnostic (diag_instinct_entry_points_ceiling.py /
    # diag_instinct_fragmentation_real.py / diag_instinct_ef_search.py,
    # scratch) found entry_points had already plateaued at this project's
    # own benchmarked scales -- recall flat within noise across a 16x range
    # (256->4096) while wall-clock only got worse -- and that permanent
    # graph fragmentation (this project's earlier-established root cause
    # for a DIFFERENT, controlled scenario) explains only 1-9% of the
    # alive population's unreachability here, far short of the ~46-47%
    # recall gap. ef_search was the real, previously-untested lever: fixed
    # at 128 in every prior run, sweeping it alone took recall from
    # ~0.53->0.99 (synth) / ~0.54->0.98 (real) at m=100,L=20. entry_points
    # fixed at 256 (the old sweep's safe upper end -- confirmed to not
    # cost recall at any scale, per the ceiling diagnostic).
    # (2026-07-21g) candidate_instinct_use_nav_proxy is now DEFAULT ON in
    # CorrTrack/Candidates (heap-priority navigation via a cheap sign-
    # Hamming proxy instead of the real dot product, real dot computed
    # lazily only on pop -- see docs/implementation_log.md's 2026-07-21(b)/
    # (f) entries) -- no longer crossed here as a toggle. A follow-up
    # diagnostic (2026-07-21g) found the proxy's coarser ordering needs a
    # HIGHER ef to match the exact-scored traversal's recall at the same ef
    # (e.g. m=100/L=20: proxy off hits recall=0.9456 at ef=512; proxy on
    # needs ef=896 for recall=0.9598) -- but even after paying that ef
    # premium, true dot-product computations are still ~39% LOWER at
    # matched-or-better recall (5.47M vs 8.96M) -- a robust win, not an
    # artifact of comparing at equal ef. The existing ef grid already
    # reaches high enough (1024 clears 0.95 at the hardest tested config)
    # -- no widening needed.
    "instinct": [
        dict(candidate_ann_m=24, candidate_ann_z=96, candidate_ann_ef=ef,
             candidate_instinct_query_mode="threshold", candidate_instinct_entry_points=256,
             candidate_instinct_use_nav_proxy=True)
        for ef in (128, 256, 512, 1024)
    ],
    # (2026-07-20) n_bands widened from {32,64,96,128} -- at 200 steps (vs
    # the 40-60 steps this grid was originally calibrated on), n_bands=128
    # no longer cleared the 0.95 recall target at one config (synth
    # m=100,L=20 -- 0.930), flagged as an honest caveat and left
    # unaddressed at the time. 32 dropped (never won at any tested config)
    # in favor of 160/192 to actually close that gap.
    # (2026-07-21g) candidate_apply_hamming_filter is now DEFAULT ON in
    # CorrTrack/Candidates -- no longer crossed here as a toggle. A
    # follow-up "push max_frac tighter" diagnostic (2026-07-21g) found
    # max_frac=0.40 is close to a hard FLOOR, not a conservative choice
    # with headroom: recall craters sharply below it at every L tested
    # (e.g. m=100/L=20: recall 0.9608 at frac=0.40 -> 0.9270 at frac=0.34,
    # already below the 0.95 target well before frac=0.30). Kept at 0.40,
    # not swept lower. Also checked whether n_bands has room beyond 192:
    # recall keeps climbing with wider n_bands, but since 192 already
    # clears 0.95 at the hardest tested config, going wider only adds
    # dot-product cost for no benefit under the min-relative-wall-among-
    # qualifiers selection rule -- grid range left unchanged.
    "lsh_sign_dot": [
        dict(candidate_lsh_n_bands=nb, candidate_apply_hamming_filter=True,
             candidate_hamming_filter_max_frac=0.40)
        for nb in (64, 96, 128, 160, 192)
    ],
    "circular_grouped_lsh": [
        dict(candidate_circular_group_size=7, candidate_circular_n_groups=ng,
             candidate_circular_sectors=16, candidate_circular_probe_radius=1)
        for ng in (12, 24, 48)
    ],
    "bptree_mixed": [
        dict(candidate_bptree_mixed_n_dims=64, candidate_bptree_mixed_theta=theta)
        for theta in (0.90, 0.91, 0.92, 0.93, 0.95)
    ],
    # (2026-07-22) lsh_grid_dot (SignLSHBandIndex, band_mode="grid") -- the
    # human's original grid-bucketing idea (partition sketch dims into
    # bands, quantize raw values into cells per band) combined with
    # lsh_sign_dot's own AND-within-band/OR-across-band structure instead
    # of a flat "f percent of groups agree" vote. n_bands fixed at 128
    # (this project's own established typical winning band count, per
    # multiple prior tuning-grid results at production scale) while
    # cell_width is swept -- the one genuinely new tunable this backend
    # introduces, with no (m,L,gamma)-driven auto-derivation formula yet
    # (unlike candidate_cosine_threshold's gamma-from-tau relation). Range
    # chosen relative to this project's own established per-coordinate
    # natural spread on mean_l2-normalized K=64 sketches (std~0.124, see
    # the "tree pruning essentially never fires" finding) -- spans well
    # below and well above that scale. See
    # docs/implementation_log.md's 2026-07-22 "lsh_grid_dot" entry.
    "lsh_grid_dot": [
        dict(candidate_lsh_n_bands=128, candidate_grid_cell_width=cw)
        for cw in (0.02, 0.05, 0.1, 0.2, 0.4)
    ],
}
UNTUNED_BACKENDS = ["bptree", "sorted_arrays_bs", "lsh_hamming_exact"]


def run_variant(train_data, proxy_ids, n_series, window_size, window_step, n_lags,
                gamma, corr_threshold, backend, n_steps, n_vectors, kwargs):
    """Same as bench_output_sensitivity_table.run_one but with explicit kwargs
    instead of the untuned backend_kwargs() default."""
    from library_corrtrack_parallel import CorrTrack
    import numpy as np
    import time
    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    common = dict(
        window_size=window_size, basic_window=basic_window, window_step=window_step,
        n_vectors=n_vectors, n_lags=n_lags, grid_dimension=8, cell_size=1.0,
        seed=2468, seed_toggle=1357, freq_threshold=0, corr_threshold=corr_threshold,
        neg_corr=True, exec="sequential", parallel_sketch=False, parallel_candidates=False,
        parallel_validation=False, sketch_norm="mean_l2", candidate_similarity="cosine",
    )
    ct = CorrTrack(candidate_backend=backend, candidate_cosine_threshold=gamma,
                   **common, **kwargs)
    t0 = time.perf_counter()
    for start in range(0, n_steps * window_step, window_step):
        data_step = train_data[:, start:start + window_step].copy()
        data_step[0, :] = np.arange(start, start + data_step.shape[1], dtype=np.int64)
        ct.run(data_step, proxy_ids, verbose=False, testing=False, corr_val=True, monitor=False)
    wall_time = time.perf_counter() - t0
    return dict(
        true_output_pairs=int(ct.validated_candidates),
        enumerated_pairs=int(getattr(ct, "candidate_search_enumerated_candidates", 0)),
        prefiltered_pairs=int(getattr(ct, "candidate_search_unique_pre_dot_pairs", 0)),
        dot_valid_pairs=int(ct.total_candidates),
        true_dot_computations=true_dot_computations(ct, backend, kwargs),
        pair_examinations=int(ct.tested_candidates),
        wall_time_s=wall_time,
        backend_effective=getattr(ct, "candidate_backend_effective", backend),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-step", type=int, default=16)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--n-lags-values", type=str, default="64,144,304")
    parser.add_argument("--n-vectors-values", type=str, default="64")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/output_sensitivity_tuned"))
    args = parser.parse_args()
    args.result_folder.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _load_module(args.dataset_config, "bench_dataset_config")
    cps._apply_dataset_config(dataset_cfg)
    country, var, data, ids = next(cps.iter_datasets())
    n_var = cps._get_cfg_attr(dataset_cfg, "N_VARS", "N_SERIES")
    n_year = cps._get_cfg_attr(dataset_cfg, "N_YEARS", "N_OBS")
    n_var = n_var[0] if isinstance(n_var, (list, tuple)) else n_var
    n_year = n_year[0] if isinstance(n_year, (list, tuple)) else n_year
    train_data, proxy_ids = cps.prepare_training_data(data, ids, n_year, n_var, args.train_ratio)
    proxy_ids = [str(x) for x in proxy_ids]
    n_series = train_data.shape[0] - 1
    print(f"Dataset: {country}/{var} m={n_series}")

    n_lags_values = [int(x) for x in args.n_lags_values.split(",")]
    n_vectors_values = [int(x) for x in args.n_vectors_values.split(",")]

    rows = []
    detail_rows = []
    for n_vectors in n_vectors_values:
        for n_lags in n_lags_values:
            L = n_lagged_windows(n_lags, args.window_step)
            max_pairs = n_series * n_series * L
            print(f"  m={n_series} K={n_vectors} n_lags={n_lags} (L={L}) reference=sorted_arrays_bs ...", flush=True)
            ref = run_one(train_data, proxy_ids, n_series, args.window_size, args.window_step,
                          n_lags, args.gamma, args.corr_threshold, "sorted_arrays_bs", args.n_steps,
                          n_vectors=n_vectors)
            z_ref = ref["true_output_pairs"]
            ref_wall = ref["wall_time_s"]
            print(f"    Z_reference={z_ref} wall={ref_wall:.3f}s")

            def make_row(backend, res, tuned_params):
                z_found = res["true_output_pairs"]
                return dict(
                    m=n_series, K=n_vectors, L=L, n_lags=n_lags, backend=backend,
                    backend_effective=res["backend_effective"], tuned_params=tuned_params,
                    true_output_pairs_reference=z_ref, true_output_pairs_found=z_found,
                    recall_vs_reference=z_found / z_ref if z_ref else float("nan"),
                    enumerated_pairs=res["enumerated_pairs"], prefiltered_pairs=res["prefiltered_pairs"],
                    dot_valid_pairs=res["dot_valid_pairs"], pair_examinations=res["pair_examinations"],
                    # (2026-07-21) true_dot_computations: REAL dot-product count, distinct
                    # from dot_valid_pairs (survivors of the internal dot+gamma pre-check).
                    # See true_dot_computations()'s docstring in bench_output_sensitivity_table.py.
                    true_dot_computations=res.get("true_dot_computations", res["dot_valid_pairs"]),
                    wall_time_s=res["wall_time_s"],
                    relative_wall=res["wall_time_s"] / ref_wall if ref_wall else float("nan"),
                    enum_amplification=res["enumerated_pairs"] / z_ref if z_ref else float("nan"),
                    prefilter_amplification=res["prefiltered_pairs"] / z_ref if z_ref else float("nan"),
                    candidate_amplification=res["dot_valid_pairs"] / z_ref if z_ref else float("nan"),
                    true_dot_amplification=res.get("true_dot_computations", res["dot_valid_pairs"]) / z_ref if z_ref else float("nan"),
                    examinations_per_output=res["pair_examinations"] / z_ref if z_ref else float("nan"),
                    runtime_per_output_ms=1000.0 * res["wall_time_s"] / z_ref if z_ref else float("nan"),
                    output_density=z_ref / max_pairs if max_pairs else float("nan"),
                )

            # untuned/exact backends: one run each
            row = make_row("sorted_arrays_bs", ref, "")
            rows.append(row)
            detail_rows.append(row)
            for backend in UNTUNED_BACKENDS:
                if backend == "sorted_arrays_bs":
                    continue
                print(f"  m={n_series} K={n_vectors} L={L} backend={backend} (untuned) ...", flush=True)
                res = run_one(train_data, proxy_ids, n_series, args.window_size, args.window_step,
                              n_lags, args.gamma, args.corr_threshold, backend, args.n_steps,
                              n_vectors=n_vectors)
                row = make_row(backend, res, "")
                rows.append(row)
                detail_rows.append(row)
                print(f"    recall={row['recall_vs_reference']:.4f} relative_wall={row['relative_wall']:.3f}")

            # tunable backends: sweep, keep the best
            for backend, variants in TUNING_GRID.items():
                best_row = None
                for kwargs in variants:
                    label = ",".join(f"{k.split('_')[-1]}={v}" for k, v in kwargs.items()
                                      if k not in ("candidate_ann_m", "candidate_ann_z", "candidate_instinct_entry_points",
                                                    "candidate_instinct_query_mode", "candidate_circular_group_size",
                                                    "candidate_circular_sectors", "candidate_circular_probe_radius",
                                                    "candidate_bptree_mixed_n_dims", "candidate_hamming_filter_max_frac",
                                                    "candidate_apply_hamming_filter", "candidate_instinct_use_nav_proxy"))
                    print(f"  m={n_series} K={n_vectors} L={L} backend={backend} tuning {label} ...", flush=True)
                    try:
                        res = run_variant(train_data, proxy_ids, n_series, args.window_size, args.window_step,
                                           n_lags, args.gamma, args.corr_threshold, backend, args.n_steps,
                                           n_vectors, kwargs)
                    except Exception as exc:  # noqa: BLE001
                        print(f"    FAILED: {exc}")
                        continue
                    row = make_row(backend, res, label)
                    detail_rows.append(row)
                    print(f"    recall={row['recall_vs_reference']:.4f} relative_wall={row['relative_wall']:.3f}")
                    meets = row["recall_vs_reference"] >= args.target_recall
                    if best_row is None:
                        best_row = row
                    else:
                        best_meets = best_row["recall_vs_reference"] >= args.target_recall
                        if meets and best_meets:
                            if row["relative_wall"] < best_row["relative_wall"]:
                                best_row = row
                        elif meets and not best_meets:
                            best_row = row
                        elif not meets and not best_meets:
                            if row["recall_vs_reference"] > best_row["recall_vs_reference"]:
                                best_row = row
                if best_row is not None:
                    rows.append(best_row)
                    print(f"    -> WINNER {backend}: {best_row['tuned_params']} "
                          f"recall={best_row['recall_vs_reference']:.4f} relative_wall={best_row['relative_wall']:.3f}")

    df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)
    print("\n" + df.to_string(index=False))
    out = args.result_folder / "output_sensitivity_table.csv"
    df.to_csv(out, index=False)
    detail_out = args.result_folder / "tuning_sweep_detail.csv"
    detail_df.to_csv(detail_out, index=False)
    print(f"\nSaved: {out}")
    print(f"Saved: {detail_out}")


if __name__ == "__main__":
    main()
