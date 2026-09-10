"""Output-sensitivity comparison table across backends, varying BOTH m
(series count) and L (n_lagged_windows, via n_lags), on synthetic and real
data (2026-07-15) -- see docs/implementation_log.md's "PROJECT DIRECTION
CHANGE" entry. For each (dataset, m, L, backend) combination, runs the real
compiled CorrTrack pipeline for --n-steps real streaming steps and reports:

  true_output_pairs        = corrtrack.validated_candidates (Z, exact)

  Three metrics, three DIFFERENT, CONSISTENTLY-DEFINED stages of the same
  pipeline -- each means the SAME thing for every backend (2026-07-16,
  corrected a second time; the "prefiltered_pairs" definition below was
  ALSO conflating stages across architectures until this fix -- see
  docs/implementation_log.md's "three consistent enumeration metrics" and
  "prefiltered_pairs really means unique_pre_dot_pairs" entries):

  enumerated_pairs         = corrtrack.candidate_search_enumerated_candidates -- the TRUE number
                             of candidates the search process visited/considered, full stop. For
                             full-scan backends (lsh_hamming_exact, circular_grouped_lsh) this is
                             genuinely alive_count per query (every candidate visited
                             unconditionally) -- the real O(N) cost.
  prefiltered_pairs        = corrtrack.candidate_search_unique_pre_dot_pairs -- candidates that
                             survived EVERY cheap filter (coordinate/bound checks, any backend-
                             specific confirmation-loop/Hamming-distance/sector-match stage, and
                             deduplication) and are about to enter an internal cheap cosine
                             pre-check, the last checkpoint before the expensive final validation.
                             (2026-07-16 fix: previously sourced from
                             candidate_search_index_candidates, a coarser, EARLIER checkpoint that
                             for the "single-stage" backends -- bptree, sorted_arrays_bs,
                             lsh_sign_dot, instinct -- was wrongly set equal to enumerated_pairs,
                             hiding a real ~2-10x cheap-filter cascade those backends do have. Every
                             backend's Cython kernel already computes this exact value; it just
                             wasn't being read. See docs/implementation_log.md.)
  dot_valid_pairs          = corrtrack.total_candidates -- candidates that ALSO survived the
                             internal cheap cosine pre-check and are valid inputs to the FINAL
                             exact Pearson/dot validation (i.e. how many real correlation checks
                             get computed). The LAST stage, always <= prefiltered_pairs. (Renamed
                             from "dot_checks" 2026-07-16 -- the old name implied "checks
                             performed," but a dot check IS performed on every prefiltered_pairs
                             candidate; this number is genuinely a smaller, later-stage headcount,
                             not a rejection count, so "dot_valid_pairs" is the honest name.)
  pair_examinations        = corrtrack.tested_candidates (pairs run through validation -- equals
                             dot_valid_pairs whenever every emitted candidate gets validated, which
                             is every configuration this script runs).
  wall_time_s              = measured wall-clock for the whole run
  enum_amplification       = enumerated_pairs / Z_reference -- "did I basically brute-force it" --
                             the genuine enumeration signal, safe to compare ACROSS architectures.
  candidate_amplification  = dot_valid_pairs / Z_reference -- "how much extra FINAL-VALIDATION
                             work (false positives that reached the expensive exact check) per
                             true output" -- the proportion of exact-validation work spent on
                             pairs that turn out not to be true correlations.
  examinations_per_output  = pair_examinations / Z_reference
  runtime_per_output       = wall_time_s / Z_reference
  output_density           = Z_reference / (m^2 * L)
  recall_vs_reference      = true_output_pairs / Z_reference
  brute_force_wall_time_s  = wall-clock of candidate_backend="brute_force" at the SAME (m, L, K) --
                             a real, no-index, direct-scan reference, run once per (m, L, K)
                             alongside sorted_arrays_bs, reported on every row for direct comparison
  speedup_vs_brute_force   = brute_force_wall_time_s / wall_time_s

(2026-07-15 addendum) `n_vectors` (K, sketch dimension) is now swept too, via
--n-vectors-values, to test the human's hypothesis that amplification drops
as K grows (a real, richer sketch should discriminate true from false
candidates more precisely, tightening the touched-candidate/true-output
ratio) -- not assumed, measured per (m, L, K, backend) combination.

All three counters are the project's own existing, already-cumulative
CorrTrack attributes (not new instrumentation) -- see
docs/implementation_log.md for the exact lines that increment them.

IMPORTANT: each backend's own `validated_candidates` reflects what THAT
backend found, which conflates the backend's own recall with the output-
size question -- approximate backends (lsh_sign_dot, circular_grouped_lsh,
instinct) can silently under-report Z if used as their own denominator.
Per docs/implementation_log.md's own flagged caveat ("amplification must
always be reported alongside recall, never alone"), this script computes
Z_reference ONCE per (m, L) via the exact `sorted_arrays_bs` backend, uses
THAT fixed value as the denominator for every backend's ratios, and reports
each backend's own recall_vs_reference as a separate column.

Usage:
    PYTHONPATH="$(pwd)" python3 -u corrtrack_release_dev/bench_output_sensitivity_table.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --backends bptree,sorted_arrays_bs,instinct,lsh_sign_dot,lsh_hamming_exact,circular_grouped_lsh \\
        --n-lags-values 16,168 --n-steps 60 \\
        --result-folder tmp_artifacts/output_sensitivity_synth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack
from diag_gamma_reliability import _load_module


def n_lagged_windows(n_lags, window_step):
    return int(n_lags) // int(window_step) + 1


def backend_kwargs(backend):
    if backend == "instinct":
        return dict(candidate_ann_m=24, candidate_ann_z=96, candidate_ann_ef=128,
                    candidate_instinct_query_mode="threshold", candidate_instinct_entry_points=64)
    if backend == "lsh_sign_dot":
        return dict(candidate_lsh_n_bands=64)
    if backend == "lsh_hamming_exact":
        return dict()
    if backend == "circular_grouped_lsh":
        return dict(candidate_circular_group_size=7, candidate_circular_n_groups=24,
                    candidate_circular_sectors=16, candidate_circular_probe_radius=1)
    if backend == "bptree_mixed":
        # (2026-07-15, final) n_dims=64 (matching diag_bptree_mixed_ideas.py's
        # original diagnostic) was briefly reduced to n_dims=8 to fight a
        # real wall-clock regression -- reverted once the human clarified
        # the best stable accuracy is at n_dims=64. The real fix was
        # architectural (MultiDimThetaIndex's single-anchor-tree redesign,
        # see its class docstring in candidate_kernels.pyx): n_dims=64/
        # theta=0.9 now costs only 1.18-1.29x plain bptree's wall-clock
        # (down from 2.2-2.7x) at 98.3-98.4% recall, with no accuracy
        # trade-off vs a smaller n_dims -- see docs/implementation_log.md's
        # "bptree_mixed: single-anchor-tree redesign" entry.
        return dict(candidate_bptree_mixed_n_dims=64, candidate_bptree_mixed_theta=0.9)
    return dict()


def true_dot_computations(ct, backend, kwargs=None):
    """Real dot-product computation count -- distinct from dot_valid_pairs
    (candidate_search_dot_valid_pairs / ct.total_candidates), which counts
    SURVIVORS of each backend's internal dot+gamma pre-check, not how many
    real dots were actually computed to get there.

    This distinction is invisible for most backends but matters a lot for
    two 2026-07-21 opt-in features: lsh_sign_dot's candidate_apply_hamming_filter
    gates the real dot computation itself (skips it for far-apart
    candidates) but is tuned to almost never change WHO survives the
    downstream gamma check -- so dot_valid_pairs barely moves even when the
    filter cuts real dot computations substantially. instinct's
    candidate_instinct_use_nav_proxy restructures WHEN the real dot is
    computed (lazily, only for alive pops) rather than how many candidates
    reach validation -- enumerated_pairs (nodes popped) already equals the
    true dot count when the proxy is on, but badly UNDERcounts it when the
    proxy is off (real cost is candidate_search_instinct_nodes_scored, a
    counter that existed since 2026-07-17's root-cause investigation but
    was never wired into CorrTrack's aggregated stats until 2026-07-21 --
    see docs/implementation_log.md's "true dot-product-computation metric"
    entry for the full derivation and the discrepancy this closes).

    (2026-07-21j) hexact_dot_checks/cgrp_dot_checks/mdt_dot_checks were
    already computed by HammingExactIndex/CircularGroupedExactIndex/
    MultiDimThetaIndex and returned by Candidates.candidate_search_stats(),
    but -- same gap instinct_nodes_scored had -- never aggregated onto
    CorrTrack. Wired through (library_corrtrack_parallel.py), closing the
    fallback for those 3 backends too; only real full-scan backends with
    no separate cheap-filter stage at all (nothing left to wire) still
    have no distinct true-dot notion, hence the general fallback below.
    """
    kwargs = kwargs or {}
    if backend == "lsh_sign_dot":
        return int(getattr(ct, "candidate_search_lsh_dot_checks", 0))
    if backend == "lsh_hamming_exact":
        return int(getattr(ct, "candidate_search_hexact_dot_checks", 0))
    if backend == "circular_grouped_lsh":
        return int(getattr(ct, "candidate_search_cgrp_dot_checks", 0))
    if backend == "bptree_mixed":
        return int(getattr(ct, "candidate_search_mdt_dot_checks", 0))
    if backend == "instinct":
        if kwargs.get("candidate_instinct_use_nav_proxy"):
            return int(getattr(ct, "candidate_search_enumerated_candidates", 0))
        return int(getattr(ct, "candidate_search_instinct_nodes_scored", 0))
    if backend in ("bptree", "sorted_arrays_bs"):
        return int(getattr(ct, "candidate_search_dot_checks", 0))
    return int(getattr(ct, "total_candidates", 0))


def run_one(train_data, proxy_ids, n_series, window_size, window_step, n_lags,
            gamma, corr_threshold, backend, n_steps, n_vectors=64):
    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    common = dict(
        window_size=window_size, basic_window=basic_window, window_step=window_step,
        n_vectors=n_vectors, n_lags=n_lags, grid_dimension=8, cell_size=1.0,
        seed=2468, seed_toggle=1357, freq_threshold=0, corr_threshold=corr_threshold,
        neg_corr=True, exec="sequential", parallel_sketch=False, parallel_candidates=False,
        parallel_validation=False, sketch_norm="mean_l2", candidate_similarity="cosine",
    )
    kwargs = backend_kwargs(backend)
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
    parser.add_argument("--n-lags-values", type=str, default="16,168")
    parser.add_argument("--n-vectors-values", type=str, default="64")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--backends", type=str,
                         default="bptree,sorted_arrays_bs,instinct,lsh_sign_dot,lsh_hamming_exact,circular_grouped_lsh")
    parser.add_argument("--skip-brute-force", action="store_true",
                         help="Skip the candidate_backend='brute_force' reference run (it can be slow at large m*L).")
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/output_sensitivity"))
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
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    rows = []
    for n_vectors in n_vectors_values:
        for n_lags in n_lags_values:
            L = n_lagged_windows(n_lags, args.window_step)
            max_pairs = n_series * n_series * L
            print(f"  m={n_series} K={n_vectors} n_lags={n_lags} (L={L}) reference=sorted_arrays_bs ...", flush=True)
            ref = run_one(train_data, proxy_ids, n_series, args.window_size, args.window_step,
                          n_lags, args.gamma, args.corr_threshold, "sorted_arrays_bs", args.n_steps,
                          n_vectors=n_vectors)
            z_ref = ref["true_output_pairs"]
            print(f"    Z_reference={z_ref}")

            bf_wall_time = float("nan")
            if not args.skip_brute_force:
                print(f"  m={n_series} K={n_vectors} n_lags={n_lags} (L={L}) backend=brute_force (brute force) ...", flush=True)
                try:
                    bf_res = run_one(train_data, proxy_ids, n_series, args.window_size, args.window_step,
                                      n_lags, args.gamma, args.corr_threshold, "brute_force", args.n_steps,
                                      n_vectors=n_vectors)
                    bf_wall_time = bf_res["wall_time_s"]
                    print(f"    brute_force_wall_time_s={bf_wall_time:.3f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"    brute force FAILED: {exc}")

            for backend in backends:
                print(f"  m={n_series} K={n_vectors} n_lags={n_lags} (L={L}) backend={backend} ...", flush=True)
                if backend == "sorted_arrays_bs":
                    res = ref
                elif backend == "brute_force":
                    if args.skip_brute_force:
                        continue
                    res = bf_res
                else:
                    try:
                        res = run_one(train_data, proxy_ids, n_series, args.window_size, args.window_step,
                                       n_lags, args.gamma, args.corr_threshold, backend, args.n_steps,
                                       n_vectors=n_vectors)
                    except Exception as exc:  # noqa: BLE001 -- report and continue the sweep
                        print(f"    FAILED: {exc}")
                        continue
                z_found = res["true_output_pairs"]
                row = dict(
                    m=n_series, K=n_vectors, L=L, n_lags=n_lags, backend=backend,
                    backend_effective=res["backend_effective"],
                    true_output_pairs_reference=z_ref,
                    true_output_pairs_found=z_found,
                    recall_vs_reference=z_found / z_ref if z_ref else float("nan"),
                    enumerated_pairs=res["enumerated_pairs"],
                    prefiltered_pairs=res["prefiltered_pairs"],
                    dot_valid_pairs=res["dot_valid_pairs"],
                    pair_examinations=res["pair_examinations"],
                    wall_time_s=res["wall_time_s"],
                    brute_force_wall_time_s=bf_wall_time,
                    speedup_vs_brute_force=bf_wall_time / res["wall_time_s"] if res["wall_time_s"] else float("nan"),
                    enum_amplification=res["enumerated_pairs"] / z_ref if z_ref else float("nan"),
                    prefilter_amplification=res["prefiltered_pairs"] / z_ref if z_ref else float("nan"),
                    candidate_amplification=res["dot_valid_pairs"] / z_ref if z_ref else float("nan"),
                    examinations_per_output=res["pair_examinations"] / z_ref if z_ref else float("nan"),
                    runtime_per_output_ms=1000.0 * res["wall_time_s"] / z_ref if z_ref else float("nan"),
                    output_density=z_ref / max_pairs if max_pairs else float("nan"),
                )
                rows.append(row)
                print(f"    Z_found={z_found} recall={row['recall_vs_reference']:.4f} "
                      f"enumerated={res['enumerated_pairs']} prefiltered={res['prefiltered_pairs']} "
                      f"dot_valid_pairs={res['dot_valid_pairs']} "
                      f"wall={res['wall_time_s']:.3f}s amp={row['candidate_amplification']:.2f} "
                      f"enum_amp={row['enum_amplification']:.2f} speedup_vs_bf={row['speedup_vs_brute_force']:.2f}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = args.result_folder / "output_sensitivity_table.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
