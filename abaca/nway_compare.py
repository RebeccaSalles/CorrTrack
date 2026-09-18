"""N-way comparison of every competitor arm against bruteforce on one dataset config.

Generalizes abaca/fourway_compare.py (kept as the frozen 2026-09-12 script) to the eight arms
of the competitor plan (docs/competitor_implementation_plan.md), with the section 8 policies
applied automatically:

- neg_corr=True run: arms tagged ``not_available`` (ParCorr/CSZ, CorrJoin) are reported N/A
  instead of run; ``enabled_by_us`` / ``specified`` arms run and carry their tag in the output.
- n_lags > 0: TSUBASA and CorrJoin are synchronous-only and are reported N/A.
- Counters (total_candidates / tested / correlated), candidate_precision = correlated / total_candidates
  (precision before validation; the density at T for an all-pairs arm) and candidate_specificity =
  1 - (candidates - TP) / (universe - positives) (the fraction of uncorrelated pair-windows the candidate
  stage rejected) are the primary comparison; wall time is
  secondary. Since 2026-09-17 the competitor indexes run their candidate loops in
  competitor_kernels (Cython); an arm is marked "py" only when that extension is missing.

Arms: bruteforce exact_stomp filcorr tsubasa braid thinbraid corrtrack parcorr csz statstream corrjoin

    python abaca/nway_compare.py --dataset-config experiment_dataset_motes_temperature.py \
        --arms all --window-size 96 --window-step 12 --n-lags 0 --corr-threshold 0.9 \
        --n-obs 4000 --out /tmp/nway_motes.json

CorrTrack's own parameters come from ``--best-params <json>`` (its hyperopt output) when
given, else untuned defaults (stated in the output). Competitor knobs come from
``--competitor-params <dir>`` (abaca/tune_competitors.py output, the CSZ protocol) when a
``best_params_<arm>.json`` exists there, else the paper defaults recorded in
experiment_run_exec_param.py, unless overridden with ``--set key=value``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from pathlib import Path

REPO = os.environ.get("REPO_DIR", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, REPO)
os.chdir(REPO)

import corrtrack_run_bruteforce as bfmod  # noqa: E402
from library_corrtrack_parallel import CorrTrack, run_and_log_bruteforce, run_and_log_corrtrack, _HAVE_COMPETITOR_KERNELS  # noqa: E402
from abaca.dataset_profile import profile_dataset  # noqa: E402
import resource  # noqa: E402

ALL_ARMS = ("bruteforce", "exact_stomp", "filcorr", "tsubasa", "braid", "thinbraid", "corrtrack", "parcorr", "csz", "statstream", "corrjoin")
PATTERN_A = {"bruteforce": "bruteforce", "exact_stomp": "exact_stomp", "filcorr": "filcorr", "tsubasa": "tsubasa", "braid": "braid", "thinbraid": "braid"}
SYNC_ONLY = {"tsubasa", "corrjoin"}
NEG_NOT_AVAILABLE = {"parcorr", "csz", "corrjoin"}
PURE_PYTHON_INDEX = {"parcorr", "csz", "statstream", "corrjoin"}


def _parse_set(items):
    out = {}
    for kv in items or ():
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--window-size", type=int, default=168)
    ap.add_argument("--window-step", type=int, default=12)
    ap.add_argument("--basic-window", type=int, default=None)
    ap.add_argument("--n-lags", type=int, default=0)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--neg-corr", action="store_true")
    ap.add_argument("--monitor", action="store_true",
                    help="(2026-09-18) run CorrTrack's monitoring step after the correlations of EVERY arm; off by default in the "
                         "comparative campaign (monitoring is CorrTrack's own feature and gets its own experiment)")
    ap.add_argument("--preprocess", action="store_true",
                    help="(2026-09-18) run the cell on first differences (returns): applied to the bruteforce truth and to every arm alike, "
                         "and forced onto the hyperopt/tuned parameters so no arm sees a different problem")
    ap.add_argument("--n-series", type=int, default=None, help="override the config's N_SERIES/N_VARS[0]")
    ap.add_argument("--n-obs", type=int, default=None, help="override the config's N_OBS/N_YEARS[0]")
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--best-params", default=None, help="CorrTrack hyperopt best_params_corrtrack.json (required for the corrtrack arm unless --allow-untuned)")
    ap.add_argument("--allow-untuned", action="store_true",
                    help="run the corrtrack arm with default parameters (plumbing smoke only; the output is labelled UNTUNED and is not quotable)")
    ap.add_argument("--n-vectors", type=int, default=32)
    ap.add_argument("--competitor-params", default=None,
                    help="directory holding best_params_<arm>.json from abaca/tune_competitors.py (CSZ protocol); "
                         "when a file exists for an arm its knobs replace the paper defaults")
    ap.add_argument("--set", action="append", default=[], help="competitor knob override key=value (json), e.g. parcorr_c=0.5")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    knobs = _parse_set(args.set)
    arms = list(ALL_ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    if "bruteforce" not in arms:
        arms.insert(0, "bruteforce")

    cfg_dataset = bfmod._load_dataset_config(Path(args.dataset_config))
    bfmod._apply_dataset_config(cfg_dataset)
    country, var, data, ids = next(bfmod.iter_datasets())
    n_obs = args.n_obs if args.n_obs is not None else bfmod.N_YEARS[0]
    n_var = args.n_series if args.n_series is not None else bfmod.N_VARS[0]
    train_ratio = args.train_ratio if args.train_ratio is not None else bfmod.TRAIN_RATIO
    test_data, ids_n_var = bfmod.prepare_test_data(data, ids, n_obs, n_var, train_ratio, tuning_mode="sampling")
    label = args.label or f"{bfmod._dataset_slug(country, var)}_{len(ids_n_var)}_{test_data.shape[1]}"
    print(f"dataset: {label} n_series={len(ids_n_var)} n_obs={test_data.shape[1]} W={args.window_size} step={args.window_step} "
          f"n_lags={args.n_lags} T={args.corr_threshold} neg_corr={args.neg_corr} preprocess={args.preprocess} monitor={args.monitor}", flush=True)

    base = dict(window_size=args.window_size, window_step=args.window_step, basic_window=args.basic_window, n_lags=args.n_lags,
                corr_threshold=args.corr_threshold, neg_corr=args.neg_corr, preprocess=bool(args.preprocess), exec="sequential", parallel_sketch=False,
                parallel_candidates=False, parallel_validation=False, max_workers=0, monitor=bool(args.monitor), track_min_dist=True,
                artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
                save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")
    if args.best_params and os.path.exists(args.best_params):
        ct_params = json.load(open(args.best_params))
        ct_source = args.best_params
    elif "corrtrack" in arms and not args.allow_untuned:
        # (2026-09-18, user) CorrTrack is tuned by its own hyperopt in every execution; an untuned
        # corrtrack row is a smoke of the plumbing, never a result, so it must be asked for explicitly
        raise SystemExit("the corrtrack arm needs --best-params <optim>/best_params_corrtrack.json (run corrtrack_param_search.py / "
                         "hyperopt_corrtrack.oar for this cell first); pass --allow-untuned only for a plumbing smoke")
    else:
        ct_params = dict(n_vectors=args.n_vectors, seed=2468, seed_toggle=1357, preprocess=False)
        ct_source = "UNTUNED defaults (--allow-untuned: smoke, not quotable)"
    ct_params = dict(ct_params, preprocess=bool(args.preprocess))        # the cell's preprocess wins over the hyperopt file's
    common = dict(n_vectors=args.n_vectors, seed=2468, seed_toggle=1357, preprocess=bool(args.preprocess))
    pattern_b = {
        "corrtrack": ct_params,
        "parcorr": dict(common, data_representation="sketch_proj", candidate_backend="parcorr_grid", parcorr_k=knobs.get("parcorr_k", 2),
                        parcorr_f=knobs.get("parcorr_f", 0.7), parcorr_c=knobs.get("parcorr_c", 0.7), parcorr_neighbor_probe=False),
        "csz": dict(common, data_representation="sketch_proj", candidate_backend="parcorr_grid", parcorr_k=knobs.get("parcorr_k", 2),
                    parcorr_f=knobs.get("parcorr_f", 0.7), parcorr_c=knobs.get("parcorr_c", 0.7), parcorr_neighbor_probe=True),
        "statstream": dict(common, data_representation="sketch_dft", candidate_backend="statstream_grid",
                           statstream_n_coeffs=knobs.get("statstream_n_coeffs", 16), statstream_index_dims=knobs.get("statstream_index_dims", 4)),
        "corrjoin": dict(common, data_representation="sketch_paa_svd", candidate_backend="corrjoin_double_filter",
                         corrjoin_ks=knobs.get("corrjoin_ks", 15), corrjoin_ke=knobs.get("corrjoin_ke", 30), corrjoin_kb=knobs.get("corrjoin_kb", 3)),
    }
    tuned = {}
    if args.competitor_params:
        for arm in pattern_b:
            f = Path(args.competitor_params) / f"best_params_{arm}.json"
            if f.exists():
                bp = {k: v for k, v in json.load(open(f)).items() if not k.startswith("_")}
                pattern_b[arm] = dict(pattern_b[arm], **bp, preprocess=bool(args.preprocess))
                tuned[arm] = str(f)
    pattern_a_extra = {
        "filcorr": dict(filcorr_fs=knobs.get("filcorr_fs", 0.0), filcorr_ft=knobs.get("filcorr_ft", 0.5), filcorr_sampling_rate=knobs.get("filcorr_sampling_rate", 1.0)),
        "braid": dict(braid_b=knobs.get("braid_b", 16), braid_gamma=knobs.get("braid_gamma", 0.4), braid_thin=False, braid_report_mode=knobs.get("braid_report_mode", "all_lags")),
        "thinbraid": dict(braid_b=knobs.get("braid_b", 16), braid_gamma=knobs.get("braid_gamma", 0.4), braid_thin=True, braid_thin_d0=knobs.get("braid_thin_d0", 400), braid_report_mode=knobs.get("braid_report_mode", "all_lags")),
    }

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm in arms:
            if args.neg_corr and arm in NEG_NOT_AVAILABLE:
                results[arm] = {"status": "N/A", "reason": "negative correlation not available in the method (section 8 item 1)"}
                print(f"{arm:12s} N/A (neg_corr not available)", flush=True)
                continue
            if args.n_lags > 0 and arm in SYNC_ONLY:
                results[arm] = {"status": "N/A", "reason": "synchronous-only method, n_lags > 0"}
                print(f"{arm:12s} N/A (synchronous only)", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                if arm in PATTERN_A:
                    cfg = dict(base, baseline_mode=PATTERN_A[arm], **pattern_a_extra.get(arm, {}))
                    record, _, flags = run_and_log_bruteforce(label, test_data, ids_n_var, cfg, f"{tmp}/{arm}.csv", metadata={"nodes": 0},
                                                              recall_by_window=True, verbose=False, testing=False)
                else:
                    record, _, flags = run_and_log_corrtrack(label, test_data, ids_n_var, base, pattern_b[arm], f"{tmp}/{arm}.csv",
                                                             metadata={"nodes": 0, "alg": arm}, recall_by_window=True, corr_val=True,
                                                             monitor=bool(args.monitor), verbose=False, testing=False)
            except Exception as exc:  # noqa: BLE001 - one failing arm must not kill the battery
                results[arm] = {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
                print(f"{arm:12s} ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            wall = time.perf_counter() - t0
            r = {k: record.get(k) for k in ("correlated", "total_candidates", "tested", "candidate_precision", "sk_time", "cand_time", "val_time", "monit_time",
                                            "n_steps", "step_time_min", "step_time_q1", "step_time_median", "step_time_q3", "step_time_max",
                                            "step_time_whisker_lo", "step_time_whisker_hi", "step_time_outliers", "step_time_mean",
                                            "candidate_search_entries_touched", "candidate_search_blocks_touched", "lsh_candidates_touched",
                                            "supports_neg_corr", "n_vectors", "candidate_backend", "data_representation",
                                            "parcorr_k", "parcorr_f", "parcorr_c", "parcorr_cell_size", "statstream_n_coeffs", "statstream_index_dims",
                                            "statstream_eps", "corrjoin_ks", "corrjoin_ke", "corrjoin_kb", "corrjoin_eps1", "corrjoin_eps2",
                                            "braid_b", "braid_gamma", "braid_thin", "filcorr_fs", "filcorr_ft")}
            # process peak RSS (MB) after the arm: monotone across arms, so report the increment as a coarse memory signal
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            # (2026-09-17) the competitor indexes run their hot loops in competitor_kernels when it is built
            r.update(status="ok", wall=wall, flags=flags, pure_python_index=(arm in PURE_PYTHON_INDEX and not _HAVE_COMPETITOR_KERNELS), peak_rss_mb_after=rss,
                     candidate_time_per_pair_window_us=(1e6 * record["cand_time"] / record["total_candidates"]) if record.get("total_candidates") else None)
            results[arm] = r
            print(f"{arm:12s} done: wall={wall:.2f}s correlated={r['correlated']} total={r['total_candidates']} tested={r['tested']} "
                  f"neg_corr_tag={r['supports_neg_corr']}", flush=True)

    bf = results["bruteforce"]
    try:
        meta = None
        loader = getattr(bfmod.DATA_LOADER, "keywords", {}) or {}
        if loader.get("name"):
            from datasets.competitor_loader import load_raw
            meta = load_raw(loader["name"])[2]
    except Exception:  # noqa: BLE001
        meta = None
    prof_data = test_data
    if args.preprocess:                                   # the arms see first differences; profile that space
        prof_data = np.vstack([test_data[:1, 1:], np.diff(np.asarray(test_data[1:], dtype=np.float64), axis=1)])
    profile = profile_dataset(prof_data, args.window_size, args.window_step, bf_record=bf if bf.get("status") == "ok" else None, meta=meta)
    profile["preprocess"] = bool(args.preprocess)
    print(f"\ndataset profile: density@T={profile.get('density_at_threshold')} low-freq energy share={profile['low_frequency_energy_share_mean']} "
          f"(white noise {profile['white_noise_reference']:.3f}) lag1 autocorr={profile['lag1_autocorr_mean']} constant windows={profile['constant_window_fraction']:.4f}", flush=True)
    for arm, r in results.items():
        if r.get("status") != "ok" or arm == "bruteforce":
            continue
        m = CorrTrack.compute_metrics_bf(r["flags"], bf["flags"], windows=True, total_pairs_bf=bf.get("total_candidates") or bf.get("tested"))
        r.update(precision=m.get("precision"), recall=m.get("recall"), f1=m.get("f1_score"))
        # (2026-09-18, user) candidate-stage specificity: of the pair-windows that are NOT correlated at T
        # (the bruteforce universe minus its positives), the fraction the arm did not put forward as a
        # candidate. FP = candidates - true positives (validation is exact, so the arm's correlated count is
        # its TP); specificity = 1 - FP / (U - P). 1.0 means the candidate stage never wastes a validation.
        U = int(bf.get("total_candidates") or bf.get("tested") or 0); P = int(bf.get("correlated") or 0)
        tp = int(r.get("correlated") or 0); fp = max(int(r.get("total_candidates") or 0) - tp, 0)
        r.update(universe_pair_windows=U, positives=P, candidate_false_positives=fp,
                 candidate_specificity=(1.0 - fp / (U - P)) if U > P else None, candidate_fpr=(fp / (U - P)) if U > P else None)

    print("\n=== SUMMARY ===")
    print(f"{'arm':12s} {'status':>6s} {'wall_s':>8s} {'speedup':>8s} {'correlated':>11s} {'total_cand':>11s} {'tested':>9s} {'recall':>7s} {'cand_prec':>9s} {'cand_spec':>9s} {'step_med_ms':>11s} {'step_q3_ms':>10s} {'neg_corr_tag':>14s} {'index':>6s}")
    for arm in arms:
        r = results[arm]
        if r["status"] != "ok":
            print(f"{arm:12s} {r['status']:>6s}  {r['reason']}")
            continue
        sp = bf["wall"] / r["wall"] if r["wall"] else float("nan")
        rec = f"{r['recall']:.4f}" if r.get("recall") is not None else "   -  "
        prec = f"{r['candidate_precision']:.4f}" if r.get("candidate_precision") is not None else "    -    "
        spec = f"{r['candidate_specificity']:.4f}" if r.get("candidate_specificity") is not None else "    -    "
        p50 = f"{1e3 * r['step_time_median']:.3f}" if r.get('step_time_median') is not None else "-"
        p99 = f"{1e3 * r['step_time_q3']:.3f}" if r.get('step_time_q3') is not None else "-"
        print(f"{arm:12s} {'ok':>6s} {r['wall']:8.2f} {sp:7.2f}x {r['correlated']:11d} {r['total_candidates']:11d} {r['tested']:9d} {rec:>7s} {prec:>9s} {spec:>9s} {p50:>11s} {p99:>10s} "
              f"{str(r['supports_neg_corr']):>14s} {'py' if r['pure_python_index'] else 'cy/np':>6s}")
    print(f"\ncorrtrack params: {ct_source}; competitor knob overrides: {knobs or 'none'}; "
          f"CSZ-protocol tuned arms: {tuned or 'none (paper defaults)'}")
    if args.out:
        out = {"dataset": label, "dataset_profile": profile, "config": vars(args), "corrtrack_params_source": ct_source, "competitor_params_tuned": tuned,
               "arms": {a: {k: v for k, v in r.items() if k != "flags"} for a, r in results.items()}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
