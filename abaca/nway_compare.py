"""N-way comparison of every competitor arm against bruteforce on one dataset config.

Generalizes abaca/fourway_compare.py (kept as the frozen 2026-09-12 script) to the eight arms
of the competitor plan (docs/competitor_implementation_plan.md), with the section 8 policies
applied automatically:

- neg_corr=True run: arms tagged ``not_available`` (ParCorr/CSZ, CorrJoin) are reported N/A
  instead of run; ``enabled_by_us`` / ``specified`` arms run and carry their tag in the output.
- n_lags > 0: every arm runs; the four whose papers are synchronous (ParCorr, CSZ, CorrJoin, TSUBASA)
  carry supports_lags="enabled_by_us" in their rows, the disclosure that the lagged capability is ours.
- Counters (total_candidates / tested / correlated), candidate_precision = correlated / total_candidates
  (precision before validation; the density at T for an all-pairs arm) and candidate_specificity =
  1 - (candidates - TP) / (universe - positives) (the fraction of uncorrelated pair-windows the candidate
  stage rejected) are the primary comparison; wall time is
  secondary. Since 2026-09-17 the competitor indexes run their candidate loops in
  competitor_kernels (Cython); an arm is marked "py" only when that extension is missing.

Arms: bruteforce bf_incremental filcorr tsubasa braid thinbraid corrtrack parcorr csz statstream corrjoin

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
from library_corrtrack_parallel import CorrTrack, run_and_log_bruteforce, run_and_log_corrtrack, _HAVE_COMPETITOR_KERNELS, NumericCorrelatedFlags, _canonicalize_rows, _rows_as_void_keys  # noqa: E402
from abaca.dataset_profile import profile_dataset  # noqa: E402
from abaca.resource_probe import run_isolated, idle_power  # noqa: E402
import socket  # noqa: E402

ALL_ARMS = ("bruteforce", "bf_incremental", "filcorr", "tsubasa", "braid", "thinbraid", "corrtrack", "corrtrack_hamming", "parcorr", "csz", "statstream", "corrjoin")
# (2026-09-21, user) corrtrack_hamming: CorrTrack with the lsh_hamming_exact backend (hamming_exact + dot gate), tuned by its own
# hyperopt on experiment_run_param_grid_campaign_hamming.py, so the tables carry both backends like the m=500 sweeps
PATTERN_A = {"bruteforce": "bruteforce", "bf_incremental": "bf_incremental", "filcorr": "filcorr", "tsubasa": "tsubasa", "braid": "braid", "thinbraid": "braid"}
# (2026-09-23, user) TSUBASA and CorrJoin now run on lagged cells too: neither needed a new mechanism
# (CorrJoin's filters are time-agnostic; TSUBASA's Lemma 1 holds between segments shifted by whole
# basic windows), so the four pruning-style arms are treated alike, with the extension disclosed per
# arm through supports_lags ("enabled_by_us"). Nothing is synchronous-only any more; the set is kept
# so a future arm can declare itself so.
SYNC_ONLY: set[str] = set()
NEG_NOT_AVAILABLE = {"parcorr", "csz", "corrjoin"}
PURE_PYTHON_INDEX = {"parcorr", "csz", "statstream", "corrjoin"}
# arms whose REPORTED correlation is approximate by design (their `correlated` is what they reported, not a TP count)
APPROX_REPORT = {"braid", "thinbraid", "statstream"}
# (2026-09-19) memory: correlated sets above this size travel child -> parent as memory-mapped .npy files; the
# budget is the node's RAM minus headroom (mercantour3: 192 GB), used only to flag a cell in the log/JSON
LARGE_SET_BYTES = int(float(os.environ.get("NWAY_LARGE_SET_GB", "1")) * 2**30)
MEMORY_BUDGET_GB = float(os.environ.get("NWAY_MEMORY_BUDGET_GB", "150"))


def _parse_set(items):
    out = {}
    for kv in items or ():
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _metrics_partitioned(pred_flags, gt_flags, n_parts: int) -> dict:
    """(2026-09-19, memory) recall / precision / F1 of two large correlated sets computed in `n_parts` partitions
    by min(s1, s2) mod n_parts (invariant under the canonical pair order), so the 40-byte sort keys and their
    unique/isin temporaries (~3x the sets) exist for one partition at a time. Same arithmetic as
    CorrTrack.compute_metrics_bf_numeric for the positive-correlation case: unique canonical rows on both sides,
    tp = |pred and gt|."""
    p_rows, _ = pred_flags.correlated_rows(); g_rows, _ = gt_flags.correlated_rows()
    p_rows = np.asarray(p_rows); g_rows = np.asarray(g_rows)
    p_key = np.minimum(p_rows[:, 0], p_rows[:, 1]) % n_parts if p_rows.shape[0] else np.zeros(0, dtype=np.int64)
    g_key = np.minimum(g_rows[:, 0], g_rows[:, 1]) % n_parts if g_rows.shape[0] else np.zeros(0, dtype=np.int64)
    tp = n_p = n_g = 0
    for k in range(n_parts):
        pk = _rows_as_void_keys(_canonicalize_rows(np.ascontiguousarray(p_rows[p_key == k], dtype=np.int64))) if p_rows.shape[0] else np.zeros(0)
        gk = _rows_as_void_keys(_canonicalize_rows(np.ascontiguousarray(g_rows[g_key == k], dtype=np.int64))) if g_rows.shape[0] else np.zeros(0)
        pk = np.unique(pk) if pk.size else pk; gk = np.unique(gk) if gk.size else gk
        n_p += int(pk.size); n_g += int(gk.size)
        if pk.size and gk.size:
            tp += int(np.count_nonzero(np.isin(pk, gk, assume_unique=True)))
        del pk, gk
    precision = tp / n_p if n_p else 0.0
    recall = tp / n_g if n_g else (1.0 if n_p == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1, "partitions": n_parts}


def _score_against_bruteforce(arm: str, r: dict, flags, bf: dict, bf_flags) -> None:
    """recall / precision / F1 of the arm's correlated set against bruteforce's, and the candidate-stage specificity."""
    total_bytes = _correlated_set_bytes(flags) + _correlated_set_bytes(bf_flags)
    if total_bytes > LARGE_SET_BYTES and hasattr(flags, "correlated_rows") and hasattr(bf_flags, "correlated_rows"):
        n_parts = 1
        while n_parts * LARGE_SET_BYTES < 3 * total_bytes:
            n_parts *= 2
        m = _metrics_partitioned(flags, bf_flags, max(8, n_parts))
        r["metrics_partitions"] = m["partitions"]
    else:
        m = CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf.get("total_candidates") or bf.get("tested"))
    r.update(precision=m.get("precision"), recall=m.get("recall"), f1=m.get("f1_score"))
    # (2026-09-18, user) candidate-stage specificity: of the pair-windows that are NOT correlated at T
    # (the bruteforce universe minus its positives), the fraction the arm did not put forward as a
    # candidate. FP = candidates - true positives; specificity = 1 - FP / (U - P). 1.0 means the
    # candidate stage never wastes a validation. TP = recall * P (2026-09-19: for an arm that reports an
    # approximate correlation, BRAID/ThinBRAID/StatStream, `correlated` is what it REPORTED, not its TP)
    U = int(bf.get("total_candidates") or bf.get("tested") or 0); P = int(bf.get("correlated") or 0)
    cand = int(r.get("total_candidates") or 0)
    lower_bound = False
    if cand >= U:                                   # no candidate stage: every negative is put forward
        fp = U - P
    elif arm in APPROX_REPORT:                      # reported TP <= candidate TP, so this FP is an upper bound
        fp = max(cand - int(round((m.get("recall") or 0.0) * P)), 0); lower_bound = True
    else:                                           # exact validation: the arm's correlated count is its TP
        fp = max(cand - int(r.get("correlated") or 0), 0)
    r.update(universe_pair_windows=U, positives=P, candidate_false_positives=fp, candidate_specificity_is_lower_bound=lower_bound,
             candidate_specificity=(1.0 - fp / (U - P)) if U > P else None, candidate_fpr=(fp / (U - P)) if U > P else None)


def _correlated_set_bytes(flags) -> int:
    """size of a NumericCorrelatedFlags (rows + corrs) or a legacy dict, for the memory log."""
    try:
        rows, corrs = flags.correlated_rows()
        return int(np.asarray(rows).nbytes + np.asarray(corrs).nbytes)
    except Exception:  # noqa: BLE001
        try:
            return int(len(flags)) * 64
        except Exception:  # noqa: BLE001
            return 0


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
    ap.add_argument("--eval-span", choices=("holdout", "full"), default="holdout",
                    help="(2026-09-21, user) which rows the arms run on: the holdout after TRAIN_RATIO (campaign default) "
                         "or the whole stream; tuning always used the first TRAIN_RATIO, so 'full' includes the calibration span")
    ap.add_argument("--best-params", default=None, help="CorrTrack hyperopt best_params_corrtrack.json (required for the corrtrack arm unless --allow-untuned)")
    ap.add_argument("--best-params-hamming", default=None, help="hyperopt best_params_corrtrack.json of the lsh_hamming_exact grid (required for the corrtrack_hamming arm unless --allow-untuned)")
    ap.add_argument("--allow-untuned", action="store_true",
                    help="run the corrtrack arm with default parameters (plumbing smoke only; the output is labelled UNTUNED and is not quotable)")
    ap.add_argument("--n-vectors", type=int, default=32)
    ap.add_argument("--competitor-params", default=None,
                    help="directory holding best_params_<arm>.json from abaca/tune_competitors.py (CSZ protocol); "
                         "when a file exists for an arm its knobs replace the paper defaults")
    ap.add_argument("--set", action="append", default=[], help="competitor knob override key=value (json), e.g. parcorr_c=0.5")
    ap.add_argument("--no-isolate", action="store_true", help="run the arms in-process (debugging); memory figures are then monotone across arms")
    ap.add_argument("--rss-interval", type=float, default=0.05, help="RSS sampling interval (s) for the mean-RSS figure")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--cell", default=None, help="campaign cell name, stored in the JSON for the aggregator")
    args = ap.parse_args()
    knobs = _parse_set(args.set)
    # (2026-09-23) `exact_stomp` is the pre-rename id of `bf_incremental`; accepted so older job scripts,
    # manifests and notebooks keep running (the library's baseline-mode resolver does the same)
    _ARM_ALIASES = {"exact_stomp": "bf_incremental", "stomp": "bf_incremental", "incremental": "bf_incremental"}
    arms = (list(ALL_ARMS) if args.arms == "all"
            else [_ARM_ALIASES.get(a.strip(), a.strip()) for a in args.arms.split(",") if a.strip()])
    if "bruteforce" not in arms:
        arms.insert(0, "bruteforce")

    cfg_dataset = bfmod._load_dataset_config(Path(args.dataset_config))
    bfmod._apply_dataset_config(cfg_dataset)
    country, var, data, ids = next(bfmod.iter_datasets())
    n_obs = args.n_obs if args.n_obs is not None else bfmod.N_YEARS[0]
    n_var = args.n_series if args.n_series is not None else bfmod.N_VARS[0]
    train_ratio = args.train_ratio if args.train_ratio is not None else bfmod.TRAIN_RATIO
    test_data, ids_n_var = bfmod.prepare_test_data(data, ids, n_obs, n_var, train_ratio, tuning_mode="sampling")
    if args.eval_span == "full":
        # the whole stream (same rows and series selection as prepare_test_data, without the train_ratio cut)
        rows = bfmod._select_rows(data.shape[0], n_obs)
        test_data = np.transpose(np.c_[data[rows, 0], data[rows, 1:(n_var + 1)]])
    label = args.label or f"{bfmod._dataset_slug(country, var)}_{len(ids_n_var)}_{test_data.shape[1]}"
    print(f"dataset: {label} n_series={len(ids_n_var)} n_obs={test_data.shape[1]} W={args.window_size} step={args.window_step} "
          f"n_lags={args.n_lags} T={args.corr_threshold} neg_corr={args.neg_corr} preprocess={args.preprocess} monitor={args.monitor} eval_span={args.eval_span}", flush=True)

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
    if args.best_params_hamming and os.path.exists(args.best_params_hamming):
        ct_params_h = json.load(open(args.best_params_hamming)); ct_source_h = args.best_params_hamming
    elif "corrtrack_hamming" in arms and not args.allow_untuned:
        raise SystemExit("the corrtrack_hamming arm needs --best-params-hamming <optim>/best_params_corrtrack.json from the "
                         "lsh_hamming_exact hyperopt grid; pass --allow-untuned only for a plumbing smoke")
    else:
        ct_params_h = dict(n_vectors=args.n_vectors, seed=2468, seed_toggle=1357, candidate_backend="lsh_hamming_exact")
        ct_source_h = "UNTUNED defaults (--allow-untuned: smoke, not quotable)"
    ct_params_h = dict(ct_params_h, preprocess=bool(args.preprocess), candidate_backend="lsh_hamming_exact")
    common = dict(n_vectors=args.n_vectors, seed=2468, seed_toggle=1357, preprocess=bool(args.preprocess))
    # (2026-09-24) StatStream's digest is 2n real values and Lemma 7 needs 2n <= W, which the sketch raises
    # on: at W=30 the paper-default n=16 makes the arm fail outright rather than report a row. n is capped
    # here at the largest legal digest for THIS cell, and the clamp is printed, because a silently reduced
    # knob is exactly the kind of thing that must not travel into a table unannounced. Tuned values are
    # clamped too: a best_params file fitted at another window would otherwise take the arm down mid-run.
    ss_n_max = max(1, int(args.window_size) // 2)
    ss_n_coeffs = int(knobs.get("statstream_n_coeffs", 16))
    if ss_n_coeffs > ss_n_max:
        print(f"statstream: n_coeffs {ss_n_coeffs} -> {ss_n_max} (Lemma 7 needs 2n <= W={args.window_size})", flush=True)
        ss_n_coeffs = ss_n_max
    pattern_b = {
        "corrtrack": ct_params,
        "corrtrack_hamming": ct_params_h,
        "parcorr": dict(common, data_representation="sketch_proj", candidate_backend="parcorr_grid", parcorr_k=knobs.get("parcorr_k", 2),
                        parcorr_f=knobs.get("parcorr_f", 0.7), parcorr_c=knobs.get("parcorr_c", 0.7), parcorr_neighbor_probe=False),
        "csz": dict(common, data_representation="sketch_proj", candidate_backend="parcorr_grid", parcorr_k=knobs.get("parcorr_k", 2),
                    parcorr_f=knobs.get("parcorr_f", 0.7), parcorr_c=knobs.get("parcorr_c", 0.7), parcorr_neighbor_probe=True),
        "statstream": dict(common, data_representation="sketch_dft", candidate_backend="statstream_grid",
                           statstream_n_coeffs=ss_n_coeffs, statstream_index_dims=knobs.get("statstream_index_dims", 4),
                           **{k: knobs[k] for k in ("statstream_report", "statstream_bw_coeffs", "statstream_tolerance") if k in knobs}),
        "corrjoin": dict(common, data_representation="sketch_paa_svd", candidate_backend="corrjoin_double_filter",
                         corrjoin_ks=knobs.get("corrjoin_ks", 15), corrjoin_ke=knobs.get("corrjoin_ke", 30), corrjoin_kb=knobs.get("corrjoin_kb", 3)),
    }
    tuned = {}
    if args.competitor_params:
        for arm in pattern_b:
            f = Path(args.competitor_params) / f"best_params_{arm}.json"
            if f.exists():
                # the tuned file may carry its own preprocess (the tuning ran in the cell's space); the cell's value wins
                bp = {k: v for k, v in json.load(open(f)).items() if not k.startswith("_") and k != "preprocess"}
                if arm == "statstream" and int(bp.get("statstream_n_coeffs", 0)) > ss_n_max:
                    # a file fitted at another window: clamp rather than let Lemma 7 take the arm down
                    print(f"statstream: tuned n_coeffs {bp['statstream_n_coeffs']} -> {ss_n_max} "
                          f"(Lemma 7 needs 2n <= W={args.window_size})", flush=True)
                    bp["statstream_n_coeffs"] = ss_n_max
                pattern_b[arm] = dict(pattern_b[arm], **bp, preprocess=bool(args.preprocess))
                tuned[arm] = str(f)
    pattern_a_extra = {
        "filcorr": dict(filcorr_fs=knobs.get("filcorr_fs", 0.0), filcorr_ft=knobs.get("filcorr_ft", 0.5), filcorr_sampling_rate=knobs.get("filcorr_sampling_rate", 1.0)),
        "braid": dict(braid_b=knobs.get("braid_b", 16), braid_gamma=knobs.get("braid_gamma", 0.4), braid_thin=False, braid_report_mode=knobs.get("braid_report_mode", "all_lags")),
        "thinbraid": dict(braid_b=knobs.get("braid_b", 16), braid_gamma=knobs.get("braid_gamma", 0.4), braid_thin=True, braid_thin_d0=knobs.get("braid_thin_d0", 400), braid_report_mode=knobs.get("braid_report_mode", "all_lags")),
    }

    # (2026-09-19) energy: node package power at rest, measured once per battery; None when RAPL is not readable
    idle = idle_power(2.0) if not args.no_isolate else {"idle_power_w": None, "energy_source": None, "rapl_domains": []}
    node = {"hostname": socket.gethostname(), "cpu_count": os.cpu_count(), "oar_job_id": os.environ.get("OAR_JOB_ID"),
            "cpuset": os.environ.get("OAR_CPUSET"), "affinity_cores": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, **idle}
    print(f"node: {node['hostname']} affinity_cores={node['affinity_cores']} idle_power_w={idle['idle_power_w']} energy_source={idle['energy_source']}", flush=True)

    results = {}
    bf_flags = None
    # (2026-09-22) ignore_cleanup_errors: the large correlated sets are memory-mapped .npy files in this directory and
    # NFS cannot unlink a still-mapped file (.nfsXXXX placeholders), so the cleanup raised OSError after every arm had
    # run and the JSON was never written (12 dense m=500 cells). The wrapper removes the directory after the process exits.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for arm in arms:
            if args.neg_corr and arm in NEG_NOT_AVAILABLE:
                results[arm] = {"status": "N/A", "reason": "negative correlation not available in the method (section 8 item 1)"}
                print(f"{arm:17s} N/A (neg_corr not available)", flush=True)
                continue
            if args.n_lags > 0 and arm in SYNC_ONLY:
                results[arm] = {"status": "N/A", "reason": "synchronous-only method, n_lags > 0"}
                print(f"{arm:17s} N/A (synchronous only)", flush=True)
                continue
            arm_dir = f"{tmp}/{arm}"
            os.makedirs(arm_dir, exist_ok=True)

            def _run_arm(arm=arm, arm_dir=arm_dir):
                if arm in PATTERN_A:
                    cfg = dict(base, baseline_mode=PATTERN_A[arm], **pattern_a_extra.get(arm, {}))
                    record, _, flags = run_and_log_bruteforce(label, test_data, ids_n_var, cfg, f"{arm_dir}/{arm}.csv", metadata={"nodes": 0},
                                                              recall_by_window=True, verbose=False, testing=False)
                else:
                    record, _, flags = run_and_log_corrtrack(label, test_data, ids_n_var, base, pattern_b[arm], f"{arm_dir}/{arm}.csv",
                                                             metadata={"nodes": 0, "alg": arm}, recall_by_window=True, corr_val=True,
                                                             monitor=bool(args.monitor), verbose=False, testing=False)
                # (2026-09-19, memory) a large correlated set goes back to the parent as .npy files (memory-mapped
                # there) instead of through the pipe, which would hold a pickled copy in both processes
                if hasattr(flags, "correlated_rows") and _correlated_set_bytes(flags) > LARGE_SET_BYTES:
                    rows_, corrs_ = flags.correlated_rows()
                    np.save(f"{arm_dir}/correlated_rows.npy", np.asarray(rows_)); np.save(f"{arm_dir}/correlated_corrs.npy", np.asarray(corrs_))
                    flags = ("npy", f"{arm_dir}/correlated_rows.npy", f"{arm_dir}/correlated_corrs.npy")
                return record, flags

            t0 = time.perf_counter()
            try:
                # (2026-09-19, user) every arm in its own forked child: peak/mean RSS, storage I/O and
                # CPU time are the arm's own (abaca/resource_probe.py); one failing arm must not kill the battery
                (record, flags), resources = run_isolated(_run_arm, artifact_dir=arm_dir, isolate=not args.no_isolate, interval=args.rss_interval)
            except Exception as exc:  # noqa: BLE001
                results[arm] = {"status": "ERROR", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0]}", "traceback": str(exc)}
                print(f"{arm:17s} ERROR {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)
                continue
            wall = time.perf_counter() - t0
            if isinstance(flags, tuple) and flags and flags[0] == "npy":
                flags = NumericCorrelatedFlags(np.load(flags[1], mmap_mode="r"), np.load(flags[2], mmap_mode="r"))
            r = {k: record.get(k) for k in ("correlated", "total_candidates", "tested", "candidate_precision", "sk_time", "cand_time", "val_time", "monit_time",
                                            "runtime", "artifact_time", "n_steps", "step_time_min", "step_time_q1", "step_time_median", "step_time_q3", "step_time_max",
                                            "step_time_whisker_lo", "step_time_whisker_hi", "step_time_outliers", "step_time_mean",
                                            "candidate_search_entries_touched", "candidate_search_blocks_touched", "lsh_candidates_touched",
                                            "supports_neg_corr", "supports_lags", "n_vectors", "candidate_backend", "data_representation",
                                            "parcorr_k", "parcorr_f", "parcorr_c", "parcorr_cell_size", "statstream_n_coeffs", "statstream_index_dims",
                                            "statstream_eps", "corrjoin_ks", "corrjoin_ke", "corrjoin_kb", "corrjoin_eps1", "corrjoin_eps2",
                                            "braid_b", "braid_gamma", "braid_thin", "filcorr_fs", "filcorr_ft")}
            # phase times (s): sk/cand/val/monit are the arm's own stage clocks, runtime the in-loop time with
            # artifact I/O subtracted, other = runtime - sum(phases) (bookkeeping), wall the child's total
            # (setup + run + artifact I/O), wall_outer the parent's view including the fork/pickle overhead
            phases = {k: float(record.get(k) or 0.0) for k in ("sk_time", "cand_time", "val_time", "monit_time")}
            r["other_time"] = max(float(record.get("runtime") or 0.0) - sum(phases.values()), 0.0)
            r["phase_fractions"] = {k: (v / record["runtime"]) for k, v in phases.items()} if record.get("runtime") else None
            # (2026-09-17) the competitor indexes run their hot loops in competitor_kernels when it is built
            r.update(status="ok", wall=resources["wall_s"], wall_outer=wall,
                     pure_python_index=(arm in PURE_PYTHON_INDEX and not _HAVE_COMPETITOR_KERNELS),
                     candidate_time_per_pair_window_us=(1e6 * record["cand_time"] / record["total_candidates"]) if record.get("total_candidates") else None,
                     resources=resources, correlated_set_mb=_correlated_set_bytes(flags) / 2**20)
            results[arm] = r
            # (2026-09-19, memory) the correlated sets live in memory: score each arm as soon as it finishes and
            # drop its set, so the parent holds bruteforce's set plus ONE arm's at a time (not all eleven). The
            # per-arm size is logged (correlated_set_mb) for the campaign's memory audit.
            if arm == "bruteforce":
                bf_flags = flags
                set_gb = r["correlated_set_mb"] / 1024.0
                # the parent will hold this set for the whole battery plus one arm's set and the metrics'
                # sort temporaries (~3x the two sets); say so when it is large
                r["memory_projection_gb"] = 3.0 * 2.0 * set_gb
                print(f"bruteforce correlated set: {set_gb:.2f} GB ({r['correlated']} pair-windows); metrics projection ~{r['memory_projection_gb']:.1f} GB"
                      + ("  ** ABOVE THE MEMORY BUDGET **" if r["memory_projection_gb"] > MEMORY_BUDGET_GB else ""), flush=True)
            elif bf_flags is not None:
                t_m = time.perf_counter()
                _score_against_bruteforce(arm, r, flags, results["bruteforce"], bf_flags)
                r["metrics_time"] = time.perf_counter() - t_m
            del flags, record
            print(f"{arm:17s} done: wall={r['wall']:.2f}s correlated={r['correlated']} total={r['total_candidates']} tested={r['tested']} "
                  f"peak_rss={resources['peak_rss_mb']:.0f}MB (+{(resources['peak_rss_delta_mb'] or 0):.0f}) mean_rss={(resources['mean_rss_mb'] or 0):.0f}MB "
                  f"io_w={(resources['io_write_mb'] or 0):.1f}MB neg_corr_tag={r['supports_neg_corr']}", flush=True)

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
    print("\n=== SUMMARY ===")
    print(f"{'arm':17s} {'status':>6s} {'wall_s':>8s} {'speedup':>8s} {'correlated':>11s} {'total_cand':>11s} {'tested':>9s} {'recall':>7s} {'cand_prec':>9s} {'cand_spec':>9s} {'step_med_ms':>11s} {'step_q3_ms':>10s} {'peakMB':>7s} {'meanMB':>7s} {'neg_corr_tag':>14s} {'index':>6s}")
    for arm in arms:
        r = results[arm]
        if r["status"] != "ok":
            print(f"{arm:17s} {r['status']:>6s}  {r['reason']}")
            continue
        sp = bf["wall"] / r["wall"] if r["wall"] else float("nan")
        rec = f"{r['recall']:.4f}" if r.get("recall") is not None else "   -  "
        prec = f"{r['candidate_precision']:.4f}" if r.get("candidate_precision") is not None else "    -    "
        spec = f"{r['candidate_specificity']:.4f}" if r.get("candidate_specificity") is not None else "    -    "
        p50 = f"{1e3 * r['step_time_median']:.3f}" if r.get('step_time_median') is not None else "-"
        p99 = f"{1e3 * r['step_time_q3']:.3f}" if r.get('step_time_q3') is not None else "-"
        rs = r.get("resources", {})
        pk = f"{rs['peak_rss_delta_mb']:.0f}" if rs.get("peak_rss_delta_mb") is not None else "-"
        mn = f"{rs['mean_rss_delta_mb']:.0f}" if rs.get("mean_rss_delta_mb") is not None else "-"
        print(f"{arm:17s} {'ok':>6s} {r['wall']:8.2f} {sp:7.2f}x {r['correlated']:11d} {r['total_candidates']:11d} {r['tested']:9d} {rec:>7s} {prec:>9s} {spec:>9s} {p50:>11s} {p99:>10s} "
              f"{pk:>7s} {mn:>7s} {str(r['supports_neg_corr']):>14s} {'py' if r['pure_python_index'] else 'cy/np':>6s}")
    print(f"\ncorrtrack params: {ct_source}; competitor knob overrides: {knobs or 'none'}; "
          f"CSZ-protocol tuned arms: {tuned or 'none (paper defaults)'}")
    if args.out:
        out = {"dataset": label, "cell": args.cell, "dataset_profile": profile, "config": vars(args), "node": node, "corrtrack_params_source": ct_source, "corrtrack_hamming_params_source": ct_source_h, "competitor_params_tuned": tuned,
               "arms": results}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
