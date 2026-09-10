"""LSH cost-model sweep -- merged design serving two purposes at once:

1. The supervisor's ask: calibrate a cost model for CorrTrack (Pearson,
   candidate_backend="lsh_approx") at small-to-moderate scale and project
   speedup to m=10,000 series, using candidate_lsh_target_occupancy as the
   scale-governing knob swept in the outer grid, with L (alive/lagged
   windows) swept explicitly alongside m since both drive complexity
   independently.
2. A publication-grade speedup/recall scaling study: real measured
   wall-clock speedup vs. brute force, plus recall/precision (via
   CorrTrack.compute_metrics_bf), across (m, L, corr_prop).

(2026-08-25) n_vectors, candidate_lsh_n_bands, and gamma
(candidate_cosine_threshold) were all found to independently affect
recall (see docs/implementation_log.md's 2026-08-25 (b) entry) and are no
longer fixed/coincidentally-derived constants -- they're tuned per cell
by calibrate_hyperparams (staged: structural (n_vectors, n_bands) search,
then gamma fine-tuning), with the selected values recorded per row in the
CSV (n_vectors_selected/n_bands_selected/gamma_selected).

See /home/rsalles/.claude/plans/misty-stargazing-liskov.md for the full
approved design. This script only RUNS the grid and writes one combined
CSV; analyze_lsh_cost_sweep.py does the curve-fitting/cross-validation.

Everything here runs in-process (no subprocess/CLI shelling): synthetic
data via datasets.synth_loader.load_dataset, real data via the existing
fr_air_temperature_121_1 dataset config's DATA_LOADER, both normalized to
CorrTrack's (m+1, n) time-as-row-0 layout via corrtrack_param_search.
prepare_training_data (confirmed to give nested, nod deterministic
increasing-m subsamples for real data). Ground truth is
run_and_log_bruteforce's own in-memory .correlated-equivalent return
value (corr_flags) -- no CSV round-trip needed for the actual metric.
"""
import argparse
import csv
import gc
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

import corrtrack_param_search as cps
from corrtrack_param_search import prepare_training_data
from datasets.synth_loader import load_dataset as load_synth_dataset
from library_corrtrack_parallel import CorrTrack, run_and_log_bruteforce

# Synthetic data has no calendar years -- "count" mode treats n_year as a
# plain row-count limit inside corrtrack_param_search._select_rows.
cps.OBS_MODE = "count"

REAL_DATASET_MODULE = "experiment_dataset_fr_air_temperature_121_1"
REAL_MAX_M = 121

RESULT_DIR = Path("tmp_artifacts/lsh_cost_sweep")
SYNTH_CACHE_ROOT = RESULT_DIR / "synth_cache"

CSV_COLUMNS = [
    "cell_id", "dataset_kind", "nested_subsample",
    "m", "L", "n_lags", "window_step", "window_size", "corr_threshold",
    "target_occupancy", "n_steps", "seed",
    "corr_prop_requested", "corr_prop_achieved", "z_used", "achieved_z",
    # (2026-08-25) n_vectors/n_bands are now TUNED per cell (see
    # calibrate_hyperparams), not fixed constants -- the actual selected
    # value must be recorded per row, not just once in the manifest, per
    # the user's explicit "I should know what values were set to them."
    "n_vectors_selected", "n_bands_selected", "n_vectors_bands_calibrated",
    "n_vectors_bands_met_target_recall", "n_vectors_bands_trials_tried",
    "sketch_time", "candidate_time", "validation_time", "monitor_time",
    "smart_wall_time", "validated_candidates", "total_candidates", "tested_candidates",
    "bf_measured", "bf_runtime", "bf_cand_time", "bf_val_time", "bf_monit_time",
    "bf_tested", "bf_correlated", "bf_n_gt_for_calibration",
    "gamma_calibrated", "gamma_default", "gamma_selected", "gamma_trials_tried",
    "gamma_met_target_recall", "target_recall",
    "speedup", "precision", "recall", "f1_score", "peak_rss_gb",
]

TARGET_RECALL = 0.95
MIN_GT_FOR_CALIBRATION = 10  # below this, calibration is flagged low-confidence
# (2026-08-24) Widening this to include 0.30-0.45 offsets was tested
# directly and gave IDENTICAL recall/gamma_selected to the narrower set
# below -- confirms the LSH band-width floor (target_occupancy-derived)
# was already reached by offset 0.25, so looser gammas retrieve nothing
# extra. Kept narrow: fewer trials = less memory/time pressure per cell
# (each trial builds a full CorrTrack instance), with zero loss of the
# recall ceiling those wider trials would have found anyway.
# The gamma (candidate_cosine_threshold) search space, expressed ONLY as offsets below
# corr_threshold -- never as absolute gamma values. Every gamma the calibrators try is
# gamma = max(0, corr_threshold - offset) for offset in this tuple; both calibrate_gamma
# (per-cell, vs. real brute force) and calibrate_via_proxy_hyperopt (proxy-anchor) build
# their trial grid this way, so the meaningful quantity is always the margin below the
# threshold, not a raw cosine number. The no-search fallback picks the single
# THEORETICAL_FALLBACK_GAMMA_OFFSET (0.20) out of this same set. Tightest (offset 0.0,
# i.e. gamma == corr_threshold) last.
GAMMA_TRIAL_OFFSETS = (0.25, 0.20, 0.15, 0.10, 0.05, 0.0)

# (2026-08-25) n_vectors and candidate_lsh_n_bands both independently
# affect recall (see docs/implementation_log.md's 2026-08-25 (b) entry) and
# were previously fixed/unswept (n_vectors coincidentally tied to m,
# n_bands left at CorrTrack's constructor default). Per the user's explicit
# instruction, both are now TUNED per cell in calibrate_hyperparams, jointly
# with gamma. User-specified search space.
N_VECTORS_GRID = (16, 32, 64)
N_BANDS_GRID = (64, 128, 256)
# Fallback for cells with no brute-force ground truth to tune against
# (m > max_m_bruteforce) -- the middle of each grid, so timing-only cells
# share one fixed, representative, documented choice rather than each
# guessing independently.
DEFAULT_N_VECTORS = N_VECTORS_GRID[len(N_VECTORS_GRID) // 2]
DEFAULT_N_BANDS = N_BANDS_GRID[len(N_BANDS_GRID) // 2]


def back_solve_z(corr_prop: float, m: int, n: int, template_len: int) -> float:
    """Inverts make_corr_dataset's pair_target = z*m*n/(2*p) relation to
    corr_prop = pair_target / (m*(m-1)/2) -- see docs/implementation_
    log.md's entry for this sweep for the derivation. Clamped to [0,1]
    (make_corr_dataset asserts 0<=z<=1)."""
    if m < 2:
        return 0.0
    z = corr_prop * (m - 1) * template_len / n
    return max(0.0, min(1.0, z))


def load_synthetic_cell(m: int, n: int, corr_prop: float, window_size: int, seed: int, max_lag: int,
                         threshold: float = 0.7, window_step: int = 16):
    # (2026-08-27b) SECOND, deeper bug in the same area, found immediately
    # after fixing the one above via the LHS sweep's smoke test: fixing
    # `threshold` alone did NOT fix cell 21 (m=86, L=8, corr_threshold=0.708)
    # -- it still showed bf_correlated=0, i.e. even EXHAUSTIVE brute force
    # found none of the 17 injected pairs, despite the generator's own
    # correlated.csv recording all 17 at |r| in [0.72, 0.9999] -- well above
    # threshold. Root cause: `window_step` was never forwarded to
    # generator_params either, so synth_loader.py's own default (1) always
    # won, meaning the generator was free to place a correlated template
    # starting at ANY integer offset (its own placement grid is
    # `range(0, n-p+1, window_step)`), while CorrTrack's actual validation
    # only ever evaluates windows starting at multiples of the REAL
    # window_step (16 throughout this whole sweep family). A template
    # placed at a non-16-aligned offset (e.g. time1=10267, not a multiple
    # of 16) can fall between every window CorrTrack actually checks,
    # diluting/misaligning the correlation in every window that DOES get
    # evaluated -- enough, in this cell, to wash out all 17 pairs even
    # though several were designed at r>0.9. This mismatch has been present
    # in every prior sweep script using this function too (main sweep, OFAT
    # follow-up) -- it evidently doesn't always cause TOTAL failure (their
    # recall was usually 90%+), but this is exactly the kind of silent,
    # partial, hard-to-attribute degradation CLAUDE.md's "verify wiring
    # end-to-end" guidance warns about, not something to leave in place now
    # that it's been caught outright. Fixed by forwarding the REAL
    # window_step so the generator's own placement grid matches the
    # validation stride exactly -- no more possible misalignment, by
    # construction, not by luck. Default changed from the (never-correct)
    # synth_loader default of 1 to 16, matching this whole sweep family's
    # actual, only-ever-used window_step -- existing callers that don't
    # pass it explicitly now get the CORRECT default instead of the
    # previously-silent wrong one, which changes their cache stem (safe --
    # `_stem_from_params` already hashes window_step in, so this triggers
    # fresh, correctly-aligned regeneration, not a stale-cache collision).
    # (2026-08-27) `threshold` previously wasn't forwarded to generator_params
    # at all, so make_corr_dataset/synth_loader's own default (0.7) always
    # won -- harmless as long as every caller's corr_threshold WAS 0.7 (true
    # of every sweep run so far), but a real bug once corr_threshold starts
    # varying per cell (the LHS sweep's own corr_threshold axis): injected
    # pairs were calibrated to survive a 0.7 bar regardless of the cell's
    # actual, possibly much tighter, validation threshold -- caught via the
    # experiment_lsh_lhs_sweep.py smoke test (cell 21, corr_threshold=0.708,
    # bf_correlated=0 despite the generator's own meta.json reporting 17
    # pairs injected). _stem_from_params already hashes `threshold` into the
    # cache stem (see datasets/synth_loader.py), so this is a pure bug fix,
    # not a cache-invalidation concern: passing the real threshold, once
    # threaded through, addresses different (m, corr_prop, seed, threshold)
    # cells with the freshly-correct cache key, not a stale one.
    z = back_solve_z(corr_prop, m, n, window_size)
    # (2026-08-24) num_templates defaults to a fixed 4 in synth_loader.py
    # if left unset -- far below pair_goal at any of this sweep's m/
    # corr_prop combinations, which reuses templates across unrelated
    # series and creates real but unintended correlation between them
    # (see docs/implementation_log.md's 2026-08-24 (c)/(d) entries).
    # "auto" gives every placement its own never-reused template.
    #
    # max_lag was previously left unset (None -> unbounded gap between a
    # pair's two placements), while brute force only ever searches within
    # n_lags=(L-1)*window_step -- most injected pairs had a gap exceeding
    # that radius and were structurally undetectable regardless of
    # correlation quality (the exact bug diagnosed in docs/implementation_
    # log.md's 2026-08-24 (c) entry, there found in an ad-hoc test script;
    # here found baked into this sweep itself, previously masked by
    # template-collision noise making bf_correlated look nonzero for the
    # wrong reason). Bounded to the SMALLEST L in the grid (not this
    # cell's own L) because one cached dataset is reused across every L
    # value sharing the same (m, corr_prop, seed) -- constraining to the
    # smallest L's n_lags keeps every injected pair detectable at every L
    # tested against it, so recall differences across L reflect the
    # algorithm, not which pairs happened to fall in/out of range.
    data, ids = load_synth_dataset(
        f"synth_m{m}",
        f"cp{corr_prop}_s{seed}",
        cache_root=str(SYNTH_CACHE_ROOT),
        generator_params={
            "m": m, "n": n, "w": window_size, "z": z, "seed": seed,
            "num_templates": "auto", "max_lag": max_lag,
            # (2026-08-25) synth_loader.py's own default is corr_sign="pos"
            # -- left unset, every injected pair would be positively
            # correlated only, while run_smart_cell/run_bruteforce_once
            # both set neg_corr=True (negative correlations are supposed
            # to be detected too). "both" makes the injected ground truth
            # actually exercise that capability instead of leaving it
            # silently untested.
            "corr_sign": "both",
            "threshold": threshold,
            "window_step": window_step,
        },
    )
    # (2026-08-27) NOTE, not fixed here: this directory can hold datasets
    # generated at DIFFERENT thresholds for the same (m, corr_prop, seed)
    # (each gets its own uniquely-stemmed .npz, since _stem_from_params
    # hashes threshold in) -- but this glob takes matches[0] regardless of
    # which threshold generated it. Currently inert for every caller in
    # this sweep family (each cell's (m, corr_prop) pair is either fixed
    # across a whole run at one threshold, or -- in the LHS sweep -- a
    # continuous per-cell draw unique enough that this directory holds
    # exactly one meta.json in practice). Would need tightening if a
    # future caller revisits the same (m, corr_prop, seed) at multiple
    # thresholds in one run.
    stem_meta = _find_meta_json(SYNTH_CACHE_ROOT / f"synth_m{m}_cp{corr_prop}_s{seed}")
    achieved_z = None
    pairs_used = None
    if stem_meta is not None:
        with open(stem_meta) as f:
            meta = json.load(f)
        achieved_z = meta.get("achieved_z")
        pairs_used = meta.get("pairs_used")
    total_series_pairs = m * (m - 1) / 2 if m >= 2 else 1
    achieved_corr_prop = (pairs_used / total_series_pairs) if pairs_used is not None else None
    return data, ids, z, achieved_z, achieved_corr_prop


def _find_meta_json(dataset_dir: Path):
    if not dataset_dir.is_dir():
        return None
    matches = list(dataset_dir.glob("*_meta.json"))
    return matches[0] if matches else None


_REAL_DATA_CACHE = {}


def load_real_cell(m: int):
    if "raw" not in _REAL_DATA_CACHE:
        import importlib
        cfg = importlib.import_module(REAL_DATASET_MODULE)
        data, ids = cfg.DATA_LOADER(cfg.COUNTRIES[0], cfg.VARIABLES[0])
        _REAL_DATA_CACHE["raw"] = (data, ids)
    data, ids = _REAL_DATA_CACHE["raw"]
    m = min(m, len(ids))
    train_data, ids_n_var = prepare_training_data(data, ids, n_year=data.shape[0], n_var=m, train_ratio=1.0)
    return train_data, list(ids_n_var)


def run_smart_cell(train_data, ids, window_size, window_step, n_lags, corr_threshold,
                    target_occupancy, n_steps, seed, candidate_cosine_threshold=None,
                    n_vectors=None, candidate_lsh_n_bands=64, candidate_lsh_n_bands_tolerance=None,
                    target_recall=0.95):
    # (2026-08-25) n_vectors previously defaulted to len(ids) (== m) --
    # coincidental, not principled: n_vectors is the sketch's random-
    # projection dimensionality (a JL-embedding-style accuracy/cost knob),
    # unrelated in principle to series count, and this project's own
    # experiment_run_param_grid.py already treats it as an independently-
    # tunable hyperparameter. Tying it to m confounded this sweep's own
    # m-scaling measurements (see docs/implementation_log.md's 2026-08-25
    # (b) entry) -- now an explicit, calibrated parameter (see
    # calibrate_hyperparams). The len(ids) fallback is kept ONLY for any
    # caller that doesn't tune it (there should be none left).
    if n_vectors is None:
        n_vectors = len(ids)
    ct = CorrTrack(
        window_size=window_size, basic_window=window_step, window_step=window_step,
        n_vectors=n_vectors, n_lags=n_lags, corr_threshold=corr_threshold, neg_corr=True,
        preprocess=False, exec="sequential", data_representation="sketch_proj",
        candidate_backend="lsh_approx", candidate_lsh_target_occupancy=target_occupancy,
        candidate_lsh_n_bands=candidate_lsh_n_bands,
        # (2026-08-30) When set, overrides candidate_lsh_n_bands with the
        # closed-form LSH-banding minimum (b_min) times this tolerance --
        # see candidate_kernels.pyx's SignLSHBandIndex._n_bands_tolerance.
        # None (default) is fully inert -- every existing caller that
        # doesn't pass this keeps its exact prior behavior.
        candidate_lsh_n_bands_tolerance=candidate_lsh_n_bands_tolerance,
        target_recall=target_recall,
        # (2026-08-25) Explicit per the user's decision, not left to
        # CorrTrack's own constructor defaults: Hamming pre-filter OFF
        # (it's a lossy, gamma-independent filter stage found during the
        # recall-parameter audit -- see docs/implementation_log.md's
        # 2026-08-25 (b) entry), dot-gamma filter always ON (the real
        # recall gate this sweep tunes via gamma), candidate volume
        # per query left unbounded.
        candidate_apply_hamming_filter=False,
        candidate_apply_dot_gamma_filter=True,
        candidate_lsh_max_candidates_per_query=0,
        candidate_cosine_threshold=candidate_cosine_threshold,
        validation_metric="pearson", seed=seed, seed_toggle=seed + 1,
    )
    pos = 0
    max_steps = (train_data.shape[1] - window_size) // window_step
    n_steps = min(n_steps, max(1, max_steps))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        chunk = train_data[:, pos:pos + window_step]
        pos += window_step
        ct.run(chunk, ids, verbose=False, testing=False, corr_val=True, monitor=True)
    wall_time = time.perf_counter() - t0
    return ct, wall_time, n_steps


def run_bruteforce_once(cache_key, train_data, ids, window_size, window_step, n_lags,
                         corr_threshold, output_csv):
    # (2026-08-24) Previously memoized by cache_key across calls, but
    # cache_key includes (m, L, n_steps, seed, corr_prop) -- every real
    # call in this grid's design is already unique, so the cache was NEVER
    # actually hit; it just accumulated every cell's full corr_flags dict
    # in memory for the whole run's lifetime with zero reuse benefit. A
    # real contributor to the SIGKILL (exit 137) seen scaling m up.
    # cache_key is kept as a parameter for now (harmless, callers still
    # pass it) but no longer used to memoize.
    base_config = dict(
        window_size=window_size, window_step=window_step, basic_window=window_step,
        n_lags=n_lags, corr_threshold=corr_threshold, neg_corr=True, exec="sequential",
        parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
        max_workers=0, baseline_mode="bruteforce", monitor=True, track_min_dist=True,
        artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
        save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False,
        testing=False, validation_metric="pearson",
    )
    record, _runtime_parts, corr_flags = run_and_log_bruteforce(
        "sweep", train_data, ids, base_config, output_csv,
        metadata={"nodes": 0}, recall_by_window=True,
    )
    return record, corr_flags


def calibrate_gamma(cell_data, ids, window_size, window_step, n_lags, corr_threshold,
                     target_occupancy, n_steps, seed, bf_corr_flags, bf_tested,
                     n_vectors, candidate_lsh_n_bands, candidate_lsh_n_bands_tolerance=None,
                     target_recall=0.95):
    """Per-cell gamma (candidate_cosine_threshold) calibration against REAL
    brute-force ground truth -- not CorrTrack_optimize's proxy-anchor
    machinery. Tried that first: it samples a handful of RANDOM anchor
    positions and only sees whichever few true positives happen to land
    there -- at sparse corr_prop this gave n_gt=1 (recall could only ever
    be 0.0 or 1.0, not a real statistic), and reaching adequate power would
    have needed anchor_count in the hundreds (each costing O(m^2*L) to
    build), making calibration itself a dominant, unpredictable cost.
    Brute force already sees EVERY true positive in the full cell_data
    span by construction (it's exhaustive) -- reusing bf_corr_flags (which
    the cell needs anyway) as calibration ground truth gets full
    statistical power for free, and each gamma trial only costs one extra
    cheap smart-CorrTrack pass, not a new ground-truth computation.
    Returns (best_ct, best_wall_time, best_gamma, met_target, n_trials)."""
    default_gamma = float(corr_threshold)
    trial_gammas = sorted({max(0.0, default_gamma - off) for off in GAMMA_TRIAL_OFFSETS})

    # (2026-08-24) Each trial builds a full CorrTrack instance; keeping all
    # of them alive simultaneously (as the original design did, via a
    # `trials` list of (gamma, ct, ...) tuples) held every trial's full
    # internal state in memory at once across the whole cell -- a real
    # contributor to a SIGKILL (exit 137, consistent with OOM) hit while
    # scaling from m=100 to m=300. Extract only the small set of scalar
    # fields each row actually needs immediately after each trial, then
    # explicitly drop the CorrTrack object and force collection before the
    # next trial, so only ONE full instance is ever live at a time.
    trials = []
    for gamma in trial_gammas:
        ct, wall_time, n_steps_run = run_smart_cell(
            cell_data, ids, window_size, window_step, n_lags, corr_threshold,
            target_occupancy, n_steps, seed, candidate_cosine_threshold=gamma,
            n_vectors=n_vectors, candidate_lsh_n_bands=candidate_lsh_n_bands,
            candidate_lsh_n_bands_tolerance=candidate_lsh_n_bands_tolerance,
            target_recall=target_recall,
        )
        metrics = CorrTrack.compute_metrics_bf(
            ct, bf_corr_flags, windows=True, total_pairs_bf=bf_tested,
        )
        recall = metrics.get("recall")
        trials.append({
            "gamma": gamma, "wall_time": wall_time, "n_steps_run": n_steps_run,
            "recall": recall, "metrics": metrics,
            "sketch_time": ct.sketch_time, "candidate_time": ct.candidate_time,
            "validation_time": ct.validation_time, "monitor_time": ct.monitor_time,
            "validated_candidates": ct.validated_candidates,
            "total_candidates": ct.total_candidates, "tested_candidates": ct.tested_candidates,
        })
        del ct
        gc.collect()

    meeting_target = [t for t in trials if t["recall"] is not None and t["recall"] >= TARGET_RECALL]
    if meeting_target:
        # Among gammas that hit the target, the TIGHTEST (largest gamma)
        # gives the smallest candidate volume -- mirrors
        # _apply_proxy_anchor_selection's own "meet the recall floor, then
        # maximize filtering" principle.
        best = max(meeting_target, key=lambda t: t["gamma"])
        met_target = True
    else:
        # Nothing hit the target -- best effort, report the highest recall
        # achieved and flag it honestly rather than silently picking one.
        best = max(trials, key=lambda t: (t["recall"] if t["recall"] is not None else -1.0))
        met_target = False

    return best, met_target, len(trials), default_gamma


PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR = 3_000_000
# (2026-09-08) Even ONE anchor's own pair universe must be built (real correlation computed
# for every pair) before any row-budget capping can help -- recommend_proxy_pair_row_budget's
# own max(1, budget//est) always tries at least 1 anchor, regardless of how far over budget
# that single anchor is. This was originally 2,000,000, calibrated against a ~25us/pair CPU-TIME
# measurement of CorrTrack_optimize._prepare_proxy_anchor_reference's OWN Python loop (a real,
# pre-existing bottleneck in the production hyperopt pipeline, not something specific to this
# sweep -- see docs/implementation_log.md's 2026-09-08 entries). That loop's CPU cost has since
# been fixed: the per-pair correlation math was already Cython (fast_corr_and_dist) but called
# once per pair in a Python loop; window-level validity (is_near_constant/is_structurally_
# spiked -- the SAME filtering the real streaming run applies, deliberately preserved exactly)
# was also recomputed via per-pair dict lookups. Both are now batched (fast_corr_and_dist_batch,
# vectorized validity), verified byte-for-byte identical to the original algorithm on 4 test
# cases including a cap-binding edge case. Real measured CPU rate: ~4us/pair (was ~25us/pair, a
# real ~6x reduction).
#
# This constant was FIRST raised to 15,000,000 on the strength of that time win alone (~60s
# worst case) -- a real mistake, caught by a real production incident, not by further analysis:
# a live cell (m=296, L=55, ~2.4M pairs/anchor, needing several anchors after adaptive
# expansion) drove this host to 3.9GB of swap and made it grind for 10+ minutes on what should
# have been a ~2min job. Measured the actual cause directly (tracemalloc): pair_keys/
# key_to_indices holds ~400 bytes per pair row (Python tuple + string objects), UNCHANGED by
# the Cython fix -- that fix cut CPU time, not the fundamentally Python-object-based memory
# footprint of the per-row output bookkeeping (deliberately left as Python -- see the
# _prepare_proxy_anchor_reference rewrite's own comment on why key_to_indices' representation
# wasn't changed). At 15,000,000 pairs that is ~6GB -- more than this 7.8GB host has to spare
# once its other usage is accounted for. Memory, not CPU time, is the real binding constraint,
# and the time-based win doesn't relax it. Reset to 3,000,000 (~1.2GB for the reference itself,
# leaving headroom for the rest of CorrTrack_optimize's own memory needs -- the 24 trial
# evaluations' candidate search plus the caller's own final brute-force+measured run) -- modestly
# above the original 2,000,000, not the originally-planned 15,000,000.
#
# (2026-09-08, later same day) SUPERSEDED as a skip-gate: the user pointed out that skipping
# real calibration at large m defeats the point of a method meant to be scalable AND accurate
# -- gamma/occupancy don't actually depend on series count (only on corr_threshold/n_vectors/
# data characteristics), so CorrTrack_optimize now subsamples series internally
# (_proxy_series_subsample_cap/_proxy_stratified_series_sample, stratified by variance/
# kurtosis/lag-1-autocorrelation so a heterogeneous real dataset's diversity is preserved, not
# a uniform random draw) whenever the full population would exceed this same budget. n_bands
# itself is unaffected -- always sized analytically for the TRUE n_series. This constant is now
# passed through as that subsample budget (series_subsample_max_pairs), not a threshold that
# skips calibration -- real hyperopt is attempted at every scale.
#
# (2026-09-08, still later) Second real bug, caught by the SAME end-to-end smoke test that
# verified the subsampling fix above: both sweep scripts were passing this exact constant as
# BOTH the per-anchor subsample budget (max_feasible_pairs_per_anchor, correct) AND the TOTAL
# pair_row_hard_ceiling (wrong) -- e.g. calibrate_via_proxy_hyperopt(...,
# pair_row_hard_ceiling=PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR). estimate_proxy_series_
# subsample_cap's own binary search picks the LARGEST effective_n_series whose single-anchor
# cost stays <= the subsample budget -- by construction that single anchor's cost sits right up
# against the budget. Reusing the SAME number as the TOTAL ceiling then left zero room for a
# 2nd anchor (anchors_affordable = hard_ceiling // est_pairs_per_anchor = 1), silently
# defeating the whole point of the fix: a smoke test at (m=296, L=55) showed effective_n_series
# correctly dropping to 234 (series_subsampled=True) yet proxy_anchor_count/
# anchors_affordable_at_budget stuck at 1, identical to before the fix, even though a
# standalone recommend_proxy_pair_row_budget(234, ...) call with a properly-separated ceiling
# confirmed 6 anchors should be affordable.
#
# Fixed by decoupling the two budgets: a new PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING is what's now
# passed as pair_row_hard_ceiling, while this constant continues to bound the per-anchor
# subsample size alone.
#
# (2026-09-08, still later) First tried a 3x multiplier (9,000,000) on memory grounds alone
# (pair_keys/key_to_indices costs ~400 bytes/pair-row regardless of anchor count, so 3 anchors
# at the per-anchor cap would cost ~3.6GB for the reference, comfortably under this 7.8GB host)
# -- but a full end-to-end run at the exact incident cell (m=296, L=55, effective_n_series=234,
# est_pairs_per_anchor=2,984,085, i.e. right at the per-anchor cap) measured a real, severe TIME
# regression: 3,042s (>50min) for anchor_count=3 vs. 220s for anchor_count=1 -- a ~14x blowup
# from a 3x anchor increase, with proxy_n_gt staying at 1 (identical statistical power to the
# single-anchor case). Peak RSS was fine (3.3GB, no swap) -- this constant's own memory
# reasoning was correct -- but wall-clock time is NOT linear in anchor count once a single
# anchor's own pair count is already near the per-anchor budget (most likely the 24-trial LSH
# sweep's own candidate-retrieval/validation cost grows worse than linearly with total reference
# size in that regime; not yet root-caused further given the time budget). Confirmed via
# _get_proxy_anchor_optim_params (library_corrtrack_parallel.py) that this was NOT the adaptive-
# anchor expansion loop compounding (only one round of the 24-trial grid ran, matching the log's
# exact "24/24 completed" count) -- the cost is intrinsic to one round at anchor_count=3 with a
# near-cap per-anchor size, not repeated expansion.
#
# Fixed by lowering the multiplier to 1.5x (4,500,000): re-verified at the exact same inputs
# that this reproduces anchors_affordable=1 (i.e. the already-measured-safe 220s/1-anchor
# behavior, no regression) precisely AT this pathological near-cap corner, while still affording
# far more anchors (23 at m=100/L=20, 244 at m=40/L=12 -- both well below the per-anchor cap)
# for cells where extra anchors are cheap and can genuinely add statistical power. This is a
# real, disclosed limitation, not a bug: at very large m combined with very low corr_prop, a
# single anchor already costs close to the memory/time budget, and this method does not chase
# further anchors past that point -- matching calibrate_gamma's own documented "struggles at low
# corr_prop" limitation, now extended honestly to the proxy-hyperopt path too.
PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING = int(1.5 * PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR)


def calibrate_via_proxy_hyperopt(cell_data, ids, window_size, window_step, n_lags, corr_threshold,
                                  target_occupancy_grid, n_series, seed, output_csv_prefix, dataset_id,
                                  target_recall=TARGET_RECALL, n_vectors=None,
                                  pair_row_hard_ceiling=None, max_anchor_count=None,
                                  max_feasible_pairs_per_anchor=PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR):
    """(2026-09-08) Calibrates (candidate_lsh_target_occupancy, candidate_cosine_threshold)
    using the REAL production hyperopt pipeline -- CorrTrack_optimize.get_optim_params, i.e.
    proxy-anchor sampling with exact LOCAL ground truth (computed fresh per anchor, not
    borrowed from a full-length brute-force run) -- not calibrate_gamma/calibrate_hyperparams's
    own brute-force-based mechanism above. Built at the user's explicit request, after they
    pointed out that always calibrating against full brute-force ground truth doesn't
    "showcase how CorrTrack will handle actual data" (in real deployment you don't have that
    ground truth either) and asked to validate their own hyperopt pipeline directly, including
    its real limitations.

    KNOWN, DOCUMENTED LIMITATION, embraced deliberately (user's explicit choice after being
    shown this tradeoff): calibrate_gamma's own docstring above records that proxy-anchor
    hyperopt was tried for this exact synthetic generator before and found to struggle at low
    corr_prop -- few true positives per anchor sample can make the recall estimate
    degenerate (0%/100%, not a real statistic), and reaching more statistical power costs
    O(m^2*L) per additional anchor. This function does NOT work around that -- an
    underpowered/degenerate result at a sparse cell is a real, honest finding about the
    hyperopt pipeline's own limits, not something to hide. Cost is still bounded: anchor
    expansion stops at proxy_max_pair_rows (sized below from the real (n_series, n_lags,
    window_step) via recommend_proxy_pair_row_budget), the same real resource ceiling
    production uses, not a fixed guess.

    (2026-09-10) A SECOND regime where the proxy recall estimate is unreliable, found by a
    controlled experiment on real dense data (fr_air_temperature, 121 series -- see
    docs/implementation_log.md's 2026-09-10 (r) entry): at HIGH effective correlation density,
    the proxy recall estimate systematically UNDER-reports. The earlier gamma-stability run saw
    proxy_recall_lb/ub around 0.16-0.62 and proxy_feasible=False for every combo on that
    dataset, yet the same selected gamma, measured against full brute-force ground truth, gave
    ~97% recall. So proxy_feasible=False / a low proxy_recall_* on this function's output does
    NOT mean "this config will miss the target" -- on dense data it routinely does hit it. The
    flag is still honest about the calibration's OWN confidence; it just isn't a reliable
    predictor of real recall in the dense regime. Not worked around here (a bias correction or
    more anchors would be a real algorithmic change to the calibration, out of scope, and the
    function still SELECTS a good combo -- the loosest gamma -- despite the pessimism, so
    downstream behavior is unaffected; only the reported confidence flag is misleading).

    n_vectors is FIXED (not searched) per the user's explicit statement -- "n_vectors is fixed
    in this experiment". The two knobs actually swept, matching calibrate_nbands_occupancy's
    own two-parameter scope exactly (just via the real hyperopt mechanism instead of an ad hoc
    grid search): candidate_lsh_target_occupancy (target_occupancy_grid) and
    candidate_cosine_threshold (GAMMA_TRIAL_OFFSETS below corr_threshold, same trial set
    calibrate_gamma uses). candidate_lsh_n_bands_tolerance is left at CorrTrack's own AUTO
    default (1.0) throughout -- nothing left to search there since the 2026-09-03 overlap
    correction, exactly as experiment_run_param_grid.py's own production PARAM_GRID already
    assumes.

    pair_row_hard_ceiling/max_anchor_count override the real resource ceiling
    (recommend_proxy_pair_row_budget) that bounds anchor-reference-construction cost --
    default None means "use production's own defaults" (20,000,000 / DEFAULT_OPTIM_PROXY_
    MAX_ANCHOR_COUNT). (2026-09-08) Added after a real, measured timing check: reference
    construction is a pure-Python nested loop over every pair in every sampled anchor's own
    local universe (est_pairs_per_anchor ~ n_series^2 * L/2, see estimate_proxy_pairs_per_
    anchor) -- at m=300, L=64 this alone took minutes and ~4.4GB RSS even capped at the
    default 20M-row ceiling (which itself only affords ~3-4 anchors at that scale). Under a
    hard wall-clock deadline, callers facing many cells across a wide (m, L) sweep should pass
    a smaller ceiling here to keep this bounded, at the cost of fewer anchors (less
    statistical power) -- the same "let it struggle where it struggles" tradeoff already
    accepted for recall, now extended to time/memory too.

    Returns a dict with the winning combo's real params (n_bands read back as the exact
    theoretical value _run_corrtrack_proxy_anchor computes via compute_lsh_sizing, not
    estimated) plus proxy diagnostics (proxy_recall_lb/ub, proxy_underpowered, proxy_feasible,
    n_trials, trial_grid_path), or None if get_optim_params found no selectable combination at
    all (a real, disclosed failure mode -- the caller decides how to handle it, not this
    function)."""
    from corrtrack_param_search import (
        DEFAULT_OPTIM_PROXY_RANDOM_SEED, DEFAULT_OPTIM_PROXY_BOOTSTRAP_ENABLED,
        DEFAULT_OPTIM_PROXY_BOOTSTRAP_REPEATS, DEFAULT_OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL,
        DEFAULT_OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS, DEFAULT_OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED,
        DEFAULT_OPTIM_PROXY_ANCHOR_COUNT, DEFAULT_OPTIM_PROXY_MAX_ANCHOR_COUNT,
        DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_FACTOR, DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS,
        DEFAULT_OPTIM_PROXY_TIMING_REPEATS, DEFAULT_OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE,
        DEFAULT_OPTIM_PROXY_PAIR_ROW_HARD_CEILING, DEFAULT_RECALL_FALLBACK_NEAR_RATIO,
        DEFAULT_SPEEDUP_NEAR_RATIO,
    )
    from library_corrtrack_parallel import (
        CorrTrack_optimize, recommend_proxy_pair_row_budget, estimate_proxy_pairs_per_anchor,
        estimate_proxy_series_subsample_cap,
    )

    n_vectors = int(n_vectors) if n_vectors else DEFAULT_N_VECTORS
    hard_ceiling = int(pair_row_hard_ceiling) if pair_row_hard_ceiling else DEFAULT_OPTIM_PROXY_PAIR_ROW_HARD_CEILING
    anchor_cap = int(max_anchor_count) if max_anchor_count else DEFAULT_OPTIM_PROXY_MAX_ANCHOR_COUNT
    series_subsample_budget = int(max_feasible_pairs_per_anchor) if max_feasible_pairs_per_anchor else 3_000_000

    # (2026-09-08) No longer a skip-gate -- CorrTrack_optimize now subsamples series
    # internally (_proxy_series_subsample_cap/_proxy_stratified_series_sample) whenever the
    # full population would make even one anchor's own pair universe impractical, so real
    # calibration is now attempted at every scale instead of falling back to theoretical
    # sizing whenever m is large. series_subsample_max_pairs (below) is that subsample budget.
    #
    # Anchor-count/row budgeting must size off the EFFECTIVE, post-subsampling series count,
    # not the true n_series -- a real bug caught by a real smoke test: at (m=296, L=55), sizing
    # off the true m capped anchor_count to 1 (since est_pairs_per_anchor there is 4.77M), even
    # though subsampling had already made each anchor far cheaper -- needlessly starving
    # statistical power exactly where the fix was supposed to restore it.
    effective_n_series = estimate_proxy_series_subsample_cap(
        n_series, n_lags, window_step, series_subsample_budget,
    )
    max_pair_rows, est_pairs_per_anchor, anchors_affordable = recommend_proxy_pair_row_budget(
        effective_n_series, n_lags, window_step,
        max_anchor_count=anchor_cap,
        hard_ceiling=hard_ceiling,
    )
    proxy_config = {
        "anchor_count": min(int(DEFAULT_OPTIM_PROXY_ANCHOR_COUNT), int(anchors_affordable)),
        "max_pair_rows": max_pair_rows,
        "distance_cache_max_rows": max_pair_rows,
        "series_subsample_max_pairs": series_subsample_budget,
        "random_seed": DEFAULT_OPTIM_PROXY_RANDOM_SEED,
        "bootstrap_enabled": DEFAULT_OPTIM_PROXY_BOOTSTRAP_ENABLED,
        "bootstrap_repeats": DEFAULT_OPTIM_PROXY_BOOTSTRAP_REPEATS,
        "bootstrap_confidence_level": DEFAULT_OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL,
        "bootstrap_min_gt_events": DEFAULT_OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS,
        "eval_mode": "cached_distances",
        "adaptive_anchor_enabled": DEFAULT_OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED,
        "max_anchor_count": anchor_cap,
        "anchor_expand_factor": DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_FACTOR,
        "anchor_expand_max_rounds": DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS,
        "timing_repeats": DEFAULT_OPTIM_PROXY_TIMING_REPEATS,
        "candidate_rate_close_tolerance": DEFAULT_OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE,
    }

    # The search axis IS the offset below corr_threshold, never an absolute gamma. We resolve
    # each offset to an absolute candidate_cosine_threshold only because that is the parameter
    # CorrTrack_optimize -> CorrTrack accepts (there is no offset knob inside CorrTrack); the
    # winning offset is recovered and reported below so the result stays offset-native.
    gamma_by_offset = {
        off: round(max(0.0, float(corr_threshold) - off), 6) for off in sorted(set(GAMMA_TRIAL_OFFSETS))
    }
    offset_by_gamma = {g: off for off, g in gamma_by_offset.items()}
    gamma_trials = sorted(set(gamma_by_offset.values()))
    param_grid = {
        "n_vectors": [n_vectors],
        "seed": [int(seed)],
        "seed_toggle": [int(seed) + 1],
        "candidate_lsh_target_occupancy": list(target_occupancy_grid),
        "candidate_cosine_threshold": gamma_trials,
    }

    optimizer = CorrTrack_optimize(
        cell_data, ids, window_size, window_step, n_lags, corr_threshold,
        recall_by_window=True, alg="corrtrack", neg_corr=True, corr_val=False,
        exec="sequential", proxy_config=proxy_config,
        # (2026-09-08) Must match run_smart_cell's own explicit choice (Hamming pre-filter
        # OFF throughout this whole sweep -- see run_smart_cell's own comment) or the
        # calibrated combo would be tuned under a DIFFERENT filter configuration than the
        # one the final run actually uses.
        candidate_apply_hamming_filter=False,
        candidate_apply_dot_gamma_filter=True,
        candidate_lsh_max_candidates_per_query=0,
    )
    # (2026-09-08) `optimizer` holds the full proxy-anchor reference (pair_keys/key_to_indices,
    # potentially millions of rows -- see the subsampling comments above) plus every one of the
    # 24 trial grid's own LSH index state. It is a local variable that would normally be freed by
    # refcounting the moment this function returns, EXCEPT that CorrTrack_optimize's internal
    # object graph (index <-> cached-distance <-> trial-record back-references) plausibly
    # contains reference cycles, which only Python's cyclic GC (not refcounting) can reclaim --
    # and cyclic GC runs on its own schedule, not necessarily before the caller's very next
    # allocation. Caught by a real crash: a worker subprocess (3GB RLIMIT_AS) failed on a TRIVIAL
    # 7.48 MiB numpy allocation immediately after calibration, in the real production run that
    # follows in the SAME process -- the only explanation is that calibration's own memory was
    # still resident, not that the production run itself needed more than 3GB at that scale.
    # Fixed by explicitly deleting `optimizer` and forcing a `gc.collect()` on every return path,
    # so the calibration phase's memory is actually reclaimed before the caller's real run starts
    # building its own large structures in the same process.
    try:
        best_df = optimizer.get_optim_params(
            param_grid, output_csv_prefix, dataset_id, run=True,
            target_recall=target_recall,
            recall_fallback_near_ratio=DEFAULT_RECALL_FALLBACK_NEAR_RATIO,
            speedup_near_ratio=DEFAULT_SPEEDUP_NEAR_RATIO,
        )
    except ValueError as exc:
        # (2026-09-08) get_optim_params raises when NOTHING is selectable at all -- a real,
        # disclosed failure mode (e.g. every combo errored), not swallowed silently.
        print(f"[proxy_hyperopt] {dataset_id}: {exc}")
        del optimizer
        gc.collect()
        return None

    if best_df is None or best_df.empty:
        del optimizer
        gc.collect()
        return None
    best = best_df.iloc[0].to_dict()
    trial_grid_path = output_csv_prefix + "_corrtrack.csv"
    _best_gamma = _to_float_or_none(best.get("candidate_cosine_threshold"))
    _best_offset = offset_by_gamma.get(round(_best_gamma, 6)) if _best_gamma is not None else None
    if _best_offset is None and _best_gamma is not None:
        _best_offset = round(float(corr_threshold) - _best_gamma, 4)
    result = {
        "n_bands": _to_int_or_none(best.get("candidate_lsh_n_bands")),
        "target_occupancy": _to_float_or_none(best.get("candidate_lsh_target_occupancy")),
        "gamma": _best_gamma,
        "gamma_offset": _best_offset,
        "n_vectors": n_vectors,
        "proxy_recall_lb": _to_float_or_none(best.get("proxy_recall_lb")),
        "proxy_recall_ub": _to_float_or_none(best.get("proxy_recall_ub")),
        "proxy_feasible": bool(best.get("proxy_feasible")),
        "proxy_underpowered": bool(best.get("proxy_underpowered")),
        "proxy_n_gt": _to_int_or_none(best.get("proxy_n_gt")),
        "proxy_anchor_count": _to_int_or_none(best.get("proxy_anchor_count")),
        "n_grid_combos": len(gamma_trials) * len(list(target_occupancy_grid)),
        "trial_grid_path": trial_grid_path,
        "anchors_affordable_at_budget": int(anchors_affordable),
        "est_pairs_per_anchor": int(est_pairs_per_anchor),
        "effective_n_series": int(effective_n_series),
        "series_subsampled": bool(effective_n_series < n_series),
    }
    del optimizer, best_df
    gc.collect()
    return result


# (2026-09-10) Gamma offset for the no-hyperopt fallback. An experiment on real dense data
# (fr_air_temperature, 121 series, see docs/implementation_log.md's 2026-09-10 (r) entry)
# measured real recall vs. gamma at corr_threshold in {0.70, 0.80, 0.90}: gamma = corr_threshold
# EXACTLY (offset 0) gave only ~72-76% recall -- well below the 0.95 target -- while an offset of
# ~0.15-0.25 below the threshold landed on a ~97% recall plateau at every threshold tested.
# The old fallback used gamma = corr_threshold (offset 0), which is where CorrTrack's own
# candidate_cosine_threshold=None would resolve, and is exactly the pathological case. 0.20 is
# the middle of the measured plateau-entry band.
THEORETICAL_FALLBACK_GAMMA_OFFSET = 0.20
assert THEORETICAL_FALLBACK_GAMMA_OFFSET in GAMMA_TRIAL_OFFSETS, (
    "the fallback gamma offset must be one of the offsets the calibrators also try, "
    "so search and no-search paths stay on the same gamma ladder"
)


def theoretical_sizing_fallback(m, L, corr_threshold, n_vectors=None, target_occupancy=3.0,
                                 target_recall=TARGET_RECALL):
    """(2026-09-08) Used when calibrate_via_proxy_hyperopt finds nothing selectable at all
    (get_optim_params raises -- e.g. every combo errored). Since CorrTrack_optimize now
    subsamples series internally, this is no longer the common "m is too large" path it
    started as -- real hyperopt (on a subsample when needed) is attempted at every scale, so
    this fallback should now be rare, not routine. Picks (candidate_lsh_target_occupancy,
    candidate_cosine_threshold) via the closed-form formula directly instead of any search:
    target_occupancy at the project's own calibrated default (3.0, not tuned -- there's no
    cheap way to tune it without a real search), and gamma (the dot-product GATE applied AFTER
    band retrieval) at corr_threshold minus THEORETICAL_FALLBACK_GAMMA_OFFSET (see that
    constant's comment: gamma = corr_threshold exactly gives only ~75% recall on dense data,
    since sketch dot products are noisy estimates of true correlation and a genuine
    threshold-level pair can score below the threshold at the gate; a ~0.20 offset lands on the
    ~97% recall plateau). n_bands is still sized for corr_threshold ITSELF, not the offset gamma
    -- n_bands governs which pairs reliably COLLIDE in a band, which should target the real
    correlation level of interest; sizing it for the looser gamma inflates n_bands ~7x for no
    added recall the gate offset doesn't already give. Returns the same shape as
    calibrate_via_proxy_hyperopt's real result (with proxy_* fields None/False) so callers can
    merge them into one row schema."""
    from candidate_kernels import compute_lsh_sizing
    n_vectors = int(n_vectors) if n_vectors else DEFAULT_N_VECTORS
    n_lagged_windows = int(L)
    fallback_gamma = max(0.0, float(corr_threshold) - THEORETICAL_FALLBACK_GAMMA_OFFSET)
    _band_width, n_bands = compute_lsh_sizing(
        int(m), n_lagged_windows, float(target_occupancy), float(corr_threshold),
        n_vectors, float(target_recall), 1.0,
    )
    return {
        "n_bands": n_bands, "target_occupancy": float(target_occupancy),
        "gamma": fallback_gamma, "gamma_offset": float(THEORETICAL_FALLBACK_GAMMA_OFFSET),
        "n_vectors": n_vectors,
        "proxy_recall_lb": None, "proxy_recall_ub": None, "proxy_feasible": None,
        "proxy_underpowered": None, "proxy_n_gt": None, "proxy_anchor_count": None,
        "n_grid_combos": 0, "trial_grid_path": None,
        "anchors_affordable_at_budget": None, "est_pairs_per_anchor": None,
        "effective_n_series": None, "series_subsampled": None,
    }


def _to_int_or_none(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def calibrate_hyperparams(cell_data, ids, window_size, window_step, n_lags, corr_threshold,
                           target_occupancy, n_steps, seed, bf_corr_flags, bf_tested):
    """(2026-08-25) Two-stage tuning of (n_vectors, candidate_lsh_n_bands,
    gamma) against REAL brute-force ground truth -- same rationale as
    calibrate_gamma (full statistical power for free, reusing work the
    cell needs anyway), extended to two more independently recall-
    affecting parameters found during the recall-parameter audit (see
    docs/implementation_log.md's 2026-08-25 (b) entry).

    A full joint grid over (n_vectors x n_bands x gamma) was considered
    and rejected as too expensive given the user's own stated time
    constraint (~9x this sweep's prior per-cell cost). Staged instead:

    Stage 1 -- structural (n_vectors, n_bands) search: for every grid
    combo, run with gamma forced to 0.0 (candidate_apply_dot_gamma_filter
    stays on, but fabs(score)>=0.0 is always true, so it rejects nothing)
    -- this isolates the LSH retrieval stage's OWN recall ceiling
    (governed by n_vectors/n_bands/target_occupancy), decoupled from
    gamma's own filtering. Keep combos reaching TARGET_RECALL; among
    those, pick the one touching the fewest candidates (fastest) --
    mirrors calibrate_gamma's own "meet the floor, then minimize work"
    principle, just measured via tested_candidates instead of via gamma
    tightness directly.

    Stage 2 -- gamma fine-tuning: within the winning (n_vectors, n_bands),
    reuse calibrate_gamma UNCHANGED to find the tightest gamma still
    meeting TARGET_RECALL.

    Total cost: len(N_VECTORS_GRID)*len(N_BANDS_GRID) stage-1 trials plus
    calibrate_gamma's own len(GAMMA_TRIAL_OFFSETS) stage-2 trials -- ~2.5x
    calibrate_gamma alone, not ~9x a full joint grid.
    """
    stage1_trials = []
    for n_vectors, n_bands in itertools.product(N_VECTORS_GRID, N_BANDS_GRID):
        ct, wall_time, n_steps_run = run_smart_cell(
            cell_data, ids, window_size, window_step, n_lags, corr_threshold,
            target_occupancy, n_steps, seed, candidate_cosine_threshold=0.0,
            n_vectors=n_vectors, candidate_lsh_n_bands=n_bands,
        )
        metrics = CorrTrack.compute_metrics_bf(
            ct, bf_corr_flags, windows=True, total_pairs_bf=bf_tested,
        )
        stage1_trials.append({
            "n_vectors": n_vectors, "n_bands": n_bands,
            "recall": metrics.get("recall"), "tested_candidates": ct.tested_candidates,
        })
        del ct
        gc.collect()

    meeting_target = [t for t in stage1_trials if t["recall"] is not None and t["recall"] >= TARGET_RECALL]
    if meeting_target:
        best_structural = min(meeting_target, key=lambda t: t["tested_candidates"])
        structural_met_target = True
    else:
        best_structural = max(stage1_trials, key=lambda t: (t["recall"] if t["recall"] is not None else -1.0))
        structural_met_target = False

    n_vectors_sel = best_structural["n_vectors"]
    n_bands_sel = best_structural["n_bands"]

    best_gamma_trial, gamma_met_target, n_gamma_trials, gamma_default = calibrate_gamma(
        cell_data, ids, window_size, window_step, n_lags, corr_threshold,
        target_occupancy, n_steps, seed, bf_corr_flags, bf_tested,
        n_vectors=n_vectors_sel, candidate_lsh_n_bands=n_bands_sel,
    )

    return {
        "n_vectors": n_vectors_sel, "n_bands": n_bands_sel,
        "structural_met_target": structural_met_target,
        "n_structural_trials": len(stage1_trials),
        "gamma_trial": best_gamma_trial, "gamma_met_target": gamma_met_target,
        "n_gamma_trials": n_gamma_trials, "gamma_default": gamma_default,
    }


def run_one_cell(writer, cell_id, dataset_kind, nested_subsample, m, L, window_step,
                  window_size, corr_threshold, target_occupancy, n_steps, seed,
                  corr_prop_requested, train_data, ids, max_m_bruteforce, bf_output_dir,
                  z_used=None, achieved_z=None, corr_prop_achieved=None):
    n_lags = (L - 1) * window_step
    # (2026-08-24) run_and_log_bruteforce's own internal loop
    # (execute_corrtrack_pass) processes EVERY window_step-sized chunk
    # across the WHOLE array it's given (`for start in range(0,
    # data.shape[1]-window_step+1, window_step)`), not a bounded n_steps
    # count -- unlike run_smart_cell, which explicitly loops only n_steps
    # times. Passing the full, generously-sized cached dataset (sized to
    # cover the LONGEST n_steps/L combination, for caching/reuse across
    # cells) directly into brute force silently made it process far more
    # internal steps than intended -- a real bug caught by the smoke test
    # timing out. Slice BOTH runs down to the exact span this cell's
    # n_steps needs before running either, so brute force and the smart
    # run are measured over the identical, bounded data span.
    span_needed = window_size + n_steps * window_step
    cell_data = train_data[:, :span_needed]
    default_gamma = float(corr_threshold)

    row = {
        "cell_id": cell_id, "dataset_kind": dataset_kind, "nested_subsample": nested_subsample,
        "m": m, "L": L, "n_lags": n_lags, "window_step": window_step, "window_size": window_size,
        "corr_threshold": corr_threshold, "target_occupancy": target_occupancy,
        "n_steps": n_steps, "seed": seed,
        "corr_prop_requested": corr_prop_requested, "corr_prop_achieved": corr_prop_achieved,
        "z_used": z_used, "achieved_z": achieved_z,
        "n_vectors_selected": DEFAULT_N_VECTORS, "n_bands_selected": DEFAULT_N_BANDS,
        "n_vectors_bands_calibrated": False, "n_vectors_bands_met_target_recall": None,
        "n_vectors_bands_trials_tried": 0,
        "bf_measured": False, "bf_runtime": None, "bf_cand_time": None, "bf_val_time": None,
        "bf_monit_time": None, "bf_tested": None, "bf_correlated": None, "bf_n_gt_for_calibration": None,
        "gamma_calibrated": False, "gamma_default": default_gamma, "gamma_selected": default_gamma,
        "gamma_trials_tried": 0, "gamma_met_target_recall": None, "target_recall": TARGET_RECALL,
        "speedup": None, "precision": None, "recall": None, "f1_score": None,
    }

    bf_record = None
    bf_corr_flags = None
    if m <= max_m_bruteforce:
        bf_cache_key = (dataset_kind, m, L, n_steps, seed, round(corr_prop_requested or 0.0, 6))
        bf_output_csv = str(bf_output_dir / f"bf_{dataset_kind}_m{m}_L{L}_n{n_steps}_s{seed}.csv")
        bf_record, bf_corr_flags = run_bruteforce_once(
            bf_cache_key, cell_data, ids, window_size, window_step, n_lags,
            corr_threshold, bf_output_csv,
        )
        row["bf_measured"] = True
        row["bf_runtime"] = bf_record.get("runtime")
        row["bf_cand_time"] = bf_record.get("cand_time")
        row["bf_val_time"] = bf_record.get("val_time")
        row["bf_monit_time"] = bf_record.get("monit_time")
        row["bf_tested"] = bf_record.get("tested")
        row["bf_correlated"] = bf_record.get("correlated")
        row["bf_n_gt_for_calibration"] = bf_record.get("correlated")

    if bf_corr_flags is not None:
        result = calibrate_hyperparams(
            cell_data, ids, window_size, window_step, n_lags, corr_threshold,
            target_occupancy, n_steps, seed, bf_corr_flags, bf_record.get("tested"),
        )
        best = result["gamma_trial"]
        row["n_vectors_selected"] = result["n_vectors"]
        row["n_bands_selected"] = result["n_bands"]
        row["n_vectors_bands_calibrated"] = True
        row["n_vectors_bands_met_target_recall"] = result["structural_met_target"]
        row["n_vectors_bands_trials_tried"] = result["n_structural_trials"]
        row["gamma_calibrated"] = True
        row["gamma_selected"] = best["gamma"]
        row["gamma_trials_tried"] = result["n_gamma_trials"]
        row["gamma_met_target_recall"] = result["gamma_met_target"]
        row["precision"] = best["metrics"].get("precision")
        row["recall"] = best["metrics"].get("recall")
        row["f1_score"] = best["metrics"].get("f1_score")
        row["n_steps"] = best["n_steps_run"]
        row["sketch_time"] = best["sketch_time"]
        row["candidate_time"] = best["candidate_time"]
        row["validation_time"] = best["validation_time"]
        row["monitor_time"] = best["monitor_time"]
        row["smart_wall_time"] = best["wall_time"]
        row["validated_candidates"] = best["validated_candidates"]
        row["total_candidates"] = best["total_candidates"]
        row["tested_candidates"] = best["tested_candidates"]
        if best["wall_time"] > 0 and bf_record.get("runtime"):
            row["speedup"] = bf_record["runtime"] / best["wall_time"]
    else:
        # No brute-force ground truth available at this m (too large to
        # measure exactly) -- can't calibrate anything against it, fall
        # back to the default gamma and the documented default (n_vectors,
        # n_bands) (see DEFAULT_N_VECTORS/DEFAULT_N_BANDS -- the middle of
        # each grid, already set in `row` above); speed is still measured,
        # recall/precision are not.
        ct, smart_wall_time, n_steps_run = run_smart_cell(
            cell_data, ids, window_size, window_step, n_lags, corr_threshold,
            target_occupancy, n_steps, seed, candidate_cosine_threshold=default_gamma,
            n_vectors=DEFAULT_N_VECTORS, candidate_lsh_n_bands=DEFAULT_N_BANDS,
        )
        row["n_steps"] = n_steps_run
        row["sketch_time"] = ct.sketch_time
        row["candidate_time"] = ct.candidate_time
        row["validation_time"] = ct.validation_time
        row["monitor_time"] = ct.monitor_time
        row["smart_wall_time"] = smart_wall_time
        row["validated_candidates"] = ct.validated_candidates
        row["total_candidates"] = ct.total_candidates
        row["tested_candidates"] = ct.tested_candidates
        del ct
        gc.collect()

    del bf_record, bf_corr_flags
    gc.collect()
    # (2026-08-25) resource.getrusage(RUSAGE_SELF).ru_maxrss is this
    # process's own peak RSS (KB on Linux) since it started -- since each
    # cell already runs in its own fresh subprocess (see run_worker_cell),
    # this is a clean, precise per-(cell, n_steps) reading with one caveat:
    # a worker subprocess calls run_one_cell twice (long then short
    # n_steps), and ru_maxrss is monotonic non-decreasing over the
    # process's lifetime, so the SECOND call's reading is really "peak
    # across both calls so far," not that call's own isolated peak. Still
    # useful for spotting which cells are memory-heavy -- see the new
    # --worker-memory-limit-gb safety net this is meant to help tune.
    try:
        import resource as _resource
        row["peak_rss_gb"] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        row["peak_rss_gb"] = None
    writer.writerow(row)
    return row


def build_grid(args):
    """Factorial + OFAT design per the approved plan: a small full
    factorial over (m, L) at one representative (corr_prop, target_occ),
    plus one-factor-at-a-time extensions for each axis's own scaling law,
    plus a full corr_prop/target_occupancy sweep at one representative
    (m, L) point. Real-data anchors are appended separately (measured
    corr_prop, not swept).

    (2026-08-24) m_values may include "timing-only" points above
    max_m_bruteforce (run_one_cell already falls back to default-gamma,
    no-brute-force timing when m exceeds it -- see run_one_cell's own
    comment). Those points still feed the m sweep (needed for the
    sk_time/cand_time~f(m) cost-model fit reaching towards m=10,000) but
    must NOT become mid_m, since mid_m anchors the L/corr_prop/occupancy
    OFAT sweeps, which need real recall/precision, not timing-only rows.
    """
    m_values = args.m_values
    l_values = args.l_values
    corr_prop_values = args.corr_prop_values
    occ_values = args.target_occupancy_values
    full_m_values = [m for m in m_values if m <= args.max_m_bruteforce] or m_values
    mid_m = full_m_values[len(full_m_values) // 2]
    mid_l = l_values[len(l_values) // 2]
    mid_cp = corr_prop_values[len(corr_prop_values) // 2]
    mid_occ = occ_values[0]

    cells = []
    seen = set()

    def add(m, L, corr_prop, occ):
        key = (m, L, round(corr_prop, 6), occ)
        if key in seen:
            return
        seen.add(key)
        cells.append({"m": m, "L": L, "corr_prop": corr_prop, "target_occupancy": occ})

    # (a) core m x L factorial at representative corr_prop/occupancy
    for m in (m_values[0], m_values[-1]):
        for L in (l_values[0], l_values[-1]):
            add(m, L, mid_cp, mid_occ)

    # (b) OFAT: full m sweep at mid L; full L sweep at mid m
    for m in m_values:
        add(m, mid_l, mid_cp, mid_occ)
    for L in l_values:
        add(mid_m, L, mid_cp, mid_occ)

    # (c) full corr_prop / target_occupancy sweep at the mid (m, L) point
    for cp in corr_prop_values:
        add(mid_m, mid_l, cp, mid_occ)
    for occ in occ_values:
        add(mid_m, mid_l, mid_cp, occ)

    return cells


def _cell_resume_key(spec):
    """Matches a pending cell spec to rows already written to the CSV by a
    prior (possibly interrupted) run of the SAME command -- see
    _load_completed_cell_counts. Deliberately excludes n_steps: a cell is
    considered done once BOTH its n_steps rows exist, checked via the
    count (>=2), not by matching the exact recorded n_steps value (which
    can differ slightly from what was requested if run_smart_cell had to
    truncate against a short data span)."""
    corr_prop = spec.get("corr_prop")
    return (
        spec["dataset_kind"], str(spec["m"]), str(spec["L"]),
        "" if corr_prop is None else str(corr_prop),
        str(spec["target_occupancy"]),
    )


def _load_completed_cell_counts(output_csv):
    """(2026-08-25) Repeated OS-level restarts (WSL restarts, full machine
    restarts) have killed this sweep mid-run multiple times; every prior
    relaunch started from a truncated CSV, silently discarding already-
    completed work and wasting wall-clock time re-doing it before ever
    reaching cells not yet attempted. Reads any existing output file's own
    rows to find out which cells (by _cell_resume_key, ignoring n_steps)
    already have both their n_steps rows written, so main() can skip
    re-dispatching them. Returns {} (resume as a no-op) on any read
    failure -- a corrupt/partial CSV should never block a fresh attempt."""
    path = Path(output_csv)
    if not path.exists():
        return {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (
                    row.get("dataset_kind"), row.get("m"), row.get("L"),
                    row.get("corr_prop_requested") or "", row.get("target_occupancy"),
                )
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        return {}
    return counts


def _migrate_csv_header_if_needed(output_csv, columns):
    """(2026-08-25) Resume mode deliberately never truncates/rewrites an
    existing output file -- that's the whole point -- but that also means
    if CSV_COLUMNS gains new columns (as it did today, peak_rss_gb) between
    when a file was first created and a later resume of the same command,
    the on-disk header silently goes stale relative to the columns new
    rows actually write (confirmed happening in practice: header had 47
    fields, rows being appended had 48). ONLY safe to call from main()
    BEFORE any worker subprocess is dispatched -- at that point nothing
    else holds this file open, so rewriting just the header line carries
    no race/corruption risk (unlike doing this while a worker's `open(...,
    "a")` handle is live, which could lose whatever that worker appends
    next). Only ever APPENDS columns are assumed (never reorders/removes)
    -- under that discipline, existing rows simply read back with the new
    trailing column(s) blank via csv.DictReader's own restval handling,
    which is exactly correct (those rows never had that data)."""
    path = Path(output_csv)
    if not path.exists():
        return
    with open(path, newline="") as fh:
        lines = fh.readlines()
    if not lines:
        return
    current_header = lines[0].rstrip("\r\n")
    expected_header = ",".join(columns)
    if current_header == expected_header:
        return
    print(f"Migrating stale CSV header in {output_csv}: "
          f"{len(current_header.split(','))} -> {len(columns)} columns "
          f"(CSV_COLUMNS changed since this file was created; existing data rows are "
          f"untouched, only the header line is corrected).")
    lines[0] = expected_header + "\n"
    with open(path, "w", newline="") as fh:
        fh.writelines(lines)


def write_run_manifest(args, timing_only_m):
    """(2026-08-25) The CSV records every SWEPT parameter per-row, but
    several consequential parameters are FIXED across the whole run and
    were never columns at all -- data_representation/candidate_backend/
    validation_metric/neg_corr (run_smart_cell), the synthetic base
    process and corr_sign (datasets/synth_loader.py's own defaults, never
    overridden by load_synthetic_cell), and every CLI arg controlling the
    grid itself. Without this, reproducing or correctly interpreting the
    CSV later requires cross-referencing this exact version of the
    script's source -- user asked "what is being saved exactly" and
    "I will need to analyse that in the end", so this writes everything
    needed to interpret the CSV standalone, once per run, next to it."""
    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_csv": args.output_csv,
        "git_commit": _current_git_commit(),
        "cli_args": vars(args),
        "derived": {
            "timing_only_m_values": timing_only_m,
            "max_m_bruteforce": args.max_m_bruteforce,
            "synth_max_lag": (min(args.l_values) - 1) * args.window_step,
        },
        "fixed_corrtrack_config": {
            "data_representation": "sketch_proj",
            "candidate_backend": "lsh_approx",
            "validation_metric": "pearson",
            "neg_corr": True,
            "preprocess": False,
            "exec": "sequential",
            "candidate_apply_dot_gamma_filter": True,
            "candidate_apply_hamming_filter": False,
            "candidate_lsh_max_candidates_per_query": 0,
            "note": "(2026-08-25) n_vectors, candidate_lsh_n_bands, and gamma "
                    "(candidate_cosine_threshold) were all found to independently affect "
                    "recall (see docs/implementation_log.md's 2026-08-25 (b) entry) and are "
                    "no longer fixed here -- see hyperparameter_tuning below for the search "
                    "space; the SELECTED value per cell is in the CSV's n_vectors_selected/"
                    "n_bands_selected/gamma_selected columns. The Hamming pre-filter (a "
                    "separate, lossy, gamma-independent filter stage found during the same "
                    "audit) is explicitly OFF by user decision; dot-gamma filter always ON "
                    "(the real recall gate this sweep tunes); candidate volume per query left "
                    "unbounded.",
        },
        "hyperparameter_tuning": {
            "method": "staged: (n_vectors, n_bands) structural search at gamma=0.0 "
                      "(isolates the LSH retrieval stage's own recall ceiling), then "
                      "calibrate_gamma's existing gamma-trial mechanism within the winning "
                      "combo -- see calibrate_hyperparams' own docstring for the full "
                      "rationale, including why a full joint grid over all three was "
                      "rejected as too expensive.",
            "n_vectors_grid": list(N_VECTORS_GRID),
            "n_bands_grid": list(N_BANDS_GRID),
            "gamma_trial_offsets": list(GAMMA_TRIAL_OFFSETS),
            "target_recall": TARGET_RECALL,
            "default_n_vectors_for_timing_only_cells": DEFAULT_N_VECTORS,
            "default_n_bands_for_timing_only_cells": DEFAULT_N_BANDS,
        },
        "fixed_bruteforce_config": {
            "baseline_mode": "bruteforce",
            "exec": "sequential",
            "parallel_sketch": False, "parallel_candidates": False, "parallel_validation": False,
        },
        "fixed_synthetic_generator_config": {
            "base_proc": {"type": "ar1", "phi": 0.6, "sigma": 1.0},
            "corr_sign": "both",
            "threshold": 0.7,
            "volatility_equalizer": None,
            "num_templates": "auto",
            "note": "base_proc/threshold are datasets/synth_loader.py's own setdefault() "
                    "values -- load_synthetic_cell never overrides them, so every synthetic "
                    "cell in this sweep uses stationary AR(1) data regardless of m/L/"
                    "corr_prop. corr_sign is explicitly set to 'both' by load_synthetic_cell "
                    "(2026-08-25 fix -- matches neg_corr=True, so negative correlations are "
                    "actually exercised by the ground truth, not just structurally supported "
                    "but never tested).",
        },
        "real_dataset_module": REAL_DATASET_MODULE,
    }
    manifest_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Wrote {manifest_path}")


def _current_git_commit():
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5, check=False,
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="LSH cost-model sweep (supervisor extrapolation + publication study).")
    parser.add_argument("--smoke", action="store_true", help="Run only the tiny smoke-test grid, both dataset kinds.")
    parser.add_argument("--m-values", type=int, nargs="+", default=[50, 100, 200, 300, 1000, 3000],
                         help="m points below --max-m-bruteforce get real brute-force ground truth "
                         "(recall/precision/gamma calibration); points above it run timing-only "
                         "(speed/cost-model rows only, default gamma, no recall/precision) -- see "
                         "run_one_cell. Kept <=300 for the brute-force-paired points per the user's "
                         "own machine constraint; 1000/3000 are timing-only extensions for the "
                         "cost-model fit's reach towards m=10,000.")
    parser.add_argument("--l-values", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--corr-prop-values", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.10])
    parser.add_argument("--target-occupancy-values", type=float, nargs="+", default=[3.0, 10.0])
    parser.add_argument("--real-m-values", type=int, nargs="+", default=[20, 50, 100, 121])
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-step", type=int, default=16)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--n-steps-long", type=int, default=2000)
    parser.add_argument("--n-steps-short", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-m-bruteforce", type=int, default=300,
                         help="Real brute-force baseline is only measured for m at or below this "
                         "(also the user's stated feasible ceiling for their machine); larger m runs "
                         "timing-only and relies on the fitted closed-form cost model instead (real "
                         "brute force at m=3000, L=64 would be prohibitively slow/memory-heavy).")
    parser.add_argument("--output-csv", type=str, default=str(RESULT_DIR / "lsh_cost_sweep.csv"))
    parser.add_argument("--fresh", action="store_true",
                         help="Truncate --output-csv and start over, even if it already has "
                         "rows from a prior (possibly interrupted) run. Default is to RESUME: "
                         "skip cells whose both n_steps rows are already present. Added "
                         "2026-08-25 after repeated OS-level restarts (WSL, full machine) each "
                         "killed the sweep mid-run and every previous relaunch silently "
                         "discarded already-completed work by truncating the CSV.")
    parser.add_argument("--worker-memory-limit-gb", type=float, default=6.0,
                         help="Hard RLIMIT_AS cap (GB) applied to each worker cell subprocess "
                         "before it starts -- a single cell exceeding this fails cleanly and is "
                         "skipped (existing WORKER FAILED handling) instead of exhausting the "
                         "whole machine's RAM+swap and crashing WSL. Default 6.0 chosen for this "
                         "machine's observed 7.8GB RAM (~3GB free) + 8GB swap, leaving headroom "
                         "for the orchestrator process and the rest of the system; pass 0 to "
                         "disable. Added 2026-08-25 after repeated WSL crashes traced to OOM.")
    parser.add_argument("--worker-cell-spec", type=str, default=None,
                         help=argparse.SUPPRESS)  # internal: subprocess worker mode, JSON spec
    args = parser.parse_args()

    if args.worker_cell_spec is not None:
        run_worker_cell(json.loads(args.worker_cell_spec))
        return

    if args.smoke:
        # 500 > max_m_bruteforce (300, default) -- exercises the
        # timing-only (no brute force, default gamma) fallback path too,
        # so the smoke test catches breakage in that path before a full run.
        args.m_values = [100, 300, 500]
        args.l_values = [8, 32]
        args.corr_prop_values = [0.01]
        args.target_occupancy_values = [3.0]
        args.real_m_values = [20, 50]
        # (2026-08-24) n_steps=30 (this override's original value) gave a
        # data span (window_size + 30*window_step = 736 samples) too short
        # for calibration/recall to be representative -- confirmed via a
        # direct A/B: the SAME (m=100, L=8) cell went from recall~0.32 at
        # n_steps=30 to recall~0.97 at n_steps=500, matching a separate,
        # earlier finding that recall itself (not just calibration) needs
        # enough steps for injected true positives to actually be in view.
        # 150/60 is short enough to stay a fast smoke test while still
        # being long enough for recall/calibration diagnostics to mean
        # something.
        args.n_steps_long = 150
        args.n_steps_short = 60
        args.output_csv = str(RESULT_DIR / "lsh_cost_sweep_smoke.csv")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    bf_output_dir = RESULT_DIR / "bf_runs"
    bf_output_dir.mkdir(parents=True, exist_ok=True)

    cells = build_grid(args)
    n_synth = args.window_size + args.n_steps_long * args.window_step

    real_l_values = [args.l_values[0], args.l_values[len(args.l_values) // 2]]

    timing_only_m = sorted(m for m in args.m_values if m > args.max_m_bruteforce)
    print(f"Grid: {len(cells)} synthetic cells + "
          f"{len(args.real_m_values) * len(real_l_values)} real-data anchor cells, "
          f"each x {{long, short}} n_steps, each in its own subprocess "
          f"(2026-08-24: isolates each cell's memory -- a real, confirmed SIGKILL/"
          f"exit-137 was hit scaling m up within one long-running process, even "
          f"after explicit del+gc.collect() cleanup between trials/cells; per-cell "
          f"subprocess isolation sidesteps it regardless of the exact retention "
          f"source, matching this project's own 'fix the failure mode, not just "
          f"the symptom' discipline when root-causing further wasn't worth the time; "
          f"it also means a failed cell -- e.g. one of these larger m values being "
          f"infeasible on this machine -- just logs 'WORKER FAILED' and is skipped, "
          f"every other cell's result stays in the CSV)."
          + (f" Timing-only (no brute force, default gamma, speed/cost-model rows "
             f"only -- m > max_m_bruteforce={args.max_m_bruteforce}): m in {timing_only_m}."
             if timing_only_m else ""))

    completed_counts = {} if args.fresh else _load_completed_cell_counts(args.output_csv)
    if not args.fresh and completed_counts:
        print(f"Resuming: found {len(completed_counts)} already-attempted cell(s) in "
              f"{args.output_csv} from a prior run -- cells with both n_steps rows present "
              f"will be skipped. Pass --fresh to discard this and start over.")

    if args.fresh or not Path(args.output_csv).exists():
        with open(args.output_csv, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
    else:
        # Safe here specifically: no worker subprocess has been dispatched
        # yet this run, so nothing else holds the file open.
        _migrate_csv_header_if_needed(args.output_csv, CSV_COLUMNS)

    write_run_manifest(args, timing_only_m)

    # (2026-08-24) Bounded to the grid's SMALLEST L (not any individual
    # cell's own L) since one cached synthetic dataset is reused across
    # every L value sharing the same (m, corr_prop, seed) -- see
    # load_synthetic_cell's own comment for why this must be the min, not
    # per-cell.
    synth_max_lag = (min(args.l_values) - 1) * args.window_step

    shared = dict(
        window_size=args.window_size, window_step=args.window_step,
        corr_threshold=args.corr_threshold, seed=args.seed,
        n_steps_long=args.n_steps_long, n_steps_short=args.n_steps_short,
        max_m_bruteforce=args.max_m_bruteforce, output_csv=args.output_csv,
        bf_output_dir=str(bf_output_dir), n_synth=n_synth, synth_max_lag=synth_max_lag,
    )

    # (2026-08-25) User: "run from smallest to bigger experiments so if it
    # takes too long, I have some results to show my supervisors" --
    # cells previously ran in build_grid's own design-driven order (the
    # core m x L factorial puts the LARGEST m near the front, right after
    # the smallest, since it iterates (m_values[0], m_values[-1])). Sort
    # all cells (synthetic + real, combined into one list) before
    # dispatch instead: full (brute-force-paired) cells first -- these are
    # the ones with a complete, presentable recall/precision/speedup
    # story -- ascending by (m, L) within that group, then timing-only
    # cells (m > max_m_bruteforce, cost-model rows only, nothing
    # presentable on their own yet without the cost-model fit) last,
    # also ascending by (m, L). Per-cell subprocess isolation + the
    # incrementally-flushed CSV (see the Grid: message above) already
    # mean an interrupted or cut-off run keeps everything completed so
    # far, in this now-deliberately-smallest-first order.
    pending = []
    for cell in cells:
        pending.append(dict(shared, dataset_kind="synthetic",
                             m=cell["m"], L=cell["L"], corr_prop=cell["corr_prop"],
                             target_occupancy=cell["target_occupancy"]))
    for m in args.real_m_values:
        for L in real_l_values:
            pending.append(dict(shared, dataset_kind="real", m=m, L=L, corr_prop=None,
                                 target_occupancy=args.target_occupancy_values[0]))

    pending.sort(key=lambda spec: (spec["m"] > args.max_m_bruteforce, spec["m"], spec["L"]))

    if completed_counts:
        n_before = len(pending)
        pending = [spec for spec in pending if completed_counts.get(_cell_resume_key(spec), 0) < 2]
        skipped = n_before - len(pending)
        if skipped:
            print(f"Skipping {skipped} cell(s) already fully completed (both n_steps rows present).")

    for cell_id, spec in enumerate(pending, start=1):
        spec["cell_id"] = cell_id
        _run_worker_subprocess(spec, memory_limit_gb=args.worker_memory_limit_gb)

    print(f"Wrote {args.output_csv}")


def _make_memory_limiter(limit_gb):
    """(2026-08-25) User's WSL crashed repeatedly from OOM (confirmed via
    their own debugging: 7.8GB RAM, only ~3GB free, before they raised
    swap to 8GB) -- per-cell subprocess isolation already contains a
    LEAKED cell's memory to that one process, but does nothing to cap a
    SINGLE cell's own peak if it alone is large enough to exhaust the
    whole machine before the OS even gets a chance to reclaim it on exit.
    A hard RLIMIT_AS (virtual address space) ceiling on each worker
    subprocess, applied via preexec_fn before exec, turns "this one cell's
    allocation pattern blows past what the machine can hold" into a
    contained allocation failure inside that ONE subprocess -- caught by
    the existing WORKER FAILED/skip handling in _run_worker_subprocess
    exactly like any other cell failure, instead of taking the whole WSL
    VM down with it. Returns None (no limiting) if limit_gb is falsy."""
    if not limit_gb or limit_gb <= 0:
        return None
    import resource
    limit_bytes = int(limit_gb * (1024 ** 3))

    def _apply():
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return _apply


def _run_worker_subprocess(spec, memory_limit_gb=None):
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, os.path.abspath(__file__), "--worker-cell-spec", json.dumps(spec)],
        check=False, preexec_fn=_make_memory_limiter(memory_limit_gb),
    )
    tag = f"m={spec['m']} L={spec['L']} corr_prop={spec.get('corr_prop')}"
    if result.returncode != 0:
        # (2026-08-25) A worker killed by hitting the memory limit or by
        # the OS OOM-killer both show up here as a nonzero/negative
        # returncode -- can't always distinguish which from the exit code
        # alone (RLIMIT_AS failures can surface as a Python MemoryError
        # traceback -> exit 1, or as a raw SIGSEGV in C-extension code
        # that doesn't handle malloc failure gracefully -> negative
        # signal-coded returncode); flagged either way so the user isn't
        # left guessing whether it was this new safety net or something
        # else.
        suspected_oom = result.returncode < 0 or result.returncode == 1
        print(f"[cell {spec['cell_id']}] {spec['dataset_kind']} {tag} -> "
              f"WORKER FAILED (exit {result.returncode}"
              f"{', possibly hit the memory limit or an OOM kill' if suspected_oom and memory_limit_gb else ''}"
              f"), skipping this cell")


def run_worker_cell(spec):
    """Runs inside a fresh subprocess (per _run_worker_subprocess) so any
    per-cell memory growth is reclaimed by the OS on exit, regardless of
    its exact source -- see main()'s own comment for why this exists."""
    bf_output_dir = Path(spec["bf_output_dir"])
    with open(spec["output_csv"], "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer = _FlushingWriter(writer, fh)

        if spec["dataset_kind"] == "synthetic":
            m, L, corr_prop, occ = spec["m"], spec["L"], spec["corr_prop"], spec["target_occupancy"]
            data, ids, z_used, achieved_z, achieved_cp = load_synthetic_cell(
                m, spec["n_synth"], corr_prop, spec["window_size"], spec["seed"], spec["synth_max_lag"],
                threshold=spec["corr_threshold"], window_step=spec["window_step"],
            )
            train_data, ids_n = prepare_training_data(
                data, ids, n_year=spec["n_synth"], n_var=m, train_ratio=1.0,
            )
            for n_steps in (spec["n_steps_long"], spec["n_steps_short"]):
                row = run_one_cell(
                    writer, spec["cell_id"], "synthetic", False, m, L, spec["window_step"],
                    spec["window_size"], spec["corr_threshold"], occ, n_steps, spec["seed"],
                    corr_prop, train_data, list(ids_n), spec["max_m_bruteforce"], bf_output_dir,
                    z_used=z_used, achieved_z=achieved_z, corr_prop_achieved=achieved_cp,
                )
                print(f"[cell {spec['cell_id']}] synthetic m={m} L={L} corr_prop={corr_prop} "
                      f"occ={occ} n_steps={n_steps} -> speedup={row['speedup']} recall={row['recall']}")
        else:
            m, L, occ = spec["m"], spec["L"], spec["target_occupancy"]
            train_data, ids_n = load_real_cell(m)
            for n_steps in (spec["n_steps_long"], spec["n_steps_short"]):
                row = run_one_cell(
                    writer, spec["cell_id"], "real", True, m, L, spec["window_step"],
                    spec["window_size"], spec["corr_threshold"], occ, n_steps, spec["seed"],
                    None, train_data, ids_n, spec["max_m_bruteforce"], bf_output_dir,
                )
                print(f"[cell {spec['cell_id']}] real m={m} L={L} n_steps={n_steps} -> "
                      f"speedup={row['speedup']} recall={row['recall']}")



class _FlushingWriter:
    """Wraps a csv.DictWriter to flush after every row (incremental write,
    per the plan) and coerce None -> '' for clean CSV output."""

    def __init__(self, writer, fh):
        self._writer = writer
        self._fh = fh

    def writerow(self, row):
        clean = {k: ("" if v is None else v) for k, v in row.items()}
        self._writer.writerow(clean)
        self._fh.flush()


if __name__ == "__main__":
    main()
