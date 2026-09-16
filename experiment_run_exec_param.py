"""Execution-level configuration for CorrTrack experiments.

This module exposes all CorrTrack and CorrTrack_compare parameters alongside
sensible experiment defaults. Required runtime values (data arrays, ids, and
param grids) remain provided by the caller.
"""

# -----------------------------------------------------------------------------
# CorrTrack / CorrTrack_compare core parameters.
# -----------------------------------------------------------------------------
# WINDOW_SIZE selects single- vs multi-window execution by itself: a scalar
# (or single-element list) runs plain single-window CorrTrack; a list of
# >=2 distinct sizes (e.g. [64, 256]) runs CorrTrackMultiWindow instead --
# shares one incremental sketch computation across all requested sizes
# rather than running independent CorrTrack instances per size. A derived
# shorter-size sketch is NOT bit-identical to a standalone same-size run
# (the "tail-toggle convention", intentional -- see docs/implementation_
# log.md's 2026-07-28(a) entry). (The separate WINDOW_SIZES parameter was
# dropped 2026-07-30 -- WINDOW_SIZE alone now covers both cases.)
WINDOW_SIZE = 7 * 24
WINDOW_STEP = 12
BASIC_WINDOW = None
N_LAGS = 7 * 24
CORR_THRESHOLD = 0.7

NEG_CORR = True

# Baseline used by corrtrack_run_bruteforce / full experiments.
# - "bruteforce": stable exhaustive baseline.
# - "exact_stomp": exact incremental rolling-dot baseline for benchmarking.
# - "filcorr": Zhong/Souza/Mueen (ICDM 2020) competitor -- band-pass Pearson via
#   Parseval's identity on FFT coefficients. See FILCORR_FS/FT below and
#   Candidates_BF_FilCorr in library_corrtrack_parallel.py.
BASELINE_MODE = "bruteforce"
# (2026-09-16) "tsubasa" is also accepted: Xu/Liu/Nargesian SIGMOD 2022, exact Pearson
#   from per-basic-window sketches (their Lemma 1). Uses BASIC_WINDOW as the segment
#   size; has no lag support, so it requires N_LAGS = 0 (the arm refuses otherwise).

# FilCorr pass-band (baseline_mode="filcorr" only). fs=0.0/ft=0.5 (full band,
# DC removed) is mathematically identical to standard Pearson -- keep this the
# default so the baseline is comparable to bruteforce's ground truth; only
# narrow the band for a deliberate band-pass-correlation experiment (a
# different quantity, not to be scored against the unfiltered ground truth).
FILCORR_FS = 0.0
FILCORR_FT = 0.5
FILCORR_SAMPLING_RATE = 1.0
# BRAID / ThinBRAID (baseline_mode="braid"; Sakurai et al. SIGMOD 2005 / TKDD 2010).
#   Paper values. 2*BRAID_B > N_LAGS makes BRAID exact (level 0 covers every lag).
BRAID_B = 16
BRAID_GAMMA = 0.4
BRAID_THIN = False
BRAID_THIN_D0 = 400
BRAID_REPORT_MODE = "all_lags"

# CorrTrack main-run validation control.
CORR_VAL = True

# CorrTrack monitoring control (main/brute-force/comparison stages).
MONITOR = True

# Track minimum pairwise distance during validation (used by pair_min_dist / recall_min).
# Disable to skip min-distance bookkeeping while keeping correlation validation enabled.
TRACK_MIN_DIST = True

# Optional dev validation path. It maintains repeated candidate relations in a
# Cython cache and turns on incremental validation only when the repeated-pair
# rate is high enough. Default remains off because it is workload-sensitive.
HYBRID_VALIDATION = False
HYBRID_VALIDATION_MIN_REPEAT_RATE = 0.25
HYBRID_VALIDATION_DISABLE_RATE = 0.125
HYBRID_VALIDATION_EMA_ALPHA = 0.25
HYBRID_VALIDATION_MIN_CANDIDATES = 256

TARGET_RECALL = 0.95
# If no hyperopt row reaches TARGET_RECALL, keep rows whose recall is at least
# this fraction of the best available recall before applying speedup/cand_w.
RECALL_FALLBACK_NEAR_RATIO = 0.98
# Keep hyperopt candidates whose speedup is at least this fraction of the best
# feasible speedup, then pick the one with the lowest candidate count.
SPEEDUP_NEAR_RATIO = 0.98
TRAIN_RATIO = 0.3

# Proxy-anchor is the only supported hyper-parameter search strategy in this
# stable release. The first TRAIN_RATIO fraction is the admissible causal prefix
# for tuning; the remainder is the holdout used by the end-to-end comparison.
OPTIM_TUNING_MODE = "sampling"
OPTIM_HYPEROPT_STRATEGY = "proxy_anchor"

# -----------------------------------------------------------------------------
# Proxy-anchor hyper-parameter tuning controls.
# -----------------------------------------------------------------------------
# Proxy-anchor samples random causal anchor windows from the admissible train
# prefix, computes exact Pearson labels for each anchor-vs-history pair universe
# once, then evaluates each sketch hyperparameter setting by asking whether the
# CorrTrack candidate search would return those pairs.
#
# Selection rule:
# 1. keep configurations whose bootstrap recall lower bound reaches
#    TARGET_RECALL;
# 2. among feasible configurations, minimize candidate rate
#    candidates / sampled pair rows;
# 3. break ties with higher bootstrap precision and specificity.
#
# Bootstrap unit:
# - anchor positions, not individual pair rows. Each anchor contributes all
#   pair rows generated by its current windows against lagged historical windows.
#   This keeps the confidence interval tied to the actual sampling design.
#
# Safety knobs:
# - OPTIM_PROXY_ANCHOR_COUNT requests how many random anchor positions to sample. 128 is a
#   reasonable moderate default: at typical densities it clears OPTIM_PROXY_BOOTSTRAP_
#   MIN_GT_EVENTS with comfortable margin (78 ground-truth events were found from just 14
#   anchors in this project's own m=150 testing) without being wastefully large. Whether it's
#   actually ENOUGH for a given dataset depends on that dataset's own correlation density,
#   which isn't known ahead of time -- OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED below is the real
#   safety net for that, not this number itself.
# - The pair-row budget that caps total proxy matrix size (formerly a flat
#   OPTIM_PROXY_MAX_PAIR_ROWS constant here) is no longer something to set directly.
#   corrtrack_param_search.py's main() now computes it per-dataset from the actual (m, L) via
#   recommend_proxy_pair_row_budget (library_corrtrack_parallel.py), because a single fixed
#   number silently truncates anchor sampling at some scales (the old flat 50,000 affords less
#   than ONE full anchor's own pair universe at m=150, L=32 -- an est_pairs_per_anchor of
#   ~709K -- which is exactly what happened before this was fixed) while being needlessly
#   small at others. OPTIM_PROXY_PAIR_ROW_HARD_CEILING below is the one thing left to set --
#   a genuine resource budget, not a value derivable from (m, L). If many series make one
#   anchor too expensive relative to that ceiling, the resolver reduces the number of anchors
#   used (down to a floor of 1) rather than silently building an oversized matrix -- and now
#   prints when that happens, rather than only leaving a note in a results CSV.
# - proxy evaluation uses cached sketch distances (sized to match the computed pair-row
#   budget, not a separately-set cache limit -- no reason for the cache to be smaller than the
#   reference it's caching). The streaming BST fallback
#   remains available internally, but release experiments use the cached path.
# - adaptive anchors double the anchor count when the selected configuration has
#   mean bootstrap recall above TARGET_RECALL but the lower confidence bound is
#   still below it, or when the selected row is underpowered. This is a
#   pre-declared sequential sampling rule inside the train prefix, not holdout
#   selection. (2026-09-03) This safety net was previously often INERT in practice: expanding
#   anchor_count accomplishes nothing if OPTIM_PROXY_MAX_PAIR_ROWS is already the binding
#   constraint (observed directly -- an expansion log line fired, the achieved anchor count
#   never actually changed). Fixed by the same per-dataset computation above, which sizes the
#   row budget for OPTIM_PROXY_MAX_ANCHOR_COUNT (this expansion's own ceiling), not just the
#   initial OPTIM_PROXY_ANCHOR_COUNT request.
OPTIM_PROXY_ANCHOR_COUNT = 128
OPTIM_PROXY_RANDOM_SEED = 2468
OPTIM_PROXY_BOOTSTRAP_ENABLED = True
OPTIM_PROXY_BOOTSTRAP_REPEATS = 1000
OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.90
OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS = 30
OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED = True
OPTIM_PROXY_MAX_ANCHOR_COUNT = 256
# (2026-09-03) The one number in this whole chain that's a genuine resource-budget choice, not
# something derivable from (m, L) -- caps how many total proxy pair-rows a single dataset's
# reference construction is allowed to use, regardless of how many anchors that would
# otherwise take to reach OPTIM_PROXY_MAX_ANCHOR_COUNT. 20,000,000 was chosen because
# 10,000,000 rows (half this ceiling) built its reference and ran a full 24-combo proxy-anchor
# hyperopt sweep in ~5 minutes wall-clock in this project's own testing -- a real, checked
# data point, not a guess. Each pair row costs real memory during reference construction (a
# handful of plain Python objects per row, before anything is packed into compact numpy
# arrays), not just wall-clock time, so raise this only with your own available memory/time
# budget in mind -- large (m, L) will correctly use FEWER than OPTIM_PROXY_MAX_ANCHOR_COUNT
# anchors once this ceiling binds, printed visibly by corrtrack_param_search.py rather than
# silently, which is the intended degrade-gracefully behavior, not a bug to raise this to fix.
OPTIM_PROXY_PAIR_ROW_HARD_CEILING = 20_000_000
OPTIM_PROXY_ANCHOR_EXPAND_FACTOR = 2.0
OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS = 1
# How many times to re-run each proxy trial's full sketch+candidate-search
# purely for timing stability (mean is taken; recall/precision/stats are
# deterministic given the fixed seed, so computed only once). Higher =
# less noisy real-time tie-break below, at ~timing_repeats-x proxy-phase
# wall-clock cost.
OPTIM_PROXY_TIMING_REPEATS = 3
# Relative tolerance defining "reasonably close" to the best achievable
# candidate rate among feasible configs. Candidate-count minimization stays
# the dominant hyperopt objective; within this fraction of the best rate,
# real measured execution time (proxy_search_time_total) breaks the tie
# instead of further candidate-count refinements.
OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE = 0.05

# Artifact persistence for run scripts that write out intermediate outputs.
ARTIFACT_MODE = "buffered"
ARTIFACT_BUFFER_MAX_ROWS = 150000
ARTIFACT_MERGE_MODE = "chunks" #merged|chunks
SAVE_ONLY_REQUIRED_ARTIFACTS = True
SAVE_MAXLAG_ARTIFACTS = False
DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = True

VERBOSE = False
TESTING = False

# =============================================================================
# Data representation
# =============================================================================
# -- General / shared --
# "auto" (default -- picks per VALIDATION_METRIC: sketch_proj for pearson,
# sketch_concordance for spearman/kendall, sketch_multichannel for
# dist_corr), "raw" (no representation, exhaustive pairwise enumeration --
# only valid with CANDIDATE_BACKEND "auto"/"brute_force"), "sketch_proj",
# "sketch_concordance", or "sketch_multichannel".
DATA_REPRESENTATION = "auto"

# -- sketch_concordance (VALIDATION_METRIC in {"spearman", "kendall"}) --
# Uses fixed multiscale GAPS (compares x[t] to x[t-g]); only the
# window_step newly-valid comparisons are computed each step (amortized
# O(window_step), not recomputed from scratch), and is inherently robust
# to nonstationary drift by construction.
#
# output_dim = sum(window_size - g for g in gaps) scales linearly with
# window_size when CONCORDANCE_N_GAPS is fixed. CONCORDANCE_N_GAPS=None
# (the default) instead auto-derives the number of gaps from
# CONCORDANCE_TARGET_DIM so output_dim stays roughly bounded regardless of
# window_size -- set an explicit int for a fixed gap count instead
# (bypasses CONCORDANCE_TARGET_DIM).
CONCORDANCE_N_GAPS = None
CONCORDANCE_TARGET_DIM = 738
# The adjacent-timestamp gap (gap=1) has the highest weight but is, on real
# autocorrelated data, the worst tau estimator (local jitter dominates);
# gaps near window_size have the highest variance (as few as 1 sample).
# CONCORDANCE_MIN_GAP/CONCORDANCE_MIN_CAPACITY exclude both extremes. See
# concordance_sketch.py's multiscale_gaps docstring.
CONCORDANCE_MIN_GAP = 8
CONCORDANCE_MIN_CAPACITY = 16
# None auto-derives from CORR_THRESHOLD via derive_concordance_multichannel_
# gamma_from_corr_threshold (a flat constant for genuine multi-gap unions,
# calibrated at ONE config only -- window_size=256, corr_threshold=0.5 --
# see concordance_sketch.py's own docstring for the disclosed, unverified
# assumption about how this should scale with corr_threshold; sweep this
# explicitly rather than trusting the default at other thresholds).
CONCORDANCE_MULTICHANNEL_GAMMA = None

# -- sketch_multichannel (requires VALIDATION_METRIC="dist_corr") --
# See distance_corr_sketch.py's module docstring. K is a real, swept
# hyperparameter -- K=8 is a verified sweet spot (reliable, low-seed-
# variance, cheaper than naive exact dCor); K<=4 is cheaper still but
# measurably less robust for periodic relationships.
DISTANCE_CORR_SKETCH_K = 8
DISTANCE_CORR_SKETCH_FREQ_LOW = 0.1
DISTANCE_CORR_SKETCH_FREQ_HIGH = 10.0
DISTANCE_CORR_SKETCH_FREQ_SEED = 42
# None auto-derives via derive_gate_tau_from_corr_threshold (empirically
# fit, window-size-aware) -- the tier-2 K^2 gate's own threshold.
DISTANCE_CORR_SKETCH_GATE_TAU = None

# =============================================================================
# Candidate search
# =============================================================================
# -- General / shared --
# "auto" (default -- resolves to "lsh_approx"), "lsh_approx",
# "hamming_exact", or "brute_force" (forces DATA_REPRESENTATION to behave
# as "raw"). Recommended: leave at "auto" for production use.
CANDIDATE_BACKEND = "auto"
# gamma, the cosine-similarity bar retrieved candidates must clear
# (candidate_similarity is fixed to "cosine" -- the only mode with a
# working search radius for any current backend). None auto-derives from
# CORR_THRESHOLD.
CANDIDATE_COSINE_THRESHOLD = None

# -- lsh_approx backend --
# SignLSHBandIndex: splits each window's sign(w_hat) pattern into N_BANDS
# independent random bands; a candidate is touched if it shares a bucket
# with the query in ANY band (also checking the bitwise complement, for
# negative correlations). Precision is exact by construction when
# CANDIDATE_APPLY_DOT_GAMMA_FILTER=True (default); recall is an empirical
# property of the data's sign-bit geometry, not a mathematical guarantee.
# band_width is not configurable here -- it is auto-sized from (series
# count) * (n_lagged_windows) so bucket occupancy stays roughly constant
# as scale grows (see CANDIDATE_LSH_TARGET_OCCUPANCY below).
CANDIDATE_LSH_N_BANDS = 64
# SignLSHBandIndex's band_width auto-sizing target (band_width =
# ceil(log2(m*L/target_occupancy))) -- see candidate_kernels.pyx's
# _finalize_sizing. Distinct from CANDIDATE_LSH_N_BANDS above.
CANDIDATE_LSH_TARGET_OCCUPANCY = 3.0
# Final dot+gamma gate after index retrieval (candidate_kernels.pyx's
# _passes_dot_gamma_gate). Default True: every returned pair has passed a
# real dot+gamma check. Disabling it lets every index-retrieved candidate
# flow straight to validation.
CANDIDATE_APPLY_DOT_GAMMA_FILTER = True
# SignLSHBandIndex's cheap full-vector sign-Hamming pre-filter, gating the
# real dot product. Default on (recall-safe, cuts real dot-product work).
CANDIDATE_APPLY_HAMMING_FILTER = True
CANDIDATE_HAMMING_FILTER_MAX_FRAC = 0.40
# SignLSHBandIndex's per-query candidate examination budget cap (0 =
# unlimited). No safe non-zero default is known (band iteration carries no
# similarity ranking) -- kept as an available dial only.
CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY = 0

# -- hamming_exact backend --
# HammingExactIndex: exact packed-bit Hamming pre-filter over the whole
# alive population, no bands/buckets. None auto-derives the Hamming touch
# threshold from gamma via the SimHash relation on the first query
# (recommended); an int overrides it.
CANDIDATE_HAMMING_THRESHOLD = None

# =============================================================================
# Validation
# =============================================================================
# -- General / shared --
# "pearson" (default) stays on the fully bulk-vectorized Cython path
# (_cy_validate_corr_rows, one call handles every candidate row at once).
# "spearman"/"kendall"/"dist_corr" route through _validate_numeric_rows_
# nonlinear instead: still a per-row Python loop (row slicing, constant/
# spike checks, threshold checks), but the metric computation itself inside
# each row is Cython (kendall_tau_cy/spearman_rho_cy/distance_correlation_
# 1d_fast_cy for dist_corr_algorithm="fast") -- no scipy or pure-Python
# Fenwick-tree hotpath remains, but this is not yet the same fully bulk-
# vectorized path Pearson has.
VALIDATION_METRIC = "pearson"

# -- dist_corr --
# "naive" (default) or "fast" (Huo & Szekely-style exact O(w log w)) -- both
# EXACT, not approximate; "fast" only wins wall-clock above roughly w=200-256.
DIST_CORR_ALGORITHM = "naive"

# -- spearman / kendall --
# Xiao (2017) online validator -- opt-in (default off), APPROXIMATE
# alternative to the exact scipy-based per-row validation path -- maintains
# a small per-pair count matrix, updated incrementally as the window
# slides instead of recomputing from the full window every step. Real
# accuracy tradeoff, not free -- see docs/implementation_log.md's
# 2026-07-29(a)/(b) entries for the measured threshold-classification
# mismatch rates (0% on stationary AR1, 1.23%-5.76% on nonstationary
# random-walk data depending on VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_
# THRESHOLD). Never claimed to reproduce the exact scipy value bit-for-bit.
VALIDATION_INCREMENTAL_APPROX = False
# None auto-selects Xiao's own rule-of-thumb (30 for spearman, 100 for kendall).
VALIDATION_INCREMENTAL_M1 = None
VALIDATION_INCREMENTAL_M2 = None
VALIDATION_INCREMENTAL_MAX_AGE_STEPS = 64
# HBR (2009)-style adaptive-update pattern: proactively refresh cutpoints
# once a pair's window mean has drifted this many standard deviations from
# where they were last derived, rather than only reacting to outright
# degeneracy (the reactive NaN fallback stays in place as a backstop
# regardless). None disables the proactive check (reactive-only).
VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD = 0.5

# =============================================================================
# Execution
# =============================================================================
PARALLEL_SKETCH = False
PARALLEL_CANDIDATES = False
PARALLEL_VALIDATION = None
MAX_WORKERS = 0
