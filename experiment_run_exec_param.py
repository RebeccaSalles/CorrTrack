"""Execution-level configuration for CorrTrack experiments.

This module exposes all CorrTrack and CorrTrack_compare parameters alongside
sensible experiment defaults. Required runtime values (data arrays, ids, and
param grids) remain provided by the caller.
"""

# -----------------------------------------------------------------------------
# CorrTrack / CorrTrack_compare core parameters.
# -----------------------------------------------------------------------------
WINDOW_SIZE = 7 * 24
WINDOW_STEP = 12
BASIC_WINDOW = None
N_LAGS = 7 * 24
CORR_THRESHOLD = 0.7

NEG_CORR = False

# CorrTrack main-run validation control.
CORR_VAL = True

# CorrTrack monitoring control (main/brute-force/comparison stages).
MONITOR = True

# Track minimum pairwise distance during validation (used by pair_min_dist / recall_min).
# Disable to skip min-distance bookkeeping while keeping correlation validation enabled.
TRACK_MIN_DIST = True

# Hyper-parameter search validation control.
CORR_VAL_OPTIM = False

# CorrTrack_compare-specific controls.
RECALL_BY_WINDOW = True
TRAIN_RATIO = 0.3
TARGET_RECALL = 0.95
# If no hyperopt row reaches TARGET_RECALL, keep rows whose recall is at least
# this fraction of the best available recall before applying speedup/cand_w.
RECALL_FALLBACK_NEAR_RATIO = 0.98
# Keep hyperopt candidates whose speedup is at least this fraction of the best
# feasible speedup, then pick the one with the lowest candidate count.
SPEEDUP_NEAR_RATIO = 0.98

# Artifact persistence for run scripts that write out intermediate outputs.
ARTIFACT_MODE = "buffered"
ARTIFACT_BUFFER_MAX_ROWS = 150000
SAVE_ONLY_REQUIRED_ARTIFACTS = True
SAVE_MAXLAG_ARTIFACTS = True
DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = True

VERBOSE = False
TESTING = False

PARALLEL_SKETCH = False
PARALLEL_CANDIDATES = False
PARALLEL_VALIDATION = None
MAX_WORKERS = 0
