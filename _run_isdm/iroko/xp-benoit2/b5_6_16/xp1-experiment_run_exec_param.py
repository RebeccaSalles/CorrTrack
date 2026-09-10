"""Execution-level configuration for CorrTrack experiments.

This module exposes all CorrTrack and CorrTrack_compare parameters alongside
sensible experiment defaults. Required runtime values (data arrays, ids, and
param grids) remain provided by the caller.
"""


PARALLEL_SKETCH     = True
PARALLEL_CANDIDATES = True
PARALLEL_VALIDATION = True
MAX_WORKERS         = 2

CANDIDATE_TASK_BATCHING        = 4    # plafond = n_nodes*2 = 4 tâches max
MIN_SERIES_PER_SKETCH_TASK     = 32   # ~3-4 tâches sketch sur 2 workers (work-stealing)
VALIDATION_ADAPTIVE_CHUNKING   = 4    # 2 workers × 4 = 8 chunks val
MONITOR_OUT_BATCHING           = 64


TRAIN_RATIO = 0.3333
TARGET_RECALL = 0.95

PLOT = True
ARTIFACT_MODE = "buffered"
ARTIFACT_BUFFER_MAX_ROWS = 150000
SAVE_ONLY_REQUIRED_ARTIFACTS = True
SAVE_MAXLAG_ARTIFACTS = True
DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = True

TRAIN_DISTANCES = False
BASIC_WINDOW = None
RECALL_BY_WINDOW = True
VERBOSE = False
TESTING = False

MONITOR = True
TRACK_MIN_DIST = True
CORR_VAL_OPTIM = False


RECALL_FALLBACK_NEAR_RATIO = 0.98
SPEEDUP_NEAR_RATIO = 0.98

WINDOW_SIZE = 7 * 24
N_LAGS = 7 * 24 * 100
WINDOW_STEP = 12
CORR_THRESHOLD = 0.8
CORR_VAL = True
NEG_CORR = True

