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

# CorrTrack_compare-specific controls.
CORR_VAL = True
RECALL_BY_WINDOW = True
TRAIN_RATIO = 0.3
TARGET_RECALL = 0.95

# Artifact persistence for run scripts that write out intermediate outputs.
ARTIFACT_MODE = "iterative"

VERBOSE = False
TESTING = False

PARALLEL_SKETCH = False
PARALLEL_CANDIDATES = False
PARALLEL_VALIDATION = None
MAX_WORKERS = 0
