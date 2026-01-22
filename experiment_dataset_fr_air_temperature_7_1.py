"""Dataset-level configuration for CorrTrack experiments."""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["air_temperature"]
N_VARS = [7]
N_YEARS = [1]

WINDOW_SIZE = 96
WINDOW_STEP = 12
N_LAGS = 48
CORR_THRESHOLD = 0.7
NEG_CORR = False
TRAIN_RATIO = 1.0
TARGET_RECALL = 0.9

DATA_LOADER = partial(load_dataset, root="datasets/asos-airports")

# Optional per-phase parallel defaults (set to True/False to override CLI defaults).
PARALLEL_SKETCH = None
PARALLEL_CANDIDATES = None
PARALLEL_VALIDATION = None
