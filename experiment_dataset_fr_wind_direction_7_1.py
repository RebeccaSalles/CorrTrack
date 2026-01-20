"""Dataset-level configuration for Corracle experiments."""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["wind_direction"]
N_VARS = [7]
N_YEARS = [1]
MODES = ["nD"]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")

# Optional per-phase parallel defaults (set to True/False to override CLI defaults).
PARALLEL_SKETCH = None
PARALLEL_CANDIDATES = None
PARALLEL_VALIDATION = None
