"""Dataset-level configuration for Corracle experiments.

(2026-07-28) VARIABLES switched from "wind_direction" to "wind_speed" per
explicit instruction -- module/file name kept as-is.
"""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["wind_speed"]
N_VARS = [7]
N_YEARS = [1]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
