"""Dataset-level configuration for Corracle experiments."""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["wind_direction"]
N_VARS = [7]
N_YEARS = [1]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
