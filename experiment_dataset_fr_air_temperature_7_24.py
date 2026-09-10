"""Dataset-level configuration for CorrTrack experiments -- isolates the
time-range effect from the series-count effect (2026-07-13 temporal-
continuity diagnostic). Same 7 stations as the original _7_1 config, but 24
years of data, matching the _121_24 variant's time range.
"""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["air_temperature"]
N_VARS = [7]
N_YEARS = [24]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
