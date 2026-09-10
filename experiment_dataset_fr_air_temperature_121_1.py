"""Dataset-level configuration for CorrTrack experiments -- isolates the
series-count effect from the time-range effect (2026-07-13 temporal-
continuity diagnostic). Same 121 series as the _121_24 variant, but only
1 year of data, matching the original 7-station config's time range.
"""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["air_temperature"]
N_VARS = [121]
N_YEARS = [1]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
