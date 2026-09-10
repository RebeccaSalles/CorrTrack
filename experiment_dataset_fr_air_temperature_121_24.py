"""Dataset-level configuration for CorrTrack experiments -- large-scale variant.

(2026-07-13) New file, not a modification of experiment_dataset_fr_air_temperature_7_1.py
-- created for the temporal-continuity diagnostic's large-scale runs
(diag_temporal_continuity_oracle.py) without changing the existing 7-station/
1-year config any other script or benchmark relies on. 121 is the real cap
for this dataset (fr-air_temperature.csv has 121 station columns, confirmed
directly against the data); 24 is the real cap for years (~24.27 years of
hourly data available -- prepare_training_data clamps to what's on disk if
N_YEARS*365*24 exceeds it).
"""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["air_temperature"]
N_VARS = [121]
N_YEARS = [24]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
