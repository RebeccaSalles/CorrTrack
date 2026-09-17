"""Dataset-level configuration: motes_temperature (competitor-paper dataset, phase 0f).

BRAID: Intel Lab Motes, temperature, 31 s epochs; physical lags of 202 and 224 min = ~390 and ~433 epochs.
Loaded through datasets/competitor_loader.py from datasets/competitor/motes_temperature.npz (or
tmp_artifacts/motes_temperature/motes_temperature.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/motes_temperature"
DATASET = ["motes_temperature"]
N_SERIES = [27]
N_OBS = [65516]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="motes_temperature")
