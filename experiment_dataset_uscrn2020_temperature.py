"""Dataset-level configuration: uscrn2020_temperature (competitor-paper dataset, phase 0f).

TSUBASA: NOAA USCRN hourly 2020, T_HR_AVG.
Loaded through datasets/competitor_loader.py from datasets/competitor/uscrn2020_temperature.npz (or
tmp_artifacts/uscrn2020_temperature/uscrn2020_temperature.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/uscrn2020_temperature"
DATASET = ["uscrn2020_temperature"]
N_SERIES = [153]
N_OBS = [8784]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="uscrn2020_temperature")
