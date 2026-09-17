"""Dataset-level configuration: uscrn2020_precipitation (competitor-paper dataset, phase 0f).

TSUBASA: NOAA USCRN hourly 2020, P_CALC.
Loaded through datasets/competitor_loader.py from datasets/competitor/uscrn2020_precipitation.npz (or
tmp_artifacts/uscrn2020_precipitation/uscrn2020_precipitation.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/uscrn2020_precipitation"
DATASET = ["uscrn2020_precipitation"]
N_SERIES = [145]
N_OBS = [8784]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="uscrn2020_precipitation")
