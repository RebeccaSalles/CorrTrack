"""Dataset-level configuration: uscrn2020_humidity (competitor-paper dataset, phase 0f).

TSUBASA: NOAA USCRN hourly 2020, RH_HR_AVG.
Loaded through datasets/competitor_loader.py from datasets/competitor/uscrn2020_humidity.npz (or
tmp_artifacts/uscrn2020_humidity/uscrn2020_humidity.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/uscrn2020_humidity"
DATASET = ["uscrn2020_humidity"]
N_SERIES = [135]
N_OBS = [8784]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="uscrn2020_humidity")
