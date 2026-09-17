"""Dataset-level configuration: yellowstone_bp3_7 (competitor-paper dataset, phase 0f).

FilCorr section VI: 28 WY stations, 100 Hz, 3-7 Hz band-passed; paper used 20 s windows (W=2000) and 10 s lag (N_LAGS=1000).
Loaded through datasets/competitor_loader.py from datasets/competitor/yellowstone_bp3_7.npz (or
tmp_artifacts/yellowstone_bp3_7/yellowstone_bp3_7.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/yellowstone_bp3_7"
DATASET = ["yellowstone_bp3_7"]
N_SERIES = [28]
N_OBS = [120000]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="yellowstone_bp3_7")
