"""Dataset-level configuration: yellowstone_raw (competitor-paper dataset, phase 0f).

FilCorr section VI: raw counts, same stations; let FilCorr apply its own band.
Loaded through datasets/competitor_loader.py from datasets/competitor/yellowstone_raw.npz (or
tmp_artifacts/yellowstone_raw/yellowstone_raw.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/yellowstone_raw"
DATASET = ["yellowstone_raw"]
N_SERIES = [28]
N_OBS = [120000]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="yellowstone_raw")
