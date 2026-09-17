"""Dataset-level configuration: motes_humidity (competitor-paper dataset, phase 0f).

BRAID: Intel Lab Motes, humidity.
Loaded through datasets/competitor_loader.py from datasets/competitor/motes_humidity.npz (or
tmp_artifacts/motes_humidity/motes_humidity.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/motes_humidity"
DATASET = ["motes_humidity"]
N_SERIES = [31]
N_OBS = [65516]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="motes_humidity")
