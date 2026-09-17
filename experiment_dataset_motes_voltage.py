"""Dataset-level configuration: motes_voltage (competitor-paper dataset, phase 0f).

BRAID: Intel Lab Motes, battery voltage.
Loaded through datasets/competitor_loader.py from datasets/competitor/motes_voltage.npz (or
tmp_artifacts/motes_voltage/motes_voltage.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/motes_voltage"
DATASET = ["motes_voltage"]
N_SERIES = [42]
N_OBS = [65516]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="motes_voltage")
