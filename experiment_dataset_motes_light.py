"""Dataset-level configuration: motes_light (competitor-paper dataset, phase 0f).

BRAID: Intel Lab Motes, light.
Loaded through datasets/competitor_loader.py from datasets/competitor/motes_light.npz (or
tmp_artifacts/motes_light/motes_light.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/motes_light"
DATASET = ["motes_light"]
N_SERIES = [42]
N_OBS = [65516]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="motes_light")
