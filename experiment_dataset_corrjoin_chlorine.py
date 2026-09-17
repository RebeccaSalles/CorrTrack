"""Dataset-level configuration: corrjoin_chlorine (competitor-paper dataset, phase 0f).

CorrJoin: EPANET chlorine.
Loaded through datasets/competitor_loader.py from datasets/competitor/corrjoin_chlorine.npz (or
tmp_artifacts/corrjoin_chlorine/corrjoin_chlorine.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/corrjoin_chlorine"
DATASET = ["corrjoin_chlorine"]
N_SERIES = [4830]
N_OBS = [2040]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="corrjoin_chlorine")
