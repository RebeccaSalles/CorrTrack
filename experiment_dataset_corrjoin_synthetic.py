"""Dataset-level configuration: corrjoin_synthetic (competitor-paper dataset, phase 0f).

CorrJoin: authors' random-walk file.
Loaded through datasets/competitor_loader.py from datasets/competitor/corrjoin_synthetic.npz (or
tmp_artifacts/corrjoin_synthetic/corrjoin_synthetic.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/corrjoin_synthetic"
DATASET = ["corrjoin_synthetic"]
N_SERIES = [5000]
N_OBS = [4080]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="corrjoin_synthetic")
