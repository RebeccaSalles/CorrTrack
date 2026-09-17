"""Dataset-level configuration: corrjoin_random (competitor-paper dataset, phase 0f).

CorrJoin: authors' i.i.d. uniform file (uncooperative bonus).
Loaded through datasets/competitor_loader.py from datasets/competitor/corrjoin_random.npz (or
tmp_artifacts/corrjoin_random/corrjoin_random.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/corrjoin_random"
DATASET = ["corrjoin_random"]
N_SERIES = [5000]
N_OBS = [4080]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="corrjoin_random")
