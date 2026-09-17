"""Dataset-level configuration: corrjoin_gas (competitor-paper dataset, phase 0f).

CorrJoin: gas sensor array.
Loaded through datasets/competitor_loader.py from datasets/competitor/corrjoin_gas.npz (or
tmp_artifacts/corrjoin_gas/corrjoin_gas.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/corrjoin_gas"
DATASET = ["corrjoin_gas"]
N_SERIES = [5120]
N_OBS = [3600]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="corrjoin_gas")
