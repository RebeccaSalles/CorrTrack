"""Dataset-level configuration: corrjoin_stock (competitor-paper dataset, phase 0f).

CorrJoin: daily stock prices; paper W=1020, ks=15, ke=30.
Loaded through datasets/competitor_loader.py from datasets/competitor/corrjoin_stock.npz (or
tmp_artifacts/corrjoin_stock/corrjoin_stock.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/corrjoin_stock"
DATASET = ["corrjoin_stock"]
N_SERIES = [3878]
N_OBS = [1259]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="corrjoin_stock")
