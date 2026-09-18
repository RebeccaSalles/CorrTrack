"""Dataset-level configuration: statstream_rw_m5000_T20000 (competitor-paper dataset, phase 0f).

StatStream section 5 random walks; generate with datasets/fetch/gen_statstream_randomwalk.py --m 5000 --T 20000.
Loaded through datasets/competitor_loader.py from datasets/competitor/statstream_rw_m5000_T20000.npz (or
tmp_artifacts/statstream_rw_m5000_T20000/statstream_rw_m5000_T20000.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/statstream_rw_m5000_T20000"
DATASET = ["statstream_rw_m5000_T20000"]
N_SERIES = [5000]
N_OBS = [20000]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="statstream_rw_m5000_T20000")
