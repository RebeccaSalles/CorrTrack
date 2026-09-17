"""Dataset-level configuration: wikipedia (one of the six real sets of the 2026-09 CorrTrack
sweeps, added to the competitor campaign 2026-09-17 at the user's request).

Wikipedia daily page views, 88 articles.
Loaded through datasets/competitor_loader.py from tmp_artifacts/<folder>/wikipedia.npz; OBS_MODE="count".
Historical protocol (implementation log 2026-09-14 methodology note): W=30, step=3, N_LAGS=15.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/wikipedia"
DATASET = ["wikipedia"]
N_SERIES = [88]
N_OBS = [1827]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="wikipedia")
