"""Dataset-level configuration: acwi_capweighted (one of the six real sets of the 2026-09 CorrTrack
sweeps, added to the competitor campaign 2026-09-17 at the user's request).

MSCI ACWI constituents, cap-weighted selection, daily.
Loaded through datasets/competitor_loader.py from tmp_artifacts/<folder>/acwi_capweighted.npz; OBS_MODE="count".
Historical protocol (implementation log 2026-09-14 methodology note): W=60, step=5, N_LAGS=20.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/acwi_capweighted"
DATASET = ["acwi_capweighted"]
N_SERIES = [263]
N_OBS = [2452]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="acwi_capweighted")
