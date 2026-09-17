"""Dataset-level configuration: streamflow (one of the six real sets of the 2026-09 CorrTrack
sweeps, added to the competitor campaign 2026-09-17 at the user's request).

USGS California daily streamflow, 538 gauges.
Loaded through datasets/competitor_loader.py from tmp_artifacts/<folder>/ca_streamflow.npz; OBS_MODE="count".
Historical protocol (implementation log 2026-09-14 methodology note): W=30, step=3, N_LAGS=15.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/streamflow"
DATASET = ["streamflow"]
N_SERIES = [538]
N_OBS = [2192]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="ca_streamflow")
