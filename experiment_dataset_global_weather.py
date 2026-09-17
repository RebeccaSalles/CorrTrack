"""Dataset-level configuration: global_weather (one of the six real sets of the 2026-09 CorrTrack
sweeps, added to the competitor campaign 2026-09-17 at the user's request).

Daily weather, 100 stations worldwide.
Loaded through datasets/competitor_loader.py from tmp_artifacts/<folder>/global_weather.npz; OBS_MODE="count".
Historical protocol (implementation log 2026-09-14 methodology note): W=30, step=3, N_LAGS=15.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/global_weather"
DATASET = ["global_weather"]
N_SERIES = [100]
N_OBS = [5113]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="global_weather")
