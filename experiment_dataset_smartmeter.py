"""Dataset-level configuration: smartmeter (one of the six real sets of the 2026-09 CorrTrack
sweeps, added to the competitor campaign 2026-09-17 at the user's request).

Half-hourly smart-meter consumption, 510 households (W=48 is one day).
Loaded through datasets/competitor_loader.py from tmp_artifacts/<folder>/smartmeter.npz; OBS_MODE="count".
Historical protocol (implementation log 2026-09-14 methodology note): W=48, step=8, N_LAGS=16.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/smartmeter"
DATASET = ["smartmeter"]
N_SERIES = [510]
N_OBS = [30577]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="smartmeter")
