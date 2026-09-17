"""Dataset-level configuration: sp500 (one of the six real sets of the 2026-09 CorrTrack
sweeps, added to the competitor campaign 2026-09-17 at the user's request).

S&P 500 daily closes, 492 tickers (finance_sectors/sp500.npz). Stand-in for the licensed TAQ / CRSP / Yahoo sets of StatStream, CSZ and ParCorr.
Loaded through datasets/competitor_loader.py from tmp_artifacts/<folder>/sp500.npz; OBS_MODE="count".
Historical protocol (implementation log 2026-09-14 methodology note): W=60, step=5, N_LAGS=20.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/sp500"
DATASET = ["sp500"]
N_SERIES = [492]
N_OBS = [1255]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="sp500")
