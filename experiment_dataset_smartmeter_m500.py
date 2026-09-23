"""Dataset-level configuration: smartmeter_m500 (2026-09-21).

London smart meters, half-hourly (full native T), the deterministic-prefix m=500 / T=2768 cut of build_m500_cuts.py used by the m=500
tables (docs/tables_m500_and_filcorr_sweep_2026-09-19.md), so the competitor campaign can run
the same cells under its own protocol (hyperopt / CSZ tuning on the calibration span, N-way
comparison on the holdout, monitoring off). Loaded through datasets/competitor_loader.py from
tmp_artifacts/smartmeter_m500/smartmeter_m500.npz; OBS_MODE="count".
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/smartmeter_m500"
DATASET = ["smartmeter_m500"]
N_SERIES = [500]
N_OBS = [27649]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="smartmeter_m500")
