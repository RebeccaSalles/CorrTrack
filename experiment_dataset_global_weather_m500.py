"""Dataset-level configuration: global_weather_m500 (2026-09-21).

global weather stations, daily, the deterministic-prefix m=500 / T=2768 cut of build_m500_cuts.py used by the m=500
tables (docs/tables_m500_and_filcorr_sweep_2026-09-19.md), so the competitor campaign can run
the same cells under its own protocol (hyperopt / CSZ tuning on the calibration span, N-way
comparison on the holdout, monitoring off). Loaded through datasets/competitor_loader.py from
tmp_artifacts/weather_m500/weather_m500.npz; OBS_MODE="count".
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/global_weather_m500"
DATASET = ["global_weather_m500"]
N_SERIES = [500]
N_OBS = [2768]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="weather_m500")
