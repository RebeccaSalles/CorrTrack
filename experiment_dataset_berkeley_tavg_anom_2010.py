"""Dataset-level configuration: berkeley_tavg_anom_2010 (competitor-paper dataset, phase 0f).

TSUBASA: Berkeley Earth daily anomalies 2010-2019, 18,520 land cells (N_SERIES capped at the 2k battery target; raise to 18520 for the scalability run).
Loaded through datasets/competitor_loader.py from datasets/competitor/berkeley_tavg_anom_2010.npz (or
tmp_artifacts/berkeley_tavg_anom_2010/berkeley_tavg_anom_2010.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/berkeley_tavg_anom_2010"
DATASET = ["berkeley_tavg_anom_2010"]
N_SERIES = [2000]
N_OBS = [3652]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="berkeley_tavg_anom_2010")
