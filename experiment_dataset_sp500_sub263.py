"""Dataset-level configuration: sp500_sub263 (competitor-paper dataset, phase 0f).

stand-in for the licensed NYSE TAQ (StatStream), CRSP (CSZ) and Yahoo (ParCorr) stock sets; built 2026-09-15, tmp_artifacts.
Loaded through datasets/competitor_loader.py from datasets/competitor/sp500_sub263.npz (or
tmp_artifacts/sp500_sub263/sp500_sub263.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/sp500_sub263"
DATASET = ["sp500_sub263"]
N_SERIES = [263]
N_OBS = [1255]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="sp500_sub263")
