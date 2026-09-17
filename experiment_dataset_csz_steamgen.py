"""Dataset-level configuration: csz_steamgen (competitor-paper dataset, phase 0f).

CSZ: DaISy steam generator, 8 channels (raw channels; see fetch_csz_daisy.py --chunk).
Loaded through datasets/competitor_loader.py from datasets/competitor/csz_steamgen.npz (or
tmp_artifacts/csz_steamgen/csz_steamgen.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/csz_steamgen"
DATASET = ["csz_steamgen"]
N_SERIES = [8]
N_OBS = [9600]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="csz_steamgen")
