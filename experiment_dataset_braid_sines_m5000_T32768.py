"""Dataset-level configuration: braid_sines_m5000_T32768 (competitor-paper dataset, phase 0f).

BRAID section 6.1 Sines (approximation); gen_braid_synthetic.py --family sines --m 5000.
Loaded through datasets/competitor_loader.py from datasets/competitor/braid_sines_m5000_T32768.npz (or
tmp_artifacts/braid_sines_m5000_T32768/braid_sines_m5000_T32768.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/braid_sines_m5000_T32768"
DATASET = ["braid_sines_m5000_T32768"]
N_SERIES = [5000]
N_OBS = [32768]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="braid_sines_m5000_T32768")
