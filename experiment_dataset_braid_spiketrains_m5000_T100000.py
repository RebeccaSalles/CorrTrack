"""Dataset-level configuration: braid_spiketrains_m5000_T100000 (competitor-paper dataset, phase 0f).

BRAID section 6.1 SpikeTrains (approximation); gen_braid_synthetic.py --family spiketrains --m 5000.
Loaded through datasets/competitor_loader.py from datasets/competitor/braid_spiketrains_m5000_T100000.npz (or
tmp_artifacts/braid_spiketrains_m5000_T100000/braid_spiketrains_m5000_T100000.npz); OBS_MODE="count" so N_OBS is a row count, not years.
Registry: datasets/competitor_sources.md.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/braid_spiketrains_m5000_T100000"
DATASET = ["braid_spiketrains_m5000_T100000"]
N_SERIES = [5000]
N_OBS = [100000]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="braid_spiketrains_m5000_T100000")
