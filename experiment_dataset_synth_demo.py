from functools import partial

from datasets.synth_loader import load_dataset as load_synth_dataset

RESULT_FOLDER = "synthetic/tests"
DATASET = ["synthetic"]
N_SERIES = [8]
N_OBS = [2000]
OBS_MODE = "count"  # interpret N_OBS as absolute number of rows
MODES = ["nD"]

SYNTH_PARAMS = {
    "m": 12,
    "n": 4000,
    "z": 0.3,
    "w": 96,
    "s": 12,
    "threshold": 0.75,
    "corr_sign": "both",
    "base_proc": {"type": "ar1", "phi": 0.6, "sigma": 1.0},
    "max_lag": 48,
    "lag_step": 12,
    "seed": 123,
}

DATA_LOADER = partial(
    load_synth_dataset,
    cache_root="datasets/synth_outputs",
    generator_params=SYNTH_PARAMS,
    refresh=False,
)
