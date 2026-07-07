from functools import partial

from datasets.synth_loader import load_dataset as load_synth_dataset

RESULT_FOLDER = "synthetic/tests"
DATASET = ["synthetic"]
N_SERIES = [50]
N_OBS = [25856]
OBS_MODE = "count"  # interpret N_OBS as absolute number of rows

SYNTH_PARAMS = {
    "m": 50,
    "n": 25856,
    "z": 0.2,
    "w": 256,
    "template_len": 256,
    "window_step": 16,  # keep aligned with evaluation WINDOW_STEP / CLI overrides
    "max_lag": 100 * 256,  # keep aligned with evaluation N_LAGS / CLI overrides
    "num_templates": 24,
    "threshold": 0.7,
    "corr_sign": "both",
    
    #Stationary
    #"base_proc": {"type": "ar1", "phi": 0.7, "sigma": 1.0},
    #"base_proc": {"type": "wn", "sigma": 1.3},
    #"base_proc": {"type": "lagged_seasonal_ar", "phi_short": 0.2, "phi_long": 0.6, "season_lag": 36, "sigma": 1.3},
    #"base_proc": {"type": "ou", "theta": 0.4, "mu": 0.0, "dt": 1.0, "sigma": 1.0},
    #"base_proc": {"type": "seasonal_arima", "phi": 0.7, "theta": -0.4, "season_period": 96, "season_amplitude": 1.2, "sigma": 0.9},

    #Nonstationary
    "base_proc": {"type": "rw", "sigma": 1.0, "obs_sigma": 2.0},
    #"base_proc": {"type": "rw_seasonal_drift", "drift": 0.04, "season_amplitude": 0.6, "season_period": 288, "sigma": 0.8, "obs_sigma": 1.0},
    #"base_proc": {"type": "trend_poly", "phi": 0.5, "sigma": 0.7}, #*special
    #"base_proc": {"type": "integrated_seasonal", "season_lag": 48, "phi": 0.4, "psi": 0.5, "sigma": 1.0, "obs_sigma": 0.0},
     
    "volatility_equalizer": {"window": 96, "target_std": 1.0, "min_std": 1.0},
    "seed": 123,
    "hash_seed": 0,
}

DATA_LOADER = partial(
    load_synth_dataset,
    cache_root="datasets/synth_outputs",
    generator_params=SYNTH_PARAMS,
    refresh=False,
)
