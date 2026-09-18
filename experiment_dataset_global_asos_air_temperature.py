"""Dataset-level configuration: global ASOS air_temperature, hourly, 181+ countries (fetched 2026-09-17 on Abaca
by the companion session: tmp_artifacts/global_asos/global_air_temperature.npz, 3,6xx stations x 137,688 hours,
2011-2026, 72% missing overall). Selection at load time: the last 2 years (17,520 h) and the stations with
>= 90% coverage over that span, best-covered first (~670 stations); remaining gaps forward-filled.
Hourly weather regime of the campaign: W = 168 (a week), step 12, L_max = 5.
"""

from functools import partial

from datasets.competitor_loader import load_dataset

RESULT_FOLDER = "competitor_exp/global_asos_air_temperature"
DATASET = ["global_asos_air_temperature"]
N_SERIES = [600]
N_OBS = [17520]
OBS_MODE = "count"

DATA_LOADER = partial(load_dataset, name="global_air_temperature", last_obs=17520, min_coverage=0.9)
