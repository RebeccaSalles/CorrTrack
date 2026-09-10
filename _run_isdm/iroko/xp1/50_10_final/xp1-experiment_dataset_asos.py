"""Dataset-level configuration for CorrTrack experiments."""

from functools import partial

from datasets.asos_loader import load_dataset


RESULT_FOLDER = "/storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp1/50_10_final/results/"
COUNTRIES = ["fr","br"]
VARIABLES = ["air_temperature","wind_speed","relative_humidity","pressure"]
N_SERIES = [50]
N_YEARS = [10]

DATA_LOADER = partial(load_dataset, root="datasets/asos-airports")
