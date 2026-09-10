"""Dataset-level configuration for CorrTrack experiments."""

from functools import partial

from datasets.asos_loader import load_dataset


RESULT_FOLDER = "/storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp-denis/noval_ws12_5_6_final/results/"
COUNTRIES = ["fr","br"]
VARIABLES = ["air_temperature","wind_speed","relative_humidity","pressure"]
N_SERIES = [5]
N_YEARS = [6]

DATA_LOADER = partial(load_dataset, root="datasets/asos-airports")
