"""Dataset-level configuration for CorrTrack experiments."""

from functools import partial

from datasets.asos_loader import load_dataset


RESULT_FOLDER = "/storage/simple/projects/iroko-lirmm/CorrTrack/_run_isdm/iroko/xp-benoit_Tmp/b10_10_1/results/"
COUNTRIES = ["fr","br"]
VARIABLES = ["air_temperature"]
N_SERIES = [10]
N_YEARS = [10]

DATA_LOADER = partial(load_dataset, root="datasets/asos-airports")
