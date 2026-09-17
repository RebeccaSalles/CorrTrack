"""Dataset-level configuration: ASOS Brazil airports, flights, 109 stations, hourly, last 1 year(s).
Competitor campaign entry (2026-09-17, user's request to include the ASOS sets); same loader and
folder convention as experiment_dataset_fr_air_temperature_121_1.py.
"""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["br"]
VARIABLES = ["flights"]
N_VARS = [109]
N_YEARS = [1]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
