"""Dataset-level configuration for CorrTrack experiments -- wind_speed
counterpart to experiment_dataset_fr_air_temperature_121_1.py, for the
same series-count-effect comparison (2026-07-24). Same 121 series, 1 year
of data, matching the air_temperature variant's time range so the two are
directly comparable.

(2026-07-28) VARIABLES switched from "wind_direction" to "wind_speed" per
explicit instruction -- module/file name kept as-is (still says
wind_direction) since it's referenced by name throughout
docs/implementation_log.md's history and several benchmark scripts; only
the loaded variable changed.
"""

from functools import partial

from datasets.asos_loader import load_dataset

RESULT_FOLDER = "asos_exp/tests"
COUNTRIES = ["fr"]
VARIABLES = ["wind_speed"]
N_VARS = [121]
N_YEARS = [1]

DATA_LOADER = partial(load_dataset, root="correlation/asos-airports")
