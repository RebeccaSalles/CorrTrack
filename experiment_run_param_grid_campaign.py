"""CorrTrack hyper-parameter grid for the competitor campaign (2026-09-19).

The default grid (experiment_run_param_grid.py) fixes the dot-gate offset at 0.20 and the LSH target
occupancy at 3.0 and searches only n_vectors x hamming fraction (8 settings). The m = 500 calibrations
of the other thread (hamming_exact_compare_m500.py) swept occupancy {2, 3, 5, 8, 12} x offset {0 ..
0.25} and found the fastest recall >= 0.95 settings at occupancy 2 to 5 and offset 0.05 to 0.10 for T
>= 0.9, i.e. a much tighter gate than 0.20. The campaign therefore searches those two axes too, with
CorrTrack's own proxy hyperopt on the training span (the settings that share a sketch reuse it, so the
extra axes cost far less than proportionally). Everything else is the default grid's.
"""
from experiment_run_param_grid import PARAM_GRID as _BASE

PARAM_GRID = dict(_BASE)
PARAM_GRID.update({
    "n_vectors": [16, 32, 64],
    "candidate_cosine_threshold_offset": [0.0, 0.05, 0.10, 0.15, 0.20],
    "candidate_lsh_target_occupancy": [2.0, 3.0, 5.0, 8.0, 12.0],
    "candidate_hamming_filter_max_frac": [None, 0.40],
})
