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
    # 2 x 4 x 5 x 2 = 80 settings: 150 took 40 min locally at m = 492 (8 took 7 min), the pick was
    # n_vectors 64, offset 0.05, occupancy 8; n_vectors 16 and offset 0.15 never won in the pilots
    # (2026-09-24, user) the sketch width is fixed at 64 rather than tuned per cell. The 32-vector
    # option was selected in 7 cells (all smartmeter, lsh backend) and 6 of them then missed the 0.95
    # recall target on the full stream (0.900 to 0.938) while posting speedups up to 16.4x: the proxy
    # recall of the narrower sketch is optimistic, so the selection bought speed with misses. Against
    # 136 cells at 64 vectors, only 5 missed the target. Fixing it also removes a degree of freedom
    # from our own method that the competitors keep.
    "n_vectors": [64],
    "candidate_cosine_threshold_offset": [0.0, 0.05, 0.10, 0.20],
    "candidate_lsh_target_occupancy": [2.0, 3.0, 5.0, 8.0, 12.0],
    "candidate_hamming_filter_max_frac": [None, 0.40],
    # (2026-09-19) incremental validation of repeated candidate pairs, on when W >= 120 (see
    # HYBRID_VALIDATION_AUTO_MIN_WINDOW); exact, so not a searched axis
    "hybrid_validation": ["auto"],
})
