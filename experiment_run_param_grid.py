"""Run-level configuration holding only the hyper-parameter grid."""

PARAM_GRID = {
    "n_vectors": [8, 16, 32, 64],
    "cell_size": [1.0],
    "freq_threshold": [0.0, 0.5, 1.0],
    "warmup_size": [1],
    "preprocess": [True, False],
    "nodes": [0],
    "seed": [2468],
    "seed_toggle": [1357],
    "grid_dimension": [0],
}
