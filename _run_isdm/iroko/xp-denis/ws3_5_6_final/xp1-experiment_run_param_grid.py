"""Run-level configuration holding only the hyper-parameter grid."""

PARAM_GRID = {
    "n_vectors": [8, 16, 32, 64],
    "cell_size": [0.25, 0.5, 0.75, 1, 1.25],  # strech rate
    "freq_threshold": [0],  # not needed in sequential

    "preprocess": [False],
    "nodes": [0],  # automatic cores-1
    "seed": [2468],
    "seed_toggle": [1357],
    "grid_dimension": [1],
    "sketch_norm": ["mean_l2"],
    "candidate_backend": ["bptree"]
}
