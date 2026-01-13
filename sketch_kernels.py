import numpy as np


def compute_series_dots(window_blocks, weights):
    return np.einsum("sbw,bvw->sbv", window_blocks, weights, optimize=True)
