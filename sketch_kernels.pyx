# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

np.import_array()


def compute_series_dots(double[:, :, ::1] window_blocks,
                        double[:, :, ::1] weights):
    """Compute per-series, per-basic-window dot products."""
    cdef Py_ssize_t n_series = window_blocks.shape[0]
    cdef Py_ssize_t n_basic = window_blocks.shape[1]
    cdef Py_ssize_t basic_window = window_blocks.shape[2]
    cdef Py_ssize_t w_basic = weights.shape[0]
    cdef Py_ssize_t n_vectors = weights.shape[1]
    cdef Py_ssize_t w_window = weights.shape[2]

    if w_basic != n_basic or w_window != basic_window:
        raise ValueError("weights shape mismatch")

    cdef np.ndarray[np.float64_t, ndim=3] out = np.empty((n_series, n_basic, n_vectors), dtype=np.float64)
    cdef double[:, :, ::1] out_mv = out
    cdef Py_ssize_t s, b, v, w
    cdef double acc

    with nogil:
        for s in range(n_series):
            for b in range(n_basic):
                for v in range(n_vectors):
                    acc = 0.0
                    for w in range(basic_window):
                        acc += window_blocks[s, b, w] * weights[b, v, w]
                    out_mv[s, b, v] = acc
    return out
