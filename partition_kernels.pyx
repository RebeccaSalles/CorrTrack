# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

np.import_array()


def build_partitions(double[:, ::1] matrix,
                     int grid_dimension,
                     unsigned char[:] const_flags):
    """Build per-grid chunk/norm/const arrays from the sketch matrix."""
    cdef Py_ssize_t n_rows = matrix.shape[0]
    cdef Py_ssize_t n_dim = matrix.shape[1]
    cdef Py_ssize_t n_grids
    if grid_dimension <= 0:
        raise ValueError("grid_dimension must be positive")
    n_grids = n_dim // grid_dimension
    if n_grids <= 0 or n_rows == 0:
        return (
            np.empty((0, 0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.uint8),
        )

    cdef np.ndarray[np.float64_t, ndim=3] chunks = np.empty(
        (n_grids, n_rows, grid_dimension),
        dtype=np.float64,
    )
    cdef np.ndarray[np.float64_t, ndim=2] norms = np.empty(
        (n_grids, n_rows),
        dtype=np.float64,
    )
    cdef np.ndarray[np.uint8_t, ndim=2] is_const = np.empty(
        (n_grids, n_rows),
        dtype=np.uint8,
    )

    cdef double[:, :, ::1] chunks_mv = chunks
    cdef double[:, ::1] norms_mv = norms
    cdef unsigned char[:, ::1] const_mv = is_const
    cdef Py_ssize_t g, r, d, base
    cdef double acc, val, norm

    with nogil:
        for g in range(n_grids):
            base = g * grid_dimension
            for r in range(n_rows):
                acc = 0.0
                for d in range(grid_dimension):
                    val = matrix[r, base + d]
                    chunks_mv[g, r, d] = val
                    acc += val * val
                norm = sqrt(acc)
                norms_mv[g, r] = norm
                if const_flags[r] != 0 or norm == 0.0:
                    const_mv[g, r] = 1
                else:
                    const_mv[g, r] = 0

    return chunks, norms, is_const
