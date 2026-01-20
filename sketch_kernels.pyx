# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

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

    for s in range(n_series):
        for b in range(n_basic):
            for v in range(n_vectors):
                out_mv[s, b, v] = 0.0
                for w in range(basic_window):
                    out_mv[s, b, v] += window_blocks[s, b, w] * weights[b, v, w]
    return out


def build_sketch_matrix(double[:, :, ::1] window_blocks,
                        double[:, :, ::1] weights,
                        long[:] perm,
                        double[:] signs,
                        int norm_mode):
    """Compute series_dots, raw (orth-applied), and normalized sketch matrix."""
    cdef Py_ssize_t n_series = window_blocks.shape[0]
    cdef Py_ssize_t n_basic = window_blocks.shape[1]
    cdef Py_ssize_t basic_window = window_blocks.shape[2]
    cdef Py_ssize_t w_basic = weights.shape[0]
    cdef Py_ssize_t n_vectors = weights.shape[1]
    cdef Py_ssize_t w_window = weights.shape[2]

    if w_basic != n_basic or w_window != basic_window:
        raise ValueError("weights shape mismatch")

    cdef np.ndarray[np.float64_t, ndim=3] dots = np.empty(
        (n_series, n_basic, n_vectors),
        dtype=np.float64,
    )
    cdef np.ndarray[np.float64_t, ndim=2] raw = np.empty(
        (n_series, n_vectors),
        dtype=np.float64,
    )
    cdef np.ndarray[np.float64_t, ndim=2] norm = np.empty(
        (n_series, n_vectors),
        dtype=np.float64,
    )
    cdef double[:, :, ::1] dots_mv = dots
    cdef double[:, ::1] raw_mv = raw
    cdef double[:, ::1] norm_mv = norm
    cdef bint use_orth = perm.shape[0] == n_vectors and signs.shape[0] == n_vectors

    cdef Py_ssize_t s, b, v, w
    cdef double acc, mean, var_acc, denom, centered
    cdef double *tmp = NULL

    if n_vectors > 0:
        tmp = <double *>malloc(n_vectors * sizeof(double))
        if tmp == NULL:
            raise MemoryError()

    with nogil:
        for s in range(n_series):
            for b in range(n_basic):
                for v in range(n_vectors):
                    acc = 0.0
                    for w in range(basic_window):
                        acc += window_blocks[s, b, w] * weights[b, v, w]
                    dots_mv[s, b, v] = acc
            for v in range(n_vectors):
                acc = 0.0
                for b in range(n_basic):
                    acc += dots_mv[s, b, v]
                raw_mv[s, v] = acc

            if use_orth:
                for v in range(n_vectors):
                    tmp[v] = raw_mv[s, v]
                for v in range(n_vectors):
                    raw_mv[s, v] = tmp[perm[v]] * signs[v]

            mean = 0.0
            for v in range(n_vectors):
                mean += raw_mv[s, v]
            if n_vectors > 0:
                mean /= n_vectors

            var_acc = 0.0
            for v in range(n_vectors):
                centered = raw_mv[s, v] - mean
                var_acc += centered * centered

            if norm_mode == 1:
                denom = sqrt(var_acc)
            else:
                denom = sqrt(var_acc / n_vectors) if n_vectors > 0 else 0.0

            if denom <= 0.0 or denom != denom:
                for v in range(n_vectors):
                    norm_mv[s, v] = 0.0
            else:
                for v in range(n_vectors):
                    centered = raw_mv[s, v] - mean
                    norm_mv[s, v] = centered / denom

    if tmp != NULL:
        free(tmp)

    return dots, raw, norm


def apply_orth_and_normalize(double[:, ::1] raw_matrix,
                             long[:] perm,
                             double[:] signs,
                             int norm_mode):
    """Apply orthogonal transform (perm/signs) and normalize a raw sketch matrix."""
    cdef Py_ssize_t n_series = raw_matrix.shape[0]
    cdef Py_ssize_t n_vectors = raw_matrix.shape[1]
    cdef bint use_orth = perm.shape[0] == n_vectors and signs.shape[0] == n_vectors

    cdef np.ndarray[np.float64_t, ndim=2] raw = np.empty(
        (n_series, n_vectors),
        dtype=np.float64,
    )
    cdef np.ndarray[np.float64_t, ndim=2] norm = np.empty(
        (n_series, n_vectors),
        dtype=np.float64,
    )
    cdef double[:, ::1] raw_mv = raw
    cdef double[:, ::1] norm_mv = norm
    cdef Py_ssize_t s, v
    cdef double mean, var_acc, denom, centered
    cdef double *tmp = NULL

    if n_vectors > 0:
        tmp = <double *>malloc(n_vectors * sizeof(double))
        if tmp == NULL:
            raise MemoryError()

    with nogil:
        for s in range(n_series):
            for v in range(n_vectors):
                raw_mv[s, v] = raw_matrix[s, v]

            if use_orth:
                for v in range(n_vectors):
                    tmp[v] = raw_mv[s, v]
                for v in range(n_vectors):
                    raw_mv[s, v] = tmp[perm[v]] * signs[v]

            mean = 0.0
            for v in range(n_vectors):
                mean += raw_mv[s, v]
            if n_vectors > 0:
                mean /= n_vectors

            var_acc = 0.0
            for v in range(n_vectors):
                centered = raw_mv[s, v] - mean
                var_acc += centered * centered

            if norm_mode == 1:
                denom = sqrt(var_acc)
            else:
                denom = sqrt(var_acc / n_vectors) if n_vectors > 0 else 0.0

            if denom <= 0.0 or denom != denom:
                for v in range(n_vectors):
                    norm_mv[s, v] = 0.0
            else:
                for v in range(n_vectors):
                    centered = raw_mv[s, v] - mean
                    norm_mv[s, v] = centered / denom

    if tmp != NULL:
        free(tmp)

    return raw, norm


def compute_constant_flags(double[:] sum1,
                           double[:] sum2,
                           double[:] sum3,
                           double[:] sum4,
                           long n,
                           double std_thresh=1e-3,
                           double kurt_thresh=5.0):
    """Compute near-constant and spiked flags from raw moment sums."""
    cdef Py_ssize_t n_series = sum1.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] const_flags = np.empty(
        n_series,
        dtype=np.uint8,
    )
    cdef np.ndarray[np.uint8_t, ndim=1] spiked_flags = np.empty(
        n_series,
        dtype=np.uint8,
    )
    cdef unsigned char[:] const_mv = const_flags
    cdef unsigned char[:] spiked_mv = spiked_flags
    cdef Py_ssize_t i
    cdef double mean, var_sum, mu4_sum, var, kurt
    cdef double n_d = <double>n
    cdef double var_thresh = std_thresh * std_thresh * n_d

    with nogil:
        for i in range(n_series):
            if n <= 0:
                const_mv[i] = 1
                spiked_mv[i] = 0
                continue
            mean = sum1[i] / n_d
            var_sum = sum2[i] - (sum1[i] * sum1[i]) / n_d
            if var_sum < 0.0:
                var_sum = 0.0
            if var_sum <= var_thresh:
                const_mv[i] = 1
            else:
                const_mv[i] = 0

            if var_sum <= 0.0:
                spiked_mv[i] = 0
                continue
            mu4_sum = (
                sum4[i]
                - 4.0 * mean * sum3[i]
                + 6.0 * (mean * mean) * sum2[i]
                - 4.0 * (mean * mean * mean) * sum1[i]
                + n_d * (mean * mean * mean * mean)
            )
            var = var_sum / n_d
            if var <= 0.0:
                spiked_mv[i] = 0
                continue
            kurt = (mu4_sum / n_d) / (var * var) - 3.0
            spiked_mv[i] = 1 if kurt > kurt_thresh else 0

    return const_flags, spiked_flags
