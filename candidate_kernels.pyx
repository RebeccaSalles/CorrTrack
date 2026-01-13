# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, realloc, free
from libc.stdint cimport int64_t

np.import_array()


cdef inline Py_ssize_t _bisect_left(double[:] values, double x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    cdef double v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_right(double[:] values, double x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    cdef double v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_candidate_pairs(double[:] values,
                         long[:] value_window_idx,
                         double[:] recent_values,
                         long[:] recent_window_idx,
                         long[:] win_sid_idx,
                         long[:] win_time,
                         double tau):
    """Return list of (window_idx, other_idx) pairs within +/- tau range."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t i, j, left, right
    cdef double val, lower, upper
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef bint failed = False
    if buf == NULL:
        raise MemoryError()

    if n_recent == 0 or n_values == 0:
        free(buf)
        return []

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                if count >= cap:
                    cap = cap * 2
                    buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                    if buf == NULL:
                        failed = True
                        break
                buf[2 * count] = <int64_t>ridx
                buf[2 * count + 1] = <int64_t>other_idx
                count += 1
            if failed:
                break

    if failed:
        free(buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
    free(buf)
    return pairs


def fast_corr_and_dist(double[:] x, double[:] y):
    """Compute Pearson correlation and Euclidean distance for two vectors."""
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        return float("nan"), float("inf"), (0, 0.0, 0.0, 0.0, 0.0)
    cdef Py_ssize_t i
    cdef double sx = 0.0
    cdef double sy = 0.0
    cdef double sum_xy = 0.0
    cdef double sum_xx = 0.0
    cdef double sum_yy = 0.0
    cdef double xi, yi
    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sy += yi
        sum_xy += xi * yi
        sum_xx += xi * xi
        sum_yy += yi * yi
    cdef double mean_x = sx / n
    cdef double mean_y = sy / n
    cdef double var_x = sum_xx - (sx * sx) / n
    cdef double var_y = sum_yy - (sy * sy) / n
    if var_x < 0.0:
        var_x = 0.0
    if var_y < 0.0:
        var_y = 0.0
    cdef double denom = sqrt(var_x * var_y)
    cdef double corr
    if denom == 0.0:
        corr = float("nan")
    else:
        corr = (sum_xy - (sx * sy) / n) / denom
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
    cdef double dist_sq = sum_xx + sum_yy - 2.0 * sum_xy
    if dist_sq < 0.0:
        dist_sq = 0.0
    cdef double dist = sqrt(dist_sq)
    return corr, dist, (n, mean_x, mean_y, var_x, var_y)


def validate_corr_batch(double[:, :] x,
                        double[:, :] y,
                        double corr_threshold,
                        bint neg_corr,
                        double std_thresh=1e-3,
                        double kurt_thresh=5.0):
    """Validate batches of x/y vectors; return list of tuples."""
    cdef Py_ssize_t n_items = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    if y.shape[0] != n_items or y.shape[1] != n:
        raise ValueError("x and y must have the same shape")

    cdef Py_ssize_t i, j
    cdef double sx, sy, sum_xy, sum_xx, sum_yy
    cdef double sum_xxx, sum_yyy, sum_xxxx, sum_yyyy
    cdef double xi, yi
    cdef double mean_x, mean_y, var_x, var_y
    cdef double denom, corr, dist_sq, dist
    cdef double mu4_x, mu4_y, varx, vary, kurt_x, kurt_y
    cdef bint is_const, is_spiked, is_corr

    results = []

    for i in range(n_items):
        if n == 0:
            results.append((False, float("nan"), float("inf"), True, False))
            continue
        sx = sy = 0.0
        sum_xy = sum_xx = sum_yy = 0.0
        sum_xxx = sum_yyy = 0.0
        sum_xxxx = sum_yyyy = 0.0
        for j in range(n):
            xi = x[i, j]
            yi = y[i, j]
            sx += xi
            sy += yi
            sum_xy += xi * yi
            sum_xx += xi * xi
            sum_yy += yi * yi
            sum_xxx += xi * xi * xi
            sum_yyy += yi * yi * yi
            sum_xxxx += xi * xi * xi * xi
            sum_yyyy += yi * yi * yi * yi

        mean_x = sx / n
        mean_y = sy / n
        var_x = sum_xx - (sx * sx) / n
        var_y = sum_yy - (sy * sy) / n
        if var_x < 0.0:
            var_x = 0.0
        if var_y < 0.0:
            var_y = 0.0

        is_const = (var_x <= (std_thresh * std_thresh) * n) or (var_y <= (std_thresh * std_thresh) * n)
        if is_const:
            results.append((False, float("nan"), float("inf"), True, False))
            continue

        is_spiked = False
        if n >= 4:
            varx = var_x / n
            if varx > 0.0:
                mu4_x = (
                    sum_xxxx
                    - 4.0 * mean_x * sum_xxx
                    + 6.0 * (mean_x * mean_x) * sum_xx
                    - 4.0 * (mean_x * mean_x * mean_x) * sx
                    + n * (mean_x * mean_x * mean_x * mean_x)
                )
                kurt_x = (mu4_x / n) / (varx * varx) - 3.0
                if kurt_x > kurt_thresh:
                    is_spiked = True
            vary = var_y / n
            if not is_spiked and vary > 0.0:
                mu4_y = (
                    sum_yyyy
                    - 4.0 * mean_y * sum_yyy
                    + 6.0 * (mean_y * mean_y) * sum_yy
                    - 4.0 * (mean_y * mean_y * mean_y) * sy
                    + n * (mean_y * mean_y * mean_y * mean_y)
                )
                kurt_y = (mu4_y / n) / (vary * vary) - 3.0
                if kurt_y > kurt_thresh:
                    is_spiked = True

        if is_spiked:
            results.append((False, float("nan"), float("inf"), False, True))
            continue

        denom = sqrt(var_x * var_y)
        if denom == 0.0:
            corr = float("nan")
        else:
            corr = (sum_xy - (sx * sy) / n) / denom
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0

        dist_sq = sum_xx + sum_yy - 2.0 * sum_xy
        if dist_sq < 0.0:
            dist_sq = 0.0
        dist = sqrt(dist_sq)

        is_corr = False
        if corr == corr:
            if neg_corr:
                is_corr = fabs(corr) >= corr_threshold
            else:
                is_corr = corr >= corr_threshold

        results.append((is_corr, corr, dist, False, False))

    return results
