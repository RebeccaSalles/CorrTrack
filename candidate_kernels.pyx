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


def find_candidate_pairs_full(double[:] values,
                              long[:] value_window_idx,
                              double[:] recent_values,
                              long[:] recent_window_idx,
                              long[:] win_sid_idx,
                              long[:] win_time,
                              double[:, :] entry_vectors,
                              double[:, :] recent_vectors,
                              double tau):
    """Return list of (window_idx, other_idx) pairs within +/- tau and L2 check."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if entry_vectors.shape[0] != n_values:
        raise ValueError("entry_vectors must align with values")
    if recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent_vectors must align with recent_values")
    cdef Py_ssize_t i, j, d, left, right
    cdef double val, lower, upper
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef double tau_sq = tau * tau
    cdef double diff, acc
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef bint failed = False
    if buf == NULL:
        raise MemoryError()

    if n_recent == 0 or n_values == 0 or n_dim == 0 or tau < 0.0:
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
                acc = 0.0
                for d in range(n_dim):
                    diff = recent_vectors[i, d] - entry_vectors[j, d]
                    acc += diff * diff
                    if acc > tau_sq:
                        break
                if acc > tau_sq:
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


def enumerate_candidate_rows(double[:, ::1] data,
                             long[:] window_index,
                             long[:] ref_indices,
                             int window_size,
                             int window_step,
                             double std_thresh=1e-3,
                             long shard_start=-1,
                             long shard_end=-1):
    cdef Py_ssize_t n_series = data.shape[0]
    cdef Py_ssize_t n_cols = data.shape[1]
    cdef Py_ssize_t window_count = n_cols - window_size + 1
    if window_size <= 0 or window_count <= 0:
        return None
    if ref_indices.shape[0] == 0:
        return None

    cdef int step = window_step if window_step > 0 else 1
    cdef Py_ssize_t last_idx = window_count - 1
    if last_idx % step != 0:
        return None

    cdef double threshold = (std_thresh * std_thresh) * window_size
    cdef Py_ssize_t step_count = (window_count + step - 1) // step
    cdef Py_ssize_t cap = n_series * step_count
    if cap <= 0:
        return None

    cdef int64_t *valid_k = <int64_t *>malloc(cap * sizeof(int64_t))
    cdef int64_t *valid_j = <int64_t *>malloc(cap * sizeof(int64_t))
    cdef unsigned char *seeds = <unsigned char *>malloc(n_series * sizeof(unsigned char))
    if valid_k == NULL or valid_j == NULL or seeds == NULL:
        if valid_k != NULL:
            free(valid_k)
        if valid_j != NULL:
            free(valid_j)
        if seeds != NULL:
            free(seeds)
        raise MemoryError()

    cdef Py_ssize_t s, j
    cdef double sum_val, sum_sq, val, outgoing, incoming, var_sum
    cdef Py_ssize_t count = 0
    for s in range(n_series):
        seeds[s] = 0
        sum_val = 0.0
        sum_sq = 0.0
        for j in range(window_size):
            val = data[s, j]
            sum_val += val
            sum_sq += val * val
        for j in range(window_count):
            if j > 0:
                outgoing = data[s, j - 1]
                incoming = data[s, j + window_size - 1]
                sum_val += incoming - outgoing
                sum_sq += incoming * incoming - outgoing * outgoing
            var_sum = sum_sq - (sum_val * sum_val) / window_size
            if var_sum < 0.0:
                var_sum = 0.0
            if (j % step) == 0 and var_sum > threshold:
                if count < cap:
                    valid_k[count] = <int64_t>s
                    valid_j[count] = <int64_t>j
                    count += 1
            if j == last_idx and var_sum > threshold:
                seeds[s] = 1

    if count == 0:
        free(valid_k)
        free(valid_j)
        free(seeds)
        return None

    cdef bint has_seed = False
    cdef long s_idx
    for j in range(ref_indices.shape[0]):
        s_idx = ref_indices[j]
        if s_idx < 0 or s_idx >= n_series:
            continue
        if seeds[s_idx] != 0:
            has_seed = True
            break
    if not has_seed:
        free(valid_k)
        free(valid_j)
        free(seeds)
        return None

    cdef Py_ssize_t out_cap = 1024
    if out_cap < count:
        out_cap = count
    cdef int64_t *out_buf = <int64_t *>malloc(out_cap * 5 * sizeof(int64_t))
    if out_buf == NULL:
        free(valid_k)
        free(valid_j)
        free(seeds)
        raise MemoryError()

    cdef Py_ssize_t out_count = 0
    cdef long curr_start = window_index[last_idx]
    cdef int64_t k_idx, j_idx
    cdef bint apply_shard = shard_start >= 0 and shard_end >= 0
    for j in range(ref_indices.shape[0]):
        s_idx = ref_indices[j]
        if s_idx < 0 or s_idx >= n_series:
            continue
        if seeds[s_idx] == 0:
            continue
        for j_idx in range(count):
            k_idx = valid_k[j_idx]
            if valid_j[j_idx] == last_idx and k_idx <= s_idx:
                continue
            if apply_shard:
                if k_idx < shard_start or k_idx >= shard_end:
                    continue
            if out_count >= out_cap:
                out_cap = out_cap * 2
                out_buf = <int64_t *>realloc(out_buf, out_cap * 5 * sizeof(int64_t))
                if out_buf == NULL:
                    free(valid_k)
                    free(valid_j)
                    free(seeds)
                    raise MemoryError()
            out_buf[5 * out_count] = <int64_t>s_idx
            out_buf[5 * out_count + 1] = k_idx
            out_buf[5 * out_count + 2] = <int64_t>curr_start
            out_buf[5 * out_count + 3] = <int64_t>window_index[valid_j[j_idx]]
            out_buf[5 * out_count + 4] = <int64_t>window_size
            out_count += 1

    free(valid_k)
    free(valid_j)
    free(seeds)

    if out_count == 0:
        free(out_buf)
        return None

    cdef np.ndarray[np.int64_t, ndim=2] out = np.empty((out_count, 5), dtype=np.int64)
    for j in range(out_count):
        out[j, 0] = out_buf[5 * j]
        out[j, 1] = out_buf[5 * j + 1]
        out[j, 2] = out_buf[5 * j + 2]
        out[j, 3] = out_buf[5 * j + 3]
        out[j, 4] = out_buf[5 * j + 4]

    free(out_buf)
    return out



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
