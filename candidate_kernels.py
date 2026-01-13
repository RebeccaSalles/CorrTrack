"""Fallback pure-Python kernels for candidate search.

The Cython version (candidate_kernels.pyx) can be compiled to speed up
range-search loops. This module keeps the same API so imports work without
building extensions.
"""

from bisect import bisect_left, bisect_right
import numpy as np


def _is_near_constant_stats(var_sum, n, std_thresh=1e-3):
    if n <= 0:
        return True
    return var_sum <= (std_thresh ** 2) * n


def _is_structurally_spiked_stats(x, mean, var_sum, n, kurt_thresh=5.0, mu4_sum=None):
    if n < 4 or var_sum <= 0.0:
        return False

    if mu4_sum is None:
        if x is None:
            return False
        centered = np.asarray(x, dtype=np.float64) - mean
        mu4 = float(np.sum(centered ** 4))
    else:
        mu4 = float(mu4_sum)

    var = var_sum / n
    if var <= 0.0:
        return False
    kurt = (mu4 / n) / (var * var) - 3.0
    return kurt > kurt_thresh


def find_candidate_pairs(values, value_window_idx, recent_values, recent_window_idx, win_sid_idx, win_time, tau):
    pairs = []
    values = np.asarray(values, dtype=np.float64).tolist()
    value_window_idx = np.asarray(value_window_idx, dtype=np.int64).tolist()
    recent_values = np.asarray(recent_values, dtype=np.float64).tolist()
    recent_window_idx = np.asarray(recent_window_idx, dtype=np.int64).tolist()
    win_sid_idx = np.asarray(win_sid_idx, dtype=np.int64).tolist()
    win_time = np.asarray(win_time, dtype=np.int64).tolist()
    if not recent_values or not values:
        return pairs
    for val, ridx in zip(recent_values, recent_window_idx):
        lower = val - tau
        upper = val + tau
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        sid_r = win_sid_idx[ridx]
        time_r = win_time[ridx]
        for j in range(left, right):
            other_idx = value_window_idx[j]
            if other_idx == ridx:
                continue
            if win_sid_idx[other_idx] == sid_r and win_time[other_idx] == time_r:
                continue
            pairs.append((ridx, other_idx))
    return pairs


def fast_corr_and_dist(x, y):

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.nan, np.inf, (0, 0.0, 0.0, 0.0, 0.0)

    sx = float(x.sum(dtype=np.float64))
    sy = float(y.sum(dtype=np.float64))
    sum_xy = float(np.dot(x, y))
    sum_xx = float(np.dot(x, x))
    sum_yy = float(np.dot(y, y))

    mean_x = sx / n
    mean_y = sy / n
    var_x = sum_xx - (sx * sx) / n
    var_y = sum_yy - (sy * sy) / n
    var_x = max(var_x, 0.0)
    var_y = max(var_y, 0.0)

    denom = np.sqrt(var_x * var_y)
    if denom == 0.0:
        corr = np.nan
    else:
        cov = sum_xy - (sx * sy) / n
        corr = cov / denom if denom else np.nan
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0

    dist_sq = sum_xx + sum_yy - 2.0 * sum_xy
    dist = np.sqrt(max(dist_sq, 0.0))
    return corr, dist, (n, mean_x, mean_y, var_x, var_y)


def validate_corr_batch(x_batch, y_batch, corr_threshold, neg_corr, std_thresh=1e-3, kurt_thresh=5.0):
    x_arr = np.asarray(x_batch, dtype=np.float64)
    y_arr = np.asarray(y_batch, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    results = []
    for i in range(x_arr.shape[0]):
        x = x_arr[i]
        y = y_arr[i]
        corr, dist, stats = fast_corr_and_dist(x, y)
        n, mean_x, mean_y, var_x, var_y = stats
        is_const = _is_near_constant_stats(var_x, n, std_thresh) or _is_near_constant_stats(var_y, n, std_thresh)
        if is_const:
            results.append((False, float("nan"), float("inf"), True, False))
            continue
        is_spiked = (
            _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh)
            or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh)
        )
        if is_spiked:
            results.append((False, float("nan"), float("inf"), False, True))
            continue
        if neg_corr:
            is_corr = (not np.isnan(corr)) and (abs(corr) >= corr_threshold)
        else:
            is_corr = (not np.isnan(corr)) and (corr >= corr_threshold)
        results.append((is_corr, corr, dist, False, False))
    return results
