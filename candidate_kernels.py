"""Fallback pure-Python kernels for candidate search.

The Cython version (candidate_kernels.pyx) can be compiled to speed up
range-search loops. This module keeps the same API so imports work without
building extensions.
"""

from bisect import bisect_left, bisect_right
import numpy as np


class BalancedIndex:
    """Fallback balanced-index API compatible with the Cython implementation."""

    _LOW_SENTINEL = -2**63
    _HIGH_SENTINEL = 2**63 - 1

    def __init__(self, n_vectors=0, initial_capacity=1024, seed=0):
        self.n_vectors = max(0, int(n_vectors or 0))
        self._entries = []  # sorted (value, entry_id, window_idx)
        self._meta = {}  # entry_id -> (value, window_idx, vector|None)
        self._next_id = 0

    def insert(self, value, window_idx, vector=None):
        value = float(value)
        window_idx = int(window_idx)
        entry_id = self._next_id
        self._next_id += 1
        vec = None
        if self.n_vectors > 0 and vector is not None:
            vec = np.asarray(vector, dtype=np.float64).ravel()
            if vec.shape[0] != self.n_vectors:
                raise ValueError("vector size does not match BalancedIndex dimension")
        rec = (value, int(entry_id), window_idx)
        pos = bisect_left(self._entries, rec)
        self._entries.insert(pos, rec)
        self._meta[int(entry_id)] = (value, window_idx, vec)
        return int(entry_id)

    def remove(self, entry_id):
        entry_id = int(entry_id)
        meta = self._meta.pop(entry_id, None)
        if meta is None:
            return False
        value = float(meta[0])
        pos = bisect_left(self._entries, (value, entry_id, self._LOW_SENTINEL))
        while pos < len(self._entries) and self._entries[pos][0] == value:
            if self._entries[pos][1] == entry_id:
                self._entries.pop(pos)
                return True
            pos += 1
        return False

    def find_pairs(self, recent_entry_ids, win_sid_idx, win_time, tau):
        tau = float(tau)
        if tau < 0.0 or not self._entries:
            return []
        entries = self._entries
        sid_idx = np.asarray(win_sid_idx, dtype=np.int64)
        times = np.asarray(win_time, dtype=np.int64)
        pairs = []
        for raw_id in np.asarray(recent_entry_ids, dtype=np.int64):
            entry_id = int(raw_id)
            meta = self._meta.get(entry_id)
            if meta is None:
                continue
            value, ridx, _ = meta
            if ridx < 0 or ridx >= sid_idx.shape[0]:
                continue
            sid_r = int(sid_idx[ridx])
            time_r = int(times[ridx])
            lower = value - tau
            upper = value + tau
            left = bisect_left(entries, (lower, self._LOW_SENTINEL, self._LOW_SENTINEL))
            right = bisect_right(entries, (upper, self._HIGH_SENTINEL, self._HIGH_SENTINEL))
            for _v, other_entry_id, other_idx in entries[left:right]:
                if other_entry_id == entry_id:
                    continue
                if other_idx < 0 or other_idx >= sid_idx.shape[0]:
                    continue
                sid_o = int(sid_idx[other_idx])
                time_o = int(times[other_idx])
                if sid_o == sid_r and time_o == time_r:
                    continue
                pairs.append((int(ridx), int(other_idx)))
        return pairs

    def find_pairs_full(self, recent_entry_ids, win_sid_idx, win_time, tau):
        tau = float(tau)
        if tau < 0.0 or not self._entries or self.n_vectors <= 0:
            return []
        tau_sq = tau * tau
        entries = self._entries
        sid_idx = np.asarray(win_sid_idx, dtype=np.int64)
        times = np.asarray(win_time, dtype=np.int64)
        pairs = []
        for raw_id in np.asarray(recent_entry_ids, dtype=np.int64):
            entry_id = int(raw_id)
            meta = self._meta.get(entry_id)
            if meta is None:
                continue
            value, ridx, vec_r = meta
            if vec_r is None:
                continue
            if ridx < 0 or ridx >= sid_idx.shape[0]:
                continue
            sid_r = int(sid_idx[ridx])
            time_r = int(times[ridx])
            lower = value - tau
            upper = value + tau
            left = bisect_left(entries, (lower, self._LOW_SENTINEL, self._LOW_SENTINEL))
            right = bisect_right(entries, (upper, self._HIGH_SENTINEL, self._HIGH_SENTINEL))
            for _v, other_entry_id, other_idx in entries[left:right]:
                if other_entry_id == entry_id:
                    continue
                other_meta = self._meta.get(int(other_entry_id))
                if other_meta is None:
                    continue
                vec_o = other_meta[2]
                if vec_o is None:
                    continue
                if other_idx < 0 or other_idx >= sid_idx.shape[0]:
                    continue
                sid_o = int(sid_idx[other_idx])
                time_o = int(times[other_idx])
                if sid_o == sid_r and time_o == time_r:
                    continue
                diff = vec_r - vec_o
                if float(np.dot(diff, diff)) > tau_sq:
                    continue
                pairs.append((int(ridx), int(other_idx)))
        return pairs


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


def find_candidate_pairs_full(
    values,
    value_window_idx,
    recent_values,
    recent_window_idx,
    win_sid_idx,
    win_time,
    entry_vectors,
    recent_vectors,
    tau,
):
    pairs = []
    values = np.asarray(values, dtype=np.float64)
    value_window_idx = np.asarray(value_window_idx, dtype=np.int64)
    recent_values = np.asarray(recent_values, dtype=np.float64)
    recent_window_idx = np.asarray(recent_window_idx, dtype=np.int64)
    win_sid_idx = np.asarray(win_sid_idx, dtype=np.int64)
    win_time = np.asarray(win_time, dtype=np.int64)
    entry_vectors = np.asarray(entry_vectors, dtype=np.float64)
    recent_vectors = np.asarray(recent_vectors, dtype=np.float64)
    if values.size == 0 or recent_values.size == 0 or tau < 0.0:
        return pairs
    tau_sq = float(tau) * float(tau)
    for i, (val, ridx) in enumerate(zip(recent_values, recent_window_idx)):
        lower = val - tau
        upper = val + tau
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        sid_r = win_sid_idx[ridx]
        time_r = win_time[ridx]
        vec_r = recent_vectors[i]
        for j in range(left, right):
            other_idx = value_window_idx[j]
            if other_idx == ridx:
                continue
            if win_sid_idx[other_idx] == sid_r and win_time[other_idx] == time_r:
                continue
            diff = vec_r - entry_vectors[j]
            if float(np.dot(diff, diff)) > tau_sq:
                continue
            pairs.append((int(ridx), int(other_idx)))
    return pairs


def enumerate_candidate_rows(
    data,
    window_index,
    ref_indices,
    window_size,
    window_step,
    std_thresh=1e-3,
    shard_start=-1,
    shard_end=-1,
):
    data = np.asarray(data)
    n_series, n_cols = data.shape
    window_count = n_cols - window_size + 1
    if window_count <= 0:
        return None

    working_mask = _compute_nonconst_mask(
        data.astype(np.float64, copy=False),
        window_size,
        std_thresh=std_thresh,
    )
    if working_mask.size == 0:
        return None

    step = window_step if window_step > 0 else 1
    step_mask = (np.arange(window_count) % step) == 0
    valid_mask = working_mask & step_mask
    last_idx = window_count - 1
    seeds_mask = valid_mask[:, last_idx]
    if not np.any(seeds_mask):
        return None

    valid_k_all, valid_j_all = np.nonzero(valid_mask)
    if valid_k_all.size == 0:
        return None

    j_start_times = np.asarray(window_index, dtype=np.int64)[valid_j_all]
    curr_start = int(np.asarray(window_index, dtype=np.int64)[last_idx])
    rows_accum = []

    ref_indices = np.asarray(list(ref_indices), dtype=np.int64)
    for s_idx in ref_indices:
        if s_idx >= n_series or not seeds_mask[s_idx]:
            continue

        mask_sel = np.ones(valid_k_all.shape[0], dtype=bool)
        mask_sel &= ~((valid_k_all <= s_idx) & (valid_j_all == last_idx))
        if not mask_sel.any():
            continue

        indices = np.nonzero(mask_sel)[0]
        k_sel = valid_k_all[indices]
        if shard_start >= 0 and shard_end >= 0:
            shard_mask = (k_sel >= shard_start) & (k_sel < shard_end)
            if not shard_mask.any():
                continue
            indices = indices[shard_mask]
            k_sel = k_sel[shard_mask]
        if indices.size == 0:
            continue

        rows_accum.append(
            np.column_stack(
                [
                    np.full(indices.size, s_idx, dtype=np.int64),
                    valid_k_all[indices].astype(np.int64, copy=False),
                    np.full(indices.size, curr_start, dtype=np.int64),
                    j_start_times[indices],
                    np.full(indices.size, window_size, dtype=np.int64),
                ]
            )
        )

    if not rows_accum:
        return None

    return np.vstack(rows_accum)



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
