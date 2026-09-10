"""Exact O(n log n) univariate distance correlation, following the approach
of Huo & Szekely (2016) "Fast Computing for Distance Covariance"
(Technometrics) -- part 4 of the 2026-07-27 roadmap.

(2026-07-29) This project's existing `_distance_correlation_1d` (in
library_corrtrack_parallel.py) is the naive O(w^2) double-centering
estimator -- correct, but scales quadratically with window size. This
module computes the EXACT same quantity (not an approximation, unlike
Xiao's online Spearman/Kendall) in O(w log w), for 1-D series only.

IMPORTANT DISCLOSURE: the source paper's own text was not available in this
session -- only its name and general description (an O(n log n) algorithm
for 1-D distance covariance). The algorithm below was independently derived
from the standard double-centering identity for distance covariance/
correlation, not transcribed from the paper. It is verified here by direct,
exhaustive comparison against the existing brute-force _distance_
correlation_1d (many random trials, with and without ties, at multiple
sample sizes) rather than trusted on derivation alone -- per this project's
"do not claim recall preservation unless the benchmark verifies it" rule,
extended here to "do not claim exactness without verifying it directly."

Derivation sketch: for the double-centered distance matrices A (from
|x_i-x_j|) and B (from |y_i-y_j|), sum_ij A_ij*B_ij expands (see this
module's git history / docs/implementation_log.md's 2026-07-29 entry for
the full term-by-term expansion) to:

    sum_ij A_ij B_ij = Sigma_ab - 2n*S + n^2 * abar * bbar

where Sigma_ab = sum_ij |x_i-x_j||y_i-y_j| (the hard O(n^2) term, reduced to
O(n log n) below via a Fenwick-tree sweep in x-sorted order), S = sum_i
(a_i. * b_i.) (row means, each computable in O(n log n) via sorting +
prefix sums, independent of the cross term), and abar/bbar are the grand
means of the two raw distance matrices. The variance terms (dvar_x, dvar_y)
use the same identity with a=b, which simplifies further since
sum_ij|x_i-x_j|^2 has a trivial O(n) closed form.
"""

import numpy as np


class _FenwickTree:
    """1-indexed internally; supports point update and prefix-sum query."""

    def __init__(self, size):
        self.n = int(size)
        self.tree = np.zeros(self.n + 1, dtype=np.float64)

    def update(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, index):
        # sum of elements with rank <= index (0-indexed index)
        if index < 0:
            return 0.0
        i = min(index + 1, self.n)
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s


def _row_sums_abs_diff(v):
    # For each i, sum_j |v_i - v_j|, for ALL i, in O(n log n): sort once,
    # then a running prefix-sum gives each row's sum in O(1).
    n = v.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order].astype(np.float64, copy=False)
    prefix = np.cumsum(v_sorted)
    total = prefix[-1]
    k = np.arange(n, dtype=np.float64)
    # row_sum[k] = v0*(2k+2-n) + total - 2*prefix[k], for the k-th smallest
    # value (0-indexed k); see module docstring for the derivation.
    row_sorted = v_sorted * (2.0 * k + 2.0 - n) + total - 2.0 * prefix
    row = np.empty(n, dtype=np.float64)
    row[order] = row_sorted
    return row


def _dense_rank(v):
    _uniq, inv = np.unique(v, return_inverse=True)
    return inv, int(_uniq.shape[0])


def _cross_term_sum_ab(x, y):
    # Sigma_ab = sum_{i,j} |x_i-x_j| * |y_i-y_j| (ordered pairs, i can equal
    # j -- those terms are 0 anyway). Processes points sorted by x; for
    # i<j in that order, |x_i-x_j| = x_j-x_i, and |y_i-y_j| is split by
    # whether y_i <= y_j or not, tracked via 4 Fenwick trees over y-rank
    # (count, sum_x, sum_y, sum_xy of already-inserted points) -- see
    # module docstring.
    n = x.shape[0]
    if n < 2:
        return 0.0
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order].astype(np.float64, copy=False)
    y_sorted = y[order].astype(np.float64, copy=False)
    y_rank, n_ranks = _dense_rank(y_sorted)

    cnt = _FenwickTree(n_ranks)
    sumx = _FenwickTree(n_ranks)
    sumy = _FenwickTree(n_ranks)
    sumxy = _FenwickTree(n_ranks)

    total_f = 0.0
    last_rank = n_ranks - 1
    for j in range(n):
        if j > 0:
            xj = x_sorted[j]
            yj = y_sorted[j]
            rj = int(y_rank[j])

            c_le = cnt.prefix_sum(rj)
            sx_le = sumx.prefix_sum(rj)
            sy_le = sumy.prefix_sum(rj)
            sxy_le = sumxy.prefix_sum(rj)

            c_all = float(j)
            sx_all = sumx.prefix_sum(last_rank)
            sy_all = sumy.prefix_sum(last_rank)
            sxy_all = sumxy.prefix_sum(last_rank)

            c_gt = c_all - c_le
            sx_gt = sx_all - sx_le
            sy_gt = sy_all - sy_le
            sxy_gt = sxy_all - sxy_le

            g1 = yj * c_le - sy_le + sy_gt - yj * c_gt
            g2 = yj * sx_le - sxy_le + sxy_gt - yj * sx_gt
            total_f += xj * g1 - g2

        rj_ins = int(y_rank[j])
        cnt.update(rj_ins, 1.0)
        sumx.update(rj_ins, x_sorted[j])
        sumy.update(rj_ins, y_sorted[j])
        sumxy.update(rj_ins, x_sorted[j] * y_sorted[j])

    return 2.0 * total_f


def distance_correlation_1d_fast(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n < 2:
        return np.nan
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.shape[0])
    if n < 2:
        return np.nan

    row_a = _row_sums_abs_diff(x)
    row_b = _row_sums_abs_diff(y)
    abar_i = row_a / n
    bbar_i = row_b / n
    abar = float(np.sum(row_a)) / (n * n)
    bbar = float(np.sum(row_b)) / (n * n)
    s_ab = float(np.sum(abar_i * bbar_i))
    s_aa = float(np.sum(abar_i * abar_i))
    s_bb = float(np.sum(bbar_i * bbar_i))

    sigma_ab = _cross_term_sum_ab(x, y)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sigma_aa = 2.0 * n * float(np.sum(x * x)) - 2.0 * sum_x * sum_x
    sigma_bb = 2.0 * n * float(np.sum(y * y)) - 2.0 * sum_y * sum_y

    dcov2 = (sigma_ab / (n * n)) - (2.0 * s_ab / n) + abar * bbar
    dvar_x = (sigma_aa / (n * n)) - (2.0 * s_aa / n) + abar * abar
    dvar_y = (sigma_bb / (n * n)) - (2.0 * s_bb / n) + bbar * bbar

    denom = np.sqrt(max(dvar_x, 0.0) * max(dvar_y, 0.0))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    dcorr2 = max(dcov2, 0.0) / denom
    return float(np.sqrt(max(0.0, min(1.0, dcorr2))))
