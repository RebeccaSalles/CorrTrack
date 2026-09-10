"""Online (incremental) Spearman/Kendall correlation, ported from Xiao (2017)
"An Online Algorithm for Nonparametric Correlations".

(2026-07-29) Part of the Kendall's-tau/distance-correlation implementation
thread (part 4 of the 2026-07-27 roadmap) -- see docs/implementation_log.md.
Maintains a small m1 x m2 count matrix per (series-pair, lag) approximating
the joint distribution of a window via fixed value-space cutpoints; both
metrics are recomputed from that matrix in O(m1*m2), independent of window
size. As the window slides by window_step, the matrix is updated
incrementally (O(window_step * log(m)) per step) instead of rebuilding from
the full window every time -- this is what corrtrack_release_dev's own
non-Pearson validation path (_validate_numeric_rows_nonlinear) lacked: an
equivalent to Pearson's own basic-window incremental decomposition.

THIS IS AN APPROXIMATION, not an exact computation -- the count matrix is a
coarsened (binned) view of the joint distribution, so results will not
match scipy.stats.spearmanr/kendalltau bit-for-bit (Xiao's own paper reports
L1 error on the order of 1e-3 to 1e-2 depending on cutpoint count). It is
also only as good as the FIXED cutpoints derived once when a pair-lag is
first seen: if the series' marginal distribution drifts substantially
afterward (a genuinely nonstationary process, e.g. a random walk), the
bins can become a poor fit for later windows. Neither of these is
addressed here (a drift-adaptive cutpoint refresh, mirroring HBR's
adaptive-update pattern, is a disclosed, not-yet-taken follow-up) -- this
is why the feature is opt-in (validation_incremental_approx=False by
default), never a silent replacement for the exact scipy path.
"""

import math

import numpy as np


def derive_cutpoints(values, m):
    m = max(1, int(m))
    if m <= 1:
        return np.empty(0, dtype=np.float64)
    probs = np.linspace(0.0, 1.0, m + 1)[1:-1]
    cutpoints = np.quantile(np.asarray(values, dtype=np.float64), probs)
    return np.ascontiguousarray(cutpoints, dtype=np.float64)


def _bin_indices(values, cutpoints):
    return np.searchsorted(cutpoints, np.asarray(values, dtype=np.float64), side="right")


def _spearman_from_matrix(M, nrow, ncol, n):
    m1, m2 = M.shape
    r_row = np.zeros(m1, dtype=np.float64)
    r = 0.0
    for k in range(m1):
        if nrow[k] == 0:
            r_row[k] = r
        else:
            r_row[k] = ((r + 1) + (r + nrow[k])) / 2.0
            r += nrow[k]
    r_col = np.zeros(m2, dtype=np.float64)
    r = 0.0
    for k in range(m2):
        if ncol[k] == 0:
            r_col[k] = r
        else:
            r_col[k] = ((r + 1) + (r + ncol[k])) / 2.0
            r += ncol[k]

    r_row_star = r_row - (n + 1) / 2.0
    r_col_star = r_col - (n + 1) / 2.0
    denom_row = math.sqrt(float(np.sum(nrow * r_row_star ** 2)))
    denom_col = math.sqrt(float(np.sum(ncol * r_col_star ** 2)))
    if denom_row <= 0.0 or denom_col <= 0.0 or not math.isfinite(denom_row) or not math.isfinite(denom_col):
        return np.nan
    r_row_star = r_row_star / denom_row
    r_col_star = r_col_star / denom_col
    corr = float(r_row_star @ M.astype(np.float64) @ r_col_star)
    return max(-1.0, min(1.0, corr))


def _kendall_from_matrix(M, nrow, ncol, n):
    m1, m2 = M.shape
    Mf = M.astype(np.float64)
    N = np.zeros((m1, m2), dtype=np.float64)
    for i in range(1, m1):
        for j in range(1, m2):
            if j == 1:
                N[i, j] = Mf[i - 1, j - 1]
            else:
                N[i, j] = N[i, j - 1] + Mf[i - 1, j - 1]
    for i in range(1, m1):
        N[i, :] = N[i, :] + N[i - 1, :]

    P = float(np.sum(Mf * N))
    row_sq = np.sum(Mf ** 2, axis=1)
    col_sq = np.sum(Mf ** 2, axis=0)
    T = float(np.sum(nrow.astype(np.float64) ** 2 - row_sq)) / 2.0
    U = float(np.sum(ncol.astype(np.float64) ** 2 - col_sq)) / 2.0
    B = float(np.sum(Mf * (Mf - 1.0))) / 2.0
    # (2026-07-29) Total pairs of distinct points is n*(n-1)/2 -- Xiao's own
    # paper writes (n+1)*n/2 here, which does not reproduce scipy's
    # kendalltau (verified directly against scipy on synthetic data with and
    # without ties before shipping this); n*(n-1)/2 is the standard
    # combinatorial total (P+Q+T+U+B must partition all pairs of distinct
    # points) and matches scipy exactly on tie-free data.
    total_pairs = n * (n - 1) / 2.0
    Q = total_pairs - P - T - U - B
    denom = math.sqrt(max((P + Q + T) * (P + Q + U), 0.0))
    if denom <= 0.0 or not math.isfinite(denom):
        return np.nan
    corr = (P - Q) / denom
    return max(-1.0, min(1.0, corr))


class XiaoOnlineCorrState:
    """Per-(pair,lag) incremental Spearman/Kendall state, keyed externally
    (callers pass their own key, e.g. CorrTrack._normalize_pair_key's
    (s1,s2,lag) convention)."""

    def __init__(self, max_age_steps=64):
        self._states = {}
        self._step = 0
        self.max_age_steps = max(1, int(max_age_steps))

    def touch_step(self):
        self._step += 1

    def size(self):
        return len(self._states)

    def prune(self):
        cutoff = self._step - self.max_age_steps
        stale = [k for k, v in self._states.items() if v["last_step_seen"] < cutoff]
        for k in stale:
            del self._states[k]
        return len(stale)

    def get_corr(self, key, x, y, t1, window_step, metric, m1=None, m2=None, cutpoint_refresh_threshold=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = x.shape[0]
        if m1 is None:
            m1 = 30 if metric == "spearman" else 100
        if m2 is None:
            m2 = m1
        m1 = max(2, int(m1))
        m2 = max(2, int(m2))

        state = self._states.get(key)
        reusable = (
            state is not None
            and state["metric"] == metric
            and state["m1"] == m1
            and state["m2"] == m2
            and state["window_size"] == w
            and state["last_t1"] is not None
            and (t1 - state["last_t1"]) == window_step
            and 0 < window_step < w
        )
        if reusable and cutpoint_refresh_threshold is not None:
            # (2026-07-29) HBR (Zhang et al. 2009)-style adaptive-update
            # pattern, Section 4.2/Fig 7: rather than waiting for the fixed
            # cutpoints to degenerate outright (the reactive NaN fallback
            # below), proactively check whether the window's own mean has
            # drifted more than cutpoint_refresh_threshold standard
            # deviations away from where the cutpoints were last derived --
            # if so, refresh now rather than let accuracy silently degrade
            # first. xi=0.5 is HBR's own mid-range choice (their Fig 12
            # experiment swept 0.4-1.2); 0 disables the proactive check
            # (reactive-only, the prior behavior) and None also disables it.
            if not self._within_drift_tolerance(state, x, y, cutpoint_refresh_threshold):
                reusable = False
        if reusable:
            self._update_incremental(state, x, y, window_step)
        else:
            state = self._build_fresh(x, y, m1, m2, metric)
            self._states[key] = state
        state["last_t1"] = t1
        state["last_step_seen"] = self._step

        corr = self._corr_from_state(state, metric)
        if not math.isfinite(corr) and reusable:
            # (2026-07-29) Fixed cutpoints derived long ago can go stale on a
            # genuinely nonstationary series (e.g. a random walk): once the
            # window has drifted far enough outside the original cutpoint
            # range, every point collapses into one extreme bin, nrow/ncol
            # lose all variance, and the correlation formula's denominator
            # degenerates to 0 -- reproduced directly against a synthetic
            # random-walk series before adding this fallback (see
            # docs/implementation_log.md's 2026-07-29 entry). Re-derive
            # cutpoints from the CURRENT window (the same one-time O(w log
            # w) cost a fresh build always pays) rather than return NaN.
            state = self._build_fresh(x, y, m1, m2, metric)
            state["last_t1"] = t1
            state["last_step_seen"] = self._step
            self._states[key] = state
            corr = self._corr_from_state(state, metric)
        return corr

    @staticmethod
    def _within_drift_tolerance(state, x, y, threshold):
        threshold = float(threshold)
        cur_mean_x = float(np.mean(x))
        cur_mean_y = float(np.mean(y))
        rel_x = abs(cur_mean_x - state["ref_mean_x"]) / state["ref_std_x"] if state["ref_std_x"] > 0.0 else 0.0
        rel_y = abs(cur_mean_y - state["ref_mean_y"]) / state["ref_std_y"] if state["ref_std_y"] > 0.0 else 0.0
        return rel_x <= threshold and rel_y <= threshold

    @staticmethod
    def _corr_from_state(state, metric):
        if metric == "spearman":
            return _spearman_from_matrix(state["M"], state["nrow"], state["ncol"], state["n"])
        return _kendall_from_matrix(state["M"], state["nrow"], state["ncol"], state["n"])

    @staticmethod
    def _build_fresh(x, y, m1, m2, metric):
        cx = derive_cutpoints(x, m1)
        cy = derive_cutpoints(y, m2)
        row_idx = _bin_indices(x, cx)
        col_idx = _bin_indices(y, cy)
        M = np.zeros((m1, m2), dtype=np.int64)
        np.add.at(M, (row_idx, col_idx), 1)
        nrow = M.sum(axis=1)
        ncol = M.sum(axis=0)
        n = int(M.sum())
        return {
            "cx": cx, "cy": cy, "M": M, "nrow": nrow, "ncol": ncol, "n": n,
            "m1": m1, "m2": m2, "metric": metric, "window_size": int(x.shape[0]),
            "last_t1": None, "last_step_seen": None,
            "prev_x": x, "prev_y": y,
            "ref_mean_x": float(np.mean(x)), "ref_std_x": float(np.std(x)),
            "ref_mean_y": float(np.mean(y)), "ref_std_y": float(np.std(y)),
        }

    @staticmethod
    def _update_incremental(state, x, y, window_step):
        cx, cy = state["cx"], state["cy"]
        M, nrow, ncol = state["M"], state["nrow"], state["ncol"]
        prev_x, prev_y = state["prev_x"], state["prev_y"]
        w = x.shape[0]

        exp_i = _bin_indices(prev_x[:window_step], cx)
        exp_j = _bin_indices(prev_y[:window_step], cy)
        np.add.at(M, (exp_i, exp_j), -1)
        np.add.at(nrow, exp_i, -1)
        np.add.at(ncol, exp_j, -1)

        new_i = _bin_indices(x[w - window_step:], cx)
        new_j = _bin_indices(y[w - window_step:], cy)
        np.add.at(M, (new_i, new_j), 1)
        np.add.at(nrow, new_i, 1)
        np.add.at(ncol, new_j, 1)

        state["prev_x"] = x
        state["prev_y"] = y
