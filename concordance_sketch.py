"""Incremental, nonstationary-robust candidate-search representation for
Kendall's-tau-style concordance screening.

(2026-07-29) Part 4 of the 2026-07-27 roadmap, built in direct response to
the question "can't I use my incremental sketching algorithm and
nonstationary normalization for screening Kendall candidates?" Answered no
for the EXISTING linear sketch directly (it's a linear functional of the
normalized raw values, s_x = M @ x_normalized for a fixed M -- cosine
similarity of two such sketches approximates Pearson correlation
specifically because Pearson correlation IS a cosine of centered vectors,
an equivalence that does not hold for Kendall's tau, a function of the
SIGNS of pairwise differences, not a linear functional of the values) --
but yes with a real redesign, reusing the ARCHITECTURAL PATTERN (basic
incremental block bookkeeping: compute only what's new each step, drop
only what's expired, never rescan the whole window) rather than the linear
sketch's own machinery directly.

GlobalOrdinalTransformer (global_ordinal_backend.py, merged 2026-07-28)
already does this in spirit but samples WINDOW-RELATIVE OFFSET pairs
(i, j), which point at different absolute timestamps every time the
window slides -- nothing to reuse across steps. This module instead fixes
a small set of GAPS g (multiscale, short to long) and compares x[t] to
x[t-g] for every valid absolute t within the window -- since a given
(t, t-g) comparison, once computed, never changes (it depends only on two
fixed historical values), only the `window_step` NEWLY arrived
comparisons need computing each step; the rest are carried over.

Verified before building the incremental machinery (see
docs/implementation_log.md's 2026-07-29 entry): cosine similarity of two
series' concatenated multiscale fixed-gap sign vectors is an excellent
Kendall's-tau estimator (correlation 0.9995 with true tau across 300
synthetic trials spanning linear, monotonic-nonlinear, anti-correlated,
and independent relationships) -- and, because sign(x[t]-x[t-g]) is
scale/location-invariant, this representation is inherently robust to
nonstationary drift, unlike Xiao (2017)'s value-binning validator (which
needed a reactive rebuild-on-degeneracy fix for exactly this failure mode).

Benchmarked before committing to this design (per this project's own
"Correct != beneficial" discipline): GlobalOrdinalTransformer's existing
per-step cost is already ~350us for 50 series at window=256 (it gathers
only `num_pairs` positions, not the whole window) -- i.e. speed was never
the bottleneck here. The value of this module is the nonstationary
robustness property and the literal reuse of the incremental-bookkeeping
architecture, not a wall-clock win over GlobalOrdinalTransformer.

(2026-07-29f) The 0.9995 figure above is from i.i.d.-noise synthetic data
only. On REAL, autocorrelated data (hourly wind-speed series), this
representation's accuracy was measurably worse than synthetic testing
predicted (mean|cos-true_tau| 0.116 vs 0.027) until `multiscale_gaps`'s
`min_gap`/`min_capacity` defaults were raised from 1/none to 8/16 -- see
that function's docstring and docs/implementation_log.md's 2026-07-29(f)
entry for the full diagnosis (short gaps carry the most weight AND are
the most biased on autocorrelated data; the longest gaps carry the least
weight but are the highest-variance). Fixed to 0.057 at window_size=256,
target_dim=738 (matching the synthetic-predicted order of magnitude),
with no synthetic regression. Disclosed cost: hitting the same target_dim
under the new floor needs more gaps (4 instead of 3 at window_size=256,
output_dim 674 instead of 496), which narrows this representation's
wall-clock edge over GlobalOrdinalTransformer on real data from ~2.8x to
~1.6x faster (838us vs ordinal's ~1361us) -- still faster, just less so;
an honest tradeoff, not hidden.
"""

import numpy as np


def multiscale_gaps(window_size, n_gaps=16, min_gap=8, min_capacity=16):
    """Choose gaps in [min_gap, window_size - min_capacity], log-spaced.

    (2026-07-29f) `min_gap` and `min_capacity` both DEFAULTED HIGHER than
    their original values (1 and "none"/window_size-1) after diagnosing a
    real accuracy problem on real (autocorrelated) wind-speed data: with
    whole-vector L2 normalization, each gap's weight in the combined
    cosine estimator is proportional to its capacity (window_size - g).
    `min_gap=1` (adjacent-timestamp comparisons) is both the HIGHEST-
    capacity gap (most weight) and, on smooth physically-autocorrelated
    data, the WORST tau estimator (local jitter dominates the true sign),
    dragging the combined estimate down. The largest gaps (near
    window_size) have the OPPOSITE problem: capacity collapses toward 1
    sample, making that gap's own cosine contribution extremely
    high-variance. `min_capacity` bounds `max_gap = window_size -
    min_capacity` to keep every included gap's estimator reasonably
    low-variance. Measured on real data (window_size=256, target_dim=738):
    combined mean|cos-true_tau| improved from 0.116 (min_gap=1, no
    capacity floor) to 0.067 (min_gap=8, min_capacity=16) -- with NO
    regression on synthetic i.i.d. data (0.0298 vs 0.0298, unchanged,
    since i.i.d. data has no gap-1-specific bias to exclude). Considered
    and rejected: per-gap equal-weighting instead of this capacity floor
    -- measured WORSE (0.145-0.256) because it overweights the lowest-
    capacity (highest-variance) gap instead of excluding it. See
    docs/implementation_log.md's 2026-07-29(f) entry.
    """
    window_size = int(window_size)
    min_gap = max(1, int(min_gap))
    min_capacity = max(1, int(min_capacity))
    max_gap = max(min_gap, window_size - min_capacity)
    if max_gap <= min_gap:
        return np.array([min_gap], dtype=np.int64)
    n_gaps = max(1, int(n_gaps))
    gaps = np.unique(np.round(np.geomspace(min_gap, max_gap, n_gaps)).astype(np.int64))
    gaps = gaps[(gaps >= min_gap) & (gaps <= max_gap) & (gaps < window_size)]
    if gaps.size == 0:
        gaps = np.array([min_gap], dtype=np.int64)
    return gaps


def derive_concordance_candidate_tau(corr_threshold):
    """Rough, EMPIRICALLY-fit default for tier 1's OWN retrieval threshold
    (`concordance_candidate_tau`) -- the cosine-similarity bar on
    ConcordanceSketchState.vectors()'s fixed-gap sign representation.

    (2026-07-31) Direct response to a real recall regression found via a
    full real-data recall/precision sweep at large windows (window_size in
    {1024, 2048, 4096}, real fr_wind_direction_121_1 data, M=24,
    validation_metric="spearman", corr_threshold=0.5): the previous default
    came from `_derive_candidate_tau`, a formula SHARED with
    global_ordinal_comparisons -- `(2/pi)*asin(corr_threshold)`, giving
    0.333 at corr_threshold=0.5. That formula is theoretically correct for
    global_ordinal_comparisons' representation (many random within-window
    position-pairs converging to the population-level correlation) but does
    NOT transfer to concordance's fixed-gap representation, especially once
    n_gaps floors at 1 (starting at window_size=1024, see multiscale_gaps'
    own docstring) -- confirmed via a touched-vs-rejected diagnostic that
    100% of concordance's missed true positives at window_size=1024 were
    never even retrieved by tier 1 (zero threshold-side gate rejections),
    and via a direct candidate_tau sweep: loosening tau from 0.333 to 0.1-
    0.2 recovered recall from 75.1%/67.6% to 95.7%/100% at window_size
    1024/2048, with precision staying 100% throughout every tau value
    tested (tier 2's exact validation protects precision, not this
    threshold -- same division of labor as distance_corr_sketch_candidate_tau).

    A larger, more reliable measurement (6 overlapping windows per
    window_size, M=30, real data) of true-positive (|spearman rho|>=0.7)
    vs. independent (|rho|<0.2) pair cosine similarity on vectors() gave:
        window_size=256:  true_pos min=0.249 10th_pct=0.280 | indep 90th_pct=0.120 95th_pct=0.148
        window_size=1024: true_pos min=0.357 10th_pct=0.500 | indep 90th_pct=0.169 95th_pct=0.193
        window_size=2048: true_pos min=0.333 10th_pct=0.468 | indep 90th_pct=0.221 95th_pct=0.253
        window_size=4096: true_pos min=0.447 10th_pct=0.510 | indep 90th_pct=0.249 95th_pct=0.272
    Notably, unlike distance_corr_sketch_gate_tau's clean 1/sqrt(window_size)
    noise-floor scaling law, here BOTH true-positive and independent-pair
    percentiles drift UPWARD with window_size (not down) -- consistent with
    this project's own documented finding (2026-07-29(f)) that fixed-gap
    comparisons are sensitive to autocorrelation structure in real data,
    not just i.i.d. sampling noise. No clean window-size-dependent scaling
    law was found to hold across all four window sizes; forcing one would
    overfit to this dataset's own autocorrelation profile. Fixed instead to
    a small, flat (window-size-independent), corr_threshold-scaled constant
    -- comfortably below every measured true-positive minimum (0.249-0.447)
    and above typical (median) independent-pair similarity, the same
    "small constant, comfortable margin, let tier 2 protect precision"
    design as derive_candidate_tau_from_corr_threshold. Disclosed,
    unverified assumption: at large window_size (>=2048), independent-pair
    upper percentiles (0.22-0.27) exceed this tau, so more non-matching
    pairs will reach tier 2 than at window_size=256 -- a disclosed
    precision/speed tradeoff (recall-preserving, not precision-risking,
    since tier 2 still filters exactly), not a hidden gap. Always
    overridable directly via concordance_candidate_tau.
    """
    threshold = max(0.0, min(1.0, float(corr_threshold)))
    return max(0.02, 0.05 + 0.2 * threshold)


class _RingGapBuffer:
    """Amortized O(k)-per-append rolling window over the last `capacity`
    columns, for an append of k new columns each call.

    (2026-07-29) A first version of ConcordanceSketchState used
    np.concatenate([kept, new_part]) to slide the buffer -- this SILENTLY
    cost O(capacity) per step (a full copy of the retained data), not
    O(window_step), despite only computing window_step new values --
    caught by direct benchmarking (update time grew with window_size
    instead of staying flat), not by inspection. This class fixes it with
    the standard "double buffer, compact only when full" ring-buffer
    trick -- the same amortization argument behind dynamic array
    doubling: allocate 2x the needed capacity, append into the free tail
    in O(k), and only when the tail is exhausted (every ~capacity/k
    appends) pay a single O(capacity) compaction -- averaging out to O(k)
    per append.

    (2026-07-30) Also tracks a running per-series sum and sum-of-squares
    over the CURRENT capacity-window, updated incrementally in append() by
    adding the k newly-arrived values' contribution and subtracting the k
    just-expired values' contribution -- both O(k), not O(capacity). This
    lets callers get mean()/sum_sq() (and therefore the L2 norm of the
    mean-centered window, via the sum_sq - n*mean^2 identity) without a
    separate O(capacity) reduction pass over view() -- direct response to
    "can we make the mean-centering + L2-normalization step incremental
    too" (distance_corr_sketch.py's vectors()/channel_vectors() and this
    module's own vectors() both used to call .mean()/np.linalg.norm() on
    the full view() every step). Floating-point drift risk of a naive
    running sum (well-known for long-running incremental sums) is bounded
    by recomputing sum/sum_sq FRESH from the live window at every
    compaction (already happening every ~capacity/k appends regardless,
    for the ring-buffer slide itself) -- verified directly (see
    docs/implementation_log.md's 2026-07-30(o) entry) that this keeps
    drift against a from-scratch recompute at machine-epsilon level over
    300+ sliding steps, not just assumed safe.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._alloc = max(1, 2 * self.capacity)
        self._buf = None
        self._write_pos = self.capacity
        self._running_sum = None
        self._running_sum_sq = None

    def reset(self, initial_window):
        # initial_window: shape (n_series, capacity)
        n_series = initial_window.shape[0]
        self._buf = np.zeros((n_series, self._alloc), dtype=np.float64)
        self._buf[:, :self.capacity] = initial_window
        self._write_pos = self.capacity
        self._running_sum = initial_window.sum(axis=1)
        self._running_sum_sq = np.einsum("ij,ij->i", initial_window, initial_window)

    def append(self, new_values):
        # new_values: shape (n_series, k), k <= capacity
        if self._buf is None or new_values.shape[0] != self._buf.shape[0]:
            raise ValueError("_RingGapBuffer.append called before reset (or series count changed)")
        k = new_values.shape[1]
        old_values = self.view()[:, :k]
        self._running_sum += new_values.sum(axis=1) - old_values.sum(axis=1)
        self._running_sum_sq += (
            np.einsum("ij,ij->i", new_values, new_values)
            - np.einsum("ij,ij->i", old_values, old_values)
        )
        compacted = self._write_pos + k > self._alloc
        if compacted:
            live = self._buf[:, self._write_pos - self.capacity:self._write_pos]
            self._buf[:, :self.capacity] = live
            self._write_pos = self.capacity
        self._buf[:, self._write_pos:self._write_pos + k] = new_values
        self._write_pos += k
        # (2026-07-30) Bound running-sum floating-point drift: whenever
        # compaction just happened above, recompute sum/sum_sq fresh from
        # the POST-append view() (which correctly reflects this call's
        # old-dropped/new-added slide) -- compaction already touches the
        # full live window (the copy), so this adds no new O(capacity)
        # work beyond what compaction already costs, and resets any
        # accumulated rounding error to exact machine precision. Must
        # happen AFTER new_values is written and _write_pos updated, using
        # an explicit `compacted` flag rather than `live` directly -- a
        # first version recomputed from `live`, the PRE-this-call window,
        # which does not yet include new_values or exclude old_values and
        # silently reverted the correct incremental update above (a real
        # bug caught by this class's own bit-exactness regression test,
        # not by inspection).
        if compacted:
            current = self.view()
            self._running_sum = current.sum(axis=1)
            self._running_sum_sq = np.einsum("ij,ij->i", current, current)

    def view(self):
        return self._buf[:, self._write_pos - self.capacity:self._write_pos]

    def sum_sq(self):
        """O(1) per-series sum of squares over the current capacity-window."""
        return self._running_sum_sq

    def mean(self):
        """O(1) per-series mean over the current capacity-window."""
        return self._running_sum / self.capacity

    def norm_of_centered(self):
        """O(1) per-series L2 norm of (view() - mean()), via the identity
        sum((x-mu)^2) = sum(x^2) - capacity*mu^2 -- avoids a second
        O(capacity) reduction pass after materializing the centered array."""
        mu = self.mean()
        var_sum = self._running_sum_sq - self.capacity * mu * mu
        return np.sqrt(np.maximum(var_sum, 0.0))

    def norm_of_raw(self):
        """O(1) per-series L2 norm of view() itself (no centering) --
        sqrt(sum_sq), for callers (like ConcordanceSketchState) that
        normalize without mean-centering."""
        return np.sqrt(np.maximum(self._running_sum_sq, 0.0))


class ConcordanceSketchState:
    """Maintains, per series and per fixed gap, a rolling buffer of
    sign(x[t]-x[t-g]) values across the current window, updated
    incrementally as the window slides by window_step -- each step
    computes only the `window_step` newly-valid comparisons (an O(window_
    step) numpy op, independent of window_size) and appends them into a
    _RingGapBuffer (amortized O(window_step) per append, not the O(window_
    size) a naive concatenate-based rebuild would cost). Falls back to a
    full recompute (still correct, just not incremental for that step)
    whenever the shape changes, the series count changes, or -- for the
    very largest configured gaps -- the lookback needed for the newest
    comparisons would reach before the start of the currently buffered
    window.

    Honest complexity note: this makes the COMPARISON BOOKKEEPING itself
    O(window_step) amortized per step, a genuine algorithmic-complexity
    improvement over recomputing all window_size-g comparisons from
    scratch every step. It does NOT make the full per-step candidate-
    search cost sub-O(output_dim): vectors() must still touch every
    dimension once (to concatenate/normalize/hand to the LSH index), and
    that step is inherently O(output_dim), the same complexity order
    GlobalOrdinalTransformer already has for its own num_pairs-sized
    representation. The win is specifically in avoiding O(window_size)
    RECOMPUTE of comparisons that haven't changed -- not in eliminating
    the unavoidable cost of presenting a full window_size-scaled vector
    to the index.
    """

    def __init__(self, window_size, n_gaps=16, min_gap=8, min_capacity=16, target_dim=None):
        self.window_size = int(window_size)
        min_gap = max(1, int(min_gap))
        min_capacity = max(1, int(min_capacity))
        if target_dim is not None:
            # (2026-07-29) output_dim = sum(window_size - g for g in gaps) ~=
            # n_gaps * window_size (since most gaps are << window_size) --
            # unlike GlobalOrdinalTransformer's num_pairs, which is a FIXED
            # budget set independently of window_size (via the Hoeffding
            # recall bound), this representation's dimension previously
            # scaled LINEARLY with window_size, verified directly to cost
            # 2.6x-20.5x more per step than GlobalOrdinalTransformer at
            # window_size=256/1024 despite genuinely cheaper incremental
            # bookkeeping -- the O(output_dim) cost of producing/normalizing/
            # inserting the full vector dominates once output_dim grows.
            # Fixing this by SUBSAMPLING POSITIONS within each gap would
            # require absolute-time-anchored stride bookkeeping to keep the
            # ring buffer's incremental correctness intact (real, avoidable
            # risk to already-verified-correct code); reducing the NUMBER OF
            # GAPS as window_size grows achieves the same goal (bounded,
            # window-size-independent output_dim) by reusing the existing,
            # unmodified, already bit-exact-verified per-gap incremental
            # machinery -- just changes how many gaps get chosen upfront.
            # Two-pass estimate. The arithmetic mean of the two capacity
            # extremes (window_size - min_gap, min_capacity) systematically
            # UNDERESTIMATES the true average dim of a log-spaced gap set,
            # because log-spacing clusters more points toward the small-gap
            # (large-dim) end -- verified this caused a real ~43% output_dim
            # overshoot (1057 vs target 738) at window_size=256 once
            # min_gap/min_capacity were raised for the 2026-07-29f accuracy
            # fix. Fixed with one cheap extra pass: build a first-guess gap
            # set from the rough estimate, measure its ACTUAL average dim,
            # then re-derive n_gaps from that -- converges to within ~10% of
            # target_dim (674 vs 738), not exact, but target_dim was always
            # an approximate budget, not a hard one.
            rough_avg = max(1, (self.window_size - min_gap + min_capacity) / 2.0)
            n_gaps_rough = max(1, round(float(target_dim) / rough_avg))
            probe_gaps = multiscale_gaps(self.window_size, n_gaps=n_gaps_rough, min_gap=min_gap, min_capacity=min_capacity)
            actual_avg = max(1.0, float(np.mean(self.window_size - probe_gaps)))
            n_gaps = max(1, round(float(target_dim) / actual_avg))
        self.gaps = multiscale_gaps(self.window_size, n_gaps=n_gaps, min_gap=min_gap, min_capacity=min_capacity)
        self.output_dim = int(sum(self.window_size - int(g) for g in self.gaps))
        self._ring_buffers = None
        self._n_series = None

    def _fresh_buffers(self, window_data):
        n_series = window_data.shape[0]
        ring_buffers = {}
        for g in self.gaps:
            g = int(g)
            comp = np.sign(window_data[:, g:] - window_data[:, :-g])
            rb = _RingGapBuffer(capacity=self.window_size - g)
            rb.reset(comp)
            ring_buffers[g] = rb
        self._ring_buffers = ring_buffers
        self._n_series = n_series

    def update(self, window_data, window_step):
        n_series, w = window_data.shape
        window_step = int(window_step)
        if (
            self._ring_buffers is None
            or self._n_series != n_series
            or w != self.window_size
            or window_step <= 0
        ):
            self._fresh_buffers(window_data)
            return

        for g in self.gaps:
            g = int(g)
            rb = self._ring_buffers.get(g)
            capacity = w - g
            start_new_lo = w - window_step - g
            if rb is None or window_step >= capacity or start_new_lo < 0:
                comp = np.sign(window_data[:, g:] - window_data[:, :-g])
                rb = _RingGapBuffer(capacity=capacity)
                rb.reset(comp)
                self._ring_buffers[g] = rb
                continue
            new_part = np.sign(
                window_data[:, w - window_step:] - window_data[:, start_new_lo:w - g]
            )
            rb.append(new_part)

    def vectors(self):
        # (2026-07-29f) Considered, MEASURED, and REJECTED: per-gap L2
        # normalization before concatenation (equal-weight-per-gap,
        # mathematically reduces to mean_g cos(a_g, b_g)). Motivation was a
        # real diagnosed problem -- with a single whole-vector
        # normalization, each gap's contribution is weighted by its raw
        # dimension count (window_size - g), which is LARGEST for the
        # SMALLEST gap, and on real autocorrelated wind-speed data gap=1
        # (51% of dims) was also the WORST single-gap tau estimator
        # (mean|diff|=0.196) -- but equal-weighting was verified to make
        # the combined estimate WORSE (0.145-0.256 vs 0.115-0.067,
        # depending on min_gap), because it gives the LOWEST-capacity gap
        # (near window_size, as few as 1 sample) the SAME vote as
        # high-capacity gaps, amplifying that gap's estimator variance
        # instead of averaging it out. Dimension-proportional weighting is
        # closer to inverse-variance weighting for gaps of comparable
        # quality -- the real fix for gap=1's BIAS (not variance) is
        # excluding it via min_gap, and for the near-window_size gaps'
        # VARIANCE is capping max_gap to keep a capacity floor -- both
        # handled in multiscale_gaps(), not here. See docs/
        # implementation_log.md's 2026-07-29(f) entry.
        # (2026-07-30) L2 norm of a concatenation is the Pythagorean sum of
        # each part's own norm (sum((a,b))^2 = sum(a^2) + sum(b^2)) -- so
        # the WHOLE-vector norm needed here can be built from each gap's
        # already-incrementally-tracked sum_sq() (O(1) each) instead of a
        # second O(output_dim) reduction pass over the concatenated array
        # (np.linalg.norm(raw, axis=1) previously). raw itself must still
        # be materialized once (O(output_dim), unavoidable -- the LSH index
        # needs the actual values), but that's now the only full-dimension
        # pass in this method.
        ring_buffers = [self._ring_buffers[int(g)] for g in self.gaps]
        parts = [rb.view() for rb in ring_buffers]
        raw = np.concatenate(parts, axis=1)
        total_sum_sq = np.sum([rb.sum_sq() for rb in ring_buffers], axis=0)
        norms = np.sqrt(np.maximum(total_sum_sq, 0.0)).reshape(-1, 1)
        out = np.zeros_like(raw)
        valid = (norms[:, 0] > 0.0) & np.isfinite(norms[:, 0])
        if np.any(valid):
            out[valid] = raw[valid] / norms[valid]
        return out

    def gap_vectors(self):
        """Per-gap counterpart to vectors(), for the multi-gap union-
        retrieval tier-1 (see ConcordanceMultiGapIndex below). Returns a
        list of (n_series, window_size-gap) arrays, one per configured
        gap, each independently L2-normalized PER GAP -- unlike vectors()'s
        single global normalize across the whole concatenation. No
        centering needed (sign values already have no meaningful mean to
        remove, same as vectors()'s own design).

        (2026-07-31) Direct structural analog of DistanceCorrSketchState.
        channel_vectors(): concatenating all gaps into one globally-
        normalized vector dilutes whichever single gap actually carries
        the real signal for a given pair, especially for true positives
        near corr_threshold at small window_size (verified directly --
        see ConcordanceMultiGapIndex's own docstring and docs/
        implementation_log.md's 2026-07-31 entry).
        """
        out = []
        for g in self.gaps:
            rb = self._ring_buffers[int(g)]
            raw = rb.view()
            norms = rb.norm_of_raw().reshape(-1, 1)
            normed = np.zeros_like(raw)
            valid = (norms[:, 0] > 0.0) & np.isfinite(norms[:, 0])
            if np.any(valid):
                normed[valid] = raw[valid] / norms[valid]
            out.append(normed)
        return out


def derive_concordance_multichannel_gamma_from_corr_threshold(corr_threshold, n_channels=None):
    """Rough, EMPIRICALLY-fit default for ConcordanceMultiGapIndex's
    per-gap retrieval threshold (the cosine-similarity bar applied to
    EACH usable gap's own index, analogous to distance_corr_sketch's
    derive_multichannel_gamma_from_corr_threshold).

    (2026-07-31) Calibrated at ONE config only (real data,
    fr_wind_direction_121_1, window_size=256, M=24, corr_threshold=0.5,
    spearman): best-single-USABLE-gap (capacity >= window_size//4,
    excluding the noisy low-capacity gap) |cosine| for true positives
    (|spearman rho|>=corr_threshold) has 10th percentile ~0.416 (mean
    0.688, min 0.335); independent pairs (|rho|<0.2) have 95th percentile
    ~0.223 (mean 0.103) -- a clean, non-overlapping gap. gamma=0.28 sits
    roughly midway between the two, not at either edge. Disclosed,
    unverified assumption: whether/how this should scale with
    corr_threshold (weaker true positives at a lower threshold plausibly
    have a weaker per-gap signal too, the same concern
    derive_gate_tau_from_corr_threshold's own window-size scaling
    addressed for distance_corr_sketch) has NOT been independently
    checked -- flat until verified otherwise, mirroring
    derive_multichannel_gamma_from_corr_threshold's own disclosed
    limitation. Always overridable directly.

    `n_channels` (the number of gaps that actually survive
    ConcordanceMultiGapIndex's own min_channel_capacity filter) matters: a
    real regression was found and fixed via direct benchmarking, not
    assumed -- at window_size>=1024, multiscale_gaps' own n_gaps floor
    (see multiscale_gaps' docstring) leaves exactly ONE usable gap, and a
    "max over 1 term" union is just that one term's own cosine --
    mathematically the SAME statistic the single-vector
    incremental_concordance backend already uses (ConcordanceSketchState.
    vectors(), n_gaps=1 case), not the different "max over several gaps"
    order statistic the 0.28 constant above was calibrated for. Applying
    0.28 there regardless REGRESSED recall relative to the (already-
    tuned) single-vector backend: 88.8% vs 95.7% at window_size=1024,
    84.7% vs 100% at window_size=2048 (real-data benchmark, corr_
    threshold=0.5). Fixed by delegating to derive_concordance_candidate_
    tau (the single-vector backend's own default) whenever n_channels<=1,
    reusing its already-verified calibration instead of a mismatched one.
    """
    threshold = max(0.0, min(1.0, float(corr_threshold)))
    if n_channels is not None and int(n_channels) <= 1:
        return derive_concordance_candidate_tau(threshold)
    return 0.28


class ConcordanceMultiGapIndex:
    """Per-gap union-based tier-1 retrieval for incremental_concordance,
    an alternative to the single combined-vector cosine retrieval used by
    ConcordanceSketchState.vectors() + one SignLSHBandIndex.

    (2026-07-31) Direct structural analog of
    DistanceCorrSketchMultiChannelIndex (distance_corr_sketch.py),
    applied to concordance's fixed-gap representation instead of
    distance_corr_sketch's RFF channels. Motivated by a real, measured
    recall gap: at window_size=256 (small windows, where multiscale_gaps
    picks n_gaps=4 rather than the n_gaps=1 floor that kicks in at
    window_size>=1024), true positives near corr_threshold have a weak
    COMBINED-vector cosine (mean 0.152, with 35.7% actually NEGATIVE) --
    concatenating all gaps into one globally-normalized vector dilutes
    whichever single gap actually carries the real signal for a given
    pair, exactly the diagnosis that motivated distance_corr_sketch's own
    multichannel redesign.

    Verified directly (docs/implementation_log.md's 2026-07-31 entry)
    that the best SINGLE-gap |cosine| alignment is dramatically stronger
    -- but ONLY once the lowest-capacity gap is EXCLUDED from the union:
    including it, independent pairs' best-of-all-gaps cosine also spikes
    high by chance (95th percentile 0.641, overlapping true positives'
    10th percentile 0.416), because that gap has as few as ~16 samples
    and taking a max over gaps lets its own high sampling variance leak
    into the "best-gap" signal for BOTH true positives and independent
    pairs equally. Excluding gaps below `min_channel_capacity` (default
    window_size // 4) fixes this: true positives' 10th percentile becomes
    0.416, independent pairs' 95th percentile becomes 0.223 -- clean
    separation, no overlap.

    Design: maintain one SignLSHBandIndex per USABLE gap (capacity >=
    min_channel_capacity), each over that gap's own L2-normalized sign
    vector (ConcordanceSketchState.gap_vectors(), NOT the single combined
    vectors()). A candidate pair is proposed if ANY single usable gap's
    index retrieves it (a union across gaps), not requiring the whole
    concatenated vector to align. At window_size>=1024, where n_gaps
    already floors to 1, this degenerates to exactly ONE index --
    functionally identical to the existing single-vector representation
    there, so this redesign only changes behavior where multiple gaps
    actually exist (small windows); it is not expected to regress the
    already-good window_size>=1024 recall (verified, not just assumed --
    see the same log entry).

    Reuses SignLSHBandIndex (candidate_kernels.pyx) unchanged, once per
    usable gap -- no new Cython code needed, mirroring
    DistanceCorrSketchMultiChannelIndex's own precedent exactly.
    """

    def __init__(
        self,
        gaps,
        window_size,
        min_channel_capacity=None,
        n_lagged_windows=3,
        n_bands=64,
        target_occupancy=10.0,
        band_seed=8161,
        apply_dot_filter=True,
        index_cls=None,
        apply_hamming_filter=True,
        hamming_max_frac=0.40,
    ):
        from candidate_kernels import SignLSHBandIndex, HammingExactIndex

        self.window_size = max(1, int(window_size))
        all_gaps = [int(g) for g in gaps]
        floor = (
            int(min_channel_capacity)
            if min_channel_capacity is not None
            else max(1, self.window_size // 4)
        )
        usable = [g for g in all_gaps if (self.window_size - g) >= floor]
        if not usable:
            # Never leave the union empty -- fall back to the single
            # smallest (highest-capacity) gap rather than silently
            # retrieving nothing for every pair.
            usable = [min(all_gaps)]
        self.gaps = usable
        self.index_cls = index_cls if index_cls is not None else SignLSHBandIndex
        # Large prime offset per gap so each gap's random band projections
        # are independent, not shared/correlated across gaps -- mirrors
        # DistanceCorrSketchMultiChannelIndex's own per-channel band_seed
        # offset exactly. Inert for HammingExactIndex (no bands, nothing
        # to seed independently -- its retrieval is exact, not
        # probabilistic).
        # (2026-07-31, Phase 2) Generalized to accept an alternate
        # index_cls (e.g. HammingExactIndex), mirroring
        # DistanceCorrSketchMultiChannelIndex's own generalization exactly
        # -- see that class's docstring for the verified API-compatibility
        # details this relies on.
        if self.index_cls is HammingExactIndex:
            self.indices = {
                g: HammingExactIndex(
                    n_vectors=self.window_size - g,
                    initial_capacity=1024,
                    apply_dot_filter=bool(apply_dot_filter),
                )
                for g in self.gaps
            }
        else:
            self.indices = {
                g: self.index_cls(
                    n_vectors=self.window_size - g,
                    initial_capacity=1024,
                    n_lagged_windows=int(n_lagged_windows),
                    n_bands=int(n_bands),
                    target_occupancy=float(target_occupancy),
                    apply_dot_filter=bool(apply_dot_filter),
                    band_seed=int(band_seed) + 7919 * i,
                    # (2026-07-31, touched_frac investigation) Previously
                    # never passed at all -- silently fell back to
                    # SignLSHBandIndex's own raw Cython default (False),
                    # UNLIKE single-vector concordance's shared Candidates-
                    # level index, which correctly inherits CorrTrack's
                    # own project-wide default (True). Confirmed via direct
                    # A/B (forcing candidate_apply_hamming_filter=False on
                    # the single-vector path made its touched_candidates
                    # count jump to EXACTLY match multichannel's, 2402==
                    # 2402, at window_size=4096) that this single missing
                    # parameter fully explained the touched_frac gap
                    # between the two representations -- recall was
                    # unaffected either way (100% both), confirming this
                    # was a pure efficiency gap, not a correctness one. See
                    # docs/implementation_log.md's 2026-07-31 entry.
                    apply_hamming_filter=bool(apply_hamming_filter),
                    hamming_max_frac=(None if hamming_max_frac is None else float(hamming_max_frac)),
                )
                for i, g in enumerate(self.gaps)
            }
        # (2026-07-31) window_idx must be a GLOBALLY unique, monotonically
        # growing identifier ACROSS EVERY STEP for each individual index
        # -- same requirement as DistanceCorrSketchMultiChannelIndex's own
        # counter (confirmed there by reading SignLSHBandIndex.insert_many
        # directly: it indexes straight into per-entry metadata arrays).
        # One shared counter is fine here too, since each gap's index only
        # needs uniqueness within itself, not across different gaps'
        # indices.
        self._next_window_idx = 0

    def notify_expected_n_series(self, m):
        # (2026-07-31, Phase 2) HammingExactIndex has no band-sizing
        # concept at all -- only call this on index classes that expose it.
        for idx in self.indices.values():
            if hasattr(idx, "notify_expected_n_series"):
                idx.notify_expected_n_series(int(m))

    def insert_and_query(
        self, gap_vectors_by_gap, sid_idx, time_idx, window_size_arr, sid_rank, gamma, tau, neg_corr
    ):
        """gap_vectors_by_gap: list aligned with ConcordanceSketchState.gaps
        (as returned by gap_vectors()), each a (n_new, window_size-gap)
        array -- one row per newly-arrived (series, window) entry this
        step, in the SAME order as sid_idx/time_idx/window_size_arr/
        sid_rank. Inserts each usable gap's new entries, then immediately
        queries each gap's own just-inserted entries against that gap's
        alive index, unioning + deduping the resulting (s1_idx, s2_idx,
        t1, t2, w) rows across all usable gaps. Returns a numeric-rows
        array ready to feed straight into CorrTrack._candidate_numeric_rows.
        """
        sid_idx = np.ascontiguousarray(sid_idx, dtype=np.int64)
        time_idx = np.ascontiguousarray(time_idx, dtype=np.int64)
        window_size_arr = np.ascontiguousarray(window_size_arr, dtype=np.int64)
        sid_rank = np.ascontiguousarray(sid_rank, dtype=np.int64)
        n_new = sid_idx.shape[0]
        if n_new == 0:
            return np.empty((0, 5), dtype=np.int64)
        window_idx = np.arange(self._next_window_idx, self._next_window_idx + n_new, dtype=np.int64)
        self._next_window_idx += n_new

        all_rows = []
        for pos, g in enumerate(self.gaps):
            idx = self.indices[g]
            vecs = np.ascontiguousarray(gap_vectors_by_gap[pos], dtype=np.float64)
            entry_ids = idx.insert_many(None, window_idx, vecs, sid_idx, time_idx, window_size_arr, sid_rank)
            entry_ids = np.ascontiguousarray(entry_ids, dtype=np.int64)
            if entry_ids.size == 0:
                continue
            if neg_corr:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            rows = np.asarray(rows, dtype=np.int64)
            if rows.size:
                all_rows.append(rows.reshape((-1, 5)))
        if not all_rows:
            return np.empty((0, 5), dtype=np.int64)
        combined = np.vstack(all_rows)
        return np.unique(combined, axis=0)

    def drop_before_time(self, min_valid_time):
        for idx in self.indices.values():
            idx.drop_before_time(int(min_valid_time))
