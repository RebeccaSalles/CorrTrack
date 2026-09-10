"""Incremental candidate-search representation + validation-time refinement
gate for distance-correlation screening.

(2026-07-30) Built in direct response to "can't I use my incremental sketch
+ nonstationary normalization to search for distance correlations?" Answered
no for a SINGLE linear sketch or a SINGLE sign-comparison sketch (the
Kendall/concordance trick): distance correlation is a quadratic form over
the FULL pairwise distance matrix within a window (a U-statistic), not
reducible to one fixed linear functional or one sign-comparison vector the
way Pearson/Kendall are.

The real, grounded avenue found instead: distance covariance is a special
case of HSIC (Sejdinovic, Sriperumbudur, Gretton & Fukumizu 2013,
"Equivalence of distance-based and RKHS-based statistics in hypothesis
testing"), and HSIC has a standard random-Fourier-features (RFF)
approximation (in the spirit of Rahimi & Recht 2007). Concretely: for a
shared set of K random frequencies w_1..w_K, each series X gets 2K derived
"channels" -- cos(w_k * X_t) and sin(w_k * X_t) for every window timepoint
t. Each channel is itself just another length-window_size series, so the
EXISTING incremental-sketch architecture (append-new/drop-expired ring
buffers) applies directly, with no new incremental machinery invented.

Two representations are built from these channels, verified separately
against this project's own already-checked naive `_distance_correlation_1d`
(library_corrtrack_parallel.py) before writing any incremental machinery,
per this project's "verify before trusting" discipline:

1. TIER 1 (`DistanceCorrSketchState.vectors()`): concatenate all 2K
   per-channel-mean-centered channels into one vector per series, L2-
   normalize ONCE. Cheap (one dot product per candidate touch, same cost
   shape as every other candidate_backend), feeds the EXISTING
   `lsh_sign_dot` bucket-retrieval machinery unchanged. Verified reliable
   but not on its own sufficient (Spearman rank-correlation with true dCor
   only ~0.71-0.79 on real/synthetic data, and unreliable specifically for
   periodic/sinusoidal relationships at some K) -- used only as a coarse,
   sub-linear pre-filter, not the final screening decision.

2. TIER 2 (`distance_corr_sketch_proxy`): the full K^2 grid of channel-pair
   cosine similarities between the tier-1 SURVIVORS' raw channels,
   aggregated via sqrt(mean of squares) -- the direct RFF-HSIC estimator.
   Verified MUCH more reliable: Spearman 0.96-0.98 across linear,
   monotonic-nonlinear, non-monotonic (quadratic), and periodic (sinusoidal)
   relationships on BOTH synthetic and real (wind_speed) data, stable
   across 8 different random frequency-set seeds at K=8 (std <= 0.017 per
   relationship type). Real-data recall/precision at a working threshold
   (dCor >= 0.5 ground truth): 91.2% recall / 85.1% precision while
   touching only 22.6% of all pairs (vs. 100% for brute force) -- a
   genuine, measured candidate-search result, not just a rank-correlation
   number. This tier is NOT indexable via the existing single-vector LSH
   machinery (it inherently needs BOTH series' full channel sets, not a
   fixed-size per-series sketch) -- it runs as a validation-time
   REFINEMENT GATE on tier-1's (already sub-linear) survivors, computed
   fresh from the raw window slices already available at that point (no
   incremental state needed there -- only tier 1 needs to be incremental,
   since tier 2 only ever runs on the few candidates tier 1 already
   narrowed down).

A real, decisive finding on K (number of frequencies), found only because
the human pushed back on an initially-premature "not beneficial"
conclusion at K=24 (48 channels): reliability does NOT require large K.
At K=24 the tier-2 proxy cost 4.4-5.4x MORE than the naive exact dCor
computation it's meant to screen before -- a genuine "correct but not
beneficial" result. At K=8, reliability stays excellent (0.95-0.98, low
variance) while wall-time drops to ~0.77x of naive exact dCor -- already
cheaper even as an unoptimized Python prototype. Profiling why: at K=8,
58% of the (already-favorable) cost is a pure-Python loop making 256
separate small `np.dot()` calls (2K x 2K = 16x16) -- exactly the pattern
this project's existing Cython kernels (e.g. SignLSHBandIndex's inner
comparison loops) exist to eliminate via one fused compiled loop. K=8 is
this module's default; K remains a real, swept hyperparameter (like
`concordance_n_gaps`/`ordinal_num_pairs`), not a fixed constant -- K=2-4
are cheaper still but measurably less robust specifically for periodic
relationships (K=2: sinusoidal Spearman mean 0.87, min 0.78 across seeds;
K=8: mean 0.95, min 0.93).

Honest disclosure on the frequency-sampling distribution: the theoretically
"correct" measure for exact (population-level) distance covariance's own
characteristic-function integral involves a specific, delicate near-origin
regularization that was not independently re-derived here (a genuine risk
of subtly-wrong math from memory alone). Rather than risk an unverified
theoretical derivation, frequencies are drawn from a simple, standard
log-spaced-scale x standard-normal construction (mirroring
`concordance_sketch.py`'s own multiscale-gap philosophy) and the resulting
proxy's RELIABILITY was verified empirically end-to-end against the exact
naive estimator -- the same "verify the statistic directly, don't trust
the derivation alone" discipline used for the Kendall/concordance sketch.
"""

import math

import numpy as np

from concordance_sketch import _RingGapBuffer

try:
    from candidate_kernels import distance_corr_sketch_proxy_cy as _distance_corr_sketch_proxy_cy
except ImportError:
    _distance_corr_sketch_proxy_cy = None


def multiscale_freqs(K, low=0.1, high=10.0, seed=42):
    """K random frequencies, log-spaced base scales x standard-normal sign/
    magnitude -- mirrors `concordance_sketch.multiscale_gaps`'s own
    "hedge across scales, don't assume one" philosophy applied to
    frequency (rather than gap) selection. Fixed once per
    `DistanceCorrSketchState`/gate call via `seed`, shared across ALL
    series (like the base sketch's own random projection directions), not
    re-drawn per pair.
    """
    K = max(1, int(K))
    rng = np.random.default_rng(seed)
    scales = np.geomspace(max(1e-6, float(low)), max(float(low) + 1e-6, float(high)), K)
    return rng.normal(size=K) * scales


def _channels_for_window(window_data, freqs):
    # window_data: (n_series, w). Returns a list of 2K arrays, each
    # (n_series, w) -- cos/sin of each frequency applied to every series'
    # raw values, NOT yet mean-centered (centering happens per-channel at
    # vectors()/proxy time, since the relevant mean is the CURRENT window's,
    # which shifts as the window slides -- exactly the same nonstationary-
    # normalization concern the base Pearson sketch already handles).
    chans = []
    for w in freqs:
        chans.append(np.cos(w * window_data))
        chans.append(np.sin(w * window_data))
    return chans


class DistanceCorrSketchState:
    """Tier-1 incremental representation: 2K ring buffers per series (one
    cos and one sin channel per frequency), updated incrementally as the
    window slides -- each step computes only the `window_step` newly-
    arrived channel values and appends them (reusing `concordance_sketch`'s
    already bit-exact-verified `_RingGapBuffer`, not a new ring-buffer
    implementation), falling back to a full rebuild whenever the shape
    changes. `vectors()` returns the cheap, LSH-indexable concatenated
    representation (tier 1 only) -- the reliable tier-2 refinement
    (`distance_corr_sketch_proxy`, below) is computed separately, fresh,
    at validation time on tier-1's survivors.
    """

    def __init__(self, window_size, K=8, freq_low=0.1, freq_high=10.0, freq_seed=42):
        self.window_size = int(window_size)
        self.K = max(1, int(K))
        self.freq_low = float(freq_low)
        self.freq_high = float(freq_high)
        self.freq_seed = int(freq_seed)
        self.freqs = multiscale_freqs(self.K, low=self.freq_low, high=self.freq_high, seed=self.freq_seed)
        self.output_dim = int(2 * self.K * self.window_size)
        self._ring_buffers = None  # list of 2K _RingGapBuffer, one per channel
        self._n_series = None

    def _fresh_buffers(self, window_data):
        chans = _channels_for_window(window_data, self.freqs)
        ring_buffers = []
        for chan in chans:
            rb = _RingGapBuffer(capacity=self.window_size)
            rb.reset(chan)
            ring_buffers.append(rb)
        self._ring_buffers = ring_buffers
        self._n_series = window_data.shape[0]

    def update(self, window_data, window_step):
        n_series, w = window_data.shape
        window_step = int(window_step)
        if (
            self._ring_buffers is None
            or self._n_series != n_series
            or w != self.window_size
            or window_step <= 0
            or window_step >= self.window_size
        ):
            self._fresh_buffers(window_data)
            return

        new_slice = window_data[:, w - window_step:]
        idx = 0
        for wf in self.freqs:
            new_cos = np.cos(wf * new_slice)
            new_sin = np.sin(wf * new_slice)
            self._ring_buffers[idx].append(new_cos)
            self._ring_buffers[idx + 1].append(new_sin)
            idx += 2

    def vectors(self):
        # Tier 1: per-channel mean-center (a real bug was caught and fixed
        # here during verification -- centering the WHOLE concatenated
        # vector once, instead of each channel separately, created a
        # spurious ~0.38 similarity between fully INDEPENDENT series
        # regardless of sample size; per-channel centering fixed it, mean
        # shrinking properly with n as expected), concatenate, ONE global
        # L2-normalize (verified design, matches the "diagonal proxy"
        # tested throughout this investigation).
        #
        # (2026-07-30) mean()/norm_of_centered() are O(1) per-series
        # accessors on _RingGapBuffer's own incrementally-tracked running
        # sum/sum_sq -- replaces the two O(window_size) reduction passes
        # (.mean(axis=1), then np.linalg.norm on the centered array) that
        # used to run here every step. The GLOBAL norm across all 2K
        # channels is the Pythagorean sum of each channel's own centered
        # norm (||concat(a,b)||^2 = ||a||^2 + ||b||^2), so it's built from
        # 2K O(1) lookups, not a second full-dimension scan. Materializing
        # `raw` itself (the actual centered values) remains one unavoidable
        # O(output_dim) pass -- the LSH index needs the real vector, not
        # just its norm.
        parts = []
        total_var = 0.0
        for rb in self._ring_buffers:
            mean = rb.mean()
            parts.append(rb.view() - mean[:, None])
            total_var = total_var + (rb.sum_sq() - rb.capacity * mean * mean)
        raw = np.concatenate(parts, axis=1)
        norms = np.sqrt(np.maximum(total_var, 0.0)).reshape(-1, 1)
        out = np.zeros_like(raw)
        valid = (norms[:, 0] > 0.0) & np.isfinite(norms[:, 0])
        if np.any(valid):
            out[valid] = raw[valid] / norms[valid]
        return out

    def channel_vectors(self):
        """(2026-07-30) Per-channel counterpart to vectors(), for the
        multi-channel union-retrieval tier-1 (see
        DistanceCorrSketchMultiChannelIndex below). Returns a list of 2K
        (n_series, window_size) arrays, each independently mean-centered
        AND L2-normalized PER CHANNEL (unlike vectors()'s single global
        normalize) -- verified directly (docs/implementation_log.md's
        2026-07-30(m) entry) that per-channel diagonal cosine alignment for
        genuine true positives is dramatically stronger (median ~0.9996,
        10th percentile ~0.756 on real data at dCor>=0.7) than the
        concatenated vector's own cosine (~0.06-0.13 mean), because
        concatenating all 2K channels into one vector before normalizing
        dilutes whichever single channel carries the real signal.

        Uses the same O(1) mean()/norm_of_centered() accessors as
        vectors() -- see that method's own comment for why this avoids the
        two O(window_size) reduction passes .mean(axis=1)/np.linalg.norm()
        previously ran here, per channel, every step.
        """
        out = []
        for rb in self._ring_buffers:
            mean = rb.mean()
            centered = rb.view() - mean[:, None]
            norms = rb.norm_of_centered().reshape(-1, 1)
            normed = np.zeros_like(centered)
            valid = (norms[:, 0] > 0.0) & np.isfinite(norms[:, 0])
            if np.any(valid):
                normed[valid] = centered[valid] / norms[valid]
            out.append(normed)
        return out


def distance_corr_sketch_proxy(x, y, freqs):
    """Tier-2 refinement gate: the full K^2 (well, (2K)x(2K)) grid of
    per-channel-pair cosine similarities between x's and y's RFF channels,
    aggregated via sqrt(mean of squares) -- the direct RFF-HSIC estimator
    for distance-covariance-style general dependence. Computed fresh from
    raw x/y (the window slices already sliced out by the caller at
    validation time) -- no incremental state needed here, since this only
    ever runs on the few candidates tier 1 has already narrowed down, not
    on every possible pair.

    (2026-07-30) Cython-only, by explicit instruction ("all gates for all
    backends should be Cython, I want no Python"): the Python prototype's
    own profiling found 58% of its cost was pure-Python-loop overhead from
    256 separate small np.dot() calls at K=8 -- ported to
    candidate_kernels.distance_corr_sketch_proxy_cy (one fused nogil loop,
    verified bit-identical to ~1e-14 against the retired Python version
    across 500 random trials spanning linear/monotonic-nonlinear/non-
    monotonic/independent relationships). No Python fallback -- raises if
    the compiled extension is unavailable, matching this project's existing
    strict pattern for candidate_backend="flat"/"lsh_sign_dot".

    Verified (see module docstring): Spearman rank-correlation with true
    (naive, exact) distance correlation of 0.96-0.98 across linear,
    monotonic-nonlinear, non-monotonic, and periodic relationships, on
    both synthetic and real data, at K=8. Real-data recall/precision at a
    dCor>=0.5 ground-truth threshold: 91.2%/85.1% while touching only
    22.6% of all pairs.
    """
    if _distance_corr_sketch_proxy_cy is None:
        raise RuntimeError(
            "distance_corr_sketch_proxy requires the compiled candidate_kernels "
            "Cython extension (distance_corr_sketch_proxy_cy); the Python "
            "fallback has been removed."
        )
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    return float(_distance_corr_sketch_proxy_cy(x, y, freqs))


def derive_candidate_tau_from_corr_threshold(corr_threshold):
    """Rough, EMPIRICALLY-fit default for tier 1's OWN retrieval threshold
    (`distance_corr_sketch_candidate_tau`) -- the cosine-similarity bar on
    the cheap, CONCATENATED (not the reliable K^2) representation.

    (2026-07-30) A real bug found via a full real-data recall/precision
    sweep (not caught by the earlier, narrower checks): the original
    default (half of corr_threshold, e.g. 0.25 at corr_threshold=0.5) was
    calibrated by analogy to Pearson/ordinal/concordance's OWN unit-norm
    sketches, where the candidate-search threshold and corr_threshold
    genuinely share a scale. That analogy does NOT hold here -- verified
    directly on real data: true-positive pairs' tier-1 cosine similarity
    has mean/median only ~0.06-0.13 even at dCor>=0.6, and the 10th
    percentile needed for >=90% recall is close to ZERO (-0.003 to 0.011
    across dcor_tau=0.3-0.6) -- a completely different, much smaller
    natural scale than corr_threshold itself. The old default filtered out
    94-99% of true positives before tier 2 ever got a chance to see them
    (measured recall 16-26% end-to-end, despite 100% precision -- the
    bottleneck was entirely tier 1, confirmed since distance_corr_sketch_
    gate_rejected stayed 0 throughout). Fixed to a small, deliberately loose
    constant with a comfortable safety margin above the measured 10th
    percentiles (tier 2's gate is what protects precision, not this
    threshold) -- NOT independently re-verified for window-size-scaling
    the way the gate_tau fix was (a disclosed, unverified assumption, not
    a silently-hidden gap). Always overridable directly via candidate_tau.
    """
    threshold = max(0.0, min(1.0, float(corr_threshold)))
    return max(0.01, 0.02 + 0.02 * threshold)


def derive_gate_tau_from_corr_threshold(corr_threshold, window_size=256):
    """Rough, EMPIRICALLY-fit (not theoretically derived or guaranteed)
    default mapping from the project's usual dCor `corr_threshold`
    (ground-truth dCor value considered "correlated") to a reasonable
    default `distance_corr_sketch_gate_tau` (the tier-2 proxy threshold).

    (2026-07-30) Originally calibrated ONLY at window_size=256 (real-data
    sweep, K=8, fr_wind_direction_121_1/wind_speed):
        dcor_tau -> best proxy threshold for >=90% recall
        0.3 -> 0.114, 0.4 -> 0.140, 0.5 -> 0.162, 0.6 -> 0.179
    giving gate_tau_256 ~= 0.095 + 0.12 * corr_threshold. Caught by direct
    testing (not assumed), a real gap: this statistic's noise floor for
    INDEPENDENT pairs scales with window_size, and a threshold calibrated
    at w=256 alone is far too lenient at smaller windows -- verified
    directly, independent-pair mean proxy * sqrt(window_size) is
    essentially CONSTANT (~0.98-1.01) across w=32..1024 (a clean 1/sqrt(w)
    noise-floor scaling law, consistent with ordinary sampling-noise
    intuition for this kind of statistic). Fixed by rescaling the w=256
    calibration by sqrt(256/window_size) rather than using it verbatim at
    every window size. Still a documented STARTING POINT, not a proof --
    the same honesty this project already applies to HammingExactIndex's
    own disclosed-as-heuristic SimHash margin. Always overridable directly
    via an explicit gate_tau.
    """
    threshold = max(0.0, min(1.0, float(corr_threshold)))
    gate_tau_at_256 = max(0.02, 0.095 + 0.12 * threshold)
    w = max(1, int(window_size))
    return gate_tau_at_256 * math.sqrt(256.0 / w)


def derive_multichannel_gamma_from_corr_threshold(corr_threshold):
    """Rough, EMPIRICALLY-fit (NOT independently re-verified across
    corr_threshold the way the K=8 default itself was) starting point for
    DistanceCorrSketchMultiChannelIndex's per-channel cosine gamma.

    (2026-07-30) Calibrated at ONE config only (real data,
    fr_wind_direction_121_1, window_size=256, K=8, corr_threshold=0.7):
    true positives' best single-channel (same-frequency, diagonal) cosine
    alignment has 10th percentile ~0.756 (mean 0.94, median 0.9996);
    independent pairs' 90th percentile is only ~0.196 -- essentially no
    overlap. gamma=0.5 (comfortably inside that gap, not at either edge)
    gave 98.9% recall at only ~27.6% of the true total-pairs count
    touched; looser gamma (down to 0.05) barely improved recall further
    (99.4%) while touching far more (~3.6x more pairs) -- 0.5 is close to
    the actual measured optimum at THIS config, not an arbitrary round
    number. Disclosed, unverified assumption: whether/how this should
    scale with corr_threshold (weaker true positives at a lower threshold
    plausibly have a weaker per-channel signal too, the same concern
    derive_gate_tau_from_corr_threshold's own window-size scaling
    addressed for the K^2 gate) has NOT been independently checked the way
    that scaling law was -- this is a flat, threshold-independent value
    until verified otherwise. Always overridable directly.
    """
    return 0.5


class DistanceCorrSketchMultiChannelIndex:
    """Per-channel union-based tier-1 retrieval, an alternative to the
    single-vector concatenated-cosine retrieval used by
    candidate_backend="distance_corr_sketch" itself.

    (2026-07-30) Direct response to "I want smart filtering with no all
    pair enumeration": tier-1's own parameter space (target_occupancy,
    n_bands, candidate_tau) was confirmed EXHAUSTED -- no sub-linear
    operating point recovers meaningfully more recall without touched_frac
    degenerating to 1.0 (see docs/implementation_log.md's 2026-07-30(l)
    entry). Root cause: concatenating all 2K per-frequency channels into
    ONE vector and taking ONE global cosine dilutes whichever single
    channel actually carries the real signal for a given pair. Verified
    directly (2026-07-30(m) entry) that the best SINGLE-channel (same
    frequency, diagonal only) cosine alignment between two genuinely
    dependent series is dramatically stronger and cleanly separated from
    independent pairs: real-data mean/median ~0.94/0.9996 (10th percentile
    0.756) for dCor>=0.7 pairs vs. mean/median ~0.12/0.10 (90th percentile
    0.196) for dCor<0.2 pairs -- essentially no overlap, versus the
    concatenated vector's own ~0.06-0.13 mean signal.

    Design: maintain 2K SEPARATE SignLSHBandIndex instances (one per
    channel, each over the plain window_size-dimensional channel vector,
    L2-normalized PER CHANNEL via DistanceCorrSketchState.channel_vectors()
    -- NOT the single concatenated-and-globally-normalized vectors()). A
    candidate pair is proposed if ANY single channel's index retrieves it
    (a union across channels), not requiring the whole concatenated vector
    to align -- exactly the property the signal-strength check above says
    should work far better than one combined index. Each channel gets an
    independently-seeded band structure (band_seed offset by a large prime
    per channel) so the 2K indices are not simply computing the same bands
    on correlated data.

    Reuses SignLSHBandIndex (candidate_kernels.pyx) unchanged, 2K times --
    no new Cython code needed here, since the existing kernel already
    returns full (s1_idx, s2_idx, t1, t2, w) numeric rows directly from a
    query, not opaque entry ids requiring a separate reverse-lookup.

    (2026-07-31, Phase 2) Generalized to accept an alternate `index_cls`
    (e.g. HammingExactIndex) in place of the default SignLSHBandIndex --
    verified directly (not assumed) that both classes expose an API-
    identical surface for everything this wrapper needs (insert_many,
    find_pair_rows_full_cosine[_signed], drop_before_time) except
    HammingExactIndex has no notify_expected_n_series (no band-sizing
    concept, since it has no bands at all) -- handled by only calling it
    when the underlying index class actually exposes it. HammingExactIndex
    also has no band_seed/n_bands/target_occupancy/n_lagged_windows
    concept (no bands, no probabilistic retrieval), so those constructor
    kwargs are only passed to SignLSHBandIndex.
    """

    def __init__(
        self,
        n_channels,
        window_size,
        n_lagged_windows=3,
        n_bands=64,
        target_occupancy=10.0,
        band_seed=4242,
        apply_dot_filter=True,
        index_cls=None,
    ):
        from candidate_kernels import SignLSHBandIndex, HammingExactIndex

        self.n_channels = max(1, int(n_channels))
        self.window_size = max(1, int(window_size))
        self.n_lagged_windows = max(1, int(n_lagged_windows))
        self.index_cls = index_cls if index_cls is not None else SignLSHBandIndex
        # Large prime offset per channel so each channel's random band
        # projections are independent, not shared/correlated across
        # channels -- the whole point of a union-across-channels retrieval
        # is that each channel gets its OWN independent chance to catch a
        # pair, not 2K copies of the same hash structure. Inert for
        # HammingExactIndex (no bands, nothing to seed independently --
        # its retrieval is exact, not probabilistic).
        if self.index_cls is HammingExactIndex:
            self.indices = [
                HammingExactIndex(
                    n_vectors=self.window_size,
                    initial_capacity=1024,
                    apply_dot_filter=bool(apply_dot_filter),
                )
                for ch in range(self.n_channels)
            ]
        else:
            self.indices = [
                self.index_cls(
                    n_vectors=self.window_size,
                    initial_capacity=1024,
                    n_lagged_windows=self.n_lagged_windows,
                    n_bands=int(n_bands),
                    target_occupancy=float(target_occupancy),
                    apply_dot_filter=bool(apply_dot_filter),
                    band_seed=int(band_seed) + 7919 * ch,
                )
                for ch in range(self.n_channels)
        ]
        # (2026-07-30) window_idx must be a GLOBALLY unique, monotonically
        # growing identifier across every step -- confirmed by reading
        # SignLSHBandIndex.insert_many directly: it uses window_idx as a
        # direct array index into its own per-entry metadata cache
        # (win_sid_idx/win_time/win_size), so reusing small values across
        # steps would silently overwrite older, still-alive entries'
        # metadata. Matches Candidates._get_or_create_window_idx's own
        # ever-growing-counter convention exactly, just tracked locally
        # here instead of shared across a whole Candidates instance.
        self._next_window_idx = 0

    def notify_expected_n_series(self, m):
        # (2026-07-31, Phase 2) HammingExactIndex has no band-sizing
        # concept at all (no bands, exact Hamming scan instead) -- only
        # call this on index classes that actually expose it.
        for idx in self.indices:
            if hasattr(idx, "notify_expected_n_series"):
                idx.notify_expected_n_series(int(m))

    def insert_and_query(
        self, channel_vectors, sid_idx, time_idx, window_size_arr, sid_rank, gamma, tau, neg_corr
    ):
        """channel_vectors: list of n_channels arrays, each (n_new,
        window_size) -- one row per newly-arrived (series, window) entry
        this step, in the SAME order as sid_idx/time_idx/window_size_arr/
        sid_rank. Inserts each channel's new entries, then immediately
        queries each channel's own just-inserted entries (as the "recent"
        set) against that channel's alive index, unioning + deduping the
        resulting (s1_idx, s2_idx, t1, t2, w) rows across all channels.
        Returns a numeric-rows array ready to feed straight into
        CorrTrack._candidate_numeric_rows.
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
        for ch, idx in enumerate(self.indices):
            vecs = np.ascontiguousarray(channel_vectors[ch], dtype=np.float64)
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
        for idx in self.indices:
            idx.drop_before_time(int(min_valid_time))
