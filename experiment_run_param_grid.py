"""Run-level configuration holding CorrTrack hyper-parameter grids."""

PARAM_GRID = {
    # =========================================================================
    # Data representation
    # =========================================================================
    # -- General / shared --
    "n_vectors": [8, 16, 32, 64],
    "preprocess": [False],
    "seed": [2468],
    "seed_toggle": [1357],
    # "auto" (default) picks per validation_metric, matching this grid's own
    # candidate_backend/validation_metric choices below; set explicitly
    # (values: "auto", "raw", "sketch_proj", "sketch_concordance",
    # "sketch_multichannel") to override.
    "data_representation": ["auto"],

    # -- sketch_concordance (validation_metric in {"spearman", "kendall"}) --
    # concordance_n_gaps=[None] (default) auto-derives from
    # concordance_target_dim so output_dim stays roughly bounded across
    # window sizes instead of scaling linearly with window_size.
    "concordance_n_gaps": [None],
    "concordance_target_dim": [738],
    # gap=1 is both the highest-weight AND (on autocorrelated real data)
    # worst tau estimator; see concordance_sketch.py's multiscale_gaps
    # docstring.
    "concordance_min_gap": [8],
    "concordance_min_capacity": [16],
    # None auto-derives from corr_threshold (a flat constant calibrated at
    # ONE config only -- window_size=256, corr_threshold=0.5 -- see
    # concordance_sketch.py's own disclosed, unverified assumption about
    # how this should scale with corr_threshold). Sweep explicitly at any
    # other corr_threshold rather than trusting the default.
    "concordance_multichannel_gamma": [None],

    # -- sketch_multichannel (requires validation_metric="dist_corr") --
    # K is a real, swept hyperparameter -- K=8 is a verified sweet spot
    # (reliable, low-variance, cheaper than naive exact dCor); see
    # distance_corr_sketch.py.
    "distance_corr_sketch_k": [8],
    "distance_corr_sketch_freq_low": [0.1],
    "distance_corr_sketch_freq_high": [10.0],
    "distance_corr_sketch_freq_seed": [42],
    # [None] auto-derives from corr_threshold via derive_multichannel_
    # gamma_from_corr_threshold (calibrated only at corr_threshold=0.7 --
    # see distance_corr_sketch.py). Real-data recall/precision/touched_frac
    # at this default: ~96%/100%/~25% of all pairs (M=36/72/121).
    "distance_corr_sketch_multichannel_gamma": [None],
    # Independent tier-2 K^2 gate toggle -- default True; set False to
    # isolate whether tier-1 alone, or the gate alone, does the filtering.
    # Not swept in this production grid.
    "distance_corr_sketch_apply_tier2_gate": [True],

    # =========================================================================
    # Candidate search
    # =========================================================================
    # -- General / shared --
    # "candidate_backend" is the search/index mechanism ("auto"/"lsh_approx"/
    # "hamming_exact"/"brute_force"). "lsh_approx" is this project's
    # established default. "hamming_exact" is a known, disclosed alternative
    # at FIXED n_vectors=64 (beats lsh_approx there) but is NOT safe to sweep
    # with a small n_vectors floor -- its fixed-margin auto-threshold
    # under-recalls at small n_vectors (see docs/implementation_log.md's
    # 2026-07-24(c) entry).
    "candidate_backend": ["lsh_approx"],
    # The retrieval gate is set as a MARGIN below corr_threshold, not an absolute cosine
    # value, so this grid stays correct at any corr_threshold instead of silently pinning
    # one gamma. Resolved to candidate_cosine_threshold = max(0, corr_threshold - offset)
    # inside CorrTrack.__init__. 0.20 is the measured recall-plateau entry on real dense
    # data (see experiment_lsh_cost_sweep.THEORETICAL_FALLBACK_GAMMA_OFFSET and
    # docs/implementation_log.md 2026-09-10 (r)). Set an absolute
    # "candidate_cosine_threshold" here instead to override.
    "candidate_cosine_threshold_offset": [0.20],

    # -- lsh_approx backend --
    # (2026-08-30) n_bands = ceil(tolerance * b_min) is computed directly in
    # CorrTrack/SignLSHBandIndex from the closed-form LSH-banding recall
    # formula (candidate_kernels.pyx's _n_bands_tolerance) -- see
    # docs/implementation_log.md's 2026-08-30 entries for the original
    # derivation.
    #
    # (2026-09-03) No longer swept here: CorrTrack's own default for
    # candidate_lsh_n_bands_tolerance is now AUTO (tolerance=1.0 against
    # the overlap-corrected b_min formula -- see candidate_kernels.pyx's
    # _corrected_b_min derivation and docs/implementation_log.md's
    # 2026-09-03 entry), which fixed a real, verified bias the old formula
    # had (systematically overestimated recall by +7.0 points on average).
    # Sweeping {1.0, 1.5, 2.0} as a hedge against that bias is no longer
    # needed -- there's nothing left to tune to reach target_recall, so
    # this key is omitted entirely (falls through to CorrTrack's own
    # default rather than being pinned to one grid value here).
    #
    # SignLSHBandIndex's band_width auto-sizing target -- kept at the
    # default here (not swept in this production grid).
    "candidate_lsh_target_occupancy": [3.0],
    # SignLSHBandIndex's cheap full-vector sign-Hamming pre-filter, gating
    # the real dot product. Default on (recall-safe, real ~9-36% cut in
    # dot-product work) -- not swept as a toggle. max_frac at its validated
    # default.
    "candidate_apply_hamming_filter": [True],
    "candidate_hamming_filter_max_frac": [None,0.40],
    # SignLSHBandIndex's per-query candidate examination budget cap
    # (0=unlimited) -- no safe non-zero default is known (band iteration
    # carries no similarity ranking); kept off, not swept.
    "candidate_lsh_max_candidates_per_query": [0],
    "candidate_apply_dot_gamma_filter": [True],

    # -- hamming_exact backend --
    "candidate_hamming_threshold": [None],

    # =========================================================================
    # Validation
    # =========================================================================
    # -- General / shared --
    # "pearson" kept as the single default -- swapping to spearman/kendall/
    # dist_corr changes the experiment's entire semantics (not just a
    # candidate-search tuning knob), so it's set explicitly per-experiment.
    "validation_metric": ["pearson"],
    # Optional dev validation path: maintains repeated candidate relations
    # in a Cython cache and turns on incremental validation only when the
    # repeated-pair rate is high enough. Default off (workload-sensitive).
    "hybrid_validation": [False],
    "hybrid_validation_min_repeat_rate": [0.25],
    "hybrid_validation_disable_rate": [0.125],
    "hybrid_validation_ema_alpha": [0.25],
    "hybrid_validation_min_candidates": [256],

    # -- dist_corr --
    "dist_corr_algorithm": ["naive"],

    # -- spearman / kendall --
    # Xiao (2017) online validator -- opt-in, APPROXIMATE alternative to the
    # exact per-row scipy path. Kept off here; see docs/implementation_log.md's
    # 2026-07-29(a)/(b) entries for the measured accuracy tradeoffs before
    # enabling for a real run.
    "validation_incremental_approx": [False],
    "validation_incremental_m1": [None],
    "validation_incremental_m2": [None],
    "validation_incremental_max_age_steps": [64],
    "validation_incremental_cutpoint_refresh_threshold": [0.5],

    # =========================================================================
    # Execution
    # =========================================================================
    "nodes": [0],
}
