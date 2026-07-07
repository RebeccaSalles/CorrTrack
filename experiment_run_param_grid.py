"""Run-level configuration holding CorrTrack hyper-parameter grids."""

PARAM_GRID = {
    "n_vectors": [8, 16, 32, 64],
    # In cosine mode, gamma is the active sketch threshold. cell_size is kept
    # as a legacy stretch placeholder for L2-mode compatibility.
    "cell_size": [1.0],
    "preprocess": [False],
    "nodes": [0],
    "seed": [2468],
    "seed_toggle": [1357],
    "grid_dimension": [8],
    "sketch_norm": ["mean_l2"],
    # (2026-07-07) "instinct" added so the experimental approximate graph
    # backend can actually be exercised through the hyperopt/comparison
    # pipeline -- opt-in only (non-recall-guaranteed correctness contract).
    # See docs/implementation_log.md, "InstinctIndex: experimental
    # approximate graph backend" + the 2026-07-07 scale-test follow-up.
    # IMPORTANT correction to the original small-scale finding: at small
    # scale (~80-400 live windows) instinct was slower than sorted_arrays_bs
    # at every setting tried. At realistic scale matching real n_lags/
    # window_step (~3300 concurrently live windows: 300 series x
    # (168/16+1) lagged windows), it was FASTER (2.1x-2.7x) -- but only once
    # candidate_instinct_entry_points was scaled up from the small
    # defaults (8-64) tried earlier to 256-512. Root cause: entry_points
    # (the graph's traversal seed set, refreshed "keep first N then
    # round-robin overwrite") was simply too small relative to the number
    # of distinct correlated clusters/topics in the data for the
    # best-first search to ever reach every cluster, *regardless* of
    # ef_search budget -- raising ef_search alone (128->512) barely moved
    # recall (0.825->0.917) while erasing the speed advantage entirely, but
    # raising entry_points alone (24->512) reached recall==1.0 (exact match
    # vs. sorted_arrays_bs, not just close) at *better* speed than smaller
    # entry_points settings. candidate_ann_m/candidate_ann_z below are
    # dead/no-op for sorted_arrays_bs and bptree; they only take effect for
    # "instinct" (reused as max_degree/ef_insert). This grid's values are
    # NOT necessarily right for the human's real data -- the right
    # entry_points scale depends on how many distinct correlated
    # clusters/topics their data actually has, which this synthetic
    # benchmark's 30-cluster structure doesn't necessarily match.
    "candidate_backend": ["sorted_arrays_bs"], #"sorted_arrays_bs", "bptree", "instinct"
    "candidate_ann_m": [24],
    "candidate_ann_z": [96],
    "candidate_ann_ef": [128],
    "candidate_instinct_query_mode": ["threshold"], #"threshold", "topk", "hybrid"
    "candidate_instinct_top_k": [256],
    "candidate_instinct_min_candidates": [64],
    "candidate_instinct_entry_points": [1024,2048],
    "candidate_similarity": ["cosine"], #"l2"|"cosine"
    "candidate_cosine_threshold": [0.55, 0.6, 0.7],
    "candidate_bucket_width": [None],
    "candidate_block_size_steps": [8],
    "candidate_block_index_dims": [4,8,16], #4,8,16
    "candidate_key_mode": ["first"], #"first", "random_sign", "sampled_sketch"
    "candidate_key_seed": [2468],
    # (2026-07-06) Part 1: cone/angular block-level upper-bound pruning,
    # candidate_backend="sorted_arrays_bs" + cosine only (bptree ignores this
    # entirely -- half the sweep below is a no-op for it, doubling hyperopt
    # runtime for that half; accepted since the human explicitly asked for
    # this to be swept). Only benefits data with real angular clustering
    # tighter than candidate_cosine_threshold; verified safe (never changes
    # which pairs are found) either way. Watch proxy_search_blocks_pruned_by_ub
    # in the optim CSV to see whether it actually fires on this dataset.
    # candidate_bound_dims/candidate_bound_dim_selection/enable_row_ub_pruning
    # are deliberately NOT swept here -- row-level pruning showed ~0
    # wall-clock benefit in testing (see docs/implementation_log.md), so
    # sweeping it would only add hyperopt runtime for no expected signal.
    #
    # (2026-07-06) IMPORTANT correction to the note above: enable_block_ub_pruning
    # alone does nothing on real streaming data -- blocks were found to be
    # formed by pure arrival order, unrelated to angular similarity, so the
    # cone bound was essentially always loose (0 blocks pruned for every
    # hyperparameter combination, confirmed even with real 0.99-correlated
    # data). block_similarity_assignment (online "leader" clustering at
    # block-close time) is *required* for enable_block_ub_pruning to have any
    # chance of firing. Swept here so real runs can verify it, but this
    # codebase's own benchmarking found it's NOT a confirmed net wall-clock
    # win even once genuinely firing (real pruning, real dot_checks
    # reduction, but ~1.2x-1.5x *slower* wall-clock at n_vectors up to 64 --
    # the range this grid's n_vectors sweeps; results were mixed/inconclusive
    # at much higher n_vectors, 128 vs 256 disagreed). Watch
    # candidate_search_blocks_pruned_by_ub AND the actual runtime column
    # together, not just the former -- see docs/implementation_log.md,
    # "block-level cone pruning: root cause found... then a full flat-array
    # rewrite of the block storage".
    "enable_block_ub_pruning": [True],
    "block_similarity_assignment": [True],
    "max_open_blocks": [4,8,16], #4,8,16
    "hybrid_validation": [False],
    "hybrid_validation_min_repeat_rate": [0.25],
    "hybrid_validation_disable_rate": [0.125],
    "hybrid_validation_ema_alpha": [0.25],
    "hybrid_validation_min_candidates": [256],
}
