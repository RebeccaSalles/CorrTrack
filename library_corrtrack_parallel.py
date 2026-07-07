import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import bisect
import csv
import time
import datetime
import os
import ast
import json
import heapq
import pandas as pd
from itertools import repeat
from collections import defaultdict
import math
import warnings
from scipy.stats import norm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Iterator, Optional, Sequence
import traceback

CSV_DELIMITER = ";"

try:
    from dask import delayed, compute
    from dask.threaded import get as dask_threaded_get
    _HAS_DASK = True
except ImportError:  # pragma: no cover
    delayed = compute = dask_threaded_get = None
    _HAS_DASK = False

try:
    import candidate_kernels as _cand_kernels
    _cand_kernel_file = str(getattr(_cand_kernels, "__file__", ""))
    if not (
        _cand_kernel_file.endswith(".so")
        or _cand_kernel_file.endswith(".pyd")
        or ".so" in os.path.basename(_cand_kernel_file)
    ):
        raise ImportError("compiled candidate_kernels extension is required")
    _cy_find_candidate_pairs = _cand_kernels.find_candidate_pairs
    _cy_find_candidate_pairs_unique = getattr(_cand_kernels, "find_candidate_pairs_unique", None)
    _cy_find_candidate_pairs_full = getattr(_cand_kernels, "find_candidate_pairs_full", None)
    _cy_find_candidate_pairs_full_parallel = getattr(_cand_kernels, "find_candidate_pairs_full_parallel", None)
    _cy_find_candidate_pairs_with_dist = getattr(_cand_kernels, "find_candidate_pairs_with_dist", None)
    _cy_find_candidate_pairs_full_with_dist = getattr(_cand_kernels, "find_candidate_pairs_full_with_dist", None)
    _cy_find_candidate_pairs_full_with_dist_parallel = getattr(_cand_kernels, "find_candidate_pairs_full_with_dist_parallel", None)
    _cy_find_candidate_pairs_bucketed = getattr(_cand_kernels, "find_candidate_pairs_bucketed", None)
    _cy_find_candidate_pairs_bucketed_with_dist = getattr(_cand_kernels, "find_candidate_pairs_bucketed_with_dist", None)
    _cy_find_candidate_pairs_bucketed_full = getattr(_cand_kernels, "find_candidate_pairs_bucketed_full", None)
    _cy_find_candidate_pairs_bucketed_full_with_dist = getattr(_cand_kernels, "find_candidate_pairs_bucketed_full_with_dist", None)
    _cy_balanced_index_cls = getattr(_cand_kernels, "BalancedIndex", None)
    _cy_blocked_lazy_index_cls = getattr(_cand_kernels, "BlockedLazyIndex", None)
    _cy_bucketed_multi_index_cls = getattr(_cand_kernels, "BucketedMultiIndex", None)
    _cy_instinct_index_cls = getattr(_cand_kernels, "InstinctIndex", None)
    _cy_enumerate_candidate_rows = getattr(_cand_kernels, "enumerate_candidate_rows", None)
    _cy_fast_corr_and_dist = _cand_kernels.fast_corr_and_dist
    _cy_validate_corr_batch = _cand_kernels.validate_corr_batch
    _cy_validate_corr_rows = getattr(_cand_kernels, "validate_corr_rows", None)
    _cy_hybrid_cache_cls = getattr(_cand_kernels, "HybridValidationCache", None)
    _HAS_CYTHON_KERNELS = True
except Exception:  # pragma: no cover
    _cy_find_candidate_pairs = None
    _cy_find_candidate_pairs_unique = None
    _cy_find_candidate_pairs_full = None
    _cy_find_candidate_pairs_full_parallel = None
    _cy_find_candidate_pairs_with_dist = None
    _cy_find_candidate_pairs_full_with_dist = None
    _cy_find_candidate_pairs_full_with_dist_parallel = None
    _cy_find_candidate_pairs_bucketed = None
    _cy_find_candidate_pairs_bucketed_with_dist = None
    _cy_find_candidate_pairs_bucketed_full = None
    _cy_find_candidate_pairs_bucketed_full_with_dist = None
    _cy_balanced_index_cls = None
    _cy_blocked_lazy_index_cls = None
    _cy_bucketed_multi_index_cls = None
    _cy_instinct_index_cls = None
    _cy_enumerate_candidate_rows = None
    _cy_fast_corr_and_dist = None
    _cy_validate_corr_batch = None
    _cy_validate_corr_rows = None
    _cy_hybrid_cache_cls = None
    _HAS_CYTHON_KERNELS = False

try:
    from sketch_kernels import compute_series_dots as _cy_compute_series_dots
    from sketch_kernels import build_sketch_matrix as _cy_build_sketch_matrix
    from sketch_kernels import apply_orth_and_normalize as _cy_apply_orth_and_normalize
    from sketch_kernels import incremental_combine_and_normalize as _cy_incremental_combine_and_normalize
    from sketch_kernels import compute_constant_flags as _cy_compute_constant_flags
except Exception:  # pragma: no cover
    _cy_compute_series_dots = None
    _cy_build_sketch_matrix = None
    _cy_apply_orth_and_normalize = None
    _cy_incremental_combine_and_normalize = None
    _cy_compute_constant_flags = None

try:
    from partition_kernels import build_partitions as _cy_build_partitions
    from partition_kernels import build_partition_values as _cy_build_partition_values
except Exception:  # pragma: no cover
    _cy_build_partitions = None
    _cy_build_partition_values = None

try:
    from monitor_kernels import NumericMonitorState as _cy_numeric_monitor_state_cls
except Exception:  # pragma: no cover
    _cy_numeric_monitor_state_cls = None


RUN_RESULT_COLUMNS: Sequence[str] = (
    "dataset_id",
    "run_kind",
    "mode",
    "alg",
    "optim",
    "baseline_mode",
    "n_ts",
    "n_w",
    "total_w",
    "mem_w",
    "nodes",
    "exec_mode",
    "parallel_sketch",
    "parallel_candidates",
    "parallel_validation",
    "workers_total",
    "workers_sketch",
    "workers_candidates",
    "workers_validation",
    "window_size",
    "window_step",
    "basic_window",
    "n_lags",
    "seed",
    "seed_toggle",
    "preprocess",
    "sketch_norm",
    "candidate_backend",
    "candidate_bucket_width",
    "candidate_block_size_steps",
    "candidate_block_index_dims",
    "candidate_bound_dims",
    "candidate_bound_dim_selection",
    "enable_block_ub_pruning",
    "enable_row_ub_pruning",
    "block_similarity_assignment",
    "max_open_blocks",
    "candidate_instinct_query_mode",
    "candidate_instinct_top_k",
    "candidate_instinct_min_candidates",
    "candidate_instinct_entry_points",
    "candidate_similarity",
    "candidate_cosine_threshold",
    "candidate_parallel_mode",
    "candidate_key_mode",
    "candidate_key_seed",
    "candidate_lsh_radius",
    "candidate_ann_m",
    "candidate_ann_z",
    "candidate_ann_ef",
    "hybrid_validation",
    "hybrid_validation_min_repeat_rate",
    "hybrid_validation_disable_rate",
    "hybrid_validation_ema_alpha",
    "hybrid_validation_min_candidates",
    "hybrid_validation_attempts",
    "hybrid_validation_hits",
    "hybrid_validation_steps_active",
    "numeric_rows",
    "corr_threshold",
    "grid_max",
    "cell_stretch",
    "cell_size",
    "n_vectors",
    "grid_dimension",
    "freq_threshold",
    "sk_time",
    "cand_time",
    "val_time",
    "monit_time",
    "runtime",
    "artifact_time",
    "correlated",
    "tested",
    "total_candidates",
    "candidate_search_index_candidates",
    "candidate_search_valid_index_candidates",
    "candidate_search_unique_index_candidates",
    "candidate_search_duplicate_index_candidates",
    "candidate_search_unique_pre_dot_pairs",
    "candidate_search_duplicate_pre_dot_pairs",
    "candidate_search_after_coord",
    "candidate_search_partial_checks",
    "candidate_search_after_partial",
    "candidate_search_after_similarity",
    "candidate_search_dot_checks",
    "candidate_search_distance_checks",
    "candidate_search_blocks_visited",
    "candidate_search_blocks_pruned_by_ub",
    "candidate_search_rows_in_surviving_blocks",
    "candidate_search_dot_checks_saved_by_row_ub",
    "candidate_search_instinct_visited_nodes",
    "candidate_search_instinct_visited_live_nodes",
    "candidate_search_instinct_dead_nodes_skipped",
    "candidate_search_instinct_edges_scanned",
    "candidate_search_instinct_candidates_returned",
    "candidate_search_instinct_threshold_candidates",
    "candidate_search_instinct_topk_candidates",
    "candidate_search_instinct_queries_with_too_few_live_nodes",
    "candidate_search_instinct_query_time",
    "candidate_search_instinct_insert_time",
    "candidate_search_instinct_num_nodes_total",
    "candidate_search_instinct_num_nodes_alive",
    "candidate_search_instinct_best_score_seen",
    "candidate_search_instinct_mean_score_returned",
    "candidate_search_instinct_dead_node_ratio",
    "pair_min_dist",
    "artifact_path",
)

OPTIM_RESULT_COLUMNS: Sequence[str] = (
    "dataset_id",
    "alg",
    "hyperopt_strategy",
    "tuning_mode",
    "sampling_design_id",
    "sampling_profile",
    "sampling_profile_notes",
    "sampling_requested_block_q",
    "sampling_requested_n_blocks",
    "sampling_requested_placement_mode",
    "sampling_requested_random_seed",
    "sampling_requested_allow_block_overlap",
    "sampling_n_sampled_blocks",
    "sampling_block_windows",
    "sampling_block_span",
    "sampling_total_sampled_timestamps",
    "sampling_coverage_train",
    "sampling_warning",
    "n_ts",
    "n_w",
    "total_w",
    "mem_w",
    "nodes",
    "exec_mode",
    "parallel_sketch",
    "parallel_candidates",
    "parallel_validation",
    "workers_total",
    "workers_sketch",
    "workers_candidates",
    "workers_validation",
    "window_size",
    "window_step",
    "basic_window",
    "n_lags",
    "seed",
    "seed_toggle",
    "preprocess",
    "sketch_norm",
    "candidate_backend",
    "candidate_bucket_width",
    "candidate_block_size_steps",
    "candidate_block_index_dims",
    "candidate_bound_dims",
    "candidate_bound_dim_selection",
    "enable_block_ub_pruning",
    "enable_row_ub_pruning",
    "block_similarity_assignment",
    "max_open_blocks",
    "candidate_instinct_query_mode",
    "candidate_instinct_top_k",
    "candidate_instinct_min_candidates",
    "candidate_instinct_entry_points",
    "candidate_similarity",
    "candidate_cosine_threshold",
    "candidate_parallel_mode",
    "candidate_key_mode",
    "candidate_key_seed",
    "candidate_lsh_radius",
    "candidate_ann_m",
    "candidate_ann_z",
    "candidate_ann_ef",
    "hybrid_validation",
    "hybrid_validation_min_repeat_rate",
    "hybrid_validation_disable_rate",
    "hybrid_validation_ema_alpha",
    "hybrid_validation_min_candidates",
    "hybrid_validation_attempts",
    "hybrid_validation_hits",
    "hybrid_validation_steps_active",
    "numeric_rows",
    "corr_threshold",
    "grid_max",
    "cell_stretch",
    "cell_size",
    "n_vectors",
    "grid_dimension",
    "freq_threshold",
    "cand_time_bf",
    "val_time_bf",
    "monit_time_bf",
    "runtime_bf",
    "artifact_time_bf",
    "sk_time",
    "cand_time",
    "val_time",
    "monit_time",
    "runtime",
    "optim_search_time",
    "optim_bootstrap_time",
    "optim_eval_time",
    "artifact_time",
    "speedup",
    "speedup_ceil",
    "rel_speedup_eff",
    "corr_w_bf",
    "corr_w",
    "tested_w_bf",
    "tested_w",
    "cand_w_bf",
    "cand_w",
    "candidate_search_index_candidates",
    "candidate_search_valid_index_candidates",
    "candidate_search_unique_index_candidates",
    "candidate_search_duplicate_index_candidates",
    "candidate_search_unique_pre_dot_pairs",
    "candidate_search_duplicate_pre_dot_pairs",
    "candidate_search_after_coord",
    "candidate_search_partial_checks",
    "candidate_search_after_partial",
    "candidate_search_after_similarity",
    "candidate_search_dot_checks",
    "candidate_search_distance_checks",
    "candidate_search_blocks_visited",
    "candidate_search_blocks_pruned_by_ub",
    "candidate_search_rows_in_surviving_blocks",
    "candidate_search_dot_checks_saved_by_row_ub",
    "candidate_search_instinct_visited_nodes",
    "candidate_search_instinct_visited_live_nodes",
    "candidate_search_instinct_dead_nodes_skipped",
    "candidate_search_instinct_edges_scanned",
    "candidate_search_instinct_candidates_returned",
    "candidate_search_instinct_threshold_candidates",
    "candidate_search_instinct_topk_candidates",
    "candidate_search_instinct_queries_with_too_few_live_nodes",
    "candidate_search_instinct_query_time",
    "candidate_search_instinct_insert_time",
    "candidate_search_instinct_num_nodes_total",
    "candidate_search_instinct_num_nodes_alive",
    "candidate_search_instinct_best_score_seen",
    "candidate_search_instinct_mean_score_returned",
    "candidate_search_instinct_dead_node_ratio",
    "corr_prop",
    "waste_val_bf",
    "waste_val",
    "rel_waste_red",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "precision_neg",
    "recall_neg",
    "f1_neg",
    "precision",
    "recall",
    "specificity",
    "recall_min",
    "f1",
    "aucroc",
    "pr_auc",
    "bootstrap_enabled",
    "bootstrap_repeats",
    "bootstrap_confidence_level",
    "bootstrap_min_gt_events",
    "bootstrap_block_summary_json",
    "bootstrap_gt_support",
    "bootstrap_recall_lb",
    "bootstrap_recall_med",
    "bootstrap_recall_ub",
    "bootstrap_precision_lb",
    "bootstrap_precision_med",
    "bootstrap_precision_ub",
    "bootstrap_specificity_lb",
    "bootstrap_specificity_med",
    "bootstrap_specificity_ub",
    "bootstrap_speedup_lb",
    "bootstrap_speedup_med",
    "bootstrap_speedup_ub",
    "bootstrap_recall_ci_width",
    "bootstrap_speedup_ci_width",
    "bootstrap_feasible",
    "bootstrap_underpowered",
    "bootstrap_selected",
    "bootstrap_clear_winner",
    "bootstrap_speedup_gap_lb",
    "bootstrap_speedup_gap_med",
    "bootstrap_sample_adequate",
    "bootstrap_warning",
    "proxy_anchor_count_requested",
    "proxy_anchor_count",
    "proxy_anchor_expansion_iteration",
    "proxy_anchor_expansion_reason",
    "proxy_anchor_seed",
    "proxy_max_pair_rows",
    "proxy_eval_mode",
    "proxy_distance_cache_used",
    "proxy_distance_cache_rows",
    "proxy_distance_cache_grids",
    "proxy_distance_prefilter_max_tau_sq",
    "proxy_distance_prefilter_total_entries",
    "proxy_distance_prefilter_entries",
    "proxy_distance_prefilter_fraction",
    "proxy_n_pairs",
    "proxy_n_gt",
    "proxy_candidate_rate",
    "proxy_search_index_candidates",
    "proxy_search_valid_index_candidates",
    "proxy_search_unique_index_candidates",
    "proxy_search_duplicate_index_candidates",
    "proxy_search_unique_pre_dot_pairs",
    "proxy_search_duplicate_pre_dot_pairs",
    "proxy_search_after_coord",
    "proxy_search_partial_checks",
    "proxy_search_after_partial",
    "proxy_search_after_dot",
    "proxy_search_dot_checks",
    "proxy_search_distance_checks",
    "proxy_search_similarity_work_rate",
    "proxy_search_index_rate",
    "proxy_search_dot_work_rate",
    "proxy_search_cascade_reject_rate",
    "proxy_search_objective_rate",
    "proxy_search_time_total",
    "proxy_timing_repeats",
    "proxy_bootstrap_enabled",
    "proxy_bootstrap_repeats",
    "proxy_bootstrap_confidence_level",
    "proxy_bootstrap_min_gt_events",
    "proxy_gt_support",
    "proxy_recall_lb",
    "proxy_recall_mean",
    "proxy_recall_med",
    "proxy_recall_ub",
    "proxy_precision_lb",
    "proxy_precision_mean",
    "proxy_precision_med",
    "proxy_precision_ub",
    "proxy_specificity_lb",
    "proxy_specificity_mean",
    "proxy_specificity_med",
    "proxy_specificity_ub",
    "proxy_candidate_rate_lb",
    "proxy_candidate_rate_mean",
    "proxy_candidate_rate_med",
    "proxy_candidate_rate_ub",
    "proxy_feasible",
    "proxy_underpowered",
    "proxy_selected",
    "proxy_sample_adequate",
    "proxy_warning",
    "status",
    "error",
)

COMPARISON_COLUMNS: Sequence[str] = (
    "dataset_id",
    "mode",
    "alg",
    "optim",
    "baseline_mode",
    "n_ts",
    "n_w",
    "total_w",
    "mem_w",
    "nodes",
    "exec_mode",
    "parallel_sketch",
    "parallel_candidates",
    "parallel_validation",
    "workers_total",
    "workers_sketch",
    "workers_candidates",
    "workers_validation",
    "window_size",
    "window_step",
    "basic_window",
    "n_lags",
    "seed",
    "seed_toggle",
    "preprocess",
    "sketch_norm",
    "candidate_backend",
    "candidate_bucket_width",
    "candidate_block_size_steps",
    "candidate_block_index_dims",
    "candidate_bound_dims",
    "candidate_bound_dim_selection",
    "enable_block_ub_pruning",
    "enable_row_ub_pruning",
    "block_similarity_assignment",
    "max_open_blocks",
    "candidate_instinct_query_mode",
    "candidate_instinct_top_k",
    "candidate_instinct_min_candidates",
    "candidate_instinct_entry_points",
    "candidate_similarity",
    "candidate_cosine_threshold",
    "candidate_parallel_mode",
    "candidate_key_mode",
    "candidate_key_seed",
    "candidate_lsh_radius",
    "candidate_ann_m",
    "candidate_ann_z",
    "candidate_ann_ef",
    "hybrid_validation",
    "hybrid_validation_min_repeat_rate",
    "hybrid_validation_disable_rate",
    "hybrid_validation_ema_alpha",
    "hybrid_validation_min_candidates",
    "hybrid_validation_attempts",
    "hybrid_validation_hits",
    "hybrid_validation_steps_active",
    "numeric_rows",
    "corr_threshold",
    "grid_max",
    "cell_stretch",
    "cell_size",
    "n_vectors",
    "grid_dimension",
    "freq_threshold",
    "cand_time_bf",
    "val_time_bf",
    "monit_time_bf",
    "runtime_bf",
    "artifact_time_bf",
    "sk_time",
    "cand_time",
    "val_time",
    "monit_time",
    "runtime",
    "artifact_time",
    "speedup",
    "speedup_ceil",
    "rel_speedup_eff",
    "corr_w_bf",
    "corr_w",
    "tested_w_bf",
    "tested_w",
    "cand_w_bf",
    "cand_w",
    "candidate_search_index_candidates",
    "candidate_search_valid_index_candidates",
    "candidate_search_unique_index_candidates",
    "candidate_search_duplicate_index_candidates",
    "candidate_search_unique_pre_dot_pairs",
    "candidate_search_duplicate_pre_dot_pairs",
    "candidate_search_after_coord",
    "candidate_search_partial_checks",
    "candidate_search_after_partial",
    "candidate_search_after_similarity",
    "candidate_search_dot_checks",
    "candidate_search_distance_checks",
    "candidate_search_blocks_visited",
    "candidate_search_blocks_pruned_by_ub",
    "candidate_search_rows_in_surviving_blocks",
    "candidate_search_dot_checks_saved_by_row_ub",
    "candidate_search_instinct_visited_nodes",
    "candidate_search_instinct_visited_live_nodes",
    "candidate_search_instinct_dead_nodes_skipped",
    "candidate_search_instinct_edges_scanned",
    "candidate_search_instinct_candidates_returned",
    "candidate_search_instinct_threshold_candidates",
    "candidate_search_instinct_topk_candidates",
    "candidate_search_instinct_queries_with_too_few_live_nodes",
    "candidate_search_instinct_query_time",
    "candidate_search_instinct_insert_time",
    "candidate_search_instinct_num_nodes_total",
    "candidate_search_instinct_num_nodes_alive",
    "candidate_search_instinct_best_score_seen",
    "candidate_search_instinct_mean_score_returned",
    "candidate_search_instinct_dead_node_ratio",
    "corr_prop",
    "waste_val_bf",
    "waste_val",
    "rel_waste_red",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "precision_neg",
    "recall_neg",
    "f1_neg",
    "precision",
    "recall",
    "specificity",
    "recall_min",
    "f1",
    "aucroc",
    "pr_auc",
    "maxlag_precision",
    "maxlag_recall",
    "maxlag_f1",
    "maxlag_diff_mean",
    "maxlag_diff_std",
)


_RAW_RUN_RESULT_COLUMNS = RUN_RESULT_COLUMNS
_RAW_OPTIM_RESULT_COLUMNS = OPTIM_RESULT_COLUMNS
_RAW_COMPARISON_COLUMNS = COMPARISON_COLUMNS

_FIXED_OR_OBSOLETE_OUTPUT_COLUMNS = {
    "candidate_parallel_mode",
    "freq_threshold",
    "numeric_rows",
    "optim_bootstrap_time",
    "proxy_eval_mode",
    # (2026-07-05) candidate-search metrics consolidation: these are pure
    # aliases of other still-exposed columns in every backend --
    # unique_index_candidates == valid_index_candidates always (both
    # increment together at the same site in every scan implementation);
    # after_coord == valid_index_candidates always; partial_checks ==
    # after_coord always (once VPTree's now-fixed per-node-visit overload of
    # partial_checks is excluded -- see docs/implementation_log.md). Dropping
    # them here (not by editing the column tuples directly) keeps
    # _RAW_*_COLUMNS/_filter_comparison_row's positional zip correct.
    "candidate_search_unique_index_candidates",
    "candidate_search_after_coord",
    "candidate_search_partial_checks",
    "proxy_search_unique_index_candidates",
    "proxy_search_after_coord",
    "proxy_search_partial_checks",
    # (2026-07-05) vptree/angular_lsh/dynamic_graph_ann backends removed at
    # the human's request (kept: sorted_arrays_bs, bptree). These params were
    # only ever read by those three dropped backends -- dropped from CSV
    # output the same safe way as above (constructors/record-builders still
    # accept and default them internally, unused in practice now that those
    # backends can no longer be selected -- see docs/implementation_log.md,
    # "backend removal: blocked_lazy/vptree/angular_lsh/dynamic_graph_ann").
    "candidate_lsh_radius",
    "candidate_ann_m",
    "candidate_ann_z",
    "candidate_ann_ef",
}


def _drop_fixed_or_obsolete_output_columns(columns: Sequence[str]) -> Sequence[str]:
    return tuple(
        col for col in columns
        if (
            col not in _FIXED_OR_OBSOLETE_OUTPUT_COLUMNS
            and not col.startswith("bootstrap_")
            and not col.startswith("proxy_bootstrap_")
        )
    )


RUN_RESULT_COLUMNS = _drop_fixed_or_obsolete_output_columns(RUN_RESULT_COLUMNS)
OPTIM_RESULT_COLUMNS = _drop_fixed_or_obsolete_output_columns(OPTIM_RESULT_COLUMNS)
COMPARISON_COLUMNS = _drop_fixed_or_obsolete_output_columns(COMPARISON_COLUMNS)


def _filter_comparison_row(row: Sequence) -> list:
    return [
        value for col, value in zip(_RAW_COMPARISON_COLUMNS, row)
        if col in COMPARISON_COLUMNS
    ]


class CSVStreamWriter:
    """Helper that appends rows to a CSV file while ensuring headers exist."""

    def __init__(self, path: str, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = tuple(columns)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._delimiter = CSV_DELIMITER
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh, delimiter=self._delimiter)
        self._writer.writerow(self.columns)
        self._fh.flush()

    def write_row(self, values: Sequence[object]) -> None:
        if len(values) != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} columns ({self.columns}), got {len(values)} items"
            )
        self._writer.writerow(values)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "CSVStreamWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        self.close()
        return None


def _row_from_mapping(columns: Sequence[str], data: dict) -> list:
    row = []
    for column in columns:
        value = data.get(column, "")
        if column in {"artifact_time", "artifact_time_bf"} and value in (None, ""):
            value = 0.0
        row.append(value)
    return row


def _remove_artifact_files(prefix: Optional[str]) -> None:
    if not prefix:
        return
    suffixes = (
        "_correlated.csv",
        "_neg_pairs.csv",
        "_candidates.csv",
        "_status.csv",
        "_anomalies.csv",
        "_max_lag_correlated.csv",
    )
    for suffix in suffixes:
        path = f"{prefix}{suffix}"
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    base_dir = os.path.dirname(os.path.abspath(prefix))
    base_name = os.path.basename(prefix)
    try:
        for name in os.listdir(base_dir):
            if not name.startswith(base_name + "_"):
                continue
            if ".chunk" not in name:
                continue
            path = os.path.join(base_dir, name)
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        pass

def _run_batch_static(func, batch):
    """Top-level batch helper to avoid bound-method pickling issues."""
    return [func(item) for item in batch]

def _detect_csv_delimiter(path: str, default: str = CSV_DELIMITER) -> str:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            header = ""
            for line in handle:
                header = line.strip()
                if header:
                    break
            if header:
                has_semicolon = ";" in header
                has_comma = "," in header
                if has_semicolon and not has_comma:
                    return ";"
                if has_comma and not has_semicolon:
                    return ","
            sample = header + handle.read(4096)
    except Exception:
        return default
    if not sample:
        return default
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        if dialect.delimiter:
            return dialect.delimiter
    except Exception:
        pass
    if ";" in sample and sample.count(";") >= sample.count(","):
        return ";"
    return default


def _correlated_codebook_path(csv_path: str) -> str:
    if csv_path.endswith("_correlated.csv"):
        return csv_path[: -len("_correlated.csv")] + "_correlated_codes.csv"
    return f"{csv_path}.codes.csv"


def _write_correlated_codebook(csv_path: str, code_map: dict) -> None:
    code_path = _correlated_codebook_path(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(code_path)), exist_ok=True)
    with open(code_path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=CSV_DELIMITER)
        writer.writerow(["code", "id"])
        for id_value, code in sorted(code_map.items(), key=lambda item: item[1]):
            writer.writerow([code, id_value])


def _load_correlated_codebook(csv_path: str) -> dict:
    code_path = _correlated_codebook_path(csv_path)
    if not os.path.exists(code_path):
        return {}
    delimiter = _detect_csv_delimiter(code_path, default=CSV_DELIMITER)
    codebook = {}
    try:
        with open(code_path, newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                code = row.get("code")
                id_value = row.get("id")
                if code not in (None, "") and id_value is not None:
                    codebook[str(code)] = id_value
    except OSError:
        return {}
    return codebook


def execute_corrtrack_pass(
    run_kind: str,
    dataset_id: str,
    corrtrack,
    data: np.ndarray,
    ids: Sequence[str],
    metadata: dict,
    corr_val: bool,
    recall_by_window: bool,
    artifact_prefix: Optional[str] = None,
    artifact_mode: str = "iterative",
    artifact_merge_mode: str = "merged",
    verbose: bool = False,
    testing: bool = False,
    monitor: bool = True,
    step_observer=None,
) -> tuple[dict, Optional[tuple], Optional[np.ndarray]]:
    """
    Execute a CorrTrack pass (brute-force or main) and collect runtime statistics.

    Returns:
        record: mapping aligned with RUN_RESULT_COLUMNS.
        runtime_parts: tuple(sketch, candidate, validation, monitor) when applicable.
        corr_flags: correlation indicators for downstream metrics.
    """
    if run_kind not in {"bf", "corrtrack"}:
        raise ValueError(f"Unsupported run_kind '{run_kind}'")

    corr_val = _coerce_to_bool(corr_val, default=True)
    recall_by_window = True
    monitor = _coerce_to_bool(monitor, default=True)
    if not corr_val:
        monitor = False
        if hasattr(corrtrack, "track_min_dist"):
            corrtrack.track_min_dist = False

    record = {key: None for key in RUN_RESULT_COLUMNS}
    record.update(metadata or {})
    record["dataset_id"] = dataset_id
    record["run_kind"] = run_kind
    record["artifact_path"] = artifact_prefix or ""
    record["baseline_mode"] = getattr(corrtrack, "baseline_mode", metadata.get("baseline_mode"))

    n_ts = data.shape[0] - 1
    window_size = corrtrack.window_size
    window_step = corrtrack.window_step
    n_w = int(np.floor((data.shape[1] - window_size) / window_step) + 1)
    total_w = n_ts * n_w

    record["n_ts"] = n_ts
    record["n_w"] = n_w
    record["total_w"] = total_w
    record["mem_w"] = None
    record["window_size"] = window_size
    record["window_step"] = window_step
    record["basic_window"] = corrtrack.basic_window
    record["n_lags"] = corrtrack.n_lags
    record["corr_threshold"] = corrtrack.corr_threshold
    record["grid_max"] = getattr(corrtrack, "grid_max", None)
    record["cell_stretch"] = getattr(corrtrack, "cell_stretch", None)
    record["cell_size"] = getattr(corrtrack, "cell_size", None)
    record["n_vectors"] = getattr(corrtrack, "n_vectors", None)
    record["grid_dimension"] = getattr(corrtrack, "grid_dimension", None)
    record["freq_threshold"] = getattr(corrtrack, "freq_threshold", None)
    record["candidate_backend"] = getattr(
        corrtrack,
        "candidate_backend_effective",
        getattr(corrtrack, "candidate_backend", None),
    )
    record["candidate_bucket_width"] = getattr(corrtrack, "candidate_bucket_width", None)
    record["candidate_block_size_steps"] = getattr(corrtrack, "candidate_block_size_steps", None)
    record["candidate_block_index_dims"] = getattr(corrtrack, "candidate_block_index_dims", None)
    # (2026-07-06) Part 1 -- see docs/implementation_log.md.
    record["candidate_bound_dims"] = getattr(corrtrack, "candidate_bound_dims", None)
    record["candidate_bound_dim_selection"] = getattr(corrtrack, "candidate_bound_dim_selection", None)
    record["enable_block_ub_pruning"] = getattr(corrtrack, "enable_block_ub_pruning", None)
    record["enable_row_ub_pruning"] = getattr(corrtrack, "enable_row_ub_pruning", None)
    record["block_similarity_assignment"] = getattr(corrtrack, "block_similarity_assignment", None)
    record["max_open_blocks"] = getattr(corrtrack, "max_open_blocks", None)
    record["candidate_instinct_query_mode"] = getattr(corrtrack, "candidate_instinct_query_mode", None)
    record["candidate_instinct_top_k"] = getattr(corrtrack, "candidate_instinct_top_k", None)
    record["candidate_instinct_min_candidates"] = getattr(corrtrack, "candidate_instinct_min_candidates", None)
    record["candidate_instinct_entry_points"] = getattr(corrtrack, "candidate_instinct_entry_points", None)
    record.update(_candidate_runtime_record_fields(corrtrack))
    record.update(_candidate_search_record_fields(corrtrack))
    record["candidate_parallel_mode"] = getattr(corrtrack, "candidate_parallel_mode", None)
    record["candidate_key_mode"] = getattr(corrtrack, "candidate_key_mode", None)
    record["candidate_key_seed"] = getattr(corrtrack, "candidate_key_seed", None)
    record["candidate_lsh_radius"] = getattr(corrtrack, "candidate_lsh_radius", None)
    record["candidate_ann_m"] = getattr(corrtrack, "candidate_ann_m", None)
    record["candidate_ann_z"] = getattr(corrtrack, "candidate_ann_z", None)
    record["candidate_ann_ef"] = getattr(corrtrack, "candidate_ann_ef", None)
    record["numeric_rows"] = getattr(corrtrack, "numeric_rows", None)
    record.update(_hybrid_validation_record_fields(corrtrack))
    record["exec_mode"] = getattr(corrtrack, "exec", None)
    record["parallel_sketch"] = getattr(corrtrack, "parallel_sketch", None)
    record["parallel_candidates"] = getattr(corrtrack, "parallel_candidates", None)
    record["parallel_validation"] = getattr(corrtrack, "parallel_validation", None)
    record["workers_total"] = getattr(corrtrack, "n_nodes", None)
    record["workers_sketch"] = getattr(corrtrack, "n_sketch_nodes", None)
    record["workers_candidates"] = getattr(corrtrack, "n_candidate_nodes", None)
    if getattr(corrtrack, "parallel_validation", False):
        record["workers_validation"] = getattr(corrtrack, "n_nodes", None)
    else:
        record["workers_validation"] = 1

    artifact_active = bool(artifact_prefix)
    artifact_buffer_max_rows = metadata.get("artifact_buffer_max_rows")
    if artifact_buffer_max_rows in (None, ""):
        artifact_buffer_max_rows = getattr(corrtrack, "_artifact_buffer_max_rows", 250000)
    artifact_policy = _resolve_artifact_save_policy(
        recall_by_window,
        metadata.get("save_only_required_artifacts"),
        metadata.get("save_maxlag_artifacts"),
    )

    artifact_mode_value = (artifact_mode or "iterative").lower()
    if artifact_mode_value not in ("iterative", "final", "buffered"):
        artifact_mode_value = "iterative"
    effective_artifact_mode = artifact_mode_value if artifact_active else "final"
    artifact_merge_mode_value = _resolve_artifact_merge_mode(
        metadata.get("artifact_merge_mode", artifact_merge_mode),
        default="merged",
    )

    corrtrack._configure_artifact_runtime(
        artifact_mode=effective_artifact_mode,
        artifact_buffer_max_rows=artifact_buffer_max_rows,
        artifact_merge_mode=artifact_merge_mode_value,
        recall_by_window=True,
        save_correlated=artifact_policy["save_correlated"],
        save_status=artifact_policy["save_status"],
        save_anomalies=artifact_policy["save_anomalies"],
        save_maxlag=artifact_policy["save_maxlag"],
    )
    corrtrack._step_observer_enabled = bool(step_observer)
    corrtrack._validated_step = {}
    corrtrack._online_window_metrics_only = bool(
        step_observer is not None
        and run_kind == "corrtrack"
        and recall_by_window
        and not corr_val
        and not monitor
        and not artifact_active
    )
    if artifact_active:
        base_dir = os.path.dirname(os.path.abspath(artifact_prefix))
        os.makedirs(base_dir, exist_ok=True)
        corrtrack._start_artifact_logging(artifact_prefix)

    artifact_per_iteration = artifact_active and effective_artifact_mode == "iterative"
    artifact_chunked = artifact_active and effective_artifact_mode in {"iterative", "buffered"}

    artifact_time_total = 0.0
    artifact_time_overlap = 0.0
    artifact_bookkeeping_before = getattr(corrtrack, "artifact_bookkeeping_time", 0.0)

    start_time = time.time()
    for start in range(0, data.shape[1] - window_step + 1, window_step):
        chunk = data[:, start : (start + window_step)]
        if run_kind == "bf":
            corrtrack.run_bf(chunk, ids, verbose=verbose, testing=testing, corr_val=True, monitor=monitor)
        else:
            corrtrack.run(chunk, ids, verbose=verbose, testing=testing, corr_val=corr_val, monitor=monitor)
        if step_observer is not None:
            _t0 = time.time()
            step_observer(corrtrack)
            elapsed = time.time() - _t0
            artifact_time_total += elapsed
            artifact_time_overlap += elapsed
        if artifact_active and artifact_chunked:
            _t0 = time.time()
            flushed = corrtrack._maybe_flush_artifact_buffers(force=artifact_per_iteration)
            elapsed = time.time() - _t0
            if flushed:
                artifact_time_total += elapsed
                artifact_time_overlap += elapsed
    end_time = time.time()

    if artifact_active and artifact_chunked:
        _t0 = time.time()
        corrtrack._finalize_buffered_artifacts()
        artifact_time_total += time.time() - _t0
    elif artifact_active:
        _t0 = time.time()
        if getattr(corrtrack, "_artifact_save_correlated", True):
            corrtrack._save_correlated(f"{artifact_prefix}_correlated.csv")
        if getattr(corrtrack, "_artifact_save_status", True):
            corrtrack._save_monitor_status(f"{artifact_prefix}_status.csv")
        if getattr(corrtrack, "_artifact_save_anomalies", True):
            corrtrack._save_anomalies(f"{artifact_prefix}_anomalies.csv")
        artifact_time_total += time.time() - _t0

    runtime = end_time - start_time
    runtime_parts = None
    if run_kind == "bf":
        record["sk_time"] = 0.0
        record["cand_time"] = getattr(corrtrack, "candidate_time", 0.0)
        record["val_time"] = getattr(corrtrack, "validation_time", 0.0)
        record["monit_time"] = getattr(corrtrack, "monitor_time", 0.0)
        runtime_parts = (
            0.0,
            record["cand_time"],
            record["val_time"],
            record["monit_time"],
        )
    else:
        runtime = max(runtime, 0.0)
        record["sk_time"] = getattr(corrtrack, "sketch_time", 0.0)
        record["cand_time"] = getattr(corrtrack, "candidate_time", 0.0)
        record["val_time"] = getattr(corrtrack, "validation_time", 0.0)
        record["monit_time"] = getattr(corrtrack, "monitor_time", 0.0)
        runtime_parts = (
            record["sk_time"],
            record["cand_time"],
            record["val_time"],
            record["monit_time"],
        )

    runtime = max(runtime - artifact_time_overlap, 0.0)

    if artifact_active and getattr(corrtrack, "_artifact_save_maxlag", True):
        _t0 = time.time()
        corrtrack._save_max_lag_correlated(f"{artifact_prefix}_max_lag_correlated.csv")
        artifact_time_total += time.time() - _t0

    artifact_bookkeeping_delta = max(
        getattr(corrtrack, "artifact_bookkeeping_time", 0.0) - artifact_bookkeeping_before,
        0.0,
    )
    artifact_time_total += artifact_bookkeeping_delta
    artifact_time_overlap += artifact_bookkeeping_delta
    runtime = max(runtime - artifact_bookkeeping_delta, 0.0)

    record["runtime"] = runtime
    record["artifact_time"] = artifact_time_total
    if hasattr(corrtrack, "n_lagged_windows"):
        record["mem_w"] = corrtrack.n_lagged_windows - 1
    record["correlated"] = getattr(corrtrack, "validated_candidates", 0)
    record["tested"] = getattr(corrtrack, "tested_candidates", 0)
    record["total_candidates"] = getattr(corrtrack, "total_candidates", 0)
    record.update(_candidate_search_record_fields(corrtrack))
    pair_min_dist = getattr(corrtrack, "pair_min_dist", None)
    record["pair_min_dist"] = str(pair_min_dist) if pair_min_dist is not None else ""
    record.update(_hybrid_validation_record_fields(corrtrack))

    corrtrack._step_observer_enabled = False
    corrtrack._online_window_metrics_only = False
    corrtrack._validated_step = {}

    if artifact_chunked:
        corr_flags = None
    elif recall_by_window:
        corr_flags = getattr(corrtrack, "correlated", None)
    else:
        corr_flags = corrtrack.get_correlation_flags(data.shape[1], 0)

    return record, runtime_parts, corr_flags


def run_and_log_bruteforce(
    dataset_id: str,
    data: np.ndarray,
    ids: Sequence[str],
    base_config: dict,
    output_csv: str,
    metadata: Optional[dict] = None,
    recall_by_window: bool = True,
    verbose: Optional[bool] = None,
    testing: Optional[bool] = None,
    artifact_prefix: Optional[str] = None,
) -> tuple[dict, tuple, Optional[np.ndarray]]:
    metadata = dict(metadata or {})
    baseline_mode = _resolve_baseline_mode(base_config.get("baseline_mode", "bruteforce"))
    metadata.setdefault("mode", "bf")
    metadata.setdefault("alg", metadata.get("alg", "bf"))
    metadata.setdefault("optim", metadata.get("optim", "baseline"))
    if baseline_mode != "bruteforce" and metadata.get("optim") in {"baseline", "bf_baseline"}:
        metadata["optim"] = f"{baseline_mode}_baseline"
    metadata.setdefault("baseline_mode", baseline_mode)
    metadata.setdefault("artifact_buffer_max_rows", base_config.get("artifact_buffer_max_rows"))
    metadata.setdefault("artifact_merge_mode", base_config.get("artifact_merge_mode"))
    metadata.setdefault(
        "save_only_required_artifacts",
        base_config.get("save_only_required_artifacts"),
    )
    metadata.setdefault("save_maxlag_artifacts", base_config.get("save_maxlag_artifacts"))

    artifact_mode = base_config.get("artifact_mode", "iterative")
    artifact_merge_mode = base_config.get("artifact_merge_mode", "merged")
    monitor = _coerce_to_bool(base_config.get("monitor", True), default=True)

    corrtrack = CorrTrack(
        window_size=base_config["window_size"],
        basic_window=base_config.get("basic_window"),
        window_step=base_config["window_step"],
        n_vectors=1,
        n_lags=base_config["n_lags"],
        grid_dimension=1,
        cell_size=1,
        seed=None,
        seed_toggle=None,
        corr_threshold=base_config["corr_threshold"],
        neg_corr=base_config.get("neg_corr", False),
        preprocess=False,
        exec=base_config.get("exec", "thread"),
        max_workers=base_config.get("max_workers", 0),
        parallel_sketch=base_config.get("parallel_sketch"),
        parallel_candidates=base_config.get("parallel_candidates"),
        parallel_validation=base_config.get("parallel_validation"),
        track_min_dist=base_config.get("track_min_dist", True),
    )
    corrtrack.baseline_mode = baseline_mode

    if verbose is None:
        verbose = _coerce_to_bool(base_config.get("verbose", False))
    if testing is None:
        testing = _coerce_to_bool(base_config.get("testing", False))

    record, runtime_parts, corr_flags = execute_corrtrack_pass(
        "bf",
        dataset_id,
        corrtrack,
        data,
        ids,
        metadata,
        corr_val=True,
        recall_by_window=True,
        artifact_prefix=artifact_prefix,
        artifact_mode=artifact_mode,
        artifact_merge_mode=artifact_merge_mode,
        verbose=bool(verbose),
        testing=bool(testing),
        monitor=bool(monitor),
    )
    record["preprocess"] = False
    record["seed"] = None
    record["seed_toggle"] = None
    record["sketch_norm"] = getattr(corrtrack, "sketch_norm", None)
    record["nodes"] = metadata.get("nodes", base_config.get("max_workers"))
    record["freq_threshold"] = getattr(corrtrack, "freq_threshold", None)

    with CSVStreamWriter(output_csv, RUN_RESULT_COLUMNS) as writer:
        writer.write_row(_row_from_mapping(RUN_RESULT_COLUMNS, record))
    return record, runtime_parts, corr_flags


def run_and_log_corrtrack(
    dataset_id: str,
    data: np.ndarray,
    ids: Sequence[str],
    base_config: dict,
    run_params: dict,
    output_csv: str,
    metadata: Optional[dict] = None,
    recall_by_window: bool = True,
    corr_val: bool = True,
    monitor: bool = True,
    verbose: Optional[bool] = None,
    testing: Optional[bool] = None,
    artifact_prefix: Optional[str] = None,
) -> tuple[dict, tuple, Optional[np.ndarray]]:
    metadata = dict(metadata or {})
    metadata.setdefault("mode", "main")
    metadata.setdefault("alg", metadata.get("alg"))
    metadata.setdefault("optim", metadata.get("optim", "main"))
    metadata.setdefault("artifact_buffer_max_rows", run_params.get("artifact_buffer_max_rows") or base_config.get("artifact_buffer_max_rows"))
    metadata.setdefault(
        "artifact_merge_mode",
        run_params.get("artifact_merge_mode", base_config.get("artifact_merge_mode")),
    )
    metadata.setdefault(
        "save_only_required_artifacts",
        run_params.get("save_only_required_artifacts", base_config.get("save_only_required_artifacts")),
    )
    metadata.setdefault(
        "save_maxlag_artifacts",
        run_params.get("save_maxlag_artifacts", base_config.get("save_maxlag_artifacts")),
    )

    def _to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    artifact_mode = (
        run_params.get("artifact_mode")
        or base_config.get("artifact_mode")
        or "iterative"
    )
    artifact_merge_mode = (
        run_params.get("artifact_merge_mode")
        or base_config.get("artifact_merge_mode")
        or "merged"
    )

    n_vectors = _to_int(run_params.get("n_vectors"))
    grid_dimension = _to_int(run_params.get("grid_dimension"))
    cell_stretch = _resolve_cell_stretch(
        run_params.get("cell_stretch"),
        run_params.get("cell_size"),
        base_config["corr_threshold"],
        n_vectors,
        grid_dimension,
        full_vector_candidates=True,
    )
    if cell_stretch is None or cell_stretch <= 0.0:
        cell_stretch = 1.0

    feature_kwargs = _extract_feature_overrides({**base_config, **run_params})
    run_seed = _to_int(run_params.get("seed"))
    seed_toggle = _to_int(run_params.get("seed_toggle"))

    corrtrack = CorrTrack(
        window_size=base_config["window_size"],
        basic_window=base_config.get("basic_window"),
        window_step=base_config["window_step"],
        n_vectors=n_vectors,
        n_lags=base_config["n_lags"],
        grid_dimension=grid_dimension,
        cell_size=cell_stretch,
        seed=run_seed,
        seed_toggle=seed_toggle,
        freq_threshold=_to_float(run_params.get("freq_threshold")),
        corr_threshold=base_config["corr_threshold"],
        neg_corr=base_config.get("neg_corr", False),
        preprocess=run_params.get("preprocess"),
        exec=base_config.get("exec", "thread"),
        max_workers=base_config.get("max_workers", 0),
        parallel_sketch=base_config.get("parallel_sketch"),
        parallel_candidates=base_config.get("parallel_candidates"),
        parallel_validation=base_config.get("parallel_validation"),
        track_min_dist=base_config.get("track_min_dist", True),
        **feature_kwargs,
    )

    if verbose is None:
        verbose = _coerce_to_bool(base_config.get("verbose", False))
    if testing is None:
        testing = _coerce_to_bool(base_config.get("testing", False))

    record, runtime_parts, corr_flags = execute_corrtrack_pass(
        "corrtrack",
        dataset_id,
        corrtrack,
        data,
        ids,
        metadata,
        corr_val=corr_val,
        recall_by_window=True,
        artifact_prefix=artifact_prefix,
        artifact_mode=artifact_mode,
        artifact_merge_mode=artifact_merge_mode,
        verbose=bool(verbose),
        testing=bool(testing),
        monitor=bool(monitor),
    )

    record["preprocess"] = run_params.get("preprocess")
    record["seed"] = run_seed
    record["seed_toggle"] = seed_toggle
    record["sketch_norm"] = corrtrack.sketch_norm
    record["candidate_backend"] = getattr(
        corrtrack,
        "candidate_backend_effective",
        getattr(corrtrack, "candidate_backend", None),
    )
    record["candidate_bucket_width"] = getattr(corrtrack, "candidate_bucket_width", None)
    record["candidate_block_size_steps"] = getattr(corrtrack, "candidate_block_size_steps", None)
    record["candidate_block_index_dims"] = getattr(corrtrack, "candidate_block_index_dims", None)
    # (2026-07-06) Part 1 -- see docs/implementation_log.md.
    record["candidate_bound_dims"] = getattr(corrtrack, "candidate_bound_dims", None)
    record["candidate_bound_dim_selection"] = getattr(corrtrack, "candidate_bound_dim_selection", None)
    record["enable_block_ub_pruning"] = getattr(corrtrack, "enable_block_ub_pruning", None)
    record["enable_row_ub_pruning"] = getattr(corrtrack, "enable_row_ub_pruning", None)
    record["block_similarity_assignment"] = getattr(corrtrack, "block_similarity_assignment", None)
    record["max_open_blocks"] = getattr(corrtrack, "max_open_blocks", None)
    record["candidate_instinct_query_mode"] = getattr(corrtrack, "candidate_instinct_query_mode", None)
    record["candidate_instinct_top_k"] = getattr(corrtrack, "candidate_instinct_top_k", None)
    record["candidate_instinct_min_candidates"] = getattr(corrtrack, "candidate_instinct_min_candidates", None)
    record["candidate_instinct_entry_points"] = getattr(corrtrack, "candidate_instinct_entry_points", None)
    record.update(_candidate_runtime_record_fields(corrtrack))
    record.update(_candidate_search_record_fields(corrtrack))
    record["candidate_parallel_mode"] = getattr(corrtrack, "candidate_parallel_mode", None)
    record["candidate_key_mode"] = getattr(corrtrack, "candidate_key_mode", None)
    record["candidate_key_seed"] = getattr(corrtrack, "candidate_key_seed", None)
    record["candidate_lsh_radius"] = getattr(corrtrack, "candidate_lsh_radius", None)
    record["candidate_ann_m"] = getattr(corrtrack, "candidate_ann_m", None)
    record["candidate_ann_z"] = getattr(corrtrack, "candidate_ann_z", None)
    record["candidate_ann_ef"] = getattr(corrtrack, "candidate_ann_ef", None)
    record["numeric_rows"] = getattr(corrtrack, "numeric_rows", None)
    record["freq_threshold"] = _to_float(run_params.get("freq_threshold"))
    record["n_vectors"] = n_vectors
    record["grid_dimension"] = grid_dimension
    record["cell_stretch"] = cell_stretch
    record["nodes"] = metadata.get("nodes", base_config.get("max_workers"))

    with CSVStreamWriter(output_csv, RUN_RESULT_COLUMNS) as writer:
        writer.write_row(_row_from_mapping(RUN_RESULT_COLUMNS, record))

    return record, runtime_parts, corr_flags


def _coerce_to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _resolve_artifact_save_policy(
    recall_by_window,
    save_only_required_artifacts=False,
    save_maxlag_artifacts=True,
):
    save_only_required = _coerce_to_bool(save_only_required_artifacts, default=False)
    save_maxlag = _coerce_to_bool(save_maxlag_artifacts, default=True)

    if save_only_required:
        return {
            "save_correlated": bool(recall_by_window),
            "save_status": False,
            "save_anomalies": not bool(recall_by_window),
            "save_maxlag": save_maxlag,
        }

    return {
        "save_correlated": True,
        "save_status": True,
        "save_anomalies": True,
        "save_maxlag": save_maxlag,
    }


def _resolve_artifact_merge_mode(value, default="merged"):
    if value is None or value == "":
        value = default
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
    else:
        key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "merge": "merged",
        "merged": "merged",
        "final": "merged",
        "chunks": "chunks",
        "chunk": "chunks",
        "chunked": "chunks",
        "unmerged": "chunks",
        "skip": "chunks",
        "none": "chunks",
    }
    return aliases.get(key, aliases.get(str(default).strip().lower(), "merged"))


def _resolve_parallel_flag(value, default=False) -> bool:
    return _coerce_to_bool(value, default=bool(default))


def _resolve_candidate_backend(value, default="auto"):
    if value is None:
        value = default
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = str(value).strip().lower()
    aliases = {
        "tree": "bptree",
        "bst": "bptree",
        "balanced": "bptree",
        "bucket": "sorted_arrays_bs",
        "bucketed": "sorted_arrays_bs",
        "bucketed_bptree": "sorted_arrays_bs",
        "bucketed_tree": "sorted_arrays_bs",
        "hash_buckets": "sorted_arrays_bs",
        "hash-buckets": "sorted_arrays_bs",
        "bucketed_multi": "sorted_arrays_bs",
        "sorted_bs": "sorted_arrays_bs",
        "sorted_array_bs": "sorted_arrays_bs",
        # (2026-07-05) renamed from "bucketed_bst" -- the backend never used a
        # BST (binary search tree); it's flat sorted arrays per key dimension
        # plus binary search (bisect). Kept as a backward-compat alias so
        # existing configs/artifacts referencing the old name keep working.
        # See docs/implementation_log.md.
        "bucketed_bst": "sorted_arrays_bs",
        # (2026-07-06) InstinctIndex -- experimental approximate graph
        # backend. See docs/implementation_log.md, "InstinctIndex:
        # experimental approximate graph backend".
        "graph": "instinct",
        "instinct_index": "instinct",
    }
    key = aliases.get(key, key)
    if key in {"hamming", "hamming_multi", "hamming-multi", "hamming_mi", "hamming_multiindex", "simhash", "sign_code", "sign-code"}:
        raise ValueError("candidate_backend='hamming_multiindex' has been removed")
    if key in {"pivot", "pivots", "pivot_sorted", "pivot-sorted", "pivot_sorted_array", "pivot-sorted-array", "pivot_array", "pivot-array"}:
        raise ValueError("candidate_backend='pivot_sorted_array' has been removed")
    if key in {"blocked_lazy", "blocked", "lazy_blocked", "blocked-lazy", "blocked_sorted", "blocked-sorted"}:
        raise ValueError("candidate_backend='blocked_lazy' has been removed")
    if key in {"vptree", "vp", "vp_tree", "vp-tree"}:
        raise ValueError("candidate_backend='vptree' has been removed")
    if key in {"angular_lsh", "angular", "angular-lsh", "multi_projection", "multi-projection", "projection_lsh", "projection-lsh"}:
        raise ValueError("candidate_backend='angular_lsh' has been removed")
    if key in {"dynamic_graph_ann", "graph_ann", "dynamic-graph-ann", "nsw"}:
        raise ValueError("candidate_backend='dynamic_graph_ann' has been removed")
    if key not in {"flat", "bptree", "sorted_arrays_bs", "auto", "instinct"}:
        key = default
    return key


_BLOCKED_INDEX_BACKENDS = {"sorted_arrays_bs"}
# (2026-07-06) InstinctIndex -- experimental approximate graph backend, NOT
# recall-preserving by construction (unlike every other backend). See
# docs/implementation_log.md, "InstinctIndex: experimental approximate graph
# backend". Deliberately kept out of _BLOCKED_INDEX_BACKENDS (a distinct
# dispatch marker, its own `_instinct_index` attribute, not `_blocked_index`).
_INSTINCT_INDEX_BACKENDS = {"instinct"}


def _resolve_candidate_similarity(value, default="l2"):
    if value is None:
        value = default
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = str(value).strip().lower()
    aliases = {
        "euclidean": "l2",
        "distance": "l2",
        "dist": "l2",
        "cos": "cosine",
        "dot": "cosine",
        "dot_product": "cosine",
        "dot-product": "cosine",
    }
    key = aliases.get(key, key)
    if key not in {"l2", "cosine"}:
        key = default
    return key


def _resolve_candidate_key_mode(value, default="first"):
    if value is None:
        value = default
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
    else:
        key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "first_dimension": "first",
        "coord0": "first",
        "coordinate": "first",
        "projection": "random_sign",
        "single_projection": "random_sign",
        "random_projection": "random_sign",
        "random-sign": "random_sign",
        "sign_projection": "random_sign",
        "sample": "sampled_sketch",
        "sampled": "sampled_sketch",
        "sampled_projection": "sampled_sketch",
        "sampled-sketch": "sampled_sketch",
        "sketch_sample": "sampled_sketch",
    }
    key = aliases.get(key, key)
    if key not in {"first", "random_sign", "sampled_sketch"}:
        key = default
    return key


def _candidate_gamma_from_tau(tau):
    tau = _to_float_safe(tau)
    if tau is None:
        return None
    return max(-1.0, min(1.0, 1.0 - (float(tau) * float(tau)) / 2.0))


def _candidate_tau_from_gamma(gamma):
    gamma = _to_float_safe(gamma)
    if gamma is None:
        return None
    gamma = max(-1.0, min(1.0, float(gamma)))
    return math.sqrt(max(0.0, 2.0 - 2.0 * gamma))


def _resolve_candidate_parallel_mode(value, default="recent_shards"):
    if value is None:
        value = default
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = str(value).strip().lower()
    aliases = {
        "recent": "recent_shards",
        "recent-shards": "recent_shards",
        "recent_windows": "recent_shards",
        "recent-windows": "recent_shards",
        "query_shards": "recent_shards",
        "query-shards": "recent_shards",
        "full_vector_shards": "recent_shards",
        "full-vector-shards": "recent_shards",
    }
    key = aliases.get(key, key)
    if key not in {"recent_shards"}:
        key = default
    return key


def _resolve_baseline_mode(value, default="bruteforce"):
    if value is None:
        value = default
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = str(value).strip().lower()
    key = key.replace("-", "_")
    aliases = {
        "bf": "bruteforce",
        "brute": "bruteforce",
        "brute_force": "bruteforce",
        "exact": "exact_stomp",
        "stomp": "exact_stomp",
        "exact_stomp_baseline": "exact_stomp",
        "incremental": "exact_stomp",
        "incremental_bf": "exact_stomp",
        "incremental_bruteforce": "exact_stomp",
    }
    key = aliases.get(key, key)
    if key not in {"bruteforce", "exact_stomp"}:
        key = default
    return key


def _uses_full_vector_candidate_radius(*_args, **_kwargs):
    return True


def _normalize_exec_mode(value, default="thread"):
    """Map user-provided execution policy strings onto supported modes."""
    if value is None:
        return default

    key = value
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = str(value).strip().lower()

    aliases = {
        "parallel": "thread",
        "threads": "thread",
        "process": "thread",
        "processes": "thread",
    }
    resolved = aliases.get(key, key)

    valid = {"sequential", "thread"}
    return resolved if resolved in valid else default


def _to_int_safe(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time_key_to_int64(value):
    """Convert window-start keys to int64 for compact partition arrays.

    Most experimental data uses integer positional starts. Real ASOS data can
    carry datetime starts through lower-level sketch objects, so encode those as
    nanoseconds while preserving existing integer behavior.
    """

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return 0
        return int(value.astype("datetime64[ns]").astype(np.int64))
    if isinstance(value, datetime.datetime):
        return int(np.datetime64(value, "ns").astype(np.int64))
    if isinstance(value, datetime.date):
        dt = datetime.datetime.combine(value, datetime.time())
        return int(np.datetime64(dt, "ns").astype(np.int64))
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            parsed = np.datetime64(value, "ns")
            if np.isnat(parsed):
                return 0
            return int(parsed.astype(np.int64))
        except Exception:
            return 0


def _to_float_safe(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator, denominator):
    num = _to_float_safe(numerator)
    den = _to_float_safe(denominator)
    if num is None or den is None:
        return float("nan")
    if math.isnan(num) or math.isnan(den) or den == 0.0:
        return float("nan")
    return num / den


def _compute_speedup_ceil(
    cand_time_bf,
    val_time_bf,
    monit_time_bf,
    sk_time,
    cand_time,
    corr_w_bf,
    cand_w_bf,
):
    cand_time_bf_f = _to_float_safe(cand_time_bf)
    val_time_bf_f = _to_float_safe(val_time_bf)
    monit_time_bf_f = _to_float_safe(monit_time_bf)
    sk_time_f = _to_float_safe(sk_time)
    cand_time_f = _to_float_safe(cand_time)
    ratio = _safe_div(corr_w_bf, cand_w_bf)
    if any(
        value is None
        for value in (
            cand_time_bf_f,
            val_time_bf_f,
            monit_time_bf_f,
            sk_time_f,
            cand_time_f,
        )
    ):
        return float("nan")
    if any(
        math.isnan(value)
        for value in (
            cand_time_bf_f,
            val_time_bf_f,
            monit_time_bf_f,
            sk_time_f,
            cand_time_f,
            ratio,
        )
    ):
        return float("nan")
    denom = sk_time_f + cand_time_f + val_time_bf_f * ratio + monit_time_bf_f
    if denom == 0.0 or math.isnan(denom):
        return float("nan")
    num = cand_time_bf_f + val_time_bf_f + monit_time_bf_f
    return num / denom


def _extract_feature_overrides(params):
    overrides = {}
    norm = params.get("sketch_norm")
    if norm is not None:
        overrides["sketch_norm"] = norm
    backend = params.get("candidate_backend")
    if backend is not None:
        overrides["candidate_backend"] = _resolve_candidate_backend(backend, default="auto")
    bucket_width = params.get("candidate_bucket_width")
    if bucket_width is not None:
        try:
            overrides["candidate_bucket_width"] = float(bucket_width)
        except (TypeError, ValueError):
            if bucket_width in ("", "none", "None"):
                overrides["candidate_bucket_width"] = None
    block_steps = params.get("candidate_block_size_steps")
    if block_steps is not None:
        try:
            overrides["candidate_block_size_steps"] = int(block_steps)
        except (TypeError, ValueError):
            pass
    block_dims = params.get("candidate_block_index_dims")
    if block_dims is not None:
        try:
            overrides["candidate_block_index_dims"] = int(block_dims)
        except (TypeError, ValueError):
            pass
    similarity = params.get("candidate_similarity")
    if similarity is not None:
        overrides["candidate_similarity"] = _resolve_candidate_similarity(similarity, default="l2")
    key_mode = params.get("candidate_key_mode")
    if key_mode is not None:
        overrides["candidate_key_mode"] = _resolve_candidate_key_mode(key_mode, default="first")
    key_seed = params.get("candidate_key_seed")
    if key_seed is not None:
        seed_val = _to_int_safe(key_seed)
        overrides["candidate_key_seed"] = seed_val
    lsh_radius = params.get("candidate_lsh_radius")
    if lsh_radius is not None:
        radius_val = _to_int_safe(lsh_radius)
        if radius_val is not None:
            overrides["candidate_lsh_radius"] = max(0, min(3, int(radius_val)))
    ann_m = params.get("candidate_ann_m")
    if ann_m is not None:
        ann_m_val = _to_int_safe(ann_m)
        if ann_m_val is not None:
            overrides["candidate_ann_m"] = max(1, int(ann_m_val))
    ann_z = params.get("candidate_ann_z")
    if ann_z is not None:
        ann_z_val = _to_int_safe(ann_z)
        if ann_z_val is not None:
            overrides["candidate_ann_z"] = max(1, int(ann_z_val))
    ann_ef = params.get("candidate_ann_ef")
    if ann_ef is not None:
        ann_ef_val = _to_int_safe(ann_ef)
        if ann_ef_val is not None:
            overrides["candidate_ann_ef"] = max(1, int(ann_ef_val))
    gamma = params.get("candidate_cosine_threshold")
    if gamma is not None:
        gamma_val = _to_float_safe(gamma)
        if gamma_val is not None:
            overrides["candidate_cosine_threshold"] = float(gamma_val)
    # (2026-07-06) Part 1: see docs/implementation_log.md. Both pruning
    # flags default off (opt-in) -- only pass through when explicitly set.
    bound_dims = params.get("candidate_bound_dims")
    if bound_dims is not None:
        bound_dims_val = _to_int_safe(bound_dims)
        if bound_dims_val is not None:
            overrides["candidate_bound_dims"] = max(0, int(bound_dims_val))
    bound_dim_selection = params.get("candidate_bound_dim_selection")
    if bound_dim_selection is not None:
        overrides["candidate_bound_dim_selection"] = str(bound_dim_selection)
    enable_block_ub = params.get("enable_block_ub_pruning")
    if enable_block_ub is not None:
        overrides["enable_block_ub_pruning"] = _coerce_to_bool(enable_block_ub, default=False)
    enable_row_ub = params.get("enable_row_ub_pruning")
    if enable_row_ub is not None:
        overrides["enable_row_ub_pruning"] = _coerce_to_bool(enable_row_ub, default=False)
    # (2026-07-06) Part 1 follow-up: block_similarity_assignment/max_open_blocks
    # -- see docs/implementation_log.md, "block-level cone pruning:
    # similarity-aware block assignment". Also opt-in/pass-through-only.
    block_sim_assign = params.get("block_similarity_assignment")
    if block_sim_assign is not None:
        overrides["block_similarity_assignment"] = _coerce_to_bool(block_sim_assign, default=False)
    max_open = params.get("max_open_blocks")
    if max_open is not None:
        max_open_val = _to_int_safe(max_open)
        if max_open_val is not None:
            overrides["max_open_blocks"] = max(1, int(max_open_val))
    # (2026-07-06) InstinctIndex -- experimental approximate graph backend,
    # candidate_backend="instinct" only. See docs/implementation_log.md,
    # "InstinctIndex: experimental approximate graph backend".
    instinct_query_mode = params.get("candidate_instinct_query_mode")
    if instinct_query_mode is not None:
        overrides["candidate_instinct_query_mode"] = str(instinct_query_mode)
    instinct_top_k = params.get("candidate_instinct_top_k")
    if instinct_top_k is not None:
        top_k_val = _to_int_safe(instinct_top_k)
        if top_k_val is not None:
            overrides["candidate_instinct_top_k"] = max(1, int(top_k_val))
    instinct_min_candidates = params.get("candidate_instinct_min_candidates")
    if instinct_min_candidates is not None:
        min_cand_val = _to_int_safe(instinct_min_candidates)
        if min_cand_val is not None:
            overrides["candidate_instinct_min_candidates"] = max(1, int(min_cand_val))
    instinct_entry_points = params.get("candidate_instinct_entry_points")
    if instinct_entry_points is not None:
        entry_points_val = _to_int_safe(instinct_entry_points)
        if entry_points_val is not None:
            overrides["candidate_instinct_entry_points"] = max(1, int(entry_points_val))
    hybrid_kwargs = _resolve_hybrid_validation_kwargs(overrides=params)
    for key, value in hybrid_kwargs.items():
        overrides[key] = value
    return overrides


_HYBRID_VALIDATION_DEFAULTS = {
    "hybrid_validation": False,
    "hybrid_validation_min_repeat_rate": 0.25,
    "hybrid_validation_disable_rate": None,
    "hybrid_validation_ema_alpha": 0.25,
    "hybrid_validation_min_candidates": 256,
}


def _is_missing_param_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _resolve_hybrid_validation_kwargs(defaults=None, overrides=None):
    defaults = defaults or {}
    overrides = overrides or {}

    def pick(name):
        if name in overrides and not _is_missing_param_value(overrides.get(name)):
            return overrides.get(name)
        if name in defaults and not _is_missing_param_value(defaults.get(name)):
            return defaults.get(name)
        return _HYBRID_VALIDATION_DEFAULTS.get(name)

    return {
        "hybrid_validation": _coerce_to_bool(pick("hybrid_validation"), default=False),
        "hybrid_validation_min_repeat_rate": pick("hybrid_validation_min_repeat_rate"),
        "hybrid_validation_disable_rate": pick("hybrid_validation_disable_rate"),
        "hybrid_validation_ema_alpha": pick("hybrid_validation_ema_alpha"),
        "hybrid_validation_min_candidates": pick("hybrid_validation_min_candidates"),
    }


def _hybrid_validation_record_fields(source):
    if source is None:
        return {}
    return {
        "hybrid_validation": getattr(source, "hybrid_validation", None),
        "hybrid_validation_min_repeat_rate": getattr(source, "hybrid_validation_min_repeat_rate", None),
        "hybrid_validation_disable_rate": getattr(source, "hybrid_validation_disable_rate", None),
        "hybrid_validation_ema_alpha": getattr(source, "hybrid_validation_ema_alpha", None),
        "hybrid_validation_min_candidates": getattr(source, "hybrid_validation_min_candidates", None),
        "hybrid_validation_attempts": getattr(source, "hybrid_validation_attempts", None),
        "hybrid_validation_hits": getattr(source, "hybrid_validation_hits", None),
        "hybrid_validation_steps_active": getattr(source, "hybrid_validation_steps_active", None),
    }


def _candidate_runtime_record_fields(source):
    if source is None:
        return {}
    return {
        "candidate_similarity": getattr(source, "candidate_similarity", None),
        "candidate_cosine_threshold": getattr(source, "candidate_cosine_threshold", None),
    }


def _candidate_search_record_fields(source):
    if source is None:
        return {}
    return {
        "candidate_search_index_candidates": getattr(source, "candidate_search_index_candidates", None),
        "candidate_search_valid_index_candidates": getattr(source, "candidate_search_valid_index_candidates", None),
        "candidate_search_unique_index_candidates": getattr(source, "candidate_search_unique_index_candidates", None),
        "candidate_search_duplicate_index_candidates": getattr(source, "candidate_search_duplicate_index_candidates", None),
        "candidate_search_unique_pre_dot_pairs": getattr(source, "candidate_search_unique_pre_dot_pairs", None),
        "candidate_search_duplicate_pre_dot_pairs": getattr(source, "candidate_search_duplicate_pre_dot_pairs", None),
        "candidate_search_after_coord": getattr(source, "candidate_search_after_coord", None),
        "candidate_search_partial_checks": getattr(source, "candidate_search_partial_checks", None),
        "candidate_search_after_partial": getattr(source, "candidate_search_after_partial", None),
        "candidate_search_after_similarity": getattr(source, "candidate_search_after_similarity", None),
        "candidate_search_dot_checks": getattr(source, "candidate_search_dot_checks", None),
        "candidate_search_distance_checks": getattr(source, "candidate_search_distance_checks", None),
        # (2026-07-06) Part 1 metrics -- see docs/implementation_log.md.
        "candidate_search_blocks_visited": getattr(source, "candidate_search_blocks_visited", None),
        "candidate_search_blocks_pruned_by_ub": getattr(source, "candidate_search_blocks_pruned_by_ub", None),
        "candidate_search_rows_in_surviving_blocks": getattr(source, "candidate_search_rows_in_surviving_blocks", None),
        "candidate_search_dot_checks_saved_by_row_ub": getattr(source, "candidate_search_dot_checks_saved_by_row_ub", None),
        # (2026-07-06) InstinctIndex metrics -- see docs/implementation_log.md,
        # "InstinctIndex: experimental approximate graph backend".
        "candidate_search_instinct_visited_nodes": getattr(source, "candidate_search_instinct_visited_nodes", None),
        "candidate_search_instinct_visited_live_nodes": getattr(source, "candidate_search_instinct_visited_live_nodes", None),
        "candidate_search_instinct_dead_nodes_skipped": getattr(source, "candidate_search_instinct_dead_nodes_skipped", None),
        "candidate_search_instinct_edges_scanned": getattr(source, "candidate_search_instinct_edges_scanned", None),
        "candidate_search_instinct_candidates_returned": getattr(source, "candidate_search_instinct_candidates_returned", None),
        "candidate_search_instinct_threshold_candidates": getattr(source, "candidate_search_instinct_threshold_candidates", None),
        "candidate_search_instinct_topk_candidates": getattr(source, "candidate_search_instinct_topk_candidates", None),
        "candidate_search_instinct_queries_with_too_few_live_nodes": getattr(source, "candidate_search_instinct_queries_with_too_few_live_nodes", None),
        "candidate_search_instinct_query_time": getattr(source, "candidate_search_instinct_query_time", None),
        "candidate_search_instinct_insert_time": getattr(source, "candidate_search_instinct_insert_time", None),
        "candidate_search_instinct_num_nodes_total": getattr(source, "candidate_search_instinct_num_nodes_total", None),
        "candidate_search_instinct_num_nodes_alive": getattr(source, "candidate_search_instinct_num_nodes_alive", None),
        "candidate_search_instinct_best_score_seen": getattr(source, "candidate_search_instinct_best_score_seen", None),
        "candidate_search_instinct_mean_score_returned": getattr(source, "candidate_search_instinct_mean_score_returned", None),
        "candidate_search_instinct_dead_node_ratio": getattr(source, "candidate_search_instinct_dead_node_ratio", None),
    }


def _coerce_int_list(value, default=None, *, field_name="value"):
    if value is None or value == "":
        value = default
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        value = parsed
    if isinstance(value, (np.integer, int)):
        items = [int(value)]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]

    out = []
    seen = set()
    for item in items:
        try:
            item_int = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must contain positive integers.") from None
        if item_int <= 0:
            raise ValueError(f"{field_name} must contain positive integers.")
        if item_int not in seen:
            out.append(item_int)
            seen.add(item_int)
    return out


def _compute_base_cell_size(corr_threshold, n_vectors):
    if corr_threshold is None:
        return None
    n_vec = _to_float_safe(n_vectors)
    if n_vec is None or n_vec <= 0.0:
        return None
    try:
        return math.sqrt((1.0 - float(corr_threshold)) * 2.0) / math.sqrt(n_vec)
    except (ValueError, ZeroDivisionError):
        return None


def _resolve_cell_stretch(
    stretch_value,
    cell_size_value,
    corr_threshold,
    n_vectors,
    grid_dimension,
    default=1.0,
    full_vector_candidates=False,
):
    stretch = _to_float_safe(stretch_value)
    if stretch is not None and stretch > 0.0:
        return stretch

    size = _to_float_safe(cell_size_value)
    if size is None:
        return default

    base = _compute_base_cell_size(corr_threshold, n_vectors)
    n_vec = _to_float_safe(n_vectors)
    if base is None or base <= 0.0 or n_vec is None or n_vec <= 0.0:
        return default

    if full_vector_candidates:
        radius_scale = math.sqrt(n_vec)
    else:
        grid_dim = _to_float_safe(grid_dimension)
        if grid_dim is None or grid_dim <= 0.0:
            grid_dim = 1.0
        radius_scale = math.sqrt(grid_dim)

    if radius_scale <= 0.0:
        return default

    stretch = size / (base * radius_scale)
    if not math.isfinite(stretch) or stretch <= 0.0:
        return default
    return stretch


def _fast_corr_and_dist(x, y, return_stats=False):
    """Compute Pearson correlation and Euclidean distance without temporary arrays.

    When ``return_stats`` is True, also returns (n, mean_x, mean_y, var_x, var_y),
    where var_* are the summed squared deviations (n * variance).
    """
    if _cy_fast_corr_and_dist is not None:
        try:
            corr, dist, stats = _cy_fast_corr_and_dist(
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
            )
            if return_stats:
                return corr, dist, stats
            return corr, dist
        except Exception:
            pass
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = x.size
    if n == 0:
        if return_stats:
            return np.nan, np.inf, (0, 0.0, 0.0, 0.0, 0.0)
        return np.nan, np.inf

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

    denom = math.sqrt(var_x * var_y)

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
    dist = math.sqrt(max(dist_sq, 0.0))

    if return_stats:
        return corr, dist, (n, mean_x, mean_y, var_x, var_y)
    return corr, dist


def _metric_distance_from_corr(corr):
    if corr is None or not np.isfinite(corr):
        return np.inf
    return math.sqrt(max(2.0 * (1.0 - abs(float(corr))), 0.0))


def _passes_corr_threshold(corr, corr_threshold, neg_corr):
    if corr is None or np.isnan(corr):
        return False
    if neg_corr:
        return abs(float(corr)) >= corr_threshold
    return float(corr) >= corr_threshold


def _compute_series_dots(window_blocks, weights):
    kernel = os.environ.get("CORRTRACK_SKETCH_KERNEL")
    if kernel:
        kernel = kernel.strip().lower()
    else:
        kernel = "cython" if _cy_compute_series_dots is not None else "numpy"
    if kernel == "cython" and _cy_compute_series_dots is not None:
        try:
            wb = np.ascontiguousarray(window_blocks, dtype=np.float64)
            wt = np.ascontiguousarray(weights, dtype=np.float64)
            return _cy_compute_series_dots(wb, wt)
        except Exception:
            pass
    return np.einsum("sbw,bvw->sbv", window_blocks, weights, optimize=True)


def _is_near_constant_stats(var_sum, n, std_thresh=1e-3):
    if std_thresh is None:
        std_thresh = 1e-3
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


def _sketch_worker(payload):
    """Execute a sketch node update."""
    corrtrack, node_index, node_new, ids_subset, verbose, testing = payload
    sketch_node = corrtrack.sketch_nodes[node_index]
    sketch_node._sid_lookup = corrtrack.series_ids
    sketch_node._profile_callback = corrtrack._profile_add if getattr(corrtrack, "profile_enabled", False) else None
    sketches, partitions = sketch_node.run(
        node_new,
        ids_subset,
        verbose=verbose,
        testing=testing,
        distribute=False,
    )
    return node_index, sketch_node, sketches, partitions


def _grid_worker(payload):
    """Execute a grid node run."""
    corrtrack, grid_index, n_ids, verbose, testing = payload[:5]
    try:
        numeric_rows = bool(payload[5])
    except (IndexError, TypeError):
        numeric_rows = True
    grid_node = corrtrack.grid_nodes[grid_index]
    result = grid_node.run(n_ids, verbose=verbose, testing=testing, numeric_rows=numeric_rows)
    return grid_index, grid_node, result


def _bf_worker(payload):
    """Execute brute-force candidate generation for a shard."""
    if len(payload) >= 8:
        index, bf_node, window_step, ids, verbose, testing, ref_ids, numeric_rows = payload[:8]
    else:
        index, bf_node, window_step, ids, verbose, testing, ref_ids = payload
        numeric_rows = True
    result = bf_node.run(
        window_step,
        ids,
        verbose=verbose,
        testing=testing,
        ref_ids=ref_ids,
        numeric_rows=bool(numeric_rows),
    )
    return index, bf_node, result


def _compute_nonconst_mask(data, window_size, std_thresh=None):
    csum = np.cumsum(data, axis=1, dtype=np.float64)
    csum = np.pad(csum, ((0, 0), (1, 0)), mode="constant")
    csum_sq = np.cumsum(data * data, axis=1, dtype=np.float64)
    csum_sq = np.pad(csum_sq, ((0, 0), (1, 0)), mode="constant")
    window_sums = csum[:, window_size:] - csum[:, :-window_size]
    window_sums_sq = csum_sq[:, window_size:] - csum_sq[:, :-window_size]
    var_sum = window_sums_sq - (window_sums * window_sums) / window_size
    if std_thresh is None:
        std_thresh = 1e-3
    threshold = (float(std_thresh) ** 2) * window_size
    return np.maximum(var_sum, 0.0) > threshold


def _enumerate_candidate_rows(
    data,
    window_index,
    ref_indices,
    window_size,
    window_step,
    mask=None,
    std_thresh=None,
    shard_start=None,
    shard_end=None,
):
    if _cy_enumerate_candidate_rows is not None and mask is None:
        std_val = 1e-3 if std_thresh is None else float(std_thresh)
        shard_start_val = -1 if shard_start is None else int(shard_start)
        shard_end_val = -1 if shard_end is None else int(shard_end)
        ref_idx_arr = np.asarray(list(ref_indices), dtype=np.int64)
        if ref_idx_arr.size == 0:
            return None
        return _cy_enumerate_candidate_rows(
            np.ascontiguousarray(data, dtype=np.float64),
            np.ascontiguousarray(window_index, dtype=np.int64),
            ref_idx_arr,
            int(window_size),
            int(window_step),
            std_val,
            shard_start_val,
            shard_end_val,
        )

    data = np.asarray(data)
    n_series, n_cols = data.shape
    window_count = n_cols - window_size + 1
    if window_count <= 0:
        return None

    working_mask = mask
    if working_mask is not None:
        working_mask = np.asarray(working_mask, dtype=bool)
        working_mask = working_mask.reshape(n_series, window_count)
    else:
        working_mask = _compute_nonconst_mask(
            data.astype(np.float64, copy=False),
            window_size,
            std_thresh=std_thresh,
        )

    if working_mask.size == 0:
        return None

    step_mask = (np.arange(window_count) % max(1, window_step)) == 0
    valid_mask = working_mask & step_mask
    last_idx = window_count - 1
    seeds_mask = valid_mask[:, last_idx]
    if not np.any(seeds_mask):
        return None

    valid_k_all, valid_j_all = np.nonzero(valid_mask)
    if valid_k_all.size == 0:
        return None

    j_start_times = window_index[valid_j_all].astype(np.int64, copy=False)
    curr_start = int(window_index[last_idx])
    rows_accum = []

    for s_idx in ref_indices:
        if s_idx >= n_series or not seeds_mask[s_idx]:
            continue

        mask_sel = np.ones(valid_k_all.shape[0], dtype=bool)
        mask_sel &= ~((valid_k_all <= s_idx) & (valid_j_all == last_idx))
        if not mask_sel.any():
            continue

        indices = np.nonzero(mask_sel)[0]
        k_sel = valid_k_all[indices]
        if shard_start is not None and shard_end is not None:
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


def _normalize_bf_key(pair):
    id1, id2, t1, t2, w = pair

    if id1 == id2:
        return (id1, id2, max(t1, t2), min(t1, t2), w)

    if t1 == t2:
        return tuple(sorted([id1, id2])) + (t1, t2, w)

    if t1 < t2:
        return (id2, id1, t2, t1, w)
    return (id1, id2, t1, t2, w)




def _normalize_window_metric_key(id1, id2, t1, t2, window_size):
    return _normalize_bf_key((id1, id2, t1, t2, window_size))


def _normalize_pair_lag_key(id1, id2, lag):
    ids = tuple(sorted((id1, id2)))
    return ids + (lag,)


def _marker_to_label(marker):
    if marker == 1:
        return "into"
    if marker == -1:
        return "out_of"
    if marker == 0:
        return "changed_sign"
    return str(marker)


def _label_to_marker(label):
    if label == "into":
        return 1
    if label == "out_of":
        return -1
    if label == "changed_sign":
        return 0
    try:
        return int(label)
    except (TypeError, ValueError):
        return 0


def _to_sortable_time_value(value):
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return ""
        return str(value.astype("datetime64[ns]"))
    if isinstance(value, (datetime.datetime, datetime.date)):
        return str(np.datetime64(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return ""
        return float(value)
    return str(value)


def _window_sort_key(pair, window_size):
    id1, id2, t1, t2 = pair
    norm = _normalize_window_metric_key(id1, id2, t1, t2, window_size)
    return (
        str(norm[0]),
        str(norm[1]),
        _to_sortable_time_value(norm[2]),
        _to_sortable_time_value(norm[3]),
    )


def _status_sort_key(row):
    id1, id2, lag, t1, t2, _duration, _sign = row
    norm = _normalize_pair_lag_key(id1, id2, lag)
    return (
        str(norm[0]),
        str(norm[1]),
        _to_sortable_time_value(norm[2]),
        _to_sortable_time_value(t1),
        _to_sortable_time_value(t2),
    )


def _anomaly_sort_key(row):
    id1, id2, lag, time_val, marker = row
    norm = _normalize_pair_lag_key(id1, id2, lag)
    return (
        str(norm[0]),
        str(norm[1]),
        _to_sortable_time_value(norm[2]),
        _to_sortable_time_value(time_val),
        int(marker),
    )


def _maxlag_sort_key(key):
    id1, id2, t1_idx = key
    return (str(id1), str(id2), _to_sortable_time_value(t1_idx))


def _compute_total_time_from_record(record):
    try:
        n_w = int(float(record.get("n_w", 0) or 0))
        window_size = int(float(record.get("window_size", 0) or 0))
        window_step = int(float(record.get("window_step", 0) or 0))
    except (TypeError, ValueError):
        return 0
    if n_w <= 0 or window_size <= 0 or window_step <= 0:
        return 0
    return ((n_w - 1) * window_step) + window_size


def _intervals_total_length(intervals):
    total = 0
    for start, end in intervals:
        total += max(int(end) - int(start), 0)
    return total


def _intervals_overlap_length(lhs, rhs):
    i = j = 0
    overlap = 0
    while i < len(lhs) and j < len(rhs):
        start = max(int(lhs[i][0]), int(rhs[j][0]))
        end = min(int(lhs[i][1]), int(rhs[j][1]))
        if end > start:
            overlap += end - start
        if int(lhs[i][1]) <= int(rhs[j][1]):
            i += 1
        else:
            j += 1
    return overlap


def _welford_update(count, mean, m2, value):
    count += 1
    delta = value - mean
    mean += delta / count
    delta2 = value - mean
    m2 += delta * delta2
    return count, mean, m2


def _safe_prec_recall_from_counts(tp, pred_total, gt_total):
    precision = tp / pred_total if pred_total > 0 else 0.0
    recall = tp / gt_total if gt_total > 0 else 0.0
    return precision, recall


def _signed_precision_from_ambiguous_fp(tp_same_sign, tp_opposite_sign, pred_total):
    denom = max(int(pred_total or 0) - int(tp_opposite_sign or 0), 0)
    return int(tp_same_sign or 0) / denom if denom > 0 else 0.0


def _empty_stream_metrics(include_sign=True):
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "f1_score": 0.0,
        "aucroc": float("nan"),
        "pr_auc": float("nan"),
        "recall_min": None,
    }
    if include_sign:
        metrics.update({
            "precision_pos": 0.0,
            "recall_pos": 0.0,
            "f1_score_pos": 0.0,
            "precision_neg": 0.0,
            "recall_neg": 0.0,
            "f1_score_neg": 0.0,
        })
    return metrics


def _current_candidate_anchor(corrtrack):
    window_index = getattr(corrtrack, "window_index", None)
    window_data = getattr(corrtrack, "window_data", None)
    window_size = getattr(corrtrack, "window_size", None)
    if window_index is None or window_data is None or window_size is None:
        return None
    try:
        window_count = int(window_data.shape[1]) - int(window_size) + 1
    except (AttributeError, TypeError, ValueError):
        return None
    if window_count <= 0 or len(window_index) < window_count:
        return None
    try:
        return int(window_index[window_count - 1])
    except (TypeError, ValueError, IndexError):
        return None


class _OptimizerBFWindowReferenceWriter:
    def __init__(self, csv_path, window_size):
        self.csv_path = csv_path
        self.window_size = window_size
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._handle = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._handle, delimiter=CSV_DELIMITER)
        self._writer.writerow(["id1", "id2", "time1_idx", "time2_idx", "sign"])
        self.gt_total = 0
        self.gt_pos = 0
        self.gt_neg = 0

    def observe(self, corrtrack):
        validated = getattr(corrtrack, "_validated_step", None) or {}
        if not validated:
            return
        rows = []
        for pair, corr in validated.items():
            norm = _normalize_window_metric_key(pair[0], pair[1], pair[2], pair[3], pair[4])
            sign = 1 if float(corr) > 0 else -1
            rows.append((str(norm[0]), str(norm[1]), int(norm[2]), int(norm[3]), int(sign)))
        rows.sort(key=lambda row: (row[2], row[0], row[1], row[3]))
        self._writer.writerows(rows)
        self.gt_total += len(rows)
        for row in rows:
            if row[4] > 0:
                self.gt_pos += 1
            else:
                self.gt_neg += 1

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        try:
            handle.close()
        finally:
            self._handle = None


class _OptimizerOnlineWindowMetrics:
    def __init__(
        self,
        reference_csv,
        window_size,
        gt_total,
        gt_pos,
        gt_neg,
        total_pairs_bf=None,
        pair_min_dist=None,
    ):
        self.reference_csv = reference_csv
        self.window_size = window_size
        self.gt_total = int(gt_total or 0)
        self.gt_pos = int(gt_pos or 0)
        self.gt_neg = int(gt_neg or 0)
        try:
            self.total_pairs_bf = int(total_pairs_bf) if total_pairs_bf is not None else None
        except (TypeError, ValueError):
            self.total_pairs_bf = None
        self.recall_min_key = None
        if pair_min_dist is not None and isinstance(pair_min_dist, tuple) and len(pair_min_dist) >= 5:
            self.recall_min_key = _normalize_window_metric_key(
                pair_min_dist[0],
                pair_min_dist[1],
                pair_min_dist[2],
                pair_min_dist[3],
                pair_min_dist[4],
            )
        self.recall_min_hit = False
        self.pred_total = 0
        self.tp = 0
        self.tp_pos = 0
        self.tp_neg = 0
        self._file = open(reference_csv, newline="")
        self._reader = csv.DictReader(self._file, delimiter=_detect_csv_delimiter(reference_csv, default=CSV_DELIMITER))
        self._next_row = next(self._reader, None)

    @staticmethod
    def _parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _load_group(self, anchor):
        group = {}
        while self._next_row is not None:
            row_anchor = self._parse_int(self._next_row.get("time1_idx"))
            if row_anchor is None:
                self._next_row = next(self._reader, None)
                continue
            if row_anchor < anchor:
                self._next_row = next(self._reader, None)
                continue
            if row_anchor > anchor:
                break
            id1 = self._next_row.get("id1")
            id2 = self._next_row.get("id2")
            time2 = self._parse_int(self._next_row.get("time2_idx"))
            sign = self._parse_int(self._next_row.get("sign"))
            if id1 is not None and id2 is not None and time2 is not None and sign is not None:
                group[(id1, id2, row_anchor, time2, self.window_size)] = sign
            self._next_row = next(self._reader, None)
        return group

    def observe(self, corrtrack):
        anchor = _current_candidate_anchor(corrtrack)
        if anchor is None:
            return
        gt_group = self._load_group(anchor)
        predicted = set()
        for pair in getattr(corrtrack, "candidates", {}) or {}:
            if not isinstance(pair, tuple) or len(pair) < 5:
                continue
            predicted.add(_normalize_window_metric_key(pair[0], pair[1], pair[2], pair[3], pair[4]))
        self.pred_total += len(predicted)
        if self.recall_min_key is not None and self.recall_min_key in predicted:
            self.recall_min_hit = True
        for key in predicted:
            sign = gt_group.get(key)
            if sign is None:
                continue
            self.tp += 1
            if sign > 0:
                self.tp_pos += 1
            else:
                self.tp_neg += 1

    def metrics(self):
        metrics = _empty_stream_metrics(include_sign=True)
        precision, recall = _safe_prec_recall_from_counts(self.tp, self.pred_total, self.gt_total)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recall_pos = self.tp_pos / self.gt_pos if self.gt_pos > 0 else 0.0
        recall_neg = self.tp_neg / self.gt_neg if self.gt_neg > 0 else 0.0
        precision_pos = _signed_precision_from_ambiguous_fp(self.tp_pos, self.tp_neg, self.pred_total)
        precision_neg = _signed_precision_from_ambiguous_fp(self.tp_neg, self.tp_pos, self.pred_total)
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0.0
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
        specificity = 0.0
        if self.total_pairs_bf is not None:
            negatives = max(self.total_pairs_bf - self.gt_total, 0)
            if negatives > 0:
                fp = max(self.pred_total - self.tp, 0)
                tn = max(negatives - fp, 0)
                specificity = tn / negatives
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1,
                "precision_pos": precision_pos,
                "recall_pos": recall_pos,
                "f1_score_pos": f1_pos,
                "precision_neg": precision_neg,
                "recall_neg": recall_neg,
                "f1_score_neg": f1_neg,
                "recall_min": None if self.recall_min_key is None else int(self.recall_min_hit),
                "aucroc": float("nan"),
                "pr_auc": float("nan"),
            }
        )
        return metrics

    def close(self):
        handle = getattr(self, "_file", None)
        if handle is None:
            return
        try:
            handle.close()
        finally:
            self._file = None
            self._reader = None
            self._next_row = None


def _corr_validation_batch_worker(payload):
    """Validate a batch of candidate pairs.

    payload keys (current shape, built by CorrTrack._iter_validation_payloads):
      - pairs, rows ([s1, s2, t1, t2, w] int64 array), data, base_index
      - corr_threshold, neg_corr, corr_val
      - optionally current_window_sums/_sq/_cu/_qu + current_window_size
        ("point 1": lets validate_corr_rows skip re-accumulating whichever
        side of a pair is the just-arrived current window, entirely inside
        its own nogil loop -- see docs/implementation_log.md)

    Legacy payload keys (items: list of {"pair","x","y"} dicts) are also
    supported, for any caller not using the rows-based shape above.
    Returns list of tuples (pair, is_corr, corr, dist, is_constant, is_spiked)
    """
    corr_threshold = payload["corr_threshold"]
    neg_corr = payload["neg_corr"]
    corr_val = payload.get("corr_val", True)
    std_thresh = 1e-3

    rows = payload.get("rows")
    if rows is not None:
        pairs = payload.get("pairs") or []
        if not corr_val:
            return [(pair, True, 1.0, 0.0, False, False) for pair in pairs]
        if not pairs:
            return []
        data = payload["data"]
        base_index = payload["base_index"]
        if _cy_validate_corr_rows is not None:
            try:
                accepted_mask, corrs, dists, consts, spikes = _cy_validate_corr_rows(
                    np.ascontiguousarray(data, dtype=np.float64),
                    rows,
                    int(base_index),
                    float(corr_threshold),
                    bool(neg_corr),
                    std_thresh,
                    5.0,
                    current_window_sums=payload.get("current_window_sums"),
                    current_window_sums_sq=payload.get("current_window_sums_sq"),
                    current_window_sums_cu=payload.get("current_window_sums_cu"),
                    current_window_sums_qu=payload.get("current_window_sums_qu"),
                    current_window_size=payload.get("current_window_size", -1),
                )
                return [
                    (
                        pair,
                        bool(accepted_mask[i]),
                        float(corrs[i]),
                        float(dists[i]),
                        bool(consts[i]),
                        bool(spikes[i]),
                    )
                    for i, pair in enumerate(pairs)
                ]
            except Exception:
                pass

        results = []
        for i, pair in enumerate(pairs):
            s1, s2, t1, t2, w = (int(v) for v in rows[i])
            start1 = t1 - base_index
            start2 = t2 - base_index
            x = data[s1, start1:start1 + w]
            y = data[s2, start2:start2 + w]
            pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
            n, mean_x, mean_y, var_x, var_y = stats
            if _is_near_constant_stats(var_x, n, std_thresh=std_thresh) or _is_near_constant_stats(var_y, n, std_thresh=std_thresh):
                results.append((pair, False, np.nan, np.inf, True, False))
                continue
            if (
                _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh=5.0)
                or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh=5.0)
            ):
                results.append((pair, False, np.nan, np.inf, False, True))
                continue
            is_correlated = _passes_corr_threshold(pair_corr, corr_threshold, neg_corr)
            results.append((pair, is_correlated, pair_corr, pair_dist, False, False))
        return results

    items = payload.get("items")
    if items is None:
        return []

    if not corr_val:
        return [
            (item["pair"], True, 1.0, 0.0, False, False)
            for item in items
        ]
    results = []

    if _cy_validate_corr_batch is not None:
        try:
            x_batch = np.asarray([item["x"] for item in items], dtype=np.float64)
            y_batch = np.asarray([item["y"] for item in items], dtype=np.float64)
            if x_batch.ndim == 2 and y_batch.shape == x_batch.shape:
                eff_std_thresh = float(std_thresh) if std_thresh is not None else 1e-3
                batch_results = _cy_validate_corr_batch(
                    x_batch,
                    y_batch,
                    float(corr_threshold),
                    bool(neg_corr),
                    eff_std_thresh,
                )
                for item, entry in zip(items, batch_results):
                    is_correlated, pair_corr, pair_dist, is_const, is_spiked = entry
                    results.append((item["pair"], is_correlated, pair_corr, pair_dist, is_const, is_spiked))
                return results
        except Exception:
            results = []

    try:
        for item in items:
            pair = item["pair"]

            x = item["x"]
            y = item["y"]

            pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
            n, mean_x, mean_y, var_x, var_y = stats

            if _is_near_constant_stats(var_x, n, std_thresh=std_thresh) or _is_near_constant_stats(var_y, n, std_thresh=std_thresh):
                results.append((pair, False, np.nan, np.inf, True, False))
                continue

            if (
                _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh=5.0)
                or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh=5.0)
            ):
                results.append((pair, False, np.nan, np.inf, False, True))
                continue

            is_correlated = _passes_corr_threshold(pair_corr, corr_threshold, neg_corr)

            results.append((pair, is_correlated, pair_corr, pair_dist, False, False))

        return results
    finally:
        pass
#from numba import njit, prange


# ==== Parallel runtime hygiene and policies (auto-injected) ====
import os as _os_parallel_guard
_os_parallel_guard.environ.setdefault("OMP_NUM_THREADS", "1")
_os_parallel_guard.environ.setdefault("MKL_NUM_THREADS", "1")
_os_parallel_guard.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os_parallel_guard.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os_parallel_guard.environ.setdefault("MKL_CBWR", "COMPATIBLE")
_os_parallel_guard.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
# ================================================================

class CorrTrack:
    def __init__(self,window_size,basic_window,window_step,n_vectors,n_lags,grid_dimension,cell_size,seed=2468,seed_toggle=1357,freq_threshold=0.7,corr_threshold=0.7,neg_corr=False,preprocess=False,exec="parallel",max_workers=0,sketch_norm="z",candidate_backend="auto",candidate_bucket_width=None,candidate_block_size_steps=None,candidate_block_index_dims=None,candidate_similarity="l2",candidate_cosine_threshold=None,candidate_parallel_mode="recent_shards",candidate_key_mode="first",candidate_key_seed=None,candidate_lsh_radius=None,candidate_ann_m=None,candidate_ann_z=None,candidate_ann_ef=None,parallel_sketch=None,parallel_candidates=None,parallel_validation=None,track_min_dist=True,hybrid_validation=False,hybrid_validation_min_repeat_rate=0.25,hybrid_validation_disable_rate=None,hybrid_validation_ema_alpha=0.25,hybrid_validation_min_candidates=256,numeric_rows=True,validation_current_window_cache=False,candidate_bound_dims=None,candidate_bound_dim_selection="variance",enable_block_ub_pruning=False,enable_row_ub_pruning=False,block_similarity_assignment=False,max_open_blocks=4,candidate_instinct_query_mode="hybrid",candidate_instinct_top_k=256,candidate_instinct_min_candidates=64,candidate_instinct_entry_points=8):
        
        if basic_window is not None and window_size % basic_window != 0:
            raise TypeError("Window size (",window_size,") is not divisable by basic window size (",basic_window,")")
        if basic_window is not None and basic_window % window_step != 0:
            raise TypeError("Basic window size (",basic_window,") is not divisable by window step (",window_step,")")
        #if window_step > 0 and n_lags % window_step != 0:
        #    raise TypeError("Number of lags (",n_lags,") is not divisable by window step (",window_step,")")
        exec_mode = _normalize_exec_mode(exec, default="thread")
        parallel_default = exec_mode == "thread"
        parallel_sketch_flag = _resolve_parallel_flag(parallel_sketch, False)
        parallel_candidates_flag = _resolve_parallel_flag(parallel_candidates, False)
        parallel_validation_flag = _resolve_parallel_flag(parallel_validation, parallel_default)
        candidate_parallel_mode = "recent_shards"

        grid_dimension = 1 if grid_dimension is None or grid_dimension <= 0 else int(grid_dimension)
        if n_vectors is None or n_vectors <= 0:
            raise ValueError("n_vectors must be a positive integer.")
        # Parameters features
        self.neg_corr = neg_corr
        # Parameters data
        self.window_data = None
        self.window_index = None
        self.window_startTimes = None
        self._cw_raw_sums = None
        self._cw_raw_sums_sq = None
        self._cw_raw_sums_cu = None
        self._cw_raw_sums_qu = None
        self._cw_raw_moments_fresh = False
        self.series_ids = {}
        self.map_ids = []
        self.ids = None
        self.datetime_index = None
        self.datetime_lookup = {} #TODO: save correlation logs to file
        self.preprocess = preprocess
        self.sketch_norm = str(sketch_norm) if sketch_norm is not None else "z"
        self.candidate_backend = _resolve_candidate_backend(candidate_backend, default="auto")
        self.candidate_bucket_width = _to_float_safe(candidate_bucket_width)
        block_steps = _to_int_safe(candidate_block_size_steps)
        if block_steps is None or block_steps <= 0:
            block_steps = 32
        self.candidate_block_size_steps = int(block_steps)
        block_dims = _to_int_safe(candidate_block_index_dims)
        if block_dims is None or block_dims <= 0:
            block_dims = 1
        self.candidate_block_index_dims = int(block_dims)
        # (2026-07-06) Part 1: see Candidates.__init__ for the full
        # rationale and docs/implementation_log.md. Both pruning flags
        # default False (opt-in) -- row-level pruning showed ~0 wall-clock
        # benefit in testing (real dot-check reduction, but the bound
        # computation cost offsets it); block-level (cone/angular) pruning
        # showed a real 4-5x speedup on data with genuine angular
        # clustering tighter than gamma, and safely no-ops otherwise.
        bound_dims_val = _to_int_safe(candidate_bound_dims)
        self.candidate_bound_dims = max(0, bound_dims_val) if bound_dims_val is not None else 0
        self.candidate_bound_dim_selection = str(candidate_bound_dim_selection or "variance").lower()
        self.enable_block_ub_pruning = bool(enable_block_ub_pruning)
        self.enable_row_ub_pruning = bool(enable_row_ub_pruning)
        self.block_similarity_assignment = bool(block_similarity_assignment)
        self.max_open_blocks = max(1, _to_int_safe(max_open_blocks) or 4)
        # (2026-07-06) InstinctIndex candidate_backend params -- see
        # docs/implementation_log.md. Approximate, opt-in backend only;
        # inert for every other candidate_backend value.
        self.candidate_instinct_query_mode = str(candidate_instinct_query_mode or "hybrid").lower()
        self.candidate_instinct_top_k = max(1, _to_int_safe(candidate_instinct_top_k) or 256)
        self.candidate_instinct_min_candidates = max(1, _to_int_safe(candidate_instinct_min_candidates) or 64)
        self.candidate_instinct_entry_points = max(1, _to_int_safe(candidate_instinct_entry_points) or 8)
        self.candidate_similarity = _resolve_candidate_similarity(candidate_similarity, default="l2")
        self._candidate_cosine_threshold_input = _to_float_safe(candidate_cosine_threshold)
        self.candidate_parallel_mode = candidate_parallel_mode
        self.candidate_key_mode = _resolve_candidate_key_mode(candidate_key_mode, default="first")
        self.candidate_key_seed = _to_int_safe(candidate_key_seed)
        lsh_radius = _to_int_safe(candidate_lsh_radius)
        if lsh_radius is None:
            lsh_radius = 0
        self.candidate_lsh_radius = max(0, min(3, int(lsh_radius)))
        ann_m = _to_int_safe(candidate_ann_m)
        if ann_m is None or ann_m <= 0:
            ann_m = 16
        self.candidate_ann_m = max(1, int(ann_m))
        ann_z = _to_int_safe(candidate_ann_z)
        if ann_z is None or ann_z <= 0:
            ann_z = 256
        self.candidate_ann_z = max(1, int(ann_z))
        ann_ef = _to_int_safe(candidate_ann_ef)
        if ann_ef is None or ann_ef <= 0:
            ann_ef = max(64, self.candidate_ann_z)
        self.candidate_ann_ef = max(1, int(ann_ef))
        self.numeric_rows = True
        self.track_min_dist = _coerce_to_bool(track_min_dist, default=True)
        # (2026-07-06) "point 1" on/off toggle: a pure performance lever (no
        # effect on which pairs validate as correlated -- verified
        # identical output either way), not a scientific parameter, so
        # deliberately kept out of the CSV/grid/hyperopt column machinery
        # used for hybrid_validation's own tuning knobs. Defaults to False
        # (opt-in), NOT True: benchmarking on the default (non-hybrid)
        # validation path showed it is often a net *loss* there -- the
        # per-pair Python-level bookkeeping this adds to
        # _iter_validation_payloads (building x_series/y_series/
        # x_is_current/y_is_current at Python speed) frequently costs more
        # than the Cython-level accumulation it saves, except at large
        # window sizes. Only clearly worth enabling after measuring it help
        # on the specific workload at hand. See docs/implementation_log.md.
        self.validation_current_window_cache = _coerce_to_bool(validation_current_window_cache, default=False)
        self.hybrid_validation = _coerce_to_bool(hybrid_validation, default=False)
        self.hybrid_validation_min_repeat_rate = _to_float_safe(hybrid_validation_min_repeat_rate)
        if self.hybrid_validation_min_repeat_rate is None:
            self.hybrid_validation_min_repeat_rate = 0.25
        self.hybrid_validation_min_repeat_rate = min(max(float(self.hybrid_validation_min_repeat_rate), 0.0), 1.0)
        disable_rate = _to_float_safe(hybrid_validation_disable_rate)
        if disable_rate is None:
            disable_rate = self.hybrid_validation_min_repeat_rate * 0.5
        self.hybrid_validation_disable_rate = min(max(float(disable_rate), 0.0), 1.0)
        alpha = _to_float_safe(hybrid_validation_ema_alpha)
        if alpha is None:
            alpha = 0.25
        self.hybrid_validation_ema_alpha = min(max(float(alpha), 0.0), 1.0)
        min_candidates = _to_int_safe(hybrid_validation_min_candidates)
        if min_candidates is None:
            min_candidates = 256
        self.hybrid_validation_min_candidates = max(1, int(min_candidates))
        # Parameters lags
        self.window_size = window_size

        self.window_step = None
        if window_step > 0:
            self.window_step = window_step

        if basic_window is None:
            self.basic_window = CorrTrack.get_bst_basic_window(self.window_size,self.window_step) #divides window_size
        else:
            self.basic_window = basic_window

        if window_step == 0:
            self.window_step = self.basic_window

        self.curr_window_size = None
        self.n_lags = (n_lags // self.window_step) * self.window_step #divisible by window_step
        self.n_lagged_windows = self.n_lags//self.window_step+1
        # Parameter sketches
        self.seed = seed if seed is not None else int(np.random.SeedSequence().entropy)
        self.seed_toggle = seed_toggle
        self.n_vectors = n_vectors
        # Parameters nodes
        self.exec = exec_mode
        self.parallel_sketch = parallel_sketch_flag
        self.parallel_candidates = parallel_candidates_flag
        self.parallel_validation = parallel_validation_flag
        self.parallel_any = self.parallel_sketch or self.parallel_candidates or self.parallel_validation

        self.n_nodes = max_workers
        if not self.parallel_any:
            self.n_nodes = 1
        elif max_workers == 0:
            self.n_nodes = max(1, (os.cpu_count() or 1))
        self.n_sketch_nodes = self.n_nodes if self.parallel_sketch else 1
        self.n_candidate_nodes = self.n_nodes if self.parallel_candidates else 1
        self.sketch_nodes = []
        self.grid_nodes = []
        self.brute_force_nodes = []
        self.exact_stomp_bf_node = None
        self.baseline_mode = "bruteforce"
        self._thread_pool = None
        self._thread_pool_workers = None
        # Parameters grids
        self.recent_shard_candidates = bool(
            self.parallel_candidates and self.candidate_parallel_mode == "recent_shards"
        )
        self.full_vector_candidates = True
        self.grid_dimension = int(self.n_vectors)
        self.n_grids = 1
        self.freq_pairs = {}
        self.candidates = {}
        self._candidate_numeric_rows = None
        self._last_candidate_numeric_rows = None
        self.validated = {}
        self.correlated = {}
        self.corr_attention_in = {}
        self.corr_attention_out = {}
        self.sketch_mean = 0
        self.sketch_std = np.sqrt(self.window_size/self.n_vectors)
        self.corr_threshold = corr_threshold

        base = _compute_base_cell_size(corr_threshold, self.n_vectors)
        if base is None:
            base = np.sqrt((1.0 - corr_threshold) * 2.0) / np.sqrt(self.n_vectors)
        grid_dim = _to_float_safe(self.grid_dimension)
        if grid_dim is None or grid_dim <= 0.0:
            grid_dim = 1.0
        try:
            grid_adjust = math.sqrt(grid_dim)
        except ValueError:
            grid_adjust = 1.0
        stretch = cell_size if cell_size is not None else 1.0
        try:
            stretch = float(stretch)
        except (TypeError, ValueError):
            stretch = 1.0
        if stretch <= 0.0:
            stretch = 1.0

        self.cell_stretch = stretch
        local_cell_size = base * stretch * grid_adjust
        full_cell_size = base * stretch * math.sqrt(float(self.n_vectors))
        self.cell_size = full_cell_size
        if self._candidate_cosine_threshold_input is None:
            self.candidate_cosine_threshold = _candidate_gamma_from_tau(self.cell_size)
        else:
            self.candidate_cosine_threshold = max(
                -1.0,
                min(1.0, float(self._candidate_cosine_threshold_input)),
            )
        self.grid_max = min(1.0, 3.0/np.sqrt(self.n_vectors))

        # Retain the constructor argument for old callers, but disable frequency gating.
        self.freq_threshold = 0

        #self.freq_threshold = self._compute_weight_threshold()

        # Parameters meta analysis
        self.hist_sketches = []
        self._artifact_state = {
            "prefix": None,
            "headers": set(),
            "correlated": set(),
            "neg_pairs": set(),
            "candidates": {},
            "status": {},
            "anomalies": set(),
        }
        self._artifact_mode_runtime = "final"
        self._artifact_merge_mode = "merged"
        self._artifact_buffer_max_rows = 250000
        self._artifact_buffered = False
        self._artifact_recall_by_window = True
        self._artifact_save_correlated = True
        self._artifact_save_status = True
        self._artifact_save_anomalies = True
        self._artifact_save_maxlag = True
        self._spill_correlated = {}
        self._spill_correlated_rows = []
        self._spill_correlated_row_count = 0
        self._spill_correlated_chunks = []
        self._spill_correlated_chunk_index = 0
        self._spill_status_chunks = []
        self._spill_status_chunk_index = 0
        self._spill_anomaly_chunks = []
        self._spill_anomaly_chunk_index = 0
        self._monitor_state = _cy_numeric_monitor_state_cls() if _cy_numeric_monitor_state_cls is not None else None
        self._maxlag_state = {}
        self._step_observer_enabled = False
        self._validated_step = {}
        self._validated_numeric_rows_step = None
        self._validated_numeric_corrs_step = None
        self._validated_numeric_count_step = 0
        self._validated_numeric_capacity_step = 0
        self._online_window_metrics_only = False
        self.sketches = {}
        self.brute_force_steps = 0
        self.constant_candidates = 0
        self.total_candidates = 0
        self.tested_candidates = 0
        self.validated_candidates = 0
        self.candidate_search_index_candidates = 0
        self.candidate_search_valid_index_candidates = 0
        self.candidate_search_unique_index_candidates = 0
        self.candidate_search_duplicate_index_candidates = 0
        self.candidate_search_unique_pre_dot_pairs = 0
        self.candidate_search_duplicate_pre_dot_pairs = 0
        self.candidate_search_after_coord = 0
        self.candidate_search_partial_checks = 0
        self.candidate_search_after_partial = 0
        self.candidate_search_after_similarity = 0
        self.candidate_search_dot_checks = 0
        self.candidate_search_distance_checks = 0
        # (2026-07-06) Part 1 metrics -- see docs/implementation_log.md.
        self.candidate_search_blocks_visited = 0
        self.candidate_search_blocks_pruned_by_ub = 0
        self.candidate_search_rows_in_surviving_blocks = 0
        self.candidate_search_dot_checks_saved_by_row_ub = 0
        # (2026-07-06) InstinctIndex metrics -- see docs/implementation_log.md,
        # "InstinctIndex: experimental approximate graph backend".
        self.candidate_search_instinct_visited_nodes = 0
        self.candidate_search_instinct_visited_live_nodes = 0
        self.candidate_search_instinct_dead_nodes_skipped = 0
        self.candidate_search_instinct_edges_scanned = 0
        self.candidate_search_instinct_candidates_returned = 0
        self.candidate_search_instinct_threshold_candidates = 0
        self.candidate_search_instinct_topk_candidates = 0
        self.candidate_search_instinct_queries_with_too_few_live_nodes = 0
        self.candidate_search_instinct_query_time = 0.0
        self.candidate_search_instinct_insert_time = 0.0
        self.candidate_search_instinct_num_nodes_total = 0
        self.candidate_search_instinct_num_nodes_alive = 0
        self.candidate_search_instinct_best_score_seen = 0.0
        self.candidate_search_instinct_mean_score_returned = 0.0
        self.candidate_search_instinct_dead_node_ratio = 0.0
        self._hybrid_validation_rate_ema = 0.0
        self._hybrid_validation_active = False
        self._hybrid_validation_cache = _cy_hybrid_cache_cls() if _cy_hybrid_cache_cls is not None else None
        self.hybrid_validation_attempts = 0
        self.hybrid_validation_hits = 0
        self.hybrid_validation_steps_active = 0
        #Parameters time
        self.sketch_time = 0
        self.candidate_time = 0
        self.validation_time = 0
        self.monitor_time = 0
        self.artifact_bookkeeping_time = 0.0

        self.profile_enabled = bool(int(os.environ.get("CORRTRACK_PROFILE", "0")))
        self.profile_print_every = max(0, int(os.environ.get("CORRTRACK_PROFILE_EVERY", "0") or 0))
        self.profile_path = os.environ.get("CORRTRACK_PROFILE_PATH", "").strip()
        self.profile_silent = bool(int(os.environ.get("CORRTRACK_PROFILE_SILENT", "0") or 0))
        self._profile_steps = 0
        self._profile_stats = defaultdict(float)
        self._profile_counts = defaultdict(int)
        self._profile_metrics = defaultdict(float)
        self._profile_metric_counts = defaultdict(int)

        self.min_dist = np.inf
        self.pair_min_dist = None

        # Filter tuning defaults
        self.sign_prefilter_scale = 1.3
        self.sign_prefilter_extra = 1
        # Neighbor search removed; keep margin for compatibility with debug tooling.
        self.neighbor_margin = 0.0

        # Instantiating corrtrack objects
        for g in range(self.n_grids):
            node = Candidates(
                self.n_lagged_windows,
                self.grid_dimension,
                self.cell_size,
                self.grid_max,
                self.freq_threshold,
                self.corr_threshold,
                self.n_vectors,
                self.sketch_std,
                self.n_grids,
                self.neg_corr,
                full_vector=self.full_vector_candidates,
                sign_prefilter_scale=self.sign_prefilter_scale,
                sign_prefilter_extra=self.sign_prefilter_extra,
                seed=(self.seed + g) if self.seed is not None else g,
                candidate_backend=self.candidate_backend,
                candidate_bucket_width=self.candidate_bucket_width,
                candidate_block_size_steps=self.candidate_block_size_steps,
                candidate_block_index_dims=self.candidate_block_index_dims,
                candidate_similarity=self.candidate_similarity,
                candidate_cosine_threshold=self.candidate_cosine_threshold,
                candidate_key_mode=self.candidate_key_mode,
                candidate_key_seed=(self.candidate_key_seed + g) if self.candidate_key_seed is not None else None,
                candidate_lsh_radius=self.candidate_lsh_radius,
                candidate_ann_m=self.candidate_ann_m,
                candidate_ann_z=self.candidate_ann_z,
                candidate_ann_ef=self.candidate_ann_ef,
                return_distances=False,
                candidate_bound_dims=self.candidate_bound_dims,
                candidate_bound_dim_selection=self.candidate_bound_dim_selection,
                enable_block_ub_pruning=self.enable_block_ub_pruning,
                enable_row_ub_pruning=self.enable_row_ub_pruning,
                block_similarity_assignment=self.block_similarity_assignment,
                max_open_blocks=self.max_open_blocks,
                candidate_instinct_query_mode=self.candidate_instinct_query_mode,
                candidate_instinct_top_k=self.candidate_instinct_top_k,
                candidate_instinct_min_candidates=self.candidate_instinct_min_candidates,
                candidate_instinct_entry_points=self.candidate_instinct_entry_points,
            )
            self.grid_nodes.append(node)
        if self.grid_nodes:
            self.candidate_backend_effective = getattr(
                self.grid_nodes[0],
                "_candidate_backend",
                self.candidate_backend,
            )
        else:
            self.candidate_backend_effective = self.candidate_backend

    def configure_filters(self, sign_scale=None, sign_extra=None):
        if sign_scale is not None:
            self.sign_prefilter_scale = float(sign_scale)
        if sign_extra is not None:
            self.sign_prefilter_extra = int(sign_extra)

        for node in self.grid_nodes:
            if sign_scale is not None:
                node.sign_prefilter_scale = float(sign_scale)
            if sign_extra is not None:
                node.sign_prefilter_extra = int(sign_extra)

    def choose_grid_dimension(K: int) -> int:
        if K is None or K <= 0:
            raise ValueError("K must be a positive integer")

        # Prefer divisors close to sqrt(K); cap the search window to keep grids manageable.
        max_dim = min(K, 32)
        divisors = [d for d in range(2, max_dim + 1) if K % d == 0]

        if not divisors:
            # Fallbacks for cases where K has no small divisors: allow a single grid (d = K)
            return K

        target = max(2, min(max_dim, int(round(math.sqrt(K)))))
        # Choose the divisor nearest to the target; on ties prefer the larger divisor (fewer grids).
        return min(divisors, key=lambda d: (abs(d - target), -d))

    @staticmethod
    def _expected_cell_recall(cell_width: float, sigma: float, dim: int) -> float:
        if dim <= 0 or sigma <= 0.0:
            return 1.0
        per_axis = math.erf(cell_width / (2.0 * math.sqrt(2.0) * sigma))
        per_axis = max(0.0, min(1.0, per_axis))
        return per_axis ** dim


    @staticmethod
    def _chunk_sequence(sequence, chunk_size):
        """Yield slices of *sequence* with at most *chunk_size* elements."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        total = len(sequence)
        for start in range(0, total, chunk_size):
            yield sequence[start:start + chunk_size]

    def _profile_add(self, key: str, elapsed: float):
        if not self.profile_enabled:
            return
        self._profile_stats[key] += elapsed
        self._profile_counts[key] += 1

    def _profile_add_metric(self, key: str, value: float):
        if not self.profile_enabled:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        self._profile_metrics[key] += value
        self._profile_metric_counts[key] += 1

    def _profile_tick(self):
        if not self.profile_enabled or self.profile_print_every <= 0:
            return
        self._profile_steps += 1
        if self._profile_steps % self.profile_print_every == 0:
            self._print_profile_stats()

    def _print_profile_stats(self):
        if not self.profile_enabled:
            return
        lines = []
        for key in sorted(self._profile_stats):
            total = self._profile_stats[key]
            count = max(1, self._profile_counts.get(key, 0))
            avg = total / count
            lines.append(f"{key}: total={total:.4f}s count={count} avg={avg:.6f}s")
        if not lines:
            return
        message = "[CorrTrack profile] " + " | ".join(lines)
        if self.profile_path:
            try:
                os.makedirs(os.path.dirname(self.profile_path) or ".", exist_ok=True)
                with open(self.profile_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
            except Exception:
                pass
        if not self.profile_silent:
            print(message)
        metric_lines = []
        for key in sorted(self._profile_metrics):
            total = self._profile_metrics[key]
            count = max(1, self._profile_metric_counts.get(key, 0))
            avg = total / count
            if abs(total - round(total)) < 1e-9:
                total_text = str(int(round(total)))
            else:
                total_text = f"{total:.4f}"
            metric_lines.append(f"{key}: total={total_text} count={count} avg={avg:.3f}")
        if not metric_lines:
            return
        metric_message = "[CorrTrack profile metrics] " + " | ".join(metric_lines)
        if self.profile_path:
            try:
                os.makedirs(os.path.dirname(self.profile_path) or ".", exist_ok=True)
                with open(self.profile_path, "a", encoding="utf-8") as handle:
                    handle.write(metric_message + "\n")
            except Exception:
                pass
        if not self.profile_silent:
            print(metric_message)

    def _log_parallel_diag(
        self,
        label,
        mode,
        items_len,
        requested_workers,
        effective_workers,
        chunk_size=None,
    ):
        if not self.profile_enabled:
            return
        logged = getattr(self, "_parallel_diag_logged", None)
        if logged is None:
            logged = set()
            self._parallel_diag_logged = logged
        key = (label, mode)
        if key in logged:
            return
        logged.add(key)
        parts = [
            f"{label}: mode={mode}",
            f"items={items_len}",
            f"requested_workers={requested_workers}",
            f"effective_workers={effective_workers}",
        ]
        if chunk_size is not None:
            parts.append(f"chunk={chunk_size}")
        parts.extend(
            [
                f"n_nodes={getattr(self, 'n_nodes', None)}",
                f"n_sketch_nodes={getattr(self, 'n_sketch_nodes', None)}",
                f"n_candidate_nodes={getattr(self, 'n_candidate_nodes', None)}",
                f"n_grids={getattr(self, 'n_grids', None)}",
                f"map_ids={len(getattr(self, 'map_ids', []) or [])}",
                f"series={len(getattr(self, 'series_ids', {}) or {})}",
            ]
        )
        message = "[CorrTrack parallel] " + " | ".join(parts)
        if self.profile_path:
            try:
                os.makedirs(os.path.dirname(self.profile_path) or ".", exist_ok=True)
                with open(self.profile_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
            except Exception:
                pass
        if not self.profile_silent:
            print(message)


    @staticmethod
    def _run_batch(func, batch):
        """Apply *func* to each item in *batch* and return the collected results."""
        return [func(item) for item in batch]


    def _parallel_map(self,func,iterable,mode: str = "thread",max_workers: int = None,chunksize: int = None,preserve_order: bool = True):
        items = list(iterable)
        if not items:
            return []

        if mode not in {"thread", "sequential"}:
            mode = "thread"

        items_len = len(items)

        if mode == "sequential":
            self._log_parallel_diag(
                getattr(func, "__name__", "callable"),
                mode,
                items_len,
                max_workers,
                1,
            )
            return [func(x) for x in items]

        requested_workers = max_workers
        if max_workers is None or max_workers <= 0:
            max_workers = max(1, (os.cpu_count() or 1))
        max_workers = max(1, min(max_workers, items_len))

        if max_workers <= 1:
            self._log_parallel_diag(
                getattr(func, "__name__", "callable"),
                mode,
                items_len,
                requested_workers,
                max_workers,
            )
            return [func(x) for x in items]

        if chunksize is None:
            chunk_size = 0
        else:
            try:
                chunk_size = int(chunksize)
            except (TypeError, ValueError):
                chunk_size = 0

        if chunk_size <= 0:
            auto = len(items) // (max_workers * 4)
            chunk_size = 1 if auto <= 1 else min(auto, 64)

        chunk_size = max(1, min(chunk_size, len(items)))
        self._log_parallel_diag(
            getattr(func, "__name__", "callable"),
            mode,
            items_len,
            requested_workers,
            max_workers,
            chunk_size=chunk_size,
        )

        use_dask = _HAS_DASK and delayed is not None

        if not use_dask:
            executor_cls = ThreadPoolExecutor
            executor = None
            managed_executor = True

            def _shutdown_executor(ex):
                if ex is not None and managed_executor:
                    try:
                        ex.shutdown(wait=True, cancel_futures=True)
                    except Exception:
                        pass

            try:
                if mode == "thread":
                    executor = self._get_thread_pool(max_workers)
                    managed_executor = False
                if executor is None:
                    executor = executor_cls(max_workers=max_workers)

                if preserve_order:
                    if chunk_size == 1:
                        return list(executor.map(func, items))

                    batches = list(self._chunk_sequence(items, chunk_size))
                    results = []
                    runner = _run_batch_static
                    for batch_result in executor.map(runner, repeat(func), batches):
                        results.extend(batch_result)
                    return results

                if chunk_size == 1:
                    futures = [executor.submit(func, item) for item in items]
                    results = []
                    for future in as_completed(futures):
                        results.append(future.result())
                    return results

                batches = list(self._chunk_sequence(items, chunk_size))
                runner = _run_batch_static
                futures = [executor.submit(runner, func, batch) for batch in batches]
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())
                return results
            except Exception as exc:
                if getattr(self, "verbose", False):
                    print(f"[parallel_map:{mode}] Executor fallback to sequential due to: {exc}")
                self._reset_thread_pool()
                return [func(x) for x in items]
            finally:
                _shutdown_executor(executor)

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1))

        scheduler_get = dask_threaded_get
        if scheduler_get is None:
            return [func(x) for x in items]

        if max_workers is not None:
            scheduler_get = partial(scheduler_get, num_workers=max_workers)

        try:
            tasks = [delayed(func)(item) for item in items]
            results = compute(*tasks, scheduler=scheduler_get)
            return list(results)
        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"[parallel_map:{mode}] Falling back to sequential due to: {e}")
            return [func(x) for x in items]

    def _get_thread_pool(self, max_workers: int):
        """Reuse a thread pool between calls to avoid frequent spin-up."""
        if max_workers is None or max_workers <= 0:
            max_workers = max(1, (os.cpu_count() or 1))

        pool = getattr(self, "_thread_pool", None)
        if pool is not None and self._thread_pool_workers != max_workers:
            self._reset_thread_pool()
            pool = None

        if pool is None:
            pool = ThreadPoolExecutor(max_workers=max_workers)
            self._thread_pool = pool
            self._thread_pool_workers = max_workers
        return pool

    def _reset_thread_pool(self):
        pool = getattr(self, "_thread_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self._thread_pool = None
        self._thread_pool_workers = None
        

    def _compute_weight_threshold(self):
        if self.cell_size is None:
            return None
        
        match_std = np.sqrt(2 * self.sketch_std**2 * (1 - self.corr_threshold))

        # Probability that two values fall into the same bucket (centered on zero)
        prob = norm.cdf(self.cell_size / 2, scale=match_std) - norm.cdf(-self.cell_size / 2, scale=match_std)

        # Central bucket weight under sketch distribution
        bucket_prob = norm.cdf(self.cell_size / 2, scale=self.sketch_std) - norm.cdf(-self.cell_size / 2, scale=self.sketch_std)
        central_weight = 1.0 / bucket_prob if bucket_prob > 0 else float('inf')

        # Expected weight sum for a truly correlated pair
        expected_sum = self.n_vectors * prob * central_weight
        return expected_sum
    
    def _is_datetime(data):
        if len(data) >0:
            dt = data[0]
        else:
            dt = data
        return np.issubdtype(type(dt), np.datetime64) or np.issubdtype(type(dt), np.dtype('datetime64[ns]')) or np.issubdtype(type(dt), datetime.datetime)

    def _datetime_to_index(self, dates):
        if CorrTrack._is_datetime(dates):
            dts = np.array(dates, dtype='datetime64[ns]')

            last_idx = -1
            if self.window_index is not None and len(self.window_index) > 0:
                last_idx = self.window_index[-1]
            
            int_seq = np.arange(last_idx+1, last_idx+1+len(dts))

            self.datetime_lookup.update(dict(zip(int_seq,dts)))
        else:
            int_seq = dates
        return np.array(int_seq)        
    
    def _safe_index_to_datetime(self, idx):
        try:
            if hasattr(self, 'datetime_lookup') and isinstance(idx, (int, np.integer)) and idx in self.datetime_lookup:
                return self.datetime_lookup[idx]
            return idx
        except Exception:
            return idx

    def _index_to_datetime(self, index):
        if isinstance(index, (int, np.integer, float, np.float64)):
            index = [index]
        index = np.asarray(index)
        
        return np.array([self.datetime_lookup[int(i)] for i in index], dtype='datetime64[ns]')
    
    def _update_curr_data(self,new_data_step,ids):
        prev_window_data = self.window_data
        prev_curr_window_size = self.curr_window_size
        new_data_step_index = new_data_step[0,:]
        new_data_step_values = new_data_step[1:,:]
        if self.datetime_index is None:
            self.datetime_index = CorrTrack._is_datetime(new_data_step_index)        
        if self.datetime_index:
            new_data_step_index = self._datetime_to_index(new_data_step_index)

        if self.window_data is None:
            self.window_index = new_data_step_index
            self.window_data = new_data_step_values
        else:
            if self.window_data.shape[1] >= (self.n_lags+self.window_size):
                self.window_index = self.window_index[self.window_step:]
                self.window_data = self.window_data[:,self.window_step:]
                self._cleanup_datetime_lookups()
            self.window_index = np.append(self.window_index,new_data_step_index)
            self.window_data = np.append(self.window_data,new_data_step_values,axis=1)

        self.ids = ids
        if len(self.series_ids) == 0 or len(self.series_ids) != len(ids):
            series_indexes = {}
            for i, val in enumerate(ids):
                series_indexes[val]= i
            self.series_ids = series_indexes # same order of the series in new_data_step

        self.window_startTimes = np.arange(int(self.window_index[0]), int(self.window_index[-1]) + 1, int(self.window_step))

        self._map_series_to_nodes()

        self._update_curr_window_size()
        # (2026-07-06) Lazy, not unconditional: _cw_raw_sums* is only ever
        # read by _validate_pairs_hybrid_cython (gated on hybrid_validation
        # being enabled), so recomputing it here on every step regardless of
        # whether hybrid_validation is even on would be pure wasted work in
        # the (current default) non-hybrid configuration. Just mark it
        # stale; _validate_pairs_hybrid_cython recomputes on first actual use
        # per step. See docs/implementation_log.md.
        self._cw_raw_moments_fresh = False

    def _cleanup_datetime_lookups(self):
        if not hasattr(self, "datetime_lookup") or not self.datetime_lookup:
            return

        # The earliest timestamp still in memory
        if hasattr(self, "window_times") and len(self.window_index) > 0:
            min_time = self.window_index[0]
            # Drop anything older than that
            to_remove = [t for t in self.datetime_lookup if t < min_time]
            for t in to_remove:
                del self.datetime_lookup[t]
    
    def _curr_time(self):
        return self.window_index[-1]+1

    def _map_series_to_nodes(self):
        ids = list(self.series_ids.keys())
        n_ids = len(ids)
        if not self.parallel_sketch:
            self.n_sketch_nodes = 1
            self.map_ids = [ids]
            return

        self.n_sketch_nodes = min(self.n_nodes, n_ids) if n_ids > 0 else 0
        if not getattr(self, "_parallel_warned", False):
            if self.parallel_sketch and n_ids > 1 and self.n_sketch_nodes < 2:
                warnings.warn(
                    "Parallel mode requested but only one sketch node is available; "
                    "increase max_workers or provide more series to enable sketch parallelism.",
                    RuntimeWarning,
                )
            if self.parallel_candidates and self.n_grids < 2:
                warnings.warn(
                    "Parallel mode requested but n_grids=1; reduce grid_dimension or "
                    "increase n_vectors to enable grid parallelism.",
                    RuntimeWarning,
                )
            self._parallel_warned = True

        flat_map_ids = [i for node_ids in self.map_ids for i in node_ids]
        if len(self.map_ids) == 0 or len(set(flat_map_ids) ^ set(ids)) > 0:
            n_series_node = math.ceil(len(ids) / self.n_sketch_nodes) if self.n_sketch_nodes > 0 else len(ids)
            if n_series_node % 1 != 0:
                n_series_node = n_series_node // 1 + 1
            self.map_ids = []
            for node in range(self.n_sketch_nodes):
                start = node * n_series_node
                end = start + n_series_node
                self.map_ids.append(ids[start:end])
    
    def _preprocess_data(self, data, series_index):       
        t = np.asarray(data)
        if self.preprocess:
            diff_t = t[:, 1:] - t[:, :-1]                         # 1. Differencing
            #clipped_t = np.clip(diff_t, -3, 3)                   # 2. Linear clipping
            t = diff_t

        return t

    def is_structurally_spiked(
        x=None,
        kurt_thresh=5,
        *,
        mean=None,
        var_sum=None,
        n=None,
        mu4_sum=None,
    ):
        if var_sum is not None and n is not None and mu4_sum is not None:
            ref_mean = float(mean) if mean is not None else 0.0
            return _is_structurally_spiked_stats(
                None,
                ref_mean,
                max(var_sum, 0.0),
                int(n),
                kurt_thresh=kurt_thresh,
                mu4_sum=mu4_sum,
            )

        if x is None:
            return False
        x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.size
        if n < 4:
            return False
        mean = float(np.mean(x_arr))
        var_sum = float(np.dot(x_arr - mean, x_arr - mean))
        return _is_structurally_spiked_stats(
            x_arr,
            mean,
            max(var_sum, 0.0),
            n,
            kurt_thresh=kurt_thresh,
        )

    def is_near_constant(x=None, std_thresh=1e-3, *, var_sum=None, n=None):
        if var_sum is not None and n is not None:
            return _is_near_constant_stats(max(var_sum, 0.0), int(n), std_thresh=std_thresh)

        if x is None:
            return True
        x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.size
        if n == 0:
            return True
        mean = float(np.mean(x_arr))
        var_sum = float(np.dot(x_arr - mean, x_arr - mean))
        return _is_near_constant_stats(max(var_sum, 0.0), n, std_thresh=std_thresh)

    def _validate_corr(self, pair, corr_val=True):
        if not corr_val:
            return (pair, True, 1, 0.0)

        ids = (pair[0],pair[1])
        startTimes = (pair[2],pair[3])
        window_size = pair[4]
        series_index = self.series_ids[ids[0]]        
        startTime_index = int(startTimes[0] - self.window_index[0])
        x = self.window_data[series_index,startTime_index:startTime_index+window_size].astype(np.float64, copy=False)
        series_index = self.series_ids[ids[1]]
        startTime_index = int(startTimes[1] - self.window_index[0])
        y = self.window_data[series_index,startTime_index:startTime_index+window_size].astype(np.float64, copy=False)

        pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
        n, mean_x, mean_y, var_x, var_y = stats

        if _is_near_constant_stats(var_x, n, std_thresh=1e-3) or _is_near_constant_stats(var_y, n, std_thresh=1e-3):
            return (pair, False, np.nan, np.inf)

        if (
            _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh=5.0)
            or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh=5.0)
        ):
            return (pair, False, np.nan, np.inf)

        is_correlated = _passes_corr_threshold(pair_corr, self.corr_threshold, self.neg_corr)

        return (pair, is_correlated, pair_corr, pair_dist)
    
    def _in_corr(self,pair, corr_value=None):
        raise RuntimeError(
            "Monitoring uses monitor_kernels.NumericMonitorState; "
            "legacy Python-object monitoring is disabled"
        )
    
    def _out_corr(self, item):
        """
        Compute the (time, -1) anomaly for a given previous_correlation entry without mutating state.
        item: (pair_key, status_list) where status_list == [t1, t2, w, last_len]
        Returns: (pair_key, (time, -1))
        """
        raise RuntimeError(
            "Monitoring uses monitor_kernels.NumericMonitorState; "
            "legacy Python-object monitoring is disabled"
        )

    def _configure_artifact_runtime(
        self,
        artifact_mode="final",
        artifact_buffer_max_rows=None,
        artifact_merge_mode="merged",
        recall_by_window=True,
        save_correlated=True,
        save_status=True,
        save_anomalies=True,
        save_maxlag=True,
    ):
        mode_value = (artifact_mode or "final").lower()
        if mode_value not in {"iterative", "final", "buffered"}:
            mode_value = "final"
        self._artifact_mode_runtime = mode_value
        self._artifact_merge_mode = _resolve_artifact_merge_mode(artifact_merge_mode, default="merged")
        self._artifact_recall_by_window = bool(recall_by_window)
        self._artifact_save_correlated = bool(save_correlated)
        self._artifact_save_status = bool(save_status)
        self._artifact_save_anomalies = bool(save_anomalies)
        self._artifact_save_maxlag = bool(save_maxlag)
        try:
            max_rows = int(artifact_buffer_max_rows) if artifact_buffer_max_rows is not None else self._artifact_buffer_max_rows
        except (TypeError, ValueError):
            max_rows = self._artifact_buffer_max_rows
        self._artifact_buffer_max_rows = max(1, max_rows)
        # Both buffered and iterative modes should flow through the sorted
        # chunk writer so downstream streaming comparison can rely on global
        # canonical ordering.
        self._artifact_buffered = mode_value in {"iterative", "buffered"}

        self._spill_correlated = {}
        self._spill_correlated_rows = []
        self._spill_correlated_row_count = 0
        self._spill_correlated_chunks = []
        self._spill_correlated_chunk_index = 0
        self._spill_status_chunks = []
        self._spill_status_chunk_index = 0
        self._spill_anomaly_chunks = []
        self._spill_anomaly_chunk_index = 0
        if _cy_numeric_monitor_state_cls is None:
            raise RuntimeError(
                "Monitoring requires compiled monitor_kernels.NumericMonitorState; "
                "Python monitoring state is disabled"
            )
        self._monitor_state = _cy_numeric_monitor_state_cls()
        self._maxlag_state = {}

    def _artifact_runtime_flush_needed(self):
        if not self._artifact_buffered:
            return False
        limit = max(1, int(getattr(self, "_artifact_buffer_max_rows", 1)))
        monitor_state = getattr(self, "_monitor_state", None)
        pending_status = int(monitor_state.pending_status_count()) if monitor_state is not None else 0
        pending_anomalies = int(monitor_state.pending_anomaly_count()) if monitor_state is not None else 0
        return (
            (
                getattr(self, "_artifact_save_correlated", True)
                and (
                    len(getattr(self, "_spill_correlated", {}))
                    + int(getattr(self, "_spill_correlated_row_count", 0) or 0)
                ) >= limit
            )
            or (
                getattr(self, "_artifact_save_status", True)
                and pending_status >= limit
            )
            or (
                getattr(self, "_artifact_save_anomalies", True)
                and pending_anomalies >= limit
            )
        )

    def _record_correlated(self, pair, corr, retain_validated=True):
        self._record_correlated_batch({pair: corr}, retain_validated=retain_validated)

    def _record_correlated_batch(self, accepted_map, retain_validated=True, record_artifact=True):
        if not accepted_map:
            return
        if getattr(self, "_step_observer_enabled", False):
            self._validated_step.update(accepted_map)
        if retain_validated:
            self.validated.update(accepted_map)
        if record_artifact and getattr(self, "_artifact_buffered", False):
            t0 = time.perf_counter()
            if getattr(self, "_artifact_save_correlated", True):
                self._spill_correlated.update(accepted_map)
            if getattr(self, "_artifact_save_maxlag", True):
                for pair, corr in accepted_map.items():
                    self._update_maxlag_state(pair, corr)
            self.artifact_bookkeeping_time += time.perf_counter() - t0
        elif record_artifact:
            if getattr(self, "_artifact_save_correlated", True):
                self.correlated.update(accepted_map)
            if getattr(self, "_artifact_save_maxlag", True):
                for pair, corr in accepted_map.items():
                    self._update_maxlag_state(pair, corr)

    def _update_maxlag_state(self, pair, corr):
        id1, id2, t1, t2, _window = pair
        if corr is None:
            return
        if isinstance(corr, float) and math.isnan(corr):
            return

        timepts = np.array([t1, t2])
        try:
            lag = int(abs(timepts.max() - timepts.min()))
        except TypeError:
            lag = int(abs(np.int64(timepts.max()) - np.int64(timepts.min())))

        if self.n_lags is not None and lag > self.n_lags:
            return

        pair_ids = tuple(sorted((id1, id2)))
        if pair_ids[0] == id1:
            time1 = t1
        else:
            time1 = t2

        key = (pair_ids[0], pair_ids[1], int(time1))
        score = abs(float(corr))
        prev = self._maxlag_state.get(key)
        if prev is None or score > prev["score"] or (score == prev["score"] and lag < prev["lag"]):
            self._maxlag_state[key] = {
                "max_corr": corr,
                "lag": lag,
                "score": score,
                "time1_raw": time1,
            }

    def _iter_validation_payloads(self, pairs, chunk_size):
        # (2026-07-06) Rewritten to build a numeric [s1, s2, t1, t2, w] rows
        # array instead of pre-slicing/copying x_batch/y_batch per pair --
        # validate_corr_rows (candidate_kernels.pyx) takes `data` + `rows`
        # directly and does the slicing itself inside its own nogil loop, so
        # this Python loop only resolves the (string) series ids to integer
        # indices and packs the row tuple: cheaper than the old
        # slice-and-copy approach even before "point 1" is considered.
        #
        # "point 1": every candidate pair has at least one side anchored at
        # the just-arrived current window; validate_corr_rows can read that
        # side's sx/sum_xx/sum_xxx/sum_xxxx from a per-series cache instead
        # of re-accumulating them, entirely inside its nogil loop -- unlike
        # the earlier validate_corr_batch extension (still available, opt-in
        # via validation_current_window_cache, see _validate_pairs_standard),
        # no per-pair Python-level bookkeeping is needed here at all, since
        # validate_corr_rows already receives s1/s2/t1/t2/w per row and can
        # compute "is this side current" itself at C speed. This was the
        # human's explicit ask after the validate_corr_batch version showed
        # the Python-level bookkeeping cost often exceeded the Cython-level
        # saving. See docs/implementation_log.md.
        base_index = int(self.window_index[0])
        ids_lookup = self.series_ids
        data = self.window_data
        n_cols = data.shape[1]
        cw_cache_kwargs = self._current_window_cache_kwargs()

        batch_pairs = []
        batch_rows = []
        for pair in pairs:
            id1, id2, t1, t2, window_size = pair
            s1 = ids_lookup.get(id1)
            s2 = ids_lookup.get(id2)
            if s1 is None or s2 is None:
                continue
            start1 = t1 - base_index
            start2 = t2 - base_index
            if start1 < 0 or start2 < 0 or start1 + window_size > n_cols or start2 + window_size > n_cols:
                continue
            batch_pairs.append(pair)
            batch_rows.append((s1, s2, t1, t2, window_size))
            if len(batch_pairs) >= chunk_size:
                yield self._build_validation_payload(batch_pairs, batch_rows, base_index, data, cw_cache_kwargs)
                batch_pairs = []
                batch_rows = []

        if batch_pairs:
            yield self._build_validation_payload(batch_pairs, batch_rows, base_index, data, cw_cache_kwargs)

    def _build_validation_payload(self, batch_pairs, batch_rows, base_index, data, cw_cache_kwargs):
        payload = {
            "pairs": list(batch_pairs),
            "rows": np.asarray(batch_rows, dtype=np.int64).reshape((-1, 5)),
            "data": data,
            "base_index": base_index,
            "corr_threshold": self.corr_threshold,
            "neg_corr": self.neg_corr,
            "corr_val": True,
        }
        payload.update(cw_cache_kwargs)
        return payload

    def _buffered_emit_anomaly(self, key, marker):
        raise RuntimeError(
            "Legacy Python-object monitoring artifacts are disabled; "
            "use monitor_kernels.NumericMonitorState"
        )

    def _buffered_finalize_status(self, key, status):
        raise RuntimeError(
            "Legacy Python-object monitoring artifacts are disabled; "
            "use monitor_kernels.NumericMonitorState"
        )

    def _buffered_in_corr(self, pair, corr):
        raise RuntimeError(
            "Buffered monitoring uses monitor_kernels.NumericMonitorState; "
            "legacy Python-object monitoring is disabled"
        )

    def _monitor_corr_buffered(self):
        raise RuntimeError(
            "Buffered monitoring uses monitor_kernels.NumericMonitorState; "
            "legacy Python-object monitoring is disabled"
        )

    def _monitor_corr_buffered_numeric(self, rows, corrs):
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape((-1, 5)))
        corrs = np.ascontiguousarray(np.asarray(corrs, dtype=np.float64).reshape((-1,)))

        monitor_state = getattr(self, "_monitor_state", None)
        if monitor_state is None:
            if _cy_numeric_monitor_state_cls is None:
                raise RuntimeError(
                    "Monitoring requires compiled monitor_kernels.NumericMonitorState; "
                    "Python monitoring state is disabled"
                )
            monitor_state = _cy_numeric_monitor_state_cls()
            self._monitor_state = monitor_state
        if _cy_numeric_monitor_state_cls is None:
            raise RuntimeError(
                "Monitoring requires compiled monitor_kernels.NumericMonitorState; "
                "Python monitoring state is disabled"
            )
        t0 = time.perf_counter() if self.profile_enabled else None
        monitor_state.update(
            rows,
            corrs,
            int(self.window_step),
            bool(getattr(self, "_artifact_save_status", True)),
            bool(getattr(self, "_artifact_save_anomalies", True)),
        )
        if t0 is not None:
            self._profile_add("monitor.numeric_state_cython", time.perf_counter() - t0)
            # Diagnostic-only (2026-07-03): break down update() into its four
            # internal sections to localize the full-Cython monitor regression.
            # See docs/implementation_log.md / tasks/current_task.md.
            snap = monitor_state.profile_snapshot()
            prev = getattr(self, "_monitor_profile_prev", (0.0, 0.0, 0.0, 0.0, 0))
            self._profile_add("monitor.capacity_check", max(snap[0] - prev[0], 0.0))
            self._profile_add("monitor.row_loop", max(snap[1] - prev[1], 0.0))
            self._profile_add("monitor.closeout", max(snap[2] - prev[2], 0.0))
            self._profile_add("monitor.swap", max(snap[3] - prev[3], 0.0))
            self._profile_add_metric("monitor.active_count", int(monitor_state.active_count()))
            self._monitor_profile_prev = snap
            branches = monitor_state.row_branch_snapshot()
            bprev = getattr(self, "_monitor_branch_prev", (0, 0, 0, 0))
            self._profile_add_metric("monitor.rows_new", max(branches[0] - bprev[0], 0))
            self._profile_add_metric("monitor.rows_early_unchanged", max(branches[1] - bprev[1], 0))
            self._profile_add_metric("monitor.rows_extend", max(branches[2] - bprev[2], 0))
            self._profile_add_metric("monitor.rows_transition", max(branches[3] - bprev[3], 0))
            self._monitor_branch_prev = branches
            self._profile_add_metric("monitor.occupied_count", int(monitor_state.occupied_count()))
            self._profile_add_metric("monitor.capacity", int(monitor_state.capacity()))
        return True

    def _flush_correlated_chunk(self):
        if not getattr(self, "_artifact_save_correlated", True):
            return False
        if not self._spill_correlated and not getattr(self, "_spill_correlated_rows", []):
            return False
        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return False
        self._spill_correlated_chunk_index += 1
        path = f"{prefix}_correlated.chunk{self._spill_correlated_chunk_index:06d}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ids_by_idx = self._ids_by_numeric_index()
        id_to_code = {
            str(id_value): idx
            for idx, id_value in enumerate(ids_by_idx)
            if id_value is not None
        }
        rows = []
        for pair, corr in self._spill_correlated.items():
            id1_code = id_to_code.get(str(pair[0]))
            id2_code = id_to_code.get(str(pair[1]))
            if id1_code is None or id2_code is None:
                continue
            rows.append((int(id1_code), int(id2_code), int(pair[2]), int(pair[3]), corr, int(pair[4])))
        numeric_rows = getattr(self, "_spill_correlated_rows", [])
        if numeric_rows:
            for block in numeric_rows:
                if isinstance(block, tuple) and len(block) == 2:
                    block_rows = np.asarray(block[0], dtype=np.int64).reshape((-1, 5))
                    block_corrs = np.asarray(block[1], dtype=np.float64).reshape((-1,))
                    limit = min(block_rows.shape[0], block_corrs.shape[0])
                    row_iter = (
                        (
                            int(block_rows[i, 0]),
                            int(block_rows[i, 1]),
                            int(block_rows[i, 2]),
                            int(block_rows[i, 3]),
                            int(block_rows[i, 4]),
                            float(block_corrs[i]),
                        )
                        for i in range(limit)
                    )
                else:
                    block_rows = np.asarray(block)
                    if block_rows.ndim == 1:
                        block_rows = block_rows.reshape((1, -1))
                    row_iter = (
                        (
                            int(row[0]),
                            int(row[1]),
                            int(row[2]),
                            int(row[3]),
                            int(row[4]),
                            float(row[5]),
                        )
                        for row in block_rows
                        if len(row) >= 6
                    )
                for s1, s2, t1, t2, window_size, corr in row_iter:
                    if s1 < 0 or s2 < 0 or s1 >= len(ids_by_idx) or s2 >= len(ids_by_idx):
                        continue
                    if ids_by_idx[s1] is None or ids_by_idx[s2] is None:
                        continue
                    rows.append((int(s1), int(s2), int(t1), int(t2), float(corr), int(window_size)))
        rows = sorted(
            rows,
            key=lambda row: _window_sort_key((row[0], row[1], int(row[2]), int(row[3])), int(row[5])),
        )
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"])
            for row in rows:
                writer.writerow([row[0], row[1], int(row[2]), int(row[3]), row[4]])
        self._spill_correlated_chunks.append(path)
        self._spill_correlated = {}
        self._spill_correlated_rows = []
        self._spill_correlated_row_count = 0
        return True

    def _flush_status_chunk(self):
        if not getattr(self, "_artifact_save_status", True):
            return False
        monitor_state = getattr(self, "_monitor_state", None)
        rows = (
            monitor_state.take_status_rows()
            if monitor_state is not None
            else np.empty((0, 7), dtype=np.int64)
        )
        rows = np.asarray(rows, dtype=np.int64).reshape((-1, 7))
        if rows.shape[0] == 0:
            return False
        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return False
        ids_by_idx = self._ids_by_numeric_index()
        self._spill_status_chunk_index += 1
        path = f"{prefix}_status.chunk{self._spill_status_chunk_index:06d}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rows.shape[0] > 1:
            order = np.lexsort((rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
            rows = rows[order]
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "lag", "start_time_id1_idx", "start_time_id2_idx", "duration", "corr_sign"])
            for row in rows:
                sid1 = int(row[0])
                sid2 = int(row[1])
                id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
                id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
                writer.writerow([
                    id1,
                    id2,
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                ])
        self._spill_status_chunks.append(path)
        return True

    def _flush_anomaly_chunk(self):
        if not getattr(self, "_artifact_save_anomalies", True):
            return False
        monitor_state = getattr(self, "_monitor_state", None)
        rows = (
            monitor_state.take_anomaly_rows()
            if monitor_state is not None
            else np.empty((0, 5), dtype=np.int64)
        )
        rows = np.asarray(rows, dtype=np.int64).reshape((-1, 5))
        if rows.shape[0] == 0:
            return False
        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return False
        ids_by_idx = self._ids_by_numeric_index()
        self._spill_anomaly_chunk_index += 1
        path = f"{prefix}_anomalies.chunk{self._spill_anomaly_chunk_index:06d}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rows.shape[0] > 1:
            order = np.lexsort((rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
            rows = rows[order]
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "lag", "time_idx", "marker"])
            for row in rows:
                sid1 = int(row[0])
                sid2 = int(row[1])
                id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
                id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
                writer.writerow([id1, id2, int(row[2]), int(row[3]), int(row[4])])
        self._spill_anomaly_chunks.append(path)
        return True

    def _maybe_flush_artifact_buffers(self, force=False):
        if not self._artifact_buffered:
            return 0
        if not force and not self._artifact_runtime_flush_needed():
            return 0
        flushed = 0
        flushed += int(self._flush_correlated_chunk())
        flushed += int(self._flush_status_chunk())
        flushed += int(self._flush_anomaly_chunk())
        return flushed

    def _merge_sorted_csv_chunks(
        self,
        chunk_paths,
        output_csv,
        header,
        key_fn,
        row_transform=None,
        dedupe_key_fn=None,
        dedupe_row_fn=None,
    ):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        chunk_paths = [path for path in chunk_paths if path and os.path.exists(path)]

        def _iter_rows(path):
            with open(path, newline="") as file:
                reader = csv.reader(file, delimiter=CSV_DELIMITER)
                next(reader, None)
                for row in reader:
                    if row:
                        yield row

        def _write_sorted_rows(rows, out_path, out_header, transform_rows=False):
            with open(out_path, "w", newline="") as out_file:
                writer = csv.writer(out_file, delimiter=CSV_DELIMITER)
                writer.writerow(out_header)
                pending_row = None
                pending_dedupe_key = None
                for row in rows:
                    if dedupe_key_fn is None:
                        writer.writerow(
                            row_transform(row) if transform_rows and row_transform else row
                        )
                        continue
                    dedupe_key = dedupe_key_fn(row)
                    if pending_row is None:
                        pending_row = row
                        pending_dedupe_key = dedupe_key
                    elif dedupe_key == pending_dedupe_key:
                        if dedupe_row_fn is not None:
                            pending_row = dedupe_row_fn(pending_row, row)
                    else:
                        writer.writerow(
                            row_transform(pending_row)
                            if transform_rows and row_transform
                            else pending_row
                        )
                        pending_row = row
                        pending_dedupe_key = dedupe_key
                if pending_row is not None:
                    writer.writerow(
                        row_transform(pending_row)
                        if transform_rows and row_transform
                        else pending_row
                    )

        def _merge_two(left_path, right_path):
            left_iter = _iter_rows(left_path)
            right_iter = _iter_rows(right_path)
            left_row = next(left_iter, None)
            right_row = next(right_iter, None)
            while left_row is not None or right_row is not None:
                if right_row is None:
                    yield left_row
                    left_row = next(left_iter, None)
                elif left_row is None:
                    yield right_row
                    right_row = next(right_iter, None)
                elif key_fn(left_row) <= key_fn(right_row):
                    yield left_row
                    left_row = next(left_iter, None)
                else:
                    yield right_row
                    right_row = next(right_iter, None)

        if not chunk_paths:
            with open(output_csv, "w", newline="") as out_file:
                writer = csv.writer(out_file, delimiter=CSV_DELIMITER)
                writer.writerow(header)
            return

        active_paths = list(chunk_paths)
        temp_paths = []
        pass_index = 0
        while len(active_paths) > 1:
            pass_index += 1
            next_paths = []
            for i in range(0, len(active_paths), 2):
                left_path = active_paths[i]
                if i + 1 >= len(active_paths):
                    next_paths.append(left_path)
                    continue
                right_path = active_paths[i + 1]
                temp_path = (
                    f"{output_csv}.merge{pass_index:03d}_{len(next_paths):06d}.tmp"
                )
                _write_sorted_rows(_merge_two(left_path, right_path), temp_path, header)
                temp_paths.append(temp_path)
                next_paths.append(temp_path)
                for old_path in (left_path, right_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
            active_paths = next_paths

        try:
            _write_sorted_rows(_iter_rows(active_paths[0]), output_csv, header, transform_rows=True)
        finally:
            for path in set(active_paths + temp_paths + chunk_paths):
                if path == output_csv:
                    continue
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _write_artifact_chunk_manifest(self, prefix):
        manifest_path = f"{prefix}_artifact_chunks.json"
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        correlated_csv = f"{prefix}_correlated.csv"
        status_csv = f"{prefix}_status.csv"
        anomalies_csv = f"{prefix}_anomalies.csv"
        codebook_path = _correlated_codebook_path(correlated_csv)

        if getattr(self, "_artifact_save_correlated", True):
            ids_by_idx = self._ids_by_numeric_index()
            code_map = {
                str(id_value): int(idx)
                for idx, id_value in enumerate(ids_by_idx)
                if id_value is not None
            }
            _write_correlated_codebook(correlated_csv, code_map)
        else:
            codebook_path = ""

        manifest = {
            "merge_mode": "chunks",
            "prefix": prefix,
            "csv_delimiter": CSV_DELIMITER,
            "targets": {
                "correlated": correlated_csv,
                "status": status_csv,
                "anomalies": anomalies_csv,
            },
            "codebook_path": codebook_path,
            "chunks": {
                "correlated": list(getattr(self, "_spill_correlated_chunks", [])),
                "status": list(getattr(self, "_spill_status_chunks", [])),
                "anomalies": list(getattr(self, "_spill_anomaly_chunks", [])),
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        return manifest_path

    def _finalize_buffered_artifacts(self):
        if not self._artifact_buffered:
            return
        monitor_state = getattr(self, "_monitor_state", None)
        if monitor_state is not None:
            monitor_state.finalize(bool(getattr(self, "_artifact_save_status", True)))
        self._maybe_flush_artifact_buffers(force=True)

        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return

        if getattr(self, "_artifact_merge_mode", "merged") == "chunks":
            self._write_artifact_chunk_manifest(prefix)
            self._spill_correlated_chunks = []
            self._spill_status_chunks = []
            self._spill_anomaly_chunks = []
            return

        correlated_csv = f"{prefix}_correlated.csv"
        if getattr(self, "_artifact_save_correlated", True) and self._spill_correlated_chunks:
            code_map = {}

            def _code_for(id_value):
                key = str(id_value)
                code = code_map.get(key)
                if code is None:
                    code = len(code_map)
                    code_map[key] = code
                return code

            self._merge_sorted_csv_chunks(
                self._spill_correlated_chunks,
                correlated_csv,
                ["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"],
                key_fn=lambda row: _window_sort_key((row[0], row[1], int(row[2]), int(row[3])), self.window_size),
                dedupe_key_fn=lambda row: _normalize_bf_key((row[0], row[1], int(row[2]), int(row[3]), self.window_size)),
                dedupe_row_fn=self._choose_stronger_correlated_row,
                row_transform=lambda row: [
                    _code_for(row[0]),
                    _code_for(row[1]),
                    row[2],
                    row[3],
                    row[4],
                ],
            )
            _write_correlated_codebook(correlated_csv, code_map)
        elif getattr(self, "_artifact_save_correlated", True) and not os.path.exists(correlated_csv):
            with open(correlated_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"])
            _write_correlated_codebook(correlated_csv, {})

        status_csv = f"{prefix}_status.csv"
        if getattr(self, "_artifact_save_status", True) and self._spill_status_chunks:
            self._merge_sorted_csv_chunks(
                self._spill_status_chunks,
                status_csv,
                ["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"],
                key_fn=lambda row: _status_sort_key((row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]))),
                row_transform=lambda row: [
                    row[0],
                    row[1],
                    row[2],
                    self._safe_index_to_datetime(int(row[3])),
                    self._safe_index_to_datetime(int(row[4])),
                    row[5],
                    row[6],
                ],
            )
        elif getattr(self, "_artifact_save_status", True) and not os.path.exists(status_csv):
            with open(status_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"])

        anomalies_csv = f"{prefix}_anomalies.csv"
        if getattr(self, "_artifact_save_anomalies", True) and self._spill_anomaly_chunks:
            self._merge_sorted_csv_chunks(
                self._spill_anomaly_chunks,
                anomalies_csv,
                ["id1", "id2", "lag", "time_idx", "time", "anomaly"],
                key_fn=lambda row: _anomaly_sort_key((row[0], row[1], int(row[2]), int(row[3]), int(row[4]))),
                dedupe_key_fn=lambda row: (
                    _normalize_pair_lag_key(row[0], row[1], int(row[2]))
                    + (int(row[3]), int(row[4]))
                ),
                row_transform=lambda row: [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    self._safe_index_to_datetime(int(row[3])),
                    _marker_to_label(int(row[4])),
                ],
            )
        elif getattr(self, "_artifact_save_anomalies", True) and not os.path.exists(anomalies_csv):
            with open(anomalies_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "lag", "time_idx", "time", "anomaly"])

        self._spill_correlated_chunks = []
        self._spill_status_chunks = []
        self._spill_anomaly_chunks = []

    @staticmethod
    def _choose_stronger_correlated_row(existing_row, new_row):
        try:
            existing_corr = abs(float(existing_row[4]))
        except (TypeError, ValueError, IndexError):
            existing_corr = float("-inf")
        try:
            new_corr = abs(float(new_row[4]))
        except (TypeError, ValueError, IndexError):
            new_corr = float("-inf")
        return new_row if new_corr > existing_corr else existing_row

    def _reset_hybrid_validation_runtime(self):
        self._hybrid_validation_rate_ema = 0.0
        self._hybrid_validation_active = False
        cache = getattr(self, "_hybrid_validation_cache", None)
        if cache is not None:
            try:
                cache.clear()
            except Exception:
                self._hybrid_validation_cache = None

    def _validate_pairs_hybrid_cython(self, pairs, retain_validated, track_min_dist):
        cache = getattr(self, "_hybrid_validation_cache", None)
        if cache is None or _cy_hybrid_cache_cls is None:
            return None
        pairs = list(pairs)
        if not pairs:
            return 0, 0, []
        if self.window_data is None or self.window_index is None or len(self.window_index) == 0:
            return None

        n_items = len(pairs)
        series1 = np.empty(n_items, dtype=np.int64)
        series2 = np.empty(n_items, dtype=np.int64)
        curr_t1 = np.empty(n_items, dtype=np.int64)
        curr_t2 = np.empty(n_items, dtype=np.int64)
        window_sizes = np.empty(n_items, dtype=np.int64)
        valid_pairs = []
        fallback_pairs = []
        ids_lookup = self.series_ids

        for pair in pairs:
            try:
                id1, id2, t1, t2, window_size = pair
                idx = len(valid_pairs)
                series1[idx] = int(ids_lookup[id1])
                series2[idx] = int(ids_lookup[id2])
                curr_t1[idx] = int(t1)
                curr_t2[idx] = int(t2)
                window_sizes[idx] = int(window_size)
                valid_pairs.append(pair)
            except (KeyError, TypeError, ValueError, IndexError):
                fallback_pairs.append(pair)

        if not valid_pairs:
            return 0, 0, fallback_pairs

        n_valid = len(valid_pairs)
        # (2026-07-06) "point 1": every candidate pair has at least one side
        # anchored at the just-arrived current window. self._cw_raw_sums*
        # lets validate_pairs skip re-accumulating that side's
        # sx/sx2/sx3/sx4 across the O(w) validation loop for every such
        # pair. Recomputed lazily here (at most once per step, only when
        # this method actually runs with non-empty pairs), not
        # unconditionally in _update_curr_data -- this method is the only
        # consumer, and it is itself only ever called when
        # hybrid_validation is enabled, so an unconditional per-step
        # recompute would be pure wasted work otherwise. See
        # docs/implementation_log.md. Gated by validation_current_window_cache
        # so it can be A/B'd or disabled outright.
        use_cw_cache_opt = getattr(self, "validation_current_window_cache", False)
        if use_cw_cache_opt and not getattr(self, "_cw_raw_moments_fresh", False):
            self._recompute_current_window_raw_moments()
            self._cw_raw_moments_fresh = True
        current_window_kwargs = {}
        if (
            use_cw_cache_opt
            and self._cw_raw_sums is not None
            and self._cw_raw_sums_sq is not None
            and self._cw_raw_sums_cu is not None
            and self._cw_raw_sums_qu is not None
            and self._cw_raw_sums.shape[0] == self.window_data.shape[0]
            and self.curr_window_size
        ):
            current_window_kwargs = {
                "current_window_sums": self._cw_raw_sums,
                "current_window_sums_sq": self._cw_raw_sums_sq,
                "current_window_sums_cu": self._cw_raw_sums_cu,
                "current_window_sums_qu": self._cw_raw_sums_qu,
                "current_window_size": int(self.curr_window_size),
            }
        try:
            result = cache.validate_pairs(
                np.asarray(self.window_data, dtype=np.float64),
                series1[:n_valid],
                series2[:n_valid],
                curr_t1[:n_valid],
                curr_t2[:n_valid],
                window_sizes[:n_valid],
                int(self.window_index[0]),
                int(self.window_step),
                float(self.corr_threshold),
                bool(self.neg_corr),
                float(getattr(self, "hybrid_validation_min_repeat_rate", 0.25)),
                float(getattr(self, "hybrid_validation_disable_rate", 0.125)),
                float(getattr(self, "hybrid_validation_ema_alpha", 0.25)),
                int(getattr(self, "hybrid_validation_min_candidates", 256)),
                1e-3,
                5.0,
                **current_window_kwargs,
            )
        except Exception:
            return None

        active = bool(result.get("active", False))
        self._hybrid_validation_active = active
        self._hybrid_validation_rate_ema = float(result.get("repeat_rate_ema", 0.0))
        if active:
            self.hybrid_validation_attempts += int(result.get("repeat_count", 0) or 0)
            self.hybrid_validation_steps_active += 1

        tested = 0
        validated = 0
        accepted_now = {}
        if track_min_dist:
            min_dist, min_pair = self.min_dist, self.pair_min_dist

        for pair, entry in zip(valid_pairs, result.get("results", ())):
            ok = bool(entry[0])
            if not ok:
                fallback_pairs.append(pair)
                continue
            is_correlated = bool(entry[1])
            corr = float(entry[2])
            dist = float(entry[3])
            is_constant = bool(entry[4])
            used_hybrid = bool(entry[6]) if len(entry) > 6 else False
            tested += 1
            if used_hybrid:
                self.hybrid_validation_hits += 1
            if is_constant:
                self.constant_candidates += 1
            if track_min_dist and dist < min_dist:
                min_dist, min_pair = dist, pair
            if is_correlated:
                validated += 1
                accepted_now[pair] = corr

        self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
        if track_min_dist:
            self.min_dist, self.pair_min_dist = min_dist, min_pair
        return tested, validated, fallback_pairs

    def _validate_pairs_standard(self, pairs, worker_mode, retain_validated, track_min_dist):
        pairs = list(pairs)
        if not pairs:
            return 0, 0

        if worker_mode == "sequential":
            if _cy_validate_corr_rows is not None:
                chunk_size = 256
                tested = validated = 0
                if track_min_dist:
                    min_dist, min_pair = self.min_dist, self.pair_min_dist
                eff_std_thresh = 1e-3
                had_payload = False
                for payload in self._iter_validation_payloads(pairs, chunk_size):
                    had_payload = True
                    accepted_mask, corrs, dists, _consts, _spikes = _cy_validate_corr_rows(
                        np.ascontiguousarray(payload["data"], dtype=np.float64),
                        payload["rows"],
                        int(payload["base_index"]),
                        float(self.corr_threshold),
                        bool(self.neg_corr),
                        eff_std_thresh,
                        5.0,
                        current_window_sums=payload.get("current_window_sums"),
                        current_window_sums_sq=payload.get("current_window_sums_sq"),
                        current_window_sums_cu=payload.get("current_window_sums_cu"),
                        current_window_sums_qu=payload.get("current_window_sums_qu"),
                        current_window_size=payload.get("current_window_size", -1),
                    )
                    accepted_now = {}
                    for i, pair in enumerate(payload["pairs"]):
                        pair_corr = float(corrs[i])
                        pair_dist = float(dists[i])
                        tested += 1
                        if track_min_dist and pair_dist < min_dist:
                            min_dist, min_pair = pair_dist, pair
                        if accepted_mask[i]:
                            validated += 1
                            accepted_now[pair] = pair_corr
                    self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
                if not had_payload:
                    return 0, 0
                if track_min_dist:
                    self.min_dist, self.pair_min_dist = min_dist, min_pair
                return tested, validated

            tested = validated = 0
            if track_min_dist:
                min_dist, min_pair = self.min_dist, self.pair_min_dist
            accepted_now = {}
            for pair in pairs:
                pair, is_corr, corr, dist = self._validate_corr(pair, True)
                tested += 1
                if track_min_dist and dist < min_dist:
                    min_dist, min_pair = dist, pair
                if is_corr:
                    validated += 1
                    accepted_now[pair] = corr
            self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
            if track_min_dist:
                self.min_dist, self.pair_min_dist = min_dist, min_pair
            return tested, validated

        chunk_size = 256
        payload_iter = self._iter_validation_payloads(pairs, chunk_size)
        first_payload = next(payload_iter, None)
        if first_payload is None:
            return 0, 0

        tested = validated = 0
        if track_min_dist:
            min_dist, min_pair = self.min_dist, self.pair_min_dist
        max_workers = max(1, int(self.n_nodes or 1))
        in_flight = {}
        pool = self._get_thread_pool(max_workers)

        def _submit(payload):
            future = pool.submit(_corr_validation_batch_worker, payload)
            in_flight[future] = None

        _submit(first_payload)
        for _ in range(max_workers * 2 - 1):
            payload = next(payload_iter, None)
            if payload is None:
                break
            _submit(payload)

        while in_flight:
            future = next(as_completed(tuple(in_flight)))
            del in_flight[future]
            batch = future.result()
            accepted_now = {}
            for pair, is_correlated, corr, dist, is_constant, _ in batch:
                tested += 1
                if is_constant:
                    self.constant_candidates += 1
                if track_min_dist and dist < min_dist:
                    min_dist, min_pair = dist, pair
                if is_correlated:
                    validated += 1
                    accepted_now[pair] = corr
            self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
            next_payload = next(payload_iter, None)
            if next_payload is not None:
                _submit(next_payload)

        if track_min_dist:
            self.min_dist, self.pair_min_dist = min_dist, min_pair
        return tested, validated

    def _ids_by_numeric_index(self):
        n_ids = len(self.series_ids)
        ids_by_idx = [None] * n_ids
        for sid, idx in self.series_ids.items():
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_int < n_ids:
                ids_by_idx[idx_int] = sid
        return ids_by_idx

    def _numeric_row_to_pair(self, row, ids_by_idx=None):
        if ids_by_idx is None:
            ids_by_idx = self._ids_by_numeric_index()
        s1 = int(row[0])
        s2 = int(row[1])
        if s1 < 0 or s2 < 0 or s1 >= len(ids_by_idx) or s2 >= len(ids_by_idx):
            return None
        id1 = ids_by_idx[s1]
        id2 = ids_by_idx[s2]
        if id1 is None or id2 is None:
            return None
        return _normalize_bf_key((id1, id2, int(row[2]), int(row[3]), int(row[4])))

    def _numeric_rows_to_pair_map(self, rows, corrs):
        t0 = time.perf_counter()
        ids_by_idx = self._ids_by_numeric_index()
        rows = np.asarray(rows, dtype=np.int64).reshape((-1, 5))
        if corrs is None:
            corr_values = np.ones(rows.shape[0], dtype=np.float64)
        else:
            corr_values = np.asarray(corrs, dtype=np.float64).reshape((-1,))
        out = {}
        limit = min(rows.shape[0], corr_values.shape[0])
        for i in range(limit):
            pair = self._numeric_row_to_pair(rows[i], ids_by_idx=ids_by_idx)
            if pair is not None:
                out[pair] = float(corr_values[i])
        self.artifact_bookkeeping_time += time.perf_counter() - t0
        return out

    @staticmethod
    def _numeric_corr_arrays(rows, corrs=None):
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape((-1, 5)))
        if corrs is None:
            corr_values = np.ones(rows.shape[0], dtype=np.float64)
        else:
            corr_values = np.ascontiguousarray(np.asarray(corrs, dtype=np.float64).reshape((-1,)))
        limit = min(rows.shape[0], corr_values.shape[0])
        if limit <= 0:
            return np.empty((0, 5), dtype=np.int64), np.empty((0,), dtype=np.float64)
        return (
            np.ascontiguousarray(rows[:limit], dtype=np.int64),
            np.ascontiguousarray(corr_values[:limit], dtype=np.float64),
        )

    def _spill_correlated_numeric(self, rows, corr_values):
        if not getattr(self, "_artifact_save_correlated", True):
            return 0
        t0 = time.perf_counter()
        rows, corr_values = self._numeric_corr_arrays(rows, corr_values)
        n_rows = int(rows.shape[0])
        if n_rows:
            self._spill_correlated_rows.append((rows, corr_values))
            self._spill_correlated_row_count = int(getattr(self, "_spill_correlated_row_count", 0) or 0) + n_rows
        self.artifact_bookkeeping_time += time.perf_counter() - t0
        return n_rows

    def _update_maxlag_state_numeric(self, rows, corr_values):
        if not getattr(self, "_artifact_save_maxlag", True):
            return
        t0 = time.perf_counter()
        rows, corr_values = self._numeric_corr_arrays(rows, corr_values)
        ids_by_idx = self._ids_by_numeric_index()
        for row, corr in zip(rows, corr_values):
            pair = self._numeric_row_to_pair(row, ids_by_idx=ids_by_idx)
            if pair is not None:
                self._update_maxlag_state(pair, float(corr))
        self.artifact_bookkeeping_time += time.perf_counter() - t0

    def _record_correlated_numeric(self, rows, corrs=None, retain_validated=True):
        rows, corr_values = self._numeric_corr_arrays(rows, corrs)
        n_rows = int(rows.shape[0])
        if n_rows == 0:
            return 0

        buffered = bool(getattr(self, "_artifact_buffered", False))
        direct_correlated_artifact = buffered and bool(getattr(self, "_artifact_save_correlated", True))

        if direct_correlated_artifact:
            self._spill_correlated_numeric(rows, corr_values)

        if retain_validated:
            self._append_validated_numeric_step(rows, corr_values)

        needs_pair_map = (
            (bool(retain_validated) and not buffered)
            or bool(getattr(self, "_step_observer_enabled", False))
            or (not buffered and bool(getattr(self, "_artifact_save_correlated", True)))
        )

        accepted_map = None
        if needs_pair_map:
            accepted_map = self._numeric_rows_to_pair_map(rows, corr_values)
            if accepted_map:
                self._record_correlated_batch(
                    accepted_map,
                    retain_validated=retain_validated,
                    record_artifact=not direct_correlated_artifact,
                )

        if direct_correlated_artifact:
            self._update_maxlag_state_numeric(rows, corr_values)
        elif accepted_map is None and getattr(self, "_artifact_save_maxlag", True):
            self._update_maxlag_state_numeric(rows, corr_values)

        return len(accepted_map) if accepted_map is not None else n_rows

    def _reset_validated_numeric_step(self):
        self._validated_numeric_count_step = 0

    def _append_validated_numeric_step(self, rows, corr_values):
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape((-1, 5)))
        n_rows = int(rows.shape[0])
        if n_rows == 0:
            return
        corr_values = np.ascontiguousarray(np.asarray(corr_values, dtype=np.float64).reshape((-1,)))
        if corr_values.shape[0] != n_rows:
            fixed = np.ones((n_rows,), dtype=np.float64)
            copy_n = min(int(corr_values.shape[0]), n_rows)
            if copy_n > 0:
                fixed[:copy_n] = corr_values[:copy_n]
            corr_values = fixed
        count = int(getattr(self, "_validated_numeric_count_step", 0) or 0)
        capacity = int(getattr(self, "_validated_numeric_capacity_step", 0) or 0)
        need = count + n_rows
        if capacity < need or self._validated_numeric_rows_step is None:
            new_capacity = max(16, capacity if capacity > 0 else 0)
            while new_capacity < need:
                new_capacity *= 2
            new_rows = np.empty((new_capacity, 5), dtype=np.int64)
            new_corrs = np.empty((new_capacity,), dtype=np.float64)
            if count > 0 and self._validated_numeric_rows_step is not None:
                new_rows[:count, :] = self._validated_numeric_rows_step[:count, :]
                new_corrs[:count] = self._validated_numeric_corrs_step[:count]
            self._validated_numeric_rows_step = new_rows
            self._validated_numeric_corrs_step = new_corrs
            self._validated_numeric_capacity_step = new_capacity
        self._validated_numeric_rows_step[count:need, :] = rows
        self._validated_numeric_corrs_step[count:need] = corr_values
        self._validated_numeric_count_step = need

    def _iter_validated_numeric_step(self):
        rows = getattr(self, "_validated_numeric_rows_step", None)
        corrs = getattr(self, "_validated_numeric_corrs_step", None)
        count = int(getattr(self, "_validated_numeric_count_step", 0) or 0)
        if rows is None or count <= 0:
            return None, None
        return (
            np.ascontiguousarray(rows[:count, :], dtype=np.int64).reshape((-1, 5)),
            np.ascontiguousarray(corrs[:count], dtype=np.float64).reshape((-1,)),
        )

    def _get_validated_corr_numeric(self, rows, corr_val=True, retain_validated=True):
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape((-1, 5)))
        n_pairs = int(rows.shape[0])
        if n_pairs == 0:
            if getattr(self, "hybrid_validation", False):
                self._reset_hybrid_validation_runtime()
            return

        self.total_candidates += n_pairs

        if not corr_val:
            if getattr(self, "_online_window_metrics_only", False) and not retain_validated:
                self.tested_candidates += n_pairs
                self.validated_candidates += n_pairs
                return
            self._record_correlated_numeric(rows, None, retain_validated=retain_validated)
            self.tested_candidates += n_pairs
            self.validated_candidates += n_pairs
            return

        track_min_dist = bool(getattr(self, "track_min_dist", True))
        if not track_min_dist:
            self.min_dist = np.inf
            self.pair_min_dist = None

        if getattr(self, "hybrid_validation", False):
            hybrid_min_candidates = int(getattr(self, "hybrid_validation_min_candidates", 256) or 0)
            cache = getattr(self, "_hybrid_validation_cache", None)
            if (
                cache is not None
                and _cy_hybrid_cache_cls is not None
                and (hybrid_min_candidates <= 0 or n_pairs >= hybrid_min_candidates)
            ):
                try:
                    result = cache.validate_pairs(
                        np.ascontiguousarray(self.window_data, dtype=np.float64),
                        rows[:, 0],
                        rows[:, 1],
                        rows[:, 2],
                        rows[:, 3],
                        rows[:, 4],
                        int(self.window_index[0]),
                        int(self.window_step),
                        float(self.corr_threshold),
                        bool(self.neg_corr),
                        float(getattr(self, "hybrid_validation_min_repeat_rate", 0.25)),
                        float(getattr(self, "hybrid_validation_disable_rate", 0.125)),
                        float(getattr(self, "hybrid_validation_ema_alpha", 0.25)),
                        int(getattr(self, "hybrid_validation_min_candidates", 256)),
                        1e-3,
                        5.0,
                        **self._current_window_cache_kwargs(),
                    )
                except Exception:
                    result = None
                if result is not None:
                    active = bool(result.get("active", False))
                    self._hybrid_validation_active = active
                    self._hybrid_validation_rate_ema = float(result.get("repeat_rate_ema", 0.0))
                    if active:
                        self.hybrid_validation_attempts += int(result.get("repeat_count", 0) or 0)
                        self.hybrid_validation_steps_active += 1
                    tested = 0
                    accepted_indices = []
                    accepted_corrs = []
                    if track_min_dist:
                        min_dist, min_pair_idx = self.min_dist, None
                    for idx, entry in enumerate(result.get("results", ())):
                        ok = bool(entry[0])
                        if not ok:
                            continue
                        tested += 1
                        is_correlated = bool(entry[1])
                        corr = float(entry[2])
                        dist = float(entry[3])
                        is_constant = bool(entry[4])
                        used_hybrid = bool(entry[6]) if len(entry) > 6 else False
                        if used_hybrid:
                            self.hybrid_validation_hits += 1
                        if is_constant:
                            self.constant_candidates += 1
                        if track_min_dist and dist < min_dist:
                            min_dist = dist
                            min_pair_idx = idx
                        if is_correlated:
                            accepted_indices.append(idx)
                            accepted_corrs.append(corr)
                    if track_min_dist and min_pair_idx is not None:
                        self.min_dist = float(min_dist)
                        self.pair_min_dist = self._numeric_row_to_pair(rows[min_pair_idx])
                    if accepted_indices:
                        self._record_correlated_numeric(
                            rows[np.asarray(accepted_indices, dtype=np.int64)],
                            np.asarray(accepted_corrs, dtype=np.float64),
                            retain_validated=retain_validated,
                        )
                    self.tested_candidates += tested
                    self.validated_candidates += len(accepted_indices)
                    return
            if hybrid_min_candidates <= 0 or n_pairs >= hybrid_min_candidates:
                self._reset_hybrid_validation_runtime()
            else:
                self._hybrid_validation_active = False

        if _cy_validate_corr_rows is None:
            ids_by_idx = self._ids_by_numeric_index()
            pairs = [self._numeric_row_to_pair(row, ids_by_idx=ids_by_idx) for row in rows]
            pairs = [pair for pair in pairs if pair is not None]
            tested, validated = self._validate_pairs_standard(
                pairs,
                self.exec,
                retain_validated,
                track_min_dist,
            )
            self.tested_candidates += tested
            self.validated_candidates += validated
            return

        t0 = time.perf_counter() if getattr(self, "profile_enabled", False) else None
        accepted_mask, corrs, dists, _constants, _spiked = _cy_validate_corr_rows(
            np.ascontiguousarray(self.window_data, dtype=np.float64),
            rows,
            int(self.window_index[0]),
            float(self.corr_threshold),
            bool(self.neg_corr),
            1e-3,
            5.0,
            **self._current_window_cache_kwargs(),
        )
        if t0 is not None:
            self._profile_add("val.numeric_rows_kernel", time.perf_counter() - t0)

        accepted_mask = np.asarray(accepted_mask, dtype=np.uint8)
        corrs = np.asarray(corrs, dtype=np.float64)
        dists = np.asarray(dists, dtype=np.float64)

        if track_min_dist and dists.size:
            finite = np.isfinite(dists)
            if finite.any():
                finite_idx = np.flatnonzero(finite)
                local_idx = int(finite_idx[int(np.argmin(dists[finite_idx]))])
                local_dist = float(dists[local_idx])
                if local_dist < self.min_dist:
                    self.min_dist = local_dist
                    self.pair_min_dist = self._numeric_row_to_pair(rows[local_idx])

        accepted_idx = np.flatnonzero(accepted_mask != 0)
        if accepted_idx.size:
            self._record_correlated_numeric(rows[accepted_idx], corrs[accepted_idx], retain_validated=retain_validated)

        self.tested_candidates += n_pairs
        self.validated_candidates += int(accepted_idx.size)
    
    def _get_validated_corr(self, corr_val=True, force_mode=None, retain_validated=True):
        self.validated = {}
        self._reset_validated_numeric_step()
        if getattr(self, "_step_observer_enabled", False):
            self._validated_step = {}
        numeric_rows = getattr(self, "_candidate_numeric_rows", None)
        if numeric_rows is not None:
            self._last_candidate_numeric_rows = np.ascontiguousarray(
                np.asarray(numeric_rows, dtype=np.int64).reshape((-1, 5))
            )
            self._get_validated_corr_numeric(
                self._last_candidate_numeric_rows,
                corr_val=corr_val,
                retain_validated=retain_validated,
            )
            self._candidate_numeric_rows = None
            return
        self._last_candidate_numeric_rows = None
        if not self.candidates:
            if getattr(self, "hybrid_validation", False):
                self._reset_hybrid_validation_runtime()
            return

        candidates = self.candidates
        n_pairs = len(candidates)

        # Count candidates once, for both sequential and parallel
        self.total_candidates += n_pairs

        if not corr_val:
            if getattr(self, "hybrid_validation", False):
                self._reset_hybrid_validation_runtime()
            if getattr(self, "_online_window_metrics_only", False) and not retain_validated:
                self.tested_candidates += n_pairs
                self.validated_candidates += n_pairs
                return
            if getattr(self, "_artifact_buffered", False):
                if retain_validated:
                    validated_now = dict.fromkeys(candidates, 1.0)
                    self.validated = validated_now
                    self._record_correlated_batch(validated_now, retain_validated=False)
                else:
                    t0 = time.perf_counter()
                    spill = self._spill_correlated
                    update_maxlag = self._update_maxlag_state
                    for pair in candidates:
                        spill[pair] = 1.0
                        update_maxlag(pair, 1.0)
                    self.artifact_bookkeeping_time += time.perf_counter() - t0
            else:
                validated_now = dict.fromkeys(candidates, 1.0)
                if retain_validated:
                    self.validated = validated_now
                if getattr(self, "_artifact_save_correlated", True):
                    self.correlated.update(validated_now)
                if getattr(self, "_artifact_save_maxlag", True):
                    for pair in candidates:
                        self._update_maxlag_state(pair, 1.0)
            self.tested_candidates += n_pairs
            self.validated_candidates += n_pairs
            return

        track_min_dist = bool(getattr(self, "track_min_dist", True))
        if not track_min_dist:
            self.min_dist = np.inf
            self.pair_min_dist = None

        worker_mode = force_mode if force_mode else self.exec
        pairs = list(candidates.keys())
        if getattr(self, "hybrid_validation", False):
            min_candidates = int(getattr(self, "hybrid_validation_min_candidates", 256) or 0)
            if min_candidates <= 0 or n_pairs >= min_candidates:
                hybrid_result = self._validate_pairs_hybrid_cython(
                    pairs,
                    retain_validated,
                    track_min_dist,
                )
                if hybrid_result is not None:
                    hybrid_tested, hybrid_validated, fallback_pairs = hybrid_result
                    self.tested_candidates += hybrid_tested
                    self.validated_candidates += hybrid_validated
                    if not fallback_pairs:
                        return
                    pairs = fallback_pairs
                else:
                    self._reset_hybrid_validation_runtime()
            else:
                self._hybrid_validation_active = False

        if worker_mode == "sequential":
            if _cy_validate_corr_rows is not None:
                chunk_size = 256
                tested = validated = 0
                if track_min_dist:
                    min_dist, min_pair = self.min_dist, self.pair_min_dist
                eff_std_thresh = 1e-3
                had_payload = False
                for payload in self._iter_validation_payloads(pairs, chunk_size):
                    had_payload = True
                    accepted_mask, corrs, dists, _consts, _spikes = _cy_validate_corr_rows(
                        np.ascontiguousarray(payload["data"], dtype=np.float64),
                        payload["rows"],
                        int(payload["base_index"]),
                        float(self.corr_threshold),
                        bool(self.neg_corr),
                        eff_std_thresh,
                        5.0,
                        current_window_sums=payload.get("current_window_sums"),
                        current_window_sums_sq=payload.get("current_window_sums_sq"),
                        current_window_sums_cu=payload.get("current_window_sums_cu"),
                        current_window_sums_qu=payload.get("current_window_sums_qu"),
                        current_window_size=payload.get("current_window_size", -1),
                    )
                    accepted_now = {}
                    for i, pair in enumerate(payload["pairs"]):
                        pair_corr = float(corrs[i])
                        pair_dist = float(dists[i])
                        tested += 1
                        if track_min_dist and pair_dist < min_dist:
                            min_dist, min_pair = pair_dist, pair
                        if accepted_mask[i]:
                            validated += 1
                            accepted_now[pair] = pair_corr
                    self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
                if not had_payload:
                    return
                self.tested_candidates += tested
                self.validated_candidates += validated
                if track_min_dist:
                    self.min_dist, self.pair_min_dist = min_dist, min_pair
                return

            tested = validated = 0
            if track_min_dist:
                min_dist, min_pair = self.min_dist, self.pair_min_dist
            accepted_now = {}
            for pair in pairs:
                pair, is_corr, corr, dist = self._validate_corr(pair, True)
                tested += 1
                if track_min_dist and dist < min_dist:
                    min_dist, min_pair = dist, pair
                if is_corr:
                    validated += 1
                    accepted_now[pair] = corr
            self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
            self.tested_candidates += tested
            self.validated_candidates += validated
            if track_min_dist:
                self.min_dist, self.pair_min_dist = min_dist, min_pair
            return

        chunk_size = 256
        t0 = time.perf_counter() if self.profile_enabled else None
        payload_iter = self._iter_validation_payloads(pairs, chunk_size)
        first_payload = next(payload_iter, None)
        if t0 is not None:
            self._profile_add("val.build_items", time.perf_counter() - t0)

        if first_payload is None:
            return

        tested = validated = 0
        if track_min_dist:
            min_dist, min_pair = self.min_dist, self.pair_min_dist
        t0 = time.perf_counter() if self.profile_enabled else None
        max_workers = max(1, int(self.n_nodes or 1))
        in_flight = {}
        pool = self._get_thread_pool(max_workers)

        def _submit(payload):
            future = pool.submit(_corr_validation_batch_worker, payload)
            in_flight[future] = None

        _submit(first_payload)
        for _ in range(max_workers * 2 - 1):
            payload = next(payload_iter, None)
            if payload is None:
                break
            _submit(payload)

        while in_flight:
            future = next(as_completed(tuple(in_flight)))
            del in_flight[future]
            batch = future.result()
            accepted_now = {}
            for pair, is_correlated, corr, dist, is_constant, _ in batch:
                tested += 1
                if is_constant:
                    self.constant_candidates += 1
                if track_min_dist and dist < min_dist:
                    min_dist, min_pair = dist, pair
                if is_correlated:
                    validated += 1
                    accepted_now[pair] = corr
            self._record_correlated_batch(accepted_now, retain_validated=retain_validated)
            next_payload = next(payload_iter, None)
            if next_payload is not None:
                _submit(next_payload)
        if t0 is not None:
            self._profile_add("val.merge", time.perf_counter() - t0)

        self.tested_candidates += tested
        self.validated_candidates += validated
        if track_min_dist:
            self.min_dist, self.pair_min_dist = min_dist, min_pair

    def _monitor_corr(self, worker_mode=None):
        """
        Monitoring update.

        Monitoring requires the numeric Cython monitor state. Python monitoring
        fallback is disabled.
        """
        if _cy_numeric_monitor_state_cls is None:
            raise RuntimeError(
                "Monitoring requires compiled monitor_kernels.NumericMonitorState; "
                "Python monitoring state is disabled"
            )
        numeric_rows, numeric_corrs = self._iter_validated_numeric_step()
        if numeric_rows is None:
            monitor_state = getattr(self, "_monitor_state", None)
            has_active = (
                monitor_state is not None
                and int(monitor_state.active_count()) > 0
            )
            if self.validated:
                raise RuntimeError(
                    "Monitoring requires numeric validated rows; "
                    "Python monitoring fallback is disabled"
                )
            if has_active:
                self._monitor_corr_buffered_numeric(
                    np.empty((0, 5), dtype=np.int64),
                    np.empty((0,), dtype=np.float64),
                )
            return

        self._monitor_corr_buffered_numeric(numeric_rows, numeric_corrs)
        self.validated = {}
        self._reset_validated_numeric_step()


    def update_window_step(self,window_step):
        if self.basic_window % window_step == 0:
            if(self.verbose):
                print("\nStep ",self.window_step," to ",window_step)
            for s in range(self.n_nodes):
                self.sketch_nodes[s].update_window_step(window_step)
            self.window_step = window_step
            self.n_lagged_windows = self.n_lags//self.window_step+1
            for g in range(self.n_grids):
                self.grid_nodes[g].update_n_lagged_windows(self.n_lagged_windows)
        else:
            raise TypeError("Basic window size (",self.basic_window,") is not divisable by new window step (",window_step,")")
        return True
    
    def update_n_lags(self,n_lags):
        if n_lags % self.window_step == 0:
            self.n_lags = n_lags
            self.n_lagged_windows = self.n_lags//self.window_step+1
            for g in range(self.n_grids):
                self.grid_nodes[g].update_n_lagged_windows(self.n_lagged_windows)          
        else:
            raise TypeError("Number of lags (",n_lags,") is not divisable by window step (",self.window_step,")")
        return True

    def update_window_size(self,window_size):
        if window_size % self.basic_window == 0:
            if(self.verbose):
                print("\nWindow ",self.window_size," to ",window_size)
            self.window_size = window_size
            for s in range(self.n_nodes):
                self.sketch_nodes[s].update_window_size(self.window_size)
        else:
            raise TypeError("New window size (",window_size,") is not divisable by basic window size (",self.basic_window,")")
        return True

    def __del__(self):
        try:
            self._reset_thread_pool()
        except Exception:
            pass

    def _print_correlated(self):
        if len(self.correlated)>0:
            for pair,corr in self.correlated.items():
                t1, t2 = pair[2], pair[3]
                if self.datetime_index:
                    t1, t2 = self._index_to_datetime(t1), self._index_to_datetime(t2)
                print(pair[0]," at time ",t1," and ",pair[1]," at time ",t2,"are correlated (coef ",corr,").")
    
    def _save_correlated(self,output_csv):
        if not self.correlated:
            return

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # Keep final-mode artifacts in the same canonical order expected by the
        # streaming comparator and buffered chunk merger.
        sorted_items = sorted(
            self.correlated.items(),
            key=lambda item: _window_sort_key(
                (item[0][0], item[0][1], item[0][2], item[0][3]),
                self.window_size,
            ),
        )

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "time1_idx", "time1", "time2_idx", "time2", "corr"])
            for pair,corr in sorted_items:
                id1, id2, t1, t2, _ = pair
                t1_idx = t1
                t2_idx = t2
                if self.datetime_index:
                    t1, t2 = self._safe_index_to_datetime(t1), self._safe_index_to_datetime(t2)
                writer.writerow([id1,id2,t1_idx,t1,t2_idx,t2,corr])

    def _save_max_lag_correlated(self,output_csv):
        if not getattr(self, "_artifact_save_maxlag", True):
            return

        if self._maxlag_state:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "t1_index", "time1", "max_corr", "lag"])
                for key in sorted(self._maxlag_state, key=_maxlag_sort_key):
                    id1, id2, t1_idx = key
                    record = self._maxlag_state[key]
                    time1_display = record["time1_raw"]
                    if self.datetime_index:
                        time1_display = self._safe_index_to_datetime(time1_display)
                    writer.writerow([
                        id1,
                        id2,
                        t1_idx,
                        time1_display,
                        record["max_corr"],
                        record["lag"],
                    ])
            return

        if not self.correlated:
            return

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        def _to_sortable(value):
            if isinstance(value, (np.integer, int)):
                return int(value)
            try:
                return int(np.int64(value))
            except (TypeError, ValueError, OverflowError):
                if isinstance(value, datetime.datetime):
                    return int(value.timestamp())
                if isinstance(value, datetime.date):
                    return int(datetime.datetime.combine(value, datetime.time()).timestamp())
                return 0

        max_corr_by_pair = {}
        for (id1, id2, t1, t2, _window), corr in self.correlated.items():
            if corr is None:
                continue
            if isinstance(corr, float) and math.isnan(corr):
                continue

            timepts = np.array([t1, t2])
            try:
                lag = int(abs(timepts.max() - timepts.min()))
            except TypeError:
                lag = int(abs(np.int64(timepts.max()) - np.int64(timepts.min())))

            if self.n_lags is not None and lag > self.n_lags:
                continue

            pair_ids = tuple(sorted((id1, id2)))
            if pair_ids[0] == id1:
                time1 = t1
                time2 = t2
            else:
                time1 = t2
                time2 = t1

            t1_idx = _to_sortable(time1)

            score = abs(corr)

            pair_records = max_corr_by_pair.setdefault(pair_ids, {})
            prev = pair_records.get(t1_idx)

            if prev is None or score > prev["score"] or (
                score == prev["score"] and lag < prev["lag"]
            ):
                pair_records[t1_idx] = {
                    "max_corr": corr,
                    "lag": lag,
                    "score": score,
                    "t1_raw": time1,
                }

        if not max_corr_by_pair:
            return

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "t1_index", "time1", "max_corr", "lag"])

            for pair_ids in sorted(max_corr_by_pair):
                pair_records = max_corr_by_pair[pair_ids]
                for t1_idx in sorted(pair_records):
                    record = pair_records[t1_idx]
                    time1_display = record["t1_raw"]
                    if self.datetime_index:
                        time1_display = self._safe_index_to_datetime(time1_display)
                    writer.writerow([
                        pair_ids[0],
                        pair_ids[1],
                        t1_idx,
                        time1_display,
                        record["max_corr"],
                        record["lag"],
                    ])
                    

    def _print_monitor_status(self):
        raise RuntimeError(
            "In-memory Python monitor status printing is disabled; "
            "write status artifacts from NumericMonitorState instead"
        )
    
    def _save_monitor_status(self,output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        monitor_state = getattr(self, "_monitor_state", None)
        if monitor_state is not None:
            monitor_state.finalize(True)
            rows = monitor_state.copy_status_rows()
        else:
            rows = np.empty((0, 7), dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64).reshape((-1, 7))
        if rows.shape[0] > 1:
            order = np.lexsort((rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
            rows = rows[order]
        ids_by_idx = self._ids_by_numeric_index()
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"])
            for row in rows:
                sid1 = int(row[0])
                sid2 = int(row[1])
                id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
                id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
                lag = int(row[2])
                t1 = int(row[3])
                t2 = int(row[4])
                corr_len = int(row[5])
                corr_sign = int(row[6])
                time1 = self._safe_index_to_datetime(t1) if self.datetime_index else t1
                time2 = self._safe_index_to_datetime(t2) if self.datetime_index else t2
                writer.writerow([id1, id2, lag, time1, time2, corr_len, corr_sign])
                        

    def _print_anomalies(self):
        raise RuntimeError(
            "In-memory Python anomaly printing is disabled; "
            "write anomaly artifacts from NumericMonitorState instead"
        )
    
    def _save_candidates(self, output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(['id1', 'id2', 'time1', 'time2', 'window', 'freq'])
            for pair, freq in sorted(self.freq_pairs.items()):
                id1, id2, t1, t2, w = pair
                time1 = self._safe_index_to_datetime(t1)
                time2 = self._safe_index_to_datetime(t2)
                writer.writerow([id1, id2, time1, time2, w, freq])

    def _save_negative_pairs(self, output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(['id1', 'id2', 'time1', 'time2', 'corr'])
            for (id1, id2, t1, t2, _), corr in self.correlated.items():
                if corr <= -self.corr_threshold:
                    time1 = self._safe_index_to_datetime(t1)
                    time2 = self._safe_index_to_datetime(t2)
                    writer.writerow([id1, id2, time1, time2, corr])

    def _save_anomalies(self,output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        monitor_state = getattr(self, "_monitor_state", None)
        if monitor_state is not None:
            rows = monitor_state.copy_anomaly_rows()
        else:
            rows = np.empty((0, 5), dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64).reshape((-1, 5))
        if rows.shape[0] > 1:
            order = np.lexsort((rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
            rows = rows[order]
        ids_by_idx = self._ids_by_numeric_index()
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "lag", "time_idx", "time", "anomaly"])
            for row in rows:
                sid1 = int(row[0])
                sid2 = int(row[1])
                id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
                id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
                lag = int(row[2])
                time_idx = int(row[3])
                marker = int(row[4])
                time_display = self._safe_index_to_datetime(time_idx) if self.datetime_index else time_idx
                writer.writerow([id1, id2, lag, time_idx, time_display, _marker_to_label(marker)])

    def _artifact_write_rows(self, path, header, rows):
        if not rows:
            return
        state = getattr(self, "_artifact_state", None)
        if not state:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = path not in state["headers"] or not os.path.exists(path)
        with open(path, "a", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            if write_header:
                writer.writerow(header)
                state["headers"].add(path)
            writer.writerows(rows)

    def _load_existing_artifacts(self, prefix):
        state = self._artifact_state
        if not prefix:
            return

        def _load_correlated(path):
            delimiter = _detect_csv_delimiter(path, default=CSV_DELIMITER)
            with open(path, newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                next(reader, None)
                for row in reader:
                    if len(row) >= 7:
                        key = tuple(row[:6])
                    elif len(row) >= 5:
                        key = (row[0], row[1], row[2], row[2], row[3], row[3])
                    else:
                        continue
                    state["correlated"].add(key)

        def _load_neg_pairs(path):
            delimiter = _detect_csv_delimiter(path, default=CSV_DELIMITER)
            with open(path, newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    key = tuple(row[:4])
                    state["neg_pairs"].add(key)

        def _load_candidates(path):
            delimiter = _detect_csv_delimiter(path, default=CSV_DELIMITER)
            with open(path, newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                next(reader, None)
                for row in reader:
                    if len(row) < 6:
                        continue
                    key = tuple(row[:5])
                    try:
                        freq = float(row[5])
                    except (TypeError, ValueError):
                        freq = row[5]
                    state["candidates"][key] = freq

        def _load_status(path):
            delimiter = _detect_csv_delimiter(path, default=CSV_DELIMITER)
            with open(path, newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                next(reader, None)
                for row in reader:
                    if len(row) < 7:
                        continue
                    key = tuple(row[:5])
                    try:
                        duration = float(row[5])
                    except (TypeError, ValueError):
                        duration = row[5]
                    corr_sign = row[6]
                    state["status"][key] = (duration, corr_sign)

        def _load_anomalies(path):
            delimiter = _detect_csv_delimiter(path, default=CSV_DELIMITER)
            with open(path, newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                next(reader, None)
                for row in reader:
                    if len(row) >= 6:
                        state["anomalies"].add(tuple(row[:6]))
                    elif len(row) >= 5:
                        state["anomalies"].add((row[0], row[1], row[2], row[3], row[3], row[4]))
                    else:
                        continue

        loaders = [
            (f"{prefix}_correlated.csv", _load_correlated),
            (f"{prefix}_neg_pairs.csv", _load_neg_pairs),
            (f"{prefix}_candidates.csv", _load_candidates),
            (f"{prefix}_status.csv", _load_status),
            (f"{prefix}_anomalies.csv", _load_anomalies),
        ]

        for path, loader in loaders:
            if os.path.exists(path):
                state["headers"].add(path)
                try:
                    loader(path)
                except Exception:
                    pass

    def _start_artifact_logging(self, prefix):
        if not prefix:
            self._artifact_state = {
                "prefix": None,
                "headers": set(),
                "correlated": set(),
                "neg_pairs": set(),
                "candidates": {},
                "status": {},
                "anomalies": set(),
            }
            return

        self._reset_artifact_files(prefix)
        if getattr(self, "_artifact_buffered", False):
            self._artifact_state = {
                "prefix": prefix,
                "headers": set(),
                "correlated": set(),
                "neg_pairs": set(),
                "candidates": {},
                "status": {},
                "anomalies": set(),
            }
        else:
            self._artifact_state = {
                "prefix": prefix,
                "headers": set(),
                "correlated": set(),
                "neg_pairs": set(),
                "candidates": {},
                "status": {},
                "anomalies": set(),
            }

    def _reset_artifact_files(self, prefix):
        if not prefix:
            return
        suffixes = (
            "_correlated.csv",
            "_neg_pairs.csv",
            "_candidates.csv",
            "_status.csv",
            "_anomalies.csv",
            "_max_lag_correlated.csv",
        )
        for suffix in suffixes:
            path = f"{prefix}{suffix}"
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        base_dir = os.path.dirname(os.path.abspath(prefix))
        base_name = os.path.basename(prefix)
        try:
            for name in os.listdir(base_dir):
                if not name.startswith(base_name + "_"):
                    continue
                if ".chunk" not in name:
                    continue
                path = os.path.join(base_dir, name)
                try:
                    os.remove(path)
                except OSError:
                    pass
        except OSError:
            pass

    def _append_artifacts(self):
        state = getattr(self, "_artifact_state", None)
        if not state or not state.get("prefix"):
            return
        prefix = state["prefix"]
        if getattr(self, "_artifact_save_correlated", True):
            self._append_correlated_artifact(prefix)
        if getattr(self, "_artifact_save_status", True):
            self._append_status_artifact(prefix)
        if getattr(self, "_artifact_save_anomalies", True):
            self._append_anomalies_artifact(prefix)

    def _append_correlated_artifact(self, prefix):
        if not getattr(self, "_artifact_save_correlated", True):
            return
        state = self._artifact_state
        rows = []
        for pair, corr in self.correlated.items():
            id1, id2, t1, t2, _ = pair
            t1_display = self._safe_index_to_datetime(t1)
            t2_display = self._safe_index_to_datetime(t2)
            key = (str(id1), str(id2), str(t1), str(t1_display), str(t2), str(t2_display))
            if key in state["correlated"]:
                continue
            rows.append([id1, id2, t1, t1_display, t2, t2_display, corr])
            state["correlated"].add(key)
        if rows:
            self._artifact_write_rows(
                f"{prefix}_correlated.csv",
                ["id1", "id2", "time1_idx", "time1", "time2_idx", "time2", "corr"],
                rows,
            )

    def _append_negative_pairs_artifact(self, prefix):
        state = self._artifact_state
        rows = []
        for (id1, id2, t1, t2, _), corr in self.correlated.items():
            if corr > -self.corr_threshold:
                continue
            t1_display = self._safe_index_to_datetime(t1)
            t2_display = self._safe_index_to_datetime(t2)
            key = (str(id1), str(id2), str(t1_display), str(t2_display))
            if key in state["neg_pairs"]:
                continue
            rows.append([id1, id2, t1_display, t2_display, corr])
            state["neg_pairs"].add(key)
        if rows:
            self._artifact_write_rows(
                f"{prefix}_neg_pairs.csv",
                ["id1", "id2", "time1", "time2", "corr"],
                rows,
            )

    def _append_candidates_artifact(self, prefix):
        state = self._artifact_state
        if not getattr(self, "freq_pairs", None):
            return
        rows = []
        for pair, freq in self.freq_pairs.items():
            id1, id2, t1, t2, window = pair
            t1_display = self._safe_index_to_datetime(t1)
            t2_display = self._safe_index_to_datetime(t2)
            key = (str(id1), str(id2), str(t1_display), str(t2_display), str(window))
            prev = state["candidates"].get(key)
            try:
                freq_val = float(freq)
            except (TypeError, ValueError):
                freq_val = freq
            if prev is not None and prev == freq_val:
                continue
            state["candidates"][key] = freq_val
            rows.append([id1, id2, t1_display, t2_display, window, freq])
        if rows:
            self._artifact_write_rows(
                f"{prefix}_candidates.csv",
                ["id1", "id2", "time1", "time2", "window", "freq"],
                rows,
            )

    def _append_status_artifact(self, prefix):
        raise RuntimeError(
            "Legacy Python-object status artifact appending is disabled; "
            "use NumericMonitorState chunk/final writers"
        )

    def _append_anomalies_artifact(self, prefix):
        raise RuntimeError(
            "Legacy Python-object anomaly artifact appending is disabled; "
            "use NumericMonitorState chunk/final writers"
        )
    
    def _print_state(self,):
        if len(self.correlated)>0:
            print("\nCorrelated:")
            self._print_correlated()

    def _get_sketches(self, new_data_step, verbose, testing, worker_mode=None):
        """
        Run Sketches nodes in parallel, collect their (sketches, partitions),
        then append merged partitions once per grid to the grid nodes.
        """
        new_data_step_index = new_data_step[0, :]
        new_data_step_series = new_data_step[1:, :]

        while len(self.sketch_nodes) < self.n_sketch_nodes:
                self.sketch_nodes.append(
                    Sketches(
                        self.window_size,
                        self.basic_window,
                        self.window_step,
                        self.seed,
                        self.seed_toggle,
                        self.n_vectors,
                        self.grid_dimension,
                        self.grid_nodes,
                        self.preprocess,
                        self.neg_corr,
                        self.sketch_norm,
                        full_vector_candidates=self.full_vector_candidates,
                    )
                )
        if len(self.sketch_nodes) > self.n_sketch_nodes:
            self.sketch_nodes = self.sketch_nodes[:self.n_sketch_nodes]

        node_inputs_thread = []
        max_nodes = min(self.n_sketch_nodes, len(self.map_ids))
        for s in range(max_nodes):
            shard_ids = self.map_ids[s]
            ids_subset = [id_ for id_ in shard_ids if id_ in self.series_ids]
            if not ids_subset:
                continue
            series_idx = [self.series_ids[id_] for id_ in ids_subset]
            node_series = new_data_step_series[series_idx, :]
            node_new = np.vstack([new_data_step_index, node_series])
            node_inputs_thread.append((self, s, node_new, ids_subset, verbose, testing))

        if not node_inputs_thread:
            return {}

        worker_mode = worker_mode if worker_mode else self.exec

        dispatch_items = node_inputs_thread

        if worker_mode == "sequential":
            results = [_sketch_worker(payload) for payload in dispatch_items]
        else:
            t0 = time.perf_counter() if self.profile_enabled else None
            results = self._parallel_map(
                _sketch_worker,
                dispatch_items,
                mode=worker_mode,
                max_workers=len(dispatch_items),
                preserve_order=False,
            )
            if t0 is not None:
                self._profile_add("sketch.dispatch", time.perf_counter() - t0)

        # Merge sketches from all nodes
        merged_sketches = {}
        per_grid_partition = [[] for _ in range(self.n_grids)]
        iter_results = results

        t0 = time.perf_counter() if self.profile_enabled else None
        for entry in iter_results:
            s, node_obj, sks, parts = entry
            self.sketch_nodes[s] = node_obj
            merged_sketches.update(sks)
            for g, partition in enumerate(parts):
                if g >= self.n_grids:
                    break
                if partition is None:
                    continue
                per_grid_partition[g].append(partition)
        if t0 is not None:
            self._profile_add("sketch.merge", time.perf_counter() - t0)

        curr_time = self._curr_startTime()
        t0 = time.perf_counter() if self.profile_enabled else None
        sid_list = [sid for sid, idx in sorted(self.series_ids.items(), key=lambda item: item[1])]
        for g in range(self.n_grids):
            parts = per_grid_partition[g]
            if not parts:
                continue
            if isinstance(parts[0], dict):
                merged_partition = {}
                for part in parts:
                    merged_partition.update(part)
            else:
                if len(parts) == 1:
                    merged_partition = parts[0]
                else:
                    sid_list_parts = [p[0] for p in parts if p is not None and p[0].size > 0]
                    if not sid_list_parts:
                        continue
                    sid_idx = np.concatenate(sid_list_parts, axis=0)
                    time_arr = np.concatenate([p[1] for p in parts if p is not None and p[1].size > 0], axis=0)
                    w_arr = np.concatenate([p[2] for p in parts if p is not None and p[2].size > 0], axis=0)
                    value_arr = np.concatenate([p[3] for p in parts if p is not None and p[3].size > 0], axis=0)
                    is_const = np.concatenate([p[4] for p in parts if p is not None and p[4].size > 0], axis=0)
                    merged_partition = (sid_idx, time_arr, w_arr, value_arr, is_const)
                self.grid_nodes[g].set_sid_list(sid_list)
            self.grid_nodes[g].append_partition(curr_time, merged_partition)
        if t0 is not None:
            self._profile_add("sketch.distribute", time.perf_counter() - t0)

        return merged_sketches

    def _accumulate_candidate_search_stats(self, node_obj):
        stats_getter = getattr(node_obj, "candidate_search_stats", None)
        if stats_getter is None:
            return
        try:
            stats = stats_getter()
        except Exception:
            return
        if not stats:
            return
        index_candidates = int(stats.get("index_candidates", 0) or 0)
        valid_index_candidates = int(stats.get("valid_index_candidates", 0) or 0)
        unique_index_candidates = int(stats.get("unique_index_candidates", 0) or 0)
        duplicate_index_candidates = int(stats.get("duplicate_index_candidates", 0) or 0)
        unique_pre_dot_pairs = int(stats.get("unique_pre_dot_pairs", 0) or 0)
        duplicate_pre_dot_pairs = int(stats.get("duplicate_pre_dot_pairs", 0) or 0)
        after_coord = int(stats.get("after_coord", 0) or 0)
        partial_checks = int(stats.get("partial_checks", 0) or 0)
        after_partial = int(stats.get("after_partial", 0) or 0)
        after_similarity = int(stats.get("after_similarity", 0) or 0)
        dot_checks = int(stats.get("dot_checks", 0) or 0)
        distance_checks = int(stats.get("distance_checks", 0) or 0)
        blocks_visited = int(stats.get("blocks_visited", 0) or 0)
        blocks_pruned_by_ub = int(stats.get("blocks_pruned_by_ub", 0) or 0)
        rows_in_surviving_blocks = int(stats.get("rows_in_surviving_blocks", 0) or 0)
        dot_checks_saved_by_row_ub = int(stats.get("dot_checks_saved_by_row_ub", 0) or 0)
        self.candidate_search_index_candidates += index_candidates
        self.candidate_search_valid_index_candidates += valid_index_candidates
        self.candidate_search_unique_index_candidates += unique_index_candidates
        self.candidate_search_duplicate_index_candidates += duplicate_index_candidates
        self.candidate_search_unique_pre_dot_pairs += unique_pre_dot_pairs
        self.candidate_search_duplicate_pre_dot_pairs += duplicate_pre_dot_pairs
        self.candidate_search_after_coord += after_coord
        self.candidate_search_partial_checks += partial_checks
        self.candidate_search_after_partial += after_partial
        self.candidate_search_after_similarity += after_similarity
        self.candidate_search_dot_checks += dot_checks
        self.candidate_search_distance_checks += distance_checks
        # (2026-07-06) Part 1 metrics -- see docs/implementation_log.md.
        self.candidate_search_blocks_visited += blocks_visited
        self.candidate_search_blocks_pruned_by_ub += blocks_pruned_by_ub
        self.candidate_search_rows_in_surviving_blocks += rows_in_surviving_blocks
        self.candidate_search_dot_checks_saved_by_row_ub += dot_checks_saved_by_row_ub
        # (2026-07-06) InstinctIndex metrics -- only nonzero for
        # candidate_backend="instinct". Most fields are per-batch counts,
        # summed like every other counter above. instinct_best_score_seen
        # is a running max (summing a max would be meaningless).
        # instinct_mean_score_returned/instinct_dead_node_ratio are latest-
        # observed snapshots, not sums (a mean/ratio isn't meaningful summed
        # across batches) -- an approximation acceptable for this
        # experimental backend's v1. See docs/implementation_log.md,
        # "InstinctIndex: experimental approximate graph backend".
        self.candidate_search_instinct_visited_nodes += int(stats.get("instinct_visited_nodes", 0) or 0)
        self.candidate_search_instinct_visited_live_nodes += int(stats.get("instinct_visited_live_nodes", 0) or 0)
        self.candidate_search_instinct_dead_nodes_skipped += int(stats.get("instinct_dead_nodes_skipped", 0) or 0)
        self.candidate_search_instinct_edges_scanned += int(stats.get("instinct_edges_scanned", 0) or 0)
        self.candidate_search_instinct_candidates_returned += int(stats.get("instinct_candidates_returned", 0) or 0)
        self.candidate_search_instinct_threshold_candidates += int(stats.get("instinct_threshold_candidates", 0) or 0)
        self.candidate_search_instinct_topk_candidates += int(stats.get("instinct_topk_candidates", 0) or 0)
        self.candidate_search_instinct_queries_with_too_few_live_nodes += int(stats.get("instinct_queries_with_too_few_live_nodes", 0) or 0)
        self.candidate_search_instinct_query_time += float(stats.get("instinct_query_time", 0.0) or 0.0)
        self.candidate_search_instinct_insert_time += float(stats.get("instinct_insert_time", 0.0) or 0.0)
        self.candidate_search_instinct_num_nodes_total += int(stats.get("instinct_num_nodes_total", 0) or 0)
        self.candidate_search_instinct_num_nodes_alive += int(stats.get("instinct_num_nodes_alive", 0) or 0)
        self.candidate_search_instinct_best_score_seen = max(
            self.candidate_search_instinct_best_score_seen, float(stats.get("instinct_best_score_seen", 0.0) or 0.0)
        )
        self.candidate_search_instinct_mean_score_returned = float(stats.get("instinct_mean_score_returned", 0.0) or 0.0)
        self.candidate_search_instinct_dead_node_ratio = float(stats.get("instinct_dead_node_ratio", 0.0) or 0.0)

    def _run_grids(self, verbose, testing, worker_mode=None):
        """
        Run each grid node in parallel via run() and merge the local outputs
        into self.freq_pairs and self.candidates on the main thread.
        """
        n_ids = len(self.series_ids)
        sid_list = [sid for sid, idx in sorted(self.series_ids.items(), key=lambda item: item[1])]
        for node_obj in self.grid_nodes:
            if hasattr(node_obj, "set_sid_list"):
                node_obj.set_sid_list(sid_list)

        worker_mode = worker_mode if worker_mode else self.exec

        use_numeric_candidate_rows = (
            self.freq_threshold <= 0
            and self.n_grids == 1
        )

        thread_payloads = [
            (self, g, n_ids, verbose, testing, use_numeric_candidate_rows)
            for g in range(self.n_grids)
        ]

        payloads = thread_payloads

        if getattr(self, "recent_shard_candidates", False):
            node_obj = self.grid_nodes[0]
            node_obj._profile_callback = self._profile_add if self.profile_enabled else None
            node_obj._profile_metric_callback = self._profile_add_metric if self.profile_enabled else None
            candidate_executor = None
            uses_internal_parallel = False
            checker = getattr(node_obj, "uses_internal_parallel_candidate_search", None)
            if checker is not None:
                try:
                    uses_internal_parallel = bool(checker())
                except Exception:
                    uses_internal_parallel = False
            if worker_mode == "thread" and self.n_candidate_nodes > 1 and not uses_internal_parallel:
                candidate_executor = self._get_thread_pool(self.n_candidate_nodes)
            result = node_obj.run(
                n_ids,
                verbose=verbose,
                testing=testing,
                worker_mode=worker_mode,
                max_workers=self.n_candidate_nodes,
                executor=candidate_executor,
                numeric_rows=use_numeric_candidate_rows,
            )
            grid_results = [(0, node_obj, result)]
        else:
            # The stable release no longer supports the old "parallel grids"
            # candidate path. Non-recent-shard grids are still valid sketch
            # partitions, but they are evaluated sequentially.
            for node_obj in self.grid_nodes:
                node_obj._profile_callback = self._profile_add if self.profile_enabled else None
                node_obj._profile_metric_callback = self._profile_add_metric if self.profile_enabled else None
            grid_results = [_grid_worker(payload) for payload in payloads]

        # Merge
        self.freq_pairs = {}
        self.uncorrelated = {}
        self.candidate_dist_sq = {}
        self._candidate_numeric_rows = None

        iterable = (
            (
                g_index,
                node_obj,
                result,
            )
            for g_index, node_obj, result in grid_results
        )

        t0 = time.perf_counter() if self.profile_enabled else None
        numeric_rows_accum = None
        numeric_rows_ready = False
        for g_index, node_obj, (loc_freq, _loc_cand, loc_unc) in iterable:
            self.grid_nodes[g_index] = node_obj
            self._accumulate_candidate_search_stats(node_obj)
            node_rows = getattr(node_obj, "_candidate_numeric_rows", None)
            if use_numeric_candidate_rows and node_rows is not None:
                numeric_rows_ready = True
                node_rows = np.asarray(node_rows, dtype=np.int64)
                if node_rows.size:
                    if numeric_rows_accum is None:
                        numeric_rows_accum = node_rows.reshape((-1, 5))
                    else:
                        numeric_rows_accum = np.vstack((numeric_rows_accum, node_rows.reshape((-1, 5))))
                continue
            for k in sorted(loc_freq):
                v = float(loc_freq[k])
                self.freq_pairs[k] = self.freq_pairs.get(k, 0.0) + v
            for k in sorted(loc_unc):
                self.uncorrelated[k] = loc_unc[k]

        if self.freq_threshold <= 0:
            base_candidates = set(self.freq_pairs.keys())
        else:
            base_candidates = {
                pair
                for pair, count in self.freq_pairs.items()
                if count >= self.freq_threshold
            }
        if numeric_rows_ready:
            if numeric_rows_accum is None:
                numeric_rows_accum = np.empty((0, 5), dtype=np.int64)
            self._candidate_numeric_rows = np.ascontiguousarray(numeric_rows_accum, dtype=np.int64).reshape((-1, 5))
            base_candidates = set()
        if t0 is not None:
            self._profile_add("grid.merge", time.perf_counter() - t0)

        self.candidates = {pair: 1 for pair in base_candidates}
        
    def print_sketch_hist(self):
        flattened = [item for sublist in self.hist_sketches for item in sublist]
        print("\nMean of sketches:",np.mean(flattened),", standard deviation:",np.std(flattened))
        plt.hist(flattened)
        plt.show()
    
    def get_correlation_flags(self, total_time, tolerance=None):
        if tolerance is None:
            tolerance = self.window_size

        monitor_state = getattr(self, "_monitor_state", None)
        if monitor_state is None:
            return {}
        rows = np.asarray(monitor_state.copy_anomaly_rows(), dtype=np.int64).reshape((-1, 5))
        ids_by_idx = self._ids_by_numeric_index()

        merged_anomalies = {}
        for row in rows:
            sid1 = int(row[0])
            sid2 = int(row[1])
            id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
            id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
            norm_key = _normalize_pair_lag_key(id1, id2, int(row[2]))
            merged_anomalies.setdefault(norm_key, []).append((int(row[3]), int(row[4])))

        flags_dict = {}
        for key, events in merged_anomalies.items():
            flags = np.zeros(total_time, dtype=int)
            sorted_events = sorted(events)

            active = False
            last_out_time = None

            for time, typ in sorted_events:
                if typ == 1:
                    if not active:
                        if last_out_time is None or (time - last_out_time) > tolerance:
                            active = True
                            flags[int(time):] = 1
                elif typ == -1:
                    if active:
                        last_out_time = time
                        active = False
                        flags[int(time):] = 0

            flags_dict[key] = flags

        return flags_dict
    
    def get_anomaly_flags(self, total_time, tolerance=None):
        if tolerance is None:
            tolerance = self.window_size

        monitor_state = getattr(self, "_monitor_state", None)
        if monitor_state is None:
            return np.zeros(total_time, dtype=int)
        anomaly_rows = np.asarray(monitor_state.copy_anomaly_rows(), dtype=np.int64).reshape((-1, 5))
        status_rows = np.asarray(monitor_state.copy_status_rows(), dtype=np.int64).reshape((-1, 7))
        ids_by_idx = self._ids_by_numeric_index()

        merged_anomalies = {}
        for row in anomaly_rows:
            sid1 = int(row[0])
            sid2 = int(row[1])
            id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
            id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
            norm_key = _normalize_pair_lag_key(id1, id2, int(row[2]))
            merged_anomalies.setdefault(norm_key, []).append((int(row[3]), int(row[4])))

        status_lengths_by_key = {}
        for row in status_rows:
            sid1 = int(row[0])
            sid2 = int(row[1])
            id1 = ids_by_idx[sid1] if 0 <= sid1 < len(ids_by_idx) else sid1
            id2 = ids_by_idx[sid2] if 0 <= sid2 < len(ids_by_idx) else sid2
            norm_key = _normalize_pair_lag_key(id1, id2, int(row[2]))
            status_lengths_by_key.setdefault(norm_key, []).append((
                int(row[3]),
                int(row[4]),
                0,
                int(row[5]),
                int(row[6]),
            ))

        flags_dict = {}
        for key, events in merged_anomalies.items():
            flags = np.zeros(total_time, dtype=int)
            sorted_events = sorted(events)
            previous_in_time = None
            i = 0
            while i < len(sorted_events):
                time, typ = sorted_events[i]
                if typ == -1:  # falling out
                    out_time = time
                    in_time = None
                    # Look for next falling in
                    for j in range(i + 1, len(sorted_events)):
                        next_time, next_typ = sorted_events[j]
                        if next_typ == 1:
                            in_time = next_time
                            break
                    if in_time is None or (in_time - out_time > tolerance):
                        adjusted_out_time = out_time
                        if previous_in_time is not None:
                            last_corr_len = [
                                status[3] for status in status_lengths_by_key.get(key, [])
                                if min(status[0], status[1]) == previous_in_time
                            ]
                            if last_corr_len:
                                adjusted_out_time = max(out_time, previous_in_time + last_corr_len[0])
                        end_time = in_time if in_time is not None else total_time
                        if adjusted_out_time < end_time:
                            flags[int(adjusted_out_time):int(min(end_time, total_time))] = 1
                    i = j if in_time is not None else len(sorted_events)
                elif typ == 1:  # falling in
                    previous_in_time = time
                    i += 1
                else:
                    i += 1

            # Handle type 0 (correlation spans)
            type0_times = [int(t) for t, typ in sorted_events if typ == 0]
            for k in range(0, len(type0_times) - 1, 2):
                start_t0 = type0_times[k]
                end_t0 = type0_times[k + 1]
                if start_t0 < end_t0:
                    flags[start_t0:min(end_t0, total_time)] = 1

            flags_dict[key] = flags

        if not flags_dict:
            return np.zeros(total_time, dtype=int)
        sum_flags = np.sum(list(flags_dict.values()), axis=0)
        combined_flags = np.clip(sum_flags, 0, 1)
        return combined_flags
    
    def _curr_window(self):
        return self.window_data[:,-self.curr_window_size:]
    
    def _curr_window_step(self):
        return np.vstack([self.window_index[-self.window_step:],self.window_data[:,-self.window_step:]])
    
    def _curr_startTime(self):
        return self.window_index[-self.curr_window_size]
    
    def _update_curr_window_size(self):
        self.curr_window_size = min(self.window_size,self.window_data.shape[1])
        return None

    def _recompute_current_window_raw_moments(self):
        # (2026-07-06) "point 1": per-series sum/sum_sq/sum_cu/sum_qu over
        # the current (just-arrived) window of CorrTrack's own window_data,
        # recomputed fresh each step -- not incrementally delta-updated like
        # Sketches._raw_window_sums (a *different* object with its own,
        # differently-partitioned window_data; that cache cannot be reused
        # here). A full O(n_series * curr_window_size) recompute once per
        # step is cheap next to what it replaces: an O(w) recompute of one
        # side's moments for every candidate pair reaching exact validation,
        # of which there are typically many more than n_series per step. See
        # docs/implementation_log.md.
        if self.window_data is None or self.curr_window_size <= 0:
            self._cw_raw_sums = None
            self._cw_raw_sums_sq = None
            self._cw_raw_sums_cu = None
            self._cw_raw_sums_qu = None
            return
        tail = np.asarray(self.window_data[:, -self.curr_window_size:], dtype=np.float64)
        self._cw_raw_sums = np.sum(tail, axis=1, dtype=np.float64)
        self._cw_raw_sums_sq = np.sum(tail * tail, axis=1, dtype=np.float64)
        self._cw_raw_sums_cu = np.sum(tail * tail * tail, axis=1, dtype=np.float64)
        self._cw_raw_sums_qu = np.sum(tail * tail * tail * tail, axis=1, dtype=np.float64)

    def _current_window_cache_kwargs(self):
        # (2026-07-06) Shared "point 1" cache-kwargs builder, used by every
        # validation call site (numeric-rows hybrid and non-hybrid branches,
        # dict-based sequential branches) so they all recompute/gate this
        # exactly the same way. Lazy: recomputes at most once per step, only
        # on first actual use. Gated by validation_current_window_cache
        # (default False -- opt-in, see docs/implementation_log.md for why).
        if not getattr(self, "validation_current_window_cache", False):
            return {}
        if not getattr(self, "_cw_raw_moments_fresh", False):
            self._recompute_current_window_raw_moments()
            self._cw_raw_moments_fresh = True
        if (
            self._cw_raw_sums is None
            or self._cw_raw_sums_sq is None
            or self._cw_raw_sums_cu is None
            or self._cw_raw_sums_qu is None
            or self.window_data is None
            or self._cw_raw_sums.shape[0] != self.window_data.shape[0]
            or not self.curr_window_size
        ):
            return {}
        return {
            "current_window_sums": self._cw_raw_sums,
            "current_window_sums_sq": self._cw_raw_sums_sq,
            "current_window_sums_cu": self._cw_raw_sums_cu,
            "current_window_sums_qu": self._cw_raw_sums_qu,
            "current_window_size": int(self.curr_window_size),
        }

    def compute_metrics(predicted: np.ndarray, ground_truth: np.ndarray):
        assert predicted.shape == ground_truth.shape, "Arrays must have the same shape."
        pred = set(np.flatnonzero(np.asarray(predicted).astype(bool).ravel()))
        gt = set(np.flatnonzero(np.asarray(ground_truth).astype(bool).ravel()))
        all_idx = set(range(np.asarray(predicted).size))
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        tn = len(all_idx - (pred | gt))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        try:
            aucroc = roc_auc_score(ground_truth, predicted)
        except ValueError:
            aucroc = float('nan')  # e.g., if only one class present in ground_truth
        try:
            pr_auc  = average_precision_score(ground_truth, predicted)
        except ValueError:
            pr_auc = float('nan')
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'aucroc': aucroc,
            'pr_auc': pr_auc
        }
    
    def compute_metrics_bf(
        predicted: dict,
        ground_truth: dict,
        windows=True,
        pair_min_dist=None,
        total_pairs_bf=None,
        use_ground_truth_sign_for_tp=False,
    ):
        if windows:
            return CorrTrack._compute_metrics_bf_windows(
                predicted,
                ground_truth,
                pair_min_dist,
                total_pairs_bf=total_pairs_bf,
                use_ground_truth_sign_for_tp=use_ground_truth_sign_for_tp,
            )
        return CorrTrack.compute_metrics_bf_timestamps(predicted, ground_truth)
    
    def _compute_metrics_bf_windows(
        predicted: dict,
        ground_truth: dict,
        pair_min_dist=None,
        total_pairs_bf=None,
        use_ground_truth_sign_for_tp=False,
    ):
        def _normalize_key(key):
            id1, id2, t1, t2, w = key

            if id1 == id2:
                # Autocorrelation case
                return (id1, id2, max(t1, t2), min(t1, t2), w)

            if t1 == t2:
                # Synchronous case
                return tuple(sorted([id1, id2])) + (t1, t2, w)

            if t1 < t2:
                # Cross-series, cross-time — canonicalize by lexicographic pair
                return (id2, id1, t2, t1, w)
            else:
                return (id1, id2, t1, t2, w)

        # Normalize keys
        normalized_predicted = {}
        normalized_predicted_pos = {}
        normalized_predicted_neg = {}
        for key, value in predicted.items():
            norm_key = _normalize_key(key)
            normalized_predicted[norm_key] = value
            if value > 0:
                normalized_predicted_pos[norm_key] = value
            else:
                normalized_predicted_neg[norm_key] = value

        normalized_ground_truth = {}
        normalized_ground_truth_pos = {}
        normalized_ground_truth_neg = {}
        for key, value in ground_truth.items():
            norm_key = _normalize_key(key)
            normalized_ground_truth[norm_key] = value
            if value > 0:
                normalized_ground_truth_pos[norm_key] = value
            else:
                normalized_ground_truth_neg[norm_key] = value

        pred = set(normalized_predicted)
        gt = set(normalized_ground_truth)

        pred_pos = set(normalized_predicted_pos)
        gt_pos = set(normalized_ground_truth_pos)

        pred_neg = set(normalized_predicted_neg)
        gt_neg = set(normalized_ground_truth_neg)
        
        def _safe_prec_recall(pred_set, gt_set, total_pairs=None):
            fp = len(pred_set - gt_set)
            specificity = 0.0
            try:
                total_pairs = int(total_pairs) if total_pairs is not None else None
            except (TypeError, ValueError):
                total_pairs = None
            if total_pairs is not None:
                negatives = max(total_pairs - len(gt_set), 0)
                if negatives > 0:
                    tn = max(negatives - fp, 0)
                    specificity = tn / negatives
            if len(pred_set) == 0:
                return 0.0, (1.0 if len(gt_set) == 0 else 0.0), specificity  # precision 0, recall 0 unless both empty
            if len(gt_set) == 0:
                return 0.0, 0.0, specificity
            precision = (len(pred_set) - len(pred_set - gt_set)) / len(pred_set)
            recall    = (len(gt_set) - len(gt_set - pred_set)) / len(gt_set)
            return precision, recall, specificity

        # Union of normalized keys
        all_keys = pred | gt
        sorted_keys = sorted(all_keys)

        recall_min = None if pair_min_dist is None else int(pair_min_dist in pred)
        if not sorted_keys:
            _, _, specificity = _safe_prec_recall(pred, gt, total_pairs_bf)
            return {
                'precision': 0.0,
                'recall': 0.0,
                'specificity': specificity,
                'f1_score': 0.0,
                'aucroc': float('nan'),
                'pr_auc': float('nan'),
                'recall_min': recall_min,
                'precision_pos': 0.0,
                'recall_pos': 0.0,
                'f1_score_pos': 0.0,
                'precision_neg': 0.0,
                'recall_neg': 0.0,
                'f1_score_neg': 0.0,
            }

        # Fill missing keys with zeros
        filled_predicted = {}
        filled_ground_truth = {}
        for key in sorted_keys:
            filled_predicted[key] = normalized_predicted.get(key, 0)
            filled_ground_truth[key] = normalized_ground_truth.get(key, 0)

        # Concatenate vectors in consistent order
        predicted_array = np.hstack([filled_predicted[k] for k in sorted_keys])
        ground_truth_array = np.hstack([filled_ground_truth[k] for k in sorted_keys])

        assert predicted_array.shape == ground_truth_array.shape, "Arrays must have the same shape after concatenation."

        # overall
        precision, recall, specificity = _safe_prec_recall(pred, gt, total_pairs_bf)
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

        if use_ground_truth_sign_for_tp:
            tp_pos = len(pred & gt_pos)
            tp_neg = len(pred & gt_neg)
            precision_pos = _signed_precision_from_ambiguous_fp(tp_pos, tp_neg, len(pred))
            recall_pos = tp_pos / len(gt_pos) if gt_pos else 0.0
            precision_neg = _signed_precision_from_ambiguous_fp(tp_neg, tp_pos, len(pred))
            recall_neg = tp_neg / len(gt_neg) if gt_neg else 0.0
        else:
            precision_pos, recall_pos, _ = _safe_prec_recall(pred_pos, gt_pos)
            precision_neg, recall_neg, _ = _safe_prec_recall(pred_neg, gt_neg)
        f1_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos) if (precision_pos+recall_pos)>0 else 0.0
        f1_neg = 2*precision_neg*recall_neg/(precision_neg+recall_neg) if (precision_neg+recall_neg)>0 else 0.0

        try:
            aucroc = roc_auc_score(ground_truth_array, predicted_array)
        except ValueError:
            aucroc = float('nan')

        try:
            pr_auc = average_precision_score(ground_truth_array, predicted_array)
        except ValueError:
            pr_auc = float('nan')

        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'aucroc': aucroc,
            'pr_auc': pr_auc,
            'recall_min': recall_min,
            'precision_pos': precision_pos,
            'recall_pos': recall_pos,
            'f1_score_pos': f1_pos,
            'precision_neg': precision_neg,
            'recall_neg': recall_neg,
            'f1_score_neg': f1_neg,
        }         

    def compute_metrics_bf_timestamps(predicted: dict, ground_truth: dict):
        def normalize_key(key):
            a, b, c = key
            return tuple(sorted([a, b]) + [c])

        # Normalize keys
        normalized_predicted = {}
        for key, value in predicted.items():
            norm_key = normalize_key(key)
            normalized_predicted[norm_key] = value

        normalized_ground_truth = {}
        for key, value in ground_truth.items():
            norm_key = normalize_key(key)
            normalized_ground_truth[norm_key] = value

        # Union of normalized keys
        all_keys = set(normalized_predicted.keys()).union(normalized_ground_truth.keys())
        sorted_keys = sorted(all_keys)

        if not sorted_keys:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 0.0,
                'f1_score': 0.0,
                'aucroc': float('nan'),
                'pr_auc': float('nan'),
                'recall_min': None,
                'precision_pos': float('nan'),
                'recall_pos': float('nan'),
                'f1_score_pos': float('nan'),
                'precision_neg': float('nan'),
                'recall_neg': float('nan'),
                'f1_score_neg': float('nan'),
            }

        # Infer expected length
        sample_array = next(iter(normalized_predicted.values())) if normalized_predicted else next(iter(normalized_ground_truth.values()))
        expected_length = len(sample_array)

        # Fill missing keys with zeros
        filled_predicted = {}
        filled_ground_truth = {}
        for key in sorted_keys:
            filled_predicted[key] = normalized_predicted.get(key, np.zeros(expected_length, dtype=int))
            filled_ground_truth[key] = normalized_ground_truth.get(key, np.zeros(expected_length, dtype=int))

        # Concatenate vectors in consistent order
        predicted_array = np.hstack([filled_predicted[k] for k in sorted_keys])
        ground_truth_array = np.hstack([filled_ground_truth[k] for k in sorted_keys])

        assert predicted_array.shape == ground_truth_array.shape, "Arrays must have the same shape after concatenation."

        # Metric computations from set membership.
        pred = set(np.flatnonzero(predicted_array.astype(bool).ravel()))
        gt = set(np.flatnonzero(ground_truth_array.astype(bool).ravel()))
        all_idx = set(range(predicted_array.size))
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        tn = len(all_idx - (pred | gt))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        try:
            aucroc = roc_auc_score(ground_truth_array, predicted_array)
        except ValueError:
            aucroc = float('nan')

        try:
            pr_auc = average_precision_score(ground_truth_array, predicted_array)
        except ValueError:
            pr_auc = float('nan')

        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'aucroc': aucroc,
            'pr_auc': pr_auc,
            'recall_min': None,
            'precision_pos': float('nan'),
            'recall_pos': float('nan'),
            'f1_score_pos': float('nan'),
            'precision_neg': float('nan'),
            'recall_neg': float('nan'),
            'f1_score_neg': float('nan'),
        }

    def get_bst_basic_window(window_size: int, window_step: int = None) -> int:
        ref = np.sqrt(window_size)
        parity = window_size % 2

        # Efficiently generate divisors
        divisors = set()
        for i in range(1, int(ref) + 1):
            if window_size % i == 0:
                divisors.add(i)
                divisors.add(window_size // i)

        # Filter by parity
        divisors = [d for d in divisors if d % 2 == parity]

        # Filter by window_step if provided
        if window_step:
            divisors = [d for d in divisors if d % window_step == 0]

        # Return the closest to sqrt(window_size) or fallback
        if divisors:
            return min(divisors, key=lambda x: abs(x - ref))
        return window_step if window_step else 1

    def get_clst_divisor(divisable: int, ref: int = None) -> int:
        
        base = round(ref)
        parity = divisable % 2

        # Adjust base to match window_size parity
        if base % 2 != parity:
            base += 1

        offset = 0
        while True:
            # Check candidate on the upper side
            candidate = base + offset
            if candidate > 0 and divisable % candidate == 0:
                return candidate

            # Check candidate on the lower side
            if offset > 0:
                candidate = base - offset
                if candidate > 0 and candidate % 2 == parity and divisable % candidate == 0:
                    return candidate

            offset += 2
    
    def _run_bf_parallel(self, curr_window_step, ids, verbose=False, testing=False, worker_mode=None):
        """
        Parallel wrapper for brute-force enumeration:
        - Shards the series IDs across workers.
        - Each worker calls Candidates_BF.run(...) on its shard.
        - Results are merged on the main thread.
        """
        ids_list = list(ids) if ids is not None else []
        n_ids = len(ids_list)
        n_nodes = min(self.n_candidate_nodes, n_ids) if n_ids > 0 else 0
        if n_nodes <= 0:
            return np.empty((0, 5), dtype=np.int64)

        while len(self.brute_force_nodes) < n_nodes:
            self.brute_force_nodes.append(Candidates_BF(self.window_size,self.window_step,self.n_lags,self.corr_threshold))
        if len(self.brute_force_nodes) > n_nodes:
            self.brute_force_nodes = self.brute_force_nodes[:n_nodes]

        if n_nodes <= 1:
            id_groups = [ids_list]
        else:
            n_series_node = math.ceil(n_ids / n_nodes)
            id_groups = [
                ids_list[i * n_series_node:(i + 1) * n_series_node]
                for i in range(n_nodes)
            ]

        node_inputs = []
        for s, shard_ids in enumerate(id_groups):
            if not shard_ids:
                continue
            ref_ids = [id_ for id_ in shard_ids if id_ in self.series_ids]
            if not ref_ids:
                continue
            node_inputs.append((s, curr_window_step, ref_ids))

        if not node_inputs:
            return np.empty((0, 5), dtype=np.int64)

        if worker_mode is None:
            worker_mode = self.exec

        payloads = [
            (
                s,
                self.brute_force_nodes[s],
                curr_window_step,
                self.ids,
                verbose,
                testing,
                ref_ids,
                True,
            )
            for s, curr_window_step, ref_ids in node_inputs
        ]

        bf_results = self._parallel_map(
            _bf_worker,
            payloads,
            mode=worker_mode,
            max_workers=len(payloads) if payloads else None,
            preserve_order=False,
        )

        row_parts = []
        for s, bf_node, result in bf_results:
            if bf_node is not None:
                self.brute_force_nodes[s] = bf_node
            rows = np.asarray(result, dtype=np.int64).reshape((-1, 5))
            if rows.size:
                row_parts.append(rows)
        if row_parts:
            return np.ascontiguousarray(np.vstack(row_parts), dtype=np.int64)
        return np.empty((0, 5), dtype=np.int64)
    
    def run_bf_exact_stomp(self, new_data_step, ids, verbose, testing, corr_val=True, monitor=True):
        self.verbose = verbose
        self.testing = testing

        val_mode = "thread" if self.parallel_validation else "sequential"
        t_update = time.perf_counter() if self.profile_enabled else None
        self._update_curr_data(new_data_step, ids)
        if t_update is not None:
            self._profile_add("bf.update_curr_data", time.perf_counter() - t_update)
        self.candidates = {}
        self._last_candidate_numeric_rows = None
        self.validated = {}
        if getattr(self, "_step_observer_enabled", False):
            self._validated_step = {}

        if self.exact_stomp_bf_node is None:
            self.exact_stomp_bf_node = Candidates_BF_ExactSTOMP(
                self.window_size,
                self.window_step,
                self.n_lags,
                self.corr_threshold,
                neg_corr=self.neg_corr,
            )

        accepted_rows, accepted_corrs, n_pairs, timing = self.exact_stomp_bf_node.run(
            self._curr_window_step(),
            self.ids,
            verbose=verbose,
            testing=testing,
            track_min_dist=bool(getattr(self, "track_min_dist", True)),
            numeric_rows=True,
        )
        self.total_candidates += int(n_pairs)
        self.tested_candidates += int(n_pairs)
        self.validated_candidates += int(np.asarray(accepted_rows).reshape((-1, 5)).shape[0])
        self.candidate_time += float(timing.get("candidate_time", 0.0) or 0.0)

        if getattr(self, "track_min_dist", True):
            step_min_dist = timing.get("min_dist", np.inf)
            step_min_pair = timing.get("pair_min_dist")
            if step_min_pair is not None and step_min_dist < self.min_dist:
                self.min_dist = float(step_min_dist)
                self.pair_min_dist = step_min_pair

        bookkeeping_before = self.artifact_bookkeeping_time
        record_t0 = time.perf_counter()
        self._record_correlated_numeric(accepted_rows, accepted_corrs, retain_validated=monitor)
        bookkeeping_delta = max(self.artifact_bookkeeping_time - bookkeeping_before, 0.0)
        self.validation_time += float(timing.get("validation_time", 0.0) or 0.0)
        self.validation_time += max((time.perf_counter() - record_t0) - bookkeeping_delta, 0.0)

        if monitor:
            start_time = time.time()
            self._monitor_corr(worker_mode=val_mode)
            end_time = time.time()
            self.monitor_time += end_time - start_time

        if verbose:
            self._print_state()
        self._profile_tick()

    def run_bf(self, new_data_step, ids, verbose, testing, corr_val=True, monitor=True):
        if _resolve_baseline_mode(getattr(self, "baseline_mode", "bruteforce")) == "exact_stomp":
            self.run_bf_exact_stomp(
                new_data_step,
                ids,
                verbose=verbose,
                testing=testing,
                corr_val=corr_val,
                monitor=monitor,
            )
            return

        self.verbose = verbose
        self.testing = testing

        cand_mode = "thread" if self.parallel_candidates else "sequential"
        val_mode = "thread" if self.parallel_validation else "sequential"

        t_update = time.perf_counter() if self.profile_enabled else None
        self._update_curr_data(new_data_step, ids)
        if t_update is not None:
            self._profile_add("bf.update_curr_data", time.perf_counter() - t_update)

        # candidates
        self.candidates = {}
        self._candidate_numeric_rows = None
        self._last_candidate_numeric_rows = None
        use_numeric_rows = True
        start_time = time.time()
        
        if self.parallel_candidates:
            result = self._run_bf_parallel(
                self._curr_window_step(),
                self.ids,
                verbose=verbose,
                testing=testing,
                worker_mode=cand_mode,
            )
            self._candidate_numeric_rows = result
        else:
            if not self.brute_force_nodes:
                self.brute_force_nodes.append(Candidates_BF(self.window_size,self.window_step,self.n_lags,self.corr_threshold))
            # original single-thread path
            result = self.brute_force_nodes[0].run(
                self._curr_window_step(),
                self.ids,
                verbose,
                testing,
                ref_ids=None,
                numeric_rows=use_numeric_rows,
            )
            self._candidate_numeric_rows = result
        end_time = time.time()
        self.candidate_time += end_time - start_time

        # validation
        bookkeeping_before = self.artifact_bookkeeping_time
        start_time = time.perf_counter()
        self._get_validated_corr(corr_val, force_mode=val_mode, retain_validated=monitor)
        end_time = time.perf_counter()
        bookkeeping_delta = max(self.artifact_bookkeeping_time - bookkeeping_before, 0.0)
        self.validation_time += max((end_time - start_time) - bookkeeping_delta, 0.0)

        if monitor:
            # monitoring
            start_time = time.time()
            self._monitor_corr(worker_mode=val_mode)
            end_time = time.time()
            self.monitor_time += end_time - start_time

        if verbose:
            self._print_state()
        self._profile_tick()
    
    def _update_hist_sketches(self):
        for s in range(self.n_nodes):
            for k, v in self.sketch_nodes[s].sketches.items():
                self.hist_sketches.append(v)   

    def _update_curr_sketches(self,new_sketches):
        if len(self.sketches) > self.n_lagged_windows:
            del self.sketches[min(self.sketches.keys())]
        self.sketches[self._curr_startTime()] = {}
        self.sketches[self._curr_startTime()].update(new_sketches)
    
    def run(self, new_data_step, ids, verbose, testing, corr_val=True, monitor=True):
        self.verbose = verbose
        self.testing = testing

        self._update_curr_data(new_data_step, ids)
        sketch_mode = "thread" if self.parallel_sketch else "sequential"
        cand_mode = "thread" if self.parallel_candidates else "sequential"
        val_mode = "thread" if self.parallel_validation else "sequential"

        # 1) sketches
        start_time = time.time()
        sketches = self._get_sketches(self._curr_window_step(), verbose, testing, worker_mode=sketch_mode)
        end_time = time.time()
        self.sketch_time += end_time - start_time
        self._update_curr_sketches(sketches)

        # 2) candidates via grids
        start_time = time.time()
        self._run_grids(verbose, testing, worker_mode=cand_mode)
        end_time = time.time()
        self.candidate_time += end_time - start_time

        # 3) validation (parallel)
        bookkeeping_before = self.artifact_bookkeeping_time
        start_time = time.perf_counter()
        self._get_validated_corr(corr_val, force_mode=val_mode, retain_validated=monitor)
        end_time = time.perf_counter()
        bookkeeping_delta = max(self.artifact_bookkeeping_time - bookkeeping_before, 0.0)
        self.validation_time += max((end_time - start_time) - bookkeeping_delta, 0.0)

        if monitor:
            # 4) monitoring (parallel 'out', sequential 'in')
            start_time = time.time()
            self._monitor_corr(worker_mode=val_mode)
            end_time = time.time()
            self.monitor_time += end_time - start_time

        if verbose:
            self._print_state()
        self._profile_tick()


    
class Sketches:
    def __init__(self,window_size,basic_window,window_step,seed,seed_toggle,n_vectors,grid_dimension,grid_nodes,preprocess,neg_corr=False,sketch_norm="z",full_vector_candidates=False):
        self.verbose = None
        # Parameters windows
        self.window_size = window_size
        self.basic_window = basic_window #divides window_size
        self.window_step = window_step #divides basic_window
        self.n_basic_windows = int(window_size/basic_window)
        self.basicDots_lag = int(basic_window/window_step+1)
        self.window_data = None
        self.window_data_original = None
        self.window_index = None
        self.last_origin = None
        self.window_stats = []
        self.series_ids = []
        self.previous_startTime = None
        self.curr_window_size = window_size
        self.preprocess = preprocess
        self.sketch_norm = str(sketch_norm) if sketch_norm is not None else "z"
        self.full_vector_candidates = bool(full_vector_candidates)
        self.neg_corr = bool(neg_corr)
        self._series_window_sums = None
        self._series_window_means = None
        self._raw_window_sums = None
        self._raw_window_sums_sq = None
        self._raw_window_sums_cu = None
        self._raw_window_sums_qu = None
        self._sid_lookup = None
        self._const_flags = None
        self._spiked_flags = None
        self.is_spiked = {}
        # Parameters sketches
        self._debug_raw_sketches = {}
        self.seed = seed
        self.seed_randomVector = seed
        self.seed_toggle = seed_toggle
        self.n_vectors = n_vectors
        self.basicRandomVector = None
        self.toggleVector = None
        self.diff_toggleVector = None
        self.intermediary_diff_toggleVector = None
        self.basicDots = [] #self.basicDots[window][serie][basic_series]
        self.basicFeatureSums = []
        self.incrementable_index = []
        self.previous_incrementable_index = None
        self.sketches = {}
        self._sketch_matrix = None
        self._orth_perm = None
        self._orth_signs = None
        self._sketch_keys = []
        debug_env = os.environ.get("CORRTRACK_DEBUG_SKETCHES", "").strip().lower()
        self._debug_sketches = debug_env in ("1", "true", "yes")
        # Parameters grids
        self.grid_dimensions = grid_dimension #2D grid
        self.n_grids = int(n_vectors/self.grid_dimensions) #divides n_vectors (sketch size) for 2D grid
        self.partitions = []
        self.grid_nodes = grid_nodes
        self.is_constant = {}
        self._toggle_weights = None
        self._random_vector_sums = None
        self._profile_callback = None

    def dump_state(self):
        state = {k: v for k, v in self.__dict__.items() if k != "grid_nodes"}
        return state

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj.grid_nodes = None
        return obj

    def load_state(self, state):
        grid_nodes = getattr(self, "grid_nodes", None)
        self.__dict__.update(state)
        self.grid_nodes = grid_nodes

    def _profile_add(self, key, elapsed):
        callback = getattr(self, "_profile_callback", None)
        if callback is not None:
            callback(key, elapsed)
    
    def _newStream(self,new_data_step,ids):
        new_data_step_index = new_data_step[0,:]
        new_data_step_values = np.asarray(new_data_step[1:,:], dtype=np.float64)
        processed_new_data_step_values = self._preprocess_data(new_data_step_values)
        added_count = processed_new_data_step_values.shape[1]

        n_series = new_data_step_values.shape[0]
        reset_sums = (
            self._series_window_sums is None
            or self._series_window_sums.shape[0] != n_series
        )
        reset_raw_stats = (
            self._raw_window_sums is None
            or self._raw_window_sums.shape[0] != n_series
        )

        #Update sliding windows
        if self.window_data is None:
            self.window_data_original = new_data_step_values
            self.window_data = processed_new_data_step_values
            n_diff = self.window_data.shape[1] - new_data_step_values.shape[1]
            if n_diff < 0:
                self.window_data = np.append(self.window_data[:, n_diff][:, None], self.window_data, axis=1)
            self.window_index = new_data_step_index[-self.window_data.shape[1]:]
            self.last_origin = new_data_step_values[:, -1]
            if reset_sums:
                self._series_window_sums = np.sum(self.window_data, axis=1, dtype=np.float64)
                reset_sums = False
        else:
            prev_total_cols = self.window_data.shape[1]
            prev_curr_size = min(self.window_size, prev_total_cols)
            window_start = prev_total_cols - prev_curr_size
            removed_cols = self.window_step if self.window_step and prev_total_cols >= self.window_size else 0
            prefix_excess = max(0, window_start)
            overlap_removed = min(prev_curr_size, max(0, removed_cols - prefix_excess))

            if overlap_removed > 0:
                start = window_start
                end = start + overlap_removed
                if not reset_sums:
                    dropped = self.window_data[:, start:end]
                    self._series_window_sums -= dropped.sum(axis=1)
                if not reset_raw_stats:
                    dropped_raw = self.window_data_original[:, start:end]
                    self._apply_raw_moment_delta(dropped_raw, -1.0)

            prev_curr_after_removal = prev_curr_size - overlap_removed
            remaining_start = window_start + overlap_removed
            available_slots = max(0, self.window_size - prev_curr_after_removal)
            extra_drop_needed = max(0, added_count - available_slots)
            extra_drop_from_old = min(prev_curr_after_removal, extra_drop_needed)

            if extra_drop_from_old > 0:
                start = remaining_start
                end = start + extra_drop_from_old
                if not reset_sums:
                    dropped = self.window_data[:, start:end]
                    self._series_window_sums -= dropped.sum(axis=1)
                if not reset_raw_stats:
                    dropped_raw = self.window_data_original[:, start:end]
                    self._apply_raw_moment_delta(dropped_raw, -1.0)

            old_remaining = prev_curr_after_removal - extra_drop_from_old
            new_curr_size = min(self.window_size, old_remaining + added_count)
            included_new = max(0, min(added_count, new_curr_size - old_remaining))

            if included_new > 0:
                if not reset_sums:
                    new_slice = processed_new_data_step_values[:, -included_new:]
                    self._series_window_sums += new_slice.sum(axis=1)
                if not reset_raw_stats:
                    new_raw_slice = new_data_step_values[:, -included_new:]
                    self._apply_raw_moment_delta(new_raw_slice, 1.0)

            if self.window_data.shape[1] >= self.window_size:
                if self.window_data.shape[1] >= self.window_size + self.basic_window:
                    self.window_index = self.window_index[self.window_step:]
                self.window_data = self.window_data[:, self.window_step:]
                self.window_data_original = self.window_data_original[:, self.window_step:]

            self.window_index = np.append(self.window_index, new_data_step_index)
            self.window_data_original = np.append(self.window_data_original, new_data_step_values, axis=1)
            self.window_data = np.append(self.window_data, processed_new_data_step_values, axis=1)
            self.last_origin = new_data_step_values[:, -1]
        
        if isinstance(ids, (list, tuple)):
            series_list = list(ids)
        else:
            series_list = list(ids)
        self.series_ids = series_list
        self.window_data = self.window_data.astype(float)
        self._update_curr_window_size()
        self._update_window_mean_stats(force_recompute=reset_sums)
        self._update_window_raw_stats(force_recompute=reset_raw_stats)

        if self.curr_window_size >= self.window_size:
            self._const_flags = None
            self._spiked_flags = None
            if (
                _cy_compute_constant_flags is not None
                and self._raw_window_sums is not None
                and self._raw_window_sums_sq is not None
                and self._raw_window_sums_cu is not None
                and self._raw_window_sums_qu is not None
            ):
                const_flags, spiked_flags = _cy_compute_constant_flags(
                    np.asarray(self._raw_window_sums, dtype=np.float64),
                    np.asarray(self._raw_window_sums_sq, dtype=np.float64),
                    np.asarray(self._raw_window_sums_cu, dtype=np.float64),
                    np.asarray(self._raw_window_sums_qu, dtype=np.float64),
                    int(self.curr_window_size),
                )
                const_flags = np.asarray(const_flags, dtype=np.uint8)
                spiked_flags = np.asarray(spiked_flags, dtype=np.uint8)
                self._const_flags = const_flags
                self._spiked_flags = spiked_flags
                self.is_constant = {}
                self.is_spiked = {}
                for idx, series_id in enumerate(self.series_ids):
                    if series_id is None:
                        continue
                    self.is_constant[series_id] = bool(self._const_flags[idx])
                    self.is_spiked[series_id] = bool(self._spiked_flags[idx])
            else:
                const_flags = []
                spiked_flags = []
                for idx, series_id in enumerate(self.series_ids):
                    stats = self._raw_stats_for_series(idx)
                    if stats is None:
                        flag = True
                        spiked = False
                    else:
                        flag = CorrTrack.is_near_constant(
                            var_sum=stats["var_sum"],
                            n=stats["n"],
                            std_thresh=1e-3,
                        )
                        spiked = CorrTrack.is_structurally_spiked(
                            var_sum=stats["var_sum"],
                            n=stats["n"],
                            mu4_sum=stats["mu4_sum"],
                        )
                    if series_id is not None:
                        self.is_constant[series_id] = flag
                        self.is_spiked[series_id] = spiked
                    const_flags.append(flag)
                    spiked_flags.append(spiked)
                self._const_flags = np.asarray(const_flags, dtype=np.uint8)
                self._spiked_flags = np.asarray(spiked_flags, dtype=np.uint8)
            
    def _preprocess_data(self, data):
        t = np.asarray(data, dtype=np.float64)

        if self.preprocess:
            if self.last_origin is not None:
                data = np.append(np.transpose([self.last_origin]), data, axis=1)
            elif data.shape[1] > 0:
                # Mirror first column so differencing keeps at least one sample when warmup is missing
                data = np.append(data[:, :1], data, axis=1)
            t = np.asarray(data, dtype=np.float64)
            if t.shape[1] <= 1:
                diff_t = np.zeros((t.shape[0], 1)) if t.shape[1] else np.zeros((t.shape[0], 0))
            else:
                diff_t = t[:, 1:] - t[:, :-1]                     # 1. Differencing
            #clipped_t = np.clip(diff_t, -3, 3)                   # 2. Linear clipping
            t = diff_t

        return np.asarray(t, dtype=np.float64)

    def _sketch_feature_width(self):
        return int(self.basic_window)

    def _sketch_feature_blocks(self, window_blocks):
        features = np.asarray(window_blocks, dtype=np.float64)
        feature_sums = np.sum(features, axis=2, dtype=np.float64)
        return features, feature_sums

    def _set_feature_window_means(self, feature_sums):
        sums = np.asarray(feature_sums, dtype=np.float64)
        if sums.ndim != 2 or sums.size == 0:
            self._series_window_means = np.zeros(sums.shape[0] if sums.ndim else 0, dtype=np.float64)
            return
        width = float(self._sketch_feature_width())
        denom = max(1.0, float(sums.shape[1]) * width)
        self._series_window_means = sums.sum(axis=1, dtype=np.float64) / denom
    
    def _update_previous_startTime(self):
        if self.window_data.shape[1] >= self.window_size:
            self.previous_startTime = self._curr_startTime()
            self.previous_incrementable_index = self._curr_incrementable_startTime()
        elif self.previous_startTime is None:
            self.previous_startTime = self.window_index[0]
            self.previous_startTime_index = 0
    
    def _update_curr_window_size(self):
        self.curr_window_size = min(self.window_size, self.window_data.shape[1])
        return None
        previous_startTime_index = 0
        if self.previous_startTime is not None:
            previous_startTime_index = np.where(self.window_index==self.previous_startTime)[0][0]
            #print("\nprevious_startTime: ",self.window_index[previous_startTime_index])
        if (self.window_data.shape[1] - previous_startTime_index) < self.window_size:
            self.curr_window_size = self.window_data.shape[1] - previous_startTime_index
        elif self.window_index[-self.window_size] < self.previous_startTime:
            diff_new_window_index = self.window_index[-self.window_size] - previous_startTime_index
            self.curr_window_size = self.window_size+diff_new_window_index
        else:
            self.curr_window_size = self.window_size
    
    def _curr_window(self):
        return self.window_data[:,-self.curr_window_size:]
    
    def _curr_window_original(self):
        if self.window_data_original is None or self.curr_window_size is None:
            return None
        return self.window_data_original[:, -self.curr_window_size:]
    
    def _curr_window_times(self):
        return self.window_index[-self.curr_window_size:]
    
    def _curr_startTime(self):
        return self.window_index[-self.curr_window_size]
    
    def _curr_incrementable_startTime(self): # start time of the 2nd basic window
        pos = len(self.window_index) - self.curr_window_size + self.basic_window
        if pos < len(self.window_index):
            return self.window_index[pos]
        return self._curr_startTime() + self.basic_window

    def _update_window_mean_stats(self, force_recompute=False):
        if self.window_data is None:
            self._series_window_sums = None
            self._series_window_means = None
            return

        n_series = self.window_data.shape[0]
        if self.curr_window_size is None or self.curr_window_size <= 0:
            current_window_sum = np.zeros(n_series, dtype=np.float64)
        elif force_recompute or self._series_window_sums is None or self._series_window_sums.shape[0] != n_series:
            current_window = self._curr_window()
            current_window_sum = np.sum(current_window, axis=1, dtype=np.float64)
        else:
            current_window_sum = self._series_window_sums

        self._series_window_sums = current_window_sum
        if self.curr_window_size is None or self.curr_window_size <= 0:
            self._series_window_means = np.zeros(n_series, dtype=np.float64)
        else:
            self._series_window_means = current_window_sum / float(self.curr_window_size)

    def _apply_raw_moment_delta(self, slice_data, sign):
        if (
            slice_data is None
            or self._raw_window_sums is None
            or self._raw_window_sums_sq is None
            or self._raw_window_sums_cu is None
            or self._raw_window_sums_qu is None
        ):
            return
        arr = np.asarray(slice_data, dtype=np.float64)
        if arr.size == 0:
            return
        scale = float(sign)
        self._raw_window_sums += scale * arr.sum(axis=1, dtype=np.float64)
        self._raw_window_sums_sq += scale * np.sum(arr * arr, axis=1, dtype=np.float64)
        self._raw_window_sums_cu += scale * np.sum(arr ** 3, axis=1, dtype=np.float64)
        self._raw_window_sums_qu += scale * np.sum(arr ** 4, axis=1, dtype=np.float64)

    def _update_window_raw_stats(self, force_recompute=False):
        if self.window_data_original is None:
            self._raw_window_sums = None
            self._raw_window_sums_sq = None
            self._raw_window_sums_cu = None
            self._raw_window_sums_qu = None
            return

        n_series = self.window_data_original.shape[0]
        if self.curr_window_size is None or self.curr_window_size <= 0:
            zeros = np.zeros(n_series, dtype=np.float64)
            self._raw_window_sums = zeros.copy()
            self._raw_window_sums_sq = zeros.copy()
            self._raw_window_sums_cu = zeros.copy()
            self._raw_window_sums_qu = zeros.copy()
            return

        needs_reset = (
            force_recompute
            or self._raw_window_sums is None
            or self._raw_window_sums.shape[0] != n_series
        )
        if needs_reset:
            current_window = self._curr_window_original()
            if current_window is None:
                zeros = np.zeros(n_series, dtype=np.float64)
                self._raw_window_sums = zeros.copy()
                self._raw_window_sums_sq = zeros.copy()
                self._raw_window_sums_cu = zeros.copy()
                self._raw_window_sums_qu = zeros.copy()
                return
            arr = np.asarray(current_window, dtype=np.float64)
            self._raw_window_sums = np.sum(arr, axis=1, dtype=np.float64)
            self._raw_window_sums_sq = np.sum(arr * arr, axis=1, dtype=np.float64)
            self._raw_window_sums_cu = np.sum(arr ** 3, axis=1, dtype=np.float64)
            self._raw_window_sums_qu = np.sum(arr ** 4, axis=1, dtype=np.float64)
        else:
            # no action needed; incremental updates already applied
            return

    def _raw_stats_for_series(self, index):
        if (
            self._raw_window_sums is None
            or self._raw_window_sums_sq is None
            or self._raw_window_sums_cu is None
            or self._raw_window_sums_qu is None
            or self.curr_window_size is None
            or self.curr_window_size <= 0
        ):
            return None
        n = int(self.curr_window_size)
        if n <= 0 or index >= self._raw_window_sums.shape[0]:
            return None
        sum1 = float(self._raw_window_sums[index])
        sum2 = float(self._raw_window_sums_sq[index])
        sum3 = float(self._raw_window_sums_cu[index])
        sum4 = float(self._raw_window_sums_qu[index])
        mean = sum1 / n if n > 0 else 0.0
        var_sum = max(sum2 - (sum1 * sum1) / n, 0.0)
        mu4_sum = (
            sum4
            - 4.0 * mean * sum3
            + 6.0 * (mean ** 2) * sum2
            - 4.0 * (mean ** 3) * sum1
            + n * (mean ** 4)
        )
        return {
            "n": n,
            "mean": mean,
            "var_sum": var_sum,
            "mu4_sum": mu4_sum,
        }


    def _refresh_toggle_weights(self):
        if self.basicRandomVector is None or self.toggleVector is None:
            self._toggle_weights = None
            self._random_vector_sums = None
            return
        base = np.array(self.basicRandomVector, dtype=np.float64, copy=False)
        toggle = np.array(self.toggleVector, dtype=np.float64, copy=False)
        self._toggle_weights = toggle[:, :, None] * base[None, :, :]
        self._random_vector_sums = self._toggle_weights.sum(axis=(0, 2))

    def _mean_adjust_matrix(self, vectors):
        if (
            vectors is None
            or vectors.size == 0
            or self._series_window_means is None
            or self._random_vector_sums is None
        ):
            return vectors

        n_series = vectors.shape[0]
        if n_series == 0 or self.curr_window_size is None or self.curr_window_size <= 0:
            return vectors

        mu = np.asarray(self._series_window_means[:n_series], dtype=np.float64)
        random_sums = np.asarray(self._random_vector_sums, dtype=np.float64)
        if mu.size == 0 or random_sums.size == 0:
            return vectors
        adjustment = mu[:, None] * random_sums[None, :]
        return vectors - adjustment

    def _mean_adjust_vector(self, vector, series_idx):
        if (
            vector is None
            or self._series_window_means is None
            or self._random_vector_sums is None
            or self.curr_window_size is None
            or self.curr_window_size <= 0
            or series_idx >= len(self._series_window_means)
        ):
            return vector
        mu_value = float(self._series_window_means[series_idx])
        random_sums = np.asarray(self._random_vector_sums, dtype=np.float64)
        if random_sums.size == 0:
            return vector
        return vector - mu_value * random_sums

    def _kernel_norm_inputs(self, n_series, norm_mode):
        if norm_mode != 1:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        if self._series_window_means is None or self._random_vector_sums is None:
            return None, None
        mean_vec = np.asarray(self._series_window_means[:n_series], dtype=np.float64)
        random_sums = np.asarray(self._random_vector_sums, dtype=np.float64)
        return mean_vec, random_sums

    def _sketch_norm_mode_code(self):
        mode = (self.sketch_norm or "z").lower()
        if mode == "mean_l2":
            return 1
        return 0
    
    def _generate_randomVectors(self):
        base_rng = np.random.RandomState(self.seed_randomVector)
        feature_width = self._sketch_feature_width()
        self.basicRandomVector = (1/np.sqrt(self.n_vectors))*base_rng.choice([1,-1], size=(self.n_vectors,feature_width), replace=True)
        toggle_rng = np.random.RandomState(self.seed_toggle)
        self.toggleVector = toggle_rng.choice([1,-1], size=(self.n_basic_windows,self.n_vectors), replace=True)
        self.diff_toggleVector = self.toggleVector[1:]/self.toggleVector[:-1]
        self._refresh_toggle_weights()
        if(self.verbose):
            print("\nBasic random vectors")
            print(self.basicRandomVector)
            print("Toggle vector (one per basic window)")
            print(self.toggleVector)
    
    def _update_randomVectors(self):
        previous_n_basic_windows = len(self.toggleVector)
        diff_n_basic_windows = self.n_basic_windows - previous_n_basic_windows
        if diff_n_basic_windows > 0:
            toggle_rng = np.random.RandomState(self.seed_toggle)
            self.toggleVector = toggle_rng.choice([1,-1], size=(self.n_basic_windows,self.n_vectors), replace=True)
            new_diff = self.toggleVector[previous_n_basic_windows:]/self.toggleVector[(previous_n_basic_windows-1):-1]
            self.diff_toggleVector = np.append(self.diff_toggleVector,new_diff,axis=0)
        elif diff_n_basic_windows < 0:
            self.toggleVector = self.toggleVector[:self.n_basic_windows,:]
            if diff_n_basic_windows == -1:
                self.intermediary_diff_toggleVector = self.diff_toggleVector
            else:
                self.intermediary_diff_toggleVector = self.diff_toggleVector[:(diff_n_basic_windows+1)]
            self.diff_toggleVector = self.diff_toggleVector[:diff_n_basic_windows]
        self._refresh_toggle_weights()
        if(self.verbose):
            print("\nBasic random vectors")
            print(self.basicRandomVector)
            print("Toggle vector (one per basic window)")
            print(self.toggleVector)

    def _get_sketches(self):
        if self.curr_window_size >= self.window_size:
            if self.basicRandomVector is None:
                self._generate_randomVectors()
            if len(self.incrementable_index) > 0 and (self.incrementable_index[0] <= self._curr_startTime() or self.previous_startTime == self._curr_startTime()):
                self._incremental_sketches()
            else:
                self._sketches_from_scratch()
            self._update_previous_startTime()
            return True
        else:
            if(self.verbose):
                self._print_curr_window()
                print("\nSkip sketches (window not full)")
            self._update_previous_startTime()
            return False

    def _clean_obsolete_basicDots(self):
        while len(self.incrementable_index)>0 and self.incrementable_index[0] < self._curr_startTime():
            if(self.verbose):
                print("Delete obsolete basic dots (incrementable index ",self.incrementable_index[0],")")
            del self.basicDots[0]
            if self.basicFeatureSums:
                del self.basicFeatureSums[0]
            del self.incrementable_index[0]
        
    def _sketches_from_scratch(self):
        if(self.verbose):
            self._print_curr_window()
            print("\nSketches from scratch")
        
        if self._toggle_weights is None:
            self._refresh_toggle_weights()

        current_window = np.array(self._curr_window(), dtype=np.float64, copy=False)
        n_series = current_window.shape[0]
        self.incrementable_index.append(self._curr_incrementable_startTime())

        if n_series == 0:
            self.basicDots.append(np.empty((0, self.n_basic_windows, self.n_vectors), dtype=np.float64))
            self.basicFeatureSums.append(np.empty((0, self.n_basic_windows), dtype=np.float64))
            self.sketches = {}
            self._sketch_matrix = None
            self._sketch_keys = []
            return

        window_blocks = current_window.reshape(n_series, self.n_basic_windows, self.basic_window)
        feature_blocks, feature_sums = self._sketch_feature_blocks(window_blocks)
        self._set_feature_window_means(feature_sums)
        weights = np.array(self._toggle_weights, dtype=np.float64, copy=False)

        norm_mode = self._sketch_norm_mode_code()
        if _cy_build_sketch_matrix is not None:
            mean_vec, random_sums = self._kernel_norm_inputs(n_series, norm_mode)
            if mean_vec is not None and random_sums is not None:
                series_dots, raw_matrix, norm_matrix = _cy_build_sketch_matrix(
                    np.ascontiguousarray(feature_blocks, dtype=np.float64),
                    np.ascontiguousarray(weights, dtype=np.float64),
                    mean_vec,
                    random_sums,
                    int(norm_mode),
                )
            else:
                series_dots = _compute_series_dots(feature_blocks, weights)
                raw_matrix = np.sum(series_dots, axis=1)
                norm_matrix = self._normalize_sketch_matrix(raw_matrix)
        else:
            series_dots = _compute_series_dots(feature_blocks, weights)
            raw_matrix = np.sum(series_dots, axis=1)
            norm_matrix = self._normalize_sketch_matrix(raw_matrix)
        curr_start = self._curr_startTime()
        window_size = self.window_size

        self.basicDots.append(np.array(series_dots, dtype=np.float64, copy=True))
        self.basicFeatureSums.append(np.array(feature_sums, dtype=np.float64, copy=True))
        self.sketches = {}
        if n_series == 0:
            self._sketch_matrix = None
            return
        if self._debug_sketches or self.testing:
            for s in range(n_series):
                try:
                    self._debug_raw_sketches[(self.series_ids[s], curr_start, window_size)] = np.array(
                        raw_matrix[s], dtype=np.float64, copy=True
                    )
                except Exception:
                    pass
        self._sketch_matrix = norm_matrix
        series_ids = list(self.series_ids)
        if len(series_ids) < n_series:
            series_ids = series_ids + [None] * (n_series - len(series_ids))
        self._sketch_keys = [(series_ids[s], curr_start, window_size) for s in range(n_series)]
        self.sketches = dict(zip(self._sketch_keys, norm_matrix))
    
    def _print_incremental_intermediary_window(self):
        if(self.verbose):
            incrementable_index_index = np.where(self.window_index==self.incrementable_index[0])[0][0]
            window_incrementable_index_index = incrementable_index_index#-self.window_step
            if(self.verbose):
                print("\nIntermediary window")
                print(self.window_index[window_incrementable_index_index:(window_incrementable_index_index+self.window_size)])
                if(self.testing):
                    print("\nFirst incrementable window dots")
                    print(self.window_data[:,window_incrementable_index_index:(window_incrementable_index_index+self.window_size)])

    def _incremental_sketches_intermediary(self,diff_n_basic_windows):
        if diff_n_basic_windows < 0:
            while self.previous_incrementable_index < self.incrementable_index[0]:
                if(self.verbose):
                    print("Delete obsolete basic dots (incrementable index ",self.incrementable_index[0],")")
                del self.basicDots[0]
                if self.basicFeatureSums:
                    del self.basicFeatureSums[0]
                del self.incrementable_index[0]
            
            n_intermediary_windows = -diff_n_basic_windows*int(self.basic_window/self.window_step)
            for i in range(n_intermediary_windows):
                #clean obsolete basic dots
                if len(self.incrementable_index)>(i+1): #TODO: -w2
                    incrementable_index_index = np.where(self.window_index==self.incrementable_index[(i+1)])[0][0]
                    while len(self.incrementable_index)>0 and self.incrementable_index[0] < self.window_index[incrementable_index_index]:
                        if(self.verbose):
                            print("Delete obsolete basic dots (incrementable index ",self.incrementable_index[0],")")
                        del self.basicDots[0]
                        if self.basicFeatureSums:
                            del self.basicFeatureSums[0]
                        del self.incrementable_index[0]
                self._print_incremental_intermediary_window()
                self._print_incremental_step()
                print("No new dots computation yet")
                incrementable_index_index = np.where(self.window_index==self.incrementable_index[0])[0][0]
                new_incrementable_index = self.window_index[incrementable_index_index+self.basic_window]
                self.incrementable_index.append(new_incrementable_index)
                self.sketches = {}
                base = np.asarray(self.basicDots[0], dtype=np.float64)
                n_series = base.shape[0]
                if n_series == 0:
                    self.basicDots.append(np.empty((0, self.n_basic_windows, self.n_vectors), dtype=np.float64))
                    self.basicFeatureSums.append(np.empty((0, self.n_basic_windows), dtype=np.float64))
                    self._sketch_matrix = None
                    self._sketch_keys = []
                    continue
                curr_start = self._curr_startTime()
                window_size = self.window_size
                diff = np.asarray(self.intermediary_diff_toggleVector, dtype=np.float64)
                series_dots = base[:, 1:(self.n_basic_windows+1), :] * diff
                self.basicDots.append(series_dots)
                if self.basicFeatureSums:
                    feature_sums = np.asarray(self.basicFeatureSums[0], dtype=np.float64)[:, 1:(self.n_basic_windows+1)]
                else:
                    feature_sums = np.zeros((n_series, self.n_basic_windows), dtype=np.float64)
                self.basicFeatureSums.append(np.array(feature_sums, dtype=np.float64, copy=True))
                self._set_feature_window_means(feature_sums)
                raw_matrix = np.sum(series_dots, axis=1)

                norm_mode = self._sketch_norm_mode_code()
                if _cy_apply_orth_and_normalize is not None:
                    mean_vec, random_sums = self._kernel_norm_inputs(n_series, norm_mode)
                    if mean_vec is not None and random_sums is not None:
                        raw_matrix, norm_matrix = _cy_apply_orth_and_normalize(
                            np.ascontiguousarray(raw_matrix, dtype=np.float64),
                            mean_vec,
                            random_sums,
                            int(norm_mode),
                        )
                    else:
                        norm_matrix = self._normalize_sketch_matrix(raw_matrix)
                else:
                    norm_matrix = self._normalize_sketch_matrix(raw_matrix)

                if self._debug_sketches or self.testing:
                    for s in range(n_series):
                        try:
                            self._debug_raw_sketches[(self.series_ids[s], curr_start, window_size)] = np.array(
                                raw_matrix[s], dtype=np.float64, copy=True
                            )
                        except Exception:
                            pass
                self._sketch_matrix = norm_matrix
                series_ids = list(self.series_ids)
                if len(series_ids) < n_series:
                    series_ids = series_ids + [None] * (n_series - len(series_ids))
                self._sketch_keys = [(series_ids[s], curr_start, window_size) for s in range(n_series)]
                self.sketches = dict(zip(self._sketch_keys, norm_matrix))
            del self.basicDots[0]
            if self.basicFeatureSums:
                del self.basicFeatureSums[0]
            del self.incrementable_index[0]                

    def _print_incremental_step(self):
        if(self.verbose):
            if self.previous_startTime == self._curr_startTime():
                print("\nRepeat dots and include basic window(s)")
                if(self.testing):
                    print("\nPrevious window dots")
                    self._print_basicDots(-1)
                print("\nIncrementing from ",self._curr_startTime()," (basic window 1)")
            else:
                incrementable_index_index = np.where(self.window_index==self.incrementable_index[0])[0][0]
                window_incrementable_index_index = incrementable_index_index-self.basic_window
                window_incrementable_index = self.window_index[window_incrementable_index_index]
                print("\nIncremental dots (-",len(self.basicDots),")")
                if(self.testing):
                    print("\nFirst incrementable window dots")
                    self._print_basicDots(0)
                print("\nIncrementing from window starting at", window_incrementable_index," (basic window 2)")

    def _incremental_sketches(self):

        base_shape = np.asarray(self.basicDots[0], dtype=np.float64).shape
        previous_n_basic_windows = base_shape[1] if len(base_shape) > 1 else 0
        diff_n_basic_windows = self.n_basic_windows - previous_n_basic_windows               
        if diff_n_basic_windows < 0:
            self._incremental_sketches_intermediary(diff_n_basic_windows)
        
        n_new_basic_windows = max(diff_n_basic_windows,1)
        new_basic_data = np.array(
            self.window_data[:,-(n_new_basic_windows*self.basic_window):],
            dtype=np.float64,
            copy=False,
        )
        n_series = new_basic_data.shape[0]
        if self._toggle_weights is None:
            self._refresh_toggle_weights()

        if n_new_basic_windows > 0:
            weights_subset = np.array(
                self._toggle_weights[-n_new_basic_windows:],
                dtype=np.float64,
                copy=False,
            )
            new_blocks = new_basic_data.reshape(n_series, n_new_basic_windows, self.basic_window)
            new_feature_blocks, new_feature_sums = self._sketch_feature_blocks(new_blocks)
            new_dots = None
        else:
            weights_subset = np.empty((0, self.n_vectors, self.basic_window), dtype=np.float64)
            new_feature_blocks = np.empty((n_series, 0, self.basic_window), dtype=np.float64)
            new_dots = np.zeros((n_series, 0, self.n_vectors), dtype=np.float64)
            new_feature_sums = np.zeros((n_series, 0), dtype=np.float64)

        if(self.verbose):
            self._print_curr_window()
        
        self._clean_obsolete_basicDots()
        if not self.basicDots:
            self._sketches_from_scratch()
            return
        self._print_incremental_step()
        self.incrementable_index.append(self._curr_incrementable_startTime())
        self.sketches = {}
        same_start = self.previous_startTime == self._curr_startTime()
        curr_start = self._curr_startTime()
        window_size = self.window_size
        raw_matrix = np.empty((n_series, self.n_vectors), dtype=np.float64)
        apply_diff = False
        if same_start and len(self.basicDots) >= 2:
            base = np.ascontiguousarray(np.asarray(self.basicDots[-2], dtype=np.float64))
            diff_arg = np.empty((0, self.n_vectors), dtype=np.float64)
            if len(self.basicFeatureSums) >= 2:
                base_feature_sums = np.asarray(self.basicFeatureSums[-2], dtype=np.float64)
            else:
                base_feature_sums = np.zeros((n_series, 0), dtype=np.float64)
        else:
            base = np.ascontiguousarray(np.asarray(self.basicDots[0], dtype=np.float64)[:, 1:, :])
            diff_arg = np.ascontiguousarray(np.asarray(self.diff_toggleVector, dtype=np.float64)[:base.shape[1], :])
            apply_diff = True
            if self.basicFeatureSums:
                base_feature_sums = np.asarray(self.basicFeatureSums[0], dtype=np.float64)[:, 1:]
            else:
                base_feature_sums = np.zeros((n_series, 0), dtype=np.float64)

        norm_mode = self._sketch_norm_mode_code()
        updated = None
        norm_matrix = None
        if base_feature_sums.size and new_feature_sums.size:
            updated_feature_sums = np.concatenate((base_feature_sums, new_feature_sums), axis=1)
        elif not base_feature_sums.size:
            updated_feature_sums = base_feature_sums.reshape(n_series, 0) if not new_feature_sums.size else new_feature_sums
        else:
            updated_feature_sums = base_feature_sums
        self._set_feature_window_means(updated_feature_sums)

        if _cy_incremental_combine_and_normalize is not None:
            mean_vec, random_sums = self._kernel_norm_inputs(n_series, norm_mode)
            if mean_vec is not None and random_sums is not None:
                try:
                    t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
                    updated, raw_matrix, norm_matrix = _cy_incremental_combine_and_normalize(
                        np.ascontiguousarray(base, dtype=np.float64),
                        np.ascontiguousarray(diff_arg, dtype=np.float64),
                        np.ascontiguousarray(new_feature_blocks, dtype=np.float64),
                        np.ascontiguousarray(weights_subset, dtype=np.float64),
                        mean_vec,
                        random_sums,
                        int(norm_mode),
                        int(apply_diff),
                    )
                    if t0 is not None:
                        self._profile_add("sketch.incremental_kernel", time.perf_counter() - t0)
                except Exception:
                    updated = None
                    norm_matrix = None

        if updated is None or norm_matrix is None:
            t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
            if new_dots is None:
                new_dots = _compute_series_dots(new_feature_blocks, weights_subset)
            base_work = base * diff_arg if apply_diff else base
            if base_work.size == 0 and new_dots.size == 0:
                updated = base_work.reshape(n_series, 0, self.n_vectors)
            elif base_work.size == 0:
                updated = new_dots
            elif new_dots.size == 0:
                updated = base_work
            else:
                updated = np.concatenate((base_work, new_dots), axis=1)
            raw_matrix = np.sum(updated, axis=1)
            if _cy_apply_orth_and_normalize is not None:
                mean_vec, random_sums = self._kernel_norm_inputs(n_series, norm_mode)
                if mean_vec is not None and random_sums is not None:
                    raw_matrix, norm_matrix = _cy_apply_orth_and_normalize(
                        np.ascontiguousarray(raw_matrix, dtype=np.float64),
                        mean_vec,
                        random_sums,
                        int(norm_mode),
                    )
                else:
                    norm_matrix = self._normalize_sketch_matrix(raw_matrix)
            else:
                norm_matrix = self._normalize_sketch_matrix(raw_matrix)
            if t0 is not None:
                self._profile_add("sketch.incremental_numpy_fallback", time.perf_counter() - t0)

        self.basicDots.append(np.asarray(updated, dtype=np.float64))
        self.basicFeatureSums.append(np.array(updated_feature_sums, dtype=np.float64, copy=True))

        if self._debug_sketches or self.testing:
            for s in range(n_series):
                try:
                    self._debug_raw_sketches[(self.series_ids[s], curr_start, window_size)] = np.array(
                        raw_matrix[s], dtype=np.float64, copy=True
                    )
                except Exception:
                    pass
        self._sketch_matrix = norm_matrix
        series_ids = list(self.series_ids)
        if len(series_ids) < n_series:
            series_ids = series_ids + [None] * (n_series - len(series_ids))
        t_materialize = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        self._sketch_keys = [(series_ids[s], curr_start, window_size) for s in range(n_series)]
        self.sketches = dict(zip(self._sketch_keys, norm_matrix))
        if t_materialize is not None:
            self._profile_add("sketch.incremental_materialize", time.perf_counter() - t_materialize)

        del self.basicDots[0]
        if self.basicFeatureSums:
            del self.basicFeatureSums[0]
        del self.incrementable_index[0]

    def update_window_step(self,window_step):
        self.window_step = window_step
        self.basicDots_lag = int(self.basic_window/window_step+1)
        
    def update_window_size(self,window_size):
        self.window_size = window_size            
        self.n_basic_windows = int(window_size/self.basic_window)
        self._update_randomVectors()
    
    def _print_curr_window(self):
        print("\nCurrent window")
        print(self._curr_window_times())
        if(self.testing):
            print(self._curr_window())
    
    def _print_basicDots(self,index):
        if len(self.basicDots)>0:
            bD = self.basicDots[index]
            if isinstance(bD, np.ndarray):
                for s in range(bD.shape[0]):
                    print("Serie ",s,"(split by basic windows)")
                    res = [bw for bw in bD[s]]
                    print(" ".join(map(str, res)))
            else:
                for s in range(len(bD)):
                    print("Serie ",s,"(split by basic windows)")
                    res = [bw for bw in bD[s]] # flatten array
                    print(" ".join(map(str, res)))

    def print_sketches(self):
        if len(self.sketches)>0:
            for k, v in self.sketches.items():
                print("Serie ",k[0],", startTime ",k[1],", window size ",k[2])
                print(v)

    def print_partitions(self):
        if len(self.partitions)>0:
            for p in range(len(self.partitions)):
                print("Partition ",p+1)
                print(self.partitions[p])

    def _print_state(self):
        if len(self.basicDots)>0:
            print("Dots")
            self._print_basicDots(-1)
            print("\nSketches")
            self.print_sketches()
            print("\nPartitions")
            self.print_partitions()

    def _z_normalize_sketch(self, v):
        arr = np.array(v, dtype=np.float64, copy=False)
        if arr.size == 0:
            return arr
        mean = float(np.mean(arr))
        centered = arr - mean
        std = float(np.std(arr))
        if not np.isfinite(std) or std <= 0.0:
            return np.zeros_like(arr)
        return centered / std

    def _mean_l2_normalize_sketch(self, v):
        arr = np.array(v, dtype=np.float64, copy=False)
        if arr.size == 0:
            return arr
        centered = arr - float(np.mean(arr))
        norm = float(np.linalg.norm(centered))
        if not np.isfinite(norm) or norm <= 0.0:
            return np.zeros_like(arr)
        return centered / norm

    def _normalize_sketch(self, v):
        mode = (self.sketch_norm or "z").lower()
        if mode == "mean_l2":
            return self._mean_l2_normalize_sketch(v)
        # default to z-normalization
        return self._z_normalize_sketch(v)

    def _normalize_sketch_matrix(self, matrix):
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.size == 0:
            return arr
        if arr.ndim == 1:
            arr = arr[None, :]
        mode = (self.sketch_norm or "z").lower()
        if mode == "mean_l2":
            adjusted = self._mean_adjust_matrix(arr)
            norms = np.linalg.norm(adjusted, axis=1, keepdims=True)
            out = np.zeros_like(adjusted)
            valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 0)
            if np.any(valid):
                out[valid] = adjusted[valid] / norms[valid]
            return out
        mean = np.mean(arr, axis=1, keepdims=True)
        centered = arr - mean
        std = np.std(arr, axis=1, keepdims=True)
        out = np.zeros_like(centered)
        valid = np.isfinite(std[:, 0]) & (std[:, 0] > 0)
        if np.any(valid):
            out[valid] = centered[valid] / std[valid]
        return out

    def safe_normalize(self,v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return np.zeros_like(v)

    def _ensure_orth_transform(self):
        if (
            self._orth_perm is not None
            and self._orth_signs is not None
            and self._orth_perm.shape[0] == self.n_vectors
        ):
            return
        seed = (int(self.seed_randomVector) + 7919) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        self._orth_perm = rng.permutation(self.n_vectors)
        self._orth_signs = rng.choice([1, -1], size=self.n_vectors)

    def _apply_orth_transform(self, vector):
        if vector is None:
            return vector
        self._ensure_orth_transform()
        perm = self._orth_perm
        signs = self._orth_signs
        if (
            perm is None
            or signs is None
            or vector.shape[0] != perm.shape[0]
        ):
            return vector
        return signs * vector[perm]

    def partition_sketches(self, window_size):
        if len(self.sketches) == 0 and not self._sketch_keys:
            return

        total_dim = self.n_grids * self.grid_dimensions
        if total_dim <= 0:
            return

        keys = self._sketch_keys
        matrix = self._sketch_matrix
        if (
            matrix is None
            or not keys
            or matrix.shape[0] != len(keys)
            or any(k[2] != window_size for k in keys)
        ):
            keys = []
            rows = []
            for k, v in self.sketches.items():
                if k[2] != window_size:
                    continue
                v_arr = np.asarray(v, dtype=np.float64)
                if v_arr.size == 0:
                    continue
                keys.append(k)
                rows.append(v_arr)
            if not rows:
                return
            matrix = np.vstack(rows)

        n_rows, n_dim = matrix.shape
        if n_rows == 0 or n_dim == 0:
            return

        sid_lookup = self._sid_lookup
        if sid_lookup is None:
            sid_lookup = {sid: i for i, sid in enumerate(self.series_ids)}
        sid_idx = np.fromiter((sid_lookup.get(k[0], -1) for k in keys), dtype=np.int64, count=len(keys))
        time_arr = np.fromiter((_time_key_to_int64(k[1]) for k in keys), dtype=np.int64, count=len(keys))
        w_arr = np.fromiter((int(k[2]) for k in keys), dtype=np.int64, count=len(keys))
        if self._const_flags is not None and self._const_flags.size >= len(keys):
            const_flags = np.asarray(self._const_flags[: len(keys)], dtype=np.uint8)
        else:
            const_flags = np.array(
                [bool(self.is_constant.get(k[0], False)) for k in keys],
                dtype=np.uint8,
            )

        valid_mask = sid_idx >= 0
        if not np.all(valid_mask):
            matrix = matrix[valid_mask]
            sid_idx = sid_idx[valid_mask]
            time_arr = time_arr[valid_mask]
            w_arr = w_arr[valid_mask]
            const_flags = const_flags[valid_mask]
            keys = [k for k, keep in zip(keys, valid_mask) if keep]
            if sid_idx.size == 0:
                return

        if self.full_vector_candidates:
            # (2026-07-06) Part 2.2: matrix (self._sketch_matrix) is already
            # the per-row unit-normalized sketch produced upstream in
            # _sketches_from_scratch/_incremental_sketches (Python
            # _normalize_sketch_matrix or the Cython
            # build_sketch_matrix/apply_orth_and_normalize kernels) -- both
            # write an all-zero row exactly when the source row was
            # const/zero-norm, and divide by the row's own norm otherwise,
            # so a valid row's post-normalization norm is always exactly
            # 1.0. Checking zero-vs-nonzero here is therefore mathematically
            # equivalent to the invalid/const check already computed
            # upstream; the only fix needed is computing it once, vectorized
            # over all rows, instead of once per row inside this Python
            # loop. See docs/implementation_log.md.
            row_norms = np.linalg.norm(matrix, axis=1)
            partition = {}
            for i, key in enumerate(keys):
                vec = np.asarray(matrix[i], dtype=np.float64)
                is_const = bool(const_flags[i]) or row_norms[i] == 0.0
                partition[key] = (vec, is_const, 0.0)
            self.partitions = [partition]
            return

        n_grids = min(self.n_grids, n_dim // self.grid_dimensions)
        self.partitions = [None for _ in range(self.n_grids)]
        sid_idx_arr = np.ascontiguousarray(sid_idx, dtype=np.int64)
        time_arr = np.ascontiguousarray(time_arr, dtype=np.int64)
        w_arr = np.ascontiguousarray(w_arr, dtype=np.int64)

        if _cy_build_partitions is not None:
            chunks, _norms, is_const = _cy_build_partitions(
                np.ascontiguousarray(matrix, dtype=np.float64),
                int(self.grid_dimensions),
                np.ascontiguousarray(const_flags, dtype=np.uint8),
            )
            n_grids = min(n_grids, chunks.shape[0])
            for grid in range(n_grids):
                self.partitions[grid] = (
                    sid_idx_arr,
                    time_arr,
                    w_arr,
                    np.ascontiguousarray(chunks[grid], dtype=np.float64),
                    np.ascontiguousarray(is_const[grid], dtype=np.uint8),
                )
        else:
            for grid in range(n_grids):
                start = grid * self.grid_dimensions
                end = start + self.grid_dimensions
                chunk = matrix[:, start:end]
                if chunk.size == 0:
                    chunk_arr = np.empty((chunk.shape[0], self.grid_dimensions), dtype=np.float64)
                    is_const = const_flags.copy()
                else:
                    chunk_arr = np.ascontiguousarray(chunk, dtype=np.float64)
                    grid_norms = np.linalg.norm(chunk, axis=1)
                    grid_norms = np.where(np.isfinite(grid_norms), grid_norms, 0.0)
                    is_const = np.logical_or(const_flags != 0, grid_norms == 0.0).astype(np.uint8)
                self.partitions[grid] = (
                    sid_idx_arr,
                    time_arr,
                    w_arr,
                    chunk_arr,
                    is_const,
                )

    def distribute_partitions(self):
        if len(self.partitions)>0:
            sid_list = None
            if isinstance(self._sid_lookup, dict) and self._sid_lookup:
                sid_list = [sid for sid, idx in sorted(self._sid_lookup.items(), key=lambda item: item[1])]
            elif self.series_ids:
                sid_list = list(self.series_ids)
            for grid in range(self.n_grids):
                part = self.partitions[grid]
                if part is None:
                    continue
                if isinstance(part, tuple) and sid_list is not None:
                    self.grid_nodes[grid].set_sid_list(sid_list)
                self.grid_nodes[grid].append_partition(self._curr_startTime(), part)
            return True
        else:
            return False
    
    def run(self, new_data_step, ids, verbose=True, testing=False, distribute=True):
        self.verbose = verbose
        self.testing = testing

        self._newStream(new_data_step, ids)

        sketches = self._get_sketches()
        self.partition_sketches(self.window_size)
        if distribute:
            self.distribute_partitions()

        if(testing and sketches):
            self._print_state()

        # Return copies so the caller can safely merge
        return dict(self.sketches), list(self.partitions)

class Candidates_BF:
    def __init__(self,window_size,window_step,n_lags,corr_threshold):       
        self.verbose = None
        # Parameters windows
        self.window_size = window_size
        self.window_step = window_step #divides basic_window
        self.n_lags = n_lags
        self.window_data = None
        self.window_index = None
        self.series_ids = []
        self.curr_window_size = window_size
        # Parameters matrix
        self.corr_threshold = corr_threshold

        self.brute_force_steps = 0
        self.ref_indices = None

    def dump_state(self):
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def load_state(self, state):
        self.__dict__.update(state)
    
    def _newStream(self,new_data_step,ids):
        new_data_step_index = new_data_step[0,:]
        new_data_step_values = new_data_step[1:,:]
        #Update sliding windows
        if self.window_data is None:            
            self.window_data = new_data_step_values
            self.window_index = new_data_step_index[-self.window_data.shape[1]:]
        else:
            if self.window_data.shape[1] >= (self.n_lags+self.window_size):
                self.window_index = self.window_index[self.window_step:]
                self.window_data = self.window_data[:,self.window_step:]
            self.window_index = np.append(self.window_index,new_data_step_index)
            self.window_data = np.append(self.window_data,new_data_step_values,axis=1)
        
        self.series_ids = ids
        self.window_data = self.window_data.astype(float)
        
        self._update_curr_window_size()
    
    def _update_curr_window_size(self):
        self.curr_window_size = min(self.window_size,self.window_data.shape[1])
        return None
    def _curr_window(self):
        return self.window_data[:,-self.curr_window_size:]
    
    def _curr_window_times(self):
        return self.window_index[-self.curr_window_size:]
    
    def _curr_startTime(self):
        return self.window_index[-self.curr_window_size]

    def _print_curr_window(self):
        print("\nCurrent window")
        print(self._curr_window_times())
        if(self.testing):
            print(self._curr_window())
    
    def _normalize_key(self, key):
        id1, id2, t1, t2, w = key

        if id1 == id2:
            # Autocorrelation case
            return (id1, id2, max(t1, t2), min(t1, t2), w)

        if t1 == t2:
            # Synchronous case
            return tuple(sorted([id1, id2])) + (t1, t2, w)

        if t1 < t2:
            # Cross-series, cross-time — canonicalize by lexicographic pair
            return (id2, id1, t2, t1, w)
        else:
            return (id1, id2, t1, t2, w)
    
    def _get_candidate_rows(self):
        if self.curr_window_size < self.window_size:
            if(self.verbose):
                self._print_curr_window()
                print("\nSkip distance calculations (window not full)")
            return None

        self.brute_force_steps += 1

        m = self.window_data.shape[0]
        ref_indices = self.ref_indices if self.ref_indices is not None else range(m)
        return _enumerate_candidate_rows(
            self.window_data,
            self.window_index,
            ref_indices,
            self.window_size,
            self.window_step,
        )

    def _get_candidates(self,candidates):
        rows = self._get_candidate_rows()
        if rows is None:
            return None

        for s_idx, k_idx, start_s, start_k, w in rows:
            id1 = self.series_ids[s_idx]
            id2 = self.series_ids[k_idx]
            pair = (id1, id2, int(start_s), int(start_k), int(w))
            candidates[self._normalize_key(pair)] = 1
    
    def run(self,new_data_step,ids, verbose=True, testing=False, ref_ids=None, numeric_rows=True):
        self.verbose = verbose
        self.testing = testing

        self._newStream(new_data_step,ids)

        if ref_ids is not None:
            try:
                id_list = list(ids)
            except TypeError:
                id_list = ids
            index_map = {id_list[i]: i for i in range(len(id_list))}
            self.ref_indices = sorted(index_map[id_] for id_ in ref_ids if id_ in index_map)
        else:
            self.ref_indices = None

        if numeric_rows:
            rows = self._get_candidate_rows()
            self.ref_indices = None
            if rows is None:
                return np.empty((0, 5), dtype=np.int64)
            return np.ascontiguousarray(rows, dtype=np.int64).reshape((-1, 5))

        candidates = {}
        self._get_candidates(candidates)

        self.ref_indices = None

        return candidates

    def prepare(self, new_data_step, ids, ref_ids=None):
        self._newStream(new_data_step, ids)
        if ref_ids is not None:
            try:
                id_list = list(ids)
            except TypeError:
                id_list = ids
            index_map = {id_list[i]: i for i in range(len(id_list))}
            self.ref_indices = sorted(index_map[id_] for id_ in ref_ids if id_ in index_map)
        else:
            self.ref_indices = None


class Candidates_BF_ExactSTOMP:
    """Exact online baseline using rolling lagged dot-product matrices."""

    def __init__(self, window_size, window_step, n_lags, corr_threshold, neg_corr=False):
        self.verbose = None
        self.testing = False
        self.window_size = int(window_size)
        self.window_step = int(window_step)
        self.n_lags = (int(n_lags) // self.window_step) * self.window_step
        self.corr_threshold = float(corr_threshold)
        self.neg_corr = bool(neg_corr)
        self.window_data = None
        self.window_index = None
        self.series_ids = []
        self.curr_window_size = self.window_size
        self.std_thresh = 1e-3
        self.spike_kurt_thresh = 5.0
        self._dot_by_lag = {}
        self._last_curr_start = None
        self.full_initializations = 0
        self.incremental_updates = 0

    def dump_state(self):
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def load_state(self, state):
        self.__dict__.update(state)

    def _newStream(self, new_data_step, ids):
        new_data_step_index = np.asarray(new_data_step[0, :])
        new_data_step_values = np.asarray(new_data_step[1:, :], dtype=np.float64)
        step_len = int(new_data_step_values.shape[1])
        extra = max(self.window_step, step_len)
        capacity = self.n_lags + self.window_size + extra

        if self.window_data is None:
            self.window_data = new_data_step_values
            self.window_index = new_data_step_index[-self.window_data.shape[1]:]
        else:
            projected = self.window_data.shape[1] + step_len
            if projected > capacity:
                drop = min(self.window_data.shape[1], projected - capacity)
                if drop > 0:
                    self.window_index = self.window_index[drop:]
                    self.window_data = self.window_data[:, drop:]
            self.window_index = np.append(self.window_index, new_data_step_index)
            self.window_data = np.append(self.window_data, new_data_step_values, axis=1)

        self.series_ids = list(ids)
        self._update_curr_window_size()

    def _update_curr_window_size(self):
        self.curr_window_size = min(self.window_size, self.window_data.shape[1])

    def _curr_startTime(self):
        return self.window_index[-self.curr_window_size]

    def _time_to_pos(self, start_time):
        return int(start_time - self.window_index[0])

    def _window_prefixes(self):
        data = np.asarray(self.window_data, dtype=np.float64)

        def _prefix(values):
            out = np.cumsum(values, axis=1, dtype=np.float64)
            return np.pad(out, ((0, 0), (1, 0)), mode="constant")

        return (
            _prefix(data),
            _prefix(data * data),
            _prefix(data * data * data),
            _prefix(data * data * data * data),
        )

    def _window_stats(self, prefixes, pos):
        csum, csum_sq, csum_cu, csum_qu = prefixes
        end = pos + self.window_size
        sx = csum[:, end] - csum[:, pos]
        sx2 = csum_sq[:, end] - csum_sq[:, pos]
        sx3 = csum_cu[:, end] - csum_cu[:, pos]
        sx4 = csum_qu[:, end] - csum_qu[:, pos]
        w = float(self.window_size)
        var_sum = np.maximum(sx2 - (sx * sx) / w, 0.0)
        nonconst = var_sum > ((self.std_thresh ** 2) * self.window_size)

        mean = sx / w
        mu4 = sx4 - 4.0 * mean * sx3 + 6.0 * mean * mean * sx2 - 4.0 * mean ** 3 * sx + w * mean ** 4
        mu4 = np.maximum(mu4, 0.0)
        var = var_sum / w
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = (mu4 / w) / (var * var) - 3.0
        spiked = (self.window_size >= 4) & (var_sum > 0.0) & (kurt > self.spike_kurt_thresh)
        return sx, sx2, var_sum, nonconst, spiked

    def _dot_for_lag(self, lag, curr_pos, hist_pos, curr_start):
        data = self.window_data
        dot = self._dot_by_lag.get(lag)
        advance = None
        if self._last_curr_start is not None:
            try:
                advance = int(curr_start - self._last_curr_start)
            except Exception:
                advance = None

        can_increment = (
            dot is not None
            and advance == self.window_step
            and curr_pos - advance >= 0
            and hist_pos - advance >= 0
            and curr_pos + self.window_size <= data.shape[1]
            and hist_pos + self.window_size <= data.shape[1]
            and curr_pos + self.window_size - advance >= 0
            and hist_pos + self.window_size - advance >= 0
        )

        if can_increment:
            old_curr = data[:, curr_pos - advance : curr_pos]
            old_hist = data[:, hist_pos - advance : hist_pos]
            new_curr = data[:, curr_pos + self.window_size - advance : curr_pos + self.window_size]
            new_hist = data[:, hist_pos + self.window_size - advance : hist_pos + self.window_size]
            dot = dot - old_curr @ old_hist.T + new_curr @ new_hist.T
            self.incremental_updates += 1
        else:
            curr_window = data[:, curr_pos : curr_pos + self.window_size]
            hist_window = data[:, hist_pos : hist_pos + self.window_size]
            dot = curr_window @ hist_window.T
            self.full_initializations += 1

        self._dot_by_lag[lag] = dot
        return dot

    def _pair_for_indices(self, s_idx, k_idx, curr_start, hist_start):
        return _normalize_bf_key(
            (
                self.series_ids[int(s_idx)],
                self.series_ids[int(k_idx)],
                int(curr_start),
                int(hist_start),
                self.window_size,
            )
        )

    def _row_for_indices(self, s_idx, k_idx, curr_start, hist_start):
        return (int(s_idx), int(k_idx), int(curr_start), int(hist_start), int(self.window_size))

    def _min_pair_from_mask(self, distances, pair_mask, curr_start, hist_start):
        if not np.any(pair_mask):
            return np.inf, None
        masked = np.where(pair_mask, distances, np.inf)
        flat_idx = int(np.argmin(masked))
        min_dist = float(masked.ravel()[flat_idx])
        if not math.isfinite(min_dist):
            return np.inf, None
        s_idx, k_idx = np.unravel_index(flat_idx, masked.shape)
        return min_dist, self._pair_for_indices(s_idx, k_idx, curr_start, hist_start)

    def run(self, new_data_step, ids, verbose=True, testing=False, track_min_dist=True, numeric_rows=True):
        self.verbose = verbose
        self.testing = testing
        self._newStream(new_data_step, ids)

        empty_timing = {
            "candidate_time": 0.0,
            "validation_time": 0.0,
            "min_dist": np.inf,
            "pair_min_dist": None,
        }
        if self.curr_window_size < self.window_size:
            if numeric_rows:
                return np.empty((0, 5), dtype=np.int64), np.empty((0,), dtype=np.float64), 0, empty_timing
            return {}, 0, empty_timing

        candidate_t0 = time.perf_counter()
        data = self.window_data
        curr_start = int(self._curr_startTime())
        curr_pos = self._time_to_pos(curr_start)
        max_lag = min(self.n_lags, curr_pos)
        if max_lag < 0:
            self._last_curr_start = curr_start
            if numeric_rows:
                return np.empty((0, 5), dtype=np.int64), np.empty((0,), dtype=np.float64), 0, empty_timing
            return {}, 0, empty_timing

        prefixes = self._window_prefixes()
        sx, sx2, var_x, curr_nonconst, curr_spiked = self._window_stats(prefixes, curr_pos)
        if not np.any(curr_nonconst):
            self._last_curr_start = curr_start
            timing = {
                "candidate_time": time.perf_counter() - candidate_t0,
                "validation_time": 0.0,
                "min_dist": np.inf,
                "pair_min_dist": None,
            }
            if numeric_rows:
                return np.empty((0, 5), dtype=np.int64), np.empty((0,), dtype=np.float64), 0, timing
            return {}, 0, timing

        validation_time = 0.0
        total_pairs = 0
        accepted = {}
        accepted_rows = []
        accepted_corrs = []
        min_dist = np.inf
        min_pair = None
        m = data.shape[0]
        upper_mask = np.triu(np.ones((m, m), dtype=bool), k=1)

        for lag in range(0, max_lag + 1, self.window_step):
            hist_start = curr_start - lag
            hist_pos = curr_pos - lag
            if hist_pos < 0 or hist_pos + self.window_size > data.shape[1]:
                continue

            dot = self._dot_for_lag(lag, curr_pos, hist_pos, curr_start)
            sy, sy2, var_y, hist_nonconst, hist_spiked = self._window_stats(prefixes, hist_pos)
            pair_mask = curr_nonconst[:, None] & hist_nonconst[None, :]
            if lag == 0:
                pair_mask &= upper_mask
            pair_count = int(np.count_nonzero(pair_mask))
            total_pairs += pair_count
            if pair_count == 0:
                continue

            lag_validation_t0 = time.perf_counter()
            denom = np.sqrt(var_x[:, None] * var_y[None, :])
            cov = dot - (sx[:, None] * sy[None, :]) / float(self.window_size)
            corr = np.divide(
                cov,
                denom,
                out=np.full_like(cov, np.nan, dtype=np.float64),
                where=denom > 0.0,
            )
            corr = np.clip(corr, -1.0, 1.0)

            if track_min_dist:
                dist_sq = sx2[:, None] + sy2[None, :] - 2.0 * dot
                distances = np.sqrt(np.maximum(dist_sq, 0.0))
                step_min_dist, step_min_pair = self._min_pair_from_mask(
                    distances,
                    pair_mask,
                    curr_start,
                    hist_start,
                )
                if step_min_pair is not None and step_min_dist < min_dist:
                    min_dist = step_min_dist
                    min_pair = step_min_pair

            accept_mask = pair_mask & (~curr_spiked[:, None]) & (~hist_spiked[None, :])
            if self.neg_corr:
                accept_mask &= np.abs(corr) >= self.corr_threshold
            else:
                accept_mask &= corr >= self.corr_threshold

            rows, cols = np.nonzero(accept_mask)
            if numeric_rows:
                if rows.size:
                    accepted_rows.append(
                        np.column_stack(
                            [
                                rows.astype(np.int64, copy=False),
                                cols.astype(np.int64, copy=False),
                                np.full(rows.size, curr_start, dtype=np.int64),
                                np.full(rows.size, hist_start, dtype=np.int64),
                                np.full(rows.size, self.window_size, dtype=np.int64),
                            ]
                        )
                    )
                    accepted_corrs.append(corr[rows, cols].astype(np.float64, copy=False))
            else:
                for s_idx, k_idx in zip(rows, cols):
                    pair = self._pair_for_indices(s_idx, k_idx, curr_start, hist_start)
                    accepted[pair] = float(corr[s_idx, k_idx])
            validation_time += time.perf_counter() - lag_validation_t0

        self._last_curr_start = curr_start
        timing = {
            "candidate_time": max(time.perf_counter() - candidate_t0 - validation_time, 0.0),
            "validation_time": max(validation_time, 0.0),
            "min_dist": min_dist,
            "pair_min_dist": min_pair,
        }
        if numeric_rows:
            if accepted_rows:
                row_arr = np.ascontiguousarray(np.vstack(accepted_rows), dtype=np.int64)
                corr_arr = np.ascontiguousarray(np.concatenate(accepted_corrs), dtype=np.float64)
            else:
                row_arr = np.empty((0, 5), dtype=np.int64)
                corr_arr = np.empty((0,), dtype=np.float64)
            return row_arr, corr_arr, total_pairs, timing
        return accepted, total_pairs, timing

class Candidates:
    def __init__(self,n_lagged_windows,grid_dimension,cell_size,grid_max,freq_threshold,corr_threshold,n_vectors,sketch_std,n_grids,neg_corr,
                 sign_prefilter_scale=1.3,sign_prefilter_extra=1, seed=None, full_vector=False, candidate_backend=None,
                 candidate_bucket_width=None, candidate_block_size_steps=None, candidate_block_index_dims=None,
                 candidate_similarity="l2", candidate_cosine_threshold=None,
                 candidate_key_mode="first", candidate_key_seed=None, candidate_lsh_radius=None,
                 candidate_ann_m=None, candidate_ann_z=None, candidate_ann_ef=None, return_distances=False,
                 candidate_bound_dims=None, candidate_bound_dim_selection="variance",
                 enable_block_ub_pruning=False, enable_row_ub_pruning=False,
                 block_similarity_assignment=False, max_open_blocks=4,
                 candidate_instinct_query_mode="hybrid", candidate_instinct_top_k=256,
                 candidate_instinct_min_candidates=64, candidate_instinct_entry_points=8):
        self.verbose = None
        self.neg_corr = neg_corr
        # Parameters grids
        self.n_lagged_windows = n_lagged_windows
        self.curr_time = None
        self.grid_dimensions = 1 if grid_dimension is None or grid_dimension <= 0 else int(grid_dimension)
        self.partition = []
        self.cell_size = cell_size
        self.grid_max = grid_max
        self.sketches = {}
        self.full_vector = bool(full_vector)
        # (2026-07-06) Part 1: Cauchy-Schwarz row-level bound + cone/angular
        # block-level bound for BlockedLazyIndex/BucketedMultiIndex
        # ("sorted_arrays_bs"). Both False by default -- opt-in, not a
        # universal win (row-level pruning reduces raw dot-check counts but
        # showed ~0 wall-clock benefit in testing; block-level pruning only
        # helps when the data has real angular clustering tighter than
        # gamma, where it showed a real 4-5x speedup, but safely no-ops
        # otherwise). See docs/implementation_log.md, "Part 1" and "Part 1
        # follow-up: cone-based block bound".
        bound_dims_val = _to_int_safe(candidate_bound_dims)
        self.candidate_bound_dims = max(0, bound_dims_val) if bound_dims_val is not None else 0
        self.candidate_bound_dim_selection = str(candidate_bound_dim_selection or "variance").lower()
        self.enable_block_ub_pruning = bool(enable_block_ub_pruning)
        self.enable_row_ub_pruning = bool(enable_row_ub_pruning)
        self.block_similarity_assignment = bool(block_similarity_assignment)
        self.max_open_blocks = max(1, _to_int_safe(max_open_blocks) or 4)
        # (2026-07-06) InstinctIndex -- experimental approximate graph
        # backend, candidate_backend="instinct" only. See
        # docs/implementation_log.md, "InstinctIndex: experimental
        # approximate graph backend". Reuses candidate_ann_m/candidate_ann_z/
        # candidate_ann_ef (dead knobs left over from the removed
        # dynamic_graph_ann/angular_lsh backends, already threaded end-to-end
        # through CLI/config/CSV) as max_degree/ef_insert/ef_search.
        self.candidate_instinct_query_mode = str(candidate_instinct_query_mode or "hybrid").lower()
        self.candidate_instinct_top_k = max(1, _to_int_safe(candidate_instinct_top_k) or 256)
        self.candidate_instinct_min_candidates = max(1, _to_int_safe(candidate_instinct_min_candidates) or 64)
        self.candidate_instinct_entry_points = max(1, _to_int_safe(candidate_instinct_entry_points) or 8)
        # Flat sorted index storage (legacy backend).
        self._entries = []  # (value, window_id_key, window_id[, vector])
        self._values = []   # parallel list of values for bisect
        self._reverse_index = defaultdict(list)  # window_id -> list of inserted values (includes neg when needed)
        self._reverse_vectors = defaultdict(list)  # window_id -> list of vectors (parallel to _reverse_index)
        self._reverse_entry_ids = defaultdict(list)  # window_id -> list of backend entry ids
        self._recent_window_ids = set()
        self._recent_entry_ids = []
        self._sid_list = []
        self._window_idx = {}
        self._window_idx_numeric = {}
        self._sid_idx_map = {}
        self._sid_sort_rank_map = {}
        self._sid_idx_rank = []
        self._win_sid = []
        self._win_sid_idx = []
        self._win_sid_rank = []
        self._win_time = []
        self._win_w = []
        self._candidate_numeric_rows = None
        self._profile_callback = None
        self._profile_metric_callback = None
        self._tree_index = None
        self._blocked_index = None
        self._bucket_entries = defaultdict(list)
        self._index_version = 0
        self._candidate_arrays_cache = None
        self._candidate_arrays_cache_version = -1
        self.candidate_bucket_width = _to_float_safe(candidate_bucket_width)
        block_steps = _to_int_safe(candidate_block_size_steps)
        if block_steps is None or block_steps <= 0:
            block_steps = 32
        self.candidate_block_size_steps = int(block_steps)
        block_dims = _to_int_safe(candidate_block_index_dims)
        if block_dims is None or block_dims <= 0:
            block_dims = 1
        self.candidate_block_index_dims = int(block_dims)
        # (2026-07-06) Part 1 -- see docs/implementation_log.md.
        bound_dims_val = _to_int_safe(candidate_bound_dims)
        self.candidate_bound_dims = max(0, bound_dims_val) if bound_dims_val is not None else 0
        self.candidate_bound_dim_selection = str(candidate_bound_dim_selection or "variance").lower()
        self.enable_block_ub_pruning = bool(enable_block_ub_pruning)
        self.enable_row_ub_pruning = bool(enable_row_ub_pruning)
        self.block_similarity_assignment = bool(block_similarity_assignment)
        self.max_open_blocks = max(1, _to_int_safe(max_open_blocks) or 4)
        self.candidate_similarity = _resolve_candidate_similarity(candidate_similarity, default="l2")
        self.candidate_cosine_threshold = _to_float_safe(candidate_cosine_threshold)
        if self.candidate_cosine_threshold is None:
            self.candidate_cosine_threshold = _candidate_gamma_from_tau(self.cell_size)
        if self.candidate_cosine_threshold is None:
            self.candidate_cosine_threshold = float(self.corr_threshold)
        self.candidate_cosine_threshold = max(-1.0, min(1.0, float(self.candidate_cosine_threshold)))
        self.sketch_std = sketch_std
        self.n_vectors = n_vectors
        # Parameters thresholds
        self.freq_threshold = freq_threshold
        self.corr_threshold = corr_threshold
        self.sign_prefilter_scale = float(sign_prefilter_scale)
        self.sign_prefilter_extra = int(sign_prefilter_extra)

        self.candidate_key_mode = _resolve_candidate_key_mode(candidate_key_mode, default="first")
        self.return_distances = _coerce_to_bool(return_distances, default=False)
        self.candidate_dist_sq = {}
        lsh_radius = _to_int_safe(candidate_lsh_radius)
        if lsh_radius is None:
            lsh_radius = 0
        self.candidate_lsh_radius = max(0, min(3, int(lsh_radius)))
        ann_m = _to_int_safe(candidate_ann_m)
        if ann_m is None or ann_m <= 0:
            ann_m = 16
        self.candidate_ann_m = max(1, int(ann_m))
        ann_z = _to_int_safe(candidate_ann_z)
        if ann_z is None or ann_z <= 0:
            ann_z = 256
        self.candidate_ann_z = max(1, int(ann_z))
        ann_ef = _to_int_safe(candidate_ann_ef)
        if ann_ef is None or ann_ef <= 0:
            ann_ef = max(64, self.candidate_ann_z)
        self.candidate_ann_ef = max(1, int(ann_ef))
        seed = _to_int_safe(seed)
        if candidate_key_seed is None:
            self.candidate_key_seed = seed
        else:
            self.candidate_key_seed = _to_int_safe(candidate_key_seed)
        self._candidate_key_signs = None
        self._candidate_key_signs_mode = None
        self._candidate_key_samples = None
        self._candidate_key_samples_mode = None
        self._refresh_vector_mode()
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        backend_value = candidate_backend
        if backend_value is None:
            backend_value = os.environ.get("CORRTRACK_CANDIDATE_BACKEND", "auto")
        self._requested_candidate_backend = _resolve_candidate_backend(backend_value, default="auto")
        self._candidate_backend = "flat"
        instinct_requested = self._requested_candidate_backend == "instinct"
        if instinct_requested and self._vector_match_enabled and not self.return_distances:
            index_cls = _cy_instinct_index_cls
            if index_cls is None:
                raise RuntimeError(
                    f"candidate_backend='{self._requested_candidate_backend}' requires the compiled "
                    "candidate_kernels Cython extension"
                )
            instinct_seed = seed if seed is not None else 0
            try:
                instinct_seed = int(instinct_seed) & ((1 << 63) - 1)
            except (TypeError, ValueError):
                instinct_seed = 0
            self._instinct_index = index_cls(
                n_vectors=int(self._vector_dim),
                initial_capacity=1024,
                max_degree=int(self.candidate_ann_m),
                ef_insert=int(self.candidate_ann_z),
                ef_search=int(self.candidate_ann_ef),
                entry_points=int(self.candidate_instinct_entry_points),
                query_mode=self.candidate_instinct_query_mode,
                top_k=int(self.candidate_instinct_top_k),
                min_candidates=int(self.candidate_instinct_min_candidates),
                seed=instinct_seed,
            )
            self._candidate_backend = "instinct"
        elif instinct_requested:
            raise RuntimeError(
                f"candidate_backend='{self._requested_candidate_backend}' requires the "
                "compiled candidate_kernels Cython extension"
            )
        bucketed_requested = self._requested_candidate_backend == "sorted_arrays_bs"
        if bucketed_requested and self._vector_match_enabled and not self.return_distances:
            index_cls = _cy_bucketed_multi_index_cls
            if index_cls is None:
                raise RuntimeError(
                    f"candidate_backend='{self._requested_candidate_backend}' requires the compiled "
                    "candidate_kernels Cython extension"
                )
            block_size = max(1, int(self.candidate_block_size_steps)) * max(1, int(self.grid_dimensions))
            self._blocked_index = index_cls(
                n_vectors=int(self._vector_dim),
                block_size=int(block_size),
                index_dims=max(1, min(int(self.candidate_block_index_dims), int(self._vector_dim))),
                initial_capacity=1024,
                bound_dims=min(int(self.candidate_bound_dims), int(self._vector_dim)),
                bound_dim_selection_variance=self.candidate_bound_dim_selection != "first",
                enable_block_ub_pruning=bool(self.enable_block_ub_pruning),
                enable_row_ub_pruning=bool(self.enable_row_ub_pruning),
                block_similarity_assignment=bool(self.block_similarity_assignment),
                max_open_blocks=max(1, int(self.max_open_blocks)),
            )
            self._candidate_backend = self._requested_candidate_backend
        elif bucketed_requested:
            raise RuntimeError(
                f"candidate_backend='{self._requested_candidate_backend}' requires the "
                "compiled candidate_kernels Cython extension"
            )
        elif self._requested_candidate_backend in {"bptree", "auto", "sorted_arrays_bs"} and _cy_balanced_index_cls is not None:
            tree_seed = seed if seed is not None else 0
            try:
                tree_seed = int(tree_seed) & ((1 << 63) - 1)
            except (TypeError, ValueError):
                tree_seed = 0
            vec_dim = int(self._vector_dim) if self._vector_match_enabled else 0
            self._tree_index = _cy_balanced_index_cls(
                n_vectors=vec_dim,
                initial_capacity=1024,
                seed=tree_seed,
            )
            self._candidate_backend = "bptree"
        elif self._requested_candidate_backend in {"bptree", "auto"}:
            raise RuntimeError(
                "Cython candidate search is required, but candidate_kernels.BalancedIndex "
                "is not available"
            )
        elif self._requested_candidate_backend == "flat":
            if _cy_find_candidate_pairs is None:
                raise RuntimeError(
                    "candidate_backend='flat' requires compiled candidate_kernels "
                    "range-search functions; Python candidate search fallback is disabled"
                )
            self._candidate_backend = "flat"

    def dump_state(self):
        return dict(self.__dict__)

    def _refresh_vector_mode(self):
        grid_dim = _to_int_safe(getattr(self, "grid_dimensions", 1))
        if grid_dim is None or grid_dim <= 0:
            grid_dim = 1
        self.grid_dimensions = int(grid_dim)
        self.full_vector = bool(getattr(self, "full_vector", False))
        self._vector_match_enabled = bool(self.full_vector or self.grid_dimensions > 1)
        if self.full_vector:
            vec_dim = _to_int_safe(getattr(self, "n_vectors", 0))
        else:
            vec_dim = self.grid_dimensions
        if vec_dim is None or vec_dim <= 0:
            vec_dim = 1
        self._vector_dim = int(vec_dim)

    def _candidate_key_value(self, vector):
        keys = self._candidate_key_values(np.asarray(vector, dtype=np.float64).reshape(1, -1), 1)
        if keys.size == 0:
            return 0.0
        return float(keys[0, 0])

    def _ensure_candidate_key_signs(self, n_keys):
        n_keys = max(1, int(n_keys or 1))
        dim = max(1, int(getattr(self, "_vector_dim", 1) or 1))
        signs = getattr(self, "_candidate_key_signs", None)
        if (
            signs is not None
            and getattr(signs, "shape", None) == (n_keys, dim)
            and getattr(self, "_candidate_key_signs_mode", None) == self.candidate_key_mode
        ):
            return signs
        seed = _to_int_safe(getattr(self, "candidate_key_seed", None))
        if seed is None:
            seed = 0
        rng = np.random.default_rng(int(seed))
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_keys, dim))
        signs = np.ascontiguousarray(signs / math.sqrt(float(dim)), dtype=np.float64)
        self._candidate_key_signs = signs
        self._candidate_key_signs_mode = self.candidate_key_mode
        return signs

    def _ensure_candidate_key_samples(self, vectors, n_keys):
        n_keys = max(1, int(n_keys or 1))
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            arr = arr.reshape((arr.shape[0], -1))
        dim = int(arr.shape[1]) if arr.ndim == 2 and arr.shape[1] > 0 else max(1, int(getattr(self, "_vector_dim", 1) or 1))
        samples = getattr(self, "_candidate_key_samples", None)
        if (
            samples is not None
            and getattr(samples, "shape", None) == (n_keys, dim)
            and getattr(self, "_candidate_key_samples_mode", None) == self.candidate_key_mode
        ):
            return samples

        sample_rows = np.zeros((n_keys, dim), dtype=np.float64)
        fill = 0
        for row in arr:
            if fill >= n_keys:
                break
            row = np.asarray(row, dtype=np.float64).ravel()
            if row.size != dim:
                fixed = np.zeros(dim, dtype=np.float64)
                copy_n = min(row.size, dim)
                if copy_n > 0:
                    fixed[:copy_n] = row[:copy_n]
                row = fixed
            norm = float(np.linalg.norm(row))
            if norm <= 0.0 or not np.isfinite(norm):
                continue
            sample_rows[fill, :] = row / norm
            fill += 1
        if fill < n_keys:
            seed = _to_int_safe(getattr(self, "candidate_key_seed", None))
            if seed is None:
                seed = 0
            rng = np.random.default_rng(int(seed) + 104729)
            fallback = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_keys - fill, dim))
            sample_rows[fill:, :] = fallback / math.sqrt(float(dim))
        samples = np.ascontiguousarray(sample_rows, dtype=np.float64)
        self._candidate_key_samples = samples
        self._candidate_key_samples_mode = self.candidate_key_mode
        return samples

    def _candidate_key_values(self, vectors, n_keys=None):
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            arr = arr.reshape((arr.shape[0], -1))
        if arr.shape[0] == 0:
            return np.empty((0, max(1, int(n_keys or 1))), dtype=np.float64)
        dim = int(arr.shape[1])
        if dim <= 0:
            return np.zeros((arr.shape[0], max(1, int(n_keys or 1))), dtype=np.float64)
        if n_keys is None:
            n_keys = 1
        n_keys = max(1, int(n_keys))
        mode = _resolve_candidate_key_mode(getattr(self, "candidate_key_mode", "first"), default="first")
        if mode == "random_sign":
            signs = self._ensure_candidate_key_signs(n_keys)
            if signs.shape[1] != dim:
                self._vector_dim = dim
                signs = self._ensure_candidate_key_signs(n_keys)
            return np.ascontiguousarray(arr @ signs[:n_keys, :].T, dtype=np.float64)
        if mode == "sampled_sketch":
            samples = self._ensure_candidate_key_samples(arr, n_keys)
            if samples.shape[1] != dim:
                self._vector_dim = dim
                self._candidate_key_samples = None
                samples = self._ensure_candidate_key_samples(arr, n_keys)
            return np.ascontiguousarray(arr @ samples[:n_keys, :].T, dtype=np.float64)
        keys = np.zeros((arr.shape[0], n_keys), dtype=np.float64)
        copy_n = min(n_keys, dim)
        if copy_n > 0:
            keys[:, :copy_n] = arr[:, :copy_n]
        return np.ascontiguousarray(keys, dtype=np.float64)

    def _candidate_search_tau(self):
        if getattr(self, "candidate_similarity", "l2") == "cosine":
            tau = _candidate_tau_from_gamma(getattr(self, "candidate_cosine_threshold", None))
            if tau is not None:
                return float(tau)
        return None if self.cell_size is None else float(self.cell_size)

    def _profile_add(self, key, elapsed):
        callback = getattr(self, "_profile_callback", None)
        if callback is not None:
            callback(key, elapsed)

    def _profile_add_metric(self, key, value):
        callback = getattr(self, "_profile_metric_callback", None)
        if callback is not None:
            callback(key, value)

    def candidate_search_stats(self):
        stats = None
        if self._candidate_backend == "bptree":
            stats = getattr(getattr(self, "_tree_index", None), "last_stats", None)
        elif self._candidate_backend in _BLOCKED_INDEX_BACKENDS:
            stats = getattr(getattr(self, "_blocked_index", None), "last_stats", None)
        elif self._candidate_backend in _INSTINCT_INDEX_BACKENDS:
            stats = getattr(getattr(self, "_instinct_index", None), "last_stats", None)
        if not stats:
            return {
                "index_candidates": 0,
                "valid_index_candidates": 0,
                "unique_index_candidates": 0,
                "duplicate_index_candidates": 0,
                "unique_pre_dot_pairs": 0,
                "duplicate_pre_dot_pairs": 0,
                "after_coord": 0,
                "partial_checks": 0,
                "after_partial": 0,
                "after_similarity": 0,
                "dot_checks": 0,
                "distance_checks": 0,
                "blocks_visited": 0,
                "blocks_pruned_by_ub": 0,
                "rows_in_surviving_blocks": 0,
                "dot_checks_saved_by_row_ub": 0,
                "instinct_visited_nodes": 0,
                "instinct_visited_live_nodes": 0,
                "instinct_dead_nodes_skipped": 0,
                "instinct_edges_scanned": 0,
                "instinct_candidates_returned": 0,
                "instinct_threshold_candidates": 0,
                "instinct_topk_candidates": 0,
                "instinct_best_score_seen": 0.0,
                "instinct_mean_score_returned": 0.0,
                "instinct_query_time": 0.0,
                "instinct_insert_time": 0.0,
                "instinct_queries_with_too_few_live_nodes": 0,
                "instinct_num_nodes_total": 0,
                "instinct_num_nodes_alive": 0,
                "instinct_dead_node_ratio": 0.0,
            }
        after_similarity = stats.get("num_after_similarity", stats.get("num_after_dot", 0))
        blocks_visited = int(stats.get("num_blocks_visited", 0) or 0)
        blocks_pruned_by_ub = int(stats.get("num_blocks_pruned_by_ub", 0) or 0)
        return {
            "index_candidates": int(stats.get("num_index_candidates", 0) or 0),
            "valid_index_candidates": int(stats.get("num_valid_index_candidates", 0) or 0),
            "unique_index_candidates": int(stats.get("num_unique_index_candidates", 0) or 0),
            "duplicate_index_candidates": int(stats.get("num_duplicate_index_candidates", 0) or 0),
            "unique_pre_dot_pairs": int(stats.get("num_unique_pre_dot_pairs", 0) or 0),
            "duplicate_pre_dot_pairs": int(stats.get("num_duplicate_pre_dot_pairs", 0) or 0),
            "after_coord": int(stats.get("num_after_coord_filter", 0) or 0),
            "partial_checks": int(stats.get("num_partial_bound_checks", 0) or 0),
            "after_partial": int(stats.get("num_after_partial_bound", 0) or 0),
            "after_similarity": int(after_similarity or 0),
            "dot_checks": int(stats.get("num_dot_checks", 0) or 0),
            "distance_checks": int(stats.get("num_distance_checks", 0) or 0),
            # (2026-07-06) Part 1 metrics -- only nonzero for
            # BlockedLazyIndex/BucketedMultiIndex ("sorted_arrays_bs") with
            # enable_block_ub_pruning/enable_row_ub_pruning on. See
            # docs/implementation_log.md.
            "blocks_visited": blocks_visited,
            "blocks_pruned_by_ub": blocks_pruned_by_ub,
            "rows_in_surviving_blocks": int(stats.get("num_rows_in_surviving_blocks", 0) or 0),
            "dot_checks_saved_by_row_ub": int(stats.get("num_dot_checks_saved_by_row_ub", 0) or 0),
            # (2026-07-06) InstinctIndex metrics -- only nonzero for
            # candidate_backend="instinct". See docs/implementation_log.md,
            # "InstinctIndex: experimental approximate graph backend".
            "instinct_visited_nodes": int(stats.get("instinct_visited_nodes", 0) or 0),
            "instinct_visited_live_nodes": int(stats.get("instinct_visited_live_nodes", 0) or 0),
            "instinct_dead_nodes_skipped": int(stats.get("instinct_dead_nodes_skipped", 0) or 0),
            "instinct_edges_scanned": int(stats.get("instinct_edges_scanned", 0) or 0),
            "instinct_candidates_returned": int(stats.get("instinct_candidates_returned", 0) or 0),
            "instinct_threshold_candidates": int(stats.get("instinct_threshold_candidates", 0) or 0),
            "instinct_topk_candidates": int(stats.get("instinct_topk_candidates", 0) or 0),
            "instinct_best_score_seen": float(stats.get("instinct_best_score_seen", 0.0) or 0.0),
            "instinct_mean_score_returned": float(stats.get("instinct_mean_score_returned", 0.0) or 0.0),
            "instinct_query_time": float(stats.get("instinct_query_time", 0.0) or 0.0),
            "instinct_insert_time": float(stats.get("instinct_insert_time", 0.0) or 0.0),
            "instinct_queries_with_too_few_live_nodes": int(stats.get("instinct_queries_with_too_few_live_nodes", 0) or 0),
            "instinct_num_nodes_total": int(stats.get("instinct_num_nodes_total", 0) or 0),
            "instinct_num_nodes_alive": int(stats.get("instinct_num_nodes_alive", 0) or 0),
            "instinct_dead_node_ratio": float(stats.get("instinct_dead_node_ratio", 0.0) or 0.0),
        }

    def _profile_candidate_search_stats(self):
        stats = self.candidate_search_stats()
        for key, value in stats.items():
            self._profile_add_metric(f"cand.search.{key}", int(value or 0))

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._reverse_index = defaultdict(list, getattr(obj, "_reverse_index", {}))
        obj._reverse_vectors = defaultdict(list, getattr(obj, "_reverse_vectors", {}))
        obj._reverse_entry_ids = defaultdict(list, getattr(obj, "_reverse_entry_ids", {}))
        obj._bucket_entries = defaultdict(list, getattr(obj, "_bucket_entries", {}))
        obj._recent_entry_ids = list(getattr(obj, "_recent_entry_ids", []))
        obj._window_idx = dict(getattr(obj, "_window_idx", {}))
        obj._window_idx_numeric = dict(getattr(obj, "_window_idx_numeric", {}))
        obj._sid_idx_map = dict(getattr(obj, "_sid_idx_map", {}))
        obj._sid_sort_rank_map = dict(getattr(obj, "_sid_sort_rank_map", {}))
        obj._sid_idx_rank = list(getattr(obj, "_sid_idx_rank", []))
        obj._win_sid = list(getattr(obj, "_win_sid", []))
        obj._win_sid_idx = list(getattr(obj, "_win_sid_idx", []))
        obj._win_sid_rank = list(getattr(obj, "_win_sid_rank", []))
        obj._win_time = list(getattr(obj, "_win_time", []))
        obj._win_w = list(getattr(obj, "_win_w", []))
        obj._candidate_numeric_rows = None
        obj._profile_metric_callback = None
        obj._blocked_index = getattr(obj, "_blocked_index", None)
        obj._index_version = int(getattr(obj, "_index_version", 0) or 0)
        obj._candidate_arrays_cache = None
        obj._candidate_arrays_cache_version = -1
        requested_backend = _resolve_candidate_backend(
            getattr(obj, "_requested_candidate_backend", getattr(obj, "_candidate_backend", "auto")),
            default="auto",
        )
        obj._requested_candidate_backend = requested_backend
        if getattr(obj, "_candidate_backend", None) not in {"flat", "bptree"} | _BLOCKED_INDEX_BACKENDS:
            obj._candidate_backend = "flat"
        if obj._candidate_backend == "bptree" and _cy_balanced_index_cls is None:
            raise RuntimeError(
                "candidate_backend='bptree' requires the compiled "
                "candidate_kernels.BalancedIndex Cython extension"
            )
        if obj._candidate_backend in _BLOCKED_INDEX_BACKENDS and _cy_blocked_lazy_index_cls is None:
            raise RuntimeError(
                f"candidate_backend='{obj._candidate_backend}' requires the compiled "
                "candidate_kernels.BlockedLazyIndex Cython extension"
            )
        obj.candidate_bucket_width = _to_float_safe(getattr(obj, "candidate_bucket_width", None))
        obj.candidate_block_size_steps = int(_to_int_safe(getattr(obj, "candidate_block_size_steps", 32)) or 32)
        obj.candidate_block_index_dims = int(_to_int_safe(getattr(obj, "candidate_block_index_dims", 1)) or 1)
        obj.candidate_similarity = _resolve_candidate_similarity(getattr(obj, "candidate_similarity", "l2"), default="l2")
        gamma = _to_float_safe(getattr(obj, "candidate_cosine_threshold", None))
        if gamma is None:
            gamma = _candidate_gamma_from_tau(getattr(obj, "cell_size", None))
        if gamma is None:
            gamma = _to_float_safe(getattr(obj, "corr_threshold", 0.0)) or 0.0
        obj.candidate_cosine_threshold = max(-1.0, min(1.0, float(gamma)))
        obj._refresh_vector_mode()
        obj.candidate_key_mode = _resolve_candidate_key_mode(getattr(obj, "candidate_key_mode", "first"), default="first")
        obj.candidate_key_seed = _to_int_safe(getattr(obj, "candidate_key_seed", None))
        obj.candidate_lsh_radius = max(0, min(3, int(_to_int_safe(getattr(obj, "candidate_lsh_radius", 0)) or 0)))
        obj.candidate_ann_m = max(1, int(_to_int_safe(getattr(obj, "candidate_ann_m", 16)) or 16))
        obj.candidate_ann_z = max(1, int(_to_int_safe(getattr(obj, "candidate_ann_z", 256)) or 256))
        obj.candidate_ann_ef = max(1, int(_to_int_safe(getattr(obj, "candidate_ann_ef", max(64, obj.candidate_ann_z))) or max(64, obj.candidate_ann_z)))
        obj._candidate_key_signs = None
        obj._candidate_key_signs_mode = None
        obj._candidate_key_samples = None
        obj._candidate_key_samples_mode = None
        obj.return_distances = _coerce_to_bool(getattr(obj, "return_distances", False), default=False)
        obj.candidate_dist_sq = dict(getattr(obj, "candidate_dist_sq", {}))
        return obj

    def load_state(self, state):
        self.__dict__.update(state)
        self._reverse_index = defaultdict(list, getattr(self, "_reverse_index", {}))
        self._reverse_vectors = defaultdict(list, getattr(self, "_reverse_vectors", {}))
        self._reverse_entry_ids = defaultdict(list, getattr(self, "_reverse_entry_ids", {}))
        self._bucket_entries = defaultdict(list, getattr(self, "_bucket_entries", {}))
        self._recent_entry_ids = list(getattr(self, "_recent_entry_ids", []))
        self._window_idx = dict(getattr(self, "_window_idx", {}))
        self._window_idx_numeric = dict(getattr(self, "_window_idx_numeric", {}))
        self._sid_idx_map = dict(getattr(self, "_sid_idx_map", {}))
        self._sid_sort_rank_map = dict(getattr(self, "_sid_sort_rank_map", {}))
        self._sid_idx_rank = list(getattr(self, "_sid_idx_rank", []))
        self._win_sid = list(getattr(self, "_win_sid", []))
        self._win_sid_idx = list(getattr(self, "_win_sid_idx", []))
        self._win_sid_rank = list(getattr(self, "_win_sid_rank", []))
        self._win_time = list(getattr(self, "_win_time", []))
        self._win_w = list(getattr(self, "_win_w", []))
        self._candidate_numeric_rows = None
        self._profile_metric_callback = None
        self._blocked_index = getattr(self, "_blocked_index", None)
        self._index_version = int(getattr(self, "_index_version", 0) or 0)
        self._candidate_arrays_cache = None
        self._candidate_arrays_cache_version = -1
        requested_backend = _resolve_candidate_backend(
            getattr(self, "_requested_candidate_backend", getattr(self, "_candidate_backend", "auto")),
            default="auto",
        )
        self._requested_candidate_backend = requested_backend
        if getattr(self, "_candidate_backend", None) not in {"flat", "bptree"} | _BLOCKED_INDEX_BACKENDS:
            self._candidate_backend = "flat"
        if self._candidate_backend == "bptree" and _cy_balanced_index_cls is None:
            raise RuntimeError(
                "candidate_backend='bptree' requires the compiled "
                "candidate_kernels.BalancedIndex Cython extension"
            )
        if self._candidate_backend in _BLOCKED_INDEX_BACKENDS and _cy_blocked_lazy_index_cls is None:
            raise RuntimeError(
                f"candidate_backend='{self._candidate_backend}' requires the compiled "
                "candidate_kernels.BlockedLazyIndex Cython extension"
            )
        self.candidate_bucket_width = _to_float_safe(getattr(self, "candidate_bucket_width", None))
        self.candidate_block_size_steps = int(_to_int_safe(getattr(self, "candidate_block_size_steps", 32)) or 32)
        self.candidate_block_index_dims = int(_to_int_safe(getattr(self, "candidate_block_index_dims", 1)) or 1)
        self.candidate_similarity = _resolve_candidate_similarity(getattr(self, "candidate_similarity", "l2"), default="l2")
        gamma = _to_float_safe(getattr(self, "candidate_cosine_threshold", None))
        if gamma is None:
            gamma = _candidate_gamma_from_tau(getattr(self, "cell_size", None))
        if gamma is None:
            gamma = _to_float_safe(getattr(self, "corr_threshold", 0.0)) or 0.0
        self.candidate_cosine_threshold = max(-1.0, min(1.0, float(gamma)))
        self._refresh_vector_mode()
        self.candidate_key_mode = _resolve_candidate_key_mode(getattr(self, "candidate_key_mode", "first"), default="first")
        self.candidate_key_seed = _to_int_safe(getattr(self, "candidate_key_seed", None))
        self.candidate_lsh_radius = max(0, min(3, int(_to_int_safe(getattr(self, "candidate_lsh_radius", 0)) or 0)))
        self.candidate_ann_m = max(1, int(_to_int_safe(getattr(self, "candidate_ann_m", 16)) or 16))
        self.candidate_ann_z = max(1, int(_to_int_safe(getattr(self, "candidate_ann_z", 256)) or 256))
        self.candidate_ann_ef = max(1, int(_to_int_safe(getattr(self, "candidate_ann_ef", max(64, self.candidate_ann_z))) or max(64, self.candidate_ann_z)))
        self._candidate_key_signs = None
        self._candidate_key_signs_mode = None
        self._candidate_key_samples = None
        self._candidate_key_samples_mode = None
        self.return_distances = _coerce_to_bool(getattr(self, "return_distances", False), default=False)
        self.candidate_dist_sq = dict(getattr(self, "candidate_dist_sq", {}))

    def append_partition(self,curr_time,new_partition):
        keep_numeric_partition = (
            self._candidate_backend == "bptree" or self._candidate_backend in _BLOCKED_INDEX_BACKENDS
            and isinstance(new_partition, tuple)
            and len(new_partition) in {5, 6}
        )
        if not keep_numeric_partition:
            new_partition = self._partition_to_dict(new_partition)
        if new_partition is None:
            return
        if self.curr_time != curr_time:
            self.curr_time = curr_time
            self.partition.append(new_partition)
        else:
            current = self.partition[-1]
            if (
                keep_numeric_partition
                and isinstance(current, tuple)
                and len(current) in {5, 6}
                and len(new_partition) in {5, 6}
            ):
                self.partition[-1] = self._merge_numeric_partitions(current, new_partition)
            else:
                if not isinstance(current, dict):
                    current = self._partition_to_dict(current)
                    self.partition[-1] = current if current is not None else {}
                if not isinstance(new_partition, dict):
                    new_partition = self._partition_to_dict(new_partition)
                    if new_partition is None:
                        return
                current.update(new_partition)
        self._recent_window_ids = set()
        self._recent_entry_ids = []
        self._candidate_numeric_rows = None
        if self._candidate_backend == "bptree" and self._tree_index is not None:
            clear_recent = getattr(self._tree_index, "clear_recent", None)
            if clear_recent is not None:
                clear_recent()
        if self._candidate_backend in _BLOCKED_INDEX_BACKENDS and self._blocked_index is not None:
            clear_recent = getattr(self._blocked_index, "clear_recent", None)
            if clear_recent is not None:
                clear_recent()
        if self._candidate_backend in _INSTINCT_INDEX_BACKENDS and self._instinct_index is not None:
            clear_recent = getattr(self._instinct_index, "clear_recent", None)
            if clear_recent is not None:
                clear_recent()
        self._candidate_arrays_cache = None
        self._candidate_arrays_cache_version = -1

    @staticmethod
    def _partition_len(partition):
        if isinstance(partition, dict):
            return len(partition)
        if isinstance(partition, tuple) and len(partition) >= 5:
            sid_idx = partition[0]
            return int(len(sid_idx)) if sid_idx is not None else 0
        return 0

    @staticmethod
    def _merge_numeric_partitions(left, right):
        left = left[:5]
        right = right[:5]
        merged = []
        for lval, rval in zip(left, right):
            if lval is None:
                merged.append(rval)
            elif rval is None:
                merged.append(lval)
            else:
                merged.append(np.concatenate((np.asarray(lval), np.asarray(rval)), axis=0))
        return tuple(merged)

    def set_sid_list(self, sid_list):
        if not sid_list:
            return
        if self._sid_list == list(sid_list):
            return
        self._sid_list = list(sid_list)
        self._sid_idx_map = {sid: idx for idx, sid in enumerate(self._sid_list)}
        try:
            sorted_sids = sorted(self._sid_list)
        except TypeError:
            sorted_sids = sorted(self._sid_list, key=lambda value: str(value))
        self._sid_sort_rank_map = {sid: rank for rank, sid in enumerate(sorted_sids)}
        self._sid_idx_rank = [
            int(self._sid_sort_rank_map.get(sid, idx))
            for idx, sid in enumerate(self._sid_list)
        ]

    def _partition_to_dict(self, partition):
        if partition is None:
            return None
        if isinstance(partition, dict):
            return partition if partition else None
        if isinstance(partition, tuple) and len(partition) == 5:
            sid_idx, time_arr, w_arr, chunk_arr, is_const = partition
            if sid_idx is None or len(sid_idx) == 0:
                return None
            chunks = np.asarray(chunk_arr, dtype=np.float64)
            out = {}
            sid_list = self._sid_list
            for i in range(len(sid_idx)):
                idx = int(sid_idx[i])
                sid = sid_list[idx] if sid_list and idx < len(sid_list) else str(idx)
                key = (sid, int(time_arr[i]), int(w_arr[i]))
                if chunks.ndim == 2:
                    sketch = np.ascontiguousarray(chunks[i], dtype=np.float64)
                else:
                    sketch = float(chunks[i])
                out[key] = (sketch, bool(is_const[i]), 0.0)
            return out
        return None

    def _window_id_key(self, window_id):
        sid, start_time, window_size = window_id
        return (str(sid), int(start_time), int(window_size))

    def _get_or_create_window_idx(self, window_id):
        idx = self._window_idx.get(window_id)
        if idx is not None:
            return idx
        sid, start_time, window_size = window_id
        idx = len(self._win_sid)
        sid_idx = self._sid_idx_map.get(sid)
        if sid_idx is None:
            sid_idx = len(self._sid_idx_map)
            self._sid_idx_map[sid] = sid_idx
        sid_rank = self._sid_sort_rank_map.get(sid)
        if sid_rank is None:
            sid_rank = len(self._sid_sort_rank_map)
            self._sid_sort_rank_map[sid] = sid_rank
        self._window_idx[window_id] = idx
        self._win_sid.append(sid)
        self._win_sid_idx.append(int(sid_idx))
        self._win_sid_rank.append(int(sid_rank))
        self._win_time.append(int(start_time))
        self._win_w.append(int(window_size))
        return idx

    def _get_or_create_window_idx_numeric(self, sid_idx, start_time, window_size):
        sid_idx = int(sid_idx)
        start_time = int(start_time)
        window_size = int(window_size)
        key = (sid_idx, start_time, window_size)
        idx = self._window_idx_numeric.get(key)
        if idx is not None:
            return idx
        idx = len(self._win_sid)
        sid = self._sid_list[sid_idx] if 0 <= sid_idx < len(self._sid_list) else str(sid_idx)
        if sid_idx >= len(self._sid_idx_rank):
            rank = int(self._sid_sort_rank_map.get(sid, sid_idx))
        else:
            rank = int(self._sid_idx_rank[sid_idx])
        self._window_idx_numeric[key] = idx
        self._win_sid.append(sid)
        self._win_sid_idx.append(int(sid_idx))
        self._win_sid_rank.append(rank)
        self._win_time.append(start_time)
        self._win_w.append(window_size)
        return idx

    def _partition_min_time(self, partition):
        try:
            if isinstance(partition, tuple) and len(partition) >= 2:
                arr = np.asarray(partition[1], dtype=np.int64)
                if arr.size:
                    return int(np.min(arr))
            elif isinstance(partition, dict) and partition:
                times = [int(k[1]) for k in partition.keys()]
                if times:
                    return min(times)
        except Exception:
            return None
        return None

    def _expire_blocked_lazy_index(self):
        if self._candidate_backend not in _BLOCKED_INDEX_BACKENDS or self._blocked_index is None:
            return
        min_time = None
        if self.partition:
            min_time = self._partition_min_time(self.partition[0])
        if min_time is None:
            return
        drop_before = getattr(self._blocked_index, "drop_before_time", None)
        if drop_before is None:
            return
        t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        drop_before(int(min_time))
        if t0 is not None:
            metric = "cand.blocked_expire"
            self._profile_add(metric, time.perf_counter() - t0)

    def _expire_instinct_index(self):
        # (2026-07-06) InstinctIndex -- experimental approximate graph
        # backend, same time-based lazy-eviction contract as
        # _expire_blocked_lazy_index (drop_before_time). See
        # docs/implementation_log.md, "InstinctIndex: experimental
        # approximate graph backend".
        if self._candidate_backend not in _INSTINCT_INDEX_BACKENDS or self._instinct_index is None:
            return
        min_time = None
        if self.partition:
            min_time = self._partition_min_time(self.partition[0])
        if min_time is None:
            return
        drop_before = getattr(self._instinct_index, "drop_before_time", None)
        if drop_before is None:
            return
        t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        drop_before(int(min_time))
        if t0 is not None:
            self._profile_add("cand.instinct_expire", time.perf_counter() - t0)

    def _insert_entry(self, value, window_id, vector=None):
        key = self._window_id_key(window_id)
        start = bisect.bisect_left(self._values, value)
        end = bisect.bisect_right(self._values, value)
        if start == end:
            idx = start
        else:
            subkeys = [self._entries[i][1] for i in range(start, end)]
            offset = bisect.bisect_left(subkeys, key)
            idx = start + offset
        self._values.insert(idx, value)
        self._entries.insert(idx, (value, key, window_id, vector))
        self._index_version = int(getattr(self, "_index_version", 0) or 0) + 1
        self._candidate_arrays_cache = None
        self._candidate_arrays_cache_version = -1

    def _remove_entry(self, value, window_id):
        key = self._window_id_key(window_id)
        idx = bisect.bisect_left(self._values, value)
        while idx < len(self._values) and self._values[idx] == value:
            entry = self._entries[idx]
            entry_key = entry[1]
            entry_id = entry[2]
            if entry_key == key and entry_id == window_id:
                self._values.pop(idx)
                self._entries.pop(idx)
                self._index_version = int(getattr(self, "_index_version", 0) or 0) + 1
                self._candidate_arrays_cache = None
                self._candidate_arrays_cache_version = -1
                return True
            idx += 1
        return False

    def _use_single_index_neg_corr(self):
        return (
            bool(self.neg_corr)
            and self._candidate_backend == "bptree"
            and self._tree_index is not None
            and self._vector_match_enabled
            and not bool(getattr(self, "return_distances", False))
        )

    def _range_search_ids(self, lower, upper):
        idx = bisect.bisect_left(self._values, lower)
        n = len(self._values)
        while idx < n and self._values[idx] <= upper:
            yield self._entries[idx][2]
            idx += 1

    def _range_search_entries(self, lower, upper):
        idx = bisect.bisect_left(self._values, lower)
        n = len(self._values)
        while idx < n and self._values[idx] <= upper:
            entry = self._entries[idx]
            yield entry[2], entry[3]
            idx += 1

    def _range_search_scalar_entries(self, lower, upper):
        idx = bisect.bisect_left(self._values, lower)
        n = len(self._values)
        while idx < n and self._values[idx] <= upper:
            entry = self._entries[idx]
            yield entry[2], entry[0]
            idx += 1

    def _record_candidate_hit(self, pair_id, freq_pairs, candidates, seen_pairs, dist_pairs=None, dist_sq=None):
        if pair_id not in seen_pairs:
            seen_pairs.add(pair_id)
            freq_pairs[pair_id] = freq_pairs.get(pair_id, 0.0) + 1.0
            if freq_pairs[pair_id] >= self.freq_threshold:
                candidates[pair_id] = 1
        if dist_pairs is not None and dist_sq is not None:
            dist_sq = float(dist_sq)
            prev = dist_pairs.get(pair_id)
            if prev is None or dist_sq < prev:
                dist_pairs[pair_id] = dist_sq

    def _window_distance_sq_by_indices(self, ridx, other_idx):
        try:
            key1 = (
                self._win_sid[int(ridx)],
                int(self._win_time[int(ridx)]),
                int(self._win_w[int(ridx)]),
            )
            key2 = (
                self._win_sid[int(other_idx)],
                int(self._win_time[int(other_idx)]),
                int(self._win_w[int(other_idx)]),
            )
        except (IndexError, TypeError, ValueError):
            return None
        return self._window_distance_sq_by_keys(key1, key2)

    def _window_distance_sq_by_keys(self, key1, key2):
        v1 = self.sketches.get(key1)
        v2 = self.sketches.get(key2)
        if v1 is None or v2 is None:
            return None
        if np.isscalar(v1) and np.isscalar(v2):
            a_val = float(v1)
            b_val = float(v2)
            dist_sq = (a_val - b_val) * (a_val - b_val)
            if self.neg_corr:
                neg_dist_sq = (a_val + b_val) * (a_val + b_val)
                dist_sq = min(dist_sq, neg_dist_sq)
            return dist_sq
        a = np.asarray(v1, dtype=np.float64).ravel()
        b = np.asarray(v2, dtype=np.float64).ravel()
        if a.size == 0 or b.size == 0:
            return None
        if a.size != b.size:
            dim = max(a.size, b.size)
            a_fixed = np.zeros(dim, dtype=np.float64)
            b_fixed = np.zeros(dim, dtype=np.float64)
            a_fixed[: a.size] = a
            b_fixed[: b.size] = b
            a = a_fixed
            b = b_fixed
        diff = a - b
        dist_sq = float(np.dot(diff, diff))
        if self.neg_corr:
            neg_diff = a + b
            dist_sq = min(dist_sq, float(np.dot(neg_diff, neg_diff)))
        return dist_sq

    @staticmethod
    def _split_array_indices(n_items, max_workers):
        if n_items <= 0:
            return []
        try:
            n_workers = int(max_workers)
        except (TypeError, ValueError):
            n_workers = 1
        if n_workers <= 1:
            return [np.arange(n_items, dtype=np.int64)]
        n_workers = min(n_workers, n_items)
        return [chunk for chunk in np.array_split(np.arange(n_items, dtype=np.int64), n_workers) if chunk.size > 0]

    def estimate_recent_range_hits(self, tau):
        if self._candidate_backend != "flat":
            return None
        if not self._recent_window_ids or not self._values:
            return None
        try:
            tau = float(tau)
        except (TypeError, ValueError):
            return None
        if tau < 0.0:
            return None

        total = 0
        n_values = len(self._values)
        for window_id in self._recent_window_ids:
            values = self._reverse_index.get(window_id, ())
            if not values:
                continue
            for value in set(values):
                lower = float(value) - tau
                upper = float(value) + tau
                left = bisect.bisect_left(self._values, lower)
                right = bisect.bisect_right(self._values, upper)
                total += max(0, min(right, n_values) - left)
        return total

    def _pairs_to_local_maps(self, pairs, win_sid, win_time, win_w, want_dist):
        local_freq = {}
        local_dist = {} if want_dist else None
        seen_pairs = set()
        dummy_candidates = {}
        if not pairs:
            return local_freq, local_dist
        for item in pairs:
            if len(item) >= 3:
                ridx, other_idx, dist_sq = item[:3]
            else:
                ridx, other_idx = item[:2]
                dist_sq = None
            ridx = int(ridx)
            other_idx = int(other_idx)
            sid1 = win_sid[ridx]
            sid2 = win_sid[other_idx]
            t1 = win_time[ridx]
            t2 = win_time[other_idx]
            w = win_w[ridx]
            if dist_sq is None and want_dist:
                dist_sq = self._window_distance_sq_by_keys(
                    (sid1, int(t1), int(w)),
                    (sid2, int(t2), int(w)),
                )
            pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
            self._record_candidate_hit(
                pair_id,
                local_freq,
                dummy_candidates,
                seen_pairs,
                local_dist,
                dist_sq,
            )
        return local_freq, local_dist

    def _pairs_to_global_maps(self, pairs, win_sid, win_time, win_w, freq_pairs, candidates, dist_pairs):
        if not pairs:
            return
        want_dist = dist_pairs is not None
        seen_pairs = set()
        for item in pairs:
            if len(item) >= 3:
                ridx, other_idx, dist_sq = item[:3]
            else:
                ridx, other_idx = item[:2]
                dist_sq = None
            ridx = int(ridx)
            other_idx = int(other_idx)
            sid1 = win_sid[ridx]
            sid2 = win_sid[other_idx]
            t1 = win_time[ridx]
            t2 = win_time[other_idx]
            w = win_w[ridx]
            if dist_sq is None and want_dist:
                dist_sq = self._window_distance_sq_by_keys(
                    (sid1, int(t1), int(w)),
                    (sid2, int(t2), int(w)),
                )
            pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
            self._record_candidate_hit(
                pair_id,
                freq_pairs,
                candidates,
                seen_pairs,
                dist_pairs,
                dist_sq,
            )

    def _merge_shard_maps(self, shard_results, freq_pairs, candidates, dist_pairs):
        seen_pairs = set()
        want_dist = dist_pairs is not None
        for local_freq, local_dist in shard_results:
            for pair_id in local_freq:
                if pair_id not in seen_pairs:
                    seen_pairs.add(pair_id)
                    freq_pairs[pair_id] = freq_pairs.get(pair_id, 0.0) + 1.0
                    if freq_pairs[pair_id] >= self.freq_threshold:
                        candidates[pair_id] = 1
                if want_dist and local_dist is not None and pair_id in local_dist:
                    dist_sq = float(local_dist[pair_id])
                    prev = dist_pairs.get(pair_id)
                    if prev is None or dist_sq < prev:
                        dist_pairs[pair_id] = dist_sq

    def _rows_to_global_maps(self, rows, freq_pairs, candidates):
        if rows is None:
            return
        rows = np.asarray(rows, dtype=np.int64).reshape((-1, 5))
        if rows.size == 0:
            return
        seen_pairs = set()
        sid_list = self._sid_list
        for sid1_idx, sid2_idx, t1, t2, w in rows:
            sid1_idx = int(sid1_idx)
            sid2_idx = int(sid2_idx)
            sid1 = sid_list[sid1_idx] if 0 <= sid1_idx < len(sid_list) else str(sid1_idx)
            sid2 = sid_list[sid2_idx] if 0 <= sid2_idx < len(sid_list) else str(sid2_idx)
            pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
            self._record_candidate_hit(pair_id, freq_pairs, candidates, seen_pairs)

    def uses_internal_parallel_candidate_search(self):
        if not self._vector_match_enabled:
            return False
        if (
            self._candidate_backend == "bptree"
            and self._tree_index is not None
            and hasattr(self._tree_index, "find_pairs_full_parallel")
        ):
            return True
        if self._candidate_backend == "flat" and (
            _cy_find_candidate_pairs_full_parallel is not None
            or _cy_find_candidate_pairs_full_with_dist_parallel is not None
        ):
            return True
        return False

    def _increment_candidates_sharded(self, freq_pairs, candidates, dist_pairs=None, max_workers=1, executor=None):
        if not self._recent_window_ids:
            return False
        tau = self._candidate_search_tau()
        if tau is None:
            return False
        tau = float(tau)
        if tau < 0.0:
            return False
        if getattr(self, "candidate_similarity", "l2") == "cosine":
            return False
        try:
            max_workers = int(max_workers)
        except (TypeError, ValueError):
            max_workers = 1
        if max_workers <= 1:
            return False

        want_dist = dist_pairs is not None
        tree_ready = (
            self._candidate_backend == "bptree"
            and self._tree_index is not None
            and self._recent_entry_ids
            and len(self._win_sid) > 0
            and len(self._win_time) == len(self._win_sid)
            and len(self._win_w) == len(self._win_sid)
            and len(self._win_sid_idx) == len(self._win_sid)
        )

        if tree_ready:
            recent_entry_ids = np.asarray(self._recent_entry_ids, dtype=np.int64)
            chunks = self._split_array_indices(recent_entry_ids.shape[0], max_workers)
            if len(chunks) <= 1:
                return False
            win_sid_idx = np.asarray(self._win_sid_idx, dtype=np.int64)
            win_time = np.asarray(self._win_time, dtype=np.int64)
            win_w_arr = np.asarray(self._win_w, dtype=np.int64)
            win_w = list(self._win_w)
            win_sid = list(self._win_sid)

            if self._vector_match_enabled and hasattr(self._tree_index, "find_pairs_full_parallel"):
                if want_dist and hasattr(self._tree_index, "find_pairs_full_with_dist_parallel"):
                    pairs = self._tree_index.find_pairs_full_with_dist_parallel(
                        recent_entry_ids,
                        win_sid_idx,
                        win_time,
                        float(tau),
                        int(max_workers),
                    )
                else:
                    pairs = self._tree_index.find_pairs_full_parallel(
                        recent_entry_ids,
                        win_sid_idx,
                        win_time,
                        float(tau),
                        int(max_workers),
                    )
                self._pairs_to_global_maps(pairs, win_sid, win_time, win_w_arr, freq_pairs, candidates, dist_pairs)
                return True

            def run_tree_chunk(pos_idx):
                ids_chunk = recent_entry_ids[pos_idx]
                if self._vector_match_enabled and want_dist and hasattr(self._tree_index, "find_pairs_full_with_dist"):
                    pairs = self._tree_index.find_pairs_full_with_dist(ids_chunk, win_sid_idx, win_time, float(tau))
                elif (not self._vector_match_enabled) and want_dist and hasattr(self._tree_index, "find_pairs_with_dist"):
                    pairs = self._tree_index.find_pairs_with_dist(ids_chunk, win_sid_idx, win_time, float(tau))
                elif self._vector_match_enabled:
                    pairs = self._tree_index.find_pairs_full(ids_chunk, win_sid_idx, win_time, float(tau))
                else:
                    pairs = self._tree_index.find_pairs(ids_chunk, win_sid_idx, win_time, float(tau))
                return pairs

            owns_executor = executor is None
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=min(max_workers, len(chunks)))
            try:
                shard_results = list(executor.map(run_tree_chunk, chunks))
            finally:
                if owns_executor:
                    executor.shutdown(wait=True, cancel_futures=True)
            for pairs in shard_results:
                self._pairs_to_global_maps(pairs, win_sid, win_time, win_w_arr, freq_pairs, candidates, dist_pairs)
            return True

        if self._candidate_backend == "bptree":
            return False

        arrays = self._build_candidate_arrays()
        if arrays is None:
            return False
        (
            values,
            value_window_idx,
            recent_values,
            recent_window_idx,
            win_sid_idx,
            win_time,
            win_w,
            entry_vectors,
            recent_vectors,
            win_sid,
        ) = arrays
        chunks = self._split_array_indices(recent_values.shape[0], max_workers)
        if len(chunks) <= 1:
            return False

        if self._vector_match_enabled and (
            _cy_find_candidate_pairs_full_parallel is not None
            or _cy_find_candidate_pairs_full_with_dist_parallel is not None
        ):
            if want_dist and _cy_find_candidate_pairs_full_with_dist_parallel is not None:
                pairs = _cy_find_candidate_pairs_full_with_dist_parallel(
                    values,
                    value_window_idx,
                    recent_values,
                    recent_window_idx,
                    win_sid_idx,
                    win_time,
                    entry_vectors,
                    recent_vectors,
                    float(tau),
                    int(max_workers),
                )
            elif _cy_find_candidate_pairs_full_parallel is not None:
                pairs = _cy_find_candidate_pairs_full_parallel(
                    values,
                    value_window_idx,
                    recent_values,
                    recent_window_idx,
                    win_sid_idx,
                    win_time,
                    entry_vectors,
                    recent_vectors,
                    float(tau),
                    int(max_workers),
                )
            else:
                pairs = None
            if pairs is not None:
                self._pairs_to_global_maps(pairs, win_sid, win_time, win_w, freq_pairs, candidates, dist_pairs)
                return True

        def run_flat_chunk(pos_idx):
            rv = recent_values[pos_idx]
            rw = recent_window_idx[pos_idx]
            if self._vector_match_enabled:
                rvec = recent_vectors[pos_idx]
                if want_dist and _cy_find_candidate_pairs_full_with_dist is not None:
                    pairs = _cy_find_candidate_pairs_full_with_dist(
                        values,
                        value_window_idx,
                        rv,
                        rw,
                        win_sid_idx,
                        win_time,
                        entry_vectors,
                        rvec,
                        float(tau),
                    )
                else:
                    pairs = _cy_find_candidate_pairs_full(
                        values,
                        value_window_idx,
                        rv,
                        rw,
                        win_sid_idx,
                        win_time,
                        entry_vectors,
                        rvec,
                        float(tau),
                    )
            elif want_dist and _cy_find_candidate_pairs_with_dist is not None:
                pairs = _cy_find_candidate_pairs_with_dist(
                    values,
                    value_window_idx,
                    rv,
                    rw,
                    win_sid_idx,
                    win_time,
                    float(tau),
                )
            else:
                pairs = _cy_find_candidate_pairs(
                    values,
                    value_window_idx,
                    rv,
                    rw,
                    win_sid_idx,
                    win_time,
                    float(tau),
                )
            return pairs

        owns_executor = executor is None
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=min(max_workers, len(chunks)))
        try:
            shard_results = list(executor.map(run_flat_chunk, chunks))
        finally:
            if owns_executor:
                executor.shutdown(wait=True, cancel_futures=True)
        for pairs in shard_results:
            self._pairs_to_global_maps(pairs, win_sid, win_time, win_w, freq_pairs, candidates, dist_pairs)
        return True

    def _input_tree(self):
        if not self.partition:
            return
        last_partition = self.partition[-1]
        if (
            self._candidate_backend in _BLOCKED_INDEX_BACKENDS
            and self._blocked_index is not None
            and hasattr(self._blocked_index, "insert_many")
        ):
            if self._input_tree_blocked_lazy_batch(last_partition):
                return
        if (
            self._candidate_backend in _INSTINCT_INDEX_BACKENDS
            and self._instinct_index is not None
            and hasattr(self._instinct_index, "insert_many")
        ):
            if self._input_tree_instinct_batch(last_partition):
                return
        if (
            self._candidate_backend == "bptree"
            and self._tree_index is not None
            and hasattr(self._tree_index, "insert_many")
        ):
            if self._input_tree_bptree_batch(last_partition):
                return
        if not isinstance(last_partition, dict):
            last_partition = self._partition_to_dict(last_partition)
            if not last_partition:
                self._recent_window_ids = set()
                self._recent_entry_ids = []
                return
        self._recent_window_ids = set()
        self._recent_entry_ids = []
        for k, v in last_partition.items():
            if len(v) >= 3:
                sketch, is_constant, _norm = v[:3]
            else:
                continue
            if is_constant:
                continue
            vec = np.asarray(sketch, dtype=np.float64).ravel()
            if vec.size == 0:
                continue
            if self._vector_match_enabled and vec.size != self._vector_dim:
                fixed = np.zeros((self._vector_dim,), dtype=np.float64)
                copy_n = min(vec.size, self._vector_dim)
                if copy_n > 0:
                    fixed[:copy_n] = vec[:copy_n]
                vec = fixed
            win_idx = self._get_or_create_window_idx(k)
            sid_meta = int(self._win_sid_idx[win_idx])
            rank_meta = int(self._win_sid_rank[win_idx])
            time_meta = int(k[1])
            window_meta = int(k[2])
            if self._vector_match_enabled:
                value = self._candidate_key_value(vec)
                single_index_neg = self._use_single_index_neg_corr()
                if self._candidate_backend == "bptree" and self._tree_index is not None:
                    entry_id = self._tree_index.insert(value, win_idx, vec, sid_meta, time_meta, window_meta, rank_meta)
                    self._reverse_entry_ids[k].append(entry_id)
                    self._recent_entry_ids.append(entry_id)
                    if self.neg_corr and not single_index_neg:
                        neg_vec = -vec
                        neg_value = self._candidate_key_value(neg_vec)
                        neg_entry_id = self._tree_index.insert(neg_value, win_idx, neg_vec, sid_meta, time_meta, window_meta, rank_meta)
                        self._reverse_entry_ids[k].append(neg_entry_id)
                        self._recent_entry_ids.append(neg_entry_id)
                else:
                    self._insert_entry(value, k, vec)
                    self._reverse_index[k].append(value)
                    self._reverse_vectors[k].append(vec)
                    if self.neg_corr:
                        neg_vec = -vec
                        neg_value = self._candidate_key_value(neg_vec)
                        self._insert_entry(neg_value, k, neg_vec)
                        self._reverse_index[k].append(neg_value)
                        self._reverse_vectors[k].append(neg_vec)
                self.sketches[k] = vec
            else:
                value = self._candidate_key_value(vec)
                single_index_neg = self._use_single_index_neg_corr()
                if self._candidate_backend == "bptree" and self._tree_index is not None:
                    entry_id = self._tree_index.insert(value, win_idx, None, sid_meta, time_meta, window_meta, rank_meta)
                    self._reverse_entry_ids[k].append(entry_id)
                    self._recent_entry_ids.append(entry_id)
                    if self.neg_corr and not single_index_neg:
                        neg_value = self._candidate_key_value(-vec)
                        neg_entry_id = self._tree_index.insert(neg_value, win_idx, None, sid_meta, time_meta, window_meta, rank_meta)
                        self._reverse_entry_ids[k].append(neg_entry_id)
                        self._recent_entry_ids.append(neg_entry_id)
                else:
                    self._insert_entry(value, k)
                    self._reverse_index[k].append(value)
                    if self.neg_corr:
                        neg_value = self._candidate_key_value(-vec)
                        self._insert_entry(neg_value, k)
                        self._reverse_index[k].append(neg_value)
                self.sketches[k] = value
            self._recent_window_ids.add(k)

    def _input_tree_blocked_lazy_batch(self, last_partition):
        if isinstance(last_partition, tuple) and len(last_partition) in {5, 6}:
            return self._input_tree_blocked_lazy_batch_tuple(last_partition)
        if not isinstance(last_partition, dict):
            return False

        self._recent_window_ids = set()
        self._recent_entry_ids = []
        win_indices = []
        sid_indices = []
        sid_ranks = []
        times = []
        window_sizes = []
        vectors = []
        keys_for_entries = []

        for key, value_tuple in last_partition.items():
            if len(value_tuple) >= 3:
                sketch, is_constant, _norm = value_tuple[:3]
            else:
                continue
            if is_constant:
                continue
            vec = np.asarray(sketch, dtype=np.float64).ravel()
            if vec.size == 0:
                continue
            if vec.size != self._vector_dim:
                fixed = np.zeros((self._vector_dim,), dtype=np.float64)
                copy_n = min(vec.size, self._vector_dim)
                if copy_n > 0:
                    fixed[:copy_n] = vec[:copy_n]
                vec = fixed

            win_idx = self._get_or_create_window_idx(key)
            sid_meta = int(self._win_sid_idx[win_idx])
            rank_meta = int(self._win_sid_rank[win_idx])
            win_indices.append(win_idx)
            sid_indices.append(sid_meta)
            sid_ranks.append(rank_meta)
            times.append(int(key[1]))
            window_sizes.append(int(key[2]))
            vectors.append(vec)
            keys_for_entries.append(key)
            self._recent_window_ids.add(key)

        if not vectors:
            return True

        vector_arr = np.ascontiguousarray(np.vstack(vectors), dtype=np.float64)
        values = self._candidate_key_values(vector_arr, self.candidate_block_index_dims)
        t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        entry_ids = self._blocked_index.insert_many(
            values,
            np.ascontiguousarray(win_indices, dtype=np.int64),
            vector_arr,
            np.ascontiguousarray(sid_indices, dtype=np.int64),
            np.ascontiguousarray(times, dtype=np.int64),
            np.ascontiguousarray(window_sizes, dtype=np.int64),
            np.ascontiguousarray(sid_ranks, dtype=np.int64),
        )
        if t0 is not None:
            metric = "cand.blocked_insert"
            self._profile_add(metric, time.perf_counter() - t0)
        for key, entry_id in zip(keys_for_entries, np.asarray(entry_ids, dtype=np.int64)):
            entry_id = int(entry_id)
            self._reverse_entry_ids[key].append(entry_id)
            self._recent_entry_ids.append(entry_id)
        return True

    def _input_tree_blocked_lazy_batch_tuple(self, partition):
        sid_idx_arr, time_arr, w_arr, chunk_arr, is_const = partition[:5]
        if len(partition) >= 6:
            entry_ids = np.asarray(partition[5], dtype=np.int64).ravel()
            self._recent_entry_ids = [int(x) for x in entry_ids]
            self._recent_window_ids = set()
            return True
        if sid_idx_arr is None or len(sid_idx_arr) == 0:
            self._recent_window_ids = set()
            self._recent_entry_ids = []
            if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
                self.partition[-1] = tuple(partition[:5]) + (np.empty(0, dtype=np.int64),)
            return True

        sid_idx_arr = np.asarray(sid_idx_arr, dtype=np.int64)
        time_arr = np.asarray(time_arr, dtype=np.int64)
        w_arr = np.asarray(w_arr, dtype=np.int64)
        chunks = np.asarray(chunk_arr, dtype=np.float64)
        is_const = np.asarray(is_const, dtype=np.uint8)
        n_rows = int(sid_idx_arr.shape[0])
        if time_arr.shape[0] != n_rows or w_arr.shape[0] != n_rows or is_const.shape[0] != n_rows:
            return False
        chunks_2d = chunks.reshape((n_rows, 1)) if chunks.ndim == 1 else chunks.reshape((n_rows, chunks.shape[1]))
        if chunks_2d.shape[0] != n_rows:
            return False

        valid_mask = is_const == 0
        if not np.any(valid_mask):
            self._recent_window_ids = set()
            self._recent_entry_ids = []
            if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
                self.partition[-1] = tuple(partition[:5]) + (np.empty(0, dtype=np.int64),)
            return True

        valid_pos = np.flatnonzero(valid_mask)
        valid_chunks = np.ascontiguousarray(chunks_2d[valid_pos], dtype=np.float64)
        valid_sid = np.ascontiguousarray(sid_idx_arr[valid_pos], dtype=np.int64)
        valid_time = np.ascontiguousarray(time_arr[valid_pos], dtype=np.int64)
        valid_w = np.ascontiguousarray(w_arr[valid_pos], dtype=np.int64)

        if valid_chunks.shape[1] != self._vector_dim:
            fixed = np.zeros((valid_chunks.shape[0], self._vector_dim), dtype=np.float64)
            copy_n = min(valid_chunks.shape[1], self._vector_dim)
            if copy_n > 0:
                fixed[:, :copy_n] = valid_chunks[:, :copy_n]
            valid_chunks = fixed

        win_indices = np.empty(valid_pos.shape[0], dtype=np.int64)
        sid_ranks = np.empty(valid_pos.shape[0], dtype=np.int64)
        for out_i in range(valid_pos.shape[0]):
            sid_idx = int(valid_sid[out_i])
            win_indices[out_i] = self._get_or_create_window_idx_numeric(sid_idx, int(valid_time[out_i]), int(valid_w[out_i]))
            if sid_idx < len(self._sid_idx_rank):
                sid_ranks[out_i] = int(self._sid_idx_rank[sid_idx])
            else:
                sid = self._sid_list[sid_idx] if 0 <= sid_idx < len(self._sid_list) else str(sid_idx)
                sid_ranks[out_i] = int(self._sid_sort_rank_map.get(sid, sid_idx))

        values = self._candidate_key_values(valid_chunks, self.candidate_block_index_dims)
        t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        entry_ids = self._blocked_index.insert_many(
            values,
            np.ascontiguousarray(win_indices, dtype=np.int64),
            valid_chunks,
            np.ascontiguousarray(valid_sid, dtype=np.int64),
            np.ascontiguousarray(valid_time, dtype=np.int64),
            np.ascontiguousarray(valid_w, dtype=np.int64),
            np.ascontiguousarray(sid_ranks, dtype=np.int64),
        )
        if t0 is not None:
            metric = "cand.blocked_insert"
            self._profile_add(metric, time.perf_counter() - t0)
        entry_ids = np.ascontiguousarray(entry_ids, dtype=np.int64)
        self._recent_entry_ids = [int(x) for x in entry_ids]
        self._recent_window_ids = set(int(x) for x in win_indices)
        if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
            self.partition[-1] = tuple(partition[:5]) + (entry_ids,)
        return True

    def _input_tree_instinct_batch(self, last_partition):
        # (2026-07-06) InstinctIndex -- experimental approximate graph
        # backend. Mirrors _input_tree_blocked_lazy_batch exactly (same
        # insert_many contract), except InstinctIndex ignores the
        # coordinate-key "values" argument entirely (no sorted-array
        # indexing needed for a graph), so no _candidate_key_values(...)
        # call is needed here. See docs/implementation_log.md, "InstinctIndex:
        # experimental approximate graph backend".
        if isinstance(last_partition, tuple) and len(last_partition) in {5, 6}:
            return self._input_tree_instinct_batch_tuple(last_partition)
        if not isinstance(last_partition, dict):
            return False

        self._recent_window_ids = set()
        self._recent_entry_ids = []
        win_indices = []
        sid_indices = []
        sid_ranks = []
        times = []
        window_sizes = []
        vectors = []
        keys_for_entries = []

        for key, value_tuple in last_partition.items():
            if len(value_tuple) >= 3:
                sketch, is_constant, _norm = value_tuple[:3]
            else:
                continue
            if is_constant:
                continue
            vec = np.asarray(sketch, dtype=np.float64).ravel()
            if vec.size == 0:
                continue
            if vec.size != self._vector_dim:
                fixed = np.zeros((self._vector_dim,), dtype=np.float64)
                copy_n = min(vec.size, self._vector_dim)
                if copy_n > 0:
                    fixed[:copy_n] = vec[:copy_n]
                vec = fixed

            win_idx = self._get_or_create_window_idx(key)
            sid_meta = int(self._win_sid_idx[win_idx])
            rank_meta = int(self._win_sid_rank[win_idx])
            win_indices.append(win_idx)
            sid_indices.append(sid_meta)
            sid_ranks.append(rank_meta)
            times.append(int(key[1]))
            window_sizes.append(int(key[2]))
            vectors.append(vec)
            keys_for_entries.append(key)
            self._recent_window_ids.add(key)

        if not vectors:
            return True

        vector_arr = np.ascontiguousarray(np.vstack(vectors), dtype=np.float64)
        t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        entry_ids = self._instinct_index.insert_many(
            None,
            np.ascontiguousarray(win_indices, dtype=np.int64),
            vector_arr,
            np.ascontiguousarray(sid_indices, dtype=np.int64),
            np.ascontiguousarray(times, dtype=np.int64),
            np.ascontiguousarray(window_sizes, dtype=np.int64),
            np.ascontiguousarray(sid_ranks, dtype=np.int64),
        )
        if t0 is not None:
            self._profile_add("cand.instinct_insert", time.perf_counter() - t0)
        for key, entry_id in zip(keys_for_entries, np.asarray(entry_ids, dtype=np.int64)):
            entry_id = int(entry_id)
            self._reverse_entry_ids[key].append(entry_id)
            self._recent_entry_ids.append(entry_id)
        return True

    def _input_tree_instinct_batch_tuple(self, partition):
        sid_idx_arr, time_arr, w_arr, chunk_arr, is_const = partition[:5]
        if len(partition) >= 6:
            entry_ids = np.asarray(partition[5], dtype=np.int64).ravel()
            self._recent_entry_ids = [int(x) for x in entry_ids]
            self._recent_window_ids = set()
            return True
        if sid_idx_arr is None or len(sid_idx_arr) == 0:
            self._recent_window_ids = set()
            self._recent_entry_ids = []
            if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
                self.partition[-1] = tuple(partition[:5]) + (np.empty(0, dtype=np.int64),)
            return True

        sid_idx_arr = np.asarray(sid_idx_arr, dtype=np.int64)
        time_arr = np.asarray(time_arr, dtype=np.int64)
        w_arr = np.asarray(w_arr, dtype=np.int64)
        chunks = np.asarray(chunk_arr, dtype=np.float64)
        is_const = np.asarray(is_const, dtype=np.uint8)
        n_rows = int(sid_idx_arr.shape[0])
        if time_arr.shape[0] != n_rows or w_arr.shape[0] != n_rows or is_const.shape[0] != n_rows:
            return False
        chunks_2d = chunks.reshape((n_rows, 1)) if chunks.ndim == 1 else chunks.reshape((n_rows, chunks.shape[1]))
        if chunks_2d.shape[0] != n_rows:
            return False

        valid_mask = is_const == 0
        if not np.any(valid_mask):
            self._recent_window_ids = set()
            self._recent_entry_ids = []
            if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
                self.partition[-1] = tuple(partition[:5]) + (np.empty(0, dtype=np.int64),)
            return True

        valid_pos = np.flatnonzero(valid_mask)
        valid_chunks = np.ascontiguousarray(chunks_2d[valid_pos], dtype=np.float64)
        valid_sid = np.ascontiguousarray(sid_idx_arr[valid_pos], dtype=np.int64)
        valid_time = np.ascontiguousarray(time_arr[valid_pos], dtype=np.int64)
        valid_w = np.ascontiguousarray(w_arr[valid_pos], dtype=np.int64)

        if valid_chunks.shape[1] != self._vector_dim:
            fixed = np.zeros((valid_chunks.shape[0], self._vector_dim), dtype=np.float64)
            copy_n = min(valid_chunks.shape[1], self._vector_dim)
            if copy_n > 0:
                fixed[:, :copy_n] = valid_chunks[:, :copy_n]
            valid_chunks = fixed

        win_indices = np.empty(valid_pos.shape[0], dtype=np.int64)
        sid_ranks = np.empty(valid_pos.shape[0], dtype=np.int64)
        for out_i in range(valid_pos.shape[0]):
            sid_idx = int(valid_sid[out_i])
            win_indices[out_i] = self._get_or_create_window_idx_numeric(sid_idx, int(valid_time[out_i]), int(valid_w[out_i]))
            if sid_idx < len(self._sid_idx_rank):
                sid_ranks[out_i] = int(self._sid_idx_rank[sid_idx])
            else:
                sid = self._sid_list[sid_idx] if 0 <= sid_idx < len(self._sid_list) else str(sid_idx)
                sid_ranks[out_i] = int(self._sid_sort_rank_map.get(sid, sid_idx))

        t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
        entry_ids = self._instinct_index.insert_many(
            None,
            np.ascontiguousarray(win_indices, dtype=np.int64),
            valid_chunks,
            np.ascontiguousarray(valid_sid, dtype=np.int64),
            np.ascontiguousarray(valid_time, dtype=np.int64),
            np.ascontiguousarray(valid_w, dtype=np.int64),
            np.ascontiguousarray(sid_ranks, dtype=np.int64),
        )
        if t0 is not None:
            self._profile_add("cand.instinct_insert", time.perf_counter() - t0)
        entry_ids = np.ascontiguousarray(entry_ids, dtype=np.int64)
        self._recent_entry_ids = [int(x) for x in entry_ids]
        self._recent_window_ids = set(int(x) for x in win_indices)
        if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
            self.partition[-1] = tuple(partition[:5]) + (entry_ids,)
        return True

    def _input_tree_bptree_batch(self, last_partition):
        if isinstance(last_partition, tuple) and len(last_partition) in {5, 6}:
            return self._input_tree_bptree_batch_tuple(last_partition)
        if not isinstance(last_partition, dict):
            return False
        self._recent_window_ids = set()
        self._recent_entry_ids = []
        values = []
        win_indices = []
        sid_indices = []
        sid_ranks = []
        times = []
        window_sizes = []
        vectors = [] if self._vector_match_enabled else None
        keys_for_entries = []

        for k, v in last_partition.items():
            if len(v) >= 3:
                sketch, is_constant, _norm = v[:3]
            else:
                continue
            if is_constant:
                continue
            vec = np.asarray(sketch, dtype=np.float64).ravel()
            if vec.size == 0:
                continue
            if self._vector_match_enabled and vec.size != self._vector_dim:
                fixed = np.zeros((self._vector_dim,), dtype=np.float64)
                copy_n = min(vec.size, self._vector_dim)
                if copy_n > 0:
                    fixed[:copy_n] = vec[:copy_n]
                vec = fixed

            win_idx = self._get_or_create_window_idx(k)
            sid_meta = int(self._win_sid_idx[win_idx])
            rank_meta = int(self._win_sid_rank[win_idx])
            time_meta = int(k[1])
            window_meta = int(k[2])

            value = self._candidate_key_value(vec)
            values.append(value)
            win_indices.append(win_idx)
            sid_indices.append(sid_meta)
            sid_ranks.append(rank_meta)
            times.append(time_meta)
            window_sizes.append(window_meta)
            keys_for_entries.append(k)
            if self._vector_match_enabled:
                vectors.append(vec)

            if self.neg_corr and not self._use_single_index_neg_corr():
                neg_vec = -vec
                neg_value = self._candidate_key_value(neg_vec)
                values.append(neg_value)
                win_indices.append(win_idx)
                sid_indices.append(sid_meta)
                sid_ranks.append(rank_meta)
                times.append(time_meta)
                window_sizes.append(window_meta)
                keys_for_entries.append(k)
                if self._vector_match_enabled:
                    vectors.append(neg_vec)

            self.sketches[k] = vec if self._vector_match_enabled else value
            self._recent_window_ids.add(k)

        if not values:
            return True

        vector_arr = np.ascontiguousarray(np.vstack(vectors), dtype=np.float64) if self._vector_match_enabled else None
        entry_ids = self._tree_index.insert_many(
            np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(win_indices, dtype=np.int64),
            vector_arr,
            np.ascontiguousarray(sid_indices, dtype=np.int64),
            np.ascontiguousarray(times, dtype=np.int64),
            np.ascontiguousarray(window_sizes, dtype=np.int64),
            np.ascontiguousarray(sid_ranks, dtype=np.int64),
        )
        for k, entry_id in zip(keys_for_entries, np.asarray(entry_ids, dtype=np.int64)):
            entry_id = int(entry_id)
            self._reverse_entry_ids[k].append(entry_id)
            self._recent_entry_ids.append(entry_id)
        return True

    def _input_tree_bptree_batch_tuple(self, partition):
        sid_idx_arr, time_arr, w_arr, chunk_arr, is_const = partition[:5]
        if len(partition) >= 6:
            entry_ids = np.asarray(partition[5], dtype=np.int64).ravel()
            self._recent_entry_ids = [int(x) for x in entry_ids]
            self._recent_window_ids = set()
            return True
        if sid_idx_arr is None or len(sid_idx_arr) == 0:
            self._recent_window_ids = set()
            self._recent_entry_ids = []
            if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
                self.partition[-1] = tuple(partition[:5]) + (np.empty(0, dtype=np.int64),)
            return True

        sid_idx_arr = np.asarray(sid_idx_arr, dtype=np.int64)
        time_arr = np.asarray(time_arr, dtype=np.int64)
        w_arr = np.asarray(w_arr, dtype=np.int64)
        chunks = np.asarray(chunk_arr, dtype=np.float64)
        is_const = np.asarray(is_const, dtype=np.uint8)
        n_rows = int(sid_idx_arr.shape[0])
        if time_arr.shape[0] != n_rows or w_arr.shape[0] != n_rows or is_const.shape[0] != n_rows:
            return False
        chunks_2d = chunks.reshape((n_rows, 1)) if chunks.ndim == 1 else chunks.reshape((n_rows, chunks.shape[1]))
        if chunks_2d.shape[0] != n_rows:
            return False

        valid_mask = is_const == 0
        if not np.any(valid_mask):
            self._recent_window_ids = set()
            self._recent_entry_ids = []
            if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
                self.partition[-1] = tuple(partition[:5]) + (np.empty(0, dtype=np.int64),)
            return True

        valid_pos = np.flatnonzero(valid_mask)
        valid_chunks = np.ascontiguousarray(chunks_2d[valid_pos], dtype=np.float64)
        valid_sid = np.ascontiguousarray(sid_idx_arr[valid_pos], dtype=np.int64)
        valid_time = np.ascontiguousarray(time_arr[valid_pos], dtype=np.int64)
        valid_w = np.ascontiguousarray(w_arr[valid_pos], dtype=np.int64)

        win_indices = np.empty(valid_pos.shape[0], dtype=np.int64)
        sid_ranks = np.empty(valid_pos.shape[0], dtype=np.int64)
        keep_sketch_cache = bool(getattr(self, "return_distances", False))
        for out_i in range(valid_pos.shape[0]):
            sid_idx = int(valid_sid[out_i])
            win_indices[out_i] = self._get_or_create_window_idx_numeric(sid_idx, int(valid_time[out_i]), int(valid_w[out_i]))
            if sid_idx < len(self._sid_idx_rank):
                sid_ranks[out_i] = int(self._sid_idx_rank[sid_idx])
            else:
                sid = self._sid_list[sid_idx] if 0 <= sid_idx < len(self._sid_list) else str(sid_idx)
                sid_ranks[out_i] = int(self._sid_sort_rank_map.get(sid, sid_idx))
            if keep_sketch_cache:
                sid = self._sid_list[sid_idx] if 0 <= sid_idx < len(self._sid_list) else str(sid_idx)
                self.sketches[(sid, int(valid_time[out_i]), int(valid_w[out_i]))] = valid_chunks[out_i]

        values = np.ascontiguousarray(self._candidate_key_values(valid_chunks, 1).ravel(), dtype=np.float64)
        vectors = valid_chunks if self._vector_match_enabled else None
        if self.neg_corr and not self._use_single_index_neg_corr():
            values = np.ascontiguousarray(np.concatenate((values, -values)), dtype=np.float64)
            win_indices = np.ascontiguousarray(np.concatenate((win_indices, win_indices)), dtype=np.int64)
            valid_sid = np.ascontiguousarray(np.concatenate((valid_sid, valid_sid)), dtype=np.int64)
            valid_time = np.ascontiguousarray(np.concatenate((valid_time, valid_time)), dtype=np.int64)
            valid_w = np.ascontiguousarray(np.concatenate((valid_w, valid_w)), dtype=np.int64)
            sid_ranks = np.ascontiguousarray(np.concatenate((sid_ranks, sid_ranks)), dtype=np.int64)
            if self._vector_match_enabled:
                vectors = np.ascontiguousarray(np.vstack((valid_chunks, -valid_chunks)), dtype=np.float64)

        entry_ids = self._tree_index.insert_many(
            values,
            np.ascontiguousarray(win_indices, dtype=np.int64),
            vectors,
            np.ascontiguousarray(valid_sid, dtype=np.int64),
            np.ascontiguousarray(valid_time, dtype=np.int64),
            np.ascontiguousarray(valid_w, dtype=np.int64),
            np.ascontiguousarray(sid_ranks, dtype=np.int64),
        )
        entry_ids = np.ascontiguousarray(entry_ids, dtype=np.int64)
        self._recent_entry_ids = [int(x) for x in entry_ids]
        self._recent_window_ids = set(int(x) for x in win_indices)
        if len(self.partition) > 0 and self.partition[-1] is partition and len(partition) == 5:
            self.partition[-1] = tuple(partition[:5]) + (entry_ids,)
        return True

    def _clean_old_sketches(self):
        if len(self.partition) > self.n_lagged_windows:
            old_partition = self.partition.pop(0)
            if isinstance(old_partition, tuple) and len(old_partition) in {5, 6}:
                if self._candidate_backend in _BLOCKED_INDEX_BACKENDS and self._blocked_index is not None:
                    self._expire_blocked_lazy_index()
                    return
                if self._candidate_backend in _INSTINCT_INDEX_BACKENDS and self._instinct_index is not None:
                    self._expire_instinct_index()
                    return
                if (
                    len(old_partition) >= 6
                    and self._candidate_backend == "bptree"
                    and self._tree_index is not None
                ):
                    entry_ids = np.asarray(old_partition[5], dtype=np.int64).ravel()
                    if entry_ids.size:
                        remove_many = getattr(self._tree_index, "remove_many", None)
                        if remove_many is not None:
                            remove_many(np.ascontiguousarray(entry_ids, dtype=np.int64))
                        else:
                            for entry_id in entry_ids:
                                self._tree_index.remove(int(entry_id))
                if getattr(self, "return_distances", False):
                    old_sid_idx, old_time, old_w, _old_chunks, old_const = old_partition[:5]
                    if old_sid_idx is not None:
                        old_sid_idx = np.asarray(old_sid_idx, dtype=np.int64)
                        old_time = np.asarray(old_time, dtype=np.int64)
                        old_w = np.asarray(old_w, dtype=np.int64)
                        old_const = np.asarray(old_const, dtype=np.uint8)
                        n_old = min(old_sid_idx.shape[0], old_time.shape[0], old_w.shape[0], old_const.shape[0])
                        for old_i in range(n_old):
                            if int(old_const[old_i]) != 0:
                                continue
                            sid_idx = int(old_sid_idx[old_i])
                            sid = self._sid_list[sid_idx] if 0 <= sid_idx < len(self._sid_list) else str(sid_idx)
                            self.sketches.pop((sid, int(old_time[old_i]), int(old_w[old_i])), None)
                return
            remove_many = None
            entry_ids_to_remove = []
            if self._candidate_backend == "bptree" and self._tree_index is not None:
                remove_many = getattr(self._tree_index, "remove_many", None)
            if self._candidate_backend in _BLOCKED_INDEX_BACKENDS and self._blocked_index is not None:
                for key in old_partition:
                    self.sketches.pop(key, None)
                    self._reverse_entry_ids.pop(key, None)
                self._expire_blocked_lazy_index()
                return
            if self._candidate_backend in _INSTINCT_INDEX_BACKENDS and self._instinct_index is not None:
                for key in old_partition:
                    self.sketches.pop(key, None)
                    self._reverse_entry_ids.pop(key, None)
                self._expire_instinct_index()
                return
            for k, v in old_partition.items():
                if len(v) >= 3:
                    _sketch, is_constant, _norm = v[:3]
                else:
                    continue
                values = self._reverse_index.pop(k, [])
                self._reverse_vectors.pop(k, [])
                entry_ids = self._reverse_entry_ids.pop(k, [])
                if is_constant:
                    self.sketches.pop(k, None)
                    continue
                if self._candidate_backend == "bptree" and self._tree_index is not None:
                    if remove_many is not None:
                        entry_ids_to_remove.extend(entry_ids)
                    else:
                        for entry_id in entry_ids:
                            self._tree_index.remove(entry_id)
                else:
                    for val in values:
                        self._remove_entry(val, k)
                self.sketches.pop(k, None)
            if remove_many is not None and entry_ids_to_remove:
                remove_many(np.ascontiguousarray(entry_ids_to_remove, dtype=np.int64))

    def _update_grid(self, n_ids):
        if self.partition and self._partition_len(self.partition[-1]) >= n_ids:
            t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
            self._clean_old_sketches()
            if t0 is not None:
                self._profile_add("cand.clean_old", time.perf_counter() - t0)
                t0 = time.perf_counter()
            self._input_tree()
            if t0 is not None:
                self._profile_add("cand.input_tree", time.perf_counter() - t0)
    
    def _normalize_key(self, key):
        id1, id2, t1, t2, w = key

        if id1 == id2:
            # Autocorrelation case
            return (id1, id2, max(t1, t2), min(t1, t2), w)

        if t1 == t2:
            # Synchronous case
            return tuple(sorted([id1, id2])) + (t1, t2, w)

        if t1 < t2:
            # Cross-series, cross-time — canonicalize by lexicographic pair
            return (id2, id1, t2, t1, w)
        else:
            return (id1, id2, t1, t2, w)

    def update_n_lagged_windows(self,n_lagged_windows):
        self.n_lagged_windows = n_lagged_windows

    def _build_candidate_arrays(self):
        if self._candidate_backend != "flat":
            return None
        if not self._entries or not self._values or not self._recent_window_ids:
            return None
        if len(self._entries) != len(self._values):
            return None

        cache_version = int(getattr(self, "_index_version", 0) or 0)
        cached = getattr(self, "_candidate_arrays_cache", None)
        if cached is not None and getattr(self, "_candidate_arrays_cache_version", -1) == cache_version:
            return cached

        window_idx = self._window_idx
        win_sid = list(self._win_sid)
        win_time = np.asarray(self._win_time, dtype=np.int64)
        win_w = np.asarray(self._win_w, dtype=np.int64)
        win_sid_idx = np.asarray(self._win_sid_idx, dtype=np.int64)
        if not win_sid or win_time.shape[0] != len(win_sid) or win_w.shape[0] != len(win_sid) or win_sid_idx.shape[0] != len(win_sid):
            return None

        n_entries = len(self._entries)
        values = np.asarray(self._values, dtype=np.float64)
        value_window_idx = np.empty(n_entries, dtype=np.int64)
        entry_vectors = None
        if self._vector_match_enabled:
            entry_vectors = np.empty((n_entries, self._vector_dim), dtype=np.float64)

        for i, entry in enumerate(self._entries):
            win_idx = window_idx.get(entry[2])
            if win_idx is None:
                return None
            value_window_idx[i] = win_idx
            if entry_vectors is not None:
                vec = entry[3]
                if vec is None:
                    entry_vectors[i, :] = 0.0
                else:
                    arr = np.asarray(vec, dtype=np.float64).ravel()
                    if arr.size != self._vector_dim:
                        fixed = np.zeros((self._vector_dim,), dtype=np.float64)
                        copy_n = min(arr.size, self._vector_dim)
                        if copy_n > 0:
                            fixed[:copy_n] = arr[:copy_n]
                        arr = fixed
                    entry_vectors[i, :] = arr

        recent_values = []
        recent_window_idx = []
        recent_vectors = []
        for window_id in self._recent_window_ids:
            win_idx = window_idx.get(window_id)
            if win_idx is None:
                continue
            values_list = self._reverse_index.get(window_id, ())
            if not values_list:
                continue
            if entry_vectors is None:
                for val in values_list:
                    recent_values.append(val)
                    recent_window_idx.append(win_idx)
            else:
                vectors = self._reverse_vectors.get(window_id, ())
                if len(vectors) != len(values_list):
                    vectors = [None] * len(values_list)
                for val, vec in zip(values_list, vectors):
                    if vec is None:
                        continue
                    arr = np.asarray(vec, dtype=np.float64).ravel()
                    if arr.size != self._vector_dim:
                        fixed = np.zeros((self._vector_dim,), dtype=np.float64)
                        copy_n = min(arr.size, self._vector_dim)
                        if copy_n > 0:
                            fixed[:copy_n] = arr[:copy_n]
                        arr = fixed
                    recent_values.append(val)
                    recent_window_idx.append(win_idx)
                    recent_vectors.append(arr)

        if not recent_values:
            return None

        recent_values = np.asarray(recent_values, dtype=np.float64)
        recent_window_idx = np.asarray(recent_window_idx, dtype=np.int64)
        if entry_vectors is not None:
            recent_vectors = np.asarray(recent_vectors, dtype=np.float64)
            if recent_vectors.ndim != 2:
                recent_vectors = np.atleast_2d(recent_vectors)
        else:
            recent_vectors = None

        result = (
            values,
            value_window_idx,
            recent_values,
            recent_window_idx,
            win_sid_idx,
            win_time,
            win_w,
            entry_vectors,
            recent_vectors,
            win_sid,
        )
        self._candidate_arrays_cache = result
        self._candidate_arrays_cache_version = cache_version
        return result

    def _increment_candidates(self, freq_pairs, candidates, dist_pairs=None, numeric_rows=True):
        if not self._recent_window_ids:
            return
        tau = self._candidate_search_tau()
        if tau is None:
            return
        tau = float(tau)
        if tau < 0.0:
            return

        blocked_ready = (
            self._candidate_backend in _BLOCKED_INDEX_BACKENDS
            and self._blocked_index is not None
            and self._recent_entry_ids
            and len(self._win_sid) > 0
            and len(self._win_time) == len(self._win_sid)
            and len(self._win_w) == len(self._win_sid)
            and len(self._win_sid_idx) == len(self._win_sid)
        )

        if blocked_ready:
            if numeric_rows and dist_pairs is None:
                row_finder = None
                row_args = (np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64), float(tau))
                if self.candidate_similarity == "cosine":
                    row_args = (
                        np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64),
                        float(self.candidate_cosine_threshold),
                        float(tau),
                    )
                    if self.neg_corr and hasattr(self._blocked_index, "find_pair_rows_full_cosine_signed"):
                        row_finder = self._blocked_index.find_pair_rows_full_cosine_signed
                    elif hasattr(self._blocked_index, "find_pair_rows_full_cosine"):
                        row_finder = self._blocked_index.find_pair_rows_full_cosine
                if row_finder is None and self.neg_corr and hasattr(self._blocked_index, "find_pair_rows_full_signed"):
                    row_finder = self._blocked_index.find_pair_rows_full_signed
                    row_args = (np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64), float(tau))
                elif row_finder is None and hasattr(self._blocked_index, "find_pair_rows_full"):
                    row_finder = self._blocked_index.find_pair_rows_full
                    row_args = (np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64), float(tau))
                if row_finder is not None:
                    t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
                    self._candidate_numeric_rows = row_finder(*row_args)
                    if t0 is not None:
                        metric = "cand.blocked_numeric_rows"
                        self._profile_add(metric, time.perf_counter() - t0)
                    self._profile_candidate_search_stats()
                    return
            if dist_pairs is None:
                row_finder = None
                row_args = (np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64), float(tau))
                if self.candidate_similarity == "cosine":
                    row_args = (
                        np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64),
                        float(self.candidate_cosine_threshold),
                        float(tau),
                    )
                    if self.neg_corr and hasattr(self._blocked_index, "find_pair_rows_full_cosine_signed"):
                        row_finder = self._blocked_index.find_pair_rows_full_cosine_signed
                    elif hasattr(self._blocked_index, "find_pair_rows_full_cosine"):
                        row_finder = self._blocked_index.find_pair_rows_full_cosine
                if row_finder is None and self.neg_corr and hasattr(self._blocked_index, "find_pair_rows_full_signed"):
                    row_finder = self._blocked_index.find_pair_rows_full_signed
                    row_args = (np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64), float(tau))
                elif row_finder is None and hasattr(self._blocked_index, "find_pair_rows_full"):
                    row_finder = self._blocked_index.find_pair_rows_full
                    row_args = (np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64), float(tau))
                if row_finder is not None:
                    rows = row_finder(*row_args)
                    self._profile_candidate_search_stats()
                    self._rows_to_global_maps(rows, freq_pairs, candidates)
                    return
            return

        if self._candidate_backend in _BLOCKED_INDEX_BACKENDS:
            return

        # (2026-07-06) InstinctIndex -- experimental approximate graph
        # backend. Same dispatch shape as blocked_ready above (this
        # backend's insert_many/find_pair_rows_full_cosine[_signed]
        # interface matches BucketedMultiIndex's exactly). See
        # docs/implementation_log.md, "InstinctIndex: experimental
        # approximate graph backend".
        instinct_ready = (
            self._candidate_backend in _INSTINCT_INDEX_BACKENDS
            and self._instinct_index is not None
            and self.candidate_similarity == "cosine"
            and self.candidate_cosine_threshold is not None
            and self._recent_entry_ids
            and len(self._win_sid) > 0
            and len(self._win_time) == len(self._win_sid)
            and len(self._win_w) == len(self._win_sid)
            and len(self._win_sid_idx) == len(self._win_sid)
        )

        if instinct_ready:
            row_finder = None
            row_args = (
                np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64),
                float(self.candidate_cosine_threshold),
                float(tau),
            )
            if self.neg_corr and hasattr(self._instinct_index, "find_pair_rows_full_cosine_signed"):
                row_finder = self._instinct_index.find_pair_rows_full_cosine_signed
            elif hasattr(self._instinct_index, "find_pair_rows_full_cosine"):
                row_finder = self._instinct_index.find_pair_rows_full_cosine
            if row_finder is not None:
                if numeric_rows and dist_pairs is None:
                    t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
                    self._candidate_numeric_rows = row_finder(*row_args)
                    if t0 is not None:
                        self._profile_add("cand.instinct_numeric_rows", time.perf_counter() - t0)
                    self._profile_candidate_search_stats()
                    return
                if dist_pairs is None:
                    rows = row_finder(*row_args)
                    self._profile_candidate_search_stats()
                    self._rows_to_global_maps(rows, freq_pairs, candidates)
                    return
            return

        if self._candidate_backend in _INSTINCT_INDEX_BACKENDS:
            return

        tree_ready = (
            self._candidate_backend == "bptree"
            and self._tree_index is not None
            and self._recent_entry_ids
            and len(self._win_sid) > 0
            and len(self._win_time) == len(self._win_sid)
            and len(self._win_w) == len(self._win_sid)
            and len(self._win_sid_idx) == len(self._win_sid)
        )

        if tree_ready:
            with_dist = dist_pairs is not None
            if numeric_rows and not with_dist:
                row_finder = None
                recent_entry_ids = None
                row_args = None
                if self.candidate_similarity == "cosine":
                    if self._use_single_index_neg_corr() and hasattr(self._tree_index, "find_pair_rows_full_cosine_signed"):
                        row_finder = self._tree_index.find_pair_rows_full_cosine_signed
                        recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    elif self._vector_match_enabled and hasattr(self._tree_index, "find_pair_rows_full_cosine"):
                        row_finder = self._tree_index.find_pair_rows_full_cosine
                        recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    if row_finder is not None:
                        row_args = (float(self.candidate_cosine_threshold), float(tau))
                if row_finder is None and self._use_single_index_neg_corr() and hasattr(self._tree_index, "find_pair_rows_full_signed"):
                    row_finder = self._tree_index.find_pair_rows_full_signed
                    recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    row_args = (float(tau),)
                elif row_finder is None and self._use_single_index_neg_corr() and hasattr(self._tree_index, "find_recent_pair_rows_full_signed"):
                    row_finder = self._tree_index.find_recent_pair_rows_full_signed
                    row_args = (float(tau),)
                elif row_finder is None and self._vector_match_enabled and hasattr(self._tree_index, "find_pair_rows_full"):
                    row_finder = self._tree_index.find_pair_rows_full
                    recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    row_args = (float(tau),)
                elif row_finder is None and self._vector_match_enabled and hasattr(self._tree_index, "find_recent_pair_rows_full"):
                    row_finder = self._tree_index.find_recent_pair_rows_full
                    row_args = (float(tau),)
                elif row_finder is None and (not self._vector_match_enabled) and hasattr(self._tree_index, "find_pair_rows"):
                    row_finder = self._tree_index.find_pair_rows
                    recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    row_args = (float(tau),)
                elif row_finder is None and (not self._vector_match_enabled) and hasattr(self._tree_index, "find_recent_pair_rows"):
                    row_finder = self._tree_index.find_recent_pair_rows
                    row_args = (float(tau),)
                if row_finder is not None:
                    t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
                    if recent_entry_ids is None:
                        self._candidate_numeric_rows = row_finder(*(row_args or (float(tau),)))
                    else:
                        self._candidate_numeric_rows = row_finder(recent_entry_ids, *(row_args or (float(tau),)))
                    if t0 is not None:
                        self._profile_add("cand.tree_numeric_rows", time.perf_counter() - t0)
                    self._profile_candidate_search_stats()
                    return
            if self._use_single_index_neg_corr() and not with_dist:
                row_finder = None
                recent_entry_ids = None
                row_args = None
                if self.candidate_similarity == "cosine" and hasattr(self._tree_index, "find_pair_rows_full_cosine_signed"):
                    row_finder = self._tree_index.find_pair_rows_full_cosine_signed
                    recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    row_args = (float(self.candidate_cosine_threshold), float(tau))
                elif hasattr(self._tree_index, "find_pair_rows_full_signed"):
                    row_finder = self._tree_index.find_pair_rows_full_signed
                    recent_entry_ids = np.ascontiguousarray(self._recent_entry_ids, dtype=np.int64)
                    row_args = (float(tau),)
                elif hasattr(self._tree_index, "find_recent_pair_rows_full_signed"):
                    row_finder = self._tree_index.find_recent_pair_rows_full_signed
                    row_args = (float(tau),)
                if row_finder is not None:
                    rows = row_finder(*(row_args or (float(tau),))) if recent_entry_ids is None else row_finder(recent_entry_ids, *(row_args or (float(tau),)))
                    self._profile_candidate_search_stats()
                    self._rows_to_global_maps(rows, freq_pairs, candidates)
                    return

            pairs = None
            if self._vector_match_enabled and with_dist and hasattr(self._tree_index, "find_recent_pairs_full_with_dist"):
                pairs = self._tree_index.find_recent_pairs_full_with_dist(float(tau))
            elif (not self._vector_match_enabled) and with_dist and hasattr(self._tree_index, "find_recent_pairs_with_dist"):
                pairs = self._tree_index.find_recent_pairs_with_dist(float(tau))
            elif self._vector_match_enabled and hasattr(self._tree_index, "find_recent_pairs_full"):
                pairs = self._tree_index.find_recent_pairs_full(float(tau))
            elif (not self._vector_match_enabled) and hasattr(self._tree_index, "find_recent_pairs"):
                pairs = self._tree_index.find_recent_pairs(float(tau))
            else:
                recent_entry_ids = np.asarray(self._recent_entry_ids, dtype=np.int64)
                win_sid_idx = np.asarray(self._win_sid_idx, dtype=np.int64)
                win_time = np.asarray(self._win_time, dtype=np.int64)
                if self._vector_match_enabled and with_dist and hasattr(self._tree_index, "find_pairs_full_with_dist"):
                    pairs = self._tree_index.find_pairs_full_with_dist(
                        recent_entry_ids,
                        win_sid_idx,
                        win_time,
                        float(tau),
                    )
                elif (not self._vector_match_enabled) and with_dist and hasattr(self._tree_index, "find_pairs_with_dist"):
                    pairs = self._tree_index.find_pairs_with_dist(
                        recent_entry_ids,
                        win_sid_idx,
                        win_time,
                        float(tau),
                    )
                elif self._vector_match_enabled:
                    pairs = self._tree_index.find_pairs_full(
                        recent_entry_ids,
                        win_sid_idx,
                        win_time,
                        float(tau),
                    )
                else:
                    pairs = self._tree_index.find_pairs(
                        recent_entry_ids,
                        win_sid_idx,
                        win_time,
                        float(tau),
                    )
            if pairs:
                self._profile_candidate_search_stats()
                seen_pairs = set()
                for item in pairs:
                    if len(item) >= 3:
                        ridx, other_idx, dist_sq = item[:3]
                    else:
                        ridx, other_idx = item[:2]
                        dist_sq = self._window_distance_sq_by_indices(ridx, other_idx) if dist_pairs is not None else None
                    sid1 = self._win_sid[ridx]
                    sid2 = self._win_sid[other_idx]
                    t1 = self._win_time[ridx]
                    t2 = self._win_time[other_idx]
                    w = self._win_w[ridx]
                    if int(w) != int(self._win_w[other_idx]):
                        continue
                    pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
                    self._record_candidate_hit(pair_id, freq_pairs, candidates, seen_pairs, dist_pairs, dist_sq)
                return
            return

        if self._candidate_backend == "bptree":
            return

        cython_ready = self._candidate_backend == "flat" and self.candidate_similarity != "cosine" and (
            (
                _cy_find_candidate_pairs is not None
                and not self._vector_match_enabled
            ) or (
                _cy_find_candidate_pairs_full is not None
                and self._vector_match_enabled
            )
        )

        if cython_ready:
            arrays = self._build_candidate_arrays()
            if arrays is not None:
                (
                    values,
                    value_window_idx,
                    recent_values,
                    recent_window_idx,
                    win_sid_idx,
                    win_time,
                    win_w,
                    entry_vectors,
                    recent_vectors,
                    win_sid,
                ) = arrays
                with_dist = dist_pairs is not None
                if self._vector_match_enabled and with_dist and _cy_find_candidate_pairs_full_with_dist is not None:
                    pairs = _cy_find_candidate_pairs_full_with_dist(
                        values,
                        value_window_idx,
                        recent_values,
                        recent_window_idx,
                        win_sid_idx,
                        win_time,
                        entry_vectors,
                        recent_vectors,
                        float(tau),
                    )
                elif (not self._vector_match_enabled) and with_dist and _cy_find_candidate_pairs_with_dist is not None:
                    pairs = _cy_find_candidate_pairs_with_dist(
                        values,
                        value_window_idx,
                        recent_values,
                        recent_window_idx,
                        win_sid_idx,
                        win_time,
                        float(tau),
                    )
                elif self._vector_match_enabled:
                    pairs = _cy_find_candidate_pairs_full(
                        values,
                        value_window_idx,
                        recent_values,
                        recent_window_idx,
                        win_sid_idx,
                        win_time,
                        entry_vectors,
                        recent_vectors,
                        float(tau),
                    )
                else:
                    pairs = _cy_find_candidate_pairs(
                        values,
                        value_window_idx,
                        recent_values,
                        recent_window_idx,
                        win_sid_idx,
                        win_time,
                        float(tau),
                    )
                if pairs:
                    seen_pairs = set()
                    for item in pairs:
                        if len(item) >= 3:
                            ridx, other_idx, dist_sq = item[:3]
                        else:
                            ridx, other_idx = item[:2]
                            dist_sq = None
                        sid1 = win_sid[ridx]
                        sid2 = win_sid[other_idx]
                        t1 = win_time[ridx]
                        t2 = win_time[other_idx]
                        w = win_w[ridx]
                        if len(item) < 3 and dist_pairs is not None:
                            dist_sq = self._window_distance_sq_by_keys(
                                (sid1, int(t1), int(w)),
                                (sid2, int(t2), int(w)),
                            )
                        pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
                        self._record_candidate_hit(pair_id, freq_pairs, candidates, seen_pairs, dist_pairs, dist_sq)
                return

        tau_sq = tau * tau
        gamma = float(getattr(self, "candidate_cosine_threshold", 0.0) or 0.0)
        use_cosine = bool(self._vector_match_enabled and self.candidate_similarity == "cosine")
        seen_pairs = set()
        for window_id in self._recent_window_ids:
            values = self._reverse_index.get(window_id, ())
            if self._vector_match_enabled:
                vectors = self._reverse_vectors.get(window_id, ())
                if len(vectors) != len(values):
                    vectors = [None] * len(values)
                for value, vec in zip(values, vectors):
                    if vec is None:
                        continue
                    lower = value - tau
                    upper = value + tau
                    for other_id, other_vec in self._range_search_entries(lower, upper):
                        if other_id == window_id:
                            continue
                        if window_id[0] == other_id[0] and window_id[1] == other_id[1]:
                            continue
                        if other_vec is None:
                            continue
                        if use_cosine:
                            dot = float(np.dot(vec, other_vec))
                            if dot < gamma:
                                continue
                            dist_sq = max(0.0, 2.0 - 2.0 * dot) if dist_pairs is not None else None
                        else:
                            diff = vec - other_vec
                            dist_sq = float(np.dot(diff, diff))
                            if dist_sq > tau_sq:
                                continue
                        pair_id = self._normalize_key((window_id[0], other_id[0], window_id[1], other_id[1], window_id[2]))
                        self._record_candidate_hit(pair_id, freq_pairs, candidates, seen_pairs, dist_pairs, dist_sq)
            else:
                for value in set(values):
                    lower = value - tau
                    upper = value + tau
                    for other_id, other_value in self._range_search_scalar_entries(lower, upper):
                        if other_id == window_id:
                            continue
                        if window_id[0] == other_id[0] and window_id[1] == other_id[1]:
                            continue
                        diff = float(value) - float(other_value)
                        dist_sq = diff * diff
                        pair_id = self._normalize_key((window_id[0], other_id[0], window_id[1], other_id[1], window_id[2]))
                        self._record_candidate_hit(pair_id, freq_pairs, candidates, seen_pairs, dist_pairs, dist_sq)

    def run(self, n_ids, verbose=True, testing=False, worker_mode=None, max_workers=1, update_grid=True, executor=None, numeric_rows=True):
        self.verbose = verbose
        self.testing = testing
        self._candidate_numeric_rows = None

        if update_grid:
            t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
            self._update_grid(n_ids)
            if t0 is not None:
                self._profile_add("cand.update_grid_total", time.perf_counter() - t0)

        freq_pairs = {}
        candidates = {}
        uncorrelated = {}
        dist_pairs = {} if getattr(self, "return_distances", False) else None

        if len(self.partition)>0:
            t0 = time.perf_counter() if getattr(self, "_profile_callback", None) is not None else None
            sharded = False
            if worker_mode == "thread" and not numeric_rows:
                sharded = self._increment_candidates_sharded(
                    freq_pairs,
                    candidates,
                    dist_pairs,
                    max_workers=max_workers,
                    executor=executor,
                )
            if not sharded:
                self._increment_candidates(freq_pairs,candidates,dist_pairs,numeric_rows=bool(numeric_rows))
            if t0 is not None:
                self._profile_add("cand.increment_total", time.perf_counter() - t0)

        self.candidate_dist_sq = dist_pairs or {}

        return freq_pairs, candidates, uncorrelated


class CorrTrack_optimize:
    def __init__(
        self,
        train_data,
        ids,
        window_size,
        window_step,
        n_lags,
        corr_threshold,
        recall_by_window,
        alg,
        neg_corr,
        corr_val,
        exec="parallel",
        max_workers=0,
        sketch_norm="z",
        candidate_bucket_width=None,
        candidate_block_size_steps=None,
        candidate_block_index_dims=None,
        candidate_similarity="l2",
        candidate_cosine_threshold=None,
        candidate_parallel_mode="recent_shards",
        candidate_key_mode="first",
        candidate_key_seed=None,
        candidate_lsh_radius=None,
        candidate_ann_m=None,
        candidate_ann_z=None,
        candidate_ann_ef=None,
        candidate_bound_dims=None,
        candidate_bound_dim_selection="variance",
        enable_block_ub_pruning=False,
        enable_row_ub_pruning=False,
        block_similarity_assignment=False,
        max_open_blocks=4,
        candidate_instinct_query_mode="hybrid",
        candidate_instinct_top_k=256,
        candidate_instinct_min_candidates=64,
        candidate_instinct_entry_points=8,
        hybrid_validation=False,
        hybrid_validation_min_repeat_rate=0.25,
        hybrid_validation_disable_rate=None,
        hybrid_validation_ema_alpha=0.25,
        hybrid_validation_min_candidates=256,
        numeric_rows=True,
        verbose=False,
        testing=False,
        parallel_sketch=None,
        parallel_candidates=None,
        parallel_validation=None,
        track_min_dist=False,
        artifact_mode="buffered",
        artifact_buffer_max_rows=250000,
        artifact_merge_mode="merged",
        save_only_required_artifacts=False,
        save_maxlag_artifacts=True,
        proxy_config=None,
    ):

        self.neg_corr = neg_corr
        self.corr_val = False
        self.sketch_norm = sketch_norm or "z"
        self.candidate_bucket_width = _to_float_safe(candidate_bucket_width)
        block_steps = _to_int_safe(candidate_block_size_steps)
        if block_steps is None or block_steps <= 0:
            block_steps = 32
        self.candidate_block_size_steps = int(block_steps)
        block_dims = _to_int_safe(candidate_block_index_dims)
        if block_dims is None or block_dims <= 0:
            block_dims = 1
        self.candidate_block_index_dims = int(block_dims)
        # (2026-07-06) Part 1 -- see docs/implementation_log.md.
        bound_dims_val = _to_int_safe(candidate_bound_dims)
        self.candidate_bound_dims = max(0, bound_dims_val) if bound_dims_val is not None else 0
        self.candidate_bound_dim_selection = str(candidate_bound_dim_selection or "variance").lower()
        self.enable_block_ub_pruning = bool(enable_block_ub_pruning)
        self.enable_row_ub_pruning = bool(enable_row_ub_pruning)
        self.block_similarity_assignment = bool(block_similarity_assignment)
        self.max_open_blocks = max(1, _to_int_safe(max_open_blocks) or 4)
        # (2026-07-06) InstinctIndex candidate_backend params -- see
        # docs/implementation_log.md. Approximate, opt-in backend only.
        self.candidate_instinct_query_mode = str(candidate_instinct_query_mode or "hybrid").lower()
        self.candidate_instinct_top_k = max(1, _to_int_safe(candidate_instinct_top_k) or 256)
        self.candidate_instinct_min_candidates = max(1, _to_int_safe(candidate_instinct_min_candidates) or 64)
        self.candidate_instinct_entry_points = max(1, _to_int_safe(candidate_instinct_entry_points) or 8)
        self.candidate_similarity = _resolve_candidate_similarity(candidate_similarity, default="l2")
        self.candidate_cosine_threshold = _to_float_safe(candidate_cosine_threshold)
        self.candidate_parallel_mode = "recent_shards"
        self.candidate_key_mode = _resolve_candidate_key_mode(candidate_key_mode, default="first")
        self.candidate_key_seed = _to_int_safe(candidate_key_seed)
        self.candidate_lsh_radius = max(0, min(3, int(_to_int_safe(candidate_lsh_radius) or 0)))
        self.candidate_ann_m = max(1, int(_to_int_safe(candidate_ann_m) or 16))
        self.candidate_ann_z = max(1, int(_to_int_safe(candidate_ann_z) or 256))
        self.candidate_ann_ef = max(1, int(_to_int_safe(candidate_ann_ef) or max(64, self.candidate_ann_z)))
        hybrid_kwargs = _resolve_hybrid_validation_kwargs({
            "hybrid_validation": hybrid_validation,
            "hybrid_validation_min_repeat_rate": hybrid_validation_min_repeat_rate,
            "hybrid_validation_disable_rate": hybrid_validation_disable_rate,
            "hybrid_validation_ema_alpha": hybrid_validation_ema_alpha,
            "hybrid_validation_min_candidates": hybrid_validation_min_candidates,
        })
        self.hybrid_validation = hybrid_kwargs["hybrid_validation"]
        self.hybrid_validation_min_repeat_rate = hybrid_kwargs["hybrid_validation_min_repeat_rate"]
        self.hybrid_validation_disable_rate = hybrid_kwargs["hybrid_validation_disable_rate"]
        self.hybrid_validation_ema_alpha = hybrid_kwargs["hybrid_validation_ema_alpha"]
        self.hybrid_validation_min_candidates = hybrid_kwargs["hybrid_validation_min_candidates"]
        self.numeric_rows = True
        self.verbose = _coerce_to_bool(verbose)
        self.testing = _coerce_to_bool(testing)
        # Hyperopt always runs with min-distance tracking disabled.
        self.track_min_dist = False

        # Parameters data
        self.train_data = train_data
        self.ids = ids

        self.window_size = window_size

        self.n_lags = n_lags
        self.corr_threshold = corr_threshold

        self.ground_truth = None
        self.recall_by_window = True

        # Policy describes how the *inner* CorrTrack should run
        # (use your globals by default)
        self.exec = _normalize_exec_mode(exec, default="thread")
        self.max_workers = max_workers
        parallel_default = self.exec == "thread"
        self.parallel_sketch = _resolve_parallel_flag(parallel_sketch, False)
        self.parallel_candidates = _resolve_parallel_flag(parallel_candidates, False)
        self.parallel_validation = _resolve_parallel_flag(parallel_validation, parallel_default)
        artifact_mode_value = (artifact_mode or "buffered").lower()
        if artifact_mode_value not in {"iterative", "final", "buffered"}:
            artifact_mode_value = "buffered"
        self.artifact_mode = artifact_mode_value
        self.artifact_merge_mode = _resolve_artifact_merge_mode(artifact_merge_mode, default="merged")
        try:
            self.artifact_buffer_max_rows = max(1, int(artifact_buffer_max_rows))
        except (TypeError, ValueError):
            self.artifact_buffer_max_rows = 250000
        self.save_only_required_artifacts = _coerce_to_bool(
            save_only_required_artifacts,
            default=False,
        )
        self.save_maxlag_artifacts = _coerce_to_bool(
            save_maxlag_artifacts,
            default=True,
        )
        self.corrtrack_bf = CorrTrack(window_size=self.window_size,basic_window=None,window_step=window_step,n_vectors=1,n_lags=self.n_lags,
                                    grid_dimension=1,cell_size=1,seed=None,seed_toggle=None,corr_threshold=self.corr_threshold,
                                    neg_corr=self.neg_corr,preprocess=False,exec=self.exec,max_workers=self.max_workers,
                                    sketch_norm=self.sketch_norm,candidate_parallel_mode=self.candidate_parallel_mode,
                                    candidate_bucket_width=self.candidate_bucket_width,
                                    candidate_block_size_steps=self.candidate_block_size_steps,
                                    candidate_block_index_dims=self.candidate_block_index_dims,
                                    candidate_similarity=self.candidate_similarity,
                                    candidate_cosine_threshold=self.candidate_cosine_threshold,
                                    candidate_lsh_radius=self.candidate_lsh_radius,
                                    candidate_ann_m=self.candidate_ann_m,
                                    candidate_ann_z=self.candidate_ann_z,
                                    candidate_ann_ef=self.candidate_ann_ef,
                                    parallel_sketch=self.parallel_sketch,parallel_candidates=self.parallel_candidates,parallel_validation=self.parallel_validation,
                                    track_min_dist=self.track_min_dist,numeric_rows=self.numeric_rows)
        
        self.window_step = self.corrtrack_bf.window_step
        self.basic_window = self.corrtrack_bf.basic_window

        self.alg = alg
        # Stable release: proxy-anchor is the only hyperopt implementation.
        self.hyperopt_strategy = "proxy_anchor"
        self.tuning_mode = "sampling"
        self.proxy_config = dict(proxy_config or {})
        self.bootstrap_validation_enabled = False
        self.bootstrap_repeats = 0
        self.bootstrap_confidence_level = 0.90
        self.bootstrap_min_gt_events = 0
        self.bootstrap_random_seed = None
        self.proxy_anchor_count_requested = max(
            1,
            _to_int_safe(self.proxy_config.get("anchor_count")) or 64,
        )
        self.proxy_max_pair_rows = max(
            1,
            _to_int_safe(self.proxy_config.get("max_pair_rows")) or 250000,
        )
        self.proxy_random_seed = _to_int_safe(self.proxy_config.get("random_seed"))
        if self.proxy_random_seed is None:
            self.proxy_random_seed = self.bootstrap_random_seed
        if self.proxy_random_seed is None:
            self.proxy_random_seed = 2468
        self.proxy_bootstrap_enabled = _coerce_to_bool(
            self.proxy_config.get("bootstrap_enabled"),
            default=True,
        )
        self.proxy_bootstrap_repeats = max(
            0,
            _to_int_safe(self.proxy_config.get("bootstrap_repeats")) or self.bootstrap_repeats or 1000,
        )
        try:
            self.proxy_bootstrap_confidence_level = float(
                self.proxy_config.get("bootstrap_confidence_level", self.bootstrap_confidence_level)
            )
        except (TypeError, ValueError):
            self.proxy_bootstrap_confidence_level = 0.90
        if not np.isfinite(self.proxy_bootstrap_confidence_level):
            self.proxy_bootstrap_confidence_level = 0.90
        self.proxy_bootstrap_confidence_level = min(
            max(self.proxy_bootstrap_confidence_level, 0.0),
            0.999999,
        )
        self.proxy_bootstrap_min_gt_events = max(
            0,
            _to_int_safe(self.proxy_config.get("bootstrap_min_gt_events")) or self.bootstrap_min_gt_events or 30,
        )
        self.proxy_eval_mode = "backend_search"
        self.proxy_distance_cache_max_rows = max(
            0,
            _to_int_safe(self.proxy_config.get("distance_cache_max_rows")) or 250000,
        )
        self.proxy_adaptive_anchor_enabled = _coerce_to_bool(
            self.proxy_config.get("adaptive_anchor_enabled"),
            default=True,
        )
        self.proxy_anchor_expand_factor = _to_float_safe(
            self.proxy_config.get("anchor_expand_factor")
        )
        if self.proxy_anchor_expand_factor is None or self.proxy_anchor_expand_factor <= 1.0:
            self.proxy_anchor_expand_factor = 2.0
        self.proxy_anchor_expand_max_rounds = max(
            0,
            _to_int_safe(self.proxy_config.get("anchor_expand_max_rounds")) or 1,
        )
        self.proxy_max_anchor_count = max(
            self.proxy_anchor_count_requested,
            _to_int_safe(self.proxy_config.get("max_anchor_count"))
            or int(math.ceil(self.proxy_anchor_count_requested * self.proxy_anchor_expand_factor)),
        )
        # (2026-07-05) Real-time selection tie-break: sk_time/cand_time are
        # genuinely measured per proxy trial, but at these scales (single-
        # to double-digit ms) a single measurement is noisy. Re-run each
        # trial's full sketch+candidate-search this many times and keep the
        # mean for the timing fields (all other fields -- recall, precision,
        # stats -- are deterministic given the same seed, so only need one
        # copy). See docs/implementation_log.md, "hyperopt real-time
        # tie-break".
        self.proxy_timing_repeats = max(
            1,
            _to_int_safe(self.proxy_config.get("timing_repeats")) or 3,
        )
        # Relative tolerance defining "reasonably close" to the best
        # achievable candidate rate among feasible configs: within this
        # fraction, real measured execution time breaks the tie instead of
        # further candidate-count refinements. Candidate-count minimization
        # remains the dominant objective outside this tolerance.
        self.proxy_candidate_rate_close_tolerance = max(
            0.0,
            _to_float_safe(self.proxy_config.get("candidate_rate_close_tolerance")) or 0.05,
        )
        self._proxy_anchor_expansion_iteration = 0
        self._proxy_anchor_expansion_reason = ""
        self._proxy_reference = None

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _run_is_parallel(self) -> bool:
        return self.exec != "sequential"

    def _sampling_enabled(self) -> bool:
        return False

    def _bootstrap_enabled(self) -> bool:
        return bool(
            self.bootstrap_validation_enabled
            and self._sampling_enabled()
            and self.recall_by_window
            and self.bootstrap_repeats > 0
        )

    def _proxy_anchor_enabled(self) -> bool:
        return self.hyperopt_strategy == "proxy_anchor"

    def _proxy_bootstrap_active(self) -> bool:
        return bool(
            self._proxy_anchor_enabled()
            and self.proxy_bootstrap_enabled
            and self.proxy_bootstrap_repeats > 0
        )

    @staticmethod
    def _sampling_plan_fields(plan):
        return {
            "tuning_mode": plan.get("tuning_mode"),
            "sampling_design_id": plan.get("sampling_design_id"),
            "sampling_profile": plan.get("sampling_profile"),
            "sampling_profile_notes": plan.get("sampling_profile_notes"),
            "sampling_requested_block_q": plan.get("sampling_requested_block_q"),
            "sampling_requested_n_blocks": plan.get("sampling_requested_n_blocks"),
            "sampling_requested_placement_mode": plan.get("sampling_requested_placement_mode"),
            "sampling_requested_random_seed": plan.get("sampling_requested_random_seed"),
            "sampling_requested_allow_block_overlap": plan.get("sampling_requested_allow_block_overlap"),
            "sampling_n_sampled_blocks": plan.get("sampling_n_sampled_blocks"),
            "sampling_block_windows": plan.get("sampling_block_windows"),
            "sampling_block_span": plan.get("sampling_block_span"),
            "sampling_total_sampled_timestamps": plan.get("sampling_total_sampled_timestamps"),
            "sampling_coverage_train": plan.get("sampling_coverage_train"),
            "sampling_warning": plan.get("sampling_warning"),
        }

    def _init_optim_record(self, dataset_id, param_combo, sampling_plan=None):
        param_combo = dict(param_combo or {})
        record = {key: None for key in OPTIM_RESULT_COLUMNS}

        record["dataset_id"] = dataset_id
        record["alg"] = self.alg
        record["hyperopt_strategy"] = self.hyperopt_strategy
        effective_plan = sampling_plan or {"enabled": False, "tuning_mode": self.tuning_mode}
        record.update(self._sampling_plan_fields(effective_plan))
        record["tuning_mode"] = record.get("tuning_mode") or self.tuning_mode
        record["n_ts"] = self.train_data.shape[0] - 1
        if effective_plan.get("enabled"):
            n_w = 0
            for block in effective_plan.get("blocks", []):
                block_span = int(block.get("span_timestamps", 0) or 0)
                if block_span < self.window_size:
                    continue
                n_w += int(np.floor((block_span - self.window_size) / self.window_step) + 1)
        else:
            n_w = int(np.floor((self.train_data.shape[1] - self.window_size) / self.window_step) + 1)
        record["n_w"] = n_w
        record["total_w"] = record["n_ts"] * n_w
        record["mem_w"] = int(self.n_lags // self.window_step) if self.window_step else None

        record["nodes"] = param_combo.get("nodes")
        record["exec_mode"] = self.exec
        record["parallel_sketch"] = self.parallel_sketch
        record["parallel_candidates"] = self.parallel_candidates
        record["parallel_validation"] = self.parallel_validation
        record["workers_total"] = None
        record["workers_sketch"] = None
        record["workers_candidates"] = None
        record["workers_validation"] = None
        record["window_size"] = self.window_size
        record["window_step"] = self.window_step
        record["basic_window"] = self.basic_window
        record["n_lags"] = self.n_lags
        record["seed"] = param_combo.get("seed")
        record["seed_toggle"] = param_combo.get("seed_toggle")
        record["preprocess"] = param_combo.get("preprocess")
        record["sketch_norm"] = param_combo.get("sketch_norm")
        record["candidate_backend"] = param_combo.get("candidate_backend")
        record["candidate_bucket_width"] = param_combo.get(
            "candidate_bucket_width",
            self.candidate_bucket_width,
        )
        record["candidate_block_size_steps"] = param_combo.get(
            "candidate_block_size_steps",
            self.candidate_block_size_steps,
        )
        record["candidate_block_index_dims"] = param_combo.get(
            "candidate_block_index_dims",
            self.candidate_block_index_dims,
        )
        # (2026-07-06) Part 1 -- see docs/implementation_log.md.
        record["candidate_bound_dims"] = param_combo.get(
            "candidate_bound_dims",
            self.candidate_bound_dims,
        )
        record["candidate_bound_dim_selection"] = param_combo.get(
            "candidate_bound_dim_selection",
            self.candidate_bound_dim_selection,
        )
        record["enable_block_ub_pruning"] = param_combo.get(
            "enable_block_ub_pruning",
            self.enable_block_ub_pruning,
        )
        record["enable_row_ub_pruning"] = param_combo.get(
            "enable_row_ub_pruning",
            self.enable_row_ub_pruning,
        )
        record["block_similarity_assignment"] = param_combo.get(
            "block_similarity_assignment",
            self.block_similarity_assignment,
        )
        record["max_open_blocks"] = param_combo.get(
            "max_open_blocks",
            self.max_open_blocks,
        )
        record["candidate_instinct_query_mode"] = param_combo.get(
            "candidate_instinct_query_mode",
            self.candidate_instinct_query_mode,
        )
        record["candidate_instinct_top_k"] = param_combo.get(
            "candidate_instinct_top_k",
            self.candidate_instinct_top_k,
        )
        record["candidate_instinct_min_candidates"] = param_combo.get(
            "candidate_instinct_min_candidates",
            self.candidate_instinct_min_candidates,
        )
        record["candidate_instinct_entry_points"] = param_combo.get(
            "candidate_instinct_entry_points",
            self.candidate_instinct_entry_points,
        )
        record["candidate_similarity"] = param_combo.get(
            "candidate_similarity",
            getattr(self, "candidate_similarity", "l2"),
        )
        record["candidate_cosine_threshold"] = param_combo.get(
            "candidate_cosine_threshold",
            getattr(self, "candidate_cosine_threshold", None),
        )
        record["candidate_parallel_mode"] = param_combo.get("candidate_parallel_mode", self.candidate_parallel_mode)
        record["candidate_key_mode"] = param_combo.get("candidate_key_mode", self.candidate_key_mode)
        record["candidate_key_seed"] = param_combo.get("candidate_key_seed", self.candidate_key_seed)
        record["candidate_lsh_radius"] = param_combo.get("candidate_lsh_radius", getattr(self, "candidate_lsh_radius", 0))
        record["candidate_ann_m"] = param_combo.get("candidate_ann_m", getattr(self, "candidate_ann_m", 16))
        record["candidate_ann_z"] = param_combo.get("candidate_ann_z", getattr(self, "candidate_ann_z", 256))
        record["candidate_ann_ef"] = param_combo.get("candidate_ann_ef", getattr(self, "candidate_ann_ef", 256))
        record["numeric_rows"] = param_combo.get("numeric_rows", self.numeric_rows)
        record.update(_resolve_hybrid_validation_kwargs({
            "hybrid_validation": getattr(self, "hybrid_validation", False),
            "hybrid_validation_min_repeat_rate": getattr(self, "hybrid_validation_min_repeat_rate", 0.25),
            "hybrid_validation_disable_rate": getattr(self, "hybrid_validation_disable_rate", None),
            "hybrid_validation_ema_alpha": getattr(self, "hybrid_validation_ema_alpha", 0.25),
            "hybrid_validation_min_candidates": getattr(self, "hybrid_validation_min_candidates", 256),
        }, param_combo))
        record["hybrid_validation_attempts"] = 0
        record["hybrid_validation_hits"] = 0
        record["hybrid_validation_steps_active"] = 0
        record["corr_threshold"] = self.corr_threshold
        record["grid_max"] = param_combo.get("grid_max")
        record["cell_stretch"] = param_combo.get("cell_size")
        record["cell_size"] = None
        record["n_vectors"] = param_combo.get("n_vectors")
        record["grid_dimension"] = param_combo.get("grid_dimension")
        record["freq_threshold"] = None

        record["cand_time_bf"] = getattr(self, "candidate_time_bf", None)
        record["val_time_bf"] = getattr(self, "validation_time_bf", None)
        record["monit_time_bf"] = getattr(self, "monitor_time_bf", None)
        record["runtime_bf"] = getattr(self, "runtime_bf", None)
        record["artifact_time_bf"] = getattr(self, "artifact_time_bf", None)
        record["corr_w_bf"] = getattr(self, "correlated_bf", None)
        record["tested_w_bf"] = getattr(self, "tested_bf", None)
        record["cand_w_bf"] = getattr(self, "total_bf", None)
        record["artifact_time"] = None
        record["optim_search_time"] = None
        record["optim_bootstrap_time"] = 0.0
        record["optim_eval_time"] = None
        record["bootstrap_enabled"] = self._bootstrap_enabled()
        record["bootstrap_repeats"] = self.bootstrap_repeats if self._bootstrap_enabled() else None
        record["bootstrap_confidence_level"] = (
            self.bootstrap_confidence_level if self._bootstrap_enabled() else None
        )
        record["bootstrap_min_gt_events"] = (
            self.bootstrap_min_gt_events if self._bootstrap_enabled() else None
        )
        record["proxy_anchor_count_requested"] = self.proxy_anchor_count_requested
        record["proxy_anchor_seed"] = self.proxy_random_seed
        record["proxy_max_pair_rows"] = self.proxy_max_pair_rows
        record["proxy_eval_mode"] = "cached_distances"
        record["proxy_distance_cache_used"] = False
        record["proxy_bootstrap_enabled"] = self._proxy_bootstrap_active()
        record["proxy_bootstrap_repeats"] = (
            self.proxy_bootstrap_repeats if self._proxy_bootstrap_active() else None
        )
        record["proxy_bootstrap_confidence_level"] = (
            self.proxy_bootstrap_confidence_level if self._proxy_bootstrap_active() else None
        )
        record["proxy_bootstrap_min_gt_events"] = (
            self.proxy_bootstrap_min_gt_events if self._proxy_bootstrap_active() else None
        )
        record["status"] = "pending"
        record["error"] = ""
        return record

    def _proxy_ids(self):
        n_series = max(0, int(self.train_data.shape[0]) - 1)
        ids = list(self.ids) if self.ids is not None else [str(i) for i in range(n_series)]
        if len(ids) < n_series:
            ids = ids + [str(i) for i in range(len(ids), n_series)]
        return [str(x) for x in ids[:n_series]]

    def _proxy_start_time(self, start_idx):
        # Proxy-anchor sampling is defined over positional starts in train_data.
        # Keep candidate keys in the same coordinate system even when the raw
        # stream timestamp row contains datetime objects.
        return int(start_idx)

    @staticmethod
    def _proxy_pair_metrics_from_counts(counts):
        tp = int(counts.get("tp", 0) or 0)
        tn = int(counts.get("tn", 0) or 0)
        fp = int(counts.get("fp", 0) or 0)
        fn = int(counts.get("fn", 0) or 0)
        total = tp + tn + fp + fn
        gt_total = tp + fn
        cand_total = tp + fp
        recall = _safe_div(tp, gt_total)
        precision = _safe_div(tp, cand_total)
        specificity = _safe_div(tn, tn + fp)
        candidate_rate = _safe_div(cand_total, total)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        return {
            "total": int(total),
            "gt_total": int(gt_total),
            "cand_total": int(cand_total),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "recall": float(recall) if recall is not None else 0.0,
            "precision": float(precision) if precision is not None else 0.0,
            "specificity": float(specificity) if specificity is not None else 0.0,
            "candidate_rate": float(candidate_rate) if candidate_rate is not None else 0.0,
            "f1": float(f1) if f1 is not None else 0.0,
        }

    @staticmethod
    def _proxy_pearson(x, y):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
            return float("nan")
        x_centered = x_arr - float(np.mean(x_arr))
        y_centered = y_arr - float(np.mean(y_arr))
        var_x = float(np.dot(x_centered, x_centered))
        var_y = float(np.dot(y_centered, y_centered))
        if var_x <= 0.0 or var_y <= 0.0:
            return float("nan")
        return float(np.dot(x_centered, y_centered) / math.sqrt(var_x * var_y))

    @staticmethod
    def _proxy_window_is_valid(window):
        arr = np.asarray(window, dtype=np.float64)
        if arr.size == 0:
            return False
        if CorrTrack.is_near_constant(arr, std_thresh=1e-3):
            return False
        try:
            if CorrTrack.is_structurally_spiked(arr, kurt_thresh=5.0):
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _normalize_window_pair_key(key):
        id1, id2, t1, t2, w = key
        id1 = str(id1)
        id2 = str(id2)
        t1 = int(t1)
        t2 = int(t2)
        w = int(w)
        if id1 == id2:
            return (id1, id2, max(t1, t2), min(t1, t2), w)
        if t1 == t2:
            a, b = sorted((id1, id2))
            return (a, b, t1, t2, w)
        if t1 < t2:
            return (id2, id1, t2, t1, w)
        return (id1, id2, t1, t2, w)

    def _prepare_proxy_anchor_reference(self):
        """Build the sampled anchor/pair truth table for proxy hyperopt.

        The sampling unit is an anchor start position. For each anchor we build
        the same candidate universe used by the streaming search: current
        windows at the anchor against historical windows inside `n_lags`, with
        same-window and reversed duplicates removed. Exact Pearson labels are
        computed once and reused for every sketch hyperparameter setting.
        """

        n_series = max(0, int(self.train_data.shape[0]) - 1)
        length_data = int(self.train_data.shape[1]) if self.train_data is not None else 0
        if n_series <= 1:
            raise ValueError("Proxy-anchor hyperopt requires at least two series")
        max_start = length_data - int(self.window_size)
        if max_start < 0:
            raise ValueError("Proxy-anchor hyperopt requires data length >= window_size")

        all_starts = list(range(0, max_start + 1, int(self.window_step)))
        if not all_starts:
            raise ValueError("No valid window starts available for proxy-anchor hyperopt")
        full_history_starts = [start for start in all_starts if start >= int(self.n_lags)]
        candidate_anchors = full_history_starts or all_starts

        lag_count = max(1, int(self.n_lags // self.window_step) + 1)
        est_pairs_per_anchor = int(n_series * (n_series - 1) / 2) + int(n_series * n_series * max(lag_count - 1, 0))
        if est_pairs_per_anchor <= 0:
            est_pairs_per_anchor = 1
        capped_anchor_count = max(1, int(self.proxy_max_pair_rows // est_pairs_per_anchor))
        anchor_count = min(
            int(self.proxy_anchor_count_requested),
            len(candidate_anchors),
            capped_anchor_count,
        )
        if anchor_count <= 0:
            anchor_count = 1

        rng = np.random.default_rng(int(self.proxy_random_seed))
        if anchor_count < len(candidate_anchors):
            anchor_starts = sorted(rng.choice(candidate_anchors, size=anchor_count, replace=False).astype(int).tolist())
        else:
            anchor_starts = sorted(int(x) for x in candidate_anchors)

        proxy_ids = self._proxy_ids()
        values = np.asarray(self.train_data[1 : 1 + n_series, :], dtype=np.float64)
        window_cache = {}
        valid_cache = {}

        def get_window(series_idx, start_idx):
            key = (int(series_idx), int(start_idx))
            cached = window_cache.get(key)
            if cached is None:
                cached = values[int(series_idx), int(start_idx) : int(start_idx) + int(self.window_size)]
                window_cache[key] = cached
            return cached

        def is_valid(series_idx, start_idx):
            key = (int(series_idx), int(start_idx))
            if key not in valid_cache:
                valid_cache[key] = self._proxy_window_is_valid(get_window(series_idx, start_idx))
            return bool(valid_cache[key])

        anchors = []
        pair_keys = []
        pair_anchor_ids = []
        truth = []
        truth_sign = []
        key_to_indices = defaultdict(list)
        unique_starts = set()
        max_rows_reached = False

        for anchor_id, anchor_start in enumerate(anchor_starts):
            min_lag_start = max(0, int(anchor_start) - int(self.n_lags))
            lag_starts = list(range(int(anchor_start), min_lag_start - 1, -int(self.window_step)))
            lag_starts = sorted(set(start for start in lag_starts if 0 <= start <= max_start))
            if not lag_starts or int(anchor_start) not in lag_starts:
                continue
            anchors.append(
                {
                    "anchor_id": int(len(anchors)),
                    "start_idx": int(anchor_start),
                    "start_time": self._proxy_start_time(anchor_start),
                    "lag_starts": [int(x) for x in lag_starts],
                }
            )
            current_anchor_id = int(anchors[-1]["anchor_id"])
            for start in lag_starts:
                unique_starts.add(int(start))

            for s_current in range(n_series):
                if not is_valid(s_current, anchor_start):
                    continue
                x = get_window(s_current, anchor_start)
                for lag_start in lag_starts:
                    for s_other in range(n_series):
                        if lag_start == anchor_start and s_other <= s_current:
                            continue
                        if not is_valid(s_other, lag_start):
                            continue
                        y = get_window(s_other, lag_start)
                        corr, _dist = _fast_corr_and_dist(x, y)
                        if not np.isfinite(corr):
                            gt = False
                            sign = 0
                        elif self.neg_corr and corr <= -float(self.corr_threshold):
                            gt = True
                            sign = -1
                        elif corr >= float(self.corr_threshold):
                            gt = True
                            sign = 1
                        else:
                            gt = False
                            sign = 0
                        pair_key = self._normalize_window_pair_key(
                            (
                                proxy_ids[s_current],
                                proxy_ids[s_other],
                                self._proxy_start_time(anchor_start),
                                self._proxy_start_time(lag_start),
                                int(self.window_size),
                            )
                        )
                        row_idx = len(pair_keys)
                        pair_keys.append(pair_key)
                        pair_anchor_ids.append(current_anchor_id)
                        truth.append(bool(gt))
                        truth_sign.append(int(sign))
                        key_to_indices[pair_key].append(row_idx)
                        if len(pair_keys) >= int(self.proxy_max_pair_rows):
                            max_rows_reached = True
                            break
                    if max_rows_reached:
                        break
                if max_rows_reached:
                    break
            if max_rows_reached:
                break

        if not pair_keys:
            raise ValueError("Proxy-anchor hyperopt produced no valid pair rows")

        if max_rows_reached:
            anchors = [a for a in anchors if a["anchor_id"] in set(pair_anchor_ids)]

        return {
            "proxy_ids": proxy_ids,
            "anchors": anchors,
            "unique_starts": sorted(unique_starts),
            "pair_keys": pair_keys,
            "pair_anchor_ids": np.asarray(pair_anchor_ids, dtype=np.int64),
            "truth": np.asarray(truth, dtype=bool),
            "truth_sign": np.asarray(truth_sign, dtype=np.int8),
            "key_to_indices": dict(key_to_indices),
            "anchor_count_requested": int(self.proxy_anchor_count_requested),
            "anchor_count": int(len(anchors)),
            "max_pair_rows": int(self.proxy_max_pair_rows),
            "n_pairs": int(len(pair_keys)),
            "n_gt": int(np.sum(truth)),
            "warning": "proxy pair row cap reached; anchors were truncated" if max_rows_reached else "",
        }

    def _proxy_partitions_for_corrtrack(self, corrtrack, reference):
        partitions_by_start = {}
        proxy_ids = reference["proxy_ids"]
        for start_idx in reference["unique_starts"]:
            sketcher = Sketches(
                self.window_size,
                corrtrack.basic_window,
                self.window_step,
                corrtrack.seed,
                corrtrack.seed_toggle,
                corrtrack.n_vectors,
                corrtrack.grid_dimension,
                [],
                corrtrack.preprocess,
                neg_corr=self.neg_corr,
                sketch_norm=corrtrack.sketch_norm,
                full_vector_candidates=corrtrack.full_vector_candidates,
            )
            if corrtrack.preprocess and int(start_idx) > 0:
                try:
                    sketcher.last_origin = np.asarray(self.train_data[1:, int(start_idx) - 1], dtype=np.float64)
                except Exception:
                    sketcher.last_origin = None
            data_step = np.array(
                self.train_data[:, int(start_idx) : int(start_idx) + int(self.window_size)],
                dtype=object,
                copy=True,
            )
            data_step[0, :] = np.arange(
                int(start_idx),
                int(start_idx) + int(data_step.shape[1]),
                dtype=np.int64,
            )
            _sketches, partitions = sketcher.run(
                data_step,
                proxy_ids,
                verbose=False,
                testing=False,
                distribute=False,
            )
            partitions_by_start[int(start_idx)] = partitions
        return partitions_by_start

    def _proxy_candidate_keys_for_anchor(self, corrtrack, partitions_by_start, anchor, feature_kwargs, return_stats=False):
        anchor_ct = CorrTrack(
            window_size=self.window_size,
            basic_window=corrtrack.basic_window,
            window_step=self.window_step,
            n_vectors=corrtrack.n_vectors,
            n_lags=self.n_lags,
            grid_dimension=corrtrack.grid_dimension,
            cell_size=corrtrack.cell_stretch,
            seed=corrtrack.seed,
            seed_toggle=corrtrack.seed_toggle,
            freq_threshold=(corrtrack.freq_threshold / corrtrack.n_grids if corrtrack.n_grids else 0),
            corr_threshold=self.corr_threshold,
            neg_corr=self.neg_corr,
            preprocess=corrtrack.preprocess,
            exec="sequential",
            max_workers=self.max_workers,
            parallel_sketch=False,
            parallel_candidates=corrtrack.parallel_candidates,
            parallel_validation=False,
            track_min_dist=False,
            **feature_kwargs,
        )
        anchor_ct.series_ids = {sid: idx for idx, sid in enumerate(self._proxy_reference["proxy_ids"])}
        lag_starts = [int(x) for x in anchor.get("lag_starts", [])]
        if not lag_starts:
            return (set(), self._empty_proxy_search_stats()) if return_stats else set()
        final_start = int(anchor.get("start_idx", lag_starts[-1]))
        for start_idx in lag_starts:
            partitions = partitions_by_start.get(int(start_idx)) or []
            for grid_index, partition in enumerate(partitions):
                if partition is None or grid_index >= len(anchor_ct.grid_nodes):
                    continue
                anchor_ct.grid_nodes[grid_index].append_partition(
                    self._proxy_start_time(start_idx),
                    partition,
                )
            if int(start_idx) == final_start:
                continue
            # Candidate nodes update their searchable index only when run/update
            # is called. Feed historical lag partitions first, then collect only
            # the final anchor-step candidates below.
            for node in anchor_ct.grid_nodes:
                node._update_grid(len(anchor_ct.series_ids))
        anchor_ct._run_grids(False, False, worker_mode="sequential")
        search_stats = self._proxy_candidate_search_stats_for_corrtrack(anchor_ct)
        numeric_rows = getattr(anchor_ct, "_candidate_numeric_rows", None)
        if numeric_rows is not None:
            rows = np.asarray(numeric_rows, dtype=np.int64).reshape((-1, 5))
            if rows.size:
                proxy_ids = list(self._proxy_reference["proxy_ids"])
                candidate_keys = set()
                for sid1_idx, sid2_idx, t1, t2, w in rows:
                    sid1_idx = int(sid1_idx)
                    sid2_idx = int(sid2_idx)
                    sid1 = proxy_ids[sid1_idx] if 0 <= sid1_idx < len(proxy_ids) else str(sid1_idx)
                    sid2 = proxy_ids[sid2_idx] if 0 <= sid2_idx < len(proxy_ids) else str(sid2_idx)
                    candidate_keys.add(
                        self._normalize_window_pair_key((sid1, sid2, int(t1), int(t2), int(w)))
                    )
                return (candidate_keys, search_stats) if return_stats else candidate_keys
        candidate_keys = set(anchor_ct.candidates.keys())
        return (candidate_keys, search_stats) if return_stats else candidate_keys

    def _proxy_counts_by_anchor(self, reference, candidate_mask):
        truth = reference["truth"]
        anchor_ids = reference["pair_anchor_ids"]
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        n_anchors = max(int(reference.get("anchor_count", 0) or 0), int(anchor_ids.max()) + 1 if anchor_ids.size else 0)
        counts = np.zeros((n_anchors, 4), dtype=np.int64)  # tp, tn, fp, fn
        for anchor_id in range(n_anchors):
            mask = anchor_ids == anchor_id
            if not np.any(mask):
                continue
            cand = candidate_mask[mask]
            gt = truth[mask]
            counts[anchor_id, 0] = int(np.sum(cand & gt))
            counts[anchor_id, 1] = int(np.sum((~cand) & (~gt)))
            counts[anchor_id, 2] = int(np.sum(cand & (~gt)))
            counts[anchor_id, 3] = int(np.sum((~cand) & gt))
        return counts

    @staticmethod
    def _empty_proxy_search_stats():
        return {
            "index_candidates": 0,
            "valid_index_candidates": 0,
            "unique_index_candidates": 0,
            "duplicate_index_candidates": 0,
            "unique_pre_dot_pairs": 0,
            "duplicate_pre_dot_pairs": 0,
            "after_coord": 0,
            "partial_checks": 0,
            "after_partial": 0,
            "after_similarity": 0,
            "after_dot": 0,
            "dot_checks": 0,
            "distance_checks": 0,
            # (2026-07-06) Part 1 metrics -- see docs/implementation_log.md.
            "blocks_visited": 0,
            "blocks_pruned_by_ub": 0,
            "rows_in_surviving_blocks": 0,
            "dot_checks_saved_by_row_ub": 0,
        }

    @classmethod
    def _merge_proxy_search_stats(cls, target, source):
        if target is None:
            target = cls._empty_proxy_search_stats()
        if not source:
            return target
        for key in (
            "index_candidates",
            "valid_index_candidates",
            "unique_index_candidates",
            "duplicate_index_candidates",
            "unique_pre_dot_pairs",
            "duplicate_pre_dot_pairs",
            "after_coord",
            "partial_checks",
            "after_partial",
            "after_similarity",
            "after_dot",
            "dot_checks",
            "distance_checks",
            "blocks_visited",
            "blocks_pruned_by_ub",
            "rows_in_surviving_blocks",
            "dot_checks_saved_by_row_ub",
        ):
            value = _to_float_safe(source.get(key))
            if value is not None:
                target[key] = int(target.get(key, 0) or 0) + int(value)
        return target

    def _proxy_candidate_search_stats_for_corrtrack(self, corrtrack):
        stats = self._empty_proxy_search_stats()
        found = False
        for node in getattr(corrtrack, "grid_nodes", []) or []:
            stats_getter = getattr(node, "candidate_search_stats", None)
            if stats_getter is None:
                continue
            node_stats = stats_getter()
            if not node_stats:
                continue
            found = True
            after_similarity = int(node_stats.get("after_similarity", 0) or 0)
            stats["index_candidates"] += int(node_stats.get("index_candidates", 0) or 0)
            stats["valid_index_candidates"] += int(node_stats.get("valid_index_candidates", 0) or 0)
            stats["unique_index_candidates"] += int(node_stats.get("unique_index_candidates", 0) or 0)
            stats["duplicate_index_candidates"] += int(node_stats.get("duplicate_index_candidates", 0) or 0)
            stats["unique_pre_dot_pairs"] += int(node_stats.get("unique_pre_dot_pairs", 0) or 0)
            stats["duplicate_pre_dot_pairs"] += int(node_stats.get("duplicate_pre_dot_pairs", 0) or 0)
            stats["after_coord"] += int(node_stats.get("after_coord", 0) or 0)
            stats["partial_checks"] += int(node_stats.get("partial_checks", 0) or 0)
            stats["after_partial"] += int(node_stats.get("after_partial", 0) or 0)
            stats["after_similarity"] += after_similarity
            stats["after_dot"] += after_similarity
            stats["dot_checks"] += int(node_stats.get("dot_checks", 0) or 0)
            stats["distance_checks"] += int(node_stats.get("distance_checks", 0) or 0)
            stats["blocks_visited"] += int(node_stats.get("blocks_visited", 0) or 0)
            stats["blocks_pruned_by_ub"] += int(node_stats.get("blocks_pruned_by_ub", 0) or 0)
            stats["rows_in_surviving_blocks"] += int(node_stats.get("rows_in_surviving_blocks", 0) or 0)
            stats["dot_checks_saved_by_row_ub"] += int(node_stats.get("dot_checks_saved_by_row_ub", 0) or 0)
        return stats if found else None

    @staticmethod
    def _proxy_search_record_fields(metrics, search_stats=None):
        metrics = metrics or {}
        total = _to_float_safe(metrics.get("total"))
        if total is None or total <= 0.0:
            total = _to_float_safe(metrics.get("proxy_n_pairs"))
        candidate_rate = _to_float_safe(metrics.get("candidate_rate"))
        if candidate_rate is None:
            candidate_rate = 0.0

        stats = dict(search_stats or {})
        index_candidates = _to_float_safe(stats.get("index_candidates"))
        valid_index_candidates = _to_float_safe(stats.get("valid_index_candidates"))
        duplicate_index_candidates = _to_float_safe(stats.get("duplicate_index_candidates"))
        unique_pre_dot_pairs = _to_float_safe(stats.get("unique_pre_dot_pairs"))
        duplicate_pre_dot_pairs = _to_float_safe(stats.get("duplicate_pre_dot_pairs"))
        after_partial = _to_float_safe(stats.get("after_partial"))
        after_similarity = _to_float_safe(stats.get("after_similarity", stats.get("after_dot")))
        after_dot = _to_float_safe(stats.get("after_dot"))
        dot_checks = _to_float_safe(stats.get("dot_checks"))
        distance_checks = _to_float_safe(stats.get("distance_checks"))
        if dot_checks is None:
            dot_checks = 0.0
        if distance_checks is None:
            distance_checks = 0.0
        similarity_checks = float(dot_checks) + float(distance_checks)

        if total is None or total <= 0.0:
            index_rate = None
            dot_work_rate = None
            similarity_work_rate = None
        else:
            index_rate = None if index_candidates is None else float(index_candidates) / total
            dot_work_rate = None if dot_checks is None else float(dot_checks) / total
            similarity_work_rate = float(similarity_checks) / total

        if index_candidates is None or index_candidates <= 0.0 or after_dot is None:
            cascade_reject_rate = None
        else:
            cascade_reject_rate = 1.0 - (float(after_dot) / float(index_candidates))

        if similarity_work_rate is None:
            objective_rate = candidate_rate
        else:
            objective_rate = candidate_rate + similarity_work_rate

        return {
            "proxy_search_index_candidates": None if index_candidates is None else int(index_candidates),
            "proxy_search_valid_index_candidates": None if valid_index_candidates is None else int(valid_index_candidates),
            "proxy_search_duplicate_index_candidates": None if duplicate_index_candidates is None else int(duplicate_index_candidates),
            "proxy_search_unique_pre_dot_pairs": None if unique_pre_dot_pairs is None else int(unique_pre_dot_pairs),
            "proxy_search_duplicate_pre_dot_pairs": None if duplicate_pre_dot_pairs is None else int(duplicate_pre_dot_pairs),
            "proxy_search_after_partial": None if after_partial is None else int(after_partial),
            "proxy_search_after_dot": None if after_dot is None else int(after_dot),
            "proxy_search_dot_checks": None if dot_checks is None else int(dot_checks),
            "proxy_search_distance_checks": None if distance_checks is None else int(distance_checks),
            "proxy_search_similarity_work_rate": similarity_work_rate,
            "proxy_search_index_rate": index_rate,
            "proxy_search_dot_work_rate": dot_work_rate,
            "proxy_search_cascade_reject_rate": cascade_reject_rate,
            "proxy_search_objective_rate": objective_rate,
        }

    @staticmethod
    def _candidate_search_record_fields_from_stats(search_stats=None):
        stats = dict(search_stats or {})

        def as_int(name):
            value = _to_float_safe(stats.get(name))
            return None if value is None else int(value)

        return {
            "candidate_search_index_candidates": as_int("index_candidates"),
            "candidate_search_valid_index_candidates": as_int("valid_index_candidates"),
            "candidate_search_duplicate_index_candidates": as_int("duplicate_index_candidates"),
            "candidate_search_unique_pre_dot_pairs": as_int("unique_pre_dot_pairs"),
            "candidate_search_duplicate_pre_dot_pairs": as_int("duplicate_pre_dot_pairs"),
            "candidate_search_after_partial": as_int("after_partial"),
            "candidate_search_after_similarity": as_int("after_similarity"),
            "candidate_search_dot_checks": as_int("dot_checks"),
            "candidate_search_distance_checks": as_int("distance_checks"),
            # (2026-07-06) Part 1 metrics -- see docs/implementation_log.md.
            "candidate_search_blocks_visited": as_int("blocks_visited"),
            "candidate_search_blocks_pruned_by_ub": as_int("blocks_pruned_by_ub"),
            "candidate_search_rows_in_surviving_blocks": as_int("rows_in_surviving_blocks"),
            "candidate_search_dot_checks_saved_by_row_ub": as_int("dot_checks_saved_by_row_ub"),
        }

    def _proxy_bootstrap_summary(self, counts_by_anchor):
        total_counts = {
            "tp": int(np.sum(counts_by_anchor[:, 0])) if counts_by_anchor.size else 0,
            "tn": int(np.sum(counts_by_anchor[:, 1])) if counts_by_anchor.size else 0,
            "fp": int(np.sum(counts_by_anchor[:, 2])) if counts_by_anchor.size else 0,
            "fn": int(np.sum(counts_by_anchor[:, 3])) if counts_by_anchor.size else 0,
        }
        metrics = self._proxy_pair_metrics_from_counts(total_counts)
        summary = {
            "counts": total_counts,
            "metrics": metrics,
            "gt_support": int(metrics["gt_total"]),
            "underpowered": int(metrics["gt_total"]) < int(self.proxy_bootstrap_min_gt_events),
            "bootstrap_time": 0.0,
        }
        if not self._proxy_bootstrap_active() or counts_by_anchor.shape[0] == 0:
            return summary

        t_bootstrap = time.perf_counter()
        rng = np.random.default_rng(int(self.proxy_random_seed))
        n_anchors = int(counts_by_anchor.shape[0])
        recalls = []
        precisions = []
        specificities = []
        candidate_rates = []
        for _ in range(int(self.proxy_bootstrap_repeats)):
            sampled = rng.integers(0, n_anchors, size=n_anchors)
            sample_counts = np.sum(counts_by_anchor[sampled, :], axis=0)
            sample_metrics = self._proxy_pair_metrics_from_counts(
                {
                    "tp": int(sample_counts[0]),
                    "tn": int(sample_counts[1]),
                    "fp": int(sample_counts[2]),
                    "fn": int(sample_counts[3]),
                }
            )
            recalls.append(sample_metrics["recall"])
            precisions.append(sample_metrics["precision"])
            specificities.append(sample_metrics["specificity"])
            candidate_rates.append(sample_metrics["candidate_rate"])

        summary.update(
            {
                "recall_ci": self._bootstrap_quantiles(recalls, self.proxy_bootstrap_confidence_level),
                "recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
                "precision_ci": self._bootstrap_quantiles(precisions, self.proxy_bootstrap_confidence_level),
                "precision_mean": float(np.mean(precisions)) if precisions else float("nan"),
                "specificity_ci": self._bootstrap_quantiles(specificities, self.proxy_bootstrap_confidence_level),
                "specificity_mean": float(np.mean(specificities)) if specificities else float("nan"),
                "candidate_rate_ci": self._bootstrap_quantiles(candidate_rates, self.proxy_bootstrap_confidence_level),
                "candidate_rate_mean": float(np.mean(candidate_rates)) if candidate_rates else float("nan"),
                "bootstrap_time": time.perf_counter() - t_bootstrap,
            }
        )
        return summary

    @staticmethod
    def _bootstrap_quantiles(values, confidence_level):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")
        alpha = max(0.0, min(1.0 - float(confidence_level), 1.0))
        lower_q = alpha / 2.0
        upper_q = 1.0 - lower_q
        return (
            float(np.quantile(arr, lower_q)),
            float(np.quantile(arr, 0.5)),
            float(np.quantile(arr, upper_q)),
        )

    @staticmethod
    def _proxy_group_key_value(value):
        if isinstance(value, dict):
            return tuple(sorted((k, CorrTrack_optimize._proxy_group_key_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(CorrTrack_optimize._proxy_group_key_value(v) for v in value)
        return value

    def _proxy_cache_group_key(self, param_combo):
        """Group proxy rows whose cached-distance work is reusable."""

        sketch_fields = (
            "n_vectors",
            "seed",
            "seed_toggle",
            "preprocess",
            "grid_dimension",
            "sketch_norm",
            "candidate_backend",
            "candidate_parallel_mode",
                                        )
        return tuple((name, self._proxy_group_key_value(param_combo.get(name))) for name in sketch_fields)

    def _proxy_build_corrtrack_from_combo(self, param_combo):
        seed = _to_int_safe(param_combo.get("seed"))
        seed_toggle = _to_int_safe(param_combo.get("seed_toggle"))
        n_vectors = _to_int_safe(param_combo.get("n_vectors"))
        grid_dimension = _to_int_safe(param_combo.get("grid_dimension"))
        freq_threshold = _to_float_safe(param_combo.get("freq_threshold"))
        cell_stretch = _to_float_safe(param_combo.get("cell_size"))
        if cell_stretch is None or cell_stretch <= 0.0:
            cell_stretch = 1.0

        feature_kwargs = _extract_feature_overrides({
            "candidate_key_mode": self.candidate_key_mode,
            "candidate_key_seed": self.candidate_key_seed,
            "candidate_lsh_radius": self.candidate_lsh_radius,
            "candidate_ann_m": self.candidate_ann_m,
            "candidate_ann_z": self.candidate_ann_z,
            "candidate_ann_ef": self.candidate_ann_ef,
            "candidate_bucket_width": self.candidate_bucket_width,
            "candidate_parallel_mode": self.candidate_parallel_mode,
            "candidate_block_size_steps": self.candidate_block_size_steps,
            "candidate_block_index_dims": self.candidate_block_index_dims,
            "candidate_bound_dims": self.candidate_bound_dims,
            "candidate_bound_dim_selection": self.candidate_bound_dim_selection,
            "enable_block_ub_pruning": self.enable_block_ub_pruning,
            "enable_row_ub_pruning": self.enable_row_ub_pruning,
            "block_similarity_assignment": self.block_similarity_assignment,
            "max_open_blocks": self.max_open_blocks,
            "candidate_instinct_query_mode": self.candidate_instinct_query_mode,
            "candidate_instinct_top_k": self.candidate_instinct_top_k,
            "candidate_instinct_min_candidates": self.candidate_instinct_min_candidates,
            "candidate_instinct_entry_points": self.candidate_instinct_entry_points,
            "candidate_similarity": self.candidate_similarity,
            "candidate_cosine_threshold": self.candidate_cosine_threshold,
            "hybrid_validation": self.hybrid_validation,
            "hybrid_validation_min_repeat_rate": self.hybrid_validation_min_repeat_rate,
            "hybrid_validation_disable_rate": self.hybrid_validation_disable_rate,
            "hybrid_validation_ema_alpha": self.hybrid_validation_ema_alpha,
            "hybrid_validation_min_candidates": self.hybrid_validation_min_candidates,
            "numeric_rows": self.numeric_rows,
            **param_combo,
        })

        corrtrack = CorrTrack(
            window_size=self.window_size,
            basic_window=self.basic_window,
            window_step=self.window_step,
            n_vectors=n_vectors,
            n_lags=self.n_lags,
            grid_dimension=grid_dimension,
            cell_size=cell_stretch,
            seed=seed,
            seed_toggle=seed_toggle,
            freq_threshold=freq_threshold,
            corr_threshold=self.corr_threshold,
            neg_corr=self.neg_corr,
            preprocess=param_combo.get("preprocess"),
            exec="sequential",
            max_workers=self.max_workers,
            parallel_sketch=False,
            parallel_candidates=self.parallel_candidates,
            parallel_validation=False,
            track_min_dist=False,
            **feature_kwargs,
        )
        return corrtrack, feature_kwargs

    def _init_proxy_record_from_corrtrack(self, dataset_id, param_combo, corrtrack, eval_mode, cache_used=False):
        reference = self._proxy_reference
        record = self._init_optim_record(dataset_id, param_combo)
        record["tuning_mode"] = self.tuning_mode
        record["proxy_anchor_count_requested"] = self.proxy_anchor_count_requested
        record["proxy_anchor_count"] = reference.get("anchor_count") if reference else None
        record["proxy_anchor_expansion_iteration"] = getattr(self, "_proxy_anchor_expansion_iteration", 0)
        record["proxy_anchor_expansion_reason"] = getattr(self, "_proxy_anchor_expansion_reason", "")
        record["proxy_anchor_seed"] = self.proxy_random_seed
        record["proxy_max_pair_rows"] = self.proxy_max_pair_rows
        record["proxy_eval_mode"] = eval_mode
        record["proxy_distance_cache_used"] = bool(cache_used)
        record["proxy_n_pairs"] = reference.get("n_pairs") if reference else None
        record["proxy_n_gt"] = reference.get("n_gt") if reference else None
        record["proxy_bootstrap_enabled"] = self._proxy_bootstrap_active()
        record["proxy_bootstrap_repeats"] = self.proxy_bootstrap_repeats if self._proxy_bootstrap_active() else None
        record["proxy_bootstrap_confidence_level"] = (
            self.proxy_bootstrap_confidence_level if self._proxy_bootstrap_active() else None
        )
        record["proxy_bootstrap_min_gt_events"] = (
            self.proxy_bootstrap_min_gt_events if self._proxy_bootstrap_active() else None
        )
        record["proxy_warning"] = reference.get("warning", "") if reference else ""

        record["nodes"] = param_combo.get("nodes")
        record["seed"] = corrtrack.seed
        record["seed_toggle"] = corrtrack.seed_toggle
        record["n_vectors"] = corrtrack.n_vectors
        record["preprocess"] = corrtrack.preprocess
        record["grid_dimension"] = corrtrack.grid_dimension
        record["cell_stretch"] = corrtrack.cell_stretch
        record["freq_threshold"] = param_combo.get("freq_threshold")
        record["grid_max"] = corrtrack.grid_max
        record["cell_size"] = corrtrack.cell_size
        record["n_lags"] = corrtrack.n_lags
        record["basic_window"] = corrtrack.basic_window
        record["exec_mode"] = "sequential"
        record["parallel_sketch"] = False
        record["parallel_candidates"] = corrtrack.parallel_candidates
        record["parallel_validation"] = False
        record["candidate_backend"] = getattr(
            corrtrack,
            "candidate_backend_effective",
            getattr(corrtrack, "candidate_backend", None),
        )
        record["candidate_bucket_width"] = corrtrack.candidate_bucket_width
        record["candidate_block_size_steps"] = corrtrack.candidate_block_size_steps
        record["candidate_block_index_dims"] = corrtrack.candidate_block_index_dims
        # (2026-07-06) Part 1 -- see docs/implementation_log.md.
        record["candidate_bound_dims"] = getattr(corrtrack, "candidate_bound_dims", None)
        record["candidate_bound_dim_selection"] = getattr(corrtrack, "candidate_bound_dim_selection", None)
        record["enable_block_ub_pruning"] = getattr(corrtrack, "enable_block_ub_pruning", None)
        record["enable_row_ub_pruning"] = getattr(corrtrack, "enable_row_ub_pruning", None)
        record["block_similarity_assignment"] = getattr(corrtrack, "block_similarity_assignment", None)
        record["max_open_blocks"] = getattr(corrtrack, "max_open_blocks", None)
        record["candidate_instinct_query_mode"] = getattr(corrtrack, "candidate_instinct_query_mode", None)
        record["candidate_instinct_top_k"] = getattr(corrtrack, "candidate_instinct_top_k", None)
        record["candidate_instinct_min_candidates"] = getattr(corrtrack, "candidate_instinct_min_candidates", None)
        record["candidate_instinct_entry_points"] = getattr(corrtrack, "candidate_instinct_entry_points", None)
        record.update(_candidate_runtime_record_fields(corrtrack))
        record["candidate_parallel_mode"] = corrtrack.candidate_parallel_mode
        record["candidate_key_mode"] = corrtrack.candidate_key_mode
        record["candidate_key_seed"] = corrtrack.candidate_key_seed
        record["candidate_lsh_radius"] = corrtrack.candidate_lsh_radius
        record["candidate_ann_m"] = corrtrack.candidate_ann_m
        record["candidate_ann_z"] = corrtrack.candidate_ann_z
        record["candidate_ann_ef"] = corrtrack.candidate_ann_ef
        record["numeric_rows"] = corrtrack.numeric_rows
        record.update(_hybrid_validation_record_fields(corrtrack))
        record["workers_total"] = 1
        record["workers_sketch"] = 1
        record["workers_candidates"] = corrtrack.n_candidate_nodes
        record["workers_validation"] = 1
        record["sketch_norm"] = corrtrack.sketch_norm
        return record

    @staticmethod
    def _proxy_partition_to_window_map(partition, proxy_ids):
        if partition is None:
            return {}
        if isinstance(partition, dict):
            out = {}
            for key, value in partition.items():
                if len(value) >= 2:
                    out[(str(key[0]), int(key[1]), int(key[2]))] = (
                        np.asarray(value[0], dtype=np.float64),
                        bool(value[1]),
                    )
            return out
        if isinstance(partition, tuple) and len(partition) == 5:
            sid_idx, time_arr, w_arr, chunk_arr, is_const = partition
            chunks = np.asarray(chunk_arr, dtype=np.float64)
            const_arr = np.asarray(is_const, dtype=bool)
            out = {}
            for i in range(len(sid_idx)):
                idx = int(sid_idx[i])
                sid = str(proxy_ids[idx]) if 0 <= idx < len(proxy_ids) else str(idx)
                vec = np.asarray(chunks[i], dtype=np.float64)
                out[(sid, int(time_arr[i]), int(w_arr[i]))] = (vec, bool(const_arr[i]))
            return out
        return {}

    @staticmethod
    def _proxy_sketch_distance_sq(vec_a, vec_b, neg_corr=False):
        a = np.asarray(vec_a, dtype=np.float64).ravel()
        b = np.asarray(vec_b, dtype=np.float64).ravel()
        if a.size == 0 or b.size == 0:
            return float("inf")
        if a.size != b.size:
            dim = max(a.size, b.size)
            a_fixed = np.zeros(dim, dtype=np.float64)
            b_fixed = np.zeros(dim, dtype=np.float64)
            a_fixed[: a.size] = a
            b_fixed[: b.size] = b
            a = a_fixed
            b = b_fixed
        diff = a - b
        dist_sq = float(np.dot(diff, diff))
        if neg_corr:
            neg_diff = a + b
            dist_sq = min(dist_sq, float(np.dot(neg_diff, neg_diff)))
        return dist_sq

    def _proxy_cached_distance_matrix(self, reference, partitions_by_start):
        proxy_ids = reference.get("proxy_ids", [])
        grid_maps = []
        for partitions in partitions_by_start.values():
            for grid_index, partition in enumerate(partitions or []):
                while len(grid_maps) <= grid_index:
                    grid_maps.append({})
                grid_maps[grid_index].update(
                    self._proxy_partition_to_window_map(partition, proxy_ids)
                )
        if not grid_maps:
            raise RuntimeError("proxy cached distance mode found no sketch partitions")

        n_pairs = int(reference.get("n_pairs", 0) or 0)
        n_grids = len(grid_maps)

        distances = np.full((n_pairs, n_grids), np.inf, dtype=np.float64)
        distance_cache = {}
        full_distance_entries = 0
        for row_idx, pair_key in enumerate(reference.get("pair_keys", [])):
            cached = distance_cache.get(pair_key)
            if cached is not None:
                distances[row_idx, :] = cached
                continue
            id1, id2, t1, t2, w = pair_key
            key1 = (str(id1), int(t1), int(w))
            key2 = (str(id2), int(t2), int(w))
            row_dist = np.full((n_grids,), np.inf, dtype=np.float64)
            for grid_index, window_map in enumerate(grid_maps):
                item1 = window_map.get(key1)
                item2 = window_map.get(key2)
                if item1 is None or item2 is None:
                    continue
                vec1, const1 = item1
                vec2, const2 = item2
                if const1 or const2:
                    continue
                row_dist[grid_index] = self._proxy_sketch_distance_sq(
                    vec1,
                    vec2,
                    neg_corr=self.neg_corr,
                )
                full_distance_entries += 1
            distance_cache[pair_key] = row_dist
            distances[row_idx, :] = row_dist
        # Distance work is cached by pair_key, so report unique pair/grid
        # distances rather than duplicated anchor rows. The prefilter columns are
        # kept for CSV compatibility but left empty while prefiltering is off.
        total_entries = int(len(distance_cache) * n_grids)
        stats = {
            "max_tau_sq": None,
            "total_entries": total_entries,
            "entries": int(full_distance_entries),
            "fraction": None,
        }
        return distances, stats

    @staticmethod
    def _proxy_mask_from_cached_distances(distances, corrtrack):
        if getattr(corrtrack, "candidate_similarity", "l2") == "cosine":
            tau = _candidate_tau_from_gamma(getattr(corrtrack, "candidate_cosine_threshold", None))
        else:
            tau = _to_float_safe(getattr(corrtrack, "cell_size", None))
        if tau is None or tau < 0.0:
            return np.zeros(distances.shape[0], dtype=bool)
        hits = np.isfinite(distances) & (distances <= (float(tau) * float(tau) + 1e-12))
        hit_counts = np.sum(hits, axis=1)
        required = _to_float_safe(getattr(corrtrack, "freq_threshold", 0.0))
        if required is None or required <= 0.0:
            return hit_counts > 0
        return hit_counts >= float(required)

    def _finalize_proxy_record_from_mask(self, record, reference, candidate_mask, summary, sketch_time, candidate_time, runtime, search_stats=None):
        counts = summary["counts"]
        metrics = summary["metrics"]
        truth_sign = reference["truth_sign"]
        truth = reference["truth"]
        tp_pos = int(np.sum(candidate_mask & (truth_sign > 0)))
        tp_neg = int(np.sum(candidate_mask & (truth_sign < 0)))
        gt_pos = int(np.sum(truth_sign > 0))
        gt_neg = int(np.sum(truth_sign < 0))

        record["cand_time_bf"] = 0.0
        record["val_time_bf"] = 0.0
        record["monit_time_bf"] = 0.0
        record["runtime_bf"] = 0.0
        record["artifact_time_bf"] = 0.0
        record["corr_w_bf"] = metrics["gt_total"]
        record["tested_w_bf"] = metrics["total"]
        record["cand_w_bf"] = metrics["total"]
        record["sk_time"] = sketch_time
        record["cand_time"] = candidate_time
        record["val_time"] = 0.0
        record["monit_time"] = 0.0
        record["runtime"] = runtime
        record["optim_bootstrap_time"] = float(summary.get("bootstrap_time", 0.0) or 0.0)
        record["optim_eval_time"] = runtime
        record["optim_search_time"] = max(float(runtime or 0.0) - record["optim_bootstrap_time"], 0.0)
        record["artifact_time"] = 0.0
        record["corr_w"] = metrics["tp"]
        record["tested_w"] = metrics["cand_total"]
        record["cand_w"] = metrics["cand_total"]
        record["speedup"] = _safe_div(metrics["total"], metrics["cand_total"])
        record["speedup_ceil"] = record["speedup"]
        record["rel_speedup_eff"] = 1.0
        record["corr_prop"] = _safe_div(metrics["gt_total"], metrics["total"])
        record["waste_val_bf"] = _safe_div(metrics["total"], metrics["gt_total"])
        record["waste_val"] = _safe_div(metrics["cand_total"], metrics["tp"])
        record["rel_waste_red"] = _safe_div(record.get("waste_val_bf"), record.get("waste_val"))
        record["precision_pos"] = _signed_precision_from_ambiguous_fp(tp_pos, tp_neg, metrics["cand_total"])
        record["recall_pos"] = _safe_div(tp_pos, gt_pos)
        record["f1_pos"] = None
        record["precision_neg"] = _signed_precision_from_ambiguous_fp(tp_neg, tp_pos, metrics["cand_total"])
        record["recall_neg"] = _safe_div(tp_neg, gt_neg)
        record["f1_neg"] = None
        record["precision"] = metrics["precision"]
        record["recall"] = metrics["recall"]
        record["specificity"] = metrics["specificity"]
        record["recall_min"] = min(
            record["recall_pos"] if record["recall_pos"] is not None else metrics["recall"],
            record["recall_neg"] if record["recall_neg"] is not None else metrics["recall"],
        )
        record["f1"] = metrics["f1"]
        record["aucroc"] = None
        record["pr_auc"] = None
        record["proxy_n_pairs"] = metrics["total"]
        record["proxy_n_gt"] = metrics["gt_total"]
        record["proxy_candidate_rate"] = metrics["candidate_rate"]
        record.update(self._proxy_search_record_fields(metrics, search_stats))
        record.update(self._candidate_search_record_fields_from_stats(search_stats))
        record["proxy_gt_support"] = summary["gt_support"]
        record["proxy_underpowered"] = bool(summary["underpowered"])

        if self._proxy_bootstrap_active():
            for prefix, values in (
                ("proxy_recall", summary.get("recall_ci")),
                ("proxy_precision", summary.get("precision_ci")),
                ("proxy_specificity", summary.get("specificity_ci")),
                ("proxy_candidate_rate", summary.get("candidate_rate_ci")),
            ):
                if values is None:
                    continue
                record[f"{prefix}_lb"] = values[0]
                record[f"{prefix}_med"] = values[1]
                record[f"{prefix}_ub"] = values[2]
            record["proxy_recall_mean"] = summary.get("recall_mean")
            record["proxy_precision_mean"] = summary.get("precision_mean")
            record["proxy_specificity_mean"] = summary.get("specificity_mean")
            record["proxy_candidate_rate_mean"] = summary.get("candidate_rate_mean")
        else:
            record["proxy_recall_lb"] = metrics["recall"]
            record["proxy_recall_mean"] = metrics["recall"]
            record["proxy_recall_med"] = metrics["recall"]
            record["proxy_recall_ub"] = metrics["recall"]
            record["proxy_precision_lb"] = metrics["precision"]
            record["proxy_precision_mean"] = metrics["precision"]
            record["proxy_precision_med"] = metrics["precision"]
            record["proxy_precision_ub"] = metrics["precision"]
            record["proxy_specificity_lb"] = metrics["specificity"]
            record["proxy_specificity_mean"] = metrics["specificity"]
            record["proxy_specificity_med"] = metrics["specificity"]
            record["proxy_specificity_ub"] = metrics["specificity"]
            record["proxy_candidate_rate_lb"] = metrics["candidate_rate"]
            record["proxy_candidate_rate_mean"] = metrics["candidate_rate"]
            record["proxy_candidate_rate_med"] = metrics["candidate_rate"]
            record["proxy_candidate_rate_ub"] = metrics["candidate_rate"]
        record["status"] = "success"
        record["error"] = ""
        return record

    def _outer_iter(self, items, worker_fn, unordered=True):
        """
        Iterate outer tasks with inversion rule:
          - inner parallel  -> outer sequential
          - inner sequential-> outer parallel
        """
        items = list(items)
        inner_parallel = self._run_is_parallel()
        if inner_parallel or self.exec == "sequential" or not _HAS_DASK or delayed is None:
            for it in items:
                yield worker_fn(it)
            return

        max_workers = self.max_workers or max(1, (os.cpu_count() or 1) - 1)
        max_workers = max(1, max_workers)
        scheduler_get = dask_threaded_get
        if scheduler_get is None:
            for it in items:
                yield worker_fn(it)
            return

        scheduler = partial(scheduler_get, num_workers=max_workers)
        try:
            tasks = [delayed(worker_fn)(it) for it in items]
            results = compute(*tasks, scheduler=scheduler)
        except Exception as exc:
            if getattr(self, "verbose", False):
                print(f"[outer_iter] Dask fallback to sequential due to: {exc}")
            for it in items:
                yield worker_fn(it)
            return

        for res in results:
            yield res

    def _run_corrtrack_proxy_anchor_mean_timed(self, args):
        # (2026-07-05) `_run_corrtrack_proxy_anchor` is fully self-contained
        # (builds a fresh CorrTrack + fresh index each call, no shared
        # mutable state beyond reading self._proxy_reference), so repeating
        # it is safe and idempotent -- recall/precision/stats are
        # deterministic given the same seed, but a single wall-clock
        # measurement at these scales (single-to-double-digit ms) is noisy.
        # Keep one representative record's non-timing fields, overwrite the
        # timing fields with the mean across repeats. See
        # docs/implementation_log.md, "hyperopt real-time tie-break".
        repeats = max(1, int(self.proxy_timing_repeats))
        record = self._run_corrtrack_proxy_anchor(args)
        if repeats <= 1 or record.get("status") != "success":
            record["proxy_timing_repeats"] = 1
            return record
        sk_times = [self._to_float(record.get("sk_time")) or 0.0]
        cand_times = [self._to_float(record.get("cand_time")) or 0.0]
        runtimes = [self._to_float(record.get("runtime")) or 0.0]
        for _ in range(repeats - 1):
            repeat_record = self._run_corrtrack_proxy_anchor(args)
            if repeat_record.get("status") != "success":
                continue
            sk_times.append(self._to_float(repeat_record.get("sk_time")) or 0.0)
            cand_times.append(self._to_float(repeat_record.get("cand_time")) or 0.0)
            runtimes.append(self._to_float(repeat_record.get("runtime")) or 0.0)
        record["sk_time"] = float(np.mean(sk_times))
        record["cand_time"] = float(np.mean(cand_times))
        record["runtime"] = float(np.mean(runtimes))
        record["optim_eval_time"] = record["runtime"]
        record["optim_search_time"] = max(
            record["runtime"] - (self._to_float(record.get("optim_bootstrap_time")) or 0.0), 0.0
        )
        record["proxy_timing_repeats"] = len(cand_times)
        return record

    def _run_corrtrack_proxy_anchor(self, args):
        trial_index, param_combo, dataset_id = args
        reference = self._proxy_reference
        record = self._init_optim_record(dataset_id, param_combo)
        record["tuning_mode"] = self.tuning_mode
        record["proxy_anchor_count_requested"] = self.proxy_anchor_count_requested
        record["proxy_anchor_count"] = reference.get("anchor_count") if reference else None
        record["proxy_anchor_expansion_iteration"] = getattr(self, "_proxy_anchor_expansion_iteration", 0)
        record["proxy_anchor_expansion_reason"] = getattr(self, "_proxy_anchor_expansion_reason", "")
        record["proxy_anchor_seed"] = self.proxy_random_seed
        record["proxy_max_pair_rows"] = self.proxy_max_pair_rows
        record["proxy_eval_mode"] = "backend_search"
        record["proxy_distance_cache_used"] = False
        record["proxy_distance_cache_rows"] = None
        record["proxy_distance_cache_grids"] = None
        record["proxy_n_pairs"] = reference.get("n_pairs") if reference else None
        record["proxy_n_gt"] = reference.get("n_gt") if reference else None
        record["proxy_bootstrap_enabled"] = self._proxy_bootstrap_active()
        record["proxy_bootstrap_repeats"] = self.proxy_bootstrap_repeats if self._proxy_bootstrap_active() else None
        record["proxy_bootstrap_confidence_level"] = (
            self.proxy_bootstrap_confidence_level if self._proxy_bootstrap_active() else None
        )
        record["proxy_bootstrap_min_gt_events"] = (
            self.proxy_bootstrap_min_gt_events if self._proxy_bootstrap_active() else None
        )
        record["proxy_warning"] = reference.get("warning", "") if reference else ""

        nodes = param_combo.get("nodes")
        try:
            nodes = int(nodes) if nodes is not None else None
        except (TypeError, ValueError):
            nodes = None
        seed = param_combo.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None
        seed_toggle = param_combo.get("seed_toggle")
        try:
            seed_toggle = int(seed_toggle) if seed_toggle is not None else None
        except (TypeError, ValueError):
            seed_toggle = None
        n_vectors = param_combo.get("n_vectors")
        try:
            n_vectors = int(n_vectors) if n_vectors is not None else None
        except (TypeError, ValueError):
            n_vectors = None
        preprocess = param_combo.get("preprocess")
        grid_dimension = param_combo.get("grid_dimension")
        try:
            grid_dimension = int(grid_dimension) if grid_dimension is not None else None
        except (TypeError, ValueError):
            grid_dimension = None
        freq_threshold = param_combo.get("freq_threshold")
        try:
            freq_threshold = float(freq_threshold) if freq_threshold is not None else None
        except (TypeError, ValueError):
            freq_threshold = None
        cell_stretch = param_combo.get("cell_size")
        try:
            cell_stretch = float(cell_stretch) if cell_stretch is not None else None
        except (TypeError, ValueError):
            cell_stretch = None
        if cell_stretch is None or cell_stretch <= 0.0:
            cell_stretch = 1.0

        feature_kwargs = _extract_feature_overrides({
            "candidate_key_mode": self.candidate_key_mode,
            "candidate_key_seed": self.candidate_key_seed,
            "candidate_lsh_radius": self.candidate_lsh_radius,
            "candidate_ann_m": self.candidate_ann_m,
            "candidate_ann_z": self.candidate_ann_z,
            "candidate_ann_ef": self.candidate_ann_ef,
            "candidate_parallel_mode": self.candidate_parallel_mode,
            "candidate_bucket_width": self.candidate_bucket_width,
            "candidate_block_size_steps": self.candidate_block_size_steps,
            "candidate_block_index_dims": self.candidate_block_index_dims,
            "candidate_bound_dims": self.candidate_bound_dims,
            "candidate_bound_dim_selection": self.candidate_bound_dim_selection,
            "enable_block_ub_pruning": self.enable_block_ub_pruning,
            "enable_row_ub_pruning": self.enable_row_ub_pruning,
            "block_similarity_assignment": self.block_similarity_assignment,
            "max_open_blocks": self.max_open_blocks,
            "candidate_instinct_query_mode": self.candidate_instinct_query_mode,
            "candidate_instinct_top_k": self.candidate_instinct_top_k,
            "candidate_instinct_min_candidates": self.candidate_instinct_min_candidates,
            "candidate_instinct_entry_points": self.candidate_instinct_entry_points,
            "candidate_similarity": self.candidate_similarity,
            "candidate_cosine_threshold": self.candidate_cosine_threshold,
            "hybrid_validation": self.hybrid_validation,
            "hybrid_validation_min_repeat_rate": self.hybrid_validation_min_repeat_rate,
            "hybrid_validation_disable_rate": self.hybrid_validation_disable_rate,
            "hybrid_validation_ema_alpha": self.hybrid_validation_ema_alpha,
            "hybrid_validation_min_candidates": self.hybrid_validation_min_candidates,
            **param_combo,
        })

        record["nodes"] = nodes
        record["seed"] = seed
        record["seed_toggle"] = seed_toggle
        record["n_vectors"] = n_vectors
        record["preprocess"] = preprocess
        record["grid_dimension"] = grid_dimension
        record["cell_stretch"] = cell_stretch
        record["freq_threshold"] = freq_threshold

        t0_total = time.perf_counter()
        try:
            if reference is None:
                raise RuntimeError("Missing proxy-anchor reference")

            corrtrack = CorrTrack(
                window_size=self.window_size,
                basic_window=self.basic_window,
                window_step=self.window_step,
                n_vectors=n_vectors,
                n_lags=self.n_lags,
                grid_dimension=grid_dimension,
                cell_size=cell_stretch,
                seed=seed,
                seed_toggle=seed_toggle,
                freq_threshold=freq_threshold,
                corr_threshold=self.corr_threshold,
                neg_corr=self.neg_corr,
                preprocess=preprocess,
                exec="sequential",
                max_workers=self.max_workers,
                parallel_sketch=False,
                parallel_candidates=self.parallel_candidates,
                parallel_validation=False,
                track_min_dist=False,
                **feature_kwargs,
            )

            record["grid_max"] = corrtrack.grid_max
            record["cell_size"] = corrtrack.cell_size
            record["n_lags"] = corrtrack.n_lags
            record["basic_window"] = corrtrack.basic_window
            record["grid_dimension"] = corrtrack.grid_dimension
            record["exec_mode"] = "sequential"
            record["parallel_sketch"] = False
            record["parallel_candidates"] = corrtrack.parallel_candidates
            record["parallel_validation"] = False
            record["candidate_backend"] = getattr(
                corrtrack,
                "candidate_backend_effective",
                getattr(corrtrack, "candidate_backend", None),
            )
            record["candidate_bucket_width"] = corrtrack.candidate_bucket_width
            record["candidate_block_size_steps"] = corrtrack.candidate_block_size_steps
            record["candidate_block_index_dims"] = corrtrack.candidate_block_index_dims
            # (2026-07-06) Part 1 -- see docs/implementation_log.md.
            record["candidate_bound_dims"] = getattr(corrtrack, "candidate_bound_dims", None)
            record["candidate_bound_dim_selection"] = getattr(corrtrack, "candidate_bound_dim_selection", None)
            record["enable_block_ub_pruning"] = getattr(corrtrack, "enable_block_ub_pruning", None)
            record["enable_row_ub_pruning"] = getattr(corrtrack, "enable_row_ub_pruning", None)
            record["block_similarity_assignment"] = getattr(corrtrack, "block_similarity_assignment", None)
            record["max_open_blocks"] = getattr(corrtrack, "max_open_blocks", None)
            record["candidate_instinct_query_mode"] = getattr(corrtrack, "candidate_instinct_query_mode", None)
            record["candidate_instinct_top_k"] = getattr(corrtrack, "candidate_instinct_top_k", None)
            record["candidate_instinct_min_candidates"] = getattr(corrtrack, "candidate_instinct_min_candidates", None)
            record["candidate_instinct_entry_points"] = getattr(corrtrack, "candidate_instinct_entry_points", None)
            record.update(_candidate_runtime_record_fields(corrtrack))
            record["candidate_parallel_mode"] = corrtrack.candidate_parallel_mode
            record["candidate_key_mode"] = corrtrack.candidate_key_mode
            record["candidate_key_seed"] = corrtrack.candidate_key_seed
            record["candidate_lsh_radius"] = corrtrack.candidate_lsh_radius
            record["candidate_ann_m"] = corrtrack.candidate_ann_m
            record["candidate_ann_z"] = corrtrack.candidate_ann_z
            record["candidate_ann_ef"] = corrtrack.candidate_ann_ef
            record["numeric_rows"] = corrtrack.numeric_rows
            record.update(_hybrid_validation_record_fields(corrtrack))
            record["workers_total"] = 1
            record["workers_sketch"] = 1
            record["workers_candidates"] = corrtrack.n_candidate_nodes
            record["workers_validation"] = 1
            record["sketch_norm"] = corrtrack.sketch_norm

            t_sketch = time.perf_counter()
            partitions_by_start = self._proxy_partitions_for_corrtrack(corrtrack, reference)
            sketch_time = time.perf_counter() - t_sketch

            t_candidates = time.perf_counter()
            candidate_keys = set()
            search_stats_total = self._empty_proxy_search_stats()
            for anchor in reference.get("anchors", []):
                anchor_keys, anchor_search_stats = self._proxy_candidate_keys_for_anchor(
                    corrtrack,
                    partitions_by_start,
                    anchor,
                    feature_kwargs,
                    return_stats=True,
                )
                candidate_keys.update(anchor_keys)
                self._merge_proxy_search_stats(search_stats_total, anchor_search_stats)
            candidate_time = time.perf_counter() - t_candidates

            candidate_mask = np.zeros(int(reference["n_pairs"]), dtype=bool)
            key_to_indices = reference["key_to_indices"]
            for pair_key in candidate_keys:
                for row_idx in key_to_indices.get(self._normalize_window_pair_key(pair_key), ()):
                    candidate_mask[int(row_idx)] = True

            counts_by_anchor = self._proxy_counts_by_anchor(reference, candidate_mask)
            summary = self._proxy_bootstrap_summary(counts_by_anchor)
            counts = summary["counts"]
            metrics = summary["metrics"]
            truth_sign = reference["truth_sign"]
            truth = reference["truth"]
            tp_pos = int(np.sum(candidate_mask & (truth_sign > 0)))
            tp_neg = int(np.sum(candidate_mask & (truth_sign < 0)))
            gt_pos = int(np.sum(truth_sign > 0))
            gt_neg = int(np.sum(truth_sign < 0))

            record["cand_time_bf"] = 0.0
            record["val_time_bf"] = 0.0
            record["monit_time_bf"] = 0.0
            record["runtime_bf"] = 0.0
            record["artifact_time_bf"] = 0.0
            record["corr_w_bf"] = metrics["gt_total"]
            record["tested_w_bf"] = metrics["total"]
            record["cand_w_bf"] = metrics["total"]
            record["sk_time"] = sketch_time
            record["cand_time"] = candidate_time
            record["val_time"] = 0.0
            record["monit_time"] = 0.0
            record["runtime"] = time.perf_counter() - t0_total
            record["optim_bootstrap_time"] = float(summary.get("bootstrap_time", 0.0) or 0.0)
            record["optim_eval_time"] = record["runtime"]
            record["optim_search_time"] = max(self._to_float(record.get("runtime")) - record["optim_bootstrap_time"], 0.0)
            # (2026-07-05) Real, measured (not candidate-count-proxy) time for
            # the phases hyperopt actually runs. val_time/monit_time are never
            # measured here (hyperopt never runs full validation/monitoring),
            # so this intentionally excludes them -- see
            # docs/implementation_log.md, "hyperopt real-time tie-break".
            record["proxy_search_time_total"] = sketch_time + candidate_time
            record["artifact_time"] = 0.0
            record["corr_w"] = metrics["tp"]
            record["tested_w"] = metrics["cand_total"]
            record["cand_w"] = metrics["cand_total"]
            record["speedup"] = _safe_div(metrics["total"], metrics["cand_total"])
            record["speedup_ceil"] = record["speedup"]
            record["rel_speedup_eff"] = 1.0
            record["corr_prop"] = _safe_div(metrics["gt_total"], metrics["total"])
            record["waste_val_bf"] = _safe_div(metrics["total"], metrics["gt_total"])
            record["waste_val"] = _safe_div(metrics["cand_total"], metrics["tp"])
            record["rel_waste_red"] = _safe_div(record.get("waste_val_bf"), record.get("waste_val"))
            record["precision_pos"] = _signed_precision_from_ambiguous_fp(tp_pos, tp_neg, metrics["cand_total"])
            record["recall_pos"] = _safe_div(tp_pos, gt_pos)
            record["f1_pos"] = None
            record["precision_neg"] = _signed_precision_from_ambiguous_fp(tp_neg, tp_pos, metrics["cand_total"])
            record["recall_neg"] = _safe_div(tp_neg, gt_neg)
            record["f1_neg"] = None
            record["precision"] = metrics["precision"]
            record["recall"] = metrics["recall"]
            record["specificity"] = metrics["specificity"]
            record["recall_min"] = min(
                record["recall_pos"] if record["recall_pos"] is not None else metrics["recall"],
                record["recall_neg"] if record["recall_neg"] is not None else metrics["recall"],
            )
            record["f1"] = metrics["f1"]
            record["aucroc"] = None
            record["pr_auc"] = None
            record["proxy_n_pairs"] = metrics["total"]
            record["proxy_n_gt"] = metrics["gt_total"]
            record["proxy_candidate_rate"] = metrics["candidate_rate"]
            record.update(self._proxy_search_record_fields(metrics, search_stats_total))
            record.update(self._candidate_search_record_fields_from_stats(search_stats_total))
            record["proxy_gt_support"] = summary["gt_support"]
            record["proxy_underpowered"] = bool(summary["underpowered"])

            if self._proxy_bootstrap_active():
                for prefix, values in (
                    ("proxy_recall", summary.get("recall_ci")),
                    ("proxy_precision", summary.get("precision_ci")),
                    ("proxy_specificity", summary.get("specificity_ci")),
                    ("proxy_candidate_rate", summary.get("candidate_rate_ci")),
                ):
                    if values is None:
                        continue
                    record[f"{prefix}_lb"] = values[0]
                    record[f"{prefix}_med"] = values[1]
                    record[f"{prefix}_ub"] = values[2]
                record["proxy_recall_mean"] = summary.get("recall_mean")
                record["proxy_precision_mean"] = summary.get("precision_mean")
                record["proxy_specificity_mean"] = summary.get("specificity_mean")
                record["proxy_candidate_rate_mean"] = summary.get("candidate_rate_mean")
            else:
                record["proxy_recall_lb"] = metrics["recall"]
                record["proxy_recall_mean"] = metrics["recall"]
                record["proxy_recall_med"] = metrics["recall"]
                record["proxy_recall_ub"] = metrics["recall"]
                record["proxy_precision_lb"] = metrics["precision"]
                record["proxy_precision_mean"] = metrics["precision"]
                record["proxy_precision_med"] = metrics["precision"]
                record["proxy_precision_ub"] = metrics["precision"]
                record["proxy_specificity_lb"] = metrics["specificity"]
                record["proxy_specificity_mean"] = metrics["specificity"]
                record["proxy_specificity_med"] = metrics["specificity"]
                record["proxy_specificity_ub"] = metrics["specificity"]
                record["proxy_candidate_rate_lb"] = metrics["candidate_rate"]
                record["proxy_candidate_rate_mean"] = metrics["candidate_rate"]
                record["proxy_candidate_rate_med"] = metrics["candidate_rate"]
                record["proxy_candidate_rate_ub"] = metrics["candidate_rate"]

            record["status"] = "success"
            record["error"] = ""
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"Skipped proxy params={param_combo} due to error: {exc}")
            traceback.print_exc()

        return record

    def _run_proxy_anchor_cached_group(self, tasks, dataset_id):
        return None

    def _run_proxy_anchor_options(self, param_grid, output_csv, dataset_id):
        names = list(param_grid.keys())
        tasks = [
            (idx, dict(zip(names, vals)), dataset_id)
            for idx, vals in enumerate(itertools.product(*param_grid.values()))
        ]
        if os.path.exists(output_csv):
            os.remove(output_csv)

        try:
            total_tasks = len(tasks)
            completed = 0
            with CSVStreamWriter(output_csv, OPTIM_RESULT_COLUMNS) as writer:
                self._proxy_reference = self._prepare_proxy_anchor_reference()
                try:
                    grouped = defaultdict(list)
                    for task in tasks:
                        grouped[self._proxy_cache_group_key(task[1])].append(task)
                    task_records = {}
                    for group_tasks in grouped.values():
                        records = self._run_proxy_anchor_cached_group(group_tasks, dataset_id)
                        if records is None:
                            records = [self._run_corrtrack_proxy_anchor_mean_timed(task) for task in group_tasks]
                        for task, record in zip(group_tasks, records):
                            task_records[int(task[0])] = record
                    ordered_records = [task_records[idx] for idx in sorted(task_records)]

                    for record in ordered_records:
                        writer.write_row(_row_from_mapping(OPTIM_RESULT_COLUMNS, record))
                        completed += 1
                        status = record.get("status", "unknown")
                        eval_mode = record.get("proxy_eval_mode") or self.proxy_eval_mode
                        print(
                            f"[Optim][ProxyAnchor:{eval_mode}] "
                            f"Completed {completed}/{total_tasks} ({status})"
                        )
                finally:
                    self._proxy_reference = None
        finally:
            self._proxy_reference = None

    def _run_options(self, param_grid, output_csv, dataset_id):
        self._run_proxy_anchor_options(param_grid, output_csv, dataset_id)

    def _apply_proxy_anchor_selection(self, metrics, target_recall):
        metrics = metrics.copy()
        if metrics.empty or "hyperopt_strategy" not in metrics.columns:
            return metrics, None

        proxy_mask = metrics["hyperopt_strategy"].fillna("").astype(str).str.lower() == "proxy_anchor"
        if not proxy_mask.any():
            return metrics, None

        for column in ("proxy_feasible", "proxy_selected", "proxy_sample_adequate", "proxy_underpowered"):
            if column not in metrics.columns:
                metrics[column] = False
            metrics[column] = metrics[column].fillna(False).astype(bool)
        if "proxy_warning" not in metrics.columns:
            metrics["proxy_warning"] = ""
        else:
            metrics["proxy_warning"] = metrics["proxy_warning"].fillna("").astype(str)

        candidates = metrics[proxy_mask].copy()
        if "status" in candidates.columns:
            candidates = candidates[candidates["status"] == "success"].copy()
        if candidates.empty:
            return metrics, None

        def numeric_column(name):
            if name in candidates.columns:
                return pd.to_numeric(candidates[name], errors="coerce")
            return pd.Series(np.nan, index=candidates.index, dtype=float)

        recall_lb = numeric_column("proxy_recall_lb")
        recall_mean = numeric_column("proxy_recall_mean")
        recall_med = numeric_column("proxy_recall_med")
        recall_point = numeric_column("recall")
        recall_score = recall_lb.where(recall_lb.notna(), recall_point)
        recall_rank = recall_mean.where(recall_mean.notna(), recall_med.where(recall_med.notna(), recall_point))

        candidate_rate_med = numeric_column("proxy_candidate_rate_med")
        candidate_rate_mean = numeric_column("proxy_candidate_rate_mean")
        candidate_rate_point = numeric_column("proxy_candidate_rate")
        candidate_rate_rank = candidate_rate_mean.where(
            candidate_rate_mean.notna(),
            candidate_rate_med.where(candidate_rate_med.notna(), candidate_rate_point),
        )
        search_objective = numeric_column("proxy_search_objective_rate")
        search_similarity_work = numeric_column("proxy_search_similarity_work_rate")
        search_dot_work = numeric_column("proxy_search_dot_work_rate")
        search_index_work = numeric_column("proxy_search_index_rate")
        search_unique_pre_dot = numeric_column("proxy_search_unique_pre_dot_pairs")
        search_reject = numeric_column("proxy_search_cascade_reject_rate")
        search_objective_rank = search_objective.where(search_objective.notna(), candidate_rate_rank)
        search_similarity_work_rank = search_similarity_work.where(search_similarity_work.notna(), candidate_rate_rank)
        search_dot_work_rank = search_dot_work.where(search_dot_work.notna(), candidate_rate_rank)
        search_index_work_rank = search_index_work.where(search_index_work.notna(), search_dot_work_rank)
        search_unique_pre_dot_rank = search_unique_pre_dot.where(search_unique_pre_dot.notna(), search_dot_work_rank)
        search_reject_rank = search_reject.where(search_reject.notna(), 0.0)

        precision_mean = numeric_column("proxy_precision_mean")
        precision_med = numeric_column("proxy_precision_med")
        precision_point = numeric_column("precision")
        precision_rank = precision_mean.where(precision_mean.notna(), precision_med.where(precision_med.notna(), precision_point))

        specificity_mean = numeric_column("proxy_specificity_mean")
        specificity_med = numeric_column("proxy_specificity_med")
        specificity_point = numeric_column("specificity")
        specificity_rank = specificity_mean.where(
            specificity_mean.notna(),
            specificity_med.where(specificity_med.notna(), specificity_point),
        )

        gt_support = numeric_column("proxy_gt_support")
        min_gt = float(self.proxy_bootstrap_min_gt_events or 0)
        underpowered = gt_support.fillna(0.0) < min_gt
        feasible_mask = recall_score >= float(target_recall)

        metrics.loc[candidates.index, "proxy_underpowered"] = underpowered.astype(bool)
        metrics.loc[candidates.index, "proxy_feasible"] = feasible_mask.fillna(False).astype(bool)

        time_total_rank = numeric_column("proxy_search_time_total")

        ranking = candidates.copy()
        ranking["_recall_score"] = recall_score
        ranking["_recall_rank"] = recall_rank
        ranking["_candidate_rate_rank"] = candidate_rate_rank
        ranking["_search_similarity_work_rank"] = search_similarity_work_rank
        ranking["_search_objective_rank"] = search_objective_rank
        ranking["_search_dot_work_rank"] = search_dot_work_rank
        ranking["_search_index_work_rank"] = search_index_work_rank
        ranking["_search_unique_pre_dot_rank"] = search_unique_pre_dot_rank
        ranking["_search_reject_rank"] = search_reject_rank
        ranking["_precision_rank"] = precision_rank
        ranking["_specificity_rank"] = specificity_rank
        ranking["_underpowered_rank"] = underpowered.astype(int)
        ranking["_time_rank"] = time_total_rank

        feasible = ranking[feasible_mask.fillna(False)]
        if feasible.empty:
            max_recall = ranking["_recall_rank"].max()
            feasible = ranking[ranking["_recall_rank"] == max_recall].copy()
            metrics.loc[feasible.index, "proxy_warning"] = (
                metrics.loc[feasible.index, "proxy_warning"].astype(str)
                + "; no proxy configuration met target recall; selected among max-recall rows"
            ).str.strip("; ")

        # (2026-07-05) Candidate-count minimization remains the dominant
        # objective (unchanged "far" branch below for configs outside
        # tolerance of the best achievable candidate rate). But among
        # configs already close to that best rate, real measured execution
        # time (sk_time+cand_time -- genuinely measured, not a candidate-
        # count proxy) is a more meaningful tie-break than further
        # candidate-rate refinements, since search cost does not scale
        # purely with output candidate count (a backend's index-maintenance/
        # search cost can vary independently of how many candidates it
        # ultimately surfaces -- see docs/implementation_log.md, "hyperopt
        # real-time tie-break", and this session's sorted_arrays_bs/blocked_lazy
        # finding: 6x the search time for byte-identical candidate output).
        common_sort_cols = [
            "_search_similarity_work_rank",
            "_search_dot_work_rank",
            "_search_index_work_rank",
            "_search_unique_pre_dot_rank",
            "_search_objective_rank",
            "_search_reject_rank",
            "_precision_rank",
            "_specificity_rank",
            "_recall_rank",
            "cand_w",
        ]
        common_ascending = [True, True, True, True, True, False, False, False, False, True]

        not_underpowered = feasible[feasible["_underpowered_rank"] == 0]
        reference_pool = not_underpowered if not not_underpowered.empty else feasible
        best_candidate_rate = reference_pool["_candidate_rate_rank"].min()
        tolerance = float(self.proxy_candidate_rate_close_tolerance)
        if pd.notna(best_candidate_rate):
            close_mask = feasible["_candidate_rate_rank"] <= best_candidate_rate * (1.0 + tolerance)
            close_mask = close_mask.fillna(False)
        else:
            close_mask = pd.Series(False, index=feasible.index)
        near_best = feasible[close_mask]
        far = feasible[~close_mask]

        near_best_sorted = near_best.sort_values(
            ["_underpowered_rank", "_time_rank"] + common_sort_cols,
            ascending=[True, True] + common_ascending,
            na_position="last",
        )
        far_sorted = far.sort_values(
            ["_underpowered_rank", "_candidate_rate_rank"] + common_sort_cols,
            ascending=[True, True] + common_ascending,
            na_position="last",
        )
        best = pd.concat([near_best_sorted, far_sorted]).head(1)
        if best.empty:
            return metrics, None

        best_idx = best.index[0]
        metrics.loc[:, "proxy_selected"] = False
        metrics.loc[best_idx, "proxy_selected"] = True
        metrics.loc[best_idx, "proxy_sample_adequate"] = bool(
            metrics.loc[best_idx, "proxy_feasible"] and not metrics.loc[best_idx, "proxy_underpowered"]
        )
        return metrics, metrics.loc[[best_idx]]

    @staticmethod
    def _merge_metrics_for_output(all_metrics, metrics):
        if metrics is None or metrics.empty:
            return all_metrics
        for column in metrics.columns:
            if column not in all_metrics.columns:
                series = metrics[column]
                if pd.api.types.is_bool_dtype(series):
                    all_metrics[column] = pd.Series(pd.NA, index=all_metrics.index, dtype="boolean")
                elif pd.api.types.is_numeric_dtype(series):
                    all_metrics[column] = np.nan
                else:
                    all_metrics[column] = pd.Series([None] * len(all_metrics), index=all_metrics.index, dtype=object)
        for column in metrics.columns:
            series = metrics[column]
            if pd.api.types.is_bool_dtype(series) and not pd.api.types.is_bool_dtype(all_metrics[column]):
                all_metrics[column] = all_metrics[column].astype("boolean")
            elif pd.api.types.is_object_dtype(series) and not pd.api.types.is_object_dtype(all_metrics[column]):
                all_metrics[column] = all_metrics[column].astype(object)
            all_metrics.loc[metrics.index, column] = series
        return all_metrics

    def _proxy_anchor_expansion_reason_for_best(self, proxy_best, target_recall):
        if proxy_best is None or proxy_best.empty:
            return ""
        row = proxy_best.iloc[0]
        recall_mean = _to_float_safe(row.get("proxy_recall_mean"))
        if recall_mean is None or not np.isfinite(recall_mean):
            recall_mean = _to_float_safe(row.get("proxy_recall_med"))
        if recall_mean is None or not np.isfinite(recall_mean):
            recall_mean = _to_float_safe(row.get("recall"))
        recall_lb = _to_float_safe(row.get("proxy_recall_lb"))
        target = _to_float_safe(target_recall)
        if target is None:
            target = 0.0
        if recall_mean is None or not np.isfinite(recall_mean) or recall_mean < target:
            return ""
        underpowered = _coerce_to_bool(row.get("proxy_underpowered"), default=False)
        gt_support = _to_float_safe(row.get("proxy_gt_support"))
        if gt_support is None:
            gt_support = _to_float_safe(row.get("proxy_n_gt"))
        if (
            underpowered
            or (
                gt_support is not None
                and np.isfinite(gt_support)
                and gt_support < float(self.proxy_bootstrap_min_gt_events)
            )
        ):
            return "expanded_after_underpowered_proxy_recall"
        if recall_lb is not None and np.isfinite(recall_lb) and recall_lb < target:
            return "expanded_after_uncertain_proxy_recall_lower_bound"
        return ""

    def _get_proxy_anchor_optim_params(
        self,
        param_grid,
        output_csv,
        dataset_id,
        run,
        target_recall,
    ):
        rounds_done = 0
        expansion_reason = ""
        while True:
            self._proxy_anchor_expansion_iteration = int(rounds_done)
            self._proxy_anchor_expansion_reason = expansion_reason
            if run:
                self._run_proxy_anchor_options(param_grid, output_csv, dataset_id)

            delimiter = _detect_csv_delimiter(output_csv, default=",")
            all_metrics = pd.read_csv(output_csv, sep=delimiter)
            metrics = all_metrics
            if "status" in metrics.columns:
                metrics = metrics[metrics["status"] == "success"]
            if metrics.empty:
                raise ValueError(f"No successful parameter combinations found in {output_csv}")

            metrics, proxy_best = self._apply_proxy_anchor_selection(metrics, target_recall)
            all_metrics = self._merge_metrics_for_output(all_metrics, metrics)
            all_metrics.to_csv(output_csv, sep=delimiter, index=False)
            if proxy_best is None or proxy_best.empty:
                return None

            reason = self._proxy_anchor_expansion_reason_for_best(proxy_best, target_recall)
            if (
                run
                and self.proxy_adaptive_anchor_enabled
                and reason
                and rounds_done < self.proxy_anchor_expand_max_rounds
                and int(self.proxy_anchor_count_requested) < int(self.proxy_max_anchor_count)
            ):
                next_count = int(math.ceil(float(self.proxy_anchor_count_requested) * self.proxy_anchor_expand_factor))
                next_count = min(int(self.proxy_max_anchor_count), max(next_count, int(self.proxy_anchor_count_requested) + 1))
                if next_count > int(self.proxy_anchor_count_requested):
                    print(
                        "[Optim][ProxyAnchor] "
                        f"{reason}; increasing anchors "
                        f"{self.proxy_anchor_count_requested} -> {next_count}"
                    )
                    self.proxy_anchor_count_requested = next_count
                    rounds_done += 1
                    expansion_reason = reason
                continue
            return proxy_best

    def get_optim_params(
        self,
        param_grid,
        output_csv,
        dataset_id,
        run=True,
        target_recall=0.95,
        recall_fallback_near_ratio=0.98,
        speedup_near_ratio=0.98,
    ):
        output_csv = output_csv + "_" + self.alg + ".csv"
        proxy_best = self._get_proxy_anchor_optim_params(
            param_grid,
            output_csv,
            dataset_id,
            run,
            target_recall,
        )
        if proxy_best is not None and not proxy_best.empty:
            return proxy_best
        raise ValueError(f"No selectable proxy-anchor parameter combination found in {output_csv}")
        #return self.ground_truth, self.runtime_bf, CorrTrack_HyperOptim._skyline_query(metrics, ref_metrics) 

class CorrTrack_compare:
    def __init__(self,train_data,test_data,ids,window_size,window_step,basic_window,n_lags,corr_threshold,param_grid,recall_by_window,neg_corr,corr_val,algs=None, exec="parallel", max_workers=0, sketch_norm="z", candidate_bucket_width=None, candidate_block_size_steps=None, candidate_block_index_dims=None, candidate_similarity="l2", candidate_cosine_threshold=None, candidate_parallel_mode="recent_shards", candidate_key_mode="first", candidate_key_seed=None, candidate_lsh_radius=None, candidate_ann_m=None, candidate_ann_z=None, candidate_ann_ef=None, candidate_bound_dims=None, candidate_bound_dim_selection="variance", enable_block_ub_pruning=False, enable_row_ub_pruning=False, block_similarity_assignment=False, max_open_blocks=4, candidate_instinct_query_mode="hybrid", candidate_instinct_top_k=256, candidate_instinct_min_candidates=64, candidate_instinct_entry_points=8, numeric_rows=True, verbose=False, testing=False, parallel_sketch=None, parallel_candidates=None, parallel_validation=None, monitor=True, track_min_dist=True, tuning_mode="sampling"):
        
        self.neg_corr = neg_corr
        self.corr_val = _coerce_to_bool(corr_val, default=True)
        self.monitor = _coerce_to_bool(monitor)
        self.track_min_dist = _coerce_to_bool(track_min_dist, default=True)
        if not self.corr_val:
            self.monitor = False
            self.track_min_dist = False
        self.sketch_norm = sketch_norm or "z"
        self.candidate_bucket_width = _to_float_safe(candidate_bucket_width)
        block_steps = _to_int_safe(candidate_block_size_steps)
        if block_steps is None or block_steps <= 0:
            block_steps = 32
        self.candidate_block_size_steps = int(block_steps)
        block_dims = _to_int_safe(candidate_block_index_dims)
        if block_dims is None or block_dims <= 0:
            block_dims = 1
        self.candidate_block_index_dims = int(block_dims)
        # (2026-07-06) Part 1 -- see docs/implementation_log.md.
        bound_dims_val = _to_int_safe(candidate_bound_dims)
        self.candidate_bound_dims = max(0, bound_dims_val) if bound_dims_val is not None else 0
        self.candidate_bound_dim_selection = str(candidate_bound_dim_selection or "variance").lower()
        self.enable_block_ub_pruning = bool(enable_block_ub_pruning)
        self.enable_row_ub_pruning = bool(enable_row_ub_pruning)
        self.block_similarity_assignment = bool(block_similarity_assignment)
        self.max_open_blocks = max(1, _to_int_safe(max_open_blocks) or 4)
        # (2026-07-06) InstinctIndex candidate_backend params -- see
        # docs/implementation_log.md. Approximate, opt-in backend only.
        self.candidate_instinct_query_mode = str(candidate_instinct_query_mode or "hybrid").lower()
        self.candidate_instinct_top_k = max(1, _to_int_safe(candidate_instinct_top_k) or 256)
        self.candidate_instinct_min_candidates = max(1, _to_int_safe(candidate_instinct_min_candidates) or 64)
        self.candidate_instinct_entry_points = max(1, _to_int_safe(candidate_instinct_entry_points) or 8)
        self.candidate_similarity = _resolve_candidate_similarity(candidate_similarity, default="l2")
        self.candidate_cosine_threshold = _to_float_safe(candidate_cosine_threshold)
        self.candidate_parallel_mode = "recent_shards"
        self.candidate_key_mode = _resolve_candidate_key_mode(candidate_key_mode, default="first")
        self.candidate_key_seed = _to_int_safe(candidate_key_seed)
        self.candidate_lsh_radius = max(0, min(3, int(_to_int_safe(candidate_lsh_radius) or 0)))
        self.candidate_ann_m = max(1, int(_to_int_safe(candidate_ann_m) or 16))
        self.candidate_ann_z = max(1, int(_to_int_safe(candidate_ann_z) or 256))
        self.candidate_ann_ef = max(1, int(_to_int_safe(candidate_ann_ef) or max(64, self.candidate_ann_z)))
        self.numeric_rows = True
        self.verbose = _coerce_to_bool(verbose)
        self.testing = _coerce_to_bool(testing)

        # Parameters data
        self.train_data = train_data
        self.test_data = test_data
        self.ids = ids

        self.window_size = window_size

        self.n_lags = n_lags
        self.corr_threshold = corr_threshold
        self.param_grid = param_grid

        self.ground_truth = None
        self.recall_by_window = True

        self.algs = algs or ["nD"]
        self.tuning_mode = "sampling"
        self.exec = _normalize_exec_mode(exec, default="thread")
        self.max_workers = max_workers
        parallel_default = self.exec == "thread"
        self.parallel_sketch = _resolve_parallel_flag(parallel_sketch, False)
        self.parallel_candidates = _resolve_parallel_flag(parallel_candidates, False)
        self.parallel_validation = _resolve_parallel_flag(parallel_validation, parallel_default)

        self.corrtrack_bf = CorrTrack(window_size=self.window_size,basic_window=basic_window,window_step=window_step,n_vectors=1,n_lags=self.n_lags,
                                    grid_dimension=1,cell_size=1,seed=None,seed_toggle=None,
                                    corr_threshold=self.corr_threshold,neg_corr=self.neg_corr,preprocess=False,
                                    exec=self.exec,max_workers=self.max_workers,
                                    sketch_norm=self.sketch_norm,candidate_parallel_mode=self.candidate_parallel_mode,
                                    candidate_bucket_width=self.candidate_bucket_width,
                                    candidate_block_size_steps=self.candidate_block_size_steps,
                                    candidate_block_index_dims=self.candidate_block_index_dims,
                                    candidate_similarity=self.candidate_similarity,
                                    candidate_cosine_threshold=self.candidate_cosine_threshold,
                                    candidate_lsh_radius=self.candidate_lsh_radius,
                                    candidate_ann_m=self.candidate_ann_m,
                                    candidate_ann_z=self.candidate_ann_z,
                                    candidate_ann_ef=self.candidate_ann_ef,
                                    parallel_sketch=self.parallel_sketch,parallel_candidates=self.parallel_candidates,parallel_validation=self.parallel_validation,
                                    track_min_dist=self.track_min_dist,numeric_rows=self.numeric_rows)
        
        self.window_step = self.corrtrack_bf.window_step
        self.basic_window = self.corrtrack_bf.basic_window
        self.n_lags = self.corrtrack_bf.n_lags

        # runtime stats placeholders
        self.candidate_time_bf = 0.0
        self.validation_time_bf = 0.0
        self.monitor_time_bf = 0.0
        self.runtime_bf = 0.0
        self.correlated_bf = 0.0
        self.tested_bf = 0.0
        self.total_bf = 0.0
        self.pair_min_dist_bf = None
        self.correlated_w = 0.0
        self.tested_w = 0.0
        self.total_w = 0.0
        self.candidate_search_index_candidates = 0
        self.candidate_search_valid_index_candidates = 0
        self.candidate_search_unique_index_candidates = 0
        self.candidate_search_duplicate_index_candidates = 0
        self.candidate_search_unique_pre_dot_pairs = 0
        self.candidate_search_duplicate_pre_dot_pairs = 0
        self.candidate_search_after_coord = 0
        self.candidate_search_partial_checks = 0
        self.candidate_search_after_partial = 0
        self.candidate_search_after_similarity = 0
        self.candidate_search_dot_checks = 0
        self.candidate_search_distance_checks = 0
        # (2026-07-06) Part 1 metrics -- see docs/implementation_log.md.
        self.candidate_search_blocks_visited = 0
        self.candidate_search_blocks_pruned_by_ub = 0
        self.candidate_search_rows_in_surviving_blocks = 0
        self.candidate_search_dot_checks_saved_by_row_ub = 0
        # (2026-07-06) InstinctIndex metrics -- see docs/implementation_log.md,
        # "InstinctIndex: experimental approximate graph backend".
        self.candidate_search_instinct_visited_nodes = 0
        self.candidate_search_instinct_visited_live_nodes = 0
        self.candidate_search_instinct_dead_nodes_skipped = 0
        self.candidate_search_instinct_edges_scanned = 0
        self.candidate_search_instinct_candidates_returned = 0
        self.candidate_search_instinct_threshold_candidates = 0
        self.candidate_search_instinct_topk_candidates = 0
        self.candidate_search_instinct_queries_with_too_few_live_nodes = 0
        self.candidate_search_instinct_query_time = 0.0
        self.candidate_search_instinct_insert_time = 0.0
        self.candidate_search_instinct_num_nodes_total = 0
        self.candidate_search_instinct_num_nodes_alive = 0
        self.candidate_search_instinct_best_score_seen = 0.0
        self.candidate_search_instinct_mean_score_returned = 0.0
        self.candidate_search_instinct_dead_node_ratio = 0.0

    def _run_is_parallel(self) -> bool:
        return self.exec != "sequential"
    
    def _outer_iter(self, items, worker_fn, unordered=False):
        items = list(items)
        inner_parallel = self._run_is_parallel()
        if inner_parallel or self.exec == "sequential" or not _HAS_DASK or delayed is None:
            for it in items:
                yield worker_fn(it)
            return

        max_workers = self.max_workers or max(1, (os.cpu_count() or 1) - 1)
        max_workers = max(1, max_workers)
        scheduler_get = dask_threaded_get
        if scheduler_get is None:
            for it in items:
                yield worker_fn(it)
            return

        scheduler = partial(scheduler_get, num_workers=max_workers)
        try:
            tasks = [delayed(worker_fn)(it) for it in items]
            results = compute(*tasks, scheduler=scheduler)
        except Exception as exc:
            if getattr(self, "verbose", False):
                print(f"[outer_iter] Dask fallback to sequential due to: {exc}")
            for it in items:
                yield worker_fn(it)
            return

        for res in results:
            yield res
    
    def _mode_run(self,mode,alg,path,prefix,nodes,seed,seed_toggle,n_vectors,grid_dimension,cell_size,grid_max,freq_threshold,preprocess,feature_overrides=None):
        length_data = self.test_data.shape[1]
        data_stream = self.test_data

        # Instantiating corrtrack objects
        overrides = {
            "candidate_block_size_steps": self.candidate_block_size_steps,
            "candidate_block_index_dims": self.candidate_block_index_dims,
            "candidate_bound_dims": self.candidate_bound_dims,
            "candidate_bound_dim_selection": self.candidate_bound_dim_selection,
            "enable_block_ub_pruning": self.enable_block_ub_pruning,
            "enable_row_ub_pruning": self.enable_row_ub_pruning,
            "block_similarity_assignment": self.block_similarity_assignment,
            "max_open_blocks": self.max_open_blocks,
            "candidate_bucket_width": self.candidate_bucket_width,
            "candidate_parallel_mode": self.candidate_parallel_mode,
            "candidate_key_mode": self.candidate_key_mode,
            "candidate_key_seed": self.candidate_key_seed,
            "candidate_lsh_radius": self.candidate_lsh_radius,
            "candidate_ann_m": self.candidate_ann_m,
            "candidate_ann_z": self.candidate_ann_z,
            "candidate_ann_ef": self.candidate_ann_ef,
            "candidate_similarity": self.candidate_similarity,
            "candidate_cosine_threshold": self.candidate_cosine_threshold,
            "numeric_rows": self.numeric_rows,
        }
        overrides.update(feature_overrides or {})
        corrtrack = CorrTrack(window_size=self.window_size,basic_window=self.basic_window,window_step=self.window_step,n_vectors=n_vectors,n_lags=self.n_lags,
                            grid_dimension=grid_dimension,cell_size=cell_size,seed=seed,seed_toggle=seed_toggle,
                            freq_threshold=freq_threshold,corr_threshold=self.corr_threshold,neg_corr=self.neg_corr,preprocess=preprocess,
                            exec=self.exec,max_workers=nodes,
                            parallel_sketch=self.parallel_sketch,parallel_candidates=self.parallel_candidates,parallel_validation=self.parallel_validation,
                            track_min_dist=self.track_min_dist,
                            **overrides)
        if mode == "main":
            #Running
            start_time = time.time()
            for start in range(0, length_data - self.window_step + 1, self.window_step):
                chunk = data_stream[:, start:start + self.window_step]
                corrtrack.run(
                    chunk,
                    self.ids,
                    verbose=self.verbose,
                    testing=self.testing,
                    corr_val=self.corr_val,
                    monitor=self.monitor,
                )
            end_time = time.time()
            runtime = end_time - start_time
            runtime_parts = (corrtrack.sketch_time,corrtrack.candidate_time,corrtrack.validation_time,corrtrack.monitor_time)
            self.tested_w = corrtrack.tested_candidates
            self.correlated_w = corrtrack.validated_candidates
            self.total_w = corrtrack.total_candidates
            self.candidate_search_index_candidates = getattr(corrtrack, "candidate_search_index_candidates", 0)
            self.candidate_search_valid_index_candidates = getattr(corrtrack, "candidate_search_valid_index_candidates", 0)
            self.candidate_search_unique_index_candidates = getattr(corrtrack, "candidate_search_unique_index_candidates", 0)
            self.candidate_search_duplicate_index_candidates = getattr(corrtrack, "candidate_search_duplicate_index_candidates", 0)
            self.candidate_search_unique_pre_dot_pairs = getattr(corrtrack, "candidate_search_unique_pre_dot_pairs", 0)
            self.candidate_search_duplicate_pre_dot_pairs = getattr(corrtrack, "candidate_search_duplicate_pre_dot_pairs", 0)
            self.candidate_search_after_coord = getattr(corrtrack, "candidate_search_after_coord", 0)
            self.candidate_search_partial_checks = getattr(corrtrack, "candidate_search_partial_checks", 0)
            self.candidate_search_after_partial = getattr(corrtrack, "candidate_search_after_partial", 0)
            self.candidate_search_after_similarity = getattr(corrtrack, "candidate_search_after_similarity", 0)
            self.candidate_search_dot_checks = getattr(corrtrack, "candidate_search_dot_checks", 0)
            self.candidate_search_distance_checks = getattr(corrtrack, "candidate_search_distance_checks", 0)
            self.candidate_search_blocks_visited = getattr(corrtrack, "candidate_search_blocks_visited", 0)
            self.candidate_search_blocks_pruned_by_ub = getattr(corrtrack, "candidate_search_blocks_pruned_by_ub", 0)
            self.candidate_search_rows_in_surviving_blocks = getattr(corrtrack, "candidate_search_rows_in_surviving_blocks", 0)
            self.candidate_search_dot_checks_saved_by_row_ub = getattr(corrtrack, "candidate_search_dot_checks_saved_by_row_ub", 0)
            self.candidate_search_instinct_visited_nodes = getattr(corrtrack, "candidate_search_instinct_visited_nodes", 0)
            self.candidate_search_instinct_visited_live_nodes = getattr(corrtrack, "candidate_search_instinct_visited_live_nodes", 0)
            self.candidate_search_instinct_dead_nodes_skipped = getattr(corrtrack, "candidate_search_instinct_dead_nodes_skipped", 0)
            self.candidate_search_instinct_edges_scanned = getattr(corrtrack, "candidate_search_instinct_edges_scanned", 0)
            self.candidate_search_instinct_candidates_returned = getattr(corrtrack, "candidate_search_instinct_candidates_returned", 0)
            self.candidate_search_instinct_threshold_candidates = getattr(corrtrack, "candidate_search_instinct_threshold_candidates", 0)
            self.candidate_search_instinct_topk_candidates = getattr(corrtrack, "candidate_search_instinct_topk_candidates", 0)
            self.candidate_search_instinct_queries_with_too_few_live_nodes = getattr(corrtrack, "candidate_search_instinct_queries_with_too_few_live_nodes", 0)
            self.candidate_search_instinct_query_time = getattr(corrtrack, "candidate_search_instinct_query_time", 0.0)
            self.candidate_search_instinct_insert_time = getattr(corrtrack, "candidate_search_instinct_insert_time", 0.0)
            self.candidate_search_instinct_num_nodes_total = getattr(corrtrack, "candidate_search_instinct_num_nodes_total", 0)
            self.candidate_search_instinct_num_nodes_alive = getattr(corrtrack, "candidate_search_instinct_num_nodes_alive", 0)
            self.candidate_search_instinct_best_score_seen = getattr(corrtrack, "candidate_search_instinct_best_score_seen", 0.0)
            self.candidate_search_instinct_mean_score_returned = getattr(corrtrack, "candidate_search_instinct_mean_score_returned", 0.0)
            self.candidate_search_instinct_dead_node_ratio = getattr(corrtrack, "candidate_search_instinct_dead_node_ratio", 0.0)
        elif mode == "bf":
            start_time = time.time()
            for start in range(0, length_data - self.window_step + 1, self.window_step):
                chunk = data_stream[:, start:start + self.window_step]
                corrtrack.run_bf(
                    chunk,
                    self.ids,
                    verbose=self.verbose,
                    testing=self.testing,
                    corr_val=True,
                    monitor=self.monitor,
                )
            end_time = time.time()
            runtime = end_time - start_time
            runtime_parts = (0,corrtrack.candidate_time,corrtrack.validation_time,corrtrack.monitor_time)
            self.candidate_time_bf = corrtrack.candidate_time
            self.validation_time_bf = corrtrack.validation_time
            self.monitor_time_bf = corrtrack.monitor_time
            self.runtime_bf = runtime
            self.correlated_bf = corrtrack.validated_candidates
            self.tested_bf = corrtrack.tested_candidates
            self.total_bf = corrtrack.total_candidates
            self.pair_min_dist_bf = corrtrack.pair_min_dist

        artifact_time = 0.0

        _artifact_path = os.path.join(path, prefix + "_correlated.csv")
        _t0 = time.time()
        corrtrack._save_correlated(_artifact_path)
        artifact_time += time.time() - _t0

        _artifact_path = os.path.join(path, prefix + "_max_lag_correlated.csv")
        _t0 = time.time()
        corrtrack._save_max_lag_correlated(_artifact_path)
        artifact_time += time.time() - _t0

        _artifact_path = os.path.join(path, prefix + "_status.csv")
        _t0 = time.time()
        corrtrack._save_monitor_status(_artifact_path)
        artifact_time += time.time() - _t0

        _artifact_path = os.path.join(path, prefix + "_anomalies.csv")
        _t0 = time.time()
        corrtrack._save_anomalies(_artifact_path)
        artifact_time += time.time() - _t0
        #print(f"Time elapsed: {runtime:.2f} sec")

        #if mode == "main":
        #    corrtrack.print_sketch_hist()

        #Get flags based on brute-force
        if self.recall_by_window:
            corr_flags = corrtrack.correlated
        else:
            corr_flags = corrtrack.get_correlation_flags(length_data,0)

        if mode == "bf":
            self.artifact_time_bf = artifact_time

        return runtime_parts, runtime, artifact_time, corr_flags

    def _optim(self,hyper_param_csv,dataset_id,run,alg,target_recall=0.95):
        corrtrack_ho = CorrTrack_optimize(
            self.train_data,
            self.ids,
            self.window_size,
            self.window_step,
            self.n_lags,
            self.corr_threshold,
            self.recall_by_window,
            alg,
            self.neg_corr,
            self.corr_val,
            exec=self.exec,
            verbose=self.verbose,
            testing=self.testing,
            candidate_bucket_width=self.candidate_bucket_width,
            candidate_block_size_steps=self.candidate_block_size_steps,
            candidate_block_index_dims=self.candidate_block_index_dims,
            candidate_similarity=self.candidate_similarity,
            candidate_cosine_threshold=self.candidate_cosine_threshold,
            candidate_parallel_mode=self.candidate_parallel_mode,
            candidate_key_mode=self.candidate_key_mode,
            candidate_key_seed=self.candidate_key_seed,
            candidate_lsh_radius=self.candidate_lsh_radius,
            candidate_ann_m=self.candidate_ann_m,
            candidate_ann_z=self.candidate_ann_z,
            candidate_ann_ef=self.candidate_ann_ef,
            track_min_dist=self.track_min_dist,
        )

        return corrtrack_ho.get_optim_params(self.param_grid,hyper_param_csv,dataset_id,run,target_recall)

    def _load_maxlag_csv(self, csv_path):
        records = {}
        if not csv_path or not os.path.exists(csv_path):
            return records

        try:
            with open(csv_path, newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    id1 = row.get("id1")
                    id2 = row.get("id2")
                    time1 = row.get("time1")
                    t1_idx = row.get("t1_index")
                    corr_val = row.get("max_corr")
                    if id1 is None or id2 is None or corr_val is None:
                        continue
                    key_time = self._normalize_time_key(time1, t1_idx)
                    if key_time is None:
                        continue
                    try:
                        value = float(corr_val)
                    except (TypeError, ValueError):
                        continue
                    records[(id1, id2, key_time)] = value
        except FileNotFoundError:
            return {}

        return records

    def _compute_maxlag_metrics(self, bf_csv, candidate_csv):
        bf_iter = self._iter_maxlag_rows(bf_csv)
        candidate_iter = self._iter_maxlag_rows(candidate_csv)

        bf_row = next(bf_iter, None)
        candidate_row = next(candidate_iter, None)

        tp = 0
        bf_total = 0
        candidate_total = 0
        diff_count = 0
        diff_mean = 0.0
        diff_m2 = 0.0

        while bf_row is not None or candidate_row is not None:
            if candidate_row is None:
                bf_total += 1
                bf_row = next(bf_iter, None)
                continue
            if bf_row is None:
                candidate_total += 1
                candidate_row = next(candidate_iter, None)
                continue

            bf_key, bf_val = bf_row
            candidate_key, candidate_val = candidate_row

            if bf_key == candidate_key:
                bf_total += 1
                candidate_total += 1
                tp += 1
                diff = bf_val - candidate_val
                if math.isclose(diff, 0.0, rel_tol=1e-10, abs_tol=1e-12):
                    diff = 0.0
                else:
                    diff = round(diff, 12)
                diff_count, diff_mean, diff_m2 = _welford_update(diff_count, diff_mean, diff_m2, diff)
                bf_row = next(bf_iter, None)
                candidate_row = next(candidate_iter, None)
            elif bf_key < candidate_key:
                bf_total += 1
                bf_row = next(bf_iter, None)
            else:
                candidate_total += 1
                candidate_row = next(candidate_iter, None)

        fp = max(candidate_total - tp, 0)
        fn = max(bf_total - tp, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        mean_diff = float("nan")
        std_diff = float("nan")
        if diff_count > 0:
            mean_diff = float(diff_mean)
            std_diff = float(math.sqrt(diff_m2 / diff_count))

        return precision, recall, f1, mean_diff, std_diff

    @staticmethod
    def _normalize_record(row):
        normalized = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            normalized[key] = value
        return normalized

    @staticmethod
    def _parse_time_key_value(value):
        if value in (None, "", "nan", "NaN"):
            return None
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                return None
            return value.astype("datetime64[ns]")
        if isinstance(value, datetime.datetime):
            return np.datetime64(value)
        if isinstance(value, datetime.date):
            return np.datetime64(datetime.datetime.combine(value, datetime.time()))
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if math.isnan(value):
                return None
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            lowered = stripped.lower()
            if lowered in {"nan", "nat"}:
                return None
            try:
                if "." in stripped:
                    numeric = float(stripped)
                    if math.isnan(numeric):
                        return None
                    return int(numeric)
                return int(stripped)
            except ValueError:
                pass
            try:
                dt = np.datetime64(stripped)
            except ValueError:
                try:
                    numeric = float(stripped)
                except ValueError:
                    return stripped
                else:
                    if math.isnan(numeric):
                        return None
                    return int(numeric)
            else:
                if np.isnat(dt):
                    return None
                return dt.astype("datetime64[ns]")
        return value

    @staticmethod
    def _normalize_time_key(primary, fallback=None):
        key = CorrTrack_compare._parse_time_key_value(primary)
        if key is not None:
            return key
        if fallback is not None:
            return CorrTrack_compare._parse_time_key_value(fallback)
        return None

    @staticmethod
    def _parse_pair_min_dist(value):
        if value in (None, "", "nan", "None"):
            return None
        if isinstance(value, tuple):
            return value
        try:
            return tuple(ast.literal_eval(value))
        except (ValueError, SyntaxError, TypeError):
            return None

    @staticmethod
    def _coerce_time_value(value):
        if value in (None, "", "nan"):
            return value
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if math.isnan(value):
                return value
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return stripped
            try:
                if "." in stripped:
                    return float(stripped)
                return int(stripped)
            except ValueError:
                return stripped
        return value

    def _canonical_series_id(self, value):
        if value is None:
            return None
        ids = getattr(self, "ids", None)
        if ids is not None:
            try:
                ids_source = id(ids)
            except TypeError:
                ids_source = None
            ids_list = getattr(self, "_canonical_ids_list", None)
            id_lookup = getattr(self, "_canonical_id_lookup", None)
            if getattr(self, "_canonical_ids_source", None) != ids_source:
                try:
                    ids_list = list(ids)
                except TypeError:
                    ids_list = None
                id_lookup = {}
                if ids_list is not None:
                    for idx, id_value in enumerate(ids_list):
                        try:
                            id_lookup[id_value] = idx
                        except TypeError:
                            pass
                        id_lookup[str(id_value)] = idx
                self._canonical_ids_source = ids_source
                self._canonical_ids_list = ids_list
                self._canonical_id_lookup = id_lookup
            if ids_list is not None:
                if id_lookup:
                    try:
                        direct_idx = id_lookup.get(value)
                    except TypeError:
                        direct_idx = None
                    if direct_idx is None:
                        direct_idx = id_lookup.get(str(value))
                    if direct_idx is not None:
                        return int(direct_idx)
                idx = None
                if isinstance(value, (np.integer, int)):
                    idx = int(value)
                elif isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        try:
                            as_float = float(stripped)
                            if as_float.is_integer():
                                idx = int(as_float)
                        except (TypeError, ValueError):
                            idx = None
                if idx is not None and 0 <= idx < len(ids_list):
                    return int(idx)
        return value

    def _canonical_window_key(self, id1, id2, time1, time2):
        return _normalize_window_metric_key(
            self._canonical_series_id(id1),
            self._canonical_series_id(id2),
            time1,
            time2,
            self.window_size,
        )

    def _canonical_pair_min_dist_key(self, pair):
        if pair is None or not isinstance(pair, tuple) or len(pair) < 5:
            return None
        return self._canonical_window_key(pair[0], pair[1], pair[2], pair[3])

    def _correlated_csv_path(self, artifact_prefix):
        if not artifact_prefix:
            return None
        if artifact_prefix.endswith("_correlated.csv"):
            return artifact_prefix
        return f"{artifact_prefix}_correlated.csv"

    def _artifact_chunk_manifest_path(self, artifact_prefix):
        if not artifact_prefix:
            return None
        if artifact_prefix.endswith("_artifact_chunks.json"):
            return artifact_prefix
        if artifact_prefix.endswith("_correlated.csv"):
            artifact_prefix = artifact_prefix[: -len("_correlated.csv")]
        return f"{artifact_prefix}_artifact_chunks.json"

    def _load_correlated_codebook_path(self, codebook_path):
        if not codebook_path or not os.path.exists(codebook_path):
            return {}
        delimiter = _detect_csv_delimiter(codebook_path, default=CSV_DELIMITER)
        codebook = {}
        with open(codebook_path, newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                code = row.get("code")
                id_value = row.get("id")
                if code not in (None, "") and id_value is not None:
                    codebook[str(code)] = id_value
        return codebook

    def _anomalies_csv_path(self, artifact_prefix):
        if not artifact_prefix:
            return None
        if artifact_prefix.endswith("_anomalies.csv"):
            return artifact_prefix
        return f"{artifact_prefix}_anomalies.csv"

    def _iter_correlated_rows(self, artifact_prefix):
        csv_path = self._correlated_csv_path(artifact_prefix)
        input_paths = []
        codebook = {}
        if csv_path and os.path.exists(csv_path):
            input_paths = [csv_path]
            codebook = _load_correlated_codebook(csv_path)
        else:
            manifest_path = self._artifact_chunk_manifest_path(artifact_prefix)
            if not manifest_path or not os.path.exists(manifest_path):
                return
            with open(manifest_path, encoding="utf-8") as file:
                manifest = json.load(file)
            input_paths = [
                path
                for path in manifest.get("chunks", {}).get("correlated", [])
                if path and os.path.exists(path)
            ]
            codebook_path = manifest.get("codebook_path")
            if codebook_path:
                codebook = self._load_correlated_codebook_path(codebook_path)
            elif csv_path:
                codebook = _load_correlated_codebook(csv_path)
        if not input_paths:
            return

        pending_key = None
        pending_corr = None
        for input_path in input_paths:
            delimiter = _detect_csv_delimiter(input_path, default=CSV_DELIMITER)
            with open(input_path, newline="") as file:
                reader = csv.DictReader(file, delimiter=delimiter)
                for row in reader:
                    id1 = row.get("id1")
                    id2 = row.get("id2")
                    if id1 is None:
                        code = row.get("id1_code")
                        id1 = codebook.get(str(code), code)
                    if id2 is None:
                        code = row.get("id2_code")
                        id2 = codebook.get(str(code), code)
                    if id1 is None or id2 is None:
                        continue
                    time1 = self._coerce_time_value(row.get("time1_idx"))
                    if time1 in (None, "", "nan"):
                        time1 = self._coerce_time_value(row.get("time1"))
                    time2 = self._coerce_time_value(row.get("time2_idx"))
                    if time2 in (None, "", "nan"):
                        time2 = self._coerce_time_value(row.get("time2"))
                    corr_val = row.get("corr")
                    try:
                        corr_val = float(corr_val)
                    except (TypeError, ValueError):
                        continue
                    key = self._canonical_window_key(id1, id2, time1, time2)
                    if pending_key is None:
                        pending_key, pending_corr = key, corr_val
                        continue
                    if key == pending_key:
                        if abs(corr_val) > abs(pending_corr):
                            pending_corr = corr_val
                        continue
                    yield pending_key, pending_corr
                    pending_key, pending_corr = key, corr_val
        if pending_key is not None:
            yield pending_key, pending_corr

    def _iter_maxlag_rows(self, csv_path):
        if not csv_path or not os.path.exists(csv_path):
            return
        delimiter = _detect_csv_delimiter(csv_path, default=CSV_DELIMITER)
        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            pending_key = None
            pending_corr = None
            for row in reader:
                id1 = row.get("id1")
                id2 = row.get("id2")
                corr_val = row.get("max_corr")
                if id1 is None or id2 is None or corr_val is None:
                    continue
                id1 = self._canonical_series_id(id1)
                id2 = self._canonical_series_id(id2)
                key_time = self._normalize_time_key(row.get("time1"), row.get("t1_index"))
                if key_time is None:
                    continue
                try:
                    corr_val = float(corr_val)
                except (TypeError, ValueError):
                    continue
                key = (id1, id2, key_time)
                if pending_key is None:
                    pending_key, pending_corr = key, corr_val
                    continue
                if key == pending_key:
                    if abs(corr_val) > abs(pending_corr):
                        pending_corr = corr_val
                    continue
                yield pending_key, pending_corr
                pending_key, pending_corr = key, corr_val
            if pending_key is not None:
                yield pending_key, pending_corr

    def _stream_window_metrics_from_artifacts(
        self,
        ground_truth_prefix,
        predicted_prefix,
        pair_min_dist=None,
        total_pairs_bf=None,
        use_ground_truth_sign_for_tp=False,
    ):
        metrics = _empty_stream_metrics(include_sign=True)

        def collect_rows(prefix):
            collected = {}
            for key, corr_val in self._iter_correlated_rows(prefix):
                prev = collected.get(key)
                if prev is None or abs(corr_val) > abs(prev):
                    collected[key] = corr_val
            return collected

        gt_rows = collect_rows(ground_truth_prefix)
        pred_rows = collect_rows(predicted_prefix)

        gt_total = len(gt_rows)
        pred_total = len(pred_rows)
        gt_pos = sum(1 for value in gt_rows.values() if value > 0)
        gt_neg = gt_total - gt_pos
        pred_pos = sum(1 for value in pred_rows.values() if value > 0)
        pred_neg = pred_total - pred_pos
        tp_keys = gt_rows.keys() & pred_rows.keys()
        tp = len(tp_keys)
        tp_pos = tp_neg = 0

        recall_min_key = self._canonical_pair_min_dist_key(pair_min_dist)
        recall_min_hit = recall_min_key is not None and recall_min_key in pred_rows

        for key in tp_keys:
            gt_val = gt_rows[key]
            pred_val = pred_rows[key]
            if use_ground_truth_sign_for_tp:
                if gt_val > 0:
                    tp_pos += 1
                else:
                    tp_neg += 1
            else:
                if gt_val > 0 and pred_val > 0:
                    tp_pos += 1
                if gt_val <= 0 and pred_val <= 0:
                    tp_neg += 1

        fp = max(pred_total - tp, 0)
        precision, recall = _safe_prec_recall_from_counts(tp, pred_total, gt_total)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if use_ground_truth_sign_for_tp:
            precision_pos = _signed_precision_from_ambiguous_fp(tp_pos, tp_neg, pred_total)
            recall_pos = tp_pos / gt_pos if gt_pos > 0 else 0.0
        else:
            precision_pos, recall_pos = _safe_prec_recall_from_counts(tp_pos, pred_pos, gt_pos)
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0.0

        if use_ground_truth_sign_for_tp:
            precision_neg = _signed_precision_from_ambiguous_fp(tp_neg, tp_pos, pred_total)
            recall_neg = tp_neg / gt_neg if gt_neg > 0 else 0.0
        else:
            precision_neg, recall_neg = _safe_prec_recall_from_counts(tp_neg, pred_neg, gt_neg)
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0

        specificity = 0.0
        try:
            total_pairs = int(total_pairs_bf) if total_pairs_bf is not None else None
        except (TypeError, ValueError):
            total_pairs = None
        if total_pairs is not None:
            negatives = max(total_pairs - gt_total, 0)
            if negatives > 0:
                tn = max(negatives - fp, 0)
                specificity = tn / negatives

        metrics.update({
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "recall_min": None if recall_min_key is None else int(recall_min_hit),
            "precision_pos": precision_pos,
            "recall_pos": recall_pos,
            "f1_score_pos": f1_pos,
            "precision_neg": precision_neg,
            "recall_neg": recall_neg,
            "f1_score_neg": f1_neg,
        })
        return metrics

    def _iter_anomaly_interval_groups(self, artifact_prefix, total_time, tolerance=None):
        csv_path = self._anomalies_csv_path(artifact_prefix)
        if not csv_path or not os.path.exists(csv_path):
            return
        if tolerance is None:
            tolerance = self.window_size
        delimiter = _detect_csv_delimiter(csv_path, default=CSV_DELIMITER)
        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            current_key = None
            active = False
            last_out_time = None
            start_time = None
            intervals = []
            last_event = None
            for row in reader:
                id1 = row.get("id1")
                id2 = row.get("id2")
                lag_val = row.get("lag")
                if id1 is None or id2 is None or lag_val in (None, ""):
                    continue
                try:
                    lag = int(float(lag_val))
                except (TypeError, ValueError):
                    continue
                key = _normalize_pair_lag_key(id1, id2, lag)
                time_val = self._normalize_time_key(row.get("time_idx"), row.get("time"))
                if time_val is None:
                    continue
                if isinstance(time_val, (np.datetime64, datetime.datetime, datetime.date)):
                    continue
                marker = _label_to_marker(row.get("anomaly") or row.get("marker"))

                if current_key is not None and key != current_key:
                    if active and start_time is not None:
                        intervals.append((int(start_time), int(total_time)))
                    yield current_key, intervals
                    current_key = key
                    active = False
                    last_out_time = None
                    start_time = None
                    intervals = []
                elif current_key is None:
                    current_key = key

                time_idx = int(time_val)
                if marker == 1:
                    event = (key, time_idx, marker)
                    if event == last_event:
                        continue
                    last_event = event
                    if not active:
                        if last_out_time is None or (time_idx - last_out_time) > tolerance:
                            active = True
                            start_time = time_idx
                elif marker == -1:
                    event = (key, time_idx, marker)
                    if event == last_event:
                        continue
                    last_event = event
                    if active and start_time is not None:
                        intervals.append((int(start_time), int(time_idx)))
                        active = False
                        last_out_time = time_idx
                else:
                    last_event = (key, time_idx, marker)

            if current_key is not None:
                if active and start_time is not None:
                    intervals.append((int(start_time), int(total_time)))
                yield current_key, intervals

    def _stream_timestamp_metrics_from_artifacts(self, ground_truth_prefix, predicted_prefix, total_time):
        metrics = _empty_stream_metrics(include_sign=True)
        if total_time <= 0:
            return metrics

        gt_iter = self._iter_anomaly_interval_groups(ground_truth_prefix, total_time)
        pred_iter = self._iter_anomaly_interval_groups(predicted_prefix, total_time)
        gt_group = next(gt_iter, None)
        pred_group = next(pred_iter, None)

        gt_total = 0
        pred_total = 0
        tp = 0
        union_keys = 0

        while gt_group is not None or pred_group is not None:
            if pred_group is None:
                union_keys += 1
                gt_total += _intervals_total_length(gt_group[1])
                gt_group = next(gt_iter, None)
                continue
            if gt_group is None:
                union_keys += 1
                pred_total += _intervals_total_length(pred_group[1])
                pred_group = next(pred_iter, None)
                continue

            gt_key, gt_intervals = gt_group
            pred_key, pred_intervals = pred_group

            if gt_key == pred_key:
                union_keys += 1
                gt_len = _intervals_total_length(gt_intervals)
                pred_len = _intervals_total_length(pred_intervals)
                overlap = _intervals_overlap_length(gt_intervals, pred_intervals)
                gt_total += gt_len
                pred_total += pred_len
                tp += overlap
                gt_group = next(gt_iter, None)
                pred_group = next(pred_iter, None)
            elif gt_key < pred_key:
                union_keys += 1
                gt_total += _intervals_total_length(gt_intervals)
                gt_group = next(gt_iter, None)
            else:
                union_keys += 1
                pred_total += _intervals_total_length(pred_intervals)
                pred_group = next(pred_iter, None)

        fp = max(pred_total - tp, 0)
        fn = max(gt_total - tp, 0)
        total_points = max(union_keys * int(total_time), 0)
        tn = max(total_points - tp - fp - fn, 0)
        precision, recall = _safe_prec_recall_from_counts(tp, pred_total, gt_total)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics.update({
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "precision_pos": float("nan"),
            "recall_pos": float("nan"),
            "f1_score_pos": float("nan"),
            "precision_neg": float("nan"),
            "recall_neg": float("nan"),
            "f1_score_neg": float("nan"),
            "recall_min": None,
        })
        return metrics

    def _load_correlated_dict(self, artifact_prefix):
        if not artifact_prefix:
            return {}

        csv_path = artifact_prefix
        if not csv_path.endswith("_correlated.csv"):
            csv_path = f"{artifact_prefix}_correlated.csv"
        if not os.path.exists(csv_path):
            return {}

        records = {}
        try:
            with open(csv_path, newline="") as file:
                delimiter = _detect_csv_delimiter(csv_path, default=CSV_DELIMITER)
                reader = csv.DictReader(file, delimiter=delimiter)
                for row in reader:
                    id1 = row.get("id1")
                    id2 = row.get("id2")
                    if id1 is None or id2 is None:
                        continue
                    time1 = self._coerce_time_value(row.get("time1_idx"))
                    if time1 in (None, "", "nan"):
                        time1 = self._coerce_time_value(row.get("time1"))
                    time2 = self._coerce_time_value(row.get("time2_idx"))
                    if time2 in (None, "", "nan"):
                        time2 = self._coerce_time_value(row.get("time2"))
                    corr_val = row.get("corr")
                    try:
                        corr_val = float(corr_val)
                    except (TypeError, ValueError):
                        continue
                    key = (id1, id2, time1, time2, self.window_size)
                    records[key] = corr_val
        except FileNotFoundError:
            return {}

        return records

    def _load_run_record(self, csv_path, dataset_id, run_kind=None, alg=None):
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing run record: {csv_path}")

        delimiter = _detect_csv_delimiter(csv_path, default=CSV_DELIMITER)
        df = pd.read_csv(csv_path, sep=delimiter)
        subset = df[df["dataset_id"] == dataset_id] if "dataset_id" in df.columns else df
        if run_kind and "run_kind" in subset.columns:
            subset = subset[subset["run_kind"] == run_kind]
        if alg and "alg" in subset.columns:
            subset = subset[subset["alg"] == alg]
        if subset.empty:
            raise ValueError(f"No matching record found for dataset_id={dataset_id} (run_kind={run_kind}, alg={alg}) in {csv_path}")
        row = subset.iloc[-1].to_dict()
        return self._normalize_record(row)

    @staticmethod
    def _format_metric(value):
        if value in (None, ""):
            return ""
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            if math.isnan(value):
                return "nan"
            if value == 0.0:
                return "0.0000"
            abs_val = abs(value)
            if abs_val != 0.0 and (abs_val < 1e-4 or abs_val >= 1e4):
                return f"{value:.3e}"
            return f"{value:.4f}"
        return str(value)

    def _build_comparison_row(
        self,
        dataset_id,
        record,
        bf_record,
        metrics,
        runtime_bf,
        artifact_time_bf,
        runtime,
        artifact_time,
        speedup,
        maxlag_metrics,
    ):
        precision_pos = metrics["precision_pos"]
        recall_pos = metrics["recall_pos"]
        f1_pos = metrics["f1_score_pos"]
        precision_neg = metrics["precision_neg"]
        recall_neg = metrics["recall_neg"]
        f1_neg = metrics["f1_score_neg"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        specificity = metrics["specificity"]
        recall_min = metrics["recall_min"]
        f1 = metrics["f1_score"]
        aucroc = metrics["aucroc"]
        pr_auc = metrics["pr_auc"]

        maxlag_precision, maxlag_recall, maxlag_f1, maxlag_diff_mean, maxlag_diff_std = maxlag_metrics

        if record.get("cell_stretch") in (None, "", "nan"):
            record["cell_stretch"] = _resolve_cell_stretch(
                record.get("cell_stretch"),
                record.get("cell_size"),
                record.get("corr_threshold"),
                record.get("n_vectors"),
                record.get("grid_dimension"),
                default=None,
                full_vector_candidates=True,
            )

        def as_int_str(value, default="0"):
            if value in (None, "", "nan"):
                return default
            try:
                return str(int(float(value)))
            except (TypeError, ValueError):
                return str(value)

        def as_optional_int(value):
            if value in (None, "", "nan"):
                return ""
            try:
                return str(int(float(value)))
            except (TypeError, ValueError):
                return str(value)

        fmt = self._format_metric
        corr_w_bf = bf_record.get("correlated")
        cand_w_bf = bf_record.get("total_candidates")
        corr_w = record.get("correlated")
        cand_w = record.get("total_candidates")
        speedup_ceil = _compute_speedup_ceil(
            bf_record.get("cand_time"),
            bf_record.get("val_time"),
            bf_record.get("monit_time"),
            record.get("sk_time"),
            record.get("cand_time"),
            corr_w_bf,
            cand_w_bf,
        )
        rel_speedup_eff = _safe_div(speedup, speedup_ceil)
        corr_prop = _safe_div(corr_w_bf, cand_w_bf)
        waste_val_bf = _safe_div(cand_w_bf, corr_w_bf)
        waste_val = _safe_div(cand_w, corr_w)
        rel_waste_red = _safe_div(waste_val_bf, waste_val)

        row = [
            dataset_id,
            record.get("mode", "main") or "main",
            record.get("alg", ""),
            record.get("optim", ""),
            bf_record.get("baseline_mode", ""),
            as_int_str(record.get("n_ts")),
            as_int_str(record.get("n_w")),
            as_int_str(record.get("total_w")),
            as_optional_int(record.get("mem_w")),
            as_optional_int(record.get("nodes")),
            str(record.get("exec_mode", "")),
            str(record.get("parallel_sketch", "")),
            str(record.get("parallel_candidates", "")),
            str(record.get("parallel_validation", "")),
            as_optional_int(record.get("workers_total")),
            as_optional_int(record.get("workers_sketch")),
            as_optional_int(record.get("workers_candidates")),
            as_optional_int(record.get("workers_validation")),
            as_optional_int(record.get("window_size")),
            as_optional_int(record.get("window_step")),
            as_optional_int(record.get("basic_window")),
            as_optional_int(record.get("n_lags")),
            as_optional_int(record.get("seed")),
            as_optional_int(record.get("seed_toggle")),
            str(record.get("preprocess")),
            str(record.get("sketch_norm")),
            str(record.get("candidate_backend")),
            fmt(record.get("candidate_bucket_width")),
            as_optional_int(record.get("candidate_block_size_steps")),
            as_optional_int(record.get("candidate_block_index_dims")),
            as_optional_int(record.get("candidate_bound_dims")),
            str(record.get("candidate_bound_dim_selection", "")),
            str(record.get("enable_block_ub_pruning", "")),
            str(record.get("enable_row_ub_pruning", "")),
            str(record.get("block_similarity_assignment", "")),
            as_optional_int(record.get("max_open_blocks")),
            str(record.get("candidate_instinct_query_mode", "")),
            as_optional_int(record.get("candidate_instinct_top_k")),
            as_optional_int(record.get("candidate_instinct_min_candidates")),
            as_optional_int(record.get("candidate_instinct_entry_points")),
            str(record.get("candidate_similarity", "")),
            fmt(record.get("candidate_cosine_threshold")),
            str(record.get("candidate_parallel_mode", "")),
            str(record.get("candidate_key_mode", "")),
            as_optional_int(record.get("candidate_key_seed")),
            as_optional_int(record.get("candidate_lsh_radius")),
            as_optional_int(record.get("candidate_ann_m")),
            as_optional_int(record.get("candidate_ann_z")),
            as_optional_int(record.get("candidate_ann_ef")),
            str(record.get("hybrid_validation", "")),
            fmt(record.get("hybrid_validation_min_repeat_rate")),
            fmt(record.get("hybrid_validation_disable_rate")),
            fmt(record.get("hybrid_validation_ema_alpha")),
            as_optional_int(record.get("hybrid_validation_min_candidates")),
            as_optional_int(record.get("hybrid_validation_attempts")),
            as_optional_int(record.get("hybrid_validation_hits")),
            as_optional_int(record.get("hybrid_validation_steps_active")),
            str(record.get("numeric_rows", "")),
            fmt(record.get("corr_threshold")),
            fmt(record.get("grid_max")),
            fmt(record.get("cell_stretch")),
            fmt(record.get("cell_size")),
            as_optional_int(record.get("n_vectors")),
            as_optional_int(record.get("grid_dimension")),
            fmt(record.get("freq_threshold")),
            fmt(bf_record.get("cand_time")),
            fmt(bf_record.get("val_time")),
            fmt(bf_record.get("monit_time")),
            fmt(runtime_bf),
            fmt(artifact_time_bf),
            fmt(record.get("sk_time")),
            fmt(record.get("cand_time")),
            fmt(record.get("val_time")),
            fmt(record.get("monit_time")),
            fmt(runtime),
            fmt(artifact_time),
            fmt(speedup),
            fmt(speedup_ceil),
            fmt(rel_speedup_eff),
            as_int_str(corr_w_bf),
            as_int_str(corr_w),
            as_int_str(bf_record.get("tested")),
            as_int_str(record.get("tested")),
            as_int_str(cand_w_bf),
            as_int_str(cand_w),
            as_optional_int(record.get("candidate_search_index_candidates")),
            as_optional_int(record.get("candidate_search_valid_index_candidates")),
            as_optional_int(record.get("candidate_search_unique_index_candidates")),
            as_optional_int(record.get("candidate_search_duplicate_index_candidates")),
            as_optional_int(record.get("candidate_search_unique_pre_dot_pairs")),
            as_optional_int(record.get("candidate_search_duplicate_pre_dot_pairs")),
            as_optional_int(record.get("candidate_search_after_coord")),
            as_optional_int(record.get("candidate_search_partial_checks")),
            as_optional_int(record.get("candidate_search_after_partial")),
            as_optional_int(record.get("candidate_search_after_similarity")),
            as_optional_int(record.get("candidate_search_dot_checks")),
            as_optional_int(record.get("candidate_search_distance_checks")),
            as_optional_int(record.get("candidate_search_blocks_visited")),
            as_optional_int(record.get("candidate_search_blocks_pruned_by_ub")),
            as_optional_int(record.get("candidate_search_rows_in_surviving_blocks")),
            as_optional_int(record.get("candidate_search_dot_checks_saved_by_row_ub")),
            as_optional_int(record.get("candidate_search_instinct_visited_nodes")),
            as_optional_int(record.get("candidate_search_instinct_visited_live_nodes")),
            as_optional_int(record.get("candidate_search_instinct_dead_nodes_skipped")),
            as_optional_int(record.get("candidate_search_instinct_edges_scanned")),
            as_optional_int(record.get("candidate_search_instinct_candidates_returned")),
            as_optional_int(record.get("candidate_search_instinct_threshold_candidates")),
            as_optional_int(record.get("candidate_search_instinct_topk_candidates")),
            as_optional_int(record.get("candidate_search_instinct_queries_with_too_few_live_nodes")),
            fmt(record.get("candidate_search_instinct_query_time")),
            fmt(record.get("candidate_search_instinct_insert_time")),
            as_optional_int(record.get("candidate_search_instinct_num_nodes_total")),
            as_optional_int(record.get("candidate_search_instinct_num_nodes_alive")),
            fmt(record.get("candidate_search_instinct_best_score_seen")),
            fmt(record.get("candidate_search_instinct_mean_score_returned")),
            fmt(record.get("candidate_search_instinct_dead_node_ratio")),
            fmt(corr_prop),
            fmt(waste_val_bf),
            fmt(waste_val),
            fmt(rel_waste_red),
            fmt(precision_pos),
            fmt(recall_pos),
            fmt(f1_pos),
            fmt(precision_neg),
            fmt(recall_neg),
            fmt(f1_neg),
            fmt(precision),
            fmt(recall),
            fmt(specificity),
            fmt(recall_min),
            fmt(f1),
            fmt(aucroc),
            fmt(pr_auc),
            fmt(maxlag_precision),
            fmt(maxlag_recall),
            fmt(maxlag_f1),
            fmt(maxlag_diff_mean),
            fmt(maxlag_diff_std),
        ]
        return _filter_comparison_row(row)

    def compare_from_artifacts(
        self,
        dataset_id,
        bf_run_csv,
        corrtrack_run_files,
        output_csv,
        delete_main_artifacts_after_compare=False,
    ):
        bf_record = self._load_run_record(bf_run_csv, dataset_id, run_kind="bf")
        bf_prefix = bf_record.get("artifact_path")
        self.pair_min_dist_bf = self._parse_pair_min_dist(bf_record.get("pair_min_dist"))

        def to_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        self.candidate_time_bf = to_float(bf_record.get("cand_time"))
        self.validation_time_bf = to_float(bf_record.get("val_time"))
        self.monitor_time_bf = to_float(bf_record.get("monit_time"))
        self.runtime_bf = to_float(bf_record.get("runtime"))
        self.artifact_time_bf = to_float(bf_record.get("artifact_time"))
        self.correlated_bf = int(float(bf_record.get("correlated", 0) or 0))
        self.tested_bf = int(float(bf_record.get("tested", 0) or 0))
        self.total_bf = int(float(bf_record.get("total_candidates", 0) or 0))

        total_time = 0
        try:
            n_w = int(float(bf_record.get("n_w", 0) or 0))
            window_step = int(float(bf_record.get("window_step", 0) or 0))
            window_size = int(float(bf_record.get("window_size", self.window_size) or self.window_size))
            if n_w > 0 and window_step > 0 and window_size > 0:
                total_time = (n_w - 1) * window_step + window_size
        except (TypeError, ValueError):
            total_time = 0

        bf_maxlag_csv = f"{bf_prefix}_max_lag_correlated.csv" if bf_prefix else None

        metric_keys = (
            "precision_pos",
            "recall_pos",
            "f1_score_pos",
            "precision_neg",
            "recall_neg",
            "f1_score_neg",
            "precision",
            "recall",
            "specificity",
            "recall_min",
            "f1_score",
            "aucroc",
            "pr_auc",
        )

        def build_empty_metrics():
            return {key: math.nan for key in metric_keys}

        results = []
        cleanup_prefixes = set()
        if isinstance(corrtrack_run_files, dict):
            run_iter = corrtrack_run_files.items()
        else:
            run_iter = corrtrack_run_files

        for alg, run_csv in run_iter:
            record = self._load_run_record(run_csv, dataset_id, alg=alg)
            prefix = record.get("artifact_path")
            if self.recall_by_window:
                metrics = self._stream_window_metrics_from_artifacts(
                    bf_prefix,
                    prefix,
                    pair_min_dist=self.pair_min_dist_bf,
                    total_pairs_bf=self.tested_bf,
                    use_ground_truth_sign_for_tp=bool(self.neg_corr and not self.corr_val),
                )
            else:
                metrics = self._stream_timestamp_metrics_from_artifacts(
                    bf_prefix,
                    prefix,
                    total_time,
                )

            runtime = to_float(record.get("runtime"))
            speedup = (self.runtime_bf / runtime) if runtime else float("inf")

            artifact_time = to_float(record.get("artifact_time"))

            maxlag_metrics = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
            if bf_maxlag_csv and prefix and os.path.exists(bf_maxlag_csv):
                candidate_maxlag_csv = f"{prefix}_max_lag_correlated.csv"
                if os.path.exists(candidate_maxlag_csv):
                    maxlag_metrics = self._compute_maxlag_metrics(bf_maxlag_csv, candidate_maxlag_csv)

            row = self._build_comparison_row(
                dataset_id,
                record,
                bf_record,
                metrics,
                self.runtime_bf,
                self.artifact_time_bf,
                runtime,
                artifact_time,
                speedup,
                maxlag_metrics,
            )
            results.append(row)
            if prefix:
                cleanup_prefixes.add(prefix)

        base_dir = os.path.dirname(os.path.abspath(bf_run_csv))
        filcorr_run_csv = os.path.join(base_dir, "filcorr_run.csv")
        filcorr_maxlag_csv = os.path.join(base_dir, "filcorr_max_lag_correlated.csv")

        if os.path.exists(filcorr_run_csv):
            try:
                filcorr_record = self._load_run_record(filcorr_run_csv, dataset_id, run_kind="filcorr")
            except (FileNotFoundError, ValueError):
                filcorr_record = None

            if filcorr_record is not None:
                filcorr_record = dict(filcorr_record)

                prefix = filcorr_record.get("artifact_path")
                if prefix == bf_prefix:
                    prefix = None

                if prefix:
                    if self.recall_by_window:
                        fil_metrics = self._stream_window_metrics_from_artifacts(
                            bf_prefix,
                            prefix,
                            pair_min_dist=self.pair_min_dist_bf,
                            total_pairs_bf=self.tested_bf,
                            use_ground_truth_sign_for_tp=False,
                        )
                    else:
                        fil_metrics = self._stream_timestamp_metrics_from_artifacts(
                            bf_prefix,
                            prefix,
                            total_time,
                        )
                else:
                    fil_metrics = build_empty_metrics()

                runtime = to_float(filcorr_record.get("runtime"))
                speedup = (self.runtime_bf / runtime) if runtime else float("inf")
                artifact_time = to_float(filcorr_record.get("artifact_time"))

                maxlag_metrics = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
                if bf_maxlag_csv and os.path.exists(bf_maxlag_csv) and os.path.exists(filcorr_maxlag_csv):
                    maxlag_metrics = self._compute_maxlag_metrics(bf_maxlag_csv, filcorr_maxlag_csv)

                filcorr_record["mode"] = "filcorr"
                filcorr_record["alg"] = "filcorr"
                filcorr_record.setdefault("optim", "filcorr")

                row = self._build_comparison_row(
                    dataset_id,
                    filcorr_record,
                    bf_record,
                    fil_metrics,
                    self.runtime_bf,
                    self.artifact_time_bf,
                    runtime,
                    artifact_time,
                    speedup,
                    maxlag_metrics,
                )
                results.append(row)

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(COMPARISON_COLUMNS)
            writer.writerows(results)

        if _coerce_to_bool(delete_main_artifacts_after_compare, default=False):
            for prefix in sorted(cleanup_prefixes):
                _remove_artifact_files(prefix)

        return results

    def _parallel_mode_run(self,args):
        dataset_id, mode, alg, path, prefix, bst, runtime_bf, corr_flags_bf = args
        n_vectors = _to_int_safe(bst.get("n_vectors"))
        grid_dimension = _to_int_safe(bst.get("grid_dimension"))
        cell_stretch = _resolve_cell_stretch(
            bst.get("cell_stretch"),
            bst.get("cell_size"),
            self.corr_threshold,
            n_vectors,
            grid_dimension,
            full_vector_candidates=True,
        )
        if cell_stretch is None or cell_stretch <= 0.0:
            cell_stretch = 1.0

        bst["n_vectors"] = n_vectors
        bst["grid_dimension"] = grid_dimension
        bst["cell_stretch"] = cell_stretch

        feature_kwargs = _extract_feature_overrides(bst)

        runtime_parts, runtime, artifact_time, corr_flags = self._mode_run(
            mode, alg, path, prefix,
            bst["nodes"], bst["seed"], bst["seed_toggle"],
            n_vectors, grid_dimension,
            cell_stretch, bst["grid_max"],
            None, bst["preprocess"],
            feature_kwargs,
        )
        speedup = runtime_bf / runtime if runtime else float("inf")
        metrics = CorrTrack.compute_metrics_bf(
            corr_flags,
            corr_flags_bf,
            self.recall_by_window,
            self.pair_min_dist_bf,
            total_pairs_bf=self.tested_bf,
            use_ground_truth_sign_for_tp=bool(self.neg_corr and not self.corr_val),
        )
        bf_artifact_time = getattr(self, "artifact_time_bf", 0.0)

        bf_maxlag_csv = os.path.join(path, "bf_max_lag_correlated.csv")
        candidate_maxlag_csv = os.path.join(path, f"{prefix}_max_lag_correlated.csv")
        if os.path.exists(bf_maxlag_csv) and os.path.exists(candidate_maxlag_csv):
            maxlag_precision, maxlag_recall, maxlag_f1, maxlag_diff_mean, maxlag_diff_std = self._compute_maxlag_metrics(
                bf_maxlag_csv,
                candidate_maxlag_csv,
            )
        else:
            maxlag_precision = float("nan")
            maxlag_recall = float("nan")
            maxlag_f1 = float("nan")
            maxlag_diff_mean = float("nan")
            maxlag_diff_std = float("nan")

        def _format_float(value):
            if isinstance(value, float) and math.isnan(value):
                return "nan"
            if value == 0.0:
                return "0.0000"
            abs_val = abs(value)
            if abs_val != 0.0 and (abs_val < 1e-4 or abs_val >= 1e4):
                return f"{value:.3e}"
            return f"{value:.4f}"

        def _format_optional_float(value):
            if value in (None, ""):
                return ""
            return _format_float(float(value))

        speedup_ceil = _compute_speedup_ceil(
            self.candidate_time_bf,
            self.validation_time_bf,
            self.monitor_time_bf,
            runtime_parts[0],
            runtime_parts[1],
            self.correlated_bf,
            self.total_bf,
        )
        rel_speedup_eff = _safe_div(speedup, speedup_ceil)
        corr_prop = _safe_div(self.correlated_bf, self.total_bf)
        waste_val_bf = _safe_div(self.total_bf, self.correlated_bf)
        waste_val = _safe_div(self.total_w, self.correlated_w)
        rel_waste_red = _safe_div(waste_val_bf, waste_val)

        n_ts = self.test_data.shape[0]-1
        n_w = (np.floor((self.test_data.shape[1]-self.window_size)/self.window_step) + 1)
        total_w = n_ts*n_w
        mem_w = self.n_lags//self.window_step

        param_keys = [
            "nodes",
            "exec_mode",
            "parallel_sketch",
            "parallel_candidates",
            "parallel_validation",
            "workers_total",
            "workers_sketch",
            "workers_candidates",
            "workers_validation",
            "window_size",
            "window_step",
            "basic_window",
            "n_lags",
            "seed",
            "seed_toggle",
            "preprocess",
            "sketch_norm",
            "candidate_backend",
            "candidate_bucket_width",
            "candidate_block_size_steps",
            "candidate_block_index_dims",
            "candidate_bound_dims",
            "candidate_bound_dim_selection",
            "enable_block_ub_pruning",
            "enable_row_ub_pruning",
            "block_similarity_assignment",
            "max_open_blocks",
            "candidate_instinct_query_mode",
            "candidate_instinct_top_k",
            "candidate_instinct_min_candidates",
            "candidate_instinct_entry_points",
                                            "candidate_similarity",
            "candidate_cosine_threshold",
            "candidate_parallel_mode",
            "candidate_key_mode",
            "candidate_key_seed",
            "candidate_lsh_radius",
            "candidate_ann_m",
            "candidate_ann_z",
            "candidate_ann_ef",
            "hybrid_validation",
            "hybrid_validation_min_repeat_rate",
            "hybrid_validation_disable_rate",
            "hybrid_validation_ema_alpha",
            "hybrid_validation_min_candidates",
            "hybrid_validation_attempts",
            "hybrid_validation_hits",
            "hybrid_validation_steps_active",
            "numeric_rows",
            "corr_threshold",
            "grid_max",
            "cell_stretch",
            "cell_size",
            "n_vectors",
            "grid_dimension",
            "freq_threshold",
        ]
        effective_nodes = 1
        if self.exec != "sequential":
            if nodes in (None, 0):
                effective_nodes = max(1, (os.cpu_count() or 1))
            else:
                effective_nodes = max(1, int(nodes))
        param_values = []
        for key in param_keys:
            if key == "exec_mode":
                param_values.append(self.exec)
            elif key == "parallel_sketch":
                param_values.append(self.parallel_sketch)
            elif key == "parallel_candidates":
                param_values.append(self.parallel_candidates)
            elif key == "parallel_validation":
                param_values.append(self.parallel_validation)
            elif key == "workers_total":
                param_values.append(effective_nodes)
            elif key == "workers_sketch":
                if self.parallel_sketch:
                    param_values.append(min(effective_nodes, len(self.ids)))
                else:
                    param_values.append(1)
            elif key == "workers_candidates":
                param_values.append(effective_nodes if self.parallel_candidates else 1)
            elif key == "workers_validation":
                param_values.append(effective_nodes if self.parallel_validation else 1)
            elif key == "candidate_key_mode":
                param_values.append(bst.get(key, self.candidate_key_mode))
            elif key == "candidate_key_seed":
                param_values.append(bst.get(key, self.candidate_key_seed))
            elif key == "candidate_lsh_radius":
                param_values.append(bst.get(key, getattr(self, "candidate_lsh_radius", 0)))
            elif key == "candidate_ann_m":
                param_values.append(bst.get(key, getattr(self, "candidate_ann_m", 16)))
            elif key == "candidate_ann_z":
                param_values.append(bst.get(key, getattr(self, "candidate_ann_z", 256)))
            elif key == "candidate_ann_ef":
                param_values.append(bst.get(key, getattr(self, "candidate_ann_ef", 256)))
            elif key == "candidate_parallel_mode":
                param_values.append(bst.get(key, self.candidate_parallel_mode))
            elif key == "candidate_bucket_width":
                param_values.append(bst.get(key, self.candidate_bucket_width))
            elif key == "candidate_block_size_steps":
                param_values.append(bst.get(key, self.candidate_block_size_steps))
            elif key == "candidate_block_index_dims":
                param_values.append(bst.get(key, self.candidate_block_index_dims))
            elif key == "candidate_bound_dims":
                param_values.append(bst.get(key, getattr(self, "candidate_bound_dims", 0)))
            elif key == "candidate_bound_dim_selection":
                param_values.append(bst.get(key, getattr(self, "candidate_bound_dim_selection", "variance")))
            elif key == "enable_block_ub_pruning":
                param_values.append(bst.get(key, getattr(self, "enable_block_ub_pruning", False)))
            elif key == "enable_row_ub_pruning":
                param_values.append(bst.get(key, getattr(self, "enable_row_ub_pruning", False)))
            elif key == "block_similarity_assignment":
                param_values.append(bst.get(key, getattr(self, "block_similarity_assignment", False)))
            elif key == "max_open_blocks":
                param_values.append(bst.get(key, getattr(self, "max_open_blocks", 4)))
            elif key == "candidate_instinct_query_mode":
                param_values.append(bst.get(key, getattr(self, "candidate_instinct_query_mode", "hybrid")))
            elif key == "candidate_instinct_top_k":
                param_values.append(bst.get(key, getattr(self, "candidate_instinct_top_k", 256)))
            elif key == "candidate_instinct_min_candidates":
                param_values.append(bst.get(key, getattr(self, "candidate_instinct_min_candidates", 64)))
            elif key == "candidate_instinct_entry_points":
                param_values.append(bst.get(key, getattr(self, "candidate_instinct_entry_points", 8)))
            elif key == "candidate_similarity":
                param_values.append(bst.get(key, getattr(self, "candidate_similarity", "l2")))
            elif key == "candidate_cosine_threshold":
                param_values.append(bst.get(key, getattr(self, "candidate_cosine_threshold", "")))
            elif key == "candidate_search_index_candidates":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_valid_index_candidates":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_unique_index_candidates":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_duplicate_index_candidates":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_unique_pre_dot_pairs":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_duplicate_pre_dot_pairs":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_after_coord":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_partial_checks":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_after_partial":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_after_similarity":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_dot_checks":
                param_values.append(bst.get(key, ""))
            elif key == "candidate_search_distance_checks":
                param_values.append(bst.get(key, ""))
            elif key in {
                "candidate_search_blocks_visited",
                "candidate_search_blocks_pruned_by_ub",
                "candidate_search_rows_in_surviving_blocks",
                "candidate_search_dot_checks_saved_by_row_ub",
    "candidate_search_instinct_visited_nodes",
    "candidate_search_instinct_visited_live_nodes",
    "candidate_search_instinct_dead_nodes_skipped",
    "candidate_search_instinct_edges_scanned",
    "candidate_search_instinct_candidates_returned",
    "candidate_search_instinct_threshold_candidates",
    "candidate_search_instinct_topk_candidates",
    "candidate_search_instinct_queries_with_too_few_live_nodes",
    "candidate_search_instinct_query_time",
    "candidate_search_instinct_insert_time",
    "candidate_search_instinct_num_nodes_total",
    "candidate_search_instinct_num_nodes_alive",
    "candidate_search_instinct_best_score_seen",
    "candidate_search_instinct_mean_score_returned",
    "candidate_search_instinct_dead_node_ratio",
            }:
                param_values.append(bst.get(key, ""))
            elif key == "hybrid_validation":
                param_values.append(bst.get(key, getattr(self, "hybrid_validation", False)))
            elif key == "hybrid_validation_min_repeat_rate":
                param_values.append(bst.get(key, getattr(self, "hybrid_validation_min_repeat_rate", 0.25)))
            elif key == "hybrid_validation_disable_rate":
                param_values.append(bst.get(key, getattr(self, "hybrid_validation_disable_rate", "")))
            elif key == "hybrid_validation_ema_alpha":
                param_values.append(bst.get(key, getattr(self, "hybrid_validation_ema_alpha", 0.25)))
            elif key == "hybrid_validation_min_candidates":
                param_values.append(bst.get(key, getattr(self, "hybrid_validation_min_candidates", 256)))
            elif key in {"hybrid_validation_attempts", "hybrid_validation_hits", "hybrid_validation_steps_active"}:
                param_values.append(bst.get(key, 0))
            elif key == "numeric_rows":
                param_values.append(bst.get(key, getattr(self, "numeric_rows", True)))
            else:
                param_values.append(bst.get(key))

        row = [
            dataset_id, mode, alg, "recall_speedup", getattr(self, "baseline_mode", ""), n_ts, n_w, total_w, mem_w,
        ] + param_values + [
            f"{self.candidate_time_bf:.4f}",f"{self.validation_time_bf:.4f}",f"{self.monitor_time_bf:.4f}",f"{self.runtime_bf:.4f}",f"{bf_artifact_time:.4f}",
            f"{runtime_parts[0]:.4f}",f"{runtime_parts[1]:.4f}",f"{runtime_parts[2]:.4f}",f"{runtime_parts[3]:.4f}",
            f"{runtime:.4f}", f"{artifact_time:.4f}", f"{speedup:.4f}",
            _format_float(speedup_ceil),
            _format_float(rel_speedup_eff),
            int(self.correlated_bf), int(self.correlated_w),
            int(self.tested_bf), int(self.tested_w),
            int(self.total_bf), int(self.total_w),
            int(getattr(self, "candidate_search_index_candidates", 0) or 0),
            int(getattr(self, "candidate_search_valid_index_candidates", 0) or 0),
            int(getattr(self, "candidate_search_unique_index_candidates", 0) or 0),
            int(getattr(self, "candidate_search_duplicate_index_candidates", 0) or 0),
            int(getattr(self, "candidate_search_unique_pre_dot_pairs", 0) or 0),
            int(getattr(self, "candidate_search_duplicate_pre_dot_pairs", 0) or 0),
            int(getattr(self, "candidate_search_after_coord", 0) or 0),
            int(getattr(self, "candidate_search_partial_checks", 0) or 0),
            int(getattr(self, "candidate_search_after_partial", 0) or 0),
            int(getattr(self, "candidate_search_after_similarity", 0) or 0),
            int(getattr(self, "candidate_search_dot_checks", 0) or 0),
            int(getattr(self, "candidate_search_distance_checks", 0) or 0),
            int(getattr(self, "candidate_search_blocks_visited", 0) or 0),
            int(getattr(self, "candidate_search_blocks_pruned_by_ub", 0) or 0),
            int(getattr(self, "candidate_search_rows_in_surviving_blocks", 0) or 0),
            int(getattr(self, "candidate_search_dot_checks_saved_by_row_ub", 0) or 0),
            int(getattr(self, "candidate_search_instinct_visited_nodes", 0) or 0),
            int(getattr(self, "candidate_search_instinct_visited_live_nodes", 0) or 0),
            int(getattr(self, "candidate_search_instinct_dead_nodes_skipped", 0) or 0),
            int(getattr(self, "candidate_search_instinct_edges_scanned", 0) or 0),
            int(getattr(self, "candidate_search_instinct_candidates_returned", 0) or 0),
            int(getattr(self, "candidate_search_instinct_threshold_candidates", 0) or 0),
            int(getattr(self, "candidate_search_instinct_topk_candidates", 0) or 0),
            int(getattr(self, "candidate_search_instinct_queries_with_too_few_live_nodes", 0) or 0),
            _format_float(getattr(self, "candidate_search_instinct_query_time", 0.0) or 0.0),
            _format_float(getattr(self, "candidate_search_instinct_insert_time", 0.0) or 0.0),
            int(getattr(self, "candidate_search_instinct_num_nodes_total", 0) or 0),
            int(getattr(self, "candidate_search_instinct_num_nodes_alive", 0) or 0),
            _format_float(getattr(self, "candidate_search_instinct_best_score_seen", 0.0) or 0.0),
            _format_float(getattr(self, "candidate_search_instinct_mean_score_returned", 0.0) or 0.0),
            _format_float(getattr(self, "candidate_search_instinct_dead_node_ratio", 0.0) or 0.0),
            _format_float(corr_prop),
            _format_float(waste_val_bf),
            _format_float(waste_val),
            _format_float(rel_waste_red),
            f"{metrics['precision_pos']:.4f}",f"{metrics['recall_pos']:.4f}",f"{metrics['f1_score_pos']:.4f}",
            f"{metrics['precision_neg']:.4f}",f"{metrics['recall_neg']:.4f}",f"{metrics['f1_score_neg']:.4f}",
            f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['specificity']:.4f}",
            _format_optional_float(metrics['recall_min']), f"{metrics['f1_score']:.4f}", f"{metrics['aucroc']:.4f}",
            f"{metrics['pr_auc']:.4f}"
        ] + [
            f"{maxlag_precision:.4f}",
            f"{maxlag_recall:.4f}",
            f"{maxlag_f1:.4f}",
            _format_float(maxlag_diff_mean),
            _format_float(maxlag_diff_std),
        ]
        return _filter_comparison_row(row)

    def compare(
        self,
        hyper_param_csv,
        output_csv,
        dataset_id,
        run=True,
        target_recall=0.95,
        bf_run_csv=None,
        corrtrack_run_files=None,
    ):
        if not run and bf_run_csv and corrtrack_run_files:
            return self.compare_from_artifacts(
                dataset_id,
                bf_run_csv,
                corrtrack_run_files,
                output_csv,
            )
        path = os.path.dirname(os.path.abspath(output_csv))

        bst = {}
        for alg in self.algs:
            best_param_comb = self._optim(hyper_param_csv, dataset_id, run, alg, target_recall)
            bst[alg] = best_param_comb.iloc[0].to_dict()

        data_stream = self.test_data

        runtime_parts, runtime_bf, artifact_time_bf, corr_flags_bf = self._mode_run(
            "bf",
            None,
            path,
            "bf",
            0,
            None,
            None,
            1,
            1,
            None,
            None,
            None,
            None,
            None,
        )
        print("Run Brute-Force, Finished in ",runtime_bf)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        args_list = [
            (
                dataset_id,
                "main",
                alg,
                path,
                "main_" + alg + "_recall_speedup",
                bst[alg],
                runtime_bf,
                corr_flags_bf,
            )
            for alg in self.algs
        ]

        results = list(self._outer_iter(args_list, self._parallel_mode_run, unordered=False))

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(COMPARISON_COLUMNS)
            writer.writerows(results)


#DONE: include timestamp
#DONE: set n lags independent of n of windows
#DONE: different seeds for random vector and toggle vector
#DONE: track startTime of windows
#DONE: dictionary with ids(serie,window_startTime)-sketches
#DONE: optimization of window_data structure
#DONE: dynamic change of window_step
#DONE: dynamic change of window_size
#DONE: increasing window_step
#TODO: bug fix: reducing window_size
#TODO: bug fix: many nodes with window step smaller than basic window

#DONE: check definition of grid size
#DONE: memory of sketches just on grid nodes
#DONE: incremental update of sketches on grid nodes
#DONE: negative correlations (compute pair combinations from the same cell and also the inverted cell grids)
#DONE: preprocessing/normalization
#DONE: correlation monitoring
#DONE: anomaly detection
#DONE: save correlation logs to file (with real times)
#DONE: clean up old datetime_lookups

#ARCHIVED: recompute sketches when min-max changes
#ARCHIVED: adaptive normalization (the probability of min-max change decreases over time as the range stabilizes)
#ARCHIVED: incremental min and max update
#ARCHIVED: get upper/lower bounds of the grid from min and max
#DONE: histogram of sketches
#DONE: incremental update of grids
#ARCHIVED: test stationarity for performing 2nd order diff

#TODO: weighting based on probability of sketches 
#TODO: observe super populated cell grids
#TODO: get subcells of super populated cell grids

#TODO: forgetting mechanism
