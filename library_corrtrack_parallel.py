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
import heapq
import pandas as pd
from itertools import combinations, product, repeat
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
    _cy_find_candidate_pairs = _cand_kernels.find_candidate_pairs
    _cy_find_candidate_pairs_unique = getattr(_cand_kernels, "find_candidate_pairs_unique", None)
    _cy_find_candidate_pairs_full = getattr(_cand_kernels, "find_candidate_pairs_full", None)
    _cy_balanced_index_cls = getattr(_cand_kernels, "BalancedIndex", None)
    _cy_enumerate_candidate_rows = getattr(_cand_kernels, "enumerate_candidate_rows", None)
    _cy_fast_corr_and_dist = _cand_kernels.fast_corr_and_dist
    _cy_validate_corr_batch = _cand_kernels.validate_corr_batch
    _HAS_CYTHON_KERNELS = True
except Exception:  # pragma: no cover
    _cy_find_candidate_pairs = None
    _cy_find_candidate_pairs_unique = None
    _cy_find_candidate_pairs_full = None
    _cy_balanced_index_cls = None
    _cy_enumerate_candidate_rows = None
    _cy_fast_corr_and_dist = None
    _cy_validate_corr_batch = None
    _HAS_CYTHON_KERNELS = False

try:
    from sketch_kernels import compute_series_dots as _cy_compute_series_dots
    from sketch_kernels import build_sketch_matrix as _cy_build_sketch_matrix
    from sketch_kernels import apply_orth_and_normalize as _cy_apply_orth_and_normalize
    from sketch_kernels import compute_constant_flags as _cy_compute_constant_flags
except Exception:  # pragma: no cover
    _cy_compute_series_dots = None
    _cy_build_sketch_matrix = None
    _cy_apply_orth_and_normalize = None
    _cy_compute_constant_flags = None

try:
    from partition_kernels import build_partitions as _cy_build_partitions
    from partition_kernels import build_partition_values as _cy_build_partition_values
except Exception:  # pragma: no cover
    _cy_build_partitions = None
    _cy_build_partition_values = None

try:
    from monitor_kernels import monitor_step as _cy_monitor_step
except Exception:  # pragma: no cover
    _cy_monitor_step = None


RUN_RESULT_COLUMNS: Sequence[str] = (
    "dataset_id",
    "run_kind",
    "mode",
    "alg",
    "optim",
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
    "pair_min_dist",
    "artifact_path",
)

OPTIM_RESULT_COLUMNS: Sequence[str] = (
    "dataset_id",
    "alg",
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
    "status",
    "error",
)

COMPARISON_COLUMNS: Sequence[str] = (
    "dataset_id",
    "mode",
    "alg",
    "optim",
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

    record = {key: None for key in RUN_RESULT_COLUMNS}
    record.update(metadata or {})
    record["dataset_id"] = dataset_id
    record["run_kind"] = run_kind
    record["artifact_path"] = artifact_prefix or ""

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

    artifact_mode_value = (artifact_mode or "iterative").lower()
    if artifact_mode_value not in ("iterative", "final", "buffered"):
        artifact_mode_value = "iterative"
    effective_artifact_mode = artifact_mode_value if artifact_active else "final"

    corrtrack._configure_artifact_runtime(
        artifact_mode=effective_artifact_mode,
        artifact_buffer_max_rows=artifact_buffer_max_rows,
        recall_by_window=recall_by_window,
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
    artifact_buffered = artifact_active and effective_artifact_mode == "buffered"

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
        if artifact_active and artifact_buffered:
            _t0 = time.time()
            flushed = corrtrack._maybe_flush_artifact_buffers()
            elapsed = time.time() - _t0
            if flushed:
                artifact_time_total += elapsed
                artifact_time_overlap += elapsed
        elif artifact_active and artifact_per_iteration:
            _t0 = time.time()
            corrtrack._append_artifacts()
            elapsed = time.time() - _t0
            artifact_time_total += elapsed
            artifact_time_overlap += elapsed
    end_time = time.time()

    if artifact_active and artifact_buffered:
        _t0 = time.time()
        corrtrack._finalize_buffered_artifacts()
        artifact_time_total += time.time() - _t0
    elif artifact_active and not artifact_per_iteration:
        _t0 = time.time()
        corrtrack._append_artifacts()
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

    if artifact_active:
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
    pair_min_dist = getattr(corrtrack, "pair_min_dist", None)
    record["pair_min_dist"] = str(pair_min_dist) if pair_min_dist is not None else ""

    corrtrack._step_observer_enabled = False
    corrtrack._online_window_metrics_only = False
    corrtrack._validated_step = {}

    if artifact_buffered:
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
    metadata.setdefault("mode", "bf")
    metadata.setdefault("alg", metadata.get("alg", "bf"))
    metadata.setdefault("optim", metadata.get("optim", "baseline"))
    metadata.setdefault("artifact_buffer_max_rows", base_config.get("artifact_buffer_max_rows"))

    artifact_mode = base_config.get("artifact_mode", "iterative")
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
        recall_by_window=recall_by_window,
        artifact_prefix=artifact_prefix,
        artifact_mode=artifact_mode,
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

    n_vectors = _to_int(run_params.get("n_vectors"))
    grid_dimension = _to_int(run_params.get("grid_dimension"))
    cell_stretch = _resolve_cell_stretch(
        run_params.get("cell_stretch"),
        run_params.get("cell_size"),
        base_config["corr_threshold"],
        n_vectors,
        grid_dimension,
        full_vector_candidates=not _resolve_parallel_flag(
            base_config.get("parallel_candidates"),
            False,
        ),
    )
    if cell_stretch is None or cell_stretch <= 0.0:
        cell_stretch = 1.0

    feature_kwargs = _extract_feature_overrides(run_params)
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
        recall_by_window=recall_by_window,
        artifact_prefix=artifact_prefix,
        artifact_mode=artifact_mode,
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
    }
    key = aliases.get(key, key)
    if key not in {"flat", "bptree", "auto"}:
        key = default
    return key


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
    return overrides


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
    grid_dim = _to_float_safe(grid_dimension)
    if grid_dim is None or grid_dim <= 0.0:
        grid_dim = 1.0
    if full_vector_candidates:
        grid_adjust = 1.0
    else:
        try:
            grid_adjust = math.sqrt(grid_dim)
        except ValueError:
            grid_adjust = 1.0

    if base is None or base <= 0.0 or grid_adjust <= 0.0:
        return default

    stretch = size / (base * grid_adjust)
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
    corrtrack, grid_index, n_ids, verbose, testing = payload
    grid_node = corrtrack.grid_nodes[grid_index]
    result = grid_node.run(n_ids, verbose=verbose, testing=testing)
    return grid_index, grid_node, result


def _bf_worker(payload):
    """Execute brute-force candidate generation for a shard."""
    index, bf_node, window_step, ids, verbose, testing, ref_ids = payload
    result = bf_node.run(window_step, ids, verbose=verbose, testing=testing, ref_ids=ref_ids)
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
                "precision_pos": float("nan"),
                "recall_pos": recall_pos,
                "f1_score_pos": float("nan"),
                "precision_neg": float("nan"),
                "recall_neg": recall_neg,
                "f1_score_neg": float("nan"),
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

    payload keys:
      - items: list of metadata dicts, or
      - pairs, x_batch, y_batch: pre-built batch arrays
      - corr_threshold, neg_corr, corr_val
    Returns list of tuples (pair, is_corr, corr, dist, is_constant, is_spiked)
    """
    items = payload.get("items")
    if items is None:
        pairs = payload.get("pairs") or []
        x_batch = payload.get("x_batch")
        y_batch = payload.get("y_batch")
        items = []
        if x_batch is None or y_batch is None:
            return []
        for pair, x, y in zip(pairs, x_batch, y_batch):
            items.append({"pair": pair, "x": x, "y": y})
    corr_threshold = payload["corr_threshold"]
    neg_corr = payload["neg_corr"]
    corr_val = payload.get("corr_val", True)
    std_thresh = 1e-3

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

            if neg_corr:
                is_correlated = (not np.isnan(pair_corr)) and (abs(pair_corr) >= corr_threshold)
            else:
                is_correlated = (not np.isnan(pair_corr)) and (pair_corr >= corr_threshold)

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
    def __init__(self,window_size,basic_window,window_step,n_vectors,n_lags,grid_dimension,cell_size,seed=2468,seed_toggle=1357,freq_threshold=0.7,corr_threshold=0.7,neg_corr=False,preprocess=False,exec="parallel",max_workers=0,sketch_norm="z",candidate_backend="auto",parallel_sketch=None,parallel_candidates=None,parallel_validation=None,track_min_dist=True):
        
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

        grid_dimension = 1 if grid_dimension is None or grid_dimension <= 0 else int(grid_dimension)
        if n_vectors is None or n_vectors <= 0:
            raise ValueError("n_vectors must be a positive integer.")
        if parallel_candidates_flag and n_vectors % grid_dimension != 0:
            raise TypeError("Number of random vectors (",n_vectors,") is not divisable by the grid dimension (",grid_dimension,")")
        
        # Parameters features
        self.neg_corr = neg_corr
        # Parameters data
        self.window_data = None
        self.window_index = None
        self.window_startTimes = None
        self.series_ids = {}
        self.map_ids = []
        self.ids = None
        self.datetime_index = None
        self.datetime_lookup = {} #TODO: save correlation logs to file
        self.preprocess = preprocess
        self.sketch_norm = str(sketch_norm) if sketch_norm is not None else "z"
        self.candidate_backend = _resolve_candidate_backend(candidate_backend, default="auto")
        self.track_min_dist = _coerce_to_bool(track_min_dist, default=True)
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
        self._thread_pool = None
        self._thread_pool_workers = None
        # Parameters grids
        self.full_vector_candidates = not self.parallel_candidates
        if self.full_vector_candidates:
            self.grid_dimension = int(self.n_vectors)
            self.n_grids = 1
        else:
            self.grid_dimension = grid_dimension
            if self.n_vectors is not None:
                self.n_grids = int(self.n_vectors // self.grid_dimension)
            else:
                self.n_grids = 1
        self.freq_pairs = {}
        self.candidates = {}
        self.validated = {}
        self.correlated = {}
        self.corr_lengths = {}
        self.previous_correlations = {}
        self.corr_attention_in = {}
        self.corr_attention_out = {}
        self.corr_anomalies = {}
        self.sketch_mean = 0
        self.sketch_std = np.sqrt(self.window_size/self.n_vectors)
        self.corr_threshold = corr_threshold

        base = _compute_base_cell_size(corr_threshold, self.n_vectors)
        if base is None:
            base = np.sqrt((1.0 - corr_threshold) * 2.0) / np.sqrt(self.n_vectors)
        grid_dim = _to_float_safe(self.grid_dimension)
        if grid_dim is None or grid_dim <= 0.0:
            grid_dim = 1.0
        if self.full_vector_candidates:
            grid_adjust = 1.0
        else:
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
        self.cell_size = base * stretch * grid_adjust
        if self.full_vector_candidates:
            self.cell_size *= math.sqrt(self.n_vectors)
        else:
            self.cell_size = min(self.cell_size, 0.5)
        self.grid_max = min(1.0, 3.0/np.sqrt(self.n_vectors))

        # Parameters thresholds
        if self.full_vector_candidates:
            self.freq_threshold = 0
        elif freq_threshold is not None:
            self.freq_threshold = freq_threshold * self.n_grids
        else:
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
        self._artifact_buffer_max_rows = 250000
        self._artifact_buffered = False
        self._artifact_recall_by_window = True
        self._spill_correlated = {}
        self._spill_correlated_chunks = []
        self._spill_correlated_chunk_index = 0
        self._spill_status_rows = []
        self._spill_status_chunks = []
        self._spill_status_chunk_index = 0
        self._spill_anomaly_rows = []
        self._spill_anomaly_chunks = []
        self._spill_anomaly_chunk_index = 0
        self._active_corr_lengths = {}
        self._maxlag_state = {}
        self._step_observer_enabled = False
        self._validated_step = {}
        self._online_window_metrics_only = False
        self.sketches = {}
        self.brute_force_steps = 0
        self.constant_candidates = 0
        self.total_candidates = 0
        self.tested_candidates = 0
        self.validated_candidates = 0
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

        if self.neg_corr:
            is_correlated = (not np.isnan(pair_corr)) and (abs(pair_corr) >= self.corr_threshold)
        else:
            is_correlated = (not np.isnan(pair_corr)) and (pair_corr >= self.corr_threshold)

        return (pair, is_correlated, pair_corr, pair_dist)
    
    def _in_corr(self,pair, corr_value=None):
        #monitor continuous and fallen into correlation
        timepts = np.array([pair[2],pair[3]])
        window_size = pair[4]
        corr_lag = max(timepts) - min(timepts)
        if corr_value is None:
            corr_value = self.correlated[pair]
        corr_sign = (1 if corr_value >= 0 else -1)
        key = (pair[0],pair[1],corr_lag) #ids = (pair[0],pair[1])
        key2 = (pair[1],pair[0],corr_lag)
        if key not in self.corr_lengths.keys() and key2 not in self.corr_lengths.keys():
            self.corr_lengths[key] = []
            self.corr_lengths[key].append([timepts[0], timepts[1], window_size, window_size, corr_sign])
            self.corr_anomalies[key] = []
            self.corr_anomalies[key].append((min(timepts),1))
        else:
            if key2 in self.corr_lengths.keys():
                key = key2
            last_corr = self.corr_lengths[key][-1]
            first_timepts = np.array([last_corr[0],last_corr[1]])
            last_window_size = last_corr[2]
            last_corr_length = last_corr[3]
            last_corr_sign = last_corr[4]
            curr_time = max(timepts+window_size)
            next_corr_time = max(first_timepts+last_corr_length+self.window_step)
            if curr_time < next_corr_time:
                return None
            elif curr_time == next_corr_time and window_size == last_window_size and corr_sign == last_corr_sign:
                self.corr_lengths[key][-1][3] += self.window_step
            else:
                self.corr_lengths[key].append([timepts[0], timepts[1], window_size, window_size, corr_sign])
                if corr_sign == last_corr_sign:
                    self.corr_anomalies[key].append((min(timepts),1))
                else:
                    self.corr_anomalies[key].append((min(timepts),0))
        return {key:self.corr_lengths[key][-1]}
    
    def _out_corr(self, item):
        """
        Compute the (time, -1) anomaly for a given previous_correlation entry without mutating state.
        item: (pair_key, status_list) where status_list == [t1, t2, w, last_len]
        Returns: (pair_key, (time, -1))
        """
        pair, status = item
        timepts = np.array([status[0], status[1]])
        window_size = status[2]
        last_corr_length = status[3]
        time = int(np.min(timepts + last_corr_length - (window_size - self.window_step)))
        return (pair, (time, -1))

    def _configure_artifact_runtime(self, artifact_mode="final", artifact_buffer_max_rows=None, recall_by_window=True):
        mode_value = (artifact_mode or "final").lower()
        if mode_value not in {"iterative", "final", "buffered"}:
            mode_value = "final"
        self._artifact_mode_runtime = mode_value
        self._artifact_recall_by_window = bool(recall_by_window)
        try:
            max_rows = int(artifact_buffer_max_rows) if artifact_buffer_max_rows is not None else self._artifact_buffer_max_rows
        except (TypeError, ValueError):
            max_rows = self._artifact_buffer_max_rows
        self._artifact_buffer_max_rows = max(1, max_rows)
        self._artifact_buffered = mode_value == "buffered"

        self._spill_correlated = {}
        self._spill_correlated_chunks = []
        self._spill_correlated_chunk_index = 0
        self._spill_status_rows = []
        self._spill_status_chunks = []
        self._spill_status_chunk_index = 0
        self._spill_anomaly_rows = []
        self._spill_anomaly_chunks = []
        self._spill_anomaly_chunk_index = 0
        self._active_corr_lengths = {}
        self._maxlag_state = {}

    def _artifact_runtime_flush_needed(self):
        if not self._artifact_buffered:
            return False
        limit = max(1, int(getattr(self, "_artifact_buffer_max_rows", 1)))
        return (
            len(getattr(self, "_spill_correlated", {})) >= limit
            or len(getattr(self, "_spill_status_rows", [])) >= limit
            or len(getattr(self, "_spill_anomaly_rows", [])) >= limit
        )

    def _record_correlated(self, pair, corr, retain_validated=True):
        self._record_correlated_batch({pair: corr}, retain_validated=retain_validated)

    def _record_correlated_batch(self, accepted_map, retain_validated=True):
        if not accepted_map:
            return
        if getattr(self, "_step_observer_enabled", False):
            self._validated_step.update(accepted_map)
        if retain_validated:
            self.validated.update(accepted_map)
        if getattr(self, "_artifact_buffered", False):
            t0 = time.perf_counter()
            self._spill_correlated.update(accepted_map)
            for pair, corr in accepted_map.items():
                self._update_maxlag_state(pair, corr)
            self.artifact_bookkeeping_time += time.perf_counter() - t0
        else:
            self.correlated.update(accepted_map)

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
        base_index = self.window_index[0]
        ids_lookup = self.series_ids
        data = self.window_data

        batch_pairs = []
        batch_x = []
        batch_y = []
        for pair in pairs:
            id1, id2, t1, t2, window_size = pair
            start1 = int(t1 - base_index)
            start2 = int(t2 - base_index)
            x = data[ids_lookup[id1], start1:start1 + window_size]
            y = data[ids_lookup[id2], start2:start2 + window_size]
            if x.size != window_size or y.size != window_size:
                continue
            batch_pairs.append(pair)
            batch_x.append(x.astype(np.float64, copy=False))
            batch_y.append(y.astype(np.float64, copy=False))
            if len(batch_pairs) >= chunk_size:
                yield {
                    "pairs": list(batch_pairs),
                    "x_batch": np.asarray(batch_x, dtype=np.float64),
                    "y_batch": np.asarray(batch_y, dtype=np.float64),
                    "corr_threshold": self.corr_threshold,
                    "neg_corr": self.neg_corr,
                    "corr_val": True,
                }
                batch_pairs = []
                batch_x = []
                batch_y = []

        if batch_pairs:
            yield {
                "pairs": list(batch_pairs),
                "x_batch": np.asarray(batch_x, dtype=np.float64),
                "y_batch": np.asarray(batch_y, dtype=np.float64),
                "corr_threshold": self.corr_threshold,
                "neg_corr": self.neg_corr,
                "corr_val": True,
            }

    def _buffered_emit_anomaly(self, key, marker):
        id1, id2, lag = key
        if isinstance(marker, tuple):
            time_val, kind = marker
        else:
            time_val, kind = marker, 0
        self._spill_anomaly_rows.append((id1, id2, lag, int(time_val), int(kind)))

    def _buffered_finalize_status(self, key, status):
        if key is None or status is None:
            return
        id1, id2, lag = key
        t1, t2, _w, corr_len, corr_sign = status
        self._spill_status_rows.append((id1, id2, lag, int(t1), int(t2), int(corr_len), int(corr_sign)))

    def _buffered_in_corr(self, pair, corr):
        timepts = np.array([pair[2], pair[3]])
        window_size = pair[4]
        corr_lag = max(timepts) - min(timepts)
        corr_sign = (1 if corr >= 0 else -1)
        key = (pair[0], pair[1], corr_lag)
        key2 = (pair[1], pair[0], corr_lag)

        if key not in self._active_corr_lengths and key2 not in self._active_corr_lengths:
            status = [int(timepts[0]), int(timepts[1]), int(window_size), int(window_size), int(corr_sign)]
            self._active_corr_lengths[key] = status
            self._buffered_emit_anomaly(key, (int(min(timepts)), 1))
            return key, status

        if key2 in self._active_corr_lengths:
            key = key2

        last_corr = self._active_corr_lengths[key]
        first_timepts = np.array([last_corr[0], last_corr[1]])
        last_window_size = last_corr[2]
        last_corr_length = last_corr[3]
        last_corr_sign = last_corr[4]
        curr_time = max(timepts + window_size)
        next_corr_time = max(first_timepts + last_corr_length + self.window_step)

        if curr_time < next_corr_time:
            return None

        if curr_time == next_corr_time and window_size == last_window_size and corr_sign == last_corr_sign:
            last_corr[3] += self.window_step
            return key, last_corr

        self._buffered_finalize_status(key, list(last_corr))
        status = [int(timepts[0]), int(timepts[1]), int(window_size), int(window_size), int(corr_sign)]
        self._active_corr_lengths[key] = status
        marker = 1 if corr_sign == last_corr_sign else 0
        self._buffered_emit_anomaly(key, (int(min(timepts)), marker))
        return key, status

    def _monitor_corr_buffered(self):
        new_in = {}
        if len(self.validated) > 0:
            for pair, corr in self.validated.items():
                new_pair = self._buffered_in_corr(pair, corr)
                if new_pair is not None:
                    key, status = new_pair
                    new_in[key] = [status[0], status[1], status[2], status[3]]
                    if key in self.previous_correlations:
                        del self.previous_correlations[key]

        if len(self.previous_correlations) > 0:
            for pair_key, rec in [self._out_corr(item) for item in self.previous_correlations.items()]:
                self._buffered_emit_anomaly(pair_key, rec)
                active_status = self._active_corr_lengths.pop(pair_key, None)
                if active_status is not None:
                    self._buffered_finalize_status(pair_key, list(active_status))

        self.previous_correlations = new_in

    def _flush_correlated_chunk(self):
        if not self._spill_correlated:
            return False
        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return False
        self._spill_correlated_chunk_index += 1
        path = f"{prefix}_correlated.chunk{self._spill_correlated_chunk_index:06d}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = sorted(
            self._spill_correlated.items(),
            key=lambda item: _window_sort_key((item[0][0], item[0][1], item[0][2], item[0][3]), item[0][4]),
        )
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "time1_idx", "time2_idx", "corr"])
            for pair, corr in rows:
                writer.writerow([pair[0], pair[1], int(pair[2]), int(pair[3]), corr])
        self._spill_correlated_chunks.append(path)
        self._spill_correlated = {}
        return True

    def _flush_status_chunk(self):
        if not self._spill_status_rows:
            return False
        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return False
        self._spill_status_chunk_index += 1
        path = f"{prefix}_status.chunk{self._spill_status_chunk_index:06d}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = sorted(self._spill_status_rows, key=_status_sort_key)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "lag", "start_time_id1_idx", "start_time_id2_idx", "duration", "corr_sign"])
            writer.writerows(rows)
        self._spill_status_chunks.append(path)
        self._spill_status_rows = []
        return True

    def _flush_anomaly_chunk(self):
        if not self._spill_anomaly_rows:
            return False
        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return False
        self._spill_anomaly_chunk_index += 1
        path = f"{prefix}_anomalies.chunk{self._spill_anomaly_chunk_index:06d}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = sorted(self._spill_anomaly_rows, key=_anomaly_sort_key)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER)
            writer.writerow(["id1", "id2", "lag", "time_idx", "marker"])
            writer.writerows(rows)
        self._spill_anomaly_chunks.append(path)
        self._spill_anomaly_rows = []
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

    def _merge_sorted_csv_chunks(self, chunk_paths, output_csv, header, key_fn, row_transform=None):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as out_file:
            writer = csv.writer(out_file, delimiter=CSV_DELIMITER)
            writer.writerow(header)

            files = []
            readers = []
            heap = []
            try:
                for index, path in enumerate(chunk_paths):
                    handle = open(path, newline="")
                    reader = csv.reader(handle, delimiter=CSV_DELIMITER)
                    next(reader, None)
                    row = next(reader, None)
                    if row is not None:
                        heap.append((key_fn(row), index, row))
                    files.append(handle)
                    readers.append(reader)

                heapq.heapify(heap)
                while heap:
                    _sort_key, index, row = heapq.heappop(heap)
                    writer.writerow(row_transform(row) if row_transform else row)
                    next_row = next(readers[index], None)
                    if next_row is not None:
                        heapq.heappush(heap, (key_fn(next_row), index, next_row))
            finally:
                for handle in files:
                    try:
                        handle.close()
                    except Exception:
                        pass

        for path in chunk_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def _finalize_buffered_artifacts(self):
        if not self._artifact_buffered:
            return
        self._maybe_flush_artifact_buffers(force=True)

        prefix = self._artifact_state.get("prefix")
        if not prefix:
            return

        for key, status in list(self._active_corr_lengths.items()):
            self._buffered_finalize_status(key, list(status))
        self._active_corr_lengths = {}
        self._maybe_flush_artifact_buffers(force=True)

        correlated_csv = f"{prefix}_correlated.csv"
        if self._spill_correlated_chunks:
            self._merge_sorted_csv_chunks(
                self._spill_correlated_chunks,
                correlated_csv,
                ["id1", "id2", "time1_idx", "time1", "time2_idx", "time2", "corr"],
                key_fn=lambda row: _window_sort_key((row[0], row[1], int(row[2]), int(row[3])), self.window_size),
                row_transform=lambda row: [
                    row[0],
                    row[1],
                    row[2],
                    self._safe_index_to_datetime(int(row[2])) if str(row[2]).strip() != "" else row[2],
                    row[3],
                    self._safe_index_to_datetime(int(row[3])) if str(row[3]).strip() != "" else row[3],
                    row[4],
                ],
            )
        elif not os.path.exists(correlated_csv):
            with open(correlated_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "time1_idx", "time1", "time2_idx", "time2", "corr"])

        status_csv = f"{prefix}_status.csv"
        if self._spill_status_chunks:
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
        elif not os.path.exists(status_csv):
            with open(status_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"])

        anomalies_csv = f"{prefix}_anomalies.csv"
        if self._spill_anomaly_chunks:
            self._merge_sorted_csv_chunks(
                self._spill_anomaly_chunks,
                anomalies_csv,
                ["id1", "id2", "lag", "time_idx", "time", "anomaly"],
                key_fn=lambda row: _anomaly_sort_key((row[0], row[1], int(row[2]), int(row[3]), int(row[4]))),
                row_transform=lambda row: [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    self._safe_index_to_datetime(int(row[3])),
                    _marker_to_label(int(row[4])),
                ],
            )
        elif not os.path.exists(anomalies_csv):
            with open(anomalies_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "lag", "time_idx", "time", "anomaly"])

        self._spill_correlated_chunks = []
        self._spill_status_chunks = []
        self._spill_anomaly_chunks = []
    
    def _get_validated_corr(self, corr_val=True, force_mode=None, retain_validated=True):
        self.validated = {}
        if getattr(self, "_step_observer_enabled", False):
            self._validated_step = {}
        if not self.candidates:
            return

        candidates = self.candidates
        n_pairs = len(candidates)

        # Count candidates once, for both sequential and parallel
        self.total_candidates += n_pairs

        if not corr_val:
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
                self.correlated.update(validated_now)
            self.tested_candidates += n_pairs
            self.validated_candidates += n_pairs
            return

        track_min_dist = bool(getattr(self, "track_min_dist", True))
        if not track_min_dist:
            self.min_dist = np.inf
            self.pair_min_dist = None

        pairs = candidates.keys()

        worker_mode = force_mode if force_mode else self.exec

        if worker_mode == "sequential":
            if _cy_validate_corr_batch is not None:
                chunk_size = 256
                tested = validated = 0
                if track_min_dist:
                    min_dist, min_pair = self.min_dist, self.pair_min_dist
                eff_std_thresh = 1e-3
                had_payload = False
                for payload in self._iter_validation_payloads(pairs, chunk_size):
                    had_payload = True
                    x_batch = payload["x_batch"]
                    y_batch = payload["y_batch"]
                    batch_results = _cy_validate_corr_batch(
                        x_batch,
                        y_batch,
                        float(self.corr_threshold),
                        bool(self.neg_corr),
                        eff_std_thresh,
                    )
                    accepted_now = {}
                    for pair, entry in zip(payload["pairs"], batch_results):
                        is_correlated, pair_corr, pair_dist, _is_const, _is_spiked = entry
                        tested += 1
                        if track_min_dist and pair_dist < min_dist:
                            min_dist, min_pair = pair_dist, pair
                        if is_correlated:
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

        Fast path: one-pass Cython kernel (when monitor_kernels is available).
        Fallback: Python hybrid path with sequential 'in' and parallel 'out'.
        """
        if getattr(self, "_artifact_buffered", False):
            self._monitor_corr_buffered()
            return

        if _cy_monitor_step is not None:
            t0 = time.perf_counter() if self.profile_enabled else None
            validated_items = tuple(self.validated.items()) if self.validated else ()
            new_in = _cy_monitor_step(
                validated_items,
                self.correlated,
                self.corr_lengths,
                self.corr_anomalies,
                self.previous_correlations,
                int(self.window_step),
            )
            if t0 is not None:
                self._profile_add("monitor.kernel", time.perf_counter() - t0)
            self.previous_correlations = new_in
            return

        new_in = {}
        if len(self.validated) > 0:
            t0 = time.perf_counter() if self.profile_enabled else None
            # sequential 'in' to preserve state coupling
            for pair, corr in self.validated.items():
                new_pair = self._in_corr(pair, corr)
                if new_pair is not None:
                    new_in.update(new_pair)
                    key = list(new_pair.keys())[0]
                    if key in self.previous_correlations:
                        del self.previous_correlations[key]
            if t0 is not None:
                self._profile_add("monitor.in", time.perf_counter() - t0)

        # parallel 'out' (independent across keys)
        if len(self.previous_correlations) > 0:
            items = list(self.previous_correlations.items())
            worker_mode = worker_mode if worker_mode else self.exec
            # compute anomaly records in parallel
            t0 = time.perf_counter() if self.profile_enabled else None
            out_records = self._parallel_map(
                self._out_corr,
                items,
                mode=worker_mode,
                max_workers=self.n_nodes,
                preserve_order=False,
            )
            if t0 is not None:
                self._profile_add("monitor.out_dispatch", time.perf_counter() - t0)
            # apply on main thread
            t0 = time.perf_counter() if self.profile_enabled else None
            for pair_key, rec in out_records:
                if pair_key not in self.corr_anomalies:
                    self.corr_anomalies[pair_key] = []
                self.corr_anomalies[pair_key].append(rec)
            if t0 is not None:
                self._profile_add("monitor.out_apply", time.perf_counter() - t0)

        # finalize 'previous_correlations'
        self.previous_correlations = new_in


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

        # sort lexicographically by (id1, id2, t1)
        sorted_items = sorted(
            self.correlated.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
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
        if getattr(self, "_artifact_buffered", False) and self._maxlag_state:
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
        if len(self.corr_lengths)>0:
            for key,value in self.corr_lengths.items():
                ids = (key[0],key[1])
                lag = key[2]
                for status in value:
                    timepts = (status[0],status[1])
                    if self.datetime_index:
                        timepts = self._index_to_datetime(timepts)
                    corr_len = status[3]
                    corr_sign = status[4]
                    if lag == 0:
                        print(ids[0]," and ",ids[1]," are correlated since time ",timepts[0]," (duration: ",corr_len,", sign: ",corr_sign,").")
                    else:
                        print(ids[0]," and ",ids[1]," are correlated with lag of ",lag,"time points since time ",min(timepts)," (duration: ",corr_len,", sign: ",corr_sign,").")
    
    def _save_monitor_status(self,output_csv):
        if len(self.corr_lengths)>0:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"])
                for key,value in self.corr_lengths.items():
                    ids = (key[0],key[1])
                    lag = key[2]
                    for t1,t2,w,corr_len,corr_sign in value:
                        time1 = t1
                        time2 = t2
                        if self.datetime_index:
                            time1 = self._safe_index_to_datetime(t1)
                            time2 = self._safe_index_to_datetime(t2)
                        writer.writerow([ids[0],ids[1],lag,time1,time2,corr_len,corr_sign])
                        

    def _print_anomalies(self):
        if len(self.corr_anomalies)>0:
            for key,value in self.corr_anomalies.items():
                ids = (key[0],key[1])
                lag = key[2]
                for status in value:
                    time = status[0]
                    if self.datetime_index:
                        time = self._index_to_datetime(time)
                    type = status[1]
                    if type == 1:
                        if lag == 0:
                            print(ids[0]," and ",ids[1]," have fallen into correlation at time ",time)
                        else:
                            print(ids[0]," and ",ids[1]," have fallen into correlation at time ",time," (lag: ",lag,").")
                    elif type == -1:
                        if lag == 0:
                            print(ids[0]," and ",ids[1]," have fallen out of correlation at time ",time)
                        else:
                            print(ids[0]," and ",ids[1]," have fallen out of correlation at time ",time," (lag: ",lag,").")
                    else:
                        if lag == 0:
                            print(ids[0]," and ",ids[1]," have changed sign of correlation at time ",time)
                        else:
                            print(ids[0]," and ",ids[1]," have changed sign of correlation at time ",time," (lag: ",lag,").")
    
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
        if len(self.corr_anomalies)>0:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1", "id2", "lag", "time_idx", "time", "anomaly"])
                for key,value in self.corr_anomalies.items():
                    ids = (key[0],key[1])
                    lag = key[2]
                    for time,type in value:
                        time_idx = time
                        time_display = time
                        if self.datetime_index:
                            time_display = self._safe_index_to_datetime(time)
                        if type == 1:
                            writer.writerow([ids[0],ids[1],lag,time_idx,time_display,"into"])
                        elif type == -1:
                            writer.writerow([ids[0],ids[1],lag,time_idx,time_display,"out_of"])
                        elif type == 0:
                            writer.writerow([ids[0],ids[1],lag,time_idx,time_display,"changed_sign"])

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
        self._append_correlated_artifact(prefix)
        self._append_status_artifact(prefix)
        self._append_anomalies_artifact(prefix)

    def _append_correlated_artifact(self, prefix):
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
        state = self._artifact_state
        if not getattr(self, "corr_lengths", None):
            return
        rows = []
        for key, entries in self.corr_lengths.items():
            ids = (key[0], key[1])
            lag = key[2]
            for t1, t2, _w, corr_len, corr_sign in entries:
                time1 = self._safe_index_to_datetime(t1)
                time2 = self._safe_index_to_datetime(t2)
                tracker_key = (str(ids[0]), str(ids[1]), str(lag), str(time1), str(time2))
                prev = state["status"].get(tracker_key)
                current = (corr_len, str(corr_sign))
                if prev == current:
                    continue
                state["status"][tracker_key] = current
                rows.append([ids[0], ids[1], lag, time1, time2, corr_len, corr_sign])
        if rows:
            self._artifact_write_rows(
                f"{prefix}_status.csv",
                ["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"],
                rows,
            )

    def _append_anomalies_artifact(self, prefix):
        state = self._artifact_state
        if not getattr(self, "corr_anomalies", None):
            return
        rows = []
        for key, entries in self.corr_anomalies.items():
            ids = (key[0], key[1])
            lag = key[2]
            for time, kind in entries:
                time_val = self._safe_index_to_datetime(time)
                if kind == 1:
                    label = "into"
                elif kind == -1:
                    label = "out_of"
                elif kind == 0:
                    label = "changed_sign"
                else:
                    label = str(kind)
                tracker_key = (str(ids[0]), str(ids[1]), str(lag), str(time), str(time_val), label)
                if tracker_key in state["anomalies"]:
                    continue
                state["anomalies"].add(tracker_key)
                rows.append([ids[0], ids[1], lag, time, time_val, label])
        if rows:
            self._artifact_write_rows(
                f"{prefix}_anomalies.csv",
                ["id1", "id2", "lag", "time_idx", "time", "anomaly"],
                rows,
            )
    
    def _print_state(self,):
        if len(self.correlated)>0:
            print("\nCorrelated:")
            self._print_correlated()
            print("\nStatus:")
            self._print_monitor_status()
            print("\nAnomalies:")
            self._print_anomalies()

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
    
    def _run_grids(self, verbose, testing, worker_mode=None):
        """
        Run each grid node in parallel via run() and merge the local outputs
        into self.freq_pairs and self.candidates on the main thread.
        """
        n_ids = len(self.series_ids)

        worker_mode = worker_mode if worker_mode else self.exec

        thread_payloads = [
            (self, g, n_ids, verbose, testing)
            for g in range(self.n_grids)
        ]

        payloads = thread_payloads

        if worker_mode == "sequential":
            grid_results = [_grid_worker(payload) for payload in payloads]
        else:
            t0 = time.perf_counter() if self.profile_enabled else None
            grid_results = self._parallel_map(
                _grid_worker,
                payloads,
                mode=worker_mode,
                max_workers=self.n_nodes,
                preserve_order=False,
            )
            if t0 is not None:
                self._profile_add("grid.dispatch", time.perf_counter() - t0)

        # Merge
        self.freq_pairs = {}
        self.uncorrelated = {}

        iterable = (
            (
                g_index,
                node_obj,
                result,
            )
            for g_index, node_obj, result in grid_results
        )

        t0 = time.perf_counter() if self.profile_enabled else None
        for g_index, node_obj, (loc_freq, _loc_cand, loc_unc) in iterable:
            self.grid_nodes[g_index] = node_obj
            for k in sorted(loc_freq):
                v = float(loc_freq[k])
                self.freq_pairs[k] = self.freq_pairs.get(k, 0.0) + v
            for k in sorted(loc_unc):
                self.uncorrelated[k] = loc_unc[k]
        if t0 is not None:
            self._profile_add("grid.merge", time.perf_counter() - t0)

        # Recompute candidates after frequencies from all grids are merged so
        # that threshold checks see the combined vote count (matches the
        # original sequential implementation behaviour).
        if self.freq_threshold <= 0:
            base_candidates = set(self.freq_pairs.keys())
        else:
            base_candidates = {
                pair
                for pair, count in self.freq_pairs.items()
                if count >= self.freq_threshold
            }

        self.candidates = {pair: 1 for pair in base_candidates}
        
    def print_sketch_hist(self):
        flattened = [item for sublist in self.hist_sketches for item in sublist]
        print("\nMean of sketches:",np.mean(flattened),", standard deviation:",np.std(flattened))
        plt.hist(flattened)
        plt.show()
    
    def get_correlation_flags(self, total_time, tolerance=None):
        if tolerance is None:
            tolerance = self.window_size

        # Merge events by normalized key
        merged_anomalies = {}
        for key, events in self.corr_anomalies.items():
            norm_key = tuple(sorted([key[0], key[1]]) + [key[2]])
            merged_anomalies.setdefault(norm_key, []).extend(events)

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

        merged_anomalies = {}
        for key, events in self.corr_anomalies.items():
            norm_key = tuple(sorted([key[0], key[1]]) + [key[2]])
            merged_anomalies.setdefault(norm_key, []).extend(events)

        merged_corr_lengths = {}
        for key, values in self.corr_lengths.items():
            norm_key = tuple(sorted([key[0], key[1]]) + [key[2]])
            merged_corr_lengths.setdefault(norm_key, []).extend(values)

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
                                status[3] for status in merged_corr_lengths.get(key, [])
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
    ):
        if windows:
            return CorrTrack._compute_metrics_bf_windows(
                predicted,
                ground_truth,
                pair_min_dist,
                total_pairs_bf=total_pairs_bf,
            )
        return CorrTrack.compute_metrics_bf_timestamps(predicted, ground_truth)
    
    def _compute_metrics_bf_windows(predicted: dict, ground_truth: dict, pair_min_dist=None, total_pairs_bf=None):
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

        # pos
        precision_pos, recall_pos, _ = _safe_prec_recall(pred_pos, gt_pos)
        f1_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos) if (precision_pos+recall_pos)>0 else 0.0

        # neg
        precision_neg, recall_neg, _ = _safe_prec_recall(pred_neg, gt_neg)
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

        # Metric computations from set representation.
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
            return {}

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
            return {}

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

        merged = {}
        for s, bf_node, result in bf_results:
            if bf_node is not None:
                self.brute_force_nodes[s] = bf_node
            for k in sorted(result):
                merged[k] = result[k]
        return merged
    
    def run_bf(self, new_data_step, ids, verbose, testing, corr_val=True, monitor=True):
        self.verbose = verbose
        self.testing = testing

        cand_mode = "thread" if self.parallel_candidates else "sequential"
        val_mode = "thread" if self.parallel_validation else "sequential"

        self._update_curr_data(new_data_step, ids)

        # candidates
        self.candidates = {}
        start_time = time.time()
        
        if self.parallel_candidates:
            self.candidates = self._run_bf_parallel(
                self._curr_window_step(),
                self.ids,
                verbose=verbose,
                testing=testing,
                worker_mode=cand_mode,
            )
        else:
            if not self.brute_force_nodes:
                self.brute_force_nodes.append(Candidates_BF(self.window_size,self.window_step,self.n_lags,self.corr_threshold))
            # original single-thread path
            self.candidates = self.brute_force_nodes[0].run(self._curr_window_step(), self.ids, verbose, testing, ref_ids=None)
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
        return self.window_index[-self.curr_window_size+self.basic_window]

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
    
    def _generate_randomVectors(self):
        base_rng = np.random.RandomState(self.seed_randomVector)
        self.basicRandomVector = (1/np.sqrt(self.n_vectors))*base_rng.choice([1,-1], size=(self.n_vectors,self.basic_window), replace=True)
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
            self.sketches = {}
            self._sketch_matrix = None
            self._sketch_keys = []
            return

        window_blocks = current_window.reshape(n_series, self.n_basic_windows, self.basic_window)
        weights = np.array(self._toggle_weights, dtype=np.float64, copy=False)

        norm_mode = 1 if (self.sketch_norm or "z").lower() == "mean_l2" else 0
        if _cy_build_sketch_matrix is not None:
            mean_vec, random_sums = self._kernel_norm_inputs(n_series, norm_mode)
            if mean_vec is not None and random_sums is not None:
                series_dots, raw_matrix, norm_matrix = _cy_build_sketch_matrix(
                    np.ascontiguousarray(window_blocks, dtype=np.float64),
                    np.ascontiguousarray(weights, dtype=np.float64),
                    mean_vec,
                    random_sums,
                    int(norm_mode),
                )
            else:
                series_dots = _compute_series_dots(window_blocks, weights)
                raw_matrix = np.sum(series_dots, axis=1)
                norm_matrix = self._normalize_sketch_matrix(raw_matrix)
        else:
            series_dots = _compute_series_dots(window_blocks, weights)
            raw_matrix = np.sum(series_dots, axis=1)
            norm_matrix = self._normalize_sketch_matrix(raw_matrix)
        curr_start = self._curr_startTime()
        window_size = self.window_size

        self.basicDots.append(np.array(series_dots, dtype=np.float64, copy=True))
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
                    self._sketch_matrix = None
                    self._sketch_keys = []
                    continue
                curr_start = self._curr_startTime()
                window_size = self.window_size
                diff = np.asarray(self.intermediary_diff_toggleVector, dtype=np.float64)
                series_dots = base[:, 1:(self.n_basic_windows+1), :] * diff
                self.basicDots.append(series_dots)
                raw_matrix = np.sum(series_dots, axis=1)

                norm_mode = 1 if (self.sketch_norm or "z").lower() == "mean_l2" else 0
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
            new_dots = _compute_series_dots(new_blocks, weights_subset)
        else:
            new_dots = np.zeros((n_series, 0, self.n_vectors), dtype=np.float64)

        if(self.verbose):
            self._print_curr_window()
        
        self._clean_obsolete_basicDots()
        self._print_incremental_step()
        self.incrementable_index.append(self._curr_incrementable_startTime())
        self.sketches = {}
        same_start = self.previous_startTime == self._curr_startTime()
        curr_start = self._curr_startTime()
        window_size = self.window_size
        raw_matrix = np.empty((n_series, self.n_vectors), dtype=np.float64)
        if same_start and len(self.basicDots) >= 2:
            base = np.asarray(self.basicDots[-2], dtype=np.float64)
        else:
            base_source = np.asarray(self.basicDots[0], dtype=np.float64)[:, 1:, :]
            base = base_source * np.asarray(self.diff_toggleVector, dtype=np.float64)

        if base.size == 0 and new_dots.size == 0:
            updated = base.reshape(n_series, 0, self.n_vectors)
        elif base.size == 0:
            updated = new_dots
        elif new_dots.size == 0:
            updated = base
        else:
            updated = np.concatenate((base, new_dots), axis=1)
        self.basicDots.append(updated)
        raw_matrix = np.sum(updated, axis=1)

        norm_mode = 1 if (self.sketch_norm or "z").lower() == "mean_l2" else 0
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
        norm = float(np.linalg.norm(arr))
        if not np.isfinite(norm) or norm <= 0.0:
            return np.zeros_like(arr)
        return arr / norm

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
        time_arr = np.fromiter((int(k[1]) for k in keys), dtype=np.int64, count=len(keys))
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
            partition = {}
            for i, key in enumerate(keys):
                vec = np.asarray(matrix[i], dtype=np.float64)
                is_const = bool(const_flags[i]) or np.linalg.norm(vec) == 0.0
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
    
    def _get_candidates(self,candidates):
        if self.curr_window_size < self.window_size:
            if(self.verbose):
                self._print_curr_window()
                print("\nSkip distance calculations (window not full)")
            return None

        self.brute_force_steps += 1

        m = self.window_data.shape[0]
        ref_indices = self.ref_indices if self.ref_indices is not None else range(m)
        rows = _enumerate_candidate_rows(
            self.window_data,
            self.window_index,
            ref_indices,
            self.window_size,
            self.window_step,
        )
        if rows is None:
            return None

        for s_idx, k_idx, start_s, start_k, w in rows:
            id1 = self.series_ids[s_idx]
            id2 = self.series_ids[k_idx]
            pair = (id1, id2, int(start_s), int(start_k), int(w))
            candidates[self._normalize_key(pair)] = 1
    
    def run(self,new_data_step,ids, verbose=True, testing=False, ref_ids=None):
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

class Candidates:
    def __init__(self,n_lagged_windows,grid_dimension,cell_size,grid_max,freq_threshold,corr_threshold,n_vectors,sketch_std,n_grids,neg_corr,
                 sign_prefilter_scale=1.3,sign_prefilter_extra=1, seed=None, full_vector=False, candidate_backend=None):
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
        self._sid_idx_map = {}
        self._win_sid = []
        self._win_sid_idx = []
        self._win_time = []
        self._win_w = []
        self._tree_index = None
        # Parameters thresholds
        self.freq_threshold = freq_threshold
        self.corr_threshold = corr_threshold
        self.sign_prefilter_scale = float(sign_prefilter_scale)
        self.sign_prefilter_extra = int(sign_prefilter_extra)

        self.sketch_std = sketch_std
        self.n_vectors = n_vectors
        self._refresh_vector_mode()
        seed = _to_int_safe(seed)
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        backend_value = candidate_backend
        if backend_value is None:
            backend_value = os.environ.get("CORRTRACK_CANDIDATE_BACKEND", "auto")
        self._requested_candidate_backend = _resolve_candidate_backend(backend_value, default="auto")
        self._candidate_backend = "flat"
        if self._requested_candidate_backend in {"bptree", "auto"} and _cy_balanced_index_cls is not None:
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
        elif self._requested_candidate_backend == "flat":
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

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._reverse_index = defaultdict(list, getattr(obj, "_reverse_index", {}))
        obj._reverse_vectors = defaultdict(list, getattr(obj, "_reverse_vectors", {}))
        obj._reverse_entry_ids = defaultdict(list, getattr(obj, "_reverse_entry_ids", {}))
        obj._recent_entry_ids = list(getattr(obj, "_recent_entry_ids", []))
        obj._window_idx = dict(getattr(obj, "_window_idx", {}))
        obj._sid_idx_map = dict(getattr(obj, "_sid_idx_map", {}))
        obj._win_sid = list(getattr(obj, "_win_sid", []))
        obj._win_sid_idx = list(getattr(obj, "_win_sid_idx", []))
        obj._win_time = list(getattr(obj, "_win_time", []))
        obj._win_w = list(getattr(obj, "_win_w", []))
        requested_backend = _resolve_candidate_backend(
            getattr(obj, "_requested_candidate_backend", getattr(obj, "_candidate_backend", "auto")),
            default="auto",
        )
        obj._requested_candidate_backend = requested_backend
        if getattr(obj, "_candidate_backend", None) not in {"flat", "bptree"}:
            obj._candidate_backend = "flat"
        if obj._candidate_backend == "bptree" and _cy_balanced_index_cls is None:
            obj._candidate_backend = "flat"
            obj._tree_index = None
        obj._refresh_vector_mode()
        return obj

    def load_state(self, state):
        self.__dict__.update(state)
        self._reverse_index = defaultdict(list, getattr(self, "_reverse_index", {}))
        self._reverse_vectors = defaultdict(list, getattr(self, "_reverse_vectors", {}))
        self._reverse_entry_ids = defaultdict(list, getattr(self, "_reverse_entry_ids", {}))
        self._recent_entry_ids = list(getattr(self, "_recent_entry_ids", []))
        self._window_idx = dict(getattr(self, "_window_idx", {}))
        self._sid_idx_map = dict(getattr(self, "_sid_idx_map", {}))
        self._win_sid = list(getattr(self, "_win_sid", []))
        self._win_sid_idx = list(getattr(self, "_win_sid_idx", []))
        self._win_time = list(getattr(self, "_win_time", []))
        self._win_w = list(getattr(self, "_win_w", []))
        requested_backend = _resolve_candidate_backend(
            getattr(self, "_requested_candidate_backend", getattr(self, "_candidate_backend", "auto")),
            default="auto",
        )
        self._requested_candidate_backend = requested_backend
        if getattr(self, "_candidate_backend", None) not in {"flat", "bptree"}:
            self._candidate_backend = "flat"
        if self._candidate_backend == "bptree" and _cy_balanced_index_cls is None:
            self._candidate_backend = "flat"
            self._tree_index = None
        self._refresh_vector_mode()

    def append_partition(self,curr_time,new_partition):
        new_partition = self._partition_to_dict(new_partition)
        if new_partition is None:
            return
        if self.curr_time != curr_time:
            self.curr_time = curr_time
            self.partition.append(new_partition)
        else:
            self.partition[-1].update(new_partition)
        self._recent_window_ids = set()
        self._recent_entry_ids = []

    def set_sid_list(self, sid_list):
        if not sid_list:
            return
        if self._sid_list == list(sid_list):
            return
        self._sid_list = list(sid_list)

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
        self._window_idx[window_id] = idx
        self._win_sid.append(sid)
        self._win_sid_idx.append(int(sid_idx))
        self._win_time.append(int(start_time))
        self._win_w.append(int(window_size))
        return idx

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
                return True
            idx += 1
        return False

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

    def _input_tree(self):
        if not self.partition:
            return
        last_partition = self.partition[-1]
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
            if self._vector_match_enabled:
                value = float(vec[0])
                if self._candidate_backend == "bptree" and self._tree_index is not None:
                    entry_id = self._tree_index.insert(value, win_idx, vec)
                    self._reverse_entry_ids[k].append(entry_id)
                    self._recent_entry_ids.append(entry_id)
                    if self.neg_corr:
                        neg_vec = -vec
                        neg_value = -value
                        neg_entry_id = self._tree_index.insert(neg_value, win_idx, neg_vec)
                        self._reverse_entry_ids[k].append(neg_entry_id)
                        self._recent_entry_ids.append(neg_entry_id)
                else:
                    self._insert_entry(value, k, vec)
                    self._reverse_index[k].append(value)
                    self._reverse_vectors[k].append(vec)
                    if self.neg_corr:
                        neg_vec = -vec
                        neg_value = -value
                        self._insert_entry(neg_value, k, neg_vec)
                        self._reverse_index[k].append(neg_value)
                        self._reverse_vectors[k].append(neg_vec)
                self.sketches[k] = vec
            else:
                value = float(vec[0])
                if self._candidate_backend == "bptree" and self._tree_index is not None:
                    entry_id = self._tree_index.insert(value, win_idx, None)
                    self._reverse_entry_ids[k].append(entry_id)
                    self._recent_entry_ids.append(entry_id)
                    if self.neg_corr:
                        neg_value = -value
                        neg_entry_id = self._tree_index.insert(neg_value, win_idx, None)
                        self._reverse_entry_ids[k].append(neg_entry_id)
                        self._recent_entry_ids.append(neg_entry_id)
                else:
                    self._insert_entry(value, k)
                    self._reverse_index[k].append(value)
                    if self.neg_corr:
                        neg_value = -value
                        self._insert_entry(neg_value, k)
                        self._reverse_index[k].append(neg_value)
                self.sketches[k] = value
            self._recent_window_ids.add(k)

    def _clean_old_sketches(self):
        if len(self.partition) > self.n_lagged_windows:
            old_partition = self.partition.pop(0)
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
                    for entry_id in entry_ids:
                        self._tree_index.remove(entry_id)
                else:
                    for val in values:
                        self._remove_entry(val, k)
                self.sketches.pop(k, None)

    def _update_grid(self, n_ids):
        if self.partition and len(self.partition[-1]) >= n_ids:
            self._clean_old_sketches()
            self._input_tree()
    
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

        window_idx = {}
        win_sid = []
        win_time = []
        win_w = []
        sid_idx_map = {}
        win_sid_idx = []

        for entry in self._entries:
            window_id = entry[2]
            if window_id in window_idx:
                continue
            idx = len(win_sid)
            window_idx[window_id] = idx
            sid, start_time, window_size = window_id
            win_sid.append(sid)
            win_time.append(int(start_time))
            win_w.append(int(window_size))
            sid_idx = sid_idx_map.get(sid)
            if sid_idx is None:
                sid_idx = len(sid_idx_map)
                sid_idx_map[sid] = sid_idx
            win_sid_idx.append(sid_idx)

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

        return (
            values,
            value_window_idx,
            recent_values,
            recent_window_idx,
            np.asarray(win_sid_idx, dtype=np.int64),
            np.asarray(win_time, dtype=np.int64),
            np.asarray(win_w, dtype=np.int64),
            entry_vectors,
            recent_vectors,
            win_sid,
        )

    def _increment_candidates(self, freq_pairs, candidates):
        if not self._recent_window_ids:
            return
        if self.cell_size is None:
            return
        tau = float(self.cell_size)
        if tau < 0.0:
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
            recent_entry_ids = np.asarray(self._recent_entry_ids, dtype=np.int64)
            win_sid_idx = np.asarray(self._win_sid_idx, dtype=np.int64)
            win_time = np.asarray(self._win_time, dtype=np.int64)
            if self._vector_match_enabled:
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
                seen_pairs = set()
                for ridx, other_idx in pairs:
                    sid1 = self._win_sid[ridx]
                    sid2 = self._win_sid[other_idx]
                    t1 = self._win_time[ridx]
                    t2 = self._win_time[other_idx]
                    w = self._win_w[ridx]
                    pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
                    if pair_id in seen_pairs:
                        continue
                    seen_pairs.add(pair_id)
                    freq_pairs[pair_id] = freq_pairs.get(pair_id, 0.0) + 1.0
                    if freq_pairs[pair_id] >= self.freq_threshold:
                        candidates[pair_id] = 1
            return

        if self._candidate_backend == "bptree":
            return

        cython_ready = self._candidate_backend == "flat" and (
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
                if self._vector_match_enabled:
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
                    for ridx, other_idx in pairs:
                        sid1 = win_sid[ridx]
                        sid2 = win_sid[other_idx]
                        t1 = win_time[ridx]
                        t2 = win_time[other_idx]
                        w = win_w[ridx]
                        pair_id = self._normalize_key((sid1, sid2, int(t1), int(t2), int(w)))
                        if pair_id in seen_pairs:
                            continue
                        seen_pairs.add(pair_id)
                        freq_pairs[pair_id] = freq_pairs.get(pair_id, 0.0) + 1.0
                        if freq_pairs[pair_id] >= self.freq_threshold:
                            candidates[pair_id] = 1
                return

        tau_sq = tau * tau
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
                        diff = vec - other_vec
                        if np.dot(diff, diff) > tau_sq:
                            continue
                        pair_id = self._normalize_key((window_id[0], other_id[0], window_id[1], other_id[1], window_id[2]))
                        if pair_id in seen_pairs:
                            continue
                        seen_pairs.add(pair_id)
                        freq_pairs[pair_id] = freq_pairs.get(pair_id, 0.0) + 1.0
                        if freq_pairs[pair_id] >= self.freq_threshold:
                            candidates[pair_id] = 1
            else:
                for value in set(values):
                    lower = value - tau
                    upper = value + tau
                    for other_id in self._range_search_ids(lower, upper):
                        if other_id == window_id:
                            continue
                        if window_id[0] == other_id[0] and window_id[1] == other_id[1]:
                            continue
                        pair_id = self._normalize_key((window_id[0], other_id[0], window_id[1], other_id[1], window_id[2]))
                        if pair_id in seen_pairs:
                            continue
                        seen_pairs.add(pair_id)
                        freq_pairs[pair_id] = freq_pairs.get(pair_id, 0.0) + 1.0
                        if freq_pairs[pair_id] >= self.freq_threshold:
                            candidates[pair_id] = 1

    def run(self, n_ids, verbose=True, testing=False):
        self.verbose = verbose
        self.testing = testing

        self._update_grid(n_ids)

        freq_pairs = {}
        candidates = {}
        uncorrelated = {}

        if len(self.partition)>0:
            self._increment_candidates(freq_pairs,candidates)

        return freq_pairs, candidates, uncorrelated


class CorrTrack_optimize:
    def __init__(self,train_data,ids,window_size,window_step,n_lags,corr_threshold,recall_by_window,alg,neg_corr,corr_val, exec="parallel",max_workers=0, sketch_norm="z", verbose=False, testing=False, parallel_sketch=None, parallel_candidates=None, parallel_validation=None, track_min_dist=False, artifact_mode="buffered", artifact_buffer_max_rows=250000):

        self.neg_corr = neg_corr
        self.corr_val = corr_val
        self.sketch_norm = sketch_norm or "z"
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
        self.recall_by_window = recall_by_window

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
        try:
            self.artifact_buffer_max_rows = max(1, int(artifact_buffer_max_rows))
        except (TypeError, ValueError):
            self.artifact_buffer_max_rows = 250000
        self._optim_artifact_dir = None
        self._bf_artifact_prefix = None
        self._bf_record = None
        self._metric_helper = None
        self._bf_online_reference_csv = None
        self._bf_online_stats = None
        self.corrtrack_bf = CorrTrack(window_size=self.window_size,basic_window=None,window_step=window_step,n_vectors=1,n_lags=self.n_lags,
                                    grid_dimension=1,cell_size=1,seed=None,seed_toggle=None,corr_threshold=self.corr_threshold,
                                    neg_corr=self.neg_corr,preprocess=False,exec=self.exec,max_workers=self.max_workers,
                                    sketch_norm=self.sketch_norm,parallel_sketch=self.parallel_sketch,parallel_candidates=self.parallel_candidates,parallel_validation=self.parallel_validation,
                                    track_min_dist=self.track_min_dist)
        
        self.window_step = self.corrtrack_bf.window_step
        self.basic_window = self.corrtrack_bf.basic_window

        self.alg = alg

    def _artifact_metric_helper(self):
        helper = getattr(self, "_metric_helper", None)
        if helper is None:
            helper = CorrTrack_compare.__new__(CorrTrack_compare)
            helper.window_size = self.window_size
            helper.recall_by_window = self.recall_by_window
            self._metric_helper = helper
        return helper

    def _artifact_prefix(self, stem):
        if not self._optim_artifact_dir:
            raise RuntimeError("Optimizer artifact directory is not configured")
        return os.path.join(self._optim_artifact_dir, stem)

    def _use_online_window_metrics(self):
        return bool(self.recall_by_window and not self.corr_val)

    def _monitor_for_metrics(self):
        return not self.recall_by_window

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _run_is_parallel(self) -> bool:
        return self.exec != "sequential"
    
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
    
    def _get_bf_ground_truth(self, dataset_id):
        """
        Compute brute-force ground truth through the buffered artifact execution path.
        """
        bf_prefix = self._artifact_prefix("bf")
        online_writer = None
        if self._use_online_window_metrics():
            self._bf_online_reference_csv = f"{bf_prefix}_online_windows.csv"
            online_writer = _OptimizerBFWindowReferenceWriter(
                self._bf_online_reference_csv,
                self.window_size,
            )
        metadata = {
            "mode": "bf",
            "alg": "bf",
            "optim": "baseline",
            "artifact_buffer_max_rows": self.artifact_buffer_max_rows,
        }

        try:
            bf_record, _runtime_parts, _corr_flags = execute_corrtrack_pass(
                "bf",
                dataset_id,
                self.corrtrack_bf,
                self.train_data,
                self.ids,
                metadata,
                corr_val=True,
                recall_by_window=self.recall_by_window,
                artifact_prefix=bf_prefix,
                artifact_mode=self.artifact_mode,
                verbose=self.verbose,
                testing=self.testing,
                monitor=self._monitor_for_metrics(),
                step_observer=online_writer.observe if online_writer is not None else None,
            )
        finally:
            if online_writer is not None:
                online_writer.close()

        self._bf_record = bf_record
        self._bf_artifact_prefix = bf_prefix
        self.runtime_bf = self._to_float(bf_record.get("runtime"))
        self.artifact_time_bf = self._to_float(bf_record.get("artifact_time"))
        self.candidate_time_bf = self._to_float(bf_record.get("cand_time"))
        self.validation_time_bf = self._to_float(bf_record.get("val_time"))
        self.monitor_time_bf = self._to_float(bf_record.get("monit_time"))
        self.correlated_bf = int(float(bf_record.get("correlated", 0) or 0))
        self.tested_bf = int(float(bf_record.get("tested", 0) or 0))
        self.total_bf = int(float(bf_record.get("total_candidates", 0) or 0))
        self.pair_min_dist_bf = self._artifact_metric_helper()._parse_pair_min_dist(
            bf_record.get("pair_min_dist")
        )
        if online_writer is not None:
            self._bf_online_stats = {
                "gt_total": online_writer.gt_total,
                "gt_pos": online_writer.gt_pos,
                "gt_neg": online_writer.gt_neg,
            }
        else:
            self._bf_online_reference_csv = None
            self._bf_online_stats = None
        print("Run Brute-Force, Finished in ", self.runtime_bf)
    
    def _init_optim_record(self, dataset_id, param_combo):
        record = {key: None for key in OPTIM_RESULT_COLUMNS}

        record["dataset_id"] = dataset_id
        record["alg"] = self.alg
        record["n_ts"] = self.train_data.shape[0] - 1
        n_w = int(np.floor((self.train_data.shape[1] - self.window_size) / self.window_step) + 1)
        record["n_w"] = n_w
        record["total_w"] = record["n_ts"] * n_w
        record["mem_w"] = None

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
        record["corr_threshold"] = self.corr_threshold
        record["grid_max"] = param_combo.get("grid_max")
        record["cell_stretch"] = param_combo.get("cell_size")
        record["cell_size"] = None
        record["n_vectors"] = param_combo.get("n_vectors")
        record["grid_dimension"] = param_combo.get("grid_dimension")
        record["freq_threshold"] = param_combo.get("freq_threshold")

        record["cand_time_bf"] = getattr(self, "candidate_time_bf", None)
        record["val_time_bf"] = getattr(self, "validation_time_bf", None)
        record["monit_time_bf"] = getattr(self, "monitor_time_bf", None)
        record["runtime_bf"] = getattr(self, "runtime_bf", None)
        record["artifact_time_bf"] = getattr(self, "artifact_time_bf", None)
        record["corr_w_bf"] = getattr(self, "correlated_bf", None)
        record["tested_w_bf"] = getattr(self, "tested_bf", None)
        record["cand_w_bf"] = getattr(self, "total_bf", None)
        record["artifact_time"] = None

        record["status"] = "pending"
        record["error"] = ""
        return record
    
    def _run_corrtrack(self, args):
        trial_index, param_combo, dataset_id = args
        record = self._init_optim_record(dataset_id, param_combo)

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

        feature_kwargs = _extract_feature_overrides(param_combo)

        record["nodes"] = nodes
        record["seed"] = seed
        record["seed_toggle"] = seed_toggle
        record["n_vectors"] = n_vectors
        record["preprocess"] = preprocess
        record["grid_dimension"] = grid_dimension
        record["cell_stretch"] = cell_stretch
        record["freq_threshold"] = freq_threshold

        length_data = self.train_data.shape[1]
        use_online_window_metrics = self._use_online_window_metrics()
        trial_prefix = None if use_online_window_metrics else self._artifact_prefix(f"trial_{int(trial_index):06d}")
        online_metrics = None

        try:
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
                exec=self.exec,
                max_workers=self.max_workers,
                parallel_sketch=self.parallel_sketch,
                parallel_candidates=self.parallel_candidates,
                parallel_validation=self.parallel_validation,
                track_min_dist=self.track_min_dist,
                **feature_kwargs,
            )

            record["grid_max"] = corrtrack.grid_max
            record["cell_size"] = corrtrack.cell_size
            record["n_lags"] = corrtrack.n_lags
            record["basic_window"] = corrtrack.basic_window
            record["grid_dimension"] = corrtrack.grid_dimension
            record["exec_mode"] = corrtrack.exec
            record["parallel_sketch"] = corrtrack.parallel_sketch
            record["parallel_candidates"] = corrtrack.parallel_candidates
            record["parallel_validation"] = corrtrack.parallel_validation
            record["candidate_backend"] = getattr(
                corrtrack,
                "candidate_backend_effective",
                getattr(corrtrack, "candidate_backend", None),
            )
            record["workers_total"] = corrtrack.n_nodes
            record["workers_sketch"] = corrtrack.n_sketch_nodes
            record["workers_candidates"] = corrtrack.n_candidate_nodes
            record["workers_validation"] = corrtrack.n_nodes if corrtrack.parallel_validation else 1
            metadata = {
                "mode": "optim",
                "alg": self.alg,
                "optim": "recall_speedup",
                "nodes": nodes,
                "seed": seed,
                "seed_toggle": seed_toggle,
                "preprocess": preprocess,
                "sketch_norm": self.sketch_norm,
                "candidate_backend": param_combo.get("candidate_backend"),
                "freq_threshold": freq_threshold,
                "artifact_buffer_max_rows": self.artifact_buffer_max_rows,
            }

            if use_online_window_metrics:
                if not self._bf_online_reference_csv or not self._bf_online_stats:
                    raise RuntimeError("Missing optimizer online BF reference for window metrics")
                online_metrics = _OptimizerOnlineWindowMetrics(
                    self._bf_online_reference_csv,
                    self.window_size,
                    self._bf_online_stats.get("gt_total", 0),
                    self._bf_online_stats.get("gt_pos", 0),
                    self._bf_online_stats.get("gt_neg", 0),
                    total_pairs_bf=self.tested_bf,
                    pair_min_dist=self.pair_min_dist_bf,
                )

            run_record, _runtime_parts, _corr_flags = execute_corrtrack_pass(
                "corrtrack",
                dataset_id,
                corrtrack,
                self.train_data,
                self.ids,
                metadata,
                corr_val=self.corr_val,
                recall_by_window=self.recall_by_window,
                artifact_prefix=trial_prefix,
                artifact_mode=self.artifact_mode,
                verbose=self.verbose,
                testing=self.testing,
                monitor=self._monitor_for_metrics(),
                step_observer=online_metrics.observe if online_metrics is not None else None,
            )

            copy_fields = (
                "mem_w",
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
            )
            for field in copy_fields:
                if field in run_record:
                    record[field] = run_record.get(field)

            record["corr_w"] = run_record.get("correlated")
            record["tested_w"] = run_record.get("tested")
            record["cand_w"] = run_record.get("total_candidates")

            runtime = self._to_float(record.get("runtime"))
            runtime_bf = self._to_float(record.get("runtime_bf"))
            record["speedup"] = (runtime_bf / runtime) if runtime > 0 else float("inf")

            record["speedup_ceil"] = _compute_speedup_ceil(
                record.get("cand_time_bf"),
                record.get("val_time_bf"),
                record.get("monit_time_bf"),
                record.get("sk_time"),
                record.get("cand_time"),
                record.get("corr_w_bf"),
                record.get("cand_w_bf"),
            )
            record["rel_speedup_eff"] = _safe_div(record.get("speedup"), record.get("speedup_ceil"))
            record["corr_prop"] = _safe_div(record.get("corr_w_bf"), record.get("cand_w_bf"))
            record["waste_val_bf"] = _safe_div(record.get("cand_w_bf"), record.get("corr_w_bf"))
            record["waste_val"] = _safe_div(record.get("cand_w"), record.get("corr_w"))
            record["rel_waste_red"] = _safe_div(record.get("waste_val_bf"), record.get("waste_val"))

            metric_helper = self._artifact_metric_helper()
            if online_metrics is not None:
                metrics = online_metrics.metrics()
            elif self.recall_by_window:
                metrics = metric_helper._stream_window_metrics_from_artifacts(
                    self._bf_artifact_prefix,
                    trial_prefix,
                    pair_min_dist=self.pair_min_dist_bf,
                    total_pairs_bf=self.tested_bf,
                )
            else:
                metrics = metric_helper._stream_timestamp_metrics_from_artifacts(
                    self._bf_artifact_prefix,
                    trial_prefix,
                    length_data,
                )

            record["precision_pos"] = metrics["precision_pos"]
            record["recall_pos"] = metrics["recall_pos"]
            record["f1_pos"] = metrics["f1_score_pos"]
            record["precision_neg"] = metrics["precision_neg"]
            record["recall_neg"] = metrics["recall_neg"]
            record["f1_neg"] = metrics["f1_score_neg"]
            record["precision"] = metrics["precision"]
            record["recall"] = metrics["recall"]
            record["specificity"] = metrics["specificity"]
            record["recall_min"] = metrics["recall_min"]
            record["f1"] = metrics["f1_score"]
            record["aucroc"] = metrics["aucroc"]
            record["pr_auc"] = metrics["pr_auc"]

            record["status"] = "success"
            record["error"] = ""
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"Skipped params={param_combo} due to error: {exc}")
            traceback.print_exc()
        finally:
            if online_metrics is not None:
                online_metrics.close()
            if trial_prefix:
                _remove_artifact_files(trial_prefix)

        return record

    def _run_options(self, param_grid, output_csv, dataset_id):
        names = list(param_grid.keys())
        tasks = [
            (idx, dict(zip(names, vals)), dataset_id)
            for idx, vals in enumerate(itertools.product(*param_grid.values()))
        ]

        self._optim_artifact_dir = os.path.splitext(os.path.abspath(output_csv))[0] + "_artifacts"
        os.makedirs(self._optim_artifact_dir, exist_ok=True)
        self._bf_artifact_prefix = None
        self._bf_record = None
        self._metric_helper = None
        try:
            self._get_bf_ground_truth(dataset_id)

            if os.path.exists(output_csv):
                os.remove(output_csv)

            total_tasks = len(tasks)
            completed = 0
            with CSVStreamWriter(output_csv, OPTIM_RESULT_COLUMNS) as writer:
                for record in self._outer_iter(tasks, self._run_corrtrack, unordered=True):
                    if not record:
                        continue
                    writer.write_row(_row_from_mapping(OPTIM_RESULT_COLUMNS, record))
                    completed += 1
                    status = record.get("status", "unknown")
                    print(f"[Optim] Completed {completed}/{total_tasks} ({status})")
        finally:
            _remove_artifact_files(self._bf_artifact_prefix)
            if self._bf_online_reference_csv:
                try:
                    os.remove(self._bf_online_reference_csv)
                except OSError:
                    pass
            try:
                if self._optim_artifact_dir and os.path.isdir(self._optim_artifact_dir) and not os.listdir(self._optim_artifact_dir):
                    os.rmdir(self._optim_artifact_dir)
            except OSError:
                pass
            self._bf_artifact_prefix = None
            self._bf_record = None
            self._bf_online_reference_csv = None
            self._bf_online_stats = None
    
    def get_optim_params(self, param_grid, output_csv, dataset_id, run=True, target_recall=0.95):
        output_csv = output_csv + "_" + self.alg + ".csv"
        if run:
            self._run_options(param_grid, output_csv, dataset_id)

        delimiter = _detect_csv_delimiter(output_csv, default=",")
        metrics = pd.read_csv(output_csv, sep=delimiter)
        if "status" in metrics.columns:
            metrics = metrics[metrics["status"] == "success"]
        if metrics.empty:
            raise ValueError(f"No successful parameter combinations found in {output_csv}")

        subset = metrics[(metrics["speedup"] > 1) & (metrics["freq_threshold"] < 1)]
        if not subset.empty:
            recall_subset = subset[subset["recall"] > target_recall]
            if recall_subset.empty:
                max_recall = subset["recall"].max()
                recall_subset = subset[subset["recall"] == max_recall]
            bst_params = recall_subset.sort_values("speedup", ascending=False).head(1)
        else:
            bst_params = metrics.sort_values("speedup", ascending=False).head(1)

        return bst_params
        #return self.ground_truth, self.runtime_bf, CorrTrack_HyperOptim._skyline_query(metrics, ref_metrics) 

class CorrTrack_compare:
    def __init__(self,train_data,test_data,ids,window_size,window_step,basic_window,n_lags,corr_threshold,param_grid,recall_by_window,neg_corr,corr_val,algs=None, exec="parallel", max_workers=0, sketch_norm="z", verbose=False, testing=False, parallel_sketch=None, parallel_candidates=None, parallel_validation=None, monitor=True, track_min_dist=True):
        
        self.neg_corr = neg_corr
        self.corr_val = corr_val
        self.monitor = _coerce_to_bool(monitor)
        self.track_min_dist = _coerce_to_bool(track_min_dist, default=True)
        self.sketch_norm = sketch_norm or "z"
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
        self.recall_by_window = recall_by_window

        self.algs = algs or ["nD"]
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
                                    sketch_norm=self.sketch_norm,parallel_sketch=self.parallel_sketch,parallel_candidates=self.parallel_candidates,parallel_validation=self.parallel_validation,
                                    track_min_dist=self.track_min_dist)
        
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
        overrides = feature_overrides or {}
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

    def _canonical_window_key(self, id1, id2, time1, time2):
        return _normalize_window_metric_key(id1, id2, time1, time2, self.window_size)

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

    def _anomalies_csv_path(self, artifact_prefix):
        if not artifact_prefix:
            return None
        if artifact_prefix.endswith("_anomalies.csv"):
            return artifact_prefix
        return f"{artifact_prefix}_anomalies.csv"

    def _iter_correlated_rows(self, artifact_prefix):
        csv_path = self._correlated_csv_path(artifact_prefix)
        if not csv_path or not os.path.exists(csv_path):
            return
        delimiter = _detect_csv_delimiter(csv_path, default=CSV_DELIMITER)
        with open(csv_path, newline="") as file:
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
                yield self._canonical_window_key(id1, id2, time1, time2), corr_val

    def _iter_maxlag_rows(self, csv_path):
        if not csv_path or not os.path.exists(csv_path):
            return
        delimiter = _detect_csv_delimiter(csv_path, default=CSV_DELIMITER)
        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                id1 = row.get("id1")
                id2 = row.get("id2")
                corr_val = row.get("max_corr")
                if id1 is None or id2 is None or corr_val is None:
                    continue
                key_time = self._normalize_time_key(row.get("time1"), row.get("t1_index"))
                if key_time is None:
                    continue
                try:
                    corr_val = float(corr_val)
                except (TypeError, ValueError):
                    continue
                yield (id1, id2, key_time), corr_val

    def _stream_window_metrics_from_artifacts(self, ground_truth_prefix, predicted_prefix, pair_min_dist=None, total_pairs_bf=None):
        metrics = _empty_stream_metrics(include_sign=True)
        gt_iter = self._iter_correlated_rows(ground_truth_prefix)
        pred_iter = self._iter_correlated_rows(predicted_prefix)
        gt_row = next(gt_iter, None)
        pred_row = next(pred_iter, None)

        gt_total = 0
        pred_total = 0
        tp = 0
        gt_pos = gt_neg = pred_pos = pred_neg = 0
        tp_pos = tp_neg = 0

        recall_min_key = self._canonical_pair_min_dist_key(pair_min_dist)
        recall_min_hit = False

        while gt_row is not None or pred_row is not None:
            if pred_row is None:
                gt_key, gt_val = gt_row
                gt_total += 1
                if gt_val > 0:
                    gt_pos += 1
                else:
                    gt_neg += 1
                gt_row = next(gt_iter, None)
                continue
            if gt_row is None:
                pred_key, pred_val = pred_row
                pred_total += 1
                if pred_val > 0:
                    pred_pos += 1
                else:
                    pred_neg += 1
                if recall_min_key is not None and pred_key == recall_min_key:
                    recall_min_hit = True
                pred_row = next(pred_iter, None)
                continue

            gt_key, gt_val = gt_row
            pred_key, pred_val = pred_row

            if gt_key == pred_key:
                gt_total += 1
                pred_total += 1
                tp += 1
                if gt_val > 0:
                    gt_pos += 1
                else:
                    gt_neg += 1
                if pred_val > 0:
                    pred_pos += 1
                else:
                    pred_neg += 1
                if gt_val > 0 and pred_val > 0:
                    tp_pos += 1
                if gt_val <= 0 and pred_val <= 0:
                    tp_neg += 1
                if recall_min_key is not None and pred_key == recall_min_key:
                    recall_min_hit = True
                gt_row = next(gt_iter, None)
                pred_row = next(pred_iter, None)
            elif gt_key < pred_key:
                gt_total += 1
                if gt_val > 0:
                    gt_pos += 1
                else:
                    gt_neg += 1
                gt_row = next(gt_iter, None)
            else:
                pred_total += 1
                if pred_val > 0:
                    pred_pos += 1
                else:
                    pred_neg += 1
                if recall_min_key is not None and pred_key == recall_min_key:
                    recall_min_hit = True
                pred_row = next(pred_iter, None)

        fp = max(pred_total - tp, 0)
        precision, recall = _safe_prec_recall_from_counts(tp, pred_total, gt_total)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_pos, recall_pos = _safe_prec_recall_from_counts(tp_pos, pred_pos, gt_pos)
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0.0

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
                    if not active:
                        if last_out_time is None or (time_idx - last_out_time) > tolerance:
                            active = True
                            start_time = time_idx
                elif marker == -1:
                    if active and start_time is not None:
                        intervals.append((int(start_time), int(time_idx)))
                        active = False
                        last_out_time = time_idx

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
                full_vector_candidates=not _resolve_parallel_flag(
                    record.get("parallel_candidates"),
                    False,
                ),
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

        return [
            dataset_id,
            record.get("mode", "main") or "main",
            record.get("alg", ""),
            record.get("optim", ""),
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

    def compare_from_artifacts(self, dataset_id, bf_run_csv, corrtrack_run_files, output_csv):
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

            maxlag_metrics = (0.0, 0.0, 0.0, float("nan"), float("nan"))
            if bf_maxlag_csv and prefix:
                candidate_maxlag_csv = f"{prefix}_max_lag_correlated.csv"
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

                maxlag_metrics = (0.0, 0.0, 0.0, float("nan"), float("nan"))
                if bf_maxlag_csv and os.path.exists(filcorr_maxlag_csv):
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
            full_vector_candidates=not self.parallel_candidates,
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
            bst["freq_threshold"], bst["preprocess"],
            feature_kwargs,
        )
        speedup = runtime_bf / runtime if runtime else float("inf")
        metrics = CorrTrack.compute_metrics_bf(
            corr_flags,
            corr_flags_bf,
            self.recall_by_window,
            self.pair_min_dist_bf,
            total_pairs_bf=self.tested_bf,
        )
        bf_artifact_time = getattr(self, "artifact_time_bf", 0.0)

        bf_maxlag_csv = os.path.join(path, "bf_max_lag_correlated.csv")
        candidate_maxlag_csv = os.path.join(path, f"{prefix}_max_lag_correlated.csv")
        maxlag_precision, maxlag_recall, maxlag_f1, maxlag_diff_mean, maxlag_diff_std = self._compute_maxlag_metrics(
            bf_maxlag_csv,
            candidate_maxlag_csv,
        )

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
            else:
                param_values.append(bst.get(key))

        return [
            dataset_id, mode, alg, "recall_speedup", n_ts, n_w, total_w, mem_w,
        ] + param_values + [
            f"{self.candidate_time_bf:.4f}",f"{self.validation_time_bf:.4f}",f"{self.monitor_time_bf:.4f}",f"{self.runtime_bf:.4f}",f"{bf_artifact_time:.4f}",
            f"{runtime_parts[0]:.4f}",f"{runtime_parts[1]:.4f}",f"{runtime_parts[2]:.4f}",f"{runtime_parts[3]:.4f}",
            f"{runtime:.4f}", f"{artifact_time:.4f}", f"{speedup:.4f}",
            _format_float(speedup_ceil),
            _format_float(rel_speedup_eff),
            int(self.correlated_bf), int(self.correlated_w),
            int(self.tested_bf), int(self.tested_w),
            int(self.total_bf), int(self.total_w),
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
            return self.compare_from_artifacts(dataset_id, bf_run_csv, corrtrack_run_files, output_csv)
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
