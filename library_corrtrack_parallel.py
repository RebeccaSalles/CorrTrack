import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import csv
import time
import datetime
import os
import ast
import pandas as pd
import itertools
from itertools import combinations, product, repeat
from collections import defaultdict
import math
from scipy.stats import norm
from multiprocessing import shared_memory
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Iterable, Optional, Sequence

try:
    from dask import delayed, compute
    from dask.threaded import get as dask_threaded_get
    from dask.multiprocessing import get as dask_multiprocessing_get
    _HAS_DASK = True
except ImportError:  # pragma: no cover
    delayed = compute = dask_threaded_get = dask_multiprocessing_get = None
    _HAS_DASK = False


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
    "window_size",
    "window_step",
    "basic_window",
    "n_lags",
    "warmup_size",
    "seed",
    "seed_toggle",
    "preprocess",
    "extra_filters",
    "corr_threshold",
    "grid_max",
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
    "window_size",
    "window_step",
    "basic_window",
    "n_lags",
    "warmup_size",
    "seed",
    "seed_toggle",
    "preprocess",
    "extra_filters",
    "corr_threshold",
    "grid_max",
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
    "corr_w_bf",
    "corr_w",
    "tested_w_bf",
    "tested_w",
    "cand_w_bf",
    "cand_w",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "precision_neg",
    "recall_neg",
    "f1_neg",
    "precision",
    "recall",
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
    "window_size",
    "window_step",
    "basic_window",
    "n_lags",
    "warmup_size",
    "seed",
    "seed_toggle",
    "preprocess",
    "extra_filters",
    "corr_threshold",
    "grid_max",
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
    "corr_w_bf",
    "corr_w",
    "tested_w_bf",
    "tested_w",
    "cand_w_bf",
    "cand_w",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "precision_neg",
    "recall_neg",
    "f1_neg",
    "precision",
    "recall",
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

        file_exists = os.path.exists(path)
        needs_header = True
        if file_exists and os.path.getsize(path) > 0:
            existing_header: Optional[Sequence[str]] = None
            try:
                with open(path, newline="") as existing_file:
                    reader = csv.reader(existing_file)
                    existing_header = next(reader, None)
            except Exception:
                existing_header = None

            if existing_header:
                if len(existing_header) != len(self.columns) or tuple(existing_header) != self.columns:
                    tmp_path = f"{path}.tmp"
                    with open(path, newline="") as existing_file, open(tmp_path, "w", newline="") as tmp_file:
                        reader = csv.reader(existing_file)
                        writer = csv.writer(tmp_file)
                        writer.writerow(self.columns)
                        first_row = True
                        for row in reader:
                            if first_row:
                                first_row = False
                                continue
                            adjusted = list(row)
                            if len(adjusted) < len(self.columns):
                                adjusted.extend([""] * (len(self.columns) - len(adjusted)))
                            elif len(adjusted) > len(self.columns):
                                adjusted = adjusted[: len(self.columns)]
                            writer.writerow(adjusted)
                    os.replace(tmp_path, path)
                    needs_header = False
                else:
                    needs_header = False
            else:
                needs_header = True

        self._fh = open(path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if needs_header:
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
    return [data.get(column, "") for column in columns]


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
    record["cell_size"] = getattr(corrtrack, "cell_size", None)
    record["n_vectors"] = getattr(corrtrack, "n_vectors", None)
    record["grid_dimension"] = getattr(corrtrack, "grid_dimension", None)
    record["freq_threshold"] = getattr(corrtrack, "freq_threshold", None)

    artifact_active = bool(artifact_prefix)
    if artifact_active:
        base_dir = os.path.dirname(os.path.abspath(artifact_prefix))
        os.makedirs(base_dir, exist_ok=True)
        corrtrack._start_artifact_logging(artifact_prefix)

    artifact_mode_value = (artifact_mode or "iterative").lower()
    if artifact_mode_value not in ("iterative", "final"):
        artifact_mode_value = "iterative"
    artifact_per_iteration = artifact_mode_value == "iterative"

    artifact_time_total = 0.0
    artifact_time_overlap = 0.0

    start_time = time.time()
    for start in range(0, data.shape[1] - window_step + 1, window_step):
        chunk = data[:, start : (start + window_step)]
        if run_kind == "bf":
            corrtrack.run_bf(chunk, ids, verbose=False, testing=False, corr_val=True)
        else:
            corrtrack.run(chunk, ids, verbose=False, testing=False, corr_val=corr_val)
        if artifact_active and artifact_per_iteration:
            _t0 = time.time()
            corrtrack._append_artifacts()
            elapsed = time.time() - _t0
            artifact_time_total += elapsed
            artifact_time_overlap += elapsed
    end_time = time.time()

    if artifact_active and not artifact_per_iteration:
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
        runtime -= getattr(corrtrack, "train_dist_time", 0.0)
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

    record["runtime"] = runtime
    record["artifact_time"] = artifact_time_total
    if hasattr(corrtrack, "n_lagged_windows"):
        record["mem_w"] = corrtrack.n_lagged_windows - 1
    record["correlated"] = getattr(corrtrack, "validated_candidates", 0)
    record["tested"] = getattr(corrtrack, "tested_candidates", 0)
    record["total_candidates"] = getattr(corrtrack, "total_candidates", 0)
    pair_min_dist = getattr(corrtrack, "pair_min_dist", None)
    record["pair_min_dist"] = str(pair_min_dist) if pair_min_dist is not None else ""

    if recall_by_window:
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
    artifact_prefix: Optional[str] = None,
) -> tuple[dict, tuple, Optional[np.ndarray]]:
    metadata = dict(metadata or {})
    metadata.setdefault("mode", "bf")
    metadata.setdefault("alg", metadata.get("alg", "bf"))
    metadata.setdefault("optim", metadata.get("optim", "baseline"))

    artifact_mode = base_config.get("artifact_mode", "iterative")

    corrtrack = CorrTrack(
        window_size=base_config["window_size"],
        basic_window=base_config.get("basic_window"),
        window_step=base_config["window_step"],
        n_vectors=1,
        n_lags=base_config["n_lags"],
        grid_dimension=1,
        cell_size=1,
        warmup_data=None,
        seed=None,
        seed_toggle=None,
        corr_threshold=base_config["corr_threshold"],
        neg_corr=base_config.get("neg_corr", False),
        preprocess=False,
        extra_filter=False,
        exec=base_config.get("exec", "thread"),
        max_workers=base_config.get("max_workers", 0),
    )

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
    )
    record["extra_filters"] = _coerce_to_bool(base_config.get("extra_filter", False))
    record["preprocess"] = False
    record["extra_filters"] = False
    record["seed"] = None
    record["seed_toggle"] = None
    record["warmup_size"] = None
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
    artifact_prefix: Optional[str] = None,
) -> tuple[dict, tuple, Optional[np.ndarray]]:
    metadata = dict(metadata or {})
    metadata.setdefault("mode", "main")
    metadata.setdefault("alg", metadata.get("alg"))
    metadata.setdefault("optim", metadata.get("optim", "main"))

    warmup_ratio = run_params.get("warmup_size")
    try:
        warmup_ratio = float(warmup_ratio) if warmup_ratio is not None else None
    except (TypeError, ValueError):
        warmup_ratio = None

    length_data = data.shape[1]
    if warmup_ratio is not None:
        warmup_len = round(warmup_ratio * length_data)
        if warmup_len <= 0:
            warmup_len = 1
        warmup_len = min(length_data, warmup_len)
        warmup_data = data[:, :warmup_len]
    else:
        warmup_data = None

    extra_filter = _coerce_to_bool(run_params.get("extra_filters", base_config.get("extra_filter", False)))

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

    corrtrack = CorrTrack(
        window_size=base_config["window_size"],
        basic_window=base_config.get("basic_window"),
        window_step=base_config["window_step"],
        n_vectors=_to_int(run_params.get("n_vectors")),
        n_lags=base_config["n_lags"],
        grid_dimension=_to_int(run_params.get("grid_dimension")),
        cell_size=_to_float(run_params.get("cell_size")),
        warmup_data=warmup_data,
        seed=_to_int(run_params.get("seed")),
        seed_toggle=_to_int(run_params.get("seed_toggle")),
        freq_threshold=_to_float(run_params.get("freq_threshold")),
        corr_threshold=base_config["corr_threshold"],
        neg_corr=base_config.get("neg_corr", False),
        preprocess=run_params.get("preprocess"),
        extra_filter=extra_filter,
        exec=base_config.get("exec", "thread"),
        max_workers=base_config.get("max_workers", 0),
    )

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
    )

    record["warmup_size"] = warmup_ratio
    record["extra_filters"] = extra_filter
    record["preprocess"] = run_params.get("preprocess")
    record["seed"] = _to_int(run_params.get("seed"))
    record["seed_toggle"] = _to_int(run_params.get("seed_toggle"))
    record["freq_threshold"] = _to_float(run_params.get("freq_threshold"))
    record["n_vectors"] = _to_int(run_params.get("n_vectors"))
    record["grid_dimension"] = _to_int(run_params.get("grid_dimension"))
    record["cell_size"] = _to_float(run_params.get("cell_size"))
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
        "processes": "process",
    }
    resolved = aliases.get(key, key)

    valid = {"sequential", "thread", "process"}
    return resolved if resolved in valid else default


def _fast_corr_and_dist(x, y, return_stats=False):
    """Compute Pearson correlation and Euclidean distance without temporary arrays.

    When ``return_stats`` is True, also returns (n, mean_x, mean_y, var_x, var_y),
    where var_* are the summed squared deviations (n * variance).
    """
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


def _is_near_constant_stats(var_sum, n, std_thresh=1e-3):
    if n <= 0:
        return True
    return var_sum <= (std_thresh ** 2) * n


def _is_structurally_spiked_stats(x, mean, var_sum, n, kurt_thresh=5.0):
    if n < 4 or var_sum <= 0.0:
        return False

    centered = np.asarray(x, dtype=np.float64) - mean
    mu4 = float(np.sum(centered ** 4))
    var = var_sum / n
    if var <= 0.0:
        return False
    kurt = (mu4 / n) / (var * var) - 3.0
    return kurt > kurt_thresh


def _sketch_worker(payload):
    """Execute a sketch node update.

    Thread payload: (corrtrack, node_index, node_new, ids_subset, verbose, testing[, perm])
    Process payload: {
        "index": int,
        "state": dict,
        "node_new": np.ndarray,
        "ids_subset": list,
        "verbose": bool,
        "testing": bool,
        "perm": Optional[list[int]],
        "distribute": bool,
    }
    Returns thread tuple or process dict (branch handled by caller).
    """

    if isinstance(payload, dict):
        perm = payload.get("perm")
        if perm is not None:
            perm = np.asarray(perm, dtype=int)
        sketch_node = Sketches.from_state(payload["state"])
        sketches, partitions = sketch_node.run(
            payload["node_new"],
            payload["ids_subset"],
            verbose=payload.get("verbose", False),
            testing=payload.get("testing", False),
            perm=perm,
            distribute=payload.get("distribute", True),
        )
        return {
            "index": payload["index"],
            "state": sketch_node.dump_state(),
            "sketches": sketches,
            "partitions": partitions,
        }

    if len(payload) == 6:
        corrtrack, node_index, node_new, ids_subset, verbose, testing = payload
        perm = None
    else:
        corrtrack, node_index, node_new, ids_subset, verbose, testing, perm = payload
    if perm is not None:
        perm = np.asarray(perm, dtype=int)
    sketch_node = corrtrack.sketch_nodes[node_index]
    sketches, partitions = sketch_node.run(
        node_new,
        ids_subset,
        verbose=verbose,
        testing=testing,
        perm=perm,
        distribute=False,
    )
    return node_index, sketch_node, sketches, partitions


def _grid_worker(payload):
    """Execute a grid node run.

    Thread payload: (corrtrack, grid_index, n_ids, verbose, testing)
    Process payload: {
        "index": int,
        "state": dict,
        "n_ids": int,
        "verbose": bool,
        "testing": bool,
    }
    Returns thread tuple or process dict (branch handled by caller).
    """
    if isinstance(payload, dict):
        grid_node = Candidates.from_state(payload["state"])
        result = grid_node.run(payload["n_ids"], verbose=payload.get("verbose", False), testing=payload.get("testing", False))
        return {
            "index": payload["index"],
            "state": grid_node.dump_state(),
            "result": result,
        }

    corrtrack, grid_index, n_ids, verbose, testing = payload
    grid_node = corrtrack.grid_nodes[grid_index]
    result = grid_node.run(n_ids, verbose=verbose, testing=testing)
    return grid_index, grid_node, result


def _bf_worker(payload):
    """Execute brute-force candidate generation for a shard.

    Thread payload: (index, bf_node, window_step, ids, verbose, testing, ref_ids)
    Process payload: {
        "index": int,
        "state": dict,
        "window_step": np.ndarray,
        "ids": list,
        "verbose": bool,
        "testing": bool,
        "ref_ids": Optional[list],
    }
    Returns thread tuple or process dict (branch handled by caller).
    """
    if isinstance(payload, dict):
        bf_node = Candidates_BF.from_state(payload["state"])
        result = bf_node.run(
            payload["window_step"],
            payload["ids"],
            verbose=payload.get("verbose", False),
            testing=payload.get("testing", False),
            ref_ids=payload.get("ref_ids"),
        )
        return {
            "index": payload["index"],
            "state": bf_node.dump_state(),
            "result": result,
        }

    index, bf_node, window_step, ids, verbose, testing, ref_ids = payload
    result = bf_node.run(window_step, ids, verbose=verbose, testing=testing, ref_ids=ref_ids)
    return index, bf_node, result


def _compute_nonconst_mask(data, window_size):
    csum = np.cumsum(data, axis=1, dtype=np.float64)
    csum = np.pad(csum, ((0, 0), (1, 0)), mode="constant")
    csum_sq = np.cumsum(data * data, axis=1, dtype=np.float64)
    csum_sq = np.pad(csum_sq, ((0, 0), (1, 0)), mode="constant")
    window_sums = csum[:, window_size:] - csum[:, :-window_size]
    window_sums_sq = csum_sq[:, window_size:] - csum_sq[:, :-window_size]
    var_sum = window_sums_sq - (window_sums * window_sums) / window_size
    threshold = (1e-3 ** 2) * window_size
    return np.maximum(var_sum, 0.0) > threshold


def _enumerate_candidate_rows(
    data,
    window_index,
    ref_indices,
    window_size,
    window_step,
    mask=None,
    shard_start=None,
    shard_end=None,
):
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
        working_mask = _compute_nonconst_mask(data.astype(np.float64, copy=False), window_size)

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


def _bf_worker_process(payload):
    shared = payload.get("shared")
    if shared is None:
        return {"index": payload.get("index"), "result": {}}

    shm = shared_memory.SharedMemory(name=shared["name"])
    mask_shm = None
    try:
        data = np.ndarray(shared["shape"], dtype=np.dtype(shared["dtype"]), buffer=shm.buf)
        window_index = np.asarray(payload["window_index"], dtype=int)
        ids_list = payload["ids_list"]
        ref_indices = payload["ref_indices"]
        window_size = int(payload["window_size"])
        window_step = int(payload["window_step"])

        mask_spec = payload.get("mask")
        if mask_spec and "name" in mask_spec:
            mask_shm = shared_memory.SharedMemory(name=mask_spec["name"])
            mask = np.ndarray(mask_spec["shape"], dtype=np.dtype(mask_spec["dtype"]), buffer=mask_shm.buf)
            mask = mask.astype(bool, copy=False)
        else:
            mask = None

        rows = _enumerate_candidate_rows(
            data,
            window_index,
            ref_indices,
            window_size,
            window_step,
            mask=mask,
            shard_start=payload.get("shard_start"),
            shard_end=payload.get("shard_end"),
        )

        if rows is None:
            result = None
        else:
            shm_result = shared_memory.SharedMemory(create=True, size=rows.nbytes)
            np.ndarray(rows.shape, dtype=rows.dtype, buffer=shm_result.buf)[:] = rows
            result = {
                "name": shm_result.name,
                "shape": rows.shape,
                "dtype": rows.dtype.str,
            }
            shm_result.close()

        return {"index": payload["index"], "result": result}
    finally:
        shm.close()
        if mask_shm is not None:
            mask_shm.close()


def _train_distance_worker(payload):
    """Process-safe worker for training distance statistics."""

    shared = payload.get("shared")
    shm = None
    window_data = payload.get("window_data")

    try:
        if shared is not None:
            shm = shared_memory.SharedMemory(name=shared["name"])
            window_data = np.ndarray(shared["shape"], dtype=np.dtype(shared["dtype"]), buffer=shm.buf)

        pair, sign = payload["item"]
        id1, id2, t1, t2, w = pair
        base_index = payload["base_index"]
        series_ids = payload["series_ids"]

        try:
            ix = series_ids[id1]
            iy = series_ids[id2]
            start1 = int(t1 - base_index)
            start2 = int(t2 - base_index)
            x = window_data[ix, start1:start1 + w].astype(np.float64, copy=False)
            y = window_data[iy, start2:start2 + w].astype(np.float64, copy=False)
        except Exception:
            return {"counts": {"seen": 0, "skipped": 1, "constants": 0}}

        pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
        n, mean_x, mean_y, var_x, var_y = stats

        if _is_near_constant_stats(var_x, n) or _is_near_constant_stats(var_y, n):
            return {"counts": {"seen": 1, "skipped": 0, "constants": 1}}

        if (
            _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh=5.0)
            or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh=5.0)
        ):
            return {"counts": {"seen": 1, "skipped": 1, "constants": 0}}

        preprocess = payload.get("preprocess", False)
        warmup_means = payload.get("warmup_means") or []
        warmup_stds = payload.get("warmup_stds") or []

        def _z_norm(sample, idx):
            arr = np.asarray(sample, dtype=np.float64)
            if preprocess:
                if arr.size >= 2:
                    arr = np.clip(np.diff(arr), -3, 3)
                else:
                    arr = np.zeros_like(arr)
            mean = warmup_means[idx] if idx < len(warmup_means) else None
            std = warmup_stds[idx] if idx < len(warmup_stds) else None
            mean = mean if mean is not None else 0.0
            std = std if std is not None and std > 0 else 1.0
            return (arr - mean) / std

        nx = _z_norm(x, ix)
        ny = _z_norm(y, iy)
        norm_pair_dist = float(np.sqrt(np.sum((nx - ny) ** 2)))

        dist_sk = None
        dist_norm_sk = None
        est = None
        est_diff = None

        sk1 = payload.get("sk1")
        sk2 = payload.get("sk2")
        if sk1 is not None and sk2 is not None:
            try:
                sk1 = np.asarray(sk1, dtype=float)
                sk2 = np.asarray(sk2, dtype=float)
                dist_sk = float(np.linalg.norm(sk1 - sk2))
                n1 = np.linalg.norm(sk1)
                n2 = np.linalg.norm(sk2)
                if n1 > 0 and n2 > 0:
                    dist_norm_sk = float(np.linalg.norm(sk1 / n1 - sk2 / n2))
                    est = float(np.dot(sk1, sk2) / (n1 * n2))
                    est_diff = float(est - pair_corr)
            except Exception:
                pass

        corr_threshold = payload["corr_threshold"]
        pos = pair_corr >= corr_threshold if not np.isnan(pair_corr) else False
        neg = pair_corr <= -corr_threshold if not np.isnan(pair_corr) else False

        result = {
            "dists_pos": [], "dists_neg": [], "dists_nonc": [],
            "dists_norm_pos": [], "dists_norm_neg": [], "dists_norm_nonc": [],
            "dists_sk_pos": [], "dists_sk_neg": [], "dists_sk_nonc": [],
            "dists_norm_sk_pos": [], "dists_norm_sk_neg": [], "dists_norm_sk_nonc": [],
            "est_pos": [], "est_neg": [], "est_nonc": [],
            "est_diffs": [],
            "counts": {"seen": 1, "skipped": 0, "constants": 0},
        }

        if pos:
            result["dists_pos"].append(pair_dist)
            result["dists_norm_pos"].append(norm_pair_dist)
        elif neg:
            result["dists_neg"].append(pair_dist)
            result["dists_norm_neg"].append(norm_pair_dist)
        else:
            result["dists_nonc"].append(pair_dist)
            result["dists_norm_nonc"].append(norm_pair_dist)

        if dist_sk is not None:
            if pos:
                result["dists_sk_pos"].append(dist_sk)
            elif neg:
                result["dists_sk_neg"].append(dist_sk)
            else:
                result["dists_sk_nonc"].append(dist_sk)

        if dist_norm_sk is not None:
            if pos:
                result["dists_norm_sk_pos"].append(dist_norm_sk)
            elif neg:
                result["dists_norm_sk_neg"].append(dist_norm_sk)
            else:
                result["dists_norm_sk_nonc"].append(dist_norm_sk)

        if est is not None:
            if pos:
                result["est_pos"].append(est)
            elif neg:
                result["est_neg"].append(est)
            else:
                result["est_nonc"].append(est)
        if est_diff is not None:
            result["est_diffs"].append(est_diff)

        return result
    finally:
        if shm is not None:
            shm.close()


def _corr_validation_batch_worker(payload):
    """Validate a batch of candidate pairs.

    payload keys:
      - items: list of metadata dicts
      - corr_threshold, neg_corr, corr_val
      - shared: optional shared-memory spec
    Returns list of tuples (pair, is_corr, corr, dist, is_constant, is_spiked)
    """
    items = payload["items"]
    corr_threshold = payload["corr_threshold"]
    neg_corr = payload["neg_corr"]
    corr_val = payload.get("corr_val", True)
    shared = payload.get("shared")

    if not corr_val:
        return [
            (item["pair"], True, 1.0, 0.0, False, False)
            for item in items
        ]

    shm = None
    data = None
    results = []

    try:
        if shared:
            shm = shared_memory.SharedMemory(name=shared["name"])
            data = np.ndarray(shared["shape"], dtype=np.dtype(shared["dtype"]), buffer=shm.buf)

        for item in items:
            pair = item["pair"]

            if data is not None:
                idx1 = item["idx1"]
                idx2 = item["idx2"]
                start1 = item["start1"]
                start2 = item["start2"]
                window = item["window"]
                x = data[idx1, start1:start1 + window]
                y = data[idx2, start2:start2 + window]
            else:
                x = item["x"]
                y = item["y"]

            pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
            n, mean_x, mean_y, var_x, var_y = stats

            if _is_near_constant_stats(var_x, n) or _is_near_constant_stats(var_y, n):
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
        if shm is not None:
            shm.close()
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
    def __init__(self,window_size,basic_window,window_step,n_vectors,n_lags,grid_dimension,cell_size,warmup_data,seed=2468,seed_toggle=1357,freq_threshold=0.7,corr_threshold=0.7,neg_corr=False,preprocess=False,extra_filter=False,exec="parallel",max_workers=0):
        
        if basic_window is not None and window_size % basic_window != 0:
            raise TypeError("Window size (",window_size,") is not divisable by basic window size (",basic_window,")")
        if basic_window is not None and basic_window % window_step != 0:
            raise TypeError("Basic window size (",basic_window,") is not divisable by window step (",window_step,")")
        #if window_step > 0 and n_lags % window_step != 0:
        #    raise TypeError("Number of lags (",n_lags,") is not divisable by window step (",window_step,")")
        if grid_dimension == 0:
            if n_vectors is None or n_vectors <= 0:
                raise ValueError("n_vectors must be a positive integer.")
            grid_dimension = CorrTrack.choose_grid_dimension(n_vectors)
        elif grid_dimension < 0:
            raise ValueError("grid_dimension must be a non-negative integer.")

        if n_vectors is None or n_vectors <= 0:
            raise ValueError("n_vectors must be a positive integer.")
        if grid_dimension == 0 or n_vectors % grid_dimension != 0:
            raise TypeError("Number of random vectors (",n_vectors,") is not divisable by the grid dimension (",grid_dimension,")")
        
        # Parameters features
        self.neg_corr = neg_corr
        # Parameters data
        self.warmup_data = warmup_data
        self.window_data = None
        self.window_index = None
        self.window_startTimes = None
        self.series_ids = {}
        self.map_ids = []
        self.ids = None
        self.warmup_means = None
        self.warmup_stds = None
        self._warmup_mean_map = {}
        self._warmup_std_map = {}
        self.datetime_index = None
        self.datetime_lookup = {} #TODO: save correlation logs to file
        self.preprocess = preprocess
        self.extra_filter = _coerce_to_bool(extra_filter)
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
        self._perm_rng = np.random.default_rng(self.seed)
        self.seed_toggle = seed_toggle
        self.n_vectors = n_vectors
        # Parameters nodes
        self.exec = _normalize_exec_mode(exec, default="thread")
        self.n_nodes = max_workers
        if self.exec not in ("thread", "process"):
            self.n_nodes = 1
        elif max_workers == 0:
            self.n_nodes = max(1, (os.cpu_count() or 1))
        self.n_sketch_nodes = self.n_nodes
        self.sketch_nodes = []
        self.grid_nodes = []
        self.brute_force_nodes = []
        self._shared_window = None
        self._shared_window_shape = None
        self._shared_window_dtype = None
        self._shared_window_size = 0
        # Parameters grids
        self.grid_dimension = grid_dimension
        if self.n_vectors is not None:
            self.n_grids = int(self.n_vectors//self.grid_dimension)
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

        base = np.sqrt((1.0 - corr_threshold) / 2.0) / np.sqrt(self.n_vectors)
        shrink = np.sqrt(np.minimum(1.0, self.grid_dimension / np.sqrt(self.n_vectors)))

        stretch = cell_size if cell_size is not None else 1.0
        try:
            stretch = float(stretch)
        except (TypeError, ValueError):
            stretch = 1.0
        if stretch <= 0.0:
            stretch = 1.0

        self.cell_stretch = stretch
        self.cell_size = base * shrink * stretch
        self.cell_size = min(self.cell_size, 0.5)
        self.grid_max = self.window_size/np.sqrt(self.n_vectors) #3*self.sketch_std
        # Parameters thresholds
        if freq_threshold is not None:
            self.freq_threshold = freq_threshold*self.n_vectors
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
        self.sketches = {}
        self.corr_dist_pos = []
        self.corr_dist_neg = []
        self.noncorr_dist = []
        self.corr_norm_dist_pos = []
        self.corr_norm_dist_neg = []
        self.noncorr_norm_dist = []        
        self.corr_dist_norm_sk_pos = []
        self.corr_dist_norm_sk_neg = []
        self.noncorr_dist_norm_sk = []        
        self.corr_dist_sk_pos = []
        self.corr_dist_sk_neg = []
        self.noncorr_dist_sk = []
        self.corr_est_pos = []
        self.corr_est_neg = []
        self.noncorr_est = []
        self.corr_est_diffs = []
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
        self.train_dist_time = 0  

        self.min_dist = np.inf
        self.pair_min_dist = None

        # Filter tuning defaults
        self.sign_prefilter_scale = 1.3
        self.sign_prefilter_extra = 1
        self.neighbor_margin = 0.1 if self.neg_corr else 0.08

        #getting warmup stats
        self._warmup()

        # Instantiating corrtrack objects
        for g in range(self.n_grids):
            neighbor_margin = self.neighbor_margin if self.neg_corr else 0.06
            self.grid_nodes.append(Candidates(
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
                sign_prefilter_scale=self.sign_prefilter_scale,
                sign_prefilter_extra=self.sign_prefilter_extra,
                neighbor_margin=min(neighbor_margin, 0.1),
                extra_filter=self.extra_filter,
            ))

    def configure_filters(self, sign_scale=None, sign_extra=None, neighbor_margin=None):
        if sign_scale is not None:
            self.sign_prefilter_scale = float(sign_scale)
        if sign_extra is not None:
            self.sign_prefilter_extra = int(sign_extra)
        if neighbor_margin is not None:
            self.neighbor_margin = float(neighbor_margin)

        for node in self.grid_nodes:
            if sign_scale is not None:
                node.sign_prefilter_scale = float(sign_scale)
            if sign_extra is not None:
                node.sign_prefilter_extra = int(sign_extra)
            if neighbor_margin is not None:
                margin = float(neighbor_margin)
                if not self.neg_corr:
                    margin = max(0.06, margin)
                node.neighbor_margin = min(margin, 0.1)
    
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


    @staticmethod
    def _run_batch(func, batch):
        """Apply *func* to each item in *batch* and return the collected results."""
        return [func(item) for item in batch]


    def _parallel_map(self,func,iterable,mode: str = "thread",max_workers: int = None,chunksize: int = 1,preserve_order: bool = True):
        items = list(iterable)
        if not items:
            return []

        if mode not in {"thread", "process", "sequential"}:
            mode = "thread"

        if mode == "sequential":
            return [func(x) for x in items]

        if max_workers is None or max_workers <= 0:
            max_workers = max(1, (os.cpu_count() or 1))
        max_workers = max(1, min(max_workers, len(items)))

        if max_workers <= 1:
            return [func(x) for x in items]

        if chunksize is None:
            chunksize = 0
        try:
            chunk_size = int(chunksize)
        except (TypeError, ValueError):
            chunk_size = 0

        if mode == "process":
            # Avoid complicating pickling by batching in process mode.
            chunk_size = 1
        elif chunk_size <= 0:
            auto = len(items) // (max_workers * 4)
            chunk_size = 1 if auto <= 1 else min(auto, 64)

        chunk_size = max(1, min(chunk_size, len(items)))

        use_dask = _HAS_DASK and delayed is not None

        if not use_dask:
            executor_cls = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
            try:
                with executor_cls(max_workers=max_workers) as executor:
                    if preserve_order:
                        if chunk_size == 1:
                            return list(executor.map(func, items))

                        batches = list(self._chunk_sequence(items, chunk_size))
                        results = []
                        for batch_result in executor.map(self._run_batch, repeat(func), batches):
                            results.extend(batch_result)
                        return results

                    if chunk_size == 1:
                        futures = [executor.submit(func, item) for item in items]
                        results = []
                        for future in as_completed(futures):
                            results.append(future.result())
                        return results

                    batches = list(self._chunk_sequence(items, chunk_size))
                    futures = [executor.submit(self._run_batch, func, batch) for batch in batches]
                    results = []
                    for future in as_completed(futures):
                        results.extend(future.result())
                    return results
            except Exception as exc:
                if getattr(self, "verbose", False):
                    print(f"[parallel_map:{mode}] Executor fallback to sequential due to: {exc}")
                return [func(x) for x in items]

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1))

        scheduler_get = dask_threaded_get if mode == "thread" else dask_multiprocessing_get
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
        
    def _release_shared_window(self):
        if self._shared_window is not None:
            try:
                self._shared_window.close()
                self._shared_window.unlink()
            except FileNotFoundError:
                pass
            except AttributeError:
                pass
            self._shared_window = None
            self._shared_window_shape = None
            self._shared_window_dtype = None
            self._shared_window_size = 0

    def _ensure_shared_window(self):
        if self.exec != "process":
            return None

        data = np.asarray(self.window_data, dtype=float)
        if data.size == 0:
            return None

        shape = data.shape
        dtype = data.dtype
        size = data.nbytes

        if self._shared_window is None or size > self._shared_window_size:
            self._release_shared_window()
            shm = shared_memory.SharedMemory(create=True, size=size)
            self._shared_window = shm
            self._shared_window_size = size
        else:
            shm = self._shared_window

        np.ndarray(shape, dtype=dtype, buffer=shm.buf)[:] = data
        self._shared_window_shape = shape
        self._shared_window_dtype = dtype.str
        return shm

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
    
    def _warmup(self): #can be done in parallel
        if self.warmup_data is not None and self.warmup_data.size > 0:
            self.datetime_index = CorrTrack._is_datetime(self.warmup_data[0,:])
            warmup_data = self.warmup_data[1:,:]
            preprocessed_warmup_data = warmup_data
            if self.preprocess:
                diff_warmup_data = np.diff(warmup_data, axis=1).astype(float)
                preprocessed_warmup_data = np.clip(diff_warmup_data, -3, 3)
            self.warmup_means = [np.mean(serie) for serie in preprocessed_warmup_data]
            self.warmup_stds = [np.std(serie) for serie in preprocessed_warmup_data]

    def _update_curr_data(self,new_data_step,ids):
        
        new_data_step_index = new_data_step[0,:]
        if self.datetime_index is None:
            self.datetime_index = CorrTrack._is_datetime(new_data_step_index)        
        if self.datetime_index:
            new_data_step_index = self._datetime_to_index(new_data_step_index)

        if self.window_data is None:
            self.window_index = new_data_step_index
            self.window_data = new_data_step[1:,:]
        else:
            if self.window_data.shape[1] >= (self.n_lags+self.window_size):
                self.window_index = self.window_index[self.window_step:]
                self.window_data = self.window_data[:,self.window_step:]
                self._cleanup_datetime_lookups()
            self.window_index = np.append(self.window_index,new_data_step_index)
            self.window_data = np.append(self.window_data,new_data_step[1:,:],axis=1)

        self.ids = ids
        if len(self.series_ids) == 0 or len(self.series_ids) != len(ids):
            series_indexes = {}
            for i, val in enumerate(ids):
                series_indexes[val]= i
            self.series_ids = series_indexes # same order of the series in new_data_step

        if self.warmup_means is not None:
            self._warmup_mean_map = {
                id_: self.warmup_means[idx] if idx < len(self.warmup_means) else None
                for id_, idx in self.series_ids.items()
            }
        else:
            self._warmup_mean_map = {}

        if self.warmup_stds is not None:
            self._warmup_std_map = {
                id_: self.warmup_stds[idx] if idx < len(self.warmup_stds) else None
                for id_, idx in self.series_ids.items()
            }
        else:
            self._warmup_std_map = {}

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
        if self.exec == "sequential":
            self.n_nodes = 1
            self.n_sketch_nodes = 1
            self.map_ids = [ids]
            return

        self.n_sketch_nodes = min(self.n_nodes, n_ids) if n_ids > 0 else 0

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
            clipped_t = np.clip(diff_t, -3, 3)                   # 2. Linear clipping
            t = clipped_t

        fallback_std = 1.0
        z_normalized = np.empty_like(t)

        i = series_index
        mean = self.warmup_means[i] if self.warmup_means[i] is not None else 0.0
        std = self.warmup_stds[i] if self.warmup_stds[i] is not None and self.warmup_stds[i] > 0 else fallback_std
        z_normalized = (t - mean) / std

        return z_normalized

    def is_structurally_spiked(x, kurt_thresh=5):
        x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.size
        if n < 4:
            return False
        mean = float(np.mean(x_arr))
        var_sum = float(np.dot(x_arr - mean, x_arr - mean))
        return _is_structurally_spiked_stats(x_arr, mean, max(var_sum, 0.0), n, kurt_thresh=kurt_thresh)

    def is_near_constant(x, std_thresh=1e-3):
        x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.size
        if n == 0:
            return True
        mean = float(np.mean(x_arr))
        var_sum = float(np.dot(x_arr - mean, x_arr - mean))
        return _is_near_constant_stats(max(var_sum, 0.0), n, std_thresh=std_thresh)

    def _train_distance_item(self, item):
        """
        Compute training stats for a single candidate 'item' = (pair_key, sign).

        Returns a small dict of lists you can merge later:
        - *_dist* lists (raw, z-normed, sketch-based)
        - corr_est_* lists (cosine sim of sketches if available)
        - corr_est_diffs (estimated - true)
        - counts (seen, skipped, constants)
        """
        pair, sign = item  # sign is present in candidates dict (1 / -1)
        try:
            id1, id2, t1, t2, w = pair
            ix = self.series_ids[id1]
            iy = self.series_ids[id2]

            # Locate slices relative to current window buffer
            s1 = int(t1 - self.window_index[0])
            s2 = int(t2 - self.window_index[0])
            x = self.window_data[ix, s1:s1 + w].astype(np.float64, copy=False)
            y = self.window_data[iy, s2:s2 + w].astype(np.float64, copy=False)
        except Exception:
            return {"counts": {"seen": 0, "skipped": 1, "constants": 0}}

        pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
        n, mean_x, mean_y, var_x, var_y = stats

        if _is_near_constant_stats(var_x, n) or _is_near_constant_stats(var_y, n):
            return {"counts": {"seen": 1, "skipped": 0, "constants": 1}}

        if (
            _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh=5.0)
            or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh=5.0)
        ):
            return {"counts": {"seen": 1, "skipped": 1, "constants": 0}}

        # Truth (raw + z-normalized)
        nx = self._preprocess_data(x, ix)
        ny = self._preprocess_data(y, iy)
        norm_pair_dist = float(np.sqrt(np.sum((nx - ny) ** 2)))

        # Sketch-based distances (if available)
        dist_sk = None
        dist_norm_sk = None
        est = None
        est_diff = None

        sk1 = self.sketches.get((id1, t1))
        sk2 = self.sketches.get((id2, t2))
        if sk1 is not None and sk2 is not None:
            try:
                sk1 = np.asarray(sk1, dtype=float)
                sk2 = np.asarray(sk2, dtype=float)
                dist_sk = float(np.linalg.norm(sk1 - sk2))
                n1 = np.linalg.norm(sk1)
                n2 = np.linalg.norm(sk2)
                if n1 > 0 and n2 > 0:
                    dist_norm_sk = float(np.linalg.norm(sk1 / n1 - sk2 / n2))
                    est = float(np.dot(sk1, sk2) / (n1 * n2))  # cosine similarity
                    est_diff = float(est - pair_corr)
            except Exception:
                pass

        # Route to the right buckets based on true corr sign thresholds
        pos = pair_corr >= self.corr_threshold if not np.isnan(pair_corr) else False
        neg = pair_corr <= -self.corr_threshold if not np.isnan(pair_corr) else False

        payload = {
            "dists_pos": [], "dists_neg": [], "dists_nonc": [],
            "dists_norm_pos": [], "dists_norm_neg": [], "dists_norm_nonc": [],
            "dists_sk_pos": [], "dists_sk_neg": [], "dists_sk_nonc": [],
            "dists_norm_sk_pos": [], "dists_norm_sk_neg": [], "dists_norm_sk_nonc": [],
            "est_pos": [], "est_neg": [], "est_nonc": [],
            "est_diffs": [],
            "counts": {"seen": 1, "skipped": 0, "constants": 0},
        }

        # Raw / z-norm distances
        if pos:
            payload["dists_pos"].append(pair_dist)
            payload["dists_norm_pos"].append(norm_pair_dist)
        elif neg:
            payload["dists_neg"].append(pair_dist)
            payload["dists_norm_neg"].append(norm_pair_dist)
        else:
            payload["dists_nonc"].append(pair_dist)
            payload["dists_norm_nonc"].append(norm_pair_dist)

        # Sketch-based, if computed
        if dist_sk is not None:
            if pos:   payload["dists_sk_pos"].append(dist_sk)
            elif neg: payload["dists_sk_neg"].append(dist_sk)
            else:     payload["dists_sk_nonc"].append(dist_sk)

        if dist_norm_sk is not None:
            if pos:   payload["dists_norm_sk_pos"].append(dist_norm_sk)
            elif neg: payload["dists_norm_sk_neg"].append(dist_norm_sk)
            else:     payload["dists_norm_sk_nonc"].append(dist_norm_sk)

        if est is not None:
            if pos:   payload["est_pos"].append(est)
            elif neg: payload["est_neg"].append(est)
            else:     payload["est_nonc"].append(est)
        if est_diff is not None:
            payload["est_diffs"].append(est_diff)

        return payload
    
    def _merge_train_distances(self, shard_payloads):
        # Ensure accumulators exist (these match names used elsewhere in your file)
        if not hasattr(self, "corr_dist_pos"): self.corr_dist_pos = []
        if not hasattr(self, "corr_dist_neg"): self.corr_dist_neg = []
        if not hasattr(self, "noncorr_dist"): self.noncorr_dist = []

        if not hasattr(self, "corr_norm_dist_pos"): self.corr_norm_dist_pos = []
        if not hasattr(self, "corr_norm_dist_neg"): self.corr_norm_dist_neg = []
        if not hasattr(self, "noncorr_norm_dist"): self.noncorr_norm_dist = []

        if not hasattr(self, "corr_dist_sk_pos"): self.corr_dist_sk_pos = []
        if not hasattr(self, "corr_dist_sk_neg"): self.corr_dist_sk_neg = []
        if not hasattr(self, "noncorr_dist_sk"): self.noncorr_dist_sk = []

        if not hasattr(self, "corr_dist_norm_sk_pos"): self.corr_dist_norm_sk_pos = []
        if not hasattr(self, "corr_dist_norm_sk_neg"): self.corr_dist_norm_sk_neg = []
        if not hasattr(self, "noncorr_dist_norm_sk"): self.noncorr_dist_norm_sk = []

        if not hasattr(self, "corr_est_pos"): self.corr_est_pos = []
        if not hasattr(self, "corr_est_neg"): self.corr_est_neg = []
        if not hasattr(self, "noncorr_est"): self.noncorr_est = []
        if not hasattr(self, "corr_est_diffs"): self.corr_est_diffs = []

        if not hasattr(self, "constant_candidates"): self.constant_candidates = 0
        if not hasattr(self, "total_candidates"): self.total_candidates = 0

        for p in shard_payloads:
            if not p: 
                continue

            # Raw
            self.corr_dist_pos.extend(p.get("dists_pos", []))
            self.corr_dist_neg.extend(p.get("dists_neg", []))
            self.noncorr_dist.extend(p.get("dists_nonc", []))

            # Z-norm
            self.corr_norm_dist_pos.extend(p.get("dists_norm_pos", []))
            self.corr_norm_dist_neg.extend(p.get("dists_norm_neg", []))
            self.noncorr_norm_dist.extend(p.get("dists_norm_nonc", []))

            # Sketch
            self.corr_dist_sk_pos.extend(p.get("dists_sk_pos", []))
            self.corr_dist_sk_neg.extend(p.get("dists_sk_neg", []))
            self.noncorr_dist_sk.extend(p.get("dists_sk_nonc", []))

            # Sketch (normed)
            self.corr_dist_norm_sk_pos.extend(p.get("dists_norm_sk_pos", []))
            self.corr_dist_norm_sk_neg.extend(p.get("dists_norm_sk_neg", []))
            self.noncorr_dist_norm_sk.extend(p.get("dists_norm_sk_nonc", []))

            # Estimates
            self.corr_est_pos.extend(p.get("est_pos", []))
            self.corr_est_neg.extend(p.get("est_neg", []))
            self.noncorr_est.extend(p.get("est_nonc", []))
            self.corr_est_diffs.extend(p.get("est_diffs", []))

            counts = p.get("counts", {})
            self.constant_candidates += counts.get("constants", 0)
            # Count every evaluated item as total_seen; your outer loop also updates tested/total elsewhere
            self.total_candidates += counts.get("seen", 0)

    def _train_distances(self):
        """
        Parallel training pass over self.candidates.items().
        Defaults to threads (NumPy releases the GIL; also avoids pickling issues).
        """
        if not getattr(self, "candidates", None):
            return

        items = list(self.candidates.items())
        worker_mode = self.exec

        if worker_mode == "process":
            base_index = int(self.window_index[0])
            shared_spec = None
            window_data = None

            shm = self._ensure_shared_window()
            if shm is not None:
                shared_spec = {
                    "name": shm.name,
                    "shape": self._shared_window_shape,
                    "dtype": self._shared_window_dtype,
                }
            else:
                window_data = np.array(self.window_data, dtype=np.float64, copy=False)

            warmup_means = list(self.warmup_means) if self.warmup_means is not None else []
            warmup_stds = list(self.warmup_stds) if self.warmup_stds is not None else []
            series_ids = dict(self.series_ids)

            payloads = []
            for pair, sign in items:
                id1, id2, t1, t2, _ = pair
                payloads.append({
                    "item": (pair, sign),
                    "series_ids": series_ids,
                    "base_index": base_index,
                    "corr_threshold": self.corr_threshold,
                    "preprocess": self.preprocess,
                    "warmup_means": warmup_means,
                    "warmup_stds": warmup_stds,
                    "sk1": self.sketches.get((id1, t1)),
                    "sk2": self.sketches.get((id2, t2)),
                    "shared": shared_spec,
                    "window_data": window_data,
                })

            shard_payloads = self._parallel_map(
                _train_distance_worker,
                payloads,
                mode=worker_mode,
                max_workers=self.n_nodes,
                preserve_order=False,
            )
        else:
            shard_payloads = self._parallel_map(
                self._train_distance_item,
                items,
                mode=worker_mode,
                max_workers=self.n_nodes,
                preserve_order=False,
            )
        self._merge_train_distances(shard_payloads)

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

        if _is_near_constant_stats(var_x, n) or _is_near_constant_stats(var_y, n):
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
    
    def _in_corr(self,pair):
        #monitor continuous and fallen into correlation
        timepts = np.array([pair[2],pair[3]])
        window_size = pair[4]
        corr_lag = max(timepts) - min(timepts)
        corr_sign = (1 if self.correlated[pair] >= 0 else -1)
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
    
    def _get_validated_corr(self, corr_val=True):
        self.validated = {}
        if not self.candidates:
            return

        existing = set(self.correlated.keys())
        pairs = [p for p in self.candidates.keys() if p not in existing]

        if not pairs:
            return

        # Count candidates once, for both sequential and parallel
        self.total_candidates += len(pairs)

        if not corr_val:
            for pair in pairs:
                self.correlated[pair] = 1.0
                self.validated[pair] = 1.0
            self.tested_candidates += len(pairs)
            self.validated_candidates += len(pairs)
            return

        worker_mode = self.exec

        if worker_mode == "sequential":
            tested = validated = 0
            min_dist, min_pair = self.min_dist, self.pair_min_dist
            for pair in pairs:
                pair, is_corr, corr, dist = self._validate_corr(pair, True)
                tested += 1
                if dist < min_dist:
                    min_dist, min_pair = dist, pair
                if is_corr:
                    validated += 1
                    self.correlated[pair] = corr
                    self.validated[pair] = corr
            self.tested_candidates += tested
            self.validated_candidates += validated
            self.min_dist, self.pair_min_dist = min_dist, min_pair
            return

        base_index = self.window_index[0]
        ids_lookup = self.series_ids
        data = self.window_data

        shared_spec = None
        if worker_mode == "process":
            shm = self._ensure_shared_window()
            if shm is None:
                worker_mode = "thread"
            else:
                shared_spec = {
                    "name": shm.name,
                    "shape": self._shared_window_shape,
                    "dtype": self._shared_window_dtype,
                }

        items = []
        for pair in pairs:
            id1, id2, t1, t2, window_size = pair
            start1 = int(t1 - base_index)
            start2 = int(t2 - base_index)

            if shared_spec is None:
                x = data[ids_lookup[id1], start1:start1 + window_size]
                y = data[ids_lookup[id2], start2:start2 + window_size]
                if x.size != window_size or y.size != window_size:
                    continue
                items.append({
                    "pair": pair,
                    "x": x.astype(np.float64, copy=False),
                    "y": y.astype(np.float64, copy=False),
                })
            else:
                idx1 = ids_lookup[id1]
                idx2 = ids_lookup[id2]
                if start1 < 0 or start2 < 0:
                    continue
                if start1 + window_size > data.shape[1] or start2 + window_size > data.shape[1]:
                    continue
                items.append({
                    "pair": pair,
                    "idx1": idx1,
                    "idx2": idx2,
                    "start1": start1,
                    "start2": start2,
                    "window": window_size,
                })

        if not items:
            return

        if worker_mode == "thread":
            chunk_size = 256
        else:  # process
            chunk_size = 2048

        payloads = []
        for i in range(0, len(items), chunk_size):
            payload_items = items[i:i + chunk_size]
            payload = {
                "items": payload_items,
                "corr_threshold": self.corr_threshold,
                "neg_corr": self.neg_corr,
                "corr_val": True,
            }
            if shared_spec is not None:
                payload["shared"] = shared_spec
            payloads.append(payload)

        results_batches = self._parallel_map(
            _corr_validation_batch_worker,
            payloads,
            mode=worker_mode,
            max_workers=self.n_nodes,
            preserve_order=False,
        )

        tested = validated = 0
        min_dist, min_pair = self.min_dist, self.pair_min_dist
        for batch in results_batches:
            for pair, is_correlated, corr, dist, is_constant, _ in batch:
                tested += 1
                if is_constant:
                    self.constant_candidates += 1
                if dist < min_dist:
                    min_dist, min_pair = dist, pair
                if is_correlated:
                    validated += 1
                    self.correlated[pair] = corr
                    self.validated[pair] = corr

        self.tested_candidates += tested
        self.validated_candidates += validated
        self.min_dist, self.pair_min_dist = min_dist, min_pair

    def _monitor_corr(self):
        """
        Hybrid monitoring:
        • 'In' updates: sequential (preserves per-key chain logic in _in_corr).
        • 'Out' updates: parallel over previous_correlations entries.
        """
        new_in = {}
        if len(self.validated) > 0:
            # sequential 'in' to preserve state coupling
            for pair, corr in self.validated.items():
                new_pair = self._in_corr(pair)
                if new_pair is not None:
                    new_in.update(new_pair)
                    key = list(new_pair.keys())[0]
                    if key in self.previous_correlations:
                        del self.previous_correlations[key]

        # parallel 'out' (independent across keys)
        if len(self.previous_correlations) > 0:
            items = list(self.previous_correlations.items())
            worker_mode = self.exec if self.exec != "process" else "thread"
            # compute anomaly records in parallel
            out_records = self._parallel_map(
                self._out_corr,
                items,
                mode=worker_mode,
                max_workers=self.n_nodes,
                preserve_order=False,
            )
            # apply on main thread
            for pair_key, rec in out_records:
                if pair_key not in self.corr_anomalies:
                    self.corr_anomalies[pair_key] = []
                self.corr_anomalies[pair_key].append(rec)

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
            self._release_shared_window()
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
            writer = csv.writer(file)
            writer.writerow(["id1", "id2", "time1", "time2", "corr"])
            for pair,corr in sorted_items:
                id1, id2, t1, t2, _ = pair
                if self.datetime_index:
                    t1, t2 = self._safe_index_to_datetime(t1), self._safe_index_to_datetime(t2)
                writer.writerow([id1,id2,t1,t2,corr])

    def _save_max_lag_correlated(self,output_csv):
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
            writer = csv.writer(file)
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
                writer = csv.writer(file)
                writer.writerow(["id1", "id2", "lag", "start_time_id1", "start_time_id2", "duration", "corr_sign"])
                for key,value in self.corr_lengths.items():
                    ids = (key[0],key[1])
                    lag = key[2]
                    for t1,t2,w,corr_len,corr_sign in value:
                        if self.datetime_index:
                            time1 = self._safe_index_to_datetime(t1)
                            time2 = self._safe_index_to_datetime(t2)
                        writer.writerow([ids[0],ids[1],lag,time1,time2,corr_len,corr_sign])
                        

    def _print_anomalies(self):
        if len(self.corr_lengths)>0:
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
            writer = csv.writer(file)
            writer.writerow(['id1', 'id2', 'time1', 'time2', 'window', 'freq'])
            for pair, freq in sorted(self.freq_pairs.items()):
                id1, id2, t1, t2, w = pair
                time1 = self._safe_index_to_datetime(t1)
                time2 = self._safe_index_to_datetime(t2)
                writer.writerow([id1, id2, time1, time2, w, freq])

    def _save_negative_pairs(self, output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id1', 'id2', 'time1', 'time2', 'corr'])
            for (id1, id2, t1, t2, _), corr in self.correlated.items():
                if corr <= -self.corr_threshold:
                    time1 = self._safe_index_to_datetime(t1)
                    time2 = self._safe_index_to_datetime(t2)
                    writer.writerow([id1, id2, time1, time2, corr])

    def _save_anomalies(self,output_csv):
        if len(self.corr_lengths)>0:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id1", "id2", "lag", "time", "anomaly"])
                for key,value in self.corr_anomalies.items():
                    ids = (key[0],key[1])
                    lag = key[2]
                    for time,type in value:
                        if self.datetime_index:
                            time = self._safe_index_to_datetime(time)
                        if type == 1:
                            writer.writerow([ids[0],ids[1],lag,time,"into"])
                        elif type == -1:
                            writer.writerow([ids[0],ids[1],lag,time,"out_of"])
                        elif type == 0:
                            writer.writerow([ids[0],ids[1],lag,time,"changed_sign"])

    def _artifact_write_rows(self, path, header, rows):
        if not rows:
            return
        state = getattr(self, "_artifact_state", None)
        if not state:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = path not in state["headers"] or not os.path.exists(path)
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(header)
                state["headers"].add(path)
            writer.writerows(rows)

    def _load_existing_artifacts(self, prefix):
        state = self._artifact_state
        if not prefix:
            return

        def _load_correlated(path):
            with open(path, newline="") as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    key = tuple(row[:4])
                    state["correlated"].add(key)

        def _load_neg_pairs(path):
            with open(path, newline="") as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    key = tuple(row[:4])
                    state["neg_pairs"].add(key)

        def _load_candidates(path):
            with open(path, newline="") as file:
                reader = csv.reader(file)
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
            with open(path, newline="") as file:
                reader = csv.reader(file)
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
            with open(path, newline="") as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    if len(row) < 5:
                        continue
                    state["anomalies"].add(tuple(row[:5]))

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

        state = getattr(self, "_artifact_state", None)
        if state and state.get("prefix") == prefix:
            return

        self._artifact_state = {
            "prefix": prefix,
            "headers": set(),
            "correlated": set(),
            "neg_pairs": set(),
            "candidates": {},
            "status": {},
            "anomalies": set(),
        }
        self._load_existing_artifacts(prefix)

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
            key = (str(id1), str(id2), str(t1_display), str(t2_display))
            if key in state["correlated"]:
                continue
            rows.append([id1, id2, t1_display, t2_display, corr])
            state["correlated"].add(key)
        if rows:
            self._artifact_write_rows(
                f"{prefix}_correlated.csv",
                ["id1", "id2", "time1", "time2", "corr"],
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
                tracker_key = (str(ids[0]), str(ids[1]), str(lag), str(time_val), label)
                if tracker_key in state["anomalies"]:
                    continue
                state["anomalies"].add(tracker_key)
                rows.append([ids[0], ids[1], lag, time_val, label])
        if rows:
            self._artifact_write_rows(
                f"{prefix}_anomalies.csv",
                ["id1", "id2", "lag", "time", "anomaly"],
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

    def _get_sketches(self, new_data_step, verbose, testing):
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
                    self._warmup_mean_map,
                    self._warmup_std_map,
                    self.preprocess,
                )
            )
        if len(self.sketch_nodes) > self.n_sketch_nodes:
            self.sketch_nodes = self.sketch_nodes[:self.n_sketch_nodes]

        node_inputs_thread = []
        node_inputs_process = []
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

        total_dim = self.n_grids * self.grid_dimension
        if total_dim <= 0:
            total_dim = 1
        base_perm = self._perm_rng.permutation(total_dim)
        perm_list = base_perm.tolist()

        node_inputs_thread = [inp + (perm_list,) for inp in node_inputs_thread]
        worker_mode = self.exec

        if worker_mode == "process":
            node_inputs_process = []
            for inp in node_inputs_thread:
                s = inp[1]
                node_inputs_process.append({
                    "index": s,
                    "state": self.sketch_nodes[s].dump_state(),
                    "node_new": inp[2],
                    "ids_subset": inp[3],
                    "verbose": inp[4],
                    "testing": inp[5],
                    "perm": list(perm_list),
                    "distribute": False,
                })
            dispatch_items = node_inputs_process
        else:
            dispatch_items = node_inputs_thread

        if worker_mode == "sequential":
            results = [_sketch_worker(payload) for payload in dispatch_items]
        else:
            results = self._parallel_map(
                _sketch_worker,
                dispatch_items,
                mode=worker_mode,
                max_workers=len(dispatch_items),
                preserve_order=False,
            )

        # Merge sketches from all nodes
        merged_sketches = {}
        per_grid_partition = [dict() for _ in range(self.n_grids)]
        if worker_mode == "process":
            iter_results = results
        else:
            iter_results = results

        for entry in iter_results:
            if worker_mode == "process":
                s = entry["index"]
                self.sketch_nodes[s].load_state(entry["state"])
                sks = entry["sketches"]
                parts = entry["partitions"]
            else:
                s, node_obj, sks, parts = entry
                self.sketch_nodes[s] = node_obj
            merged_sketches.update(sks)
            for g, partition in enumerate(parts):
                if g >= self.n_grids:
                    break
                per_grid_partition[g].update(partition)

        curr_time = self._curr_startTime()
        for g in range(self.n_grids):
            self.grid_nodes[g].append_partition(curr_time, per_grid_partition[g])

        return merged_sketches
    
    def _run_grids(self, verbose, testing):
        """
        Run each grid node in parallel via run() and merge the local outputs
        into self.freq_pairs and self.candidates on the main thread.
        """
        n_ids = len(self.series_ids)

        worker_mode = self.exec

        thread_payloads = [
            (self, g, n_ids, verbose, testing)
            for g in range(self.n_grids)
        ]

        if worker_mode == "process":
            payloads = [
                {
                    "index": g,
                    "state": self.grid_nodes[g].dump_state(),
                    "n_ids": n_ids,
                    "verbose": verbose,
                    "testing": testing,
                }
                for g in range(self.n_grids)
            ]
        else:
            payloads = thread_payloads

        if worker_mode == "sequential":
            grid_results = [_grid_worker(payload) for payload in payloads]
        else:
            grid_results = self._parallel_map(
                _grid_worker,
                payloads,
                mode=worker_mode,
                max_workers=self.n_nodes,
                preserve_order=False,
            )

        # Merge
        self.freq_pairs = {}
        self.uncorrelated = {}

        if worker_mode == "process":
            iterable = (
                (
                    entry["index"],
                    entry["state"],
                    entry["result"],
                )
                for entry in grid_results
            )
        else:
            iterable = (
                (
                    g_index,
                    node_obj,
                    result,
                )
                for g_index, node_obj, result in grid_results
            )

        for g_index, state_or_obj, (loc_freq, _loc_cand, loc_unc) in iterable:
            if worker_mode == "process":
                self.grid_nodes[g_index].load_state(state_or_obj)
            else:
                self.grid_nodes[g_index] = state_or_obj
            for k in sorted(loc_freq):
                v = float(loc_freq[k])
                self.freq_pairs[k] = self.freq_pairs.get(k, 0.0) + v
            for k in sorted(loc_unc):
                self.uncorrelated[k] = loc_unc[k]

        # Recompute candidates after frequencies from all grids are merged so
        # that threshold checks see the combined vote count (matches the
        # original sequential implementation behaviour).
        if self.freq_threshold <= 0:
            self.candidates = {pair: 1 for pair in self.freq_pairs}
        else:
            self.candidates = {
                pair: 1
                for pair, count in self.freq_pairs.items()
                if count >= self.freq_threshold
            }
        
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
        predicted_bool = predicted.astype(bool)
        ground_truth_bool = ground_truth.astype(bool)
        tp = np.sum(predicted_bool & ground_truth_bool)
        fp = np.sum(predicted_bool & ~ground_truth_bool)
        fn = np.sum(~predicted_bool & ground_truth_bool)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
            'f1_score': f1,
            'aucroc': aucroc,
            'pr_auc': pr_auc
        }
    
    def compute_metrics_bf(predicted: dict, ground_truth: dict, windows=True, pair_min_dist=None):
        if windows:
            return CorrTrack._compute_metrics_bf_windows(predicted, ground_truth, pair_min_dist)
        return CorrTrack.compute_metrics_bf_timestamps(predicted, ground_truth)
    
    def _compute_metrics_bf_windows(predicted: dict, ground_truth: dict, pair_min_dist=None):
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
        
        # Union of normalized keys
        all_keys = pred | gt
        sorted_keys = sorted(all_keys)

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

        # Binary conversion
        predicted_bool = predicted_array.astype(bool)
        ground_truth_bool = ground_truth_array.astype(bool)

        # Metric computations
        #tp = np.sum(predicted_bool & ground_truth_bool)
        #fp = np.sum(predicted_bool & ~ground_truth_bool)
        #fn = np.sum(~predicted_bool & ground_truth_bool)

        #precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        #recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        recall_min = int(pair_min_dist in pred)

        def _safe_prec_recall(pred_set, gt_set):
            if len(pred_set) == 0:
                return 0.0, (1.0 if len(gt_set) == 0 else 0.0)  # precision 0, recall 0 unless both empty
            if len(gt_set) == 0:
                return 0.0, 0.0
            precision = (len(pred_set) - len(pred_set - gt_set)) / len(pred_set)
            recall    = (len(gt_set) - len(gt_set - pred_set)) / len(gt_set)
            return precision, recall

        # overall
        precision, recall = _safe_prec_recall(pred, gt)
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

        # pos
        precision_pos, recall_pos = _safe_prec_recall(pred_pos, gt_pos)
        f1_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos) if (precision_pos+recall_pos)>0 else 0.0

        # neg
        precision_neg, recall_neg = _safe_prec_recall(pred_neg, gt_neg)
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

        # Binary conversion
        predicted_bool = predicted_array.astype(bool)
        ground_truth_bool = ground_truth_array.astype(bool)

        # Metric computations
        tp = np.sum(predicted_bool & ground_truth_bool)
        fp = np.sum(predicted_bool & ~ground_truth_bool)
        fn = np.sum(~predicted_bool & ground_truth_bool)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
            'f1_score': f1,
            'aucroc': aucroc,
            'pr_auc': pr_auc
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
    
    def run_train_distances(self,new_data_step,ids,verbose,testing):
        """
        Train distribution collector:
        1) Update buffers
        2) Parallel sketches (like `run`)
        3) Parallel BF enumeration for candidate pairs (like `run_bf`)
        4) Collect sketch-distance training stats
        Notes:
        - No validation/monitoring here by design.
        - Defaults to thread mode for portability; process mode is available
            if you use the top-level worker (see _bf_worker_train_distances).
        """
        self.verbose = verbose
        self.testing = testing

        use_parallel = self.exec in ("thread", "process")

        self._update_curr_data(new_data_step,ids)

        start_time = time.time()
        sketches = self._get_sketches(self._curr_window_step(), verbose, testing)
        end_time = time.time()
        self.sketch_time += end_time - start_time
        #self._update_hist_sketches()
        self._update_curr_sketches(sketches)

        # candidates
        self.candidates = {}
        start_time = time.time()
        if not self.brute_force_nodes:
            self.brute_force_nodes.append(Candidates_BF(self.window_size,self.window_step,self.n_lags,self.corr_threshold))
        if use_parallel:
            self.candidates = self._run_bf_parallel(self._curr_window_step(), self.ids)
        else:
            # original single-thread path
            self.candidates = self.brute_force_nodes[0].run(self._curr_window_step(), self.ids, verbose, testing, ref_indices=None)
        end_time = time.time()
        self.candidate_time += end_time - start_time

        start_time = time.time()
        self._train_distances()
        end_time = time.time()
        self.train_dist_time += end_time - start_time
    
    def _run_bf_parallel(self, curr_window_step, ids, verbose=False, testing=False):
        """
        Parallel wrapper for brute-force enumeration:
        - Shards the series IDs across workers.
        - Each worker calls Candidates_BF.run(...) on its shard.
        - Results are merged on the main thread.
        """
        while len(self.brute_force_nodes) < self.n_sketch_nodes:
            self.brute_force_nodes.append(Candidates_BF(self.window_size,self.window_step,self.n_lags,self.corr_threshold))
        if len(self.brute_force_nodes) > self.n_sketch_nodes:
            self.brute_force_nodes = self.brute_force_nodes[:self.n_sketch_nodes]

        node_inputs = []
        max_nodes = min(self.n_sketch_nodes, len(self.map_ids))
        for s in range(max_nodes):
            ref_ids = [id_ for id_ in self.map_ids[s] if id_ in self.series_ids]
            if not ref_ids:
                continue
            node_inputs.append((s, curr_window_step, ref_ids))

        if not node_inputs:
            return {}

        worker_mode = self.exec

        thread_payloads = [
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

        mask_shm = None
        if worker_mode == "process":
            shm = self._ensure_shared_window()
            if shm is None:
                worker_mode = "thread"
                payloads = thread_payloads
            else:
                shared_spec = {
                    "name": shm.name,
                    "shape": self._shared_window_shape,
                    "dtype": self._shared_window_dtype,
                }
                window_data = np.ndarray(
                    self._shared_window_shape,
                    dtype=np.dtype(self._shared_window_dtype),
                    buffer=shm.buf,
                )
                window_size = self.window_size
                n_series, n_cols = window_data.shape
                window_count = n_cols - window_size + 1
                mask_spec = None
                if window_count > 0:
                    csum = np.cumsum(window_data, axis=1, dtype=np.float64)
                    csum = np.pad(csum, ((0, 0), (1, 0)), mode="constant")
                    csum_sq = np.cumsum(window_data * window_data, axis=1, dtype=np.float64)
                    csum_sq = np.pad(csum_sq, ((0, 0), (1, 0)), mode="constant")
                    window_sums = csum[:, window_size:] - csum[:, :-window_size]
                    window_sums_sq = csum_sq[:, window_size:] - csum_sq[:, :-window_size]
                    var_sum = window_sums_sq - (window_sums * window_sums) / window_size
                    var_sum = np.maximum(var_sum, 0.0)
                    threshold = (1e-3 ** 2) * window_size
                    mask = var_sum > threshold
                    mask = mask.astype(np.bool_, copy=False)
                    mask_shm = shared_memory.SharedMemory(create=True, size=mask.nbytes)
                    np.ndarray(mask.shape, dtype=mask.dtype, buffer=mask_shm.buf)[:] = mask
                    mask_spec = {
                        "name": mask_shm.name,
                        "shape": mask.shape,
                        "dtype": mask.dtype.str,
                    }
                ids_order = list(self.ids)
                window_index_list = self.window_index.tolist()
                payloads = []
                shard_count = max(1, len(node_inputs))
                block = max(1, (n_series + shard_count - 1) // shard_count)
                for s, curr_step, ref_ids in node_inputs:
                    self.brute_force_nodes[s].prepare(curr_step, self.ids, ref_ids)
                    ref_indices = sorted(self.series_ids[id_] for id_ in ref_ids if id_ in self.series_ids)
                    start_idx = min(s * block, n_series)
                    end_idx = min(start_idx + block, n_series)
                    payloads.append({
                        "index": s,
                        "shared": shared_spec,
                        "window_index": window_index_list,
                        "ids_list": ids_order,
                        "ref_indices": ref_indices,
                        "window_size": self.window_size,
                        "window_step": self.window_step,
                        "mask": mask_spec,
                        "shard_start": start_idx,
                        "shard_end": end_idx,
                    })
        else:
            payloads = thread_payloads

        # Dispatch
        worker_func = _bf_worker_process if worker_mode == "process" else _bf_worker

        bf_results = self._parallel_map(
            worker_func,
            payloads,
            mode=worker_mode,
            max_workers=len(payloads) if payloads else None,
            preserve_order=False,
        )

        # Merge
        merged = {}
        if worker_mode == "process":
            ids_order = list(self.ids)
            for entry in bf_results:
                spec = entry.get("result")
                if not spec or "name" not in spec:
                    continue
                shm = shared_memory.SharedMemory(name=spec["name"])
                try:
                    arr = np.ndarray(spec["shape"], dtype=np.dtype(spec["dtype"]), buffer=shm.buf)
                    for row in arr:
                        idx1, idx2, t1, t2, w = (int(v) for v in row)
                        id1 = ids_order[idx1]
                        id2 = ids_order[idx2]
                        pair = _normalize_bf_key((id1, id2, t1, t2, w))
                        merged[pair] = 1
                finally:
                    shm.close()
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
            for node in self.brute_force_nodes:
                node.ref_indices = None
            if mask_shm is not None:
                try:
                    mask_shm.close()
                    mask_shm.unlink()
                except FileNotFoundError:
                    pass
        else:
            for s, bf_node, result in bf_results:
                if bf_node is not None:
                    self.brute_force_nodes[s] = bf_node
                for k in sorted(result):
                    merged[k] = result[k]
        return merged
    
    def run_bf(self, new_data_step, ids, verbose, testing, corr_val=True):
        self.verbose = verbose
        self.testing = testing

        use_parallel = self.exec in ("thread", "process")

        self._update_curr_data(new_data_step, ids)

        # candidates
        self.candidates = {}
        start_time = time.time()
        
        if use_parallel:
            self.candidates = self._run_bf_parallel(self._curr_window_step(), self.ids, verbose=verbose, testing=testing)
        else:
            if not self.brute_force_nodes:
                self.brute_force_nodes.append(Candidates_BF(self.window_size,self.window_step,self.n_lags,self.corr_threshold))
            # original single-thread path
            self.candidates = self.brute_force_nodes[0].run(self._curr_window_step(), self.ids, verbose, testing, ref_ids=None)
        end_time = time.time()
        self.candidate_time += end_time - start_time

        # validation
        start_time = time.time()
        self._get_validated_corr(corr_val)
        end_time = time.time()
        self.validation_time += end_time - start_time

        # monitoring
        start_time = time.time()
        self._monitor_corr()
        end_time = time.time()
        self.monitor_time += end_time - start_time

        if verbose:
            self._print_state()
    
    def _update_hist_sketches(self):
        for s in range(self.n_nodes):
            for k, v in self.sketch_nodes[s].sketches.items():
                self.hist_sketches.append(v)   

    def _update_curr_sketches(self,new_sketches):
        if len(self.sketches) > self.n_lagged_windows:
            del self.sketches[min(self.sketches.keys())]
        self.sketches[self._curr_startTime()] = {}
        self.sketches[self._curr_startTime()].update(new_sketches)
    
    def run(self, new_data_step, ids, verbose, testing, corr_val=True):
        self.verbose = verbose
        self.testing = testing

        self._update_curr_data(new_data_step, ids)

        # 1) sketches
        start_time = time.time()
        sketches = self._get_sketches(self._curr_window_step(), verbose, testing)
        end_time = time.time()
        self.sketch_time += end_time - start_time
        self._update_curr_sketches(sketches)

        # 2) candidates via grids
        start_time = time.time()
        self._run_grids(verbose, testing)
        end_time = time.time()
        self.candidate_time += end_time - start_time

        # 3) validation (parallel)
        start_time = time.time()
        self._get_validated_corr(corr_val)
        end_time = time.time()
        self.validation_time += end_time - start_time

        # 4) monitoring (parallel 'out', sequential 'in')
        start_time = time.time()
        self._monitor_corr()
        end_time = time.time()
        self.monitor_time += end_time - start_time

        if verbose:
            self._print_state()


    
class Sketches:
    def __init__(self,window_size,basic_window,window_step,seed,seed_toggle,n_vectors,grid_dimension,grid_nodes,warmup_mean_map,warmup_std_map,preprocess):       
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
        self.warmup_mean_map = warmup_mean_map or {}
        self.warmup_std_map = warmup_std_map or {}
        self._series_warmup_means = []
        self._series_warmup_stds = []
        self.preprocess = preprocess     
        # Parameters sketches
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
        # Parameters grids
        self.grid_dimensions = grid_dimension #2D grid
        self.n_grids = int(n_vectors/self.grid_dimensions) #divides n_vectors (sketch size) for 2D grid
        self.partitions = []
        self.grid_nodes = grid_nodes
        self.is_constant = {}
        self._toggle_weights = None

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
        new_data_step_values = new_data_step[1:,:]
        #Update sliding windows
        if self.window_data is None:
            self.window_data_original = new_data_step_values
            self.window_data = self._preprocess_data(new_data_step_values)
            n_diff = self.window_data.shape[1] - new_data_step_values.shape[1]
            if n_diff < 0:                
                self.window_data = np.append(self.window_data[:,n_diff][:, None],self.window_data,axis=1)
            self.window_index = new_data_step_index[-self.window_data.shape[1]:]
            self.last_origin = new_data_step_values[:,-1]
        else:
            if self.window_data.shape[1] >= self.window_size:
                if self.window_data.shape[1] >= self.window_size+self.basic_window:
                    self.window_index = self.window_index[self.window_step:]
                self.window_data = self.window_data[:,self.window_step:]
                self.window_data_original = self.window_data_original[:,self.window_step:]
            self.window_index = np.append(self.window_index,new_data_step_index)
            self.window_data_original = np.append(self.window_data_original,new_data_step_values,axis=1)
            processed_new_data_step_values = self._preprocess_data(new_data_step_values)
            self.window_data = np.append(self.window_data,processed_new_data_step_values,axis=1)
            self.last_origin = new_data_step_values[:,-1]
        
        if isinstance(ids, (list, tuple)):
            series_list = list(ids)
        else:
            series_list = list(ids)
        self.series_ids = series_list
        n_series = new_data_step_values.shape[0]
        self._series_warmup_means = []
        self._series_warmup_stds = []
        for i in range(n_series):
            series_id = series_list[i] if i < len(series_list) else None
            self._series_warmup_means.append(
                self.warmup_mean_map.get(series_id, None) if series_id is not None else None
            )
            self._series_warmup_stds.append(
                self.warmup_std_map.get(series_id, None) if series_id is not None else None
            )
        self.window_data = self.window_data.astype(float)
        if self.window_data.shape[1] >= self.window_size:
            for s in range(len(self.series_ids)):
                self.is_constant[self.series_ids[s]] = CorrTrack.is_near_constant(self.window_data_original[s,:])
        
        self._update_curr_window_size()
            
    def _preprocess_data(self, data):
        
        t = np.asarray(data)

        if self.preprocess:
            if self.last_origin is not None:
                data = np.append(np.transpose([self.last_origin]), data, axis=1)
            elif data.shape[1] > 0:
                # Mirror first column so differencing keeps at least one sample when warmup is missing
                data = np.append(data[:, :1], data, axis=1)
            t = np.asarray(data)
            if t.shape[1] <= 1:
                diff_t = np.zeros((t.shape[0], 1)) if t.shape[1] else np.zeros((t.shape[0], 0))
            else:
                diff_t = t[:, 1:] - t[:, :-1]                     # 1. Differencing
            clipped_t = np.clip(diff_t, -3, 3)                   # 2. Linear clipping
            t = clipped_t

        fallback_std = 1.0

        z_normalized = np.empty_like(t)

        for i in range(len(t)):
            mean = self._series_warmup_means[i] if i < len(self._series_warmup_means) else None
            std = self._series_warmup_stds[i] if i < len(self._series_warmup_stds) else None
            mean = mean if mean is not None else 0.0
            std = std if std is not None and std > 0 else fallback_std
            z_normalized[i] = (t[i] - mean) / std

        return z_normalized
    
    def _update_previous_startTime(self):
        if self.window_data.shape[1] >= self.window_size:
            self.previous_startTime = self._curr_startTime()
            self.previous_incrementable_index = self._curr_incrementable_startTime()
        elif self.previous_startTime is None:
            self.previous_startTime = self.window_index[0]
            self.previous_startTime_index = 0
    
    def _update_curr_window_size(self):
        self.curr_window_size = min(self.window_size,self.window_data.shape[1])
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
    
    def _curr_window_times(self):
        return self.window_index[-self.curr_window_size:]
    
    def _curr_startTime(self):
        return self.window_index[-self.curr_window_size]
    
    def _curr_incrementable_startTime(self): # start time of the 2nd basic window
        return self.window_index[-self.curr_window_size+self.basic_window]

    def _refresh_toggle_weights(self):
        if self.basicRandomVector is None or self.toggleVector is None:
            self._toggle_weights = None
            return
        base = np.array(self.basicRandomVector, dtype=np.float64, copy=False)
        toggle = np.array(self.toggleVector, dtype=np.float64, copy=False)
        self._toggle_weights = toggle[:, :, None] * base[None, :, :]
    
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
            self.basicDots.append([])
            self.sketches = {}
            return

        window_blocks = current_window.reshape(n_series, self.n_basic_windows, self.basic_window)
        weights = np.array(self._toggle_weights, dtype=np.float64, copy=False)

        series_dots = np.einsum('sbw,bvw->sbv', window_blocks, weights, optimize=True)
        sketch_vectors = series_dots.sum(axis=1)
        curr_start = self._curr_startTime()
        window_size = self.window_size

        self.basicDots.append([series_dots[s].copy() for s in range(n_series)])
        self.sketches = {
            (self.series_ids[s], curr_start, window_size): sketch_vectors[s].copy()
            for s in range(n_series)
        }
    
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
                self.basicDots.append([])
                self.sketches = {}
                for s in range(len(self.basicDots[0])):
                    series_dots = []
                    series_dots = self.basicDots[0][s][1:(self.n_basic_windows+1),:]
                    series_dots = np.multiply(series_dots,self.intermediary_diff_toggleVector)
                    self.basicDots[-1].append(series_dots)
                    self.sketches[(self.series_ids[s],self._curr_startTime(),self.window_size)] = np.sum(series_dots, axis=0)
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

        previous_n_basic_windows = len(self.basicDots[0][0])
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
            new_dots = np.einsum('sbw,bvw->sbv', new_blocks, weights_subset, optimize=True)
        else:
            new_dots = np.zeros((n_series, 0, self.n_vectors), dtype=np.float64)

        if(self.verbose):
            self._print_curr_window()
        
        self._clean_obsolete_basicDots()
        self._print_incremental_step()
        self.incrementable_index.append(self._curr_incrementable_startTime())
        self.basicDots.append([])
        self.sketches = {}
        same_start = self.previous_startTime == self._curr_startTime()
        curr_start = self._curr_startTime()
        window_size = self.window_size

        for s in range(n_series):
            if same_start and len(self.basicDots) >= 2:
                base = np.array(self.basicDots[-2][s], dtype=np.float64, copy=True)
            else:
                base_source = np.array(self.basicDots[0][s][1:,:], dtype=np.float64, copy=False)
                base = base_source * self.diff_toggleVector

            updated = np.concatenate((base, new_dots[s]), axis=0)
            self.basicDots[-1].append(updated)
            self.sketches[(self.series_ids[s], curr_start, window_size)] = updated.sum(axis=0)

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

    def safe_normalize(self,v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return np.zeros_like(v)
    
    def partition_sketches(self, window_size, perm=None):
        if len(self.sketches) == 0:
            return

        self.partitions = [dict() for _ in range(self.n_grids)]
        total_dim = self.n_grids * self.grid_dimensions
        if total_dim <= 0:
            return

        if perm is None:
            base_perm = self._perm_rng.permutation(total_dim)
        else:
            base_perm = np.asarray(perm, dtype=int)
            if base_perm.size < total_dim:
                repeats = int(np.ceil(total_dim / base_perm.size))
                base_perm = np.tile(base_perm, repeats)[:total_dim]

        for k, v in self.sketches.items():
            norm = np.linalg.norm(v) if not self.is_constant[k[0]] else 0
            if k[2] != window_size:
                continue

            length = v.shape[0]
            if length == 0:
                continue

            idx = base_perm % length

            for grid in range(self.n_grids):
                start = grid * self.grid_dimensions
                end = start + self.grid_dimensions
                if end > idx.shape[0]:
                    break
                indices = idx[start:end]
                self.partitions[grid][k] = (v[indices], self.is_constant[k[0]], norm)

    def distribute_partitions(self):
        if len(self.partitions)>0:
            for grid in range(self.n_grids):
                self.grid_nodes[grid].append_partition(self._curr_startTime(),self.partitions[grid])
            return True
        else:
            return False
    
    def run(self, new_data_step, ids, verbose=True, testing=False, perm=None, distribute=True):
        self.verbose = verbose
        self.testing = testing

        self._newStream(new_data_step, ids)

        sketches = self._get_sketches()
        self.partition_sketches(self.window_size, perm=perm)
        if distribute:
            self.distribute_partitions()

        if(testing and sketches):
            self._print_state()

        # Return copies so the caller can safely merge
        return dict(self.sketches), [dict(p) for p in self.partitions]

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
                 sign_prefilter_scale=1.3,sign_prefilter_extra=1,neighbor_margin=0.08,extra_filter=False):       
        self.verbose = None
        self.neg_corr = neg_corr
        # Parameters grids
        self.n_lagged_windows = n_lagged_windows
        self.curr_time = None
        self.grid_dimensions = grid_dimension #2D grid
        self.partition = []
        self.cell_size = cell_size
        self.grid_max = grid_max
        self.grid = {}
        self.sketches = {}
        self._unit_vectors = {}
        self._cell_cache = {}
        self._key_cells = {}
        self._sign_bits = {}
        self._updated_cells = set()
        self.sign_prefilter_scale = float(sign_prefilter_scale)
        self.sign_prefilter_extra = int(sign_prefilter_extra)
        self.neighbor_margin = float(neighbor_margin)
        self.extra_filter = _coerce_to_bool(extra_filter)
        # Parameters thresholds
        self.freq_threshold = freq_threshold
        self.corr_threshold = corr_threshold

        self.sketch_std = sketch_std
        self.n_vectors = n_vectors

    def dump_state(self):
        return dict(self.__dict__)

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def load_state(self, state):
        self.__dict__.update(state)

    def append_partition(self,curr_time,new_partition):
        if self.curr_time != curr_time:
            self.curr_time = curr_time
            self.partition.append(new_partition)
        else:
            self.partition[-1].update(new_partition)

    def _normalize_cell_coords(self, v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape[0] < self.grid_dimensions:
            raise ValueError("sketch dimension smaller than grid dimensionality")
        if self.neg_corr:
            normalized = 0.5 * (arr[:self.grid_dimensions] + 1.0)
        else:
            normalized = np.abs(arr[:self.grid_dimensions])
        return np.clip(normalized, 0.0, 1.0)

    def _get_cell(self, v):
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")

        coords = self._normalize_cell_coords(v)
        eps = self.cell_size * 1e-9
        return tuple(
            int(np.floor((coords[i] + eps) / self.cell_size))
            for i in range(self.grid_dimensions)
        )

    def normalized_sketch(self,sk,norm):
        eps = 1e-8
        #scale = 1.0 / (2.0 * max(norm, eps))  # from [-1,1] → [0,1]
        #return 0.5 + sk * scale
        # [-1, 1] domain
        return sk / max(norm, eps)

    def _compute_hamming_threshold(self):
        if self.grid_dimensions <= 0:
            return None
        nv = self.n_vectors if self.n_vectors is not None else 0
        if nv <= 0:
            freq_ratio = 0.0
        else:
            freq_ratio = max(0.0, min(1.0, max(self.freq_threshold, 0.0) / float(nv)))
        base_fraction = 0.45 - 0.10 * freq_ratio
        fraction = base_fraction * self.sign_prefilter_scale
        fraction = max(0.0, min(1.0, fraction))
        threshold = int(math.ceil(fraction * self.grid_dimensions + self.sign_prefilter_extra))
        return max(0, threshold)

    @staticmethod
    def _sign_prefilter(bits_a, bits_b, threshold):
        if threshold is None or bits_a is None or bits_b is None:
            return True, False

        diff = int(np.count_nonzero(np.bitwise_xor(bits_a, bits_b)))
        diff_flip = int(np.count_nonzero(np.bitwise_xor(bits_a, 1 - bits_b)))
        best = min(diff, diff_flip)
        passed = best <= threshold
        strong = passed and best <= (threshold // 2)
        return passed, strong

    def _cells_for_vector(self, vector, boundary_margin):
        cell_key = self._get_cell(vector)
        cells = {cell_key}

        if self.cell_size <= 0:
            return cells

        normalized = self._normalize_cell_coords(vector)
        raw_margin = max(boundary_margin, 0.0)
        radius = int(np.floor(raw_margin))
        fractional = raw_margin - radius
        frac_margin = fractional * self.cell_size

        base_indices = list(cell_key)
        offset_options = []

        for dim in range(self.grid_dimensions):
            idx = base_indices[dim]

            if idx < 0:
                offset_options.append({0})
                continue

            offsets = set(range(-radius, radius + 1))

            if fractional > 0:
                coord = normalized[dim]
                t = min(coord, 1.0 - coord)
                lower = idx * self.cell_size
                pos_in_cell = t - lower
                if pos_in_cell < 0.0:
                    pos_in_cell = 0.0
                if pos_in_cell > self.cell_size:
                    pos_in_cell = self.cell_size

                extra = radius + 1
                if pos_in_cell < frac_margin and idx - extra >= 0:
                    offsets.add(-extra)
                if pos_in_cell > self.cell_size - frac_margin:
                    offsets.add(extra)

            offset_options.append(offsets)

        for deltas in product(*offset_options):
            if all(delta == 0 for delta in deltas):
                continue
            neighbor = tuple(idx + delta for idx, delta in zip(base_indices, deltas))
            if any(n < 0 for n in neighbor):
                continue
            cells.add(neighbor)

        return cells

    def _input_grid(self, boundary_margin: float = 0.06):
        last_partition = self.partition[-1]
        updated_cells = set()

        norms = []
        for sk, is_constant, norm in last_partition.values():
            if not is_constant and norm is not None:
                norms.append(max(norm, 1e-8))
        shared_scale = float(np.median(norms)) if norms else 1.0

        updated_cells = set()

        for k, v in last_partition.items():
            sketch,is_constant,norm = v
            if is_constant: #if is constant
                self._key_cells.pop(k, None)
                continue
            scaled = sketch / shared_scale if shared_scale > 0 else sketch
            scaled = np.clip(scaled, -1.0, 1.0)
            sketch = scaled
            self.partition[-1][k] = (sketch,is_constant,norm)
            bits = (sketch >= 0.0).astype(np.uint8)
            self._sign_bits[k] = bits
            cells_for_key = set()
            variants = (sketch, -sketch) if self.neg_corr else (sketch,)
            for variant in variants:
                cells_for_key.update(self._cells_for_vector(variant, boundary_margin))

            self._key_cells[k] = tuple(sorted(cells_for_key))

            for cell in self._key_cells[k]:
                bucket = self.grid.setdefault(cell, [])
                bucket.append(k)
                updated_cells.add(cell)
            self.sketches[k] = sketch
            unit_vec = None
            vec = sketch
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                unit_vec = vec / vec_norm
            self._unit_vectors[k] = unit_vec
        # ensure constants still clear any stale cache entry
        for k, (_, is_constant, _) in self.partition[-1].items():
            if is_constant:
                self._unit_vectors[k] = None
                self._sign_bits.pop(k, None)
        for cell in updated_cells:
            self._cell_cache.pop(cell, None)

        self._updated_cells = updated_cells

    def _clean_old_sketches(self):
        if len(self.partition) > self.n_lagged_windows:
            old_partition = self.partition.pop(0)
            for k, v in old_partition.items():
                sketch,is_constant,norm = v
                self._unit_vectors.pop(k, None)
                self._sign_bits.pop(k, None)
                if is_constant: #if is constant
                    self._key_cells.pop(k, None)
                    continue
                cells = self._key_cells.pop(k, None)
                if cells is None:
                    fallback_cells = {self._get_cell(sketch), self._get_cell(-sketch)} if self.neg_corr else {self._get_cell(sketch)}
                    #fallback_cells = {self._get_cell(np.abs(sketch))} if self.neg_corr else {self._get_cell(sketch)}
                    cells = tuple(fallback_cells)
                for cell_key in cells:
                    bucket = self.grid.get(cell_key)
                    if bucket is None:
                        continue
                    if k in bucket:
                        bucket.remove(k)
                    self._cell_cache.pop(cell_key, None)
                    if not bucket:
                        del self.grid[cell_key]            
                del self.sketches[k]

    def _update_grid(self, n_ids):
        if self.partition and len(self.partition[-1]) >= n_ids:
            self._clean_old_sketches()
            margin = max(0.06, self.neighbor_margin)
            self._input_grid(boundary_margin=margin)
    
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
    
    def bucket_weight(self,cell):
        prob = 1.0
        for c in cell:
            left = c * self.cell_size
            right = (c + 1) * self.cell_size
            p = norm.cdf(right, scale=self.sketch_std) - norm.cdf(left, scale=self.sketch_std)
            prob *= p

        return float('inf') if prob == 0 else 1.0 / prob
    
    def _increment_candidates(self, freq_pairs, candidates):
        grid = self.grid
        if not self._updated_cells:
            return
        cells_to_scan = tuple(self._updated_cells)
        prefilter_cos = float(max(0.06, min(0.20, 0.20 * self.corr_threshold)))
        self.prefilter_cos = prefilter_cos
        unit_vectors = self._unit_vectors
        hamming_threshold = self._compute_hamming_threshold() if self.extra_filter else None

        for cell in cells_to_scan:
            series_ids = grid.get(cell)
            if not series_ids:
                continue
            if len(series_ids) < 2:
                continue

            cache_entry = self._cell_cache.get(cell)
            if cache_entry is None:
                cache_keys = []
                cache_vecs = []
                cache_bits = []
                cache_windows = []
                cache_times = []
                cache_series = []
                for key in series_ids:
                    vec = unit_vectors.get(key)
                    if vec is None:
                        continue
                    cache_keys.append(key)
                    cache_vecs.append(vec)
                    cache_bits.append(self._sign_bits.get(key))
                    cache_windows.append(key[2])
                    cache_times.append(key[1])
                    cache_series.append(key[0])
                if not cache_keys:
                    self._cell_cache[cell] = {"keys": (), "vecs": None}
                    continue
                vecs = np.stack(cache_vecs)
                bits = None
                if cache_bits and all(bit is not None for bit in cache_bits):
                    bits = np.asarray(cache_bits, dtype=np.uint8)
                cache_entry = {
                    "keys": tuple(cache_keys),
                    "vecs": vecs,
                    "bits": bits,
                    "windows": np.asarray(cache_windows, dtype=np.int64),
                    "times": np.asarray(cache_times, dtype=np.int64),
                    "series": np.asarray(cache_series, dtype=object),
                    "window_groups": None,
                }
                self._cell_cache[cell] = cache_entry
            else:
                vecs = cache_entry.get("vecs")
                if vecs is None or not cache_entry.get("keys"):
                    continue
                bits = cache_entry.get("bits")
                if bits is None:
                    cache_bits = []
                    for key in cache_entry["keys"]:
                        bit = self._sign_bits.get(key)
                        if bit is None:
                            cache_bits = None
                            break
                        cache_bits.append(bit)
                    if cache_bits:
                        bits = np.asarray(cache_bits, dtype=np.uint8)
                        cache_entry["bits"] = bits

            keys = cache_entry["keys"]
            windows = cache_entry["windows"]
            times = cache_entry["times"]
            series_arr = cache_entry["series"]
            bits = cache_entry.get("bits")

            curr_mask = times == self.curr_time
            if not np.any(curr_mask):
                continue

            window_groups = cache_entry.get("window_groups")
            if window_groups is None:
                grouped = {}
                for idx, window_size in enumerate(windows):
                    grouped.setdefault(window_size, []).append(idx)
                window_groups = {
                    w: np.asarray(idxs, dtype=np.int64)
                    for w, idxs in grouped.items()
                }
                cache_entry["window_groups"] = window_groups

            for window_size, window_idx in window_groups.items():
                if window_idx.size < 1:
                    continue

                window_curr_mask = curr_mask[window_idx]
                curr_idx = window_idx[window_curr_mask]
                if curr_idx.size == 0:
                    continue

                past_idx = window_idx[~window_curr_mask]

                curr_vecs = vecs[curr_idx]
                curr_bits = bits[curr_idx] if bits is not None else None
                curr_keys = [keys[idx] for idx in curr_idx]
                curr_series = series_arr[curr_idx]
                curr_times = times[curr_idx]

                past_bits = None
                if past_idx.size > 0:
                    past_vecs = vecs[past_idx]
                    past_bits = bits[past_idx] if bits is not None else None
                    past_keys = [keys[idx] for idx in past_idx]
                    past_series = series_arr[past_idx]
                    past_times = times[past_idx]

                    tile_size = 256
                    for tile_start in range(0, past_vecs.shape[0], tile_size):
                        tile_end = min(tile_start + tile_size, past_vecs.shape[0])
                        past_tile = past_vecs[tile_start:tile_end]
                        cos_matrix = curr_vecs @ past_tile.T

                        for i, key_a in enumerate(curr_keys):
                            cos_row = cos_matrix[i]
                            for local_j, key_b in enumerate(past_keys[tile_start:tile_end]):
                                global_j = tile_start + local_j
                                if curr_series[i] == past_series[global_j] and curr_times[i] == past_times[global_j]:
                                    continue
                                pair_id = self._normalize_key((key_a[0], key_b[0], key_a[1], key_b[1], window_size))
                                if pair_id in candidates:
                                    continue
                                bits_a = curr_bits[i] if curr_bits is not None else None
                                bits_b = past_bits[global_j] if past_bits is not None else None
                                if self.extra_filter:
                                    cos_val = float(cos_row[local_j])
                                    if (not self.neg_corr and cos_val < self.prefilter_cos) or (self.neg_corr and abs(cos_val) < self.prefilter_cos):
                                        continue
                                    if hamming_threshold is not None:
                                        passed, _ = self._sign_prefilter(bits_a, bits_b, hamming_threshold)
                                        if not passed:
                                            continue
                                freq_pairs[pair_id] = freq_pairs.get(pair_id, 0) + self.grid_dimensions
                                if freq_pairs[pair_id] >= self.freq_threshold:
                                    candidates[pair_id] = 1

                if curr_idx.size > 1:
                    cos_curr = curr_vecs @ curr_vecs.T
                    for i in range(curr_idx.size):
                        key_i = curr_keys[i]
                        for j in range(i + 1, curr_idx.size):
                            key_j = curr_keys[j]
                            if curr_series[i] == curr_series[j] and curr_times[i] == curr_times[j]:
                                continue
                            pair_id = self._normalize_key((key_i[0], key_j[0], key_i[1], key_j[1], window_size))
                            if pair_id in candidates:
                                continue
                            bits_i = curr_bits[i] if curr_bits is not None else None
                            bits_j = curr_bits[j] if curr_bits is not None else None
                            if self.extra_filter:
                                cos_val = float(cos_curr[i, j])
                                if (not self.neg_corr and cos_val < self.prefilter_cos) or (self.neg_corr and abs(cos_val) < self.prefilter_cos):
                                    continue
                                if hamming_threshold is not None:
                                    passed, _ = self._sign_prefilter(bits_i, bits_j, hamming_threshold)
                                    if not passed:
                                        continue
                            freq_pairs[pair_id] = freq_pairs.get(pair_id, 0) + self.grid_dimensions
                            if freq_pairs[pair_id] >= self.freq_threshold:
                                candidates[pair_id] = 1
        
        self._updated_cells.clear()

                #weight = self.bucket_weight(cell)
                
                #if np.sqrt(sum((self.sketches[a]-self.sketches[b])**2)) > 0.25:
                #    continue
                
                #cos_sim = np.dot(self.sketches[a],self.sketches[b])
                #freq_pairs[pair_id] = freq_pairs.get(pair_id, 0) + cos_sim #1 #weight
                #if abs(freq_pairs[pair_id]) >= self.corr_threshold-0.2:
                #    candidates[pair_id] = 1

    def update_n_lagged_windows(self,n_lagged_windows):
        self.n_lagged_windows = n_lagged_windows

    def _print_grid(self):
        for cell, series_ids in self.grid.items():
            print("Cell ",cell,": ",series_ids)
    
    def _print_freq_pairs(self,freq_pairs):
        if len(freq_pairs)>0:
            for pair,freq in freq_pairs.items():
                print("Pair ",pair,": ",freq)
    
    def _print_candidates(self,candidates):
        if len(candidates)>0:
            for pair,isCandidate in candidates.items():
                print("Pair ",pair,": ",isCandidate)

    def _print_state(self,freq_pairs,candidates):
        for d in range(self.grid_dimensions):
            if len(self.grid[d])>0:
                if(self.testing):
                    print("\nGrid:")
                    self._print_grid()
                    print("\nFrequency of pairs:")
                    self._print_freq_pairs(freq_pairs)
                    print("\nCandidates:")
                    self._print_candidates(candidates)

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
    def __init__(self,train_data,ids,window_size,window_step,n_lags,corr_threshold,recall_by_window,alg,neg_corr,corr_val, extra_filter=False, exec="parallel",max_workers=0):

        self.neg_corr = neg_corr
        self.corr_val = corr_val
        self.extra_filter = _coerce_to_bool(extra_filter)

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
        
        self.corrtrack_bf = CorrTrack(window_size=self.window_size,basic_window=None,window_step=window_step,n_vectors=1,n_lags=self.n_lags,
                                    grid_dimension=1,cell_size=1,warmup_data=None,seed=None,seed_toggle=None,corr_threshold=self.corr_threshold,
                                    neg_corr=self.neg_corr,preprocess=False,extra_filter=self.extra_filter,exec=self.exec,max_workers=self.max_workers)
        
        self.window_step = self.corrtrack_bf.window_step
        self.basic_window = self.corrtrack_bf.basic_window

        self.alg = alg
        self.path = None

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
        scheduler_get = dask_multiprocessing_get
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
    
    def _plot_corrtrack_histograms(self, corrtrack, save=True):
        bins = 50
        r = corrtrack.n_vectors

        # Create subplots
        fig, axes = plt.subplots(3, 6, figsize=(60, 10), sharex=False)
        axes = axes.flatten()

        # Determine common x-axis limits for the first 3 histograms
        dist_all = corrtrack.corr_dist_pos + corrtrack.corr_dist_neg + corrtrack.noncorr_dist
        x_min = min(dist_all)
        x_max = max(dist_all)

        # Plot positive correlation distances
        axes[0].hist(corrtrack.corr_dist_pos, bins=bins, color='green', alpha=0.7)
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_title('Distances for Positive Correlation (> 0.7)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)

        # Plot negative correlation distances
        axes[6].hist(corrtrack.corr_dist_neg, bins=bins, color='red', alpha=0.7)
        axes[6].set_xlim(x_min, x_max)
        axes[6].set_title('Distances for Negative Correlation (< -0.7)')
        axes[6].set_ylabel('Frequency')
        axes[6].grid(True)

        # Plot non-correlated distances
        axes[12].hist(corrtrack.noncorr_dist, bins=bins, color='blue', alpha=0.7)
        axes[12].set_xlim(x_min, x_max)
        axes[12].set_title('Distances for Non-Correlated (|ρ| < 0.7)')
        axes[12].set_ylabel('Frequency')
        axes[12].grid(True)

        # Determine common x-axis limits for the first 3 histograms
        dist_all = corrtrack.corr_norm_dist_pos + corrtrack.corr_norm_dist_neg + corrtrack.noncorr_norm_dist
        x_min = min(dist_all)
        x_max = max(dist_all)

        # Plot positive correlation distances
        axes[1].hist(corrtrack.corr_norm_dist_pos, bins=bins, color='green', alpha=0.7)
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_title('Normalized distances for Positive Correlation (> 0.7)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)

        # Plot negative correlation distances
        axes[7].hist(corrtrack.corr_norm_dist_neg, bins=bins, color='red', alpha=0.7)
        axes[7].set_xlim(x_min, x_max)
        axes[7].set_title('Normalized distances for Negative Correlation (< -0.7)')
        axes[7].set_ylabel('Frequency')
        axes[7].grid(True)

        # Plot non-correlated distances
        axes[13].hist(corrtrack.noncorr_norm_dist, bins=bins, color='blue', alpha=0.7)
        axes[13].set_xlim(x_min, x_max)
        axes[13].set_title('Normalized distances for Non-Correlated (|ρ| < 0.7)')
        axes[13].set_ylabel('Frequency')
        axes[13].grid(True)

        # Determine common x-axis limits for the first 3 histograms
        dist_all = corrtrack.corr_dist_sk_pos + corrtrack.corr_dist_sk_neg + corrtrack.noncorr_dist_sk
        x_min = min(dist_all)
        x_max = max(dist_all)

        # Plot positive correlation distances
        axes[2].hist(corrtrack.corr_dist_sk_pos, bins=bins, color='green', alpha=0.7)
        axes[2].set_xlim(x_min, x_max)
        axes[2].set_title('Sketch Distances for Positive Correlation (> 0.7)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True)

        # Plot negative correlation distances
        axes[8].hist(corrtrack.corr_dist_sk_neg, bins=bins, color='red', alpha=0.7)
        axes[8].set_xlim(x_min, x_max)
        axes[8].set_title('Sketch Distances for Negative Correlation (< -0.7)')
        axes[8].set_ylabel('Frequency')
        axes[8].grid(True)

        # Plot non-correlated distances
        axes[14].hist(corrtrack.noncorr_dist_sk, bins=bins, color='blue', alpha=0.7)
        axes[14].set_xlim(x_min, x_max)
        axes[14].set_title('Sketch Distances for Non-Correlated (|ρ| < 0.7)')
        axes[14].set_ylabel('Frequency')
        axes[14].grid(True)

        # Determine common x-axis limits for the first 3 histograms
        dist_all = corrtrack.corr_dist_norm_sk_pos + corrtrack.corr_dist_norm_sk_neg + corrtrack.noncorr_dist_norm_sk
        x_min = min(dist_all)
        x_max = max(dist_all)
        
        # Plot positive correlation distances
        axes[3].hist(corrtrack.corr_dist_norm_sk_pos, bins=bins, color='green', alpha=0.7)
        axes[3].set_xlim(x_min, x_max)
        axes[3].set_title('Normalized Sketch Distances for Positive Correlation (> 0.7)')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(True)

        # Plot negative correlation distances
        axes[9].hist(corrtrack.corr_dist_norm_sk_neg, bins=bins, color='red', alpha=0.7)
        axes[9].set_xlim(x_min, x_max)
        axes[9].set_title('Normalized Sketch Distances for Negative Correlation (< -0.7)')
        axes[9].set_ylabel('Frequency')
        axes[9].grid(True)

        # Plot non-correlated distances
        axes[15].hist(corrtrack.noncorr_dist_norm_sk, bins=bins, color='blue', alpha=0.7)
        axes[15].set_xlim(x_min, x_max)
        axes[15].set_title('Normalized Sketch Distances for Non-Correlated (|ρ| < 0.7)')
        axes[15].set_ylabel('Frequency')
        axes[15].grid(True)

        # Determine common x-axis limits for the first 3 histograms
        dist_all = corrtrack.corr_est_pos + corrtrack.corr_est_neg + corrtrack.noncorr_est
        x_min = min(dist_all)
        x_max = max(dist_all)

        # Plot positive correlation distances
        axes[4].hist(corrtrack.corr_est_pos, bins=bins, color='green', alpha=0.7)
        axes[4].set_xlim(x_min, x_max)
        axes[4].set_title('Cosine Similarity for Positive Correlation (> 0.7)')
        axes[4].set_ylabel('Frequency')
        axes[4].grid(True)

        # Plot negative correlation distances
        axes[10].hist(corrtrack.corr_est_neg, bins=bins, color='red', alpha=0.7)
        axes[10].set_xlim(x_min, x_max)
        axes[10].set_title('Cosine Similarity for Negative Correlation (< -0.7)')
        axes[10].set_ylabel('Frequency')
        axes[10].grid(True)

        # Plot non-correlated distances
        axes[16].hist(corrtrack.noncorr_est, bins=bins, color='blue', alpha=0.7)
        axes[16].set_xlim(x_min, x_max)
        axes[16].set_title('Cosine Similarity for Non-Correlated (|ρ| < 0.7)')
        axes[16].set_ylabel('Frequency')
        axes[16].grid(True)


        # Plot correlation estimate differences
        axes[5].hist(corrtrack.corr_est_diffs, bins=bins, color='purple', alpha=0.7)
        axes[5].set_title('Correlation Estimate Differences (estimated − true)')
        axes[5].set_xlabel('Value')
        axes[5].set_ylabel('Frequency')
        axes[5].grid(True)

        # Super title and layout
        fig.suptitle(f'Distances and Correlation Estimation (n_vectors = {r})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        n_ts = self.train_data.shape[0]-1
        n_w = (np.floor((self.train_data.shape[1]-self.window_size)/self.window_step) + 1)
        # Save or show
        if save:
            hist_dir = os.path.join(self.path, "hist")
            os.makedirs(hist_dir, exist_ok=True)
            save_path = os.path.join(hist_dir, f"corrtrack_hist_{n_ts}_{n_w}_r{r}.png")
            plt.savefig(save_path)
            print(f"[INFO] Histogram figure saved to: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def _get_bf_ground_truth(self):
        """
        Compute brute-force ground truth with the same inner-execution policy
        that CorrTrack would use (policy-relative). The outer loop is sequential.
        """
        length_data = self.train_data.shape[1]

        start_time = time.time()
        for start in range(0, length_data - self.window_step + 1, self.window_step):
            chunk = self.train_data[:, start:start + self.window_step]
            self.corrtrack_bf.run_bf(chunk, self.ids, verbose=False, testing=False, corr_val=True)
        end_time = time.time()
        runtime = end_time - start_time
        print("Run Brute-Force, Finished in ",runtime)

        # Get flags based on brute-force
        if self.recall_by_window:
            self.ground_truth = self.corrtrack_bf.correlated
        else:
            self.ground_truth = self.corrtrack_bf.get_correlation_flags(length_data, 0)

        self.runtime_bf = runtime
        self.pair_min_dist_bf = self.corrtrack_bf.pair_min_dist
    
    def _run_corrtrack_distances(self,args):
        param_combo = args

        # Parameters
        nodes = param_combo["nodes"]

        # Parameters windows
        window_size = self.window_size
        param_combo["window_size"] = window_size

        window_step = self.window_step
        param_combo["window_step"] = window_step

        basic_window = self.basic_window
        param_combo["basic_window"] = basic_window
        
        warmup_size = param_combo["warmup_size"]
        try:
            warmup_size = float(warmup_size)
        except (TypeError, ValueError):
            warmup_size = None
        try:
            warmup_size = float(warmup_size)
        except (TypeError, ValueError):
            warmup_size = None
        
        # Parameters sketches
        seed = param_combo["seed"]
        seed_toggle = param_combo["seed_toggle"]
        n_vectors = param_combo["n_vectors"]
        preprocess = param_combo["preprocess"]
        extra_filter = _coerce_to_bool(param_combo.get("extra_filters", self.extra_filter))
        param_combo["extra_filters"] = extra_filter
        
        # Parameters lags
        n_lags = self.n_lags
        param_combo["n_lags"] = n_lags
        
        # Parameters thresholds
        corr_threshold = self.corr_threshold
        param_combo["corr_threshold"] = corr_threshold
        
        try:
            length_data = self.train_data.shape[1]
            if warmup_size is not None:
                warmup_len = round(warmup_size * length_data)
                if warmup_len <= 0:
                    warmup_len = 1
                warmup_len = min(length_data, warmup_len)
                warmup_data = self.train_data[:, :warmup_len]
            else:
                warmup_data = None

            corrtrack = CorrTrack(window_size=window_size,basic_window=basic_window,window_step=window_step,n_vectors=n_vectors,n_lags=n_lags,
                                grid_dimension=1,cell_size=1,warmup_data=warmup_data,seed=seed,seed_toggle=seed_toggle,
                                freq_threshold=0,corr_threshold=corr_threshold,neg_corr=self.neg_corr,preprocess=preprocess,extra_filter=extra_filter,exec=self.exec,max_workers=self.max_workers)

            for start in range(0, length_data - self.window_step + 1, self.window_step):
                chunk = self.train_data[:, start:(start + self.window_step)]
                corrtrack.run_train_distances(chunk, self.ids, verbose=False, testing=False,policy=self.exec)
            
            #print("Distances pos_corr",corrtrack.n_vectors,min(corrtrack.corr_dist_pos),max(corrtrack.corr_dist_pos),np.mean(corrtrack.corr_dist_pos),np.std(corrtrack.corr_dist_pos))
            #print("Distances neg_corr",corrtrack.n_vectors,min(corrtrack.corr_dist_neg),max(corrtrack.corr_dist_neg),np.mean(corrtrack.corr_dist_neg),np.std(corrtrack.corr_dist_neg))
            #print("Distances noncorr",corrtrack.n_vectors,min(corrtrack.noncorr_dist),max(corrtrack.noncorr_dist),np.mean(corrtrack.noncorr_dist),np.std(corrtrack.noncorr_dist))
            #print("Corr_est ",corrtrack.n_vectors,min(corrtrack.corr_est_pos),max(corrtrack.corr_est_pos),min(corrtrack.corr_est_neg),max(corrtrack.corr_est_neg),min(corrtrack.noncorr_est),max(corrtrack.noncorr_est))
            #print("Corr diffs ",min(corrtrack.corr_est_diffs),max(corrtrack.corr_est_diffs),np.mean(corrtrack.corr_est_diffs),np.std(corrtrack.corr_est_diffs))
            #plt.hist(corrtrack.corr_est_diffs)
            #plt.show()
            self._plot_corrtrack_histograms(corrtrack)

            return {n_vectors:{
                    "corr_dist_sk_pos":np.quantile(corrtrack.corr_dist_sk_pos, 0.99),
                    "corr_dist_norm_sk_pos":np.quantile(corrtrack.corr_dist_norm_sk_pos, 0.99),
                    "corr_est_pos":np.quantile(corrtrack.corr_est_pos, 0.01)}}

        except Exception as e:
            print(f"Skipped params={param_combo} due to error: {e}")
            return None

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
        record["window_size"] = self.window_size
        record["window_step"] = self.window_step
        record["basic_window"] = self.basic_window
        record["n_lags"] = self.n_lags
        record["warmup_size"] = param_combo.get("warmup_size")
        record["seed"] = param_combo.get("seed")
        record["seed_toggle"] = param_combo.get("seed_toggle")
        record["preprocess"] = param_combo.get("preprocess")
        record["extra_filters"] = _coerce_to_bool(param_combo.get("extra_filters", self.extra_filter))
        record["corr_threshold"] = self.corr_threshold
        record["grid_max"] = param_combo.get("grid_max")
        record["cell_size"] = param_combo.get("cell_size")
        record["n_vectors"] = param_combo.get("n_vectors")
        record["grid_dimension"] = param_combo.get("grid_dimension")
        record["freq_threshold"] = param_combo.get("freq_threshold")

        record["cand_time_bf"] = getattr(self.corrtrack_bf, "candidate_time", None)
        record["val_time_bf"] = getattr(self.corrtrack_bf, "validation_time", None)
        record["monit_time_bf"] = getattr(self.corrtrack_bf, "monitor_time", None)
        record["runtime_bf"] = getattr(self, "runtime_bf", None)
        record["artifact_time_bf"] = getattr(self, "artifact_time_bf", None)
        record["corr_w_bf"] = getattr(self.corrtrack_bf, "validated_candidates", None)
        record["tested_w_bf"] = getattr(self.corrtrack_bf, "tested_candidates", None)
        record["cand_w_bf"] = getattr(self.corrtrack_bf, "total_candidates", None)
        record["artifact_time"] = None

        record["status"] = "pending"
        record["error"] = ""
        return record
    
    def _run_corrtrack(self, args):
        param_combo, dataset_id = args
        record = self._init_optim_record(dataset_id, param_combo)

        nodes = param_combo.get("nodes")
        try:
            nodes = int(nodes) if nodes is not None else None
        except (TypeError, ValueError):
            nodes = None
        warmup_size = param_combo.get("warmup_size")
        try:
            warmup_ratio = float(warmup_size)
        except (TypeError, ValueError):
            warmup_ratio = None
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
        extra_filter = _coerce_to_bool(param_combo.get("extra_filters", self.extra_filter))
        grid_dimension = param_combo.get("grid_dimension")
        try:
            grid_dimension = int(grid_dimension) if grid_dimension is not None else None
        except (TypeError, ValueError):
            grid_dimension = None
        cell_size = param_combo.get("cell_size")
        freq_threshold = param_combo.get("freq_threshold")
        try:
            freq_threshold = float(freq_threshold) if freq_threshold is not None else None
        except (TypeError, ValueError):
            freq_threshold = None
        try:
            cell_size = float(cell_size) if cell_size is not None else None
        except (TypeError, ValueError):
            cell_size = None

        record["nodes"] = nodes
        record["warmup_size"] = warmup_ratio
        record["seed"] = seed
        record["seed_toggle"] = seed_toggle
        record["n_vectors"] = n_vectors
        record["preprocess"] = preprocess
        record["extra_filters"] = extra_filter
        record["grid_dimension"] = grid_dimension
        record["cell_size"] = cell_size
        record["freq_threshold"] = freq_threshold

        length_data = self.train_data.shape[1]

        try:
            if warmup_ratio is not None:
                warmup_len = round(warmup_ratio * length_data)
                if warmup_len <= 0:
                    warmup_len = 1
                warmup_len = min(length_data, warmup_len)
                warmup_data = self.train_data[:, :warmup_len]
            else:
                warmup_data = None

            corrtrack = CorrTrack(
                window_size=self.window_size,
                basic_window=self.basic_window,
                window_step=self.window_step,
                n_vectors=n_vectors,
                n_lags=self.n_lags,
                grid_dimension=grid_dimension,
                cell_size=cell_size,
                warmup_data=warmup_data,
                seed=seed,
                seed_toggle=seed_toggle,
                freq_threshold=freq_threshold,
                corr_threshold=self.corr_threshold,
                neg_corr=self.neg_corr,
                preprocess=preprocess,
                extra_filter=extra_filter,
                exec=self.exec,
                max_workers=self.max_workers,
            )

            record["grid_max"] = corrtrack.grid_max
            record["cell_size"] = corrtrack.cell_size
            record["n_lags"] = corrtrack.n_lags
            record["basic_window"] = corrtrack.basic_window
            record["grid_dimension"] = corrtrack.grid_dimension

            start_time = time.time()
            for start in range(0, length_data - self.window_step + 1, self.window_step):
                chunk = self.train_data[:, start : (start + self.window_step)]
                corrtrack.run(chunk, self.ids, verbose=False, testing=False, corr_val=self.corr_val)
            end_time = time.time()
            runtime = end_time - start_time - corrtrack.train_dist_time
            runtime = max(runtime, 0.0)
            record["runtime"] = runtime
            record["sk_time"] = corrtrack.sketch_time
            record["cand_time"] = corrtrack.candidate_time
            record["val_time"] = corrtrack.validation_time
            record["monit_time"] = corrtrack.monitor_time
            record["corr_w"] = corrtrack.validated_candidates
            record["tested_w"] = corrtrack.tested_candidates
            record["cand_w"] = corrtrack.total_candidates
            record["mem_w"] = corrtrack.n_lagged_windows - 1

            runtime_bf = record.get("runtime_bf") or 0.0
            if runtime > 0:
                record["speedup"] = runtime_bf / runtime
            else:
                record["speedup"] = float("inf")

            if self.recall_by_window:
                corr_flags = corrtrack.correlated
            else:
                corr_flags = corrtrack.get_correlation_flags(length_data, 0)

            metrics = CorrTrack.compute_metrics_bf(
                corr_flags, self.ground_truth, self.recall_by_window, self.pair_min_dist_bf
            )

            record["precision_pos"] = metrics["precision_pos"]
            record["recall_pos"] = metrics["recall_pos"]
            record["f1_pos"] = metrics["f1_score_pos"]
            record["precision_neg"] = metrics["precision_neg"]
            record["recall_neg"] = metrics["recall_neg"]
            record["f1_neg"] = metrics["f1_score_neg"]
            record["precision"] = metrics["precision"]
            record["recall"] = metrics["recall"]
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

        return record

    def _train_distances(self, param_grid_all):
        # Filter grid (unchanged)
        param_grid = {k: v for k, v in param_grid_all.items()
                      if k not in ["grid_dimension", "cell_size", "freq_threshold"]}

        names = list(param_grid.keys())
        tasks = [dict(zip(names, vals)) for vals in itertools.product(*param_grid.values())]

        thresholds = []
        i = 0
        for result in self._outer_iter(tasks, self._run_corrtrack_distances, unordered=True):
            if result:
                n_vectors = list(result.keys())[0]
                print("Run train distances ", i, "/", len(tasks))
                print("[n_vectors = ", n_vectors, "] - Quantiles 99%:\n",
                      "Dist sketch:", result[n_vectors]["corr_dist_sk_pos"],
                      "Dist norm sketch:", result[n_vectors]["corr_dist_norm_sk_pos"],
                      "Cosine similarity:", result[n_vectors]["corr_est_pos"])
                thresholds.append(result)
                i += 1
        return thresholds

    def _run_options(self, param_grid, output_csv, dataset_id):
        self.path = os.path.dirname(os.path.abspath(output_csv))

        names = list(param_grid.keys())
        tasks = [(dict(zip(names, vals)), dataset_id) for vals in itertools.product(*param_grid.values())]

        if self.ground_truth is None:
            self._get_bf_ground_truth()

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
    
    def get_train_distances(self, param_grid, output_csv):
        self.path = os.path.dirname(os.path.abspath(output_csv))
        return self._train_distances(param_grid)
    
    def get_optim_params(self, param_grid, output_csv, dataset_id, run=True, target_recall=0.95):
        output_csv = output_csv + "_" + self.alg + ".csv"
        if run:
            self._run_options(param_grid, output_csv, dataset_id)

        metrics = pd.read_csv(output_csv)
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
    def __init__(self,train_data,test_data,ids,window_size,window_step,basic_window,n_lags,corr_threshold,param_grid,recall_by_window,neg_corr,corr_val,algs=None, exec="parallel", max_workers=0, extra_filter=False):
        
        self.neg_corr = neg_corr
        self.corr_val = corr_val
        self.extra_filter = _coerce_to_bool(extra_filter)

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

        self.corrtrack_bf = CorrTrack(window_size=self.window_size,basic_window=basic_window,window_step=window_step,n_vectors=1,n_lags=self.n_lags,
                                    grid_dimension=1,cell_size=1,warmup_data=None,seed=None,seed_toggle=None,
                                    corr_threshold=self.corr_threshold,neg_corr=self.neg_corr,preprocess=False,
                                    extra_filter=self.extra_filter,exec=self.exec,max_workers=self.max_workers)
        
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
        scheduler_get = dask_multiprocessing_get
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
    
    def _plot_top_durations(self, corr_csv, data, output_folder):
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Load the correlation metadata
        df_corr = pd.read_csv(corr_csv)
        df_corr['start_time_id1'] = pd.to_datetime(df_corr['start_time_id1'])
        df_corr['start_time_id2'] = pd.to_datetime(df_corr['start_time_id2'])

        # Sort and get top 5 durations
        top5 = df_corr.sort_values(by='duration', ascending=False).head(5)

        # Load the time series dataset
        times = np.array(data[0, :], dtype='datetime64[ns]')
        data = data[1:,:].astype(float)

        # Go through top 5
        for i, row in top5.iterrows():
            id1, id2 = row['id1'], row['id2']
            duration = row['duration']
            lag = row['lag']            
            start_time_id1 = np.datetime64(row['start_time_id1'])
            start_time_id2 = np.datetime64(row['start_time_id2'])
            start_time = min(start_time_id1,start_time_id2)
            max_start_time = max(start_time_id1,start_time_id2)

            # Find the index of the exact start_time and end_time
            start_idx = np.where(times == start_time)[0][0]
            max_start_idx = np.where(times == max_start_time)[0][0]
            start_idx_id1 = np.where(times == start_time_id1)[0][0]
            start_idx_id2 = np.where(times == start_time_id2)[0][0]

            end_time_id1 = times[min(len(times), start_idx_id1 + duration)]
            end_time_id2 = times[min(len(times), start_idx_id2 + duration)]

            # Get column indices from 'ids'
            idx1 = np.where(self.ids==id1)[0][0]
            idx2 = np.where(self.ids==id2)[0][0]

            # Compute range using index
            # Compute centralized ranges for each series
            half_window = self.window_size
            center1 = start_idx_id1 + duration // 2
            center2 = start_idx_id2 + duration // 2

            start_range_id1 = max(0, center1 - half_window)
            end_range_id1 = min(len(times), center1 + half_window)
            start_range_id2 = max(0, center2 - half_window)
            end_range_id2 = min(len(times), center2 + half_window)

            # Time windows for independent x-axes
            time_window_id1 = times[start_range_id1:end_range_id1]
            time_window_id2 = times[start_range_id2:end_range_id2]

            series1 = data[idx1, start_range_id1:end_range_id1]
            series2 = data[idx2, start_range_id2:end_range_id2]

            # Plotting with independent x-axes
            fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

            # Top plot for id1
            axs[0].plot(time_window_id1, series1, label=id1, color='blue')
            axs[0].axvline(x=start_time_id1, color='red', linestyle='--', label='Start Time')
            axs[0].axvline(x=end_time_id1 + np.timedelta64(duration, 's'), color='orange', linestyle='--', label='End Time')
            axs[0].set_ylabel(id1)
            axs[0].legend(loc='upper left')
            axs[0].grid(True)

            # Bottom plot for id2
            axs[1].plot(time_window_id2, series2, label=id2, color='green')
            axs[1].axvline(x=start_time_id2, color='red', linestyle='--', label='Start Time')
            axs[1].axvline(x=end_time_id2 + np.timedelta64(duration, 's'), color='orange', linestyle='--', label='End Time')
            axs[1].set_ylabel(id2)
            axs[1].legend(loc='upper left')
            axs[1].grid(True)

            plt.xlabel("Time (independent axes)")
            plt.tight_layout()
            filename = f"{id1.replace('/', '_')}_{id2.replace('/', '_')}_{lag}.png"
            output_csv = os.path.join(output_folder, "plots", filename)
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            plt.savefig(output_csv)
            plt.close()

    def _mode_run(self,mode,alg,path,prefix,nodes,seed,seed_toggle,n_vectors,grid_dimension,cell_size,grid_max,freq_threshold,warmup_data,preprocess,extra_filter):
        length_data = self.test_data.shape[1]
        data_stream = self.test_data
        extra_filter_flag = _coerce_to_bool(extra_filter, self.extra_filter)

        # Instantiating corrtrack objects
        corrtrack = CorrTrack(window_size=self.window_size,basic_window=self.basic_window,window_step=self.window_step,n_vectors=n_vectors,n_lags=self.n_lags,
                            grid_dimension=grid_dimension,cell_size=cell_size,warmup_data=warmup_data,seed=seed,seed_toggle=seed_toggle,
                            freq_threshold=freq_threshold,corr_threshold=self.corr_threshold,neg_corr=self.neg_corr,preprocess=preprocess,
                            extra_filter=extra_filter_flag,exec=self.exec,max_workers=nodes)
        if mode == "main":
            #Running
            start_time = time.time()
            for start in range(0, length_data - self.window_step + 1, self.window_step):
                chunk = data_stream[:, start:start + self.window_step]
                corrtrack.run(chunk,self.ids,verbose=False,testing=False,corr_val=self.corr_val)
            end_time = time.time()
            runtime = end_time - start_time
            runtime -= corrtrack.train_dist_time
            runtime_parts = (corrtrack.sketch_time,corrtrack.candidate_time,corrtrack.validation_time,corrtrack.monitor_time)
            self.tested_w = corrtrack.tested_candidates
            self.correlated_w = corrtrack.validated_candidates
            self.total_w = corrtrack.total_candidates
        elif mode == "bf":
            start_time = time.time()
            for start in range(0, length_data - self.window_step + 1, self.window_step):
                chunk = data_stream[:, start:start + self.window_step]
                corrtrack.run_bf(chunk,self.ids,verbose=False,testing=False,corr_val=True)
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
            self.pair_min_dist_bf = None

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
        corrtrack_ho = CorrTrack_optimize(self.train_data,self.ids,self.window_size,self.window_step,self.n_lags,self.corr_threshold,self.recall_by_window,alg,self.neg_corr,self.corr_val,extra_filter=self.extra_filter,exec=self.exec)

        return corrtrack_ho.get_optim_params(self.param_grid,hyper_param_csv,dataset_id,run,target_recall)

    def _slice_warmup(self, data, warmup_ratio):
        if warmup_ratio is None:
            return None

        try:
            warmup_ratio = float(warmup_ratio)
        except (TypeError, ValueError):
            return None

        if warmup_ratio <= 0:
            warmup_ratio = 0

        length = data.shape[1]
        warmup_len = round(warmup_ratio * length)
        if warmup_len <= 0:
            warmup_len = 1

        warmup_len = min(length, warmup_len)
        return data[:, :warmup_len]

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
        bf_entries = self._load_maxlag_csv(bf_csv)
        candidate_entries = self._load_maxlag_csv(candidate_csv)

        bf_keys = set(bf_entries.keys())
        candidate_keys = set(candidate_entries.keys())

        tp = len(bf_keys & candidate_keys)
        fp = len(candidate_keys - bf_keys)
        fn = len(bf_keys - candidate_keys)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        mean_diff = float("nan")
        std_diff = float("nan")
        common_keys = sorted(
            bf_keys & candidate_keys,
            key=lambda item: (item[0], item[1], str(item[2])),
        )
        if common_keys:
            diffs = [bf_entries[key] - candidate_entries[key] for key in common_keys]
            diffs = [0.0 if math.isclose(d, 0.0, rel_tol=1e-10, abs_tol=1e-12) else round(d, 12) for d in diffs]
            if diffs:
                diffs_arr = np.asarray(diffs, dtype=float)
                mean_diff = float(np.mean(diffs_arr))
                std_diff = float(np.std(diffs_arr, ddof=0))

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
                reader = csv.DictReader(file)
                for row in reader:
                    id1 = row.get("id1")
                    id2 = row.get("id2")
                    if id1 is None or id2 is None:
                        continue
                    time1 = self._coerce_time_value(row.get("time1"))
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

        df = pd.read_csv(csv_path)
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
        recall_min = metrics["recall_min"]
        f1 = metrics["f1_score"]
        aucroc = metrics["aucroc"]
        pr_auc = metrics["pr_auc"]

        maxlag_precision, maxlag_recall, maxlag_f1, maxlag_diff_mean, maxlag_diff_std = maxlag_metrics

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
            as_optional_int(record.get("window_size")),
            as_optional_int(record.get("window_step")),
            as_optional_int(record.get("basic_window")),
            as_optional_int(record.get("n_lags")),
            fmt(record.get("warmup_size")),
            as_optional_int(record.get("seed")),
            as_optional_int(record.get("seed_toggle")),
            str(record.get("preprocess")),
            str(record.get("extra_filters")),
            fmt(record.get("corr_threshold")),
            fmt(record.get("grid_max")),
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
            as_int_str(bf_record.get("correlated")),
            as_int_str(record.get("correlated")),
            as_int_str(bf_record.get("tested")),
            as_int_str(record.get("tested")),
            as_int_str(bf_record.get("total_candidates")),
            as_int_str(record.get("total_candidates")),
            fmt(precision_pos),
            fmt(recall_pos),
            fmt(f1_pos),
            fmt(precision_neg),
            fmt(recall_neg),
            fmt(f1_neg),
            fmt(precision),
            fmt(recall),
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
        ground_truth = self._load_correlated_dict(bf_prefix)
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
            predicted = self._load_correlated_dict(prefix)

            metrics = CorrTrack.compute_metrics_bf(
                predicted,
                ground_truth,
                self.recall_by_window,
                self.pair_min_dist_bf,
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
                    predicted = self._load_correlated_dict(prefix)
                    fil_metrics = CorrTrack.compute_metrics_bf(
                        predicted,
                        ground_truth,
                        self.recall_by_window,
                        self.pair_min_dist_bf,
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
            writer = csv.writer(file)
            writer.writerow(COMPARISON_COLUMNS)
            writer.writerows(results)

        return results

    def _parallel_mode_run(self,args):
        dataset_id, mode, alg, path, prefix, bst, runtime_bf, corr_flags_bf, warmup_data = args
        extra_filter_flag = _coerce_to_bool(bst.get("extra_filters"), self.extra_filter)
        runtime_parts, runtime, artifact_time, corr_flags = self._mode_run(
            mode, alg, path, prefix,
            bst["nodes"], bst["seed"], bst["seed_toggle"],
            bst["n_vectors"], bst["grid_dimension"],
            bst["cell_size"], bst["grid_max"],
            bst["freq_threshold"], warmup_data, bst["preprocess"], extra_filter_flag
        )
        speedup = runtime_bf / runtime if runtime else float("inf")
        metrics = CorrTrack.compute_metrics_bf(corr_flags, corr_flags_bf,self.recall_by_window,self.pair_min_dist_bf)
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

        n_ts = self.test_data.shape[0]-1
        n_w = (np.floor((self.test_data.shape[1]-self.window_size)/self.window_step) + 1)
        total_w = n_ts*n_w
        mem_w = self.n_lags//self.window_step

        param_keys = [
            "nodes","window_size","window_step","basic_window","n_lags","warmup_size",
            "seed","seed_toggle","preprocess","extra_filters","corr_threshold",
            "grid_max","cell_size","n_vectors","grid_dimension","freq_threshold"
        ]
        param_values = []
        for key in param_keys:
            if key == "extra_filters":
                param_values.append(extra_filter_flag)
            else:
                param_values.append(bst.get(key))

        return [
            dataset_id, mode, alg, "recall_speedup", n_ts, n_w, total_w, mem_w,
        ] + param_values + [
            f"{self.candidate_time_bf:.4f}",f"{self.validation_time_bf:.4f}",f"{self.monitor_time_bf:.4f}",f"{self.runtime_bf:.4f}",f"{bf_artifact_time:.4f}",
            f"{runtime_parts[0]:.4f}",f"{runtime_parts[1]:.4f}",f"{runtime_parts[2]:.4f}",f"{runtime_parts[3]:.4f}",
            f"{runtime:.4f}", f"{artifact_time:.4f}", f"{speedup:.4f}",
            int(self.correlated_bf), int(self.correlated_w),
            int(self.tested_bf), int(self.tested_w),
            int(self.total_bf), int(self.total_w),
            f"{metrics['precision_pos']:.4f}",f"{metrics['recall_pos']:.4f}",f"{metrics['f1_score_pos']:.4f}",
            f"{metrics['precision_neg']:.4f}",f"{metrics['recall_neg']:.4f}",f"{metrics['f1_score_neg']:.4f}",
            f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['recall_min']:.4f}",
            f"{metrics['f1_score']:.4f}", f"{metrics['aucroc']:.4f}",
            f"{metrics['pr_auc']:.4f}"
        ] + [
            f"{maxlag_precision:.4f}",
            f"{maxlag_recall:.4f}",
            f"{maxlag_f1:.4f}",
            _format_float(maxlag_diff_mean),
            _format_float(maxlag_diff_std),
        ]

    def plot(self,status_csv):
        path = os.path.dirname(os.path.abspath(status_csv))        
        self._plot_top_durations(status_csv, self.test_data, path)
        
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

        runtime_parts, runtime_bf, artifact_time_bf, corr_flags_bf = self._mode_run("bf", None, path, "bf", 0, None, None, 1, 1, None, None, None, None, None, self.extra_filter)
        print("Run Brute-Force, Finished in ",runtime_bf)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        args_list = [
            (dataset_id, "main", alg, path, "main_" + alg + "_recall_speedup",
             bst[alg], runtime_bf, corr_flags_bf,
             self._slice_warmup(self.train_data, bst[alg].get("warmup_size")))
            for alg in self.algs
        ]

        results = list(self._outer_iter(args_list, self._parallel_mode_run, unordered=False))

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
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
#DONE: warmup within
#DONE: incremental update of grids
#ARCHIVED: test stationarity for performing 2nd order diff

#TODO: weighting based on probability of sketches 
#TODO: observe super populated cell grids
#TODO: get subcells of super populated cell grids

#TODO: forgetting mechanism
