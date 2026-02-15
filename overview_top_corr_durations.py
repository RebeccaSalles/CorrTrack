"""CLI for plotting top correlated durations from CorrTrack run outputs."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library_corrtrack_parallel import _detect_csv_delimiter


def _load_module(config_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


DEFAULT_EXEC_PARAM_CONFIG = Path(__file__).with_name("experiment_run_exec_param.py")
_DEFAULT_EXEC_CFG = _load_module(DEFAULT_EXEC_PARAM_CONFIG, "experiment_exec_defaults")

DEFAULT_WINDOW_SIZE = getattr(_DEFAULT_EXEC_CFG, "WINDOW_SIZE", 7 * 24)
DEFAULT_WINDOW_STEP = getattr(_DEFAULT_EXEC_CFG, "WINDOW_STEP", 12)
DEFAULT_BASIC_WINDOW = getattr(_DEFAULT_EXEC_CFG, "BASIC_WINDOW", None)
DEFAULT_N_LAGS = getattr(_DEFAULT_EXEC_CFG, "N_LAGS", 7 * 24)
DEFAULT_CORR_THRESHOLD = getattr(_DEFAULT_EXEC_CFG, "CORR_THRESHOLD", 0.7)

DEFAULT_PARALLEL_SKETCH = getattr(_DEFAULT_EXEC_CFG, "PARALLEL_SKETCH", False)
DEFAULT_PARALLEL_CANDIDATES = getattr(_DEFAULT_EXEC_CFG, "PARALLEL_CANDIDATES", False)
DEFAULT_PARALLEL_VALIDATION = getattr(_DEFAULT_EXEC_CFG, "PARALLEL_VALIDATION", None)
DEFAULT_PARALLEL = any(
    val is True for val in (DEFAULT_PARALLEL_SKETCH, DEFAULT_PARALLEL_CANDIDATES, DEFAULT_PARALLEL_VALIDATION)
)
DEFAULT_EXEC_MODE = "thread" if DEFAULT_PARALLEL else "sequential"
DEFAULT_NEG_CORR = getattr(_DEFAULT_EXEC_CFG, "NEG_CORR", False)
DEFAULT_VERBOSE = getattr(_DEFAULT_EXEC_CFG, "VERBOSE", False)
DEFAULT_TESTING = getattr(_DEFAULT_EXEC_CFG, "TESTING", False)
DEFAULT_RESULT_FOLDER = None
DEFAULT_MAX_WORKERS = getattr(_DEFAULT_EXEC_CFG, "MAX_WORKERS", 0)

DEFAULT_DATASET_CONFIG = Path(__file__).with_name("experiment_dataset_fr_air_temperature_7_1.py")
DEFAULT_OBS_MODE = "years"

WINDOW_SIZE = DEFAULT_WINDOW_SIZE
WINDOW_STEP = DEFAULT_WINDOW_STEP
BASIC_WINDOW = DEFAULT_BASIC_WINDOW
N_LAGS = DEFAULT_N_LAGS
CORR_THRESHOLD = DEFAULT_CORR_THRESHOLD
PARALLEL = DEFAULT_PARALLEL
PARALLEL_SKETCH = DEFAULT_PARALLEL_SKETCH
PARALLEL_CANDIDATES = DEFAULT_PARALLEL_CANDIDATES
PARALLEL_VALIDATION = DEFAULT_PARALLEL_VALIDATION
EXEC_MODE = DEFAULT_EXEC_MODE
NEG_CORR = DEFAULT_NEG_CORR
VERBOSE = DEFAULT_VERBOSE
TESTING = DEFAULT_TESTING
RESULT_FOLDER = DEFAULT_RESULT_FOLDER
MAX_WORKERS = DEFAULT_MAX_WORKERS
COUNTRIES = VARIABLES = N_VARS = N_YEARS = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
OBS_MODE = DEFAULT_OBS_MODE

MODES = ["corrtrack"]


def _coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _any_parallel(*values) -> bool:
    return any(val is True for val in values)


def _resolve_cfg_value(value, cfg, attr, default):
    if value is not None:
        return value
    if hasattr(cfg, attr):
        return getattr(cfg, attr)
    return default


def _apply_parallel_defaults_from_cfg(cfg, args):
    if getattr(args, "parallel_sketch", None) is None and hasattr(cfg, "PARALLEL_SKETCH"):
        args.parallel_sketch = _coerce_optional_bool(getattr(cfg, "PARALLEL_SKETCH"))
    if getattr(args, "parallel_candidates", None) is None and hasattr(cfg, "PARALLEL_CANDIDATES"):
        args.parallel_candidates = _coerce_optional_bool(getattr(cfg, "PARALLEL_CANDIDATES"))
    if getattr(args, "parallel_validation", None) is None and hasattr(cfg, "PARALLEL_VALIDATION"):
        args.parallel_validation = _coerce_optional_bool(getattr(cfg, "PARALLEL_VALIDATION"))


def _get_cfg_attr(cfg, *names):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    raise AttributeError(f"Dataset configuration must define one of: {', '.join(names)}")


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _dataset_slug(country: str, var: str | None) -> str:
    if var is None or var == "" or var == country:
        return country
    return f"{country}_{var}"


def _effective_variable(country: str, var: str | None) -> str:
    return var if var not in (None, "") else country


def _apply_dataset_config(cfg):
    global COUNTRIES, VARIABLES, N_VARS, N_YEARS, DATA_LOADER, OBS_MODE

    COUNTRIES = _as_list(_get_cfg_attr(cfg, "COUNTRIES", "DATASET"))
    variables_attr = getattr(cfg, "VARIABLES", None)
    VARIABLES = _as_list(variables_attr) if variables_attr is not None else [None]
    N_VARS = _get_cfg_attr(cfg, "N_VARS", "N_SERIES")
    N_YEARS = _get_cfg_attr(cfg, "N_YEARS", "N_OBS")
    DATA_LOADER = getattr(cfg, "DATA_LOADER", None)
    OBS_MODE = getattr(cfg, "OBS_MODE", DEFAULT_OBS_MODE)


def _load_loader(loader_spec: str) -> Callable[[str, str], tuple[np.ndarray, np.ndarray]]:
    if ":" in loader_spec:
        module_name, attr = loader_spec.split(":", 1)
    else:
        module_name, attr = loader_spec, "load_dataset"
    module = importlib.import_module(module_name)
    loader = getattr(module, attr)
    return loader


def iter_datasets():
    if DATA_LOADER is None:
        raise RuntimeError("Dataset configuration must define DATA_LOADER")
    for var in VARIABLES:
        for country in COUNTRIES:
            var_key = _effective_variable(country, var)
            data, ids = DATA_LOADER(country, var_key)
            yield country, var, data, ids


def _select_rows(total_rows: int, span: int) -> np.ndarray:
    if OBS_MODE == "count":
        limit = max(1, min(int(span), total_rows))
        return np.arange(limit)

    one_year = 365 * 24
    start = max(0, total_rows - int(span) * one_year)
    rows = np.arange(start, total_rows)
    if start > 0:
        rows = np.r_[0, rows]
    return rows


def prepare_test_data(data, ids, n_year, n_var):
    rows = _select_rows(data.shape[0], n_year)
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    test_data = np.transpose(data_stream)
    ids_n_var = ids[: data_stream.shape[1] - 1]
    return test_data, ids_n_var


def config_folder():
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    return f"ws{WINDOW_SIZE}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"


def _load_duration_csv(path: str) -> pd.DataFrame:
    delimiter = _detect_csv_delimiter(path, default=",")
    df = pd.read_csv(path, sep=delimiter)
    if "start_time_id1" not in df.columns or "start_time_id2" not in df.columns:
        raise KeyError(
            f"CSV missing start_time_id1/start_time_id2 columns: {path}. "
            "Expected a *_status.csv artifact (monitor status output)."
        )
    return df


def _infer_time_mode(times_raw: np.ndarray) -> tuple[str, np.ndarray]:
    if np.issubdtype(times_raw.dtype, np.number):
        numeric = pd.to_numeric(times_raw, errors="coerce")
        if hasattr(numeric, "to_numpy"):
            numeric = numeric.to_numpy()
        else:
            numeric = np.asarray(numeric)
        if np.isnan(numeric).all():
            numeric = np.zeros_like(times_raw, dtype=float)
        return "numeric", numeric

    if np.issubdtype(times_raw.dtype, np.datetime64):
        return "datetime", times_raw.astype("datetime64[ns]")

    parsed = pd.to_datetime(times_raw, errors="coerce")
    if getattr(parsed.notna(), "mean", lambda: 0.0)() >= 0.8:
        if hasattr(parsed, "to_numpy"):
            return "datetime", parsed.to_numpy(dtype="datetime64[ns]")
        return "datetime", np.asarray(parsed, dtype="datetime64[ns]")

    numeric = pd.to_numeric(times_raw, errors="coerce")
    if hasattr(numeric, "to_numpy"):
        numeric = numeric.to_numpy()
    else:
        numeric = np.asarray(numeric)
    if np.isnan(numeric).all():
        numeric = np.zeros_like(times_raw, dtype=float)
    return "numeric", numeric


def _coerce_time_values(values, mode: str) -> np.ndarray:
    if mode == "datetime":
        parsed = pd.to_datetime(values, errors="coerce")
        if hasattr(parsed, "to_numpy"):
            return parsed.to_numpy(dtype="datetime64[ns]")
        return np.asarray(parsed, dtype="datetime64[ns]")
    numeric = pd.to_numeric(values, errors="coerce")
    if hasattr(numeric, "to_numpy"):
        numeric = numeric.to_numpy()
    else:
        numeric = np.asarray(numeric)
    if np.isnan(numeric).all():
        return np.zeros_like(values, dtype=float)
    return numeric


def _find_time_index(times: np.ndarray, target) -> int:
    matches = np.where(times == target)[0]
    if matches.size:
        return int(matches[0])

    try:
        if np.issubdtype(times.dtype, np.datetime64):
            times_ns = times.astype("datetime64[ns]").astype("int64")
            target_ns = np.asarray(target).astype("datetime64[ns]").astype("int64")
        else:
            times_ns = times.astype(float)
            target_ns = float(target)
        idx = int(np.searchsorted(times_ns, target_ns))
        idx = max(0, min(idx, len(times_ns) - 1))
        return idx
    except Exception:
        return 0


def _pick_datetime_unit(range_ns: int) -> str:
    if range_ns < 1_000:
        return "ns"
    if range_ns < 1_000_000:
        return "us"
    if range_ns < 1_000_000_000:
        return "ms"
    if range_ns < 3_600_000_000_000:
        return "s"
    return "h"


def _format_time_label(value, mode: str, unit: str | None = None, range_ns: int | None = None) -> str:
    if mode == "datetime":
        try:
            if unit is None and range_ns is not None:
                unit = _pick_datetime_unit(range_ns)
            if unit is None:
                unit = "s"
            return str(np.datetime_as_string(value, unit=unit))
        except Exception:
            try:
                return str(np.asarray(value).astype("datetime64[ns]").astype("int64"))
            except Exception:
                return str(value)
    try:
        if isinstance(value, (np.floating, float)) and np.isnan(value):
            return "nan"
        hours = int(float(value))
        base = np.datetime64("1970-01-01T00:00:00")
        dt = base + np.timedelta64(hours, "h")
        return str(np.datetime_as_string(dt, unit="h"))
    except Exception:
        return str(value)


def _apply_informative_xticks(ax, time_window, start_time, end_time, mode: str):
    if time_window is None or len(time_window) == 0:
        return
    ticks = [time_window[0], start_time, end_time, time_window[-1]]
    # Deduplicate while preserving order.
    seen = set()
    unique_ticks = []
    for t in ticks:
        key = t
        if mode == "datetime":
            try:
                key = np.datetime64(t, "ns")
            except Exception:
                key = t
        if key in seen:
            continue
        seen.add(key)
        unique_ticks.append(t)
    ax.set_xticks(unique_ticks)
    range_ns = None
    unit = None
    if mode == "datetime":
        try:
            times_ns = np.asarray(time_window).astype("datetime64[ns]").astype("int64")
            if len(times_ns) >= 2:
                range_ns = int(times_ns[-1] - times_ns[0])
                unit = _pick_datetime_unit(range_ns)
        except Exception:
            pass
    ax.set_xticklabels(
        [_format_time_label(t, mode, unit=unit, range_ns=range_ns) for t in unique_ticks],
        rotation=30,
        ha="right",
    )


def plot_top_durations(corr_csv, data, ids, window_size, output_folder, *, exclude_autocorr=False, unique_pairs=False):
    os.makedirs(output_folder, exist_ok=True)

    df_corr = _load_duration_csv(corr_csv)
    df_corr = df_corr.copy()
    df_corr["duration_val"] = pd.to_numeric(df_corr.get("duration"), errors="coerce")
    df_corr = df_corr.dropna(subset=["duration_val"])

    if exclude_autocorr and "id1" in df_corr.columns and "id2" in df_corr.columns:
        df_corr = df_corr[df_corr["id1"] != df_corr["id2"]]
    if unique_pairs:
        subset_cols = [c for c in ("id1", "id2", "start_time_id1", "start_time_id2", "lag") if c in df_corr.columns]
        if subset_cols:
            df_corr = df_corr.drop_duplicates(subset=subset_cols)

    time_mode, times = _infer_time_mode(np.asarray(data[0, :]))
    data_vals = data[1:, :].astype(float)
    ids_arr = np.asarray(ids)

    sorted_rows = df_corr.sort_values(by="duration_val", ascending=False)
    start_times1 = _coerce_time_values(sorted_rows["start_time_id1"], time_mode)
    start_times2 = _coerce_time_values(sorted_rows["start_time_id2"], time_mode)

    plotted = 0
    target_plots = 5
    for (row_idx, row), start_time_id1, start_time_id2 in zip(sorted_rows.iterrows(), start_times1, start_times2):
        id1, id2 = row["id1"], row["id2"]
        duration = row.get("duration_val", 0)
        try:
            duration = int(float(duration))
        except Exception:
            duration = 0
        if duration <= 0:
            continue
        lag = row["lag"]
        start_time = min(start_time_id1, start_time_id2)
        max_start_time = max(start_time_id1, start_time_id2)

        _ = _find_time_index(times, start_time)
        _ = _find_time_index(times, max_start_time)
        start_idx_id1 = _find_time_index(times, start_time_id1)
        start_idx_id2 = _find_time_index(times, start_time_id2)

        end_idx_id1 = min(len(times) - 1, start_idx_id1 + duration)
        end_idx_id2 = min(len(times) - 1, start_idx_id2 + duration)
        end_time_id1 = times[end_idx_id1]
        end_time_id2 = times[end_idx_id2]

        idx1 = np.where(ids_arr == id1)[0][0]
        idx2 = np.where(ids_arr == id2)[0][0]

        half_window = window_size
        start_range_id1 = max(0, start_idx_id1 - half_window)
        end_range_id1 = min(len(times), end_idx_id1 + half_window)
        start_range_id2 = max(0, start_idx_id2 - half_window)
        end_range_id2 = min(len(times), end_idx_id2 + half_window)

        # If we clipped on one side, try to extend the other side so the lines
        # are not glued to the plot boundary when there is room.
        if start_range_id1 == 0:
            pad = max(0, half_window - start_idx_id1)
            end_range_id1 = min(len(times), end_range_id1 + pad)
        if end_range_id1 == len(times):
            pad = max(0, half_window - (len(times) - 1 - end_idx_id1))
            start_range_id1 = max(0, start_range_id1 - pad)
        if start_range_id2 == 0:
            pad = max(0, half_window - start_idx_id2)
            end_range_id2 = min(len(times), end_range_id2 + pad)
        if end_range_id2 == len(times):
            pad = max(0, half_window - (len(times) - 1 - end_idx_id2))
            start_range_id2 = max(0, start_range_id2 - pad)

        # Fallback to a window around the start index if we ended up with empty slices.
        if start_range_id1 >= end_range_id1:
            start_range_id1 = max(0, start_idx_id1 - half_window)
            end_range_id1 = min(len(times), start_idx_id1 + half_window)
        if start_range_id2 >= end_range_id2:
            start_range_id2 = max(0, start_idx_id2 - half_window)
            end_range_id2 = min(len(times), start_idx_id2 + half_window)

        time_window_id1 = times[start_range_id1:end_range_id1]
        time_window_id2 = times[start_range_id2:end_range_id2]

        series1 = data_vals[idx1, start_range_id1:end_range_id1]
        series2 = data_vals[idx2, start_range_id2:end_range_id2]

        if series1.size == 0:
            start_range_id1 = max(0, min(start_idx_id1, len(times) - 1))
            end_range_id1 = min(len(times), start_range_id1 + 1)
            time_window_id1 = times[start_range_id1:end_range_id1]
            series1 = data_vals[idx1, start_range_id1:end_range_id1]
        if series2.size == 0:
            start_range_id2 = max(0, min(start_idx_id2, len(times) - 1))
            end_range_id2 = min(len(times), start_range_id2 + 1)
            time_window_id2 = times[start_range_id2:end_range_id2]
            series2 = data_vals[idx2, start_range_id2:end_range_id2]
        if series1.size == 0 or series2.size == 0:
            continue

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

        axs[0].plot(time_window_id1, series1, label=id1, color="blue")
        axs[0].axvline(x=start_time_id1, color="red", linestyle="--", label="Start Time")
        axs[0].axvline(
            x=end_time_id1,
            color="orange",
            linestyle="--",
            label="End Time",
        )
        axs[0].set_ylabel(id1)
        _apply_informative_xticks(axs[0], time_window_id1, start_time_id1, end_time_id1, time_mode)
        axs[0].legend(loc="upper left")
        axs[0].grid(True)

        axs[1].plot(time_window_id2, series2, label=id2, color="green")
        axs[1].axvline(x=start_time_id2, color="red", linestyle="--", label="Start Time")
        axs[1].axvline(
            x=end_time_id2,
            color="orange",
            linestyle="--",
            label="End Time",
        )
        axs[1].set_ylabel(id2)
        _apply_informative_xticks(axs[1], time_window_id2, start_time_id2, end_time_id2, time_mode)
        axs[1].legend(loc="upper left")
        axs[1].grid(True)

        plt.xlabel("Time (independent axes)")
        plt.tight_layout()
        filename = (
            f"{id1.replace('/', '_')}_{id2.replace('/', '_')}_{lag}_"
            f"{start_idx_id1}_{start_idx_id2}_d{duration}_r{row_idx}.png"
        )
        output_path = os.path.join(output_folder, "plots", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        plotted += 1
        if plotted >= target_plots:
            break


def main():
    parser = argparse.ArgumentParser(description="Plot top correlated durations from CorrTrack run outputs.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to dataset configuration module.",
    )
    parser.add_argument(
        "--exec-param-config",
        type=Path,
        default=DEFAULT_EXEC_PARAM_CONFIG,
        help="Path to execution parameter configuration module.",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Override RESULT_FOLDER from the dataset config.",
    )
    parser.add_argument(
        "--loader",
        type=str,
        default=None,
        help="Python path to dataset loader function (module:callable). Overrides config DATA_LOADER.",
    )
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--basic-window", type=int, default=None)
    parser.add_argument("--n-lags", type=int, default=None)
    parser.add_argument("--corr-threshold", type=float, default=None)
    parser.add_argument("--parallel", dest="parallel", action="store_true")
    parser.add_argument("--sequential", dest="parallel", action="store_false")
    parser.add_argument("--parallel-sketch", dest="parallel_sketch", action="store_true")
    parser.add_argument("--sequential-sketch", dest="parallel_sketch", action="store_false")
    parser.add_argument("--parallel-candidates", dest="parallel_candidates", action="store_true")
    parser.add_argument("--sequential-candidates", dest="parallel_candidates", action="store_false")
    parser.add_argument("--parallel-validation", dest="parallel_validation", action="store_true")
    parser.add_argument("--sequential-validation", dest="parallel_validation", action="store_false")
    parser.add_argument("--neg-corr", dest="neg_corr", action="store_true")
    parser.add_argument("--no-neg-corr", dest="neg_corr", action="store_false")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--testing", dest="testing", action="store_true")
    parser.add_argument("--no-testing", dest="testing", action="store_false")
    parser.add_argument(
        "--exclude-autocorr",
        action="store_true",
        help="Exclude autocorrelation rows (id1 == id2) when selecting top durations.",
    )
    parser.add_argument(
        "--unique-pairs",
        action="store_true",
        help="Drop duplicate (id1,id2,start_time_id1,start_time_id2,lag) rows before selecting top durations.",
    )
    parser.set_defaults(
        parallel=None,
        parallel_sketch=None,
        parallel_candidates=None,
        parallel_validation=None,
        neg_corr=None,
        verbose=None,
        testing=None,
    )
    args = parser.parse_args()

    cfg_exec = _load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = _load_module(args.dataset_config, "experiment_dataset")
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_exec, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD, RESULT_FOLDER
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, DATA_LOADER, MAX_WORKERS
    global VERBOSE, TESTING

    RESULT_FOLDER = _resolve_cfg_value(args.result_folder, cfg_dataset, "RESULT_FOLDER", DEFAULT_RESULT_FOLDER)
    WINDOW_SIZE = _resolve_cfg_value(args.window_size, cfg_exec, "WINDOW_SIZE", DEFAULT_WINDOW_SIZE)
    WINDOW_STEP = _resolve_cfg_value(args.window_step, cfg_exec, "WINDOW_STEP", DEFAULT_WINDOW_STEP)
    BASIC_WINDOW = _resolve_cfg_value(args.basic_window, cfg_exec, "BASIC_WINDOW", DEFAULT_BASIC_WINDOW)
    N_LAGS = _resolve_cfg_value(args.n_lags, cfg_exec, "N_LAGS", DEFAULT_N_LAGS)
    CORR_THRESHOLD = _resolve_cfg_value(args.corr_threshold, cfg_exec, "CORR_THRESHOLD", DEFAULT_CORR_THRESHOLD)
    PARALLEL = args.parallel
    PARALLEL_SKETCH = _resolve_cfg_value(args.parallel_sketch, cfg_exec, "PARALLEL_SKETCH", DEFAULT_PARALLEL_SKETCH)
    PARALLEL_CANDIDATES = _resolve_cfg_value(
        args.parallel_candidates, cfg_exec, "PARALLEL_CANDIDATES", DEFAULT_PARALLEL_CANDIDATES
    )
    PARALLEL_VALIDATION = _resolve_cfg_value(
        args.parallel_validation, cfg_exec, "PARALLEL_VALIDATION", DEFAULT_PARALLEL_VALIDATION
    )
    if PARALLEL is None:
        PARALLEL = _any_parallel(PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION)
    EXEC_MODE = "thread" if _any_parallel(PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION) else "sequential"
    NEG_CORR = _resolve_cfg_value(args.neg_corr, cfg_exec, "NEG_CORR", DEFAULT_NEG_CORR)
    MAX_WORKERS = _resolve_cfg_value(None, cfg_exec, "MAX_WORKERS", DEFAULT_MAX_WORKERS)
    VERBOSE = _resolve_cfg_value(args.verbose, cfg_exec, "VERBOSE", DEFAULT_VERBOSE)
    TESTING = _resolve_cfg_value(args.testing, cfg_exec, "TESTING", DEFAULT_TESTING)
    if args.loader:
        DATA_LOADER = _load_loader(args.loader)
    if DATA_LOADER is None:
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")
    if RESULT_FOLDER is None:
        raise RuntimeError("Dataset config must define RESULT_FOLDER or provide --result-folder.")

    for country, var, data, ids in iter_datasets():
        for n_year in N_YEARS:
            for n_var in N_VARS:
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                test_data, ids_n_var = prepare_test_data(data, ids, n_year, n_var)

                base_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder())
                for alg in MODES:
                    status_csv = os.path.join(base_dir, f"main_{alg}_status.csv")
                    if not os.path.exists(status_csv):
                        raise FileNotFoundError(
                            f"Status CSV not found: {status_csv}. "
                            "Run corrtrack with artifact logging to generate *_status.csv."
                        )
                    output_dir = os.path.join(base_dir, "top_corr_durations", alg)
                    plot_top_durations(
                        status_csv,
                        test_data,
                        ids_n_var,
                        WINDOW_SIZE,
                        output_dir,
                        exclude_autocorr=args.exclude_autocorr,
                        unique_pairs=args.unique_pairs,
                    )
                    print(f"[INFO] Plots saved under: {output_dir}")


if __name__ == "__main__":
    main()
