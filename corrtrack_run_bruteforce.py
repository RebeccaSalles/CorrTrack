import os
import argparse
import importlib
import importlib.util
import numpy as np
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import run_and_log_bruteforce


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
DEFAULT_MONITOR = getattr(_DEFAULT_EXEC_CFG, "MONITOR", True)
DEFAULT_TRACK_MIN_DIST = getattr(_DEFAULT_EXEC_CFG, "TRACK_MIN_DIST", True)
DEFAULT_RECALL_BY_WINDOW = getattr(_DEFAULT_EXEC_CFG, "RECALL_BY_WINDOW", True)
DEFAULT_ARTIFACT_MODE = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_MODE", "iterative")
DEFAULT_ARTIFACT_BUFFER_MAX_ROWS = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_BUFFER_MAX_ROWS", 250000)
DEFAULT_SAVE_ONLY_REQUIRED_ARTIFACTS = getattr(_DEFAULT_EXEC_CFG, "SAVE_ONLY_REQUIRED_ARTIFACTS", False)
DEFAULT_SAVE_MAXLAG_ARTIFACTS = getattr(_DEFAULT_EXEC_CFG, "SAVE_MAXLAG_ARTIFACTS", True)
DEFAULT_VERBOSE = getattr(_DEFAULT_EXEC_CFG, "VERBOSE", False)
DEFAULT_TESTING = getattr(_DEFAULT_EXEC_CFG, "TESTING", False)
DEFAULT_RESULT_FOLDER = None
DEFAULT_MAX_WORKERS = getattr(_DEFAULT_EXEC_CFG, "MAX_WORKERS", 0)

DEFAULT_DATASET_CONFIG = Path(__file__).with_name(
    "experiment_dataset_fr_air_temperature_7_1.py"
)
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
MONITOR = DEFAULT_MONITOR
TRACK_MIN_DIST = DEFAULT_TRACK_MIN_DIST
RECALL_BY_WINDOW = DEFAULT_RECALL_BY_WINDOW
ARTIFACT_MODE = DEFAULT_ARTIFACT_MODE
ARTIFACT_BUFFER_MAX_ROWS = DEFAULT_ARTIFACT_BUFFER_MAX_ROWS
SAVE_ONLY_REQUIRED_ARTIFACTS = DEFAULT_SAVE_ONLY_REQUIRED_ARTIFACTS
SAVE_MAXLAG_ARTIFACTS = DEFAULT_SAVE_MAXLAG_ARTIFACTS
VERBOSE = DEFAULT_VERBOSE
TESTING = DEFAULT_TESTING
RESULT_FOLDER = DEFAULT_RESULT_FOLDER
MAX_WORKERS = DEFAULT_MAX_WORKERS
COUNTRIES = VARIABLES = N_VARS = N_YEARS = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
OBS_MODE = DEFAULT_OBS_MODE


def _load_dataset_config(config_path: Path):
    return _load_module(config_path, "experiment_dataset")


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


def build_base_config():
    return {
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "basic_window": BASIC_WINDOW,
        "n_lags": N_LAGS,
        "corr_threshold": CORR_THRESHOLD,
        "neg_corr": NEG_CORR,
        "exec": EXEC_MODE,
        "parallel_sketch": PARALLEL_SKETCH,
        "parallel_candidates": PARALLEL_CANDIDATES,
        "parallel_validation": PARALLEL_VALIDATION,
        "max_workers": MAX_WORKERS,
        "monitor": MONITOR,
        "track_min_dist": TRACK_MIN_DIST,
        "artifact_mode": ARTIFACT_MODE,
        "artifact_buffer_max_rows": ARTIFACT_BUFFER_MAX_ROWS,
        "save_only_required_artifacts": SAVE_ONLY_REQUIRED_ARTIFACTS,
        "save_maxlag_artifacts": SAVE_MAXLAG_ARTIFACTS,
        "verbose": VERBOSE,
        "testing": TESTING,
    }


def iter_datasets():
    if DATA_LOADER is None:
        raise RuntimeError("Dataset config must define DATA_LOADER")
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


def prepare_stream(data, ids, n_year, n_var):
    rows = _select_rows(data.shape[0], n_year)
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    ids_n_var = ids[: data_stream.shape[1] - 1]
    return data_stream, ids_n_var


def config_folder():
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    return f"ws{WINDOW_SIZE}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run CorrTrack brute-force baseline.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to the dataset configuration module.",
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
    parser.add_argument("--monitor", dest="monitor", action="store_true")
    parser.add_argument("--no-monitor", dest="monitor", action="store_false")
    parser.add_argument("--track-min-dist", dest="track_min_dist", action="store_true")
    parser.add_argument("--no-track-min-dist", dest="track_min_dist", action="store_false")
    parser.add_argument("--recall-by-window", dest="recall_by_window", action="store_true")
    parser.add_argument(
        "--artifact-mode",
        choices=("iterative", "final", "buffered"),
        default=None,
        help="Persist artifacts after each iteration (iterative), only once after the run (final), or spill in bounded chunks (buffered).",
    )
    parser.add_argument("--artifact-buffer-max-rows", type=int, default=None)
    parser.add_argument(
        "--save-only-required-artifacts",
        dest="save_only_required_artifacts",
        action="store_true",
    )
    parser.add_argument(
        "--save-all-artifacts",
        dest="save_only_required_artifacts",
        action="store_false",
    )
    parser.add_argument(
        "--save-maxlag-artifacts",
        dest="save_maxlag_artifacts",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-maxlag-artifacts",
        dest="save_maxlag_artifacts",
        action="store_false",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--testing", dest="testing", action="store_true")
    parser.add_argument("--no-testing", dest="testing", action="store_false")
    parser.add_argument(
        "--no-recall-by-window",
        dest="recall_by_window",
        action="store_false",
        help="Disable recall-by-window mode (enabled by default).",
    )
    parser.set_defaults(
        parallel=None,
        parallel_sketch=None,
        parallel_candidates=None,
        parallel_validation=None,
        neg_corr=None,
        monitor=None,
        track_min_dist=None,
        recall_by_window=None,
        artifact_mode=None,
        artifact_buffer_max_rows=None,
        save_only_required_artifacts=None,
        save_maxlag_artifacts=None,
        verbose=None,
        testing=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_exec = _load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = _load_dataset_config(args.dataset_config)
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_exec, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, MONITOR, TRACK_MIN_DIST, RECALL_BY_WINDOW, ARTIFACT_MODE, ARTIFACT_BUFFER_MAX_ROWS, SAVE_ONLY_REQUIRED_ARTIFACTS, SAVE_MAXLAG_ARTIFACTS
    global DATA_LOADER, RESULT_FOLDER, MAX_WORKERS, VERBOSE, TESTING

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
    MONITOR = _resolve_cfg_value(args.monitor, cfg_exec, "MONITOR", DEFAULT_MONITOR)
    TRACK_MIN_DIST = _resolve_cfg_value(
        args.track_min_dist, cfg_exec, "TRACK_MIN_DIST", DEFAULT_TRACK_MIN_DIST
    )
    RECALL_BY_WINDOW = _resolve_cfg_value(
        args.recall_by_window, cfg_exec, "RECALL_BY_WINDOW", DEFAULT_RECALL_BY_WINDOW
    )
    ARTIFACT_MODE = _resolve_cfg_value(args.artifact_mode, cfg_exec, "ARTIFACT_MODE", DEFAULT_ARTIFACT_MODE)
    ARTIFACT_BUFFER_MAX_ROWS = _resolve_cfg_value(
        args.artifact_buffer_max_rows, cfg_exec, "ARTIFACT_BUFFER_MAX_ROWS", DEFAULT_ARTIFACT_BUFFER_MAX_ROWS
    )
    SAVE_ONLY_REQUIRED_ARTIFACTS = _resolve_cfg_value(
        args.save_only_required_artifacts,
        cfg_exec,
        "SAVE_ONLY_REQUIRED_ARTIFACTS",
        DEFAULT_SAVE_ONLY_REQUIRED_ARTIFACTS,
    )
    SAVE_MAXLAG_ARTIFACTS = _resolve_cfg_value(
        args.save_maxlag_artifacts,
        cfg_exec,
        "SAVE_MAXLAG_ARTIFACTS",
        DEFAULT_SAVE_MAXLAG_ARTIFACTS,
    )
    MAX_WORKERS = _resolve_cfg_value(None, cfg_exec, "MAX_WORKERS", DEFAULT_MAX_WORKERS)
    VERBOSE = _resolve_cfg_value(args.verbose, cfg_exec, "VERBOSE", DEFAULT_VERBOSE)
    TESTING = _resolve_cfg_value(args.testing, cfg_exec, "TESTING", DEFAULT_TESTING)

    if args.loader:
        DATA_LOADER = _load_loader(args.loader)
    if DATA_LOADER is None:
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")
    if RESULT_FOLDER is None:
        raise RuntimeError("Dataset config must define RESULT_FOLDER or provide --result-folder.")

    base_config = build_base_config()

    for country, var, data, ids in iter_datasets():
        for n_year in N_YEARS:
            for n_var in N_VARS:
                data_stream, ids_n_var = prepare_stream(data, ids, n_year, n_var)
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                output_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder())
                os.makedirs(output_dir, exist_ok=True)

                metadata = {
                    "alg": "bf",
                    "mode": "bf",
                    "optim": "bf_baseline",
                    "nodes": base_config.get("max_workers"),
                }

                test_data = np.transpose(data_stream)
                output_csv = os.path.join(output_dir, "bf_run.csv")

                run_and_log_bruteforce(
                    dataset_id,
                    test_data,
                    ids_n_var,
                    base_config,
                    output_csv,
                    metadata=metadata,
                    recall_by_window=RECALL_BY_WINDOW,
                    verbose=VERBOSE,
                    testing=TESTING,
                    artifact_prefix=os.path.join(output_dir, "bf"),
                )


if __name__ == "__main__":
    main()
