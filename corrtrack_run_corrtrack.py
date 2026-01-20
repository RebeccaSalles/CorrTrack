import argparse
import importlib
import importlib.util
import json
import os
import numpy as np
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import run_and_log_corrtrack

DEFAULT_WINDOW_SIZE = 7 * 24
DEFAULT_WINDOW_STEP = 12
DEFAULT_BASIC_WINDOW = None
DEFAULT_N_LAGS = 7 * 24
DEFAULT_CORR_THRESHOLD = 0.7

DEFAULT_PARALLEL = False
DEFAULT_PARALLEL_SKETCH = False
DEFAULT_PARALLEL_CANDIDATES = False
DEFAULT_PARALLEL_VALIDATION = None
DEFAULT_EXEC_MODE = "thread" if DEFAULT_PARALLEL else "sequential"
DEFAULT_NEG_CORR = False
DEFAULT_CORR_VAL = True
DEFAULT_EXTRA_FILTER = False
DEFAULT_RECALL_BY_WINDOW = True
DEFAULT_ARTIFACT_MODE = "iterative"

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
CORR_VAL = DEFAULT_CORR_VAL
EXTRA_FILTER = DEFAULT_EXTRA_FILTER
RECALL_BY_WINDOW = DEFAULT_RECALL_BY_WINDOW
ARTIFACT_MODE = DEFAULT_ARTIFACT_MODE
RESULT_FOLDER = None
COUNTRIES = VARIABLES = N_VARS = N_YEARS = MODES = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
OBS_MODE = DEFAULT_OBS_MODE


def _load_dataset_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("experiment_dataset", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("experiment_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _any_parallel(*values) -> bool:
    return any(val is True for val in values)


def _apply_parallel_defaults_from_cfg(cfg, args):
    if getattr(args, "parallel_sketch", None) is None and hasattr(cfg, "PARALLEL_SKETCH"):
        args.parallel_sketch = _coerce_optional_bool(getattr(cfg, "PARALLEL_SKETCH"))
    if getattr(args, "parallel_candidates", None) is None and hasattr(cfg, "PARALLEL_CANDIDATES"):
        args.parallel_candidates = _coerce_optional_bool(getattr(cfg, "PARALLEL_CANDIDATES"))
    if getattr(args, "parallel_validation", None) is None and hasattr(cfg, "PARALLEL_VALIDATION"):
        args.parallel_validation = _coerce_optional_bool(getattr(cfg, "PARALLEL_VALIDATION"))


def _resolve_parallel(cfg) -> bool:
    if hasattr(cfg, "PARALLEL"):
        return bool(getattr(cfg, "PARALLEL"))
    if hasattr(cfg, "EXEC_MODE"):
        mode = str(getattr(cfg, "EXEC_MODE", "")).strip().lower()
        return mode not in {"sequential", "seq", "false", "0"}
    return DEFAULT_PARALLEL


def _apply_run_config(cfg):
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, CORR_VAL, EXTRA_FILTER, RECALL_BY_WINDOW
    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD

    PARALLEL = _resolve_parallel(cfg)
    PARALLEL_SKETCH = _coerce_optional_bool(getattr(cfg, "PARALLEL_SKETCH", None))
    PARALLEL_CANDIDATES = _coerce_optional_bool(getattr(cfg, "PARALLEL_CANDIDATES", None))
    PARALLEL_VALIDATION = _coerce_optional_bool(getattr(cfg, "PARALLEL_VALIDATION", None))
    EXEC_MODE = "thread" if _any_parallel(PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION) else "sequential"
    NEG_CORR = cfg.NEG_CORR
    CORR_VAL = cfg.CORR_VAL
    EXTRA_FILTER = cfg.EXTRA_FILTER
    RECALL_BY_WINDOW = cfg.RECALL_BY_WINDOW
    WINDOW_SIZE = cfg.WINDOW_SIZE
    WINDOW_STEP = cfg.WINDOW_STEP
    BASIC_WINDOW = getattr(cfg, "BASIC_WINDOW", None)
    N_LAGS = cfg.N_LAGS
    CORR_THRESHOLD = cfg.CORR_THRESHOLD


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
    global RESULT_FOLDER, COUNTRIES, VARIABLES, N_VARS, N_YEARS, MODES, DATA_LOADER, OBS_MODE

    RESULT_FOLDER = cfg.RESULT_FOLDER
    COUNTRIES = _as_list(_get_cfg_attr(cfg, "COUNTRIES", "DATASET"))
    variables_attr = getattr(cfg, "VARIABLES", None)
    VARIABLES = _as_list(variables_attr) if variables_attr is not None else [None]
    N_VARS = _get_cfg_attr(cfg, "N_VARS", "N_SERIES")
    N_YEARS = _get_cfg_attr(cfg, "N_YEARS", "N_OBS")
    MODES = cfg.MODES
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


def load_best_params(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


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


def build_base_config():
    return {
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "basic_window": BASIC_WINDOW,
        "n_lags": N_LAGS,
        "corr_threshold": CORR_THRESHOLD,
        "neg_corr": NEG_CORR,
        "extra_filter": EXTRA_FILTER,
        "exec": EXEC_MODE,
        "parallel_sketch": PARALLEL_SKETCH,
        "parallel_candidates": PARALLEL_CANDIDATES,
        "parallel_validation": PARALLEL_VALIDATION,
        "max_workers": 0,
        "artifact_mode": ARTIFACT_MODE,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CorrTrack main algorithm with selected params.")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to dataset configuration module.",
    )
    parser.add_argument(
        "--loader",
        type=str,
        default=None,
        help="Python path to dataset loader function (module:callable). Overrides config DATA_LOADER.",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--window-step", type=int, default=DEFAULT_WINDOW_STEP)
    parser.add_argument("--basic-window", type=int, default=DEFAULT_BASIC_WINDOW)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
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
    parser.add_argument("--corr-val", dest="corr_val", action="store_true")
    parser.add_argument("--no-corr-val", dest="corr_val", action="store_false")
    parser.add_argument("--extra-filter", dest="extra_filter", action="store_true")
    parser.add_argument("--no-extra-filter", dest="extra_filter", action="store_false")
    parser.add_argument("--recall-by-window", dest="recall_by_window", action="store_true")
    parser.add_argument("--no-recall-by-window", dest="recall_by_window", action="store_false")
    parser.add_argument(
        "--artifact-mode",
        choices=("iterative", "final"),
        default=DEFAULT_ARTIFACT_MODE,
        help="Persist artifacts after each iteration (iterative) or only once after the run (final).",
    )
    parser.set_defaults(
        neg_corr=DEFAULT_NEG_CORR,
        corr_val=DEFAULT_CORR_VAL,
        extra_filter=DEFAULT_EXTRA_FILTER,
        recall_by_window=DEFAULT_RECALL_BY_WINDOW,
    )
    parser.set_defaults(
        parallel=DEFAULT_PARALLEL,
        parallel_sketch=DEFAULT_PARALLEL_SKETCH,
        parallel_candidates=DEFAULT_PARALLEL_CANDIDATES,
        parallel_validation=DEFAULT_PARALLEL_VALIDATION,
    )
    args = parser.parse_args()

    cfg_dataset = _load_dataset_config(args.dataset_config)
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_dataset, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, CORR_VAL, EXTRA_FILTER, RECALL_BY_WINDOW, ARTIFACT_MODE, DATA_LOADER

    WINDOW_SIZE = args.window_size
    WINDOW_STEP = args.window_step
    BASIC_WINDOW = args.basic_window
    N_LAGS = args.n_lags
    CORR_THRESHOLD = args.corr_threshold
    PARALLEL = args.parallel
    PARALLEL_SKETCH = args.parallel_sketch
    PARALLEL_CANDIDATES = args.parallel_candidates
    PARALLEL_VALIDATION = args.parallel_validation
    EXEC_MODE = "thread" if _any_parallel(PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION) else "sequential"
    NEG_CORR = args.neg_corr
    CORR_VAL = args.corr_val
    EXTRA_FILTER = args.extra_filter
    RECALL_BY_WINDOW = args.recall_by_window
    ARTIFACT_MODE = args.artifact_mode
    if args.loader:
        DATA_LOADER = _load_loader(args.loader)
    if DATA_LOADER is None:
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")

    base_config = build_base_config()

    for country, var, data, ids in iter_datasets():
        for n_year in N_YEARS:
            for n_var in N_VARS:
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                test_data, ids_n_var = prepare_test_data(data, ids, n_year, n_var)

                base_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder())
                optim_dir = os.path.join(base_dir, "optim")
                os.makedirs(base_dir, exist_ok=True)

                for alg in MODES:
                    params_path = os.path.join(optim_dir, f"best_params_{alg}.json")
                    if not os.path.exists(params_path):
                        raise FileNotFoundError(f"Best parameter file not found: {params_path}")
                    params = load_best_params(params_path)

                    output_csv = os.path.join(base_dir, f"corrtrack_run_{alg}.csv")
                    metadata = {
                        "alg": alg,
                        "mode": "main",
                        "optim": "best",
                        "nodes": params.get("nodes"),
                    }

                    run_and_log_corrtrack(
                        dataset_id,
                        test_data,
                        ids_n_var,
                        base_config,
                        params,
                        output_csv,
                        metadata=metadata,
                        recall_by_window=RECALL_BY_WINDOW,
                        corr_val=CORR_VAL,
                        artifact_prefix=os.path.join(base_dir, f"main_{alg}"),
                    )


if __name__ == "__main__":
    main()
