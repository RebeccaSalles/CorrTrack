import argparse
import importlib
import importlib.util
import json
import os
import numpy as np
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import CorrTrack_optimize

DEFAULT_PARAM_GRID_CONFIG = Path(__file__).with_name(
    "experiment_run_param_grid.py"
)
DEFAULT_DATASET_CONFIG = Path(__file__).with_name(
    "experiment_dataset_fr_air_temperature_7_1.py"
)
DEFAULT_OBS_MODE = "years"

DEFAULT_WINDOW_SIZE = 7 * 24
DEFAULT_WINDOW_STEP = 12
DEFAULT_BASIC_WINDOW = None
DEFAULT_N_LAGS = 7 * 24
DEFAULT_CORR_THRESHOLD = 0.7

DEFAULT_PARALLEL = False
DEFAULT_EXEC_MODE = "thread" if DEFAULT_PARALLEL else "sequential"
DEFAULT_NEG_CORR = False
DEFAULT_CORR_VAL = True
DEFAULT_EXTRA_FILTER = False
DEFAULT_RECALL_BY_WINDOW = True
DEFAULT_TARGET_RECALL = 0.95
DEFAULT_TRAIN_RATIO = 0.3

PARALLEL = DEFAULT_PARALLEL
EXEC_MODE = DEFAULT_EXEC_MODE
NEG_CORR = DEFAULT_NEG_CORR
CORR_VAL = DEFAULT_CORR_VAL
EXTRA_FILTER = DEFAULT_EXTRA_FILTER
RECALL_BY_WINDOW = DEFAULT_RECALL_BY_WINDOW
TARGET_RECALL = DEFAULT_TARGET_RECALL
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
WINDOW_STEP = DEFAULT_WINDOW_STEP
BASIC_WINDOW = DEFAULT_BASIC_WINDOW
N_LAGS = DEFAULT_N_LAGS
CORR_THRESHOLD = DEFAULT_CORR_THRESHOLD
TRAIN_RATIO = DEFAULT_TRAIN_RATIO

RESULT_FOLDER = None
COUNTRIES = VARIABLES = N_VARS = N_YEARS = MODES = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
PARAM_GRID = None
OBS_MODE = DEFAULT_OBS_MODE


def _load_module(config_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_param_grid(config_path: Path):
    module = _load_module(config_path, "experiment_param_grid")
    return module.PARAM_GRID


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
    loader = getattr(cfg, "DATA_LOADER", None)
    if loader is None:
        raise RuntimeError("Dataset configuration must define DATA_LOADER")
    DATA_LOADER = loader
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
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")
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


def prepare_training_data(data, ids, n_year, n_var, train_ratio=1.0):
    rows = _select_rows(data.shape[0], n_year)
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    length_data = data_stream.shape[0]
    train_end = round(train_ratio * length_data)
    train_data = np.transpose(data_stream[:train_end, :])
    ids_n_var = ids[: data_stream.shape[1] - 1]
    return train_data, ids_n_var


def config_folder():
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    return f"ws{WINDOW_SIZE}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"


def best_params_to_json(best_row):
    obj = {}
    for key, value in best_row.items():
        if isinstance(value, (np.generic,)):
            value = value.item()
        obj[key] = value
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run CorrTrack hyper-parameter search.")
    parser.add_argument(
        "--param-grid-config",
        type=Path,
        default=DEFAULT_PARAM_GRID_CONFIG,
        help="Path to configuration module providing PARAM_GRID.",
    )
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
    parser.add_argument("--neg-corr", dest="neg_corr", action="store_true")
    parser.add_argument("--no-neg-corr", dest="neg_corr", action="store_false")
    parser.add_argument("--corr-val", dest="corr_val", action="store_true")
    parser.add_argument("--no-corr-val", dest="corr_val", action="store_false")
    parser.add_argument("--extra-filter", dest="extra_filter", action="store_true")
    parser.add_argument("--no-extra-filter", dest="extra_filter", action="store_false")
    parser.add_argument("--recall-by-window", dest="recall_by_window", action="store_true")
    parser.add_argument("--no-recall-by-window", dest="recall_by_window", action="store_false")
    parser.set_defaults(
        parallel=DEFAULT_PARALLEL,
        neg_corr=DEFAULT_NEG_CORR,
        corr_val=DEFAULT_CORR_VAL,
        extra_filter=DEFAULT_EXTRA_FILTER,
        recall_by_window=DEFAULT_RECALL_BY_WINDOW,
    )
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    args = parser.parse_args()

    cfg_dataset = _load_module(args.dataset_config, "experiment_dataset")
    _apply_dataset_config(cfg_dataset)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD
    global PARALLEL, EXEC_MODE, NEG_CORR, CORR_VAL, EXTRA_FILTER, RECALL_BY_WINDOW
    global TARGET_RECALL, TRAIN_RATIO, PARAM_GRID, DATA_LOADER

    WINDOW_SIZE = args.window_size
    WINDOW_STEP = args.window_step
    BASIC_WINDOW = args.basic_window
    N_LAGS = args.n_lags
    CORR_THRESHOLD = args.corr_threshold
    PARALLEL = args.parallel
    EXEC_MODE = "thread" if PARALLEL else "sequential"
    NEG_CORR = args.neg_corr
    CORR_VAL = args.corr_val
    EXTRA_FILTER = args.extra_filter
    RECALL_BY_WINDOW = args.recall_by_window
    TARGET_RECALL = args.target_recall
    TRAIN_RATIO = args.train_ratio
    PARAM_GRID = _load_param_grid(args.param_grid_config)
    if args.loader:
        DATA_LOADER = _load_loader(args.loader)
    if DATA_LOADER is None:
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")

    for country, var, data, ids in iter_datasets():
        for n_year in N_YEARS:
            for n_var in N_VARS:
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                train_data, ids_n_var = prepare_training_data(data, ids, n_year, n_var, TRAIN_RATIO)

                output_dir = os.path.join(
                    "correlation",
                    RESULT_FOLDER,
                    dataset_id,
                    config_folder(),
                    "optim",
                )
                os.makedirs(output_dir, exist_ok=True)

                for alg in MODES:
                    optimizer = CorrTrack_optimize(
                        train_data,
                        ids_n_var,
                        WINDOW_SIZE,
                        WINDOW_STEP,
                        N_LAGS,
                        CORR_THRESHOLD,
                        RECALL_BY_WINDOW,
                        alg,
                        NEG_CORR,
                        CORR_VAL,
                        extra_filter=EXTRA_FILTER,
                        exec=EXEC_MODE,
                    )

                    output_prefix = os.path.join(output_dir, f"corrtrack_optim_{dataset_id}")
                    best_df = optimizer.get_optim_params(
                        PARAM_GRID,
                        output_prefix,
                        dataset_id,
                        run=True,
                        target_recall=TARGET_RECALL,
                    )

                    best_params = best_df.iloc[0].to_dict()
                    selected_path = os.path.join(output_dir, f"best_params_{alg}.json")
                    with open(selected_path, "w", encoding="utf-8") as fp:
                        json.dump(best_params_to_json(best_params), fp, indent=2)


if __name__ == "__main__":
    main()
