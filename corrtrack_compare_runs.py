import argparse
import importlib
import importlib.util
import os
import numpy as np
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import CorrTrack_compare

DEFAULT_WINDOW_SIZE = 7 * 24
DEFAULT_WINDOW_STEP = 12
DEFAULT_BASIC_WINDOW = None
DEFAULT_N_LAGS = 7 * 24
DEFAULT_CORR_THRESHOLD = 0.7

DEFAULT_EXEC_MODE = "sequential"
DEFAULT_NEG_CORR = False
DEFAULT_CORR_VAL = True
DEFAULT_EXTRA_FILTER = False
DEFAULT_RECALL_BY_WINDOW = True
DEFAULT_TRAIN_RATIO = 0.3

DEFAULT_DATASET_CONFIG = Path(__file__).with_name(
    "experiment_dataset_fr_air_temperature_7_1.py"
)

WINDOW_SIZE = DEFAULT_WINDOW_SIZE
WINDOW_STEP = DEFAULT_WINDOW_STEP
BASIC_WINDOW = DEFAULT_BASIC_WINDOW
N_LAGS = DEFAULT_N_LAGS
CORR_THRESHOLD = DEFAULT_CORR_THRESHOLD
EXEC_MODE = DEFAULT_EXEC_MODE
NEG_CORR = DEFAULT_NEG_CORR
CORR_VAL = DEFAULT_CORR_VAL
EXTRA_FILTER = DEFAULT_EXTRA_FILTER
RECALL_BY_WINDOW = DEFAULT_RECALL_BY_WINDOW
TRAIN_RATIO = DEFAULT_TRAIN_RATIO
RESULT_FOLDER = None
COUNTRIES = VARIABLES = N_VARS = N_YEARS = MODES = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None


def _load_dataset_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("experiment_dataset", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _apply_dataset_config(cfg):
    global RESULT_FOLDER, COUNTRIES, VARIABLES, N_VARS, N_YEARS, MODES, DATA_LOADER

    RESULT_FOLDER = cfg.RESULT_FOLDER
    COUNTRIES = cfg.COUNTRIES
    VARIABLES = cfg.VARIABLES
    N_VARS = cfg.N_VARS
    N_YEARS = cfg.N_YEARS
    MODES = cfg.MODES
    DATA_LOADER = getattr(cfg, "DATA_LOADER", None)


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
            data, ids = DATA_LOADER(country, var)
            yield country, var, data, ids


def prepare_data(data, ids, n_year, n_var, train_ratio):
    one_year = 365 * 24
    rows = np.r_[0, np.arange(data.shape[0] - n_year * one_year, data.shape[0])]
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    length_data = data_stream.shape[0]
    train_end = round(train_ratio * length_data)
    train_data = np.transpose(data_stream[:train_end, :])
    test_data = np.transpose(data_stream)
    ids_n_var = ids[: data_stream.shape[1] - 1]
    return train_data, test_data, ids_n_var


def config_folder():
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    return f"ws{WINDOW_SIZE}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"


def parse_args():
    parser = argparse.ArgumentParser(description="Build CorrTrack comparison reports.")
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
    parser.add_argument("--exec-mode", type=str, default=DEFAULT_EXEC_MODE)
    parser.add_argument("--neg-corr", dest="neg_corr", action="store_true")
    parser.add_argument("--no-neg-corr", dest="neg_corr", action="store_false")
    parser.add_argument("--corr-val", dest="corr_val", action="store_true")
    parser.add_argument("--no-corr-val", dest="corr_val", action="store_false")
    parser.add_argument("--extra-filter", dest="extra_filter", action="store_true")
    parser.add_argument("--no-extra-filter", dest="extra_filter", action="store_false")
    parser.add_argument("--recall-by-window", dest="recall_by_window", action="store_true")
    parser.add_argument("--no-recall-by-window", dest="recall_by_window", action="store_false")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.set_defaults(
        neg_corr=DEFAULT_NEG_CORR,
        corr_val=DEFAULT_CORR_VAL,
        extra_filter=DEFAULT_EXTRA_FILTER,
        recall_by_window=DEFAULT_RECALL_BY_WINDOW,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_dataset = _load_dataset_config(args.dataset_config)
    _apply_dataset_config(cfg_dataset)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD
    global EXEC_MODE, NEG_CORR, CORR_VAL, EXTRA_FILTER, RECALL_BY_WINDOW, TRAIN_RATIO, DATA_LOADER

    WINDOW_SIZE = args.window_size
    WINDOW_STEP = args.window_step
    BASIC_WINDOW = args.basic_window
    N_LAGS = args.n_lags
    CORR_THRESHOLD = args.corr_threshold
    EXEC_MODE = args.exec_mode
    NEG_CORR = args.neg_corr
    CORR_VAL = args.corr_val
    EXTRA_FILTER = args.extra_filter
    RECALL_BY_WINDOW = args.recall_by_window
    TRAIN_RATIO = args.train_ratio
    if args.loader:
        DATA_LOADER = _load_loader(args.loader)
    if DATA_LOADER is None:
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")

    for country, var, data, ids in iter_datasets():
        for n_year in N_YEARS:
            for n_var in N_VARS:
                dataset_id = f"{country}_{var}_{n_var}_{n_year}"
                train_data, test_data, ids_n_var = prepare_data(data, ids, n_year, n_var, TRAIN_RATIO)

                base_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder())
                bf_run_csv = os.path.join(base_dir, "bf_run.csv")
                corrtrack_run_files = {
                    alg: os.path.join(base_dir, f"corrtrack_run_{alg}.csv") for alg in MODES
                }
                output_csv = os.path.join(base_dir, f"corrtrack_metrics_{dataset_id}.csv")

                cc = CorrTrack_compare(
                    train_data,
                    test_data,
                    ids_n_var,
                    WINDOW_SIZE,
                    WINDOW_STEP,
                    BASIC_WINDOW,
                    N_LAGS,
                    CORR_THRESHOLD,
                    {},
                    RECALL_BY_WINDOW,
                    NEG_CORR,
                    CORR_VAL,
                    MODES,
                    exec=EXEC_MODE,
                    extra_filter=EXTRA_FILTER,
                )

                cc.compare_from_artifacts(dataset_id, bf_run_csv, corrtrack_run_files, output_csv)


if __name__ == "__main__":
    main()
