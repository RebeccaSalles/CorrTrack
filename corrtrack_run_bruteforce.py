import os
import argparse
import importlib
import importlib.util
import json
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
DEFAULT_BASELINE_MODE = getattr(_DEFAULT_EXEC_CFG, "BASELINE_MODE", "bruteforce")
# FilCorr (baseline_mode="filcorr") competitor knobs. fs=0.0/ft=0.5 (full
# band, DC removed) is mathematically identical to standard Pearson -- the
# setting to use so this baseline is comparable to the bruteforce ground
# truth. See Candidates_BF_FilCorr in library_corrtrack_parallel.py.
DEFAULT_FILCORR_FS = getattr(_DEFAULT_EXEC_CFG, "FILCORR_FS", 0.0)
DEFAULT_FILCORR_FT = getattr(_DEFAULT_EXEC_CFG, "FILCORR_FT", 0.5)
DEFAULT_FILCORR_SAMPLING_RATE = getattr(_DEFAULT_EXEC_CFG, "FILCORR_SAMPLING_RATE", 1.0)
DEFAULT_BRAID_B = getattr(_DEFAULT_EXEC_CFG, "BRAID_B", 16)
DEFAULT_BRAID_GAMMA = getattr(_DEFAULT_EXEC_CFG, "BRAID_GAMMA", 0.4)
DEFAULT_BRAID_THIN = getattr(_DEFAULT_EXEC_CFG, "BRAID_THIN", False)
DEFAULT_BRAID_THIN_D0 = getattr(_DEFAULT_EXEC_CFG, "BRAID_THIN_D0", 400)
DEFAULT_BRAID_REPORT_MODE = getattr(_DEFAULT_EXEC_CFG, "BRAID_REPORT_MODE", "all_lags")
BRAID_B = DEFAULT_BRAID_B
BRAID_GAMMA = DEFAULT_BRAID_GAMMA
BRAID_THIN = DEFAULT_BRAID_THIN
BRAID_THIN_D0 = DEFAULT_BRAID_THIN_D0
BRAID_REPORT_MODE = DEFAULT_BRAID_REPORT_MODE
DEFAULT_VALIDATION_METRIC = getattr(_DEFAULT_EXEC_CFG, "VALIDATION_METRIC", "pearson")
DEFAULT_TRAIN_RATIO = getattr(_DEFAULT_EXEC_CFG, "TRAIN_RATIO", 0.3)
DEFAULT_OPTIM_TUNING_MODE = "sampling"
DEFAULT_ARTIFACT_MODE = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_MODE", "iterative")
DEFAULT_ARTIFACT_BUFFER_MAX_ROWS = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_BUFFER_MAX_ROWS", 250000)
DEFAULT_ARTIFACT_MERGE_MODE = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_MERGE_MODE", "merged")
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
BASELINE_MODE = DEFAULT_BASELINE_MODE
FILCORR_FS = DEFAULT_FILCORR_FS
FILCORR_FT = DEFAULT_FILCORR_FT
FILCORR_SAMPLING_RATE = DEFAULT_FILCORR_SAMPLING_RATE
VALIDATION_METRIC = DEFAULT_VALIDATION_METRIC
TRAIN_RATIO = DEFAULT_TRAIN_RATIO
OPTIM_TUNING_MODE = DEFAULT_OPTIM_TUNING_MODE
ARTIFACT_MODE = DEFAULT_ARTIFACT_MODE
ARTIFACT_BUFFER_MAX_ROWS = DEFAULT_ARTIFACT_BUFFER_MAX_ROWS
ARTIFACT_MERGE_MODE = DEFAULT_ARTIFACT_MERGE_MODE
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


def _load_tuned_validation_metric(optim_dir: str):
    """Read validation_metric from the hyperopt step's own best_params_
    corrtrack.json, if present -- lets the brute-force ground-truth
    baseline automatically match whatever metric hyperopt actually tuned
    for, instead of requiring --validation-metric to be kept in manual sync
    across corrtrack_run_bruteforce.py/experiment_run_exec_param.py/
    experiment_run_param_grid.py by hand. Returns None (falls back to the
    CLI/config default) if the file doesn't exist yet (e.g. brute-force run
    standalone, before any hyperparameter search), can't be parsed, or
    predates this field being recorded."""
    params_path = os.path.join(optim_dir, "best_params_corrtrack.json")
    if not os.path.isfile(params_path):
        return None
    try:
        with open(params_path, "r", encoding="utf-8") as fp:
            params = json.load(fp)
    except (OSError, ValueError):
        return None
    value = params.get("validation_metric")
    if not value or (isinstance(value, float) and value != value):  # NaN from a CSV-sourced JSON
        return None
    return str(value)


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
        "baseline_mode": BASELINE_MODE,
        "filcorr_fs": FILCORR_FS,
        "filcorr_ft": FILCORR_FT,
        "filcorr_sampling_rate": FILCORR_SAMPLING_RATE,
        "braid_b": BRAID_B,
        "braid_gamma": BRAID_GAMMA,
        "braid_thin": BRAID_THIN,
        "braid_thin_d0": BRAID_THIN_D0,
        "braid_report_mode": BRAID_REPORT_MODE,
        "validation_metric": VALIDATION_METRIC,
        "monitor": MONITOR,
        "track_min_dist": TRACK_MIN_DIST,
        "artifact_mode": ARTIFACT_MODE,
        "artifact_buffer_max_rows": ARTIFACT_BUFFER_MAX_ROWS,
        "artifact_merge_mode": ARTIFACT_MERGE_MODE,
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


def prepare_test_data(data, ids, n_year, n_var, train_ratio, tuning_mode="sampling"):
    rows = _select_rows(data.shape[0], n_year)
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    length_data = data_stream.shape[0]
    train_end = round(train_ratio * length_data)
    tuning_mode = "sampling"
    if train_end <= 0:
        raise ValueError(
            f"train_ratio={train_ratio} leaves no train data for brute-force evaluation; "
            "choose a value greater than 0."
        )
    if train_end >= length_data:
        raise ValueError(
            f"train_ratio={train_ratio} leaves no holdout test data for brute-force evaluation; "
            "choose a value strictly between 0 and 1."
        )
    test_data = np.transpose(data_stream[train_end:, :])
    ids_n_var = ids[: data_stream.shape[1] - 1]
    return test_data, ids_n_var


def config_folder():
    window_slug = "-".join(str(v) for v in WINDOW_SIZE) if isinstance(WINDOW_SIZE, (list, tuple)) else str(WINDOW_SIZE)
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    slug = f"ws{window_slug}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"
    return str(Path(Path(__file__).resolve().parent.name) / slug)


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
    parser.add_argument("--recall-by-window", dest="recall_by_window", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-recall-by-window", dest="recall_by_window", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--monitor", dest="monitor", action="store_true")
    parser.add_argument("--no-monitor", dest="monitor", action="store_false")
    parser.add_argument("--track-min-dist", dest="track_min_dist", action="store_true")
    parser.add_argument("--no-track-min-dist", dest="track_min_dist", action="store_false")
    parser.add_argument(
        "--baseline-mode",
        choices=("bruteforce", "exact_stomp", "filcorr", "tsubasa", "braid"),
        default=None,
        help="Exact baseline implementation for the brute-force stage. 'filcorr' is the "
        "Zhong/Souza/Mueen (ICDM 2020) competitor -- see --filcorr-fs/--filcorr-ft. "
        "'tsubasa' is Xu/Liu/Nargesian (SIGMOD 2022): exact Pearson from per-basic-window "
        "sketches (Lemma 1); no lag support, requires n_lags=0. 'braid' is Sakurai et al. "
        "(SIGMOD 2005 / TKDD 2010): geometric lag probing + smoothing + spline; see --braid-*.",
    )
    parser.add_argument(
        "--filcorr-fs", type=float, default=None,
        help="FilCorr pass-band lower bound (baseline_mode=filcorr only). Default 0.0 "
        "(full band, DC removed -- mathematically identical to standard Pearson, the "
        "setting comparable to this project's bruteforce ground truth).",
    )
    parser.add_argument(
        "--filcorr-ft", type=float, default=None,
        help="FilCorr pass-band upper bound (baseline_mode=filcorr only). Default 0.5 "
        "(Nyquist -- full band).",
    )
    parser.add_argument(
        "--filcorr-sampling-rate", type=float, default=None,
        help="FilCorr sampling frequency f (Hz); set fs/ft as fractions of Nyquist by "
        "leaving this at its default (1.0).",
    )
    parser.add_argument("--braid-b", type=int, default=None,
        help="BRAID coefficients per level (baseline_mode=braid). Paper: 16. "
             "2*b > n_lags makes BRAID exact (level 0 covers every lag).")
    parser.add_argument("--braid-gamma", type=float, default=None,
        help="BRAID lag-detection threshold on |R(l)| (Definition 1). Paper: 0.4.")
    parser.add_argument("--braid-thin", dest="braid_thin", action="store_true", default=None,
        help="Use ThinBRAID (TKDD 2010): per-series random projections instead of per-pair sums.")
    parser.add_argument("--braid-thin-d0", type=int, default=None,
        help="ThinBRAID projection dimension at level 0 (d_h = d0 / 2^h). Paper: 400.")
    parser.add_argument("--braid-report-mode", choices=("all_lags", "braid"), default=None,
        help="'all_lags': thresholded pairs at every harness lag (common-denominator comparison). "
             "'braid': one row per pair at its earliest local max of |R| >= gamma (their output).")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument(
        "--validation-metric",
        choices=("pearson", "spearman", "kendall", "dist_corr"),
        default=None,
        help="Validation metric for the brute-force ground-truth baseline. Must match the "
        "main run's --validation-metric for recall/precision comparisons and wall-clock "
        "baselines to be meaningful -- otherwise this baseline validates a different "
        "notion of 'correlated' than the tuned run does.",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("iterative", "final", "buffered"),
        default=None,
        help="Persist artifacts after each iteration (iterative), only once after the run (final), or spill in bounded chunks (buffered).",
    )
    parser.add_argument("--artifact-buffer-max-rows", type=int, default=None)
    parser.add_argument(
        "--artifact-merge-mode",
        choices=("merged", "chunks"),
        default=None,
        help="Merge chunked artifact CSVs at finalize time, or leave chunks plus a manifest.",
    )
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
    parser.set_defaults(
        parallel=None,
        parallel_sketch=None,
        parallel_candidates=None,
        parallel_validation=None,
        neg_corr=None,
        recall_by_window=None,
        monitor=None,
        track_min_dist=None,
        baseline_mode=None,
        filcorr_fs=None,
        filcorr_ft=None,
        filcorr_sampling_rate=None,
        braid_b=None,
        braid_gamma=None,
        braid_thin=None,
        braid_thin_d0=None,
        braid_report_mode=None,
        artifact_mode=None,
        artifact_buffer_max_rows=None,
        artifact_merge_mode=None,
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
    global EXEC_MODE, NEG_CORR, MONITOR, TRACK_MIN_DIST, BASELINE_MODE, ARTIFACT_MODE, ARTIFACT_BUFFER_MAX_ROWS, ARTIFACT_MERGE_MODE, SAVE_ONLY_REQUIRED_ARTIFACTS, SAVE_MAXLAG_ARTIFACTS
    global FILCORR_FS, FILCORR_FT, FILCORR_SAMPLING_RATE
    global BRAID_B, BRAID_GAMMA, BRAID_THIN, BRAID_THIN_D0, BRAID_REPORT_MODE
    global DATA_LOADER, RESULT_FOLDER, MAX_WORKERS, VERBOSE, TESTING, TRAIN_RATIO, OPTIM_TUNING_MODE
    global VALIDATION_METRIC

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
    BASELINE_MODE = _resolve_cfg_value(args.baseline_mode, cfg_exec, "BASELINE_MODE", DEFAULT_BASELINE_MODE)
    FILCORR_FS = _resolve_cfg_value(args.filcorr_fs, cfg_exec, "FILCORR_FS", DEFAULT_FILCORR_FS)
    FILCORR_FT = _resolve_cfg_value(args.filcorr_ft, cfg_exec, "FILCORR_FT", DEFAULT_FILCORR_FT)
    FILCORR_SAMPLING_RATE = _resolve_cfg_value(
        args.filcorr_sampling_rate, cfg_exec, "FILCORR_SAMPLING_RATE", DEFAULT_FILCORR_SAMPLING_RATE
    )
    BRAID_B = int(_resolve_cfg_value(args.braid_b, cfg_exec, "BRAID_B", DEFAULT_BRAID_B))
    BRAID_GAMMA = float(_resolve_cfg_value(args.braid_gamma, cfg_exec, "BRAID_GAMMA", DEFAULT_BRAID_GAMMA))
    BRAID_THIN = bool(_resolve_cfg_value(args.braid_thin, cfg_exec, "BRAID_THIN", DEFAULT_BRAID_THIN))
    BRAID_THIN_D0 = int(_resolve_cfg_value(args.braid_thin_d0, cfg_exec, "BRAID_THIN_D0", DEFAULT_BRAID_THIN_D0))
    BRAID_REPORT_MODE = str(_resolve_cfg_value(args.braid_report_mode, cfg_exec, "BRAID_REPORT_MODE", DEFAULT_BRAID_REPORT_MODE))
    VALIDATION_METRIC = _resolve_cfg_value(
        args.validation_metric, cfg_exec, "VALIDATION_METRIC", DEFAULT_VALIDATION_METRIC
    )
    TRAIN_RATIO = _resolve_cfg_value(args.train_ratio, cfg_exec, "TRAIN_RATIO", DEFAULT_TRAIN_RATIO)
    OPTIM_TUNING_MODE = "sampling"
    ARTIFACT_MODE = _resolve_cfg_value(args.artifact_mode, cfg_exec, "ARTIFACT_MODE", DEFAULT_ARTIFACT_MODE)
    ARTIFACT_BUFFER_MAX_ROWS = _resolve_cfg_value(
        args.artifact_buffer_max_rows, cfg_exec, "ARTIFACT_BUFFER_MAX_ROWS", DEFAULT_ARTIFACT_BUFFER_MAX_ROWS
    )
    ARTIFACT_MERGE_MODE = _resolve_cfg_value(
        args.artifact_merge_mode, cfg_exec, "ARTIFACT_MERGE_MODE", DEFAULT_ARTIFACT_MERGE_MODE
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
                test_data, ids_n_var = prepare_test_data(
                    data,
                    ids,
                    n_year,
                    n_var,
                    TRAIN_RATIO,
                    tuning_mode=OPTIM_TUNING_MODE,
                )
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                output_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder())
                os.makedirs(output_dir, exist_ok=True)

                dataset_base_config = dict(base_config)
                if args.validation_metric is None:
                    # No explicit --validation-metric override: prefer
                    # whatever the hyperparameter-search step actually
                    # tuned for over this script's own CLI/config default,
                    # so the ground-truth baseline can never silently
                    # validate a different metric than the tuned run does.
                    tuned_metric = _load_tuned_validation_metric(os.path.join(output_dir, "optim"))
                    if tuned_metric is not None:
                        dataset_base_config["validation_metric"] = tuned_metric

                metadata = {
                    "alg": "bf",
                    "mode": "bf",
                    "optim": "bf_baseline",
                    "nodes": base_config.get("max_workers"),
                }

                output_csv = os.path.join(output_dir, "bf_run.csv")

                run_and_log_bruteforce(
                    dataset_id,
                    test_data,
                    ids_n_var,
                    dataset_base_config,
                    output_csv,
                    metadata=metadata,
                    recall_by_window=True,
                    verbose=VERBOSE,
                    testing=TESTING,
                    artifact_prefix=os.path.join(output_dir, "bf"),
                )


if __name__ == "__main__":
    main()
