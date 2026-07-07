import argparse
import importlib
import importlib.util
import json
import os
import numpy as np
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import CorrTrack_optimize


def _load_module(config_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


DEFAULT_EXEC_PARAM_CONFIG = Path(__file__).with_name(
    "experiment_run_exec_param.py"
)
_DEFAULT_EXEC_CFG = _load_module(DEFAULT_EXEC_PARAM_CONFIG, "experiment_exec_defaults")

DEFAULT_PARAM_GRID_CONFIG = Path(__file__).with_name(
    "experiment_run_param_grid.py"
)
DEFAULT_DATASET_CONFIG = Path(__file__).with_name(
    "experiment_dataset_fr_air_temperature_7_1.py"
)
DEFAULT_OBS_MODE = "years"

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
DEFAULT_ARTIFACT_MODE = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_MODE", "buffered")
DEFAULT_ARTIFACT_BUFFER_MAX_ROWS = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_BUFFER_MAX_ROWS", 250000)
DEFAULT_ARTIFACT_MERGE_MODE = getattr(_DEFAULT_EXEC_CFG, "ARTIFACT_MERGE_MODE", "merged")
DEFAULT_SAVE_ONLY_REQUIRED_ARTIFACTS = getattr(_DEFAULT_EXEC_CFG, "SAVE_ONLY_REQUIRED_ARTIFACTS", False)
DEFAULT_SAVE_MAXLAG_ARTIFACTS = getattr(_DEFAULT_EXEC_CFG, "SAVE_MAXLAG_ARTIFACTS", True)
DEFAULT_TARGET_RECALL = getattr(_DEFAULT_EXEC_CFG, "TARGET_RECALL", 0.95)
DEFAULT_RECALL_FALLBACK_NEAR_RATIO = getattr(_DEFAULT_EXEC_CFG, "RECALL_FALLBACK_NEAR_RATIO", 0.98)
DEFAULT_SPEEDUP_NEAR_RATIO = getattr(_DEFAULT_EXEC_CFG, "SPEEDUP_NEAR_RATIO", 0.98)
DEFAULT_TRAIN_RATIO = getattr(_DEFAULT_EXEC_CFG, "TRAIN_RATIO", 0.3)
DEFAULT_OPTIM_TUNING_MODE = "sampling"
DEFAULT_OPTIM_PROXY_ANCHOR_COUNT = getattr(_DEFAULT_EXEC_CFG, "OPTIM_PROXY_ANCHOR_COUNT", 64)
DEFAULT_OPTIM_PROXY_MAX_PAIR_ROWS = getattr(_DEFAULT_EXEC_CFG, "OPTIM_PROXY_MAX_PAIR_ROWS", 250000)
DEFAULT_OPTIM_PROXY_RANDOM_SEED = getattr(_DEFAULT_EXEC_CFG, "OPTIM_PROXY_RANDOM_SEED", 2468)
DEFAULT_OPTIM_PROXY_BOOTSTRAP_ENABLED = getattr(_DEFAULT_EXEC_CFG, "OPTIM_PROXY_BOOTSTRAP_ENABLED", True)
DEFAULT_OPTIM_PROXY_BOOTSTRAP_REPEATS = getattr(_DEFAULT_EXEC_CFG, "OPTIM_PROXY_BOOTSTRAP_REPEATS", 1000)
DEFAULT_OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL",
    0.90,
)
DEFAULT_OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS",
    30,
)
DEFAULT_OPTIM_PROXY_DISTANCE_CACHE_MAX_ROWS = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_DISTANCE_CACHE_MAX_ROWS",
    250000,
)
DEFAULT_OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED",
    True,
)
DEFAULT_OPTIM_PROXY_MAX_ANCHOR_COUNT = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_MAX_ANCHOR_COUNT",
    max(128, int(DEFAULT_OPTIM_PROXY_ANCHOR_COUNT) * 2),
)
DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_FACTOR = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_ANCHOR_EXPAND_FACTOR",
    2.0,
)
DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS",
    1,
)
# (2026-07-05) How many times to re-run each proxy trial's full sketch+
# candidate-search purely for timing stability (mean taken; other fields
# are deterministic so only computed once). Higher = less noisy real-time
# tie-break, at the cost of ~Nx hyperopt wall-clock time.
DEFAULT_OPTIM_PROXY_TIMING_REPEATS = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_TIMING_REPEATS",
    3,
)
# Relative tolerance defining "reasonably close" to the best achievable
# candidate rate among feasible configs -- within this fraction, real
# measured execution time breaks the tie instead of further candidate-count
# refinements. Candidate-count minimization remains the dominant objective
# outside this tolerance.
DEFAULT_OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE = getattr(
    _DEFAULT_EXEC_CFG,
    "OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE",
    0.05,
)
DEFAULT_VERBOSE = getattr(_DEFAULT_EXEC_CFG, "VERBOSE", False)
DEFAULT_TESTING = getattr(_DEFAULT_EXEC_CFG, "TESTING", False)
DEFAULT_RESULT_FOLDER = None
DEFAULT_MAX_WORKERS = getattr(_DEFAULT_EXEC_CFG, "MAX_WORKERS", 0)
DEFAULT_CANDIDATE_BUCKET_WIDTH = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_BUCKET_WIDTH", None)
DEFAULT_CANDIDATE_BLOCK_SIZE_STEPS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_BLOCK_SIZE_STEPS", 32)
DEFAULT_CANDIDATE_BLOCK_INDEX_DIMS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_BLOCK_INDEX_DIMS", 1)
DEFAULT_CANDIDATE_N_PIVOTS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_N_PIVOTS", 8)
DEFAULT_CANDIDATE_N_PROBE_PIVOTS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_N_PROBE_PIVOTS", 2)
DEFAULT_CANDIDATE_PIVOT_SELECTION = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_PIVOT_SELECTION", "random_unit")
DEFAULT_CANDIDATE_PIVOT_SEED = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_PIVOT_SEED", 2468)
DEFAULT_CANDIDATE_SIMILARITY = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_SIMILARITY", "l2")
DEFAULT_CANDIDATE_COSINE_THRESHOLD = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_COSINE_THRESHOLD", None)
DEFAULT_CANDIDATE_HAMMING_Z = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_HAMMING_Z", 3.0)
DEFAULT_CANDIDATE_HAMMING_HMAX = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_HAMMING_HMAX", None)
DEFAULT_CANDIDATE_HAMMING_GROUPS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_HAMMING_GROUPS", 8)
DEFAULT_CANDIDATE_FILTER_HAMMING = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_FILTER_HAMMING", True)
DEFAULT_CANDIDATE_FILTER_COSINE = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_FILTER_COSINE", True)
DEFAULT_HYBRID_VALIDATION = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION", False)
DEFAULT_HYBRID_VALIDATION_MIN_REPEAT_RATE = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_MIN_REPEAT_RATE", 0.25)
DEFAULT_HYBRID_VALIDATION_DISABLE_RATE = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_DISABLE_RATE", None)
DEFAULT_HYBRID_VALIDATION_EMA_ALPHA = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_EMA_ALPHA", 0.25)
DEFAULT_HYBRID_VALIDATION_MIN_CANDIDATES = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_MIN_CANDIDATES", 256)

PARALLEL = DEFAULT_PARALLEL
PARALLEL_SKETCH = DEFAULT_PARALLEL_SKETCH
PARALLEL_CANDIDATES = DEFAULT_PARALLEL_CANDIDATES
PARALLEL_VALIDATION = DEFAULT_PARALLEL_VALIDATION
EXEC_MODE = DEFAULT_EXEC_MODE
NEG_CORR = DEFAULT_NEG_CORR
ARTIFACT_MODE = DEFAULT_ARTIFACT_MODE
ARTIFACT_BUFFER_MAX_ROWS = DEFAULT_ARTIFACT_BUFFER_MAX_ROWS
ARTIFACT_MERGE_MODE = DEFAULT_ARTIFACT_MERGE_MODE
SAVE_ONLY_REQUIRED_ARTIFACTS = DEFAULT_SAVE_ONLY_REQUIRED_ARTIFACTS
SAVE_MAXLAG_ARTIFACTS = DEFAULT_SAVE_MAXLAG_ARTIFACTS
TARGET_RECALL = DEFAULT_TARGET_RECALL
RECALL_FALLBACK_NEAR_RATIO = DEFAULT_RECALL_FALLBACK_NEAR_RATIO
SPEEDUP_NEAR_RATIO = DEFAULT_SPEEDUP_NEAR_RATIO
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
WINDOW_STEP = DEFAULT_WINDOW_STEP
BASIC_WINDOW = DEFAULT_BASIC_WINDOW
N_LAGS = DEFAULT_N_LAGS
CORR_THRESHOLD = DEFAULT_CORR_THRESHOLD
TRAIN_RATIO = DEFAULT_TRAIN_RATIO
OPTIM_TUNING_MODE = DEFAULT_OPTIM_TUNING_MODE
OPTIM_PROXY_CONFIG = {
    "anchor_count": DEFAULT_OPTIM_PROXY_ANCHOR_COUNT,
    "max_pair_rows": DEFAULT_OPTIM_PROXY_MAX_PAIR_ROWS,
    "random_seed": DEFAULT_OPTIM_PROXY_RANDOM_SEED,
    "bootstrap_enabled": DEFAULT_OPTIM_PROXY_BOOTSTRAP_ENABLED,
    "bootstrap_repeats": DEFAULT_OPTIM_PROXY_BOOTSTRAP_REPEATS,
    "bootstrap_confidence_level": DEFAULT_OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL,
    "bootstrap_min_gt_events": DEFAULT_OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS,
    "eval_mode": "cached_distances",
    "distance_cache_max_rows": DEFAULT_OPTIM_PROXY_DISTANCE_CACHE_MAX_ROWS,
    "adaptive_anchor_enabled": DEFAULT_OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED,
    "max_anchor_count": DEFAULT_OPTIM_PROXY_MAX_ANCHOR_COUNT,
    "anchor_expand_factor": DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_FACTOR,
    "anchor_expand_max_rounds": DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS,
    "timing_repeats": DEFAULT_OPTIM_PROXY_TIMING_REPEATS,
    "candidate_rate_close_tolerance": DEFAULT_OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE,
}
VERBOSE = DEFAULT_VERBOSE
TESTING = DEFAULT_TESTING

RESULT_FOLDER = DEFAULT_RESULT_FOLDER
MAX_WORKERS = DEFAULT_MAX_WORKERS
CANDIDATE_BUCKET_WIDTH = DEFAULT_CANDIDATE_BUCKET_WIDTH
CANDIDATE_BLOCK_SIZE_STEPS = DEFAULT_CANDIDATE_BLOCK_SIZE_STEPS
CANDIDATE_BLOCK_INDEX_DIMS = DEFAULT_CANDIDATE_BLOCK_INDEX_DIMS
CANDIDATE_N_PIVOTS = DEFAULT_CANDIDATE_N_PIVOTS
CANDIDATE_N_PROBE_PIVOTS = DEFAULT_CANDIDATE_N_PROBE_PIVOTS
CANDIDATE_PIVOT_SELECTION = DEFAULT_CANDIDATE_PIVOT_SELECTION
CANDIDATE_PIVOT_SEED = DEFAULT_CANDIDATE_PIVOT_SEED
CANDIDATE_SIMILARITY = DEFAULT_CANDIDATE_SIMILARITY
CANDIDATE_COSINE_THRESHOLD = DEFAULT_CANDIDATE_COSINE_THRESHOLD
CANDIDATE_HAMMING_Z = DEFAULT_CANDIDATE_HAMMING_Z
CANDIDATE_HAMMING_HMAX = DEFAULT_CANDIDATE_HAMMING_HMAX
CANDIDATE_HAMMING_GROUPS = DEFAULT_CANDIDATE_HAMMING_GROUPS
CANDIDATE_FILTER_HAMMING = DEFAULT_CANDIDATE_FILTER_HAMMING
CANDIDATE_FILTER_COSINE = DEFAULT_CANDIDATE_FILTER_COSINE
HYBRID_VALIDATION = DEFAULT_HYBRID_VALIDATION
HYBRID_VALIDATION_MIN_REPEAT_RATE = DEFAULT_HYBRID_VALIDATION_MIN_REPEAT_RATE
HYBRID_VALIDATION_DISABLE_RATE = DEFAULT_HYBRID_VALIDATION_DISABLE_RATE
HYBRID_VALIDATION_EMA_ALPHA = DEFAULT_HYBRID_VALIDATION_EMA_ALPHA
HYBRID_VALIDATION_MIN_CANDIDATES = DEFAULT_HYBRID_VALIDATION_MIN_CANDIDATES
COUNTRIES = VARIABLES = N_VARS = N_YEARS = None
MODES = ["corrtrack"]
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
PARAM_GRID = None
OBS_MODE = DEFAULT_OBS_MODE


def _load_grid_config(config_path: Path):
    module = _load_module(config_path, "experiment_param_grid")
    return module.PARAM_GRID


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


def _validate_tuning_split(tuning_mode, train_ratio):
    ratio = float(train_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(
            "Proxy-anchor hyperparameter tuning requires a true holdout remainder, "
            f"so train_ratio must be strictly between 0 and 1; got train_ratio={train_ratio}."
        )


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
    window_slug = "-".join(str(v) for v in WINDOW_SIZE) if isinstance(WINDOW_SIZE, (list, tuple)) else str(WINDOW_SIZE)
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    slug = f"ws{window_slug}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"
    return str(Path(Path(__file__).resolve().parent.name) / slug)


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
    parser.add_argument(
        "--artifact-mode",
        choices=("iterative", "final", "buffered"),
        default=None,
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
    parser.add_argument("--hybrid-validation", dest="hybrid_validation", action="store_true")
    parser.add_argument("--no-hybrid-validation", dest="hybrid_validation", action="store_false")
    parser.add_argument("--hybrid-validation-min-repeat-rate", type=float, default=None)
    parser.add_argument("--hybrid-validation-disable-rate", type=float, default=None)
    parser.add_argument("--hybrid-validation-ema-alpha", type=float, default=None)
    parser.add_argument("--hybrid-validation-min-candidates", type=int, default=None)
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
        hybrid_validation=None,
        artifact_mode=None,
        artifact_buffer_max_rows=None,
        artifact_merge_mode=None,
        save_only_required_artifacts=None,
        save_maxlag_artifacts=None,
        verbose=None,
        testing=None,
    )
    parser.add_argument("--target-recall", type=float, default=None)
    parser.add_argument("--recall-fallback-near-ratio", type=float, default=None)
    parser.add_argument("--speedup-near-ratio", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    args = parser.parse_args()

    cfg_exec = _load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = _load_module(args.dataset_config, "experiment_dataset")
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_exec, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, ARTIFACT_MODE, ARTIFACT_BUFFER_MAX_ROWS, ARTIFACT_MERGE_MODE, SAVE_ONLY_REQUIRED_ARTIFACTS, SAVE_MAXLAG_ARTIFACTS
    global TARGET_RECALL, RECALL_FALLBACK_NEAR_RATIO, SPEEDUP_NEAR_RATIO, TRAIN_RATIO, OPTIM_TUNING_MODE, OPTIM_PROXY_CONFIG, PARAM_GRID, DATA_LOADER, RESULT_FOLDER, MAX_WORKERS, CANDIDATE_BUCKET_WIDTH, CANDIDATE_BLOCK_SIZE_STEPS, CANDIDATE_BLOCK_INDEX_DIMS, CANDIDATE_N_PIVOTS, CANDIDATE_N_PROBE_PIVOTS, CANDIDATE_PIVOT_SELECTION, CANDIDATE_PIVOT_SEED, CANDIDATE_SIMILARITY, CANDIDATE_COSINE_THRESHOLD, CANDIDATE_HAMMING_Z, CANDIDATE_HAMMING_HMAX, CANDIDATE_HAMMING_GROUPS, CANDIDATE_FILTER_HAMMING, CANDIDATE_FILTER_COSINE
    global VERBOSE, TESTING, HYBRID_VALIDATION, HYBRID_VALIDATION_MIN_REPEAT_RATE, HYBRID_VALIDATION_DISABLE_RATE, HYBRID_VALIDATION_EMA_ALPHA, HYBRID_VALIDATION_MIN_CANDIDATES

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
    TARGET_RECALL = _resolve_cfg_value(args.target_recall, cfg_exec, "TARGET_RECALL", DEFAULT_TARGET_RECALL)
    RECALL_FALLBACK_NEAR_RATIO = _resolve_cfg_value(
        args.recall_fallback_near_ratio,
        cfg_exec,
        "RECALL_FALLBACK_NEAR_RATIO",
        DEFAULT_RECALL_FALLBACK_NEAR_RATIO,
    )
    SPEEDUP_NEAR_RATIO = _resolve_cfg_value(
        args.speedup_near_ratio,
        cfg_exec,
        "SPEEDUP_NEAR_RATIO",
        DEFAULT_SPEEDUP_NEAR_RATIO,
    )
    TRAIN_RATIO = _resolve_cfg_value(args.train_ratio, cfg_exec, "TRAIN_RATIO", DEFAULT_TRAIN_RATIO)
    OPTIM_TUNING_MODE = "sampling"
    _validate_tuning_split(OPTIM_TUNING_MODE, TRAIN_RATIO)
    OPTIM_PROXY_CONFIG = {
        "anchor_count": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_ANCHOR_COUNT",
            DEFAULT_OPTIM_PROXY_ANCHOR_COUNT,
        ),
        "max_pair_rows": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_MAX_PAIR_ROWS",
            DEFAULT_OPTIM_PROXY_MAX_PAIR_ROWS,
        ),
        "random_seed": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_RANDOM_SEED",
            DEFAULT_OPTIM_PROXY_RANDOM_SEED,
        ),
        "bootstrap_enabled": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_BOOTSTRAP_ENABLED",
            DEFAULT_OPTIM_PROXY_BOOTSTRAP_ENABLED,
        ),
        "bootstrap_repeats": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_BOOTSTRAP_REPEATS",
            DEFAULT_OPTIM_PROXY_BOOTSTRAP_REPEATS,
        ),
        "bootstrap_confidence_level": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL",
            DEFAULT_OPTIM_PROXY_BOOTSTRAP_CONFIDENCE_LEVEL,
        ),
        "bootstrap_min_gt_events": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS",
            DEFAULT_OPTIM_PROXY_BOOTSTRAP_MIN_GT_EVENTS,
        ),
        "eval_mode": "cached_distances",
        "distance_cache_max_rows": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_DISTANCE_CACHE_MAX_ROWS",
            DEFAULT_OPTIM_PROXY_DISTANCE_CACHE_MAX_ROWS,
        ),
        "adaptive_anchor_enabled": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED",
            DEFAULT_OPTIM_PROXY_ADAPTIVE_ANCHORS_ENABLED,
        ),
        "max_anchor_count": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_MAX_ANCHOR_COUNT",
            DEFAULT_OPTIM_PROXY_MAX_ANCHOR_COUNT,
        ),
        "anchor_expand_factor": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_ANCHOR_EXPAND_FACTOR",
            DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_FACTOR,
        ),
        "anchor_expand_max_rounds": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS",
            DEFAULT_OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS,
        ),
        "timing_repeats": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_TIMING_REPEATS",
            DEFAULT_OPTIM_PROXY_TIMING_REPEATS,
        ),
        "candidate_rate_close_tolerance": _resolve_cfg_value(
            None,
            cfg_exec,
            "OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE",
            DEFAULT_OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE,
        ),
    }
    MAX_WORKERS = _resolve_cfg_value(None, cfg_exec, "MAX_WORKERS", DEFAULT_MAX_WORKERS)
    CANDIDATE_BUCKET_WIDTH = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_BUCKET_WIDTH", DEFAULT_CANDIDATE_BUCKET_WIDTH)
    CANDIDATE_BLOCK_SIZE_STEPS = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_BLOCK_SIZE_STEPS", DEFAULT_CANDIDATE_BLOCK_SIZE_STEPS)
    CANDIDATE_BLOCK_INDEX_DIMS = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_BLOCK_INDEX_DIMS", DEFAULT_CANDIDATE_BLOCK_INDEX_DIMS)
    CANDIDATE_N_PIVOTS = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_N_PIVOTS", DEFAULT_CANDIDATE_N_PIVOTS)
    CANDIDATE_N_PROBE_PIVOTS = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_N_PROBE_PIVOTS", DEFAULT_CANDIDATE_N_PROBE_PIVOTS)
    CANDIDATE_PIVOT_SELECTION = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_PIVOT_SELECTION", DEFAULT_CANDIDATE_PIVOT_SELECTION)
    CANDIDATE_PIVOT_SEED = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_PIVOT_SEED", DEFAULT_CANDIDATE_PIVOT_SEED)
    CANDIDATE_SIMILARITY = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_SIMILARITY", DEFAULT_CANDIDATE_SIMILARITY)
    CANDIDATE_COSINE_THRESHOLD = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_COSINE_THRESHOLD", DEFAULT_CANDIDATE_COSINE_THRESHOLD)
    CANDIDATE_HAMMING_Z = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_HAMMING_Z", DEFAULT_CANDIDATE_HAMMING_Z)
    CANDIDATE_HAMMING_HMAX = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_HAMMING_HMAX", DEFAULT_CANDIDATE_HAMMING_HMAX)
    CANDIDATE_HAMMING_GROUPS = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_HAMMING_GROUPS", DEFAULT_CANDIDATE_HAMMING_GROUPS)
    CANDIDATE_FILTER_HAMMING = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_FILTER_HAMMING", DEFAULT_CANDIDATE_FILTER_HAMMING)
    CANDIDATE_FILTER_COSINE = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_FILTER_COSINE", DEFAULT_CANDIDATE_FILTER_COSINE)
    HYBRID_VALIDATION = _resolve_cfg_value(args.hybrid_validation, cfg_exec, "HYBRID_VALIDATION", DEFAULT_HYBRID_VALIDATION)
    HYBRID_VALIDATION_MIN_REPEAT_RATE = _resolve_cfg_value(args.hybrid_validation_min_repeat_rate, cfg_exec, "HYBRID_VALIDATION_MIN_REPEAT_RATE", DEFAULT_HYBRID_VALIDATION_MIN_REPEAT_RATE)
    HYBRID_VALIDATION_DISABLE_RATE = _resolve_cfg_value(args.hybrid_validation_disable_rate, cfg_exec, "HYBRID_VALIDATION_DISABLE_RATE", DEFAULT_HYBRID_VALIDATION_DISABLE_RATE)
    HYBRID_VALIDATION_EMA_ALPHA = _resolve_cfg_value(args.hybrid_validation_ema_alpha, cfg_exec, "HYBRID_VALIDATION_EMA_ALPHA", DEFAULT_HYBRID_VALIDATION_EMA_ALPHA)
    HYBRID_VALIDATION_MIN_CANDIDATES = _resolve_cfg_value(args.hybrid_validation_min_candidates, cfg_exec, "HYBRID_VALIDATION_MIN_CANDIDATES", DEFAULT_HYBRID_VALIDATION_MIN_CANDIDATES)
    VERBOSE = _resolve_cfg_value(args.verbose, cfg_exec, "VERBOSE", DEFAULT_VERBOSE)
    TESTING = _resolve_cfg_value(args.testing, cfg_exec, "TESTING", DEFAULT_TESTING)
    PARAM_GRID = _load_grid_config(args.param_grid_config)
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
                        True,
                        alg,
                        NEG_CORR,
                        False,
                        exec=EXEC_MODE,
                        max_workers=MAX_WORKERS,
                        candidate_bucket_width=CANDIDATE_BUCKET_WIDTH,
                        candidate_block_size_steps=CANDIDATE_BLOCK_SIZE_STEPS,
                        candidate_block_index_dims=CANDIDATE_BLOCK_INDEX_DIMS,
                        candidate_similarity=CANDIDATE_SIMILARITY,
                        candidate_cosine_threshold=CANDIDATE_COSINE_THRESHOLD,
                        hybrid_validation=HYBRID_VALIDATION,
                        hybrid_validation_min_repeat_rate=HYBRID_VALIDATION_MIN_REPEAT_RATE,
                        hybrid_validation_disable_rate=HYBRID_VALIDATION_DISABLE_RATE,
                        hybrid_validation_ema_alpha=HYBRID_VALIDATION_EMA_ALPHA,
                        hybrid_validation_min_candidates=HYBRID_VALIDATION_MIN_CANDIDATES,
                        verbose=VERBOSE,
                        testing=TESTING,
                        parallel_sketch=PARALLEL_SKETCH,
                        parallel_candidates=PARALLEL_CANDIDATES,
                        parallel_validation=PARALLEL_VALIDATION,
                        track_min_dist=False,
                        artifact_mode=ARTIFACT_MODE,
                        artifact_buffer_max_rows=ARTIFACT_BUFFER_MAX_ROWS,
                        artifact_merge_mode=ARTIFACT_MERGE_MODE,
                        save_only_required_artifacts=SAVE_ONLY_REQUIRED_ARTIFACTS,
                        save_maxlag_artifacts=SAVE_MAXLAG_ARTIFACTS,
                        proxy_config=OPTIM_PROXY_CONFIG,
                    )

                    output_prefix = os.path.join(output_dir, f"corrtrack_optim_{dataset_id}")
                    best_df = optimizer.get_optim_params(
                        PARAM_GRID,
                        output_prefix,
                        dataset_id,
                        run=True,
                        target_recall=TARGET_RECALL,
                        recall_fallback_near_ratio=RECALL_FALLBACK_NEAR_RATIO,
                        speedup_near_ratio=SPEEDUP_NEAR_RATIO,
                    )

                    best_params = best_df.iloc[0].to_dict()
                    selected_path = os.path.join(output_dir, f"best_params_{alg}.json")
                    with open(selected_path, "w", encoding="utf-8") as fp:
                        json.dump(best_params_to_json(best_params), fp, indent=2)


if __name__ == "__main__":
    main()
