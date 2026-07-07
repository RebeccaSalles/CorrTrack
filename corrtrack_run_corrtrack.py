import argparse
import importlib
import importlib.util
import json
import os
import numpy as np
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import run_and_log_corrtrack


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
DEFAULT_CORR_VAL = getattr(_DEFAULT_EXEC_CFG, "CORR_VAL", True)
DEFAULT_MONITOR = getattr(_DEFAULT_EXEC_CFG, "MONITOR", True)
DEFAULT_TRACK_MIN_DIST = getattr(_DEFAULT_EXEC_CFG, "TRACK_MIN_DIST", True)
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
# (2026-07-06) Part 1: Cauchy-Schwarz row-level bound + cone-based
# block-level bound for candidate_backend="sorted_arrays_bs". Both pruning
# flags default off (opt-in) -- see docs/implementation_log.md for why
# (row-level pruning showed ~0 wall-clock benefit in testing; block-level
# pruning showed a real 4-5x speedup, but only when the data has genuine
# angular clustering tighter than the correlation threshold).
DEFAULT_CANDIDATE_BOUND_DIMS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_BOUND_DIMS", None)
DEFAULT_CANDIDATE_BOUND_DIM_SELECTION = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_BOUND_DIM_SELECTION", "variance")
DEFAULT_ENABLE_BLOCK_UB_PRUNING = getattr(_DEFAULT_EXEC_CFG, "ENABLE_BLOCK_UB_PRUNING", False)
DEFAULT_ENABLE_ROW_UB_PRUNING = getattr(_DEFAULT_EXEC_CFG, "ENABLE_ROW_UB_PRUNING", False)
# (2026-07-06) Part 1 follow-up: block_similarity_assignment/max_open_blocks
# -- online "leader" clustering at block-close time, so enable_block_ub_pruning
# actually has a chance to fire on real (arrival-order-unrelated-to-similarity)
# streaming data. Off by default -- opt-in, and NOT a clear net win in this
# session's own benchmarking (real, measured pruning, but still 1.2x-1.5x
# slower wall-clock at typical n_vectors). See docs/implementation_log.md,
# "block-level cone pruning: root cause found... then a full flat-array rewrite".
DEFAULT_BLOCK_SIMILARITY_ASSIGNMENT = getattr(_DEFAULT_EXEC_CFG, "BLOCK_SIMILARITY_ASSIGNMENT", False)
DEFAULT_MAX_OPEN_BLOCKS = getattr(_DEFAULT_EXEC_CFG, "MAX_OPEN_BLOCKS", 4)
# (2026-07-06) InstinctIndex -- experimental approximate graph candidate
# backend (candidate_backend="instinct"). Opt-in only, NOT in the default
# candidate_backend sweep -- see docs/implementation_log.md, "InstinctIndex:
# experimental approximate graph backend". These 4 params are inert for
# every other candidate_backend value.
DEFAULT_CANDIDATE_INSTINCT_QUERY_MODE = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_INSTINCT_QUERY_MODE", "hybrid")
DEFAULT_CANDIDATE_INSTINCT_TOP_K = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_INSTINCT_TOP_K", 256)
DEFAULT_CANDIDATE_INSTINCT_MIN_CANDIDATES = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_INSTINCT_MIN_CANDIDATES", 64)
DEFAULT_CANDIDATE_INSTINCT_ENTRY_POINTS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_INSTINCT_ENTRY_POINTS", 8)

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
MONITOR = DEFAULT_MONITOR
TRACK_MIN_DIST = DEFAULT_TRACK_MIN_DIST
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
CANDIDATE_BOUND_DIMS = DEFAULT_CANDIDATE_BOUND_DIMS
CANDIDATE_BOUND_DIM_SELECTION = DEFAULT_CANDIDATE_BOUND_DIM_SELECTION
ENABLE_BLOCK_UB_PRUNING = DEFAULT_ENABLE_BLOCK_UB_PRUNING
ENABLE_ROW_UB_PRUNING = DEFAULT_ENABLE_ROW_UB_PRUNING
BLOCK_SIMILARITY_ASSIGNMENT = DEFAULT_BLOCK_SIMILARITY_ASSIGNMENT
MAX_OPEN_BLOCKS = DEFAULT_MAX_OPEN_BLOCKS
CANDIDATE_INSTINCT_QUERY_MODE = DEFAULT_CANDIDATE_INSTINCT_QUERY_MODE
CANDIDATE_INSTINCT_TOP_K = DEFAULT_CANDIDATE_INSTINCT_TOP_K
CANDIDATE_INSTINCT_MIN_CANDIDATES = DEFAULT_CANDIDATE_INSTINCT_MIN_CANDIDATES
CANDIDATE_INSTINCT_ENTRY_POINTS = DEFAULT_CANDIDATE_INSTINCT_ENTRY_POINTS
COUNTRIES = VARIABLES = N_VARS = N_YEARS = None
MODES = ["corrtrack"]
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


def prepare_test_data(data, ids, n_year, n_var, train_ratio, tuning_mode="sampling"):
    rows = _select_rows(data.shape[0], n_year)
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    length_data = data_stream.shape[0]
    train_end = round(train_ratio * length_data)
    tuning_mode = "sampling"
    if train_end <= 0:
        raise ValueError(
            f"train_ratio={train_ratio} leaves no train data for CorrTrack evaluation; "
            "choose a value greater than 0."
        )
    if train_end >= length_data:
        raise ValueError(
            f"train_ratio={train_ratio} leaves no holdout test data for CorrTrack evaluation; "
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
        "candidate_bucket_width": CANDIDATE_BUCKET_WIDTH,
        "candidate_block_size_steps": CANDIDATE_BLOCK_SIZE_STEPS,
        "candidate_block_index_dims": CANDIDATE_BLOCK_INDEX_DIMS,
        "candidate_similarity": CANDIDATE_SIMILARITY,
        "candidate_cosine_threshold": CANDIDATE_COSINE_THRESHOLD,
        "hybrid_validation": HYBRID_VALIDATION,
        "hybrid_validation_min_repeat_rate": HYBRID_VALIDATION_MIN_REPEAT_RATE,
        "hybrid_validation_disable_rate": HYBRID_VALIDATION_DISABLE_RATE,
        "hybrid_validation_ema_alpha": HYBRID_VALIDATION_EMA_ALPHA,
        "hybrid_validation_min_candidates": HYBRID_VALIDATION_MIN_CANDIDATES,
        "candidate_bound_dims": CANDIDATE_BOUND_DIMS,
        "candidate_bound_dim_selection": CANDIDATE_BOUND_DIM_SELECTION,
        "enable_block_ub_pruning": ENABLE_BLOCK_UB_PRUNING,
        "enable_row_ub_pruning": ENABLE_ROW_UB_PRUNING,
        "block_similarity_assignment": BLOCK_SIMILARITY_ASSIGNMENT,
        "max_open_blocks": MAX_OPEN_BLOCKS,
        "candidate_instinct_query_mode": CANDIDATE_INSTINCT_QUERY_MODE,
        "candidate_instinct_top_k": CANDIDATE_INSTINCT_TOP_K,
        "candidate_instinct_min_candidates": CANDIDATE_INSTINCT_MIN_CANDIDATES,
        "candidate_instinct_entry_points": CANDIDATE_INSTINCT_ENTRY_POINTS,
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


def main():
    parser = argparse.ArgumentParser(description="Run CorrTrack main algorithm with selected params.")
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
    parser.add_argument("--corr-val", dest="corr_val", action="store_true")
    parser.add_argument("--no-corr-val", dest="corr_val", action="store_false")
    parser.add_argument("--monitor", dest="monitor", action="store_true")
    parser.add_argument("--no-monitor", dest="monitor", action="store_false")
    parser.add_argument("--track-min-dist", dest="track_min_dist", action="store_true")
    parser.add_argument("--no-track-min-dist", dest="track_min_dist", action="store_false")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--hybrid-validation", dest="hybrid_validation", action="store_true")
    parser.add_argument("--no-hybrid-validation", dest="hybrid_validation", action="store_false")
    parser.add_argument("--hybrid-validation-min-repeat-rate", type=float, default=None)
    parser.add_argument("--hybrid-validation-disable-rate", type=float, default=None)
    parser.add_argument("--hybrid-validation-ema-alpha", type=float, default=None)
    parser.add_argument("--hybrid-validation-min-candidates", type=int, default=None)
    parser.add_argument(
        "--candidate-bound-dims",
        type=int,
        default=None,
        help="Part 1: number of dims used for the Cauchy-Schwarz row bound / cone block bound "
        "(candidate_backend='sorted_arrays_bs' only). 0 or unset disables both.",
    )
    parser.add_argument(
        "--candidate-bound-dim-selection",
        choices=("variance", "first"),
        default=None,
        help="How to choose the bound dims: highest-variance (default) or the first N dims.",
    )
    parser.add_argument("--enable-block-ub-pruning", dest="enable_block_ub_pruning", action="store_true",
                        help="Part 1: cone/angular block-level pruning. Off by default; only helps when the "
                        "data has real angular clustering tighter than the correlation threshold.")
    parser.add_argument("--no-enable-block-ub-pruning", dest="enable_block_ub_pruning", action="store_false")
    parser.add_argument("--enable-row-ub-pruning", dest="enable_row_ub_pruning", action="store_true",
                        help="Part 1: Cauchy-Schwarz row-level pruning. Off by default; requires "
                        "--candidate-bound-dims > 0, and showed ~0 wall-clock benefit in testing.")
    parser.add_argument("--no-enable-row-ub-pruning", dest="enable_row_ub_pruning", action="store_false")
    parser.add_argument("--block-similarity-assignment", dest="block_similarity_assignment", action="store_true",
                        help="Part 1 follow-up: group each closing block's rows by online 'leader' "
                        "clustering instead of arrival order, so --enable-block-ub-pruning has a real "
                        "chance to fire. Off by default -- required for block-level pruning to do "
                        "anything on real streaming data (arrival order is unrelated to similarity), "
                        "but NOT a confirmed net wall-clock win in this codebase's own benchmarking "
                        "(real pruning fires, still ~1.2x-1.5x slower at typical n_vectors) -- see "
                        "docs/implementation_log.md before enabling for a real run.")
    parser.add_argument("--no-block-similarity-assignment", dest="block_similarity_assignment", action="store_false")
    parser.add_argument(
        "--max-open-blocks",
        type=int,
        default=None,
        help="Part 1 follow-up: number of concurrent online 'leader' clusters per closing block "
        "when --block-similarity-assignment is set. Default 4.",
    )
    parser.add_argument(
        "--candidate-instinct-query-mode",
        choices=("threshold", "topk", "hybrid"),
        default=None,
        help="InstinctIndex (candidate_backend=instinct) query mode: threshold-only, top-k-only, "
        "or hybrid (min_candidates floor + threshold). Experimental, approximate backend -- "
        "see docs/implementation_log.md. Default 'hybrid'.",
    )
    parser.add_argument(
        "--candidate-instinct-top-k",
        type=int,
        default=None,
        help="InstinctIndex: max candidates returned per query in topk/hybrid mode. Default 256.",
    )
    parser.add_argument(
        "--candidate-instinct-min-candidates",
        type=int,
        default=None,
        help="InstinctIndex: minimum candidates to keep in hybrid mode even below the gamma "
        "threshold. Default 64.",
    )
    parser.add_argument(
        "--candidate-instinct-entry-points",
        type=int,
        default=None,
        help="InstinctIndex: number of graph entry points maintained for best-first search. "
        "Default 8.",
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
        corr_val=None,
        monitor=None,
        track_min_dist=None,
        hybrid_validation=None,
        enable_block_ub_pruning=None,
        enable_row_ub_pruning=None,
        block_similarity_assignment=None,
        artifact_mode=None,
        artifact_buffer_max_rows=None,
        artifact_merge_mode=None,
        save_only_required_artifacts=None,
        save_maxlag_artifacts=None,
        verbose=None,
        testing=None,
    )
    args = parser.parse_args()

    cfg_exec = _load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = _load_dataset_config(args.dataset_config)
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_exec, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD, RESULT_FOLDER
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, CORR_VAL, MONITOR, TRACK_MIN_DIST, TRAIN_RATIO, OPTIM_TUNING_MODE, ARTIFACT_MODE, ARTIFACT_BUFFER_MAX_ROWS, ARTIFACT_MERGE_MODE, SAVE_ONLY_REQUIRED_ARTIFACTS, SAVE_MAXLAG_ARTIFACTS, DATA_LOADER, MAX_WORKERS, CANDIDATE_BUCKET_WIDTH, CANDIDATE_BLOCK_SIZE_STEPS, CANDIDATE_BLOCK_INDEX_DIMS, CANDIDATE_N_PIVOTS, CANDIDATE_N_PROBE_PIVOTS, CANDIDATE_PIVOT_SELECTION, CANDIDATE_PIVOT_SEED, CANDIDATE_SIMILARITY, CANDIDATE_COSINE_THRESHOLD, CANDIDATE_HAMMING_Z, CANDIDATE_HAMMING_HMAX, CANDIDATE_HAMMING_GROUPS, CANDIDATE_FILTER_HAMMING, CANDIDATE_FILTER_COSINE, HYBRID_VALIDATION, HYBRID_VALIDATION_MIN_REPEAT_RATE, HYBRID_VALIDATION_DISABLE_RATE, HYBRID_VALIDATION_EMA_ALPHA, HYBRID_VALIDATION_MIN_CANDIDATES, CANDIDATE_BOUND_DIMS, CANDIDATE_BOUND_DIM_SELECTION, ENABLE_BLOCK_UB_PRUNING, ENABLE_ROW_UB_PRUNING, BLOCK_SIMILARITY_ASSIGNMENT, MAX_OPEN_BLOCKS, CANDIDATE_INSTINCT_QUERY_MODE, CANDIDATE_INSTINCT_TOP_K, CANDIDATE_INSTINCT_MIN_CANDIDATES, CANDIDATE_INSTINCT_ENTRY_POINTS
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
    CORR_VAL = _resolve_cfg_value(args.corr_val, cfg_exec, "CORR_VAL", DEFAULT_CORR_VAL)
    MONITOR = _resolve_cfg_value(args.monitor, cfg_exec, "MONITOR", DEFAULT_MONITOR)
    TRACK_MIN_DIST = _resolve_cfg_value(
        args.track_min_dist, cfg_exec, "TRACK_MIN_DIST", DEFAULT_TRACK_MIN_DIST
    )
    if not CORR_VAL:
        MONITOR = False
        TRACK_MIN_DIST = False
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
    CANDIDATE_BOUND_DIMS = _resolve_cfg_value(args.candidate_bound_dims, cfg_exec, "CANDIDATE_BOUND_DIMS", DEFAULT_CANDIDATE_BOUND_DIMS)
    CANDIDATE_BOUND_DIM_SELECTION = _resolve_cfg_value(args.candidate_bound_dim_selection, cfg_exec, "CANDIDATE_BOUND_DIM_SELECTION", DEFAULT_CANDIDATE_BOUND_DIM_SELECTION)
    ENABLE_BLOCK_UB_PRUNING = _resolve_cfg_value(args.enable_block_ub_pruning, cfg_exec, "ENABLE_BLOCK_UB_PRUNING", DEFAULT_ENABLE_BLOCK_UB_PRUNING)
    ENABLE_ROW_UB_PRUNING = _resolve_cfg_value(args.enable_row_ub_pruning, cfg_exec, "ENABLE_ROW_UB_PRUNING", DEFAULT_ENABLE_ROW_UB_PRUNING)
    BLOCK_SIMILARITY_ASSIGNMENT = _resolve_cfg_value(args.block_similarity_assignment, cfg_exec, "BLOCK_SIMILARITY_ASSIGNMENT", DEFAULT_BLOCK_SIMILARITY_ASSIGNMENT)
    MAX_OPEN_BLOCKS = _resolve_cfg_value(args.max_open_blocks, cfg_exec, "MAX_OPEN_BLOCKS", DEFAULT_MAX_OPEN_BLOCKS)
    CANDIDATE_INSTINCT_QUERY_MODE = _resolve_cfg_value(args.candidate_instinct_query_mode, cfg_exec, "CANDIDATE_INSTINCT_QUERY_MODE", DEFAULT_CANDIDATE_INSTINCT_QUERY_MODE)
    CANDIDATE_INSTINCT_TOP_K = _resolve_cfg_value(args.candidate_instinct_top_k, cfg_exec, "CANDIDATE_INSTINCT_TOP_K", DEFAULT_CANDIDATE_INSTINCT_TOP_K)
    CANDIDATE_INSTINCT_MIN_CANDIDATES = _resolve_cfg_value(args.candidate_instinct_min_candidates, cfg_exec, "CANDIDATE_INSTINCT_MIN_CANDIDATES", DEFAULT_CANDIDATE_INSTINCT_MIN_CANDIDATES)
    CANDIDATE_INSTINCT_ENTRY_POINTS = _resolve_cfg_value(args.candidate_instinct_entry_points, cfg_exec, "CANDIDATE_INSTINCT_ENTRY_POINTS", DEFAULT_CANDIDATE_INSTINCT_ENTRY_POINTS)
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
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                test_data, ids_n_var = prepare_test_data(
                    data,
                    ids,
                    n_year,
                    n_var,
                    TRAIN_RATIO,
                    tuning_mode=OPTIM_TUNING_MODE,
                )

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
                        recall_by_window=True,
                        corr_val=CORR_VAL,
                        monitor=MONITOR,
                        verbose=VERBOSE,
                        testing=TESTING,
                        artifact_prefix=os.path.join(base_dir, f"main_{alg}"),
                    )


if __name__ == "__main__":
    main()
