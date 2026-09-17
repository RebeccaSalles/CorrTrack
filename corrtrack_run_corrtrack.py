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
DEFAULT_CANDIDATE_COSINE_THRESHOLD = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_COSINE_THRESHOLD", None)
DEFAULT_HYBRID_VALIDATION = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION", False)
DEFAULT_HYBRID_VALIDATION_MIN_REPEAT_RATE = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_MIN_REPEAT_RATE", 0.25)
DEFAULT_HYBRID_VALIDATION_DISABLE_RATE = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_DISABLE_RATE", None)
DEFAULT_HYBRID_VALIDATION_EMA_ALPHA = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_EMA_ALPHA", 0.25)
DEFAULT_HYBRID_VALIDATION_MIN_CANDIDATES = getattr(_DEFAULT_EXEC_CFG, "HYBRID_VALIDATION_MIN_CANDIDATES", 256)
# Candidate search: two orthogonal parameters.
#   data_representation: "auto" (default, picks per validation_metric --
#     pearson->sketch_proj, spearman/kendall->sketch_concordance,
#     dist_corr->sketch_multichannel), "raw" (no representation, exhaustive
#     pairwise enumeration), "sketch_proj", "sketch_concordance", or
#     "sketch_multichannel".
#   candidate_backend: "auto" (default, resolves to "lsh_approx"),
#     "lsh_approx", "hamming_exact", or "brute_force".
DEFAULT_DATA_REPRESENTATION = getattr(_DEFAULT_EXEC_CFG, "DATA_REPRESENTATION", "auto")
DEFAULT_CANDIDATE_BACKEND = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_BACKEND", "auto")
# SignLSHBandIndex's band_width auto-sizing target -- see
# candidate_kernels.pyx's _finalize_sizing.
DEFAULT_CANDIDATE_LSH_TARGET_OCCUPANCY = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_LSH_TARGET_OCCUPANCY", 3.0)
# "pearson" (default) stays on the fast Cython path. "spearman"/"kendall"/
# "dist_corr" route to a dedicated per-row Python path.
DEFAULT_VALIDATION_METRIC = getattr(_DEFAULT_EXEC_CFG, "VALIDATION_METRIC", "pearson")
# "naive" (default) or "fast" (exact O(w log w) vs. naive O(w^2), both
# EXACT) -- only meaningful for validation_metric="dist_corr".
DEFAULT_DIST_CORR_ALGORITHM = getattr(_DEFAULT_EXEC_CFG, "DIST_CORR_ALGORITHM", "naive")
# Concordance sketch params -- only meaningful for data_representation=
# "sketch_concordance".
DEFAULT_CONCORDANCE_N_GAPS = getattr(_DEFAULT_EXEC_CFG, "CONCORDANCE_N_GAPS", None)
DEFAULT_CONCORDANCE_TARGET_DIM = getattr(_DEFAULT_EXEC_CFG, "CONCORDANCE_TARGET_DIM", 738)
DEFAULT_CONCORDANCE_MIN_GAP = getattr(_DEFAULT_EXEC_CFG, "CONCORDANCE_MIN_GAP", 8)
DEFAULT_CONCORDANCE_MIN_CAPACITY = getattr(_DEFAULT_EXEC_CFG, "CONCORDANCE_MIN_CAPACITY", 16)
# Config-file-only (no CLI flag), matching distance_corr_sketch_
# multichannel_gamma's own precedent below.
DEFAULT_CONCORDANCE_MULTICHANNEL_GAMMA = getattr(
    _DEFAULT_EXEC_CFG, "CONCORDANCE_MULTICHANNEL_GAMMA", None
)
# Distance-correlation sketch params -- only meaningful for data_
# representation="sketch_multichannel" (requires validation_metric=
# "dist_corr").
DEFAULT_DISTANCE_CORR_SKETCH_K = getattr(_DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_K", 8)
DEFAULT_DISTANCE_CORR_SKETCH_FREQ_LOW = getattr(_DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_FREQ_LOW", 0.1)
DEFAULT_DISTANCE_CORR_SKETCH_FREQ_HIGH = getattr(_DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_FREQ_HIGH", 10.0)
DEFAULT_DISTANCE_CORR_SKETCH_FREQ_SEED = getattr(_DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_FREQ_SEED", 42)
DEFAULT_DISTANCE_CORR_SKETCH_GATE_TAU = getattr(_DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_GATE_TAU", None)
# Config-file-only (no CLI flag), matching candidate_tau/gate_tau's own
# precedent above.
DEFAULT_DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA = getattr(
    _DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA", None
)
# Independent tier-2 K^2 gate toggle -- config-file-only (no CLI flag).
DEFAULT_DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE = getattr(
    _DEFAULT_EXEC_CFG, "DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE", True
)
# Xiao (2017) online Spearman/Kendall validator -- opt-in, APPROXIMATE
# alternative to the exact per-row scipy validation path.
DEFAULT_VALIDATION_INCREMENTAL_APPROX = getattr(_DEFAULT_EXEC_CFG, "VALIDATION_INCREMENTAL_APPROX", False)
DEFAULT_VALIDATION_INCREMENTAL_M1 = getattr(_DEFAULT_EXEC_CFG, "VALIDATION_INCREMENTAL_M1", None)
DEFAULT_VALIDATION_INCREMENTAL_M2 = getattr(_DEFAULT_EXEC_CFG, "VALIDATION_INCREMENTAL_M2", None)
DEFAULT_VALIDATION_INCREMENTAL_MAX_AGE_STEPS = getattr(
    _DEFAULT_EXEC_CFG, "VALIDATION_INCREMENTAL_MAX_AGE_STEPS", 64
)
DEFAULT_VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD = getattr(
    _DEFAULT_EXEC_CFG, "VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD", 0.5
)
# SignLSHBandIndex (candidate_backend="lsh_approx", the default): precision
# is exact by construction when CANDIDATE_APPLY_DOT_GAMMA_FILTER=True (the
# default); recall is an empirical property of the data's sign-bit
# geometry, not a mathematical guarantee. Inert for every other
# candidate_backend value.
DEFAULT_CANDIDATE_LSH_N_BANDS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_LSH_N_BANDS", 64)
# (2026-09-17) ParCorr / Cole-Shasha-Zhao arm (--candidate-backend parcorr_grid)
DEFAULT_PARCORR_K = getattr(_DEFAULT_EXEC_CFG, "PARCORR_K", 2)
DEFAULT_PARCORR_F = getattr(_DEFAULT_EXEC_CFG, "PARCORR_F", 0.7)
DEFAULT_PARCORR_C = getattr(_DEFAULT_EXEC_CFG, "PARCORR_C", 0.7)
DEFAULT_PARCORR_NEIGHBOR_PROBE = getattr(_DEFAULT_EXEC_CFG, "PARCORR_NEIGHBOR_PROBE", False)
# (2026-09-17) StatStream arm (--data-representation sketch_dft --candidate-backend statstream_grid)
DEFAULT_STATSTREAM_N_COEFFS = getattr(_DEFAULT_EXEC_CFG, "STATSTREAM_N_COEFFS", 16)
DEFAULT_STATSTREAM_INDEX_DIMS = getattr(_DEFAULT_EXEC_CFG, "STATSTREAM_INDEX_DIMS", 4)
DEFAULT_STATSTREAM_APPLY_DFT_FILTER = getattr(_DEFAULT_EXEC_CFG, "STATSTREAM_APPLY_DFT_FILTER", True)
# (2026-09-17) CorrJoin arm (--data-representation sketch_paa_svd --candidate-backend corrjoin_double_filter)
DEFAULT_CORRJOIN_KS = getattr(_DEFAULT_EXEC_CFG, "CORRJOIN_KS", 15)
DEFAULT_CORRJOIN_KE = getattr(_DEFAULT_EXEC_CFG, "CORRJOIN_KE", 30)
DEFAULT_CORRJOIN_KB = getattr(_DEFAULT_EXEC_CFG, "CORRJOIN_KB", 3)
DEFAULT_CANDIDATE_APPLY_DOT_GAMMA_FILTER = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_APPLY_DOT_GAMMA_FILTER", True)
# HammingExactIndex (candidate_backend="hamming_exact"): exact packed-bit
# Hamming pre-filter, no bands/buckets. None auto-derives the Hamming touch
# threshold from gamma via the SimHash relation on first query; inert for
# every other candidate_backend value.
DEFAULT_CANDIDATE_HAMMING_THRESHOLD = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_HAMMING_THRESHOLD", None)
# SignLSHBandIndex Hamming pre-filter / budget cap. Opt-in, default off.
DEFAULT_CANDIDATE_APPLY_HAMMING_FILTER = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_APPLY_HAMMING_FILTER", False)
DEFAULT_CANDIDATE_HAMMING_FILTER_MAX_FRAC = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_HAMMING_FILTER_MAX_FRAC", 0.40)
DEFAULT_CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY", 0)

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
CANDIDATE_COSINE_THRESHOLD = DEFAULT_CANDIDATE_COSINE_THRESHOLD
HYBRID_VALIDATION = DEFAULT_HYBRID_VALIDATION
HYBRID_VALIDATION_MIN_REPEAT_RATE = DEFAULT_HYBRID_VALIDATION_MIN_REPEAT_RATE
HYBRID_VALIDATION_DISABLE_RATE = DEFAULT_HYBRID_VALIDATION_DISABLE_RATE
HYBRID_VALIDATION_EMA_ALPHA = DEFAULT_HYBRID_VALIDATION_EMA_ALPHA
HYBRID_VALIDATION_MIN_CANDIDATES = DEFAULT_HYBRID_VALIDATION_MIN_CANDIDATES
DATA_REPRESENTATION = DEFAULT_DATA_REPRESENTATION
CANDIDATE_BACKEND = DEFAULT_CANDIDATE_BACKEND
CANDIDATE_LSH_TARGET_OCCUPANCY = DEFAULT_CANDIDATE_LSH_TARGET_OCCUPANCY
VALIDATION_METRIC = DEFAULT_VALIDATION_METRIC
DIST_CORR_ALGORITHM = DEFAULT_DIST_CORR_ALGORITHM
CONCORDANCE_N_GAPS = DEFAULT_CONCORDANCE_N_GAPS
CONCORDANCE_TARGET_DIM = DEFAULT_CONCORDANCE_TARGET_DIM
CONCORDANCE_MIN_GAP = DEFAULT_CONCORDANCE_MIN_GAP
CONCORDANCE_MIN_CAPACITY = DEFAULT_CONCORDANCE_MIN_CAPACITY
CONCORDANCE_MULTICHANNEL_GAMMA = DEFAULT_CONCORDANCE_MULTICHANNEL_GAMMA
DISTANCE_CORR_SKETCH_K = DEFAULT_DISTANCE_CORR_SKETCH_K
DISTANCE_CORR_SKETCH_FREQ_LOW = DEFAULT_DISTANCE_CORR_SKETCH_FREQ_LOW
DISTANCE_CORR_SKETCH_FREQ_HIGH = DEFAULT_DISTANCE_CORR_SKETCH_FREQ_HIGH
DISTANCE_CORR_SKETCH_FREQ_SEED = DEFAULT_DISTANCE_CORR_SKETCH_FREQ_SEED
DISTANCE_CORR_SKETCH_GATE_TAU = DEFAULT_DISTANCE_CORR_SKETCH_GATE_TAU
DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA = DEFAULT_DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA
DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE = DEFAULT_DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE
VALIDATION_INCREMENTAL_APPROX = DEFAULT_VALIDATION_INCREMENTAL_APPROX
VALIDATION_INCREMENTAL_M1 = DEFAULT_VALIDATION_INCREMENTAL_M1
VALIDATION_INCREMENTAL_M2 = DEFAULT_VALIDATION_INCREMENTAL_M2
VALIDATION_INCREMENTAL_MAX_AGE_STEPS = DEFAULT_VALIDATION_INCREMENTAL_MAX_AGE_STEPS
VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD = DEFAULT_VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD
CANDIDATE_LSH_N_BANDS = DEFAULT_CANDIDATE_LSH_N_BANDS
PARCORR_K = DEFAULT_PARCORR_K
PARCORR_F = DEFAULT_PARCORR_F
PARCORR_C = DEFAULT_PARCORR_C
PARCORR_NEIGHBOR_PROBE = DEFAULT_PARCORR_NEIGHBOR_PROBE
STATSTREAM_N_COEFFS = DEFAULT_STATSTREAM_N_COEFFS
STATSTREAM_INDEX_DIMS = DEFAULT_STATSTREAM_INDEX_DIMS
STATSTREAM_APPLY_DFT_FILTER = DEFAULT_STATSTREAM_APPLY_DFT_FILTER
CORRJOIN_KS = DEFAULT_CORRJOIN_KS
CORRJOIN_KE = DEFAULT_CORRJOIN_KE
CORRJOIN_KB = DEFAULT_CORRJOIN_KB
CANDIDATE_APPLY_DOT_GAMMA_FILTER = DEFAULT_CANDIDATE_APPLY_DOT_GAMMA_FILTER
CANDIDATE_HAMMING_THRESHOLD = DEFAULT_CANDIDATE_HAMMING_THRESHOLD
CANDIDATE_APPLY_HAMMING_FILTER = DEFAULT_CANDIDATE_APPLY_HAMMING_FILTER
CANDIDATE_HAMMING_FILTER_MAX_FRAC = DEFAULT_CANDIDATE_HAMMING_FILTER_MAX_FRAC
CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY = DEFAULT_CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY
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


def _parse_window_size(value):
    # (2026-07-30) WINDOW_SIZE itself selects single- vs multi-window
    # execution now (WINDOW_SIZES dropped) -- "256" -> 256 (plain CorrTrack),
    # "64,256" -> [64, 256] (CorrTrackMultiWindow). A single-element list
    # (e.g. "256," or a one-item WINDOW_SIZE list in the config file) is
    # normalized back to a scalar so it takes the plain-CorrTrack path too.
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"invalid --window-size value: {value!r}")
    sizes = [int(p) for p in parts]
    return sizes[0] if len(sizes) == 1 else sizes


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
        "data_representation": DATA_REPRESENTATION,
        "candidate_backend": CANDIDATE_BACKEND,
        "candidate_cosine_threshold": CANDIDATE_COSINE_THRESHOLD,
        "hybrid_validation": HYBRID_VALIDATION,
        "hybrid_validation_min_repeat_rate": HYBRID_VALIDATION_MIN_REPEAT_RATE,
        "hybrid_validation_disable_rate": HYBRID_VALIDATION_DISABLE_RATE,
        "hybrid_validation_ema_alpha": HYBRID_VALIDATION_EMA_ALPHA,
        "hybrid_validation_min_candidates": HYBRID_VALIDATION_MIN_CANDIDATES,
        "validation_metric": VALIDATION_METRIC,
        "dist_corr_algorithm": DIST_CORR_ALGORITHM,
        "concordance_n_gaps": CONCORDANCE_N_GAPS,
        "concordance_target_dim": CONCORDANCE_TARGET_DIM,
        "concordance_min_gap": CONCORDANCE_MIN_GAP,
        "concordance_min_capacity": CONCORDANCE_MIN_CAPACITY,
        "concordance_multichannel_gamma": CONCORDANCE_MULTICHANNEL_GAMMA,
        "distance_corr_sketch_k": DISTANCE_CORR_SKETCH_K,
        "distance_corr_sketch_freq_low": DISTANCE_CORR_SKETCH_FREQ_LOW,
        "distance_corr_sketch_freq_high": DISTANCE_CORR_SKETCH_FREQ_HIGH,
        "distance_corr_sketch_freq_seed": DISTANCE_CORR_SKETCH_FREQ_SEED,
        "distance_corr_sketch_gate_tau": DISTANCE_CORR_SKETCH_GATE_TAU,
        "distance_corr_sketch_multichannel_gamma": DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA,
        "distance_corr_sketch_apply_tier2_gate": DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE,
        "validation_incremental_approx": VALIDATION_INCREMENTAL_APPROX,
        "validation_incremental_m1": VALIDATION_INCREMENTAL_M1,
        "validation_incremental_m2": VALIDATION_INCREMENTAL_M2,
        "validation_incremental_max_age_steps": VALIDATION_INCREMENTAL_MAX_AGE_STEPS,
        "validation_incremental_cutpoint_refresh_threshold": VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD,
        "candidate_lsh_n_bands": CANDIDATE_LSH_N_BANDS,
        "parcorr_k": PARCORR_K,
        "parcorr_f": PARCORR_F,
        "parcorr_c": PARCORR_C,
        "parcorr_neighbor_probe": PARCORR_NEIGHBOR_PROBE,
        "statstream_n_coeffs": STATSTREAM_N_COEFFS,
        "statstream_index_dims": STATSTREAM_INDEX_DIMS,
        "statstream_apply_dft_filter": STATSTREAM_APPLY_DFT_FILTER,
        "corrjoin_ks": CORRJOIN_KS,
        "corrjoin_ke": CORRJOIN_KE,
        "corrjoin_kb": CORRJOIN_KB,
        "candidate_lsh_target_occupancy": CANDIDATE_LSH_TARGET_OCCUPANCY,
        "candidate_apply_dot_gamma_filter": CANDIDATE_APPLY_DOT_GAMMA_FILTER,
        "candidate_hamming_threshold": CANDIDATE_HAMMING_THRESHOLD,
        "candidate_apply_hamming_filter": CANDIDATE_APPLY_HAMMING_FILTER,
        "candidate_hamming_filter_max_frac": CANDIDATE_HAMMING_FILTER_MAX_FRAC,
        "candidate_lsh_max_candidates_per_query": CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY,
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
    parser.add_argument(
        "--window-size",
        type=_parse_window_size,
        default=None,
        help="Single window size (e.g. 256), or a comma-separated list of >=2 sizes "
        "(e.g. 64,256) to run CorrTrackMultiWindow instead of plain CorrTrack.",
    )
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
        "--data-representation",
        default=None,
        help="Sketch representation: 'auto' (default, picks per --validation-metric: sketch_proj "
        "for pearson, sketch_concordance for spearman/kendall, sketch_multichannel for dist_corr), "
        "'raw' (no representation -- exhaustive pairwise enumeration; requires "
        "--candidate-backend auto/brute_force), 'sketch_proj', 'sketch_concordance', "
        "'sketch_multichannel', 'sketch_dft' (StatStream digest: first n DFT coefficients of the "
        "normalized window; pairs with --candidate-backend statstream_grid), or 'sketch_paa_svd' (CorrJoin: "
        "PAA_ks | PAA_ke of the normalized window; pairs with corrjoin_double_filter). Orthogonal to --candidate-backend.",
    )
    parser.add_argument(
        "--candidate-backend",
        default=None,
        help="Search/index mechanism: 'auto' (default, resolves to lsh_approx), 'lsh_approx', "
        "'hamming_exact', 'brute_force' (exhaustive; forces data-representation to 'raw'), or "
        "'parcorr_grid' (ParCorr / Cole-Shasha-Zhao competitor: sketch split into k-dim group "
        "grids, vote across a fraction f; see --parcorr-*; requires neg_corr=False). "
        "Orthogonal to --data-representation.",
    )
    parser.add_argument("--parcorr-k", type=int, default=None,
        help="parcorr_grid: sketch coordinates per group grid. Paper: 2 (n_vectors must be divisible).")
    parser.add_argument("--parcorr-f", type=float, default=None,
        help="parcorr_grid: fraction of grids that must co-locate a pair. Paper: 0.7, calibrated to 0.95 recall.")
    parser.add_argument("--parcorr-c", type=float, default=None,
        help="parcorr_grid: Cole-Shasha-Zhao distance multiplier; cell side = c*sqrt(2(1-T)). Published sweep [0.1, 1.3]; default 0.7 is a starting point to calibrate, not a result.")
    parser.add_argument("--statstream-n-coeffs", type=int, default=None,
        help="sketch_dft/statstream_grid: DFT coefficients kept (2n real dims). Paper: 16 (swept 16-40).")
    parser.add_argument("--statstream-index-dims", type=int, default=None,
        help="statstream_grid: first h dims of the DFT cube used for the grid (3^h neighbour probes). Paper gives no value; default 4.")
    parser.add_argument("--no-statstream-dft-filter", dest="statstream_apply_dft_filter", action="store_false", default=None,
        help="statstream_grid: disable the n-approximate DFT-distance post-filter (grid only; Theorem 2 anchor).")
    parser.add_argument("--corrjoin-ks", type=int, default=None,
        help="sketch_paa_svd: PAA frames feeding the SVD. Paper: 15. Must divide window_size.")
    parser.add_argument("--corrjoin-ke", type=int, default=None,
        help="sketch_paa_svd: PAA frames for the Euclidean filter. Paper: 30. Must divide window_size.")
    parser.add_argument("--corrjoin-kb", type=int, default=None,
        help="corrjoin_double_filter: SVD dimensions kept for the bucketing grid. Paper: 3.")
    parser.add_argument("--parcorr-neighbor-probe", dest="parcorr_neighbor_probe", action="store_true", default=None,
        help="parcorr_grid: Cole-Shasha-Zhao variant -- probe neighbouring cells and test the group radius (ParCorr default is same-cell only).")
    parser.add_argument(
        "--candidate-lsh-n-bands",
        type=int,
        default=None,
        help="SignLSHBandIndex: number of independent random bands (OR-retrieval -- a candidate "
        "is touched if it matches in ANY band). Default 64.",
    )
    parser.add_argument(
        "--candidate-lsh-target-occupancy",
        type=float,
        default=None,
        help="SignLSHBandIndex: target avg entries per bucket used to auto-size "
        "band_width = ceil(log2(m*L/target_occupancy)), clamped to [3, min(24, n_vectors)]. "
        "Default 3.0 (unchanged historical behavior).",
    )
    parser.add_argument(
        "--candidate-hamming-threshold",
        type=int,
        default=None,
        help="HammingExactIndex: Hamming-distance touch threshold (bits). Omit to auto-derive "
        "from gamma via the SimHash relation on first query (recommended). Only meaningful for "
        "candidate_backend=lsh_hamming_exact.",
    )
    parser.add_argument(
        "--candidate-apply-hamming-filter",
        dest="candidate_apply_hamming_filter",
        action="store_true",
        default=None,
        help="SignLSHBandIndex: cheap full-vector sign-Hamming pre-filter between "
        "prefiltered_pairs and the real dot product (same packed-bit representation as "
        "lsh_hamming_exact). Opt-in, default off. See docs/implementation_log.md's "
        "2026-07-21(a) entry (real dot products cut 33-75%% at recall preserved, max_frac "
        "0.40-0.45). Only meaningful for candidate_backend=lsh_approx (the default).",
    )
    parser.add_argument(
        "--no-candidate-apply-hamming-filter",
        dest="candidate_apply_hamming_filter",
        action="store_false",
    )
    parser.add_argument(
        "--candidate-hamming-filter-max-frac",
        type=float,
        default=None,
        help="SignLSHBandIndex: max fraction of sign bits allowed to differ to survive the "
        "Hamming pre-filter. Default 0.40. Only meaningful when "
        "--candidate-apply-hamming-filter is set.",
    )
    parser.add_argument(
        "--candidate-lsh-max-candidates-per-query",
        type=int,
        default=None,
        help="SignLSHBandIndex: per-query candidate examination budget cap (0=unlimited). "
        "Diagnosed real but data-dependent -- no safe non-zero default found. Only "
        "meaningful for candidate_backend=lsh_approx (the default).",
    )
    parser.add_argument(
        "--validation-metric",
        choices=("pearson", "spearman", "kendall", "dist_corr"),
        default=None,
        help="Part 3 (2026-07-28b): validation metric. 'pearson' (default) stays on the fast "
        "Cython bulk-validation path unchanged; the other three route through a dedicated "
        "per-row Python path.",
    )
    parser.add_argument(
        "--dist-corr-algorithm",
        choices=("naive", "fast"),
        default=None,
        help="Part 4 (2026-07-29b): 'naive' (default, O(w^2)) or 'fast' (Huo & Szekely-style "
        "exact O(w log w) -- both exact, 'fast' only wins wall-clock above ~w=200-256). Only "
        "meaningful with --validation-metric dist_corr.",
    )
    parser.add_argument(
        "--concordance-n-gaps",
        type=int,
        default=None,
        help="Explicit number of fixed multiscale gaps for the concordance sketch. Omit "
        "(default) to auto-derive from --concordance-target-dim instead, so output "
        "dimension stays roughly bounded regardless of window size. Only meaningful with "
        "data-representation=sketch_concordance.",
    )
    parser.add_argument(
        "--concordance-target-dim",
        type=int,
        default=None,
        help="Target output dimension for the concordance sketch (default 738, matching "
        "GlobalOrdinalTransformer's own default budget) -- n_gaps is derived from this so "
        "output_dim stays roughly constant across window sizes instead of scaling linearly "
        "with window_size. Ignored if --concordance-n-gaps is set explicitly.",
    )
    parser.add_argument(
        "--concordance-min-gap",
        type=int,
        default=None,
        help="Default 8 (2026-07-29f: raised from 1 -- gap=1 is both the highest-weight and, "
        "on autocorrelated real data, the worst tau estimator; see concordance_sketch.py).",
    )
    parser.add_argument(
        "--concordance-min-capacity",
        type=int,
        default=None,
        help="Default 16. Caps the largest usable gap at window_size - min_capacity, keeping "
        "every gap's per-step sample count (and thus its own cosine estimate's variance) "
        "bounded -- fixes a real accuracy problem on real data (see concordance_sketch.py's "
        "multiscale_gaps docstring and docs/implementation_log.md's 2026-07-29(f) entry).",
    )
    parser.add_argument(
        "--distance-corr-sketch-k",
        type=int,
        default=None,
        help="DistanceCorrSketchState (candidate_backend=distance_corr_sketch, requires "
        "validation_metric=dist_corr): number of random frequencies (a real, swept "
        "hyperparameter). Default 8 -- a verified sweet spot (reliable, low-variance across "
        "random-frequency seeds, and cheaper than naive exact dCor even unoptimized in "
        "Python). See distance_corr_sketch.py's module docstring.",
    )
    parser.add_argument(
        "--distance-corr-sketch-freq-low",
        type=float,
        default=None,
        help="Lower bound of the log-spaced frequency-scale range. Default 0.1.",
    )
    parser.add_argument(
        "--distance-corr-sketch-freq-high",
        type=float,
        default=None,
        help="Upper bound of the log-spaced frequency-scale range. Default 10.0.",
    )
    parser.add_argument(
        "--distance-corr-sketch-freq-seed",
        type=int,
        default=None,
        help="Random seed for the shared frequency set (fixed once per CorrTrack instance, "
        "like the base sketch's own projection-direction seed). Default 42.",
    )
    parser.add_argument(
        "--validation-incremental-approx",
        dest="validation_incremental_approx",
        action="store_true",
        default=None,
        help="Part 4 (2026-07-29a/b): Xiao (2017) online Spearman/Kendall validator -- opt-in, "
        "APPROXIMATE alternative to the exact per-row scipy path. Default off. See "
        "docs/implementation_log.md's 2026-07-29(a)/(b) entries for measured accuracy "
        "tradeoffs (0%% mismatch on stationary data, 1.23%%-5.76%% on nonstationary data "
        "depending on --validation-incremental-cutpoint-refresh-threshold).",
    )
    parser.add_argument(
        "--no-validation-incremental-approx", dest="validation_incremental_approx", action="store_false"
    )
    parser.add_argument(
        "--validation-incremental-m1",
        type=int,
        default=None,
        help="Xiao validator count-matrix size (dim 1). Omit to auto-select (30 for spearman, "
        "100 for kendall).",
    )
    parser.add_argument("--validation-incremental-m2", type=int, default=None, help="Same as m1, dim 2.")
    parser.add_argument(
        "--validation-incremental-max-age-steps", type=int, default=None, help="Default 64."
    )
    parser.add_argument(
        "--validation-incremental-cutpoint-refresh-threshold",
        type=float,
        default=None,
        help="HBR-style adaptive cutpoint refresh (2026-07-29b): proactively re-derive cutpoints "
        "once a pair's window mean drifts this many std devs from where they were last "
        "derived. Default 0.5. Set to a large value or handle via config as None to disable "
        "(reactive-only, the original 2026-07-29(a) fallback).",
    )
    parser.add_argument(
        "--candidate-apply-dot-gamma-filter",
        dest="candidate_apply_dot_gamma_filter",
        action="store_true",
        default=None,
        help="Apply the final full-dot+gamma gate after index retrieval, before validation. "
        "Only meaningful for candidate_backend=lsh_approx (the default). Default True (recall-safe). "
        "Disabling it lets every index-retrieved candidate flow straight to validation with "
        "zero dot products computed at this stage -- widens what validation must filter.",
    )
    parser.add_argument(
        "--no-candidate-apply-dot-gamma-filter",
        dest="candidate_apply_dot_gamma_filter",
        action="store_false",
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
        candidate_apply_hamming_filter=None,
        validation_incremental_approx=None,
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
    global EXEC_MODE, NEG_CORR, CORR_VAL, MONITOR, TRACK_MIN_DIST, TRAIN_RATIO, OPTIM_TUNING_MODE, ARTIFACT_MODE, ARTIFACT_BUFFER_MAX_ROWS, ARTIFACT_MERGE_MODE, SAVE_ONLY_REQUIRED_ARTIFACTS, SAVE_MAXLAG_ARTIFACTS, DATA_LOADER, MAX_WORKERS, CANDIDATE_COSINE_THRESHOLD, HYBRID_VALIDATION, HYBRID_VALIDATION_MIN_REPEAT_RATE, HYBRID_VALIDATION_DISABLE_RATE, HYBRID_VALIDATION_EMA_ALPHA, HYBRID_VALIDATION_MIN_CANDIDATES, CANDIDATE_LSH_N_BANDS, CANDIDATE_APPLY_DOT_GAMMA_FILTER, CANDIDATE_HAMMING_THRESHOLD, CANDIDATE_APPLY_HAMMING_FILTER, CANDIDATE_HAMMING_FILTER_MAX_FRAC, CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY
    global PARCORR_K, PARCORR_F, PARCORR_C, PARCORR_NEIGHBOR_PROBE
    global STATSTREAM_N_COEFFS, STATSTREAM_INDEX_DIMS, STATSTREAM_APPLY_DFT_FILTER
    global CORRJOIN_KS, CORRJOIN_KE, CORRJOIN_KB
    global DATA_REPRESENTATION, CANDIDATE_BACKEND, CANDIDATE_LSH_TARGET_OCCUPANCY, VALIDATION_METRIC, DIST_CORR_ALGORITHM, CONCORDANCE_N_GAPS, CONCORDANCE_TARGET_DIM, CONCORDANCE_MIN_GAP, CONCORDANCE_MIN_CAPACITY, CONCORDANCE_MULTICHANNEL_GAMMA, DISTANCE_CORR_SKETCH_K, DISTANCE_CORR_SKETCH_FREQ_LOW, DISTANCE_CORR_SKETCH_FREQ_HIGH, DISTANCE_CORR_SKETCH_FREQ_SEED, DISTANCE_CORR_SKETCH_GATE_TAU, DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA, DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE, VALIDATION_INCREMENTAL_APPROX, VALIDATION_INCREMENTAL_M1, VALIDATION_INCREMENTAL_M2, VALIDATION_INCREMENTAL_MAX_AGE_STEPS, VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD
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
    CANDIDATE_COSINE_THRESHOLD = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_COSINE_THRESHOLD", DEFAULT_CANDIDATE_COSINE_THRESHOLD)
    HYBRID_VALIDATION = _resolve_cfg_value(args.hybrid_validation, cfg_exec, "HYBRID_VALIDATION", DEFAULT_HYBRID_VALIDATION)
    HYBRID_VALIDATION_MIN_REPEAT_RATE = _resolve_cfg_value(args.hybrid_validation_min_repeat_rate, cfg_exec, "HYBRID_VALIDATION_MIN_REPEAT_RATE", DEFAULT_HYBRID_VALIDATION_MIN_REPEAT_RATE)
    HYBRID_VALIDATION_DISABLE_RATE = _resolve_cfg_value(args.hybrid_validation_disable_rate, cfg_exec, "HYBRID_VALIDATION_DISABLE_RATE", DEFAULT_HYBRID_VALIDATION_DISABLE_RATE)
    HYBRID_VALIDATION_EMA_ALPHA = _resolve_cfg_value(args.hybrid_validation_ema_alpha, cfg_exec, "HYBRID_VALIDATION_EMA_ALPHA", DEFAULT_HYBRID_VALIDATION_EMA_ALPHA)
    HYBRID_VALIDATION_MIN_CANDIDATES = _resolve_cfg_value(args.hybrid_validation_min_candidates, cfg_exec, "HYBRID_VALIDATION_MIN_CANDIDATES", DEFAULT_HYBRID_VALIDATION_MIN_CANDIDATES)
    DATA_REPRESENTATION = _resolve_cfg_value(args.data_representation, cfg_exec, "DATA_REPRESENTATION", DEFAULT_DATA_REPRESENTATION)
    CANDIDATE_BACKEND = _resolve_cfg_value(args.candidate_backend, cfg_exec, "CANDIDATE_BACKEND", DEFAULT_CANDIDATE_BACKEND)
    CANDIDATE_LSH_N_BANDS = _resolve_cfg_value(args.candidate_lsh_n_bands, cfg_exec, "CANDIDATE_LSH_N_BANDS", DEFAULT_CANDIDATE_LSH_N_BANDS)
    PARCORR_K = int(_resolve_cfg_value(args.parcorr_k, cfg_exec, "PARCORR_K", DEFAULT_PARCORR_K))
    PARCORR_F = float(_resolve_cfg_value(args.parcorr_f, cfg_exec, "PARCORR_F", DEFAULT_PARCORR_F))
    PARCORR_C = float(_resolve_cfg_value(args.parcorr_c, cfg_exec, "PARCORR_C", DEFAULT_PARCORR_C))
    PARCORR_NEIGHBOR_PROBE = bool(_resolve_cfg_value(args.parcorr_neighbor_probe, cfg_exec, "PARCORR_NEIGHBOR_PROBE", DEFAULT_PARCORR_NEIGHBOR_PROBE))
    STATSTREAM_N_COEFFS = int(_resolve_cfg_value(args.statstream_n_coeffs, cfg_exec, "STATSTREAM_N_COEFFS", DEFAULT_STATSTREAM_N_COEFFS))
    STATSTREAM_INDEX_DIMS = int(_resolve_cfg_value(args.statstream_index_dims, cfg_exec, "STATSTREAM_INDEX_DIMS", DEFAULT_STATSTREAM_INDEX_DIMS))
    STATSTREAM_APPLY_DFT_FILTER = bool(_resolve_cfg_value(args.statstream_apply_dft_filter, cfg_exec, "STATSTREAM_APPLY_DFT_FILTER", DEFAULT_STATSTREAM_APPLY_DFT_FILTER))
    CORRJOIN_KS = int(_resolve_cfg_value(args.corrjoin_ks, cfg_exec, "CORRJOIN_KS", DEFAULT_CORRJOIN_KS))
    CORRJOIN_KE = int(_resolve_cfg_value(args.corrjoin_ke, cfg_exec, "CORRJOIN_KE", DEFAULT_CORRJOIN_KE))
    CORRJOIN_KB = int(_resolve_cfg_value(args.corrjoin_kb, cfg_exec, "CORRJOIN_KB", DEFAULT_CORRJOIN_KB))
    CANDIDATE_LSH_TARGET_OCCUPANCY = _resolve_cfg_value(args.candidate_lsh_target_occupancy, cfg_exec, "CANDIDATE_LSH_TARGET_OCCUPANCY", DEFAULT_CANDIDATE_LSH_TARGET_OCCUPANCY)
    CANDIDATE_APPLY_DOT_GAMMA_FILTER = _resolve_cfg_value(args.candidate_apply_dot_gamma_filter, cfg_exec, "CANDIDATE_APPLY_DOT_GAMMA_FILTER", DEFAULT_CANDIDATE_APPLY_DOT_GAMMA_FILTER)
    CANDIDATE_HAMMING_THRESHOLD = _resolve_cfg_value(args.candidate_hamming_threshold, cfg_exec, "CANDIDATE_HAMMING_THRESHOLD", DEFAULT_CANDIDATE_HAMMING_THRESHOLD)
    CANDIDATE_APPLY_HAMMING_FILTER = _resolve_cfg_value(args.candidate_apply_hamming_filter, cfg_exec, "CANDIDATE_APPLY_HAMMING_FILTER", DEFAULT_CANDIDATE_APPLY_HAMMING_FILTER)
    CANDIDATE_HAMMING_FILTER_MAX_FRAC = _resolve_cfg_value(args.candidate_hamming_filter_max_frac, cfg_exec, "CANDIDATE_HAMMING_FILTER_MAX_FRAC", DEFAULT_CANDIDATE_HAMMING_FILTER_MAX_FRAC)
    CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY = _resolve_cfg_value(args.candidate_lsh_max_candidates_per_query, cfg_exec, "CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY", DEFAULT_CANDIDATE_LSH_MAX_CANDIDATES_PER_QUERY)
    VALIDATION_METRIC = _resolve_cfg_value(args.validation_metric, cfg_exec, "VALIDATION_METRIC", DEFAULT_VALIDATION_METRIC)
    DIST_CORR_ALGORITHM = _resolve_cfg_value(args.dist_corr_algorithm, cfg_exec, "DIST_CORR_ALGORITHM", DEFAULT_DIST_CORR_ALGORITHM)
    CONCORDANCE_N_GAPS = _resolve_cfg_value(args.concordance_n_gaps, cfg_exec, "CONCORDANCE_N_GAPS", DEFAULT_CONCORDANCE_N_GAPS)
    CONCORDANCE_TARGET_DIM = _resolve_cfg_value(args.concordance_target_dim, cfg_exec, "CONCORDANCE_TARGET_DIM", DEFAULT_CONCORDANCE_TARGET_DIM)
    CONCORDANCE_MIN_GAP = _resolve_cfg_value(args.concordance_min_gap, cfg_exec, "CONCORDANCE_MIN_GAP", DEFAULT_CONCORDANCE_MIN_GAP)
    CONCORDANCE_MIN_CAPACITY = _resolve_cfg_value(args.concordance_min_capacity, cfg_exec, "CONCORDANCE_MIN_CAPACITY", DEFAULT_CONCORDANCE_MIN_CAPACITY)
    CONCORDANCE_MULTICHANNEL_GAMMA = _resolve_cfg_value(None, cfg_exec, "CONCORDANCE_MULTICHANNEL_GAMMA", DEFAULT_CONCORDANCE_MULTICHANNEL_GAMMA)
    DISTANCE_CORR_SKETCH_K = _resolve_cfg_value(args.distance_corr_sketch_k, cfg_exec, "DISTANCE_CORR_SKETCH_K", DEFAULT_DISTANCE_CORR_SKETCH_K)
    DISTANCE_CORR_SKETCH_FREQ_LOW = _resolve_cfg_value(args.distance_corr_sketch_freq_low, cfg_exec, "DISTANCE_CORR_SKETCH_FREQ_LOW", DEFAULT_DISTANCE_CORR_SKETCH_FREQ_LOW)
    DISTANCE_CORR_SKETCH_FREQ_HIGH = _resolve_cfg_value(args.distance_corr_sketch_freq_high, cfg_exec, "DISTANCE_CORR_SKETCH_FREQ_HIGH", DEFAULT_DISTANCE_CORR_SKETCH_FREQ_HIGH)
    DISTANCE_CORR_SKETCH_FREQ_SEED = _resolve_cfg_value(args.distance_corr_sketch_freq_seed, cfg_exec, "DISTANCE_CORR_SKETCH_FREQ_SEED", DEFAULT_DISTANCE_CORR_SKETCH_FREQ_SEED)
    DISTANCE_CORR_SKETCH_GATE_TAU = _resolve_cfg_value(None, cfg_exec, "DISTANCE_CORR_SKETCH_GATE_TAU", DEFAULT_DISTANCE_CORR_SKETCH_GATE_TAU)
    DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA = _resolve_cfg_value(None, cfg_exec, "DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA", DEFAULT_DISTANCE_CORR_SKETCH_MULTICHANNEL_GAMMA)
    DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE = _resolve_cfg_value(None, cfg_exec, "DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE", DEFAULT_DISTANCE_CORR_SKETCH_APPLY_TIER2_GATE)
    VALIDATION_INCREMENTAL_APPROX = _resolve_cfg_value(args.validation_incremental_approx, cfg_exec, "VALIDATION_INCREMENTAL_APPROX", DEFAULT_VALIDATION_INCREMENTAL_APPROX)
    VALIDATION_INCREMENTAL_M1 = _resolve_cfg_value(args.validation_incremental_m1, cfg_exec, "VALIDATION_INCREMENTAL_M1", DEFAULT_VALIDATION_INCREMENTAL_M1)
    VALIDATION_INCREMENTAL_M2 = _resolve_cfg_value(args.validation_incremental_m2, cfg_exec, "VALIDATION_INCREMENTAL_M2", DEFAULT_VALIDATION_INCREMENTAL_M2)
    VALIDATION_INCREMENTAL_MAX_AGE_STEPS = _resolve_cfg_value(args.validation_incremental_max_age_steps, cfg_exec, "VALIDATION_INCREMENTAL_MAX_AGE_STEPS", DEFAULT_VALIDATION_INCREMENTAL_MAX_AGE_STEPS)
    VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD = _resolve_cfg_value(args.validation_incremental_cutpoint_refresh_threshold, cfg_exec, "VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD", DEFAULT_VALIDATION_INCREMENTAL_CUTPOINT_REFRESH_THRESHOLD)
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
