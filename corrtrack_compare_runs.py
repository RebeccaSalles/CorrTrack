import argparse
import ast
import csv
import importlib
import importlib.util
import os
import shutil
import numpy as np
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from library_corrtrack_parallel import CorrTrack_compare

CSV_DELIMITER = ";"

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
DEFAULT_VERBOSE = getattr(_DEFAULT_EXEC_CFG, "VERBOSE", False)
DEFAULT_TESTING = getattr(_DEFAULT_EXEC_CFG, "TESTING", False)
DEFAULT_DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = getattr(
    _DEFAULT_EXEC_CFG,
    "DELETE_MAIN_ARTIFACTS_AFTER_COMPARE",
    False,
)
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
DEFAULT_CANDIDATE_HAMMING_GROUPS = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_HAMMING_GROUPS", 8)
DEFAULT_CANDIDATE_FILTER_HAMMING = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_FILTER_HAMMING", True)
DEFAULT_CANDIDATE_FILTER_COSINE = getattr(_DEFAULT_EXEC_CFG, "CANDIDATE_FILTER_COSINE", True)

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
VERBOSE = DEFAULT_VERBOSE
TESTING = DEFAULT_TESTING
DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = DEFAULT_DELETE_MAIN_ARTIFACTS_AFTER_COMPARE
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
CANDIDATE_HAMMING_GROUPS = DEFAULT_CANDIDATE_HAMMING_GROUPS
CANDIDATE_FILTER_HAMMING = DEFAULT_CANDIDATE_FILTER_HAMMING
CANDIDATE_FILTER_COSINE = DEFAULT_CANDIDATE_FILTER_COSINE
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


def prepare_data(data, ids, n_year, n_var, train_ratio, tuning_mode="sampling"):
    rows = _select_rows(data.shape[0], n_year)
    data_stream = np.c_[data[rows, 0], data[rows, 1 : (n_var + 1)]]
    length_data = data_stream.shape[0]
    train_end = round(train_ratio * length_data)
    tuning_mode = "sampling"
    if train_end <= 0:
        raise ValueError(
            f"train_ratio={train_ratio} leaves no train data for comparison; "
            "choose a value greater than 0."
        )
    if train_end >= length_data:
        raise ValueError(
            f"train_ratio={train_ratio} leaves no holdout test data for comparison; "
            "choose a value strictly between 0 and 1."
        )
    train_data = np.transpose(data_stream[:train_end, :])
    test_data = np.transpose(data_stream[train_end:, :])
    ids_n_var = ids[: data_stream.shape[1] - 1]
    return train_data, test_data, ids_n_var


def config_folder():
    window_slug = "-".join(str(v) for v in WINDOW_SIZE) if isinstance(WINDOW_SIZE, (list, tuple)) else str(WINDOW_SIZE)
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    slug = f"ws{window_slug}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"
    return str(Path(Path(__file__).resolve().parent.name) / slug)


def parse_args():
    parser = argparse.ArgumentParser(description="Build CorrTrack comparison reports.")
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
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--testing", dest="testing", action="store_true")
    parser.add_argument("--no-testing", dest="testing", action="store_false")
    parser.add_argument(
        "--delete-main-artifacts-after-compare",
        dest="delete_main_artifacts_after_compare",
        action="store_true",
    )
    parser.add_argument(
        "--keep-main-artifacts-after-compare",
        dest="delete_main_artifacts_after_compare",
        action="store_false",
    )
    parser.add_argument(
        "--filcorr-results",
        type=Path,
        default=None,
        help=(
            "Optional directory containing FilCorr CSV outputs. When provided, "
            "the matching files are merged via integrate_filcorr_results.py and "
            "the resulting filcorr_run.csv is stored alongside the comparison artifacts."
        ),
    )
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
        verbose=None,
        testing=None,
        delete_main_artifacts_after_compare=None,
    )
    return parser.parse_args()


def _extract_timeseries_ids(bf_run_path: Path) -> set[str]:
    if not bf_run_path.exists():
        print(f"[corrtrack_compare_runs] Skipping FilCorr integration: missing {bf_run_path}")
        return set()

    with bf_run_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=CSV_DELIMITER)
        if not reader.fieldnames or "artifact_path" not in reader.fieldnames:
            print(
                f"[corrtrack_compare_runs] Skipping FilCorr integration: "
                f"'artifact_path' column not found in {bf_run_path}"
            )
            return set()

        ids: set[str] = set()
        for row in reader:
            artifact = row.get("pair_min_dist") or row.get("artifact_path")
            if not artifact:
                continue
            try:
                artifact_tuple = ast.literal_eval(artifact)
            except (ValueError, SyntaxError):
                continue
            if isinstance(artifact_tuple, (list, tuple)):
                for item in artifact_tuple[:2]:
                    if isinstance(item, str):
                        ids.add(item)
        if not ids:
            print(f"[corrtrack_compare_runs] No time-series ids were extracted from {bf_run_path}")
        return ids


def _filter_filcorr_files(results_dir: Path, ts_ids: set[str]) -> list[Path]:
    if not results_dir.exists():
        print(f"[corrtrack_compare_runs] FilCorr results directory not found: {results_dir}")
        return []
    if not ts_ids:
        return []
    matches = [
        path
        for path in sorted(results_dir.glob("*.csv"))
        if any(ts_id in path.name for ts_id in ts_ids)
    ]
    if not matches:
        ids_slug = ", ".join(sorted(ts_ids))
        print(
            f"[corrtrack_compare_runs] No FilCorr CSV files in {results_dir} matched "
            f"the extracted ids: {ids_slug}"
        )
    return matches


def _run_integrate_filcorr(
    filtered_files: list[Path],
    output_path: Path,
    country: str,
    variable: str,
) -> None:
    if not filtered_files:
        return

    script_path = Path(__file__).resolve().with_name("integrate_filcorr_results.py")
    if not script_path.exists():
        legacy_path = (
            Path(__file__).resolve().parent.parent / "correlation" / "asos_exp" / "integrate_filcorr_results.py"
        )
        if legacy_path.exists():
            script_path = legacy_path
        else:
            raise FileNotFoundError(f"integrate_filcorr_results.py not found near {script_path}")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for src_file in filtered_files:
            shutil.copy(src_file, tmp_dir / src_file.name)

        cmd = [
            sys.executable,
            str(script_path),
            "--results-dir",
            str(tmp_dir),
            "--output",
            str(output_path),
            "--country",
            country,
            "--variable",
            variable,
        ]
        env = os.environ.copy()
        repo_dir = Path(__file__).resolve().parent
        root_dir = repo_dir.parent
        extra_paths = [str(repo_dir), str(root_dir)]
        existing_py = env.get("PYTHONPATH")
        parts = [p for p in extra_paths if p]
        if existing_py:
            parts.append(existing_py)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        subprocess.run(cmd, check=True, env=env)

    print(
        f"[corrtrack_compare_runs] Integrated {len(filtered_files)} FilCorr CSV files "
        f"into {output_path}"
    )


def main():
    args = parse_args()
    cfg_exec = _load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = _load_dataset_config(args.dataset_config)
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_exec, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD, RESULT_FOLDER, MAX_WORKERS, CANDIDATE_BUCKET_WIDTH, CANDIDATE_BLOCK_SIZE_STEPS, CANDIDATE_BLOCK_INDEX_DIMS, CANDIDATE_N_PIVOTS, CANDIDATE_N_PROBE_PIVOTS, CANDIDATE_PIVOT_SELECTION, CANDIDATE_PIVOT_SEED, CANDIDATE_SIMILARITY, CANDIDATE_COSINE_THRESHOLD, CANDIDATE_HAMMING_Z, CANDIDATE_HAMMING_GROUPS, CANDIDATE_FILTER_HAMMING, CANDIDATE_FILTER_COSINE
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, CORR_VAL, MONITOR, TRACK_MIN_DIST, TRAIN_RATIO, OPTIM_TUNING_MODE, DATA_LOADER
    global VERBOSE, TESTING, DELETE_MAIN_ARTIFACTS_AFTER_COMPARE

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
    CANDIDATE_HAMMING_GROUPS = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_HAMMING_GROUPS", DEFAULT_CANDIDATE_HAMMING_GROUPS)
    CANDIDATE_FILTER_HAMMING = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_FILTER_HAMMING", DEFAULT_CANDIDATE_FILTER_HAMMING)
    CANDIDATE_FILTER_COSINE = _resolve_cfg_value(None, cfg_exec, "CANDIDATE_FILTER_COSINE", DEFAULT_CANDIDATE_FILTER_COSINE)
    VERBOSE = _resolve_cfg_value(args.verbose, cfg_exec, "VERBOSE", DEFAULT_VERBOSE)
    TESTING = _resolve_cfg_value(args.testing, cfg_exec, "TESTING", DEFAULT_TESTING)
    DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = _resolve_cfg_value(
        args.delete_main_artifacts_after_compare,
        cfg_exec,
        "DELETE_MAIN_ARTIFACTS_AFTER_COMPARE",
        DEFAULT_DELETE_MAIN_ARTIFACTS_AFTER_COMPARE,
    )
    if args.loader:
        DATA_LOADER = _load_loader(args.loader)
    if DATA_LOADER is None:
        raise RuntimeError("Dataset loader is not configured. Provide DATA_LOADER in config or --loader option.")
    if RESULT_FOLDER is None:
        raise RuntimeError("Dataset config must define RESULT_FOLDER or provide --result-folder.")

    filcorr_results_dir = args.filcorr_results.resolve() if args.filcorr_results else None

    for country, var, data, ids in iter_datasets():
        for n_year in N_YEARS:
            for n_var in N_VARS:
                slug = _dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                train_data, test_data, ids_n_var = prepare_data(
                    data,
                    ids,
                    n_year,
                    n_var,
                    TRAIN_RATIO,
                    tuning_mode=OPTIM_TUNING_MODE,
                )

                base_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder())
                bf_run_csv = os.path.join(base_dir, "bf_run.csv")
                corrtrack_run_files = {
                    alg: os.path.join(base_dir, f"corrtrack_run_{alg}.csv") for alg in MODES
                }
                output_csv = os.path.join(base_dir, f"corrtrack_metrics_{dataset_id}.csv")
                if filcorr_results_dir is not None:
                    bf_ids = _extract_timeseries_ids(Path(bf_run_csv))
                    matched = _filter_filcorr_files(filcorr_results_dir, bf_ids)
                    if matched:
                        output_filcorr = Path(base_dir) / "filcorr_run.csv"
                        _run_integrate_filcorr(matched, output_filcorr, country, var)

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
                    True,
                    NEG_CORR,
                    CORR_VAL,
                    MODES,
                    exec=EXEC_MODE,
                    max_workers=MAX_WORKERS,
                    candidate_similarity=CANDIDATE_SIMILARITY,
                    candidate_cosine_threshold=CANDIDATE_COSINE_THRESHOLD,
                    verbose=VERBOSE,
                    testing=TESTING,
                    parallel_sketch=PARALLEL_SKETCH,
                    parallel_candidates=PARALLEL_CANDIDATES,
                    parallel_validation=PARALLEL_VALIDATION,
                    monitor=MONITOR,
                    track_min_dist=TRACK_MIN_DIST,
                    tuning_mode=OPTIM_TUNING_MODE,
                )

                cc.compare_from_artifacts(
                    dataset_id,
                    bf_run_csv,
                    corrtrack_run_files,
                    output_csv,
                    delete_main_artifacts_after_compare=DELETE_MAIN_ARTIFACTS_AFTER_COMPARE,
                )


if __name__ == "__main__":
    main()
