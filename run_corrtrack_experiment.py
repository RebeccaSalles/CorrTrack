import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

DEFAULT_PARAM_GRID_CONFIG = Path(__file__).with_name(
    "experiment_run_param_grid.py"
)
DEFAULT_EXEC_PARAM_CONFIG = Path(__file__).with_name(
    "experiment_run_exec_param.py"
)
DEFAULT_DATASET_CONFIG = Path(__file__).with_name(
    "experiment_dataset_fr_air_temperature_7_1.py"
)

STEPS = [
    ("Brute-force baseline", "corrtrack_run_bruteforce.py", "brute", False),
    ("Hyper-parameter search", "corrtrack_param_search.py", "param", True),
    ("CorrTrack main run", "corrtrack_run_corrtrack.py", "corrtrack", False),
    ("Comparison report", "corrtrack_compare_runs.py", "compare", False),
]


def _load_module(config_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def run_step(
    description: str,
    script: Path,
    dataset_config: Path,
    param_grid_config: Optional[Path],
    exec_param_config: Optional[Path],
    extra_args: list[str],
):
    cmd = [sys.executable, str(script), "--dataset-config", str(dataset_config)]
    if param_grid_config is not None:
        cmd.extend(["--param-grid-config", str(param_grid_config)])
    if exec_param_config is not None:
        cmd.extend(["--exec-param-config", str(exec_param_config)])
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n[RUN] {description}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


ARG_SPECS: dict[str, tuple[int, set[str]]] = {
    "--window-size": (1, {"brute", "param", "corrtrack", "compare"}),
    "--window-step": (1, {"brute", "param", "corrtrack", "compare"}),
    "--basic-window": (1, {"brute", "param", "corrtrack", "compare"}),
    "--n-lags": (1, {"brute", "param", "corrtrack", "compare"}),
    "--corr-threshold": (1, {"brute", "param", "corrtrack", "compare"}),
    "--result-folder": (1, {"brute", "param", "corrtrack", "compare"}),
    "--verbose": (0, {"brute", "param", "corrtrack", "compare"}),
    "--no-verbose": (0, {"brute", "param", "corrtrack", "compare"}),
    "--testing": (0, {"brute", "param", "corrtrack", "compare"}),
    "--no-testing": (0, {"brute", "param", "corrtrack", "compare"}),
    "--parallel": (0, {"brute", "param", "corrtrack", "compare"}),
    "--sequential": (0, {"brute", "param", "corrtrack", "compare"}),
    "--parallel-sketch": (0, {"brute", "param", "corrtrack", "compare"}),
    "--sequential-sketch": (0, {"brute", "param", "corrtrack", "compare"}),
    "--parallel-candidates": (0, {"brute", "param", "corrtrack", "compare"}),
    "--sequential-candidates": (0, {"brute", "param", "corrtrack", "compare"}),
    "--parallel-validation": (0, {"brute", "param", "corrtrack", "compare"}),
    "--sequential-validation": (0, {"brute", "param", "corrtrack", "compare"}),
    "--neg-corr": (0, {"brute", "param", "corrtrack", "compare"}),
    "--no-neg-corr": (0, {"brute", "param", "corrtrack", "compare"}),
    "--monitor": (0, {"brute", "corrtrack", "compare"}),
    "--no-monitor": (0, {"brute", "corrtrack", "compare"}),
    "--track-min-dist": (0, {"brute", "corrtrack", "compare"}),
    "--no-track-min-dist": (0, {"brute", "corrtrack", "compare"}),
    "--baseline-mode": (1, {"brute"}),
    "--hybrid-validation": (0, {"param", "corrtrack"}),
    "--no-hybrid-validation": (0, {"param", "corrtrack"}),
    "--hybrid-validation-min-repeat-rate": (1, {"param", "corrtrack"}),
    "--hybrid-validation-disable-rate": (1, {"param", "corrtrack"}),
    "--hybrid-validation-ema-alpha": (1, {"param", "corrtrack"}),
    "--hybrid-validation-min-candidates": (1, {"param", "corrtrack"}),
    "--corr-val": (0, {"corrtrack"}),
    "--no-corr-val": (0, {"corrtrack"}),
    "--corr-val-optim": (0, {"param"}),
    "--no-corr-val-optim": (0, {"param"}),
    "--recall-by-window": (0, {"brute", "param", "corrtrack", "compare"}),
    "--no-recall-by-window": (0, {"brute", "param", "corrtrack", "compare"}),
    "--target-recall": (1, {"param"}),
    "--recall-fallback-near-ratio": (1, {"param"}),
    "--train-ratio": (1, {"brute", "param", "corrtrack", "compare"}),
    "--artifact-mode": (1, {"brute", "param", "corrtrack"}),
    "--artifact-buffer-max-rows": (1, {"brute", "param", "corrtrack"}),
    "--artifact-merge-mode": (1, {"brute", "param", "corrtrack"}),
    "--save-only-required-artifacts": (0, {"brute", "param", "corrtrack"}),
    "--save-all-artifacts": (0, {"brute", "param", "corrtrack"}),
    "--save-maxlag-artifacts": (0, {"brute", "param", "corrtrack"}),
    "--no-save-maxlag-artifacts": (0, {"brute", "param", "corrtrack"}),
    "--delete-main-artifacts-after-compare": (0, {"compare"}),
    "--keep-main-artifacts-after-compare": (0, {"compare"}),
    "--loader": (1, {"brute", "param", "corrtrack", "compare"}),
}


def split_args(extra_args: list[str]) -> dict[str, list[str]]:
    step_args = {key: [] for key in {"brute", "param", "corrtrack", "compare"}}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        spec = ARG_SPECS.get(arg)
        if spec is None:
            raise SystemExit(f"Unrecognized experiment argument '{arg}'")
        value_count, targets = spec
        values = extra_args[i + 1 : i + 1 + value_count]
        if len(values) < value_count:
            raise SystemExit(f"Argument '{arg}' expects {value_count} value(s)")
        payload = [arg, *values]
        for target in targets:
            step_args[target].extend(payload)
        i += 1 + value_count
    return step_args


def _extract_forwarded_value(extra_args: list[str], flag: str):
    i = 0
    last_value = None
    while i < len(extra_args):
        arg = extra_args[i]
        spec = ARG_SPECS.get(arg)
        if spec is None:
            raise SystemExit(f"Unrecognized experiment argument '{arg}'")
        value_count, _targets = spec
        values = extra_args[i + 1 : i + 1 + value_count]
        if len(values) < value_count:
            raise SystemExit(f"Argument '{arg}' expects {value_count} value(s)")
        if arg == flag and value_count == 1:
            last_value = values[0]
        i += 1 + value_count
    return last_value


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get_cfg_attr(cfg, *names, default=None):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def _dataset_slug(country, var):
    if var is None or var == "" or var == country:
        return str(country)
    return f"{country}_{var}"


def _extract_flag_state(extra_args: list[str], true_flag: str, false_flag: str, default=None):
    state = default
    for arg in extra_args:
        if arg == true_flag:
            state = True
        elif arg == false_flag:
            state = False
    return state


def _config_folder_slug(exec_cfg, extra_args: list[str], version_name: str) -> Path:
    window_size = _extract_forwarded_value(extra_args, "--window-size")
    if window_size is None:
        window_size = getattr(exec_cfg, "WINDOW_SIZE", "unknown")
    window_step = _extract_forwarded_value(extra_args, "--window-step")
    if window_step is None:
        window_step = getattr(exec_cfg, "WINDOW_STEP", "unknown")
    n_lags = _extract_forwarded_value(extra_args, "--n-lags")
    if n_lags is None:
        n_lags = getattr(exec_cfg, "N_LAGS", "unknown")
    corr_threshold = _extract_forwarded_value(extra_args, "--corr-threshold")
    if corr_threshold is None:
        corr_threshold = getattr(exec_cfg, "CORR_THRESHOLD", "unknown")

    parallel = _extract_flag_state(extra_args, "--parallel", "--sequential", None)
    parallel_sketch = _extract_flag_state(
        extra_args,
        "--parallel-sketch",
        "--sequential-sketch",
        getattr(exec_cfg, "PARALLEL_SKETCH", False),
    )
    parallel_candidates = _extract_flag_state(
        extra_args,
        "--parallel-candidates",
        "--sequential-candidates",
        getattr(exec_cfg, "PARALLEL_CANDIDATES", False),
    )
    parallel_validation = _extract_flag_state(
        extra_args,
        "--parallel-validation",
        "--sequential-validation",
        getattr(exec_cfg, "PARALLEL_VALIDATION", None),
    )
    if parallel is None:
        parallel = any(val is True for val in (parallel_sketch, parallel_candidates, parallel_validation))
    exec_mode = "thread" if any(
        val is True for val in (parallel, parallel_sketch, parallel_candidates, parallel_validation)
    ) else "sequential"

    if isinstance(window_size, (list, tuple)):
        window_slug = "-".join(str(v) for v in window_size)
    else:
        window_slug = str(window_size)
    threshold_slug = str(corr_threshold).replace(".", "p")
    slug = f"ws{window_slug}_step{window_step}_lags{n_lags}_thr{threshold_slug}_exec{exec_mode}"
    return Path(version_name) / slug


def _refresh_resolved_artifacts(
    dataset_config_path: Path,
    exec_param_config_path: Path,
    base_dir: Path,
    extra_args: list[str],
):
    cfg_dataset = _load_module(dataset_config_path, "experiment_dataset_refresh")
    cfg_exec = _load_module(exec_param_config_path, "experiment_exec_refresh")
    result_folder = _extract_forwarded_value(extra_args, "--result-folder")
    if result_folder is None:
        result_folder = getattr(cfg_dataset, "RESULT_FOLDER", None)
    if not result_folder:
        raise RuntimeError("Cannot refresh artifacts without RESULT_FOLDER or --result-folder.")

    countries = _as_list(_get_cfg_attr(cfg_dataset, "COUNTRIES", "DATASET", default=[]))
    variables_attr = getattr(cfg_dataset, "VARIABLES", None)
    variables = _as_list(variables_attr) if variables_attr is not None else [None]
    n_vars = _as_list(_get_cfg_attr(cfg_dataset, "N_VARS", "N_SERIES", default=[]))
    n_years = _as_list(_get_cfg_attr(cfg_dataset, "N_YEARS", "N_OBS", default=[]))
    config_rel = _config_folder_slug(cfg_exec, extra_args, base_dir.name)

    deleted = []
    for var in variables:
        for country in countries:
            slug = _dataset_slug(country, var)
            for n_year in n_years:
                for n_var in n_vars:
                    target = Path("correlation") / str(result_folder) / f"{slug}_{n_var}_{n_year}" / config_rel
                    if target.exists():
                        shutil.rmtree(target)
                        deleted.append(target)
    if deleted:
        for target in deleted:
            print(f"[REFRESH] Removed {target}", flush=True)
    else:
        print(
            f"[REFRESH] No existing artifacts matched {Path('correlation') / str(result_folder) / '*' / config_rel}",
            flush=True,
        )


def _resolved_result_dirs(
    dataset_config_path: Path,
    exec_param_config_path: Path,
    base_dir: Path,
    extra_args: list[str],
) -> list[tuple[str, Path]]:
    cfg_dataset = _load_module(dataset_config_path, "experiment_dataset_outputs")
    cfg_exec = _load_module(exec_param_config_path, "experiment_exec_outputs")
    result_folder = _extract_forwarded_value(extra_args, "--result-folder")
    if result_folder is None:
        result_folder = getattr(cfg_dataset, "RESULT_FOLDER", None)
    if not result_folder:
        return []

    countries = _as_list(_get_cfg_attr(cfg_dataset, "COUNTRIES", "DATASET", default=[]))
    variables_attr = getattr(cfg_dataset, "VARIABLES", None)
    variables = _as_list(variables_attr) if variables_attr is not None else [None]
    n_vars = _as_list(_get_cfg_attr(cfg_dataset, "N_VARS", "N_SERIES", default=[]))
    n_years = _as_list(_get_cfg_attr(cfg_dataset, "N_YEARS", "N_OBS", default=[]))
    config_rel = _config_folder_slug(cfg_exec, extra_args, base_dir.name)

    result_dirs: list[tuple[str, Path]] = []
    for var in variables:
        for country in countries:
            slug = _dataset_slug(country, var)
            for n_year in n_years:
                for n_var in n_vars:
                    dataset_id = f"{slug}_{n_var}_{n_year}"
                    result_dirs.append(
                        (
                            dataset_id,
                            Path("correlation") / str(result_folder) / dataset_id / config_rel,
                        )
                    )
    return result_dirs


def _has_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _step_outputs_complete(result_dirs: list[tuple[str, Path]], step_key: str) -> bool:
    if not result_dirs:
        return False
    for dataset_id, result_dir in result_dirs:
        if step_key == "brute":
            expected = [result_dir / "bf_run.csv"]
        elif step_key == "param":
            expected = [
                result_dir / "optim" / "best_params_corrtrack.json",
                result_dir / "optim" / f"corrtrack_optim_{dataset_id}_corrtrack.csv",
            ]
        elif step_key == "corrtrack":
            expected = [result_dir / "corrtrack_run_corrtrack.csv"]
        elif step_key == "compare":
            expected = [result_dir / f"corrtrack_metrics_{dataset_id}.csv"]
        else:
            return False
        if not all(_has_nonempty_file(path) for path in expected):
            return False
    return True


def _validate_holdout_split(exec_param_config_path: Path, extra_args: list[str]):
    cfg_exec = _load_module(exec_param_config_path, "experiment_exec_pipeline")
    train_ratio = getattr(cfg_exec, "TRAIN_RATIO", 0.3)

    override_train_ratio = _extract_forwarded_value(extra_args, "--train-ratio")
    if override_train_ratio is not None:
        train_ratio = float(override_train_ratio)

    train_ratio = float(train_ratio)
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            "Proxy-anchor CorrTrack evaluation requires a true holdout remainder for brute-force, "
            "CorrTrack, and comparison stages, "
            f"so train_ratio must be strictly between 0 and 1; got train_ratio={train_ratio}."
        )
    if train_ratio <= 0.0:
        raise ValueError(f"train_ratio must be greater than 0; got train_ratio={train_ratio}.")


def main():
    parser = argparse.ArgumentParser(description="Run the full CorrTrack experiment pipeline.")
    parser.add_argument(
        "--param-grid-config",
        type=Path,
        default=DEFAULT_PARAM_GRID_CONFIG,
        help="Path to configuration module providing PARAM_GRID (used by param search).",
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
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--testing", dest="testing", action="store_true")
    parser.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(verbose=None, testing=None)
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="Remove the resolved dataset/config result directories before running the pipeline.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing the experiment scripts (defaults to this file's directory).",
    )
    args, extra_args = parser.parse_known_args()

    param_grid_config_path = args.param_grid_config.resolve()
    if not param_grid_config_path.exists():
        raise FileNotFoundError(f"Param grid configuration file not found: {param_grid_config_path}")

    dataset_config_path = args.dataset_config.resolve()
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {dataset_config_path}")

    exec_param_config_path = args.exec_param_config.resolve()
    if not exec_param_config_path.exists():
        raise FileNotFoundError(
            f"Execution parameter configuration file not found: {exec_param_config_path}"
        )

    base_dir = args.base_dir.resolve()

    if args.result_folder is not None:
        extra_args = [*extra_args, "--result-folder", args.result_folder]
    if args.verbose is True:
        extra_args = [*extra_args, "--verbose"]
    elif args.verbose is False:
        extra_args = [*extra_args, "--no-verbose"]
    if args.testing is True:
        extra_args = [*extra_args, "--testing"]
    elif args.testing is False:
        extra_args = [*extra_args, "--no-testing"]

    _validate_holdout_split(exec_param_config_path, extra_args)
    if args.refresh_artifacts:
        _refresh_resolved_artifacts(dataset_config_path, exec_param_config_path, base_dir, extra_args)
    result_dirs = _resolved_result_dirs(dataset_config_path, exec_param_config_path, base_dir, extra_args)
    step_arg_map = split_args(extra_args)

    for description, script_name, step_key, needs_param_grid in STEPS:
        script_path = (base_dir / script_name).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")
        if not args.refresh_artifacts and _step_outputs_complete(result_dirs, step_key):
            print(f"\n[SKIP] {description}: existing artifacts found", flush=True)
            continue
        run_step(
            description,
            script_path,
            dataset_config_path,
            param_grid_config_path if needs_param_grid else None,
            exec_param_config_path,
            step_arg_map[step_key],
        )

if __name__ == "__main__":
    main()
