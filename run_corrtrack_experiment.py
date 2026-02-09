import argparse
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
    print(f"\n[RUN] {description}: {' '.join(cmd)}")
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
    "--corr-val": (0, {"param", "corrtrack"}),
    "--no-corr-val": (0, {"param", "corrtrack"}),
    "--recall-by-window": (0, {"brute", "param", "corrtrack", "compare"}),
    "--no-recall-by-window": (0, {"brute", "param", "corrtrack", "compare"}),
    "--target-recall": (1, {"param"}),
    "--train-ratio": (1, {"param", "compare"}),
    "--artifact-mode": (1, {"brute", "corrtrack"}),
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

    step_arg_map = split_args(extra_args)

    for description, script_name, step_key, needs_param_grid in STEPS:
        script_path = (base_dir / script_name).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")
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
