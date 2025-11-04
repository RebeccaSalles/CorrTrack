# CorrTrack Experiment Pipeline

This repository contains a modular pipeline for running the CorrTrack correlation engine against arbitrary datasets. The workflow is designed to let you:

* Load any dataset (via a configurable loader module)
* Run a brute-force baseline
* Perform CorrTrack hyper-parameter search
* Execute CorrTrack using the best parameters
* Compare CorrTrack’s output against the brute-force ground truth

The sections below describe the repository structure, configuration model, and how to run the pipeline end-to-end or stage-by-stage.

---

## Repository Layout

```
corrtrack_release/
├─ README.md
├─ corrtrack_run_bruteforce.py   # Stage 1: brute-force baseline
├─ corrtrack_param_search.py     # Stage 2: CorrTrack hyper-param sweep
├─ corrtrack_run_corrtrack.py    # Stage 3: CorrTrack execution with chosen params
├─ corrtrack_compare_runs.py     # Stage 4: metrics + comparison reports
├─ run_corrtrack_experiment.py   # Orchestrates the four stages
├─ library_corrtrack_parallel.py # CorrTrack implementation & shared helpers
├─ experiment_dataset_*.py       # Dataset configuration modules
├─ experiment_run_param_grid.py  # Hyper-parameter grid definition
├─ load_data_asos.py             # Back-compat shim for the ASOS loader
└─ datasets/
   ├─ __init__.py
   └─ asos_loader.py             # Example dataset loader (ASOS airports CSVs)
```

Utility modules such as `load_data_asos.py` are retained for backwards compatibility, but the new dataset loader API lives under `datasets/`.

---

## Configuration Overview

CorrTrack relies on two configuration sources:

1. **Dataset configuration (`experiment_dataset_*.py`)**

   Defines which dataset(s) to run, where to store artifacts, and which loader should be used. Example (`experiment_dataset_fr_air_temperature_7_1.py`):

   ```python
   from functools import partial
   from datasets.asos_loader import load_dataset

   RESULT_FOLDER = "asos_exp/tests"
   COUNTRIES = ["fr"]
   VARIABLES = ["air_temperature"]
   N_VARS = [7]
   N_YEARS = [1]
   MODES = ["nD"]

   DATA_LOADER = partial(load_dataset, root="datasets/asos-airports")  # falls back to correlation/asos-airports if present
   ```

  _You can create additional dataset configs for other sources. The only requirement is that `DATA_LOADER` points to a callable that takes `(country, variable, **kwargs)` and returns `(data: np.ndarray, ids: np.ndarray)`._

Place your dataset files under `datasets/asos-airports/` using the `<country>-<variable>.csv` naming pattern (the loader caches `.npz` exports beside the CSV). If you have an existing layout under `correlation/asos-airports/`, the loader will automatically fall back to it.

---
## Requirements

- **Python 3.10+** (tested with 3.12)
- Python packages: `numpy`, `pandas`, `scipy`, `scikit-learn`
- Install with `pip install numpy pandas scipy scikit-learn` (or via a `requirements.txt`).
- Ensure the project directory is on `PYTHONPATH` before running commands.

Suggested setup:

```bash
cd corrtrack_release
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn
export PYTHONPATH=$(pwd)
```

---

2. **Run hyper-parameter grid (`experiment_run_param_grid.py`)**

   Holds the parameter combinations to test during the CorrTrack optimization stage. Only `PARAM_GRID` is expected:

   ```python
   PARAM_GRID = {
       "n_vectors": [8, 16, 32, 64],
       "cell_size": [1.0],        # acts as the stretch multiplier
       "freq_threshold": [0.0, 0.5, 1.0],
       "warmup_size": [1],
       "preprocess": [True, False],
       "nodes": [0],
       "seed": [2468],
       "seed_toggle": [1357],
       "grid_dimension": [0],
   }
   ```

   The grid is loaded by `corrtrack_param_search.py` during the hyper-parameter sweep.

---

## Global Defaults & CLI Overrides

Each stage script defines default CorrTrack settings (window size, lag count, correlation threshold, execution mode, etc.) and exposes them as CLI options. For example:

```
--window-size           default 7*24
--window-step           default 12
--n-lags                default 7*24
--corr-threshold        default 0.7
--exec-mode             default "sequential"
--neg-corr / --no-neg-corr
--extra-filter / --no-extra-filter
--recall-by-window / --no-recall-by-window
--artifact-mode         (brute-force & corrtrack runs) "iterative" | "final"
--target-recall         (hyper-param search) default 0.95
--train-ratio           (hyper-param search & comparison) default 0.3
```

Artifacts are written under `<RESULT_FOLDER>/<dataset_id>/ws…_exec<mode>/…`, so runs with different exec-mode values do not collide.

### Full parameter reference

All scripts accept the following shared options:

| Option | Description |
| --- | --- |
| `--dataset-config PATH` | Dataset configuration module (defaults to the ASOS example). |
| `--loader module:callable` | Optional override for the dataset loader function; defaults to the `DATA_LOADER` exported by the dataset config. |
| `--window-size`, `--window-step`, `--basic-window`, `--n-lags` | Sliding-window geometry (defaults shown above). |
| `--corr-threshold` | Minimum correlation absolute value. |
| `--exec-mode` | Execution policy (`sequential`, `thread`, `process`). |
| `--neg-corr` / `--no-neg-corr` | Enable or disable mining negative correlations. |
| `--extra-filter` / `--no-extra-filter` | Toggle the candidate pre-filtering stage. |
| `--recall-by-window` / `--no-recall-by-window` | Whether recall is computed per time window or globally. |

Stage-specific parameters:

| Script | Additional options |
| --- | --- |
| `corrtrack_run_bruteforce.py`, `corrtrack_run_corrtrack.py` | `--artifact-mode {iterative,final}` to control when artifacts are persisted. |
| `corrtrack_param_search.py`, `run_corrtrack_experiment.py` | `--param-grid-config PATH` (hyper-parameter grid), `--target-recall`, `--train-ratio`. |
| `corrtrack_compare_runs.py` | `--train-ratio` to control the train/test split used during the metrics recomputation. |

Any extra flags given to `run_corrtrack_experiment.py` are filtered and forwarded only to the stages that understand them.

---

## End-to-End Pipeline

The orchestrator invokes the four stages in order, forwarding shared overrides. Basic invocation:

```
python3 run_corrtrack_experiment.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --param-grid-config experiment_run_param_grid.py \
  --loader datasets.asos_loader:load_dataset
```

### Adding overrides

> **Environment note**  
> All commands assume you are inside the `corrtrack_release/` directory and that the repository root is on `PYTHONPATH`.  
> Run `cd corrtrack_release` then `export PYTHONPATH=$(pwd)` (or prefix commands with `PYTHONPATH=$(pwd)`) before executing the examples below.

Any extra flags you pass to the orchestrator are dispatched automatically to the relevant stages. For example:

```
python3 run_corrtrack_experiment.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --param-grid-config experiment_run_param_grid.py \
  --loader datasets.asos_loader:load_dataset \
  --corr-threshold 0.75 \
  --exec-mode process \
  --train-ratio 0.5 \
  --target-recall 0.9 \
  --artifact-mode final
```

* `--corr-threshold`, `--exec-mode`, etc. are applied to every stage that understands them.
* `--target-recall` goes only to the hyper-parameter search.
* `--train-ratio` affects the parameter search and comparison steps.
* `--artifact-mode` is forwarded to brute-force and CorrTrack runs, switching between incremental writes (“iterative”) and single final dumps (“final”).

---

## Running Individual Stages

Useful for debugging or partial reruns. Ensure the same dataset config/loader is used to keep outputs aligned.

### 1. Brute-force baseline

```
python3 corrtrack_run_bruteforce.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --loader datasets.asos_loader:load_dataset \
  --artifact-mode final
```

_Outputs_: `bf_run.csv`, incremental artifact CSVs (correlated pairs, candidates, anomalies, etc.). Set `--artifact-mode iterative` to append after each window.

### 2. Hyper-parameter sweep

```
python3 corrtrack_param_search.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --param-grid-config experiment_run_param_grid.py \
  --loader datasets.asos_loader:load_dataset \
  --target-recall 0.95 \
  --train-ratio 0.3
```

Creates `optim/<dataset_id>/corrtrack_optim_<dataset_id>.csv` with one row per combination plus `best_params_*.json` summaries.

### 3. CorrTrack main run

```
python3 corrtrack_run_corrtrack.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --loader datasets.asos_loader:load_dataset \
  --artifact-mode final
```

Runs CorrTrack using the best parameters chosen in the previous step. Results go to `corrtrack_run_<alg>.csv` and associated artifact CSVs.

### 4. Comparison

```
python3 corrtrack_compare_runs.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --loader datasets.asos_loader:load_dataset \
  --train-ratio 0.3
```

Combines brute-force and CorrTrack outputs, producing `corrtrack_metrics_<dataset_id>.csv` with accuracy and performance metrics.

---

## Swapping Datasets

1. Implement a loader function that returns `(data_array, id_array)` for each dataset.
2. Create a dataset config module that sets `DATA_LOADER` to your loader (optionally partial’ed with arguments) and enumerates the loops (countries, variables, etc.).
3. Point the CLI at the new config and loader via `--dataset-config` and (if needed) `--loader module:callable`.

No code changes are required beyond the new config and loader.

---

## Tips

* **Incremental writing:** Use `--artifact-mode iterative` for per-window CSV updates; `final` writes only once at the end.
* **Result paths include exec-mode** (`…_exec<mode>`), preventing collisions across runs with different execution policies.
* **Backward compatibility:** `load_data_asos.py` now defers to the new loader. Legacy scripts that import it remain functional.
* **Validation:** Each stage ensures prerequisites (loader, best params, etc.) exist and will exit with a clear message if not.

---

## License & Contributions

Feel free to adapt the loaders, configs, and parameter grids for your own datasets. Improvements are welcome—open an issue or pull request with details of new datasets or features you add.
