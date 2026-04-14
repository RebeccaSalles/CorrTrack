<img src="header.png" alt="CorrTrack" width="1100" align="center"/>

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
repo-root/
├─ README.md
├─ corrtrack_run_bruteforce.py        # Stage 1: brute-force baseline
├─ corrtrack_param_search.py          # Stage 2: CorrTrack hyper-param sweep
├─ corrtrack_run_corrtrack.py         # Stage 3: CorrTrack execution with chosen params
├─ corrtrack_compare_runs.py          # Stage 4: metrics + comparison reports
├─ debug_corrtrack.py                 # Debug runner with instrumented CorrTrack passes
├─ integrate_filcorr_results.py       # Utility to merge FilCorr CSV outputs
├─ plot_correlated_windows_example.py # Visualize correlated synthetic windows
├─ synth_corr_gen.py                  # Synthetic correlated-series generator
├─ run_corrtrack_experiment.py        # Orchestrates the four stages
├─ library_corrtrack_parallel.py      # CorrTrack implementation & shared helpers
├─ experiment_dataset_*.py            # Dataset configuration modules
├─ experiment_run_exec_param.py       # Execution defaults (windowing, thresholds, runtime)
├─ experiment_run_param_grid.py       # Hyper-parameter grid definition
├─ load_data_asos.py                  # Back-compat for the ASOS loader
└─ datasets/
   ├─ __init__.py
   ├─ asos_loader.py                  # Example dataset loader (ASOS airports CSVs)
   └─ synth_loader.py                 # Synthetic dataset loader + cache manager
```

Utility modules such as `load_data_asos.py` are retained for backwards compatibility, but the new dataset loader API lives under `datasets/`.

---
## Requirements

- **Python 3.10+** (tested with 3.12)
- Python packages: `numpy` (1.x), `pandas`, `scipy`, `scikit-learn`, `matplotlib` (for debug/plot helpers)
- Install with `pip install numpy pandas scipy scikit-learn matplotlib` (or via a `requirements.txt`).
- Ensure the project directory is on `PYTHONPATH` before running commands.

Suggested setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib
```

---

## Optional Cython Kernels

The repository includes optional Cython kernels for candidate search, sketch routines, partition utilities, and monitoring updates (`candidate_kernels.pyx`, `sketch_kernels.pyx`, `partition_kernels.pyx`, `monitor_kernels.pyx`). When compiled, they are imported automatically; otherwise the NumPy/Python implementations are used.

Build in place:

```bash
python3 -m pip install cython numpy
python3 setup_cython.py build_ext --inplace
```

To force the sketch kernel (when available), set:

```
CORRTRACK_SKETCH_KERNEL=cython
```

---

## Configuration Overview

CorrTrack relies on three configuration sources:

1. **Dataset configuration (`experiment_dataset_*.py`)**

   Defines which dataset(s) to run, where to store artifacts (`RESULT_FOLDER`), and which loader should be used. Example (`experiment_dataset_fr_air_temperature_7_1.py`):

   ```python
   from functools import partial
   from datasets.asos_loader import load_dataset

   RESULT_FOLDER = "asos_exp/tests"
   COUNTRIES = ["fr"]
   VARIABLES = ["air_temperature"]
   N_VARS = [7]
   N_YEARS = [1]

   DATA_LOADER = partial(load_dataset, root="datasets/asos-airports")  # falls back to correlation/asos-airports if present
   ```

  You can create additional dataset configs for other sources. The only requirement is that `DATA_LOADER` points to a callable that takes `(country, variable, **kwargs)` and returns `(data: np.ndarray, ids: np.ndarray)`.

Place your dataset files under `datasets/asos-airports/` using the `<country>-<variable>.csv` naming pattern (the loader caches `.npz` exports beside the CSV). If you have an existing layout under `correlation/asos-airports/`, the loader will automatically fall back to it.

> **Naming flexibility**  
> Dataset configs can export either `N_VARS`/`N_YEARS` (legacy) or the synonymous `N_SERIES`/`N_OBS`. When the optional `OBS_MODE = "count"` flag is set, CorrTrack interprets the `N_OBS` entries as absolute row counts instead of calendar years, which is handy for synthetic data. Likewise, you can replace `COUNTRIES` with a simple `DATASET` list (e.g., `["synthetic"]`) and omit `VARIABLES` entirely when no secondary grouping is needed.

2. **Execution parameters (`experiment_run_exec_param.py`)**

   Holds the run defaults shared by every stage. Example:

   ```python
   WINDOW_SIZE = 7 * 24
   WINDOW_STEP = 12
   BASIC_WINDOW = None
   N_LAGS = 7 * 24
   CORR_THRESHOLD = 0.7

   NEG_CORR = False
   CORR_VAL = True
   CORR_VAL_OPTIM = False
   TRACK_MIN_DIST = True
   MONITOR = True
   RECALL_BY_WINDOW = True
   TRAIN_RATIO = 0.3
   TARGET_RECALL = 0.95
   ARTIFACT_MODE = "buffered"
   ARTIFACT_BUFFER_MAX_ROWS = 250000
   SAVE_ONLY_REQUIRED_ARTIFACTS = True
   SAVE_MAXLAG_ARTIFACTS = False
   DELETE_MAIN_ARTIFACTS_AFTER_COMPARE = False

   PARALLEL_SKETCH = False
   PARALLEL_CANDIDATES = False
   PARALLEL_VALIDATION = None
   MAX_WORKERS = 0

   VERBOSE = False
   TESTING = False
   ```

   `CORR_VAL` controls validation for CorrTrack main/comparison runs, while `CORR_VAL_OPTIM` controls validation during hyper-parameter search only.
   `TRACK_MIN_DIST` controls min-distance bookkeeping (`pair_min_dist` / `recall_min`) for brute-force/main/compare stages.
   Hyper-parameter search always forces `TRACK_MIN_DIST=False`.
   `ARTIFACT_MODE` controls when run artifacts are flushed to disk:
   - `iterative`: append after each iteration
   - `final`: keep in memory and write once at the end
   - `buffered`: keep a bounded in-memory buffer and flush when `ARTIFACT_BUFFER_MAX_ROWS` is reached

   `ARTIFACT_BUFFER_MAX_ROWS` is a row-count threshold used only in `buffered` mode. Lower it to reduce peak memory; raise it to reduce flush frequency.

   `SAVE_ONLY_REQUIRED_ARTIFACTS=True` tells the run stages to write only the artifact family needed for the selected metric mode:
   - `RECALL_BY_WINDOW=True`: keep `_correlated.csv`, skip `_anomalies.csv` and `_status.csv`
   - `RECALL_BY_WINDOW=False`: keep `_anomalies.csv`, skip `_correlated.csv` and `_status.csv`

   `SAVE_MAXLAG_ARTIFACTS=False` skips `_max_lag_correlated.csv`. When disabled, max-lag comparison columns are reported as `nan`.

   `DELETE_MAIN_ARTIFACTS_AFTER_COMPARE=True` removes the main CorrTrack artifact bundle after `corrtrack_compare_runs.py` successfully writes the metrics CSV.

   Hyper-parameter search has an additional optimizer-only fast path: when `RECALL_BY_WINDOW=True` and `CORR_VAL_OPTIM=False`, CorrTrack trials are compared online against brute-force ground truth instead of writing trial `_correlated.csv` artifacts. In that mode, exact overall metrics are still reported, `recall_pos` / `recall_neg` are exact BF-sign-stratified recall values, and signed precision/F1 metrics are reported as `nan`.

   Execution mode is derived from the per-phase flags above: if any of `PARALLEL_*` is `True`, the run uses threads; otherwise it is sequential. You can override these values per run with CLI flags, or swap the file via `--exec-param-config`. Output location (`RESULT_FOLDER`) lives in the dataset config and can be overridden with `--result-folder`.

3. **Run hyper-parameter grid (`experiment_run_param_grid.py`)**

   Holds the parameter combinations to test during the CorrTrack optimization stage. Only `PARAM_GRID` is expected:

   ```python
   PARAM_GRID = {
       "n_vectors": [8, 16, 32, 64],
       "cell_size": [1, 2, 3],       # stretch multiplier over the base cell size
       "freq_threshold": [0.3, 0.5, 0.7],
       "preprocess": [False],
       "nodes": [0],                 # 0 => auto (use available cores)
       "seed": [2468],
       "seed_toggle": [1357],
       "grid_dimension": [1],        # must divide n_vectors
       "sketch_norm": ["mean_l2"],   # "z" or "mean_l2"
   }
   ```

   The grid is loaded by `corrtrack_param_search.py` during the hyper-parameter sweep.

   Parameter notes:
   - `cell_size` scales the base grid cell width derived from `corr_threshold` and `n_vectors`.
   - `grid_dimension` must divide `n_vectors`; set `1` for a single grid.
   - `sketch_norm` controls sketch normalization (`z` or `mean_l2`).
   
---

## Synthetic datasets

Need synthetic data for development? The repository bundles `synth_corr_gen.py`, a flexible generator that injects correlated windows on top of a configurable base process (AR(1), white noise, random walk, OU, seasonal ARIMA, lagged seasonal AR, trend polynomial, integrated seasonal, or seasonal drift). It emits an `.npz` matrix plus correlated pair metadata and a JSON summary; an optional volatility equalizer can normalize local window variance before injection.

```
python3 synth_corr_gen.py \
  --save-dir datasets/synth_outputs/synthetic \
  --m 16 --n 8000 --z 0.25 --w 96 --template-len 96 --num-templates 4 \
  --threshold 0.8 --corr-sign both \
  --base-proc '{"type":"ar1","phi":0.7,"sigma":1.0}' \
  --volatility-equalizer '{"window":96,"target_std":1.0,"min_std":0.05}' \
  --seed 123 \
  --hash-seed 0
```

To plug synthetic data into the CorrTrack pipeline, use the loader in `datasets/synth_loader.py`. It wraps the same generator and caches the results under `datasets/synth_outputs/<dataset_id>/`. A ready-to-run config lives at `experiment_dataset_synth_demo.py`; the core bits look like:

```python
from functools import partial
from datasets.synth_loader import load_dataset as load_synth_dataset

RESULT_FOLDER = "synthetic/tests"
DATASET = ["synthetic"]
N_SERIES = [8]        # must be <= m
N_OBS = [2000]        # number of rows to keep
OBS_MODE = "count"    # treat N_OBS entries as absolute row counts

SYNTH_PARAMS = {
    "m": 8,
    "n": 2000,
    "z": 0.5,
    "w": 96,
    "template_len": 96,  # defaults to w when omitted
    "num_templates": 4,
    "threshold": 0.75,
    "corr_sign": "pos",
    "base_proc": {"type": "trend_poly", "phi": 0.5, "sigma": 0.7},
    "volatility_equalizer": {"window": 96, "target_std": 1.0, "min_std": 0.05},
    "seed": 1235,
    "hash_seed": 0,
}

DATA_LOADER = partial(
    load_synth_dataset,
    cache_root="datasets/synth_outputs",
    generator_params=SYNTH_PARAMS,
    refresh=True,  # set False to reuse cached outputs
)
```

With `OBS_MODE = "count"`, CorrTrack slices the generated matrix so that `N_SERIES` controls how many series (columns 1..N) are retained while `N_OBS` limits the number of rows (always including the timestamp column at index 0). This makes it trivial to resize scenarios without regenerating the raw synthetic file. Execution defaults such as window sizes and thresholds live in `experiment_run_exec_param.py`, while output folders live in the dataset config (or can be overridden via `--result-folder`).

Synthetic configs don’t need `VARIABLES`; the optional `DATASET` list (defaulting to `["synthetic"]`) is only used to namespace cached artifacts. Need multiple scenarios? Add more dataset labels to that list or reintroduce `VARIABLES` for additional granularity. You can still override the loader from the CLI via `--loader datasets.synth_loader:load_dataset`.

#### Generator flag reference

`python3 synth_corr_gen.py` accepts the following key options:

| Flag | Description |
| --- | --- |
| `--save-dir` | Folder where every output file is written. |
| `--m` | Number of synthetic series. |
| `--n` | Length (observations) per series. |
| `--z` | Target fraction of total samples that belong to correlated pairs (0–1). |
| `--w` | Evaluation window length (must be divisible by `--template-len`). |
| `--template-len` | Length `p` of each correlated template; defaults to `--w`. |
| `--num-templates` | How many distinct templates to sample and reuse. |
| `--threshold` | Minimum Pearson `r`; actual `r*` is sampled uniformly in `[threshold, 1]`. |
| `--corr-sign` | Correlation sign to inject: `pos`, `neg`, or `both`. |
| `--base-proc` | JSON string describing the base process (e.g., `{"type":"ar1","phi":0.6,"sigma":1.0}`). |
| `--volatility-equalizer` | JSON string to normalize local window variance (e.g., `{"window":96,"target_std":1.0,"min_std":0.05}`). |
| `--seed` | RNG seed. |
| `--hash-seed` | Optional `PYTHONHASHSEED` value. |

Outputs are auto-named using  
`synt_[stat|nonstat]_<proc>_corr<rate>_m<m>_w<w>_p<p>_g<num_templates>_sign<corr_sign>_thr<threshold>`,
where the `stat`/`nonstat` tag is inferred from the `base_proc` type. They include:

- `<stem>.npz` – data matrix `(n, m+1)` where column 0 stores the 1..n index, columns 1..m store the series (`S1..Sm`).
- `<stem>_correlated.csv` – rows of `id1,id2,time1,time2,corr`.
- `<stem>_meta.json` – aggregate metadata (achieved `z`, attempts, etc.).

### Plotting correlated windows

Use `plot_correlated_windows_example.py` to visualize injected windows for any subset of series. Windows whose partners are visible in the plot share a unique color across both series (with annotations); spans whose partner is outside the view are drawn in a light neutral shade to avoid confusion.

Example command (matching the dataset generated above):

```bash
python3 plot_correlated_windows_example.py \
  --data-npz  datasets/synth_outputs/synthetic/synt_stat_ar1_corr0p25_m16_w96_p96_g4_signboth_thr0p8.npz \
  --correlated-csv  datasets/synth_outputs/synthetic/synt_stat_ar1_corr0p25_m16_w96_p96_g4_signboth_thr0p8_correlated.csv \
  --series s1,s4,s5 \
  --time-min 0 --time-max 500 \
  --window-size 96 \
  --output tmp_artifacts/correlated_windows_example.png
```

Adjust the series list and time span to highlight other segments.

---

## Global Defaults & CLI Overrides

Each stage script reads the execution defaults from `experiment_run_exec_param.py` and exposes them as CLI options. Dataset configs only control data selection. When a flag is omitted, the exec config value is used; if the exec config omits a setting, the script-level fallback applies. For example:

```
--exec-param-config     execution defaults module
--result-folder         override RESULT_FOLDER from the dataset config
--window-size           default from exec config (fallback 7*24)
--window-step           default from exec config (fallback 12)
--basic-window          default from exec config (fallback auto; divides window-size)
--n-lags                default from exec config (fallback 7*24)
--corr-threshold        default from exec config (fallback 0.7)
--parallel / --sequential (global default, see per-phase flags below)
--parallel-sketch / --sequential-sketch
--parallel-candidates / --sequential-candidates
--parallel-validation / --sequential-validation
--neg-corr / --no-neg-corr (default from exec config)
--corr-val / --no-corr-val (corrtrack + compare)
--corr-val-optim / --no-corr-val-optim (param search)
--track-min-dist / --no-track-min-dist (brute-force + corrtrack + compare)
--recall-by-window / --no-recall-by-window
--artifact-mode         stage-dependent; "iterative" | "final" | "buffered"
--artifact-buffer-max-rows  row threshold for buffered artifact flushing
--save-only-required-artifacts / --save-all-artifacts
--save-maxlag-artifacts / --no-save-maxlag-artifacts
--target-recall         (hyper-param search) default from exec config (fallback 0.95)
--recall-fallback-tolerance  (hyper-param search) if target recall is unmet, keep rows within this absolute recall distance from the best available recall; default from exec config (fallback 0.01)
--speedup-near-ratio    (hyper-param search) keep rows within this fraction of the best speedup before minimizing cand_w; default from exec config (fallback 0.98)
--train-ratio           (hyper-param search & comparison) default from exec config (fallback 0.3)
--delete-main-artifacts-after-compare / --keep-main-artifacts-after-compare
--verbose / --no-verbose
--testing / --no-testing
```

Artifacts are written under `correlation/<RESULT_FOLDER>/<dataset_id>/ws…_exec<mode>/…`, so runs with different execution policies do not collide. The execution mode is `thread` if any phase runs in parallel, otherwise `sequential`.

In buffered mode, CorrTrack spills sorted chunks to disk and merges them at the end. The final `_correlated.csv` and `_anomalies.csv` outputs are therefore stream-friendly and can be compared without loading the whole file into memory.

### Full parameter reference

Stage scripts (`corrtrack_run_bruteforce.py`, `corrtrack_param_search.py`, `corrtrack_run_corrtrack.py`, `corrtrack_compare_runs.py`, `debug_corrtrack.py`) accept the following shared options:

| Option | Description |
| --- | --- |
| `--dataset-config PATH` | Dataset configuration module (defaults to the ASOS example). |
| `--exec-param-config PATH` | Execution defaults module (defaults to `experiment_run_exec_param.py`). |
| `--result-folder NAME` | Override `RESULT_FOLDER` from the dataset config. |
| `--loader module:callable` | Optional override for the dataset loader function; defaults to the `DATA_LOADER` exported by the dataset config. |
| `--window-size`, `--window-step`, `--basic-window`, `--n-lags` | Sliding-window geometry; `basic-window` must divide `window-size` (auto when omitted). |
| `--corr-threshold` | Minimum correlation absolute value. |
| `--parallel` / `--sequential` | Global default for execution mode (applies to all phases unless overridden by per-phase flags). |
| `--parallel-sketch` / `--sequential-sketch` | Enable or disable parallelism for sketch computation only. |
| `--parallel-candidates` / `--sequential-candidates` | Enable or disable parallelism for candidate generation only. |
| `--parallel-validation` / `--sequential-validation` | Enable or disable parallelism for candidate validation only. |
| `--neg-corr` / `--no-neg-corr` | Enable or disable mining negative correlations. |
| `--track-min-dist` / `--no-track-min-dist` | Enable or disable min-distance bookkeeping (`pair_min_dist` and `recall_min` support) while keeping correlation validation on. |
| `--recall-by-window` / `--no-recall-by-window` | Whether recall is computed per time window or globally. |
| `--verbose` / `--no-verbose` | Enable or disable verbose logging in CorrTrack execution. |
| `--testing` / `--no-testing` | Enable or disable CorrTrack testing paths. |

CorrTrack validation options:

| Option | Description |
| --- | --- |
| `--corr-val` / `--no-corr-val` | Enable or skip correlation validation for CorrTrack main/comparison runs (disabling is faster but less accurate). |
| `--corr-val-optim` / `--no-corr-val-optim` | Enable or skip correlation validation during hyper-parameter search only. |
| `--track-min-dist` / `--no-track-min-dist` | Applies to brute-force/main/compare stages; hyper-parameter search always keeps this off. |

Stage-specific parameters:

| Script | Additional options |
| --- | --- |
| `corrtrack_run_bruteforce.py`, `corrtrack_run_corrtrack.py` | `--artifact-mode {iterative,final,buffered}`, `--artifact-buffer-max-rows N`, `--save-only-required-artifacts/--save-all-artifacts`, `--save-maxlag-artifacts/--no-save-maxlag-artifacts`. |
| `corrtrack_param_search.py` | `--param-grid-config PATH` (hyper-parameter grid), `--target-recall`, `--recall-fallback-tolerance`, `--speedup-near-ratio`, `--train-ratio`, `--corr-val-optim/--no-corr-val-optim`, `--artifact-mode {iterative,final,buffered}`, `--artifact-buffer-max-rows N`, `--save-only-required-artifacts/--save-all-artifacts`, `--save-maxlag-artifacts/--no-save-maxlag-artifacts`. |
| `corrtrack_compare_runs.py` | `--train-ratio` for the metrics split, `--filcorr-results` to collate FilCorr CSV outputs before comparison, `--delete-main-artifacts-after-compare/--keep-main-artifacts-after-compare`. |
| `run_corrtrack_experiment.py` | `--param-grid-config PATH` and `--base-dir PATH` to locate scripts. |
| `debug_corrtrack.py` | `--param-grid-config PATH`, `--exec-param-config PATH`, `--result-folder`, `--samples-per-class`, `--output-dir`, `--skip-initial-run`, `--refresh-artifacts`, `--artifact-mode`, `--verbose/--no-verbose`, `--testing/--no-testing`. |

Any extra flags given to `run_corrtrack_experiment.py` are filtered and forwarded only to the stages that understand them. In particular, `--corr-val` is forwarded to the CorrTrack run stage, while `--corr-val-optim` is forwarded to the hyper-parameter search stage.

### Per-phase execution control

Each CorrTrack phase can run sequentially or in threads, independently. The per-phase flags take priority over the global `--parallel/--sequential` default:

```
python3 corrtrack_run_corrtrack.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --parallel-sketch \
  --sequential-candidates \
  --parallel-validation
```

This example uses threads for sketches and validation, while keeping candidate generation sequential. The output folder will still be tagged as `exec<mode>`, where `<mode>` is `thread` whenever any phase is parallel.

---

## End-to-End Pipeline

The orchestrator invokes the four stages in order, forwarding shared overrides. Basic invocation:

```
python3 run_corrtrack_experiment.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --param-grid-config experiment_run_param_grid.py \
  --exec-param-config experiment_run_exec_param.py \
  --loader datasets.asos_loader:load_dataset
```

Or, to exercise the synthetic demo:

```
python3 run_corrtrack_experiment.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --param-grid-config experiment_run_param_grid.py \
  --exec-param-config experiment_run_exec_param.py \
  --result-folder synthetic/tests \
  --corr-threshold 0.75 \
  --sequential \
  --train-ratio 0.4 \
  --target-recall 0.9 \
  --artifact-mode final
```

### Adding overrides

> **Environment note**  
> All commands assume you are inside the repository root.

Any extra flags you pass to the orchestrator are dispatched automatically to the relevant stages. For example:

```
python3 run_corrtrack_experiment.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --param-grid-config experiment_run_param_grid.py \
  --exec-param-config experiment_run_exec_param.py \
  --loader datasets.asos_loader:load_dataset \
  --corr-threshold 0.75 \
  --sequential \
  --train-ratio 0.5 \
  --target-recall 0.9 \
  --artifact-mode final
```

* `--corr-threshold`, `--parallel/--sequential`, etc. are applied to every stage that understands them.
* `--target-recall` goes only to the hyper-parameter search.
* `--recall-fallback-tolerance` goes only to the hyper-parameter search.
* `--corr-val-optim` goes only to the hyper-parameter search; `--corr-val` goes only to the CorrTrack run when using the orchestrator.
* `--track-min-dist` is forwarded to brute-force, CorrTrack, and comparison stages; hyper-parameter search always keeps min-distance tracking disabled.
* `--train-ratio` affects the parameter search and comparison steps.
* `--artifact-mode` and `--artifact-buffer-max-rows` are forwarded to brute-force, hyper-parameter search, and CorrTrack main-run stages.
* `--save-only-required-artifacts` / `--save-all-artifacts` and `--save-maxlag-artifacts` / `--no-save-maxlag-artifacts` are forwarded to brute-force, hyper-parameter search, and CorrTrack main-run stages.
* `--delete-main-artifacts-after-compare` is forwarded only to the comparison stage.

---

## Running Individual Stages

Useful for debugging or partial reruns. Ensure the same dataset config/loader is used to keep outputs aligned.

### 1. Brute-force baseline

```
python3 corrtrack_run_bruteforce.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --exec-param-config experiment_run_exec_param.py \
  --loader datasets.asos_loader:load_dataset \
  --artifact-mode final
```

_Outputs_: `bf_run.csv` plus the artifact CSVs enabled by your artifact policy. With `SAVE_ONLY_REQUIRED_ARTIFACTS=True`, brute-force writes only `_correlated.csv` for window-based recall or only `_anomalies.csv` for timestamp-based recall; `_status.csv` is skipped, and `_max_lag_correlated.csv` is written only when `SAVE_MAXLAG_ARTIFACTS=True`. `buffered` is usually the best default for large runs; `iterative` appends after each window, while `final` writes only once at the end.

Synthetic variant:

```
python3 corrtrack_run_bruteforce.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --exec-param-config experiment_run_exec_param.py \
  --result-folder synthetic/tests \
  --window-size 168 \
  --window-step 12 \
  --basic-window 12 \
  --n-lags 168 \
  --corr-threshold 0.7 \
  --sequential \
  --neg-corr \
  --artifact-mode final \
  --recall-by-window
```

### 2. Hyper-parameter sweep

```
python3 corrtrack_param_search.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --param-grid-config experiment_run_param_grid.py \
  --exec-param-config experiment_run_exec_param.py \
  --loader datasets.asos_loader:load_dataset \
  --target-recall 0.95 \
  --train-ratio 0.3
```

Creates `optim/<dataset_id>/corrtrack_optim_<dataset_id>.csv` with one row per combination plus `best_params_*.json` summaries.

`best_params_*.json` is selected from the feasible rows (`speedup > 1` and `freq_threshold < 1`, or all rows if that feasible subset is empty), then filtered to `recall >= target_recall`. If no row reaches the recall target, the selector falls back to the best recall available and keeps every row within `RECALL_FALLBACK_TOLERANCE` of that value (CLI: `--recall-fallback-tolerance`, default `0.01`). From that recall-qualified set, the optimizer keeps only the near-best speedup band defined by `speedup >= best_speedup * SPEEDUP_NEAR_RATIO` (CLI: `--speedup-near-ratio`, default `0.98`) and then chooses the row with the lowest `cand_w`; higher `speedup` breaks ties.

When `RECALL_BY_WINDOW=True` and `CORR_VAL_OPTIM=False`, parameter search automatically switches to the optimizer-online path. In that mode:
- brute-force still builds one exact ground-truth reference
- CorrTrack trials are compared online per iteration and do not write trial `_correlated.csv` artifacts
- `artifact_time_bf` / `artifact_time` include this online reference / compare work
- `precision`, `recall`, `specificity`, `f1`, and `recall_min` remain exact
- `recall_pos` / `recall_neg` are exact BF-sign-stratified recall values
- `precision_pos`, `precision_neg`, `f1_pos`, and `f1_neg` are reported as `nan`
- `aucroc` and `pr_auc` are `nan` in the online / streamed paths

New performance diagnostics in the CSV:
- `speedup_ceil = (cand_time_bf + val_time_bf + monit_time_bf) / (sk_time + cand_time + val_time_bf * (corr_w_bf / cand_w_bf) + monit_time_bf)`
- `rel_speedup_eff = speedup / speedup_ceil`
- `corr_prop = corr_w_bf / cand_w_bf`
- `waste_val_bf = cand_w_bf / corr_w_bf`
- `waste_val = cand_w / corr_w`
- `rel_waste_red = waste_val_bf / waste_val`

Synthetic variant:

```
python3 corrtrack_param_search.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --param-grid-config experiment_run_param_grid.py \
  --exec-param-config experiment_run_exec_param.py \
  --result-folder synthetic/tests \
  --window-size 168 \
  --window-step 12 \
  --basic-window 12 \
  --n-lags 168 \
  --corr-threshold 0.7 \
  --sequential \
  --neg-corr \
  --corr-val-optim \
  --recall-by-window \
  --target-recall 0.9 \
  --train-ratio 0.4
```

### 3. CorrTrack main run

```
python3 corrtrack_run_corrtrack.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --exec-param-config experiment_run_exec_param.py \
  --loader datasets.asos_loader:load_dataset \
  --artifact-mode final
```

Runs CorrTrack using the best parameters chosen in the previous step. Results go to `corrtrack_run_<alg>.csv` and the artifact CSVs enabled by your artifact policy.

Synthetic variant:

```
python3 corrtrack_run_corrtrack.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --exec-param-config experiment_run_exec_param.py \
  --result-folder synthetic/tests \
  --window-size 168 \
  --window-step 12 \
  --basic-window 12 \
  --n-lags 168 \
  --corr-threshold 0.7 \
  --sequential \
  --neg-corr \
  --corr-val \
  --recall-by-window \
  --artifact-mode final
```

### 4. Comparison

```
python3 corrtrack_compare_runs.py \
  --dataset-config experiment_dataset_fr_air_temperature_7_1.py \
  --exec-param-config experiment_run_exec_param.py \
  --loader datasets.asos_loader:load_dataset \
  --train-ratio 0.3 \
  --filcorr-results ../correlation/asos_exp/tests/filcorr_res
```

Combines brute-force and CorrTrack outputs, producing `corrtrack_metrics_<dataset_id>.csv` with accuracy and performance metrics. Artifact comparison is streamed from the sorted CSV outputs, so large `_correlated.csv` files no longer need to be loaded fully into memory. When `--filcorr-results` is provided, the step also filters FilCorr CSVs whose names include any brute-force time-series ids, merges them via `integrate_filcorr_results.py`, and emits `filcorr_run.csv` alongside the comparison artifacts for easier downstream analysis. If `DELETE_MAIN_ARTIFACTS_AFTER_COMPARE=True` or `--delete-main-artifacts-after-compare` is set, the main CorrTrack artifact bundle is removed after the metrics CSV is written successfully.

The comparison CSV includes the same derived diagnostics as the optim CSV (`speedup_ceil`, `rel_speedup_eff`, `corr_prop`, `waste_val_bf`, `waste_val`, `rel_waste_red`) so the performance ratios stay aligned across stages. In streamed comparison mode, `aucroc` and `pr_auc` are reported as `nan`.

Synthetic variant:

```
python3 corrtrack_compare_runs.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --exec-param-config experiment_run_exec_param.py \
  --result-folder synthetic/tests \
  --window-size 168 \
  --window-step 12 \
  --basic-window 12 \
  --n-lags 168 \
  --corr-threshold 0.7 \
  --sequential \
  --neg-corr \
  --corr-val \
  --recall-by-window \
  --train-ratio 0.3
```

> Want to run the integration manually? Use the bundled helper:
>
> ```
> python3 integrate_filcorr_results.py \
>   --results-dir ../correlation/asos_exp/tests/filcorr_res \
>   --output ../correlation/asos_exp/tests/filcorr_res/filcorr_max_lag_correlated.csv \
>   --country fr \
>   --variable air_temperature
> ```

---

## Debugging with `debug_corrtrack.py`

`debug_corrtrack.py` replays the CorrTrack pipeline with extra instrumentation. It can reuse existing artifacts or refresh them, then samples representative TP/FP/TN windows and reruns CorrTrack with detailed logging.

Example:

```
python3 debug_corrtrack.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --param-grid-config experiment_run_param_grid.py \
  --exec-param-config experiment_run_exec_param.py \
  --result-folder synthetic/tests \
  --samples-per-class 5 \
  --output-dir tmp_artifacts/corrtrack_debug \
  --refresh-artifacts \
  --verbose
```

Key options:

| Option | Description |
| --- | --- |
| `--dataset-config PATH` | Dataset configuration module (required). |
| `--param-grid-config PATH` | Hyper-parameter grid (required). |
| `--exec-param-config PATH` | Execution defaults module. |
| `--result-folder NAME` | Override `RESULT_FOLDER` from the dataset config. |
| `--window-size`, `--window-step`, `--basic-window`, `--n-lags`, `--corr-threshold` | Window geometry overrides. |
| `--parallel/--sequential`, `--parallel-*` | Execution policy overrides. |
| `--neg-corr/--no-neg-corr` | Enable or disable negative correlation discovery. |
| `--extra-filter/--no-extra-filter` | Enable or disable extra filtering logic. |
| `--recall-by-window/--no-recall-by-window` | Control recall mode. |
| `--artifact-mode {iterative,final,buffered}` | Artifact persistence mode. |
| `--loader module:callable` | Override the dataset loader. |
| `--verbose/--no-verbose` | Toggle verbose logging. |
| `--testing/--no-testing` | Toggle testing paths. |
| `--samples-per-class N` | How many windows to sample per class. |
| `--output-dir PATH` | Where to write debug reports/plots. |
| `--skip-initial-run` | Skip the pipeline rerun even when artifacts are missing. |
| `--refresh-artifacts` | Rerun the full pipeline before debugging. |

---

## Profiling

### CORRTRACK_PROFILE

Set environment variables to enable lightweight timing stats inside CorrTrack (sketch dispatch/merge, grid dispatch/merge, validation, monitoring). Example:

```
CORRTRACK_PROFILE=1 \
CORRTRACK_PROFILE_EVERY=25 \
CORRTRACK_PROFILE_PATH=tmp_artifacts/corrtrack_profile.log \
python3 corrtrack_run_corrtrack.py \
  --dataset-config experiment_dataset_synth_demo.py \
  --exec-param-config experiment_run_exec_param.py \
  --artifact-mode final
```

Add `CORRTRACK_PROFILE_SILENT=1` to suppress stdout while still appending to the profile log.

---

## Swapping Datasets

1. Implement a loader function that returns `(data_array, id_array)` for each dataset.
2. Create a dataset config module that sets `DATA_LOADER` to your loader (optionally partial’ed with arguments) and enumerates the loops (countries, variables, etc.).
3. Point the CLI at the new config and loader via `--dataset-config` and (if needed) `--loader module:callable`.

No code changes are required beyond the new config and loader.

---

## Tips

* **Artifact persistence:** Use `--artifact-mode buffered` for large runs, `iterative` for per-window CSV updates, and `final` to write only once at the end.
* **Buffered flushing:** Tune `ARTIFACT_BUFFER_MAX_ROWS` or `--artifact-buffer-max-rows` to trade memory for fewer disk flushes.
* **Optimizer-online mode:** With `RECALL_BY_WINDOW=True` and `CORR_VAL_OPTIM=False`, hyper-parameter search compares trials online against BF and avoids writing trial `_correlated.csv` artifacts.
* **Result paths include execution mode** (`…_exec<mode>`), preventing collisions across runs with different execution policies.
* **Backward compatibility:** `load_data_asos.py` now defers to the new loader. Legacy scripts that import it remain functional.
* **Validation:** Each stage ensures prerequisites (loader, best params, etc.) exist and will exit with a clear message if not.

---

## License & Contributions

Feel free to adapt the loaders, configs, and parameter grids for your own datasets. Improvements are welcome—open an issue or pull request with details of new datasets or features you add.
