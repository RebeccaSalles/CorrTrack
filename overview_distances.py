"""CLI for collecting CorrTrack distance distributions and histograms.

This script mirrors the dataset/exec/param-grid configuration used by the
other CorrTrack entrypoints, but writes outputs under a dedicated
`distances/` folder.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import os
import time
import traceback
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from library_corrtrack_parallel import (
    CorrTrack,
    Candidates_BF,
    _extract_feature_overrides,
    _fast_corr_and_dist,
    _is_near_constant_stats,
    _is_structurally_spiked_stats,
)

try:
    from dask import delayed, compute
    from dask.threaded import get as dask_threaded_get
    _HAS_DASK = True
except Exception:
    delayed = compute = dask_threaded_get = None
    _HAS_DASK = False


def _load_module(config_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


DEFAULT_EXEC_PARAM_CONFIG = Path(__file__).with_name("experiment_run_exec_param.py")
_DEFAULT_EXEC_CFG = _load_module(DEFAULT_EXEC_PARAM_CONFIG, "experiment_exec_defaults")

DEFAULT_PARAM_GRID_CONFIG = Path(__file__).with_name("experiment_run_param_grid.py")
DEFAULT_DATASET_CONFIG = Path(__file__).with_name("experiment_dataset_fr_air_temperature_7_1.py")
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
DEFAULT_TRAIN_RATIO = getattr(_DEFAULT_EXEC_CFG, "TRAIN_RATIO", 0.3)
DEFAULT_VERBOSE = getattr(_DEFAULT_EXEC_CFG, "VERBOSE", False)
DEFAULT_TESTING = getattr(_DEFAULT_EXEC_CFG, "TESTING", False)
DEFAULT_RESULT_FOLDER = None
DEFAULT_MAX_WORKERS = getattr(_DEFAULT_EXEC_CFG, "MAX_WORKERS", 0)

PARALLEL = DEFAULT_PARALLEL
PARALLEL_SKETCH = DEFAULT_PARALLEL_SKETCH
PARALLEL_CANDIDATES = DEFAULT_PARALLEL_CANDIDATES
PARALLEL_VALIDATION = DEFAULT_PARALLEL_VALIDATION
EXEC_MODE = DEFAULT_EXEC_MODE
NEG_CORR = DEFAULT_NEG_CORR
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
WINDOW_STEP = DEFAULT_WINDOW_STEP
BASIC_WINDOW = DEFAULT_BASIC_WINDOW
N_LAGS = DEFAULT_N_LAGS
CORR_THRESHOLD = DEFAULT_CORR_THRESHOLD
TRAIN_RATIO = DEFAULT_TRAIN_RATIO
VERBOSE = DEFAULT_VERBOSE
TESTING = DEFAULT_TESTING

RESULT_FOLDER = DEFAULT_RESULT_FOLDER
MAX_WORKERS = DEFAULT_MAX_WORKERS
COUNTRIES = VARIABLES = N_VARS = N_YEARS = None
DATA_LOADER: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None
PARAM_GRID = None
OBS_MODE = DEFAULT_OBS_MODE


def _load_param_grid(config_path: Path):
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
    threshold_slug = str(CORR_THRESHOLD).replace(".", "p")
    exec_slug = str(EXEC_MODE or "unknown").replace(" ", "_")
    return f"ws{WINDOW_SIZE}_step{WINDOW_STEP}_lags{N_LAGS}_thr{threshold_slug}_exec{exec_slug}"


class DistanceOverview:
    def __init__(self, corrtrack: CorrTrack):
        self.corrtrack = corrtrack
        self.corr_dist_pos: list[float] = []
        self.corr_dist_neg: list[float] = []
        self.noncorr_dist: list[float] = []
        self.corr_norm_dist_pos: list[float] = []
        self.corr_norm_dist_neg: list[float] = []
        self.noncorr_norm_dist: list[float] = []
        self.corr_dist_sk_pos: list[float] = []
        self.corr_dist_sk_neg: list[float] = []
        self.noncorr_dist_sk: list[float] = []
        self.corr_dist_norm_sk_pos: list[float] = []
        self.corr_dist_norm_sk_neg: list[float] = []
        self.noncorr_dist_norm_sk: list[float] = []
        self.corr_est_pos: list[float] = []
        self.corr_est_neg: list[float] = []
        self.noncorr_est: list[float] = []
        self.corr_est_diffs: list[float] = []
        self.constant_candidates = 0
        self.total_candidates = 0
        self.train_dist_time = 0.0

    def _train_distance_item(self, item):
        ct = self.corrtrack
        pair, sign = item
        try:
            id1, id2, t1, t2, w = pair
            ix = ct.series_ids[id1]
            iy = ct.series_ids[id2]

            s1 = int(t1 - ct.window_index[0])
            s2 = int(t2 - ct.window_index[0])
            x = ct.window_data[ix, s1 : s1 + w].astype(np.float64, copy=False)
            y = ct.window_data[iy, s2 : s2 + w].astype(np.float64, copy=False)
        except Exception:
            return {"counts": {"seen": 0, "skipped": 1, "constants": 0}}

        pair_corr, pair_dist, stats = _fast_corr_and_dist(x, y, return_stats=True)
        n, mean_x, mean_y, var_x, var_y = stats

        if _is_near_constant_stats(var_x, n, std_thresh=1e-3) or _is_near_constant_stats(var_y, n, std_thresh=1e-3):
            return {"counts": {"seen": 1, "skipped": 0, "constants": 1}}

        if (
            _is_structurally_spiked_stats(x, mean_x, var_x, n, kurt_thresh=5.0)
            or _is_structurally_spiked_stats(y, mean_y, var_y, n, kurt_thresh=5.0)
        ):
            return {"counts": {"seen": 1, "skipped": 1, "constants": 0}}

        nx = ct._preprocess_data(x, ix)
        ny = ct._preprocess_data(y, iy)
        norm_pair_dist = float(np.sqrt(np.sum((nx - ny) ** 2)))

        dist_sk = None
        dist_norm_sk = None
        est = None
        est_diff = None

        sk1 = None
        sk2 = None
        sketches_t1 = ct.sketches.get(t1)
        if isinstance(sketches_t1, dict):
            sk1 = sketches_t1.get((id1, t1, w))
            if sk1 is None:
                sk1 = sketches_t1.get((id1, t1))
        else:
            sk1 = ct.sketches.get((id1, t1, w))
            if sk1 is None:
                sk1 = ct.sketches.get((id1, t1))

        sketches_t2 = ct.sketches.get(t2)
        if isinstance(sketches_t2, dict):
            sk2 = sketches_t2.get((id2, t2, w))
            if sk2 is None:
                sk2 = sketches_t2.get((id2, t2))
        else:
            sk2 = ct.sketches.get((id2, t2, w))
            if sk2 is None:
                sk2 = ct.sketches.get((id2, t2))
        if sk1 is not None and sk2 is not None:
            try:
                sk1 = np.asarray(sk1, dtype=float)
                sk2 = np.asarray(sk2, dtype=float)
                dist_sk = float(np.linalg.norm(sk1 - sk2))
                n1 = np.linalg.norm(sk1)
                n2 = np.linalg.norm(sk2)
                if n1 > 0 and n2 > 0:
                    dist_norm_sk = float(np.linalg.norm(sk1 / n1 - sk2 / n2))
                    est = float(np.dot(sk1, sk2) / (n1 * n2))
                    est_diff = float(est - pair_corr)
            except Exception:
                pass

        pos = pair_corr >= ct.corr_threshold if not np.isnan(pair_corr) else False
        neg = pair_corr <= -ct.corr_threshold if not np.isnan(pair_corr) else False

        payload = {
            "dists_pos": [],
            "dists_neg": [],
            "dists_nonc": [],
            "dists_norm_pos": [],
            "dists_norm_neg": [],
            "dists_norm_nonc": [],
            "dists_sk_pos": [],
            "dists_sk_neg": [],
            "dists_sk_nonc": [],
            "dists_norm_sk_pos": [],
            "dists_norm_sk_neg": [],
            "dists_norm_sk_nonc": [],
            "est_pos": [],
            "est_neg": [],
            "est_nonc": [],
            "est_diffs": [],
            "counts": {"seen": 1, "skipped": 0, "constants": 0},
        }

        if pos:
            payload["dists_pos"].append(pair_dist)
            payload["dists_norm_pos"].append(norm_pair_dist)
        elif neg:
            payload["dists_neg"].append(pair_dist)
            payload["dists_norm_neg"].append(norm_pair_dist)
        else:
            payload["dists_nonc"].append(pair_dist)
            payload["dists_norm_nonc"].append(norm_pair_dist)

        if dist_sk is not None:
            if pos:
                payload["dists_sk_pos"].append(dist_sk)
            elif neg:
                payload["dists_sk_neg"].append(dist_sk)
            else:
                payload["dists_sk_nonc"].append(dist_sk)

        if dist_norm_sk is not None:
            if pos:
                payload["dists_norm_sk_pos"].append(dist_norm_sk)
            elif neg:
                payload["dists_norm_sk_neg"].append(dist_norm_sk)
            else:
                payload["dists_norm_sk_nonc"].append(dist_norm_sk)

        if est is not None:
            if pos:
                payload["est_pos"].append(est)
            elif neg:
                payload["est_neg"].append(est)
            else:
                payload["est_nonc"].append(est)
        if est_diff is not None:
            payload["est_diffs"].append(est_diff)

        return payload

    def _merge_train_distances(self, shard_payloads):
        for p in shard_payloads:
            if not p:
                continue

            self.corr_dist_pos.extend(p.get("dists_pos", []))
            self.corr_dist_neg.extend(p.get("dists_neg", []))
            self.noncorr_dist.extend(p.get("dists_nonc", []))

            self.corr_norm_dist_pos.extend(p.get("dists_norm_pos", []))
            self.corr_norm_dist_neg.extend(p.get("dists_norm_neg", []))
            self.noncorr_norm_dist.extend(p.get("dists_norm_nonc", []))

            self.corr_dist_sk_pos.extend(p.get("dists_sk_pos", []))
            self.corr_dist_sk_neg.extend(p.get("dists_sk_neg", []))
            self.noncorr_dist_sk.extend(p.get("dists_sk_nonc", []))

            self.corr_dist_norm_sk_pos.extend(p.get("dists_norm_sk_pos", []))
            self.corr_dist_norm_sk_neg.extend(p.get("dists_norm_sk_neg", []))
            self.noncorr_dist_norm_sk.extend(p.get("dists_norm_sk_nonc", []))

            self.corr_est_pos.extend(p.get("est_pos", []))
            self.corr_est_neg.extend(p.get("est_neg", []))
            self.noncorr_est.extend(p.get("est_nonc", []))
            self.corr_est_diffs.extend(p.get("est_diffs", []))

            counts = p.get("counts", {})
            self.constant_candidates += counts.get("constants", 0)
            self.total_candidates += counts.get("seen", 0)

    def _train_distances(self, worker_mode=None):
        ct = self.corrtrack
        if not getattr(ct, "candidates", None):
            return

        items = list(ct.candidates.items())
        worker_mode = worker_mode if worker_mode else ct.exec

        shard_payloads = ct._parallel_map(
            self._train_distance_item,
            items,
            mode=worker_mode,
            max_workers=ct.n_nodes,
            preserve_order=False,
        )
        self._merge_train_distances(shard_payloads)

    def run_step(self, new_data_step, ids, verbose=False, testing=False):
        ct = self.corrtrack
        ct.verbose = verbose
        ct.testing = testing

        sketch_mode = "thread" if ct.parallel_sketch else "sequential"
        cand_mode = "thread" if ct.parallel_candidates else "sequential"
        val_mode = "thread" if ct.parallel_validation else "sequential"

        ct._update_curr_data(new_data_step, ids)

        start_time = time.time()
        sketches = ct._get_sketches(ct._curr_window_step(), verbose, testing, worker_mode=sketch_mode)
        ct.sketch_time += time.time() - start_time
        ct._update_curr_sketches(sketches)

        ct.candidates = {}
        start_time = time.time()
        if not ct.brute_force_nodes:
            ct.brute_force_nodes.append(Candidates_BF(ct.window_size, ct.window_step, ct.n_lags, ct.corr_threshold))
        if ct.parallel_candidates:
            ct.candidates = ct._run_bf_parallel(ct._curr_window_step(), ct.ids, worker_mode=cand_mode)
        else:
            ct.candidates = ct.brute_force_nodes[0].run(ct._curr_window_step(), ct.ids, verbose, testing, ref_ids=None)
        ct.candidate_time += time.time() - start_time

        start_time = time.time()
        self._train_distances(worker_mode=val_mode)
        self.train_dist_time += time.time() - start_time


def _plot_corrtrack_histograms(distances: DistanceOverview, corrtrack: CorrTrack, train_data, output_dir, save=True):
    bins = 50
    r = corrtrack.n_vectors

    fig, axes = plt.subplots(3, 6, figsize=(60, 10), sharex=False)
    axes = axes.flatten()

    dist_all = distances.corr_dist_pos + distances.corr_dist_neg + distances.noncorr_dist
    x_min = min(dist_all)
    x_max = max(dist_all)

    axes[0].hist(distances.corr_dist_pos, bins=bins, color="green", alpha=0.7)
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_title("Distances for Positive Correlation (> 0.7)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    axes[6].hist(distances.corr_dist_neg, bins=bins, color="red", alpha=0.7)
    axes[6].set_xlim(x_min, x_max)
    axes[6].set_title("Distances for Negative Correlation (< -0.7)")
    axes[6].set_ylabel("Frequency")
    axes[6].grid(True)

    axes[12].hist(distances.noncorr_dist, bins=bins, color="blue", alpha=0.7)
    axes[12].set_xlim(x_min, x_max)
    axes[12].set_title("Distances for Non-Correlated (|ρ| < 0.7)")
    axes[12].set_ylabel("Frequency")
    axes[12].grid(True)

    dist_all = distances.corr_norm_dist_pos + distances.corr_norm_dist_neg + distances.noncorr_norm_dist
    x_min = min(dist_all)
    x_max = max(dist_all)

    axes[1].hist(distances.corr_norm_dist_pos, bins=bins, color="green", alpha=0.7)
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_title("Normalized distances for Positive Correlation (> 0.7)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    axes[7].hist(distances.corr_norm_dist_neg, bins=bins, color="red", alpha=0.7)
    axes[7].set_xlim(x_min, x_max)
    axes[7].set_title("Normalized distances for Negative Correlation (< -0.7)")
    axes[7].set_ylabel("Frequency")
    axes[7].grid(True)

    axes[13].hist(distances.noncorr_norm_dist, bins=bins, color="blue", alpha=0.7)
    axes[13].set_xlim(x_min, x_max)
    axes[13].set_title("Normalized distances for Non-Correlated (|ρ| < 0.7)")
    axes[13].set_ylabel("Frequency")
    axes[13].grid(True)

    dist_all = distances.corr_dist_sk_pos + distances.corr_dist_sk_neg + distances.noncorr_dist_sk
    x_min = min(dist_all)
    x_max = max(dist_all)

    axes[2].hist(distances.corr_dist_sk_pos, bins=bins, color="green", alpha=0.7)
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_title("Sketch Distances for Positive Correlation (> 0.7)")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True)

    axes[8].hist(distances.corr_dist_sk_neg, bins=bins, color="red", alpha=0.7)
    axes[8].set_xlim(x_min, x_max)
    axes[8].set_title("Sketch Distances for Negative Correlation (< -0.7)")
    axes[8].set_ylabel("Frequency")
    axes[8].grid(True)

    axes[14].hist(distances.noncorr_dist_sk, bins=bins, color="blue", alpha=0.7)
    axes[14].set_xlim(x_min, x_max)
    axes[14].set_title("Sketch Distances for Non-Correlated (|ρ| < 0.7)")
    axes[14].set_ylabel("Frequency")
    axes[14].grid(True)

    dist_all = distances.corr_dist_norm_sk_pos + distances.corr_dist_norm_sk_neg + distances.noncorr_dist_norm_sk
    x_min = min(dist_all)
    x_max = max(dist_all)

    axes[3].hist(distances.corr_dist_norm_sk_pos, bins=bins, color="green", alpha=0.7)
    axes[3].set_xlim(x_min, x_max)
    axes[3].set_title("Normalized Sketch Distances for Positive Correlation (> 0.7)")
    axes[3].set_ylabel("Frequency")
    axes[3].grid(True)

    axes[9].hist(distances.corr_dist_norm_sk_neg, bins=bins, color="red", alpha=0.7)
    axes[9].set_xlim(x_min, x_max)
    axes[9].set_title("Normalized Sketch Distances for Negative Correlation (< -0.7)")
    axes[9].set_ylabel("Frequency")
    axes[9].grid(True)

    axes[15].hist(distances.noncorr_dist_norm_sk, bins=bins, color="blue", alpha=0.7)
    axes[15].set_xlim(x_min, x_max)
    axes[15].set_title("Normalized Sketch Distances for Non-Correlated (|ρ| < 0.7)")
    axes[15].set_ylabel("Frequency")
    axes[15].grid(True)

    dist_all = distances.corr_est_pos + distances.corr_est_neg + distances.noncorr_est
    x_min = min(dist_all)
    x_max = max(dist_all)

    axes[4].hist(distances.corr_est_pos, bins=bins, color="green", alpha=0.7)
    axes[4].set_xlim(x_min, x_max)
    axes[4].set_title("Cosine Similarity for Positive Correlation (> 0.7)")
    axes[4].set_ylabel("Frequency")
    axes[4].grid(True)

    axes[10].hist(distances.corr_est_neg, bins=bins, color="red", alpha=0.7)
    axes[10].set_xlim(x_min, x_max)
    axes[10].set_title("Cosine Similarity for Negative Correlation (< -0.7)")
    axes[10].set_ylabel("Frequency")
    axes[10].grid(True)

    axes[16].hist(distances.noncorr_est, bins=bins, color="blue", alpha=0.7)
    axes[16].set_xlim(x_min, x_max)
    axes[16].set_title("Cosine Similarity for Non-Correlated (|ρ| < 0.7)")
    axes[16].set_ylabel("Frequency")
    axes[16].grid(True)

    axes[5].hist(distances.corr_est_diffs, bins=bins, color="purple", alpha=0.7)
    axes[5].set_title("Correlation Estimate Differences (estimated − true)")
    axes[5].set_xlabel("Value")
    axes[5].set_ylabel("Frequency")
    axes[5].grid(True)

    fig.suptitle(f"Distances and Correlation Estimation (n_vectors = {r})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    n_ts = train_data.shape[0] - 1
    n_w = int(np.floor((train_data.shape[1] - corrtrack.window_size) / corrtrack.window_step) + 1)
    if save:
        hist_dir = os.path.join(output_dir, "hist")
        os.makedirs(hist_dir, exist_ok=True)
        save_path = os.path.join(hist_dir, f"corrtrack_hist_{n_ts}_{n_w}_r{r}.png")
        plt.savefig(save_path)
        print(f"[INFO] Histogram figure saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _run_is_parallel(exec_mode: str) -> bool:
    return exec_mode != "sequential"


def _outer_iter(items, worker_fn, exec_mode, max_workers=0, unordered=True):
    items = list(items)
    inner_parallel = _run_is_parallel(exec_mode)
    if inner_parallel or exec_mode == "sequential" or not _HAS_DASK or delayed is None:
        for it in items:
            yield worker_fn(it)
        return

    max_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
    max_workers = max(1, max_workers)
    scheduler_get = dask_threaded_get
    if scheduler_get is None:
        for it in items:
            yield worker_fn(it)
        return

    scheduler = lambda dsk, keys: scheduler_get(dsk, keys, num_workers=max_workers)
    try:
        tasks = [delayed(worker_fn)(it) for it in items]
        results = compute(*tasks, scheduler=scheduler)
    except Exception as exc:
        print(f"[outer_iter] Dask fallback to sequential due to: {exc}")
        for it in items:
            yield worker_fn(it)
        return

    for res in results:
        yield res


def _run_corrtrack_distances(param_combo, train_data, ids, output_dir):
    try:
        window_size = WINDOW_SIZE
        window_step = WINDOW_STEP
        basic_window = BASIC_WINDOW
        n_lags = N_LAGS
        corr_threshold = CORR_THRESHOLD

        seed = param_combo["seed"]
        seed_toggle = param_combo["seed_toggle"]
        n_vectors = param_combo["n_vectors"]
        preprocess = param_combo["preprocess"]

        feature_kwargs = _extract_feature_overrides(param_combo)

        corrtrack = CorrTrack(
            window_size=window_size,
            basic_window=basic_window,
            window_step=window_step,
            n_vectors=n_vectors,
            n_lags=n_lags,
            seed=seed,
            seed_toggle=seed_toggle,
            freq_threshold=0,
            corr_threshold=corr_threshold,
            neg_corr=NEG_CORR,
            preprocess=preprocess,
            exec=EXEC_MODE,
            max_workers=MAX_WORKERS,
            parallel_sketch=PARALLEL_SKETCH,
            parallel_candidates=PARALLEL_CANDIDATES,
            parallel_validation=PARALLEL_VALIDATION,
            **feature_kwargs,
        )

        distances = DistanceOverview(corrtrack)

        length_data = train_data.shape[1]
        for start in range(0, length_data - window_step + 1, window_step):
            chunk = train_data[:, start : (start + window_step)]
            distances.run_step(chunk, ids, verbose=VERBOSE, testing=TESTING)

        _plot_corrtrack_histograms(distances, corrtrack, train_data, output_dir)

        return {
            "params": dict(param_combo),
            "n_vectors": n_vectors,
            "corr_dist_sk_pos": float(np.quantile(distances.corr_dist_sk_pos, 0.99)),
            "corr_dist_norm_sk_pos": float(np.quantile(distances.corr_dist_norm_sk_pos, 0.99)),
            "corr_est_pos": float(np.quantile(distances.corr_est_pos, 0.01)),
        }

    except Exception as exc:
        print(f"Skipped params={param_combo} due to error: {exc}")
        traceback.print_exc()
        return None


def _train_distances(param_grid_all, train_data, ids, output_dir):
    param_grid = {k: v for k, v in param_grid_all.items() if k not in ["grid_dimension", "cell_size", "freq_threshold"]}

    names = list(param_grid.keys())
    tasks = [dict(zip(names, vals)) for vals in itertools.product(*param_grid.values())]

    thresholds = []
    i = 0
    for result in _outer_iter(tasks, lambda params: _run_corrtrack_distances(params, train_data, ids, output_dir), EXEC_MODE, MAX_WORKERS, unordered=True):
        if result:
            n_vectors = result["n_vectors"]
            print("Run train distances ", i, "/", len(tasks))
            print(
                "[n_vectors = ",
                n_vectors,
                "] - Quantiles 99%:\n",
                "Dist sketch:",
                result["corr_dist_sk_pos"],
                "Dist norm sketch:",
                result["corr_dist_norm_sk_pos"],
                "Cosine similarity:",
                result["corr_est_pos"],
            )
            thresholds.append(result)
            i += 1
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Run CorrTrack distance overview with brute-force candidates.")
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
        verbose=None,
        testing=None,
    )
    parser.add_argument("--train-ratio", type=float, default=None)
    args = parser.parse_args()

    cfg_exec = _load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = _load_module(args.dataset_config, "experiment_dataset")
    _apply_dataset_config(cfg_dataset)
    _apply_parallel_defaults_from_cfg(cfg_exec, args)

    global WINDOW_SIZE, WINDOW_STEP, BASIC_WINDOW, N_LAGS, CORR_THRESHOLD
    global PARALLEL, PARALLEL_SKETCH, PARALLEL_CANDIDATES, PARALLEL_VALIDATION
    global EXEC_MODE, NEG_CORR, TRAIN_RATIO, PARAM_GRID, DATA_LOADER, RESULT_FOLDER, MAX_WORKERS
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
    TRAIN_RATIO = _resolve_cfg_value(args.train_ratio, cfg_exec, "TRAIN_RATIO", DEFAULT_TRAIN_RATIO)
    MAX_WORKERS = _resolve_cfg_value(None, cfg_exec, "MAX_WORKERS", DEFAULT_MAX_WORKERS)
    VERBOSE = _resolve_cfg_value(args.verbose, cfg_exec, "VERBOSE", DEFAULT_VERBOSE)
    TESTING = _resolve_cfg_value(args.testing, cfg_exec, "TESTING", DEFAULT_TESTING)
    PARAM_GRID = _load_param_grid(args.param_grid_config)
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

                base_dir = os.path.join("correlation", RESULT_FOLDER, dataset_id, config_folder(), "distances")
                os.makedirs(base_dir, exist_ok=True)

                results = _train_distances(PARAM_GRID, train_data, ids_n_var, base_dir)
                summary_path = os.path.join(base_dir, f"corrtrack_distances_{dataset_id}.json")
                with open(summary_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "dataset_id": dataset_id,
                            "window_size": WINDOW_SIZE,
                            "window_step": WINDOW_STEP,
                            "n_lags": N_LAGS,
                            "corr_threshold": CORR_THRESHOLD,
                            "results": results,
                        },
                        handle,
                        indent=2,
                    )
                print(f"[INFO] Distance summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
