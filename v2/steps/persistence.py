"""Step 7 — persisting the results to disk (CSV).

Writes into `output_dir`:
  * correlated.csv : id1, t1, id2, t2, lag, corr
  * episodes.csv   : id1, id2, lag, start, last, length, sign   (if monitoring)
  * anomalies.csv  : id1, id2, lag, time, kind                  (if monitoring)
  * summary.csv    : metric, value (counters + runtime + per-phase timings)
"""

import csv
import os


def save_summary(output_dir, summary):
    """Write summary.csv: one (metric, value) row per useful datum of the run."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key, value in summary.items():
            w.writerow([key, value])


def _save_timeseries(output_dir, filename, rows, columns):
    """Write a time series (list of dicts) into a CSV. A header row alone when
    `rows` is empty (useful for viz: the file exists, the curve is empty)."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])


_THROUGHPUT_COLS = [
    "t", "frac", "w_done", "n_candidates", "n_tested", "n_correlated",
    "windows_per_s", "windows_per_s_cum",
    "sketch_per_s", "sketch_per_s_cum",
    "candidate_per_s", "candidate_per_s_cum",
    "validate_per_s", "validate_per_s_cum",
    "monitor_per_s", "monitor_per_s_cum",
]
_RESOURCES_COLS = ["t", "cpu_pct", "mem_mb", "power_cores"]


def save_throughput(output_dir, rows):
    """throughput.csv: the throughput = f(time) curve (one point per checkpoint).

    Columns: wall time `t`, processed fraction, cumulative counters, and for
    each stream (windows/sketch/candidate/validate/monitor) the instantaneous
    throughput `<stream>_per_s` and the cumulative one `<stream>_per_s_cum`."""
    _save_timeseries(output_dir, "throughput.csv", rows, _THROUGHPUT_COLS)


def save_resources(output_dir, rows):
    """resources.csv: time series of CPU %, RSS memory (MB), energy proxy
    (busy cores). Empty/header only when psutil is unavailable."""
    _save_timeseries(output_dir, "resources.csv", rows, _RESOURCES_COLS)


_PHASE_METRICS_COLS = [
    "phase", "calls", "time_total",
    "time_min", "time_median", "time_max", "time_mean", "time_std",
    "time_p05", "time_p95",
    "n_in_total", "n_out_total", "selectivity",
    "in_overall", "in_min", "in_median", "in_max",
    "out_overall", "out_min", "out_median", "out_max",
]


def save_phase_metrics(output_dir, rows):
    """phase_metrics.csv: one row per STEP (sketch/index/select/
    validate_sketches/validate/monitor…).

    For each step: number of calls, total time, distribution of the time PER
    CALL (min/median/max/mean/std/p05/p95), cumulated input/output elements,
    selectivity (output/input) and input/output throughput (effective `overall`
    = Σelements/Σtime, plus min/median/max of the per-call throughput). This is
    the basis for computing the per-step speedup between two runs."""
    _save_timeseries(output_dir, "phase_metrics.csv", rows, _PHASE_METRICS_COLS)


def save_results(output_dir, correlated, episodes=None, anomalies=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "correlated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id1", "t1", "id2", "t2", "lag", "corr"])
        for a, b, corr, lag in correlated:
            w.writerow([a[0], a[1], b[0], b[1], lag, corr])

    if episodes is not None:
        with open(os.path.join(output_dir, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id1", "id2", "lag", "start", "last", "length", "sign"])
            for (id1, id2, lag), start, last, length, sign in episodes:
                w.writerow([id1, id2, lag, start, last, length, sign])

    if anomalies is not None:
        with open(os.path.join(output_dir, "anomalies.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id1", "id2", "lag", "time", "kind"])
            for (id1, id2, lag), time, kind in anomalies:
                w.writerow([id1, id2, lag, time, kind])
