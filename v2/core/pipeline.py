"""Orchestration shared by both modes.

  * "corrtrack" : read -> windows -> sketch -> index -> select
                  -> validate_sketches -> validate
  * "bf"        : read -> windows -> candidates(brute force) -> validate

Both modes reuse reading, windowing and validation. Only the candidate
generation differs. All the "hot" computation goes through the backend
(python/parallel/cython/cuda swap through config.backend).

`step` lets the pipeline run only up to a given step (inclusive).
"""

import os
import time
from dataclasses import asdict

from ..backends import get_backend, is_sharded
from ..steps import persistence, selection, validation
from ..steps.index import make_index
from ..steps.io import read_csv, subset
from ..steps.monitoring import Monitor
from ..steps.sketch import make_sketcher
from ..steps.windows import iter_windows
from . import _static_shard
from .log import get_logger
from .profiling import ResourceSampler, Throughput, Timings
from .qos import set_qos

# Sharded mode (static partition): memory guard. The FULL index lives in RAM
# (no eviction); beyond this budget we fall back to serial streaming.
# Overridable through CORRTRACK_STATIC_SHARD_MAX_GB.
STATIC_SHARD_MAX_BYTES = int(float(os.environ.get("CORRTRACK_STATIC_SHARD_MAX_GB", "4")) * 1e9)

CORRTRACK_STEPS = [
    "read", "windows", "sketch", "index",
    "select", "validate_sketches", "validate", "monitor",
]
BF_STEPS = ["read", "windows", "candidates", "validate", "monitor"]


def steps_for(mode):
    return CORRTRACK_STEPS if mode == "corrtrack" else BF_STEPS


def _sketch_and_insert_serial(ids, win, config, store, sketcher, index,
                              sketches_out, mode, limit, timings):
    """Sequential sketch+insert phase — historical code, moved as-is.

    Returns the list of `current_keys` of the window.
    """
    current_keys = []
    with timings.track("sketch" if mode == "corrtrack" else "ingest") as rec:
        for i, sid in enumerate(ids):
            # v1 alignment: skip near-constant windows (corr undefined) — they
            # never correlate and only inflate the candidate count.
            if config.std_threshold > 0.0 and float(win.block[i].std()) < config.std_threshold:
                continue
            key = (sid, win.start_time)
            entry = {"time": win.start_time, "raw": win.block[i]}
            if mode == "corrtrack":
                entry["sketch"] = sketcher.sketch(sid, win.start_index, win.block[i])
                sketches_out.append((key, entry["sketch"]))
            store[key] = entry
            current_keys.append(key)
        # input = series of the window; output = sketches/raws kept (≠ when the
        # std_threshold filter applies). Throughput = series processed per second.
        rec.io(n_in=len(ids), n_out=len(current_keys))

    if mode == "corrtrack" and limit >= CORRTRACK_STEPS.index("index"):
        with timings.track("index") as rec:
            for key in current_keys:
                index.insert(key, store[key]["sketch"])
            rec.io(n_in=len(current_keys), n_out=len(current_keys))
    return current_keys


def run(config, path, mode="corrtrack", step="all"):
    order = steps_for(mode)
    target = order[-1] if step in (None, "all") else step
    if target not in order:
        raise ValueError(f"unknown step {target!r} for mode {mode!r}: {order}")
    limit = order.index(target)

    # logs go to <output>/run.log by default (unless an explicit --log-file is given)
    log_file = config.log_file or (os.path.join(config.output, "run.log")
                                   if config.output else "")
    log = get_logger(config.log_level, log_file)
    log.info("=== run: mode=%s, target step='%s' ===", mode, target)
    _log_params(log, config, mode)

    set_qos(config.cores)  # macOS P/E hint for this (main) process
    timings = Timings()
    # one backend per phase (may be identical; "" inherits config.backend)
    be_kw = dict(max_workers=config.workers, cores=config.cores)
    sketch_be = get_backend(config.backend_for("sketch"), **be_kw)
    candidate_be = get_backend(config.backend_for("candidate"), **be_kw)
    validate_be = get_backend(config.backend_for("validate"), **be_kw)
    bs, bc, bv = (config.backend_for("sketch"), config.backend_for("candidate"),
                  config.backend_for("validate"))
    if mode == "bf":               # bf: only the validation computes
        backend_tag = f"{mode} back={bv}"
    elif bs == bc == bv:           # same backend everywhere
        backend_tag = f"{mode} back={bv}"
    else:                          # per-phase backend
        backend_tag = f"{mode} back=sk:{bs}/cand:{bc}/val:{bv}"

    # --- step 'read' ---
    log.info("step 'read': loading CSV %s", path)
    with timings.track("read"):
        ids, times, data = read_csv(path)
        if config.n_series or config.n_years or config.train_ratio < 1.0:
            ids, times, data = subset(ids, times, data, config.n_series,
                                      config.n_years, config.obs_mode, config.train_ratio)
            log.info("  filter: n_series=%s n_years=%s (mode=%s) train_ratio=%s",
                     config.n_series or "all", config.n_years or "all",
                     config.obs_mode, config.train_ratio)
    log.info("  -> %d series, %d observations (%.3fs)",
             len(ids), len(times), timings.as_dict()["read"])
    if limit == 0:
        log.info("=== done (step 'read') ===")
        return _result(mode, target, timings, n_series=len(ids), n_obs=len(times))

    # --- step 'windows' : slice the stream into sliding sub-windows ---
    log.info("step 'windows': slicing into sub-windows (size=%d, step=%d)",
             config.window_size, config.window_step)
    with timings.track("windows"):
        windows = list(iter_windows(data, times, config.window_size, config.window_step))
    log.info("  -> %d windows (%.3fs)", len(windows), timings.as_dict()["windows"])
    if limit == 1:
        log.info("=== done (step 'windows') ===")
        return _result(mode, target, timings, n_windows=len(windows))

    # the streaming loop runs every remaining step, window by window
    loop_steps = order[2:limit + 1]
    log.info("streaming %d windows through steps: %s",
             len(windows), " -> ".join(loop_steps))
    report_every = config.log_every  # log a checkpoint every N seconds (wall time)
    last_log = time.perf_counter()
    last_w = 0
    prev_agg = {k: 0.0 for k in _V1_ORDER}  # phase timings at the previous checkpoint
    prev_elapsed = 0.0
    prev_ncand = 0                          # candidate count at the previous checkpoint

    # REAL-TIME instruments: throughput curve (per phase) + resources (CPU/mem/energy).
    # `throughput` is fed at each checkpoint; `resources` samples in the background.
    throughput = Throughput(len(windows))
    resources = ResourceSampler().start()

    # --- streaming steps ---
    store = {}
    # sharded mode (`sharded_<base>`): window-parallel static partition (see
    # _static_shard). `workers` threads, each in charge of a slice of windows,
    # do their whole share (sketch then select+validate) — no per-window
    # re-dispatch. Relies on the FULL index in RAM → memory guard, otherwise
    # fall back to serial streaming. Only enabled when the target reaches
    # `validate`.
    sharded = mode == "corrtrack" and is_sharded(config.backend)
    n_workers = max(1, config.workers or (os.cpu_count() or 1))
    needs_validate = (mode == "corrtrack"
                      and limit >= CORRTRACK_STEPS.index("validate"))
    use_static = sharded and n_workers > 1 and needs_validate
    if use_static:
        est = _static_shard.estimated_bytes(len(ids), len(windows),
                                            config.window_size, config.n_vectors)
        if est > STATIC_SHARD_MAX_BYTES:
            log.info("sharded: full index ~%.2f GB > %.2f GB (guard) "
                     "-> falling back to serial streaming", est / 1e9,
                     STATIC_SHARD_MAX_BYTES / 1e9)
            use_static = False
    elif sharded and n_workers > 1 and not needs_validate:
        log.info("sharded: target '%s' does not reach validate -> serial streaming",
                 target)

    if mode == "corrtrack" and not use_static:
        index = make_index(config, candidate_be)
    else:
        index = None
    sketcher = (make_sketcher(config, sketch_be)
                if mode == "corrtrack" and not use_static else None)
    sketch_insert_fn = _sketch_and_insert_serial
    if use_static:
        log.info("sharded (static partition): %d workers, %d windows, index %s",
                 n_workers, len(windows), config.index_backend)

    sketches, candidates, prevalidated, correlated = [], [], [], []
    # MEMORY: only RETAIN the candidates/prevalidated lists when the target
    # returns them; otherwise only the COUNT is kept. On dense bf (e.g. 25_10),
    # the cumulated `candidates` list reaches hundreds of millions of tuples
    # (cand=458M observed) → 32 GB OOM. The counter is enough for n_candidates.
    n_candidates = n_prevalidated = 0
    keep_candidates = target in ("select", "candidates")
    keep_prevalidated = target == "validate_sketches"
    monitor = Monitor(config.window_step)

    if use_static:
        # sharded mode: window-parallel static partition (see _static_shard).
        res = _static_shard.run(config, sketch_be, candidate_be, validate_be,
                                ids, windows, n_workers, timings)
        correlated = res["correlated"]
        n_candidates = res["n_candidates"]
        n_prevalidated = res["n_tested"]
        if limit >= CORRTRACK_STEPS.index("monitor"):
            with timings.track("monitor") as rec:
                for item in res["per_window"]:
                    if item is not None:               # non-empty window
                        monitor.update(item[1], item[0])
                rec.io(n_in=len(correlated))
        log.info("  static-shard: cand=%d tested=%d corr=%d",
                 n_candidates, n_prevalidated, len(correlated))

    # streaming loop (disabled in static sharded mode)
    for w_idx, win in (() if use_static else enumerate(windows)):
        current_keys = sketch_insert_fn(
            ids, win, config, store, sketcher, index, sketches,
            mode, limit, timings,
        )

        if mode == "corrtrack":
            pairs = []
            if limit >= CORRTRACK_STEPS.index("select"):
                with timings.track("select") as rec:
                    pairs = selection.select_candidates(index, store, current_keys, config.n_lags)
                    rec.io(n_in=len(current_keys), n_out=len(pairs))
                n_candidates += len(pairs)
                if keep_candidates:
                    candidates.extend(pairs)
            if limit >= CORRTRACK_STEPS.index("validate_sketches"):
                with timings.track("validate_sketches") as rec:
                    n_in = len(pairs)
                    pairs = selection.validate_sketches(
                        pairs, store, config.sketch_threshold, candidate_be, config.neg_corr)
                    rec.io(n_in=n_in, n_out=len(pairs))
                n_prevalidated += len(pairs)
                if keep_prevalidated:
                    prevalidated.extend(pairs)
            if limit >= CORRTRACK_STEPS.index("validate"):
                with timings.track("validate") as rec:
                    win_corr = validation.validate(
                        pairs, store, config.corr_threshold, validate_be, config.neg_corr)
                    rec.io(n_in=len(pairs), n_out=len(win_corr))
                correlated.extend(win_corr)
                if limit >= CORRTRACK_STEPS.index("monitor"):
                    with timings.track("monitor") as rec:
                        monitor.update(win_corr, win.start_time)
                        rec.io(n_in=len(win_corr))
        else:  # bf
            with timings.track("candidates") as rec:
                pairs = selection.enumerate_candidates(store, current_keys, config.n_lags)
                rec.io(n_in=len(current_keys), n_out=len(pairs))
            n_candidates += len(pairs)
            if keep_candidates:
                candidates.extend(pairs)
            if limit >= BF_STEPS.index("validate"):
                with timings.track("validate") as rec:
                    win_corr = validation.validate(
                        pairs, store, config.corr_threshold, validate_be, config.neg_corr)
                    rec.io(n_in=len(pairs), n_out=len(win_corr))
                correlated.extend(win_corr)
                if limit >= BF_STEPS.index("monitor"):
                    with timings.track("monitor") as rec:
                        monitor.update(win_corr, win.start_time)
                        rec.io(n_in=len(win_corr))

        with timings.track("evict"):
            _evict(store, index, _min_time(win.start_time, config.n_lags))

        now = time.perf_counter()
        if now - last_log >= report_every or w_idx == len(windows) - 1:
            agg = _v1_timings(timings)
            elapsed = timings.elapsed()
            keys = [k for k in _V1_ORDER if agg[k] > 0]
            d_agg = {k: agg[k] - prev_agg[k] for k in _V1_ORDER}
            dt = now - last_log
            rate_w = (w_idx - last_w) / dt if dt > 0 else 0.0          # windows/s (recent)
            rate_c = (n_candidates - prev_ncand) / dt if dt > 0 else 0.0  # candidates/s
            eta = (len(windows) - 1 - w_idx) / rate_w if rate_w > 0 else float("inf")
            # real-time throughput curve: one sample per checkpoint
            throughput.snapshot(
                elapsed, w_idx + 1,
                counts={"candidate": n_candidates,
                        "validate": n_prevalidated if mode == "corrtrack" else n_candidates,
                        "monitor": len(correlated)},
                phase_times={"sketch": agg["sk_time"], "candidate": agg["cand_time"],
                             "validate": agg["val_time"], "monitor": agg["monit_time"]})
            log.info("  CP: %d/%d (t=%s) | %s | cand=%d corr=%d | %.1f win/s %.0f cand/s "
                     "| cum: %s | Δ: %s | ETA %s",
                     w_idx + 1, len(windows), win.start_time, backend_tag,
                     n_candidates, len(correlated), rate_w, rate_c,
                     _agg_inline(agg, elapsed, keys),
                     _agg_inline(d_agg, elapsed - prev_elapsed, keys), _fmt_eta(eta))
            last_log, last_w = now, w_idx
            prev_agg, prev_elapsed, prev_ncand = agg, elapsed, n_candidates

    log.info("=== done: %d correlated / %d candidates over %d windows in %.3fs ===",
             len(correlated), n_candidates, len(windows), timings.elapsed())
    _log_timings(log, timings)

    # --- closing the real-time instruments (throughput + resources) ---
    resources.stop()
    if not throughput.samples:
        # static sharded mode (no per-window checkpoint): a final sample so the
        # global throughput and a consistent min=max=median are available.
        _agg = _v1_timings(timings)
        throughput.snapshot(
            timings.elapsed(), len(windows),
            counts={"candidate": n_candidates,
                    "validate": n_prevalidated if mode == "corrtrack" else n_candidates,
                    "monitor": len(correlated)},
            phase_times={"sketch": _agg["sk_time"], "candidate": _agg["cand_time"],
                         "validate": _agg["val_time"], "monitor": _agg["monit_time"]})
    tput_summary = throughput.summary()
    res_summary = resources.summary()
    # PER-STEP metrics (time + input/output throughput per computation) — the
    # basis of the per-step speedup between two runs (see Timings.phase_metrics).
    phase_metrics = timings.phase_metrics()
    rt = {"throughput": tput_summary, "resources": res_summary,
          "phase_metrics": phase_metrics}
    if config.output:
        persistence.save_throughput(config.output, throughput.rows())
        persistence.save_resources(config.output, resources.rows())
        persistence.save_phase_metrics(config.output, _phase_metrics_rows(phase_metrics))
    _log_realtime(log, tput_summary, res_summary)
    _log_phase_metrics(log, phase_metrics)

    if target == "sketch":
        return _result(mode, target, timings, n_sketches=len(sketches), sketches=sketches)
    if target == "index":
        return _result(mode, target, timings, n_indexed=len(sketches))
    if target in ("select", "candidates"):
        return _result(mode, target, timings, n_candidates=n_candidates, candidates=candidates)
    if target == "validate_sketches":
        return _result(mode, target, timings, n_prevalidated=n_prevalidated,
                       n_candidates=n_candidates, candidates=prevalidated)
    if target == "validate":
        if config.output:
            persistence.save_results(config.output, correlated)
        return _result(mode, target, timings, n_correlated=len(correlated),
                       correlated=correlated, **rt)
    # target == "monitor"
    episodes, anomalies = monitor.finalize()
    # tested = pairs actually correlation-tested (corrtrack: after the sketch
    # filter; bf: every candidate, no sketch filter).
    n_tested = n_prevalidated if mode == "corrtrack" else n_candidates
    if config.output:
        persistence.save_results(config.output, correlated, episodes, anomalies)
        # summary.csv = parameters used (param.*) + results + timings
        summary = {f"param.{k}": v for k, v in asdict(config).items()}
        summary.update({"mode": mode, "step": target,
                        "n_series": len(ids), "n_windows": len(windows),
                        "n_candidates": n_candidates, "n_tested": n_tested,
                        "n_correlated": len(correlated), "n_episodes": len(episodes),
                        "n_anomalies": len(anomalies), "runtime": timings.elapsed()})
        summary.update(_v1_timings(timings))
        summary.update(_realtime_flat(tput_summary, res_summary))  # throughput + resources (flat)
        summary.update(_phase_metrics_flat(phase_metrics))         # per-step metrics (flat)
        persistence.save_summary(config.output, summary)
        log.info("results written to %s/ (correlated, episodes, anomalies, summary, "
                 "throughput, resources, run.log)", config.output)
    return _result(mode, target, timings, n_correlated=len(correlated),
                   n_candidates=n_candidates, n_tested=n_tested,
                   n_series=len(ids), n_windows=len(windows),
                   n_episodes=len(episodes), n_anomalies=len(anomalies),
                   correlated=correlated, episodes=episodes, anomalies=anomalies, **rt)


# map the fine-grained v2 phases onto the canonical v1 ("method 1") phases,
# same names as the benchmark columns (sk_time / cand_time / val_time / monit_time).
_V1_PHASE = {
    "sketch": "sk_time",
    "index": "cand_time", "select": "cand_time", "validate_sketches": "cand_time",
    "candidates": "cand_time",
    "validate": "val_time",
    "monitor": "monit_time",
    "read": "setup", "windows": "setup", "evict": "setup", "ingest": "setup",
}
_V1_ORDER = ["sk_time", "cand_time", "val_time", "monit_time", "setup"]
_V1_ABBR = {"sk_time": "sk", "cand_time": "cand", "val_time": "val",
            "monit_time": "monit", "setup": "setup"}


def _v1_timings(timings):
    """Aggregate v2 phases into the v1 canonical phases (dict, _V1_ORDER keys)."""
    agg = {k: 0.0 for k in _V1_ORDER}
    for name, secs in timings.as_dict().items():
        agg[_V1_PHASE.get(name, "setup")] += secs
    return agg


def _agg_inline(agg, total, keys):
    """Compact one-line timing for a v1-phase aggregate over the given keys."""
    parts = [f"{_V1_ABBR[k]}={agg[k]:.3f}" for k in keys]
    return " ".join(parts) + f" Σ={total:.2f}s"


def _fmt_eta(seconds):
    """Human-readable ETA ('1h23m', '4m12s', '45s', or '?' if unknown)."""
    if seconds != seconds or seconds == float("inf"):  # nan or inf
        return "?"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _log_params(log, config, mode):
    """Log a synthesis of the effective parameters at startup."""
    log.info("params | data: n_series=%s n_years=%s obs_mode=%s",
             config.n_series or "all", config.n_years or "all", config.obs_mode)
    log.info("params | windows: size=%d step=%d n_lags=%d basic_window=%d",
             config.window_size, config.window_step, config.n_lags, config.basic_window)
    log.info("params | thresholds: corr=%s neg_corr=%s sketch=%s",
             config.corr_threshold, config.neg_corr, config.sketch_threshold)
    if mode == "corrtrack":
        log.info("params | sketch: n_vectors=%d seed=%d | index=%s grid_cell=%s "
                 "grid_n_tables=%d grid_n_coords=%d grid_vote_min=%d query_radius=%s",
                 config.n_vectors, config.seed, config.index_backend, config.grid_cell,
                 config.grid_n_tables, config.grid_n_coords, config.grid_vote_min,
                 config.query_radius)
        log.info("params | backends: sketch=%s candidate=%s validate=%s (workers=%d)",
                 config.backend_for("sketch"), config.backend_for("candidate"),
                 config.backend_for("validate"), config.workers)
    else:
        log.info("params | backend: validate=%s (workers=%d)",
                 config.backend_for("validate"), config.workers)
    log.info("params | output=%s log_every=%ss",
             config.output or "(none)", config.log_every)


def _log_timings(log, timings):
    """Log per-phase timings using the v1 phase names; fine v2 detail at DEBUG."""
    total = timings.elapsed()
    agg = _v1_timings(timings)
    log.info("per-phase timings (v1 phases):")
    for name in _V1_ORDER:
        pct = 100.0 * agg[name] / total if total else 0.0
        log.info("      %-10s %9.4fs  %5.1f%%", name, agg[name], pct)
    log.info("      %-10s %9.4fs  %5.1f%%", "runtime", total, 100.0)
    # fine-grained v2 sub-steps (only when --log-level debug)
    for name, secs in sorted(timings.as_dict().items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * secs / total if total else 0.0
        log.debug("      [v2] %-18s %9.4fs  %5.1f%%", name, secs, pct)


# throughput streams exposed in summary.csv (flat keys `tput_<stream>_<stat>`)
_TPUT_STREAMS = ("windows", "sketch", "candidate", "validate", "monitor")
_TPUT_STATS = ("min", "max", "median", "mean", "p05", "p95")


def _realtime_flat(tput_summary, res_summary):
    """Flatten throughput + resources into scalar keys for summary.csv.

    Throughput: `tput_<stream>_{min,max,median,mean,p05,p95,overall}` (windows/s
    for `windows`, one unit per step). Resources: `cpu_pct_*`, `mem_mb_*`,
    `mem_peak_mb`, `power_cores_*`, `energy_cpu_seconds`. Everything is
    NaN-safe.
    """
    out = {"tput_n_samples": tput_summary.get("n_samples", 0)}
    for stream in _TPUT_STREAMS:
        st = tput_summary.get(stream, {})
        for stat in _TPUT_STATS:
            out[f"tput_{stream}_{stat}"] = st.get(stat, float("nan"))
        out[f"tput_{stream}_overall"] = st.get("overall", float("nan"))
    cpu, mem, power = (res_summary.get("cpu_pct", {}), res_summary.get("mem_mb", {}),
                       res_summary.get("power_cores", {}))
    for stat in ("min", "max", "median", "mean"):
        out[f"cpu_pct_{stat}"] = cpu.get(stat, float("nan"))
        out[f"mem_mb_{stat}"] = mem.get(stat, float("nan"))
    for stat in ("min", "max", "median"):
        out[f"power_cores_{stat}"] = power.get(stat, float("nan"))
    out["mem_peak_mb"] = res_summary.get("mem_peak_mb", float("nan"))
    out["energy_cpu_seconds"] = res_summary.get("energy_cpu_seconds", float("nan"))
    out["resources_available"] = res_summary.get("available", False)
    out["resources_n_samples"] = res_summary.get("n_samples", 0)
    return out


def _log_realtime(log, tput_summary, res_summary):
    """Log the real-time block: per-stream throughput (min/median/max) + resources."""
    log.info("real-time throughput (instantaneous, min/median/max | overall):")
    for stream in _TPUT_STREAMS:
        st = tput_summary.get(stream, {})
        if not st or st.get("n", 0) == 0:
            continue
        log.info("      %-10s %9.1f / %9.1f / %9.1f %-6s | overall %9.1f",
                 stream, st.get("min", float("nan")), st.get("median", float("nan")),
                 st.get("max", float("nan")), st.get("unit", ""),
                 st.get("overall", float("nan")))
    cpu, mem = res_summary.get("cpu_pct", {}), res_summary.get("mem_mb", {})
    if res_summary.get("available"):
        log.info("      resources | cpu%% min/med/max %.0f/%.0f/%.0f | mem MB "
                 "min/med/peak %.0f/%.0f/%.0f | energy ~%.1f core·s",
                 cpu.get("min", float("nan")), cpu.get("median", float("nan")),
                 cpu.get("max", float("nan")), mem.get("min", float("nan")),
                 mem.get("median", float("nan")), res_summary.get("mem_peak_mb", float("nan")),
                 res_summary.get("energy_cpu_seconds", float("nan")))
    else:
        log.info("      resources | psutil unavailable — energy ~%.1f core·s, "
                 "peak mem ~%.0f MB (getrusage)",
                 res_summary.get("energy_cpu_seconds", float("nan")),
                 res_summary.get("mem_peak_mb", float("nan")))


# STEPS displayed in the per-step metrics (canonical order: corrtrack + bf).
# Steps absent from a run are ignored; unknown ones are appended at the end
# (robustness).
_PHASE_METRIC_ORDER = ("read", "windows", "sketch", "ingest", "index", "select",
                       "validate_sketches", "candidates", "validate", "monitor", "evict")


def _ordered_phases(pm):
    """Steps of `pm` in canonical order, followed by any unknown ones."""
    known = [p for p in _PHASE_METRIC_ORDER if p in pm]
    return known + [p for p in pm if p not in _PHASE_METRIC_ORDER]


def _phase_metrics_rows(pm):
    """Rows for phase_metrics.csv (one per step)."""
    rows = []
    for name in _ordered_phases(pm):
        m = pm[name]
        t, it, ot = m["time"], (m["in_tput"] or {}), (m["out_tput"] or {})
        nan = float("nan")
        rows.append({
            "phase": name, "calls": m["calls"], "time_total": m["time_total"],
            "time_min": t["min"], "time_median": t["median"], "time_max": t["max"],
            "time_mean": t["mean"], "time_std": t["std"],
            "time_p05": t["p05"], "time_p95": t["p95"],
            "n_in_total": m["n_in_total"], "n_out_total": m["n_out_total"],
            "selectivity": m["selectivity"],
            "in_overall": m["in_overall"], "in_min": it.get("min", nan),
            "in_median": it.get("median", nan), "in_max": it.get("max", nan),
            "out_overall": m["out_overall"], "out_min": ot.get("min", nan),
            "out_median": ot.get("median", nan), "out_max": ot.get("max", nan),
        })
    return rows


def _phase_metrics_flat(pm):
    """Flatten into scalar keys `phase_<step>_*` for summary.csv (per-step speedup)."""
    out = {}
    nan = float("nan")
    for name in _ordered_phases(pm):
        m = pm[name]
        t = m["time"]
        p = f"phase_{name}_"
        out[p + "calls"] = m["calls"]
        out[p + "time_total"] = m["time_total"]
        out[p + "time_min"] = t["min"]
        out[p + "time_median"] = t["median"]
        out[p + "time_max"] = t["max"]
        out[p + "n_in"] = m["n_in_total"] if m["n_in_total"] is not None else nan
        out[p + "n_out"] = m["n_out_total"] if m["n_out_total"] is not None else nan
        out[p + "selectivity"] = m["selectivity"]
        out[p + "in_overall"] = m["in_overall"]
        out[p + "out_overall"] = m["out_overall"]
    return out


def _log_phase_metrics(log, pm):
    """Log the per-step metrics: time/call (min/med/max) + in→out throughput."""
    if not pm:
        return
    log.info("per-step metrics (t/call min/med/max ms | in->out items/s overall | sel):")
    for name in _ordered_phases(pm):
        m = pm[name]
        t = m["time"]
        sel = m["selectivity"]
        sel_s = f"{sel:.3f}" if sel == sel else "  -  "
        log.info("      %-18s n=%-6d t/call %8.3f/%8.3f/%8.3f ms | %11.0f->%-11.0f it/s | sel %s",
                 name, m["calls"],
                 t["min"] * 1e3, t["median"] * 1e3, t["max"] * 1e3,
                 m["in_overall"], m["out_overall"], sel_s)


def _evict(store, index, min_time):
    if min_time is None:
        return
    for key in [k for k, v in store.items() if v["time"] < min_time]:
        del store[key]
        if index is not None:
            index.remove(key)


def _min_time(current_time, n_lags):
    try:
        return current_time - n_lags
    except TypeError:
        return None  # non-numeric time: no eviction (skeleton)


def _result(mode, step, timings, **payload):
    out = {"mode": mode, "step": step}
    out.update(payload)
    out["runtime"] = timings.elapsed()
    out["timings"] = timings.as_dict()
    out["_timings"] = timings  # objet complet (pour .summary()), exclu de l'affichage compact
    return out
