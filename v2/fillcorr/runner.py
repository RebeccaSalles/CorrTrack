"""FilCorr runner — orchestrates the pipeline in `filcorr` mode.

Reuses the v2 CSV reading and windowing (`steps.io`, `steps.windows`):
- reads the CSV, applies the filtering (`n_series`, `n_years`, `train_ratio`,
  `std_threshold` on the windows);
- slides sub-windows of size `window_size`, step `window_step`;
- for each window: computes the band FFT of every series (through the backend);
- enumerates every pair `(X@t, Y@t')` such that `|t − t'| ≤ n_lags`;
- validates through the Parseval correlation (batch backend);
- keeps those with `|corr| ≥ corr_threshold` (signed lag = `t_A − t_B`).

The output is 100% aligned with bf/corrtrack for a direct comparison: same files
(`correlated.csv`, `episodes.csv`, `anomalies.csv`, `summary.csv`, `run.log`),
same keys in `result.json` (`n_series`, `n_windows`, `n_candidates`,
`n_tested`, `n_correlated`, `n_episodes`, `n_anomalies`, `runtime`,
`timings`), same canonical v1 phases in the `summary` (`sk_time`, `cand_time`,
`val_time`, `monit_time`, `setup`), same log formats (parameter synthesis at
startup + `cum`/`Δ`/`ETA` checkpoints + per-phase breakdown at the end). Reuses
the `_v1_timings`, `_agg_inline`, `_fmt_eta`, `_log_timings` helpers of
`core.pipeline` to guarantee bit-for-bit format parity.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

from ..core.log import get_logger
from ..core.pipeline import (
    _V1_ORDER,
    _agg_inline,
    _fmt_eta,
    _log_timings,
    _v1_timings,
)
from ..core.profiling import Timings
from ..core.qos import set_qos
from ..steps import persistence
from ..steps.io import read_csv, subset
from ..steps.monitoring import Monitor
from ..steps.windows import iter_windows
from .algo import FilCorrParams, band_indices
from .backends import get_backend


def _filcorr_params_from(cfg, overrides: Optional[dict] = None) -> FilCorrParams:
    """Build `FilCorrParams` from (cfg + overrides).

    Looks the FilCorr attributes up first in `overrides` (an optional dict passed
    by the JSON pipeline), then in `cfg` (which can be extended outside Config by
    setting ad-hoc attributes). Defaults: no filtering (DC only).
    """
    def g(name, default):
        if overrides and name in overrides:
            return overrides[name]
        return getattr(cfg, name, default)
    return FilCorrParams(
        fs=float(g("filcorr_fs", 0.0)),
        ft=float(g("filcorr_ft", 0.5)),
        sampling_rate=float(g("filcorr_sampling_rate", 1.0)),
    )


def _be_name(cfg, overrides: Optional[dict] = None) -> str:
    if overrides and "filcorr_backend" in overrides:
        return overrides["filcorr_backend"]
    name = getattr(cfg, "filcorr_backend", None)
    if name:
        return name
    # fallback: map the standard `backend` to its filcorr equivalent.
    std = (getattr(cfg, "validate_backend", "") or cfg.backend or "vectorized")
    if "parallel" in std:
        return "parallel"
    if "cython" in std:
        return "cython"
    if std in ("python", "vectorized", "mps"):
        return std
    return "vectorized"


def _log_filcorr_params(log, cfg, fp, backend_name):
    """Parameter synthesis at startup — same lines as `core.pipeline._log_params`
    so that grepping the logs finds the same keys across the 3 modes.
    """
    log.info("params | data: n_series=%s n_years=%s obs_mode=%s",
             cfg.n_series or "all", cfg.n_years or "all", cfg.obs_mode)
    log.info("params | windows: size=%d step=%d n_lags=%d basic_window=%d",
             cfg.window_size, cfg.window_step, cfg.n_lags, cfg.basic_window)
    log.info("params | thresholds: corr=%s neg_corr=%s sketch=%s",
             cfg.corr_threshold, cfg.neg_corr, cfg.sketch_threshold)
    log.info("params | filcorr: fs=%s ft=%s sampling_rate=%s backend=%s",
             fp.fs, fp.ft, fp.sampling_rate, backend_name)
    log.info("params | backend: validate=%s (workers=%d)",
             cfg.backend_for("validate"), cfg.workers)
    log.info("params | output=%s log_every=%ss",
             cfg.output or "(none)", cfg.log_every)


def run(cfg, path: str, mode: str = "filcorr", step: str = "all",
        overrides: Optional[dict] = None) -> dict:
    """Run FilCorr (offline) with the same signature as `core.pipeline.run`.

    Steps: `read`, `windows`, `band_fft`, `pairs`, `validate`, `monitor`.
    `step` can limit the execution (the steps beyond it are skipped).
    """
    # Phases named in alignment with `core.pipeline._V1_PHASE`:
    #   sketch     -> sk_time    (our band FFT)
    #   candidates -> cand_time  (pair enumeration)
    # → `_v1_timings` aggregates them correctly without a post-hoc alias.
    order = ["read", "windows", "sketch", "candidates", "validate", "monitor"]
    target = order[-1] if step in (None, "all") else step
    if target not in order:
        raise ValueError(f"unknown step {target!r} for mode 'filcorr': {order}")
    limit = order.index(target)

    log_file = cfg.log_file or (os.path.join(cfg.output, "run.log")
                                if cfg.output else "")
    log = get_logger(cfg.log_level, log_file)
    log.info("=== run: mode=filcorr, target step='%s' ===", target)
    fp = _filcorr_params_from(cfg, overrides)
    backend_name = _be_name(cfg, overrides)
    _log_filcorr_params(log, cfg, fp, backend_name)

    set_qos(cfg.cores)
    timings = Timings()
    be = get_backend(backend_name, max_workers=cfg.workers, cores=cfg.cores)
    backend_tag = f"filcorr back={backend_name}"

    # --- read ---
    log.info("step 'read': loading CSV %s", path)
    with timings.track("read"):
        ids, times, data = read_csv(path)
        if cfg.n_series or cfg.n_years or cfg.train_ratio < 1.0:
            ids, times, data = subset(ids, times, data, cfg.n_series,
                                      cfg.n_years, cfg.obs_mode, cfg.train_ratio)
            log.info("  filter: n_series=%s n_years=%s (mode=%s) train_ratio=%s",
                     cfg.n_series or "all", cfg.n_years or "all",
                     cfg.obs_mode, cfg.train_ratio)
    log.info("  -> %d series, %d observations (%.3fs)",
             len(ids), len(times), timings.as_dict()["read"])
    if limit == 0:
        log.info("=== done (step 'read') ===")
        return _result(target, timings, n_series=len(ids), n_obs=len(times))

    # --- windows ---
    log.info("step 'windows': slicing into sub-windows (size=%d, step=%d)",
             cfg.window_size, cfg.window_step)
    with timings.track("windows"):
        windows = list(iter_windows(data, times, cfg.window_size, cfg.window_step))
    log.info("  -> %d windows (%.3fs)", len(windows), timings.as_dict()["windows"])
    if limit == 1:
        log.info("=== done (step 'windows') ===")
        return _result(target, timings, n_windows=len(windows), n_series=len(ids))

    lb, ub = band_indices(cfg.window_size, fp)
    B = ub - lb
    log.info("streaming %d windows through steps: %s",
             len(windows), " -> ".join(order[2:limit + 1]))
    log.info("filcorr band: lb=%d ub=%d B=%d (m=%d) -> ratio %.1f%% kept",
             lb, ub, B, cfg.window_size, 100.0 * B / cfg.window_size)

    # store: key (sid, t) -> {"time": t, "band": Wband ou _RawWindows row}
    store: dict = {}
    correlated: list = []
    candidates_count = 0
    monitor = Monitor(cfg.window_step)

    report_every = cfg.log_every
    last_log = time.perf_counter()
    last_w = 0
    prev_agg = {k: 0.0 for k in _V1_ORDER}
    prev_elapsed = 0.0
    prev_ncand = 0

    for w_idx, win in enumerate(windows):
        # std_threshold: filtering happens BEFORE the FFT to stay aligned with
        # bf/corrtrack v2 (near-constant windows ignored). Preserves v1 alignment.
        with timings.track("ingest"):
            if cfg.std_threshold > 0.0:
                kept_idx = [i for i in range(len(ids))
                            if float(win.block[i].std()) >= cfg.std_threshold]
            else:
                kept_idx = list(range(len(ids)))

        current_keys = []
        with timings.track("sketch"):
            if kept_idx:
                sub = win.block[kept_idx] if len(kept_idx) < len(ids) else win.block
                band = be.band_fft(sub, lb, ub)   # (k, B) ou _RawWindows
            else:
                band = None
        for slot, i in enumerate(kept_idx):
            sid = ids[i]
            key = (sid, win.start_time)
            store[key] = {"time": win.start_time, "band": _row(band, slot)}
            current_keys.append(key)

        # enumeration of the pairs (X@t, Y@t') with |t-t'| <= n_lags
        with timings.track("candidates"):
            pairs = _enumerate_pairs(current_keys, store, cfg.n_lags)
        candidates_count += len(pairs)

        if limit >= order.index("validate") and pairs:
            with timings.track("validate"):
                win_corr = _validate_pairs(pairs, store, be, cfg.corr_threshold,
                                           cfg.neg_corr, backend_name)
            correlated.extend(win_corr)
            if limit >= order.index("monitor"):
                with timings.track("monitor"):
                    monitor.update(win_corr, win.start_time)

        with timings.track("evict"):
            _evict(store, _min_time(win.start_time, cfg.n_lags))

        now = time.perf_counter()
        if now - last_log >= report_every or w_idx == len(windows) - 1:
            agg = _v1_timings(timings)
            elapsed = timings.elapsed()
            keys = [k for k in _V1_ORDER if agg[k] > 0]
            d_agg = {k: agg[k] - prev_agg[k] for k in _V1_ORDER}
            dt = now - last_log
            rate_w = (w_idx - last_w) / dt if dt > 0 else 0.0
            rate_c = (candidates_count - prev_ncand) / dt if dt > 0 else 0.0
            eta = (len(windows) - 1 - w_idx) / rate_w if rate_w > 0 else float("inf")
            log.info("  CP: %d/%d (t=%s) | %s | cand=%d corr=%d | %.1f win/s %.0f cand/s "
                     "| cum: %s | Δ: %s | ETA %s",
                     w_idx + 1, len(windows), win.start_time, backend_tag,
                     candidates_count, len(correlated), rate_w, rate_c,
                     _agg_inline(agg, elapsed, keys),
                     _agg_inline(d_agg, elapsed - prev_elapsed, keys), _fmt_eta(eta))
            last_log, last_w = now, w_idx
            prev_agg, prev_elapsed, prev_ncand = agg, elapsed, candidates_count

    log.info("=== done: %d correlated / %d candidates over %d windows in %.3fs ===",
             len(correlated), candidates_count, len(windows), timings.elapsed())
    _log_timings(log, timings)

    # cleanly close the pool (parallel) — not critical, but avoids warnings
    if hasattr(be, "close"):
        try:
            be.close()
        except Exception:
            pass

    n_tested = candidates_count
    episodes, anomalies = ([], [])
    if target == "monitor":
        episodes, anomalies = monitor.finalize()

    if cfg.output and target in ("validate", "monitor"):
        if target == "monitor":
            persistence.save_results(cfg.output, correlated, episodes, anomalies)
        else:
            persistence.save_results(cfg.output, correlated)
        summary = {f"param.{k}": v for k, v in asdict(cfg).items()}
        summary.update({
            "mode": "filcorr", "step": target,
            "param.filcorr_fs": fp.fs, "param.filcorr_ft": fp.ft,
            "param.filcorr_sampling_rate": fp.sampling_rate,
            "param.filcorr_backend": backend_name,
            "n_series": len(ids), "n_windows": len(windows),
            "n_candidates": candidates_count, "n_tested": n_tested,
            "n_correlated": len(correlated),
            "n_episodes": len(episodes), "n_anomalies": len(anomalies),
            "runtime": timings.elapsed(),
        })
        # canonical v1 phases — exactly the same keys as bf/corrtrack
        # (sk_time / cand_time / val_time / monit_time / setup) through _v1_timings.
        summary.update(_v1_timings(timings))
        persistence.save_summary(cfg.output, summary)
        log.info("results written to %s/ (correlated, episodes, anomalies, summary, run.log)",
                 cfg.output)

    if target == "validate":
        return _result(target, timings, n_correlated=len(correlated),
                       n_candidates=candidates_count, n_tested=n_tested,
                       n_series=len(ids), n_windows=len(windows),
                       correlated=correlated)
    if target == "monitor":
        return _result(target, timings, n_correlated=len(correlated),
                       n_candidates=candidates_count, n_tested=n_tested,
                       n_series=len(ids), n_windows=len(windows),
                       n_episodes=len(episodes), n_anomalies=len(anomalies),
                       correlated=correlated, episodes=episodes, anomalies=anomalies)
    return _result(target, timings,
                   n_correlated=len(correlated), n_candidates=candidates_count,
                   n_tested=n_tested, n_series=len(ids), n_windows=len(windows))


# ----------------------------------------------------------------------- helpers


def _row(band, i):
    """Extrait la `i`e ligne d'un bloc FFT (ou conteneur backend-specific)."""
    if band is None:
        raise IndexError("empty band block")
    return band[i]


def _enumerate_pairs(current_keys, store, n_lags):
    """Every (X@t, Y@t') pair, without duplicates.

    Identical to `selection.enumerate_candidates` (bf v2): the current windows
    are walked against the whole store, intra-window duplicates are skipped and
    the order is canonicalized. Returns a LIST (already duplicate-free by
    construction thanks to the intra-current `cand < key` filter) to preserve the
    deterministic order the tests expect.
    """
    pairs = []
    current = set(current_keys)
    for key in current_keys:
        for cand in store:
            if cand == key:
                continue
            if cand in current and cand < key:
                continue
            if abs(store[key]["time"] - store[cand]["time"]) > n_lags:
                continue
            pairs.append((key, cand) if key <= cand else (cand, key))
    return pairs


def _validate_pairs(pairs, store, backend, threshold, neg_corr, backend_name):
    """Compute the batched Parseval correlation and filter by threshold."""
    if not pairs:
        return []
    if backend_name == "python":
        from .backends import _RawWindows
        lb = ub = 0
        rows_x, rows_y = [], []
        for a, b in pairs:
            wx_obj = store[a]["band"]
            wy_obj = store[b]["band"]
            rows_x.append(wx_obj.raw if isinstance(wx_obj, _RawWindows) else wx_obj)
            rows_y.append(wy_obj.raw if isinstance(wy_obj, _RawWindows) else wy_obj)
            if isinstance(wx_obj, _RawWindows):
                lb, ub = wx_obj.lb, wx_obj.ub
        WX = _RawWindows(np.array(rows_x), lb, ub)
        WY = _RawWindows(np.array(rows_y), lb, ub)
    else:
        WX = np.array([store[a]["band"] for a, _ in pairs])
        WY = np.array([store[b]["band"] for _, b in pairs])
    corrs = backend.correlate_pairs(WX, WY)
    out = []
    for (a, b), corr in zip(pairs, corrs):
        if corr != corr:           # NaN
            continue
        ok = abs(corr) >= threshold if neg_corr else corr >= threshold
        if ok:
            out.append((a, b, float(corr), int(a[1] - b[1])))
    return out


def _evict(store, min_time):
    if min_time is None:
        return
    for key in [k for k, v in store.items() if v["time"] < min_time]:
        del store[key]


def _min_time(current_time, n_lags):
    try:
        return current_time - n_lags
    except TypeError:
        return None


def _result(step, timings, **payload):
    out = {"mode": "filcorr", "step": step}
    out.update(payload)
    out["runtime"] = timings.elapsed()
    # The native phases are already named 'sketch'/'candidates' → `_v1_timings`
    # (used by benchmark / save_summary) aggregates them as-is into
    # sk_time/cand_time. The raw dict is reproduced and a 'setup' alias is
    # computed, exposed directly in res["timings"] for consistency with the
    # bf/corrtrack result.json (load_result only looks at the aliased keys).
    tdict = dict(timings.as_dict())
    tdict["setup"] = sum(tdict.get(k, 0.0) for k in ("read", "windows", "ingest", "evict"))
    out["timings"] = tdict
    out["_timings"] = timings
    return out
