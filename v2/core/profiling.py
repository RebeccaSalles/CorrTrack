"""Lightweight per-phase profiling, to spot the bottlenecks.

Accumulates the time spent in each pipeline phase (sketch, index, select,
validate…) over every window, the number of calls, and produces a table sorted
by decreasing time (with the share of the total).

Usage:
    timings = Timings()
    with timings.track("sketch"):
        ...
    print(timings.summary(total=timings.elapsed()))

Two complementary instruments for **real-time** analysis:

  * `Throughput`: time series of the throughput (global windows/s + per-phase
    throughput) sampled at each checkpoint, with a min/max/median/… summary and
    the full curve to plot throughput = f(time).
  * `ResourceSampler`: background thread sampling CPU %, RSS memory and an
    energy proxy (busy CPU cores) over the process + its children, to show how
    the system behaves under load (min/max/median).
"""

import os
import statistics
import threading
import time
from contextlib import contextmanager


class _CallIO:
    """I/O collector returned by `Timings.track(...)`.

    The step code declares there the number of elements processed as INPUT and
    produced as OUTPUT for the current call (typically one window):

        with timings.track("select") as rec:
            pairs = select_candidates(...)
            rec.io(n_in=len(current_keys), n_out=len(pairs))

    Without a call to `io()`, the step only records its duration (the
    input/output throughputs stay unavailable for that step, but the per-call
    time is still measured).
    """

    __slots__ = ("n_in", "n_out")

    def __init__(self):
        self.n_in = None
        self.n_out = None

    def io(self, n_in=None, n_out=None):
        if n_in is not None:
            self.n_in = int(n_in)
        if n_out is not None:
            self.n_out = int(n_out)


class Timings:
    def __init__(self):
        self.totals = {}        # phase name -> cumulated seconds
        self.counts = {}        # phase name -> number of calls
        self.samples = {}       # phase name -> list of per-call (dt, n_in, n_out)
        self._order = []        # order of first appearance
        self._start = time.perf_counter()

    @contextmanager
    def track(self, name):
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
            self.samples[name] = []
            self._order.append(name)
        rec = _CallIO()
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1
            self.samples[name].append((dt, rec.n_in, rec.n_out))

    def elapsed(self):
        """Wall time elapsed since creation (total runtime)."""
        return time.perf_counter() - self._start

    def as_dict(self):
        """Phases in order of appearance -> seconds (for the machine output)."""
        return {n: self.totals[n] for n in self._order}

    def summary(self, total=None):
        """Readable table, sorted by decreasing time (bottleneck first)."""
        if total is None:
            total = self.elapsed()
        rows = sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)
        width = max((len(n) for n in self.totals), default=5)
        lines = [f"{'phase':<{width}}  {'calls':>7}  {'time(s)':>10}  {'%':>6}"]
        lines.append("-" * (width + 29))
        for name, secs in rows:
            pct = 100.0 * secs / total if total else 0.0
            lines.append(f"{name:<{width}}  {self.counts[name]:>7}  {secs:>10.4f}  {pct:>5.1f}%")
        lines.append("-" * (width + 29))
        lines.append(f"{'runtime':<{width}}  {'':>7}  {total:>10.4f}  {100.0:>5.1f}%")
        return "\n".join(lines)

    def phase_metrics(self):
        """Per-STEP distribution over every call (≈ per window).

        Designed to compare two backends on the SAME step (per-step speedup):
        the per-call throughput is the natural unit, uncontaminated by the other
        phases. For each step actually executed, returns a dict:

          calls       : number of calls (windows processed by the step).
          time        : _stats of the per-call DURATIONS, in seconds
                        (min/max/median/mean/std/p05/p95) → time of each computation.
          time_total  : sum of the durations (= `totals[name]`).
          in_tput     : _stats of the per-call INPUT THROUGHPUT (n_in/dt, elements/s).
          out_tput    : _stats of the per-call OUTPUT THROUGHPUT (n_out/dt, elements/s).
          n_in_total  / n_out_total : cumulated input / output elements.
          in_overall  / out_overall : Σn_in / Σtime,  Σn_out / Σtime (effective throughput).
          selectivity : n_out_total / n_in_total (pass-through rate of the step).

        `in_tput`/`out_tput`/`n_*` are None when the step declared no I/O (see
        `_CallIO.io`). `time` is always present.
        """
        out = {}
        for name in self._order:
            samples = self.samples.get(name)
            if not samples:
                continue
            durs = [dt for dt, _ni, _no in samples]
            has_in = any(ni is not None for _dt, ni, _no in samples)
            has_out = any(no is not None for _dt, _ni, no in samples)
            ins = [ni / dt for dt, ni, _no in samples
                   if ni is not None and dt > 1e-12]
            outs = [no / dt for dt, _ni, no in samples
                    if no is not None and dt > 1e-12]
            n_in_tot = sum(ni for _dt, ni, _no in samples if ni is not None)
            n_out_tot = sum(no for _dt, _ni, no in samples if no is not None)
            t_tot = sum(durs)
            out[name] = {
                "calls": len(samples),
                "time": _stats(durs),
                "time_total": t_tot,
                "in_tput": _stats(ins) if has_in else None,
                "out_tput": _stats(outs) if has_out else None,
                "n_in_total": n_in_tot if has_in else None,
                "n_out_total": n_out_tot if has_out else None,
                "in_overall": (n_in_tot / t_tot) if (has_in and t_tot > 0) else float("nan"),
                "out_overall": (n_out_tot / t_tot) if (has_out and t_tot > 0) else float("nan"),
                "selectivity": (n_out_tot / n_in_tot)
                               if (has_in and has_out and n_in_tot) else float("nan"),
            }
        return out


# ---------------------------------------------------------------------------
# THROUGHPUT — real-time analysis
# ---------------------------------------------------------------------------

def _stats(values):
    """Statistical summary of a list of throughputs (min/max/median/mean/std/p05/p95).

    Ignores NaN/inf. Returns a dict of NaN when the list is empty after
    filtering.
    """
    xs = sorted(v for v in values
                if v is not None and v == v and v not in (float("inf"), float("-inf")))
    if not xs:
        nan = float("nan")
        return {"min": nan, "max": nan, "median": nan, "mean": nan,
                "std": nan, "p05": nan, "p95": nan, "n": 0}

    def _pct(p):
        if len(xs) == 1:
            return xs[0]
        k = p * (len(xs) - 1)
        lo = int(k)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {
        "min": xs[0], "max": xs[-1], "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "p05": _pct(0.05), "p95": _pct(0.95), "n": len(xs),
    }


class Throughput:
    """Throughput time series for the real-time analysis.

    At each checkpoint (see the streaming loop of `core.pipeline`),
    `snapshot(...)` is called with the cumulated counters (windows processed,
    candidates, tested pairs, correlations) and the cumulated per-phase times.
    From those we derive:

      * an **instantaneous** throughput over the interval elapsed since the
        previous checkpoint (what a real-time stream sees at that moment);
      * a **cumulative** throughput since the start (smoothed trend).

    Streams tracked (unit → instantaneous throughput definition):
      windows    : windows/s    = Δwindows / Δwall_time      (global system throughput)
      sketch     : windows/s    = Δwindows / Δsketch_time    (capacity of the step)
      candidate  : candidates/s = Δcandidates / Δcand_time
      validate   : pairs/s      = Δpairs   / Δval_time
      monitor    : corr/s       = Δcorr    / Δmonit_time

    The `windows` stream is relative to **wall** time (real-time metric: is the
    system keeping up with the arrival rate of the windows?). The per-step
    streams are relative to the time **spent inside the step**: this is the raw
    capacity of the step, independent of what the others do. `summary()`
    summarizes each stream by min/max/median/mean/std/p05/p95 over the
    instantaneous samples.
    """

    # (internal key, unit label)
    STREAMS = (("windows", "win/s"), ("sketch", "win/s"), ("candidate", "cand/s"),
               ("validate", "pair/s"), ("monitor", "corr/s"))

    def __init__(self, total_windows=0):
        self.total_windows = int(total_windows or 0)
        self.samples = []      # list of dicts (one row per checkpoint)
        self._prev = None      # last raw state (for the deltas)

    def snapshot(self, elapsed, w_done, counts, phase_times):
        """Record a throughput sample.

        elapsed     : wall time elapsed since the start (s).
        w_done      : number of windows processed so far.
        counts      : cumulated dict {candidate, validate, monitor} (units).
        phase_times : cumulated dict {sketch, candidate, validate, monitor} (s).
        """
        cur = {"t": float(elapsed), "w": int(w_done),
               "counts": dict(counts), "pt": dict(phase_times)}
        prev = self._prev or {"t": 0.0, "w": 0,
                              "counts": {k: 0 for k in counts},
                              "pt": {k: 0.0 for k in phase_times}}
        dt = cur["t"] - prev["t"]
        frac = (cur["w"] / self.total_windows) if self.total_windows else float("nan")

        inst, cum = {}, {}
        # global "windows" stream → relative to wall time
        dw = cur["w"] - prev["w"]
        inst["windows"] = (dw / dt) if dt > 0 else float("nan")
        cum["windows"] = (cur["w"] / cur["t"]) if cur["t"] > 0 else float("nan")
        # per-step streams → relative to the time spent inside the step
        unit_src = {"sketch": ("w", None), "candidate": ("counts", "candidate"),
                    "validate": ("counts", "validate"), "monitor": ("counts", "monitor")}
        for key, (src, sub) in unit_src.items():
            if src == "w":
                u_cur, u_prev = cur["w"], prev["w"]
            else:
                u_cur = cur["counts"].get(sub, 0)
                u_prev = prev["counts"].get(sub, 0)
            pt_cur = cur["pt"].get(key, 0.0)
            pt_prev = prev["pt"].get(key, 0.0)
            d_pt = pt_cur - pt_prev
            inst[key] = ((u_cur - u_prev) / d_pt) if d_pt > 1e-9 else float("nan")
            cum[key] = (u_cur / pt_cur) if pt_cur > 1e-9 else float("nan")

        self.samples.append({
            "t": cur["t"], "frac": frac, "w_done": cur["w"],
            "n_candidates": cur["counts"].get("candidate", 0),
            "n_tested": cur["counts"].get("validate", 0),
            "n_correlated": cur["counts"].get("monitor", 0),
            "inst": inst, "cum": cum,
        })
        self._prev = cur
        return self.samples[-1]

    def overall(self):
        """Effective average throughput per stream = total units / total wall time."""
        if not self.samples:
            return {}
        last = self.samples[-1]
        t = last["t"] or float("nan")
        return {
            "windows": (last["w_done"] / t) if t else float("nan"),
            "candidate": (last["n_candidates"] / t) if t else float("nan"),
            "validate": (last["n_tested"] / t) if t else float("nan"),
            "monitor": (last["n_correlated"] / t) if t else float("nan"),
        }

    def summary(self):
        """Per-stream summary: min/max/median/mean/std/p05/p95 of the instantaneous
        throughput, plus `overall` (total units / wall time). Empty when there is
        no sample."""
        out = {"n_samples": len(self.samples)}
        overall = self.overall()
        for key, unit in self.STREAMS:
            vals = [s["inst"].get(key) for s in self.samples]
            st = _stats(vals)
            st["unit"] = unit
            if key in overall:
                st["overall"] = overall[key]
            out[key] = st
        return out

    def rows(self):
        """Rows ready for throughput.csv (the throughput = f(time) curve)."""
        out = []
        for s in self.samples:
            row = {"t": s["t"], "frac": s["frac"], "w_done": s["w_done"],
                   "n_candidates": s["n_candidates"], "n_tested": s["n_tested"],
                   "n_correlated": s["n_correlated"]}
            for key, _unit in self.STREAMS:
                row[f"{key}_per_s"] = s["inst"].get(key)
                row[f"{key}_per_s_cum"] = s["cum"].get(key)
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# RESOURCES — CPU / memory / energy (proxy), background thread
# ---------------------------------------------------------------------------

class ResourceSampler:
    """Sample CPU %, RSS memory and an energy proxy in the background.

    A daemon thread reads, every `interval` seconds (0.25 s by default), the
    usage of the process **and of its children** (useful for the
    parallel/sharded backends that fork workers). Three streams:

      cpu_pct     : aggregated CPU % (may exceed 100% = several cores).
      mem_mb      : resident memory (RSS) in MB.
      power_cores : energy proxy = CPU cores actually busy over the interval
                    ( ΔCPU_time / Δwall_time ). Integrated over the duration it
                    gives the core·seconds ≈ energy (at constant TDP).

    Negligible cost (a thread sleeping 99% of the time). When `psutil` is
    missing, we fall back to `resource.getrusage` at shutdown: only the peak
    memory and the total core·seconds are then available (no time series).

    Energy in real Joules (RAPL/powermetrics) requires privileges on most
    platforms (sudo on macOS) → deliberately NOT attempted, so as not to add any
    cost; `power_cores` is its privilege-free proxy.
    """

    def __init__(self, interval=0.25):
        self.interval = float(interval)
        self.t = []            # wall time of each sample
        self.cpu_pct = []
        self.mem_mb = []
        self.power_cores = []  # cores busy over the interval (power proxy)
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._t0 = None
        self._last_t = None
        self._last_cpu_s = None
        self._cpu_seconds_fallback = float("nan")
        self._peak_mem_fallback = float("nan")
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
        except Exception:
            self._psutil = None

    # -- process + children collection -------------------------------------
    def _read(self):
        """(aggregated cpu %, RSS MB, cumulated CPU time s) over process + children."""
        ps = self._psutil
        procs = [self._proc]
        try:
            procs += self._proc.children(recursive=True)
        except Exception:
            pass
        cpu_pct = 0.0
        mem = 0
        cpu_s = 0.0
        for p in procs:
            try:
                cpu_pct += p.cpu_percent(interval=None)
                mem += p.memory_info().rss
                ct = p.cpu_times()
                cpu_s += ct.user + ct.system
            except Exception:
                continue
        return cpu_pct, mem / 1e6, cpu_s

    def _loop(self):
        # 1st cpu_percent call: primes the counter (returns 0), so it is ignored.
        try:
            self._read()
        except Exception:
            pass
        while not self._stop.wait(self.interval):
            try:
                now = time.perf_counter()
                cpu_pct, mem_mb, cpu_s = self._read()
                self.t.append(now - self._t0)
                self.cpu_pct.append(cpu_pct)
                self.mem_mb.append(mem_mb)
                if self._last_t is not None:
                    d_wall = now - self._last_t
                    d_cpu = cpu_s - self._last_cpu_s
                    if d_wall > 0:
                        self.power_cores.append(max(0.0, d_cpu / d_wall))
                self._last_t, self._last_cpu_s = now, cpu_s
            except Exception:
                continue

    def start(self):
        self._t0 = time.perf_counter()
        if self._psutil is None:
            return self          # no thread: getrusage fallback at shutdown
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # always-available safety net: getrusage (self + children)
        try:
            import resource
            ru_self = resource.getrusage(resource.RUSAGE_SELF)
            ru_ch = resource.getrusage(resource.RUSAGE_CHILDREN)
            self._cpu_seconds_fallback = (ru_self.ru_utime + ru_self.ru_stime
                                          + ru_ch.ru_utime + ru_ch.ru_stime)
            # ru_maxrss: KB on Linux, bytes on macOS → normalized to MB.
            unit = 1e6 if os.uname().sysname == "Darwin" else 1e3
            self._peak_mem_fallback = max(ru_self.ru_maxrss, ru_ch.ru_maxrss) / unit
        except Exception:
            pass
        return self

    def cpu_seconds(self):
        """Total core·seconds ≈ energy consumed (proxy, at constant TDP)."""
        # prefers getrusage (exact); otherwise integrates power_cores·dt.
        if self._cpu_seconds_fallback == self._cpu_seconds_fallback:
            return self._cpu_seconds_fallback
        if self.power_cores and self.t:
            return statistics.fmean(self.power_cores) * (self.t[-1] or 0.0)
        return float("nan")

    def summary(self):
        """Resource summary: cpu_pct/mem_mb/power_cores (min/max/median/…) +
        peak memory + energy proxy (core·s). NaN-safe for every config."""
        cpu = _stats(self.cpu_pct)
        mem = _stats(self.mem_mb)
        power = _stats(self.power_cores)
        peak = mem["max"]
        if (peak != peak) and self._peak_mem_fallback == self._peak_mem_fallback:
            peak = self._peak_mem_fallback
        return {
            "available": self._psutil is not None,
            "n_samples": len(self.t),
            "cpu_pct": cpu,
            "mem_mb": mem,
            "mem_peak_mb": peak,
            "power_cores": power,
            "energy_cpu_seconds": self.cpu_seconds(),
        }

    def rows(self):
        """Resource time series (for resources.csv)."""
        out = []
        for i, t in enumerate(self.t):
            out.append({
                "t": t,
                "cpu_pct": self.cpu_pct[i] if i < len(self.cpu_pct) else float("nan"),
                "mem_mb": self.mem_mb[i] if i < len(self.mem_mb) else float("nan"),
                "power_cores": self.power_cores[i - 1] if 0 < i <= len(self.power_cores) else float("nan"),
            })
        return out