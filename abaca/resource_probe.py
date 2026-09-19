"""Per-arm resource accounting for the competitor campaign (2026-09-19, user request).

Every arm of a battery is run in a forked child process so that its memory and disk
figures are its own:

  peak_rss_mb        ru_maxrss of the child (the process high-water mark; includes the
                     dataset and the interpreter inherited at fork, which is the same
                     baseline for every arm of the battery)
  rss_before_mb      RSS at the start of the arm (the shared baseline)
  peak_rss_delta_mb  peak_rss_mb - rss_before_mb: the arm's own footprint
  mean_rss_mb        time-weighted mean of RSS sampled every `interval` seconds
  mean_rss_delta_mb  mean_rss_mb - rss_before_mb
  rss_samples        number of samples behind the mean
  io_read_mb /       bytes the arm read from / wrote to storage (/proc/self/io read_bytes
  io_write_mb        and write_bytes of the child, which start at zero at fork; this is
                     the storage traffic, not the syscall byte count)
  artifact_mb        size of the files the arm left in its artifact directory
  cpu_user_s /       ru_utime / ru_stime of the child
  cpu_sys_s
  wall_s             wall-clock of the arm inside the child (setup + run + artifact I/O)
  t_start_epoch /    absolute interval of the arm (Unix time), to join with a node-level power
  t_end_epoch        series (Grid'5000 kwollect: oarsub -t monitor=..., see aggregate_campaign.py)
  energy_j           (2026-09-19, user) energy over the arm from the Intel RAPL powercap
                     counters (sum over the package domains; wrap handled), plus per-domain
                     `energy_domains_j` and `energy_dram_j` when the DRAM sub-domain is
                     exposed. RAPL is a NODE-wide (socket) counter: it is the arm's energy
                     only on a node the job holds alone, which is why the campaign's N-way
                     jobs take a whole host. `energy_source` says where the number came
                     from ("rapl_powercap" or None when the counters are not readable, as
                     on a kernel that makes energy_uj root-only); `mean_power_w` =
                     energy_j / wall_s. The idle baseline measured once per battery
                     (`idle_power_w`, see idle_power) lets the aggregator report the
                     dynamic energy energy_j - idle_power_w * wall_s.

In-process fallback (`isolate=False`, or a platform without fork): the sampler still
runs but the peak is then monotone across arms and is reported as `peak_rss_mb_after`
only. The sampler reads /proc/self/statm; on a platform without /proc the RSS fields
are None. Sampling cost: one small file read per interval (default 50 ms).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import resource
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_mb() -> float | None:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * _PAGE / 2**20
    except (OSError, IndexError, ValueError):
        return None


def io_bytes() -> tuple[int | None, int | None]:
    try:
        vals = {}
        with open("/proc/self/io") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                vals[k.strip()] = int(v)
        return vals.get("read_bytes"), vals.get("write_bytes")
    except (OSError, ValueError):
        return None, None


def rapl_domains() -> list[tuple[str, str, int]]:
    """(name, path, max_range_uj) of the readable RAPL powercap domains (packages and their sub-domains)."""
    out = []
    root = Path("/sys/class/powercap")
    if not root.is_dir():
        return out
    for d in sorted(root.glob("intel-rapl:*")):
        try:
            name = (d / "name").read_text().strip()
            int((d / "energy_uj").read_text())
            mx = int((d / "max_energy_range_uj").read_text())
        except (OSError, ValueError):
            continue
        out.append((f"{d.name}:{name}", str(d), mx))
    return out


def rapl_read(domains) -> dict[str, int] | None:
    vals = {}
    for name, path, _ in domains:
        try:
            vals[name] = int(Path(path, "energy_uj").read_text())
        except (OSError, ValueError):
            return None
    return vals


def rapl_delta_j(domains, before, after) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    out = {}
    for name, _, mx in domains:
        d = after[name] - before[name]
        if d < 0:                                  # counter wrapped
            d += mx
        out[name] = d / 1e6
    return out


def _energy_fields(domains, before, after, wall) -> dict[str, Any]:
    dom = rapl_delta_j(domains, before, after)
    if not dom:
        return {"energy_j": None, "energy_dram_j": None, "energy_domains_j": None, "energy_source": None, "mean_power_w": None}
    # package domains are the top level (intel-rapl:N); sub-domains (intel-rapl:N:M) are parts of them
    pkg = sum(v for k, v in dom.items() if k.count(":") == 2)
    dram = sum(v for k, v in dom.items() if k.count(":") == 3 and k.endswith(":dram"))
    return {"energy_j": pkg, "energy_dram_j": dram if dram > 0 else None, "energy_domains_j": dom, "energy_source": "rapl_powercap",
            "mean_power_w": (pkg / wall) if wall > 0 else None}


def idle_power(seconds: float = 2.0) -> dict[str, Any]:
    """Node package power while this process sleeps; None when RAPL is not readable."""
    domains = rapl_domains()
    b = rapl_read(domains) if domains else None
    time.sleep(seconds)
    a = rapl_read(domains) if domains else None
    f = _energy_fields(domains, b, a, seconds)
    return {"idle_power_w": f["mean_power_w"], "idle_seconds": seconds, "energy_source": f["energy_source"],
            "rapl_domains": [d[0] for d in domains]}


def dir_bytes(path: str | os.PathLike | None) -> int | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if not p.is_dir():
        return None
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class RssSampler:
    """Background thread; time-weighted mean and sampled max of the process RSS."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.n = 0
        self.max_mb = None
        self._area = 0.0
        self._elapsed = 0.0

    def _run(self) -> None:
        last_t = time.perf_counter()
        last_v = rss_mb()
        while not self._stop.wait(self.interval):
            now = time.perf_counter(); v = rss_mb()
            if last_v is not None:
                self._area += last_v * (now - last_t); self._elapsed += now - last_t
            if v is not None:
                self.n += 1
                self.max_mb = v if self.max_mb is None else max(self.max_mb, v)
            last_t, last_v = now, v
        now = time.perf_counter(); v = rss_mb()
        if last_v is not None:
            self._area += last_v * (now - last_t); self._elapsed += now - last_t
        if v is not None:
            self.n += 1
            self.max_mb = v if self.max_mb is None else max(self.max_mb, v)

    def start(self) -> "RssSampler":
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        mean = self._area / self._elapsed if self._elapsed > 0 else None
        return {"mean_rss_mb": mean, "sampled_max_rss_mb": self.max_mb, "rss_samples": self.n}


def measure(fn: Callable[..., Any], *args: Any, artifact_dir: str | None = None, interval: float = 0.05, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Run fn(*args, **kwargs) in THIS process under the sampler; return (result, resources)."""
    r0, w0 = io_bytes()
    before = rss_mb()
    domains = rapl_domains()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    sampler = RssSampler(interval).start()
    e0 = rapl_read(domains) if domains else None
    t_epoch0 = time.time()
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        wall = time.perf_counter() - t0
        e1 = rapl_read(domains) if domains else None
        stats = sampler.stop()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    r1, w1 = io_bytes()
    peak = ru1.ru_maxrss / 1024.0
    res = {"rss_before_mb": before, "peak_rss_mb": peak, "peak_rss_delta_mb": (peak - before) if before is not None else None,
           "mean_rss_mb": stats["mean_rss_mb"], "mean_rss_delta_mb": (stats["mean_rss_mb"] - before) if (before is not None and stats["mean_rss_mb"] is not None) else None,
           "sampled_max_rss_mb": stats["sampled_max_rss_mb"], "rss_samples": stats["rss_samples"], "rss_sample_interval_s": interval,
           "io_read_mb": ((r1 - r0) / 2**20) if (r0 is not None and r1 is not None) else None,
           "io_write_mb": ((w1 - w0) / 2**20) if (w0 is not None and w1 is not None) else None,
           "artifact_mb": (dir_bytes(artifact_dir) or 0) / 2**20 if artifact_dir is not None else None,
           "cpu_user_s": ru1.ru_utime - ru0.ru_utime, "cpu_sys_s": ru1.ru_stime - ru0.ru_stime, "wall_s": wall,
           # absolute interval of the arm, for joining with node-level monitoring series (kwollect power)
           "t_start_epoch": t_epoch0, "t_end_epoch": t_epoch0 + wall}
    res.update(_energy_fields(domains, e0, e1, wall))
    return result, res


def _child(conn, fn, args, kwargs, artifact_dir, interval):
    try:
        result, res = measure(fn, *args, artifact_dir=artifact_dir, interval=interval, **kwargs)
        conn.send(("ok", result, res))
    except BaseException as exc:  # noqa: BLE001 - the parent decides
        conn.send(("error", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
    finally:
        conn.close()


def run_isolated(fn: Callable[..., Any], *args: Any, artifact_dir: str | None = None, interval: float = 0.05,
                 isolate: bool = True, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Run fn in a forked child with resource accounting; returns (result, resources).

    Raises RuntimeError with the child's traceback when fn raised there. `isolate=False`
    (or no fork on this platform) runs fn in-process with the same accounting, with
    `isolated=False` in the returned dict.
    """
    if not isolate or "fork" not in mp.get_all_start_methods():
        result, res = measure(fn, *args, artifact_dir=artifact_dir, interval=interval, **kwargs)
        res["isolated"] = False
        return result, res
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, fn, args, kwargs, artifact_dir, interval))
    proc.start()
    child.close()
    try:
        msg = parent.recv()
    except EOFError:
        proc.join()
        raise RuntimeError(f"isolated run died without a result (exit code {proc.exitcode}; out of memory?)")
    proc.join()
    if msg[0] != "ok":
        raise RuntimeError(f"{msg[1]}\n--- child traceback ---\n{msg[2]}")
    _, result, res = msg
    res["isolated"] = True
    res["child_exit_code"] = proc.exitcode
    return result, res
