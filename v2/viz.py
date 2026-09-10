"""Interactive viz launcher: explore correlated windows produced by a pipeline run.

Run:
    python -m v2.viz --results /Users/benoit/test/results [--dataset DATA.csv] [--port 8050]
    python -m v2.viz --pipeline /Users/benoit/test/pipeline-run.json [--dataset OVERRIDE.csv]

Remote (over SSH, key-based auth required):
    python -m v2.viz --pipeline user@host:/abs/path/pipeline-run.json
  The viewer runs locally; results are rsync'd into ~/.corrtrack/cache/remote/
  and refreshed every 5s in the background.

Design notes:
- Pure Flask + pandas + matplotlib. No CDN, no plotly.
- Nothing is preloaded: every panel is filled by an explicit user action.
- The source CSV is sliced per request (`usecols=[id]`, `skiprows`, `nrows`).
- `correlated.csv` is read with pandas chunks for pagination/filtering.

CSV conventions (see steps/io.py):
- ASOS layout : col 0 = date (str), col 1 = hour, cols 2+ = series.
  `t` in correlated.csv is the integer "hours since the first row" — which
  equals the 0-based row index when the source has 1 row per hour with no gap.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402


# ---------------------------------------------------------------------------
# Cache root — user-wide, NOT inside the repo. Survives `git clean -fdx`
# and is shared across multiple checkouts.
# ---------------------------------------------------------------------------
CACHE_ROOT = Path.home() / ".corrtrack" / "cache"
# Legacy location (used before May 2026). We migrate transparently on first
# launch so existing quality/remote caches don't get lost.
_LEGACY_CACHE = Path(__file__).parent / "cache"


def _ensure_cache_root() -> Path:
    """Create CACHE_ROOT, migrating from the legacy v2/cache/ location if
    we find content there but not in the new home."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not _LEGACY_CACHE.is_dir():
        return CACHE_ROOT
    moved: list[str] = []
    try:
        for item in _LEGACY_CACHE.iterdir():
            # airports.json stays with the repo (bundled reference data,
            # also written there by maintenance scripts). Everything else
            # is user data → migrate.
            if item.name == "airports.json":
                continue
            dst = CACHE_ROOT / item.name
            if dst.exists():
                continue  # don't clobber new-location content
            try:
                item.rename(dst)
                moved.append(item.name)
            except OSError:
                # Cross-device or other rename failure — fall back to copy.
                import shutil
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
                moved.append(item.name + " (copied)")
    except Exception as e:
        print(f"[viz] cache migration warning: {e}", file=sys.stderr)
    if moved:
        print(f"[viz] migrated cache to {CACHE_ROOT}: {', '.join(moved)}",
              file=sys.stderr)
    return CACHE_ROOT


# ---------------------------------------------------------------------------
# State / config
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    results_root: Path
    source_csv: Path | None
    # cache: pipeline_dir -> {"window_size":..., "summary":..., "source":..., "uniq":DataFrame|None}
    cache: dict[str, dict[str, Any]]
    lock: Lock
    pipeline_json: Path | None = None
    pipeline_obj: dict | None = None
    # Remote SSH mirror — None when running against local files.
    mirror: "RemoteMirror | None" = None


STATE: AppState  # set in main()


# ---------------------------------------------------------------------------
# Remote SSH mirror — rsync-backed, transparent to the rest of the app
# ---------------------------------------------------------------------------
#
# When `--pipeline user@host:/abs/path/pipeline-run.json` is passed, we mirror
# the remote tree into v2/cache/remote/<host>_<mangled-path>/ so the rest of
# the code keeps reading plain local files. A background thread re-runs rsync
# every few seconds so live runs stay observable.

@dataclass
class RemoteSpec:
    user: str | None
    host: str
    path: str  # absolute remote path

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


# `[user@]host:/abs/path` — host must be a non-empty hostname and path must be
# absolute, so we don't mis-classify Windows-style or relative paths.
_REMOTE_RE = re.compile(r"^(?:([^@:/\s]+)@)?([A-Za-z0-9._-]+):(/[^\s]+)$")


def _parse_remote(s: str | None) -> RemoteSpec | None:
    if not s:
        return None
    # Local paths win — anything that resolves as an existing file/dir locally
    # is treated as local even if the regex would match.
    p = Path(s).expanduser()
    if p.exists():
        return None
    m = _REMOTE_RE.match(s)
    if not m:
        return None
    return RemoteSpec(user=m.group(1), host=m.group(2), path=m.group(3))


class RemoteMirror:
    """Mirrors a remote tree into a local cache via ssh+rsync.

    Paths are mirrored by their absolute remote path: `/srv/foo/bar.csv` →
    `<cache>/<key>/srv/foo/bar.csv`. The rest of the app sees local files.
    """

    def __init__(self, spec: RemoteSpec, cache_root: Path,
                 *, ssh_connect_timeout: int = 30):
        self.spec = spec
        key = re.sub(r"[^A-Za-z0-9._-]", "_",
                     f"{spec.user or 'nouser'}@{spec.host}{spec.path}")
        self.local_root = cache_root / "remote" / key
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._sync_thread: threading.Thread | None = None
        self._sync_dirs: list[str] = []
        self._sync_interval = 5.0
        self._stop = threading.Event()
        # SSH connect timeout, in seconds, applied uniformly to test_ssh,
        # `ssh ...` probes (list_remote, remote_path_exists) and to the
        # `ssh` invoked by rsync via `-e`. Override via --ssh-timeout.
        self.ssh_connect_timeout = max(5, int(ssh_connect_timeout))
        # macOS now ships openrsync (BSD reimplementation) by default, and
        # older macOS versions shipped rsync 2.6.9 (2006). Neither knows
        # --info=progress2 or --no-inc-recursive. Detect and degrade.
        self._rsync_version = self._detect_rsync_version()
        if self._rsync_version >= (3, 1):
            self._progress_flags = ["--info=progress2", "--no-inc-recursive"]
        else:
            self._progress_flags = ["--progress"]
            extra = ""
            if "openrsync" in self._rsync_version_str.lower():
                extra = (" (macOS' openrsync lacks the modern --info=progress2 "
                         "global progress bar). ")
            print(f"[viz] note: local rsync = {self._rsync_version_str}"
                  f"{extra} → falling back to per-file --progress. "
                  f"Run `brew install rsync` for the nicer single-line "
                  f"global progress display.", file=sys.stderr)
        # Updated by every periodic / forced sync — surfaced via /api/sync_now
        # and the UI status bar.
        self._last_sync: dict[str, Any] = {
            "at": 0.0,         # epoch seconds of last completed cycle
            "duration": 0.0,   # seconds the last cycle took
            "n_changes": 0,    # files added/changed in the last cycle
            "n_cycles": 0,     # total cycles since startup
            "last_error": "",  # most recent error (cleared on success)
            "interval": self._sync_interval,
        }

    def remote_to_local(self, remote_abs: str) -> Path:
        return self.local_root / remote_abs.lstrip("/")

    def _ssh_opts(self) -> list[str]:
        """SSH options shared by every ssh/rsync invocation."""
        return ["-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={self.ssh_connect_timeout}",
                "-o", "StrictHostKeyChecking=accept-new"]

    def _detect_rsync_version(self) -> tuple[int, int]:
        """Best-effort parse of `rsync --version`. Returns (major, minor)
        or (0, 0) if unknown. The version string is stored on the instance
        in `_rsync_version_str` for diagnostics."""
        self._rsync_version_str = "unknown"
        try:
            r = subprocess.run(["rsync", "--version"],
                               capture_output=True, text=True, timeout=5)
        except Exception:
            return (0, 0)
        head = (r.stdout or "").splitlines()[0] if r.stdout else ""
        self._rsync_version_str = head.strip() or "unknown"
        # macOS: "openrsync: protocol version 29"
        # GNU rsync: "rsync  version 3.2.7  protocol version 31"
        m = re.search(r"version\s+(\d+)\.(\d+)", head)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))

    @staticmethod
    def _ssh_hint(target: str) -> str:
        return (f"hint: SSH key auth required. Try `ssh {target} true` from a "
                f"terminal; if that prompts for a password or fails, fix your "
                f"~/.ssh/config / authorized_keys before retrying.")

    def test_ssh(self, *, retries: int = 1) -> tuple[bool, str]:
        """One blocking SSH round-trip to verify we can talk to the host.
        Retries once on plain timeout (transient network blips are very
        common on VPNs / slow links)."""
        sub_timeout = self.ssh_connect_timeout * 2 + 5
        last_err = ""
        for attempt in range(retries + 1):
            try:
                r = subprocess.run(
                    ["ssh"] + self._ssh_opts() + [self.spec.target, "true"],
                    capture_output=True, text=True, timeout=sub_timeout,
                )
            except FileNotFoundError:
                return False, "ssh not found in PATH"
            except subprocess.TimeoutExpired:
                last_err = (f"ssh connection timed out "
                            f"(>{sub_timeout}s, ConnectTimeout={self.ssh_connect_timeout})"
                            f" — bump with --ssh-timeout N if your link is slow")
                if attempt < retries:
                    print(f"[viz] ssh test timed out (attempt {attempt+1}/"
                          f"{retries+1}) — retrying…", file=sys.stderr)
                    continue
                return False, last_err
            if r.returncode != 0:
                return False, (r.stderr.strip()
                               or f"ssh exited with code {r.returncode}")
            return True, ""
        return False, last_err or "ssh test failed"

    def _rsync(self, args: list[str], *,
               progress: bool = False,
               itemize: bool = False,
               label: str = "") -> subprocess.CompletedProcess:
        """Run rsync. With `progress=True`, rsync's native `--info=progress2`
        output is piped straight to stderr (single self-updating line via
        embedded `\\r`); we still capture stderr so errors are reported.
        With `itemize=True`, adds `-i` so the caller can count changed paths
        from stdout."""
        # -a archive, -z compress, --partial keep partial transfers for retry.
        cmd = ["rsync", "-az", "--partial",
               "-e", "ssh " + " ".join(self._ssh_opts())]
        if progress:
            cmd += list(self._progress_flags)
        if itemize:
            cmd += ["-i"]
        cmd += args
        if not progress:
            return subprocess.run(cmd, capture_output=True, text=True)
        sys.stderr.write(f"[viz] sync {label}…\n")
        sys.stderr.flush()
        # In progress mode rsync uses stdout to paint the live line. We let it
        # pass through, but we ALSO buffer it so we can surface it on failure
        # (otherwise the error message would be empty — rsync sometimes writes
        # the actual error on stdout when it can't even open the remote dir).
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out_buf: list[bytes] = []
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            try:
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
            except Exception:
                pass
            out_buf.append(chunk)
        err_b = proc.stderr.read() if proc.stderr else b""
        proc.wait()
        out = b"".join(out_buf).decode("utf-8", errors="replace")
        err = err_b.decode("utf-8", errors="replace")
        sys.stderr.write("\n")
        sys.stderr.flush()
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)

    def fetch_file(self, remote_abs: str, *, progress: bool = False) -> Path:
        local = self.remote_to_local(remote_abs)
        local.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            r = self._rsync([f"{self.spec.target}:{remote_abs}", str(local)],
                            progress=progress,
                            label=Path(remote_abs).name)
        if r.returncode != 0:
            raise RuntimeError(
                self._fmt_rsync_err(r, what=remote_abs))
        return local

    def _fmt_rsync_err(self, r: subprocess.CompletedProcess,
                       *, what: str) -> str:
        stderr_s = (r.stderr or "").strip()
        stdout_s = (r.stdout or "").strip()
        msg = stderr_s or stdout_s or "(rsync produced no output)"
        # Common rsync exit codes worth annotating.
        code_hint = {
            12: "protocol/transport error — often means the remote shell "
                 "printed garbage on stderr (a chatty .bashrc/.zshrc or "
                 "MOTD broke rsync's protocol)",
            23: "partial transfer — some files missing / unreadable",
            24: "vanished source files",
            30: "I/O timeout",
            127: "remote shell couldn't find `rsync` in PATH (try `ssh "
                 f"{self.spec.target} which rsync`; if missing, add it via "
                 "`module load rsync` in ~/.bashrc or pass --rsync-path)",
            255: "ssh failed (auth / network / closed by remote)",
        }.get(r.returncode, "")
        head = f"rsync failed (exit {r.returncode}"
        if code_hint:
            head += f" — {code_hint}"
        head += f") for {what}:"
        # When rsync says nothing, the most common HPC cause is that the
        # remote shell doesn't expose `rsync` in its non-interactive PATH.
        # Probe and surface the result so we don't leave the user guessing.
        diag = ""
        if not stderr_s and not stdout_s:
            try:
                p = subprocess.run(
                    ["ssh"] + self._ssh_opts() + [self.spec.target,
                     "command -v rsync || which rsync || echo NOT_FOUND; "
                     "echo PATH=$PATH"],
                    capture_output=True, text=True,
                    timeout=self.ssh_connect_timeout + 10)
                diag = "\nremote probe:\n  " + p.stdout.strip().replace(
                    "\n", "\n  ")
                if p.stderr.strip():
                    diag += "\n  (stderr) " + p.stderr.strip().replace(
                        "\n", "\n  ")
            except Exception as e:
                diag = f"\nremote probe failed: {e}"
        cmd_quoted = " ".join(re.sub(r"([\"' \\$`])", r"\\\1", a) or "''"
                              for a in r.args)
        return (f"{head}\n{msg}{diag}\nfailed command (copy to reproduce):\n"
                f"  {cmd_quoted}\n" + self._ssh_hint(self.spec.target))

    def remote_path_exists(self, remote_abs: str) -> tuple[bool, str]:
        """Returns (exists, kind) where kind is 'dir' / 'file' / '' (missing).
        Uses a single SSH round-trip so we can pre-flight a sync."""
        try:
            r = subprocess.run(
                ["ssh"] + self._ssh_opts() + [self.spec.target,
                 # `[ -d X ] && echo dir || ([ -f X ] && echo file || echo)`
                 f"if [ -d {remote_abs!r} ]; then echo dir; "
                 f"elif [ -f {remote_abs!r} ]; then echo file; "
                 f"else echo; fi"],
                capture_output=True, text=True,
                timeout=self.ssh_connect_timeout + 10,
            )
        except Exception:
            return False, ""
        kind = (r.stdout or "").strip()
        return (kind != ""), kind

    def sync_tree(self, remote_abs_dir: str, *,
                  excludes: tuple[str, ...] = (),
                  progress: bool = False,
                  itemize: bool = False) -> tuple[Path, list[str]]:
        """Returns (local_path, changed_paths). `changed_paths` is non-empty
        only when `itemize=True` (otherwise an empty list)."""
        local = self.remote_to_local(remote_abs_dir)
        local.mkdir(parents=True, exist_ok=True)
        args: list[str] = []
        for ex in excludes:
            args += ["--exclude", ex]
        args += [f"{self.spec.target}:{remote_abs_dir.rstrip('/')}/",
                 str(local) + "/"]
        with self._lock:
            r = self._rsync(args, progress=progress, itemize=itemize,
                            label=Path(remote_abs_dir).name or "tree")
        if r.returncode != 0:
            raise RuntimeError(self._fmt_rsync_err(r, what=remote_abs_dir))
        # rsync -i format: `<flags> <path>` per line, e.g.
        #   `>f.st...... results/run42/run.log`
        # We strip the leading 11-char itemize column to get the path.
        changes: list[str] = []
        if itemize and r.stdout:
            for ln in r.stdout.splitlines():
                if len(ln) > 12 and ln[10:11] == " ":
                    changes.append(ln[11:].strip())
        return local, changes

    @property
    def last_sync(self) -> dict[str, Any]:
        return dict(self._last_sync)

    def list_remote(self, remote_abs_dir: str, *, pattern: str = "*") -> list[str]:
        """ls -1 a remote dir. Returns basenames matching `pattern`."""
        try:
            r = subprocess.run(
                ["ssh"] + self._ssh_opts() + [self.spec.target,
                 f"cd {remote_abs_dir!r} 2>/dev/null && ls -1 {pattern} 2>/dev/null || true"],
                capture_output=True, text=True,
                timeout=self.ssh_connect_timeout + 10,
            )
        except Exception:
            return []
        if r.returncode != 0:
            return []
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]

    def start_periodic_sync(self, remote_dirs: list[str],
                             interval: float = 5.0) -> None:
        self._sync_dirs = list(remote_dirs)
        self._sync_interval = interval
        self._last_sync["interval"] = interval
        if self._sync_thread is not None:
            return
        t = threading.Thread(target=self._sync_loop, daemon=True,
                              name="viz-remote-sync")
        self._sync_thread = t
        t.start()

    def sync_now(self) -> dict[str, Any]:
        """Force one immediate sync of every tracked dir. Updates _last_sync."""
        return self._run_one_cycle()

    def _run_one_cycle(self) -> dict[str, Any]:
        t0 = time.time()
        total_changes: list[str] = []
        err = ""
        for d in list(self._sync_dirs):
            try:
                # Skip silently when the dir doesn't exist yet (typical
                # right after a pipeline launch, before the first run has
                # populated its output folder). Avoids spamming rsync errors.
                exists, kind = self.remote_path_exists(d)
                if not exists:
                    continue
                if kind != "dir":
                    raise RuntimeError(f"remote path is not a directory: {d}")
                _local, changes = self.sync_tree(d, itemize=True)
                total_changes.extend(changes)
            except Exception as e:
                msg = f"sync of {d} failed: {e}"
                err = (err + "\n" + msg).strip() if err else msg
                print(f"[viz] {msg}", file=sys.stderr)
        dt = time.time() - t0
        self._last_sync.update(
            at=time.time(),
            duration=round(dt, 3),
            n_changes=len(total_changes),
            n_cycles=self._last_sync["n_cycles"] + 1,
            last_error=err,
        )
        # Per-cycle log: silent when nothing changed (so we don't spam every
        # 5s), one line when something moved or when there's an error.
        if total_changes:
            head = ", ".join(total_changes[:3])
            extra = f" (+{len(total_changes)-3} more)" if len(total_changes) > 3 else ""
            print(f"[viz] sync: {len(total_changes)} change(s) in {dt:.1f}s — "
                  f"{head}{extra}", file=sys.stderr)
        elif err:
            pass  # already printed above
        return {**self._last_sync, "changes": total_changes}

    def _sync_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self._sync_interval)
            self._run_one_cycle()


# ---------------------------------------------------------------------------
# Helpers: file discovery & metadata
# ---------------------------------------------------------------------------

def _list_pipeline_dirs(root: Path) -> list[Path]:
    """A pipeline dir is any subdir that contains a `correlated.csv` OR a
    `run.log` (so we also see runs that are currently in progress and don't
    have outputs yet).

    Search depth 1 and 2 below `root`. The root itself is included ONLY when
    no sub-pipeline is found (so that pointing `--results` at a single
    pipeline directory still works).
    """
    def _looks_like_run(d: Path) -> bool:
        return (d / "correlated.csv").is_file() or (d / "run.log").is_file()

    out: list[Path] = []
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _looks_like_run(child):
            out.append(child)
        for sub in sorted(child.iterdir()) if child.is_dir() else []:
            if sub.is_dir() and _looks_like_run(sub):
                out.append(sub)
    if not out and _looks_like_run(root):
        out.append(root)
    seen, uniq = set(), []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


# Parse the latest "CP:" line from a run.log. Format produced by v2/pipeline.py:
#   CP: 464/717 (t=215940) | corrtrack back=python | cand=133513 corr=40476 | ... | ETA 2s
_CP_RE = re.compile(
    r"CP:\s*(\d+)/(\d+)\s*\(t=(\d+)\).*?cand=(\d+)\s*corr=(\d+).*?"
    r"([\d.]+)\s*win/s\s*([\d,]+)\s*cand/s.*?ETA\s+(\d+)s"
)
_DONE_RE = re.compile(r"===\s*done:\s*\d+\s*correlated\s*/\s*\d+\s*candidates.*?in\s*([\d.]+)s")
# Some run.log final summary lines we use as a fallback when result.json is
# missing the matching field. Example log lines:
#   "      runtime      11.4489s  100.0%"
#   "      val_time      3.3863s   29.6%"
_RUNLOG_RT_RE = re.compile(r"\bruntime\s+([\d.]+)\s*s\b", re.IGNORECASE)
_RUNLOG_PHASE_RE = re.compile(
    r"\b(read|windows|sketch|index|ingest|select|candidates|"
    r"validate_sketches|validate|monitor|evict|cand_time|val_time|"
    r"sketch_time|monit_time|setup|opt_time)"
    r"\s+([\d.]+)\s*s\b",
    re.IGNORECASE,
)


def _parse_runlog_runtime(log_path: Path) -> tuple[float, dict[str, float]]:
    """Parse the trailing summary block of a run.log to recover `runtime` and
    per-phase timings. Returns (runtime_seconds, {phase: seconds}).
    Used as a fallback when result.json is missing / has runtime=0 — the
    final 'done' block in run.log always lists the wall time even when
    the JSON write is partial."""
    if not log_path.is_file():
        return 0.0, {}
    rt = 0.0
    phases: dict[str, float] = {}
    try:
        # Tail: only the last ~200 lines matter (the summary block).
        with log_path.open(errors="ignore") as fh:
            tail = fh.readlines()[-200:]
    except Exception:
        return 0.0, {}
    for line in tail:
        m = _RUNLOG_RT_RE.search(line)
        if m and "100.0%" in line:
            try:
                rt = max(rt, float(m.group(1)))
            except ValueError:
                pass
        mp = _RUNLOG_PHASE_RE.search(line)
        if mp:
            name = mp.group(1).lower()
            try:
                phases[name] = float(mp.group(2))
            except ValueError:
                pass
    return rt, phases


def _term_progress(done: int, total: int, *, label: str = "",
                   width: int = 32, start_ts: float | None = None,
                   final: bool = False) -> None:
    """Render an in-place progress bar to stderr (terminal only).

    Writes a single line with a carriage return so it updates in place;
    pass final=True to terminate with a newline. No-op when stderr is not
    a TTY (e.g. piped logs) — there we emit a plain line every ~10% instead
    so the progress is still legible in a log file.
    """
    total = max(1, total)
    frac = min(1.0, done / total)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    eta = ""
    if start_ts is not None and done > 0 and frac < 1.0:
        elapsed = time.time() - start_ts
        remaining = elapsed / frac - elapsed
        eta = f" · ETA {remaining:4.0f}s"
    msg = f"[viz] {label} [{bar}] {done}/{total} ({frac*100:4.1f}%){eta}"
    if sys.stderr.isatty():
        sys.stderr.write("\r" + msg + ("\n" if final else ""))
        sys.stderr.flush()
    elif final or done == total:
        # Non-TTY: only log the final summary line (avoids log spam).
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def _read_config_json(pdir: Path) -> dict:
    """Read `<run_dir>/config.json` — written by the runner before the run
    starts, so it's available even when result.json hasn't been flushed yet.
    Provides authoritative backend / index / sketch values.

    The actual params live under `effective_params` (the runner's normalized
    final view); we flatten that out + the `run.params` dict (hyphenated key
    aliases) so the caller can do plain `.get('backend')` regardless of
    where the value was originally written.
    """
    cf = pdir / "config.json"
    if not cf.is_file():
        return {}
    try:
        raw = json.loads(cf.read_text())
    except Exception:
        return {}
    out: dict = {}
    eff = raw.get("effective_params") or {}
    if isinstance(eff, dict):
        out.update(eff)
    run_params = ((raw.get("run") or {}).get("params") or {})
    if isinstance(run_params, dict):
        # Normalize hyphen aliases (`index-backend` → `index_backend`).
        for k, v in run_params.items():
            nk = str(k).replace("-", "_")
            out.setdefault(nk, v)
    # Top-level fields take last priority — they may overlap.
    for k in ("backend", "index_backend", "sketch_method", "key_mode"):
        if k not in out and raw.get(k) is not None:
            out[k] = raw[k]
    return out


def _parse_run_log(log_path: Path) -> dict | None:
    """Return progress dict from the latest CP line in a run.log, or None."""
    if not log_path.is_file():
        return None
    last = None
    last_line_ts = None
    try:
        with log_path.open(errors="ignore") as f:
            for line in f:
                m = _CP_RE.search(line)
                if m:
                    last = m
                    # Capture timestamp prefix if present (YYYY-MM-DD HH:MM:SS)
                    tsm = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
                    if tsm:
                        last_line_ts = tsm.group(1)
    except Exception:
        return None
    if not last:
        return None
    cur, total = int(last.group(1)), int(last.group(2))
    cand, corr = int(last.group(4)), int(last.group(5))
    win_s = float(last.group(6))
    cand_s = float(last.group(7).replace(",", ""))
    eta = int(last.group(8))
    return {
        "current": cur, "total": total,
        "progress": cur / total if total else 0.0,
        "n_candidates": cand, "n_correlated": corr,
        "win_per_s": win_s, "cand_per_s": cand_s,
        "eta_seconds": eta,
        "last_line_at": last_line_ts,
    }


def _detect_run_status(pdir: Path) -> tuple[str, dict | None]:
    """Return (status, progress_dict|None) for a run directory.

    status ∈ {"done", "running", "failed", "starting", "unknown"}
    progress_dict from _parse_run_log() when running.
    """
    rj = pdir / "result.json"
    if rj.is_file():
        try:
            obj = json.loads(rj.read_text())
            if obj.get("runtime", 0) > 0:
                return "done", None
            # Some pipeline versions (notably sharded variants) finish but
            # leave `runtime` at 0 in result.json. If the run also wrote a
            # `correlated.csv`, it really did complete — treat as done even
            # though the wall-clock field is missing. The viewer will use
            # the sum of phase timings as a fallback elsewhere.
            if (pdir / "correlated.csv").is_file():
                return "done", None
        except Exception:
            pass
        return "failed", None
    log = pdir / "run.log"
    if log.is_file():
        prog = _parse_run_log(log)
        if prog:
            return "running", prog
        return "starting", None
    return "unknown", None


def _read_summary(pipeline_dir: Path) -> dict[str, Any]:
    """Parse summary.csv (metric,value)."""
    f = pipeline_dir / "summary.csv"
    if not f.is_file():
        return {}
    out: dict[str, Any] = {}
    with f.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            k, v = row[0], row[1]
            # try numeric / bool
            if v.lower() == "true":
                vv: Any = True
            elif v.lower() == "false":
                vv = False
            else:
                try:
                    vv = int(v) if v.lstrip("-").isdigit() else float(v)
                except ValueError:
                    vv = v
            out[k] = vv
    return out


def _read_result_json(pipeline_dir: Path) -> dict[str, Any]:
    f = pipeline_dir / "result.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _read_timeseries_csv(path: Path, cols: list[str], cap: int = 400) -> list[dict]:
    """Read a time-series CSV (throughput/resources) into a list of float dicts.
    Uniformly sub-samples down to `cap` points max (the curve stays faithful, the
    JSON payload stays light even on runs with thousands of checkpoints).
    NaN/empty -> None."""
    if not path.is_file():
        return []
    rows: list[dict] = []
    try:
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                row: dict[str, Any] = {}
                for c in cols:
                    v = r.get(c, "")
                    try:
                        row[c] = float(v) if v not in ("", "nan", "NaN") else None
                    except (TypeError, ValueError):
                        row[c] = None
                rows.append(row)
    except Exception:
        return []
    if len(rows) > cap:                       # uniform sub-sampling
        step = len(rows) / cap
        rows = [rows[min(int(i * step), len(rows) - 1)] for i in range(cap)]
    return rows


_TPUT_SERIES_COLS = ["t", "frac", "w_done", "windows_per_s", "windows_per_s_cum",
                     "sketch_per_s", "candidate_per_s", "validate_per_s",
                     "monitor_per_s"]
_RES_SERIES_COLS = ["t", "cpu_pct", "mem_mb", "power_cores"]


def _read_throughput(pipeline_dir: Path) -> list[dict]:
    """Throughput series (throughput.csv): the throughput = f(time) curve."""
    return _read_timeseries_csv(pipeline_dir / "throughput.csv", _TPUT_SERIES_COLS)


def _read_resources(pipeline_dir: Path) -> list[dict]:
    """Resource series (resources.csv): CPU %, memory MB, energy proxy."""
    return _read_timeseries_csv(pipeline_dir / "resources.csv", _RES_SERIES_COLS)


def _find_dataset_near(results_root: Path) -> Path | None:
    """Pick the `dataset` field from a pipeline-*.json next to the results."""
    for p in _pipeline_jsons(results_root):
        try:
            obj = _load_jsonc(p.read_text())
        except Exception:
            continue
        ds = obj.get("dataset")
        if ds:
            ds_path = Path(ds)
            if not ds_path.is_absolute():
                ds_path = (p.parent / ds_path).resolve()
            if ds_path.is_file():
                return ds_path
    return None


def _load_jsonc(text: str) -> Any:
    """json.loads with tolerance for `//` and `/* */` comments (outside strings)."""
    out: list[str] = []
    i, n, in_str, esc = 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True; out.append(c); i += 1; continue
        if c == "/" and i + 1 < n:
            nx = text[i + 1]
            if nx == "/":
                j = text.find("\n", i)
                i = n if j == -1 else j
                continue
            if nx == "*":
                j = text.find("*/", i + 2)
                i = n if j == -1 else j + 2
                continue
        out.append(c)
        i += 1
    return json.loads("".join(out))


def _pipeline_jsons(results_root: Path) -> list[Path]:
    """Find pipeline-*.json files near the results dir (parent + self)."""
    out = list(results_root.parent.glob("pipeline-*.json"))
    out += list(results_root.glob("pipeline-*.json"))
    # uniq
    seen, uniq = set(), []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


_RUN_META_CACHE: dict[str, dict] | None = None


def _run_meta_by_name() -> dict[str, dict]:
    """Build `name -> {mode, params, optimize}` from all pipeline-*.json files.

    Used to recover the *declared* run mode (bf vs corrtrack) which can't be
    reliably inferred from result.json alone.
    """
    global _RUN_META_CACHE
    if _RUN_META_CACHE is not None:
        return _RUN_META_CACHE
    out: dict[str, dict] = {}
    for p in _pipeline_jsons(STATE.results_root):
        try:
            obj = _load_jsonc(p.read_text())
        except Exception:
            continue
        for run in obj.get("runs", []) or []:
            name = run.get("name")
            if not name:
                continue
            out[name] = {
                "mode": run.get("mode") or "",
                "params": run.get("params") or {},
                "optimize": bool(run.get("optimize")),
                "pipeline_json": str(p),
            }
    _RUN_META_CACHE = out
    return out


def _resolve_source_for(pipeline_dir: Path) -> Path | None:
    if STATE.source_csv is not None:
        return STATE.source_csv
    return _find_dataset_near(STATE.results_root)


def _pipeline_meta(pipeline_dir: Path) -> dict[str, Any]:
    key = str(pipeline_dir)
    with STATE.lock:
        c = STATE.cache.setdefault(key, {})
    if "summary" not in c:
        c["summary"] = _read_summary(pipeline_dir)
        c["result"] = _read_result_json(pipeline_dir)
        c["window_size"] = int(c["summary"].get("param.window_size", 168) or 168)
        c["source"] = _resolve_source_for(pipeline_dir)
        c["throughput"] = _read_throughput(pipeline_dir)   # throughput = f(time) curve
        c["resources"] = _read_resources(pipeline_dir)     # CPU/mem/energy
    return c


# ---------------------------------------------------------------------------
# Source CSV header (cached): id -> column index
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[str, dict[str, int]] = {}


def _source_header(source: Path) -> dict[str, int]:
    k = str(source)
    if k in _HEADER_CACHE:
        return _HEADER_CACHE[k]
    with source.open() as fh:
        header = next(csv.reader(fh))
    mapping = {h.strip().strip('"'): i for i, h in enumerate(header)}
    _HEADER_CACHE[k] = mapping
    return mapping


def _source_ids(source: Path) -> list[str]:
    h = _source_header(source)
    # ASOS layout: first two cols are date/time, rest are series
    items = sorted(h.items(), key=lambda kv: kv[1])
    return [k for k, i in items if i >= 2]


_TIMES_CACHE: dict[str, np.ndarray] = {}
_DATETIMES_CACHE: dict[str, np.ndarray] = {}


def _source_datetimes(source: Path) -> np.ndarray:
    """datetime64[ms] for every data row of the ASOS source CSV. Cached."""
    k = str(source)
    if k in _DATETIMES_CACHE:
        return _DATETIMES_CACHE[k]
    df = pd.read_csv(source, usecols=[0, 1], dtype=str)
    dt = pd.to_datetime(df.iloc[:, 0].astype(str) + "T" + df.iloc[:, 1].astype(str),
                        format="%Y-%m-%dT%H").to_numpy().astype("datetime64[ms]")
    _DATETIMES_CACHE[k] = dt
    return dt


def _source_times(source: Path) -> np.ndarray:
    """Hours-since-first-row for every data row of the ASOS source CSV.

    The CSV may have gaps in date+time → `t` (the "hour offset" used by
    correlated.csv) is NOT the row index. Cached per source.
    """
    k = str(source)
    if k in _TIMES_CACHE:
        return _TIMES_CACHE[k]
    df = pd.read_csv(source, usecols=[0, 1], dtype=str)
    dt = pd.to_datetime(df.iloc[:, 0].astype(str) + "T" + df.iloc[:, 1].astype(str),
                        format="%Y-%m-%dT%H")
    hours = ((dt - dt.iloc[0]) // pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
    _TIMES_CACHE[k] = hours
    return hours


def _t_to_row(source: Path, t: int) -> int:
    """Find the dense row index whose hour-offset equals t (or the nearest below)."""
    hours = _source_times(source)
    idx = int(np.searchsorted(hours, t))
    if idx >= len(hours):
        return len(hours) - 1
    if int(hours[idx]) == int(t):
        return idx
    # not exactly present: fall back to the closest one
    if idx == 0:
        return 0
    return idx - 1 if abs(int(hours[idx - 1]) - t) <= abs(int(hours[idx]) - t) else idx


# ---------------------------------------------------------------------------
# Slice a window of values out of the source CSV
# ---------------------------------------------------------------------------

def _read_window_values(source: Path, series_id: str, t: int, length: int,
                        pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, values) for `series_id`, `length` consecutive DENSE rows
    starting at the row whose hour-offset is `t`, with `pad` extra dense rows
    on each side.

    `xs` is the *hour-offset* of each row (not necessarily contiguous when the
    source CSV has gaps). v2's iter_windows operates on dense rows, so we read
    dense rows here too — but we still report the actual time on the x-axis.
    """
    header = _source_header(source)
    if series_id not in header:
        raise KeyError(f"unknown series: {series_id}")
    hours = _source_times(source)
    n_rows = len(hours)
    idx = _t_to_row(source, t)
    start = max(0, idx - pad)
    end = min(n_rows, idx + length + pad)
    n = end - start
    if n <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    skip = range(1, start + 1)
    df = pd.read_csv(source, usecols=[series_id], skiprows=skip, nrows=n,
                     dtype=str, keep_default_na=False, na_values=[""])
    vals = pd.to_numeric(df[series_id], errors="coerce").to_numpy(dtype=float)
    xs = hours[start:start + len(vals)].copy()
    return xs, vals


def _window_stats(vals: np.ndarray) -> dict[str, float]:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "range": float(np.max(finite) - np.min(finite)),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


_CORR_METHODS = ("pearson", "spearman", "kendall")


def _corr_pair(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Correlation via pandas (no scipy dep). NaN if too few points or constant."""
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    sa, sb = pd.Series(a[m]), pd.Series(b[m])
    if sa.std() == 0 or sb.std() == 0:
        return float("nan")
    try:
        v = sa.corr(sb, method=method)
    except Exception:
        return float("nan")
    return float(v) if v is not None and not pd.isna(v) else float("nan")


def _best_lag_pearson(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Slide `b` against `a` over ±max_lag (clipped to overlap ≥3), return
    (best_corr_in_abs, best_lag). Lag > 0 means b is shifted right.
    """
    best = (float("nan"), 0)
    best_abs = -1.0
    n = min(len(a), len(b))
    if n < 4:
        return best
    max_lag = min(max_lag, n - 3)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[: n - lag], b[lag:n]
        else:
            aa, bb = a[-lag:n], b[: n + lag]
        c = _pearson(aa, bb)
        if np.isfinite(c) and abs(c) > best_abs:
            best_abs = abs(c)
            best = (c, lag)
    return best


# ---------------------------------------------------------------------------
# correlated.csv access (paginated, filtered, streamed)
# ---------------------------------------------------------------------------

def _correlated_path(pipeline_dir: Path) -> Path:
    return pipeline_dir / "correlated.csv"


def _correlated_count(pipeline_dir: Path) -> int:
    key = str(pipeline_dir)
    with STATE.lock:
        c = STATE.cache.setdefault(key, {})
    if "n_correlated" in c:
        return c["n_correlated"]
    # cheap line count
    p = _correlated_path(pipeline_dir)
    n = 0
    with p.open("rb") as fh:
        for _ in fh:
            n += 1
    n = max(0, n - 1)
    c["n_correlated"] = n
    return n


def _filter_chunk(df: pd.DataFrame, *, id_q: str, min_corr: float, max_corr: float,
                  lag_min: int | None, lag_max: int | None,
                  exclude_self: bool, only_self: bool,
                  anchor_id: str | None = None,
                  anchor_t: int | None = None) -> pd.DataFrame:
    if anchor_id is not None and anchor_t is not None:
        # Exact (id, t) anchor — pairs whose (id1,t1) OR (id2,t2) equals it.
        m = ((df["id1"] == anchor_id) & (df["t1"] == anchor_t)) | \
            ((df["id2"] == anchor_id) & (df["t2"] == anchor_t))
        df = df[m]
    elif anchor_id is not None:
        # Airport-only anchor — pairs where this id appears on either side,
        # regardless of the window's t value.
        m = (df["id1"] == anchor_id) | (df["id2"] == anchor_id)
        df = df[m]
    if id_q:
        q = id_q.lower()
        m = df["id1"].str.lower().str.contains(q, regex=False) | \
            df["id2"].str.lower().str.contains(q, regex=False)
        df = df[m]
    if min_corr is not None:
        df = df[df["corr"].abs() >= min_corr]
    if max_corr is not None:
        df = df[df["corr"].abs() <= max_corr]
    if lag_min is not None:
        df = df[df["lag"] >= lag_min]
    if lag_max is not None:
        df = df[df["lag"] <= lag_max]
    if exclude_self:
        df = df[df["id1"] != df["id2"]]
    if only_self:
        df = df[df["id1"] == df["id2"]]
    return df


def _iter_correlated(pipeline_dir: Path, *, filters: dict, sort: str, order: str,
                     offset: int, limit: int) -> tuple[list[dict], int]:
    """Return (rows, total_matching) using streaming filter and a small heap-style
    sort over only the filtered rows. For sorts we still need to scan, but the
    chunked read keeps memory bounded."""
    p = _correlated_path(pipeline_dir)
    matched: list[pd.DataFrame] = []
    total = 0
    chunksize = 200_000
    for chunk in pd.read_csv(p, chunksize=chunksize):
        # type clean
        for c in ("t1", "t2", "lag"):
            if c in chunk:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("Int64")
        chunk["corr"] = pd.to_numeric(chunk["corr"], errors="coerce")
        flt = _filter_chunk(chunk, **filters)
        total += len(flt)
        if not flt.empty:
            matched.append(flt)
    if not matched:
        return [], 0
    full = pd.concat(matched, ignore_index=True)
    asc = order != "desc"
    if sort == "abs_corr":
        full = full.reindex(full["corr"].abs().sort_values(ascending=asc).index)
    elif sort in ("corr", "lag", "t1", "t2", "id1", "id2"):
        full = full.sort_values(by=sort, ascending=asc, kind="mergesort")
    page = full.iloc[offset:offset + limit]
    rows = page.to_dict(orient="records")
    return rows, total


def _unique_windows(pipeline_dir: Path, *, id_q: str, top_n: int) -> list[dict]:
    """Aggregate distinct (id, t) windows across correlated.csv with cheap stats.

    Each row: {id, t, n_partners, max_abs_corr, mean_abs_corr}. Limited to top_n
    by max_abs_corr (descending) within the id filter. Streamed and aggregated.
    """
    p = _correlated_path(pipeline_dir)
    # Accumulate per (id, t)
    n_partners: dict[tuple[str, int], int] = {}
    sum_abs: dict[tuple[str, int], float] = {}
    max_abs: dict[tuple[str, int], float] = {}
    ql = id_q.lower() if id_q else ""
    for chunk in pd.read_csv(p, chunksize=200_000):
        chunk["corr"] = pd.to_numeric(chunk["corr"], errors="coerce")
        chunk["t1"] = pd.to_numeric(chunk["t1"], errors="coerce").astype("Int64")
        chunk["t2"] = pd.to_numeric(chunk["t2"], errors="coerce").astype("Int64")
        for side in ("1", "2"):
            ids = chunk[f"id{side}"].astype(str).to_numpy()
            ts = chunk[f"t{side}"].to_numpy()
            cs = chunk["corr"].to_numpy()
            for i in range(len(ids)):
                t = ts[i]
                if pd.isna(t):
                    continue
                idv = ids[i]
                if ql and ql not in idv.lower():
                    continue
                cv = abs(float(cs[i])) if not pd.isna(cs[i]) else 0.0
                key = (idv, int(t))
                n_partners[key] = n_partners.get(key, 0) + 1
                sum_abs[key] = sum_abs.get(key, 0.0) + cv
                if cv > max_abs.get(key, -1.0):
                    max_abs[key] = cv
    items = []
    for k, n in n_partners.items():
        items.append({
            "id": k[0], "t": k[1],
            "n_partners": n,
            "max_abs_corr": round(max_abs[k], 6),
            "mean_abs_corr": round(sum_abs[k] / n, 6),
        })
    items.sort(key=lambda r: r["max_abs_corr"], reverse=True)
    return items[:top_n]


# ---------------------------------------------------------------------------
# HTML / static assets (inlined; no CDN)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CorrTrack viz</title>
<script src="/static/d3.v7.min.js"></script>
<link rel="stylesheet" href="/static/leaflet.css"/>
<script src="/static/leaflet.js"></script>
<style>
:root { --bg:#0f1115; --panel:#171a21; --panel2:#1d2129; --line:#2a2f3a;
        --fg:#e6e9ef; --muted:#9aa3b2; --accent:#5aa1ff; --warn:#ffb05a;
        --ok:#7cd992; --bad:#ff7c7c; --mono:'SFMono-Regular',Menlo,Consolas,monospace; }
* { box-sizing: border-box; }
html,body { margin:0; padding:0; background:var(--bg); color:var(--fg);
            font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
header { padding:10px 14px; background:var(--panel); border-bottom:1px solid var(--line);
         display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:14px; margin:0; font-weight:600; letter-spacing:.3px; }
header .meta { color:var(--muted); font-size:12px; font-family:var(--mono); }
.layout { display:grid; grid-template-columns: 260px 1fr 320px; height:calc(100vh - 49px); }
.col { overflow:auto; }
.col.left, .col.right { background:var(--panel); border-right:1px solid var(--line); }
.col.right { border-right:none; border-left:1px solid var(--line); }
.section { padding:10px 12px; border-bottom:1px solid var(--line); }
.section h2 { font-size:11px; text-transform:uppercase; letter-spacing:.7px;
              color:var(--muted); margin:0 0 8px 0; font-weight:600; }
select, input[type=text], input[type=number] { background:var(--panel2); color:var(--fg);
              border:1px solid var(--line); border-radius:4px; padding:5px 7px;
              font:inherit; width:100%; }
/* Custom multi-select dropdown */
.ms { position:relative; display:inline-block; }
.ms-trigger { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
              border-radius:4px; padding:5px 24px 5px 8px; font:inherit; font-size:12px;
              cursor:pointer; min-width:130px; max-width:230px; text-align:left;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
              position:relative; }
.ms-trigger::after { content:'▾'; position:absolute; right:8px; top:50%; transform:translateY(-50%);
                     color:var(--muted); font-size:10px; }
.ms.ms-active .ms-trigger { border-color:var(--accent); color:var(--accent); font-weight:600; }
.ms.ms-active .ms-trigger::after { color:var(--accent); }
.ms-trigger:hover { border-color:var(--accent); }
.ms-panel { position:absolute; top:calc(100% + 2px); left:0; z-index:100;
            min-width:200px; max-width:280px; background:var(--panel);
            border:1px solid var(--line); border-radius:5px; padding:6px;
            box-shadow:0 4px 16px rgba(0,0,0,.4); }
.ms-panel[hidden] { display:none; }
.ms-search { width:100%; box-sizing:border-box; margin-bottom:4px;
             padding:3px 6px; font-size:11px; }
.ms-options { max-height:260px; overflow-y:auto; }
.ms-options label { display:flex; align-items:center; gap:6px; padding:3px 4px;
                    font-size:12px; cursor:pointer; border-radius:3px;
                    font-family:var(--mono); }
.ms-options label:hover { background:var(--panel2); }
.ms-options label.hidden { display:none; }
.ms-options input { margin:0; }
.ms-actions { display:flex; gap:4px; padding-top:6px; margin-top:4px;
              border-top:1px solid var(--line); }
input[type=number] { width:100%; }
button { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
         border-radius:4px; padding:5px 10px; cursor:pointer; font:inherit; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); border-color:var(--accent); color:#0a0e14; font-weight:600; }
button.ghost { background:transparent; }
button.small { padding:2px 6px; font-size:11px; }
.row { display:flex; gap:6px; align-items:center; }
.row > * { flex: 1 1 auto; }
.kv { display:grid; grid-template-columns: auto 1fr; gap:4px 10px;
      font-family:var(--mono); font-size:12px; }
.kv .k { color:var(--muted); }
table.t { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:11.5px; }
table.t th, table.t td { padding:3px 8px; border-bottom:1px solid var(--line);
                         text-align:left; white-space:nowrap; }
table.t th { background:var(--panel2); position:sticky; top:0; cursor:pointer;
             font-weight:600; user-select:none; color:var(--muted); font-size:11px;
             text-transform:uppercase; letter-spacing:.5px; padding:6px 8px; }
table.t th.active { color:var(--accent); }
table.t th[data-sort] { white-space:nowrap; }
table.t td.num, table.t th.num { text-align:right; font-variant-numeric:tabular-nums; }
table.t td.actions { text-align:right; padding-right:6px; }
table.t td.add-cell { cursor: copy; position: relative; }
table.t td.add-cell[data-side="a"]:hover { box-shadow: inset 0 0 0 1px rgba(124,217,146,.45); background: rgba(124,217,146,.06); }
table.t td.add-cell[data-side="b"]:hover { box-shadow: inset 0 0 0 1px rgba(255,176,90,.45); background: rgba(255,176,90,.06); }
/* Persistent marker: this window is in the basket. Dot color = basket curve color. */
.basket-dot { display:inline-block; width:9px; height:9px; border-radius:50%;
              margin-left:6px; vertical-align:-1px;
              box-shadow: 0 0 0 1.5px var(--panel),
                          0 0 0 2.5px rgba(255,255,255,.2);
              animation: dot-pop .35s ease-out; }
@keyframes dot-pop { from { transform: scale(0); } to { transform: scale(1); } }
table.t td.add-cell.in-basket { background: rgba(90,161,255,.08); }
table.t tr.pair-in-basket td:nth-child(2) { box-shadow: inset 3px 0 0 var(--ok); }
table.t tbody tr { cursor:pointer; }
table.t tbody tr:nth-child(odd) td { background:rgba(255,255,255,.012); }
table.t tbody tr:hover td { background:rgba(90,161,255,.08); }
table.t tbody tr.selected td { background:rgba(124,217,146,.10); }
table.t tbody tr.active-row td { background:rgba(90,161,255,.10); }
table.t tbody tr.sel-row td { box-shadow: inset 3px 0 0 var(--accent); }
table.t td.sel-cell { padding:2px 6px; text-align:center; cursor:default; }
table.t td.sel-cell input { cursor:pointer; }
.lag-pos { color:#88c1ff; }
.lag-neg { color:#ffb05a; }
.lag-zero { color:var(--muted); }
.pager { display:flex; gap:6px; align-items:center; padding:6px 0; font-size:12px; color:var(--muted); }
.pager .grow { flex:1; }
.tabs { display:flex; gap:0; border-bottom:1px solid var(--line); }
.tabs .tab { padding:8px 12px; cursor:pointer; border-bottom:2px solid transparent;
             color:var(--muted); font-size:12px; }
.tabs .tab:hover { color:var(--fg); }
.tabs .tab.active { color:var(--fg); border-bottom-color:var(--accent); }
.tabs .tab.home { font-weight:600; }
.tabs .tab.needs-active { display:none; }
.tabs.has-active .tab.needs-active { display:block; }
header h1:hover { color:var(--accent); }
.open-btn { display:inline-flex; align-items:center; gap:4px; padding:3px 8px; }
.open-btn svg { display:block; }
.tabbody { padding:10px 12px; }
.chip { display:inline-block; padding:1px 6px; border-radius:9px; font-size:10.5px;
        background:var(--panel2); color:var(--muted); border:1px solid var(--line); }
.chip.ok { color:var(--ok); border-color:rgba(124,217,146,.4); }
.chip.bad { color:var(--bad); border-color:rgba(255,124,124,.4); }
.chip.warn { color:var(--warn); border-color:rgba(255,176,90,.4); }
.chip.info { color:#5aa1ff; border-color:rgba(90,161,255,.4); }
/* Run status badges + progress bar */
.status-cell { display:flex; align-items:center; gap:4px; font-family:var(--mono); font-size:10.5px; }
.status-dot { width:8px; height:8px; border-radius:50%; flex:0 0 8px; }
.status-dot.done    { background:#7cd992; }
.status-dot.running { background:#5aa1ff; animation: pulse 1s infinite ease-in-out; }
.status-dot.starting{ background:#ffe066; animation: pulse 1s infinite ease-in-out; }
.status-dot.failed  { background:#ff7c7c; }
.status-dot.queued  { background:#888; }
.status-dot.unknown { background:#444; }
@keyframes pulse { 0%,100% { transform: scale(1); opacity:1; } 50% { transform: scale(1.4); opacity:0.5; } }
.progress-bar { display:inline-block; background:var(--panel2); border-radius:3px; height:6px;
                width:70px; position:relative; overflow:hidden; }
.progress-bar > span { position:absolute; left:0; top:0; bottom:0; background:#5aa1ff;
                       transition: width .3s ease-out; }
tr.row-running td { background: rgba(90,161,255,.06); }
/* Top banner for in-progress runs */
.runs-banner { margin:6px 12px; padding:8px 12px; background:rgba(90,161,255,.08);
               border:1px solid rgba(90,161,255,.3); border-radius:5px;
               display:flex; align-items:center; gap:10px; font-size:12px; }
.runs-banner .status-dot.running { width:10px; height:10px; }
/* Leaflet distance labels */
.dist-label { background:none; border:none; }
.dist-label > span { background:rgba(20,23,28,.85); color:#5aa1ff; font-family:var(--mono);
                     font-size:11px; padding:1px 6px; border-radius:3px; white-space:nowrap;
                     border:1px solid rgba(90,161,255,.4); }
.leaflet-container { font-family: -apple-system,BlinkMacSystemFont,sans-serif; }
.leaflet-popup-content-wrapper, .leaflet-popup-tip { background: var(--panel2); color: var(--fg); }
.leaflet-popup-content { margin: 8px 12px; font-size: 12px; }
button.chip { background:var(--panel2); cursor:pointer; padding:1px 8px;
              font-family:var(--mono); font-size:11px; }
button.chip:hover { filter:brightness(1.2); }
tr.cov-expand td { border-top:1px dashed var(--line) !important; }
/* Self-contained run name with inline param chips */
.rn-name { font-family:var(--mono); font-size:12px; font-weight:500; color:var(--fg);
           white-space:nowrap; }
.rn-chips { display:flex; gap:4px; flex-wrap:wrap; margin-top:2px; }
.rn-chips .rn-tag { font-family:var(--mono); font-size:10px; color:var(--muted);
                     padding:0 5px; border-radius:3px;
                     background:rgba(255,255,255,.04); border:1px solid var(--line); }
.rn-chips .rn-tag.rn-sketch { color:#5aa1ff; border-color:rgba(90,161,255,.3); }
.rn-chips .chip { padding:0 5px; font-size:10px; }
/* Mute *failed* rows; running rows must stay clearly visible */
table.t tbody tr[data-status="failed"] td,
table.t tbody tr[data-status="queued"] td,
table.t tbody tr[data-status="unknown"] td { color:var(--muted); }
table.t tbody tr[data-status="running"] td,
table.t tbody tr[data-status="starting"] td { background: rgba(90,161,255,.10); }
table.t tbody tr[data-status="running"] td:nth-child(3),
table.t tbody tr[data-status="starting"] td:nth-child(3) { font-weight:600; }
table.t tbody tr[data-done="0"] td.actions, table.t tbody tr[data-done="0"] td.sel-cell { color:inherit; }
.cov-pipe:hover { transform:translateY(-1px); }
.basket-item { background:var(--panel2); border:1px solid var(--line); border-radius:5px;
               padding:8px; margin-bottom:8px; }
.basket-item .head { display:flex; justify-content:space-between; gap:6px; align-items:center; }
.basket-item .lab { font-family:var(--mono); font-size:12px; font-weight:600; }
.basket-item .swatch { width:10px; height:10px; border-radius:50%; display:inline-block;
                       margin-right:6px; vertical-align:middle; }
.basket-item .stat { font-family:var(--mono); font-size:11px; color:var(--muted);
                     margin-top:4px; }
.basket-item .basket-date { font-family:var(--mono); font-size:11px;
                            color:var(--fg); margin-top:3px;
                            letter-spacing:.2px; }
.basket-item .partners { margin-top:8px; max-height:200px; overflow:auto;
                         padding:6px; border-top:1px dashed var(--line);
                         font-size:11px; }
.basket-item .partners .pfilter { display:flex; gap:4px; margin-bottom:6px; }
.basket-item .partners .pfilter input { font-size:11px; padding:2px 4px; }
.basket-item .partners .chip { display:inline-block; margin:2px; cursor:pointer;
                                font-family:var(--mono); }
.basket-item .partners .chip:hover { border-color:var(--accent); color:var(--fg); }
.basket-item .partners .chip .corr { color:var(--ok); font-weight:600; }
.basket-item .partners .chip .corr.neg { color:var(--warn); }
.basket-item .partners .chip.added { opacity:.4; cursor:default; }
.muted { color:var(--muted); }
.plot-wrap { padding:10px 12px; }
.plot-wrap img { max-width:100%; display:block; border:1px solid var(--line);
                 background:#fff; border-radius:4px; }
.plot-wrap img:not([src]), .plot-wrap img[src=""] { display:none; }
.plot-empty { border:1px dashed var(--line); border-radius:4px; padding:40px 20px;
              text-align:center; color:var(--muted); font-style:italic;
              background:var(--panel); }
.plotctrl { display:flex; gap:8px; align-items:center; padding:8px 12px;
            border-bottom:1px solid var(--line); flex-wrap:wrap; }
.plotctrl label { font-size:12px; color:var(--muted); display:flex; align-items:center; gap:4px; }
.empty { color:var(--muted); font-style:italic; padding:20px; text-align:center; }
.chart-grid { display:grid; gap:14px; grid-template-columns: 1fr; }
.chart-card { background:var(--panel); border:1px solid var(--line); border-radius:6px;
              padding:10px 12px; }
.chart-card h4 { margin:0 0 8px 0; font-size:11px; text-transform:uppercase;
                 letter-spacing:.6px; color:var(--muted); font-weight:600; }
.chart-card img { max-width:100%; display:block; border-radius:4px; background:#fff; }
.chart-card .svg-host { width:100%; overflow-x:auto; }
.chart-card .svg-host svg { display:block; max-width:100%; height:auto; }
.chart-card .svg-host text { fill: var(--fg); }
.chart-card .svg-host text.axis { fill: var(--muted); }
.chart-card .svg-host text.label { fill: var(--muted); font-family: var(--mono); }
.chart-card .svg-host .grid { stroke: var(--line); stroke-dasharray: 2 3; }
.chart-card .svg-host .axis-line { stroke: var(--line); }
/* D3 axes inside #plot */
#plot .d3-axis path.domain, #plot .d3-axis line { stroke: var(--line); }
#plot .d3-axis text { fill: var(--muted); font-size: 10.5px; font-family: var(--mono); }
#plot .d3-axis-label { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
#plot .d3-brush .selection { fill: var(--accent); fill-opacity: 0.15; stroke: var(--accent); }
#plot { position: relative; }
.d3-tooltip { background: rgba(20, 23, 28, 0.96); border: 1px solid var(--line);
              border-radius: 5px; padding: 6px 10px; font-size: 11px;
              color: var(--fg); box-shadow: 0 4px 14px rgba(0,0,0,0.4);
              font-family: var(--mono); min-width: 140px; z-index: 10; }
.d3-tooltip .thead { font-weight: 600; color: var(--accent); margin-bottom: 4px;
                     padding-bottom: 3px; border-bottom: 1px solid var(--line); }
.d3-tooltip .trow { display: flex; align-items: center; gap: 6px; line-height: 1.5; }
.d3-tooltip .trow .dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 9px; }
.d3-tooltip .trow .nm { flex: 1 1 auto; color: var(--fg); }
.d3-tooltip .trow .val { font-weight: 600; }
.chart-card .empty { padding:30px; text-align:center; color:var(--muted); font-style:italic; }
.res-block { border:1px solid var(--line); border-radius:5px; padding:8px 10px 4px; background:rgba(255,255,255,0.015); }
.res-title { font-size:11.5px; font-weight:600; color:var(--fg); margin-bottom:4px; letter-spacing:.3px; }
.spin { color:var(--muted); padding:8px; font-style:italic; }
.corr-section { margin:14px 12px 18px; padding:12px 14px;
                 background:var(--panel); border:1px solid var(--line); border-radius:6px; }
.corr-section .corr-section-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
                                    margin-bottom:10px; padding-bottom:8px;
                                    border-bottom:1px solid var(--line); }
.corr-section .corr-section-head h2 { margin:0; font-size:12px; text-transform:uppercase;
                                       letter-spacing:.7px; color:var(--fg); font-weight:600; }
.corr-tabs { display:flex; gap:0; border:1px solid var(--line); border-radius:4px; overflow:hidden; }
.corr-tabs .ct { padding:4px 11px; cursor:pointer; font-size:11px; background:var(--panel2);
                 color:var(--muted); border-right:1px solid var(--line); }
.corr-tabs .ct:last-child { border-right:none; }
.corr-tabs .ct:hover { color:var(--fg); }
.corr-tabs .ct.active { background:var(--accent); color:#0a0e14; font-weight:600; }
.corr-table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:5px;
                    background:var(--panel2); }
.mini-corr { font-family:var(--mono); font-size:11.5px; border-collapse:collapse; width:100%; }
.mini-corr td, .mini-corr th { padding:6px 10px; border:1px solid var(--line);
                                text-align:right; font-variant-numeric:tabular-nums;
                                white-space:nowrap; }
.mini-corr thead th { background:var(--panel2); color:var(--muted); font-weight:600;
                       font-size:10.5px; text-transform:uppercase; letter-spacing:.5px;
                       text-align:center; position:sticky; top:0; z-index:1; }
.mini-corr tbody th { background:var(--panel); color:var(--fg); font-weight:600;
                       text-align:left; position:sticky; left:0; z-index:1; }
.mini-corr td.cell { position:relative; }
.mini-corr td.cell .lag { display:block; font-size:9.5px; color:var(--muted);
                           margin-top:2px; font-weight:400; }
.mini-corr td.diag { color:var(--muted); background:var(--panel2);
                     font-size:18px; text-align:center; }
/* Compact "all" matrix: P/S/K (and optional best-lag) stacked per cell */
.mini-corr.compact td.cell { padding:5px 8px; line-height:1.38; }
.mini-corr.compact .r { display:flex; gap:8px; justify-content:flex-end;
                         align-items:center; font-size:11.5px; }
.mini-corr.compact .r + .r { margin-top:1px; }
.mini-corr.compact .r .m { min-width:14px; text-align:left; color:var(--muted);
                            font-size:9.5px; font-weight:700; text-transform:uppercase;
                            letter-spacing:.3px; }
.mini-corr.compact .r .v { font-weight:500; }
.mini-corr.compact .r.bl { border-top:1px dashed rgba(255,255,255,.15);
                            margin-top:3px; padding-top:3px; }
.mini-corr.compact .r.bl .lag { font-size:9.5px; color:var(--muted); margin-left:4px; }
.win-swatch { display:inline-block; width:9px; height:9px; border-radius:50%;
              margin-right:6px; vertical-align:-1px; }
/* Runtime ranking: HTML table with mode/backend/index/key/sketch chips,
   plus an in-row horizontal bar (replaces the old SVG ranking).             */
.rt-rank { width:100%; border-collapse:collapse; font-size:11.5px;
            font-family:var(--mono); }
.rt-rank th { text-align:left; padding:4px 6px; font-weight:600; font-size:10px;
              color:var(--muted); text-transform:uppercase; letter-spacing:.4px;
              border-bottom:1px solid var(--line); white-space:nowrap; }
.rt-rank th.sortable { cursor:pointer; user-select:none; }
.rt-rank th.sortable:hover { color:var(--accent); }
.rt-rank th.sorted { color:var(--accent); }
.rt-rank th .sort-arrow { display:inline-block; margin-left:3px;
                            opacity:.4; font-weight:700; }
.rt-rank th.sorted .sort-arrow { opacity:1; }
.rt-phase-legend .lg { cursor:pointer; padding:1px 4px; border-radius:3px; }
.rt-phase-legend .lg:hover { background:rgba(255,255,255,.06); }
.rt-phase-legend .lg.active { color:var(--accent); font-weight:600;
                                background:rgba(90,161,255,.1); }
.rt-rank td { padding:3px 6px; vertical-align:middle; white-space:nowrap;
              border-bottom:1px solid rgba(255,255,255,.04); }
.rt-rank tr.baseline td { background:rgba(255,176,90,.07); }
.rt-rank tr.fastest td { background:rgba(124,217,146,.06); }
.rt-rank tr.slower-than-baseline td.rt-col-name,
.rt-rank tr.slower-than-baseline td.rt-col-time { color:#ff5c5c; }
.rt-rank tr:hover td { background:rgba(255,255,255,.03); }
.rt-rank td.rt-col-rank { color:var(--muted); text-align:right;
                           width:32px; font-size:10px; }
.rt-rank td.rt-col-name { max-width:280px; overflow:hidden;
                          text-overflow:ellipsis; }
.rt-rank td.rt-col-bar  { width:55%; padding-left:0; padding-right:0; }
.rt-rank td.rt-col-time { text-align:right; width:80px; font-weight:600; }
.rt-bar-host { position:relative; height:18px;
                background:rgba(255,255,255,.04); border-radius:3px;
                overflow:hidden; }
.rt-bar-fill { position:absolute; left:0; top:0; bottom:0;
                border-radius:3px; opacity:.85; }
.rt-bar-baseline { position:absolute; top:0; bottom:0;
                    width:2px; background:#ff5c5c; opacity:.7;
                    pointer-events:none; }
.rt-bar-seg { position:absolute; top:0; bottom:0; opacity:.92;
               border-right:1px solid rgba(0,0,0,.25); }
.rt-bar-seg:last-child { border-right:none; }
.rt-phase-legend { display:flex; flex-wrap:wrap; gap:10px;
                    padding:6px; font-size:10px;
                    font-family:var(--mono); color:var(--muted); }
.rt-phase-legend .lg { display:inline-flex; align-items:center; gap:4px; }
.rt-phase-legend .sw { width:10px; height:10px; border-radius:2px;
                         display:inline-block; }
.rt-rank-caption { padding:4px 6px; font-size:11px; color:var(--muted);
                    display:flex; align-items:center; gap:10px; }
.rt-rank-caption .rt-baseline-pill { color:#ff5c5c; font-weight:600;
                                       font-family:var(--mono); }
.rt-rank .rt-tag { display:inline-block; padding:0px 5px; border-radius:8px;
                    font-size:10px; line-height:14px;
                    background:rgba(255,255,255,.05);
                    border:1px solid rgba(255,255,255,.12); }
.rt-rank .rt-tag.muted { color:var(--muted); opacity:.5; }
.rt-rank .rt-tag.mode-bf        { color:var(--warn); border-color:rgba(255,176,90,.4); }
.rt-rank .rt-tag.mode-corrtrack { color:var(--ok);   border-color:rgba(124,217,146,.4); }
.rt-rank .rt-tag.mode-filcorr   { color:#5aa1ff;     border-color:rgba(90,161,255,.4); }
.rt-rank .rt-tag.sketch         { color:#5aa1ff;     border-color:rgba(90,161,255,.3); }
/* `sharded_<base>` runs: render base + ⚡N marker so visual scan stays clean
   but the parallelism level is obvious. Same look in compare table + ranks. */
.shard-mark { display:inline-block; margin-left:4px; padding:0 4px;
               border-radius:3px; font-size:9px; font-weight:700;
               color:#0a0e14; background:#ffd75a; vertical-align:1px;
               font-family:var(--mono); line-height:13px; }
.shard-mark::before { content:'⚡'; margin-right:1px; }
.rt-rank .rt-tag.sharded { color:#ffd75a; border-color:rgba(255,215,90,.45);
                            background:rgba(255,215,90,.08); }
/* Remote-mode banner (only visible when --pipeline is user@host:/…) */
#remote-banner { display:flex; align-items:center; gap:10px;
                 padding:6px 14px; font-size:12px;
                 background:linear-gradient(90deg, #2a3a52 0%, #1c2a40 100%);
                 color:#cfe3ff; border-bottom:1px solid #3a5074;
                 font-family:var(--mono); }
#remote-banner .rb-icon { font-size:14px; }
#remote-banner .rb-label { font-weight:700; letter-spacing:.5px; color:#9fc1ff;
                            background:rgba(159,193,255,.12); padding:2px 7px;
                            border-radius:3px; font-size:10px; }
#remote-banner .rb-target { color:#fff; font-weight:600;
                             overflow:hidden; text-overflow:ellipsis;
                             white-space:nowrap; max-width:60vw; }
#remote-banner .rb-sep { color:#5a7aa8; }
#remote-banner .rb-status { color:#b8d3ff; }
#remote-banner .rb-status.err { color:#ff9b9b; }
#remote-banner .rb-spacer { flex:1; }
#remote-banner .rb-ctl { display:flex; align-items:center; gap:5px; color:#9fc1ff; }
#remote-banner select { background:#1c2a40; color:#fff;
                         border:1px solid #3a5074; border-radius:3px;
                         padding:2px 5px; font:inherit; font-size:11px; }
#remote-banner button.small { background:#3a5074; border-color:#5a7aa8; color:#fff; }
#remote-banner button.small:hover { background:#4a6088; border-color:#9fc1ff; }
#remote-banner button.small:disabled { opacity:.5; cursor:wait; }
#remote-banner .rb-dot { display:inline-block; width:8px; height:8px;
                          border-radius:50%; background:#5aa1ff;
                          animation: pulse 1.4s infinite ease-in-out;
                          margin-right:4px; vertical-align:-1px; }
#remote-banner .rb-dot.err { background:#ff6b6b; animation:none; }
#remote-banner .rb-dot.idle { background:#5a7aa8; animation:none; }
</style>
</head>
<body>
<div id="remote-banner" style="display:none">
  <span class="rb-icon">🌐</span>
  <span class="rb-label">REMOTE</span>
  <span class="rb-target" id="rb-target" title=""></span>
  <span class="rb-sep">·</span>
  <span class="rb-status" id="rb-status">connecting…</span>
  <span class="rb-spacer"></span>
  <label class="rb-ctl">refresh every
    <select id="rb-interval">
      <option value="2">2s</option>
      <option value="5" selected>5s</option>
      <option value="10">10s</option>
      <option value="30">30s</option>
      <option value="60">60s</option>
      <option value="0">off</option>
    </select>
  </label>
  <button class="small" id="rb-sync-now" title="force a rsync cycle right now">⇅ sync now</button>
</div>
<header>
  <h1 id="home-link" title="back to Home (overview + run stats)"
      style="cursor:pointer; user-select:none">CorrTrack — interactive viz</h1>
  <span class="meta" id="run-meta"></span>
  <div style="flex:1"></div>
  <div id="auto-refresh-bar" style="display:flex; align-items:center; gap:8px">
    <span class="chip" id="watch-chip" style="font-size:11px; display:none"></span>
    <button class="small" id="refresh-now-btn" title="refresh status immediately">↻ refresh now</button>
    <label class="muted" style="font-size:11px; display:flex; align-items:center; gap:4px; user-select:none">
      <input type="checkbox" id="auto-refresh-toggle"/>
      Auto-refresh
      <select id="auto-refresh-int" style="padding:2px 4px; font-size:11px; max-width:80px">
        <option value="3">3s</option>
        <option value="5" selected>5s</option>
        <option value="10">10s</option>
        <option value="30">30s</option>
      </select>
    </label>
    <span class="muted" id="auto-refresh-status" style="font-size:11px; min-width:90px"></span>
  </div>
</header>
<div class="layout">

  <!-- LEFT: pipeline summary -->
  <div class="col left">
    <div class="section">
      <h2>Run summary</h2>
      <div id="summary" class="kv"><span class="muted">Select a pipeline.</span></div>
    </div>
    <div class="section">
      <h2>Source</h2>
      <div id="source-info" class="kv"><span class="muted">—</span></div>
    </div>
    <!-- Phase visibility filter — single source of truth used by:
         · ② Phase breakdown (chart)
         · ⑧ Sharded gain mini-bars
         · _phaseBreakdownTipHtml / _shardedPhaseTipHtml tooltips
         Click a chip to hide/show the corresponding phase. State is
         persisted in ST.phasesOff and survives chart re-renders. -->
    <div class="section" id="phase-filter-section">
      <h2 style="display:flex; align-items:center; gap:6px">
        Phases
        <span class="muted" style="font-size:10px; font-weight:400; letter-spacing:0">(click = hide/show)</span>
      </h2>
      <div id="phase-filters" style="display:flex; flex-direction:column; gap:4px; font-size:11.5px">
        <span class="muted" style="font-size:10.5px">—</span>
      </div>
      <div style="margin-top:6px; display:flex; gap:6px">
        <button class="small ghost" id="phase-filters-reset-btn" style="font-size:10.5px">Show all</button>
        <button class="small ghost" id="phase-filters-hide-btn" style="font-size:10.5px">Hide all</button>
      </div>
    </div>
    <div class="section">
      <h2>Selection</h2>
      <div class="kv">
        <span class="k">windows</span><span id="sel-count">0</span>
        <span class="k">window_size</span><span id="sel-ws">—</span>
      </div>
      <div style="margin-top:8px" class="row">
        <button class="small ghost" id="clear-sel">Clear</button>
        <button class="small primary" id="replot">Re-plot</button>
      </div>
    </div>
  </div>

  <!-- CENTER: browser + plot -->
  <div class="col center" style="display:flex; flex-direction:column;">
    <div class="tabs">
      <div class="tab home active" data-tab="compare" title="overview · run stats · engine comparison">🏠 Home</div>
      <div class="tab needs-active" data-tab="pairs">Correlated pairs</div>
      <div class="tab needs-active" data-tab="windows" style="display:none">Unique windows</div>
      <div class="tab needs-active" data-tab="free" style="display:none">Add a free window</div>
    </div>

    <div class="tabbody" id="tab-compare">
      <div id="runs-running-banner" class="runs-banner" style="display:none"></div>
      <div class="row" style="margin-bottom:8px; align-items:flex-end">
        <div style="flex:0 0 auto"><b style="font-size:13px">All runs in <code style="font-family:var(--mono); font-size:12px; color:var(--accent)" id="cmp-root"></code></b></div>
        <div class="muted" id="cmp-summary" style="font-size:11px; margin-left:12px"></div>
        <div style="flex:1"></div>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="cmp-bf" checked/> bf</label>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="cmp-ct" checked/> corrtrack</label>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="cmp-fc" checked/> filcorr</label>
        <span class="muted" style="font-size:11px; margin-left:12px" title="reference run used to compute ×baseline. Auto = SLOWEST bf run (most conservative), else slowest run overall. Override below to pin any specific bf run as the reference.">Baseline:</span>
        <select id="cmp-baseline" style="padding:2px 6px; font-size:11px; max-width:340px"></select>
        <button class="small" id="cmp-refresh">Refresh</button>
      </div>
      <div class="row" style="margin-bottom:6px; align-items:center; gap:8px; flex-wrap:wrap">
        <span class="muted" style="font-size:11px">Backend:</span>
        <div id="ms-backend" class="ms"><button class="ms-trigger" type="button">all backends</button><div class="ms-panel" hidden><input type="text" class="ms-search" placeholder="filter…"/><div class="ms-options"></div><div class="ms-actions"><button class="small ghost ms-all" type="button">all</button><button class="small ghost ms-none" type="button">none</button></div></div></div>
        <span class="muted" style="font-size:11px">Index:</span>
        <div id="ms-index" class="ms"><button class="ms-trigger" type="button">all indexes</button><div class="ms-panel" hidden><input type="text" class="ms-search" placeholder="filter…"/><div class="ms-options"></div><div class="ms-actions"><button class="small ghost ms-all" type="button">all</button><button class="small ghost ms-none" type="button">none</button></div></div></div>
        <span class="muted" style="font-size:11px">Index key:</span>
        <div id="ms-keymode" class="ms"><button class="ms-trigger" type="button">all key modes</button><div class="ms-panel" hidden><input type="text" class="ms-search" placeholder="filter…"/><div class="ms-options"></div><div class="ms-actions"><button class="small ghost ms-all" type="button">all</button><button class="small ghost ms-none" type="button">none</button></div></div></div>
        <span class="muted" style="font-size:11px">Sketch:</span>
        <div id="ms-sketch" class="ms"><button class="ms-trigger" type="button">all sketch methods</button><div class="ms-panel" hidden><input type="text" class="ms-search" placeholder="filter…"/><div class="ms-options"></div><div class="ms-actions"><button class="small ghost ms-all" type="button">all</button><button class="small ghost ms-none" type="button">none</button></div></div></div>
        <button class="small ghost" id="cmp-filters-clear" title="reset all dimension filters">clear</button>
      </div>
      <div class="row" style="margin-bottom:8px; align-items:center; gap:8px; flex-wrap:wrap">
        <span class="muted" style="font-size:11px" title="filters require running 'Compute precision/recall' first">Quality:</span>
        <span class="muted" style="font-size:11px">precision ≥</span>
        <input id="cmp-min-prec" type="number" step="0.01" min="0" max="1" placeholder="0.00" style="width:70px"/>
        <span class="muted" style="font-size:11px">recall ≥</span>
        <input id="cmp-min-recall" type="number" step="0.01" min="0" max="1" placeholder="0.00" style="width:70px"/>
        <span class="muted" style="font-size:11px">f1 ≥</span>
        <input id="cmp-min-f1" type="number" step="0.01" min="0" max="1" placeholder="0.00" style="width:70px"/>
        <span class="muted" style="font-size:11px" title="speedup column = runtime_slowest / runtime">speedup ≥</span>
        <input id="cmp-min-speedup" type="number" step="0.1" min="0" placeholder="(any)" style="width:80px"/>
        <span class="muted" style="font-size:11px">×</span>
        <span class="muted" style="font-size:11px; margin-left:8px"><input type="checkbox" id="cmp-hide-noq"/> hide runs without quality info</span>
      </div>
      <div class="row" style="margin-bottom:8px; align-items:center; gap:6px">
        <span class="muted" style="font-size:11px">Selection:</span>
        <button class="small" id="cmp-sel-all">all</button>
        <button class="small" id="cmp-sel-none">none</button>
        <button class="small" id="cmp-sel-invert">invert</button>
        <span class="chip" id="cmp-sel-count" style="font-size:11px">0 selected</span>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="cmp-only-sel"/> show selected only</label>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="cmp-charts-sel" checked/> charts use selection</label>
        <div style="flex:1"></div>
        <button class="small" id="cmp-quality">Compute precision / recall vs baseline</button>
      </div>
      <div style="max-height:62vh; overflow:auto; border:1px solid var(--line); border-radius:4px;">
        <table class="t" id="cmp-table">
          <thead><tr>
            <th style="width:24px"><input type="checkbox" id="cmp-sel-header" title="select all visible"/></th>
            <th style="width:60px">open</th>
            <th data-sort="status" style="width:170px">status</th>
            <th data-sort="mode">mode</th>
            <th data-sort="label" style="min-width:220px" title="run directory name (under the pipeline output folder)">name</th>
            <th data-sort="backend">backend</th>
            <th data-sort="index_backend">index</th>
            <th data-sort="index_key_mode">key</th>
            <th data-sort="sketch_method">sketch</th>
            <th data-sort="runtime" class="num active">runtime ↑</th>
            <th data-sort="speedup_vs_baseline" class="num" title="speedup vs the baseline run (defaults to bf_python — same one used for precision/recall)">×baseline</th>
            <th data-sort="precision" class="num">precision</th>
            <th data-sort="recall" class="num">recall</th>
            <th data-sort="f1" class="num">f1</th>
            <th data-sort="n_candidates" class="num">candidates</th>
            <th data-sort="n_tested" class="num">tested</th>
            <th data-sort="candidate_ratio" class="num">cand/tested</th>
            <th data-sort="n_correlated" class="num">correlated</th>
            <th data-sort="win_per_s" class="num">win/s</th>
            <th data-sort="cand_per_s" class="num">cand/s</th>
            <th data-sort="tput_win_median" class="num" title="median real-time throughput (windows/s, instantaneous) — sustained processing capacity">win/s med</th>
            <th data-sort="tput_win_min" class="num" title="minimum observed real-time throughput (windows/s) — worst case for real-time tracking">win/s min</th>
            <th data-sort="mem_peak_mb" class="num" title="peak resident memory (MB)">mem MB</th>
            <th data-sort="energy_cpu_seconds" class="num" title="energy proxy = CPU core·seconds consumed (≈ energy at constant TDP)">energy</th>
            <th data-sort="t_candidates" class="num">cand_t</th>
            <th data-sort="t_validate" class="num">val_t</th>
            <th data-sort="spd_t_candidates" class="num" title="speedup of the candidate phase (select) vs baseline — see where sharding wins">cand×</th>
            <th data-sort="spd_t_validate" class="num" title="speedup of the validate phase vs baseline">val×</th>
            <th data-sort="opt_time" class="num" title="Total optim time (hyperparameter sweep). The serial detail (sum of the configs) vs parallel (makespan over the workers) is shown in the phase bars — legend 'opt sweep'.">opt_t</th>
          </tr></thead>
          <tbody><tr><td colspan="29" class="empty">Loading…</td></tr></tbody>
        </table>
      </div>
      <div class="muted" style="font-size:11px; margin-top:6px">
        Row click: <b>preview</b> stats in the left panel · <b>open</b> button: switch to that pipeline (loads Correlated pairs).
        <b>Shift+click</b> a column header to add it as a secondary/tertiary sort (e.g. mode → backend → index).
        Columns: <code>cand_t</code>/<code>val_t</code> = time spent in candidate+validate phases (sec);
        <code>cand/tested</code> = candidate ratio (lower = better pruning);
        <code>opt_t</code> = optim (sweep) time. The <code>opt sweep</code> bar (chart ② / sharded panel)
        shows the <b>serial sum</b> for a base run and the <b>makespan</b> (time of the busiest thread)
        for a parallel/sharded run — hence smaller when the grid parallelizes.
      </div>

      <!-- Charts section -->
      <div class="charts" id="charts" style="margin-top:18px; display:none">
        <div class="row" style="margin-bottom:6px; align-items:center; gap:8px">
          <b style="font-size:13px">Charts</b>
          <span class="chip" id="charts-filter-chip" style="font-size:11px">all runs</span>
          <span class="muted" style="font-size:11px">— native SVG, rendered in the browser (instant)</span>
          <div style="flex:1"></div>
          <!-- Master top-N: shared by ① runtime, ⑤ speedup, ② phases, ⑧ sharded gain.
               Avoids 4 separate selectors and keeps the cap consistent across charts. -->
          <label class="muted" style="font-size:11px; display:inline-flex; align-items:center; gap:5px"
                 title="Cap the number of rows shown in ① Runtime ranking, ⑤ Speedup, ② Phase breakdown and ⑧ Sharded gain. Applies to all four at once.">
            show
            <select id="cmp-topn" style="padding:2px 4px; font-size:11px">
              <option value="10">top 10</option>
              <option value="20" selected>top 20</option>
              <option value="30">top 30</option>
              <option value="50">top 50</option>
              <option value="100">top 100</option>
              <option value="0">all</option>
            </select>
          </label>
          <button class="small" id="charts-refresh">Re-render</button>
        </div>
        <div class="chart-grid">
          <div class="chart-card">
            <h4 style="display:flex; align-items:center; gap:10px">
              <span>① Runtime ranking</span>
              <span class="muted" style="font-weight:400; font-size:11px; margin-left:auto; text-transform:none; letter-spacing:0">
                <span class="muted" id="ch-runtime-topn-note" style="font-size:10px"></span>
              </span>
            </h4>
            <div id="ch-runtime"></div>
          </div>
          <div class="chart-card">
            <h4 style="display:flex; align-items:center; gap:10px">
              <span>⑤ Speedup vs bf_python (log scale)</span>
              <span class="muted" style="font-weight:400; font-size:11px; margin-left:auto; text-transform:none; letter-spacing:0">
                <span class="muted" id="ch-speedup-topn-note" style="font-size:10px"></span>
              </span>
            </h4>
            <div id="ch-speedup"></div>
          </div>
          <div class="chart-card"><h4>⑥ Heatmap backend × index (best speedup per cell)</h4><div id="ch-heatmap" class="svg-host"></div></div>
          <div class="chart-card"><h4>⑦ Heatmap sketch × index (best speedup per cell)</h4><div id="ch-sketch-heatmap" class="svg-host"></div></div>
          <div class="chart-card">
            <h4 style="display:flex; align-items:center; gap:10px">
              <span>⑧ Sharded gain · base vs sharded (×bf as reference)</span>
              <span class="muted" style="font-weight:400; font-size:11px; margin-left:auto; text-transform:none; letter-spacing:0">
                <span class="muted" id="ch-shard-note" style="font-size:10px"></span>
                <span class="muted" id="ch-shard-topn-note" style="font-size:10px; margin-left:6px"></span>
              </span>
            </h4>
            <div id="ch-shard-gain"></div>
          </div>
          <div class="chart-card">
            <h4 style="display:flex; align-items:center; gap:10px">
              <span>② Phase breakdown (stacked)</span>
              <span class="muted" style="font-weight:400; font-size:11px; margin-left:auto; text-transform:none; letter-spacing:0">
                <span class="muted" id="ch-phases-topn-note" style="font-size:10px"></span>
              </span>
            </h4>
            <div id="ch-phases"></div>
          </div>
          <div class="chart-card">
            <h4>③ Custom trade-off scatter (Pareto frontier highlighted)</h4>
            <div class="row" style="gap:8px; margin-bottom:6px; align-items:center; font-size:11px">
              <span class="muted">X:</span>
              <select id="tro-x" style="max-width:200px; padding:2px 6px"></select>
              <label class="muted"><input type="checkbox" id="tro-x-log" checked/> log</label>
              <span class="muted" style="margin-left:8px">Y:</span>
              <select id="tro-y" style="max-width:200px; padding:2px 6px"></select>
              <label class="muted"><input type="checkbox" id="tro-y-log"/> log</label>
              <button class="small ghost" id="tro-swap" title="swap X and Y axes">⇄</button>
            </div>
            <div id="ch-tradeoff" class="svg-host"></div>
          </div>
          <div class="chart-card"><h4>④ Throughput by backend</h4><div id="ch-throughput" class="svg-host"></div></div>
          <div class="chart-card">
            <h4>⑧ Real-time throughput over time (per run)</h4>
            <div class="row" style="gap:8px; margin-bottom:6px; align-items:center; font-size:11px">
              <span class="muted">metric:</span>
              <select id="tput-metric" style="max-width:220px; padding:2px 6px">
                <option value="windows_per_s">overall throughput (windows/s)</option>
                <option value="windows_per_s_cum">overall throughput cumulative (windows/s)</option>
                <option value="sketch_per_s">sketch phase (windows/s)</option>
                <option value="candidate_per_s">candidate phase (cand/s)</option>
                <option value="validate_per_s">validate phase (pairs/s)</option>
                <option value="monitor_per_s">monitor phase (corr/s)</option>
                <option value="cpu_pct">CPU (%)</option>
                <option value="mem_mb">memory (MB)</option>
                <option value="power_cores">energy proxy (cores busy)</option>
              </select>
              <label class="muted"><input type="checkbox" id="tput-log"/> log Y</label>
              <span class="muted" style="margin-left:8px">X:</span>
              <select id="tput-xaxis" style="padding:2px 6px">
                <option value="t">time (s)</option>
                <option value="frac">progress (%)</option>
              </select>
              <span class="muted" id="tput-note" style="font-size:10px; margin-left:6px"></span>
            </div>
            <div id="ch-tput-timeline" class="svg-host"></div>
          </div>
          <div class="chart-card" style="grid-column:1 / -1">
            <h4 style="display:flex; align-items:center; gap:10px; flex-wrap:wrap">
              <span style="flex:0 0 auto; white-space:nowrap">⑨ Resources — memory · energy · CPU · cores (min / median / max)</span>
              <span class="muted" style="font-weight:400; font-size:11px; margin-left:auto; text-transform:none; letter-spacing:0">
                D3 · whiskers = min–max, bar = median (from resources.csv)
                <span class="muted" id="ch-resources-note" style="margin-left:6px"></span>
              </span>
            </h4>
            <div id="ch-resources-grid"
                 style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px">
              <div class="res-block"><div class="res-title">Memory (MB)</div><div id="ch-res-mem" class="svg-host"></div></div>
              <div class="res-block"><div class="res-title">Energy (CPU core·s)</div><div id="ch-res-energy" class="svg-host"></div></div>
              <div class="res-block"><div class="res-title">CPU utilisation (%)</div><div id="ch-res-cpu" class="svg-host"></div></div>
              <div class="res-block"><div class="res-title">Concurrent cores</div><div id="ch-res-cores" class="svg-host"></div></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="tabbody" id="tab-pairs" style="display:none">
      <div class="row" style="margin-bottom:6px; align-items:center; gap:6px">
        <span class="muted" style="font-size:11px">Quick anchor:</span>
        <select id="p-quick-airport" style="max-width:200px; padding:3px 6px; font-size:12px"><option value="">— pick an airport (station id) —</option></select>
        <input id="p-quick-t" type="number" step="12" placeholder="t (optional, e.g. 215484)" style="width:170px; font-size:12px"/>
        <button class="small primary" id="p-quick-anchor">🔍 anchor</button>
        <span class="muted" style="font-size:11px">→ filters pairs to those touching this window (or any window of this airport when t is empty)</span>
      </div>
      <div class="row" style="margin-bottom:8px">
        <input id="p-q" type="text" placeholder="filter by id substring…" style="flex:2 1 auto"/>
        <input id="p-mincorr" type="number" step="0.01" min="0" max="1" placeholder="|corr| ≥ …"/>
        <input id="p-maxcorr" type="number" step="0.01" min="0" max="1" placeholder="|corr| ≤ …"/>
        <input id="p-lagmin" type="number" step="1" placeholder="lag ≥"/>
        <input id="p-lagmax" type="number" step="1" placeholder="lag ≤"/>
        <label class="muted" style="flex:0 0 auto"><input type="checkbox" id="p-noself" checked/>no self-pair</label>
        <label class="muted" style="flex:0 0 auto"><input type="checkbox" id="p-onlyself"/>only self</label>
        <label class="muted" style="flex:0 0 auto" title="adds the `by` column showing how many other engines detected the same pair — re-scans every correlated.csv on cold cache (~1m on 300+ runs)"><input type="checkbox" id="p-cover"/>coverage</label>
        <button id="p-search">Search</button>
      </div>
      <div id="p-anchor" style="display:none; margin-bottom:6px"></div>
      <div id="pairs-multi" style="display:none; margin-bottom:6px; padding:6px 10px; background:var(--panel2); border:1px solid var(--accent); border-radius:5px; align-items:center; gap:8px; font-size:11.5px;"></div>
      <div id="pairs-pager" class="pager"></div>
      <div style="max-height:36vh; overflow:auto; border:1px solid var(--line); border-radius:4px;">
        <table class="t" id="pairs-table">
          <thead><tr>
            <th style="width:24px"><input type="checkbox" id="pairs-sel-all" title="select all visible"/></th>
            <th data-sort="id1">window A · id</th>
            <th data-sort="t1" class="num">t</th>
            <th data-sort="id2">window B · id</th>
            <th data-sort="t2" class="num">t</th>
            <th data-sort="lag" class="num">lag</th>
            <th data-sort="abs_corr" class="active num">|corr| ↓</th>
            <th class="num">signed</th>
            <th class="num" title="how many engines (pipelines) detected this exact pair">by</th>
            <th title="add window A to basket">+A</th>
            <th title="add window B to basket">+B</th>
            <th title="add both windows to basket">+pair</th>
          </tr></thead>
          <tbody><tr><td colspan="12" class="empty">Set filters and hit Search. Click on the <b>A cells</b> (green hover) to add window A · <b>B cells</b> (orange hover) to add window B · anywhere else on the row, or <b>+ pair</b>, to add both.</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="tabbody" id="tab-windows" style="display:none">
      <div class="row" style="margin-bottom:8px">
        <input id="w-q" type="text" placeholder="filter by id substring…" style="flex:2 1 auto"/>
        <input id="w-top" type="number" step="50" min="10" max="5000" value="200" placeholder="top N"/>
        <button id="w-search">Compute / refresh</button>
        <span class="muted" style="flex:0 0 auto; font-size:11px;">(streams correlated.csv)</span>
      </div>
      <div style="max-height:42vh; overflow:auto; border:1px solid var(--line); border-radius:4px;">
        <table class="t" id="windows-table">
          <thead><tr>
            <th data-sort="id">id</th>
            <th data-sort="t" class="num">t</th>
            <th data-sort="n_partners" class="num">#partners</th>
            <th data-sort="max_abs_corr" class="active num">max|corr| ↓</th>
            <th data-sort="mean_abs_corr" class="num">mean|corr|</th>
            <th>add</th>
          </tr></thead>
          <tbody><tr><td colspan="6" class="empty">Click "Compute" to aggregate correlated pairs into per-window stats.</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="tabbody" id="tab-free" style="display:none">
      <div class="row" style="margin-bottom:6px">
        <select id="free-id" style="flex:2 1 auto"></select>
        <input id="free-t" type="number" step="1" placeholder="t (hour index)"/>
        <button id="free-add" class="primary">Add window</button>
      </div>
      <div class="muted" style="font-size:12px">
        Free windows are evaluated against the rest of your basket — useful to
        compare a chosen <em>non</em>-correlated window with the correlated ones.
      </div>
      <div id="free-stats" style="margin-top:10px"></div>
    </div>

    <div class="plotctrl">
      <label><input type="checkbox" id="opt-normalize"/> z-normalize</label>
      <label><input type="checkbox" id="opt-align" checked/> align on t=0</label>
      <label title="when not aligned, draw the full series history around the selected windows in a faded color"><input type="checkbox" id="opt-history" checked/> history (faded)</label>
      <label title="draw a marker at each sample (1 point per hour)"><input type="checkbox" id="opt-points" checked/> points</label>
      <label><input type="checkbox" id="opt-show-pad"/> context ±</label>
      <input id="opt-pad" type="number" min="0" max="2000" value="0" step="24" style="width:80px"/>
      <label><input type="checkbox" id="opt-grid" checked/> grid</label>
      <div style="flex:1"></div>
      <span class="muted" id="plot-status"></span>
    </div>
    <div class="plot-wrap">
      <div id="plot-empty" class="plot-empty">
        Pick at least one window (click a row in the table above) to render the plot.
      </div>
      <div id="plot" class="svg-host" style="background:var(--panel); border:1px solid var(--line); border-radius:4px; min-height:460px;"></div>
    </div>

    <!-- Map section: geographical view of basket stations + pairwise distances -->
    <div class="map-section" id="map-section" style="display:none; margin:14px 12px 18px; padding:12px 14px; background:var(--panel); border:1px solid var(--line); border-radius:6px;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid var(--line); flex-wrap:wrap;">
        <h2 style="margin:0; font-size:12px; text-transform:uppercase; letter-spacing:.7px; color:var(--fg); font-weight:600;">Geographical view</h2>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="map-show-lines" checked/> distance lines</label>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="map-show-labels" checked/> labels</label>
        <button class="small" id="map-fit" title="re-fit view to all markers">⊕ fit</button>
        <button class="small" id="map-expand" title="expand / collapse the map">⤢ expand</button>
        <div style="flex:1"></div>
        <span class="muted" id="map-status" style="font-size:11px"></span>
      </div>
      <div id="map-host" style="height:480px; border-radius:4px; overflow:hidden;"></div>
      <div class="muted" style="font-size:11px; margin-top:6px">
        scroll = zoom · drag = pan · double-click = zoom in · shift+drag = zoom to box · arrow keys = pan when focused
      </div>
    </div>

    <!-- Separate section: pairwise correlation matrices -->
    <div class="corr-section" id="corr-section" style="display:none">
      <div class="corr-section-head">
        <h2>Pairwise correlation</h2>
        <div class="corr-tabs" id="corr-tabs"></div>
        <label class="muted" style="font-size:11px"><input type="checkbox" id="opt-bestlag"/> compute best-lag</label>
        <span class="muted" id="corr-status" style="font-size:11px; margin-left:auto"></span>
      </div>
      <div id="corr-matrix"></div>
    </div>
  </div>

  <!-- RIGHT: basket -->
  <div class="col right">
    <div class="section">
      <h2>Basket — selected windows</h2>
      <div id="basket"><div class="empty">No windows yet.</div></div>
    </div>
  </div>

</div>

<script>
// ---------- minimal state ----------
const ST = {
  pipeline: null,
  windowSize: 168,
  source: null,
  ids: [],
  basket: [],   // {id, t, label, color, kind:'corr'|'free'|'partner'}
  pairsState: { offset:0, limit:50, sort:'abs_corr', order:'desc', anchor:null,
                selected: new Set() },
  windowsState: { sort:'max_abs_corr', order:'desc' },
};
const COLORS = ['#5aa1ff','#ffb05a','#7cd992','#ff7c7c','#c08bff','#5ddcdc','#ffe066','#ff95c5'];
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// ---------- API helpers ----------
async function api(path, params) {
  const url = new URL(path, location.origin);
  if (params) for (const [k,v] of Object.entries(params)) {
    if (v === '' || v == null) continue;
    if (Array.isArray(v)) {
      for (const item of v) {
        if (item === '' || item == null) continue;
        url.searchParams.append(k, item);
      }
    } else {
      url.searchParams.set(k, v);
    }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- bootstrap ----------
async function bootstrap() {
  const ps = await api('/api/pipelines');
  if (!ps.pipelines.length) {
    $('#run-meta').textContent = 'no pipelines found in ' + (ps.root || '?');
  } else {
    $('#run-meta').textContent = 'no pipeline opened — pick one in the Home table';
  }
  // Populate the sidebar Phases filter as the very first thing — safe now
  // that PHASE_KEYS has been initialized (bootstrap runs after parse).
  renderPhaseFilters();
  // landing page: compare runs + charts + auto precision/recall + watch chip
  await loadCompare();
  loadCharts();
  loadQuality(true);
  _refreshRunStatus();
  // Auto-refresh ON by default — kicks in immediately so the user sees live progress
  if (ST.auto.on) _startAutoRefresh();
}
$('#refresh-now-btn')?.addEventListener('click', _refreshRunStatus);
$('#phase-filters-reset-btn')?.addEventListener('click', _resetPhases);
$('#phase-filters-hide-btn')?.addEventListener('click', _hideAllPhases);
// NOTE: don't call renderPhaseFilters() here — PHASE_KEYS is declared with
// `const` later in the file, so a top-level call hits the temporal dead
// zone and silently leaves the sidebar empty. bootstrap() runs after the
// whole script is parsed (its first `await` yields a tick), which is the
// safe place to populate the sidebar.
bootstrap();

function _renderLeftPanel(info, pipelinePath, isActive) {
  $('#summary').innerHTML = renderKV(info.summary_pretty);
  $('#source-info').innerHTML = renderKV({
    'csv': info.source || '(not found — pass --dataset on CLI)',
    'series in CSV': info.ids ? info.ids.length : '?',
    'rows in CSV': info.source_rows ?? '?',
    'window_size': info.window_size,
    'n_correlated rows': info.n_correlated,
  });
}

async function previewPipeline(path) {
  // Update LEFT panels from `path`'s stats — but don't change the active
  // pipeline (so the basket and the other tabs stay anchored).
  try {
    const info = await api('/api/pipeline_info', {pipeline: path});
    _renderLeftPanel(info, path, false);
    const label = path.split('/').slice(-2).join('/');
    const active = ST.pipeline ? ST.pipeline.split('/').slice(-2).join('/') : '?';
    $('#run-meta').innerHTML = `previewing <b>${escapeHtml(label)}</b> · active: ${escapeHtml(active)}`;
  } catch (e) { /* ignore */ }
}

async function setActivePipeline(path, resetBasket) {
  ST.pipeline = path;
  document.querySelector('.tabs').classList.add('has-active');
  // FULL reset of the Correlated-pairs page so nothing from the previous
  // pipeline lingers (basket, plot, map, matrix, anchor, filters, selection).
  if (resetBasket !== false) {
    ST.basket = [];
    renderBasket();
    refreshPairsBasketIndicators();
  }
  // Wipe pairs filter state + UI
  ST.pairsState = { offset:0, limit:50, sort:'abs_corr', order:'desc',
                    anchor:null, selected: new Set() };
  ST._lastPairsRows = [];
  ST._lastCovByRow = new Map();
  for (const id of ['p-q','p-mincorr','p-maxcorr','p-lagmin','p-lagmax']) {
    const el = $('#'+id); if (el) el.value = '';
  }
  const ns = $('#p-noself'); if (ns) ns.checked = true;
  const os = $('#p-onlyself'); if (os) os.checked = false;
  const ach = $('#p-anchor'); if (ach) { ach.style.display='none'; ach.innerHTML=''; }
  const mb = $('#pairs-multi'); if (mb) { mb.style.display='none'; mb.innerHTML=''; }
  const psa = $('#pairs-sel-all'); if (psa) { psa.checked = false; psa.indeterminate = false; }
  // Wipe plot / matrix / map
  const plot = $('#plot'); if (plot) plot.innerHTML = '';
  const pe = $('#plot-empty'); if (pe) pe.style.display = '';
  const cm = $('#corr-matrix'); if (cm) cm.innerHTML = '';
  const cs = $('#corr-section'); if (cs) cs.style.display = 'none';
  const ms = $('#map-section'); if (ms) ms.style.display = 'none';
  if (ST.leafletMap) { try { ST.leafletMap.remove(); } catch {} ST.leafletMap = null; }
  ST._mapLastBasketKey = '';
  $('#plot-status').textContent = '';

  const info = await api('/api/pipeline_info', {pipeline: path});
  ST.windowSize = info.window_size || 168;
  ST.source = info.source;
  ST.ids = info.ids || [];
  _renderLeftPanel(info, path, true);
  $('#sel-ws').textContent = ST.windowSize;
  const label = path.split('/').slice(-2).join('/');
  $('#run-meta').innerHTML = `active: <b>${escapeHtml(label)}</b> · ${escapeHtml(info.short_meta || '')}`;
  // Populate quick-anchor airport dropdown with the source CSV's station ids
  const qa = $('#p-quick-airport');
  if (qa) {
    qa.innerHTML = '<option value="">— pick an airport (station id) —</option>' +
      (ST.ids || []).map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join('');
  }
  const fs = $('#free-id');
  if (fs) {
    fs.innerHTML = '';
    for (const id of (ST.ids || [])) {
      const o = document.createElement('option'); o.value = id; o.textContent = id;
      fs.appendChild(o);
    }
  }
  // Reset placeholders for empty tables
  $('#pairs-table tbody').innerHTML =
    '<tr><td colspan="12" class="empty">Set filters and hit Search.</td></tr>';
  if (ST.cmpData) renderCompare();
}

function renderKV(obj) {
  if (!obj || typeof obj !== 'object') return '<span class="muted">—</span>';
  let h = '';
  for (const [k,v] of Object.entries(obj)) {
    h += `<span class="k">${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`;
  }
  return h;
}
function escapeHtml(s){return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}

// ---------- tabs ----------
$$('.tabs .tab').forEach(t => t.addEventListener('click', () => {
  $$('.tabs .tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  for (const id of ['compare','pairs','windows','free']) {
    $('#tab-'+id).style.display = (id === t.dataset.tab) ? '' : 'none';
  }
}));
function switchTab(name) {
  $$('.tabs .tab').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  for (const id of ['compare','pairs','windows','free']) {
    $('#tab-'+id).style.display = (id === name) ? '' : 'none';
  }
  // Auto-scroll to the top of the center column so the user actually sees it
  document.querySelector('.col.center')?.scrollTo({top:0, behavior:'smooth'});
}
$('#home-link').addEventListener('click', () => switchTab('compare'));

// ---------- Auto-refresh + run-status polling ----------
ST.auto = { on:false, intervalSec:5, timer:null, lastFetchAt:null };
function _fmtAgo(ms) {
  if (ms == null) return '';
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 5) return 'just now';
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  return Math.floor(s/3600) + 'h ago';
}
function _renderStatusCell(x) {
  const s = x.status || 'unknown';
  const dot = `<span class="status-dot ${s}"></span>`;
  if (s === 'running' && x.progress) {
    const pct = Math.round(x.progress.progress * 100);
    return `<div class="status-cell">${dot}<span class="progress-bar"><span style="width:${pct}%"></span></span><span>${pct}% · ETA ${x.progress.eta_seconds}s</span></div>`;
  }
  if (s === 'starting') return `<div class="status-cell">${dot}<span>starting…</span></div>`;
  if (s === 'failed')   return `<div class="status-cell">${dot}<span>failed</span></div>`;
  if (s === 'queued')   return `<div class="status-cell">${dot}<span>queued</span></div>`;
  if (s === 'done')     return `<div class="status-cell">${dot}<span style="color:var(--muted)">done</span></div>`;
  return `<div class="status-cell">${dot}<span class="muted">${s}</span></div>`;
}
async function _refreshRunStatus() {
  try {
    const r = await api('/api/runs_status');
    // Two independent triggers for a full compare/charts/quality refresh:
    //  (1) a run transitioned active → terminal between this tick and the
    //      previous one (caught via ST.lastStatuses diff)
    //  (2) a run is reported done/failed on the live endpoint but our
    //      cached cmpData row for it is empty (runtime ≤ 0 OR backend
    //      missing) — this catches runs that finished BEFORE the page
    //      ever rendered, e.g. while loadCompare was inflight or before
    //      result.json finished being flushed to disk.
    // Without (2), a "done · 0.00s · —×" row stays frozen forever because
    // there's no transition to observe.
    const ACTIVE = new Set(['running', 'starting', 'queued']);
    const TERMINAL = new Set(['done', 'failed']);
    let anyFinished = false;
    let staleDone = false;
    const cmpByPath = ST.cmpData
      ? new Map(ST.cmpData.rows.map(x => [x.path, x]))
      : new Map();
    if (r.rows) {
      for (const row of r.rows) {
        const prev = ST.lastStatuses ? ST.lastStatuses.get(row.path) : null;
        if (prev && ACTIVE.has(prev) && TERMINAL.has(row.status)) {
          anyFinished = true;
        }
        if (TERMINAL.has(row.status)) {
          const cmp = cmpByPath.get(row.path);
          // _orig_runtime is the unmutated runtime from the server (the
          // _applyEffectiveMetrics step rewrites `runtime` itself, so test
          // the original where available).
          const cmpRt = cmp ? (cmp._orig_runtime ?? cmp.runtime ?? 0) : 0;
          const cmpHasData = cmp && cmpRt > 0 && (cmp.backend || '');
          // Retry stale-done runs AT MOST ONCE per session — _detect_run_status
          // can legitimately return "done" with runtime=0 (e.g., sharded runs
          // where result.json was written without `runtime` but a correlated.csv
          // exists). Without this guard, those rows would trigger a reload
          // every 4s forever, hammering the server for nothing.
          ST.staleRetried = ST.staleRetried || new Set();
          if (cmp && !cmpHasData && !ST.staleRetried.has(row.path)) {
            staleDone = true;
            ST.staleRetried.add(row.path);
          }
        }
        if (anyFinished && staleDone) break;
      }
    }
    ST.lastStatuses = new Map((r.rows || []).map(x => [x.path, x.status]));
    ST.runStatus = r;
    _renderRunsBanner(r);
    // If anything is running and the table is visible, surgically update its status cells
    // (no full re-render to keep selections + scroll position intact).
    if (r.rows && ST.cmpData) {
      const mp = new Map(r.rows.map(x => [x.path, x]));
      document.querySelectorAll('#cmp-table tr[data-pipeline]').forEach(tr => {
        const live = mp.get(tr.dataset.pipeline);
        if (live) {
          // status TD = index 2 (sel-cell + open + status)
          const td = tr.children[2];
          if (td) td.innerHTML = _renderStatusCell(live);
          tr.dataset.status = live.status || 'unknown';
          tr.classList.toggle('row-running', live.status === 'running' || live.status === 'starting');
        }
      });
    }
    ST.auto.lastFetchAt = Date.now();
    _updateAutoStatus();
    // Throttle: at most one full reload every 4s, so a persistently stale
    // server response (e.g., a result.json that's still half-written or a
    // genuine zero-runtime failure) doesn't pin the CPU in a reload loop.
    if (anyFinished || staleDone) {
      const now = Date.now();
      if (!ST._lastFullReload || (now - ST._lastFullReload) >= 4000) {
        ST._lastFullReload = now;
        ST.cmpQuality = null;
        await loadCompare();
        loadCharts();
        loadQuality(true);
        ST.auto._cycleCount = 0;
      }
    }
  } catch (e) { /* ignore */ }
}
function _renderRunsBanner(rs) {
  const chip = $('#watch-chip');
  const running = (rs.counts?.running || 0) + (rs.counts?.starting || 0);
  const queued = rs.counts?.queued || 0;
  if (running > 0 || queued > 0) {
    const dot = '<span class="status-dot running" style="display:inline-block; vertical-align:-1px; margin-right:6px"></span>';
    chip.innerHTML = `${dot}${running} running${queued ? ' · ' + queued + ' queued' : ''}`;
    chip.style.display = '';
    chip.title = rs.pipeline_name ? `pipeline: ${rs.pipeline_name}` : '';
  } else {
    chip.style.display = 'none';
  }
  // Big banner in the Home tab: list each currently-running run with its progress
  const banner = $('#runs-running-banner');
  if (!banner) return;
  const liveRunning = (rs.rows || []).filter(r => r.status === 'running' || r.status === 'starting');
  if (!liveRunning.length) { banner.style.display = 'none'; return; }
  banner.style.display = 'flex';
  banner.style.flexDirection = 'column';
  banner.style.gap = '4px';
  banner.innerHTML =
    `<div style="display:flex; align-items:center; gap:8px">
      <span class="status-dot running" style="width:11px; height:11px"></span>
      <b style="font-size:12px">${liveRunning.length} run${liveRunning.length>1?'s':''} in progress</b>
      ${queued ? `<span class="muted" style="font-size:11px">· ${queued} more queued</span>` : ''}
    </div>` +
    liveRunning.map(r => {
      const p = r.progress;
      const pct = p ? Math.round(p.progress * 100) : 0;
      const eta = p ? `ETA ${p.eta_seconds}s · ${p.win_per_s.toFixed(1)} win/s` : 'starting…';
      const corr = p ? ` · cand=${p.n_candidates.toLocaleString()} corr=${p.n_correlated.toLocaleString()}` : '';
      return `<div style="display:flex; align-items:center; gap:8px; font-size:11.5px; font-family:var(--mono)">
        <code style="color:var(--accent); flex:0 0 auto">${escapeHtml(r.label)}</code>
        <span class="progress-bar" style="flex:0 0 140px"><span style="width:${pct}%"></span></span>
        <span style="flex:0 0 auto">${pct}%</span>
        <span class="muted" style="flex:1">${eta}${corr}</span>
      </div>`;
    }).join('');
}
function _updateAutoStatus() {
  const s = $('#auto-refresh-status');
  if (!s) return;
  if (!ST.auto.on) { s.textContent = ''; return; }
  s.textContent = 'last ' + _fmtAgo(ST.auto.lastFetchAt) + ' · every ' + ST.auto.intervalSec + 's';
}
function _startAutoRefresh() {
  _stopAutoRefresh();
  if (!ST.auto.on) return;
  // Fire one immediately, then on interval
  _refreshRunStatus();
  ST.auto.timer = setInterval(async () => {
    await _refreshRunStatus();
    // Also re-pull /api/compare every ~30s (for new runs / finished runs to appear)
    if (!ST.auto._cycleCount) ST.auto._cycleCount = 0;
    ST.auto._cycleCount++;
    if (ST.auto._cycleCount * ST.auto.intervalSec >= 30) {
      ST.auto._cycleCount = 0;
      // Drop quality (so it recomputes for any new runs)
      ST.cmpQuality = null;
      await loadCompare();
      loadCharts();
      loadQuality(true);
    }
  }, ST.auto.intervalSec * 1000);
  _updateAutoStatus();
}
function _stopAutoRefresh() {
  if (ST.auto.timer) { clearInterval(ST.auto.timer); ST.auto.timer = null; }
  _updateAutoStatus();
}
// Update the "last Xs ago" string every second
setInterval(_updateAutoStatus, 1000);

// ---------- Remote SSH banner (top of page, only when --pipeline is user@host:/…)
const RB = { remote: false, intervalSet: false };
async function _refreshRemoteBanner() {
  const bar = document.getElementById('remote-banner');
  if (!bar) return;
  let r;
  try { r = await fetch('/api/sync_status').then(x => x.json()); }
  catch (e) { return; }
  if (!r.remote) { bar.style.display = 'none'; RB.remote = false; return; }
  RB.remote = true;
  bar.style.display = '';
  // Target text: user@host:/path  (hover shows local mirror)
  const tgt = document.getElementById('rb-target');
  tgt.textContent = (r.target || '') + ':' + (r.remote_path || '');
  tgt.title = 'local mirror: ' + (r.local_mirror || '');
  // Status line with dot
  const st = document.getElementById('rb-status');
  const ago = r.at ? _fmtAgo(r.at * 1000) : 'never';
  const dur = r.duration ? ' (' + r.duration.toFixed(1) + 's)' : '';
  const ch = r.n_changes ? ', +' + r.n_changes + ' file' + (r.n_changes>1?'s':'') : '';
  let dotCls = 'rb-dot';
  if (r.last_error) dotCls += ' err';
  else if (!r.at) dotCls += ' idle';
  st.innerHTML = '<span class="' + dotCls + '"></span>' +
                 (r.last_error
                    ? 'sync ERROR — ' + r.last_error.split('\n')[0]
                    : 'synced ' + ago + dur + ch +
                      ' · cycle #' + (r.n_cycles || 0));
  st.classList.toggle('err', !!r.last_error);
  // One-time: sync the interval dropdown to whatever the backend reports.
  if (!RB.intervalSet) {
    const sel = document.getElementById('rb-interval');
    const v = (r.interval == null ? 5 : r.interval);
    const opts = Array.from(sel.options).map(o => parseFloat(o.value));
    if (!opts.includes(v)) {
      const o = new Option(v + 's', String(v), true, true);
      sel.add(o);
    }
    sel.value = String(v);
    RB.intervalSet = true;
  }
}
setInterval(_refreshRemoteBanner, 2000);
_refreshRemoteBanner();

document.getElementById('rb-sync-now')?.addEventListener('click', async () => {
  const btn = document.getElementById('rb-sync-now');
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = '⇅ syncing…';
  try {
    const r = await fetch('/api/sync_now', {method:'POST'}).then(x => x.json());
    if (!r.ok) alert('sync failed: ' + (r.last_error || 'unknown'));
    await _refreshRemoteBanner();
    await _refreshRunStatus();
  } finally {
    btn.disabled = false; btn.textContent = old;
  }
});

document.getElementById('rb-interval')?.addEventListener('change', async (e) => {
  const v = parseFloat(e.target.value);
  await fetch('/api/sync_interval', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'seconds=' + encodeURIComponent(String(v)),
  });
  _refreshRemoteBanner();
});
$('#auto-refresh-toggle')?.addEventListener('change', e => {
  ST.auto.on = e.target.checked;
  _startAutoRefresh();
});
$('#auto-refresh-int')?.addEventListener('change', e => {
  ST.auto.intervalSec = parseInt(e.target.value, 10) || 5;
  if (ST.auto.on) _startAutoRefresh();
});

// ---------- Compare runs (landing page) ----------
// sorts is an ordered list of {key, order}; first item is primary, the rest
// are tie-breakers. Single click on a header replaces the whole list;
// shift+click appends (or flips order if the key is already in the list).
ST.cmpState = { sorts: [{key:'runtime', order:'asc'}] };
ST.cmpData = null;
const _DEFAULT_SORT_DESC = new Set([
  'mode','backend','index_backend','n_candidates','n_correlated',
  'speedup_vs_slowest','precision','recall','f1','win_per_s','cand_per_s',
]);
function _onSortClick(e, key) {
  const sorts = ST.cmpState.sorts;
  if (e.shiftKey) {
    const idx = sorts.findIndex(s => s.key === key);
    if (idx >= 0) {
      sorts[idx].order = sorts[idx].order === 'asc' ? 'desc' : 'asc';
    } else {
      sorts.push({key, order: _DEFAULT_SORT_DESC.has(key) ? 'desc' : 'asc'});
    }
  } else {
    if (sorts.length === 1 && sorts[0].key === key) {
      sorts[0].order = sorts[0].order === 'asc' ? 'desc' : 'asc';
    } else {
      ST.cmpState.sorts = [{key, order: _DEFAULT_SORT_DESC.has(key) ? 'desc' : 'asc'}];
    }
  }
  renderCompare();
}
function _renderSortHeaders() {
  const sorts = ST.cmpState.sorts;
  const multi = sorts.length > 1;
  document.querySelectorAll('#cmp-table th[data-sort]').forEach(th => {
    const key = th.dataset.sort;
    // cache the original label so we can re-apply it cleanly
    if (!th.dataset.label) th.dataset.label = th.textContent.trim();
    const idx = sorts.findIndex(s => s.key === key);
    th.classList.toggle('active', idx >= 0);
    if (idx >= 0) {
      const arrow = sorts[idx].order === 'asc' ? '↑' : '↓';
      const num = multi ? ' ' + ['①','②','③','④','⑤'][idx] : '';
      th.innerHTML = `${escapeHtml(th.dataset.label)} ${arrow}${num}`;
    } else {
      th.textContent = th.dataset.label;
    }
  });
}
ST.cmpSel = new Set();        // selected pipeline paths
ST.cmpQuality = null;          // precision/recall map keyed by path

function _cellColor(v, kind) {
  // Returns an inline `background:` style for a metric cell.
  // kind: 'speedup' (any v, neutral at 1), 'pr' (precision/recall in [0,1])
  if (v == null || Number.isNaN(v) || !Number.isFinite(v)) return '';
  if (kind === 'speedup') {
    if (v >= 10) return 'background:rgba(40,160,90,.55);color:#fff';
    if (v >= 5)  return 'background:rgba(80,190,120,.45);color:#fff';
    if (v >= 2)  return 'background:rgba(124,217,146,.40)';
    if (v >= 1)  return 'background:rgba(124,217,146,.18)';
    return 'background:rgba(255,124,124,.30)';
  }
  // precision / recall in [0,1]
  if (v >= 0.95) return 'background:rgba(40,160,90,.55);color:#fff';
  if (v >= 0.85) return 'background:rgba(124,217,146,.45)';
  if (v >= 0.70) return 'background:rgba(255,224,102,.35)';
  if (v >= 0.50) return 'background:rgba(255,176,90,.40)';
  return 'background:rgba(255,124,124,.45);color:#fff';
}
function _selChanged() {
  $('#cmp-sel-count').textContent = `${ST.cmpSel.size} selected`;
  if ($('#cmp-charts-sel').checked) loadCharts();
}
// Multi-select state. Empty Set = "all" (no filter).
ST.cmpFilters = { backend: new Set(), index_backend: new Set(),
                  index_key_mode: new Set(), sketch_method: new Set() };

function _setupMultiSelect(rootEl, key, allLabel, getValues) {
  if (rootEl.dataset.wired) return;  // wire once
  rootEl.dataset.wired = '1';
  const trigger = rootEl.querySelector('.ms-trigger');
  const panel   = rootEl.querySelector('.ms-panel');
  const optsHost = rootEl.querySelector('.ms-options');
  const search  = rootEl.querySelector('.ms-search');
  const sel = ST.cmpFilters[key];

  function refreshTrigger() {
    if (sel.size === 0) {
      trigger.textContent = 'all ' + allLabel;
    } else if (sel.size === 1) {
      trigger.textContent = [...sel][0];
    } else if (sel.size <= 3) {
      trigger.textContent = [...sel].join(' · ');
    } else {
      trigger.textContent = `${sel.size} ${allLabel} selected`;
    }
    rootEl.classList.toggle('ms-active', sel.size > 0);
  }
  function rebuildOpts() {
    const values = getValues();
    optsHost.innerHTML = values.map(v =>
      `<label data-val="${escapeHtml(v)}"><input type="checkbox" value="${escapeHtml(v)}" ${sel.has(v)?'checked':''}/>${escapeHtml(v)}</label>`
    ).join('') || '<div class="muted" style="padding:4px; font-size:11px">no values</div>';
    optsHost.querySelectorAll('input').forEach(cb => {
      cb.addEventListener('change', () => {
        if (cb.checked) sel.add(cb.value); else sel.delete(cb.value);
        refreshTrigger();
        renderCompare(); loadCharts();
      });
    });
    applySearch();
  }
  function applySearch() {
    const q = (search.value || '').toLowerCase().trim();
    optsHost.querySelectorAll('label').forEach(l => {
      l.classList.toggle('hidden', q && !l.dataset.val.toLowerCase().includes(q));
    });
  }
  trigger.addEventListener('click', e => {
    e.stopPropagation();
    // Close other panels
    document.querySelectorAll('.ms-panel').forEach(p => { if (p !== panel) p.hidden = true; });
    if (panel.hidden) {
      rebuildOpts();
      panel.hidden = false;
      search.focus();
    } else {
      panel.hidden = true;
    }
  });
  search.addEventListener('input', applySearch);
  rootEl.querySelector('.ms-all').addEventListener('click', () => {
    getValues().forEach(v => sel.add(v));
    rebuildOpts(); refreshTrigger();
    renderCompare(); loadCharts();
  });
  rootEl.querySelector('.ms-none').addEventListener('click', () => {
    sel.clear();
    rebuildOpts(); refreshTrigger();
    renderCompare(); loadCharts();
  });
  // expose for external resets
  rootEl._refresh = refreshTrigger;
  rootEl._reset = () => { sel.clear(); refreshTrigger(); };
  refreshTrigger();
}

function _populateDimensionFilters() {
  if (!ST.cmpData) return;
  // Real values only (filter out '' and 'n/a' — they're shown as a dedicated entry).
  const valsFn = (key, allowNA) => () => {
    const real = [...new Set(ST.cmpData.rows.map(r => r[key]).filter(v => v && v !== 'n/a'))].sort();
    // Include a synthetic '(n/a)' entry if at least one row has 'n/a' or empty
    if (allowNA && ST.cmpData.rows.some(r => !r[key] || r[key] === 'n/a')) {
      real.push('(n/a)');
    }
    return real;
  };
  _setupMultiSelect($('#ms-backend'), 'backend',         'backends',        valsFn('backend', false));
  _setupMultiSelect($('#ms-index'),   'index_backend',   'indexes',         valsFn('index_backend', false));
  _setupMultiSelect($('#ms-keymode'), 'index_key_mode',  'key modes',       valsFn('index_key_mode', true));
  _setupMultiSelect($('#ms-sketch'),  'sketch_method',   'sketch methods',  valsFn('sketch_method', true));
  // Drop selected values that no longer exist in the data
  for (const [key, set] of Object.entries(ST.cmpFilters)) {
    const valid = new Set([...new Set(ST.cmpData.rows.map(r => r[key]).filter(v => v && v !== 'n/a'))]);
    for (const v of [...set]) if (!valid.has(v)) set.delete(v);
  }
  for (const id of ['ms-backend','ms-index','ms-keymode','ms-sketch']) {
    $('#'+id)._refresh?.();
  }
}
// global click closes any open panel
document.addEventListener('click', e => {
  if (!e.target.closest('.ms')) {
    document.querySelectorAll('.ms-panel').forEach(p => p.hidden = true);
  }
});
// Baseline picker -----------------------------------------------------------
// The server sends `baseline_candidates` = list of {key, label, path, runtime,
// mode, backend}. The user can switch baseline at runtime; speedups are
// recomputed client-side from `runtime` so we don't round-trip the server.
function _populateBaselinePicker() {
  const sel = document.getElementById('cmp-baseline');
  if (!sel) return;
  const cands = (ST.cmpData && ST.cmpData.baseline_candidates) || [];
  // Preserve selection across reloads if the chosen baseline is still present.
  const prev = ST.cmpBaseline || 'auto';
  sel.innerHTML = cands.length
    ? cands.map(c => `<option value="${escapeHtml(c.key)}" data-path="${escapeHtml(c.path)}" `
        + `data-runtime="${c.runtime}">${escapeHtml(c.label)}</option>`).join('')
        + '<option value="none">(none — disable ×baseline)</option>'
    : '<option value="none">no candidate (no completed run yet)</option>';
  const has = cands.some(c => c.key === prev);
  sel.value = has ? prev : (cands.length ? cands[0].key : 'none');
  ST.cmpBaseline = sel.value;
}
function _applyBaselineSelection() {
  // Recompute every row's `speedup_vs_baseline` based on the selected pick.
  const r = ST.cmpData;
  if (!r || !r.rows) return;
  const sel = document.getElementById('cmp-baseline');
  const key = ST.cmpBaseline = (sel ? sel.value : 'auto');
  const cands = r.baseline_candidates || [];
  const chosen = cands.find(c => c.key === key);
  if (!chosen) {
    // none / missing → clear speedups and baseline meta
    for (const row of r.rows) row.speedup_vs_baseline = null;
    r.baseline_label = null;
    r.baseline_path = null;
    r.baseline_runtime = null;
    return;
  }
  const brt = chosen.runtime;
  for (const row of r.rows) {
    const rt = +row.runtime || 0;
    row.speedup_vs_baseline = (rt > 0 && brt > 0) ? (brt / rt) : null;
    row._baseline_label = chosen.label;
    row._baseline_path = chosen.path;
  }
  r.baseline_label = chosen.label;
  r.baseline_path = chosen.path;
  r.baseline_runtime = chosen.runtime;
}
document.addEventListener('change', e => {
  if (e.target && e.target.id === 'cmp-baseline') {
    _applyBaselineSelection();
    if (typeof renderCompare === 'function') renderCompare();
    if (typeof loadCharts === 'function') loadCharts();
    // Precision/recall are defined RELATIVE to the baseline pair-set, so a
    // baseline change must recompute them too. Drop the stale map first.
    ST.cmpQuality = null;
    if (typeof loadQuality === 'function') loadQuality(true);
  }
});

async function loadCompare() {
  const tb = $('#cmp-table tbody');
  tb.innerHTML = '<tr><td colspan="29" class="spin">Loading run stats…</td></tr>';
  try {
    const r = await api('/api/compare');
    ST.cmpData = r;
    _populateDimensionFilters();
    _populateBaselinePicker();
    _applyBaselineSelection();
    $('#cmp-root').textContent = r.root || '';
    const n = r.rows.length;
    const counts = {};
    for (const x of r.rows) counts[x.mode] = (counts[x.mode]||0) + 1;
    const parts = [`${n} runs`];
    for (const m of ['bf','corrtrack','filcorr','?']) {
      if (counts[m]) parts.push(`${counts[m]} ${m}`);
    }
    if (r.fastest_bf)         parts.push(`fastest bf=${r.fastest_bf.toFixed(2)}s`);
    if (r.fastest_corrtrack)  parts.push(`fastest corrtrack=${r.fastest_corrtrack.toFixed(2)}s`);
    if (r.fastest_filcorr)    parts.push(`fastest filcorr=${r.fastest_filcorr.toFixed(2)}s`);
    if (r.baseline_label)     parts.push(`baseline=${r.baseline_label.split('/').pop()} (${r.baseline_runtime.toFixed(2)}s)`);
    $('#cmp-summary').textContent = parts.join(' · ');
    renderCompare();
  } catch (e) {
    tb.innerHTML = `<tr><td colspan="29" class="empty">Error: ${escapeHtml(String(e))}</td></tr>`;
  }
}
function renderCompare() {
  const r = ST.cmpData;
  if (!r) return;
  const showBf = $('#cmp-bf').checked, showCt = $('#cmp-ct').checked,
        showFc = $('#cmp-fc').checked;
  const onlySel = $('#cmp-only-sel').checked;
  const F = ST.cmpFilters;
  const minPrec    = parseFloat($('#cmp-min-prec')?.value);
  const minRec     = parseFloat($('#cmp-min-recall')?.value);
  const minF1      = parseFloat($('#cmp-min-f1')?.value);
  const minSpeedup = parseFloat($('#cmp-min-speedup')?.value);
  const hideNoQ    = $('#cmp-hide-noq')?.checked;
  let rows = r.rows.filter(x => {
    if (onlySel && !ST.cmpSel.has(x.path)) return false;
    if (x.mode === 'bf' && !showBf) return false;
    if (x.mode === 'corrtrack' && !showCt) return false;
    if (x.mode === 'filcorr' && !showFc) return false;
    const _matchF = (sel, v) => {
      if (!sel.size) return true;
      const isNA = !v || v === 'n/a';
      if (isNA && sel.has('(n/a)')) return true;
      return sel.has(v || '');
    };
    if (!_matchF(F.backend,        x.backend))         return false;
    if (!_matchF(F.index_backend,  x.index_backend))   return false;
    if (!_matchF(F.index_key_mode, x.index_key_mode))  return false;
    if (!_matchF(F.sketch_method,  x.sketch_method))   return false;
    if (Number.isFinite(minSpeedup) &&
        (x.speedup_vs_baseline == null || x.speedup_vs_baseline < minSpeedup)) return false;
    return true;
  });
  // merge quality data if available
  const q = ST.cmpQuality || {};
  rows = rows.map(x => ({...x, ...(q[x.path] || {})}));
  // Recompute runtime / speedup / throughput from the user's visible-phase
  // selection so the table mirrors what the charts show.
  rows = _applyEffectiveMetrics(rows);
  // Quality filters apply AFTER merging precision/recall
  rows = rows.filter(x => {
    const hasQ = x.precision != null || x.recall != null;
    if (hideNoQ && !hasQ) return false;
    if (Number.isFinite(minPrec) && (x.precision == null || x.precision < minPrec)) return false;
    if (Number.isFinite(minRec)  && (x.recall    == null || x.recall    < minRec))  return false;
    if (Number.isFinite(minF1)   && (x.f1        == null || x.f1        < minF1))   return false;
    return true;
  });
  // Multi-level sort: iterate through ST.cmpState.sorts, each level becomes a
  // tie-breaker for the previous ones. Null/NaN sort to the end regardless.
  rows = rows.slice().sort((a,b) => {
    for (const {key, order} of ST.cmpState.sorts) {
      const av = a[key], bv = b[key];
      const aNull = av == null || Number.isNaN(av);
      const bNull = bv == null || Number.isNaN(bv);
      if (aNull && bNull) continue;
      if (aNull) return 1;
      if (bNull) return -1;
      if (av === bv) continue;
      return (av < bv ? -1 : 1) * (order === 'asc' ? 1 : -1);
    }
    return 0;
  });
  _renderSortHeaders();
  const tb = $('#cmp-table tbody');
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="29" class="empty">No runs match the filters.</td></tr>';
    return;
  }
  const fastest = Math.min(...rows.filter(x => x.runtime > 0).map(x => x.runtime));
  tb.innerHTML = rows.map(x => {
    const isFast = x.runtime > 0 && Math.abs(x.runtime - fastest) < 1e-9;
    const isActive = ST.pipeline && ST.pipeline === x.path;
    const isSel = ST.cmpSel.has(x.path);
    const cls = (isActive ? 'active-row ' : '') + (isSel ? 'sel-row' : '');
    const fmt = (v, d=2) => (v == null || Number.isNaN(v)) ? '—' : v.toFixed(d);
    const num = (v) => (v == null) ? '—' : Number(v).toLocaleString();
    const spdStyle = _cellColor(x.speedup_vs_baseline, 'speedup');
    const prStyle = _cellColor(x.precision, 'pr');
    const reStyle = _cellColor(x.recall, 'pr');
    const f1Style = _cellColor(x.f1, 'pr');
    // Disable selection + open for non-done runs (no result.json yet).
    const isDone = x.status === 'done';
    const cbHtml = isDone
      ? `<input type="checkbox" class="row-sel" ${isSel?'checked':''}/>`
      : `<input type="checkbox" disabled title="run not finished yet" style="opacity:0.4; cursor:not-allowed"/>`;
    const openHtml = isDone
      ? `<button class="small primary open-btn" title="open this pipeline (loads Correlated pairs)"><svg viewBox="0 0 16 16" width="11" height="11" fill="currentColor"><path d="M3 2v12l11-6z"/></svg>open</button>`
      : `<button class="small open-btn" disabled style="opacity:0.4; cursor:not-allowed" title="cannot open: run is ${x.status}"><svg viewBox="0 0 16 16" width="11" height="11" fill="currentColor"><path d="M3 2v12l11-6z"/></svg>open</button>`;
    // Run identity is now conveyed by mode/backend/index/key/sketch columns;
    // keep the path as a tr-level tooltip for reference.
    const shortName = x.label.split('/').pop();
    // RUN column hidden — the unique identity is given by mode+backend+index+key+sketch.
    // We keep ★ (fastest) and "active" markers in the STATUS column, and put the
    // full run name as a tr-level tooltip so it's still discoverable on hover.
    const statusExtras = (isFast ? ' <span class="chip ok" title="fastest in current filter">★</span>' : '')
                        + (isActive ? ' <span class="chip info" title="active pipeline">active</span>' : '');
    return `<tr class="${cls}" data-pipeline="${escapeHtml(x.path)}" data-done="${isDone?1:0}" data-status="${escapeHtml(x.status||'unknown')}" title="${escapeHtml(shortName)} — ${escapeHtml(x.label)}">
      <td class="sel-cell">${cbHtml}</td>
      <td class="actions">${openHtml}</td>
      <td>${_renderStatusCell(x)}${statusExtras}</td>
      <td><span class="chip ${ {bf:'warn', corrtrack:'ok', filcorr:'info', '?':''}[x.mode] ?? '' }">${escapeHtml(x.mode)}</span></td>
      <td style="font-family:var(--mono); font-size:11px" title="${escapeHtml(x.label)}">${escapeHtml(shortName)}</td>
      <td>${_backendChip(x, {asChip:false})}</td>
      <td>${escapeHtml(x.index_backend) || '<span class="muted">—</span>'}</td>
      <td>${!x.index_key_mode || x.index_key_mode === 'n/a'
            ? '<span class="muted" title="key_mode does not apply to this index">—</span>'
            : escapeHtml(x.index_key_mode)}</td>
      <td>${
        x.mode === 'filcorr'
          ? (x.filcorr_kind === 'full'
              ? `<span class="chip" style="background:rgba(80,180,255,.15); color:#9ed2ff; border:1px solid rgba(80,180,255,.45)" `
                + `title="filcorr full band (fs≤0, ft≥0.5) — equivalent to standard Pearson on the full spectrum">full</span>`
              : x.filcorr_kind === 'band'
                ? `<span class="chip" style="background:rgba(255,180,90,.15); color:#ffc977; border:1px solid rgba(255,180,90,.45)" `
                  + `title="filcorr band-pass · fs=${x.filcorr_fs ?? '?'} ft=${x.filcorr_ft ?? '?'}">band</span>`
                : '<span class="muted">?</span>')
          : (!x.sketch_method || x.sketch_method === 'n/a'
              ? '<span class="muted" title="this mode does not use sketches">—</span>'
              : `<span class="chip info" title="sketch method (default for v2 corrtrack = random_projection)">${escapeHtml(x.sketch_method)}</span>`)
      }</td>
      <td class="num">${fmt(x.runtime, 2)}s</td>
      <td class="num" style="${spdStyle}" title="${x._baseline_label ? 'vs ' + escapeHtml(x._baseline_label) : ''}">${fmt(x.speedup_vs_baseline, 2)}×</td>
      <td class="num" style="${prStyle}">${fmt(x.precision, 3)}</td>
      <td class="num" style="${reStyle}">${fmt(x.recall, 3)}</td>
      <td class="num" style="${f1Style}">${fmt(x.f1, 3)}</td>
      <td class="num">${num(x.n_candidates)}</td>
      <td class="num">${num(x.n_tested)}</td>
      <td class="num">${fmt(x.candidate_ratio, 3)}</td>
      <td class="num">${num(x.n_correlated)}</td>
      <td class="num">${fmt(x.win_per_s, 1)}</td>
      <td class="num">${fmt(x.cand_per_s, 0)}</td>
      <td class="num" title="${x.tput_win_min==null?'':'min '+fmt(x.tput_win_min,1)+' / max '+fmt(x.tput_win_max,1)+' / overall '+fmt(x.tput_win_overall,1)+' win/s'}">${x.tput_win_median==null?'–':fmt(x.tput_win_median,1)}</td>
      <td class="num">${x.tput_win_min==null?'–':fmt(x.tput_win_min,1)}</td>
      <td class="num" title="${x.cpu_pct_median==null?'':'median CPU '+fmt(x.cpu_pct_median,0)+'% / max '+fmt(x.cpu_pct_max,0)+'%'}">${x.mem_peak_mb==null?'–':fmt(x.mem_peak_mb,0)}</td>
      <td class="num" title="${x.power_cores_median==null?'':'≈'+fmt(x.power_cores_median,2)+' cores busy (median)'}">${x.energy_cpu_seconds==null?'–':fmt(x.energy_cpu_seconds,0)}</td>
      <td class="num">${fmt(x.t_candidates, 2)}s</td>
      <td class="num">${fmt(x.t_validate, 2)}s</td>
      <td class="num" style="${_cellColor(x.spd_t_candidates, 'speedup')}" title="speedup phase candidats (select) vs baseline">${x.spd_t_candidates==null?'–':fmt(x.spd_t_candidates,2)+'×'}</td>
      <td class="num" style="${_cellColor(x.spd_t_validate, 'speedup')}" title="speedup phase validate vs baseline">${x.spd_t_validate==null?'–':fmt(x.spd_t_validate,2)+'×'}</td>
      <td class="num">${fmt(x.opt_time, 1)}s</td>
    </tr>`;
  }).join('');
  tb.querySelectorAll('tr[data-pipeline]').forEach(tr => {
    const path = tr.dataset.pipeline;
    const isDone = tr.dataset.done === '1';
    const cb = tr.querySelector('input.row-sel');
    if (cb && !cb.disabled) cb.addEventListener('click', e => {
      e.stopPropagation();
      if (cb.checked) ST.cmpSel.add(path); else ST.cmpSel.delete(path);
      tr.classList.toggle('sel-row', cb.checked);
      _selChanged();
    });
    const openBtn = tr.querySelector('.open-btn');
    if (openBtn && isDone) {
      openBtn.addEventListener('click', async e => {
        e.stopPropagation();
        await setActivePipeline(path);
        switchTab('pairs');
      });
    }
    tr.addEventListener('click', async (e) => {
      if (e.target.closest('.sel-cell') || e.target.closest('.actions')) return;
      if (!isDone) return;     // no preview/open for non-completed runs
      await previewPipeline(path);
    });
  });
}
// ---------- Client-side SVG charts ----------
const MODE_COLOR = {bf:'#ffb05a', corrtrack:'#7cd992', filcorr:'#5aa1ff', '?':'#888'};
const PHASE_KEYS = [
  ['t_read', 'read', '#5aa1ff'],
  ['t_windows', 'windows', '#88c1ff'],
  ['t_ingest', 'ingest', '#c08bff'],
  ['t_candidates', 'candidates', '#ffb05a'],
  ['t_validate', 'validate', '#ff7c7c'],
  ['t_monitor', 'monitor', '#5ddcdc'],
  ['t_evict', 'evict', '#a0a0a0'],
  ['opt_time', 'opt sweep (// makespan)', '#ffe066'],
];
function escapeSvg(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

// ----- Shared tooltip + helpers for D3-powered Home charts -----
function _chartTooltipEl() {
  let tip = document.getElementById('chart-tooltip-fl');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'chart-tooltip-fl';
    tip.className = 'd3-tooltip';
    tip.style.position = 'fixed';
    tip.style.pointerEvents = 'none';
    tip.style.display = 'none';
    tip.style.zIndex = '1000';
    tip.style.maxWidth = '320px';
    document.body.appendChild(tip);
  }
  return tip;
}
function _showTip(html, ev) {
  const tip = _chartTooltipEl();
  tip.innerHTML = html;
  tip.style.display = 'block';
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = Math.max(8, x) + 'px';
  tip.style.top = Math.max(8, y) + 'px';
}
function _hideTip() {
  const tip = document.getElementById('chart-tooltip-fl');
  if (tip) tip.style.display = 'none';
}
function _rowTipHtml(r, extra) {
  const f = (k, v, fmt) => v == null ? '' :
    `<div class="trow"><span class="nm">${escapeHtml(k)}</span><span class="val">${escapeHtml(fmt ? fmt(v) : v)}</span></div>`;
  const _eb = _effBaselineRt();
  const slowerMark = _eb && r.runtime > _eb &&
                     r.path !== ST.cmpData?.baseline_path ? ' <span style="color:#ff5c5c">⚠ slower than baseline</span>' : '';
  const backendTxt = r.is_sharded
    ? `${r.backend_base || r.backend} ⚡ sharded × ${r.n_shards || '?'} threads`
    : (r.backend || '—');
  return `<div class="thead">${escapeHtml(r.label)}${slowerMark}</div>` +
    f('mode', r.mode) +
    f('backend', backendTxt) +
    f('index', r.index_backend || '—') +
    f('key', r.index_key_mode || '—') +
    f('sketch', r.sketch_method || '—') +
    f('runtime', r.runtime, v => v.toFixed(2) + 's') +
    f('×baseline', r.speedup_vs_baseline, v => '×' + v.toFixed(2)) +
    f('n_correlated', r.n_correlated, v => v.toLocaleString()) +
    f('cand/tested', r.candidate_ratio, v => v.toFixed(3)) +
    (r.precision != null ? f('precision', r.precision, v => v.toFixed(3)) : '') +
    (r.recall    != null ? f('recall',    r.recall,    v => v.toFixed(3)) : '') +
    (r.f1        != null ? f('f1',        r.f1,        v => v.toFixed(3)) : '') +
    (extra || '') +
    '<div class="trow" style="color:var(--muted); font-style:italic; padding-top:3px; border-top:1px dashed var(--line); margin-top:3px;">click = preview pipeline · shift+click = toggle selection</div>';
}
function _chartRowAction(r, ev) {
  // shift+click → toggle selection ; plain click → preview the pipeline.
  if (ev.shiftKey) {
    if (ST.cmpSel.has(r.path)) ST.cmpSel.delete(r.path);
    else ST.cmpSel.add(r.path);
    renderCompare();
    if ($('#cmp-charts-sel')?.checked) loadCharts();
    _selChanged();
  } else {
    previewPipeline(r.path).catch(() => {});
  }
}
function _filteredRows() {
  if (!ST.cmpData) return [];
  const showBf = $('#cmp-bf').checked, showCt = $('#cmp-ct').checked,
        showFc = $('#cmp-fc').checked;
  const useSel = $('#cmp-charts-sel').checked && ST.cmpSel.size > 0;
  const F = ST.cmpFilters;
  const minPrec    = parseFloat($('#cmp-min-prec')?.value);
  const minRec     = parseFloat($('#cmp-min-recall')?.value);
  const minF1      = parseFloat($('#cmp-min-f1')?.value);
  const minSpeedup = parseFloat($('#cmp-min-speedup')?.value);
  const hideNoQ    = $('#cmp-hide-noq')?.checked;
  let rows = ST.cmpData.rows.filter(r => r.runtime > 0);
  if (useSel) {
    rows = rows.filter(r => ST.cmpSel.has(r.path));
  } else {
    rows = rows.filter(r => {
      if (r.mode === 'bf' && !showBf) return false;
      if (r.mode === 'corrtrack' && !showCt) return false;
      if (r.mode === 'filcorr' && !showFc) return false;
      const _mf = (sel, v) => {
        if (!sel.size) return true;
        const isNA = !v || v === 'n/a';
        if (isNA && sel.has('(n/a)')) return true;
        return sel.has(v || '');
      };
      if (!_mf(F.backend,        r.backend))         return false;
      if (!_mf(F.index_backend,  r.index_backend))   return false;
      if (!_mf(F.index_key_mode, r.index_key_mode))  return false;
      if (!_mf(F.sketch_method,  r.sketch_method))   return false;
      if (Number.isFinite(minSpeedup) &&
          (r.speedup_vs_baseline == null || r.speedup_vs_baseline < minSpeedup)) return false;
      return true;
    });
  }
  // merge precision/recall if loaded
  if (ST.cmpQuality) {
    rows = rows.map(r => ({...r, ...(ST.cmpQuality[r.path] || {})}));
  }
  // Quality threshold filters
  rows = rows.filter(r => {
    const hasQ = r.precision != null || r.recall != null;
    if (hideNoQ && !hasQ) return false;
    if (Number.isFinite(minPrec) && (r.precision == null || r.precision < minPrec)) return false;
    if (Number.isFinite(minRec)  && (r.recall    == null || r.recall    < minRec))  return false;
    if (Number.isFinite(minF1)   && (r.f1        == null || r.f1        < minF1))   return false;
    return true;
  });
  return _applyEffectiveMetrics(rows);
}

// _applyEffectiveMetrics — rewrite runtime + derived metrics so every chart
// reflects the user's phase selection. When the user hides `validate`, we
// recompute every row's runtime as the SUM of the remaining visible phases,
// then derive speedup / throughput from that smaller wall-time. The
// untouched original is preserved as `_orig_runtime` for the rare cell
// that wants to show it (mini-bars).
function _applyEffectiveMetrics(rows) {
  if (!rows || !rows.length) return rows;
  // Baseline is resolved against the FULL data (not the filtered subset —
  // a user-applied multi-select might exclude bf_python while charts still
  // want to scale against it).
  const blPath = ST.cmpData?.baseline_path;
  const allRows = ST.cmpData?.rows || rows;
  const baseRow = blPath ? allRows.find(r => r.path === blPath) : null;
  const baseEff = baseRow ? _visiblePhaseSum(baseRow) : 0;
  return rows.map(r => {
    const eff = _visiblePhaseSum(r);
    const newRt = eff > 0 ? eff : (r.runtime || 0);
    return {
      ...r,
      _orig_runtime: r.runtime,
      runtime: newRt,
      speedup_vs_baseline: (baseEff > 0 && newRt > 0)
        ? (baseEff / newRt)
        : r.speedup_vs_baseline,
      win_per_s: (newRt > 0 && r.n_windows)
        ? (r.n_windows / newRt)
        : r.win_per_s,
      cand_per_s: (newRt > 0 && r.n_candidates)
        ? (r.n_candidates / newRt)
        : r.cand_per_s,
    };
  });
}

// Effective baseline runtime — used by charts that pin scales / reference
// lines against the bf_python wall-time. Falls back to the server-provided
// baseline when the baseline row isn't in the row set yet.
function _effBaselineRt() {
  if (!ST.cmpData) return null;
  const base = ST.cmpData.rows?.find(r => r.path === ST.cmpData.baseline_path);
  if (!base) return ST.cmpData.baseline_runtime;
  return _visiblePhaseSum(base) || ST.cmpData.baseline_runtime;
}

function loadCharts() {
  if (!ST.cmpData) return;
  $('#charts').style.display = '';
  const bf = $('#cmp-bf').checked, ct = $('#cmp-ct').checked,
        fc = $('#cmp-fc').checked;
  const useSel = $('#cmp-charts-sel').checked && ST.cmpSel.size > 0;
  const chip = $('#charts-filter-chip');
  if (chip) {
    if (useSel) {
      chip.textContent = `filtered: ${ST.cmpSel.size} selected runs`;
      chip.className = 'chip ok';
    } else {
      const modes = [bf?'bf':null, ct?'corrtrack':null, fc?'filcorr':null]
                       .filter(x => x).join(' + ');
      chip.textContent = `all ${modes || 'modes'} runs`;
      chip.className = 'chip';
    }
  }
  const rows = _filteredRows();
  drawRuntime($('#ch-runtime'), rows);
  drawSpeedup($('#ch-speedup'), rows);
  drawHeatmap($('#ch-heatmap'), rows);
  drawSketchHeatmap($('#ch-sketch-heatmap'), rows);
  drawShardedGain($('#ch-shard-gain'), rows);
  drawPhases($('#ch-phases'), rows);
  drawTradeoff($('#ch-tradeoff'), rows);
  drawThroughput($('#ch-throughput'), rows);
  drawTputTimeline($('#ch-tput-timeline'), rows);
  drawResources(rows);
}

// ⑨ Resources — 4 D3 blocks (memory / energy / CPU / cores). Each block is a
// horizontal bar chart: bar = median, whiskers = min–max, computed per run
// from its resources.csv time series (res_series). Energy is a scalar total
// (no whiskers). Runs without resource data are skipped; if none have data a
// guidance message is shown.
const _RES_METRICS = [
  {host:'ch-res-mem',    key:'mem_mb',      scalar:'mem_peak_mb',         color:'#c08bff', fmt:v=>d3.format(',.0f')(v),  unit:'MB'},
  {host:'ch-res-energy', key:null,          scalar:'energy_cpu_seconds',  color:'#ffb05a', fmt:v=>d3.format(',.0f')(v),  unit:'core·s'},
  {host:'ch-res-cpu',    key:'cpu_pct',     scalar:'cpu_pct_median',      color:'#5aa1ff', fmt:v=>d3.format('.0f')(v)+'%', unit:'%'},
  {host:'ch-res-cores',  key:'power_cores', scalar:'power_cores_median',  color:'#5ddcdc', fmt:v=>d3.format('.2f')(v),   unit:'cores'},
];
function _resStats(r, key) {
  // (median, min, max) from res_series for `key`, else null.
  const s = (r.res_series || [])
    .map(p => p && p[key]).filter(v => v != null && isFinite(v)).sort((a,b)=>a-b);
  if (!s.length) return null;
  return {med: s[Math.floor(s.length/2)], min: s[0], max: s[s.length-1]};
}
function drawResources(rows) {
  if (typeof d3 === 'undefined') return;
  // Keep only runs that carry resource data (series OR a scalar).
  const usable = (rows || []).filter(r =>
    (r.res_series && r.res_series.length) ||
    r.mem_peak_mb != null || r.energy_cpu_seconds != null);
  const note = document.getElementById('ch-resources-note');
  if (!usable.length) {
    for (const mt of _RES_METRICS) {
      const h = document.getElementById(mt.host);
      if (h) _emptyHost(h, 'no resources.csv — re-run pipeline with real-time profiling');
    }
    if (note) note.textContent = '';
    return;
  }
  // Sort by peak memory desc, cap 20 for legibility (shared order across blocks).
  const peakMem = r => { const st=_resStats(r,'mem_mb'); return st?st.max:(+r.mem_peak_mb||0); };
  const ranked = usable.slice().sort((a,b)=>peakMem(b)-peakMem(a)).slice(0, 20);
  if (note) note.textContent = ranked.length < usable.length
    ? `top ${ranked.length} of ${usable.length} (by peak mem)` : `${ranked.length} run(s)`;

  for (const mt of _RES_METRICS) {
    const host = document.getElementById(mt.host);
    if (!host) continue;
    // Build per-run datum: {name, mode, med, min, max}
    const data = ranked.map(r => {
      if (mt.key) {
        const st = _resStats(r, mt.key);
        const fb = +r[mt.scalar];
        const med = st ? st.med : (isFinite(fb) ? fb : 0);
        const lo  = st ? st.min : med;
        const hi  = st ? st.max : med;
        return {r, name:(r.label||'').split('/').pop(), mode:r.mode, med, lo, hi};
      }
      // Energy: scalar only
      const v = +r[mt.scalar] || 0;
      return {r, name:(r.label||'').split('/').pop(), mode:r.mode, med:v, lo:v, hi:v, scalar:true};
    });
    _drawResBlock(host, data, mt);
  }
}
function _drawResBlock(host, data, mt) {
  host.innerHTML = '';
  const W = host.clientWidth || 520;
  const rowH = 20, gap = 3;
  const m = {top:6, right:64, bottom:22, left:150};
  const innerW = Math.max(120, W - m.left - m.right);
  const H = m.top + m.bottom + data.length * (rowH + gap);
  const maxV = Math.max(1e-9, d3.max(data, d => d.hi));
  const x = d3.scaleLinear().domain([0, maxV]).range([0, innerW]).nice();
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
  // x gridlines + axis
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  g.selectAll('line.xg').data(x.ticks(4)).join('line')
    .attr('x1', d=>x(d)).attr('x2', d=>x(d)).attr('y1', 0)
    .attr('y2', data.length*(rowH+gap))
    .attr('stroke','var(--line)').attr('stroke-dasharray','2 3').attr('opacity',.5);
  svg.append('g').attr('transform', `translate(${m.left},${m.top + data.length*(rowH+gap)})`)
    .selectAll('text').data(x.ticks(4)).join('text')
    .attr('x', d=>x(d)).attr('y', 14).attr('text-anchor','middle')
    .attr('font-size',9).attr('fill','var(--muted)').text(d=>mt.fmt(d));
  data.forEach((d, i) => {
    const y = i*(rowH+gap);
    const row = g.append('g').attr('transform', `translate(0,${y})`).style('cursor','pointer');
    // label
    svg.append('text').attr('x', m.left-6).attr('y', m.top + y + rowH*0.7)
      .attr('text-anchor','end').attr('font-size',9.5).attr('font-family','var(--mono)')
      .attr('fill', d.mode==='bf'?'#ffb05a':(d.mode==='filcorr'?'#5ddcdc':'var(--muted)'))
      .text(_shortLabel(d.name, 22));
    // median bar
    row.append('rect').attr('x',0).attr('y',2).attr('width', Math.max(0,x(d.med)))
      .attr('height', rowH-4).attr('fill', mt.color).attr('opacity',.9)
      .attr('stroke','rgba(255,255,255,.2)').attr('stroke-width',.5);
    // min–max whisker (skip for scalar-only energy). Bright high-contrast
    // colour: the whisker spans the median bar (coloured) AND extends into
    // the dark background for the max, so it must read on both. White with a
    // dark halo (paint-order via a wider dark line underneath) stays visible
    // on purple/blue/cyan bars and on the dark panel alike.
    if (!d.scalar && d.hi > d.lo) {
      const yc = rowH/2;
      const wh = (x1, x2, y1, y2) => {
        // dark halo underneath, then bright white on top
        row.append('line').attr('x1',x1).attr('x2',x2).attr('y1',y1).attr('y2',y2)
          .attr('stroke','rgba(0,0,0,0.6)').attr('stroke-width',3.2);
        row.append('line').attr('x1',x1).attr('x2',x2).attr('y1',y1).attr('y2',y2)
          .attr('stroke','#fff').attr('stroke-opacity',0.95).attr('stroke-width',1.4);
      };
      wh(x(d.lo), x(d.hi), yc, yc);                 // horizontal span
      wh(x(d.lo), x(d.lo), yc-4, yc+4);             // left cap (min)
      wh(x(d.hi), x(d.hi), yc-4, yc+4);             // right cap (max)
    }
    // value label
    svg.append('text').attr('x', m.left + Math.max(0,x(d.med)) + 5)
      .attr('y', m.top + y + rowH*0.7).attr('font-size',9).attr('font-family','var(--mono)')
      .attr('fill','var(--fg)').text(mt.fmt(d.med));
    // hover → tooltip with min/median/max
    const tip = () => `<div class="thead">${escapeHtml(d.name)}</div>`
      + `<div class="trow"><span class="nm">median</span><span class="val">${mt.fmt(d.med)} ${mt.unit}</span></div>`
      + (d.scalar ? '' :
         `<div class="trow"><span class="nm">min</span><span class="val">${mt.fmt(d.lo)} ${mt.unit}</span></div>`
       + `<div class="trow"><span class="nm">max</span><span class="val">${mt.fmt(d.hi)} ${mt.unit}</span></div>`);
    row.append('rect').attr('x',0).attr('y',0).attr('width',innerW).attr('height',rowH)
      .attr('fill','transparent')
      .on('mouseover', ev=>_showTip(tip(), ev))
      .on('mousemove', ev=>_showTip(tip(), ev))
      .on('mouseleave', _hideTip)
      .on('click', ev=>{ _hideTip(); _chartRowAction(d.r, ev); });
  });
}

function _emptyHost(host, msg) {
  host.innerHTML = `<div class="empty">${escapeSvg(msg || 'no runs to plot — toggle filters above')}</div>`;
}
function _shortLabel(label, max) {
  if (!label) return '';
  if (label.length <= (max || 40)) return label;
  return '…' + label.slice(-(max || 40) + 1);
}
function _legendHTML(items) {
  // items: [{color, label}]
  return items.map(it =>
    `<g><rect width="12" height="12" fill="${it.color}"/><text x="18" y="10" class="axis" font-size="11">${escapeSvg(it.label)}</text></g>`
  );
}

// ---------- Shared helpers for HTML-table ranking charts ----------
// Per-chart sort state. Each click on a sortable header mutates this and
// re-renders. Defaults below match each chart's "natural" view.
const RT_SORT = {
  'ch-runtime': { key: 'runtime', dir: 'asc' },   // fastest first
  'ch-speedup': { key: '_sp',     dir: 'desc' },  // biggest speedup first
  'ch-phases':  { key: 'runtime', dir: 'asc' },   // fastest first
};
const _RT_STRING_KEYS = new Set([
  'mode', 'backend', 'index_backend', 'index_key_mode', 'sketch_method', 'label',
]);
function _rtArrow(state, key) {
  if (!state || state.key !== key) {
    return ' <span class="sort-arrow">⇅</span>';
  }
  return ' <span class="sort-arrow">' + (state.dir === 'desc' ? '▼' : '▲') + '</span>';
}
function _rtHeader(chartId, key, label, opts) {
  const st = RT_SORT[chartId];
  const sorted = st && st.key === key;
  const cls = 'sortable' + (sorted ? ' sorted' : '');
  const style = opts && opts.style ? ` style="${opts.style}"` : '';
  return `<th class="${cls}" data-chart="${chartId}" data-sort-key="${key}"${style}>`
    + escapeHtml(label) + _rtArrow(st, key) + '</th>';
}
function _rtChipHeadSortable(chartId) {
  return (
    _rtHeader(chartId, 'mode', 'mode')
    + _rtHeader(chartId, 'backend', 'backend')
    + _rtHeader(chartId, 'index_backend', 'index')
    + _rtHeader(chartId, 'index_key_mode', 'key')
    + _rtHeader(chartId, 'sketch_method', 'sketch')
  );
}
function _rtSortRows(rows, key, dir) {
  const mul = dir === 'desc' ? -1 : 1;
  return rows.slice().sort((a, b) => {
    let va = a[key], vb = b[key];
    // Treat "n/a" sentinels as missing.
    if (va === 'n/a') va = null;
    if (vb === 'n/a') vb = null;
    const aMissing = va == null || (typeof va === 'number' && Number.isNaN(va));
    const bMissing = vb == null || (typeof vb === 'number' && Number.isNaN(vb));
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;   // missing values always last, regardless of dir
    if (bMissing) return -1;
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * mul;
    return String(va).localeCompare(String(vb), undefined,
                                     {numeric: true, sensitivity: 'base'}) * mul;
  });
}
// Header-click delegation: pull chart id + sort key out of `th[data-chart]`,
// update RT_SORT, redraw that chart. Same code services all three rankings.
document.addEventListener('click', e => {
  const th = e.target.closest && e.target.closest('th.sortable[data-chart]');
  if (!th || !ST.cmpData) return;
  const chart = th.dataset.chart, key = th.dataset.sortKey;
  const st = RT_SORT[chart] || (RT_SORT[chart] = {key, dir: 'desc'});
  if (st.key === key) {
    st.dir = st.dir === 'asc' ? 'desc' : 'asc';
  } else {
    st.key = key;
    // New key → choose direction sensibly: strings asc (a→z),
    // numbers desc (biggest first), except runtime-ish which asc (fastest).
    st.dir = _RT_STRING_KEYS.has(key)
      ? 'asc'
      : (key === 'runtime' || /^t_/.test(key) || key === 'opt_time' ? 'asc' : 'desc');
  }
  if (chart === 'ch-runtime') drawRuntime(document.getElementById(chart), _filteredRows());
  else if (chart === 'ch-speedup') drawSpeedup(document.getElementById(chart), _filteredRows());
  else if (chart === 'ch-phases')  drawPhases(document.getElementById(chart), _filteredRows());
});

function _rtTag(cls, txt) {
  return txt
    ? `<span class="rt-tag ${cls}">${escapeHtml(txt)}</span>`
    : `<span class="rt-tag muted">—</span>`;
}
// Backend chip with sharded marker. `sharded_vectorized` → "vectorized ⚡8".
// Used by both the compare table cell and the chart-ranking chip cells so
// the visual stays consistent.
function _backendChip(r, opts) {
  const asChip = !opts || opts.asChip !== false;  // default: chip styling
  const base = r.backend_base || r.backend;
  if (!base) return asChip
    ? '<span class="rt-tag muted">—</span>'
    : '<span class="muted">—</span>';
  const sharded = !!r.is_sharded;
  const marker = sharded
    ? `<span class="shard-mark" title="sharded · ${r.n_shards || '?'} shards">${r.n_shards || ''}</span>`
    : '';
  const tt = sharded
    ? ` title="raw: ${escapeHtml(r.backend)} · sharded sketch+insert phase, ${r.n_shards || '?'} threads"`
    : '';
  if (asChip) {
    const cls = sharded ? 'rt-tag sharded' : 'rt-tag';
    return `<span class="${cls}"${tt}>${escapeHtml(base)}${marker}</span>`;
  }
  return `<span${tt}>${escapeHtml(base)}${marker}</span>`;
}
function _rtChipCells(r) {
  const keyTxt = (!r.index_key_mode || r.index_key_mode === 'n/a') ? '' : r.index_key_mode;
  const sketchTxt = (!r.sketch_method || r.sketch_method === 'n/a') ? '' : r.sketch_method;
  return (
    `<td>${_rtTag('mode-' + r.mode, r.mode)}</td>`
    + `<td>${_backendChip(r)}</td>`
    + `<td>${_rtTag('', r.index_backend)}</td>`
    + `<td>${_rtTag('', keyTxt)}</td>`
    + `<td>${_rtTag('sketch', sketchTxt)}</td>`
  );
}
function _rtChipHead() {
  return '<th>mode</th><th>backend</th><th>index</th><th>key</th><th>sketch</th>';
}
// _rtTopN — keep the first `topN` rows of `sorted`. The cap is now read
// from a SHARED master selector (`#cmp-topn`) so ① runtime, ⑤ speedup,
// ② phases and ⑧ sharded gain stay in sync. `noteId` identifies the per-
// chart "(X hidden of N)" status span so each card keeps its own hint.
function _rtTopN(sorted, noteId, baselinePath) {
  const masterSel = document.getElementById('cmp-topn');
  const topN = masterSel ? parseInt(masterSel.value, 10) : 20;
  let kept = sorted;
  if (topN > 0 && sorted.length > topN) {
    kept = sorted.slice(0, topN);
    if (baselinePath && !kept.some(r => r.path === baselinePath)) {
      const bf = sorted.find(r => r.path === baselinePath);
      if (bf) kept = kept.concat([bf]);
    }
  }
  const note = document.getElementById(noteId + '-note');
  if (note) {
    const hidden = sorted.length - kept.length;
    note.textContent = hidden > 0
      ? `(${hidden} hidden of ${sorted.length})`
      : `(${sorted.length} runs)`;
  }
  return kept;
}
function _rtBindRows(host, kept, extraFn) {
  // `extraFn(r) → htmlString` is appended to the tooltip body. Used by the
  // phase-breakdown chart to add per-phase timings + percentages.
  host.querySelectorAll('tr[data-pipeline]').forEach(tr => {
    tr.style.cursor = 'pointer';
    const find = () => kept.find(x => x.path === tr.dataset.pipeline);
    const tipHtml = r => _rowTipHtml(r, extraFn ? extraFn(r) : '');
    tr.addEventListener('mouseover', ev => {
      const r = find(); if (r) _showTip(tipHtml(r), ev);
    });
    tr.addEventListener('mousemove', ev => {
      const r = find(); if (r) _showTip(tipHtml(r), ev);
    });
    tr.addEventListener('mouseleave', _hideTip);
    tr.addEventListener('click', ev => {
      _hideTip();
      const r = find(); if (r) _chartRowAction(r, ev);
    });
  });
}
// Build the per-phase timing+percentage block shown in the tooltip of the
// Phase breakdown chart. Lists each non-zero phase in chronological order
// (read → windows → ingest → candidates → validate → monitor → evict →
// opt sweep), then any unattributed remainder.
function _phaseBreakdownTipHtml(r) {
  // `r.runtime` may have been rewritten by _applyEffectiveMetrics() to the
  // sum of visible phases (which can INCLUDE opt_time when opt_sweep is
  // toggled visible). For the breakdown percentages we want the ORIGINAL
  // wall-clock runtime that EXCLUDES opt_time, otherwise the "other /
  // unattrib." remainder absorbs opt_time and gets shown twice.
  const total = +r._orig_runtime || +r.runtime || 0;
  const off = _phasesOff();
  let used = 0;          // runtime phases counted toward 'used vs total'
  const lines = [];
  const fmt = (name, v, color, isOff, pctBase) => {
    // % is relative to `pctBase` (runtime by default; for opt_time we use
    // runtime too but flag the bar as ">100%" of runtime since it's not
    // part of the displayed total).
    const pct = pctBase > 0 ? (v / pctBase * 100) : 0;
    const sw = color
      ? `<span style="display:inline-block; width:9px; height:9px; `
        + `background:${color}; border-radius:2px; margin-right:5px; `
        + `vertical-align:-1px;${isOff ? 'outline:1px solid var(--muted); background:transparent' : ''}"></span>`
      : '';
    const nameStyle = isOff ? ' style="opacity:.45; text-decoration:line-through"' : '';
    return (
      `<div class="trow">`
      + `<span class="nm"${nameStyle}>${sw}${escapeHtml(name)}</span>`
      + `<span class="val" style="font-family:var(--mono)">`
        + `${v.toFixed(2)}s `
        + `<span class="muted" style="font-size:10px">(${pct.toFixed(1)}%)</span>`
      + `</span></div>`
    );
  };
  // Runtime phases first (% of runtime)
  for (const [key, name, color] of _RUNTIME_PHASE_KEYS) {
    const v = +r[key] || 0;
    if (v <= 0) continue;
    used += v;
    lines.push(fmt(name, v, color, off.has(key), total));
  }
  const rest = total - used;
  if (rest > 0.01) lines.push(fmt('other / unattrib.', rest, '#444', false, total));
  // opt sweep last — makespan of the grid over the workers (opt_eff); % vs runtime.
  const optV = _phaseVal(r, 'opt_time');
  if (optV > 0) {
    const optMeta = PHASE_KEYS.find(([k]) => k === 'opt_time');
    const optColor = optMeta ? optMeta[2] : '#ffe066';
    const serial = +r.opt_serial || optV;
    const W = Math.max(1, +r.workers || 0);
    const kind = r.opt_real ? 'measured' : `${W}thr projected`;
    const optName = (serial > optV + 0.01)
      ? `opt sweep (// ${kind} ${optV.toFixed(1)}s · serial ${serial.toFixed(1)}s)`
      : 'opt sweep (serial)';
    lines.push(fmt(optName, optV, optColor, off.has('opt_time'), total));
  }
  if (!lines.length) return '';
  const head =
    '<div class="trow" style="padding-top:6px; border-top:1px dashed var(--line); '
    + 'margin-top:4px; color:var(--muted); font-size:10px; text-transform:uppercase; '
    + 'letter-spacing:.4px; font-weight:600;">phase breakdown</div>';
  const tot =
    `<div class="trow" style="margin-top:3px; padding-top:3px; `
    + `border-top:1px dotted rgba(255,255,255,.1); font-weight:600">`
    + `<span class="nm">total</span>`
    + `<span class="val" style="font-family:var(--mono)">${total.toFixed(2)}s</span>`
    + `</div>`;
  return head + lines.join('') + tot;
}

function drawRuntime(host, rows) {
  if (!rows.length) { _emptyHost(host); return; }
  // Always compute "fastest" from the natural runtime order (so the ★ still
  // marks the literal fastest run even if the user sorted by mode etc.).
  const byRuntime = rows.slice().sort((a,b) => a.runtime - b.runtime);
  const fastest = byRuntime[0].runtime;
  const st = RT_SORT['ch-runtime'];
  const sorted = _rtSortRows(rows, st.key, st.dir);
  const baselineRt = _effBaselineRt();
  const baselinePath = ST.cmpData?.baseline_path;
  const baselineLabel = ST.cmpData?.baseline_label || '';
  const kept = _rtTopN(sorted, 'ch-runtime-topn', baselinePath);
  const xMax = kept.reduce((m, r) => Math.max(m, r.runtime),
                            baselineRt || 0) || 1;
  const baselinePct = baselineRt ? (baselineRt / xMax * 100) : null;

  const html = [];
  // Caption: tells the user what the red vertical line means.
  if (baselineRt) {
    html.push(
      '<div class="rt-rank-caption">',
      `<span>${kept.length} shown · sorted by runtime</span>`,
      '<span class="muted">·</span>',
      `<span class="rt-baseline-pill">┃ baseline = ${baselineRt.toFixed(2)}s `
        + `(${escapeHtml(baselineLabel || 'bf_python')})</span>`,
      '</div>'
    );
  }
  html.push(
    '<table class="rt-rank">',
    '<thead><tr>',
    '<th></th>',  // rank #
    _rtChipHeadSortable('ch-runtime'),
    '<th style="width:55%">runtime</th>',
    _rtHeader('ch-runtime', 'runtime', 'time', {style: 'text-align:right'}),
    '</tr></thead>',
    '<tbody>'
  );
  kept.forEach((r, i) => {
    const isBase = r.path === baselinePath;
    const isFast = r.runtime === fastest && !isBase;
    const slower = baselineRt && r.runtime > baselineRt && !isBase;
    const cls = [
      isBase ? 'baseline' : '',
      isFast ? 'fastest' : '',
      slower ? 'slower-than-baseline' : '',
      ST.cmpSel.has(r.path) ? 'sel-row' : '',
    ].filter(Boolean).join(' ');
    const pct = (r.runtime / xMax * 100).toFixed(2);
    const barColor = slower ? '#ff5c5c'
                            : (MODE_COLOR[r.mode] || '#888');
    const baselineLine = (baselinePct != null && baselinePct <= 100)
      ? `<div class="rt-bar-baseline" style="left:${baselinePct.toFixed(2)}%" title="baseline ${baselineRt.toFixed(2)}s · ${escapeHtml(baselineLabel)}"></div>`
      : '';
    const marks = (isFast ? ' ★' : '') + (slower ? ' ⚠' : '');
    html.push(
      `<tr class="${cls}" data-pipeline="${escapeHtml(r.path)}" `
        + `title="${escapeHtml(r.label)} — click to preview, ⌘/Ctrl-click to toggle selection">`
      + `<td class="rt-col-rank">${isBase ? 'BL' : '#' + (i+1)}</td>`
      + _rtChipCells(r)
      + `<td class="rt-col-bar">`
        + `<div class="rt-bar-host">`
          + `<div class="rt-bar-fill" style="width:${pct}%; background:${barColor}"></div>`
          + baselineLine
        + `</div>`
      + `</td>`
      + `<td class="rt-col-time">${r.runtime.toFixed(2)}s${marks}</td>`
      + `</tr>`
    );
  });
  html.push('</tbody></table>');
  host.innerHTML = html.join('');
  _rtBindRows(host, kept);
}
// Re-render all rank-style charts when the shared top-N selector changes
// (single master `#cmp-topn` ⇒ keeps ① / ⑤ / ② / ⑧ in sync).
document.addEventListener('change', e => {
  if (!e.target || !ST.cmpData) return;
  if (e.target.id !== 'cmp-topn') return;
  const rows = _filteredRows();
  const h1 = document.getElementById('ch-runtime');     if (h1) drawRuntime(h1, rows);
  const h2 = document.getElementById('ch-speedup');     if (h2) drawSpeedup(h2, rows);
  const h3 = document.getElementById('ch-phases');      if (h3) drawPhases(h3, rows);
  const h4 = document.getElementById('ch-shard-gain');  if (h4) drawShardedGain(h4, rows);
});

// Re-render the real-time throughput timeline when its own controls change
// (metric / log-Y / X-axis), without touching the other charts.
document.addEventListener('change', e => {
  if (!e.target || !ST.cmpData) return;
  if (!['tput-metric', 'tput-log', 'tput-xaxis'].includes(e.target.id)) return;
  const h = document.getElementById('ch-tput-timeline');
  if (h) drawTputTimeline(h, _filteredRows());
});

function drawSpeedup(host, rows) {
  if (!rows.length) { _emptyHost(host); return; }
  // Pick the baseline (bf_python preferred, then slowest bf, then global slowest).
  const bfs = rows.filter(r => r.mode === 'bf');
  const baseline = bfs.find(r => r.backend === 'python') ||
                   (bfs.length ? bfs.reduce((a,b) => b.runtime>a.runtime?b:a) : null) ||
                   rows.reduce((a,b) => b.runtime>a.runtime?b:a);
  const baseRt = baseline.runtime;
  const baselinePath = baseline.path;
  // We always compute _sp; sort key may be _sp or any other column.
  const enriched = rows.map(r => ({...r, _sp: baseRt / r.runtime}));
  const best = enriched.reduce((m, r) => Math.max(m, r._sp), 0);
  const st = RT_SORT['ch-speedup'];
  const sorted = _rtSortRows(enriched, st.key, st.dir);
  const kept = _rtTopN(sorted, 'ch-speedup-topn', baselinePath);
  // Log-scaled bar width: render 0..100% as log(_sp / minDomain) / log(maxDomain / minDomain)
  const minS = Math.max(0.05, Math.min(...kept.map(r => r._sp)) * 0.9);
  const maxS = Math.max(...kept.map(r => r._sp)) * 1.1;
  const logRange = Math.log(maxS / minS);
  const toPct = sp => Math.max(0, Math.log(Math.max(minS, sp) / minS) / logRange * 100);
  const onePct = (1 >= minS && 1 <= maxS) ? toPct(1) : null;

  const html = [
    '<div class="rt-rank-caption">',
    `<span>${kept.length} shown · baseline = ${escapeHtml(baseline.label)} (${baseRt.toFixed(2)}s)</span>`,
    '<span class="muted">·</span>',
    '<span class="rt-baseline-pill">┃ ×1 reference</span>',
    '<span class="muted">· bar length is log-scaled (×0.1 → ×100 spans the row)</span>',
    '</div>',
    '<table class="rt-rank">',
    '<thead><tr>',
    '<th></th>',
    _rtChipHeadSortable('ch-speedup'),
    '<th style="width:55%">speedup (log)</th>',
    _rtHeader('ch-speedup', '_sp', '×baseline', {style: 'text-align:right'}),
    '</tr></thead>',
    '<tbody>',
  ];
  kept.forEach((r, i) => {
    const isBase = r.path === baselinePath;
    const isFast = r._sp === best && !isBase;
    const slower = r._sp < 1 && !isBase;
    const cls = [
      isBase ? 'baseline' : '',
      isFast ? 'fastest' : '',
      slower ? 'slower-than-baseline' : '',
      ST.cmpSel.has(r.path) ? 'sel-row' : '',
    ].filter(Boolean).join(' ');
    const pct = toPct(r._sp).toFixed(2);
    const barColor = slower ? '#ff5c5c' : (MODE_COLOR[r.mode] || '#888');
    const oneLine = (onePct != null)
      ? `<div class="rt-bar-baseline" style="left:${onePct.toFixed(2)}%" title="×1 (baseline)"></div>`
      : '';
    const marks = (isFast ? ' ★' : '') + (slower ? ' ⚠' : '');
    html.push(
      `<tr class="${cls}" data-pipeline="${escapeHtml(r.path)}" title="${escapeHtml(r.label)}">`
      + `<td class="rt-col-rank">${isBase ? 'BL' : '#' + (i+1)}</td>`
      + _rtChipCells(r)
      + `<td class="rt-col-bar"><div class="rt-bar-host">`
        + `<div class="rt-bar-fill" style="width:${pct}%; background:${barColor}"></div>`
        + oneLine
      + `</div></td>`
      + `<td class="rt-col-time">×${r._sp.toFixed(2)}${marks}</td>`
      + `</tr>`
    );
  });
  html.push('</tbody></table>');
  host.innerHTML = html.join('');
  _rtBindRows(host, kept);
}

function drawPhases(host, rows) {
  if (!rows.length) { _emptyHost(host); return; }
  const st = RT_SORT['ch-phases'];
  const sorted = _rtSortRows(rows, st.key, st.dir);
  const baselineRt = _effBaselineRt();
  const baselinePath = ST.cmpData?.baseline_path;
  const baselineLabel = ST.cmpData?.baseline_label || '';
  const kept = _rtTopN(sorted, 'ch-phases-topn', baselinePath);
  // Scale = max VISIBLE phase sum (so toggles really shrink/grow bars,
  // and re-enabling `opt_time` extends the scale to accommodate it).
  const xMax = kept.reduce((m, r) => Math.max(m, _visiblePhaseSum(r)),
                            baselineRt || 0) || 1;
  const baselinePct = baselineRt ? (baselineRt / xMax * 100) : null;

  // Legend: each swatch is a click-to-sort handle. Hidden phases are still
  // displayed dimmed + struck-through so the user can see the current
  // filter state, but hide/show is controlled exclusively from the sidebar
  // (single source of truth for ST.phasesOff). Includes `opt_time` so it
  // can be sorted-by when enabled in the sidebar.
  const offSet = _phasesOff();
  const legendHtml = PHASE_KEYS.map(([key, name, color]) => {
    const active = st.key === key ? ' active' : '';
    const arrow = st.key === key ? (st.dir === 'desc' ? ' ▼' : ' ▲') : '';
    const isOff = offSet.has(key);
    const style = isOff
      ? ' style="opacity:.38; text-decoration:line-through"' : '';
    return (
      `<span class="lg${active}" data-phase-sort="${key}"${style} `
      + `title="click → sort by ${escapeHtml(name)} time${isOff ? ' · hidden — toggle in sidebar' : ''}">`
      + `<span class="sw" style="background:${color}${isOff ? '; outline:1px solid var(--muted); background:transparent' : ''}"></span>`
      + `${escapeHtml(name)}${arrow}</span>`
    );
  }).join('');
  // Phase toggle row FIRST (above the caption) — primary control for this
  // panel should be the first thing the user sees, then the contextual
  // baseline / "N shown" hint.
  const html = [
    `<div class="rt-phase-legend">${legendHtml}</div>`,
    '<div class="rt-rank-caption">',
    `<span>${kept.length} shown · click column header or phase swatch to sort</span>`,
  ];
  if (baselineRt) {
    html.push('<span class="muted">·</span>',
      `<span class="rt-baseline-pill">┃ baseline = ${baselineRt.toFixed(2)}s `
        + `(${escapeHtml(baselineLabel || 'bf_python')})</span>`);
  }
  html.push('</div>',
    '<table class="rt-rank">',
    '<thead><tr>',
    '<th></th>',
    _rtChipHeadSortable('ch-phases'),
    '<th style="width:55%">phase breakdown</th>',
    _rtHeader('ch-phases', 'runtime', 'total', {style: 'text-align:right'}),
    '</tr></thead>',
    '<tbody>',
  );
  kept.forEach((r, i) => {
    const isBase = r.path === baselinePath;
    const slower = baselineRt && r.runtime > baselineRt && !isBase;
    const cls = [
      isBase ? 'baseline' : '',
      slower ? 'slower-than-baseline' : '',
      ST.cmpSel.has(r.path) ? 'sel-row' : '',
    ].filter(Boolean).join(' ');
    // Build phase segments. Width is in % of xMax so all rows share scale.
    // Order: runtime phases (read → ... → evict) · unattributed remainder ·
    //        opt_time (last, only when toggled visible).
    // Visibility governed by ST.phasesOff, toggled from the legend.
    let leftPct = 0;
    const segs = [];
    let runtimePhaseSum = 0;
    const offSetRow = _phasesOff();
    for (const [key, name, color] of _RUNTIME_PHASE_KEYS) {
      if (offSetRow.has(key)) continue;
      const v = +r[key] || 0;
      if (v <= 0) continue;
      const w = v / xMax * 100;
      segs.push(
        `<div class="rt-bar-seg" style="left:${leftPct.toFixed(3)}%; `
          + `width:${w.toFixed(3)}%; background:${color}" `
          + `title="${escapeHtml(name)}: ${v.toFixed(2)}s"></div>`
      );
      leftPct += w;
      runtimePhaseSum += v;
    }
    // Unattributed remainder within runtime — only meaningful while all
    // runtime phases are visible (otherwise it would silently swallow the
    // hidden phases' time). Use the ORIGINAL runtime (`_orig_runtime`)
    // which excludes opt_time, otherwise we'd compute rest = opt_time and
    // render a phantom gray segment exactly the size of opt_sweep.
    const allRuntimeVisible = _RUNTIME_PHASE_KEYS.every(([k]) => !offSetRow.has(k));
    const wallRt = +r._orig_runtime || +r.runtime || 0;
    const rest = allRuntimeVisible ? Math.max(0, wallRt - runtimePhaseSum) : 0;
    if (rest > 0.01) {
      const w = rest / xMax * 100;
      segs.push(
        `<div class="rt-bar-seg" style="left:${leftPct.toFixed(3)}%; `
          + `width:${w.toFixed(3)}%; background:#444; opacity:.5" `
          + `title="other / unattributed: ${rest.toFixed(2)}s"></div>`
      );
      leftPct += w;
    }
    // opt_time last (opt-in via legend toggle). The MAKESPAN of the optim grid
    // over the run's workers (opt_eff) is displayed — small when the optim
    // parallelizes — with the serial sum + thread count recalled in the tooltip.
    if (!offSetRow.has('opt_time')) {
      const v = _phaseVal(r, 'opt_time');
      if (v > 0.01) {
        const optMeta = PHASE_KEYS.find(([k]) => k === 'opt_time');
        const color = optMeta ? optMeta[2] : '#ffe066';
        const w = v / xMax * 100;
        const serial = +r.opt_serial || v;
        const W = Math.max(1, +r.workers || 0);
        const kind = r.opt_real ? 'measured' : `${W} threads, projected`;
        const tip = (serial > v + 0.01)
          ? `opt sweep: ${v.toFixed(2)}s (// ${kind}) · ${serial.toFixed(2)}s serial · excluded from runtime`
          : `opt sweep: ${v.toFixed(2)}s (serial, 1 thread) · excluded from runtime`;
        segs.push(
          `<div class="rt-bar-seg" style="left:${leftPct.toFixed(3)}%; `
            + `width:${w.toFixed(3)}%; background:${color}" `
            + `title="${tip}"></div>`
        );
        leftPct += w;
      }
    }
    const baselineLine = (baselinePct != null && baselinePct <= 100)
      ? `<div class="rt-bar-baseline" style="left:${baselinePct.toFixed(2)}%" title="baseline ${baselineRt.toFixed(2)}s"></div>`
      : '';
    html.push(
      `<tr class="${cls}" data-pipeline="${escapeHtml(r.path)}" title="${escapeHtml(r.label)}">`
      + `<td class="rt-col-rank">${isBase ? 'BL' : '#' + (i+1)}</td>`
      + _rtChipCells(r)
      + `<td class="rt-col-bar"><div class="rt-bar-host">`
        + segs.join('')
        + baselineLine
      + `</div></td>`
      + `<td class="rt-col-time">${r.runtime.toFixed(2)}s${slower?' ⚠':''}</td>`
      + `</tr>`
    );
  });
  html.push('</tbody></table>');
  host.innerHTML = html.join('');
  _rtBindRows(host, kept, _phaseBreakdownTipHtml);
  // Per-phase legend: click sorts by that phase's time (re-click flips
  // direction). Hide/show lives in the sidebar.
  host.querySelectorAll('[data-phase-sort]').forEach(sw => {
    sw.style.cursor = 'pointer';
    sw.addEventListener('click', ev => {
      ev.stopPropagation();
      const key = sw.dataset.phaseSort;
      const cur = RT_SORT['ch-phases'];
      if (cur.key === key) {
        cur.dir = cur.dir === 'asc' ? 'desc' : 'asc';
      } else {
        cur.key = key; cur.dir = 'desc';   // largest phase-time first
      }
      drawPhases(document.getElementById('ch-phases'), _filteredRows());
    });
  });
}

// ⑧ Sharded gain — pairs each sharded run with its non-sharded twin of the
// same (backend_base, index_backend, index_key_mode, sketch_method) and shows
// the speedup factor that sharding provided on top of the base backend.
// bf_python is NOT a row here (it's not shardable) but is kept as a reference
// per row via the ×bf column.
// Inline mini stacked bar pair (base on top, sharded below), shared scale so
// the length difference between the two bars literally is the time saved by
// sharding. Color palette = PHASE_KEYS to match chart ② Phase breakdown.
// The `opt_time` (optimizer sweep) phase is excluded so the bar length matches
// the `runtime` value shown elsewhere (RUNTIME / ×BASELINE / SHARD GAIN columns
// all use `runtime` which does NOT include `opt_time`).
// Each segment's tooltip carries: phase name, absolute seconds, % of THAT
// run's total runtime, and (when both base & sharded are present) the per-
// phase delta + speedup.
const _RUNTIME_PHASE_KEYS = PHASE_KEYS.filter(([k]) => k !== 'opt_time');

// ---------- Phase visibility toggles (shared by sharded panel + chart ②) ----
// `ST.phasesOff` is a Set of phase keys ('t_read', 't_validate', ...) that the
// user has hidden via the legend swatches. Hidden phases are skipped in BOTH
// the sharded mini-bars and the chart ② Phase breakdown so the page stays
// internally consistent.
function _phasesOff() {
  // First-time init: NOTHING hidden by default — opt_sweep is now visible
  // out of the box so the user immediately sees the full picture (runtime
  // + optimization sweep) on charts ② and ⑧. Hide it via the sidebar if
  // you want to focus on the pure algorithm wall-time.
  if (!ST.phasesOff) ST.phasesOff = new Set();
  return ST.phasesOff;
}
// Value of a phase for a row. For `opt_time` we return `opt_eff` = MAKESPAN of
// the optim grid over `workers` threads (LPT) rather than the cumulated sum: the
// yellow bar equals the serial sum for a base run (1 thread) and shrinks to the
// busiest thread's time for a parallel/sharded run — showing that the optim
// parallelizes. `opt_serial` stays the serial sum.
function _phaseVal(r, key) {
  if (key === 'opt_time') return +r.opt_eff || +r.opt_serial || +r.opt_time || 0;
  return +r[key] || 0;
}
// Sum of visible phase times for a row (respects ST.phasesOff).
function _visiblePhaseSum(r) {
  if (!r) return 0;
  const off = _phasesOff();
  let s = 0;
  for (const [key] of PHASE_KEYS) {
    if (off.has(key)) continue;
    s += _phaseVal(r, key);
  }
  return s;
}
function _togglePhase(key) {
  const s = _phasesOff();
  if (s.has(key)) s.delete(key); else s.add(key);
  _rerenderPhaseConsumers();
}
function _resetPhases() {
  _phasesOff().clear();
  _rerenderPhaseConsumers();
}
function _hideAllPhases() {
  // Hide every phase at once (mark all PHASE_KEYS as off).
  const s = _phasesOff();
  for (const [k] of PHASE_KEYS) s.add(k);
  _rerenderPhaseConsumers();
}
function _rerenderPhaseConsumers() {
  // Phase visibility affects effective runtime, which feeds every chart
  // AND the compare table — so re-render everything that consumes the
  // recomputed metrics in a single sweep.
  if (ST.cmpData) {
    if (typeof renderCompare === 'function') renderCompare();
    if (typeof loadCharts === 'function')    loadCharts();
  }
  renderPhaseFilters();
}

// Build the sidebar phase-filter chip list. Single source of truth for the
// hide/show phase state — Chart ②, sharded mini-bars and tooltips all read
// from ST.phasesOff which this UI mutates.
function renderPhaseFilters() {
  const host = document.getElementById('phase-filters');
  if (!host) return;
  const off = _phasesOff();
  const rows = PHASE_KEYS.map(([key, name, color]) => {
    const isOff = off.has(key);
    return (
      `<span class="ph-toggle" data-phase-toggle="${key}" `
      + `style="cursor:pointer; user-select:none; display:flex; align-items:center; `
      + `gap:6px; padding:3px 6px; border-radius:4px; `
      + (isOff
          ? `opacity:.4; text-decoration:line-through; background:rgba(255,255,255,0.02)`
          : `background:rgba(255,255,255,0.04)`)
      + `" title="${isOff ? 'show' : 'hide'} ${escapeHtml(name)} (applies to chart ② and sharded panel)">`
      + `<span style="display:inline-block; width:12px; height:12px; border-radius:2px; `
      + `background:${color}; ${isOff ? 'outline:1px solid var(--muted); background:transparent' : ''}"></span>`
      + `<span>${escapeHtml(name)}</span></span>`
    );
  });
  host.innerHTML = rows.join('');
  // Enable/disable the bulk buttons to reflect state: "Show all" only useful
  // when something is hidden; "Hide all" only when something is visible.
  const anyOff = PHASE_KEYS.some(([k]) => off.has(k));
  const allOff = PHASE_KEYS.every(([k]) => off.has(k));
  const showBtn = document.getElementById('phase-filters-reset-btn');
  const hideBtn = document.getElementById('phase-filters-hide-btn');
  if (showBtn) { showBtn.disabled = !anyOff; showBtn.style.opacity = anyOff ? '' : '.4'; }
  if (hideBtn) { hideBtn.disabled = allOff; hideBtn.style.opacity = allOff ? '.4' : ''; }
  host.querySelectorAll('[data-phase-toggle]').forEach(el => {
    el.addEventListener('click', ev => {
      ev.stopPropagation();
      _togglePhase(el.dataset.phaseToggle);
    });
  });
}

function _phasesMiniBar(base, sharded, maxRt) {
  // Bigger dims so segments are actually readable.
  // Width is fixed; scale (`maxRt`) covers the LONGEST visible-phase-sum of
  // the pair so both bars fit and remain directly comparable.
  const W = 300, rowH = 14, gap = 4;
  const H = rowH * 2 + gap;
  const labelW = 80;       // right-side runtime label
  const sc = v => W * v / (maxRt || 1);
  function row(yy, r, tag) {
    if (!r) {
      return `<g transform="translate(0,${yy})">
        <rect width="${W}" height="${rowH}" fill="rgba(255,255,255,0.04)" rx="2"/>
        <text x="4" y="${rowH-3}" font-size="9.5" fill="var(--muted)">no ${tag} twin</text></g>`;
    }
    let left = 0, segs = '', visSum = 0;
    const off = _phasesOff();
    for (const [key, , color] of PHASE_KEYS) {
      if (off.has(key)) continue;
      const v = _phaseVal(r, key);
      if (v <= 0) continue;
      const w = sc(v);
      segs += `<rect x="${left.toFixed(1)}" y="0" width="${w.toFixed(1)}" `
            + `height="${rowH}" fill="${color}"/>`;
      left += w;
      visSum += v;
    }
    // Right label = the VISIBLE-phase total (so it matches the bar length
    // and the table's base/sharded columns when the user toggles phases).
    return `<g transform="translate(0,${yy})">${segs}
      <text x="${(left+4).toFixed(1)}" y="${rowH-3}" font-size="10" font-family="var(--mono)"
            fill="var(--muted)">${tag[0].toUpperCase()} ${visSum.toFixed(1)}s</text></g>`;
  }
  return `<svg viewBox="0 0 ${W + labelW} ${H}" width="${W + labelW}" height="${H}"
              xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:block">
    ${row(0,           base,    'base')}
    ${row(rowH + gap,  sharded, 'sharded')}
  </svg>`;
}

// Custom rich tooltip for the sharded panel — shows the FULL phase breakdown
// for BOTH base & sharded side-by-side with color swatches, percentages and
// per-phase delta. Replaces the per-segment <title> tooltips with one
// HTML-rendered tooltip per row (much more legible).
function _shardedPhaseTipHtml(base, sharded) {
  const off = _phasesOff();
  const visBase = _visiblePhaseSum(base);
  const visSh   = _visiblePhaseSum(sharded);
  const head =
    '<div class="trow" style="padding-top:6px; border-top:1px dashed var(--line); '
    + 'margin-top:4px; color:var(--muted); font-size:10px; text-transform:uppercase; '
    + 'letter-spacing:.4px; font-weight:600;">phase breakdown · base vs sharded</div>';
  const colHeader =
    '<div class="trow" style="font-size:10px; color:var(--muted)">'
    + '<span class="nm"></span>'
    + '<span class="val" style="display:flex; gap:8px; justify-content:flex-end; min-width:170px">'
    +   '<span style="width:65px; text-align:right">base</span>'
    +   '<span style="width:65px; text-align:right">sharded</span>'
    +   '<span style="width:42px; text-align:right">Δ</span>'
    + '</span></div>';
  const rowFmt = (key, name, color) => {
    const isOff = off.has(key);
    const vb = base    ? _phaseVal(base, key)    : 0;
    const vs = sharded ? _phaseVal(sharded, key) : 0;
    if (vb <= 0 && vs <= 0) return '';
    const pb = (visBase > 0 && vb > 0) ? (vb / visBase * 100) : 0;
    const ps = (visSh   > 0 && vs > 0) ? (vs / visSh   * 100) : 0;
    const delta = vb - vs;     // positive ⇒ sharded is faster
    const deltaStr = (vb > 0 && vs > 0)
      ? (delta >= 0 ? `−${delta.toFixed(2)}s` : `+${Math.abs(delta).toFixed(2)}s`)
      : '—';
    const deltaColor = (vb > 0 && vs > 0)
      ? (delta >= 0 ? 'var(--ok)' : '#ff8a8a')
      : 'var(--muted)';
    const fmtCell = (v, p) => v > 0
      ? `<span style="width:65px; text-align:right; font-family:var(--mono)">`
        + `${v.toFixed(2)}s `
        + `<span class="muted" style="font-size:9.5px">(${p.toFixed(0)}%)</span></span>`
      : `<span style="width:65px; text-align:right; color:var(--muted)">—</span>`;
    const sw =
      `<span style="display:inline-block; width:10px; height:10px; `
      + `background:${color}; border-radius:2px; margin-right:6px; vertical-align:-1px; `
      + (isOff ? 'outline:1px solid var(--muted); background:transparent' : '')
      + `"></span>`;
    const nameStyle = isOff ? ' style="opacity:.45; text-decoration:line-through"' : '';
    return (
      `<div class="trow">`
      + `<span class="nm"${nameStyle}>${sw}${escapeHtml(name)}</span>`
      + `<span class="val" style="display:flex; gap:8px; justify-content:flex-end; min-width:170px">`
        + fmtCell(vb, pb)
        + fmtCell(vs, ps)
        + `<span style="width:42px; text-align:right; font-family:var(--mono); color:${deltaColor}">${deltaStr}</span>`
      + `</span></div>`
    );
  };
  const lines = PHASE_KEYS.map(([k, n, c]) => rowFmt(k, n, c)).filter(Boolean);
  if (!lines.length) return '';
  // Totals row (visible-phase sums) + gain
  const gainNum = (visBase > 0 && visSh > 0) ? visBase / visSh : null;
  const gainStr = gainNum != null ? `×${gainNum.toFixed(2)}` : '—';
  const gainColor = gainNum != null
    ? (gainNum >= 1 ? 'var(--ok)' : '#ff8a8a')
    : 'var(--muted)';
  const tot =
    `<div class="trow" style="margin-top:3px; padding-top:3px; `
    + `border-top:1px dotted rgba(255,255,255,.15); font-weight:600">`
    + `<span class="nm">visible total</span>`
    + `<span class="val" style="display:flex; gap:8px; justify-content:flex-end; min-width:170px">`
      + `<span style="width:65px; text-align:right; font-family:var(--mono)">${visBase.toFixed(2)}s</span>`
      + `<span style="width:65px; text-align:right; font-family:var(--mono)">${visSh.toFixed(2)}s</span>`
      + `<span style="width:42px; text-align:right; font-family:var(--mono); color:${gainColor}">${gainStr}</span>`
    + `</span></div>`;
  const hint =
    '<div class="trow" style="color:var(--muted); font-size:10px; padding-top:2px">'
    + '<span class="nm" style="font-style:italic">tip</span>'
    + '<span class="val muted" style="font-style:italic">% = share of visible total · '
    + 'click legend → hide/show a phase</span></div>';
  return head + colHeader + lines.join('') + tot + hint;
}

function _shardSig(r) {
  const be = r.backend_base || r.backend || '';
  const idx = r.index_backend || '';
  const key = (r.index_key_mode === 'n/a') ? '' : (r.index_key_mode || '');
  const sk  = (r.sketch_method  === 'n/a') ? '' : (r.sketch_method  || '');
  return [be, idx, key, sk].join('|');
}
function drawShardedGain(host, rows) {
  if (!host) return;
  if (!rows.length) { _emptyHost(host, 'no rows'); return; }
  const baselineRt = _effBaselineRt();
  const baselineLabel = ST.cmpData?.baseline_label || '';

  // Group by signature. Each group keeps the best (fastest) base run and the
  // list of sharded variants (potentially several with different n_shards).
  const groups = new Map();
  for (const r of rows) {
    if (r.mode !== 'corrtrack') continue;     // only corrtrack runs are shardable
    if (!(r.runtime > 0)) continue;
    const sig = _shardSig(r);
    let g = groups.get(sig);
    if (!g) { g = {sig, base: null, sharded: [], ref: r}; groups.set(sig, g); }
    if (r.is_sharded) g.sharded.push(r);
    else if (!g.base || r.runtime < g.base.runtime) g.base = r;
  }
  // Keep only groups with at least one sharded variant.
  let pairs = [];
  for (const g of groups.values()) {
    for (const sh of g.sharded) {
      pairs.push({base: g.base, sharded: sh, sig: g.sig, ref: g.base || sh});
    }
  }
  const note = document.getElementById('ch-shard-note');
  if (!pairs.length) {
    // Build a small diagnostic showing what backends WERE seen, so the user
    // can tell "no sharded runs exist" from "detection failed".
    const seen = new Map();   // backend → count
    for (const r of rows) {
      const k = r.backend || '(empty)';
      seen.set(k, (seen.get(k) || 0) + 1);
    }
    const breakdown = Array.from(seen.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([k, n]) => `<code style="color:var(--accent)">${escapeHtml(k)}</code> ×${n}`)
      .join(', ') || '(none)';
    if (note) note.textContent = `0 sharded variants · ${seen.size} backend(s) seen`;
    host.innerHTML =
      '<div class="empty" style="text-align:left; padding:14px 18px">'
      + '<div style="font-weight:600; margin-bottom:6px">No sharded run detected.</div>'
      + '<div style="margin-bottom:8px">A run is treated as sharded when:</div>'
      + '<ul style="margin:0 0 10px 18px; padding:0; font-size:11.5px; line-height:1.5">'
      + '<li><code>param.backend</code> in <code>summary.csv</code> starts with <code>sharded_</code></li>'
      + '<li><i>or</i> the pipeline JSON declares <code>"backend": "sharded_…"</code> for the run</li>'
      + '<li><i>or</i> the run directory name contains <code>"sharded"</code></li>'
      + '</ul>'
      + `<div>Backends currently visible in the table: ${breakdown}</div>`
      + '<div class="muted" style="margin-top:8px; font-size:11px">'
      + 'Try <code>⇅ sync now</code> in the remote banner, '
      + 'or check <code>grep param.backend &lt;run&gt;/summary.csv</code> on the server.'
      + '</div></div>';
    return;
  }
  if (note) {
    const orphan = pairs.filter(p => !p.base).length;
    note.textContent = `${pairs.length} sharded variant${pairs.length>1?'s':''}`
      + (orphan ? ` · ${orphan} without a base twin (sharded-only)` : '');
  }
  // Pre-compute visible-phase sums (reflect the user's legend toggles) and
  // a visible-phase gain. These drive the displayed times AND the gain
  // column, so toggling phases really changes the comparison.
  pairs.forEach(p => {
    p.visBase = p.base    ? _visiblePhaseSum(p.base)    : 0;
    p.visSh   =             _visiblePhaseSum(p.sharded);
    p.vGain = (p.visBase > 0 && p.visSh > 0) ? (p.visBase / p.visSh) : null;
    // Fall back to runtime-based gain only when visible sums are zero (eg
    // user hid every single phase) — keeps the column non-empty.
    p.gain = p.vGain != null
      ? p.vGain
      : ((p.base && p.sharded.runtime > 0) ? (p.base.runtime / p.sharded.runtime) : null);
  });

  // Sort state (persisted across re-renders + legend toggles).
  ST.shardSort = ST.shardSort || {key:'gain', dir:'desc'};
  const sort = ST.shardSort;
  const sortVal = (p, k) => {
    // Phase-keyed sort: `phase:<phaseKey>` ⇒ sort by the SHARDED run's time
    // on that phase (most meaningful focus in this panel).
    if (k && k.startsWith('phase:')) {
      const pk = k.slice(6);
      return +(p.sharded ? p.sharded[pk] || 0 : 0);
    }
    switch (k) {
      case 'backend':  return p.sig.split('|')[0] || '';
      case 'index':    return p.sig.split('|')[1] || '';
      case 'key':      return p.sig.split('|')[2] || '';
      case 'sketch':   return p.sig.split('|')[3] || '';
      case 'base':     return p.visBase || 0;
      case 'sharded':  return p.visSh || 0;
      case 'gain':     return p.gain != null ? p.gain : -Infinity;
      case 'bf':       return baselineRt ? (baselineRt / p.sharded.runtime) : 0;
      default:         return 0;
    }
  };
  const dirMul = sort.dir === 'asc' ? 1 : -1;
  pairs.sort((a, b) => {
    const va = sortVal(a, sort.key), vb = sortVal(b, sort.key);
    if (typeof va === 'string') return dirMul * va.localeCompare(vb);
    return dirMul * ((va) - (vb));
  });
  // Best-gain badge is computed from the FULL set (before top-N clamp), so
  // the ★ stays attached to the actual global champion regardless of cap.
  const bestGain = pairs.reduce((m, p) => Math.max(m, p.gain || 0), 0);
  // Apply the shared master top-N cap (same selector that limits the other
  // rank-style charts). The "(X hidden of N)" hint goes in the card header.
  const totalPairs = pairs.length;
  const masterSel = document.getElementById('cmp-topn');
  const topN = masterSel ? parseInt(masterSel.value, 10) : 20;
  if (topN > 0 && pairs.length > topN) pairs = pairs.slice(0, topN);
  const topNote = document.getElementById('ch-shard-topn-note');
  if (topNote) {
    const hidden = totalPairs - pairs.length;
    topNote.textContent = hidden > 0
      ? `(${hidden} hidden of ${totalPairs})`
      : `(${totalPairs} variants)`;
  }

  // Sortable header helper — emits a <th> with arrow indicator + data-sort.
  const sortTh = (key, label, opts) => {
    const isActive = sort.key === key;
    const arrow = isActive ? (sort.dir === 'desc' ? ' ▼' : ' ▲') : '';
    const style = (opts && opts.style) || '';
    const title = (opts && opts.title) || `click to sort by ${label}`;
    return `<th data-shard-sort="${key}" style="cursor:pointer; user-select:none;${style}" `
      + `title="${escapeHtml(title)}">${label}<span style="color:var(--accent)">${arrow}</span></th>`;
  };

  const html = [];

  // Phase sort handles — pure sort affordance (click = sort by phase).
  // Hide/show is delegated to the sidebar's Phases section (single source
  // of truth), so this row no longer reacts to shift+click. Hidden phases
  // are still rendered dimmed here for situational awareness.
  const offSet = _phasesOff();
  html.push('<div class="rt-rank-caption phase-toggle-row" style="margin:0 0 8px 0; gap:8px; flex-wrap:wrap">');
  html.push('<span class="muted" style="font-size:10.5px">sort by phase (click):</span>');
  for (const [key, name, color] of PHASE_KEYS) {
    const isOff = offSet.has(key);
    const sortKey = 'phase:' + key;
    const isActiveSort = sort.key === sortKey;
    const arrow = isActiveSort ? (sort.dir === 'desc' ? ' ▼' : ' ▲') : '';
    html.push(
      `<span class="ph-sort" data-phase-sort="${sortKey}" `
      + `style="cursor:pointer; user-select:none; display:inline-flex; align-items:center; gap:5px; font-size:11px; `
      + `padding:2px 7px; border-radius:4px; `
      + `border:1px solid ${isActiveSort ? 'var(--accent)' : 'transparent'}; `
      + (isOff
          ? `opacity:.38; text-decoration:line-through; background:rgba(255,255,255,0.03)`
          : `opacity:1; background:rgba(255,255,255,0.04)`)
      + `" title="sort table by ${escapeHtml(name)} (sharded value)${isOff ? ' · hidden — toggle in sidebar' : ''}">`
      + `<span style="width:11px; height:11px; background:${color}; border-radius:2px; `
        + (isOff ? `outline:1px solid var(--muted); background:transparent` : '')
        + `"></span>`
      + `<span class="muted">${escapeHtml(name)}</span>`
      + (arrow ? `<span style="color:var(--accent)">${arrow}</span>` : '')
      + `</span>`
    );
  }
  html.push('</div>');

  // Caption AFTER the legend now (legend is the primary control, so it
  // belongs above; the caption is contextual annotation).
  html.push('<div class="rt-rank-caption">',
    `<span>${totalPairs} sharded variant${totalPairs>1?'s':''} found · `
      + 'comparing each to its base (same backend without `sharded_` prefix)</span>');
  if (baselineRt) {
    html.push('<span class="muted">·</span>',
      `<span class="rt-baseline-pill">┃ bf_python = ${baselineRt.toFixed(2)}s`
      + ` (${escapeHtml(baselineLabel || 'bf_python')})</span>`);
  }
  html.push('</div>');

  html.push('<table class="rt-rank">',
    '<thead><tr>',
    '<th></th>',
    sortTh('backend', 'backend'),
    sortTh('index',   'index'),
    sortTh('key',     'key'),
    sortTh('sketch',  'sketch'),
    sortTh('base',    'base',    {style:'text-align:right', title:'visible-phase time of the base run'}),
    sortTh('sharded', 'sharded', {style:'text-align:right', title:'visible-phase time of the sharded run'}),
    '<th title="per-phase breakdown · top bar = base · bottom bar = sharded · same scale · hover for full %" style="text-align:left; min-width:400px; width:400px">phases (base / sharded)</th>',
    sortTh('gain', 'shard gain', {style:'text-align:right', title:'visible-phase speedup of the sharded variant vs its non-sharded twin'}),
    baselineRt ? sortTh('bf', '×bf', {style:'text-align:right', title:'speedup of the SHARDED run vs bf_python (global reference)'}) : '',
    '</tr></thead><tbody>');

  pairs.forEach((p, i) => {
    const sh = p.sharded, base = p.base;
    const isBest = p.gain && p.gain === bestGain;
    const cls = isBest ? 'fastest' : '';
    // Show the VISIBLE-phase sum as the primary number (so toggling phases
    // changes what's displayed) and the raw runtime as a muted secondary.
    // When all phases are visible AND opt_time is hidden, visBase ≈ runtime
    // so we suppress the secondary to avoid noise.
    const _fmtTime = (vis, rt) => {
      const main = `${vis.toFixed(2)}s`;
      const showRt = Math.abs(vis - rt) > 0.01;
      const rtPart = showRt
        ? ` <span class="muted" style="font-size:10px" title="full runtime">rt ${rt.toFixed(2)}s</span>`
        : '';
      return main + rtPart;
    };
    const baseCell = base
      ? _fmtTime(p.visBase, base.runtime)
        + (baselineRt
            ? ` <span class="muted" style="font-size:10px">×${(baselineRt/base.runtime).toFixed(1)}</span>`
            : '')
      : '<span class="muted">— no twin —</span>';
    const shCell = _fmtTime(p.visSh, sh.runtime) + ' '
      + `<span class="shard-mark" title="sharded · ${sh.n_shards||'?'} threads">${sh.n_shards||''}</span>`
      + (baselineRt
          ? `<br><span class="muted" style="font-size:10px">×${(baselineRt/sh.runtime).toFixed(1)} vs bf</span>`
          : '');
    const gainCell = p.gain != null
      ? `<span style="font-weight:600; color:${p.gain >= 1 ? 'var(--ok)' : '#ff5c5c'}" `
        + `title="visible-phase gain · base ${p.visBase.toFixed(2)}s / sharded ${p.visSh.toFixed(2)}s">`
        + `×${p.gain.toFixed(2)}${isBest ? ' ★' : ''}${p.gain < 1 ? ' ⚠' : ''}</span>`
      : '<span class="muted">—</span>';
    const bfCell = baselineRt
      ? `<td style="text-align:right; font-family:var(--mono)">×${(baselineRt/sh.runtime).toFixed(1)}</td>`
      : '';
    const sigParts = p.sig.split('|');  // [be, idx, key, sk]
    const beTxt = sigParts[0];
    const idxTxt = sigParts[1];
    const keyTxt = sigParts[2];
    const skTxt = sigParts[3];
    // Phase breakdown: shared scale across base & sharded for THIS pair
    // (computed from the visible phases only so user toggles really change
    // the bar shape). Length difference = time difference of visible phases.
    const phaseMax = Math.max(
      _visiblePhaseSum(base),
      _visiblePhaseSum(sh)
    ) || 1;
    const phasesCell = _phasesMiniBar(base, sh, phaseMax);
    html.push(
      `<tr class="${cls}" data-pipeline="${escapeHtml(sh.path)}" `
        + `data-pipeline-base="${escapeHtml(base ? base.path : '')}" `
        + `title="click → preview sharded run · base = ${escapeHtml(base ? base.label : 'none')}">`
      + `<td class="rt-col-rank">#${i+1}</td>`
      + `<td>${_rtTag('', beTxt)}</td>`
      + `<td>${_rtTag('', idxTxt)}</td>`
      + `<td>${_rtTag('', keyTxt)}</td>`
      + `<td>${_rtTag('sketch', skTxt)}</td>`
      + `<td style="text-align:right; font-family:var(--mono)">${baseCell}</td>`
      + `<td style="text-align:right; font-family:var(--mono)">${shCell}</td>`
      + `<td style="padding:2px 6px">${phasesCell}</td>`
      + `<td style="text-align:right; font-family:var(--mono)">${gainCell}</td>`
      + bfCell
      + `</tr>`
    );
  });
  html.push('</tbody></table>');
  host.innerHTML = html.join('');
  // Sort handles: click on a phase chip → sort the sharded table by
  // THAT phase's sharded value (re-click flips direction). Hide/show is
  // controlled exclusively from the sidebar Phases section.
  host.querySelectorAll('[data-phase-sort]').forEach(el => {
    el.addEventListener('click', ev => {
      ev.stopPropagation();
      const k = el.dataset.phaseSort;
      const cur = ST.shardSort;
      if (cur.key === k) {
        cur.dir = cur.dir === 'asc' ? 'desc' : 'asc';
      } else {
        cur.key = k; cur.dir = 'desc';   // largest phase-time first
      }
      drawShardedGain(host, _filteredRows());
    });
  });
  const resetBtn = host.querySelector('[data-phase-reset]');
  if (resetBtn) resetBtn.addEventListener('click', ev => {
    ev.stopPropagation();
    _resetPhases();
  });
  // Sortable headers — click toggles direction on the active column, or
  // switches sort to that column (default desc for numeric, asc for text).
  host.querySelectorAll('[data-shard-sort]').forEach(th => {
    th.addEventListener('click', ev => {
      ev.stopPropagation();
      const k = th.dataset.shardSort;
      const cur = ST.shardSort;
      if (cur.key === k) {
        cur.dir = cur.dir === 'asc' ? 'desc' : 'asc';
      } else {
        cur.key = k;
        // text columns sort asc by default, numeric columns desc
        cur.dir = ['backend','index','key','sketch'].includes(k) ? 'asc' : 'desc';
      }
      drawShardedGain(host, _filteredRows());
    });
  });
  // Click on the row → preview the SHARDED run. Cmd/Ctrl-click → toggle in
  // selection. The base run is identified by `data-pipeline-base` for future
  // extensions (e.g. "open both side-by-side").
  host.querySelectorAll('tr[data-pipeline]').forEach(tr => {
    tr.style.cursor = 'pointer';
    const pair = pairs.find(p => p.sharded.path === tr.dataset.pipeline);
    if (!pair) return;
    const sh = pair.sharded;
    const base = pair.base;
    const tipHtml = () => _rowTipHtml(sh, _shardedPhaseTipHtml(base, sh));
    tr.addEventListener('mouseover', ev => _showTip(tipHtml(), ev));
    tr.addEventListener('mousemove', ev => _showTip(tipHtml(), ev));
    tr.addEventListener('mouseleave', _hideTip);
    tr.addEventListener('click', ev => {
      _hideTip();
      _chartRowAction(sh, ev);
    });
  });
}

// All numeric metrics that can be plotted on the scatter axes
const TRO_METRICS = [
  {k:'speedup_vs_baseline', label:'speedup vs baseline', defaultLog:false},
  {k:'runtime',             label:'runtime (s)',         defaultLog:false},
  {k:'precision',           label:'precision',           defaultLog:false},
  {k:'recall',              label:'recall',              defaultLog:false},
  {k:'f1',                  label:'F1',                  defaultLog:false},
  {k:'n_correlated',        label:'# correlated pairs',  defaultLog:true},
  {k:'n_candidates',        label:'# candidates',        defaultLog:true},
  {k:'n_tested',            label:'# tested',            defaultLog:true},
  {k:'candidate_ratio',     label:'cand/tested ratio',   defaultLog:false},
  {k:'win_per_s',           label:'throughput (win/s)',  defaultLog:true},
  {k:'cand_per_s',          label:'throughput (cand/s)', defaultLog:true},
];
// Default axes for the first render — chosen to give the canonical
// speed-vs-quality trade-off chart used in CorrTrack papers.
const TRO_DEFAULTS = {
  'tro-x': 'speedup_vs_baseline',  // X = how much faster than baseline (log)
  'tro-y': 'f1',                    // Y = quality (F1 captures both prec+rec)
};
function _populateTroSelects() {
  for (const id of ['tro-x', 'tro-y']) {
    const sel = $('#' + id);
    if (!sel || sel.dataset.populated) continue;
    sel.dataset.populated = '1';
    const def = TRO_DEFAULTS[id];
    // We mark `selected` in the HTML so the very first paint shows the
    // intended axis. Setting sel.value AFTER innerHTML is fragile because
    // a select with N options auto-picks option 0 — so `!sel.value` is
    // always false and the old "set if empty" check never fired, leaving
    // BOTH axes on the first metric (the bug we just fixed).
    sel.innerHTML = TRO_METRICS.map(m =>
      `<option value="${m.k}"${m.k === def ? ' selected' : ''}>`
      + `${escapeHtml(m.label)}</option>`).join('');
    // Also set the matching log toggle based on the default metric's hint
    // (speedup wants log; F1 wants linear). Only on first populate.
    const meta = TRO_METRICS.find(m => m.k === def);
    const logEl = $('#' + id + '-log');
    if (logEl && meta) logEl.checked = !!meta.defaultLog;
  }
}
// When the user picks the same metric on both axes, the scatter becomes a
// degenerate diagonal — auto-rotate the other axis to a sensible different
// metric so the chart stays meaningful.
function _troEnsureDistinct(changedId) {
  const sx = $('#tro-x'), sy = $('#tro-y');
  if (!sx || !sy || sx.value !== sy.value) return false;
  const otherId = changedId === 'tro-x' ? 'tro-y' : 'tro-x';
  const otherSel = $('#' + otherId);
  // Pick the original default for the OTHER axis if it differs, else fall
  // back to the first metric in the list that isn't the new value.
  const def = TRO_DEFAULTS[otherId];
  const fallback = (def !== sx.value)
    ? def
    : (TRO_METRICS.find(m => m.k !== sx.value)?.k || def);
  otherSel.value = fallback;
  const meta = TRO_METRICS.find(m => m.k === fallback);
  const logEl = $('#' + otherId + '-log');
  if (logEl && meta) logEl.checked = !!meta.defaultLog;
  return true;
}
function drawTradeoff(host, rows) {
  if (typeof d3 === 'undefined') { _emptyHost(host, 'd3 not loaded'); return; }
  if (!rows.length) { _emptyHost(host); return; }
  _populateTroSelects();
  const xKey = $('#tro-x')?.value || 'speedup_vs_baseline';
  const yKey = $('#tro-y')?.value || 'recall';
  const xLog = !!$('#tro-x-log')?.checked;
  const yLog = !!$('#tro-y-log')?.checked;
  const xMeta = TRO_METRICS.find(m => m.k === xKey) || TRO_METRICS[0];
  const yMeta = TRO_METRICS.find(m => m.k === yKey) || TRO_METRICS[1];

  // Chart-local mode visibility (legend-click toggle). Survives re-renders.
  ST.tradeoffOff = ST.tradeoffOff || new Set();

  // Build all points; drop rows where either dimension is null/NaN
  const allPts = rows.map(r => {
    const xv = r[xKey], yv = r[yKey];
    if (xv == null || yv == null || !Number.isFinite(xv) || !Number.isFinite(yv)) return null;
    if (xLog && xv <= 0) return null;
    if (yLog && yv <= 0) return null;
    return {x: xv, y: yv, r};
  }).filter(p => p);
  const allModes = [...new Set(allPts.map(p => p.r.mode))].sort();
  // Apply legend filter
  const pts = allPts.filter(p => !ST.tradeoffOff.has(p.r.mode));
  if (!allPts.length) {
    _emptyHost(host, `no rows have both "${xMeta.label}" and "${yMeta.label}" populated (try unchecking log or compute precision/recall first)`);
    return;
  }
  if (!pts.length) {
    _emptyHost(host, `all modes hidden via legend — click a legend entry on the right to show again`);
    return;
  }
  // Pareto only meaningful when "higher x and higher y is better" — handle both
  // for the user's chosen pair, default heuristic: bigger is better for all
  // metrics EXCEPT runtime + candidate_ratio (where lower is better).
  const xBetter = (xKey === 'runtime' || xKey === 'candidate_ratio') ? -1 : 1;
  const yBetter = (yKey === 'runtime' || yKey === 'candidate_ratio') ? -1 : 1;
  const pareto = pts.filter((p, i) =>
    !pts.some((q, j) =>
      j !== i &&
      ((xBetter * (q.x - p.x) >= 0 && yBetter * (q.y - p.y) > 0) ||
       (xBetter * (q.x - p.x) > 0  && yBetter * (q.y - p.y) >= 0))));
  const paretoSet = new Set(pareto.map(p => p.r.path));
  host.innerHTML = '';
  const W = host.clientWidth || 1100, H = 460;
  const m = {top:40, right:200, bottom:60, left:80};
  const innerW = Math.max(200, W - m.left - m.right);
  const innerH = H - m.top - m.bottom;
  // Axes — log if asked AND all values positive. Otherwise linear.
  const xMin = d3.min(pts, p => p.x), xMax = d3.max(pts, p => p.x);
  const yMin = d3.min(pts, p => p.y), yMax = d3.max(pts, p => p.y);
  const xPad = xLog ? 0 : (xMax - xMin) * 0.05 || 1;
  const yPad = yLog ? 0 : (yMax - yMin) * 0.05 || 1;
  const x = (xLog ? d3.scaleLog() : d3.scaleLinear())
    .domain(xLog ? [Math.max(1e-6, xMin) * 0.85, xMax * 1.15]
                 : [xMin - xPad, xMax + xPad])
    .range([0, innerW])
    .nice();
  const y = (yLog ? d3.scaleLog() : d3.scaleLinear())
    .domain(yLog ? [Math.max(1e-6, yMin) * 0.85, yMax * 1.15]
                 : [yMin - yPad, yMax + yPad])
    .range([innerH, 0])
    .nice();
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
  svg.append('text').attr('x', W/2).attr('y', 22).attr('text-anchor','middle')
    .attr('font-size', 13).attr('font-weight', 600).attr('fill','var(--fg)')
    .text(`${yMeta.label} ↔ ${xMeta.label} · ${pts.length} runs · hover for details`);
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  // Smart tick formatter — compact for big numbers, fixed for ratios in [0,1]
  const fmtTick = (meta) => (d) => {
    if (meta.k === 'speedup_vs_baseline') return '×' + (d % 1 === 0 ? d : d.toFixed(1));
    if (meta.k === 'runtime') return d.toFixed(d < 10 ? 1 : 0) + 's';
    if (['precision','recall','f1','candidate_ratio'].includes(meta.k)) return d.toFixed(2);
    if (d >= 1e6) return (d/1e6).toFixed(1) + 'M';
    if (d >= 1e3) return (d/1e3).toFixed(1) + 'k';
    return d.toFixed(d < 10 ? 1 : 0);
  };
  // grid
  g.selectAll('.x-grid').data(x.ticks(6)).join('line')
    .attr('class','x-grid').attr('x1', d=>x(d)).attr('x2', d=>x(d))
    .attr('y1',0).attr('y2', innerH)
    .attr('stroke','var(--line)').attr('stroke-dasharray','2 3').attr('opacity',0.5);
  g.selectAll('.y-grid').data(y.ticks(6)).join('line')
    .attr('class','y-grid').attr('x1',0).attr('x2', innerW)
    .attr('y1', d=>y(d)).attr('y2', d=>y(d))
    .attr('stroke','var(--line)').attr('stroke-dasharray','2 3').attr('opacity',0.5);
  g.append('g').attr('transform', `translate(0,${innerH})`).attr('class','d3-axis')
    .call(d3.axisBottom(x).ticks(6).tickFormat(fmtTick(xMeta)).tickSizeOuter(0));
  g.append('g').attr('class','d3-axis')
    .call(d3.axisLeft(y).ticks(6).tickFormat(fmtTick(yMeta)).tickSizeOuter(0));
  // Reference line at "perfect" if Y is a ratio in [0,1]
  if (['precision','recall','f1'].includes(yKey) && y.domain()[0] <= 1 && y.domain()[1] >= 1) {
    g.append('line').attr('x1',0).attr('x2', innerW).attr('y1', y(1)).attr('y2', y(1))
      .attr('stroke','#7cd992').attr('stroke-dasharray','4 3').attr('opacity',0.6);
  }
  // Pareto frontier path
  if (pareto.length >= 2) {
    const sorted = pareto.slice().sort((a,b) => xBetter*(a.x-b.x));
    g.append('path').datum(sorted)
      .attr('fill','none').attr('stroke','#888').attr('opacity',0.4)
      .attr('d', d3.line().x(p=>x(p.x)).y(p=>y(p.y)));
  }
  // Hover label layer — shown only for the focused point. Solves overlap.
  const focusLabel = g.append('text').attr('class','focus-label')
    .attr('pointer-events','none').attr('font-family','monospace').attr('font-size',11)
    .attr('font-weight',600).attr('fill','var(--fg)')
    .attr('stroke','#1a1d24').attr('stroke-width',3).attr('paint-order','stroke')
    .attr('display','none');
  // Pts — only HIT detection circle (larger, invisible) on top to ease hover
  g.append('g').selectAll('circle.pt').data(pts).join('circle')
    .attr('class','pt').attr('cx', p=>x(p.x)).attr('cy', p=>y(p.y))
    .attr('r', p => paretoSet.has(p.r.path) ? 7 : 4)
    .attr('fill', p => MODE_COLOR[p.r.mode] || '#888')
    .attr('stroke', p => ST.cmpSel.has(p.r.path) ? 'var(--accent)'
                       : paretoSet.has(p.r.path) ? '#222' : '#fff')
    .attr('stroke-width', p => ST.cmpSel.has(p.r.path) ? 2.5
                              : paretoSet.has(p.r.path) ? 1.5 : 0.5)
    .attr('opacity', p => paretoSet.has(p.r.path) ? 0.95 : 0.6)
    .attr('cursor','pointer')
    .on('mouseover', function(ev, p) {
      d3.select(this).attr('r', paretoSet.has(p.r.path) ? 10 : 7);
      focusLabel.attr('display', null)
        .attr('x', x(p.x) + 10).attr('y', y(p.y) - 8)
        .text(p.r.label.split('/').pop());
      const extra = paretoSet.has(p.r.path)
        ? '<div class="trow"><span class="nm">on Pareto frontier</span><span class="val">★</span></div>'
        : '';
      _showTip(_rowTipHtml(p.r, extra), ev);
    })
    .on('mousemove', (ev, p) => _showTip(_rowTipHtml(p.r), ev))
    .on('mouseleave', function(ev, p) {
      d3.select(this).attr('r', paretoSet.has(p.r.path) ? 7 : 4);
      focusLabel.attr('display','none');
      _hideTip();
    })
    .on('click', (ev, p) => { _hideTip(); _chartRowAction(p.r, ev); });
  // axis labels — reflect the user's chosen dimensions + scale
  svg.append('text').attr('x', m.left + innerW/2).attr('y', H-22)
    .attr('text-anchor','middle').attr('font-size',11).attr('fill','var(--muted)')
    .text(`${xMeta.label}${xLog ? ' — log scale' : ''}`);
  svg.append('text').attr('transform', `translate(${m.left-54}, ${m.top + innerH/2}) rotate(-90)`)
    .attr('text-anchor','middle').attr('font-size',11).attr('fill','var(--muted)')
    .text(`${yMeta.label}${yLog ? ' — log scale' : ''}`);
  // Legend — clickable: tap a mode to hide/show its points in this chart only.
  const leg = svg.append('g').attr('transform', `translate(${m.left + innerW + 20}, ${m.top})`);
  allModes.forEach((mm, i) => {
    const off = ST.tradeoffOff.has(mm);
    const ge = leg.append('g')
      .attr('transform', `translate(0, ${i*22})`)
      .attr('cursor', 'pointer')
      .attr('opacity', off ? 0.32 : 1)
      .on('click', function() {
        if (ST.tradeoffOff.has(mm)) ST.tradeoffOff.delete(mm);
        else ST.tradeoffOff.add(mm);
        drawTradeoff(host, rows);
      })
      .on('mouseover', function() { d3.select(this).attr('opacity', off ? 0.55 : 0.8); })
      .on('mouseleave', function() { d3.select(this).attr('opacity', off ? 0.32 : 1); });
    // larger clickable hitbox behind the icon+text
    ge.append('rect').attr('x', -4).attr('y', -8).attr('width', 90).attr('height', 20)
      .attr('rx', 3).attr('fill', 'transparent');
    ge.append('circle').attr('r', 7).attr('fill', MODE_COLOR[mm] || '#888')
      .attr('stroke', '#fff').attr('stroke-width', off ? 0 : 1.5);
    if (off) {
      // Strikethrough line over hidden modes
      ge.append('line').attr('x1', -7).attr('y1', 0).attr('x2', 80).attr('y2', 0)
        .attr('stroke', '#888').attr('stroke-width', 1.2);
    }
    ge.append('text').attr('x', 14).attr('y', 4).attr('font-size', 11.5)
      .attr('font-weight', off ? 400 : 600)
      .attr('fill', off ? 'var(--muted)' : 'var(--fg)')
      .attr('text-decoration', off ? 'line-through' : 'none')
      .text(mm);
    // tiny count of points in that mode
    const n = allPts.filter(p => p.r.mode === mm).length;
    ge.append('text').attr('x', 60).attr('y', 4).attr('font-size', 9.5)
      .attr('fill', 'var(--muted)').text(`(${n})`);
  });
  // perfect recall entry below modes
  const ge = leg.append('g').attr('transform', `translate(0, ${allModes.length*22 + 10})`);
  ge.append('line').attr('x1',0).attr('y1',6).attr('x2',14).attr('y2',6)
    .attr('stroke','#7cd992').attr('stroke-dasharray','3 2');
  ge.append('text').attr('x',20).attr('y',10).attr('font-size',11).attr('fill','var(--muted)').text('perfect recall');
  // hints
  leg.append('text').attr('y', allModes.length*22 + 36).attr('font-size',10)
    .attr('fill','var(--muted)').text('click legend → toggle visibility');
  leg.append('text').attr('y', allModes.length*22 + 50).attr('font-size',10)
    .attr('fill','var(--muted)').text('click point → preview · shift+click → select');
}

function drawThroughput(host, rows) {
  if (typeof d3 === 'undefined') { _emptyHost(host, 'd3 not loaded'); return; }
  if (!rows.length) { _emptyHost(host); return; }
  const groups = {};
  for (const r of rows) {
    const k = r.backend || '(none)';
    (groups[k] = groups[k] || []).push(r);
  }
  const entries = Object.entries(groups).map(([k, arr]) => {
    const ws = arr.map(r => r.win_per_s||0).filter(x => x);
    const cs = arr.map(r => r.cand_per_s||0).filter(x => x);
    const med = a => a.length ? a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)] : 0;
    return {
      backend: k, runs: arr, n: arr.length,
      winMed: med(ws), winMin: ws.length?Math.min(...ws):0, winMax: ws.length?Math.max(...ws):0,
      candMed: med(cs), candMin: cs.length?Math.min(...cs):0, candMax: cs.length?Math.max(...cs):0,
    };
  }).sort((a,b) => b.winMed - a.winMed);
  host.innerHTML = '';
  const W = host.clientWidth || 1200;
  // Larger row height for readability + a bit more bottom for the axis
  const H = Math.max(200, entries.length * 56 + 110);
  const m = {top:54, right:30, bottom:46, left:140};
  const innerW = W - m.left - m.right;
  const innerH = H - m.top - m.bottom;
  const halfW = (innerW - 50) / 2;
  const maxWin  = Math.max(1, ...entries.map(e => e.winMax));
  const maxCand = Math.max(1, ...entries.map(e => e.candMax));
  const xW = d3.scaleLinear().domain([0, maxWin]).nice().range([0, halfW]);
  const xC = d3.scaleLinear().domain([0, maxCand]).nice().range([0, halfW]);
  // y.padding(0.30) → thicker visual gap + thicker bars
  const y  = d3.scaleBand().domain(entries.map(e => e.backend)).range([0, innerH]).padding(0.30);
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
  svg.append('text').attr('x', W/2).attr('y', 22).attr('text-anchor','middle')
    .attr('font-size', 13).attr('font-weight', 600).attr('fill','var(--fg)')
    .text(`Backends · throughput (windows/s and candidates/s) · ${entries.length} backends`);
  // sub-titles
  svg.append('text').attr('x', m.left + halfW/2).attr('y', 40).attr('text-anchor','middle')
    .attr('font-size',11).attr('fill','var(--muted)').text('windows/s (median, error = min/max)');
  svg.append('text').attr('x', m.left + halfW + 30 + halfW/2).attr('y', 40).attr('text-anchor','middle')
    .attr('font-size',11).attr('fill','var(--muted)').text('candidates/s (median, error = min/max)');
  const g = svg.append('g').attr('transform', `translate(${m.left}, ${m.top})`);
  // y labels (bigger + bolder)
  g.selectAll('text.ylab').data(entries).join('text')
    .attr('class','ylab').attr('x', -10).attr('y', e => y(e.backend) + y.bandwidth()/2 + 5)
    .attr('text-anchor','end').attr('font-size',13).attr('font-family','var(--mono)')
    .attr('font-weight', 600).attr('fill','var(--fg)').text(e => e.backend);
  // gridlines for both halves (so values are easier to read off)
  g.selectAll('line.x-grid-win').data(xW.ticks(5)).join('line')
    .attr('class','x-grid-win').attr('x1', d=>xW(d)).attr('x2', d=>xW(d))
    .attr('y1', 0).attr('y2', innerH)
    .attr('stroke','var(--line)').attr('stroke-dasharray','2 3').attr('opacity',0.5);
  // axes at bottom
  g.append('g').attr('transform', `translate(0,${innerH})`).attr('class','d3-axis')
    .call(d3.axisBottom(xW).ticks(5).tickSizeOuter(0));
  g.append('g').attr('transform', `translate(${halfW+50},${innerH})`).attr('class','d3-axis')
    .call(d3.axisBottom(xC).ticks(5).tickFormat(d => d3.format(',')(d)).tickSizeOuter(0));
  // tooltip-style content
  function _thrTip(e, kind) {
    const top = e.runs.slice().sort((a,b) => (b[kind+'_per_s']||0) - (a[kind+'_per_s']||0)).slice(0,5);
    const unit = kind === 'win' ? 'win/s' : 'cand/s';
    const med = kind === 'win' ? e.winMed : e.candMed;
    const mn = kind === 'win' ? e.winMin : e.candMin;
    const mx = kind === 'win' ? e.winMax : e.candMax;
    return `<div class="thead">${escapeHtml(e.backend)} · ${unit}</div>
      <div class="trow"><span class="nm">runs</span><span class="val">${e.n}</span></div>
      <div class="trow"><span class="nm">median</span><span class="val">${med.toLocaleString(undefined,{maximumFractionDigits:1})}</span></div>
      <div class="trow"><span class="nm">min</span><span class="val">${mn.toLocaleString(undefined,{maximumFractionDigits:1})}</span></div>
      <div class="trow"><span class="nm">max</span><span class="val">${mx.toLocaleString(undefined,{maximumFractionDigits:1})}</span></div>
      <div class="trow" style="border-top:1px dashed var(--line); padding-top:3px; margin-top:3px">
        <span class="nm" style="color:var(--muted)">top runs:</span></div>
      ${top.map(r => `<div class="trow"><span class="nm">${escapeHtml(r.label.split('/').pop())}</span><span class="val">${(r[kind+'_per_s']||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span></div>`).join('')}`;
  }
  // Helper to draw one half (a "panel"): bar + capped error line + value/n label
  const drawHalf = (parent, scale, color, getMed, getMin, getMax, fmt, kind) => {
    // Error bar layer FIRST (behind the bar) — thicker line + end caps
    const eg = parent.append('g').attr('class','err-' + kind).attr('pointer-events','none');
    eg.selectAll('line.eb').data(entries).join('line').attr('class','eb')
      .attr('x1', e => scale(getMin(e))).attr('x2', e => scale(getMax(e)))
      .attr('y1', e => y(e.backend) + y.bandwidth()/2)
      .attr('y2', e => y(e.backend) + y.bandwidth()/2)
      .attr('stroke', '#000').attr('stroke-opacity', 0.55).attr('stroke-width', 2.5);
    eg.selectAll('line.cap-l').data(entries).join('line').attr('class','cap-l')
      .attr('x1', e => scale(getMin(e))).attr('x2', e => scale(getMin(e)))
      .attr('y1', e => y(e.backend) + y.bandwidth()/2 - 6)
      .attr('y2', e => y(e.backend) + y.bandwidth()/2 + 6)
      .attr('stroke', '#000').attr('stroke-opacity', 0.55).attr('stroke-width', 2);
    eg.selectAll('line.cap-r').data(entries).join('line').attr('class','cap-r')
      .attr('x1', e => scale(getMax(e))).attr('x2', e => scale(getMax(e)))
      .attr('y1', e => y(e.backend) + y.bandwidth()/2 - 6)
      .attr('y2', e => y(e.backend) + y.bandwidth()/2 + 6)
      .attr('stroke', '#000').attr('stroke-opacity', 0.55).attr('stroke-width', 2);
    // Bar (median value)
    parent.append('g').selectAll('rect.bar').data(entries).join('rect').attr('class','bar')
      .attr('x', 0).attr('y', e => y(e.backend))
      .attr('width', e => scale(getMed(e))).attr('height', y.bandwidth())
      .attr('fill', color).attr('opacity', 0.95)
      .attr('stroke', '#fff').attr('stroke-opacity', 0.25).attr('stroke-width', 1)
      .attr('cursor','help')
      .on('mouseover', (ev, e) => _showTip(_thrTip(e, kind), ev))
      .on('mousemove', (ev, e) => _showTip(_thrTip(e, kind), ev))
      .on('mouseleave', _hideTip);
    // Median value label INSIDE the bar (white text + dark stroke) when
    // the bar is wide enough, otherwise OUTSIDE the bar — and when outside
    // we MUST use a dark stroke (the page background is dark; a white
    // stroke renders the text invisible, which was the previous bug).
    parent.append('g').selectAll('text.vl').data(entries).join('text').attr('class','vl')
      .attr('x', e => scale(getMed(e)) > 60 ? scale(getMed(e)) - 6 : scale(getMed(e)) + 8)
      .attr('y', e => y(e.backend) + y.bandwidth()/2 + 5)
      .attr('text-anchor', e => scale(getMed(e)) > 60 ? 'end' : 'start')
      .attr('font-size', 13).attr('font-weight', 700)
      .attr('font-family','var(--mono)')
      // INSIDE → white text on the bar color; OUTSIDE → also light text
      // (page bg is dark) — never use page-bg color or it disappears.
      .attr('fill', '#fff')
      // INSIDE → dark stroke kills the bar bleed-through.
      // OUTSIDE → dark stroke too (page bg is dark, so a halo of dark
      // around white text still works; the previous white-on-white was
      // the unreadable case in the screenshot).
      .attr('stroke', 'rgba(0,0,0,0.85)')
      .attr('stroke-width', 2.5).attr('paint-order', 'stroke')
      .attr('pointer-events','none')
      .text(e => fmt(getMed(e)));
    // "n=N" subscript on the side, away from the bar
    parent.append('g').selectAll('text.nl').data(entries).join('text').attr('class','nl')
      .attr('x', e => scale(getMax(e)) + 10)
      .attr('y', e => y(e.backend) + y.bandwidth()/2 + 5)
      .attr('font-size', 11).attr('font-family','var(--mono)').attr('fill','var(--muted)')
      .attr('pointer-events','none')
      .text(e => `n=${e.n}`);
  };
  // Left half: windows/s
  drawHalf(g, xW, '#5aa1ff',
           e => e.winMed, e => e.winMin, e => e.winMax,
           v => v.toFixed(1), 'win');
  // Right half: candidates/s
  const gC = svg.append('g').attr('transform', `translate(${m.left + halfW + 50}, ${m.top})`);
  // gridlines for candidate panel
  gC.selectAll('line.x-grid-cand').data(xC.ticks(5)).join('line')
    .attr('class','x-grid-cand').attr('x1', d=>xC(d)).attr('x2', d=>xC(d))
    .attr('y1', 0).attr('y2', innerH)
    .attr('stroke','var(--line)').attr('stroke-dasharray','2 3').attr('opacity',0.5);
  drawHalf(gC, xC, '#ffb05a',
           e => e.candMed, e => e.candMin, e => e.candMax,
           v => d3.format(',')(Math.round(v)), 'cand');
}

// Line palette for the multi-run plot (stable, readable on a dark background).
const _TPUT_PALETTE = ['#5aa1ff','#ffb05a','#5ad19a','#ff6b9d','#b89cff','#ffd35a',
                       '#4fd0e0','#ff8a5a','#9ad14f','#e05a9a','#6ad1ff','#d1b04f'];

// THROUGHPUT = f(time) curve: one line per run. Shows how the system behaves
// over time (ramp-up, stalls, stability) — the key metric to judge REAL-TIME
// viability. Selectors: metric (global / per-step throughput / CPU / memory /
// energy), log-Y, X axis (time|%).
function drawTputTimeline(host, rows) {
  if (!host) return;
  if (typeof d3 === 'undefined') { _emptyHost(host, 'd3 not loaded'); return; }
  const metric = (document.getElementById('tput-metric')||{}).value || 'windows_per_s';
  const logY = !!(document.getElementById('tput-log')||{}).checked;
  const xField = (document.getElementById('tput-xaxis')||{}).value || 't';
  const note = document.getElementById('tput-note');
  const isRes = ['cpu_pct','mem_mb','power_cores'].includes(metric);
  // Build (run -> points) from the right series (throughput vs resources).
  const series = [];
  for (const r of rows) {
    const raw = (isRes ? r.res_series : r.tput_series) || [];
    const xf = (xField === 'frac' && !isRes) ? 'frac' : 't';
    const pts = [];
    for (const p of raw) {
      const xv = p[xf], yv = p[metric];
      if (xv == null || yv == null) continue;
      if (logY && !(yv > 0)) continue;       // log : ignore ≤0
      pts.push({ x: (xf === 'frac' ? xv * 100 : xv), y: yv });
    }
    if (pts.length >= 1) {
      pts.sort((a,b) => a.x - b.x);
      const ys = pts.map(p => p.y).sort((a,b)=>a-b);
      series.push({ label: r.label, name: (r.label||'').split('/').pop(),
                    mode: r.mode, backend: r.backend,
                    med: ys[Math.floor(ys.length/2)], pts });
    }
  }
  if (!series.length) {
    _emptyHost(host, 'no throughput series — runs without throughput.csv (re-run the pipeline) or empty metric');
    if (note) note.textContent = '';
    return;
  }
  // Cap the number of lines for legibility (fastest first).
  series.sort((a,b) => b.med - a.med);
  const CAP = 12;
  const dropped = Math.max(0, series.length - CAP);
  const shown = series.slice(0, CAP);
  if (note) note.textContent = dropped
    ? `${shown.length} runs shown (${dropped} hidden — refine the selection)`
    : `${shown.length} run(s)`;

  host.innerHTML = '';
  const W = host.clientWidth || 1200;
  const H = 420;
  const m = {top:30, right:210, bottom:46, left:70};
  const innerW = Math.max(200, W - m.left - m.right);
  const innerH = H - m.top - m.bottom;
  const allX = shown.flatMap(s => s.pts.map(p => p.x));
  const allY = shown.flatMap(s => s.pts.map(p => p.y));
  const xMax = Math.max(...allX), xMin = Math.min(0, ...allX);
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if (logY) yMin = Math.max(yMin, Math.min(...allY.filter(v => v > 0)) || 1e-3);
  const x = d3.scaleLinear().domain([xMin, xMax || 1]).range([0, innerW]).nice();
  const y = (logY ? d3.scaleLog() : d3.scaleLinear())
    .domain(logY ? [yMin * 0.9, yMax * 1.1] : [Math.min(0, yMin), yMax * 1.05])
    .range([innerH, 0]).nice();
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  const METRIC_LABEL = {
    windows_per_s:'windows/s', windows_per_s_cum:'windows/s (cumulative)',
    sketch_per_s:'sketch — windows/s', candidate_per_s:'candidates/s',
    validate_per_s:'pairs/s', monitor_per_s:'corr/s',
    cpu_pct:'CPU %', mem_mb:'memory (MB)', power_cores:'cores busy',
  };
  // gridlines Y
  g.selectAll('line.gy').data(y.ticks(6)).join('line').attr('class','gy')
    .attr('x1',0).attr('x2',innerW).attr('y1',d=>y(d)).attr('y2',d=>y(d))
    .attr('stroke','var(--line)').attr('stroke-dasharray','2 3').attr('opacity',0.5);
  // axes
  g.append('g').attr('transform',`translate(0,${innerH})`).attr('class','d3-axis')
    .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
  g.append('g').attr('class','d3-axis')
    .call(d3.axisLeft(y).ticks(6).tickFormat(d => d3.format('~s')(d)).tickSizeOuter(0));
  // titres d'axes
  svg.append('text').attr('x', m.left + innerW/2).attr('y', H - 8)
    .attr('text-anchor','middle').attr('font-size',11).attr('fill','var(--muted)')
    .text(xField === 'frac' && !isRes ? 'progress (%)' : 'wall time (s)');
  svg.append('text').attr('transform','rotate(-90)')
    .attr('x', -(m.top + innerH/2)).attr('y', 16)
    .attr('text-anchor','middle').attr('font-size',11).attr('fill','var(--muted)')
    .text((METRIC_LABEL[metric] || metric) + (logY ? ' (log)' : ''));
  const line = d3.line().x(p => x(p.x)).y(p => y(p.y)).curve(d3.curveMonotoneX);
  const color = i => _TPUT_PALETTE[i % _TPUT_PALETTE.length];
  shown.forEach((s, i) => {
    g.append('path').datum(s.pts).attr('fill','none')
      .attr('stroke', color(i)).attr('stroke-width', 1.8).attr('opacity', 0.9)
      .attr('d', line);
    // points (markers + hover)
    const tipHtml = p =>
      `<div class="thead">${escapeHtml(s.name)}</div>`
      + `<div class="trow"><span class="nm">${xField==='frac'&&!isRes?'progress':'t'}</span>`
      + `<span class="val">${p.x.toLocaleString(undefined,{maximumFractionDigits:1})}${xField==='frac'&&!isRes?' %':' s'}</span></div>`
      + `<div class="trow"><span class="nm">${escapeHtml(METRIC_LABEL[metric]||metric)}</span>`
      + `<span class="val">${p.y.toLocaleString(undefined,{maximumFractionDigits:1})}</span></div>`;
    g.append('g').selectAll('circle').data(s.pts).join('circle')
      .attr('cx', p => x(p.x)).attr('cy', p => y(p.y)).attr('r', 2.2)
      .attr('fill', color(i)).attr('opacity', 0.8).attr('cursor','help')
      .on('mouseover', (ev, p) => _showTip(tipHtml(p), ev))
      .on('mousemove', (ev, p) => _showTip(tipHtml(p), ev))
      .on('mouseleave', _hideTip);
  });
  // legend (on the right)
  const lg = svg.append('g').attr('transform', `translate(${m.left + innerW + 14}, ${m.top})`);
  shown.forEach((s, i) => {
    const row = lg.append('g').attr('transform', `translate(0, ${i * 18})`).attr('cursor','default');
    row.append('rect').attr('width',11).attr('height',11).attr('y',-9).attr('fill', color(i));
    row.append('text').attr('x',16).attr('y',0).attr('font-size',11)
      .attr('font-family','var(--mono)').attr('fill','var(--fg)')
      .text(_shortLabel(s.name, 26))
      .append('title').text(s.label);
  });
}

function _fmtDate(ms, scale) {
  const d = new Date(ms);
  const pad = n => String(n).padStart(2, '0');
  if (scale === 'day')      return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
  if (scale === 'datetime') return `${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}h`;
  if (scale === 'hour')     return `${pad(d.getHours())}h`;
  return d.toISOString();
}
function drawTimeseries(host, data, opts) {
  if (typeof d3 === 'undefined') {
    host.innerHTML = '<div class="empty">D3 not loaded — check /static/d3.v7.min.js</div>';
    return;
  }
  host.innerHTML = '';
  const W = host.clientWidth || 1280, H = 480;
  const m = {top: 36, right: 210, bottom: 60, left: 64};
  const innerW = Math.max(200, W - m.left - m.right);
  const innerH = H - m.top - m.bottom;
  const align = !!data.align, norm = !!opts.normalize;

  // ---- z-normalize ----
  function zscore(vals) {
    if (!norm) return vals;
    const f = vals.filter(v => v != null && Number.isFinite(v));
    if (f.length < 2) return vals;
    const μ = d3.mean(f), σ = d3.deviation(f);
    if (!σ) return vals;
    return vals.map(v => (v == null || !Number.isFinite(v)) ? null : (v - μ) / σ);
  }

  // ---- shape data: build [{x, y}] arrays so d3.line works cleanly ----
  // x is either relative hours (align) OR an absolute Date (else)
  // Use a reference (xs[0], dates_ms[0]) to turn hour offset into Date.
  let baseX = null, baseMs = null;
  for (const w of [...data.windows, ...(data.history_lines||[])]) {
    if (w.xs && w.xs.length && w.dates_ms && w.dates_ms.length) {
      baseX = w.xs[0]; baseMs = w.dates_ms[0]; break;
    }
  }
  const xvToDate = xv => new Date(baseMs + (xv - baseX) * 3600 * 1000);

  function shape(w) {
    const ys = zscore(w.values);
    return w.xs.map((x, i) => {
      const y = ys[i];
      if (y == null || !Number.isFinite(y)) return null;
      const xx = align ? (x - w.t) : (baseMs != null ? xvToDate(x) : x);
      return {x: xx, y, raw_x: x};
    });
  }
  const windows = data.windows.map(w => ({...w, data: shape(w)}));
  const history = (data.history_lines || []).map(h => ({...h, data: shape(h)}));

  // ---- extents ----
  const allPts = [...windows, ...history].flatMap(w => w.data.filter(d => d));
  if (!allPts.length) {
    host.innerHTML = '<div class="empty">no data points to plot</div>';
    return;
  }
  const yExtent = d3.extent(allPts, d => d.y);
  const yPad = (yExtent[1] - yExtent[0]) * 0.05 || 1;
  const yScale = d3.scaleLinear()
    .domain([yExtent[0] - yPad, yExtent[1] + yPad]).nice()
    .range([innerH, 0]);
  let xScale;
  if (align || baseMs == null) {
    const xExtent = d3.extent(allPts, d => d.x);
    const xPad = (xExtent[1] - xExtent[0]) * 0.02 || 1;
    xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPad, xExtent[1] + xPad]).nice()
      .range([0, innerW]);
  } else {
    const xExtent = d3.extent(allPts, d => d.x);
    const xPad = (+xExtent[1] - +xExtent[0]) * 0.02 || 3_600_000;
    xScale = d3.scaleUtc()
      .domain([new Date(+xExtent[0] - xPad), new Date(+xExtent[1] + xPad)]).nice()
      .range([0, innerW]);
  }
  const xScale0 = xScale.copy();  // for reset zoom

  // ---- SVG scaffolding ----
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('width', W).attr('height', H);
  // title
  const titleBits = [data.pipeline, `window_size=${data.window_size}`];
  if (data.history && !align) titleBits.push('history shown');
  svg.append('text')
    .attr('x', W/2).attr('y', 22).attr('text-anchor', 'middle')
    .attr('font-size', 13).attr('font-weight', 600)
    .attr('fill', 'var(--fg)')
    .text(titleBits.join(' · '));

  // clip for zoom
  svg.append('defs').append('clipPath').attr('id', 'plot-clip')
    .append('rect').attr('width', innerW).attr('height', innerH);
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  // plot background
  g.append('rect').attr('width', innerW).attr('height', innerH)
    .attr('fill', 'rgba(255,255,255,0.02)')
    .attr('stroke', 'var(--line)');

  // ---- axes ----
  const xAxisG = g.append('g').attr('transform', `translate(0,${innerH})`)
    .attr('class', 'd3-axis');
  const yAxisG = g.append('g').attr('class', 'd3-axis');
  const gridG = g.append('g').attr('class', 'd3-grid')
    .attr('pointer-events', 'none')
    .attr('display', opts.grid ? null : 'none');

  function drawAxes(sx) {
    const xTickFmt = align
      ? d => (d >= 0 ? '+' : '') + d + 'h'
      : (sx.domain()[1] - sx.domain()[0] > 1000*3600*24*60 ? d3.utcFormat('%Y-%m-%d')
        : sx.domain()[1] - sx.domain()[0] > 1000*3600*12  ? d3.utcFormat('%m-%d %Hh')
        :                                                   d3.utcFormat('%Hh'));
    const xAxis = d3.axisBottom(sx).ticks(8).tickFormat(xTickFmt).tickSizeOuter(0);
    const yAxis = d3.axisLeft(yScale).ticks(7).tickFormat(d => norm ? d.toFixed(2) : d.toFixed(1)).tickSizeOuter(0);
    xAxisG.call(xAxis);
    yAxisG.call(yAxis);
    // grid
    gridG.selectAll('*').remove();
    yScale.ticks(7).forEach(t => {
      gridG.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', yScale(t)).attr('y2', yScale(t))
        .attr('stroke', 'var(--line)').attr('stroke-dasharray', '2 3')
        .attr('opacity', 0.5);
    });
    sx.ticks(8).forEach(t => {
      gridG.append('line')
        .attr('x1', sx(t)).attr('x2', sx(t))
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', 'var(--line)').attr('stroke-dasharray', '2 3')
        .attr('opacity', 0.5);
    });
  }
  drawAxes(xScale);

  // axis labels
  svg.append('text').attr('class','d3-axis-label')
    .attr('transform', `translate(${m.left - 48}, ${m.top + innerH/2}) rotate(-90)`)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'var(--muted)')
    .text(norm ? 'z-score' : 'value');
  svg.append('text').attr('class','d3-axis-label')
    .attr('x', m.left + innerW/2).attr('y', H - 18)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'var(--muted)')
    .text(align ? 't (relative hours)' : 't (date)');

  // ---- drawable area (clipped, gets zoomed via re-scale) ----
  const plot = g.append('g').attr('clip-path', 'url(#plot-clip)');

  // window highlights (background)
  const hl = plot.append('g').attr('class', 'd3-hl');
  function drawHighlights(sx) {
    hl.selectAll('rect').remove();
    for (const w of windows) {
      const xa = align ? 0 : (baseMs != null ? xvToDate(w.t) : w.t);
      const xb = align ? data.window_size - 1
              : (baseMs != null ? xvToDate(w.t + data.window_size - 1) : w.t + data.window_size - 1);
      const x1 = sx(xa), x2 = sx(xb);
      hl.append('rect')
        .attr('x', Math.min(x1, x2)).attr('y', 0)
        .attr('width', Math.abs(x2 - x1)).attr('height', innerH)
        .attr('fill', w.color).attr('opacity', 0.10);
    }
  }
  drawHighlights(xScale);

  // monotone-x line generator (smooth, won't overshoot)
  function mkLine(sx) {
    return d3.line()
      .defined(d => d != null && Number.isFinite(d.y))
      .x(d => sx(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveMonotoneX);
  }

  // history (faded) — drawn first so windows are on top
  const histG = plot.append('g').attr('class', 'd3-hist');
  function drawHistory(sx) {
    histG.selectAll('path').remove();
    const ln = mkLine(sx);
    for (const h of history) {
      histG.append('path')
        .datum(h.data)
        .attr('fill', 'none')
        .attr('stroke', h.color)
        .attr('stroke-width', 1)
        .attr('stroke-opacity', 0.25)
        .attr('d', ln);
    }
  }
  drawHistory(xScale);

  // window lines + markers
  const winG = plot.append('g').attr('class', 'd3-win');
  function drawWindows(sx) {
    winG.selectAll('*').remove();
    const ln = mkLine(sx);
    for (const w of windows) {
      winG.append('path')
        .datum(w.data)
        .attr('fill', 'none')
        .attr('stroke', w.color)
        .attr('stroke-width', 1.8)
        .attr('d', ln);
      if (opts.points) {
        const r = w.data.length <= 200 ? 2.6 : w.data.length <= 600 ? 1.6 : 1.0;
        winG.selectAll('circle.pt-' + w.id.replace(/[^A-Za-z0-9_]/g,'_'))
          .data(w.data.filter(d => d != null))
          .join('circle')
          .attr('cx', d => sx(d.x))
          .attr('cy', d => yScale(d.y))
          .attr('r', r)
          .attr('fill', w.color)
          .attr('opacity', 0.9);
      }
    }
  }
  drawWindows(xScale);

  // ---- crosshair + tooltip ----
  const focus = g.append('g').attr('class', 'd3-focus').attr('display', 'none');
  focus.append('line').attr('class', 'cx-v')
    .attr('y1', 0).attr('y2', innerH)
    .attr('stroke', 'var(--accent)').attr('stroke-width', 1).attr('stroke-opacity', 0.5);
  const dotsG = focus.append('g');
  // Tooltip uses a foreignObject so we can style it like the rest of the app
  const tipDiv = d3.select(host).append('div')
    .attr('class', 'd3-tooltip')
    .style('position', 'absolute')
    .style('pointer-events', 'none')
    .style('display', 'none');
  host.style.position = 'relative';

  function bisectFor(arr) {
    return d3.bisector(d => d == null ? Infinity : d.x).left;
  }
  function showTooltip(mx, my, sx) {
    // For each series, find closest point in current x
    const xRef = sx.invert(mx);
    const rows = [];
    for (const w of windows) {
      const arr = w.data;
      if (!arr.length) continue;
      // bisect on x (handles nulls by skipping)
      const valid = arr.filter(d => d);
      if (!valid.length) continue;
      const idx = d3.bisector(d => +d.x).left(valid, +xRef);
      const pick = (idx >= valid.length ? valid[valid.length-1]
                  : idx <= 0           ? valid[0]
                  : (Math.abs(+valid[idx].x - xRef) < Math.abs(+valid[idx-1].x - xRef) ? valid[idx] : valid[idx-1]));
      rows.push({w, p: pick});
    }
    if (!rows.length) return;
    // Vertical line at closest series' x (use median-ish)
    rows.sort((a, b) => Math.abs(+a.p.x - xRef) - Math.abs(+b.p.x - xRef));
    const anchor = rows[0].p;
    focus.attr('display', null);
    focus.select('line.cx-v')
      .attr('x1', sx(anchor.x)).attr('x2', sx(anchor.x));
    dotsG.selectAll('circle').remove();
    for (const r of rows) {
      dotsG.append('circle')
        .attr('cx', sx(r.p.x)).attr('cy', yScale(r.p.y)).attr('r', 4)
        .attr('fill', r.w.color).attr('stroke', '#fff').attr('stroke-width', 1.2);
    }
    // Build tooltip content
    let header;
    if (align || baseMs == null) {
      header = (anchor.x >= 0 ? '+' : '') + Math.round(anchor.x) + 'h (relative)';
    } else {
      const d = anchor.x instanceof Date ? anchor.x : new Date(anchor.x);
      const pad = n => String(n).padStart(2,'0');
      header = `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}h`;
    }
    const rowsHtml = rows.map(r => `<div class="trow"><span class="dot" style="background:${r.w.color}"></span>
      <span class="nm">${escapeHtml(r.w.id)}</span>
      <span class="val">${r.p.y.toFixed(norm ? 3 : 2)}</span></div>`).join('');
    tipDiv.html(`<div class="thead">${escapeHtml(header)}</div>${rowsHtml}`);
    tipDiv.style('display', 'block');
    // position
    const bb = host.getBoundingClientRect();
    const xPx = sx(anchor.x) + m.left + 12;
    const yPx = my + 10;
    const tipW = tipDiv.node().offsetWidth || 0;
    tipDiv.style('left', Math.min(bb.width - tipW - 6, xPx) + 'px')
          .style('top', yPx + 'px');
  }
  function hideTooltip() {
    focus.attr('display', 'none');
    tipDiv.style('display', 'none');
  }

  // hover/touch surface (transparent, sits ON TOP of everything inside the plot area)
  const hover = g.append('rect')
    .attr('width', innerW).attr('height', innerH)
    .attr('fill', 'transparent')
    .style('cursor', 'crosshair');
  hover.on('mousemove', function(ev) {
    const [mx, my] = d3.pointer(ev, this);
    showTooltip(mx, my, currentX);
  }).on('mouseleave', hideTooltip);

  // ---- zoom via brush selection ----
  let currentX = xScale;
  const brush = d3.brushX()
    .extent([[0, 0], [innerW, innerH]])
    .on('end', brushed);
  const brushG = g.append('g').attr('class', 'd3-brush').call(brush);
  // brush should not block hover by default — make its overlay transparent to mouse
  brushG.select('.overlay').attr('pointer-events', 'none');
  function brushed(ev) {
    const sel = ev.selection;
    if (!sel) return;  // no selection → ignore
    const [x0, x1] = sel;
    if (Math.abs(x1 - x0) < 5) return;  // too small
    const domain = [currentX.invert(x0), currentX.invert(x1)];
    currentX = currentX.copy().domain(domain);
    drawAxes(currentX);
    drawHighlights(currentX);
    drawHistory(currentX);
    drawWindows(currentX);
    brushG.call(brush.move, null);
  }
  // double-click anywhere in the plot → reset zoom
  svg.on('dblclick', () => {
    currentX = xScale0;
    drawAxes(currentX);
    drawHighlights(currentX);
    drawHistory(currentX);
    drawWindows(currentX);
    hideTooltip();
  });

  // Allow brushing only when shift is held; otherwise drag does nothing and
  // hover/tooltip works smoothly. Holding shift activates the brush overlay.
  document.addEventListener('keydown', e => {
    if (e.key === 'Shift') brushG.select('.overlay').attr('pointer-events', 'all');
  });
  document.addEventListener('keyup', e => {
    if (e.key === 'Shift') brushG.select('.overlay').attr('pointer-events', 'none');
  });

  // ---- Legend (right) — 2-line entries, with toggleable visibility ----
  let ly = m.top;
  for (const w of windows) {
    const dt = (w.dates_ms && w.dates_ms.length)
      ? _fmtDate(w.dates_ms[0], (xScale0.domain()[1] - xScale0.domain()[0] > 30*24*3600*1000 || (align && (xScale0.domain()[1]-xScale0.domain()[0])>30*24)) ? 'day' : 'datetime')
      : '';
    const dtEnd = (w.dates_ms && w.dates_ms.length > 1)
      ? _fmtDate(w.dates_ms[w.dates_ms.length - 1],
                 (xScale0.domain()[1] - xScale0.domain()[0] > 30*24*3600*1000 || (align && (xScale0.domain()[1]-xScale0.domain()[0])>30*24)) ? 'day' : 'datetime')
      : '';
    const ge = svg.append('g')
      .attr('transform', `translate(${m.left + innerW + 16}, ${ly})`)
      .style('cursor', 'pointer');
    ge.append('line').attr('x1', 0).attr('y1', 7).attr('x2', 20).attr('y2', 7)
      .attr('stroke', w.color).attr('stroke-width', 2);
    ge.append('circle').attr('cx', 10).attr('cy', 7).attr('r', 2.5).attr('fill', w.color);
    ge.append('text').attr('x', 26).attr('y', 6).attr('font-size', 11)
      .attr('font-weight', 600).attr('fill', 'var(--fg)').text(w.id);
    ge.append('text').attr('x', 26).attr('y', 20).attr('font-size', 9.5)
      .attr('fill', 'var(--muted)')
      .text(`${dt}${dtEnd && dtEnd !== dt ? ' → ' + dtEnd : ''}`);
    ge.on('click', () => {
      // toggle that series visibility
      const sel = winG.selectAll('path').filter((d, i) => i === windows.indexOf(w));
      const visible = sel.attr('opacity') !== '0';
      sel.attr('opacity', visible ? 0 : 1);
      ge.attr('opacity', visible ? 0.35 : 1);
    });
    ly += 30;
  }
  if (history.length) {
    const ge = svg.append('g')
      .attr('transform', `translate(${m.left + innerW + 16}, ${ly + 4})`);
    ge.append('line').attr('x1', 0).attr('y1', 6).attr('x2', 20).attr('y2', 6)
      .attr('stroke', '#888').attr('stroke-width', 1).attr('stroke-opacity', 0.5);
    ge.append('text').attr('x', 26).attr('y', 9).attr('font-size', 10)
      .attr('fill', 'var(--muted)').text('history (faded)');
    ly += 18;
  }
  // help text
  svg.append('text')
    .attr('x', m.left + innerW + 16).attr('y', m.top + innerH - 16)
    .attr('font-size', 9.5).attr('fill', 'var(--muted)')
    .text('hover for values · shift+drag to zoom · double-click to reset');
}

function _drawGenericHeatmap(host, rows, rowKey, colKey, rowLabel, colLabel) {
  if (typeof d3 === 'undefined') { _emptyHost(host, 'd3 not loaded'); return; }
  if (!rows.length) { _emptyHost(host); return; }
  // Drop rows where either axis value is missing OR explicitly "n/a".
  // The server already remaps stray "n/a" on corrtrack runs to the real
  // v2 default (random_projection / truncate) — anything still labelled
  // "n/a" here is genuinely non-applicable (bf/filcorr sketch column).
  const _isReal = v => v && v !== 'n/a' && v !== 'N/A';
  const filtered = rows.filter(r => _isReal(r[rowKey]) && _isReal(r[colKey]));
  if (!filtered.length) {
    _emptyHost(host, `no runs have both ${rowLabel} and ${colLabel} set — try widening filters`);
    return;
  }
  const slowest = Math.max(...rows.map(r => r.runtime));
  const cells = {};
  for (const r of filtered) {
    const k = r[rowKey] + '|' + r[colKey];
    if (!cells[k] || cells[k].runtime > r.runtime) cells[k] = r;
  }
  const rowVals = [...new Set(Object.values(cells).map(r => r[rowKey]))].sort();
  const colVals = [...new Set(Object.values(cells).map(r => r[colKey]))].sort();
  const nb = rowVals.length, ni = colVals.length;
  host.innerHTML = '';
  const cellW = Math.max(80, Math.min(160, ((host.clientWidth || 1200) - 220) / Math.max(1, ni)));
  const cellH = 64;
  const m = {top:60, right:30, bottom:30, left:180};
  const W = m.left + ni * cellW + m.right;
  const H = m.top + nb * cellH + m.bottom;
  // Log color scale — Inferno truncated to [0.10, 0.60]. Stops only in the
  // dark half of the Inferno ramp → every stop has a luminance ≤ 0.40,
  // guaranteeing that white text + black stroke stays readable EVERYWHERE (no
  // green / yellow / light-orange zone like a Turbo / Plasma trim).
  //
  //  t=0.10 → #1d1147   dark indigo
  //  t=0.25 → #4f0a6c   dark purple
  //  t=0.40 → #8f1a72   dark magenta
  //  t=0.55 → #c63a4e   crimson rose
  //  t=0.60 → #d44a39   dark red-orange  ← deliberate cap, we never go beyond
  //
  // Perceptually uniform differentiation (Inferno is calibrated for that) over
  // the ×20-85 cluster that visually dominates.
  const speedups = Object.values(cells).map(r => slowest / r.runtime);
  const minS = Math.max(0.5, Math.min(...speedups));
  const maxS = Math.max(...speedups);
  const _heatRamp = t => d3.interpolateInferno(0.10 + 0.50 * t);
  const colorScale = d3.scaleSequential(_heatRamp)
                       .domain([Math.log10(minS), Math.log10(maxS)]);
  // Robust color parser: works with "rgb(r,g,b)", "rgb(r, g, b)", and
  // "#rgb"/"#rrggbb" hex (older regex broke on hex by matching the wrong
  // digit groups). Uses d3.rgb when available.
  const _parseRgb = (fill) => {
    if (typeof d3.rgb === 'function') {
      const c = d3.rgb(fill);
      if (c && !isNaN(c.r)) return [c.r, c.g, c.b];
    }
    const mm = fill.match(/rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/i);
    if (mm) return [+mm[1], +mm[2], +mm[3]];
    return null;
  };
  const txtContrast = (fill) => {
    const rgb = _parseRgb(fill); if (!rgb) return '#fff';
    // W3C relative luminance — higher → lighter background → dark text
    const lum = (0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]) / 255;
    return lum > 0.55 ? '#111' : '#fff';
  };
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
  svg.append('text').attr('x', W/2).attr('y', 22).attr('text-anchor','middle')
    .attr('font-size',13).attr('font-weight',600).attr('fill','var(--fg)')
    .text(`Best speedup per (${rowLabel} × ${colLabel}) · baseline = slowest run (${slowest.toFixed(1)}s)`);
  // Column labels
  svg.append('g').selectAll('text.col-lab').data(colVals).join('text')
    .attr('class','col-lab').attr('x', (_,j) => m.left + j*cellW + cellW/2).attr('y', m.top - 10)
    .attr('text-anchor','middle').attr('font-size',11)
    .attr('font-family','var(--mono)').attr('fill','var(--muted)').text(d => d);
  // Row labels
  svg.append('g').selectAll('text.row-lab').data(rowVals).join('text')
    .attr('class','row-lab').attr('x', m.left - 6)
    .attr('y', (_,i) => m.top + i*cellH + cellH/2 + 4)
    .attr('text-anchor','end').attr('font-size',11)
    .attr('font-family','var(--mono)').attr('fill','var(--muted)').text(d => d);
  // Cells data: flat array {row, col, x, y, r?}
  const flatCells = [];
  rowVals.forEach((rv, i) => colVals.forEach((cv, j) => {
    flatCells.push({rv, cv, i, j, r: cells[rv + '|' + cv] || null});
  }));
  const cg = svg.append('g');
  cg.selectAll('g.cell').data(flatCells).join('g')
    .attr('class','cell').attr('transform', d => `translate(${m.left + d.j*cellW},${m.top + d.i*cellH})`)
    .each(function(d) {
      const g = d3.select(this);
      if (!d.r) {
        g.append('rect').attr('width', cellW-2).attr('height', cellH-2)
          .attr('fill','#1a1d24').attr('stroke','var(--line)');
        g.append('text').attr('x', cellW/2).attr('y', cellH/2 + 4)
          .attr('text-anchor','middle').attr('font-size',14).attr('fill','#555').text('–');
        return;
      }
      const sp = slowest / d.r.runtime;
      const fill = colorScale(Math.log10(Math.max(0.1, sp)));
      const tc = txtContrast(fill);
      const stroke = tc === '#fff' ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.85)';
      g.append('rect').attr('width', cellW-2).attr('height', cellH-2)
        .attr('fill', fill).attr('cursor','pointer')
        .on('mouseover', (ev) => _showTip(_rowTipHtml(d.r,
          `<div class="trow"><span class="nm">${rowLabel}</span><span class="val">${escapeHtml(d.rv)}</span></div>
           <div class="trow"><span class="nm">${colLabel}</span><span class="val">${escapeHtml(d.cv)}</span></div>
           <div class="trow"><span class="nm">×baseline (slowest)</span><span class="val">×${sp.toFixed(2)}</span></div>`), ev))
        .on('mousemove', (ev) => _showTip(_rowTipHtml(d.r), ev))
        .on('mouseleave', _hideTip)
        .on('click', (ev) => { _hideTip(); _chartRowAction(d.r, ev); });
      // Common attrs for cell text: paint-order stroke so it stays readable
      // on any bg color. Stroke is opaque + wider for the warm gradient
      // (was 2.2 with .85 alpha → too thin to defeat mid-amber backgrounds).
      const strokeOpaque = tc === '#fff' ? '#000' : '#fff';
      const _txt = (yy, sz, weight, str) => g.append('text')
        .attr('x', cellW/2).attr('y', yy).attr('text-anchor','middle')
        .attr('font-size', sz).attr('font-weight', weight)
        .attr('fill', tc).attr('stroke', strokeOpaque).attr('stroke-width', 3)
        .attr('paint-order', 'stroke').attr('font-family','monospace')
        .attr('pointer-events','none').text(str);
      _txt(20, 12.5, 700, `×${sp.toFixed(1)}`);
      _txt(37, 11,   500, `${d.r.runtime.toFixed(1)}s`);
      _txt(53, 10,   400, `n=${(d.r.n_correlated||0).toLocaleString()}`);
    });
  svg.append('text').attr('x', W/2).attr('y', H - 8).attr('text-anchor','middle')
    .attr('font-size',10).attr('fill','var(--muted)').text(`${colLabel} →`);
}

function drawSketchHeatmap(host, rows) {
  _drawGenericHeatmap(host, rows, 'sketch_method', 'index_backend', 'sketch', 'index');
}

function drawHeatmap(host, rows) {
  _drawGenericHeatmap(host, rows, 'backend', 'index_backend', 'backend', 'index');
}
$('#cmp-refresh').addEventListener('click', async () => {
  // Drop the stale quality map so the table shows '—' until the recompute
  // finishes (avoids mixing old precision/recall with newly-added runs).
  ST.cmpQuality = null;
  // Reset the once-per-session retry guard so the user can FORCE another
  // attempt on rows that we previously gave up on (e.g. a sharded run
  // whose result.json was eventually fixed up server-side).
  ST.staleRetried = new Set();
  ST._lastFullReload = 0;
  await loadCompare();
  // Re-render ALL charts with the fresh rows (compare table is rebuilt by
  // loadCompare; charts were stale until now).
  loadCharts();
  loadQuality(true);
});
$('#charts-refresh').addEventListener('click', loadCharts);
// Trade-off chart axis controls. Listen once with delegation, and enforce
// the "X axis ≠ Y axis" invariant whenever either dropdown changes.
document.addEventListener('change', e => {
  if (!e.target) return;
  const id = e.target.id;
  if (id === 'tro-x' || id === 'tro-y' || id === 'tro-x-log' || id === 'tro-y-log') {
    if (id === 'tro-x' || id === 'tro-y') _troEnsureDistinct(id);
    drawTradeoff($('#ch-tradeoff'), _filteredRows());
  }
});
$('#tro-swap')?.addEventListener('click', () => {
  const sx = $('#tro-x'), sy = $('#tro-y');
  const xl = $('#tro-x-log'), yl = $('#tro-y-log');
  [sx.value, sy.value] = [sy.value, sx.value];
  [xl.checked, yl.checked] = [yl.checked, xl.checked];
  drawTradeoff($('#ch-tradeoff'), _filteredRows());
});
$('#cmp-bf').addEventListener('change', () => { renderCompare(); loadCharts(); });
$('#cmp-ct').addEventListener('change', () => { renderCompare(); loadCharts(); });
$('#cmp-fc').addEventListener('change', () => { renderCompare(); loadCharts(); });
// Threshold inputs (quality + speedup) — debounced live update
for (const id of ['cmp-min-prec','cmp-min-recall','cmp-min-f1','cmp-min-speedup',
                  'cmp-hide-noq']) {
  document.addEventListener('change', e => {
    if (e.target && e.target.id === id) { renderCompare(); loadCharts(); }
  });
  document.addEventListener('input', e => {
    if (e.target && e.target.id === id && e.target.type === 'number') {
      clearTimeout(ST._qDeb);
      ST._qDeb = setTimeout(() => { renderCompare(); loadCharts(); }, 200);
    }
  });
}
$('#cmp-filters-clear')?.addEventListener('click', () => {
  // Reset all multi-select dimension filters
  for (const id of ['ms-backend','ms-index','ms-keymode','ms-sketch']) {
    const el = $('#'+id); el?._reset?.();
  }
  // Threshold inputs
  for (const id of ['cmp-min-prec','cmp-min-recall','cmp-min-f1','cmp-min-speedup']) {
    const el = $('#'+id); if (el) el.value = '';
  }
  const hq = $('#cmp-hide-noq'); if (hq) hq.checked = false;
  // Mode checkboxes
  ['#cmp-bf','#cmp-ct','#cmp-fc'].forEach(s => { const el=$(s); if (el) el.checked = true; });
  renderCompare(); loadCharts();
});
$('#cmp-only-sel').addEventListener('change', renderCompare);
$('#cmp-charts-sel').addEventListener('change', loadCharts);

// Selection controls
$('#cmp-sel-all').addEventListener('click', () => {
  if (!ST.cmpData) return;
  // only add currently visible rows
  const showBf = $('#cmp-bf').checked, showCt = $('#cmp-ct').checked,
        showFc = $('#cmp-fc').checked;
  for (const x of ST.cmpData.rows) {
    const visible = (x.mode === 'bf'        && showBf)
                 || (x.mode === 'corrtrack' && showCt)
                 || (x.mode === 'filcorr'   && showFc)
                 || (x.mode !== 'bf' && x.mode !== 'corrtrack' && x.mode !== 'filcorr');
    if (visible) ST.cmpSel.add(x.path);
  }
  renderCompare(); _selChanged();
});
$('#cmp-sel-none').addEventListener('click', () => {
  ST.cmpSel.clear(); renderCompare(); _selChanged();
});
$('#cmp-sel-invert').addEventListener('click', () => {
  if (!ST.cmpData) return;
  const next = new Set();
  for (const x of ST.cmpData.rows) {
    if (!ST.cmpSel.has(x.path)) next.add(x.path);
  }
  ST.cmpSel = next; renderCompare(); _selChanged();
});
$('#cmp-sel-header').addEventListener('click', e => {
  if (e.target.checked) $('#cmp-sel-all').click();
  else $('#cmp-sel-none').click();
});

// Compute precision / recall — extracted so we can also call it automatically
async function loadQuality(silent) {
  const btn = $('#cmp-quality');
  const defaultLabel = 'Compute precision / recall vs baseline';
  btn.textContent = 'computing precision/recall… (streams every correlated.csv)';
  btn.disabled = true;
  try {
    // Pass the CURRENTLY-SELECTED baseline path so precision/recall is
    // computed against the same reference the table's ×baseline uses.
    // Without this the server fell back to its own default (which could
    // differ from the user's dropdown pick → quality vs the wrong run).
    const blPath = ST.cmpData?.baseline_path;
    const q = await api('/api/compare_quality',
                        blPath ? {baseline: blPath} : undefined);
    ST.cmpQuality = {};
    for (const r of q.rows) ST.cmpQuality[r.path] = r;
    const base = q.baseline.split('/').slice(-2).join('/');
    const cacheNote = (q.cache_hits != null && q.cache_misses != null && (q.cache_hits + q.cache_misses) > 0)
      ? ` · cache: ${q.cache_hits} hits / ${q.cache_misses} new`
      : '';
    btn.textContent = `precision/recall · baseline = ${base} (${q.n_baseline_pairs.toLocaleString()} pairs)${cacheNote} · click to recompute`;
    renderCompare();
    // Re-render charts too — the trade-off scatter (and any other
    // quality-dependent chart) is built BEFORE loadQuality finishes on
    // first launch, so its rows have no F1/precision/recall yet and the
    // chart falls back to "no rows have both axes populated". Re-rendering
    // now that ST.cmpQuality is populated unblocks it.
    if (ST.cmpData) loadCharts();
  } catch (e) {
    if (!silent) console.error('loadQuality failed', e);
    btn.textContent = 'precision/recall failed: ' + String(e).slice(0, 80);
    setTimeout(() => { btn.textContent = defaultLabel; }, 5000);
  } finally {
    btn.disabled = false;
  }
}
$('#cmp-quality').addEventListener('click', e => {
  if (e.shiftKey) {
    // Shift+click → wipe the disk cache, then recompute everything from scratch
    fetch('/api/quality_cache/clear', {method:'POST'})
      .then(() => loadQuality(false));
  } else {
    loadQuality(false);
  }
});
$$('#cmp-table th[data-sort]').forEach(th =>
  th.addEventListener('click', e => _onSortClick(e, th.dataset.sort)));

// ---------- PAIRS tab ----------
// Pretty-print a lag value (in hours) as a compact d/h or w/d string.
function _fmtLagHM(lag) {
  const sign = lag < 0 ? '−' : lag > 0 ? '+' : '';
  const abs = Math.abs(lag);
  if (abs === 0) return '0';
  if (abs >= 24*7) {
    const w = Math.floor(abs / (24*7));
    const d = Math.floor((abs % (24*7)) / 24);
    return d === 0 ? `${sign}${w}w` : `${sign}${w}w ${d}d`;
  }
  if (abs >= 24) {
    const d = Math.floor(abs / 24);
    const h = abs % 24;
    return h === 0 ? `${sign}${d}d` : `${sign}${d}d ${h}h`;
  }
  return `${sign}${abs}h`;
}
function pairsFilters() {
  const f = {
    pipeline: ST.pipeline,
    cover: $('#p-cover')?.checked ? 1 : 0,
    id_q: $('#p-q').value.trim(),
    min_corr: $('#p-mincorr').value,
    max_corr: $('#p-maxcorr').value,
    lag_min: $('#p-lagmin').value,
    lag_max: $('#p-lagmax').value,
    exclude_self: $('#p-noself').checked ? 1 : 0,
    only_self: $('#p-onlyself').checked ? 1 : 0,
    sort: ST.pairsState.sort,
    order: ST.pairsState.order,
    offset: ST.pairsState.offset,
    limit: ST.pairsState.limit,
  };
  if (ST.pairsState.anchor) {
    f.anchor_id = ST.pairsState.anchor.id;
    if (ST.pairsState.anchor.t != null) f.anchor_t = ST.pairsState.anchor.t;
  }
  return f;
}
function setAnchor(id, t) {
  // t === null → anchor on the airport id only (any window of that airport).
  ST.pairsState.anchor = {id, t: (t == null || Number.isNaN(t)) ? null : t};
  ST.pairsState.offset = 0;
  ST.pairsState.selected.clear();
  ST.pairsState.sort = 'abs_corr'; ST.pairsState.order = 'desc';
  renderAnchorChip();
  searchPairs();
  $('#tab-pairs').scrollIntoView({behavior:'smooth', block:'start'});
}
function clearAnchor() {
  ST.pairsState.anchor = null;
  ST.pairsState.offset = 0;
  ST.pairsState.selected.clear();
  renderAnchorChip();
  searchPairs();
}
function renderAnchorChip() {
  const host = $('#p-anchor');
  const a = ST.pairsState.anchor;
  if (!a) { host.style.display = 'none'; host.innerHTML = ''; return; }
  host.style.display = '';
  const isExact = a.t != null;
  const label = isExact ? `${a.id}@${a.t}` : `${a.id} (any window)`;
  const scope = isExact
    ? 'pairs touching this exact window'
    : 'pairs involving any window of this airport';
  host.innerHTML = `<span class="chip warn" style="font-size:11.5px; padding:3px 8px;">
    Showing ${scope}: <b>${escapeHtml(label)}</b>
    <a href="#" id="anchor-clear" style="margin-left:8px; color:var(--accent); text-decoration:none">× clear</a>
    ${isExact ? '<a href="#" id="anchor-add" style="margin-left:8px; color:var(--ok); text-decoration:none">+ add this window to basket</a>' : ''}
  </span>`;
  $('#anchor-clear').addEventListener('click', e => { e.preventDefault(); clearAnchor(); });
  if (isExact) {
    $('#anchor-add').addEventListener('click', e => {
      e.preventDefault();
      addToBasket({id:a.id, t:a.t, label:`${a.id}@${a.t}`, kind:'corr'});
    });
  }
}
async function searchPairs() {
  const tb = $('#pairs-table tbody');
  const cov = $('#p-cover')?.checked;
  const msg = cov
    ? 'Streaming correlated.csv… <span class="muted">(coverage enabled — first run may take ~1m to scan all pipelines)</span>'
    : 'Streaming correlated.csv…';
  tb.innerHTML = `<tr><td colspan="12" class="spin">${msg}</td></tr>`;
  try {
    const r = await api('/api/correlated', pairsFilters());
    if (!r.rows.length) {
      tb.innerHTML = '<tr><td colspan="12" class="empty">No match.</td></tr>';
    } else {
      // selection persists across the current page; reset on new search/page
      tb.innerHTML = r.rows.map((row, ri) => {
        const c = Number(row.corr).toFixed(4);
        const ac = Math.abs(row.corr).toFixed(4);
        const cls = (Math.abs(row.corr) >= 0.95) ? 'ok' : (Math.abs(row.corr) >= 0.85) ? 'warn' : 'bad';
        const lagCls = row.lag > 0 ? 'lag-pos' : row.lag < 0 ? 'lag-neg' : 'lag-zero';
        const signSym = row.corr < 0 ? '−' : '+';
        const a = {id:row.id1, t:row.t1, label:`${row.id1}@${row.t1}`, kind:'corr'};
        const b = {id:row.id2, t:row.t2, label:`${row.id2}@${row.t2}`, kind:'corr'};
        // coverage cell: count chip + copy button
        let covCell = '<td class="num"><span class="muted">—</span></td>';
        if (row.coverage) {
          const nd = row.coverage.n_detected;
          const nt = row.coverage.n_runs_total;
          const ratio = nt ? nd / nt : 0;
          const covCls = ratio >= 0.9 ? 'ok' : ratio >= 0.5 ? 'warn' : ratio >= 0.2 ? 'warn' : 'bad';
          covCell = `<td class="num"><span style="display:inline-flex; gap:3px; align-items:center">
            <button class="small chip ${covCls} cov-toggle" title="${nd} of ${nt} engines detect this pair (click to expand)" data-cov-row="${ri}">${nd}/${nt}</button>
            <button class="small ghost cov-copy" title="copy detector list to clipboard" data-cov-row="${ri}" style="padding:2px 5px;">📋</button>
          </span></td>`;
        }
        const isSel = ST.pairsState.selected.has(ri);
        return `<tr data-pair='${JSON.stringify({a,b})}' data-cov-key="${ri}" data-row-idx="${ri}" class="${isSel ? 'sel-row' : ''}">
          <td class="sel-cell"><input type="checkbox" class="row-pair-sel" ${isSel?'checked':''} data-row-idx="${ri}"/></td>
          <td class="add-cell" data-side="a" title="click to add window A to basket"><button class="small ghost" title="filter to all pairs involving airport ${escapeHtml(row.id1)} (any window)" data-anchor='${JSON.stringify({id:row.id1, t:null})}'>🔍</button> ${escapeHtml(row.id1)}</td>
          <td class="num add-cell" data-side="a" title="click to add window A to basket">${row.t1} <button class="small ghost" title="filter to pairs touching this exact window: ${escapeHtml(row.id1)}@${row.t1}" data-anchor='${JSON.stringify({id:row.id1, t:row.t1})}' style="padding:1px 4px;">🔍</button></td>
          <td class="add-cell" data-side="b" title="click to add window B to basket"><button class="small ghost" title="filter to all pairs involving airport ${escapeHtml(row.id2)} (any window)" data-anchor='${JSON.stringify({id:row.id2, t:null})}'>🔍</button> ${escapeHtml(row.id2)}</td>
          <td class="num add-cell" data-side="b" title="click to add window B to basket">${row.t2} <button class="small ghost" title="filter to pairs touching this exact window: ${escapeHtml(row.id2)}@${row.t2}" data-anchor='${JSON.stringify({id:row.id2, t:row.t2})}' style="padding:1px 4px;">🔍</button></td>
          <td class="num ${lagCls}" title="raw lag = ${row.lag} time units (hours)">
            <div>${_fmtLagHM(row.lag)}</div>
            <div class="muted" style="font-size:9.5px; line-height:1">${row.lag>0?'+':''}${row.lag}</div>
          </td>
          <td class="num"><span class="chip ${cls}">${ac}</span></td>
          <td class="num">${signSym}${c.replace('-','')}</td>
          ${covCell}
          <td class="actions"><button class="small" title="add window A only" data-side-add="a">+</button></td>
          <td class="actions"><button class="small" title="add window B only" data-side-add="b">+</button></td>
          <td class="actions"><button class="small primary" title="add both windows of this pair" data-pair-add>+ pair</button></td>
        </tr>`;
      }).join('');
      ST._lastPairsRows = r.rows;
      // Cache the coverage info so the expansion can access it
      ST._lastCovByRow = new Map(r.rows.map((row, ri) => [String(ri), row.coverage]));
      tb.querySelectorAll('tr[data-pair]').forEach(tr => {
        const p = JSON.parse(tr.dataset.pair);
        tr.querySelector('[data-pair-add]').addEventListener('click', e => {
          e.stopPropagation(); addPair(p.a, p.b, tr);
        });
        tr.querySelectorAll('[data-side-add]').forEach(btn => {
          btn.addEventListener('click', e => {
            e.stopPropagation();
            const side = btn.dataset.sideAdd;
            addToBasket(side === 'a' ? p.a : p.b);
            flashRow(tr);
          });
        });
        tr.querySelectorAll('[data-anchor]').forEach(btn =>
          btn.addEventListener('click', e => {
            e.stopPropagation();
            const a = JSON.parse(btn.dataset.anchor);
            setAnchor(a.id, a.t);
          }));
        const covBtn = tr.querySelector('.cov-toggle');
        if (covBtn) covBtn.addEventListener('click', e => {
          e.stopPropagation();
          toggleCoverageRow(tr, covBtn.dataset.covRow);
        });
        const cpyBtn = tr.querySelector('.cov-copy');
        if (cpyBtn) cpyBtn.addEventListener('click', async e => {
          e.stopPropagation();
          await copyCoverageRow(cpyBtn);
        });
        const rsel = tr.querySelector('.row-pair-sel');
        if (rsel) rsel.addEventListener('click', e => {
          e.stopPropagation();
          const ri = parseInt(rsel.dataset.rowIdx, 10);
          if (rsel.checked) ST.pairsState.selected.add(ri);
          else ST.pairsState.selected.delete(ri);
          tr.classList.toggle('sel-row', rsel.checked);
          renderPairsToolbar();
        });
        // Click routing on the row:
        //   - Cell A (id/t) → add window A only
        //   - Cell B (id/t) → add window B only
        //   - Anywhere else (lag, |corr|, signed, by) → add both
        //   - sel-cell / buttons → handled above, never reach here
        tr.addEventListener('click', e => {
          if (e.target.closest('.sel-cell')) return;
          if (e.target.closest('button')) return;
          const sideCell = e.target.closest('.add-cell');
          if (sideCell) {
            const side = sideCell.dataset.side;
            addToBasket(side === 'a' ? p.a : p.b);
            flashRow(tr);
            return;
          }
          addPair(p.a, p.b, tr);
        });
      });
      // refresh header checkbox tristate + toolbar + basket indicators
      renderPairsToolbar();
      refreshPairsBasketIndicators();
    }
    const total = r.total;
    const off = ST.pairsState.offset, lim = ST.pairsState.limit;
    const last = Math.min(off + lim, total);
    $('#pairs-pager').innerHTML = `
      <span>${total.toLocaleString()} matches · showing ${total ? (off+1) : 0}–${last}</span>
      <span class="grow"></span>
      <button class="small" id="pg-prev">‹ prev</button>
      <button class="small" id="pg-next">next ›</button>`;
    $('#pg-prev').onclick = () => { ST.pairsState.offset = Math.max(0, off - lim); ST.pairsState.selected.clear(); searchPairs(); };
    $('#pg-next').onclick = () => { if (last < total) { ST.pairsState.offset = last; ST.pairsState.selected.clear(); searchPairs(); } };
  } catch (e) {
    tb.innerHTML = `<tr><td colspan="12" class="empty">Error: ${escapeHtml(String(e))}</td></tr>`;
  }
}
function toggleCoverageRow(tr, key) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains('cov-expand')) {
    next.remove();
    return;
  }
  const cov = ST._lastCovByRow && ST._lastCovByRow.get(key);
  if (!cov) return;
  const chipClass = m => ({bf:'warn', corrtrack:'ok', filcorr:'info'}[m] || '');
  const modeCounts = {};
  for (const d of cov.detected) modeCounts[d.mode] = (modeCounts[d.mode]||0) + 1;
  const modeBreakdown = Object.entries(modeCounts)
    .map(([m, n]) => `<span class="chip ${chipClass(m)}" style="margin:0 4px">${n} ${m}</span>`)
    .join(' ');
  const chips = cov.detected
    .sort((a,b) => a.label.localeCompare(b.label))
    .map(d => `<span class="chip ${chipClass(d.mode)} cov-pipe" data-pipe-path="${escapeHtml(d.path)}" style="margin:2px; cursor:pointer" title="click to switch to this pipeline">${escapeHtml(d.label)}</span>`)
    .join(' ');
  const row = document.createElement('tr');
  row.className = 'cov-expand';
  row.innerHTML = `<td colspan="12" style="background:var(--panel); padding:8px 12px">
    <div style="font-size:11.5px; margin-bottom:6px">
      <b>${cov.n_detected}</b> of <b>${cov.n_runs_total}</b> engines detect this pair · ${modeBreakdown || '<span class="muted">no detection</span>'}
    </div>
    <div style="font-size:11px">${chips}</div>
  </td>`;
  tr.insertAdjacentElement('afterend', row);
  // wire chip click to switch active pipeline
  row.querySelectorAll('.cov-pipe').forEach(c =>
    c.addEventListener('click', async () => {
      await setActivePipeline(c.dataset.pipePath);
      searchPairs();
    }));
}
async function copyCoverageRow(btn) {
  const key = btn.dataset.covRow;
  const cov = ST._lastCovByRow && ST._lastCovByRow.get(key);
  if (!cov || !cov.detected) return;
  const lines = cov.detected
    .slice()
    .sort((a, b) => (a.mode || '').localeCompare(b.mode || '') || a.label.localeCompare(b.label))
    .map(d => `${d.mode}\t${d.label}\t${d.path}`);
  const header = `# ${cov.n_detected} of ${cov.n_runs_total} engines detected this pair`;
  const text = [header, '# mode\tlabel\tpath', ...lines].join('\n');
  const orig = btn.textContent;
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = '✓';
    setTimeout(() => { btn.textContent = orig; }, 1200);
  } catch (e) {
    // Fallback for older browsers / non-https contexts
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); btn.textContent = '✓'; } catch {}
    document.body.removeChild(ta);
    setTimeout(() => { btn.textContent = orig; }, 1200);
  }
}

function renderPairsToolbar() {
  const sel = ST.pairsState.selected;
  const bar = $('#pairs-multi');
  const head = $('#pairs-sel-all');
  // sync header checkbox: checked if all current rows are selected; indeterminate if some
  const rowsOnPage = (ST._lastPairsRows || []).length;
  const nSel = sel.size;
  if (head) {
    head.checked = (nSel > 0 && nSel >= rowsOnPage);
    head.indeterminate = (nSel > 0 && nSel < rowsOnPage);
  }
  if (!nSel) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';
  bar.innerHTML = `
    <span><b>${nSel}</b> row${nSel>1?'s':''} selected</span>
    <button class="small primary" id="msel-Bs">+ add ${nSel} window${nSel>1?'s':''} B to basket</button>
    <button class="small" id="msel-pairs">+ add A+B (${nSel*2} windows)</button>
    <div style="flex:1"></div>
    <button class="small ghost" id="msel-clear">clear selection</button>`;
  $('#msel-Bs').addEventListener('click', async () => {
    await addMultiRows(/*aSide*/ false, /*bSide*/ true);
  });
  $('#msel-pairs').addEventListener('click', async () => {
    await addMultiRows(true, true);
  });
  $('#msel-clear').addEventListener('click', () => {
    ST.pairsState.selected.clear();
    document.querySelectorAll('#pairs-table .row-pair-sel').forEach(c => { c.checked = false; });
    document.querySelectorAll('#pairs-table tr.sel-row').forEach(tr => tr.classList.remove('sel-row'));
    renderPairsToolbar();
  });
}
async function addMultiRows(aSide, bSide) {
  const rows = ST._lastPairsRows || [];
  let added = false;
  for (const ri of ST.pairsState.selected) {
    const r = rows[ri];
    if (!r) continue;
    if (aSide) {
      added = (await addToBasket({id:r.id1, t:r.t1, label:`${r.id1}@${r.t1}`, kind:'corr'}, {silent:true})) || added;
    }
    if (bSide) {
      added = (await addToBasket({id:r.id2, t:r.t2, label:`${r.id2}@${r.t2}`, kind:'corr'}, {silent:true})) || added;
    }
  }
  if (added) await replot();
}

$('#p-search').addEventListener('click', () => { ST.pairsState.offset = 0; ST.pairsState.selected.clear(); searchPairs(); });
// Quick anchor by airport (+ optional t)
$('#p-quick-anchor')?.addEventListener('click', () => {
  const id = $('#p-quick-airport')?.value;
  if (!id) { alert('Pick an airport (station id) first'); return; }
  const t = parseInt($('#p-quick-t')?.value, 10);
  if (Number.isFinite(t)) {
    setAnchor(id, t);
  } else {
    // No t given → use the substring filter for "any window of this airport"
    ST.pairsState.anchor = null;
    ST.pairsState.offset = 0;
    ST.pairsState.selected.clear();
    const inp = $('#p-q'); if (inp) inp.value = id;
    renderAnchorChip();
    searchPairs();
  }
});
// header checkbox: select/deselect all visible
document.addEventListener('change', e => {
  if (e.target && e.target.id === 'pairs-sel-all') {
    const rows = ST._lastPairsRows || [];
    if (e.target.checked) {
      for (let i = 0; i < rows.length; i++) ST.pairsState.selected.add(i);
    } else {
      ST.pairsState.selected.clear();
    }
    document.querySelectorAll('#pairs-table .row-pair-sel').forEach(c => {
      const ri = parseInt(c.dataset.rowIdx, 10);
      c.checked = ST.pairsState.selected.has(ri);
      const tr = c.closest('tr');
      if (tr) tr.classList.toggle('sel-row', c.checked);
    });
    renderPairsToolbar();
  }
});
$$('#pairs-table th[data-sort]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.sort;
  if (ST.pairsState.sort === k) {
    ST.pairsState.order = (ST.pairsState.order === 'desc') ? 'asc' : 'desc';
  } else { ST.pairsState.sort = k; ST.pairsState.order = 'desc'; }
  $$('#pairs-table th').forEach(x => x.classList.remove('active'));
  th.classList.add('active');
  ST.pairsState.offset = 0;
  searchPairs();
}));

// ---------- WINDOWS tab ----------
let WINDOWS_CACHE = [];
async function searchWindows() {
  const tb = $('#windows-table tbody');
  tb.innerHTML = '<tr><td colspan="6" class="spin">Aggregating windows…</td></tr>';
  try {
    const r = await api('/api/windows', {
      pipeline: ST.pipeline,
      id_q: $('#w-q').value.trim(),
      top_n: $('#w-top').value || 200,
    });
    WINDOWS_CACHE = r.windows;
    renderWindows();
  } catch (e) {
    tb.innerHTML = `<tr><td colspan="6" class="empty">Error: ${escapeHtml(String(e))}</td></tr>`;
  }
}
function renderWindows() {
  const tb = $('#windows-table tbody');
  const s = ST.windowsState.sort, asc = ST.windowsState.order === 'asc';
  const arr = WINDOWS_CACHE.slice().sort((a,b) => {
    const av = a[s], bv = b[s];
    if (av === bv) return 0;
    return (av < bv ? -1 : 1) * (asc ? 1 : -1);
  });
  if (!arr.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">No window matches.</td></tr>';
    return;
  }
  tb.innerHTML = arr.map(w => {
    const cls = (w.max_abs_corr >= 0.95) ? 'ok' : (w.max_abs_corr >= 0.85) ? 'warn' : 'bad';
    return `<tr>
      <td>${escapeHtml(w.id)}</td><td class="num">${w.t}</td>
      <td class="num">${w.n_partners}</td>
      <td class="num"><span class="chip ${cls}">${w.max_abs_corr.toFixed(4)}</span></td>
      <td class="num">${w.mean_abs_corr.toFixed(4)}</td>
      <td class="actions">
        <button class="small primary" data-add='${JSON.stringify({id:w.id,t:w.t,label:`${w.id}@${w.t}`,kind:'corr'})}'>+ add</button>
        <button class="small" data-partners='${JSON.stringify({id:w.id,t:w.t})}'>partners…</button>
      </td>
    </tr>`;
  }).join('');
  tb.querySelectorAll('button[data-add]').forEach(b =>
    b.addEventListener('click', () => addToBasket(JSON.parse(b.dataset.add))));
  tb.querySelectorAll('button[data-partners]').forEach(b =>
    b.addEventListener('click', () => showPartners(JSON.parse(b.dataset.partners))));
}
$('#w-search').addEventListener('click', searchWindows);
$$('#windows-table th[data-sort]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.sort;
  if (ST.windowsState.sort === k) {
    ST.windowsState.order = (ST.windowsState.order === 'desc') ? 'asc' : 'desc';
  } else { ST.windowsState.sort = k; ST.windowsState.order = 'desc'; }
  $$('#windows-table th').forEach(x => x.classList.remove('active'));
  th.classList.add('active');
  renderWindows();
}));
async function showPartners(w) {
  const r = await api('/api/partners', {pipeline: ST.pipeline, id: w.id, t: w.t});
  // open a tiny modal-ish popover inside the windows tab
  const tb = $('#windows-table tbody');
  const html = `<tr><td colspan="6" style="background:var(--panel)">
    <b>Partners of ${escapeHtml(w.id)}@${w.t}</b> (${r.partners.length}):<br>
    ${r.partners.slice(0,40).map(p => `
      <span class="chip" style="margin:2px; cursor:pointer"
        data-partner='${JSON.stringify({id:p.id,t:p.t,label:`${p.id}@${p.t}`,kind:'partner'})}'
      >${escapeHtml(p.id)}@${p.t} · ${p.corr.toFixed(3)} (lag ${p.lag}) +</span>`).join(' ')}
    ${r.partners.length > 40 ? `<span class="muted">…and ${r.partners.length-40} more</span>` : ''}
  </td></tr>`;
  tb.insertAdjacentHTML('afterbegin', html);
  tb.querySelectorAll('[data-partner]').forEach(c =>
    c.addEventListener('click', () => addToBasket(JSON.parse(c.dataset.partner))));
}

// ---------- FREE tab ----------
$('#free-add').addEventListener('click', () => {
  const id = $('#free-id').value;
  const t = parseInt($('#free-t').value, 10);
  if (!id || !Number.isFinite(t)) { alert('pick id and t'); return; }
  addToBasket({id, t, label: id+'@'+t+' (free)', kind: 'free'});
});

// ---------- Map view (basket stations) ----------
ST.airportCache = {};   // ICAO → {lat,lon,name,...}
function _icaoOf(stationId) {
  const parts = String(stationId).split('_');
  return (parts[parts.length-1] || '').toUpperCase();
}
function _haversineKm(a, b) {
  const R = 6371;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(b.lat - a.lat), dLon = toRad(b.lon - a.lon);
  const s = Math.sin(dLat/2)**2 + Math.cos(toRad(a.lat))*Math.cos(toRad(b.lat))*Math.sin(dLon/2)**2;
  return 2 * R * Math.asin(Math.sqrt(s));
}
async function _fetchAirportsFor(icaos) {
  const need = icaos.filter(c => !(c in ST.airportCache));
  if (!need.length) return;
  try {
    const r = await api('/api/airports', {icaos: need.join(',')});
    for (const [k, v] of Object.entries(r.airports || {})) ST.airportCache[k] = v;
    // Mark missing ones explicitly with null so we don't re-fetch
    for (const m of (r.missing || [])) ST.airportCache[m] = null;
  } catch (e) { /* ignore */ }
}
async function drawMap() {
  const host = $('#map-host');
  const sec  = $('#map-section');
  if (!sec || !host) return;
  if (!ST.basket.length) {
    sec.style.display = 'none';
    if (ST.leafletMap) { ST.leafletMap.remove(); ST.leafletMap = null; }
    return;
  }
  sec.style.display = '';
  if (typeof L === 'undefined') {
    host.innerHTML = '<div class="empty">leaflet not loaded — check /static/leaflet.js</div>';
    return;
  }
  // Resolve ICAOs of basket entries
  const icaos = [...new Set(ST.basket.map(w => _icaoOf(w.id)))];
  await _fetchAirportsFor(icaos);
  const markers = ST.basket.map(w => {
    const icao = _icaoOf(w.id);
    const ap = ST.airportCache[icao];
    return ap ? {w, icao, ap, lat: ap.lat, lon: ap.lon} : {w, icao, ap: null, missing: true};
  });
  const known = markers.filter(m => m.ap);
  const missing = markers.filter(m => m.missing);
  if (!known.length) {
    host.innerHTML = `<div class="empty">no airport coords for the current basket (${missing.map(m => m.icao).join(', ')})</div>`;
    return;
  }
  $('#map-status').textContent =
    `${known.length} markers · ${[...new Set(known.map(k=>k.icao))].length} airports`
    + (missing.length ? ' · ⚠ ' + missing.length + ' missing: ' + missing.map(m=>m.icao).join(', ') : '');

  // Initialise the Leaflet map once
  if (!ST.leafletMap) {
    ST.leafletMap = L.map(host, {zoomControl: true, scrollWheelZoom: true,
                                  attributionControl: true});
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19, minZoom: 2,
    }).addTo(ST.leafletMap);
    ST.leafletLayers = L.layerGroup().addTo(ST.leafletMap);
  }
  ST.leafletLayers.clearLayers();

  // Group markers by ICAO so multiple windows at same airport stack as separate circles
  const byIcao = {};
  for (const k of known) (byIcao[k.icao] = byIcao[k.icao] || []).push(k);

  // Markers
  for (const [icao, group] of Object.entries(byIcao)) {
    const ap = group[0].ap;
    group.forEach((g, i) => {
      const offsetMeters = i * 60;   // ~60m offset per stacked window to avoid overlap at low zoom
      // Convert offset meters → degrees approximately (1° ≈ 111km)
      const offLat = 0;
      const offLon = (offsetMeters / 1000) / (111 * Math.cos(ap.lat * Math.PI / 180));
      const m = L.circleMarker([ap.lat, ap.lon + offLon * (i - (group.length-1)/2)], {
        radius: 8,
        fillColor: g.w.color || '#5aa1ff', color: '#fff',
        weight: 2, opacity: 1, fillOpacity: 0.92,
      });
      const lines = [
        `<b>${ap.name}</b>`,
        `<div style="font-family:monospace; font-size:11px; color:#5aa1ff">ICAO ${icao}</div>`,
        `<div style="font-size:11px">${ap.municipality || ''}</div>`,
        `<div style="font-size:11px; color:#aaa">${ap.lat.toFixed(4)}, ${ap.lon.toFixed(4)} · elev ${ap.elevation_ft || '?'} ft</div>`,
        `<div style="font-size:11px; margin-top:4px">window: <code>${g.w.id}@${g.w.t}</code></div>`,
      ];
      m.bindPopup(lines.join(''));
      m.bindTooltip(`${ap.municipality || icao} (${icao})`, {direction: 'top', offset: [0, -8]});
      m.addTo(ST.leafletLayers);
    });
  }

  // Pairwise distance lines
  if ($('#map-show-lines').checked && known.length >= 2) {
    const seenPairs = new Set();
    for (let i = 0; i < known.length; i++) {
      for (let j = i + 1; j < known.length; j++) {
        const A = known[i], B = known[j];
        if (A.icao === B.icao) continue;
        const pairKey = [A.icao, B.icao].sort().join('|');
        if (seenPairs.has(pairKey)) continue;
        seenPairs.add(pairKey);
        const dist = _haversineKm(A, B);
        const ln = L.polyline([[A.lat, A.lon], [B.lat, B.lon]], {
          color: '#5aa1ff', weight: 2, opacity: 0.6, dashArray: '4 4',
        });
        ln.bindTooltip(
          `<b>${A.ap.name} ↔ ${B.ap.name}</b><br>` +
          `<span style="font-family:monospace">${A.icao} — ${B.icao}</span><br>` +
          `<b>${dist.toFixed(1)} km</b> (great-circle)`,
          {sticky: true}
        );
        ln.addTo(ST.leafletLayers);
        // Midpoint label
        if ($('#map-show-labels').checked) {
          const mlat = (A.lat + B.lat) / 2, mlon = (A.lon + B.lon) / 2;
          L.marker([mlat, mlon], {
            icon: L.divIcon({
              className: 'dist-label',
              html: `<span>${dist.toFixed(0)} km</span>`,
              iconSize: null,
            }),
            interactive: false,
          }).addTo(ST.leafletLayers);
        }
      }
    }
  }

  // Fit bounds with padding
  const latlngs = known.map(k => [k.lat, k.lon]);
  const bounds = L.latLngBounds(latlngs);
  if (bounds.isValid()) {
    ST.leafletMap.fitBounds(bounds, {padding: [40, 40], maxZoom: 10});
  }
  // Force redraw after layout settles (leaflet sometimes mis-sizes when section was display:none)
  setTimeout(() => ST.leafletMap && ST.leafletMap.invalidateSize(), 50);
}

// ---------- Basket ----------
async function addToBasket(w, opts) {
  opts = opts || {};
  if (ST.basket.find(x => x.id === w.id && x.t === w.t)) return false;
  w.color = COLORS[ST.basket.length % COLORS.length];
  ST.basket.push(w);
  renderBasket();
  refreshPairsBasketIndicators();
  drawMap();
  if (!opts.silent) await replot();
  return true;
}
async function addPair(a, b, tr) {
  const ra = await addToBasket(a, {silent:true});
  const rb = await addToBasket(b, {silent:true});
  if (ra || rb) await replot();
  if (tr) flashRow(tr);
}
function flashRow(tr) {
  tr.classList.add('selected');
  setTimeout(() => tr.classList.remove('selected'), 700);
}

// Persistent "in-basket" markers on the pairs table — connects each row visually
// to the corresponding curve color in the plot.
function refreshPairsBasketIndicators() {
  const tbl = document.getElementById('pairs-table');
  if (!tbl) return;
  const map = new Map();   // 'id|t' → basket entry (color, etc.)
  for (const w of ST.basket) map.set(`${w.id}|${w.t}`, w);
  tbl.querySelectorAll('tr[data-pair]').forEach(tr => {
    let pair;
    try { pair = JSON.parse(tr.dataset.pair); } catch { return; }
    const inA = map.get(`${pair.a.id}|${pair.a.t}`);
    const inB = map.get(`${pair.b.id}|${pair.b.t}`);
    // Wipe any existing dots & in-basket classes first
    tr.querySelectorAll('.basket-dot').forEach(d => d.remove());
    tr.querySelectorAll('.add-cell').forEach(c => c.classList.remove('in-basket'));
    if (inA) {
      const aCells = tr.querySelectorAll('.add-cell[data-side="a"]');
      aCells.forEach(c => c.classList.add('in-basket'));
      // Put the dot in the FIRST a-cell, right after the 🔍 button (= after id text)
      const dot = document.createElement('span');
      dot.className = 'basket-dot';
      dot.style.background = inA.color;
      dot.title = `in basket as ${inA.label || inA.id+'@'+inA.t}`;
      aCells[0]?.appendChild(dot);
    }
    if (inB) {
      const bCells = tr.querySelectorAll('.add-cell[data-side="b"]');
      bCells.forEach(c => c.classList.add('in-basket'));
      const dot = document.createElement('span');
      dot.className = 'basket-dot';
      dot.style.background = inB.color;
      dot.title = `in basket as ${inB.label || inB.id+'@'+inB.t}`;
      bCells[0]?.appendChild(dot);
    }
    tr.classList.toggle('pair-in-basket', !!(inA && inB));
  });
}
function removeFromBasket(idx) {
  ST.basket.splice(idx, 1);
  ST.basket.forEach((w, i) => w.color = COLORS[i % COLORS.length]);
  renderBasket();
  refreshPairsBasketIndicators();
  drawMap();
  replot();
}
function renderBasket() {
  const root = $('#basket');
  $('#sel-count').textContent = ST.basket.length;
  if (!ST.basket.length) { root.innerHTML = '<div class="empty">No windows yet.</div>'; return; }
  root.innerHTML = ST.basket.map((w, i) => `
    <div class="basket-item">
      <div class="head">
        <div class="lab"><span class="swatch" style="background:${w.color}"></span>${escapeHtml(w.id)}</div>
        <button class="small ghost" data-rm="${i}">×</button>
      </div>
      <div class="basket-date" data-date="${i}"><span class="muted">loading date…</span></div>
      <div class="stat" data-stats="${i}"><span class="muted">loading stats…</span></div>
      <div class="row" style="margin-top:6px; gap:4px">
        <span class="chip ${w.kind==='free'?'warn':w.kind==='partner'?'info':'ok'}">${w.kind}</span>
        <span class="chip" title="raw hour offset stored in correlated.csv">t=${w.t}</span>
      </div>
    </div>`).join('');
  root.querySelectorAll('button[data-rm]').forEach(b =>
    b.addEventListener('click', () => removeFromBasket(parseInt(b.dataset.rm,10))));
  // fetch stats + date per item
  ST.basket.forEach(async (w, i) => {
    try {
      const r = await api('/api/window', {pipeline: ST.pipeline, id: w.id, t: w.t});
      const s = r.stats;
      const el = root.querySelector(`[data-stats="${i}"]`);
      if (el) el.textContent = `μ=${s.mean?.toFixed(3) ?? '?'} σ=${s.std?.toFixed(3) ?? '?'} ` +
                               `min=${s.min?.toFixed(2) ?? '?'} max=${s.max?.toFixed(2) ?? '?'}`;
      const de = root.querySelector(`[data-date="${i}"]`);
      if (de && r.date_start_ms) {
        const dur = (r.date_end_ms - r.date_start_ms);
        const days = Math.round(dur / 86400000);
        const startFmt = _fmtDate(r.date_start_ms, days >= 1 ? 'datetime' : 'hour');
        const endFmt = _fmtDate(r.date_end_ms, days >= 1 ? 'datetime' : 'hour');
        de.innerHTML = `<span title="window covers ${days} day${days>1?'s':''}">${escapeHtml(startFmt)} → ${escapeHtml(endFmt)}</span>`;
      }
    } catch(e) { /* ignore */ }
  });
}
$('#clear-sel').addEventListener('click', () => { ST.basket = []; renderBasket(); refreshPairsBasketIndicators(); drawMap(); replot(); });
$('#replot').addEventListener('click', () => replot());
['opt-normalize','opt-align','opt-history','opt-points','opt-show-pad','opt-pad','opt-grid'].forEach(id =>
  $('#'+id).addEventListener('change', () => replot()));
['map-show-lines','map-show-labels'].forEach(id =>
  document.addEventListener('change', e => { if (e.target?.id === id) drawMap(); }));

// ---------- Plot ----------
async function replot() {
  const host = $('#plot'), empty = $('#plot-empty');
  if (!ST.basket.length) {
    host.innerHTML = '';
    host.style.display = 'none';
    empty.style.display = '';
    $('#corr-matrix').innerHTML = '';
    $('#plot-status').textContent = '';
    return;
  }
  empty.style.display = 'none';
  host.style.display = '';
  $('#plot-status').textContent = 'fetching…';
  try {
    const data = await api('/api/plot_data', {
      pipeline: ST.pipeline,
      w: ST.basket.map(w => `${w.id}|${w.t}|${w.color}`),
      align: $('#opt-align').checked ? 1 : 0,
      history: $('#opt-history').checked ? 1 : 0,
      pad: $('#opt-show-pad').checked ? ($('#opt-pad').value || 0) : 0,
    });
    drawTimeseries(host, data, {
      normalize: $('#opt-normalize').checked,
      points: $('#opt-points').checked,
      grid: $('#opt-grid').checked,
    });
    $('#plot-status').textContent = 'done';
  } catch (e) {
    host.innerHTML = `<div class="empty">plot error: ${escapeHtml(String(e))}</div>`;
    $('#plot-status').textContent = 'failed';
  }
  // also fetch pairwise corr matrices (separate section below)
  const sec = $('#corr-section');
  if (ST.basket.length >= 2) {
    sec.style.display = '';
    $('#corr-status').textContent = 'computing…';
    try {
      const params = {pipeline: ST.pipeline, w: ST.basket.map(w => `${w.id}|${w.t}`)};
      if (ST.showBestLag) params.best_lag = 1;
      const r = await api('/api/corr_matrix', params);
      ST.lastCorr = r;
      $('#corr-status').textContent = `${ST.basket.length} windows · length ${r.window_size}`;
      renderCorrPanel();
    } catch(e) {
      $('#corr-matrix').innerHTML = `<div class="empty">Error: ${escapeHtml(String(e))}</div>`;
      $('#corr-status').textContent = '';
    }
  } else {
    sec.style.display = 'none';
    ST.lastCorr = null;
  }
}
ST.corrMethod = 'pearson';
ST.showBestLag = false;
ST.lastCorr = null;

function _heatColor(v) {
  if (v == null || Number.isNaN(v)) return '';
  const a = Math.min(1, Math.abs(v));
  const rgb = v >= 0 ? '124,217,146' : '255,124,124';
  return `background:rgba(${rgb},${(a*0.55).toFixed(3)});`;
}
function _fmt(v) { return (v == null || Number.isNaN(v)) ? '–' : v.toFixed(3); }

function _labelWithSwatch(label) {
  // find the basket window whose @t matches the label (best effort)
  const w = ST.basket.find(b => `${b.id}@${b.t}` === label);
  const sw = w ? `<span class="win-swatch" style="background:${w.color}"></span>` : '';
  return `${sw}${escapeHtml(label)}`;
}
function _renderOneMatrix(labels, M, L) {
  let h = '<div class="corr-table-wrap"><table class="mini-corr"><thead><tr><th></th>' +
    labels.map(l => `<th>${_labelWithSwatch(l)}</th>`).join('') + '</tr></thead><tbody>';
  for (let i = 0; i < labels.length; i++) {
    h += `<tr><th>${_labelWithSwatch(labels[i])}</th>`;
    for (let j = 0; j < labels.length; j++) {
      const v = M[i][j];
      if (i === j) { h += `<td class="diag">·</td>`; continue; }
      const style = _heatColor(v);
      const lagStr = (L && L[i][j] != null) ? `<span class="lag">lag ${L[i][j]>0?'+':''}${L[i][j]}</span>` : '';
      h += `<td class="cell" style="${style}">${_fmt(v)}${lagStr}</td>`;
    }
    h += '</tr>';
  }
  h += '</tbody></table></div>';
  return h;
}
function _renderCompactAllMatrix(labels, matrices, bestLag) {
  const methods = ['pearson', 'spearman', 'kendall'].filter(m => m in matrices);
  let h = '<div class="corr-table-wrap"><table class="mini-corr compact"><thead><tr><th></th>' +
    labels.map(l => `<th>${_labelWithSwatch(l)}</th>`).join('') + '</tr></thead><tbody>';
  for (let i = 0; i < labels.length; i++) {
    h += `<tr><th>${_labelWithSwatch(labels[i])}</th>`;
    for (let j = 0; j < labels.length; j++) {
      if (i === j) { h += `<td class="diag">·</td>`; continue; }
      // Tint the cell by Pearson (the primary reference)
      const ref = matrices.pearson ? matrices.pearson[i][j]
                                   : matrices[methods[0]][i][j];
      const style = _heatColor(ref);
      let inner = '';
      for (const m of methods) {
        const v = matrices[m][i][j];
        const code = m === 'pearson' ? 'P' : m === 'spearman' ? 'S' : 'K';
        inner += `<div class="r"><span class="m">${code}</span><span class="v">${_fmt(v)}</span></div>`;
      }
      if (bestLag) {
        const bv = bestLag.corr[i][j];
        const bl = bestLag.lag[i][j];
        inner += `<div class="r bl"><span class="m">↔</span><span class="v">${_fmt(bv)}</span><span class="lag">lag ${bl>0?'+':''}${bl}</span></div>`;
      }
      h += `<td class="cell" style="${style}">${inner}</td>`;
    }
    h += '</tr>';
  }
  h += '</tbody></table></div>';
  return h;
}

function renderCorrPanel() {
  const r = ST.lastCorr;
  const root = $('#corr-matrix');
  if (!r) { root.innerHTML = ''; return; }
  const labels = r.labels;
  const methods = Object.keys(r.matrices);
  if (!methods.includes(ST.corrMethod) && ST.corrMethod !== '__all__' && ST.corrMethod !== '__bl__') {
    ST.corrMethod = methods[0];
  }

  // render tabs in the header (outside the matrix div)
  const tabsHost = $('#corr-tabs');
  tabsHost.innerHTML =
    methods.map(m => `<div class="ct${m===ST.corrMethod?' active':''}" data-meth="${m}">${m}</div>`).join('')
    + `<div class="ct${ST.corrMethod==='__all__'?' active':''}" data-meth="__all__">all</div>`
    + (r.best_lag ? `<div class="ct${ST.corrMethod==='__bl__'?' active':''}" data-meth="__bl__">best-lag</div>` : '');
  tabsHost.querySelectorAll('.ct').forEach(t =>
    t.addEventListener('click', () => { ST.corrMethod = t.dataset.meth; renderCorrPanel(); }));

  // body
  let h = '';
  if (ST.corrMethod === '__all__') {
    h += _renderCompactAllMatrix(labels, r.matrices, r.best_lag);
    h += `<div class="muted" style="font-size:11px; margin-top:6px">
      Each cell stacks the 3 correlation measures
      (<b>P</b>earson · <b>S</b>pearman · <b>K</b>endall) computed on the
      window slices (length ${r.window_size}, no shift). Cell tint = Pearson.
      ${r.best_lag ? 'The ↔ row shows the best-lag Pearson within ±'+r.best_lag.max_lag+' samples.' : ''}
    </div>`;
  } else if (ST.corrMethod === '__bl__' && r.best_lag) {
    h += _renderOneMatrix(labels, r.best_lag.corr, r.best_lag.lag);
    h += `<div class="muted" style="font-size:11px; margin-top:6px">Best-lag Pearson: best |corr| obtained by sliding one window over the other within ±${r.best_lag.max_lag} samples (positive lag = column shifted right).</div>`;
  } else {
    h += _renderOneMatrix(labels, r.matrices[ST.corrMethod], null);
  }
  root.innerHTML = h;
}
// global best-lag checkbox lives in the header now
document.addEventListener('change', e => {
  if (e.target && e.target.id === 'opt-bestlag') {
    ST.showBestLag = e.target.checked;
    if (ST.showBestLag && ST.corrMethod !== '__all__') ST.corrMethod = '__bl__';
    replot();
  }
});
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.route("/static/<path:fname>")
def static_file(fname: str):
    """Serve vendored static assets (D3, future CSS/JS) from v2/static/."""
    # Resolve relative to this file's directory
    root = Path(__file__).parent / "static"
    target = (root / fname).resolve()
    # Defensive: stay within the static dir
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return Response("not found", status=404)
    if not target.is_file():
        return Response("not found", status=404)
    mime = ("application/javascript" if fname.endswith(".js")
            else "text/css" if fname.endswith(".css")
            else "application/octet-stream")
    return Response(target.read_bytes(), mimetype=mime)


@app.route("/api/pipelines")
def api_pipelines():
    items = []
    for d in _list_pipeline_dirs(STATE.results_root):
        # label = path relative to results root (or full if elsewhere)
        try:
            rel = d.relative_to(STATE.results_root)
            label = str(rel) if str(rel) != "." else d.name
        except ValueError:
            label = str(d)
        items.append({"path": str(d), "label": label})
    return jsonify({"pipelines": items, "root": str(STATE.results_root)})


def _get_pipeline_arg() -> Path:
    p = Path(request.args.get("pipeline", ""))
    if not p.is_dir() or not (p / "correlated.csv").is_file():
        raise ValueError("invalid pipeline path")
    return p


def _source_row_count(source: Path) -> int:
    """Cheap (line count - header). Cached per source."""
    key = "rows:" + str(source)
    if key in _HEADER_CACHE:
        return _HEADER_CACHE[key]  # type: ignore[return-value]
    n = 0
    with source.open("rb") as fh:
        for _ in fh:
            n += 1
    n = max(0, n - 1)
    _HEADER_CACHE[key] = n  # type: ignore[assignment]
    return n


_PRETTY_KEYS = (
    "param.n_series", "param.n_years", "param.window_size", "param.window_step",
    "param.n_lags", "param.corr_threshold", "param.neg_corr",
    "param.backend", "param.index_backend", "param.std_threshold",
    "n_windows", "n_candidates", "n_correlated", "n_episodes",
    "runtime", "param.train_ratio",
)


# Defaults from v2/core/config.py: when a corrtrack run's summary.csv was emitted
# before these params existed, the algorithm still used these values internally.
_DEFAULT_SKETCH_METHOD = "random_projection"
_DEFAULT_INDEX_KEY_MODE = "truncate"
# key_mode only applies to these indexes (see v2/pipeline.py _SUPPORTED_FOR_INDEX)
_KEY_MODE_INDEXES = {"bptree", "bst", "bptree3d", "bst3d"}


def _auto_baseline(rows: list[dict]) -> dict | None:
    """Default baseline pick.

    1. **Slowest** `bf_*` run if any bf is present — this is the reference
       method (BF = brute force, the ground-truth correlation), and the
       SLOWEST one is the most conservative reference for "×baseline"
       speedup numbers. (Previous behavior pinned bf_python regardless,
       which understated speedups on pipelines where bf_python wasn't
       actually run.)
    2. Slowest run of ANY mode otherwise — so pipelines without a bf
       reference (e.g. MPS-only sweeps) still get a meaningful speedup
       column.

    The user can override this from the `#cmp-baseline` dropdown — the
    auto-pick is just the seed, and every individual bf run is exposed
    as an explicit choice in the picker (see _baseline_candidates).
    """
    bf_rows = [r for r in rows if r["mode"] == "bf" and r["runtime"] > 0]
    if bf_rows:
        return max(bf_rows, key=lambda r: r["runtime"])
    any_rows = [r for r in rows if r["runtime"] > 0]
    return max(any_rows, key=lambda r: r["runtime"]) if any_rows else None


def _baseline_candidates(rows: list[dict]) -> list[dict]:
    """Build the list of selectable baselines for the UI picker. Each entry
    is `{key, label, path, runtime, mode, backend}`. Always starts with the
    auto pick so "Auto" is the default selected option."""
    auto = _auto_baseline(rows)

    def _slowest_of(predicate) -> dict | None:
        same = [r for r in rows if r["runtime"] > 0 and predicate(r)]
        return max(same, key=lambda r: r["runtime"]) if same else None

    out: list[dict] = []
    seen: set[str] = set()

    def _push(key: str, label_pfx: str, src: dict | None):
        if not src or src["path"] in seen:
            return
        seen.add(src["path"])
        out.append({
            "key": key,
            "label": f"{label_pfx} · {src['label']} ({src['runtime']:.2f}s)",
            "path": src["path"],
            "runtime": src["runtime"],
            "mode": src["mode"],
            "backend": src["backend"],
        })

    _push("auto", "auto", auto)
    _push("slowest", "slowest overall", _slowest_of(lambda r: True))
    for m, lab in (("bf", "slowest bf"),
                    ("corrtrack", "slowest corrtrack"),
                    ("filcorr", "slowest filcorr")):
        _push(f"slowest_{m}", lab,
              _slowest_of(lambda r, mm=m: r["mode"] == mm))
    # Expose EVERY individual bf run as an explicit picker option so the
    # user can override the auto-pick precisely (e.g. pin bf_vectorized as
    # baseline when bf_python wasn't actually run, or vice versa). Sorted
    # slowest-first so the most "conservative" references appear at the
    # top of the explicit-run section.
    bf_runs = [r for r in rows if r["mode"] == "bf" and r["runtime"] > 0]
    bf_runs.sort(key=lambda r: r["runtime"], reverse=True)
    for r in bf_runs:
        _push(f"bf_{r['path']}", "bf run", r)
    return out


def _partition_makespan(times: list[float], n: int) -> float:
    """Makespan of a grid of `times` over `n` threads, using EXACTLY the sweep's
    algorithm (`benchmark._partition`: `n` CONTIGUOUS bundles balanced within ±1
    config) → max of the per-bundle sum. Faithful to the real run bundling."""
    n = max(1, min(int(n), len(times)))
    k, m = divmod(len(times), n)
    mx, i = 0.0, 0
    for j in range(n):
        sz = k + (1 if j < m else 0)
        mx = max(mx, sum(times[i:i + sz]))
        i += sz
    return mx


def _optim_serial_parallel(run_dir: Path, workers: int) -> tuple[float, float, bool]:
    """Time of a run's optim GRID, serial and parallel.

    Returns `(serial, parallel, real)`:
      • serial   = sum of the per-config runtimes (1-thread execution);
      • parallel = time with the grid spread over `workers` threads;
      • real     = True when `parallel` is a MEASURED wall clock (the grid really
        ran in parallel: `sweep_workers>1` in best_params.json), False when it is
        a projection through the sweep's exact partitioning
        (`_partition_makespan`) applied to the measured per-config runtimes.
    `workers ≤ 1` → parallel = serial. (0,0,False) when there is no sweep.
    """
    # 1) best_params.json: serial + measured wall + REAL grid parallelism
    bp = {}
    try:
        with open(run_dir / "optimize" / "best_params.json") as fh:
            bp = json.load(fh)
    except (OSError, ValueError):
        bp = {}
    serial = float(bp.get("opt_serial") or 0.0)
    sweep_workers = int(bp.get("sweep_workers") or 0)
    opt_wall = float(bp.get("opt_time") or 0.0)

    # 2) per-config runtimes (for the serial when missing + the projection)
    times: list[float] = []
    try:
        import csv as _csv
        with open(run_dir / "optimize" / "optimize_stats.csv", newline="") as fh:
            times = [float(row["runtime"]) for row in _csv.DictReader(fh, delimiter=";")
                     if row.get("status") == "success" and row.get("runtime")]
    except (OSError, KeyError, ValueError):
        times = []
    if serial <= 0:
        serial = sum(times)
    if serial <= 0:
        return 0.0, 0.0, False

    # The grid REALLY ran in parallel (GRID mode / single run) → measured wall.
    if sweep_workers > 1 and opt_wall > 0:
        return serial, opt_wall, True
    # Otherwise: faithful projection onto the RUN's workers via the sweep partition.
    W = max(1, int(workers or 0))
    if W <= 1 or len(times) <= 1:
        return serial, serial, False
    return serial, _partition_makespan(times, W), False


def _compare_rows() -> list[dict]:
    """One row per pipeline directory, cheap (summary.csv + result.json only)."""
    rows: list[dict] = []
    pipes = _list_pipeline_dirs(STATE.results_root)
    name_map = _run_meta_by_name()
    for d in pipes:
        meta = _pipeline_meta(d)
        s = meta.get("summary", {}) or {}
        r = meta.get("result", {}) or {}
        try:
            rel = d.relative_to(STATE.results_root)
            label = str(rel) if str(rel) != "." else d.name
        except ValueError:
            label = str(d)
        # Pull declared params from the pipeline JSON (when matched by dir name).
        # Keys come hyphenated in the JSON (`index-backend`, `key-mode`, …) — normalize.
        declared = name_map.get(d.name)
        declared_params_raw = (declared or {}).get("params") or {}
        dp = {k.replace("-", "_"): v for k, v in declared_params_raw.items()}
        # summary.csv values take priority, but fall back to the declared params
        # when the field is missing (new optimizer runs don't always re-emit all
        # params in summary.csv — see v2/core/benchmark.py).
        backend = (s.get("param.backend") or dp.get("backend") or "")
        index_backend = (s.get("param.index_backend") or dp.get("index_backend") or "")
        sketch_method = (s.get("param.sketch_method") or dp.get("sketch_method") or "")
        index_key_mode = (s.get("param.index_key_mode") or dp.get("key_mode") or dp.get("index_key_mode") or "")
        # Per-phase compute backends (corrtrack only) — empty means "inherit
        # from the global `backend`". See v2/README.md §"Compute backends".
        sketch_backend = (s.get("param.sketch_backend") or dp.get("sketch_backend") or "")
        candidate_backend = (s.get("param.candidate_backend") or dp.get("candidate_backend") or "")
        validate_backend = (s.get("param.validate_backend") or dp.get("validate_backend") or "")
        # filcorr has its OWN compute backend independent of the v2 default
        # `backend` field (which stays "python" for all filcorr runs). The
        # algorithm dispatcher uses param.filcorr_backend; without surfacing
        # it here the table would show every filcorr run as `python`.
        # See v2/fillcorr.md "filcorr_backend".
        filcorr_backend = (s.get("param.filcorr_backend") or dp.get("filcorr_backend") or "")
        # FilCorr band-pass bounds (fs, ft). (0.0, 0.5) = full band ≡ Pearson
        # standard. Anything narrower = true band-pass mode. We tag each
        # filcorr run with a `filcorr_kind` ∈ {"full", "band"} so it can
        # populate the SKETCH column (otherwise empty for filcorr).
        # See v2/fillcorr.md.
        def _fnum(*cands):
            for v in cands:
                if v is None or v == "": continue
                try: return float(v)
                except (TypeError, ValueError): pass
            return None
        filcorr_fs = _fnum(s.get("param.filcorr_fs"), dp.get("filcorr_fs"))
        filcorr_ft = _fnum(s.get("param.filcorr_ft"), dp.get("filcorr_ft"))
        # sharded_<base> alias (see v2/backends/__init__.py): thread-parallel
        # sketch+insert phase. Detection is layered (each layer is a fallback
        # for the previous one), with summary.csv being most authoritative:
        #   (1) param.backend = "sharded_<base>"           (canonical)
        #   (2) declared `backend` in pipeline JSON         (when summary lost it)
        #   (3) run dir name contains "sharded"             (last resort)
        # We also propagate the resolved `backend` string back to the row's
        # `backend` field so the multiselect filters can group sharded variants.
        declared_be = dp.get("backend") or ""
        if (isinstance(declared_be, str)
                and declared_be.startswith("sharded_")
                and not str(backend).startswith("sharded_")):
            backend = declared_be
        is_sharded = isinstance(backend, str) and backend.startswith("sharded_")
        if not is_sharded and "sharded" in d.name.lower():
            # Run dir name says "sharded" but neither summary nor declared
            # params do — synthesize the alias from the visible backend so
            # the rest of the UI lights up the ⚡ marker.
            base = backend or ""
            if base and not base.startswith("sharded_"):
                backend = f"sharded_{base}"
                is_sharded = True
        backend_base = backend[len("sharded_"):] if is_sharded else backend
        try:
            n_shards = int(s.get("param.workers") or s.get("param.max_workers")
                           or dp.get("workers") or dp.get("max_workers") or 0)
        except (TypeError, ValueError):
            n_shards = 0
        # Resolve defaults vs N/A according to mode + index applicability.
        # Prefer the *declared* mode from pipeline-*.json (matched by dir name);
        # fall back to the directory-name prefix convention used by the launchers.
        if declared and declared.get("mode"):
            mode = declared["mode"]
        else:
            name = d.name.lower()
            if name.startswith("corrtrack"):
                mode = "corrtrack"
            elif name.startswith("filcorr") or name.startswith("fillcorr"):
                mode = "filcorr"
            elif name.startswith("bf"):
                mode = "bf"
            else:
                mode = "?"

        # Apply defaults: corrtrack runs always carry a configured value for
        # both sketch_method and key_mode (from v2/core/config.py — even if a
        # specific index doesn't consume key_mode, the Config still has it).
        # bf / filcorr / ? don't use sketches at all → n/a.
        # NOTE: treat literal "n/a" / "none" / "—" the same as missing — some
        # old runs persist that string in summary.csv even for corrtrack, and
        # we want the heatmaps / filters to group them under the real default.
        _MISSING = {"", "n/a", "N/A", "none", "None", "—", "-"}
        if mode == "corrtrack":
            if not sketch_method or str(sketch_method) in _MISSING:
                sketch_method = _DEFAULT_SKETCH_METHOD     # = "random_projection"
            if not index_key_mode or str(index_key_mode) in _MISSING:
                index_key_mode = _DEFAULT_INDEX_KEY_MODE   # = "truncate"
        else:
            sketch_method = "n/a"
            index_key_mode = "n/a"
        timings = r.get("timings", {}) or {}
        # v1-style phase mapping (best effort): "sketch" might not exist on bf
        rt = float(r.get("runtime") or 0.0)
        # Fallback 1: some pipeline versions (notably sharded variants) write
        # a `result.json` whose top-level `runtime` field is missing or 0
        # while the per-phase `timings` ARE populated.
        if rt <= 0 and isinstance(timings, dict) and timings:
            try:
                rt = sum(float(v or 0.0) for v in timings.values()
                         if isinstance(v, (int, float)))
            except Exception:
                rt = 0.0
        # Fallback 2: result.json was never written (or was wiped) but the
        # run actually completed — the final block of `run.log` carries the
        # wall-clock runtime + per-phase times. Recover them so the row
        # doesn't display as "0.00s · —×" forever.
        log_rt, log_phases = (0.0, {})
        if rt <= 0:
            log_rt, log_phases = _parse_runlog_runtime(d / "run.log")
            if log_rt > 0:
                rt = log_rt
                # Promote into `timings` so the per-phase columns below
                # (t_read / t_validate / …) also fill in.
                for k, v in log_phases.items():
                    timings.setdefault(k, v)
        # Fallback 3: backend / index / sketch may be empty if summary.csv
        # wasn't written. Recover them from config.json (always written at
        # run start) so the row carries real labels in the table.
        if mode == "corrtrack" and (not backend or not index_backend):
            cfg = _read_config_json(d)
            if not backend:
                backend = cfg.get("backend") or backend
                backend_base = backend[len("sharded_"):] if (
                    isinstance(backend, str) and backend.startswith("sharded_")
                ) else backend
            if not index_backend:
                index_backend = cfg.get("index_backend") or index_backend
            if not sketch_method or sketch_method == _DEFAULT_SKETCH_METHOD:
                sketch_method = cfg.get("sketch_method") or sketch_method
            if not index_key_mode or index_key_mode == _DEFAULT_INDEX_KEY_MODE:
                index_key_mode = cfg.get("key_mode") or cfg.get("index_key_mode") or index_key_mode
            if not sketch_backend:
                sketch_backend = cfg.get("sketch_backend") or ""
            if not candidate_backend:
                candidate_backend = cfg.get("candidate_backend") or ""
            if not validate_backend:
                validate_backend = cfg.get("validate_backend") or ""
        # For filcorr runs, also recover the filcorr_backend from config.json
        # when summary.csv didn't expose it (e.g., when result.json was
        # partial). Same fallback for fs/ft if needed.
        if mode == "filcorr":
            if not filcorr_backend or filcorr_fs is None or filcorr_ft is None:
                cfg = _read_config_json(d)
                if not filcorr_backend:
                    filcorr_backend = cfg.get("filcorr_backend") or ""
                if filcorr_fs is None:
                    filcorr_fs = _fnum(cfg.get("filcorr_fs"))
                if filcorr_ft is None:
                    filcorr_ft = _fnum(cfg.get("filcorr_ft"))
        # Compute filcorr_kind tag: "full" when (fs≤0, ft≥0.5) ≡ Pearson
        # standard; "band" for any narrower window; "?" when bounds are
        # genuinely unknown (rare — only if BOTH summary.csv and config.json
        # are missing the params).
        if mode == "filcorr":
            if filcorr_fs is None and filcorr_ft is None:
                filcorr_kind = "?"
            else:
                fs_val = filcorr_fs if filcorr_fs is not None else 0.0
                ft_val = filcorr_ft if filcorr_ft is not None else 0.5
                filcorr_kind = "full" if (fs_val <= 0.0 and ft_val >= 0.5) else "band"
        else:
            filcorr_kind = ""
        n_w = int(r.get("n_windows") or 0)
        n_c = int(r.get("n_candidates") or 0)
        n_t = int(r.get("n_tested") or 0)
        n_ok = int(r.get("n_correlated") or 0)
        win_per_s = (n_w / rt) if rt > 0 else None
        cand_per_s = (n_c / rt) if rt > 0 else None
        status, progress = _detect_run_status(d)
        # Displayed backend depends on mode:
        #   filcorr → use `filcorr_backend` (the real compute kernel; the
        #             generic `backend` field stays at v2 default "python"
        #             for every filcorr run, which would make the column
        #             useless).
        #   bf      → fall back to "python" when unset.
        #   corrtrack → use `backend` (with `sharded_<base>` prefix kept).
        if mode == "filcorr":
            display_backend = filcorr_backend or backend or "python"
            display_backend_base = display_backend
        else:
            display_backend = backend or ("python" if mode == "bf" else "")
            display_backend_base = backend_base or ("python" if mode == "bf" else "")

        # --- REAL-TIME metrics (throughput + resources) ---
        # summary.csv first (flat keys); fall back to result.json (nested).
        rj_tput = (r.get("throughput") or {}) if isinstance(r, dict) else {}
        rj_res = (r.get("resources") or {}) if isinstance(r, dict) else {}

        def _rt(flat_key, *nested, default=None):
            v = s.get(flat_key)
            if v is not None and v != "":
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default
            node: Any = rj_tput if (nested and nested[0] in rj_tput) else rj_res
            for k in nested:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            if isinstance(node, (int, float)):
                return float(node)
            return default

        tput_win_median = _rt("tput_windows_median", "windows", "median")
        tput_win_min = _rt("tput_windows_min", "windows", "min")
        tput_win_max = _rt("tput_windows_max", "windows", "max")
        tput_win_overall = _rt("tput_windows_overall", "windows", "overall")
        cpu_pct_median = _rt("cpu_pct_median", "cpu_pct", "median")
        cpu_pct_max = _rt("cpu_pct_max", "cpu_pct", "max")
        mem_peak_mb = _rt("mem_peak_mb", "mem_peak_mb")
        energy_cpu_seconds = _rt("energy_cpu_seconds", "energy_cpu_seconds")
        power_cores_median = _rt("power_cores_median", "power_cores", "median")

        # Optim grid time, serial vs parallel (see _optim_serial_parallel).
        try:
            _workers_n = int(s.get("param.workers") or dp.get("workers") or 0)
        except (TypeError, ValueError):
            _workers_n = 0
        _opt_serial, _opt_makespan, _opt_real = _optim_serial_parallel(d, _workers_n)

        rows.append({
            "path": str(d),
            "label": label,
            "mode": mode,
            "status": status,
            "progress": progress,   # null when not running
            "backend": display_backend,
            "backend_base": display_backend_base,
            "is_sharded": is_sharded,
            "n_shards": n_shards if is_sharded else 0,
            "index_backend": index_backend,
            "sketch_method": sketch_method,
            "index_key_mode": index_key_mode,
            "sketch_backend": sketch_backend,
            "candidate_backend": candidate_backend,
            "validate_backend": validate_backend,
            "filcorr_backend": filcorr_backend,
            "filcorr_kind": filcorr_kind,
            "filcorr_fs": filcorr_fs,
            "filcorr_ft": filcorr_ft,
            "runtime": rt,
            "n_windows": n_w,
            "n_candidates": n_c,
            "n_tested": n_t,
            "n_correlated": n_ok,
            "n_episodes": int(r.get("n_episodes") or 0),
            "n_anomalies": int(r.get("n_anomalies") or 0),
            "opt_time": float(r.get("opt_time") or 0.0),
            # Optim grid time, serial vs parallel (see
            # _optim_serial_parallel). opt_eff = MEASURED wall when the grid
            # really ran in parallel (opt_real=True), otherwise a faithful
            # projection (exact sweep partition) onto the run's workers.
            "workers": _workers_n,
            "opt_serial": _opt_serial,
            "opt_eff": _opt_makespan,
            "opt_real": _opt_real,
            "win_per_s": win_per_s,
            "cand_per_s": cand_per_s,
            "candidate_ratio": (n_c / n_t) if n_t else None,
            # Map both v1-style ("sketch" / "index" / "select" /
            # "validate_sketches") and v2-style ("ingest" / "candidates")
            # phase names into the unified v2 columns. v1 runs in tested
            # result.json files split sketch-construction across `sketch` +
            # `index` (= ingest) and candidate-selection across `select` +
            # `validate_sketches` (= candidates). Without this mapping
            # ingest/candidates show up empty in the tooltip and the
            # phase-bar visually merges that time into "other".
            "t_read": float(timings.get("read") or 0.0),
            "t_windows": float(timings.get("windows") or 0.0),
            "t_ingest": float((timings.get("ingest") or 0.0)
                              + (timings.get("sketch") or 0.0)
                              + (timings.get("index") or 0.0)),
            "t_candidates": float((timings.get("candidates") or 0.0)
                                  + (timings.get("select") or 0.0)
                                  + (timings.get("validate_sketches") or 0.0)),
            "t_validate": float(timings.get("validate") or 0.0),
            "t_monitor": float(timings.get("monitor") or 0.0),
            "t_evict": float(timings.get("evict") or 0.0),
            # --- real-time metrics ---
            "tput_win_median": tput_win_median,
            "tput_win_min": tput_win_min,
            "tput_win_max": tput_win_max,
            "tput_win_overall": tput_win_overall,
            "cpu_pct_median": cpu_pct_median,
            "cpu_pct_max": cpu_pct_max,
            "mem_peak_mb": mem_peak_mb,
            "energy_cpu_seconds": energy_cpu_seconds,
            "power_cores_median": power_cores_median,
            # time series (sub-sampled) for the throughput=f(t) curve
            "tput_series": meta.get("throughput", []) or [],
            "res_series": meta.get("resources", []) or [],
        })

    # Auto-picked baseline: prefer bf_python, else slowest bf, else fall back
    # to "slowest run of any mode" so MPS-only / corrtrack-only pipelines also
    # get a meaningful speedup column. Users can pick a different baseline at
    # runtime via the UI picker — see _baseline_candidates() + /api/compare.
    baseline_row = _auto_baseline(rows)
    baseline_rt = baseline_row["runtime"] if baseline_row else None
    baseline_label = baseline_row["label"] if baseline_row else None
    baseline_path = baseline_row["path"] if baseline_row else None
    runtimes_bf = [r["runtime"] for r in rows if r["mode"] == "bf" and r["runtime"] > 0]
    runtimes_ct = [r["runtime"] for r in rows if r["mode"] == "corrtrack" and r["runtime"] > 0]
    fastest_overall = min((rt for rt in runtimes_bf + runtimes_ct), default=None)
    slowest_overall = max((rt for rt in runtimes_bf + runtimes_ct), default=None)
    for r in rows:
        rt = r["runtime"]
        if rt and fastest_overall:
            r["speedup_vs_fastest"] = fastest_overall / rt
        if rt and slowest_overall:
            r["speedup_vs_slowest"] = slowest_overall / rt
        # Primary column: speedup against the consistent baseline (bf_python by default).
        if rt and baseline_rt:
            r["speedup_vs_baseline"] = baseline_rt / rt
        else:
            r["speedup_vs_baseline"] = None
        # PER-STEP speedup vs baseline — to see where the gain comes from
        # (sharded accelerates candidates/validate but not monitor). Unified
        # phases:
        # ingest=sketch+index, candidates=select(+vsketch), validate, monitor.
        for ph in ("t_ingest", "t_candidates", "t_validate", "t_monitor"):
            bt = (baseline_row.get(ph) or 0.0) if baseline_row else 0.0
            vt = r.get(ph) or 0.0
            r["spd_" + ph] = (bt / vt) if (bt > 0 and vt > 0) else None
        r["_baseline_label"] = baseline_label
        r["_baseline_path"] = baseline_path
    return rows


@app.route("/api/sync_now", methods=["POST", "GET"])
def api_sync_now():
    """Force one rsync cycle now and return what changed. No-op (with a clear
    payload) when the viewer is running on local files."""
    if STATE.mirror is None:
        return jsonify({"ok": True, "remote": False,
                        "msg": "viewer is using local files; no sync needed"})
    info = STATE.mirror.sync_now()
    return jsonify({"ok": not info.get("last_error"), "remote": True, **info})


@app.route("/api/sync_status")
def api_sync_status():
    """Last-sync info without forcing a sync. Used by the auto-refresh bar."""
    if STATE.mirror is None:
        return jsonify({"remote": False})
    m = STATE.mirror
    return jsonify({"remote": True,
                    "target": m.spec.target,
                    "remote_path": m.spec.path,
                    "local_mirror": str(m.local_root),
                    **m.last_sync})


@app.route("/api/sync_interval", methods=["POST"])
def api_sync_interval():
    """Update the periodic-sync interval at runtime. POST seconds=N. N<=0
    pauses the loop without killing the thread (it just no-ops)."""
    if STATE.mirror is None:
        return jsonify({"ok": False, "msg": "not in remote mode"}), 400
    try:
        secs = float(request.values.get("seconds", "5"))
    except ValueError:
        return jsonify({"ok": False, "msg": "invalid seconds"}), 400
    secs = max(0.0, min(secs, 3600.0))
    STATE.mirror._sync_interval = secs if secs > 0 else 86400.0  # ~never
    STATE.mirror._last_sync["interval"] = secs
    return jsonify({"ok": True, "interval": secs})


@app.route("/api/runs_status")
def api_runs_status():
    """Lightweight per-run status — designed for cheap 5s polling.

    Returns per-pipeline {path, label, status, progress?, mode_hint?}.
    If a pipeline JSON was provided on the CLI, also adds 'queued' entries
    for declared runs that have no directory yet.
    """
    rows = []
    seen_paths = set()
    name_map = _run_meta_by_name()
    for d in _list_pipeline_dirs(STATE.results_root):
        status, progress = _detect_run_status(d)
        try:
            rel = d.relative_to(STATE.results_root)
            label = str(rel) if str(rel) != "." else d.name
        except ValueError:
            label = str(d)
        rows.append({"path": str(d), "label": label, "name": d.name,
                     "status": status, "progress": progress})
        seen_paths.add(d.name)
    # Queued: declared in pipeline JSON but no dir exists yet
    if STATE.pipeline_obj:
        pj_runs = STATE.pipeline_obj.get("runs") or []
        for run in pj_runs:
            n = run.get("name")
            if not n or n in seen_paths:
                continue
            rows.append({"path": "", "label": n, "name": n,
                         "status": "queued", "progress": None,
                         "mode_hint": run.get("mode") or ""})
    counts = {"done":0, "running":0, "starting":0, "failed":0, "queued":0, "unknown":0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    # If a JSON was given, also include the live "where" info for the running pipeline
    pj_name = STATE.pipeline_obj.get("name") if STATE.pipeline_obj else None
    return jsonify({"rows": rows, "counts": counts,
                    "pipeline_name": pj_name,
                    "pipeline_json": str(STATE.pipeline_json) if STATE.pipeline_json else None,
                    "results_root": str(STATE.results_root)})


_PAIRS_CACHE: dict[str, tuple[float, frozenset]] = {}    # path -> (mtime, pair_set)


def _read_pairs_set(pdir: Path) -> frozenset:
    """Return the set of correlated pairs for a run as a frozenset of
    canonicalized ((id_lo, t_lo), (id_hi, t_hi)) tuples.

    Cached in-memory keyed by (path, mtime) so an updated correlated.csv
    invalidates the cache automatically.
    """
    key = str(pdir)
    p = pdir / "correlated.csv"
    if not p.is_file():
        return frozenset()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _PAIRS_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    for chunk in pd.read_csv(p, usecols=["id1", "t1", "id2", "t2"],
                             chunksize=200_000):
        ids1 = chunk["id1"].astype(str).to_numpy()
        ids2 = chunk["id2"].astype(str).to_numpy()
        t1s = pd.to_numeric(chunk["t1"], errors="coerce").to_numpy()
        t2s = pd.to_numeric(chunk["t2"], errors="coerce").to_numpy()
        for i in range(len(ids1)):
            if not (np.isfinite(t1s[i]) and np.isfinite(t2s[i])):
                continue
            a = (ids1[i], int(t1s[i]))
            b = (ids2[i], int(t2s[i]))
            pairs.add((a, b) if a <= b else (b, a))
    fs = frozenset(pairs)
    _PAIRS_CACHE[key] = (mtime, fs)
    return fs


# ---- Parallel precision/recall workers --------------------------------------
# Each worker process holds ONE copy of the baseline pair-set (shipped once via
# the initializer, not per-task — pickling a 1.5M-pair set on every task would
# dwarf the compute). A task then reads one run's correlated.csv, intersects
# against the baseline, and returns the metrics dict. True multi-core speedup:
# both the pandas CSV parse AND the Python set-intersection run concurrently
# across processes.
_WORKER_BASE: frozenset = frozenset()


def _quality_worker_init(base_set: frozenset) -> None:
    global _WORKER_BASE
    _WORKER_BASE = base_set


def _quality_worker_serial(path_str: str, base: frozenset) -> tuple[str, float, dict]:
    """Compute the precision/recall entry for one run dir against an explicit
    baseline set. Used both by the process-pool worker (via the global) and
    the serial fallback."""
    pdir = Path(path_str)
    ccsv = pdir / "correlated.csv"
    try:
        mtime = ccsv.stat().st_mtime
    except OSError:
        mtime = 0.0
    try:
        run_set = _read_pairs_set(pdir)
    except Exception as e:
        return (path_str, mtime, {"precision": None, "recall": None,
                                  "error": str(e)})
    if not run_set:
        entry = {"precision": None, "recall": None, "f1": None,
                 "n_pairs": 0, "n_intersect": 0, "n_baseline": len(base)}
    else:
        inter = base & run_set
        precision = len(inter) / len(run_set)
        recall = len(inter) / len(base) if base else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        entry = {"precision": precision, "recall": recall, "f1": f1,
                 "n_pairs": len(run_set), "n_intersect": len(inter),
                 "n_baseline": len(base)}
    return (path_str, mtime, entry)


def _quality_worker(path_str: str) -> tuple[str, float, dict]:
    """Process-pool entry point: delegates to the serial worker using the
    per-process baseline set installed by _quality_worker_init."""
    return _quality_worker_serial(path_str, _WORKER_BASE)


# On-disk cache of precision/recall results (survives server restart).
# Schema: {baseline_path: {run_path: {mtime, precision, recall, f1, n_pairs, ...}}}
_QUALITY_CACHE_FILE = CACHE_ROOT / "quality.json"
_QUALITY_CACHE: dict[str, dict[str, dict]] | None = None


def _load_quality_cache() -> dict:
    global _QUALITY_CACHE
    if _QUALITY_CACHE is not None:
        return _QUALITY_CACHE
    if _QUALITY_CACHE_FILE.is_file():
        try:
            _QUALITY_CACHE = json.loads(_QUALITY_CACHE_FILE.read_text())
        except Exception:
            _QUALITY_CACHE = {}
    else:
        _QUALITY_CACHE = {}
    return _QUALITY_CACHE


def _save_quality_cache() -> None:
    if _QUALITY_CACHE is None:
        return
    try:
        _QUALITY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # ATOMIC write: serialize to a temp file in the same dir, fsync, then
        # os.replace() (atomic rename on the same filesystem). A Ctrl+C / kill
        # mid-write can no longer truncate or corrupt the real cache file —
        # the previous bug was that an interrupted write_text() left an empty
        # / partial quality.json, so the next launch loaded 0 cached entries
        # and recomputed everything from scratch.
        data = json.dumps(_QUALITY_CACHE, separators=(",", ":"))
        tmp = _QUALITY_CACHE_FILE.with_suffix(
            _QUALITY_CACHE_FILE.suffix + f".tmp.{os.getpid()}")
        with open(tmp, "w") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _QUALITY_CACHE_FILE)
    except Exception:
        # Best-effort: clean up a stray temp file if the rename failed.
        try:
            if tmp.is_file():
                tmp.unlink()
        except Exception:
            pass


def _compute_quality(baseline_path: str | None = None) -> dict:
    """Core precision/recall computation. Returns the same dict the
    /api/compare_quality route jsonifies, OR {"error": ...} on failure.
    Safe to call off the request thread (e.g. startup pre-warm) — touches
    only module-level caches under no Flask context.

    Persistent cache on disk, invalidated by correlated.csv mtime; on a
    warm cache only newly-changed runs are recomputed.
    """
    rows = _compare_rows()
    if not baseline_path:
        # Same auto-pick policy as _auto_baseline(): the SLOWEST bf run
        # (most conservative reference), else slowest run overall.
        auto = _auto_baseline(rows)
        if not auto:
            return {"error": "no run available as baseline"}
        baseline_path = auto["path"]

    cache = _load_quality_cache()
    base_entry = cache.setdefault(baseline_path, {})
    print(f"[viz] quality: baseline = {Path(baseline_path).name} "
          f"(reading pair set…)", file=sys.stderr, flush=True)
    # Baseline pairs (also cached in-memory keyed by mtime via _read_pairs_set)
    base = _read_pairs_set(Path(baseline_path))
    if not base:
        return {"error": f"empty baseline at {baseline_path}"}
    print(f"[viz] quality: baseline has {len(base):,} pairs · "
          f"comparing {len(rows)} run(s)", file=sys.stderr, flush=True)

    # Track baseline mtime + size — if either changed since the last save,
    # the cached entries vs this baseline are stale; drop them.
    base_csv = Path(baseline_path) / "correlated.csv"
    base_mtime = base_csv.stat().st_mtime if base_csv.is_file() else 0.0
    base_meta_key = "__baseline_meta__"
    saved_meta = base_entry.get(base_meta_key) or {}
    if saved_meta.get("mtime") != base_mtime or saved_meta.get("size") != len(base):
        base_entry.clear()
        base_entry[base_meta_key] = {"mtime": base_mtime, "size": len(base)}

    dirty = 0
    hits = 0
    # Partition runs: instant cache hits vs the ones needing a (slow)
    # recompute. We emit cache hits immediately and farm the misses out to a
    # process pool.
    todo: list[str] = []   # run paths needing recompute
    out_by_path: dict[str, dict] = {}
    skipped_running = 0
    for r in rows:
        pdir = Path(r["path"])
        ccsv = pdir / "correlated.csv"
        if not ccsv.is_file():
            out_by_path[r["path"]] = {
                "path": r["path"], "precision": None, "recall": None,
                "n_pairs": 0, "n_baseline": len(base)}
            continue
        # Skip runs still in progress: their correlated.csv is incomplete and
        # keeps growing, so computing precision/recall now would be wrong AND
        # immediately stale. They'll be picked up automatically once they
        # finish (the auto-refresh detects the done transition and recomputes).
        if r.get("status") in ("running", "starting"):
            out_by_path[r["path"]] = {
                "path": r["path"], "precision": None, "recall": None,
                "n_pairs": 0, "n_baseline": len(base), "_running": True}
            skipped_running += 1
            continue
        try:
            mtime = ccsv.stat().st_mtime
        except OSError:
            mtime = 0.0
        cached = base_entry.get(r["path"])
        if cached and cached.get("mtime") == mtime:
            out_by_path[r["path"]] = {
                "path": r["path"],
                **{k: v for k, v in cached.items() if k != "mtime"},
                "_cached": True}
            hits += 1
        else:
            todo.append(r["path"])
    n_todo = len(todo)
    if skipped_running:
        print(f"[viz] quality: skipping {skipped_running} run(s) still in "
              f"progress (will compute once they finish)",
              file=sys.stderr, flush=True)
    start_ts = time.time()

    if n_todo:
        # Worker count: capped at 2 to keep CPU/RAM footprint low (each
        # worker holds its own copy of the baseline pair-set + parses big
        # CSVs). Also bounded by the number of pending runs and an optional
        # QUALITY_WORKERS env override.
        try:
            _env_w = int(os.environ.get("QUALITY_WORKERS", "2"))
        except ValueError:
            _env_w = 2
        n_workers = max(1, min(n_todo, _env_w, (os.cpu_count() or 2)))
        print(f"[viz] quality: {hits} cached · {n_todo} to recompute "
              f"on {n_workers} workers", file=sys.stderr, flush=True)
        try:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_quality_worker_init,
                    initargs=(base,)) as ex:
                futs = {ex.submit(_quality_worker, p): p for p in todo}
                try:
                    for fut in as_completed(futs):
                        path_str, mtime, entry = fut.result()
                        base_entry[path_str] = {"mtime": mtime, **entry}
                        out_by_path[path_str] = {"path": path_str, **entry,
                                                 "_cached": False}
                        dirty += 1
                        _term_progress(dirty, n_todo, label="quality",
                                       start_ts=start_ts, final=(dirty == n_todo))
                        # Periodic checkpoint so an interrupt keeps progress.
                        if dirty % 10 == 0:
                            _save_quality_cache()
                except KeyboardInterrupt:
                    # Persist everything completed so far BEFORE the pool's
                    # __exit__ shutdown runs (which can hang/abort and skip
                    # the save). Cancel pending futures so shutdown is fast,
                    # then re-raise for main() to handle.
                    print(f"\n[viz] quality: interrupted at {dirty}/{n_todo} "
                          f"— saving partial progress…", file=sys.stderr, flush=True)
                    _save_quality_cache()
                    for f in futs:
                        f.cancel()
                    raise
        except Exception as e:
            # Fall back to serial in-process compute if the pool can't start
            # (e.g. restricted sandbox without fork/spawn). Slower but works.
            print(f"\n[viz] quality: process pool unavailable ({e}); "
                  f"falling back to serial", file=sys.stderr, flush=True)
            for path_str in todo:
                if path_str in out_by_path and not out_by_path[path_str].get("_cached", True):
                    continue  # already done before the failure
                _, mtime, entry = _quality_worker_serial(path_str, base)
                base_entry[path_str] = {"mtime": mtime, **entry}
                out_by_path[path_str] = {"path": path_str, **entry, "_cached": False}
                dirty += 1
                _term_progress(dirty, n_todo, label="quality",
                               start_ts=start_ts, final=(dirty == n_todo))
                if dirty % 10 == 0:
                    _save_quality_cache()
        _term_progress(dirty, n_todo, label="quality",
                       start_ts=start_ts, final=True)

    # Preserve the original row order in the response.
    out = [out_by_path[r["path"]] for r in rows if r["path"] in out_by_path]
    if dirty:
        _save_quality_cache()
    print(f"[viz] quality: done · {hits} cached / {dirty} recomputed "
          f"· {time.time()-start_ts:.1f}s", file=sys.stderr, flush=True)
    return {"rows": out, "baseline": baseline_path,
            "n_baseline_pairs": len(base),
            "cache_hits": hits, "cache_misses": dirty}


@app.route("/api/compare_quality")
def api_compare_quality():
    """Precision / recall per run vs a baseline pipeline (HTTP wrapper)."""
    result = _compute_quality(request.args.get("baseline"))
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/quality_cache/clear", methods=["POST", "GET"])
def api_quality_cache_clear():
    """Force a full recompute on the next /api/compare_quality call."""
    global _QUALITY_CACHE
    _QUALITY_CACHE = {}
    try:
        if _QUALITY_CACHE_FILE.is_file():
            _QUALITY_CACHE_FILE.unlink()
    except Exception:
        pass
    return jsonify({"cleared": True})


@app.route("/api/compare")
def api_compare():
    rows = _compare_rows()
    def _fastest(mode):
        rts = [r["runtime"] for r in rows
               if r["mode"] == mode and r["runtime"] > 0]
        return min(rts) if rts else None
    all_rts = [r["runtime"] for r in rows if r["runtime"] > 0]
    baseline = _auto_baseline(rows)
    return jsonify({
        "rows": rows,
        "fastest_bf": _fastest("bf"),
        "fastest_corrtrack": _fastest("corrtrack"),
        "fastest_filcorr": _fastest("filcorr"),
        "fastest_overall": min(all_rts) if all_rts else None,
        "baseline_label": baseline["label"] if baseline else None,
        "baseline_path":  baseline["path"]  if baseline else None,
        "baseline_runtime": baseline["runtime"] if baseline else None,
        "baseline_candidates": _baseline_candidates(rows),
        "root": str(STATE.results_root),
    })


# ---------------------------------------------------------------------------
# Compare charts (server-side matplotlib PNGs)
# ---------------------------------------------------------------------------

_MODE_COLOR = {"bf": "#ffb05a", "corrtrack": "#7cd992",
               "filcorr": "#5aa1ff", "?": "#888888"}
# Stable palette for stacked phase bars (in plotting order, bottom→top)
_PHASE_KEYS = [
    ("t_read", "read", "#5aa1ff"),
    ("t_windows", "windows", "#88c1ff"),
    ("t_ingest", "ingest", "#c08bff"),
    ("t_candidates", "candidates", "#ffb05a"),
    ("t_validate", "validate", "#ff7c7c"),
    ("t_monitor", "monitor", "#5ddcdc"),
    ("t_evict", "evict", "#a0a0a0"),
    ("opt_time", "opt sweep", "#ffe066"),
]


def _safe_float(v, default: float = float("nan")) -> float:
    """Coerce to float, returning `default` (NaN) on None / parse failure."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _png_response(fig) -> Response:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.read(), mimetype="image/png")


def _filtered_rows() -> list[dict]:
    rows = _compare_rows()
    show_bf = request.args.get("bf", "1") in ("1", "true")
    show_ct = request.args.get("ct", "1") in ("1", "true")
    show_fc = request.args.get("fc", "1") in ("1", "true")
    sel_paths = set(request.args.getlist("path"))
    out = []
    for r in rows:
        if sel_paths and r["path"] not in sel_paths:
            continue
        if r["mode"] == "bf" and not show_bf:
            continue
        if r["mode"] == "corrtrack" and not show_ct:
            continue
        if r["mode"] == "filcorr" and not show_fc:
            continue
        if r["runtime"] <= 0:
            continue
        out.append(r)
    return out


@app.route("/api/chart/<name>")
def api_chart(name: str):
    rows = _filtered_rows()
    if not rows:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No runs to plot — toggle filters above.",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_axis_off()
        return _png_response(fig)

    if name == "runtime":
        return _png_response(_chart_runtime(rows))
    if name == "phases":
        return _png_response(_chart_phases(rows))
    if name == "tradeoff":
        return _png_response(_chart_tradeoff(rows))
    if name == "throughput":
        return _png_response(_chart_throughput(rows))
    if name == "speedup":
        return _png_response(_chart_speedup(rows))
    if name == "heatmap":
        return _png_response(_chart_heatmap(rows))
    if name == "resources":
        return _png_response(_chart_resources(rows))
    return Response(f"unknown chart: {name}", status=404)


def _chart_runtime(rows: list[dict]):
    rows = sorted(rows, key=lambda r: r["runtime"])
    labels = [r["label"] for r in rows]
    values = [r["runtime"] for r in rows]
    colors = [_MODE_COLOR.get(r["mode"], "#888") for r in rows]
    h = max(3.2, 0.28 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(12, h), dpi=110)
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
    fastest = min(values)
    for i, (bar, v) in enumerate(zip(bars, values)):
        star = " ★" if v == fastest else ""
        ax.text(v, i, f"  {v:.2f}s{star}", va="center", fontsize=9, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("runtime (s) — lower is better")
    ax.set_title(f"Runtime ranking · {len(rows)} runs")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    # mode legend
    from matplotlib.patches import Patch
    legend = [Patch(color=_MODE_COLOR["bf"], label="bf"),
              Patch(color=_MODE_COLOR["corrtrack"], label="corrtrack")]
    if any(r["mode"] == "filcorr" for r in rows):
        legend.append(Patch(color=_MODE_COLOR["filcorr"], label="filcorr"))
    if any(r["mode"] == "?" for r in rows):
        legend.append(Patch(color=_MODE_COLOR["?"], label="other"))
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    return fig


def _chart_phases(rows: list[dict]):
    rows = sorted(rows, key=lambda r: r["runtime"])
    labels = [r["label"] for r in rows]
    n = len(rows)
    # taller per-row so labels stay readable when many runs are stacked
    h = max(4.0, 0.38 * n + 1.6)
    fig, ax = plt.subplots(figsize=(13, h), dpi=110)
    y = np.arange(n)
    left = np.zeros(n)
    for key, name, color in _PHASE_KEYS:
        vals = np.array([float(r.get(key) or 0.0) for r in rows])
        if vals.sum() <= 0:
            continue
        ax.barh(y, vals, left=left, color=color, label=name,
                edgecolor="white", linewidth=0.5)
        left += vals
    for i, r in enumerate(rows):
        ax.text(r["runtime"], i, f"  {r['runtime']:.1f}s",
                va="center", fontsize=9, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("time (s) — stacked phases")
    ax.set_title(f"Time breakdown per run · {n} runs")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.92)
    return fig


def _chart_speedup(rows: list[dict]):
    """Horizontal bars of speedup vs bf_python (or slowest bf as fallback).

    Speedup = baseline_runtime / row_runtime. So 10× means "10 times faster".
    """
    # Reference: prefer the explicit bf_python run (the naive reference);
    # otherwise fall back to the slowest bf run.
    bf_python = next((r for r in rows if r["mode"] == "bf"
                      and r["backend"] == "python"), None)
    if bf_python is None:
        bf_rows = [r for r in rows if r["mode"] == "bf" and r["runtime"] > 0]
        if not bf_rows:
            bf_python = max(rows, key=lambda r: r["runtime"])
        else:
            bf_python = max(bf_rows, key=lambda r: r["runtime"])
    baseline_label = bf_python["label"]
    baseline_rt = bf_python["runtime"]

    # Sort by SPEEDUP desc (fastest at the top), not by runtime; reads better.
    pairs = sorted(zip([baseline_rt / r["runtime"] for r in rows], rows),
                   key=lambda p: p[0], reverse=True)
    speedups = [p[0] for p in pairs]
    rows = [p[1] for p in pairs]
    labels = [r["label"] for r in rows]
    colors = [_MODE_COLOR.get(r["mode"], "#888") for r in rows]
    h = max(4.5, 0.40 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(14, h), dpi=110)
    y = np.arange(len(rows))
    ax.barh(y, speedups, color=colors, edgecolor="white", linewidth=0.5)
    best = max(speedups)
    for i, (s, r) in enumerate(zip(speedups, rows)):
        star = " ★" if s == best else ""
        ax.text(s, i, f"  ×{s:.2f}  ({r['runtime']:.2f}s){star}",
                va="center", fontsize=9, color="#333")
    ax.axvline(1.0, color="#888", linestyle="--", alpha=0.7,
               label=f"baseline = {baseline_label} ({baseline_rt:.1f}s)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel(f"speedup × vs {baseline_label} — log scale")
    ax.set_title(f"Speedup ranking · baseline = {baseline_label} ({baseline_rt:.1f}s)")
    ax.set_xscale("log")
    # leave room on the right for the annotations
    ax.set_xlim(left=min(0.7, min(speedups) * 0.7),
                right=max(speedups) * 2.2)
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.35)
    from matplotlib.patches import Patch
    handles = [Patch(color=_MODE_COLOR["bf"], label="bf"),
               Patch(color=_MODE_COLOR["corrtrack"], label="corrtrack")]
    if any(r["mode"] == "filcorr" for r in rows):
        handles.append(Patch(color=_MODE_COLOR["filcorr"], label="filcorr"))
    if any(r["mode"] == "?" for r in rows):
        handles.append(Patch(color=_MODE_COLOR["?"], label="other"))
    ax.legend(handles=handles + [
        plt.Line2D([0], [0], color="#888", linestyle="--", label="×1 baseline")
    ], loc="lower right", fontsize=9, framealpha=0.92)
    return fig


def _chart_heatmap(rows: list[dict]):
    """Grid (backend × index_backend) coloured by speedup vs slowest run.

    Each cell shows the *fastest* runtime found for that combination, with
    `n_correlated` annotated underneath. Empty combinations stay grey.
    """
    # Reference: slowest run overall — keeps the colour scale dataset-relative
    slowest = max(r["runtime"] for r in rows)
    cells: dict[tuple[str, str], dict] = {}
    for r in rows:
        b = r["backend"] or "(none)"
        i = r["index_backend"] or "(none)"
        key = (b, i)
        if key not in cells or r["runtime"] < cells[key]["runtime"]:
            cells[key] = r
    backends = sorted({k[0] for k in cells.keys()})
    indexes = sorted({k[1] for k in cells.keys()})
    nb, ni = len(backends), len(indexes)
    speed_grid = np.full((nb, ni), np.nan)
    rt_grid = np.full((nb, ni), np.nan)
    nc_grid = np.full((nb, ni), np.nan)
    for (b, idx), r in cells.items():
        bi, ii = backends.index(b), indexes.index(idx)
        speed_grid[bi, ii] = slowest / r["runtime"] if r["runtime"] else np.nan
        rt_grid[bi, ii] = r["runtime"]
        nc_grid[bi, ii] = r["n_correlated"]

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * ni + 4), max(3.2, 0.6 * nb + 1.5)),
                            dpi=110)
    # Log-normalise colours so the 1×→60× range is readable
    finite = speed_grid[np.isfinite(speed_grid)]
    vmin = max(0.5, finite.min()) if finite.size else 1.0
    vmax = finite.max() if finite.size else 10.0
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(speed_grid, cmap="YlGnBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(ni))
    ax.set_xticklabels(indexes, fontsize=10, family="monospace")
    ax.set_yticks(np.arange(nb))
    ax.set_yticklabels(backends, fontsize=10, family="monospace")
    # cell annotations
    for bi in range(nb):
        for ii in range(ni):
            s = speed_grid[bi, ii]
            if not np.isfinite(s):
                ax.text(ii, bi, "–", ha="center", va="center", color="#888", fontsize=9)
                continue
            rt = rt_grid[bi, ii]; nc = nc_grid[bi, ii]
            # pick text colour for contrast
            txt_color = "white" if s > vmax * 0.55 else "#222"
            ax.text(ii, bi, f"×{s:.1f}\n{rt:.1f}s\nn={int(nc):,}",
                    ha="center", va="center", fontsize=8.5, color=txt_color,
                    family="monospace", linespacing=1.15)
    ax.set_xlabel("index_backend")
    ax.set_ylabel("backend")
    ax.set_title(f"Best speedup per (backend × index) · "
                  f"baseline = slowest run ({slowest:.1f}s)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("speedup × (log)", fontsize=9)
    return fig


def _chart_tradeoff(rows: list[dict]):
    """Speedup vs recall scatter.

    Speedup baseline = slowest bf run (typical "naive python" reference).
    Recall proxy   = n_correlated / max(n_correlated across all rows).
    A perfect-recall run sits at y=1; under-recalling sketches drop below.
    """
    bf_runtimes = [r["runtime"] for r in rows if r["mode"] == "bf"]
    baseline_rt = max(bf_runtimes) if bf_runtimes else max(r["runtime"] for r in rows)
    max_corr = max((r["n_correlated"] for r in rows if r["n_correlated"]), default=1)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=110)
    # Pre-compute (x, y, row) so we can find the Pareto frontier
    pts = []
    for r in rows:
        x = baseline_rt / r["runtime"] if r["runtime"] else 0
        y = (r["n_correlated"] / max_corr) if max_corr else 0
        pts.append((x, y, r))
    # Pareto front: keep points where no other has higher x AND higher y
    pareto = []
    for x, y, r in pts:
        if not any((xx >= x and yy > y) or (xx > x and yy >= y)
                   for xx, yy, _ in pts):
            pareto.append((x, y, r))
    pareto_set = {id(r) for _, _, r in pareto}
    for x, y, r in pts:
        color = _MODE_COLOR.get(r["mode"], "#888")
        is_pareto = id(r) in pareto_set
        ax.scatter(x, y, color=color, s=90 if is_pareto else 55,
                   edgecolor="#222" if is_pareto else "white",
                   linewidth=1.0 if is_pareto else 0.6,
                   alpha=0.95 if is_pareto else 0.55, zorder=3 if is_pareto else 2)
        if is_pareto:
            ax.annotate(r["label"].split("/")[-1], (x, y),
                        xytext=(8, 3), textcoords="offset points",
                        fontsize=8.5, color="#222", family="monospace",
                        weight="bold")
    # Draw the Pareto staircase
    pareto.sort(key=lambda p: p[0])
    if len(pareto) >= 2:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, color="#888", linestyle="-", alpha=0.4,
                zorder=1, label="Pareto frontier")
    ax.axhline(1.0, color="#7cd992", linestyle="--", alpha=0.6, zorder=1,
               label="perfect recall (= best-found pair count)")
    ax.axvline(1.0, color="#888", linestyle=":", alpha=0.4, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"speedup vs slowest bf ({baseline_rt:.1f}s) — log scale, higher is better")
    ax.set_ylabel("recall proxy (n_correlated / max)")
    ax.set_title("Speed ↔ recall trade-off")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    from matplotlib.patches import Patch
    handles = [Patch(color=_MODE_COLOR["bf"], label="bf"),
               Patch(color=_MODE_COLOR["corrtrack"], label="corrtrack")]
    if any(r["mode"] == "filcorr" for r in rows):
        handles.append(Patch(color=_MODE_COLOR["filcorr"], label="filcorr"))
    if any(r["mode"] == "?" for r in rows):
        handles.append(Patch(color=_MODE_COLOR["?"], label="other"))
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="#7cd992",
              linestyle="--", label="perfect recall")],
              loc="lower right", fontsize=9, framealpha=0.92)
    ax.set_ylim(-0.05, 1.10)
    return fig


def _chart_throughput(rows: list[dict]):
    """Bars per backend showing win/s and cand/s (median, with min/max ticks)."""
    by_be: dict[str, list[dict]] = {}
    for r in rows:
        be = r["backend"] or "(none)"
        by_be.setdefault(be, []).append(r)
    items = sorted(by_be.items(),
                   key=lambda kv: np.median([x["win_per_s"] or 0 for x in kv[1]]),
                   reverse=True)
    backends = [k for k, _ in items]
    win_med, win_min, win_max = [], [], []
    cand_med, cand_min, cand_max = [], [], []
    n_per_backend = []
    for _, group in items:
        ws = [x["win_per_s"] for x in group if x["win_per_s"]]
        cs = [x["cand_per_s"] for x in group if x["cand_per_s"]]
        win_med.append(np.median(ws) if ws else 0)
        win_min.append(min(ws) if ws else 0)
        win_max.append(max(ws) if ws else 0)
        cand_med.append(np.median(cs) if cs else 0)
        cand_min.append(min(cs) if cs else 0)
        cand_max.append(max(cs) if cs else 0)
        n_per_backend.append(len(group))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(3.6, 0.32 * len(backends) + 1.5)),
                                    dpi=110)
    y = np.arange(len(backends))
    # win/s
    ax1.barh(y, win_med, color="#5aa1ff", edgecolor="white", linewidth=0.5)
    ax1.errorbar(win_med, y,
                 xerr=[np.subtract(win_med, win_min), np.subtract(win_max, win_med)],
                 fmt="none", ecolor="#333", capsize=3, linewidth=0.8)
    for i, (m, n) in enumerate(zip(win_med, n_per_backend)):
        ax1.text(m, i, f"  {m:.1f} · n={n}", va="center", fontsize=8.5, color="#333")
    ax1.set_yticks(y); ax1.set_yticklabels(backends, fontsize=9, family="monospace")
    ax1.invert_yaxis()
    ax1.set_xlabel("windows / second (median, min–max)")
    ax1.set_title("Throughput · windows/s")
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    # cand/s
    ax2.barh(y, cand_med, color="#ffb05a", edgecolor="white", linewidth=0.5)
    ax2.errorbar(cand_med, y,
                 xerr=[np.subtract(cand_med, cand_min), np.subtract(cand_max, cand_med)],
                 fmt="none", ecolor="#333", capsize=3, linewidth=0.8)
    for i, m in enumerate(cand_med):
        ax2.text(m, i, f"  {m:,.0f}", va="center", fontsize=8.5, color="#333")
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("candidates / second (median, min–max)")
    ax2.set_title("Throughput · candidates/s")
    ax2.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle("Backends — aggregate throughput "
                 f"({sum(n_per_backend)} runs across {len(backends)} backends)",
                 fontsize=11, y=1.01)
    return fig


def _chart_resources(rows: list[dict]):
    """Per-run comparison of real-time resource metrics with min/median/max:
    memory (MB), energy proxy (CPU core·seconds), CPU utilisation (%), and
    concurrent cores. Min/median/max are computed from each run's resources.csv
    time series (res_series); energy is the scalar core·seconds total.

    Runs without any resource sample are skipped. If NO run has data, an
    informative panel explains the feature must be enabled at run time.
    """
    def _series(r, key):
        return [float(p[key]) for p in (r.get("res_series") or [])
                if p.get(key) is not None and np.isfinite(_safe_float(p.get(key)))]

    # Keep only runs that actually carry resource data (series or scalar).
    usable = []
    for r in rows:
        has_series = bool(r.get("res_series"))
        has_scalar = (r.get("mem_peak_mb") is not None
                      or r.get("energy_cpu_seconds") is not None)
        if has_series or has_scalar:
            usable.append(r)

    if not usable:
        fig, ax = plt.subplots(figsize=(11, 3.2), dpi=110)
        ax.text(0.5, 0.5,
                "No real-time resource data in these runs.\n\n"
                "resources.csv (CPU / memory / energy) is written only when the\n"
                "pipeline is run with real-time profiling enabled (v2 ≥ 2026-06-03).\n"
                "Re-run the pipeline to populate memory / energy charts.",
                ha="center", va="center", transform=ax.transAxes,
                color="#888", fontsize=11, linespacing=1.6)
        ax.set_axis_off()
        return fig

    # Sort by peak memory desc, cap at 20 for legibility.
    def _peak_mem(r):
        s = _series(r, "mem_mb")
        return max(s) if s else (_safe_float(r.get("mem_peak_mb")) or 0)
    usable = sorted(usable, key=_peak_mem, reverse=True)[:20]
    labels = [r["label"].split("/")[-1] for r in usable]
    colors = [_MODE_COLOR.get(r["mode"], "#888") for r in usable]
    y = np.arange(len(usable))

    def _mmm(r, key, scalar_fallback=None):
        """Return (median, min, max) from the series, or a flat triple from
        a scalar fallback when the series is absent."""
        s = _series(r, key)
        if s:
            return float(np.median(s)), float(min(s)), float(max(s))
        v = _safe_float(r.get(scalar_fallback)) if scalar_fallback else None
        return (v, v, v) if v is not None else (0.0, 0.0, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, max(4.5, 0.42 * len(usable) + 2.2)),
                             dpi=110)
    (ax_mem, ax_energy), (ax_cpu, ax_cores) = axes

    def _errbar_panel(ax, key, scalar_fallback, color, title, xlabel, fmt="{:.0f}"):
        med, lo, hi = [], [], []
        for r in usable:
            m, a, b = _mmm(r, key, scalar_fallback)
            med.append(m); lo.append(a); hi.append(b)
        ax.barh(y, med, color=color, edgecolor="white", linewidth=0.5)
        ax.errorbar(med, y,
                    xerr=[np.subtract(med, lo), np.subtract(hi, med)],
                    fmt="none", ecolor="#333", capsize=3, linewidth=0.8)
        for i, m in enumerate(med):
            ax.text(m, i, "  " + fmt.format(m), va="center", fontsize=8, color="#333")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5, family="monospace")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Memory (MB) — min/median/max from series
    _errbar_panel(ax_mem, "mem_mb", "mem_peak_mb", "#c08bff",
                  "Memory (MB) · median, min–max", "resident memory (MB)",
                  fmt="{:.0f}")
    # Energy proxy (CPU core·seconds) — scalar, single bar (no min/max)
    energy = [(_safe_float(r.get("energy_cpu_seconds")) or 0) for r in usable]
    ax_energy.barh(y, energy, color="#ffb05a", edgecolor="white", linewidth=0.5)
    for i, e in enumerate(energy):
        ax_energy.text(e, i, f"  {e:,.0f}", va="center", fontsize=8, color="#333")
    ax_energy.set_yticks(y); ax_energy.set_yticklabels([])
    ax_energy.invert_yaxis()
    ax_energy.set_xlabel("CPU core·seconds (≈ energy at constant TDP)")
    ax_energy.set_title("Energy proxy · core·seconds", fontsize=10)
    ax_energy.grid(axis="x", linestyle=":", alpha=0.4)
    # CPU utilisation (%) — min/median/max from series
    _errbar_panel(ax_cpu, "cpu_pct", "cpu_pct_median", "#5aa1ff",
                  "CPU utilisation (%) · median, min–max", "CPU %",
                  fmt="{:.0f}%")
    # Concurrent cores (power proxy) — min/median/max from series
    _errbar_panel(ax_cores, "power_cores", "power_cores_median", "#5ddcdc",
                  "Concurrent cores · median, min–max", "cores busy",
                  fmt="{:.2f}")

    fig.suptitle(f"Real-time resources — {len(usable)} run(s) "
                 "(memory · energy · CPU · cores)", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


@app.route("/api/pipeline_info")
def api_pipeline_info():
    pdir = _get_pipeline_arg()
    meta = _pipeline_meta(pdir)
    summary = meta["summary"]
    pretty = {k.replace("param.", ""): summary[k] for k in _PRETTY_KEYS if k in summary}
    result = meta.get("result", {})
    for k in ("runtime", "n_correlated", "n_windows", "n_candidates"):
        if k in result and k not in pretty:
            pretty[k] = result[k]
    source = meta.get("source")
    ids: list[str] = []
    rows = None
    if source is not None and source.is_file():
        try:
            ids = _source_ids(source)
        except Exception:
            ids = []
        try:
            rows = _source_row_count(source)
        except Exception:
            rows = None
    backend = summary.get("param.backend") or summary.get("param.index_backend") or ""
    short_meta = f"ws={meta['window_size']} · backend={backend} · " \
                 f"|correlated|={result.get('n_correlated','?')}"
    return jsonify({
        "window_size": meta["window_size"],
        "summary_pretty": pretty,
        "source": str(source) if source else None,
        "source_rows": rows,
        "ids": ids,
        "n_correlated": result.get("n_correlated", _correlated_count(pdir)),
        "short_meta": short_meta,
    })


def _get_filters() -> dict:
    def num(name, cast=float, default=None):
        v = request.args.get(name)
        if v in (None, ""):
            return default
        try:
            return cast(v)
        except ValueError:
            return default
    anchor_id = request.args.get("anchor_id", "").strip() or None
    anchor_t = num("anchor_t", int)
    # Both None → no anchor; id only → airport-wide anchor; id+t → exact window.
    return {
        "id_q": request.args.get("id_q", "").strip(),
        "min_corr": num("min_corr", float),
        "max_corr": num("max_corr", float),
        "lag_min": num("lag_min", int),
        "lag_max": num("lag_max", int),
        "exclude_self": request.args.get("exclude_self") in ("1", "true"),
        "only_self": request.args.get("only_self") in ("1", "true"),
        "anchor_id": anchor_id,
        "anchor_t": anchor_t,
    }


@app.route("/api/correlated")
def api_correlated():
    pdir = _get_pipeline_arg()
    filters = _get_filters()
    sort = request.args.get("sort", "abs_corr")
    order = request.args.get("order", "desc")
    offset = max(0, int(request.args.get("offset", 0) or 0))
    limit = max(1, min(500, int(request.args.get("limit", 50) or 50)))
    rows, total = _iter_correlated(pdir, filters=filters, sort=sort, order=order,
                                   offset=offset, limit=limit)
    # JSON-safe
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif pd.isna(v):
                r[k] = None

    # Optional: cross-engine coverage — does each pair appear in OTHER
    # pipelines too? Uses the same pair-set cache as precision/recall, so
    # it's instant once that cache is warm.
    if request.args.get("cover") in ("1", "true") and rows:
        all_pipes = _list_pipeline_dirs(STATE.results_root)
        n_total = len(all_pipes)
        # Pre-load pair sets and lightweight metadata (mode, label) once
        name_map = _run_meta_by_name()
        def _mode_of(p):
            d = name_map.get(p.name) or {}
            if d.get("mode"): return d["mode"]
            n = p.name.lower()
            if n.startswith("corrtrack"): return "corrtrack"
            if n.startswith("filcorr") or n.startswith("fillcorr"): return "filcorr"
            if n.startswith("bf"): return "bf"
            return "?"
        meta = []  # list of (path_str, label, mode, pair_set)
        for p in all_pipes:
            try:
                rel = p.relative_to(STATE.results_root)
                label = str(rel) if str(rel) != "." else p.name
            except ValueError:
                label = str(p)
            meta.append((str(p), label, _mode_of(p), _read_pairs_set(p)))
        for r in rows:
            try:
                a = (str(r["id1"]), int(r["t1"]))
                b = (str(r["id2"]), int(r["t2"]))
                key = (a, b) if a <= b else (b, a)
            except Exception:
                r["coverage"] = None
                continue
            detected = [{"path": path, "label": label, "mode": mode}
                        for path, label, mode, s in meta if key in s]
            r["coverage"] = {
                "n_runs_total": n_total,
                "n_detected": len(detected),
                "detected": detected,
            }
    return jsonify({"rows": rows, "total": int(total)})


@app.route("/api/windows")
def api_windows():
    pdir = _get_pipeline_arg()
    id_q = request.args.get("id_q", "").strip()
    top_n = max(10, min(5000, int(request.args.get("top_n", 200) or 200)))
    items = _unique_windows(pdir, id_q=id_q, top_n=top_n)
    return jsonify({"windows": items})


@app.route("/api/partners")
def api_partners():
    """For a target window (id, t), list partner windows + correlation."""
    pdir = _get_pipeline_arg()
    iq = request.args.get("id", "")
    tq = int(request.args.get("t", 0))
    p = pdir / "correlated.csv"
    out: list[dict] = []
    for chunk in pd.read_csv(p, chunksize=200_000):
        chunk["corr"] = pd.to_numeric(chunk["corr"], errors="coerce")
        for side, other in [("1", "2"), ("2", "1")]:
            m = (chunk[f"id{side}"] == iq) & (chunk[f"t{side}"] == tq)
            if not m.any():
                continue
            sub = chunk[m]
            for _, r in sub.iterrows():
                out.append({
                    "id": str(r[f"id{other}"]),
                    "t": int(r[f"t{other}"]),
                    "lag": int(r["lag"]) if not pd.isna(r["lag"]) else 0,
                    "corr": float(r["corr"]) if not pd.isna(r["corr"]) else None,
                })
    out.sort(key=lambda x: abs(x["corr"] or 0), reverse=True)
    return jsonify({"partners": out})


_AIRPORTS_CACHE: dict | None = None


def _airports() -> dict:
    """Load airports cache (ICAO → {lat, lon, name, municipality, ...}).

    Source: OurAirports (public domain, https://ourairports.com/data/).
    Filtered to entries matching the source CSV's stations. Looked up first
    in ~/.corrtrack/cache/airports.json (user override), then in the
    bundled v2/cache/airports.json that ships with the repo.
    """
    global _AIRPORTS_CACHE
    if _AIRPORTS_CACHE is not None:
        return _AIRPORTS_CACHE
    candidates = [CACHE_ROOT / "airports.json",
                  _LEGACY_CACHE / "airports.json"]
    for p in candidates:
        if p.is_file():
            try:
                _AIRPORTS_CACHE = json.loads(p.read_text())
                return _AIRPORTS_CACHE
            except Exception:
                pass
    _AIRPORTS_CACHE = {}
    return _AIRPORTS_CACHE


def _icao_of(station_id: str) -> str:
    """Extract ICAO code from station IDs like 'Ajaccio_LFKJ' → 'LFKJ'."""
    parts = str(station_id).split("_")
    return parts[-1].upper() if parts else ""


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points in km."""
    from math import radians, sin, cos, sqrt, asin
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


@app.route("/api/airports")
def api_airports():
    """Return airport metadata. With `icaos=A,B,C` filter, returns just those.

    Each entry: {icao, name, lat, lon, municipality, type, elevation_ft}.
    """
    db = _airports()
    icaos_arg = request.args.get("icaos", "").strip()
    if icaos_arg:
        wanted = {x.strip().upper() for x in icaos_arg.split(",") if x.strip()}
        out = {k: v for k, v in db.items() if k in wanted}
        missing = sorted(wanted - set(out.keys()))
    else:
        out = db
        missing = []
    return jsonify({"airports": out, "n": len(out), "missing": missing})


@app.route("/api/distance")
def api_distance():
    """Compute the haversine distance (km) between two station IDs."""
    a = _icao_of(request.args.get("a", ""))
    b = _icao_of(request.args.get("b", ""))
    db = _airports()
    pa, pb = db.get(a), db.get(b)
    if not pa or not pb:
        return jsonify({"error": f"unknown ICAO: {a if not pa else b}",
                        "a": a, "b": b}), 404
    d = _haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])
    return jsonify({"a": pa, "b": pb, "distance_km": round(d, 2)})


@app.route("/api/window")
def api_window():
    pdir = _get_pipeline_arg()
    meta = _pipeline_meta(pdir)
    source = meta.get("source")
    if source is None:
        return jsonify({"error": "no source CSV"}), 400
    iq = request.args.get("id", "")
    tq = int(request.args.get("t", 0))
    ws = meta["window_size"]
    xs, vals = _read_window_values(source, iq, tq, ws)
    # add the start/end real-world datetimes
    date_start_ms = date_end_ms = None
    if xs.size:
        hours = _source_times(source)
        start_row = int(np.searchsorted(hours, int(xs[0])))
        dates = _source_datetimes(source)
        if start_row < len(dates):
            date_start_ms = int(dates[start_row].astype("int64"))
            end_row = min(len(dates) - 1, start_row + len(xs) - 1)
            date_end_ms = int(dates[end_row].astype("int64"))
    return jsonify({
        "id": iq, "t": tq, "window_size": ws,
        "xs": xs.tolist(),
        "values": [None if not math.isfinite(v) else v for v in vals.tolist()],
        "stats": _window_stats(vals),
        "date_start_ms": date_start_ms,
        "date_end_ms": date_end_ms,
    })


def _parse_w(w: str) -> tuple[str, int, str | None]:
    parts = w.split("|")
    sid = parts[0]
    t = int(parts[1])
    color = parts[2] if len(parts) > 2 else None
    return sid, t, color


@app.route("/api/corr_matrix")
def api_corr_matrix():
    pdir = _get_pipeline_arg()
    meta = _pipeline_meta(pdir)
    source = meta.get("source")
    if source is None:
        return jsonify({"error": "no source CSV"}), 400
    ws = meta["window_size"]
    ws_list = request.args.getlist("w")
    methods = request.args.getlist("method") or list(_CORR_METHODS)
    bl_max_lag = int(request.args.get("bl_max_lag", ws // 2) or (ws // 2))

    series, labels = [], []
    for w in ws_list:
        sid, t, _ = _parse_w(w)
        _, vals = _read_window_values(source, sid, t, ws)
        series.append(vals)
        labels.append(f"{sid}@{t}")
    n = len(series)

    out: dict[str, Any] = {"labels": labels, "matrices": {}, "best_lag": None,
                          "window_size": ws}
    for meth in methods:
        if meth not in _CORR_METHODS:
            continue
        M = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i][j] = 1.0
                else:
                    v = _corr_pair(series[i], series[j], meth)
                    M[i][j] = None if pd.isna(v) else v
        out["matrices"][meth] = M

    if request.args.get("best_lag") in ("1", "true"):
        BC = [[None] * n for _ in range(n)]
        BL = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    BC[i][j], BL[i][j] = 1.0, 0
                else:
                    c, lag = _best_lag_pearson(series[i], series[j], bl_max_lag)
                    BC[i][j] = None if not np.isfinite(c) else c
                    BL[i][j] = int(lag)
        out["best_lag"] = {"corr": BC, "lag": BL, "max_lag": bl_max_lag}
    return jsonify(out)


def _window_payload(source: Path, sid: str, t: int, length: int, pad: int = 0,
                    color: str | None = None) -> dict | None:
    """Read a slice + dates and convert to a JSON-safe payload for the SVG plot."""
    try:
        xs, vals = _read_window_values(source, sid, t, length, pad=pad)
    except Exception:
        return None
    if xs.size == 0:
        return None
    # Map this slice to the cached datetime array by row index
    hours = _source_times(source)
    start = int(np.searchsorted(hours, int(xs[0])))
    dates = _source_datetimes(source)[start:start + len(xs)]
    return {
        "id": sid, "t": int(t), "color": color or "#5aa1ff",
        "label": f"{sid}@{t}",
        "xs": [int(x) for x in xs],
        "values": [None if not np.isfinite(v) else float(v) for v in vals],
        "dates_ms": [int(d) for d in dates.astype("int64")],
    }


@app.route("/api/plot_data")
def api_plot_data():
    """JSON payload for the client-side SVG time-series plot.

    Returns the basket windows + (when not aligned) faded history lines per
    series, each with `xs` (hour offsets), `values`, and `dates_ms`
    (epoch-ms timestamps for axis formatting).
    """
    pdir = _get_pipeline_arg()
    meta = _pipeline_meta(pdir)
    source = meta.get("source")
    if source is None:
        return jsonify({"error": "no source CSV"}), 400
    ws = meta["window_size"]
    ws_list = request.args.getlist("w")
    align = request.args.get("align", "1") in ("1", "true")
    history = request.args.get("history", "1") in ("1", "true")
    pad = int(request.args.get("pad", 0) or 0)

    parsed = [_parse_w(w) for w in ws_list]
    windows: list[dict] = []
    by_id: dict[str, dict] = {}
    for sid, t, color in parsed:
        wpad = 0 if (history and not align) else pad
        p = _window_payload(source, sid, t, ws, pad=wpad, color=color)
        if p is None:
            continue
        windows.append(p)
        d = by_id.setdefault(sid, {"color": color, "ts": []})
        d["ts"].append(t)
        if d["color"] is None and color:
            d["color"] = color

    history_lines: list[dict] = []
    if history and not align and by_id:
        all_ts = [t for info in by_id.values() for t in info["ts"]]
        gtmin = min(all_ts)
        gtmax = max(all_ts) + ws
        ctx_pad = max(pad, max(24, ws // 2))
        for sid, info in by_id.items():
            eff_min = min(min(info["ts"]), gtmin)
            eff_max = max(max(info["ts"]) + ws, gtmax)
            length = eff_max - eff_min
            p = _window_payload(source, sid, eff_min, length,
                                pad=ctx_pad, color=info["color"])
            if p is None:
                continue
            history_lines.append(p)

    return jsonify({
        "pipeline": pdir.name,
        "window_size": ws,
        "align": align,
        "history": history,
        "windows": windows,
        "history_lines": history_lines,
    })


@app.route("/api/plot.png")
def api_plot_png():
    pdir = _get_pipeline_arg()
    meta = _pipeline_meta(pdir)
    source = meta.get("source")
    if source is None:
        return Response("no source CSV available; pass --dataset on launch", status=400)
    ws = meta["window_size"]
    ws_list = request.args.getlist("w")
    normalize = request.args.get("normalize") in ("1", "true")
    align = request.args.get("align", "1") in ("1", "true")
    history = request.args.get("history", "1") in ("1", "true")
    points = request.args.get("points", "1") in ("1", "true")
    grid_on = request.args.get("grid", "1") in ("1", "true")
    pad = int(request.args.get("pad", 0) or 0)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=110)
    if not ws_list:
        ax.text(0.5, 0.5, "Add windows to the basket to plot.",
                ha="center", va="center", transform=ax.transAxes,
                color="#888")
        ax.set_axis_off()
    else:
        parsed = [_parse_w(w) for w in ws_list]

        def _z(y):
            if not normalize:
                return y
            m, s = np.nanmean(y), np.nanstd(y)
            return (y - m) / s if (s and not np.isnan(s)) else y

        # History pass — only meaningful when not aligned: for each id, draw the
        # full underlying series over the union range of its selected windows
        # (faded, behind everything else). Same z-norm/pad treatment as windows.
        if history and not align:
            by_id: dict[str, dict] = {}
            for sid, t, color in parsed:
                d = by_id.setdefault(sid, {"color": color, "ts": []})
                d["ts"].append(t)
                if d["color"] is None and color:
                    d["color"] = color
            # Per-series small context pad (≈ half a window on each side); the
            # user's `pad` can extend it further.
            default_history_pad = max(24, ws // 2)
            ctx_pad = max(pad, default_history_pad)
            # Global t-range across ALL basket windows: stretch every series to
            # cover that span so we can compare them on a single time axis when
            # the windows are far apart (instead of disconnected bubbles).
            all_ts = [t for info in by_id.values() for t in info["ts"]]
            global_tmin = min(all_ts)
            global_tmax = max(all_ts) + ws
            for sid, info in by_id.items():
                eff_min = min(min(info["ts"]), global_tmin)
                eff_max = max(max(info["ts"]) + ws, global_tmax)
                length = eff_max - eff_min
                try:
                    xs_h, vals_h = _read_window_values(source, sid, eff_min,
                                                      length, pad=ctx_pad)
                except Exception:
                    continue
                ax.plot(xs_h, _z(vals_h), color=info["color"] or "#888",
                        alpha=0.22, linewidth=1.0, zorder=1)

        # Window pass — solid lines on top of history
        for sid, t, color in parsed:
            try:
                # when history is showing, the per-window pad is already in the
                # history line; keep only the highlighted slice strong
                wpad = 0 if (history and not align) else pad
                xs, vals = _read_window_values(source, sid, t, ws, pad=wpad)
            except Exception as e:
                ax.text(0.5, 0.5, f"error: {e}", ha="center", va="center",
                        transform=ax.transAxes, color="red")
                continue
            x_plot = xs - t if align else xs
            y_plot = _z(vals.copy())
            marker = "o" if points else None
            # adapt marker size to density to stay readable
            ms = 3.0 if len(y_plot) <= 200 else (2.0 if len(y_plot) <= 600 else 1.2)
            ax.plot(x_plot, y_plot, color=color or None,
                    label=f"{sid}@{t}", linewidth=1.4, zorder=3,
                    marker=marker, markersize=ms,
                    markeredgecolor=color or None,
                    markerfacecolor=color or None)
            ax.axvspan((0 if align else t), (ws - 1 if align else t + ws - 1),
                       color=color or "#888", alpha=0.10, linewidth=0, zorder=2)

        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_xlabel("t" + (" (relative)" if align else " (hour index)"))
        ax.set_ylabel("z-score" if normalize else "value")
        if grid_on:
            ax.grid(True, linestyle=":", alpha=0.4)
        title_bits = [f"{pdir.name}", f"window_size={ws}"]
        if pad:
            title_bits.append(f"pad=±{pad}")
        if history and not align:
            title_bits.append("history shown")
        ax.set_title(" · ".join(title_bits))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.read(), mimetype="image/png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    global STATE
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pipeline", default=None,
                   help="path to a pipeline-*.json (derives --results and "
                        "--dataset from its `output` and `dataset` fields). "
                        "Also accepts `user@host:/abs/path/pipeline.json` to "
                        "pull results from a remote host via SSH+rsync "
                        "(key-based auth required).")
    p.add_argument("--results", default=None,
                   help="path to the results directory to browse "
                        "(overridden by --pipeline if both are given)")
    p.add_argument("--dataset", "--source", dest="dataset", default=None,
                   metavar="DATASET",
                   help="path to the source CSV. Overrides the `dataset` field "
                        "of the --pipeline JSON when both are given. Default: "
                        "auto-detect from --pipeline or from a pipeline-*.json "
                        "sitting next to --results. (`--source` kept as a "
                        "backward-compat alias.)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-prewarm", action="store_true",
                   help="Skip the precision/recall pre-compute step and start "
                        "the web server immediately. Metrics are then computed "
                        "lazily on the first browser request instead.")
    p.add_argument("--sync-interval", type=float, default=5.0,
                   help="Remote-mode only: seconds between background rsync "
                        "cycles. Default 5.0. Set to 0 to disable.")
    p.add_argument("--ssh-timeout", type=int, default=30,
                   help="Remote-mode only: ssh ConnectTimeout in seconds "
                        "(applied to test_ssh, remote probes and rsync's "
                        "ssh transport). Default 30. Bump it if you're on "
                        "a slow link / VPN and see 'ssh connection timed out'.")
    args = p.parse_args(argv)

    # Resolve results dir + source CSV.
    # Priority order: --pipeline > --results · --dataset > pipeline.dataset.
    pipeline_obj = None
    pipeline_json_path = None
    mirror: RemoteMirror | None = None
    # Convenience: `--results pipeline-run.json` is auto-promoted to --pipeline.
    if args.results and not args.pipeline:
        rp = Path(args.results).expanduser()
        if rp.is_file() and rp.suffix == ".json":
            print(f"[viz] --results points to a JSON file → treating as --pipeline",
                  file=sys.stderr)
            args.pipeline = str(rp)
            args.results = None

    # Detect remote pipeline spec — `[user@]host:/abs/path/pipeline.json`.
    # On detection we mirror everything we need locally, then keep going as if
    # the user had passed local paths.
    remote_spec = _parse_remote(args.pipeline)
    remote_dataset_spec = _parse_remote(args.dataset)
    if remote_spec is not None:
        cache_root = _ensure_cache_root()
        mirror = RemoteMirror(remote_spec, cache_root,
                               ssh_connect_timeout=args.ssh_timeout)
        print(f"[viz] remote pipeline: {remote_spec.target}:{remote_spec.path}",
              file=sys.stderr)
        print(f"[viz] mirror dir: {mirror.local_root}", file=sys.stderr)
        ok, err = mirror.test_ssh()
        if not ok:
            print(f"[viz] SSH to {remote_spec.target} failed: {err}",
                  file=sys.stderr)
            print(f"[viz] {RemoteMirror._ssh_hint(remote_spec.target)}",
                  file=sys.stderr)
            return 2
        # 1) Fetch the pipeline JSON itself.
        try:
            local_pj = mirror.fetch_file(remote_spec.path)
        except RuntimeError as e:
            print(f"[viz] could not fetch pipeline JSON: {e}", file=sys.stderr)
            return 2
        # Also grab any sibling `pipeline-*.json` (used to recover run metadata).
        remote_pj_dir = str(Path(remote_spec.path).parent.as_posix())
        for name in mirror.list_remote(remote_pj_dir, pattern="pipeline-*.json"):
            try:
                mirror.fetch_file(f"{remote_pj_dir.rstrip('/')}/{name}")
            except RuntimeError as e:
                print(f"[viz] warn: could not fetch {name}: {e}", file=sys.stderr)
        args.pipeline = str(local_pj)
        # 2) Resolve remote output + dataset abs paths from the JSON, then
        # rewrite args.dataset / mirror them so downstream code reads locals.
        try:
            _pj_obj = _load_jsonc(local_pj.read_text())
        except Exception as e:
            print(f"[viz] failed to parse fetched pipeline JSON: {e}",
                  file=sys.stderr)
            return 2
        _out = _pj_obj.get("output") or "results"
        if Path(_out).is_absolute():
            remote_out_abs = _out
        else:
            remote_out_abs = str(
                (Path(remote_spec.path).parent / _out).as_posix())
        print(f"[viz] initial rsync from {remote_spec.target}:{remote_out_abs}",
              file=sys.stderr)
        # Pre-check: a missing remote output dir is normal at the very start
        # of a fresh pipeline run (no run has produced anything yet). We
        # surface it clearly and continue with an empty mirror — the periodic
        # sync will pick the dir up as soon as it appears.
        exists, kind = mirror.remote_path_exists(remote_out_abs)
        if not exists:
            print(f"[viz] WARNING: remote output dir does not exist yet — "
                  f"starting with empty local mirror; runs will appear as the "
                  f"pipeline creates them on the server.", file=sys.stderr)
            mirror.remote_to_local(remote_out_abs).mkdir(parents=True, exist_ok=True)
        elif kind != "dir":
            print(f"[viz] ERROR: remote path exists but is not a directory "
                  f"(kind={kind}): {remote_out_abs}", file=sys.stderr)
            return 2
        else:
            try:
                mirror.sync_tree(remote_out_abs,
                                 progress=sys.stderr.isatty())
            except RuntimeError as e:
                print(f"[viz] {e}", file=sys.stderr)
                return 2
        # The local mirror path corresponding to remote_out_abs.
        local_out = mirror.remote_to_local(remote_out_abs)
        # If `output` was absolute (would resolve wrong locally), rewrite the
        # JSON in-memory so the existing downstream code resolves correctly.
        # We do this by replacing `output` and `dataset` with their local-
        # mirror equivalents BEFORE re-running the standard parsing below.
        if Path(_out).is_absolute():
            _pj_obj["output"] = str(local_out)
        # Dataset: --dataset overrides; otherwise use JSON's dataset.
        if args.dataset is None and remote_dataset_spec is None:
            ds = _pj_obj.get("dataset")
            if ds:
                if Path(ds).is_absolute():
                    ds_abs = ds
                else:
                    ds_abs = str(
                        (Path(remote_spec.path).parent / ds).as_posix())
                print(f"[viz] fetching dataset {remote_spec.target}:{ds_abs}",
                      file=sys.stderr)
                try:
                    local_ds = mirror.fetch_file(
                        ds_abs, progress=sys.stderr.isatty())
                except RuntimeError as e:
                    print(f"[viz] dataset fetch failed: {e}", file=sys.stderr)
                    return 2
                args.dataset = str(local_ds)
                # Drop the JSON's dataset field so downstream resolution
                # doesn't compute a stale local path from a remote string.
                _pj_obj.pop("dataset", None)
        # Persist the rewritten JSON to disk so _run_meta_by_name() and
        # _find_dataset_near() see consistent local paths.
        local_pj.write_text(json.dumps(_pj_obj, indent=2))
        # Background refresh (disabled when --sync-interval=0).
        if args.sync_interval > 0:
            mirror.start_periodic_sync([remote_out_abs],
                                        interval=args.sync_interval)
            print(f"[viz] background sync every {args.sync_interval:.1f}s "
                  f"(POST /api/sync_now to force, GET /api/sync_status for state)",
                  file=sys.stderr)
        else:
            mirror._sync_dirs = [remote_out_abs]
            print(f"[viz] background sync disabled — use POST /api/sync_now",
                  file=sys.stderr)

    # Same handling if --dataset alone is remote (rare, but symmetric).
    if remote_dataset_spec is not None:
        if mirror is None:
            cache_root = _ensure_cache_root()
            mirror = RemoteMirror(remote_dataset_spec, cache_root,
                                   ssh_connect_timeout=args.ssh_timeout)
            ok, err = mirror.test_ssh()
            if not ok:
                print(f"[viz] SSH to {remote_dataset_spec.target} failed: {err}",
                      file=sys.stderr)
                print(f"[viz] {RemoteMirror._ssh_hint(remote_dataset_spec.target)}",
                      file=sys.stderr)
                return 2
        try:
            local_ds = mirror.fetch_file(
                remote_dataset_spec.path, progress=sys.stderr.isatty())
        except RuntimeError as e:
            print(f"[viz] dataset fetch failed: {e}", file=sys.stderr)
            return 2
        args.dataset = str(local_ds)

    if args.pipeline:
        pj = Path(args.pipeline).expanduser().resolve()
        if not pj.is_file():
            print(f"pipeline JSON not found: {pj}", file=sys.stderr)
            return 2
        try:
            pipeline_obj = _load_jsonc(pj.read_text())
        except Exception as e:
            print(f"failed to parse pipeline JSON: {e}", file=sys.stderr)
            return 2
        pipeline_json_path = pj
        # Mirror EXACTLY the path layout used by v2/pipeline.py:
        #     base = os.path.join(spec.get("output", "results"),
        #                         spec.get("name", "pipeline"))
        # Both fields are read from the SAME JSON the viz was given, so any
        # config (name=foo, output=bar, name omitted, output omitted, ...)
        # is handled identically by the runner and the viewer.
        out = pipeline_obj.get("output") or "results"
        base_out = (pj.parent / out).resolve() if not Path(out).is_absolute() else Path(out).resolve()
        pj_name = pipeline_obj.get("name") or "pipeline"   # same default as the runner
        results = base_out / pj_name
        if not results.is_dir() and base_out.is_dir():
            # Named subdir doesn't exist yet (pipeline not started yet) —
            # fall back to base_out so the page still loads and "queued"
            # runs show up. Logged so the user notices the difference.
            print(f"[viz] pipeline `{pj_name}` has no subdir under {base_out} "
                  f"yet — falling back to {base_out} "
                  f"(sibling pipelines may appear until the named subdir exists)",
                  file=sys.stderr)
            results = base_out
        print(f"[viz] from pipeline {pj.name}: name={pj_name} · output → {results}",
              file=sys.stderr)
        if args.dataset:
            # --dataset (or --source alias) explicitly given on the CLI → takes
            # precedence over the JSON's `dataset` field.
            print(f"[viz] --dataset override: dataset = {args.dataset}  "
                  f"(ignoring JSON dataset `{pipeline_obj.get('dataset','')}`)",
                  file=sys.stderr)
        else:
            ds = pipeline_obj.get("dataset")
            if ds:
                args.dataset = str((pj.parent / ds).resolve()
                                  if not Path(ds).is_absolute() else Path(ds))
                print(f"[viz] from pipeline {pj.name}: dataset → {args.dataset}",
                      file=sys.stderr)
    else:
        results = Path(args.results or "results").expanduser().resolve()
    if not results.is_dir():
        print(f"results dir not found: {results}", file=sys.stderr)
        return 2
    source = Path(args.dataset).expanduser().resolve() if args.dataset else None
    if source is None:
        guessed = _find_dataset_near(results)
        if guessed is not None:
            print(f"[viz] using auto-detected source CSV: {guessed}", file=sys.stderr)
            source = guessed
        else:
            print("[viz] no --dataset given and no pipeline-*.json found nearby — "
                  "window plots will be disabled until you pass --dataset",
                  file=sys.stderr)
    if source is not None and not source.is_file():
        print(f"source CSV not found: {source}", file=sys.stderr)
        return 2

    STATE = AppState(results_root=results, source_csv=source,
                     cache={}, lock=Lock(),
                     pipeline_json=pipeline_json_path,
                     pipeline_obj=pipeline_obj,
                     mirror=mirror)
    # Precision/recall cache warm-up. Two modes depending on whether any run
    # is still in progress:
    #   • ALL runs done  → BLOCKING pre-warm (metrics ready on first load).
    #   • runs in progress → BACKGROUND pre-warm (daemon thread) so the web
    #     server comes up immediately and you can watch live progress; the
    #     finished runs' metrics fill in as the thread completes, and the
    #     auto-refresh recomputes each run as it finishes.
    # In both cases the cache is checkpointed every 10 runs and resumable.
    # Skip entirely with --no-prewarm.
    def _do_prewarm():
        try:
            res = _compute_quality(None)
            if "error" in res:
                print(f"[viz] quality pre-warm skipped: {res['error']}",
                      file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            _save_quality_cache()
            print("\n[viz] quality pre-warm interrupted — partial progress "
                  "saved (resumes on next launch).", file=sys.stderr, flush=True)
        except Exception as e:  # pragma: no cover - best-effort
            print(f"[viz] quality pre-warm failed: {e}",
                  file=sys.stderr, flush=True)

    if not getattr(args, "no_prewarm", False):
        # Cheap status sweep to decide blocking vs background.
        n_active = 0
        try:
            for d in _list_pipeline_dirs(results):
                st, _ = _detect_run_status(d)
                if st in ("running", "starting"):
                    n_active += 1
        except Exception:
            n_active = 0
        if n_active > 0:
            print(f"[viz] {n_active} run(s) in progress → computing "
                  f"precision/recall in the BACKGROUND (server starts now; "
                  f"finished runs fill in live)…", file=sys.stderr, flush=True)
            threading.Thread(target=_do_prewarm, daemon=True).start()
        else:
            print("[viz] pre-computing precision/recall cache before serving "
                  "(Ctrl+C to skip — progress is saved & resumable)…",
                  file=sys.stderr, flush=True)
            _do_prewarm()

    print(f"[viz] serving on http://{args.host}:{args.port}/  "
          f"(results={results}, source={source})", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
