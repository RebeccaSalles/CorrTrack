"""Shared plumbing for the competitor-dataset fetch/generate scripts.

Every script in this folder ends by calling :func:`save_competitor_npz`, which writes the
canonical on-disk form read by ``datasets/competitor_loader.py``:

    datasets/competitor/<name>.npz
        data : float64 (m, T)   series x time, NaN where the source had no value
        ids  : str     (m,)     one label per series
        meta : str              JSON: source, fetch date, alignment choices, coverage

The ``(m, T)`` orientation matches the ``tmp_artifacts/<name>/<name>.npz`` files the
2026-09-16 Sobol sweep already built for sp500 / streamflow / smartmeter / acwi / wikipedia /
global_weather and for CorrJoin's five Drive files, so one loader serves all of them.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "datasets" / "competitor"
RAW_DIR = OUT_DIR / "raw"


def download(url: str, dest: Path, *, timeout: int = 600, retries: int = 3, quiet: bool = False) -> Path:
    """Stream ``url`` to ``dest`` unless it already exists. Returns ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if not quiet:
            print(f"  cached  {dest.name}", flush=True)
        return dest
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "corrtrack-datasets/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest.with_suffix(dest.suffix + ".part"), "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            os.replace(dest.with_suffix(dest.suffix + ".part"), dest)
            if not quiet:
                print(f"  fetched {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
            return dest
        except Exception as exc:  # noqa: BLE001 - report and retry
            last = exc
            print(f"  attempt {attempt}/{retries} failed for {url}: {exc}", file=sys.stderr, flush=True)
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def open_maybe_gzip(path: Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8", errors=None if "b" in mode else "replace")
    return open(path, mode, encoding=None if "b" in mode else "utf-8", errors=None if "b" in mode else "replace")


def forward_fill(data: np.ndarray, max_gap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill NaNs along axis 1 of an ``(m, T)`` array, then back-fill the leading run.

    ``max_gap`` bounds the length of a filled run; longer gaps stay NaN. Returns the filled
    array and a boolean ``(m, T)`` mask of positions that were *observed* (not filled).
    """
    data = np.asarray(data, dtype=np.float64)
    observed = ~np.isnan(data)
    m, T = data.shape
    out = data.copy()
    idx = np.where(observed, np.arange(T)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    rows = np.arange(m)[:, None]
    have = last >= 0
    filled = np.where(have, data[rows, np.maximum(last, 0)], np.nan)
    if max_gap is not None:
        gap = np.arange(T)[None, :] - last
        filled = np.where(have & (gap <= max_gap), filled, np.nan)
    out = filled
    # back-fill the leading NaN run with the first observation (bounded by max_gap too)
    first_obs = np.argmax(observed, axis=1)
    has_any = observed.any(axis=1)
    for i in range(m):
        if not has_any[i]:
            continue
        f = first_obs[i]
        if f > 0:
            lead = f if max_gap is None else min(f, max_gap)
            out[i, f - lead:f] = data[i, f]
    return out, observed


def coverage(observed: np.ndarray) -> np.ndarray:
    return observed.mean(axis=1)


def save_competitor_npz(name: str, data: np.ndarray, ids, meta: dict, *, out_dir: Path = OUT_DIR) -> Path:
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    ids = np.asarray([str(s) for s in ids])
    if data.ndim != 2 or data.shape[0] != ids.shape[0]:
        raise ValueError(f"data must be (m, T) with m == len(ids); got {data.shape} and {ids.shape}")
    meta = dict(meta)
    meta.setdefault("name", name)
    meta.setdefault("shape_m_T", list(data.shape))
    meta.setdefault("nan_fraction", float(np.isnan(data).mean()))
    meta.setdefault("written_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, data=data, ids=ids, meta=json.dumps(meta, indent=1, default=str))
    print(f"wrote {path}  m={data.shape[0]} T={data.shape[1]} nan={meta['nan_fraction']:.4f}", flush=True)
    return path
