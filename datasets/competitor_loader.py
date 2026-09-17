"""Loader for the competitor-paper datasets (implementation plan phase 0f).

Bridges the ``(m, T)`` npz files written by ``datasets/fetch/*.py`` (and by the 2026-09-16
Sobol sweep into ``tmp_artifacts/<name>/<name>.npz``) to the ``DATA_LOADER(country, variable)
-> (data, ids)`` contract used by ``corrtrack_run_*.py``: ``data`` is ``(T, 1 + m)`` with an
integer sample index in column 0 (the runners accept non-datetime indices, as the synthetic
loader already relies on), ``ids`` has length ``m``.

    from functools import partial
    from datasets.competitor_loader import load_dataset
    DATA_LOADER = partial(load_dataset, name="motes_temperature")

``variable`` is accepted and ignored so the partial slots into the existing config pattern;
``name`` selects the file. ``max_series`` / ``max_obs`` truncate for smoke tests. Series that are
entirely NaN are dropped; remaining NaNs are forward-filled (the fetch scripts already do this
with a bounded gap and record coverage in ``meta``; this is a last line of defence so the
runners never see NaN).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEARCH_DIRS: tuple[Path, ...] = (
    REPO / "datasets" / "competitor",
    REPO / "tmp_artifacts",
)


def resolve_path(name: str) -> Path:
    for base in SEARCH_DIRS:
        for cand in (base / f"{name}.npz", base / name / f"{name}.npz"):
            if cand.exists():
                return cand
    # the 2026-09 sweep assets are not all named after their folder (finance_sectors/sp500.npz,
    # streamflow/ca_streamflow.npz): one level of glob below tmp_artifacts
    for base in SEARCH_DIRS:
        hits = sorted(base.glob(f"*/{name}.npz"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"competitor dataset {name!r} not found under {[str(d) for d in SEARCH_DIRS]}; "
        f"run the matching script in datasets/fetch/ (see datasets/competitor_sources.md)"
    )


def load_raw(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = resolve_path(name)
    with np.load(path, allow_pickle=True) as z:
        data = np.asarray(z["data"], dtype=np.float64)
        ids = np.asarray(z["ids"]).astype(str)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if data.ndim != 2:
        raise ValueError(f"{path}: expected (m, T), got {data.shape}")
    if data.shape[0] != ids.shape[0] and data.shape[1] == ids.shape[0]:
        data = data.T  # tolerate a (T, m) file
    if data.shape[0] != ids.shape[0]:
        raise ValueError(f"{path}: data {data.shape} does not match ids {ids.shape}")
    meta.setdefault("path", str(path))
    return data, ids, meta


def _ffill_rows(data: np.ndarray) -> np.ndarray:
    m, T = data.shape
    obs = ~np.isnan(data)
    idx = np.where(obs, np.arange(T)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    out = data[np.arange(m)[:, None], np.maximum(last, 0)]
    out = np.where(last >= 0, out, np.nan)
    first = np.argmax(obs, axis=1)
    for i in range(m):
        if first[i] > 0:
            out[i, : first[i]] = data[i, first[i]]
    return out


def load_dataset(
    country: str = "",
    variable: str = "",
    *,
    name: str | None = None,
    max_series: int | None = None,
    max_obs: int | None = None,
    series_ids: Iterable[str] | None = None,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    name = name or country
    data, ids, _meta = load_raw(name)
    if series_ids is not None:
        wanted = set(map(str, series_ids))
        keep = np.array([s in wanted for s in ids], dtype=bool)
        data, ids = data[keep], ids[keep]
    alive = ~np.all(np.isnan(data), axis=1)
    if not alive.all():
        data, ids = data[alive], ids[alive]
    if max_series is not None:
        data, ids = data[: int(max_series)], ids[: int(max_series)]
    if max_obs is not None:
        data = data[:, : int(max_obs)]
    if np.isnan(data).any():
        data = _ffill_rows(data)
    T = data.shape[1]
    out = np.empty((T, 1 + data.shape[0]), dtype=dtype)
    out[:, 0] = np.arange(T, dtype=dtype)
    out[:, 1:] = data.T
    return out, np.asarray(ids, dtype=object)
