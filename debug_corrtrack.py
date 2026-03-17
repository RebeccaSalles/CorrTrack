"""Comprehensive CorrTrack debugging runner.

This script orchestrates the standard CorrTrack experiment pipeline, samples
representative FN/TP/TN window pairs from the brute-force and CorrTrack
artifacts, and then replays the CorrTrack run with detailed instrumentation to
collect per-pair diagnostics (raw/normalized windows, sketch vectors, bucket
assignments, and collision statistics). Output artifacts consist of JSON/MD
reports plus plots for each tracked pair.
"""

from __future__ import annotations

import argparse
import json
import inspect
import math
import os
import subprocess
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from library_corrtrack_parallel import (
    CorrTrack,
    _coerce_to_bool,
    _resolve_cell_stretch,
    _extract_feature_overrides,
    execute_corrtrack_pass,
)
import corrtrack_run_corrtrack as corrtrack_main

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

PairKey = Tuple[str, str, int, int, int]
SeriesKey = Tuple[str, int, int]


@dataclass
class PairSelection:
    """Represents a sampled pair and its label (FN/TP/TN)."""

    dataset_id: str
    alg: str
    label: str
    id1: str
    id2: str
    start1: int
    start2: int
    window: int
    bf_corr: Optional[float] = None
    main_corr: Optional[float] = None
    freq: Optional[float] = None
    priority: int = 0

    def canonical_key(self) -> PairKey:
        return normalize_pair_key(self.id1, self.id2, self.start1, self.start2, self.window)

    @property
    def series_keys(self) -> Tuple[SeriesKey, SeriesKey]:
        return ((self.id1, self.start1, self.window), (self.id2, self.start2, self.window))


@dataclass
class SeriesWindowRecord:
    series_id: str
    times: List[float]
    values: List[float]
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class BucketRecord:
    series_id: str
    grid_index: int
    cells: List[Tuple[Tuple[int, ...], int]]


@dataclass
class PairDebugState:
    selection: PairSelection
    raw_windows: Optional[Dict[str, object]] = None
    normalized_windows: Optional[Dict[str, object]] = None
    sketches: Optional[Dict[str, object]] = None
    buckets: Optional[Dict[str, object]] = None
    bucket_stats: Dict[int, Dict[str, object]] = field(default_factory=dict)
    corrtrack_detection: Optional[bool] = None
    corrtrack_corr: Optional[float] = None
    freq_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "dataset_id": self.selection.dataset_id,
            "algorithm": self.selection.alg,
            "label": self.selection.label,
            "id1": self.selection.id1,
            "id2": self.selection.id2,
            "start1": self.selection.start1,
            "start2": self.selection.start2,
            "window": self.selection.window,
            "bf_corr": self.selection.bf_corr,
            "main_corr": self.selection.main_corr,
            "raw": self.raw_windows,
            "normalized": self.normalized_windows,
            "sketches": self.sketches,
            "buckets": self.buckets,
            "freq_history": self.freq_history,
            "found_by_corrtrack": self.corrtrack_detection,
            "corrtrack_corr": self.corrtrack_corr,
        }
        return payload


@dataclass
class SelectionBundle:
    selections: List[PairSelection]
    bf_pairs: Dict[PairKey, float]
    main_pairs: Dict[PairKey, float]
    tn_budget: int
    samples_per_class: int
    order_map: Dict[PairKey, int]
    candidate_pairs: Set[PairKey]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def parse_float(value: str | float | int | None) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    try:
        if value.strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def parse_int(value: str | float | int | None) -> Optional[int]:
    fv = parse_float(value)
    if fv is None:
        return None
    return int(round(fv))


def normalize_pair_key(id1: str, id2: str, start1: int, start2: int, window: int) -> PairKey:
    """Normalizes pair keys following Candidates._normalize_key rules."""

    id1 = str(id1)
    id2 = str(id2)
    t1 = int(start1)
    t2 = int(start2)
    w = int(window)

    if id1 == id2:
        return (id1, id2, max(t1, t2), min(t1, t2), w)

    if t1 == t2:
        a, b = sorted((id1, id2))
        return (a, b, t1, t2, w)

    if t1 < t2:
        return (id2, id1, t2, t1, w)
    return (id1, id2, t1, t2, w)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _fmt_float(value, digits=2):
    if value is None:
        return "nan"
    try:
        return f"{float(value):.{digits}f}"
    except (ValueError, TypeError):
        return "nan"


def _compute_normalized_coords(vector: np.ndarray, grid_dim: int, neg_corr: bool) -> List[float]:
    arr = np.asarray(vector, dtype=float)
    if arr.size == 0 or grid_dim <= 0:
        return []
    slice_arr = arr[:grid_dim]
    if neg_corr:
        normalized = 0.5 * (slice_arr + 1.0)
    else:
        normalized = np.abs(slice_arr)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.tolist()


def _normalize_segment(segment: np.ndarray, neg_corr: bool) -> List[float]:
    arr = np.asarray(segment, dtype=float)
    coords = np.clip(arr, -1.0, 1.0)
    return coords.tolist()


def _flatten_coords(
    entry: Dict[str, object],
    grid_dim: Optional[int],
    grid_order: Optional[Sequence[int]] = None,
) -> Optional[List[float]]:
    grid_coords = entry.get("grid_coords")
    vector = entry.get("vector")
    if not isinstance(grid_coords, dict) or not grid_coords or not isinstance(vector, list):
        return None
    total_dim = len(vector)
    base_dim = int(entry.get("grid_dim") or grid_dim or total_dim)
    if grid_order is None:
        order = sorted(int(k) for k in grid_coords.keys())
    else:
        order = [int(k) for k in grid_order]
    coords: List[float] = []
    for g_idx in order:
        segment = grid_coords.get(g_idx)
        for offset in range(base_dim):
            if segment and offset < len(segment):
                coords.append(float(segment[offset]))
            else:
                coords.append(float("nan"))
    if all(math.isnan(c) for c in coords):
        return None
    return coords


def _format_cell_key(cell: Optional[Tuple[int, ...]]) -> str:
    if not cell:
        return "?"
    parts = []
    for value in cell:
        try:
            parts.append(str(int(value)))
        except (TypeError, ValueError):
            parts.append(str(value))
    return "(" + ",".join(parts) + ")"


def _vector_metrics(vec_a: Sequence[float], vec_b: Sequence[float]) -> Tuple[float, float, float]:
    length = max(len(vec_a), len(vec_b))
    if length == 0:
        return float("nan"), float("nan"), float("nan")
    a = np.full(length, np.nan, dtype=float)
    b = np.full(length, np.nan, dtype=float)
    a[: len(vec_a)] = np.asarray(vec_a, dtype=float)
    b[: len(vec_b)] = np.asarray(vec_b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    a = a[mask]
    b = b[mask]
    euclidean = float(np.linalg.norm(a - b))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine = float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else float("nan")
    pearson = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    return euclidean, cosine, pearson


def _loader_accepts_refresh(loader) -> bool:
    try:
        sig = inspect.signature(loader)
    except (TypeError, ValueError):
        return False
    return "refresh" in sig.parameters


def _set_loader_refresh(enable: bool) -> None:
    loader = corrtrack_main.DATA_LOADER
    if loader is None:
        return

    target = bool(enable)

    if isinstance(loader, partial):
        keywords = dict(loader.keywords or {})
        args = loader.args or ()
        func = loader.func
        if not _loader_accepts_refresh(func):
            return
        if keywords.get("refresh") == target:
            return
        keywords["refresh"] = target
        corrtrack_main.DATA_LOADER = partial(func, *args, **keywords)
        return

    if not _loader_accepts_refresh(loader):
        return

    original = loader

    def wrapper(*args, **kwargs):
        kwargs["refresh"] = target
        return original(*args, **kwargs)

    corrtrack_main.DATA_LOADER = wrapper


def _resolve_cfg_value(value, cfg, attr, default):
    if value is not None:
        return value
    if hasattr(cfg, attr):
        return getattr(cfg, attr)
    return default

# ---------------------------------------------------------------------------
# Debug monitor that observes CorrTrack internals
# ---------------------------------------------------------------------------


class DebugMonitor:
    def __init__(
        self,
        dataset_id: str,
        alg: str,
        selections: Sequence[PairSelection],
        bf_pairs: Dict[PairKey, float],
        main_pairs: Dict[PairKey, float],
        tn_budget: int,
    ) -> None:
        self.dataset_id = dataset_id
        self.alg = alg
        self.bf_pairs = bf_pairs
        self.main_pairs = main_pairs
        self.tn_budget = tn_budget
        self.pairs: Dict[PairKey, PairDebugState] = {}
        self.series_to_pairs: Dict[SeriesKey, List[PairKey]] = {}
        for selection in selections:
            self._register_pair(selection)

    def _ensure_bucket_entry(self, state: PairDebugState, grid_idx: int) -> Dict[str, object]:
        stats = state.bucket_stats.setdefault(grid_idx, {})
        stats.setdefault("series", {})
        stats.setdefault("collision_cells", {})
        stats.setdefault("votes", 0.0)
        stats.setdefault("vote_history", [])
        stats.setdefault("collisions", 0)
        stats.setdefault("coords", {})
        return stats

    def _register_pair(self, selection: PairSelection) -> None:
        key = selection.canonical_key()
        if key in self.pairs:
            return
        state = PairDebugState(selection=selection)
        if selection.freq is not None:
            state.freq_history.append(float(selection.freq))
        self.pairs[key] = state
        for sk in selection.series_keys:
            self.series_to_pairs.setdefault(sk, []).append(key)

    def _extract_series_window(
        self,
        corrtrack: CorrTrack,
        series_id: str,
        start: int,
        window: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if corrtrack.window_data is None or corrtrack.window_index is None:
            return None
        idx = corrtrack.series_ids.get(series_id)
        if idx is None:
            return None
        win_index = np.asarray(corrtrack.window_index)
        idx_matches = np.where(win_index == start)[0]
        if idx_matches.size == 0:
            return None
        start_idx = int(idx_matches[0])
        end_idx = start_idx + window
        if end_idx > corrtrack.window_data.shape[1]:
            return None
        values = corrtrack.window_data[idx, start_idx:end_idx].astype(np.float64, copy=False)
        window_slice = corrtrack.window_data[idx : idx + 1, start_idx:end_idx].astype(np.float64, copy=False)
        times = win_index[start_idx:end_idx]
        return times, values, window_slice

    @staticmethod
    def _times_to_list(times: np.ndarray) -> List[object]:
        if np.issubdtype(times.dtype, np.number):
            return times.astype(float).tolist()
        return [str(item) for item in times.tolist()]

    def observe_raw(self, corrtrack: CorrTrack) -> None:
        for key, state in self.pairs.items():
            if state.raw_windows is not None:
                continue
            sel = state.selection
            first = self._extract_series_window(corrtrack, sel.id1, sel.start1, sel.window)
            second = self._extract_series_window(corrtrack, sel.id2, sel.start2, sel.window)
            if first is None or second is None:
                continue
            times_a, values_a, window_a = first
            times_b, values_b, window_b = second
            corr, dist = self._pair_stats(values_a, values_b)
            norm_a = self._normalize_series(corrtrack, window_a, sel.id1)
            norm_b = self._normalize_series(corrtrack, window_b, sel.id2)
            norm_corr, norm_dist = self._pair_stats(norm_a, norm_b)
            state.raw_windows = {
                "series": [
                    SeriesWindowRecord(sel.id1, self._times_to_list(times_a), values_a.tolist()).__dict__,
                    SeriesWindowRecord(sel.id2, self._times_to_list(times_b), values_b.tolist()).__dict__,
                ],
                "pearson": corr,
                "euclidean": dist,
            }
            state.normalized_windows = {
                "series": [
                    {
                        "series_id": sel.id1,
                        "values": norm_a.tolist(),
                    },
                    {
                        "series_id": sel.id2,
                        "values": norm_b.tolist(),
                    },
                ],
                "pearson": norm_corr,
                "euclidean": norm_dist,
            }
            self._attach_sketch_from_corrtrack(corrtrack, state, sel.id1, sel.start1)
            self._attach_sketch_from_corrtrack(corrtrack, state, sel.id2, sel.start2)

    def _normalize_series(self, corrtrack: CorrTrack, window_matrix: np.ndarray, series_id: str) -> np.ndarray:
        idx = corrtrack.series_ids.get(series_id)
        if idx is None:
            return np.asarray(window_matrix[0], dtype=np.float64)
        normalized = corrtrack._preprocess_data(window_matrix, idx)
        if normalized.ndim == 2 and normalized.shape[0] == 1:
            return normalized[0]
        return np.asarray(normalized, dtype=np.float64)

    def _pair_stats(self, values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float]:
        if values_a.size == 0 or values_b.size == 0:
            return float("nan"), float("nan")
        corr = np.corrcoef(values_a, values_b)[0, 1]
        dist = float(np.linalg.norm(values_a - values_b))
        return float(corr), dist

    def observe_sketches(self, corrtrack: CorrTrack, sketches: Dict[Tuple[str, int, int], np.ndarray]) -> None:
        if not sketches:
            return
        for skey, vector in sketches.items():
            pairs = self.series_to_pairs.get(skey)
            if not pairs:
                continue
            for pkey in pairs:
                self._add_sketch_entry(self.pairs[pkey], skey, vector, corrtrack)

    def observe_buckets(self, corrtrack: CorrTrack) -> None:
        pair_keys = list(self.pairs.keys())
        if not pair_keys:
            return
        # Capture frequency votes for tracked pairs
        for pkey in pair_keys:
            freq = corrtrack.freq_pairs.get(pkey)
            if freq is not None:
                state = self.pairs[pkey]
                state.freq_history.append(float(freq))
        self._maybe_add_tn_pairs(corrtrack)
        grid_params = {
            "cell_size": corrtrack.cell_size,
            "freq_threshold": corrtrack.freq_threshold,
            "neighbor_margin": corrtrack.neighbor_margin,
            "grid_dimension": corrtrack.grid_dimension,
            "n_vectors": corrtrack.n_vectors,
        }
        for grid_idx, node in enumerate(corrtrack.grid_nodes):
            key_cells = getattr(node, "_key_cells", {})
            grid_map = getattr(node, "grid", {})
            if not key_cells or not grid_map:
                continue
            sketches_map = getattr(node, "sketches", {})
            for state in self.pairs.values():
                skey_a, skey_b = state.selection.series_keys
                series_hits: Dict[SeriesKey, List[Tuple[Tuple[int, ...], int, Optional[Tuple[float, ...]]]]] = {}
                for skey in (skey_a, skey_b):
                    cells = key_cells.get(skey)
                    if not cells:
                        continue
                    hits: List[Tuple[Tuple[int, ...], int, Optional[Tuple[float, ...]]]] = []
                    for cell in cells:
                        cell_key = tuple(cell)
                        occupants = grid_map.get(cell_key)
                        if not occupants:
                            continue
                        if skey in occupants:
                            density = len(occupants)
                            coords_value: Optional[Tuple[float, ...]] = None
                            vector = sketches_map.get(skey)
                            if vector is not None:
                                try:
                                    coords_no_flip = node._normalize_cell_coords(vector, flip=False)
                                except Exception:
                                    coords_no_flip = np.zeros(node.grid_dimensions, dtype=float)
                                base_cell = node._get_cell(v=vector, flip=False, coords=coords_no_flip)
                                coords_value = tuple(float(x) for x in np.asarray(coords_no_flip, dtype=float))
                                if base_cell != cell_key and getattr(node, "neg_corr", False):
                                    try:
                                        coords_flip = node._normalize_cell_coords(vector, flip=True)
                                        base_flip = node._get_cell(v=vector, flip=True, coords=coords_flip)
                                        if base_flip == cell_key:
                                            coords_value = tuple(float(x) for x in np.asarray(coords_flip, dtype=float))
                                    except Exception:
                                        pass
                            hits.append((cell_key, density, coords_value))
                    if hits:
                        series_hits[skey] = hits
                if not series_hits:
                    continue
                stats = self._ensure_bucket_entry(state, grid_idx)
                series_map = stats["series"]
                coords_store = stats.setdefault("coords", {})
                for skey, cells in series_hits.items():
                    sid = skey[0]
                    cell_counts = series_map.setdefault(sid, {})
                    for cell_key, density, coords in cells:
                        prev = cell_counts.get(cell_key, 0)
                        cell_counts[cell_key] = max(prev, int(density))
                        if coords is not None:
                            coords_store.setdefault(sid, tuple(coords))
                if len(series_hits) == 2:
                    series_keys = list(series_hits.keys())
                    sid_a = series_keys[0][0]
                    sid_b = series_keys[1][0]
                    counts_a = series_map.get(sid_a, {})
                    counts_b = series_map.get(sid_b, {})
                    common = set(counts_a.keys()).intersection(counts_b.keys())
                    if common:
                        collision_cells = stats["collision_cells"]
                        for cell in common:
                            overlap = min(counts_a.get(cell, 0), counts_b.get(cell, 0))
                            if overlap <= 0:
                                continue
                            prev = collision_cells.get(cell, 0)
                            collision_cells[cell] = max(prev, int(overlap))
        for state in self.pairs.values():
            history: Dict[str, Dict[str, object]] = {}
            for grid_idx, stats in sorted(state.bucket_stats.items()):
                series_payload = {
                    sid: { _format_cell_key(cell): int(count) for cell, count in cells.items() }
                    for sid, cells in stats.get("series", {}).items()
                }
                coll_payload = {
                    _format_cell_key(cell): int(count)
                    for cell, count in stats.get("collision_cells", {}).items()
                }
                coords_payload = {
                    sid: [float(x) for x in coords]
                    for sid, coords in (stats.get("coords") or {}).items()
                }
                entry = {
                    "series": series_payload,
                    "collision_cells": coll_payload,
                    "collisions": int(stats.get("collisions", len(stats.get("vote_history", [])))),
                    "votes": float(stats.get("votes", 0.0)),
                    "coords": coords_payload,
                }
                history[str(grid_idx)] = entry
            if history:
                state.buckets = {"grid_params": grid_params, "history": history}
            else:
                state.buckets = {"grid_params": grid_params}

    def record_grid_votes(self, grid_idx: int, freq_pairs: Dict[PairKey, float]) -> None:
        if not freq_pairs:
            return
        for pair_key, vote in freq_pairs.items():
            state = self.pairs.get(pair_key)
            if state is None:
                continue
            stats = self._ensure_bucket_entry(state, grid_idx)
            stats["votes"] = float(stats.get("votes", 0.0)) + float(vote)
            history = stats.setdefault("vote_history", [])
            history.append(float(vote))
            stats["collisions"] = len(history)

    def _maybe_add_tn_pairs(self, corrtrack: CorrTrack) -> None:
        if self.tn_budget <= 0:
            return
        sorted_pairs = sorted(
            corrtrack.freq_pairs.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        for pair, freq in sorted_pairs:
            if pair in self.pairs or pair in self.bf_pairs or pair in self.main_pairs:
                continue
            sel = PairSelection(
                dataset_id=self.dataset_id,
                alg=self.alg,
                label="TN",
                id1=str(pair[0]),
                id2=str(pair[1]),
                start1=int(pair[2]),
                start2=int(pair[3]),
                window=int(pair[4]),
                bf_corr=self.bf_pairs.get(pair),
                main_corr=self.main_pairs.get(pair),
                freq=float(freq),
                priority=len(self.pairs),
            )
            self._register_pair(sel)
            self.tn_budget -= 1
            self.observe_raw(corrtrack)
            self.observe_sketches(corrtrack, {})
            if self.tn_budget <= 0:
                break

    def _attach_sketch_from_corrtrack(self, corrtrack: CorrTrack, state: PairDebugState, series_id: str, start: int) -> None:
        key = (series_id, int(start), state.selection.window)
        vector = corrtrack.sketches.get(key)
        if vector is None:
            return
        self._add_sketch_entry(state, key, vector, corrtrack)

    def _add_sketch_entry(
        self,
        state: PairDebugState,
        skey: Tuple[str, int, int],
        vector: np.ndarray,
        corrtrack: CorrTrack,
    ) -> None:
        _debug_monitor_add_sketch_entry(self, state, skey, vector, corrtrack)

    def observe_validation(self, corrtrack: CorrTrack) -> None:
        for key, state in self.pairs.items():
            if state.corrtrack_detection is not None:
                continue
            if key in corrtrack.correlated:
                state.corrtrack_detection = True
                state.corrtrack_corr = float(corrtrack.correlated[key])
            elif key in corrtrack.candidates:
                state.corrtrack_detection = False
                state.corrtrack_corr = None

    def results(self) -> List[PairDebugState]:
        return list(self.pairs.values())

    def observe_validation(self, corrtrack: CorrTrack) -> None:
        for key, state in self.pairs.items():
            if state.corrtrack_detection is not None:
                continue
            if key in corrtrack.correlated:
                state.corrtrack_detection = True
                state.corrtrack_corr = float(corrtrack.correlated[key])
            elif key in corrtrack.candidates:
                state.corrtrack_detection = False
                state.corrtrack_corr = None

    def results(self) -> List[PairDebugState]:
        return list(self.pairs.values())


def _debug_monitor_observe_validation(self: DebugMonitor, corrtrack: CorrTrack) -> None:
    for key, state in self.pairs.items():
        if state.corrtrack_detection is not None:
            continue
        if key in corrtrack.correlated:
            state.corrtrack_detection = True
            state.corrtrack_corr = float(corrtrack.correlated[key])
        elif key in corrtrack.candidates:
            state.corrtrack_detection = False
            state.corrtrack_corr = None


def _debug_monitor_results(self: DebugMonitor) -> List[PairDebugState]:
    return list(self.pairs.values())


DebugMonitor.observe_validation = _debug_monitor_observe_validation
DebugMonitor.results = _debug_monitor_results


def _debug_monitor_add_sketch_entry(
    monitor: DebugMonitor,
    state: PairDebugState,
    skey: Tuple[str, int, int],
    vector: np.ndarray,
    corrtrack: CorrTrack,
) -> None:
    payload = state.sketches or {
        "series": [],
        "euclidean": None,
        "cosine": None,
        "pearson": None,
        "params": {
            "n_vectors": corrtrack.n_vectors,
            "grid_dimension": corrtrack.grid_dimension,
            "cell_size": corrtrack.cell_size,
            "cell_stretch": corrtrack.cell_stretch,
            "neg_corr": corrtrack.neg_corr,
        },
    }
    payload_series = payload["series"]
    if any(entry["series_id"] == skey[0] for entry in payload_series):
        state.sketches = payload
        return
    raw_map = getattr(corrtrack, "raw_sketches", {}) if corrtrack is not None else {}
    raw_vector = raw_map.get(skey, vector)
    raw_arr = np.asarray(raw_vector, dtype=float)
    arr_vector = np.asarray(vector, dtype=float)
    entry_grid_dim = int(corrtrack.grid_dimension or arr_vector.size)
    entry_payload: Dict[str, object] = {
        "series_id": skey[0],
        "start": skey[1],
        "vector": raw_arr.tolist(),
        "norm_vector": arr_vector.tolist(),
        "norm": float(np.linalg.norm(arr_vector)),
        "grid_coords": {},
        "grid_dim": entry_grid_dim,
    }
    gdim = max(1, entry_grid_dim)
    n_grids = max(1, int(math.ceil(arr_vector.size / gdim)))
    for g_idx in range(n_grids):
        start = g_idx * gdim
        segment = arr_vector[start : start + gdim]
        if segment.size < gdim:
            break
        entry_payload["grid_coords"][g_idx] = _normalize_segment(segment, corrtrack.neg_corr)
    payload_series.append(entry_payload)
    if len(payload_series) == 2:
        vec_a = np.asarray(payload_series[0].get("norm_vector") or payload_series[0]["vector"], dtype=float)
        vec_b = np.asarray(payload_series[1].get("norm_vector") or payload_series[1]["vector"], dtype=float)
        payload["euclidean"] = float(np.linalg.norm(vec_a - vec_b))
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 0 and norm_b > 0:
            payload["cosine"] = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            payload["pearson"] = float(np.corrcoef(vec_a, vec_b)[0, 1])
    state.sketches = payload


DebugMonitor._add_sketch_entry = _debug_monitor_add_sketch_entry


class DebugCorrTrack(CorrTrack):
    """CorrTrack variant that forwards internal state updates to DebugMonitor."""

    def __init__(self, *args, debugger: Optional[DebugMonitor] = None, **kwargs):
        self.debugger = debugger
        super().__init__(*args, **kwargs)
        if self.debugger:
            self._wrap_grid_nodes()

    def _wrap_grid_nodes(self) -> None:
        for idx, node in enumerate(self.grid_nodes):
            if getattr(node, "_debug_run_wrapped", False):
                continue
            original_run = node.run
            debugger = self.debugger

            def run_with_debug(self_node, *args, _orig=original_run, _idx=idx, _dbg=debugger, **kwargs):
                freq_pairs, candidates, uncorrelated = _orig(*args, **kwargs)
                if _dbg is not None:
                    _dbg.record_grid_votes(_idx, freq_pairs)
                return freq_pairs, candidates, uncorrelated

            node.run = MethodType(run_with_debug, node)
            node._debug_run_wrapped = True

    def _update_curr_data(self, new_data_step, ids):
        super()._update_curr_data(new_data_step, ids)
        if self.debugger:
            self.debugger.observe_raw(self)

    def _get_sketches(self, new_data_step, verbose, testing, worker_mode=None):
        self.raw_sketches = {}
        # Ensure sketch nodes created are debug-enabled
        for node in getattr(self, "sketch_nodes", []):
            if not hasattr(node, "_debug_raw_sketches"):
                node._debug_raw_sketches = {}
        sketches = super()._get_sketches(new_data_step, verbose, testing, worker_mode=worker_mode)
        if getattr(self, "sketch_nodes", None):
            combined_raw: Dict[Tuple[str, int, int], np.ndarray] = {}
            for node in self.sketch_nodes:
                raw_map = getattr(node, "_debug_raw_sketches", {}) or {}
                combined_raw.update(raw_map)
            self.raw_sketches = combined_raw
        if self.debugger:
            self.debugger.observe_sketches(self, sketches)
        return sketches

    def _run_grids(self, verbose, testing, worker_mode=None):
        super()._run_grids(verbose, testing, worker_mode=worker_mode)
        if self.debugger:
            self.debugger.observe_buckets(self)

    def _get_validated_corr(self, corr_val=True, force_mode=None, retain_validated=True):
        super()._get_validated_corr(
            corr_val=corr_val,
            force_mode=force_mode,
            retain_validated=retain_validated,
        )
        if self.debugger:
            self.debugger.observe_validation(self)


# ---------------------------------------------------------------------------
# Dataset context discovery and experiment invocation
# ---------------------------------------------------------------------------


@dataclass
class DatasetContext:
    dataset_id: str
    base_dir: Path
    test_data: np.ndarray
    ids: Sequence[str]


def discover_datasets(args: argparse.Namespace) -> List[DatasetContext]:
    cfg_exec = corrtrack_main._load_module(args.exec_param_config, "experiment_exec")
    cfg_dataset = corrtrack_main._load_dataset_config(args.dataset_config)
    corrtrack_main._apply_dataset_config(cfg_dataset)
    if hasattr(corrtrack_main, "_apply_parallel_defaults_from_cfg"):
        corrtrack_main._apply_parallel_defaults_from_cfg(cfg_exec, args)
    if args.loader:
        corrtrack_main.DATA_LOADER = corrtrack_main._load_loader(args.loader)
    _set_loader_refresh(getattr(args, "refresh_artifacts", False))

    corrtrack_main.RESULT_FOLDER = _resolve_cfg_value(
        args.result_folder, cfg_dataset, "RESULT_FOLDER", corrtrack_main.DEFAULT_RESULT_FOLDER
    )
    corrtrack_main.WINDOW_SIZE = _resolve_cfg_value(
        args.window_size, cfg_exec, "WINDOW_SIZE", corrtrack_main.DEFAULT_WINDOW_SIZE
    )
    corrtrack_main.WINDOW_STEP = _resolve_cfg_value(
        args.window_step, cfg_exec, "WINDOW_STEP", corrtrack_main.DEFAULT_WINDOW_STEP
    )
    corrtrack_main.BASIC_WINDOW = _resolve_cfg_value(
        args.basic_window, cfg_exec, "BASIC_WINDOW", corrtrack_main.DEFAULT_BASIC_WINDOW
    )
    corrtrack_main.N_LAGS = _resolve_cfg_value(
        args.n_lags, cfg_exec, "N_LAGS", corrtrack_main.DEFAULT_N_LAGS
    )
    corrtrack_main.CORR_THRESHOLD = _resolve_cfg_value(
        args.corr_threshold, cfg_exec, "CORR_THRESHOLD", corrtrack_main.DEFAULT_CORR_THRESHOLD
    )
    effective_parallel = bool(args.parallel) or any(
        flag is True
        for flag in (args.parallel_sketch, args.parallel_candidates, args.parallel_validation)
    )
    corrtrack_main.PARALLEL = effective_parallel
    corrtrack_main.PARALLEL_SKETCH = args.parallel_sketch
    corrtrack_main.PARALLEL_CANDIDATES = args.parallel_candidates
    corrtrack_main.PARALLEL_VALIDATION = args.parallel_validation
    parallel_any = bool(args.parallel) or any(
        flag is True
        for flag in (args.parallel_sketch, args.parallel_candidates, args.parallel_validation)
    )
    corrtrack_main.EXEC_MODE = "thread" if parallel_any else "sequential"
    corrtrack_main.NEG_CORR = _resolve_cfg_value(
        args.neg_corr, cfg_exec, "NEG_CORR", corrtrack_main.DEFAULT_NEG_CORR
    )
    corrtrack_main.RECALL_BY_WINDOW = _resolve_cfg_value(
        args.recall_by_window, cfg_exec, "RECALL_BY_WINDOW", corrtrack_main.DEFAULT_RECALL_BY_WINDOW
    )
    corrtrack_main.ARTIFACT_MODE = _resolve_cfg_value(
        args.artifact_mode, cfg_exec, "ARTIFACT_MODE", corrtrack_main.DEFAULT_ARTIFACT_MODE
    )
    corrtrack_main.VERBOSE = _resolve_cfg_value(
        args.verbose, cfg_exec, "VERBOSE", corrtrack_main.DEFAULT_VERBOSE
    )
    corrtrack_main.TESTING = _resolve_cfg_value(
        args.testing, cfg_exec, "TESTING", corrtrack_main.DEFAULT_TESTING
    )
    corrtrack_main.MAX_WORKERS = _resolve_cfg_value(
        None, cfg_exec, "MAX_WORKERS", corrtrack_main.DEFAULT_MAX_WORKERS
    )
    if corrtrack_main.RESULT_FOLDER is None:
        raise RuntimeError("Dataset config must define RESULT_FOLDER or provide --result-folder.")

    contexts: List[DatasetContext] = []
    config_folder = corrtrack_main.config_folder()
    for country, var, data, ids in corrtrack_main.iter_datasets():
        for n_year in corrtrack_main.N_YEARS:
            for n_var in corrtrack_main.N_VARS:
                slug = corrtrack_main._dataset_slug(country, var)
                dataset_id = f"{slug}_{n_var}_{n_year}"
                test_data, ids_n_var = corrtrack_main.prepare_test_data(data, ids, n_year, n_var)
                base_dir = Path("correlation") / corrtrack_main.RESULT_FOLDER / dataset_id / config_folder
                contexts.append(
                    DatasetContext(
                        dataset_id=dataset_id,
                        base_dir=base_dir,
                        test_data=test_data,
                        ids=ids_n_var,
                    )
                )
    return contexts


def run_full_pipeline(args: argparse.Namespace, passthrough: List[str]) -> None:
    if args.skip_initial_run or not args.refresh_artifacts:
        return
    script = Path(__file__).with_name("run_corrtrack_experiment.py")
    cmd = [
        sys.executable,
        str(script),
        "--dataset-config",
        str(args.dataset_config),
        "--param-grid-config",
        str(args.param_grid_config),
    ]
    if args.exec_param_config:
        cmd.extend(["--exec-param-config", str(args.exec_param_config)])
    if args.result_folder:
        cmd.extend(["--result-folder", args.result_folder])
    # Propagate core overrides so the reproducible rerun matches.
    effective_parallel = bool(args.parallel) or any(
        flag is True
        for flag in (args.parallel_sketch, args.parallel_candidates, args.parallel_validation)
    )

    if args.window_size is not None:
        cmd.extend(["--window-size", str(args.window_size)])
    if args.window_step is not None:
        cmd.extend(["--window-step", str(args.window_step)])
    if args.n_lags is not None:
        cmd.extend(["--n-lags", str(args.n_lags)])
    if args.corr_threshold is not None:
        cmd.extend(["--corr-threshold", str(args.corr_threshold)])
    cmd.append("--parallel" if effective_parallel else "--sequential")
    if args.parallel_sketch is not None:
        cmd.append("--parallel-sketch" if args.parallel_sketch else "--sequential-sketch")
    if args.parallel_candidates is not None:
        cmd.append("--parallel-candidates" if args.parallel_candidates else "--sequential-candidates")
    if args.parallel_validation is not None:
        cmd.append("--parallel-validation" if args.parallel_validation else "--sequential-validation")
    if args.neg_corr is True:
        cmd.append("--neg-corr")
    elif args.neg_corr is False:
        cmd.append("--no-neg-corr")
    if args.recall_by_window:
        cmd.append("--recall-by-window")
    if args.artifact_mode:
        cmd.extend(["--artifact-mode", args.artifact_mode])
    if args.verbose is True:
        cmd.append("--verbose")
    elif args.verbose is False:
        cmd.append("--no-verbose")
    if args.testing is True:
        cmd.append("--testing")
    elif args.testing is False:
        cmd.append("--no-testing")
    if args.loader:
        cmd.extend(["--loader", args.loader])
    cmd.extend(passthrough)

    print("[DEBUG] Running baseline experiment:\n  ", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Pair sampling (online via brute force)
# ---------------------------------------------------------------------------


def _run_bruteforce_pairs(
    ctx: DatasetContext,
    base_config: dict,
) -> Tuple[Dict[PairKey, float], CorrTrack]:
    corrtrack = CorrTrack(
        window_size=base_config["window_size"],
        basic_window=base_config.get("basic_window"),
        window_step=base_config["window_step"],
        n_vectors=1,
        n_lags=base_config["n_lags"],
        grid_dimension=1,
        cell_size=1,
        seed=None,
        seed_toggle=None,
        freq_threshold=0.0,
        corr_threshold=base_config["corr_threshold"],
        neg_corr=base_config.get("neg_corr", False),
        preprocess=False,
        exec=base_config.get("exec", "thread"),
        max_workers=base_config.get("max_workers", 0),
        parallel_sketch=base_config.get("parallel_sketch"),
        parallel_candidates=base_config.get("parallel_candidates"),
        parallel_validation=base_config.get("parallel_validation"),
    )
    metadata = {"alg": "bf", "mode": "bf", "optim": "debug"}
    execute_corrtrack_pass(
        "bf",
        ctx.dataset_id,
        corrtrack,
        ctx.test_data,
        ctx.ids,
        metadata,
        corr_val=True,
        recall_by_window=True,
        artifact_prefix=None,
        artifact_mode="final",
    )
    return dict(corrtrack.correlated), corrtrack


def _run_plain_corrtrack_pairs(
    ctx: DatasetContext,
    base_config: dict,
    alg: str,
    params: dict,
) -> Tuple[Dict[PairKey, float], Dict[PairKey, float], Set[PairKey]]:
    corrtrack = build_corrtrack_instance(ctx, base_config, params, debugger=None)
    metadata = {"alg": alg, "mode": "main", "optim": "best"}
    execute_corrtrack_pass(
        "corrtrack",
        ctx.dataset_id,
        corrtrack,
        ctx.test_data,
        ctx.ids,
        metadata,
        corr_val=True,
        recall_by_window=True,
        artifact_prefix=None,
        artifact_mode="final",
    )
    return (
        dict(corrtrack.correlated),
        dict(corrtrack.freq_pairs),
        set(corrtrack.candidates.keys()),
    )


def build_selection_bundle(
    ctx: DatasetContext,
    alg: str,
    params: dict,
    base_config: dict,
    samples_per_class: int,
) -> SelectionBundle:
    bf_pairs, bf_runner = _run_bruteforce_pairs(ctx, base_config)
    main_pairs, freq_pairs, candidate_pairs = _run_plain_corrtrack_pairs(ctx, base_config, alg, params)

    # CorrTrack clears validated candidates from self.candidates, so determine
    # FN/TP membership using the correlated outputs instead of the candidate map.
    fn_candidates = [k for k in bf_pairs.keys() if k not in main_pairs]
    tp_candidates = [k for k in bf_pairs.keys() if k in main_pairs]

    oversample = max(2, samples_per_class * 4)
    fn_keys = fn_candidates[: oversample * 2]

    tn_keys: List[PairKey] = []
    tn_freqs: Dict[PairKey, float] = {}
    fp_keys: List[PairKey] = []
    fp_freqs: Dict[PairKey, float] = {}
    for key, freq in sorted(freq_pairs.items(), key=lambda item: float(item[1]), reverse=True):
        if key in bf_pairs:
            continue
        if key in candidate_pairs:
            promoted_to_tp = False
            if bf_runner is not None:
                try:
                    _pair, is_corr, corr_val, _ = bf_runner._validate_corr(key, corr_val=True)
                except Exception:
                    is_corr = False
                    corr_val = None
                if is_corr:
                    promoted_to_tp = True
                    if corr_val is not None:
                        bf_pairs[key] = float(corr_val)
                    if key not in tp_candidates:
                        tp_candidates.append(key)
                # fall through to FP only if not promoted
            if promoted_to_tp:
                continue
            if len(fp_keys) < oversample:
                fp_keys.append(key)
                fp_freqs[key] = float(freq)
        else:
            if len(tn_keys) < oversample:
                tn_keys.append(key)
                tn_freqs[key] = float(freq)
        if len(fp_keys) >= oversample and len(tn_keys) >= oversample:
            break

    tp_keys = tp_candidates[: oversample * 2]

    selections: List[PairSelection] = []
    order_map: Dict[PairKey, int] = {}
    priority = 0
    for label, keys in (("FN", fn_keys), ("TP", tp_keys), ("TN", tn_keys), ("FP", fp_keys)):
        for key in keys:
            id1, id2, t1, t2, window = key
            if label == "TN":
                freq_map = tn_freqs
            elif label == "FP":
                freq_map = fp_freqs
            else:
                freq_map = {}
            sel = PairSelection(
                dataset_id=ctx.dataset_id,
                alg=alg,
                label=label,
                id1=id1,
                id2=id2,
                start1=t1,
                start2=t2,
                window=window,
                bf_corr=bf_pairs.get(key),
                main_corr=main_pairs.get(key),
                freq=freq_map.get(key),
                priority=priority,
            )
            selections.append(sel)
            order_map[sel.canonical_key()] = priority
            priority += 1

    tn_budget = max(0, samples_per_class - len([sel for sel in selections if sel.label == "TN"]))
    return SelectionBundle(
        selections=selections,
        bf_pairs=bf_pairs,
        main_pairs=main_pairs,
        tn_budget=tn_budget,
        samples_per_class=samples_per_class,
        order_map=order_map,
        candidate_pairs=candidate_pairs,
    )


# ---------------------------------------------------------------------------
# Instrumented rerun execution
# ---------------------------------------------------------------------------


def build_corrtrack_instance(
    ctx: DatasetContext,
    base_config: dict,
    params: dict,
    debugger: Optional[DebugMonitor] = None,
) -> DebugCorrTrack:
    def _to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    n_vectors = _to_int(params.get("n_vectors"))
    grid_dimension = _to_int(params.get("grid_dimension"))
    cell_stretch = _resolve_cell_stretch(
        params.get("cell_stretch"),
        params.get("cell_size"),
        base_config["corr_threshold"],
        n_vectors,
        grid_dimension,
    )
    if cell_stretch is None or cell_stretch <= 0.0:
        cell_stretch = 1.0

    feature_kwargs = _extract_feature_overrides(params)

    return DebugCorrTrack(
        window_size=base_config["window_size"],
        basic_window=base_config.get("basic_window"),
        window_step=base_config["window_step"],
        n_vectors=n_vectors,
        n_lags=base_config["n_lags"],
        grid_dimension=grid_dimension,
        cell_size=cell_stretch,
        seed=_to_int(params.get("seed")),
        seed_toggle=_to_int(params.get("seed_toggle")),
        freq_threshold=_to_float(params.get("freq_threshold")),
        corr_threshold=base_config["corr_threshold"],
        neg_corr=base_config.get("neg_corr", False),
        preprocess=params.get("preprocess"),
        exec=base_config.get("exec", "thread"),
        max_workers=base_config.get("max_workers", 0),
        parallel_sketch=base_config.get("parallel_sketch"),
        parallel_candidates=base_config.get("parallel_candidates"),
        parallel_validation=base_config.get("parallel_validation"),
        debugger=debugger,
        **feature_kwargs,
    )


def run_debug_pass(
    ctx: DatasetContext,
    alg: str,
    bundle: SelectionBundle,
    params: dict,
    args: argparse.Namespace,
) -> List[PairDebugState]:
    debugger = DebugMonitor(
        dataset_id=ctx.dataset_id,
        alg=alg,
        selections=bundle.selections,
        bf_pairs=bundle.bf_pairs,
        main_pairs=bundle.main_pairs,
        tn_budget=bundle.tn_budget,
    )
    base_config = corrtrack_main.build_base_config()
    corrtrack = build_corrtrack_instance(ctx, base_config, params, debugger)

    metadata = {"alg": alg, "mode": "debug", "optim": "best"}
    execute_corrtrack_pass(
        run_kind="corrtrack",
        dataset_id=ctx.dataset_id,
        corrtrack=corrtrack,
        data=ctx.test_data,
        ids=ctx.ids,
        metadata=metadata,
        corr_val=True,
        recall_by_window=True,
        artifact_prefix=None,
        artifact_mode=args.artifact_mode or "iterative",
    )

    rerun_candidates = set(corrtrack.candidates.keys())
    rerun_correlated = set(corrtrack.correlated.keys())

    states = debugger.results()
    filtered = [
        state
        for state in states
        if state.raw_windows is not None
        and state.sketches
        and len(state.sketches.get("series", [])) >= 2
    ]
    final_states: List[PairDebugState] = []
    for state in filtered:
        key = state.selection.canonical_key()
        label = state.selection.label
        if label == "TP":
            matches = key in rerun_correlated
        elif label == "FN":
            matches = key not in rerun_correlated
        elif label == "FP":
            matches = key in rerun_candidates
        elif label == "TN":
            matches = key not in rerun_candidates
        else:
            matches = True
        if matches:
            final_states.append(state)
        else:
            print(
                f"[DEBUG] Dropping {label} pair {key} (rerun mismatch)"
            )
    if len(filtered) < len(states):
        missing = len(states) - len(filtered)
        print(
            f"[DEBUG] Skipping {missing} pairs lacking captured sketches for {ctx.dataset_id}/{alg}"
        )
    filtered = final_states

    order_map = getattr(bundle, "order_map", {})
    filtered.sort(
        key=lambda s: order_map.get(s.selection.canonical_key(), s.selection.priority)
    )

    target = getattr(bundle, "samples_per_class", 0) or 0
    label_limits = {"FN": target, "TP": target, "TN": target, "FP": target}
    collected: Dict[str, List[PairDebugState]] = {label: [] for label in label_limits}
    for state in filtered:
        label = state.selection.label
        if label not in collected:
            continue
        if len(collected[label]) < label_limits[label]:
            collected[label].append(state)
        if all(len(collected[lbl]) >= limit for lbl, limit in label_limits.items()):
            break

    for label, limit in label_limits.items():
        if limit > 0 and len(collected[label]) < limit:
            print(
                f"[WARN] Only {len(collected[label])} {label} samples available (requested {limit}) for {ctx.dataset_id}/{alg}"
            )

    ordered_labels = ["FN", "TP", "TN", "FP"]
    selected_states: List[PairDebugState] = []
    for label in ordered_labels:
        selected_states.extend(collected.get(label, []))
    return selected_states


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def plot_pair(state: PairDebugState, output_dir: Path) -> None:
    if state.raw_windows is None or state.normalized_windows is None:
        return
    fig, axes = plt.subplots(5, 1, figsize=(11, 16), gridspec_kw={"height_ratios": [1.6, 1.6, 2.2, 2.2, 2.2]})
    ax_raw, ax_norm, ax_sketch, ax_coords, ax_bucket = axes
    raw_series = state.raw_windows["series"]
    norm_series = state.normalized_windows["series"]

    freq_summary = ", ".join(_fmt_float(v) for v in state.freq_history) or "—"
    gp = state.buckets.get("grid_params", {}) if state.buckets else {}
    sketch_payload = state.sketches or {}
    sketch_series = sketch_payload.get("series", [])

    def _coerce_cell_size(value) -> Optional[float]:
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0] if len(value) > 0 else None
        try:
            cell_f = float(value)
            return cell_f if cell_f > 0 else None
        except (TypeError, ValueError):
            return None

    cell_size = _coerce_cell_size(gp.get("cell_size"))
    if cell_size is None:
        cell_size = _coerce_cell_size((sketch_payload.get("params") or {}).get("cell_size"))

    for entry in raw_series:
        ax_raw.plot(entry["times"], entry["values"], label=entry["series_id"])
    ax_raw.set_title(
        f"Raw windows (Pearson={_fmt_float(state.raw_windows['pearson'])}, "
        f"Euclidean={_fmt_float(state.raw_windows['euclidean'])}) | Freq votes={freq_summary}",
        fontsize=11,
    )
    ax_raw.set_xlabel("Time index")
    ax_raw.legend(loc="upper right")

    for entry in norm_series:
        ax_norm.plot(range(len(entry.get("values", []))), entry.get("values", []), label=entry["series_id"])
    ax_norm.set_title(
        f"Normalized windows (Pearson={_fmt_float(state.normalized_windows['pearson'])}, "
        f"Euclidean={_fmt_float(state.normalized_windows['euclidean'])})"
    )
    ax_norm.set_xlabel("Sample")
    ax_norm.legend(loc="upper right")

    raw_vectors: List[np.ndarray] = []
    if len(sketch_series) >= 2:
        vec_a = np.asarray(sketch_series[0]["vector"], dtype=float)
        vec_b = np.asarray(sketch_series[1]["vector"], dtype=float)
        raw_vectors = [vec_a, vec_b]
        ax_sketch.plot(vec_a, label=sketch_series[0]["series_id"])
        ax_sketch.plot(vec_b, label=sketch_series[1]["series_id"])
        ax_sketch.set_xlabel("Sketch dimension")
        ax_sketch.legend(loc="upper right")
        ax_sketch.set_title(
            f"Raw sketch vectors (Euclidean={_fmt_float(sketch_payload.get('euclidean'))}, "
            f"Cosine={_fmt_float(sketch_payload.get('cosine'))}, "
            f"Pearson={_fmt_float(sketch_payload.get('pearson'))})",
            fontsize=11,
        )
    else:
        ax_sketch.set_title("Sketch vectors unavailable", fontsize=11)
        ax_sketch.text(0.5, 0.5, "Sketch vectors unavailable", transform=ax_sketch.transAxes, ha="center")
        ax_sketch.set_xticks([])
        ax_sketch.set_yticks([])

    grid_dim = (sketch_payload.get("params") or {}).get("grid_dimension")

    norm_vectors: List[np.ndarray] = []
    if sketch_series:
        for entry in sketch_series[:2]:
            vec = entry.get("norm_vector") or entry.get("vector")
            if vec is None:
                continue
            norm_vectors.append(np.clip(np.asarray(vec, dtype=float), -1.0, 1.0))
    if norm_vectors:
        for idx, vec in enumerate(norm_vectors):
            series_id = sketch_series[idx]["series_id"]
            xs = np.arange(len(vec))
            yerr = None if cell_size is None else np.full_like(vec, cell_size, dtype=float)
            ax_coords.errorbar(xs, vec, yerr=yerr, fmt="-o", capsize=4, label=f"{series_id} (norm)")
        coord_title = "Normalized sketch vectors"
        if cell_size:
            coord_title += f" (±{_fmt_float(cell_size)} threshold)"
        if len(norm_vectors) == 2:
            eu, co, pe = _vector_metrics(norm_vectors[0], norm_vectors[1])
            coord_title += f" | Euclid={_fmt_float(eu)} Cosine={_fmt_float(co)} Pearson={_fmt_float(pe)}"
        ax_coords.set_ylabel("Normalized value")
        ax_coords.set_xlabel("Sketch dimension")
        ax_coords.set_title(coord_title, fontsize=11)
        ax_coords.legend(loc="upper right", fontsize=8)
    else:
        ax_coords.set_title("Normalized sketch vectors unavailable", fontsize=11)
        ax_coords.text(0.5, 0.5, "No coords captured", transform=ax_coords.transAxes, ha="center")
        ax_coords.set_xticks([])
        ax_coords.set_yticks([])

    threshold = cell_size if cell_size is not None else 0.0
    freq_thr = float(gp.get("freq_threshold") or 0.0)
    if len(norm_vectors) >= 2:
        a = norm_vectors[0]
        b = norm_vectors[1]
        length = max(len(a), len(b))
        a_padded = np.full(length, np.nan, dtype=float)
        b_padded = np.full(length, np.nan, dtype=float)
        a_padded[: len(a)] = a
        b_padded[: len(b)] = b
        diffs = np.abs(a_padded - b_padded)
        hits = (~np.isnan(diffs)) & (diffs <= threshold)
        bar_heights = [2 if hit else 0 for hit in hits]
        colors = ["green" if hit else "#1f77b4" for hit in hits]
        ax_bucket.bar(range(length), bar_heights, color=colors, width=0.8)
        ax_bucket.set_xticks(range(length))
        ax_bucket.set_xticklabels([f"Dim {idx}" for idx in range(length)], rotation=0)
        ax_bucket.set_ylabel("Windows within threshold (max=2)")
        ax_bucket.set_ylim(0, 2.2)
        hit_count = int(np.sum(hits))
        title = (
            f"Distance threshold hits per dimension (grid_dim={_fmt_float(grid_dim)}, "
            f"cell_size={_fmt_float(cell_size)}) | hits={hit_count}/{length}"
        )
        ax_bucket.set_title(title, fontsize=11)
        hit_counts = hits.astype(int)
        ax_hits = ax_bucket.twinx()
        cum_hits = np.cumsum(hit_counts)
        if cum_hits.size:
            ax_hits.plot(range(length), cum_hits, color="#444444", marker="o", linestyle="-", label="Cumulative hits")
        upper_hits = max(freq_thr, float(np.max(cum_hits)) if cum_hits.size else 0.0, 1.0)
        ax_hits.set_ylim(0, upper_hits * 1.2)
        ax_hits.set_ylabel("Cumulative hits")
        legend_handles: List = []
        legend_labels: List[str] = []
        handles, labels = ax_hits.get_legend_handles_labels()
        legend_handles.extend(handles)
        legend_labels.extend(labels)
        if freq_thr > 0:
            thr_line = ax_hits.axhline(freq_thr, color="red", linestyle="--", linewidth=1, label="Hit threshold")
            legend_handles.append(thr_line)
            legend_labels.append("Hit threshold")
        if legend_handles:
            ax_hits.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)
    else:
        ax_bucket.set_title("Threshold hits unavailable", fontsize=11)
        ax_bucket.text(0.5, 0.5, "Need two sketch vectors", transform=ax_bucket.transAxes, ha="center")
        ax_bucket.set_xticks([])
        ax_bucket.set_yticks([])

    fig.suptitle(
        f"{state.selection.dataset_id} | {state.selection.alg} | {state.selection.label} "
        f"(Detection={state.corrtrack_detection}) | Pair ({state.selection.id1},{state.selection.start1}) vs ({state.selection.id2},{state.selection.start2})",
        fontsize=12,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = (
        f"{state.selection.label}_{state.selection.id1}_{state.selection.id2}_{state.selection.start1}_{state.selection.start2}.png"
    )
    plot_path = output_dir / plot_name
    if plot_path.exists():
        plot_path.unlink()
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def dump_reports(states: Sequence[PairDebugState], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "debug_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [state.to_dict() for state in states],
            handle,
            indent=None,
            separators=(",", ":"),
            default=_json_default,
        )

    md_path = output_dir / "debug_report.md"
    with md_path.open("w", encoding="utf-8") as handle:
        for state in states:
            sel = state.selection
            handle.write(f"### {sel.dataset_id} / {sel.alg} / {sel.label}\n")
            handle.write(f"Pair: ({sel.id1}, {sel.start1}) vs ({sel.id2}, {sel.start2}) window={sel.window}\n")
            if state.raw_windows:
                handle.write(f"- Raw corr/dist: {state.raw_windows['pearson']:.3f} / {state.raw_windows['euclidean']:.3f}\n")
            if state.normalized_windows:
                handle.write(
                    f"- Normalized corr/dist: {state.normalized_windows['pearson']:.3f} / {state.normalized_windows['euclidean']:.3f}\n"
                )
            if state.sketches:
                handle.write(
                    f"- Sketch dist/cosine: {state.sketches.get('euclidean')} / {state.sketches.get('cosine')}\n"
                )
            if state.buckets:
                history = state.buckets.get("history", {})
                total_collisions = sum(entry.get("collisions", 0) for entry in history.values())
                total_votes = sum(entry.get("votes", 0.0) for entry in history.values())
                handle.write(
                    f"- Bucket grids captured: {len(history)} (freq_history={state.freq_history}, total_votes={total_votes}, collisions={total_collisions})\n"
                )
            handle.write(f"- CorrTrack detected: {state.corrtrack_detection} corr={state.corrtrack_corr}\n\n")

    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for state in states:
        plot_pair(state, plots_dir)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CorrTrack debugging helper")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--param-grid-config", type=Path, required=True)
    parser.add_argument(
        "--exec-param-config",
        type=Path,
        default=corrtrack_main.DEFAULT_EXEC_PARAM_CONFIG,
        help="Path to execution parameter configuration module.",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Override RESULT_FOLDER from the dataset config.",
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
    parser.add_argument("--recall-by-window", dest="recall_by_window", action="store_true")
    parser.add_argument("--no-recall-by-window", dest="recall_by_window", action="store_false")
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
        recall_by_window=None,
        verbose=None,
        testing=None,
    )
    parser.add_argument("--artifact-mode", type=str, default=None)
    parser.add_argument("--loader", type=str, default=None)

    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp_artifacts/corrtrack_debug"))
    parser.add_argument("--skip-initial-run", action="store_true")
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="Rerun the full CorrTrack experiment pipeline before debugging",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args, passthrough = parser.parse_known_args(argv)

    cfg_exec = corrtrack_main._load_module(args.exec_param_config, "experiment_exec")
    if hasattr(corrtrack_main, "_apply_parallel_defaults_from_cfg"):
        corrtrack_main._apply_parallel_defaults_from_cfg(cfg_exec, args)
    args.parallel_sketch = _resolve_cfg_value(
        args.parallel_sketch,
        cfg_exec,
        "PARALLEL_SKETCH",
        corrtrack_main.DEFAULT_PARALLEL_SKETCH,
    )
    args.parallel_candidates = _resolve_cfg_value(
        args.parallel_candidates,
        cfg_exec,
        "PARALLEL_CANDIDATES",
        corrtrack_main.DEFAULT_PARALLEL_CANDIDATES,
    )
    args.parallel_validation = _resolve_cfg_value(
        args.parallel_validation,
        cfg_exec,
        "PARALLEL_VALIDATION",
        corrtrack_main.DEFAULT_PARALLEL_VALIDATION,
    )
    if args.parallel is None:
        args.parallel = any(
            flag is True
            for flag in (args.parallel_sketch, args.parallel_candidates, args.parallel_validation)
        )
    args.neg_corr = _resolve_cfg_value(
        args.neg_corr, cfg_exec, "NEG_CORR", corrtrack_main.DEFAULT_NEG_CORR
    )
    args.recall_by_window = _resolve_cfg_value(
        args.recall_by_window,
        cfg_exec,
        "RECALL_BY_WINDOW",
        corrtrack_main.DEFAULT_RECALL_BY_WINDOW,
    )
    args.artifact_mode = _resolve_cfg_value(
        args.artifact_mode,
        cfg_exec,
        "ARTIFACT_MODE",
        corrtrack_main.DEFAULT_ARTIFACT_MODE,
    )
    args.verbose = _resolve_cfg_value(
        args.verbose,
        cfg_exec,
        "VERBOSE",
        corrtrack_main.DEFAULT_VERBOSE,
    )
    args.testing = _resolve_cfg_value(
        args.testing,
        cfg_exec,
        "TESTING",
        corrtrack_main.DEFAULT_TESTING,
    )

    run_full_pipeline(args, passthrough)
    contexts = discover_datasets(args)
    base_config = corrtrack_main.build_base_config()

    for ctx in contexts:
        for alg in corrtrack_main.MODES:
            params_path = ctx.base_dir / "optim" / f"best_params_{alg}.json"
            if not params_path.exists():
                print(f"[WARN] Skipping {ctx.dataset_id}/{alg}: missing params at {params_path}")
                continue
            with params_path.open("r", encoding="utf-8") as handle:
                params = json.load(handle)

            bundle = build_selection_bundle(ctx, alg, params, base_config, args.samples_per_class)
            total_targets = len(bundle.selections) + bundle.tn_budget
            if total_targets == 0:
                print(f"[WARN] No candidate pairs found for {ctx.dataset_id}/{alg}; skipping.")
                continue

            print(
                f"[DEBUG] Instrumenting {ctx.dataset_id} / {alg} "
                f"for {len(bundle.selections)} initial pairs (TN pending {bundle.tn_budget})"
            )
            states = run_debug_pass(ctx, alg, bundle, params, args)
            out_dir = args.output_dir / ctx.dataset_id / alg
            dump_reports(states, out_dir)


if __name__ == "__main__":
    main()
