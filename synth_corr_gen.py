import os
import json
import math
import csv
from bisect import bisect_left, bisect_right
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

CSV_DELIMITER = ";"

def _format_rate(z: float) -> str:
    return f"{z:.2f}".replace(".", "p")


_STAT_TYPES = {
    "ar1",
    "wn",
    "white",
    "white_noise",
    "white-noise",
    "seasonal_arima",
    "lagged_seasonal_ar",
    "ou",
}

_NONSTAT_TYPES = {
    "rw",
    "randomwalk",
    "random_walk",
    "random-walk",
    "rw_seasonal_drift",
    "trend_poly",
    "integrated_seasonal",
}


def _stationarity_tag(base_proc: Optional[Dict[str, Any]]) -> str:
    if not base_proc:
        return "stat"
    t = str(base_proc.get("type", "ar1")).lower()
    if t in _STAT_TYPES:
        return "stat"
    if t in _NONSTAT_TYPES:
        return "nonstat"
    return "stat"


def _build_stem(
    m: int,
    n: int,
    w: int,
    z: float,
    corr_sign: str,
    threshold: float,
    template_len: int,
    num_templates: int,
    window_step: int,
    max_lag: Optional[int],
    base_proc: Optional[Dict[str, Any]],
) -> str:
    stat_tag = _stationarity_tag(base_proc)
    rate_tag = _format_rate(z)
    sign_tag = corr_sign.lower()
    proc_type = (base_proc or {}).get("type", "ar1")
    proc_tag = str(proc_type).lower().replace(" ", "_")
    lag_tag = "all" if max_lag is None else str(max_lag)
    stem = (
        f"synt_{stat_tag}_{proc_tag}_corr{rate_tag}_m{m}_w{w}_p{template_len}_"
        f"g{num_templates}_s{window_step}_lag{lag_tag}_"
        f"sign{sign_tag}_thr{str(threshold).replace('.', 'p')}"
    )
    return stem


def _obs_sigma_value(base_proc: Optional[Dict[str, Any]]) -> float:
    if not base_proc:
        return 0.0
    try:
        return max(0.0, float(base_proc.get("obs_sigma", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _add_observation_noise(
    X: np.ndarray,
    base_proc: Optional[Dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    obs_sigma = _obs_sigma_value(base_proc)
    if obs_sigma <= 0.0:
        return X.astype(np.float32, copy=False)
    noisy = np.asarray(X, dtype=np.float64) + rng.normal(0.0, obs_sigma, size=X.shape)
    return noisy.astype(np.float32, copy=False)


def _gen_base_series(
    m: int, n: int, base_proc: Optional[Dict[str, Any]], rng: np.random.Generator
) -> np.ndarray:
    """Generate base data of shape (m, n)."""
    if base_proc is None:
        base_proc = {"type": "ar1", "phi": 0.6, "sigma": 1.0}

    kind = str(base_proc.get("type", "ar1")).lower()
    sigma = float(base_proc.get("sigma", 1.0))

    X = np.empty((m, n), dtype=np.float32)

    if kind in ("wn", "white", "white_noise", "white-noise"):
        X[:] = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        return X

    if kind in ("rw", "randomwalk", "random_walk", "random-walk"):
        eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        X[:, 0] = eps[:, 0]
        for t in range(1, n):
            X[:, t] = X[:, t - 1] + eps[:, t]
        return _add_observation_noise(X, base_proc, rng)

    if kind == "seasonal_arima":
        phi = float(base_proc.get("phi", 0.6))
        theta = float(base_proc.get("theta", -0.3))
        period = int(base_proc.get("season_period", 200))
        amplitude = float(base_proc.get("season_amplitude", 1.0))
        phases = rng.uniform(0.0, 2 * math.pi, size=m)
        noise = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        ma_noise = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        X[:, 0] = noise[:, 0]
        for t in range(1, n):
            ma_term = theta * ma_noise[:, t - 1]
            X[:, t] = phi * X[:, t - 1] + noise[:, t] + ma_term
        t_idx = np.arange(n, dtype=np.float32)
        seasonal = amplitude * np.sin(2 * math.pi * t_idx[None, :] / max(1, period) + phases[:, None])
        return (X + seasonal).astype(np.float32)

    if kind == "lagged_seasonal_ar":
        phi_short = float(base_proc.get("phi_short", 0.3))
        phi_long = float(base_proc.get("phi_long", 0.6))
        lag = int(base_proc.get("season_lag", 24))
        lag = max(1, lag)
        eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        X[:, :lag] = eps[:, :lag]
        for t in range(lag, n):
            X[:, t] = (
                phi_short * X[:, t - 1]
                + phi_long * X[:, t - lag]
                + eps[:, t]
            )
        return X

    if kind == "ou":
        theta = float(base_proc.get("theta", 0.3))
        mean_level = float(base_proc.get("mu", 0.0))
        dt = float(base_proc.get("dt", 1.0))
        X[:, 0] = rng.normal(mean_level, sigma, size=m)
        for t in range(1, n):
            noise = rng.normal(0.0, sigma * math.sqrt(dt), size=m)
            X[:, t] = X[:, t - 1] + theta * (mean_level - X[:, t - 1]) * dt + noise
        return X.astype(np.float32)

    if kind == "rw_seasonal_drift":
        drift = float(base_proc.get("drift", 0.05))
        amplitude = float(base_proc.get("season_amplitude", 0.5))
        period = int(base_proc.get("season_period", 288))
        phases = rng.uniform(0.0, 2 * math.pi, size=m)
        eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        X[:, 0] = eps[:, 0]
        for t in range(1, n):
            seasonal = amplitude * np.sin(2 * math.pi * t / max(1, period) + phases)
            X[:, t] = X[:, t - 1] + drift + seasonal + eps[:, t]
        return _add_observation_noise(X, base_proc, rng)

    if kind == "trend_poly":
        t_norm = np.linspace(0.0, 1.0, n, dtype=np.float32)
        slopes = rng.normal(0.5, 0.2, size=m)
        curves = rng.normal(0.0, 0.1, size=m)
        trend = slopes[:, None] * t_norm[None, :] + curves[:, None] * (t_norm[None, :] ** 2)
        noise = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        ar_coeff = float(base_proc.get("phi", 0.5))
        ar_component = np.zeros((m, n), dtype=np.float32)
        for t in range(1, n):
            ar_component[:, t] = ar_coeff * ar_component[:, t - 1] + noise[:, t]
        return (trend + ar_component).astype(np.float32)

    if kind == "integrated_seasonal":
        lag = int(base_proc.get("season_lag", 24))
        lag = max(1, lag)
        phi = float(base_proc.get("phi", 0.4))
        psi = float(base_proc.get("psi", 0.5))
        eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        seasonal = np.zeros((m, n), dtype=np.float32)
        seasonal[:, :lag] = eps[:, :lag]
        for t in range(lag, n):
            seasonal[:, t] = phi * seasonal[:, t - 1] + psi * seasonal[:, t - lag] + eps[:, t]
        X[:, 0] = seasonal[:, 0]
        for t in range(1, n):
            X[:, t] = X[:, t - 1] + seasonal[:, t]
        return _add_observation_noise(X, base_proc, rng)

    # Default AR(1)
    phi = float(base_proc.get("phi", 0.6))
    eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
    X[:, 0] = eps[:, 0]
    for t in range(1, n):
        X[:, t] = phi * X[:, t - 1] + eps[:, t]
    return X


def _apply_volatility_equalizer(
    X: np.ndarray,
    config: Optional[Dict[str, Any]],
    default_window: int,
) -> None:
    """
    Normalize local volatility so that sliding windows have comparable std dev.

    Parameters in ``config``:
      - ``window``: window length used to estimate local variance (default: default_window).
      - ``target_std``: target standard deviation after normalization (default: 1.0).
      - ``min_std``: lower bound to avoid division by very small numbers (default: 1e-3).
      - ``pad_mode``: numpy.pad mode ("reflect", "edge", "constant"; default: "reflect").
    """
    if not config:
        return

    window = int(config.get("window", default_window))
    if window <= 1:
        return
    window = min(window, X.shape[1])
    target_std = float(config.get("target_std", 1.0))
    min_std = float(config.get("min_std", 1e-3))
    pad_mode = str(config.get("pad_mode", "reflect")).lower()
    if pad_mode not in {"reflect", "edge", "constant"}:
        pad_mode = "reflect"

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    kernel = np.ones(window, dtype=np.float64) / float(window)

    for idx in range(X.shape[0]):
        series = X[idx].astype(np.float64, copy=False)
        padded = np.pad(series, (pad_left, pad_right), mode=pad_mode)
        mean = np.convolve(padded, kernel, mode="valid")
        sq = np.convolve(padded * padded, kernel, mode="valid")
        var = np.maximum(sq - mean * mean, min_std * min_std)
        std = np.sqrt(var)
        scaled = series / std * target_std
        X[idx] = scaled.astype(np.float32, copy=False)


def _choose_sign(corr_sign: str, rng: np.random.Generator) -> int:
    c = corr_sign.lower()
    if c == "pos":
        return +1
    if c == "neg":
        return -1
    return +1 if rng.random() < 0.5 else -1


def _target_sign_for_pair(
    corr_sign: str,
    pair_idx: int,
    pair_goal: int,
    rng: np.random.Generator,
) -> int:
    """
    Pick desired sign at placement time.
    For corr_sign="both", force one positive and one negative pair when possible.
    """
    c = corr_sign.lower()
    if c == "pos":
        return +1
    if c == "neg":
        return -1
    if pair_goal >= 2 and pair_idx == 0:
        return +1
    if pair_goal >= 2 and pair_idx == 1:
        return -1
    return +1 if rng.random() < 0.5 else -1


def _apply_sign_to_template(
    xw: np.ndarray,
    yw: np.ndarray,
    r_xy: float,
    desired_sign: int,
) -> Tuple[np.ndarray, float]:
    """
    Mirror y around its mean when sign must be flipped.
    This preserves y mean/std and approximately flips corr(x,y) sign.
    """
    current_sign = +1 if r_xy >= 0.0 else -1
    if current_sign == desired_sign:
        return yw, r_xy

    mean_y = float(np.mean(yw, dtype=np.float64))
    yw_flipped = (2.0 * mean_y - yw.astype(np.float64, copy=False)).astype(np.float32)
    r_new = _pearson_raw(xw, yw_flipped)
    return yw_flipped, r_new


def _pearson_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation on raw data (no prior normalization)."""
    r = np.corrcoef(x, y)[0, 1]
    if r > 1.0:
        r = 1.0
    if r < -1.0:
        r = -1.0
    return float(r)


def _sample_base_correlated_template(
    base_proc: Optional[Dict[str, Any]],
    length: int,
    threshold: float,
    corr_sign: str,
    rng: np.random.Generator,
    max_attempts: int = 64,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Draw correlated windows whose marginal stats follow the configured base process.
    """
    min_std = 1e-3
    for _ in range(max_attempts):
        base = _gen_base_series(2, length, base_proc, rng)
        x_raw = base[0].astype(np.float64, copy=False)
        y_raw = base[1].astype(np.float64, copy=False)

        mx = float(x_raw.mean())
        sx = float(x_raw.std())
        my = float(y_raw.mean())
        sy = float(y_raw.std())
        if sx < min_std or sy < min_std:
            continue

        x_norm = (x_raw - mx) / sx
        eps = rng.normal(0.0, 1.0, size=length)
        r_star = threshold + rng.random() * (1.0 - threshold)
        r_star = min(r_star, 1.0)
        sign = _choose_sign(corr_sign, rng)
        noise_scale = math.sqrt(max(0.0, 1.0 - r_star * r_star))
        y_corr = sign * r_star * x_norm + noise_scale * eps
        y_new = my + sy * y_corr

        xw = x_raw.astype(np.float32, copy=False)
        yw = y_new.astype(np.float32, copy=False)
        r = _pearson_raw(xw, yw)
        if abs(r) >= threshold:
            return xw, yw, r
    return None


def _make_base_templates(
    num_templates: int,
    length: int,
    threshold: float,
    corr_sign: str,
    base_proc: Optional[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    templates: List[Tuple[np.ndarray, np.ndarray, float]] = []
    attempts = 0
    max_attempts = max(200, 20 * max(1, num_templates))
    while len(templates) < num_templates and attempts < max_attempts:
        attempts += 1
        tpl = _sample_base_correlated_template(base_proc, length, threshold, corr_sign, rng)
        if tpl is not None:
            templates.append(tpl)
    if len(templates) < num_templates:
        raise RuntimeError(
            f"Could not build {num_templates} base templates of length {length} meeting |r| >= {threshold}"
        )
    return templates


def _series_with_slots(slots_by_series: List[List[int]]) -> List[int]:
    return [idx for idx, slots in enumerate(slots_by_series) if slots]


def _pick_series_with_weight(
    series_indices: List[int],
    slots_by_series: List[List[int]],
    rng: np.random.Generator,
) -> int:
    counts = np.array([len(slots_by_series[i]) for i in series_indices], dtype=np.float64)
    probs = counts / counts.sum()
    choice = int(rng.choice(series_indices, p=probs))
    return choice


def _normalize_window_step(window_step: Optional[int]) -> int:
    step = 1 if window_step is None else int(window_step)
    if step <= 0:
        raise ValueError(f"window_step must be positive; got {window_step!r}")
    return step


def _normalize_max_lag(max_lag: Optional[int]) -> Optional[int]:
    if max_lag is None:
        return None
    lag = int(max_lag)
    if lag < 0:
        return None
    return lag


def _candidate_partner_starts(
    slots: List[int],
    start: int,
    max_lag: Optional[int],
) -> List[int]:
    if not slots:
        return []
    if max_lag is None:
        return slots
    lo = bisect_left(slots, start - max_lag)
    hi = bisect_right(slots, start + max_lag)
    return slots[lo:hi]


def _remove_overlapping_starts(
    slots: List[int],
    start: int,
    length: int,
) -> List[int]:
    if not slots:
        return []
    left_cut = start - length + 1
    right_cut = start + length - 1
    return [slot for slot in slots if slot < left_cut or slot > right_cut]


def _max_nonoverlap_slots_per_series(
    n: int,
    length: int,
    window_step: int,
) -> int:
    if length > n:
        return 0
    gap = int(math.ceil(length / float(window_step))) * window_step
    return 1 + max(0, (n - length) // gap)


def make_corr_dataset(
    save_dir: str,
    m: int,
    n: int,
    z: float,
    w: int,
    *,
    template_len: Optional[int] = None,
    num_templates: int = 4,
    threshold: float = 0.7,
    corr_sign: str = "pos",  # "pos" | "neg" | "both"
    base_proc: Optional[Dict[str, Any]] = None,
    window_step: int = 1,
    max_lag: Optional[int] = None,
    volatility_equalizer: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 7,
    hash_seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate m synthetic time series of length n with a fraction z of all samples
    participating in correlated window pairs of length ``template_len``.

    Semantics:
      - Concatenate all series into one vector of length m * n.
      - Target number of correlated pairs: floor(z * m * n / (2 * p)), where p = template_len.
      - Each pair uses two windows of length p (one per series), with starts on the
        ``window_step`` grid and optional ``max_lag`` constraint.
      - Injected windows remain non-overlapping within each series.
      - Templates are sampled from the base process so marginals match the background.
      - w is the evaluation window length and must be divisible by p (p defaults to w).
    """
    assert 0.0 <= z <= 1.0, "z must be in [0,1]"
    assert w > 0 and n > 0, "w and n must be positive"
    if template_len is None:
        template_len = w
    p = int(template_len)
    assert p > 0, "template_len must be positive"
    assert p <= n, "template_len must be <= n"
    if w % p != 0:
        raise ValueError(f"w ({w}) must be divisible by template_len ({p})")
    window_step = _normalize_window_step(window_step)
    max_lag = _normalize_max_lag(max_lag)

    if hash_seed is not None:
        os.environ["PYTHONHASHSEED"] = str(hash_seed)
    rng = np.random.default_rng(seed)

    # Base data
    X = _gen_base_series(m, n, base_proc, rng)  # shape (m, n)
    _apply_volatility_equalizer(X, volatility_equalizer, w)

    # Prepare step-aligned candidate starts. Non-overlap is enforced lazily by
    # retiring overlapping starts on a series after each placement.
    slots_by_series: List[List[int]] = [
        list(range(0, n - p + 1, window_step)) for _ in range(m)
    ]
    slots_total = sum(len(s) for s in slots_by_series)
    max_slots_per_series = _max_nonoverlap_slots_per_series(n, p, window_step)
    max_pairs = (m * max_slots_per_series) // 2

    pair_target = int(z * m * n // (2 * p))
    pair_goal = min(pair_target, max_pairs)
    if pair_goal <= 0 or slots_total < 2:
        pair_goal = 0

    templates = _make_base_templates(num_templates, p, threshold, corr_sign, base_proc, rng)

    correlated_rows: List[Tuple[str, str, int, int, float]] = []
    pairs_used = 0
    placement_attempts = 0
    max_placement_attempts = max(2000, 40 * max(1, pair_goal))
    discarded_starts = 0

    while pairs_used < pair_goal and placement_attempts < max_placement_attempts:
        placement_attempts += 1
        available_series = _series_with_slots(slots_by_series)
        if len(available_series) < 2:
            break

        i1 = _pick_series_with_weight(available_series, slots_by_series, rng)
        slots1 = slots_by_series[i1]
        if not slots1:
            continue

        start1_idx = int(rng.integers(0, len(slots1)))
        start1 = slots1[start1_idx]

        partner_series: List[int] = []
        partner_slot_sets: List[List[int]] = []
        partner_weights: List[float] = []
        for i2 in available_series:
            if i2 == i1:
                continue
            partner_slots = _candidate_partner_starts(slots_by_series[i2], start1, max_lag)
            if partner_slots:
                partner_series.append(i2)
                partner_slot_sets.append(partner_slots)
                partner_weights.append(float(len(partner_slots)))

        if not partner_series:
            del slots1[start1_idx]
            discarded_starts += 1
            continue

        partner_probs = np.asarray(partner_weights, dtype=np.float64)
        partner_probs /= partner_probs.sum()
        partner_idx = int(rng.choice(len(partner_series), p=partner_probs))
        i2 = partner_series[partner_idx]
        slots2 = slots_by_series[i2]
        partner_slots = partner_slot_sets[partner_idx]
        start2 = int(partner_slots[int(rng.integers(0, len(partner_slots)))])

        tpl = templates[int(rng.integers(0, len(templates)))]
        xw, yw, r_tpl = tpl
        desired_sign = _target_sign_for_pair(corr_sign, pairs_used, pair_goal, rng)
        yw_use, r_use = _apply_sign_to_template(xw, yw, r_tpl, desired_sign)

        X[i1, start1 : start1 + p] = xw
        X[i2, start2 : start2 + p] = yw_use

        slots_by_series[i1] = _remove_overlapping_starts(slots1, start1, p)
        slots_by_series[i2] = _remove_overlapping_starts(slots2, start2, p)

        correlated_rows.append((f"s{i1+1}", f"s{i2+1}", start1 + 1, start2 + 1, r_use))
        pairs_used += 1

    samples_correlated = pairs_used * 2 * p
    achieved_z = samples_correlated / float(m * n) if m * n > 0 else 0.0

    stem = _build_stem(
        m=m,
        n=n,
        w=w,
        z=z,
        corr_sign=corr_sign,
        threshold=threshold,
        template_len=p,
        num_templates=num_templates,
        window_step=window_step,
        max_lag=max_lag,
        base_proc=base_proc,
    )
    os.makedirs(save_dir, exist_ok=True)

    # Data .npz: (n, m+1) with first column = index 1..n, then S1..Sm
    arr = np.zeros((n, m + 1), dtype=np.float32)
    arr[:, 0] = np.arange(1, n + 1, dtype=np.float32)
    arr[:, 1:] = X.T

    data_path = os.path.join(save_dir, f"{stem}.npz")
    np.savez_compressed(data_path, arr)

    # Correlated pairs CSV
    corr_csv_path = os.path.join(save_dir, f"{stem}_correlated.csv")
    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER)
        writer.writerow(["id1", "id2", "time1", "time2", "corr"])
        for row in correlated_rows:
            writer.writerow(row)

    meta = {
        "m": m,
        "n": n,
        "w": w,
        "template_len": p,
        "num_templates": num_templates,
        "window_step": window_step,
        "max_lag": max_lag,
        "threshold": threshold,
        "corr_sign": corr_sign,
        "target_z": z,
        "achieved_z": achieved_z,
        "pair_target": pair_target,
        "pair_goal": pair_goal,
        "pairs_used": pairs_used,
        "placement_attempts": placement_attempts,
        "discarded_starts": discarded_starts,
        "samples_correlated": samples_correlated,
        "slots_total": slots_total,
        "slots_used": pairs_used * 2,
        "base_proc": base_proc,
        "volatility_equalizer": volatility_equalizer,
        "seed": seed,
        "hash_seed": hash_seed,
    }
    meta_path = os.path.join(save_dir, f"{stem}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "data_npz": data_path,
        "correlated_csv": corr_csv_path,
        "meta_json": meta_path,
        "stem": stem,
    }


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Synthetic correlated time series generator "
            "(NPZ data + CSV/JSON metadata)."
        )
    )
    p.add_argument("--save-dir", type=str, default=".", help="Folder to write outputs.")
    p.add_argument("--m", type=int, required=True, help="Number of time series.")
    p.add_argument("--n", type=int, required=True, help="Length of each time series.")
    p.add_argument(
        "--z",
        type=float,
        required=True,
        help="Target fraction of all samples that belong to correlated pairs (0..1).",
    )
    p.add_argument("--w", type=int, required=True, help="Evaluation window length (must be divisible by p).")
    p.add_argument(
        "--template-len",
        type=int,
        default=None,
        help="Length p of each correlated template (defaults to w).",
    )
    p.add_argument(
        "--num-templates",
        type=int,
        default=4,
        help="How many distinct base templates to sample and reuse.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum Pearson correlation threshold (default: 0.7).",
    )
    p.add_argument(
        "--corr-sign",
        type=str,
        choices=["pos", "neg", "both"],
        default="pos",
        help="Correlation sign to inject: pos, neg, or both (default: pos).",
    )
    p.add_argument(
        "--base-proc",
        type=str,
        default=None,
        help="JSON string describing the base process (e.g., '{\"type\":\"ar1\",\"phi\":0.6,\"sigma\":1.0}').",
    )
    p.add_argument(
        "--window-step",
        type=int,
        default=1,
        help="Start-time grid for injected pairs (default: 1).",
    )
    p.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum absolute lag |time1 - time2| for injected pairs (default: unconstrained).",
    )
    p.add_argument(
        "--volatility-equalizer",
        type=str,
        default=None,
        help="Optional JSON string with volatility equalizer config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7).",
    )
    p.add_argument(
        "--hash-seed",
        type=int,
        default=None,
        help="PYTHONHASHSEED value (default: None).",
    )
    args = p.parse_args()

    base_proc = json.loads(args.base_proc) if args.base_proc else None
    vol_eq = json.loads(args.volatility_equalizer) if args.volatility_equalizer else None

    return dict(
        save_dir=args.save_dir,
        m=args.m,
        n=args.n,
        z=args.z,
        w=args.w,
        template_len=args.template_len,
        num_templates=args.num_templates,
        threshold=args.threshold,
        corr_sign=args.corr_sign,
        base_proc=base_proc,
        window_step=args.window_step,
        max_lag=args.max_lag,
        volatility_equalizer=vol_eq,
        seed=args.seed,
        hash_seed=args.hash_seed,
    )


if __name__ == "__main__":
    kwargs = _parse_args()
    out = make_corr_dataset(**kwargs)
    print(json.dumps(out, indent=2))
