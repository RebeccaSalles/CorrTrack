import os
import json
import math
import csv
from bisect import bisect_left, bisect_right
from typing import Dict, List, Tuple, Optional, Any, Union

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
    num_templates: Union[int, str],
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
    diff_space: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Mirror y around its mean when sign must be flipped.
    This preserves y mean/std and approximately flips corr(x,y) sign.
    Mirroring around the mean negates every increment too (diff(2c - y) =
    -diff(y)), so the same level-space flip is valid regardless of which
    space r_xy/r_new is measured in -- only the reported statistic's
    representation (diff_space) needs to match how r_xy was computed by
    the caller, so nonstationary (increment-controlled) pairs report an
    honest increment-space correlation rather than a level-space one.
    """
    current_sign = +1 if r_xy >= 0.0 else -1
    if current_sign == desired_sign:
        return yw, r_xy

    mean_y = float(np.mean(yw, dtype=np.float64))
    yw_flipped = (2.0 * mean_y - yw.astype(np.float64, copy=False)).astype(np.float32)
    if diff_space:
        r_new = _pearson_raw(np.diff(xw.astype(np.float64, copy=False)), np.diff(yw_flipped.astype(np.float64, copy=False)))
    else:
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


def _apply_continuity_shift(
    segment: np.ndarray,
    prev_value: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    (2026-08-24) Shift a freshly-generated nonstationary template by a
    constant so it continues from the host series' own value just before
    the splice point, plus one ordinary fresh step, instead of jumping
    from whatever level the template happened to start at. Translation-
    invariant (diff(x + c) == diff(x)), so this preserves the controlled
    injected correlation exactly -- it only replaces the artificial
    splice-boundary discontinuity with a step statistically
    indistinguishable from any other step in the series. Empirically, the
    unshifted discontinuity averaged ~30x a typical step (see
    docs/implementation_log.md's 2026-08-24 entry) and was the dominant
    cause of injected pairs failing to validate as correlated even though
    _sample_base_correlated_template's own acceptance check always passed.
    """
    boundary_step = float(rng.normal(0.0, sigma))
    offset = (prev_value + boundary_step) - float(segment[0])
    return (segment.astype(np.float64, copy=False) + offset).astype(np.float32, copy=False)


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
    nonstat = _stationarity_tag(base_proc) == "nonstat"
    for _ in range(max_attempts):
        base = _gen_base_series(2, length, base_proc, rng)
        x_raw = base[0].astype(np.float64, copy=False)
        y_raw = base[1].astype(np.float64, copy=False)

        if nonstat:
            # (2026-08-24) Coupling LEVELS the way the stationary branch
            # below does assumes step-scale and window-scale track
            # together -- true for stationary processes, false for a
            # unit-root process, where window-level std grows with window
            # length while per-step innovation variance stays fixed. That
            # mismatch made the injected correlation collapse under
            # differencing as window length grew (empirically measured:
            # diff-space |r| fell from ~0.79 at p=32 to ~0.44 at p=512 for
            # a random walk, well below corr_threshold=0.7 -- see
            # docs/implementation_log.md's 2026-08-24 entry). Couple the
            # STEPS instead (constant scale regardless of window length)
            # and cumsum back to levels, so the controlled correlation
            # lives in the increments by construction -- coherent with
            # validating this data under preprocess=True, the regime this
            # generator's own nonstationary types are meant to be used
            # under (raw-level Pearson on unit-root series is dominated by
            # spurious correlation, per the same log entry).
            x_diffs = np.diff(x_raw)
            y_diffs_ref = np.diff(y_raw)
            mxd = float(x_diffs.mean())
            sxd = float(x_diffs.std())
            myd = float(y_diffs_ref.mean())
            syd = float(y_diffs_ref.std())
            if sxd < min_std or syd < min_std:
                continue

            x_diffs_norm = (x_diffs - mxd) / sxd
            eps = rng.normal(0.0, 1.0, size=length - 1)
            r_star = threshold + rng.random() * (1.0 - threshold)
            r_star = min(r_star, 1.0)
            sign = _choose_sign(corr_sign, rng)
            noise_scale = math.sqrt(max(0.0, 1.0 - r_star * r_star))
            y_diffs_corr = sign * r_star * x_diffs_norm + noise_scale * eps
            y_diffs = myd + syd * y_diffs_corr

            xw = x_raw.astype(np.float32, copy=False)
            yw = (float(y_raw[0]) + np.concatenate(([0.0], np.cumsum(y_diffs)))).astype(np.float32, copy=False)
            r = _pearson_raw(x_diffs.astype(np.float32, copy=False), y_diffs.astype(np.float32, copy=False))
            if abs(r) >= threshold:
                return xw, yw, r
            continue

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


def _num_templates_arg_type(value: str) -> Union[int, str]:
    if value.strip().lower() == "auto":
        return "auto"
    return int(value)


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


def resolve_pair_goal(m: int, n: int, z: float, template_len: int, window_step: int) -> int:
    """
    Single source of truth for make_corr_dataset's own pair_goal
    computation -- also needed by datasets/synth_loader.py's cache-stem
    builder, so num_templates="auto" resolves to the IDENTICAL value both
    when computing the cache lookup path and when make_corr_dataset
    actually generates/saves the file. Duplicating this formula in two
    places would risk exactly the stem mismatch (cache permanently
    missing) it exists to avoid.
    """
    p = int(template_len)
    window_step = _normalize_window_step(window_step)
    slots_total = m * len(range(0, n - p + 1, window_step))
    max_slots_per_series = _max_nonoverlap_slots_per_series(n, p, window_step)
    max_pairs = (m * max_slots_per_series) // 2
    pair_target = int(z * m * n // (2 * p))
    pair_goal = min(pair_target, max_pairs)
    if pair_goal <= 0 or slots_total < 2:
        pair_goal = 0
    return pair_goal


def make_corr_dataset(
    save_dir: str,
    m: int,
    n: int,
    z: float,
    w: int,
    *,
    template_len: Optional[int] = None,
    num_templates: Union[int, str] = 4,
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
      - num_templates="auto" sizes the template pool to the actual number of pairs
        that will be placed, so every placement gets its own never-reused template --
        avoids "phantom" correlation between unrelated series that happen to share a
        template at overlapping positions (see docs/implementation_log.md's
        2026-08-24 (c) entry). An explicit int below that count is rejected rather
        than silently under-templated.
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
    nonstat = _stationarity_tag(base_proc) == "nonstat"
    proc_sigma = float((base_proc or {}).get("sigma", 1.0))

    # Prepare step-aligned candidate starts. Non-overlap is enforced lazily by
    # retiring overlapping starts on a series after each placement.
    slots_by_series: List[List[int]] = [
        list(range(0, n - p + 1, window_step)) for _ in range(m)
    ]
    slots_total = sum(len(s) for s in slots_by_series)
    pair_goal = resolve_pair_goal(m, n, z, p, window_step)
    pair_target = int(z * m * n // (2 * p))  # kept for meta.json reporting only; pair_goal (above) is authoritative

    # (2026-08-24) num_templates="auto" resolves to pair_goal -- one
    # never-reused template per placement, eliminating the template-reuse
    # collision problem entirely (see docs/implementation_log.md's
    # 2026-08-24 (c) entry: unrelated series drawing a role from the same
    # template at overlapping positions show real, unintended correlation
    # with each other; confirmed present regardless of base_proc). An
    # explicit int below pair_goal is rejected rather than silently
    # under-templated, since that reintroduces exactly this problem.
    if isinstance(num_templates, str):
        if num_templates.strip().lower() != "auto":
            raise ValueError(f"num_templates string value must be 'auto', got {num_templates!r}")
        num_templates = pair_goal
    elif num_templates < pair_goal:
        raise ValueError(
            f"num_templates={num_templates} is less than pair_goal={pair_goal} -- this "
            f"would reuse templates across unrelated series pairs, which creates real "
            f"but unintended correlation between them (see docs/implementation_log.md's "
            f"2026-08-24 (c) entry). Pass num_templates='auto' to size this correctly, "
            f"or set num_templates >= {pair_goal} explicitly."
        )

    templates = _make_base_templates(num_templates, p, threshold, corr_sign, base_proc, rng)
    template_order = rng.permutation(len(templates)) if templates else np.empty(0, dtype=np.int64)

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

        # (2026-08-24) Draw without replacement -- template_order was
        # pre-shuffled once, and pairs_used increments exactly once per
        # successful placement, so each placement gets a distinct template
        # index (guaranteed available since len(templates) >= pair_goal).
        tpl = templates[int(template_order[pairs_used])]
        xw, yw, r_tpl = tpl
        desired_sign = _target_sign_for_pair(corr_sign, pairs_used, pair_goal, rng)
        yw_use, r_use = _apply_sign_to_template(
            xw, yw, r_tpl, desired_sign, diff_space=nonstat
        )

        if nonstat:
            if start1 > 0:
                xw = _apply_continuity_shift(xw, float(X[i1, start1 - 1]), proc_sigma, rng)
            if start2 > 0:
                yw_use = _apply_continuity_shift(yw_use, float(X[i2, start2 - 1]), proc_sigma, rng)

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


# ======================================================================
# Density-deterministic generator (2026-09-10)
# ----------------------------------------------------------------------
# `make_density_targeted_dataset` takes a TARGET EFFECTIVE TUPLE DENSITY
# (bf_correlated / bf_tested) plus the exact evaluation config, and builds
# a dataset whose measured density hits the target within a tight
# tolerance -- deterministically for a given (target, seed, base_proc,
# corr_sign, n_epochs, duty) -- together with the EXACT ground-truth
# (pair, lag, window, signed_r) set by construction, so a full brute-force
# pass is no longer needed per cell.  See
# ~/.claude/plans/misty-stargazing-liskov.md for the full design.
# ======================================================================

def _a_from_r(r: float) -> float:
    """Loading a such that two equal-loading members correlate at |r|:
    r = a^2 / (a^2 + 1)  =>  a = sqrt(r / (1 - r))."""
    r = min(max(float(r), 1e-6), 1.0 - 1e-6)
    return math.sqrt(r / (1.0 - r))


def _gen_shared_driver_signal(n: int, sigma_smooth: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Unit-variance (near-)white noise. Used as a group's shared driver c_group:
    its autocorrelation is ~0 at every nonzero lag, so a within-group pair is
    correlated at exactly one lag bucket (|o_i - o_j|) and -- crucially -- two
    INDEPENDENT group drivers stay windowed-uncorrelated over the evaluation window
    (a driver with a longer correlation length makes independent instances
    windowed-correlated above threshold: a hard geometric incompatibility at the
    window / step sizes this experiment uses, verified empirically)."""
    raw = rng.standard_normal(n)
    if float(sigma_smooth) > 0.4:
        pad = int(math.ceil(4.0 * sigma_smooth))
        k = np.arange(-pad, pad + 1, dtype=np.float64)
        kern = np.exp(-0.5 * (k / float(sigma_smooth)) ** 2)
        kern /= kern.sum()
        raw = np.convolve(np.concatenate([raw[:pad][::-1], raw, raw[-pad:][::-1]]),
                          kern, mode="same")[pad:pad + n]
    c = (raw - raw.mean()) / (raw.std() + 1e-12)
    return c.astype(np.float64)


def _build_group_schedule(n_groups: int, n_epochs: int, corr_sign: str,
                          duty: float, seed: int):
    """Per-group state timeline over `n_epochs` epochs, states in {0, +1, -1}.
    Deterministic from (group, seed). Every group is ON for the SAME number of
    epochs (n_on = round(duty * n_epochs)) so effective density stays exact; the
    schedule only varies which epochs and which signs. For corr_sign='both' at
    least one group is forced to contain a +<->- sign flip. Returns
    (states (n_groups, n_epochs) int8, events list)."""
    rng = np.random.default_rng((int(seed) & 0xFFFFFFFF) ^ 0x5C4ED<<1)
    if corr_sign == "pos":
        on_signs = [1]
    elif corr_sign == "neg":
        on_signs = [-1]
    else:
        on_signs = [1, -1]
    n_on = max(1, min(n_epochs, round(float(duty) * n_epochs)))
    states = np.zeros((n_groups, n_epochs), dtype=np.int8)
    for g in range(n_groups):
        on_epochs = np.sort(rng.choice(n_epochs, size=n_on, replace=False))
        signs = rng.choice(on_signs, size=n_on)
        if len(on_signs) == 2 and g == 0 and n_on >= 2:
            # force a genuine sign-flip transition in the first group
            signs[0], signs[1] = 1, -1
            on_epochs[:2] = np.sort(on_epochs[:2])
        for e, s in zip(on_epochs, signs):
            states[g, int(e)] = int(s)
    events = []
    for g in range(n_groups):
        for e in range(1, n_epochs):
            if states[g, e] != states[g, e - 1]:
                events.append({"group": int(g), "epoch": int(e),
                               "from": int(states[g, e - 1]), "to": int(states[g, e])})
    return states, events


def _canonical_gt_row(cur_s, hist_s, t1, t2, w):
    """(current, historical) -> canonical (later-start first, id-sorted tie), matching
    library_corrtrack_parallel._normalize_bf_key / _canonicalize_rows."""
    if t1 < t2 or (t1 == t2 and cur_s > hist_s):
        return (hist_s, cur_s, t2, t1, w)
    return (cur_s, hist_s, t1, t2, w)


def make_density_targeted_dataset(
    m: int,
    n: int,
    *,
    target_density: float,
    corr_threshold: float,
    window_size: int,
    window_step: int,
    n_lags: int,
    n_eval_steps: int,
    base_proc: Optional[Dict[str, Any]] = None,
    preprocess: bool = False,
    lag_band: int = 4,
    corr_sign: str = "pos",
    n_epochs: int = 6,
    duty: float = 1.0,
    burst_length_windows: Optional[Union[int, Tuple[int, int]]] = None,
    r_max: float = 0.99,
    corr_margin: float = 0.03,
    near_threshold_fraction: float = 0.05,
    loading_skew: str = "high",
    seed: int = 7,
    tolerance: float = 0.15,
    save_dir: Optional[str] = None,
    verify_bf: bool = True,
    _max_correction_regens: int = 3,
    group_size: Optional[int] = None,
):
    """Generate an (m, n) dataset whose brute-force effective tuple density
    (correlated (pair,lag,window) tuples / all tested tuples) equals
    `target_density` within `tolerance`, with the exact ground-truth set
    returned. See the module comment above and the plan file.

    `burst_length_windows`: None (default) -> each ON epoch is correlated for its FULL
    span (the original, persistent model). If set (an int, or a (lo, hi) range), each
    group draws its own burst length k (windows) once, and each of its ON epochs is
    correlated for only a short k-window burst centered inside that epoch, OFF the rest
    of the epoch -- i.e. many short, isolated correlation events instead of one long
    segment. Motivated by a real finding (2026-09-11): on real data, brute-force recall
    correlates far more with how many CONSECUTIVE windows a correlation persists (0.51 at
    1-2 windows -> 0.82 at 32+) than with its strength -- a discovery-latency effect the
    old persistent-only generator could never exercise.

    Returns a dict:
      data            (n, m+1) float32 -- col 0 is a 1..n index, cols 1..m the series
      ids             ["s1".."sm"]
      gt_rows         (K, 5) int64  canonical [s1,s2,t1,t2,w] over ALL eval windows
      gt_corrs        (K,)  float64 signed
      analytic_density, verified_density, group_size, n_groups, lag_band_measured,
      schedule, events, rho_hist, meta_json (if save_dir), ...
    """
    corr_sign = str(corr_sign).lower()
    if corr_sign not in ("pos", "neg", "both"):
        raise ValueError("corr_sign must be 'pos' | 'neg' | 'both'")
    step = _normalize_window_step(window_step)
    w = int(window_size)
    L_test = n_lags // step + 1
    if L_test < 1:
        raise ValueError("n_lags // window_step + 1 must be >= 1")
    b = int(max(1, min(lag_band, L_test)))
    n_on = max(1, min(n_epochs, round(float(duty) * n_epochs)))
    on_frac = n_on / float(n_epochs)
    total_pairs = m * (m - 1) / 2.0
    max_windows = (n - w) // step + 1
    if max_windows < 1:
        raise ValueError("n too short for even one evaluation window")
    n_eval = int(min(int(n_eval_steps), max_windows))

    # burst mode: an ON epoch is correlated for only ~burst_k windows out of the
    # ~windows_per_epoch it spans, not the whole epoch -- on_frac_density (used only to
    # pick a starting (ng, g), the correction loop re-solves against the real measured
    # density regardless) must reflect that much smaller true ON-window fraction.
    on_frac_density = on_frac
    if burst_length_windows is not None:
        _epoch_len_est = max(step, (n // n_epochs // step) * step)
        _windows_per_epoch = max(1, _epoch_len_est // step)
        _burst_mid = (float(burst_length_windows) if np.isscalar(burst_length_windows)
                     else float(sum(burst_length_windows)) / 2.0)
        on_frac_density = on_frac * min(1.0, _burst_mid / _windows_per_epoch)

    # Candidates_BF tests, per full step: C(m,2) synchronous tuples + m*m per NONZERO lag
    # bucket (every ordered (current, history) pair, self-pairs included). This is the
    # denominator the sweep's effective density (bf_correlated / bf_tested) divides by,
    # so the generator must target it -- NOT the (unordered, one-lag-per-pair) count.
    tested_per_step = total_pairs + m * m * (L_test - 1)
    # exact tested-tuple count Candidates_BF accumulates over the first n_eval windows
    # (lag bucket k is only available from window k onward): C(m,2)*n_eval synchronous
    # + m^2 * sum_{k=1}^{L_test-1} (n_eval - k) lagged. Used when verify_bf is off.
    _lag_terms = sum(max(0, n_eval - k) for k in range(1, L_test))
    tested_total_est = float(total_pairs) * n_eval + float(m) * m * _lag_terms

    # ---- solve (n_groups, group_size) from the analytic density formula ----
    # Canonical (unordered, deduped) planted tuples per full step ~= ng * C(g,2) * on_frac,
    # so d_exact(ng, g) = ng * g*(g-1)/2 * on_frac / tested_per_step, valid for ng*g <= m.
    # Search the (ng, g) grid: ng distinct groups of g series each, the rest uncorrelated.
    # g=2 with small ng recovers the sparse "disjoint pairs" regime; a big single group
    # (ng=1, g up to m) recovers the dense regime -- one mechanism, no special cases.
    def _d_exact(ng, gg):
        if gg < 2 or ng < 1 or ng * gg > m:
            return 0.0
        return ng * gg * (gg - 1) / 2.0 * on_frac_density / tested_per_step

    # (2026-09-18) `group_size` pins g (= degree + 1) and lets the density pick n_groups only; the
    # reachable density is then bounded by ng <= m // g, and the caller is told (RuntimeError) when the
    # requested density needs more groups than the series allow.
    if group_size is not None:
        group_size = int(group_size)
        if group_size < 2 or group_size > m:
            raise ValueError("group_size must be in [2, m]")

    def _solve_ng_g(target, model_gain=1.0):
        want = target / max(model_gain, 1e-6)
        best, best_err = (1, 2), float("inf")
        if group_size is not None:
            gg = group_size
            for ng in range(1, m // gg + 1):
                err = abs(_d_exact(ng, gg) - want)
                if err < best_err:
                    best, best_err = (ng, gg), err
            if _d_exact(m // gg, gg) < want * 0.9:
                raise RuntimeError(
                    f"density {target:.4f} is not reachable with group_size={gg}: even m//g={m // gg} groups give "
                    f"{_d_exact(m // gg, gg):.4f}; lower the density or raise the degree")
            return best
        for gg in range(2, m + 1):
            for ng in range(1, m // gg + 1):
                err = abs(_d_exact(ng, gg) - want)
                if err < best_err:
                    best, best_err = (ng, gg), err
            if _d_exact(1, gg) > want and gg > 2:
                break   # larger g only overshoots further
        return best

    n_groups_target, g = _solve_ng_g(target_density)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    # ---- shared-signal smoothing width ----
    # The shared driver c_group is white. A driver whose windowed autocorrelation
    # stayed >= corr_threshold across a lag *band* of width b*step would necessarily
    # have a correlation length ~b*step, and at the window / step sizes this experiment
    # uses (w ~ 8*step) that makes two INDEPENDENT group drivers windowed-correlated
    # well above threshold -- verified empirically, a hard geometric incompatibility,
    # not a tuning issue. So each correlated pair is planted at exactly ONE lag bucket
    # (|o_i - o_j|), and `lag_band` instead controls how widely member offsets are
    # spread within a group (jitter over b consecutive buckets) so different
    # within-group pairs sit at different lags spanning b buckets collectively.
    sigma_c = 0.0

    def _win_std(sig):
        """sqrt of the mean windowed variance (in the evaluation space) -- the scale the
        loading->correlation map is defined against."""
        s = np.asarray(sig, dtype=np.float64)
        if preprocess:
            s = np.diff(s)
        starts = np.arange(0, max(1, len(s) - w), max(step, (len(s) - w) // 64 or 1))
        vs = [s[t:t + w].var() for t in starts if t + w <= len(s)]
        return math.sqrt(max(1e-12, float(np.mean(vs)))) if vs else 1.0

    def _build(ng_local, g_local):
        _rng = np.random.default_rng((int(seed) & 0xFFFFFFFF) ^ 0xD3117)
        perm = _rng.permutation(m)
        ng_local = max(1, min(int(ng_local), m // max(2, int(g_local))))
        groups = [perm[c * g_local:(c + 1) * g_local].tolist() for c in range(ng_local)]
        groups = [grp for grp in groups if len(grp) >= 2]
        n_groups_l = len(groups)
        states, events = _build_group_schedule(n_groups_l, n_epochs, corr_sign, duty, seed)

        # step-align epoch boundaries so every transition falls on the window grid
        epoch_len = max(step, (n // n_epochs // step) * step)
        o_hi = max(0, L_test - b)   # so O_group + (b-1) stays <= L_test - 1

        # per-member idiosyncratic noise, normalised to unit windowed std in the eval space
        eps = _gen_base_series(m, n, base_proc, _rng).astype(np.float64)
        for si in range(m):
            eps[si] /= _win_std(eps[si])

        a_lo = _a_from_r(corr_threshold + corr_margin)
        a_hi = _a_from_r(r_max)

        def _draw_loadings(k):
            if loading_skew == "high":
                a = a_hi - _rng.beta(2.0, 5.0, size=k) * (a_hi - a_lo)
            elif loading_skew == "low":
                a = a_hi - _rng.beta(5.0, 2.0, size=k) * (a_hi - a_lo)
            else:
                a = _rng.uniform(a_lo, a_hi, size=k)
            n_near = int(round(near_threshold_fraction * k))
            if n_near > 0:
                a[_rng.choice(k, size=n_near, replace=False)] = a_lo
            return a

        offsets, loadings, group_of, c_signals, burst_k_by_group = {}, {}, {}, [], {}
        for gi, grp in enumerate(groups):
            c = _gen_shared_driver_signal(n, sigma_c, _rng)
            c /= _win_std(c)                       # unit windowed std, same as eps
            c_signals.append(c)
            O_group = int(_rng.integers(0, o_hi + 1))
            a = _draw_loadings(len(grp))
            for mi_local, si in enumerate(grp):
                jit = int(_rng.integers(0, b)) if b > 1 else 0
                offsets[si] = max(0, min(L_test - 1, O_group + jit))
                loadings[si] = float(a[mi_local])
                group_of[si] = gi
            if burst_length_windows is not None:
                if np.isscalar(burst_length_windows):
                    k = int(burst_length_windows)
                else:
                    lo_k, hi_k = int(burst_length_windows[0]), int(burst_length_windows[1])
                    k = int(_rng.integers(lo_k, hi_k + 1))
                burst_k_by_group[gi] = max(1, k)

        # per-group sample-level state timeline. Transitions are SHARP (state changes
        # exactly on a step-aligned epoch boundary) so the monitor event log is exact;
        # windows that straddle a transition are inherently ambiguous and are excluded
        # from the per-window GT / metric on both sides (see _verify). In burst mode, an
        # ON epoch is correlated for only a short burst_k-window span centered inside the
        # epoch (rest of the epoch OFF) instead of the whole epoch.
        s_samp_by_group = {}
        for gi in range(len(groups)):
            ss = np.empty(n, dtype=np.int8)
            for e_idx in range(n_epochs):
                lo = e_idx * epoch_len
                hi = n if e_idx == n_epochs - 1 else (e_idx + 1) * epoch_len
                s = int(states[gi, e_idx])
                if s != 0 and burst_length_windows is not None:
                    k = burst_k_by_group[gi]
                    span = min(w + (k - 1) * step, hi - lo)
                    # lo is step-aligned (epoch_len is a multiple of step); the centering
                    # offset must stay a multiple of step too, or the burst can land off
                    # the window grid entirely and contribute zero evaluable windows.
                    center_off = (max(0, (hi - lo - span) // 2) // step) * step
                    b_start = lo + center_off
                    ss[lo:hi] = 0
                    ss[b_start:b_start + span] = s
                else:
                    ss[lo:hi] = s
            s_samp_by_group[gi] = ss

        # ungrouped series: plain base process
        X = _gen_base_series(m, n, base_proc, _rng).astype(np.float64)
        for si in range(m):
            gi = group_of.get(si)
            if gi is None:
                continue
            o = offsets[si] * step        # offsets are in lag-BUCKET units
            csig = c_signals[gi]
            shifted = np.empty(n, dtype=np.float64)
            shifted[:o] = csig[0]
            shifted[o:] = csig[:n - o]
            comp = loadings[si] * shifted
            ss = s_samp_by_group[gi].astype(np.float64)
            X[si] = ss * comp + eps[si]
        return (X, groups, group_of, offsets, loadings, states, events, epoch_len,
                s_samp_by_group)

    # ---- analytic GT builder over the first `n_eval` evaluated windows ----
    # Window starts are recorded 1-based by Candidates_BF / CorrTrack, so we add 1.
    # Vectorised: for a group, the set of valid current-window starts depends only on the
    # pair's lag bucket (both members share the group state timeline), so it is computed
    # once per (group, lag) and reused across every pair at that lag.
    def _analytic_gt(groups, offsets, loadings, s_samp_by_group):
        starts = np.arange(n_eval, dtype=np.int64) * step        # 0-based current starts
        rows_blocks, corr_blocks = [], []
        for gi, grp in enumerate(groups):
            ss = s_samp_by_group[gi]
            if not np.any(ss != 0):
                continue
            chg = np.flatnonzero(np.diff(ss.astype(np.int64))) + 1   # state-change positions

            def _valid_for_lag(lag0):
                # current window [t1, t1+w), historical [t1 - lag0*step, ...); the whole
                # span must be one constant nonzero state.
                lo = starts - lag0 * step
                hi = starts + w
                ok = (lo >= 0) & (hi <= n)
                lo_c = np.clip(lo, 0, n - 1)
                s0 = ss[lo_c]
                ok &= s0 != 0
                if chg.size:
                    # a change AT lo_c itself doesn't break constancy of [lo_c, hi) (the
                    # transition already happened before this window) -- need the first
                    # change STRICTLY after lo_c, hence side="right", not "left".
                    nc = np.searchsorted(chg, lo_c, side="right")
                    has_change = (nc < chg.size) & (chg[np.minimum(nc, chg.size - 1)] < hi)
                    ok &= ~has_change
                return ok, s0

            lag_cache = {}
            # collect pairs by (cur, hist, lag0) then emit
            for ii in range(len(grp)):
                for jj in range(ii + 1, len(grp)):
                    si, sj = grp[ii], grp[jj]
                    off_diff = offsets[si] - offsets[sj]
                    cur, hist, lag0 = (si, sj, off_diff) if off_diff >= 0 else (sj, si, -off_diff)
                    if not (0 <= lag0 <= L_test - 1):
                        continue
                    r_ij = (loadings[si] * loadings[sj] /
                            math.sqrt((loadings[si] ** 2 + 1.0) * (loadings[sj] ** 2 + 1.0)))
                    if lag0 not in lag_cache:
                        lag_cache[lag0] = _valid_for_lag(lag0)
                    ok, s0 = lag_cache[lag0]
                    if not ok.any():
                        continue
                    t1 = starts[ok] + 1
                    t2 = starts[ok] - lag0 * step + 1
                    sgn = s0[ok]
                    # canonical: later-start first; here t1 >= t2 (lag0 >= 0), and on the
                    # t1 == t2 tie order by id. Emit as (cur, hist, t1, t2) then fix ties.
                    blk = np.empty((t1.size, 5), dtype=np.int64)
                    blk[:, 0] = cur; blk[:, 1] = hist
                    blk[:, 2] = t1; blk[:, 3] = t2
                    blk[:, 4] = w
                    if lag0 == 0 and cur > hist:
                        blk[:, [0, 1]] = blk[:, [1, 0]]
                    rows_blocks.append(blk)
                    corr_blocks.append(sgn.astype(np.float64) * r_ij)
        if not rows_blocks:
            return (np.empty((0, 5), dtype=np.int64), np.empty((0,), dtype=np.float64))
        return (np.vstack(rows_blocks), np.concatenate(corr_blocks))

    # ---- brute-force verification over exactly the first `n_eval` windows ----
    def _verify(X):
        from library_corrtrack_parallel import run_and_log_bruteforce
        import tempfile
        span = min(n, w + step * (n_eval - 1))
        ids = [f"s{i + 1}" for i in range(m)]
        # run_and_log_bruteforce expects (n_series + 1, n_time): row 0 is a 1..T index.
        data_row = np.vstack([np.arange(1, span + 1, dtype=np.float64),
                              X[:, :span].astype(np.float64)])
        base_config = dict(
            window_size=w, window_step=step, basic_window=step, n_lags=int(n_lags),
            corr_threshold=float(corr_threshold), neg_corr=(corr_sign != "pos"),
            exec="sequential", parallel_sketch=False, parallel_candidates=False,
            parallel_validation=False, max_workers=0, baseline_mode="bruteforce",
            monitor=True, track_min_dist=False, artifact_mode="final",
            save_only_required_artifacts=True, save_maxlag_artifacts=False,
            verbose=False, testing=False, validation_metric="pearson",
            preprocess=bool(preprocess),
        )
        with tempfile.TemporaryDirectory() as td:
            rec, _rt, flags = run_and_log_bruteforce(
                "densgen_verify", data_row, ids, base_config,
                os.path.join(td, "bf.csv"), metadata={"nodes": 0}, recall_by_window=True,
            )
        return rec, flags, span

    # ---- generate, verify, re-solve (ng, g) toward the target (deterministic) ----
    # Target analytic_density: canonical planted tuples / BF's own tested count. That is
    # the density the sweep sees when it uses the analytic GT in place of a full BF, and
    # it is smooth in (ng, g). verified_density (bf_correlated / bf_tested) is reported as
    # a cross-check but is noisier -- it also counts the transition-straddle windows BF
    # fires on, which the analytic GT (correctly) excludes as ambiguous. Switching (ng, g)
    # changes the transition structure so the model gain is not perfectly stable across
    # attempts; keep the closest-to-target attempt, not necessarily the last.
    from library_corrtrack_parallel import _canonicalize_rows, _rows_as_void_keys
    ng = n_groups_target
    tried = set()
    best = None
    for attempt in range(_max_correction_regens + 1):
        (X, groups, group_of, offsets, loadings, states, events, epoch_len,
         s_samp_by_group) = _build(ng, g)
        gt_rows, gt_corrs = _analytic_gt(groups, offsets, loadings, s_samp_by_group)
        if verify_bf:
            rec, flags, vspan = _verify(X)
            bf_tested = rec.get("tested") or 0
            bf_corr = rec.get("correlated") or 0
            verified_density = (bf_corr / bf_tested) if bf_tested else 0.0
            analytic_density = (len(gt_rows) / bf_tested) if bf_tested else 0.0
        else:
            rec, flags, vspan = None, None, min(n, w + step * (n_eval - 1))
            verified_density = float("nan")
            analytic_density = (len(gt_rows) / tested_total_est) if tested_total_est else 0.0
        rel = abs(analytic_density - target_density) / max(target_density, 1e-12)
        snap = dict(X=X, groups=groups, group_of=group_of, offsets=offsets,
                    loadings=loadings, states=states, events=events, epoch_len=epoch_len,
                    s_samp_by_group=s_samp_by_group, gt_rows=gt_rows, gt_corrs=gt_corrs,
                    rec=rec, flags=flags, vspan=vspan, verified_density=verified_density,
                    analytic_density=analytic_density, rel=rel, g=g, ng=len(groups))
        if best is None or rel < best["rel"]:
            best = snap
        tried.add((ng, g))
        if rel <= tolerance or attempt == _max_correction_regens or analytic_density <= 0.0:
            break
        model_gain = analytic_density / max(_d_exact(len(groups), g), 1e-12)
        ng_new, g_new = _solve_ng_g(target_density, model_gain=model_gain)
        if (ng_new, g_new) in tried:
            break
        ng, g = ng_new, g_new

    # effective degree: distinct partners per series among the verified correlated tuples (mean over the
    # series that have at least one), the quantity a fixed `group_size` promises as group_size - 1
    def _effective_degree(rows):
        rows = np.asarray(rows)
        if rows.size == 0:
            return 0.0
        pairs = {(int(min(a, b)), int(max(a, b))) for a, b in zip(rows[:, 0], rows[:, 1])}
        deg = {}
        for a, b in pairs:
            deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
        return float(np.mean(list(deg.values()))) if deg else 0.0
    for snap_ in ([best] if best is not None else []):
        snap_["effective_degree"] = _effective_degree(snap_["gt_rows"])

    (X, groups, group_of, offsets, loadings, states, events, epoch_len, s_samp_by_group,
     gt_rows, gt_corrs, rec, flags, vspan, verified_density, analytic_density, g) = (
        best["X"], best["groups"], best["group_of"], best["offsets"], best["loadings"],
        best["states"], best["events"], best["epoch_len"], best["s_samp_by_group"],
        best["gt_rows"], best["gt_corrs"], best["rec"], best["flags"], best["vspan"],
        best["verified_density"], best["analytic_density"], best["g"])

    # A within-group BF row whose window span straddles a state transition is inherently
    # ambiguous (partly ON, partly OFF / opposite-sign): the analytic GT never emits those,
    # and they must not count against precision. Drop them from the BF set before comparing.
    # Cross-group rows are kept -- any that survive are real false positives.
    def _determinate_mask(rows):
        group_arr = np.full(m, -1, dtype=np.int64)
        for _s, _gi in group_of.items():
            group_arr[_s] = _gi
        s1 = rows[:, 0]; s2 = rows[:, 1]
        lo = np.clip(np.minimum(rows[:, 2], rows[:, 3]) - 1, 0, n - 1)
        hi = np.clip(np.maximum(rows[:, 2], rows[:, 3]) - 1 + w, 1, n)
        g1 = group_arr[s1]; g2 = group_arr[s2]
        same = (g1 == g2) & (g1 >= 0)
        keep = np.ones(len(rows), dtype=bool)
        for gi in np.unique(g1[same]) if same.any() else ():
            ss = s_samp_by_group[gi]
            chg = np.flatnonzero(np.diff(ss.astype(np.int64))) + 1
            idx = np.flatnonzero(same & (g1 == gi))
            if chg.size == 0:
                continue
            nc = np.searchsorted(chg, lo[idx], side="right")   # see _analytic_gt's note
            has_change = (nc < chg.size) & (chg[np.minimum(nc, chg.size - 1)] < hi[idx])
            keep[idx[has_change]] = False
        return keep

    gt_precision = gt_recall = float("nan")
    n_bf_ambiguous = 0
    if verify_bf:
        bf_rows, _bf_corrs = flags.correlated_rows()
        bf_rows_det = bf_rows[_determinate_mask(bf_rows)] if len(bf_rows) else bf_rows
        gk = set(_rows_as_void_keys(_canonicalize_rows(gt_rows)).tolist()) if len(gt_rows) else set()
        bk = set(_rows_as_void_keys(_canonicalize_rows(bf_rows_det)).tolist()) if len(bf_rows_det) else set()
        tp = len(gk & bk)
        gt_precision = tp / len(bk) if bk else 1.0    # planted / determinate BF-confirmed
        gt_recall = tp / len(gk) if gk else 1.0       # planted-and-confirmed / planted
        n_bf_ambiguous = int(len(bf_rows) - len(bf_rows_det))
        if os.environ.get("DENSGEN_DEBUG"):
            print(f"[densgen] ng={len(groups)} g={g} GT={len(gt_rows)} BF={len(bf_rows)} "
                  f"BFdet={len(bf_rows_det)} ambiguous={n_bf_ambiguous} "
                  f"gt_precision={gt_precision:.4f} gt_recall={gt_recall:.4f} "
                  f"analytic_d={analytic_density:.3e} verified_d={verified_density:.3e}")
        if gt_recall < 1.0 - float(near_threshold_fraction) - 0.05:
            raise RuntimeError(
                "make_density_targeted_dataset: brute force confirmed only "
                f"{gt_recall:.1%} of the planted (pair,lag,window) tuples "
                f"(expected >= {1.0 - near_threshold_fraction - 0.05:.1%}). The loading -> "
                "correlation construction is not clearing corr_threshold -- check base_proc "
                "/ preprocess / window_size or raise corr_margin."
            )
        if gt_precision < 0.95:
            raise RuntimeError(
                "make_density_targeted_dataset: brute force reported "
                f"{1.0 - gt_precision:.1%} correlated tuples that were not planted "
                "(spurious cross-group correlation). The shared driver is not windowed-"
                "decorrelated -- reduce sigma_c or check base_proc."
            )
    result = dict(
        X=X, groups=groups, offsets=offsets, loadings=loadings, states=states,
        events=events, epoch_len=epoch_len, gt_rows=gt_rows, gt_corrs=gt_corrs,
        analytic_density=analytic_density, verified_density=verified_density,
        gt_precision=gt_precision, gt_recall=gt_recall,
        n_bf_ambiguous=n_bf_ambiguous,
        g=g, n_groups=len(groups), verify_span=vspan,
    )

    ids = [f"s{i + 1}" for i in range(m)]
    data_col = np.zeros((n, m + 1), dtype=np.float32)
    data_col[:, 0] = np.arange(1, n + 1)
    data_col[:, 1:] = result["X"].T.astype(np.float32)

    meta = {
        "mode": "density_targeted",
        "m": m, "n": n, "window_size": w, "window_step": step, "n_lags": int(n_lags),
        "n_eval_steps": int(n_eval), "L_test": L_test,
        "target_density": float(target_density),
        "analytic_density": float(result["analytic_density"]),
        "verified_density": (None if math.isnan(result["verified_density"])
                             else float(result["verified_density"])),
        "gt_precision": (None if math.isnan(result["gt_precision"])
                         else float(result["gt_precision"])),
        "gt_recall": (None if math.isnan(result["gt_recall"])
                      else float(result["gt_recall"])),
        "n_bf_ambiguous_windows": int(result["n_bf_ambiguous"]),
        "group_size": int(result["g"]), "n_groups": int(result["n_groups"]),
        "effective_degree": float(best.get("effective_degree", 0.0)), "requested_group_size": group_size,
        "lag_band": b, "corr_sign": corr_sign, "n_epochs": int(n_epochs),
        "duty": float(duty), "on_frac": float(on_frac),
        "burst_length_windows": (list(burst_length_windows)
                                 if isinstance(burst_length_windows, (list, tuple))
                                 else burst_length_windows),
        "base_proc": base_proc, "preprocess": bool(preprocess),
        "r_max": float(r_max), "corr_margin": float(corr_margin),
        "near_threshold_fraction": float(near_threshold_fraction),
        "loading_skew": loading_skew, "seed": int(seed), "tolerance": float(tolerance),
        "verify_bf": bool(verify_bf),
        "schedule": result["states"].tolist(), "events": result["events"],
        "n_ground_truth_tuples": int(len(result["gt_rows"])),
    }
    out = {
        "data": data_col, "ids": ids,
        "gt_rows": result["gt_rows"], "gt_corrs": result["gt_corrs"],
        "analytic_density": result["analytic_density"],
        "verified_density": result["verified_density"],
        "gt_precision": result["gt_precision"], "gt_recall": result["gt_recall"],
        "group_size": result["g"], "n_groups": result["n_groups"], "effective_degree": float(best.get("effective_degree", 0.0)),
        "meta": meta,
    }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stat_tag = _stationarity_tag(base_proc)
        proc_tag = str((base_proc or {}).get("type", "ar1")).lower().replace(" ", "_")
        # density spans ~1e-4..3e-2 -> a 2-decimal tag would collide; use a compact
        # mantissa-exponent form (e.g. 5.0e-3 -> "d5p0em3") so nearby targets stay distinct
        _mant, _exp = f"{float(target_density):.1e}".split("e")
        d_tag = f"{_mant.replace('.', 'p')}e{int(_exp)}".replace("-", "m")
        pp_tag = "diff" if preprocess else "raw"
        stem = (f"densgen_{stat_tag}_{proc_tag}_{pp_tag}_m{m}_w{w}_s{step}_lag{n_lags}_"
                f"d{d_tag}_sign{corr_sign}_lb{b}_e{n_epochs}_"
                f"duty{_format_rate(duty)}_thr{str(corr_threshold).replace('.', 'p')}_seed{seed}")
        np.savez_compressed(os.path.join(save_dir, f"{stem}.npz"), data_col)
        np.savez_compressed(os.path.join(save_dir, f"{stem}_gt.npz"),
                            rows=result["gt_rows"], corrs=result["gt_corrs"])
        with open(os.path.join(save_dir, f"{stem}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        out["stem"] = stem
        out["meta_json"] = os.path.join(save_dir, f"{stem}_meta.json")
    return out


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
        type=_num_templates_arg_type,
        default=4,
        help="How many distinct base templates to use, or 'auto' to size to pair_goal "
        "(one never-reused template per placement -- avoids template-reuse collisions).",
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
