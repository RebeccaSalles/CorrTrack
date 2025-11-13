import os
import json
import math
import csv
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def _format_rate(z: float) -> str:
    return f"{z:.2f}".replace(".", "p")


def _stationarity_tag(base_proc: Optional[Dict[str, Any]]) -> str:
    if not base_proc:
        return "stat"
    t = str(base_proc.get("type", "ar1")).lower()
    if t in ("ar1", "wn", "white", "white_noise", "white-noise"):
        return "stat"
    if t in ("rw", "randomwalk", "random_walk", "random-walk"):
        return "nonstat"
    return "stat"


def _build_stem(
    m: int,
    n: int,
    w: int,
    s: int,
    z: float,
    corr_sign: str,
    threshold: float,
    max_lag: int,
    base_proc: Optional[Dict[str, Any]],
) -> str:
    stat_tag = _stationarity_tag(base_proc)
    rate_tag = _format_rate(z)
    sign_tag = corr_sign.lower()
    stem = (
        f"synt_{stat_tag}_corr{rate_tag}_m{m}_w{w}_s{s}_"
        f"sign{sign_tag}_thr{str(threshold).replace('.', 'p')}_lag{max_lag}"
    )
    return stem


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
        return X

    # Default AR(1)
    phi = float(base_proc.get("phi", 0.6))
    eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
    X[:, 0] = eps[:, 0]
    for t in range(1, n):
        X[:, t] = phi * X[:, t - 1] + eps[:, t]
    return X


def _choose_sign(corr_sign: str, rng: np.random.Generator) -> int:
    c = corr_sign.lower()
    if c == "pos":
        return +1
    if c == "neg":
        return -1
    return +1 if rng.random() < 0.5 else -1


def _pearson_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation on raw data (no prior normalization)."""
    r = np.corrcoef(x, y)[0, 1]
    if r > 1.0:
        r = 1.0
    if r < -1.0:
        r = -1.0
    return float(r)


def _compute_z_max(m: int, n: int, w: int, s: int) -> Tuple[float, int]:
    """
    Compute theoretical z_max under physical non-overlap (ignoring lag constraints),
    and the max number of non-overlapping windows across all series.
    """
    W_s = (n - w) // s + 1
    if W_s <= 0 or m <= 0:
        return 0.0, 0

    g_step = math.ceil(w / s)  # min grid index spacing to avoid overlap
    max_windows_per_series = (W_s - 1) // g_step + 1
    max_windows_total = m * max_windows_per_series
    z_max = max_windows_total / (m * W_s)
    return z_max, max_windows_total


def _make_nonoverlap_slots(
    m: int, n: int, w: int, s: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """
    Return a list of (series_index, start) giving all physically non-overlapping
    window starts across all series, using grid step ceil(w/s) and a random
    offset per series for variability.
    """
    g_step = math.ceil(w / s)
    slots: List[Tuple[int, int]] = []
    for i in range(m):
        # random offset in grid units for this series
        offset_g = int(rng.integers(0, g_step))
        start = offset_g * s
        while start + w <= n:
            slots.append((i, start))
            start += g_step * s
    return slots


def make_corr_dataset(
    save_dir: str,
    m: int,
    n: int,
    z: float,
    w: int,
    s: int = 1,
    threshold: float = 0.7,
    corr_sign: str = "pos",  # "pos" | "neg" | "both"
    base_proc: Optional[Dict[str, Any]] = None,
    nonoverlap: bool = True,
    max_lag: int = 0,
    lag_step: Optional[int] = None,
    seed: Optional[int] = 7,
) -> Dict[str, str]:
    """
    Generate m synthetic time series of length n with approximately z fraction of windows
    participating in high Pearson correlation pairs (|r| >= threshold).

    Lags are counted BACK:
      time2 = time1 - lag

    When nonoverlap=True (default), physical non-overlap is enforced:
      no sample index belongs to more than one injected window in any series.
    In this mode, we:
      - compute z_max (theoretical upper bound under non-overlap),
      - clamp the effective z to min(z, z_max),
      - systematically fill non-overlapping slots to get as close as possible
        to this effective z, with |r| >= threshold guarantee.

    When nonoverlap=False (allow-overlap), we fall back to a random injection
    strategy; z is best-effort and z_max is less meaningful.
    """
    assert 0.0 <= z <= 1.0, "z must be in [0,1]"
    assert w > 0 and n > w, "n must be > w"
    assert s >= 1, "stride s must be >= 1"
    if lag_step is None:
        lag_step = s
    assert lag_step >= 1, "lag_step must be >= 1"

    rng = np.random.default_rng(seed)

    # Base data
    X = _gen_base_series(m, n, base_proc, rng)  # shape (m, n)

    # Sliding windows
    W_s = (n - w) // s + 1

    # Precompute lag grid for consistency (though systematic pairing doesn't force lags)
    if max_lag <= 0:
        lag_allowed = lambda lag: True  # no restriction if max_lag <= 0
    else:
        def lag_allowed(lag: int) -> bool:
            if abs(lag) > max_lag:
                return False
            if lag % lag_step != 0:
                return False
            return True

    # ------------------------------------------------------------------
    # NON-OVERLAP MODE (Option A, with z_max)
    # ------------------------------------------------------------------
    if nonoverlap:
        z_max, max_windows_total = _compute_z_max(m, n, w, s)
        # requested total windows in correlations (each pair uses 2)
        target_windows = int(round(z * m * W_s))
        # clamp to physical capacity
        target_windows_eff = min(target_windows, max_windows_total)
        # ensure even number (pairs use 2 windows)
        if target_windows_eff % 2 == 1:
            target_windows_eff -= 1
        if target_windows_eff < 2:
            target_windows_eff = 0

        # systematic non-overlapping slots; keep all and pick as many as needed
        slots = _make_nonoverlap_slots(m, n, w, s, rng)

        # Build adjacency lists of lag-compatible slot indices.
        adjacency: List[List[int]] = [[] for _ in range(len(slots))]
        for idx in range(len(slots)):
            start_i = slots[idx][1]
            for jdx in range(idx + 1, len(slots)):
                start_j = slots[jdx][1]
                if lag_allowed(start_i - start_j):
                    adjacency[idx].append(jdx)
                    adjacency[jdx].append(idx)

        max_pairs_target = min(target_windows_eff // 2, len(slots) // 2)
        pairs_idx: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # Deterministic-but-randomized tie-breaking to keep reproducibility per seed.
        rank = list(range(len(slots)))
        rng.shuffle(rank)
        order_rank = {node: pos for pos, node in enumerate(rank)}

        available = set(range(len(slots)))

        def available_degree(node: int) -> int:
            return sum(1 for neigh in adjacency[node] if neigh in available)

        while available and len(pairs_idx) < max_pairs_target:
            # pick the most constrained slot first
            idx = min(
                available,
                key=lambda node: (available_degree(node), order_rank[node]),
            )
            neighbors = [neigh for neigh in adjacency[idx] if neigh in available]
            if not neighbors:
                available.discard(idx)
                continue
            partner = min(
                neighbors,
                key=lambda node: (available_degree(node), order_rank[node]),
            )

            available.discard(idx)
            available.discard(partner)
            pairs_idx.append((slots[idx], slots[partner]))

        correlated_rows: List[Tuple[str, str, int, int, float]] = []
        used = np.zeros((m, n), dtype=np.uint8)  # physical occupancy

        for (i1, start1), (i2, start2) in pairs_idx:
            # Respect physical non-overlap (should already be guaranteed by construction,
            # but we double-check in case of any edge case).
            if np.any(used[i1, start1:start1 + w]) or np.any(
                used[i2, start2:start2 + w]
            ):
                continue

            # Compute lag according to counted-back convention: time2 = time1 - lag
            # Here we choose time1 = start1, time2 = start2 (both 0-based),
            # so lag = start1 - start2.
            lag = start1 - start2
            if not lag_allowed(lag):
                continue

            # Build template
            u = rng.normal(0.0, 1.0, size=w)
            u = u - u.mean()
            std_u = u.std()
            if std_u < 1e-12:
                continue
            u = u / std_u

            # Rejection sample to ensure |r| >= threshold on constructed windows
            max_inner_attempts = 20
            accepted = False
            for _ in range(max_inner_attempts):
                r_star = threshold + rng.random() * (1.0 - threshold)
                if r_star >= 1.0:
                    eps = np.zeros(w, dtype=np.float64)
                else:
                    sigma2 = 1.0 / (r_star**2) - 1.0
                    sigma2 = max(sigma2, 0.0)
                    eps = rng.normal(0.0, math.sqrt(sigma2), size=w)
                sign = _choose_sign(corr_sign, rng)
                xw = u.astype(np.float32)
                yw = (sign * u + eps).astype(np.float32)
                r = _pearson_raw(xw, yw)
                if abs(r) >= threshold:
                    accepted = True
                    break
            if not accepted:
                continue

            # Write into X
            X[i1, start1 : start1 + w] = xw
            X[i2, start2 : start2 + w] = yw

            # Mark occupancy
            used[i1, start1 : start1 + w] = 1
            used[i2, start2 : start2 + w] = 1

            # 1-based times; ids s1..sm
            correlated_rows.append(
                (f"s{i1+1}", f"s{i2+1}", start1 + 1, start2 + 1, r)
            )

        windows_used = 2 * len(correlated_rows)
        if W_s > 0 and m > 0:
            achieved_z = windows_used / (m * W_s)
        else:
            achieved_z = 0.0

        achieved_z_eff = min(achieved_z, z_max)

        stem = _build_stem(m, n, w, s, z, corr_sign, threshold, max_lag, base_proc)
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
            writer = csv.writer(f)
            writer.writerow(["id1", "id2", "time1", "time2", "corr"])
            for row in correlated_rows:
                writer.writerow(row)

        meta = {
            "m": m,
            "n": n,
            "w": w,
            "s": s,
            "threshold": threshold,
            "corr_sign": corr_sign,
            "max_lag": max_lag,
            "lag_step": lag_step,
            "target_z": z,
            "z_max_nonoverlap": z_max,
            "achieved_z": achieved_z,
            "windows_target": target_windows,
            "windows_target_eff": target_windows_eff,
            "windows_used": windows_used,
            "n_pairs": len(correlated_rows),
            "nonoverlap": True,
        }
        meta_path = os.path.join(save_dir, f"{stem}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        params = {
            "seed": seed,
            "base_proc": base_proc
            if base_proc is not None
            else {"type": "ar1", "phi": 0.6, "sigma": 1.0},
        }
        params_path = os.path.join(save_dir, f"{stem}_params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        return {
            "data_npz": data_path,
            "correlated_csv": corr_csv_path,
            "meta_json": meta_path,
            "params_json": params_path,
            "stem": stem,
        }

    # ------------------------------------------------------------------
    # OVERLAP-ALLOWED MODE (best-effort, no strong z guarantee)
    # ------------------------------------------------------------------
    # fall back to simpler random injection when nonoverlap=False
    # (not the mode you care about for tight z control)
    W_s = (n - w) // s + 1
    K_target = int(math.ceil(z * (m * W_s) / 2.0))

    correlated_rows: List[Tuple[str, str, int, int, float]] = []
    attempts = 0
    max_attempts = max(5 * max(K_target, 1), 1000)

    while len(correlated_rows) < K_target and attempts < max_attempts:
        attempts += 1
        i1 = int(rng.integers(0, m))
        i2 = int(rng.integers(0, m))
        # choose grid index for first window
        gi = int(rng.integers(0, W_s))
        start1 = gi * s
        if start1 < 0 or start1 + w > n:
            continue

        # choose lag by sampling admissible lag values uniformly
        if max_lag <= 0:
            lag = 0
        else:
            # build lag grid once (could be optimized)
            L = [l for l in range(-max_lag, max_lag + 1, lag_step)]
            lag = int(rng.choice(L))
        start2 = start1 - lag
        if start2 < 0 or start2 + w > n:
            continue

        # build template
        u = rng.normal(0.0, 1.0, size=w)
        u = u - u.mean()
        std_u = u.std()
        if std_u < 1e-12:
            continue
        u = u / std_u

        max_inner_attempts = 20
        accepted = False
        for _ in range(max_inner_attempts):
            r_star = threshold + rng.random() * (1.0 - threshold)
            if r_star >= 1.0:
                eps = np.zeros(w, dtype=np.float64)
            else:
                sigma2 = 1.0 / (r_star**2) - 1.0
                sigma2 = max(sigma2, 0.0)
                eps = rng.normal(0.0, math.sqrt(sigma2), size=w)
            sign = _choose_sign(corr_sign, rng)
            xw = u.astype(np.float32)
            yw = (sign * u + eps).astype(np.float32)
            r = _pearson_raw(xw, yw)
            if abs(r) >= threshold:
                accepted = True
                break
        if not accepted:
            continue

        X[i1, start1 : start1 + w] = xw
        X[i2, start2 : start2 + w] = yw

        correlated_rows.append((f"s{i1+1}", f"s{i2+1}", start1 + 1, start2 + 1, r))

    windows_used = 2 * len(correlated_rows)
    if W_s > 0 and m > 0:
        achieved_z = windows_used / (m * W_s)
    else:
        achieved_z = 0.0

    # z_max is not meaningful in overlap mode; we set it to 1.0 as a placeholder
    z_max = 1.0

    stem = _build_stem(m, n, w, s, z, corr_sign, threshold, max_lag, base_proc)
    os.makedirs(save_dir, exist_ok=True)

    arr = np.zeros((n, m + 1), dtype=np.float32)
    arr[:, 0] = np.arange(1, n + 1, dtype=np.float32)
    arr[:, 1:] = X.T

    data_path = os.path.join(save_dir, f"{stem}.npz")
    np.savez_compressed(data_path, arr)

    corr_csv_path = os.path.join(save_dir, f"{stem}_correlated.csv")
    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id1", "id2", "time1", "time2", "corr"])
        for row in correlated_rows:
            writer.writerow(row)

    meta = {
        "m": m,
        "n": n,
        "w": w,
        "s": s,
        "threshold": threshold,
        "corr_sign": corr_sign,
        "max_lag": max_lag,
        "lag_step": lag_step,
        "target_z": z,
        "z_max_nonoverlap": z_max,
        "achieved_z": achieved_z,
        "windows_target": int(round(z * m * W_s)),
        "windows_used": windows_used,
        "n_pairs": len(correlated_rows),
        "nonoverlap": False,
        "attempts": attempts,
    }
    meta_path = os.path.join(save_dir, f"{stem}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    params = {
        "seed": seed,
        "base_proc": base_proc
        if base_proc is not None
        else {"type": "ar1", "phi": 0.6, "sigma": 1.0},
    }
    params_path = os.path.join(save_dir, f"{stem}_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    return {
        "data_npz": data_path,
        "correlated_csv": corr_csv_path,
        "meta_json": meta_path,
        "params_json": params_path,
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
        help="Target fraction of correlated windows (0..1).",
    )
    p.add_argument("--w", type=int, required=True, help="Window length.")
    p.add_argument(
        "--s",
        type=int,
        default=1,
        help="Stride/step between window starts (default: 1).",
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
        "--base-type",
        type=str,
        choices=["ar1", "rw", "wn"],
        default="ar1",
        help="Base process: ar1 (stationary), rw (nonstationary), wn (white noise).",
    )
    p.add_argument(
        "--phi",
        type=float,
        default=0.6,
        help="AR(1) phi (only for base-type=ar1).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise sigma for base process.",
    )
    p.add_argument(
        "--nonoverlap",
        action="store_true",
        help=("Enforce physical non-overlap of injected windows (default)."),
    )
    p.add_argument(
        "--allow-overlap",
        action="store_true",
        help=("Allow overlapping injected windows (sets nonoverlap=False)."),
    )
    p.add_argument(
        "--max-lag",
        type=int,
        default=0,
        help="Maximum absolute lag (default: 0).",
    )
    p.add_argument(
        "--lag-step",
        type=int,
        default=None,
        help="Lag grid step (default: s).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7).",
    )
    args = p.parse_args()

    base_proc = {"type": args.base_type, "sigma": args.sigma}
    if args.base_type == "ar1":
        base_proc["phi"] = args.phi

    # default: nonoverlap=True; allow-overlap overrides
    nonoverlap = True
    if args.allow_overlap:
        nonoverlap = False
    if args.nonoverlap:
        nonoverlap = True

    return dict(
        save_dir=args.save_dir,
        m=args.m,
        n=args.n,
        z=args.z,
        w=args.w,
        s=args.s,
        threshold=args.threshold,
        corr_sign=args.corr_sign,
        base_proc=base_proc,
        nonoverlap=nonoverlap,
        max_lag=args.max_lag,
        lag_step=args.lag_step,
        seed=args.seed,
    )


if __name__ == "__main__":
    kwargs = _parse_args()
    out = make_corr_dataset(**kwargs)
    print(json.dumps(out, indent=2))
