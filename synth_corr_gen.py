
import os
import json
import math
import csv
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def _format_rate(z: float) -> str:
    # e.g., 0.3 -> "0p30", 1.0 -> "1p00"
    return f"{z:.2f}".replace(".", "p")


def _stationarity_tag(base_proc: Optional[Dict[str, Any]]) -> str:
    # "stat" for AR(1) and White Noise; "nonstat" for Random Walk
    if not base_proc:
        return "stat"
    t = str(base_proc.get("type", "ar1")).lower()
    if t in ("ar1", "wn", "white", "white_noise", "white-noise"):
        return "stat"
    if t in ("rw", "randomwalk", "random_walk", "random-walk"):
        return "nonstat"
    # default to stat
    return "stat"


def _build_stem(
    m: int, n: int, w: int, s: int, z: float, corr_sign: str,
    threshold: float, max_lag: int, base_proc: Optional[Dict[str, Any]]
) -> str:
    stat_tag = _stationarity_tag(base_proc)
    rate_tag = _format_rate(z)
    sign_tag = corr_sign.lower()
    stem = f"synt_{stat_tag}_corr{rate_tag}_m{m}_w{w}_s{s}_sign{sign_tag}_thr{str(threshold).replace('.', 'p')}_lag{max_lag}"
    return stem


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _gen_base_series(m: int, n: int, base_proc: Optional[Dict[str, Any]], rng: np.random.Generator) -> np.ndarray:
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
        # Random walk: x_t = x_{t-1} + epsilon_t
        eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        X[:, 0] = eps[:, 0]
        for t in range(1, n):
            X[:, t] = X[:, t-1] + eps[:, t]
        return X

    # Default AR(1): x_t = phi * x_{t-1} + eps
    phi = float(base_proc.get("phi", 0.6))
    eps = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
    X[:, 0] = eps[:, 0]
    for t in range(1, n):
        X[:, t] = phi * X[:, t-1] + eps[:, t]
    return X


def _choose_sign(corr_sign: str, rng: np.random.Generator) -> int:
    c = corr_sign.lower()
    if c == "pos":
        return +1
    if c == "neg":
        return -1
    # "both"
    return +1 if rng.random() < 0.5 else -1


def _sample_free_index(mask: np.ndarray, gmin: int, gmax: int, rng: np.random.Generator) -> Optional[int]:
    """Return a free grid index within [gmin, gmax], or None if none available."""
    if gmin > gmax:
        return None
    sub = mask[gmin:gmax+1] == 0
    if not np.any(sub):
        return None
    choices = np.nonzero(sub)[0] + gmin
    return int(rng.choice(choices))


def _pearson_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation on raw data (no prior normalization)."""
    # np.corrcoef does centering/scaling internally for the covariance matrix
    r = np.corrcoef(x, y)[0, 1]
    # Guard against tiny numerical drift
    if r > 1.0: r = 1.0
    if r < -1.0: r = -1.0
    return float(r)


def make_corr_dataset(
    save_dir: str,
    m: int,
    n: int,
    z: float,
    w: int,
    s: int = 1,
    threshold: float = 0.7,
    corr_sign: str = "pos",           # "pos" | "neg" | "both"
    base_proc: Optional[Dict[str, Any]] = None,
    nonoverlap: bool = True,
    max_lag: int = 0,
    lag_step: Optional[int] = None,
    seed: Optional[int] = 7,
) -> Dict[str, str]:
    """
    Generate m synthetic time series of length n with approximately z fraction of windows
    participating in high Pearson correlation pairs. Lags are counted BACK:
      time2 = time1 - lag
    Only the data are saved in .npz; companion files are CSV/JSON.
    Returns dict of file paths.
    """
    assert 0.0 <= z <= 1.0, "z must be in [0,1]"
    assert w > 0 and n > w, "n must be > w"
    assert s >= 1, "stride s must be >= 1"
    if lag_step is None:
        lag_step = s
    assert lag_step >= 1, "lag_step must be >= 1"
    if max_lag % lag_step != 0:
        # we will only use lags that are multiples of lag_step anyway
        pass

    rng = np.random.default_rng(seed)

    # Base data
    X = _gen_base_series(m, n, base_proc, rng)  # shape (m, n)

    # Window grid and occupancy
    W_s = (n - w) // s + 1
    occ = np.zeros((m, W_s), dtype=np.uint8)  # 0=free, 1=used

    # Target number of pair injections
    K_target = int(math.ceil(z * (m * W_s) / 2.0))

    # Lag grid (multiples of lag_step), counted BACK convention (we still treat symmetry same)
    if max_lag <= 0:
        L = np.array([0], dtype=int)
    else:
        L = np.arange(-max_lag, max_lag + 1, lag_step, dtype=int)

    # Prepare collection of correlated pairs for CSV
    correlated_rows: List[Tuple[str, str, int, int, float]] = []  # (id1,id2,time1,time2,corr)

    attempts = 0
    max_attempts = max(5 * K_target, 1000)  # generous cap for feasibility

    while len(correlated_rows) < K_target and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m))
        lag = int(rng.choice(L))

        # ensure lag is multiple of s so starts stay on grid after shift
        if lag % s != 0:
            continue
        shift = lag // s

        # Feasible g_i so that g_j = g_i - shift lies in [0, W_s-1]
        gmin = max(0, +shift)  # if shift positive, need g_i >= shift
        gmax = min(W_s - 1, W_s - 1 + shift)  # if shift negative, gmax reduces

        gi = _sample_free_index(occ[i], gmin, gmax, rng) if nonoverlap else int(
            rng.integers(gmin, gmax + 1)) if gmin <= gmax else None
        if gi is None:
            continue
        gj = gi - shift
        if gj < 0 or gj >= W_s:
            continue
        if nonoverlap and occ[j, gj] == 1:
            continue

        start_i = gi * s
        start_j = start_i - lag  # counted-back convention
        if start_i < 0 or start_i + w > n:  # guard
            continue
        if start_j < 0 or start_j + w > n:
            continue

        # Build a template u (mean 0, std 1), then construct pair hitting r* in [threshold, 1]
        u = np.asarray(np.random.default_rng(rng.integers(0, 2**63-1)).normal(0.0, 1.0, size=w), dtype=np.float64)
        u = (u - u.mean())
        std_u = u.std()
        if std_u < 1e-12:
            continue
        u = u / std_u

        r_star = threshold + rng.random() * (1.0 - threshold)  # uniform in [threshold, 1)
        if r_star >= 1.0:
            eps = np.zeros(w, dtype=np.float64)
        else:
            sigma2 = 1.0 / (r_star ** 2) - 1.0
            eps = rng.normal(0.0, math.sqrt(max(sigma2, 0.0)), size=w)

        sign = _choose_sign(corr_sign, rng)
        xw = u
        yw = sign * u + eps

        # Overwrite raw windows into series
        X[i, start_i:start_i + w] = xw.astype(np.float32)
        X[j, start_j:start_j + w] = yw.astype(np.float32)

        # Mark occupancy
        occ[i, gi] = 1
        occ[j, gj] = 1

        # Compute achieved raw Pearson
        r = _pearson_raw(
            X[i, start_i:start_i + w],
            X[j, start_j:start_j + w]
        )

        # Record 1-based times; ids as "s1".."sm"
        correlated_rows.append((f"s{i+1}", f"s{j+1}", start_i + 1, start_j + 1, r))

    # Prepare outputs and save
    stem = _build_stem(m, n, w, s, z, corr_sign, threshold, max_lag, base_proc)
    os.makedirs(save_dir, exist_ok=True)

    # Data .npz: array shape (n, m+1): first col = index 1..n; then S1..Sm
    arr = np.zeros((n, m + 1), dtype=np.float32)
    arr[:, 0] = np.arange(1, n + 1, dtype=np.float32)
    arr[:, 1:] = X.T  # each series as a column

    data_path = os.path.join(save_dir, f"{stem}.npz")
    np.savez_compressed(data_path, arr)

    # Correlated pairs CSV
    corr_csv_path = os.path.join(save_dir, f"{stem}_correlated.csv")
    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id1", "id2", "time1", "time2", "corr"])
        for row in correlated_rows:
            writer.writerow(row)

    # Meta JSON
    achieved_z = (2.0 * len(correlated_rows)) / (m * ((n - w) // s + 1)) if ((n - w) // s + 1) > 0 else 0.0
    meta = {
        "m": m, "n": n, "w": w, "s": s, "threshold": threshold, "corr_sign": corr_sign,
        "max_lag": max_lag, "lag_step": lag_step, "target_z": z, "achieved_z": achieved_z,
        "n_pairs": len(correlated_rows), "attempts": attempts
    }
    meta_path = os.path.join(save_dir, f"{stem}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Params JSON (exact base_proc and seed etc.)
    params = {
        "seed": seed,
        "base_proc": base_proc if base_proc is not None else {"type": "ar1", "phi": 0.6, "sigma": 1.0}
    }
    params_path = os.path.join(save_dir, f"{stem}_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    return {
        "data_npz": data_path,
        "correlated_csv": corr_csv_path,
        "meta_json": meta_path,
        "params_json": params_path,
        "stem": stem
    }


def example_run():
    """Small runnable example with modest sizes."""
    out = make_corr_dataset(
        save_dir=".",  # current directory
        m=20,
        n=5000,
        z=0.25,
        w=128,
        s=16,
        threshold=0.75,
        corr_sign="both",
        base_proc={"type": "ar1", "phi": 0.6, "sigma": 1.0},
        nonoverlap=True,
        max_lag=64,
        lag_step=16,
        seed=42,
    )
    return out



def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Synthetic correlated time series generator (NPZ + CSV/JSON artifacts).")
    p.add_argument("--save-dir", type=str, default=".", help="Folder to write outputs.")
    p.add_argument("--m", type=int, required=True, help="Number of time series.")
    p.add_argument("--n", type=int, required=True, help="Length of each time series.")
    p.add_argument("--z", type=float, required=True, help="Target fraction of correlated windows (0..1).")
    p.add_argument("--w", type=int, required=True, help="Window length.")
    p.add_argument("--s", type=int, default=1, help="Stride/step between window starts (default: 1).")
    p.add_argument("--threshold", type=float, default=0.7, help="Minimum Pearson correlation threshold (default: 0.7).")
    p.add_argument("--corr-sign", type=str, choices=["pos","neg","both"], default="pos",
                   help="Correlation sign to inject: pos, neg, or both (default: pos).")
    p.add_argument("--base-type", type=str, choices=["ar1","rw","wn"], default="ar1",
                   help="Base process: ar1 (stationary), rw (nonstationary), wn (white noise).")
    p.add_argument("--phi", type=float, default=0.6, help="AR(1) phi (only for base-type=ar1).")
    p.add_argument("--sigma", type=float, default=1.0, help="Noise sigma for base process.")
    p.add_argument("--nonoverlap", action="store_true", help="Enforce non-overlap at the grid level (default: true).")
    p.add_argument("--allow-overlap", action="store_true",
                   help="Allow reusing grid starts (overrides --nonoverlap).")
    p.add_argument("--max-lag", type=int, default=0, help="Maximum absolute lag (default: 0).")
    p.add_argument("--lag-step", type=int, default=None, help="Lag grid step (default: s).")
    p.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    args = p.parse_args()

    # Compose base_proc
    base_proc = {"type": args.base_type, "sigma": args.sigma}
    if args.base_type == "ar1":
        base_proc["phi"] = args.phi

    # nonoverlap logic: default True unless allow_overlap set
    nonoverlap = True
    if args.allow_overlap:
        nonoverlap = False
    elif args.nonoverlap:
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
