"""BRAID's two synthetic families (Sakurai, Papadimitriou, Faloutsos, SIGMOD 2005 / TKDD 2010
section 6.1), reproduced approximately and extended from a single lagged pair to ``m`` series.

The paper describes ``Sines`` as a mixture of sines (n = 32,768) and ``SpikeTrains`` as
periodic spike trains (period 6,500, n = 100,000), each used as a *pair* with a known lag to
test lag estimation; Sines is BRAID's zero-error case. It gives no further generative detail,
so this is a documented approximation, not a reproduction:

    Sines:        x_i(t) = sum_{k=1..3} a_ik sin(2 pi t / P_ik + phi_ik) + sigma eps
                  P drawn log-uniformly in [64, 4096]
    SpikeTrains:  x_i(t) = sum_j g((t - t_ij) / w_i) + sigma eps, spikes every ``period``
                  samples with jitter, Gaussian pulse of width w_i in [20, 60]

Series come in groups: the first ``--n-groups`` series are independent seeds, each followed by
``--copies`` lagged copies (lag drawn uniformly in [1, --max-lag]) plus fresh noise, so the
ground-truth lag of every planted pair is known and written to ``meta["planted_pairs"]``.

    python datasets/fetch/gen_braid_synthetic.py --family sines --m 2000 --T 32768
    python datasets/fetch/gen_braid_synthetic.py --family spiketrains --m 2000 --T 100000 --period 6500
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import save_competitor_npz


def _sines(rng, T):
    t = np.arange(T, dtype=np.float64)
    x = np.zeros(T)
    for _ in range(3):
        P = np.exp(rng.uniform(np.log(64.0), np.log(4096.0)))
        x += rng.uniform(0.5, 1.5) * np.sin(2.0 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
    return x


def _spikes(rng, T, period):
    x = np.zeros(T)
    w = rng.uniform(20.0, 60.0)
    t = np.arange(T, dtype=np.float64)
    start = rng.uniform(0, period)
    for centre in np.arange(start, T, period):
        c = centre + rng.normal(0.0, 0.02 * period)
        lo, hi = int(max(0, c - 5 * w)), int(min(T, c + 5 * w))
        if hi > lo:
            x[lo:hi] += np.exp(-0.5 * ((t[lo:hi] - c) / w) ** 2)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("sines", "spiketrains"), required=True)
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--T", type=int, default=None, help="default 32768 (sines) or 100000 (spiketrains)")
    ap.add_argument("--period", type=float, default=6500.0)
    ap.add_argument("--copies", type=int, default=1, help="lagged copies per seed series")
    ap.add_argument("--max-lag", type=int, default=168)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    T = args.T or (32768 if args.family == "sines" else 100000)
    rng = np.random.default_rng(args.seed)
    n_groups = int(np.ceil(args.m / (1 + args.copies)))
    data = np.empty((args.m, T))
    ids, pairs = [], []
    row = 0
    for g in range(n_groups):
        base = _sines(rng, T + args.max_lag) if args.family == "sines" else _spikes(rng, T + args.max_lag, args.period)
        data[row] = base[args.max_lag:] + args.noise * rng.normal(size=T)
        ids.append(f"g{g:05d}_seed")
        seed_row = row
        row += 1
        for c in range(args.copies):
            if row >= args.m:
                break
            lag = int(rng.integers(1, args.max_lag + 1))
            data[row] = base[args.max_lag - lag: args.max_lag - lag + T] + args.noise * rng.normal(size=T)
            ids.append(f"g{g:05d}_lag{lag}")
            pairs.append({"a": ids[seed_row], "b": ids[row], "lag": lag})
            row += 1
        if row >= args.m:
            break
    name = args.name or f"braid_{args.family}_m{args.m}_T{T}"
    save_competitor_npz(
        name, data, ids,
        {"source": "BRAID SIGMOD 2005 / TKDD 2010 section 6.1, approximated (paper gives no generator)",
         "generator": "gen_braid_synthetic.py", "family": args.family, "m": args.m, "T": T,
         "period": args.period if args.family == "spiketrains" else None, "copies": args.copies,
         "max_lag": args.max_lag, "noise": args.noise, "seed": args.seed, "planted_pairs": pairs,
         "regime": "cooperative, lag-by-construction (synthetic)"},
    )


if __name__ == "__main__":
    main()
