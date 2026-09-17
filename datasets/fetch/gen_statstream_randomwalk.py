"""StatStream's synthetic random walks (Zhu & Shasha, VLDB 2002, section 5).

    s_i(0) = 100,   s_i(t) = s_i(t-1) + (u_{i,t} - 0.5),   u ~ Uniform(0, 1) i.i.d.

Independent walks: every correlation found is spurious, which is exactly why the paper
calls this the cooperative case (high-correlation pairs abound because normalized random
walks are smooth, so a small-dimensional grid separates them well). ``--m`` and ``--T``
default to a 2k-series battery cell; the paper used up to 10,000 streams.

    python datasets/fetch/gen_statstream_randomwalk.py --m 2000 --T 20000 --seed 20260917
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import save_competitor_npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--name", default=None, help="output name (default statstream_rw_m<m>_T<T>)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    steps = rng.uniform(0.0, 1.0, size=(args.m, args.T)) - 0.5
    steps[:, 0] = 0.0
    data = 100.0 + np.cumsum(steps, axis=1)
    name = args.name or f"statstream_rw_m{args.m}_T{args.T}"
    save_competitor_npz(
        name, data, [f"rw{i:05d}" for i in range(args.m)],
        {"source": "StatStream VLDB 2002 section 5, formula reproduced", "generator": "gen_statstream_randomwalk.py",
         "m": args.m, "T": args.T, "seed": args.seed, "regime": "cooperative (smooth independent walks)"},
    )


if __name__ == "__main__":
    main()
