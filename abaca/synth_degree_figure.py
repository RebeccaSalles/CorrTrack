"""Figure of the fixed-degree m-sweep (2026-10-18, user): CorrTrack's search exponent under linear output.

A: wall clock against m (log-log) for CorrTrack-LSH at each threshold, BF_vect for reference, and dashed lines with
   the slope 1 + rho_eff that CorrTrack's own band sizing predicts, anchored at m = 500.
B: exponent against threshold: measured (wall and candidate search), the textbook 1 + rho of sign-hash LSH, and
   1 + rho_eff from compute_lsh_n_bands at 64 sign bits (and at 128 / 256 bits, with the measured ablation points
   when abaca/campaign_synth_degree.py --ablation has run).

    python abaca/synth_degree_figure.py ~/results/synth_degree docs/figures/synth_degree_scaling.png
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import candidate_kernels as ck
import method_style as ms

MS = [125, 250, 500, 1000, 2000, 4000]
TS = [0.7, 0.8, 0.9, 0.95]
L, OCC, RECALL = 6, 12.0, 0.95          # lagged windows, the tuned target occupancy, the recall target
T_RAMP = {0.7: "#eda29c", 0.8: "#da6a62", 0.9: "#c03d3e", 0.95: "#7d1f24"}    # one hue, darker = higher threshold
NV_STYLE = {64: ("-", "o"), 128: ("--", "s"), 256: (":", "^")}
CELL = re.compile(r"_m(\d+)_L6_T0p(\d+)_k55_.*?(?:_nv(\d+))?$")


def load(root):
    out = {}
    for p in glob.glob(os.path.join(root, "nway", "*", "nway.json")):
        m = CELL.search(os.path.basename(os.path.dirname(p)))
        if m:
            out[(float("0." + m.group(2)), int(m.group(1)), int(m.group(3) or 64))] = json.load(open(p))
    return out


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def rho(T):
    return math.log2(1.0 / (1.0 - math.acos(T) / math.pi))


def rho_eff(T, nv):
    """Growth exponent in N = m L of the n_bands CorrTrack sizes at threshold T from a pool of nv sign bits."""
    nb = [ck.compute_lsh_n_bands(ck.compute_lsh_band_width(m, L, OCC, nv), T, nv, RECALL, 1.0) for m in MS]
    return slope(np.array(MS) * L, nb)


def main(root, out_path):
    cells = load(root)
    wall = lambda T, a, nv=64, k="wall": [cells[(T, m, nv)]["arms"][a][k] for m in MS]   # noqa: E731
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # A: wall clock against m
    bf = np.median([wall(T, "bruteforce") for T in TS], axis=0)
    ax.plot(MS, bf, color=ms.COLOR[ms.BF], linewidth=1.6, marker="o", markersize=4)
    ax.annotate(f"BF_vect: {slope(MS, bf):.2f}", (MS[-1], bf[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=ms.COLOR[ms.BF], fontweight="bold")
    for T in TS:
        w = wall(T, "corrtrack")
        ax.plot(MS, w, color=T_RAMP[T], linewidth=2.2, marker="o", markersize=5)
        r = rho_eff(T, 64)
        anchor = MS.index(500)
        pred = [w[anchor] * (m / 500) ** (1 + r) for m in MS]
        ax.plot(MS, pred, color=T_RAMP[T], linewidth=1, linestyle=(0, (4, 3)), alpha=0.9)
        ax.annotate(f"T = {T}: {slope(MS, w):.2f}", (MS[-1], w[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=T_RAMP[T], fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(MS); ax.set_xticklabels([str(m) for m in MS])
    ax.set_xlim(100, 9000)
    ax.set_xlabel("number of series m (each with 55 correlated partners)")
    ax.set_ylabel("wall clock over the whole stream (s)")
    ax.grid(alpha=0.25, which="both", linewidth=0.5)
    ax.set_title("A. CorrTrack-LSH wall clock against m, per threshold\n(solid: measured; dashed: slope 1 + rho_eff "
                 "from its band sizing, through m = 500)", fontsize=10, loc="left")

    # B: exponent against threshold
    tt = np.linspace(0.65, 0.97, 60)
    # n_bands and band width are integers, so rho_eff is a staircase in T; a coarse grid shows its trend
    tc = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]
    bx.plot(tt, [1 + rho(T) for T in tt], color="#8a8a8a", linewidth=1.4, linestyle=(0, (5, 3)), label="textbook LSH, 1 + rho")
    nvs = sorted({k[2] for k in cells})
    for nv in sorted(set(nvs) | {64}):
        ls, mk = NV_STYLE.get(nv, ("-", "o"))
        bx.plot(tc, [1 + rho_eff(T, nv) for T in tc], color=ms.COLOR["CorrTrack-LSH"], linewidth=1.2, linestyle=ls,
                alpha=0.55 if nv != 64 else 0.9, label=f"CorrTrack band sizing, 1 + rho_eff, {nv} bits")
        pts = [T for T in TS if all((T, m, nv) in cells for m in MS)]
        if pts:
            bx.plot(pts, [slope(MS, wall(T, "corrtrack", nv)) for T in pts], linestyle="none", marker=mk, markersize=8,
                    color=ms.COLOR["CorrTrack-LSH"], markeredgecolor="white", label=f"measured CorrTrack-LSH wall, {nv} bits")
            if nv == 64:
                bx.plot(pts, [slope(MS, wall(T, "corrtrack", nv, "cand_time")) for T in pts], linestyle="none",
                        marker="o", markersize=7, markerfacecolor="white", markeredgecolor=ms.COLOR["CorrTrack-LSH"],
                        label="measured candidate search only, 64 bits")
    for a, name in (("bruteforce", ms.BF), ("bf_incremental", "BF_incr"), ("filcorr", "FilCorr")):
        bx.plot(TS, [slope(MS, wall(T, a)) for T in TS], linestyle="none", marker="D", markersize=5, color=ms.COLOR[name],
                label=f"measured {name}")
    bx.axhline(2.0, color="#cccccc", linewidth=0.8, zorder=0)
    bx.set_xlabel("correlation threshold T")
    bx.set_ylabel("exponent of m (least squares, m = 125 to 4000)")
    bx.set_ylim(1.0, 2.15); bx.set_xlim(0.65, 0.97)
    bx.grid(alpha=0.25, linewidth=0.5)
    bx.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    bx.set_title("B. Exponent of m against the threshold\n(exact arms stay at 2; CorrTrack falls with T)", fontsize=10, loc="left")

    fig.suptitle("Fixed-degree synthetic sweep: linear output, so the exponent is the search's "
                 "(AR(1), W = 60, step 6, 30 lags; recall >= 0.997 in every cell)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]), sys.argv[2])
