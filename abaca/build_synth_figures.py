"""Figures for the synthetic OFAT campaign (2026-09-28).

    python abaca/build_synth_figures.py <results-root> <figures-dir>

Writes, per base process, the four scaling curves (raw and differenced side by side) and the
per-threshold wall-clock bar chart at the largest m the threshold axis covers. Arms whose lag or
negative-correlation capability is ours rather than their authors' are hatched, so the figure says
who enabled what: the tier comes from the run's own supports_lags / supports_neg_corr field.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_results as sr

PROCS = [("ar1", "AR(1) base process"), ("rw", "random-walk base process")]
SPACES = [("raw", "raw levels"), ("differenced", "first differences")]
AXIS_TITLE = {"m": "number of series m", "L": "lagged windows L", "T": "correlation threshold T",
              "density": "target correlation density"}
LOGX = {"m": True, "L": False, "T": False, "density": True}
NAMES = [n for _, n in sr.ORDER]
# one fixed colour per arm across every figure of the study; CorrTrack's two backends share a hue
COLOR = {"bf_incr": "#4c78a8", "FilCorr": "#f58518", "TSUBASA": "#8c6d31", "BRAID": "#7b4173",
         "ThinBRAID": "#d67ab1", "CT-lsh": "#c03d3e", "CT-ham": "#e8927c", "ParCorr": "#2f8a57",
         "CSZ": "#8fbf6b", "StatStream": "#57a3c7", "CorrJoin": "#c8b44a"}
MARK = {"CT-lsh": "o", "CT-ham": "s"}
# provenance of the capability the cell exercises, as the runs record it
HATCH = {"native": "", "enabled_by_us": "//", "specified": "..", "not_available": "xx", None: ""}


def tier(cell, name):
    """Which capability this cell asks of the arm, and where that capability comes from."""
    a = cell["arms"].get(name)
    if not a:
        return None
    return a["supports_lags"] if cell["L"] > 1 else "native"


def scaling_figure(cells, proc, desc, path):
    fig, axes = plt.subplots(len(sr.AXES), 2, figsize=(12.5, 15), sharey="row")
    for i, axis in enumerate(sr.AXES):
        for j, (space, sdesc) in enumerate(SPACES):
            ax = axes[i][j]
            pts = sr.on_axis(cells, axis, proc, space)
            for name in NAMES:
                xs = [c[axis] for c in pts if c["arms"][name]]
                ys = [c["arms"][name]["speedup"] for c in pts if c["arms"][name]]
                if not xs:
                    continue
                ct = name.startswith("CT-")
                ax.plot(xs, ys, marker=MARK.get(name, "."), markersize=6 if ct else 4,
                        linewidth=2.2 if ct else 1.2, color=COLOR[name], label=name, zorder=3 if ct else 2)
            ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
            ax.set_yscale("log")
            if LOGX[axis]:
                ax.set_xscale("log")
                ax.set_xticks([c[axis] for c in pts]); ax.set_xticklabels([f"{c[axis]:g}" for c in pts])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # the decade ticks collide with the levels
            ax.set_xlabel(AXIS_TITLE[axis]); ax.grid(alpha=0.25, linewidth=0.6)
            if j == 0:
                ax.set_ylabel("speedup over brute force")
            ax.set_title(f"{AXIS_TITLE[axis]}, {sdesc}", fontsize=10)
    axes[0][1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"Synthetic scaling, {desc}: speedup over brute force (one factor at a time around "
                 f"m={sr.BASE['m']}, L={sr.BASE['L']}, T={sr.BASE['T']}, density={sr.BASE['density']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(path, dpi=140); plt.close(fig)


def threshold_bars(cells, proc, desc, path):
    """Wall-clock time per method against the threshold, at the largest m the T axis covers."""
    pool = [c for c in cells if c["proc"] == proc and all(c[k] == v for k, v in sr.BASE.items() if k != "T")]
    if not pool:
        return None
    m = max(c["m"] for c in pool)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    lo, hi = [], []
    for j, (space, sdesc) in enumerate(SPACES):
        ax = axes[j]
        pts = sorted([c for c in pool if c["space"] == space and c["m"] == m], key=lambda c: c["T"])
        width = 0.92 / (len(NAMES) + 1)
        for k, name in enumerate(["bruteforce"] + NAMES):
            xs, ys, hs = [], [], []
            for i, c in enumerate(pts):
                if name == "bruteforce":
                    xs.append(i + k * width); ys.append(c["bf_wall"]); hs.append("")
                    continue
                a = c["arms"][name]
                if not a:
                    continue
                xs.append(i + k * width); ys.append(a["wall"]); hs.append(HATCH.get(tier(c, name), ""))
            if not xs:
                continue
            lo += ys; hi += ys
            ax.bar(xs, ys, width=width, color="#222222" if name == "bruteforce" else COLOR[name],
                   hatch=hs[0] if hs else "", edgecolor="white", linewidth=0.4,
                   label=name if j == 1 else None)
        ax.set_yscale("log")
        ax.set_xticks([i + width * len(NAMES) / 2 for i in range(len(pts))])
        ax.set_xticklabels([f"{c['T']:g}" for c in pts])
        ax.set_xlabel("correlation threshold"); ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_title(sdesc, fontsize=11)
        if j == 0:
            ax.set_ylabel("wall-clock time (s, log scale)")
    for ax in axes:
        ax.set_ylim(min(lo) * 0.6, max(hi) * 1.4)
    handles = [Patch(facecolor="#222222", label="brute force")] + \
              [Patch(facecolor=COLOR[n], label=n) for n in NAMES] + \
              [Patch(facecolor="white", edgecolor="#555555", hatch="//", label="lagged search enabled by us"),
               Patch(facecolor="white", edgecolor="#555555", hatch="..", label="lagged search specified by the\nauthors, never evaluated there")]
    axes[1].legend(handles=handles, ncol=1, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"{desc}, m = {m}: wall-clock time per method against the correlation threshold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.87, 0.95))
    fig.savefig(path, dpi=140); plt.close(fig)
    return m


def main() -> None:
    root, figdir = sys.argv[1], sys.argv[2]
    os.makedirs(figdir, exist_ok=True)
    cells = sr.load(root)
    for proc, desc in PROCS:
        p = os.path.join(figdir, f"scaling_{proc}.png")
        scaling_figure(cells, proc, desc, p); print("wrote", p)
        p = os.path.join(figdir, f"threshold_bars_{proc}.png")
        m = threshold_bars(cells, proc, desc, p); print("wrote", p, f"(m={m})")


if __name__ == "__main__":
    main()
