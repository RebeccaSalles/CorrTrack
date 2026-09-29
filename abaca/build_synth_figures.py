"""Figures for the synthetic OFAT campaign (2026-09-28).

    python abaca/build_synth_figures.py <results-root> <figures-dir>

Two figures, both on raw levels, both comparing the two base processes side by side:

  synth_scaling.png         the four scaling curves (m, L, T, density), AR(1) against random walk
  synth_threshold_bars.png  wall-clock time per method by threshold at the largest m the T axis covers

(2026-09-28, user) First differences are left out of the figures and kept in the tables. Differencing
makes both processes stationary, and the differenced cells then track the raw AR(1) cells closely
(CorrTrack-LSH over the m axis: 8.30 / 13.66 / 19.38 / 24.57 raw AR(1), 8.02 / 13.60 / 20.47 / 24.41
differenced AR(1), 7.97 / 14.05 / 20.64 / 24.40 differenced random walk), so the interesting contrast
is stationary against nonstationary, which is the raw pair.

Arms whose lag capability is ours rather than their authors' are hatched, from the run's own
supports_lags field.
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
import method_style as ms
import synth_results as sr

PROCS = [("ar1", "AR(1): stationary in levels"),
         ("rw", "random walk: nonstationary, spurious correlation in levels")]
AXIS_TITLE = {"m": "number of series m", "L": "lagged windows L", "T": "correlation threshold T",
              "density": "effective correlation density"}
# the density axis is plotted at the density brute force measured, not the one the generator was asked
# for: a random walk in levels carries spurious correlation, so the two differ by up to 4.8x there
XVAL = {"density": lambda c: c["effective"]}
LOGX = {"m": True, "L": False, "T": False, "density": True}
SPACE = "raw"
RECALL_TARGET = 0.95


def tier(cell, name):
    """Which capability this cell asks of the arm, and where that capability comes from."""
    a = cell["arms"].get(name)
    if not a:
        return None
    return a["supports_lags"] if cell["L"] > 1 else "native"


def scaling_figure(cells, path):
    fig, axes = plt.subplots(len(sr.AXES), len(PROCS), figsize=(13, 15), sharey="row")
    for i, axis in enumerate(sr.AXES):
        for j, (proc, pdesc) in enumerate(PROCS):
            ax = axes[i][j]
            pts = sr.on_axis(cells, axis, proc, SPACE)
            xof = XVAL.get(axis, lambda c: c[axis])
            for name in ms.lead(ms.NAMES):
                pt = [c for c in pts if c["arms"][name]]
                if not pt:
                    continue
                ct = name.startswith("CorrTrack")
                ax.plot([xof(c) for c in pt], [c["arms"][name]["speedup"] for c in pt],
                        linewidth=2.2 if ct else 1.2, color=ms.COLOR[name], label=name, zorder=3 if ct else 2)
                # marker shape is the provenance of the capability the cell exercises, hollow when the
                # arm's recall in that cell misses the target: the same encoding as the campaign figures
                for c in pt:
                    a = c["arms"][name]
                    ax.plot(xof(c), a["speedup"], marker=ms.SHAPE[tier(c, name)], markersize=7 if ct else 5,
                            color=ms.COLOR[name], markerfacecolor=ms.COLOR[name] if (a["recall"] or 1) >= RECALL_TARGET else "white",
                            markeredgecolor=ms.COLOR[name], zorder=4 if ct else 2)
            ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
            ax.set_yscale("log")
            if LOGX[axis]:
                ax.set_xscale("log")
                ax.set_xticks([xof(c) for c in pts])
                ax.set_xticklabels([(f"{xof(c):.3g}" if axis == "density" else f"{xof(c):.0f}") for c in pts],
                                   fontsize=8)
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # the decade ticks collide with the levels
            ax.set_xlabel(AXIS_TITLE[axis]); ax.grid(alpha=0.25, linewidth=0.6)
            if j == 0:
                ax.set_ylabel("speedup over BF_vect\n(1.0 = brute force)")
            if i == 0:
                ax.set_title(pdesc, fontsize=11)
    handles = [Line2D([], [], color=ms.COLOR[n], marker="o", label=n) for n in ms.lead(ms.NAMES)]
    handles += [Line2D([], [], color="none", label=" ")]
    handles += [Line2D([], [], color="#555555", marker=ms.SHAPE[t], linestyle="none", label=ms.TIER_LABEL[t])
                for t in ("native", "enabled_by_us", "specified")]
    handles += [Line2D([], [], color="#555555", marker="o", linestyle="none", markerfacecolor="white",
                       label=f"hollow: recall below {RECALL_TARGET}")]
    axes[0][1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Synthetic scaling on raw levels: speedup over the cell's own BF_vect, one generated dataset per "
                 "point\none factor at a time around m=" + f"{sr.BASE['m']}, L={sr.BASE['L']}, T={sr.BASE['T']}, "
                 f"target density {sr.BASE['density']}; W=60, step=6, 991 windows, positive correlation only",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.87, 0.96))
    fig.savefig(path, dpi=140); plt.close(fig)


def threshold_bars(cells, path):
    """Wall-clock time per method against the threshold, at the largest m the T axis covers."""
    fig, axes = plt.subplots(1, len(PROCS), figsize=(15, 5.8), sharey=True)
    lo, hi, m = [], [], None
    for j, (proc, pdesc) in enumerate(PROCS):
        ax = axes[j]
        pool = [c for c in cells if c["proc"] == proc and c["space"] == SPACE
                and all(c[k] == v for k, v in sr.BASE.items() if k != "T")]
        m = max(c["m"] for c in pool)
        pts = sorted([c for c in pool if c["m"] == m], key=lambda c: c["T"])
        width = 0.92 / (len(ms.NAMES) + 1)
        for i, c in enumerate(pts):
            bars = [(ms.BF, c["bf_wall"], "")]
            bars += [(n, c["arms"][n]["wall"], ms.HATCH.get(tier(c, n), "")) for n in ms.NAMES if c["arms"][n]]
            bars.sort(key=lambda b: b[1])            # fastest first inside the group
            for k, (name, wall, hatch) in enumerate(bars):
                lo.append(wall); hi.append(wall)
                ax.bar(i + (k - len(bars) / 2) * width, wall, width=width, color=ms.COLOR[name],
                       hatch=hatch, edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(pts))); ax.set_xticklabels([f"{c['T']:g}" for c in pts])
        ax.set_xlabel("correlation threshold"); ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_title(pdesc, fontsize=11)
        if j == 0:
            ax.set_ylabel("wall-clock time (s, log scale)")
    for ax in axes:
        ax.set_ylim(min(lo) * 0.7, max(hi) * 1.3)
    handles = [Patch(facecolor=ms.COLOR[ms.BF], label=ms.BF + " (brute force)")]
    handles += [Patch(facecolor=ms.COLOR[n], label=n) for n in ms.lead(ms.NAMES)]
    handles += [Patch(facecolor="white", edgecolor="#555555", hatch=ms.HATCH["enabled_by_us"],
                      label="lag capability added by us"),
                Patch(facecolor="white", edgecolor="#555555", hatch=ms.HATCH["specified"],
                      label="lag capability specified,\nnever evaluated by the authors")]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"Raw levels, m = {m}: wall-clock time per method by correlation threshold, one generated dataset "
                 f"per bar\nbars sorted fastest first inside each threshold group", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.85, 0.93))
    fig.savefig(path, dpi=140); plt.close(fig)
    return m


def main() -> None:
    root, figdir = sys.argv[1], sys.argv[2]
    os.makedirs(figdir, exist_ok=True)
    cells = sr.load(root)
    p = os.path.join(figdir, "synth_scaling.png"); scaling_figure(cells, p); print("wrote", p)
    p = os.path.join(figdir, "synth_threshold_bars.png"); m = threshold_bars(cells, p); print("wrote", p, f"(m={m})")


if __name__ == "__main__":
    main()
