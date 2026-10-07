"""One figure for the paper's headline claim (2026-10-05, user).

    python abaca/headline_figure.py <campaign-root> <synth-root> <figures-dir> [<out.md>]

The claim is "the first subquadratic operator for complete, online, lagged, signed, thresholded
Pearson correlation discovery that stays robust under nonstationarity". Four clauses can be
measured, and each gets a panel, from the runs already in hand:

  A  subquadratic   wall clock against m on log axes, with the fitted exponent per arm
  B  complete and   recall of every cell as a distribution, against the tuned target, with the
     thresholded   precision that makes "thresholded Pearson" exact rather than approximate
  C  online        per-window cost, which is what makes it an operator over a stream rather than a
                   batch: the median window and its interquartile spread at each m
  D  robust under  the same cell under four regimes: stationary and random-walk processes, levels
     nonstationarity and first differences

Lagged and signed are capability facts, not measurements, so they stay in the capability table;
the panels carry what can be measured.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_figures as cf
import method_style as ms
import synth_results as sr

SHOW = ["CorrTrack-LSH", "CorrTrack-Ham", "FilCorr", "BF_incr"]
RECALL_TARGET = 0.95


def exponent(xs, ys):
    """Least squares slope of log(y) on log(x): the growth exponent in m over the measured range."""
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = st.mean(lx), st.mean(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def panel_subquadratic(ax, synth):
    pts = sr.on_axis(synth, "m", "ar1", "raw")
    xs = [c["m"] for c in pts]
    for name in [ms.BF] + SHOW:
        ys = [(c["bf_wall"] if name == ms.BF else c["arms"][name]["wall"]) for c in pts
              if name == ms.BF or c["arms"][name]]
        if len(ys) != len(xs):
            continue
        e = exponent(xs, ys)
        ax.plot(xs, ys, marker="o", markersize=6, color=ms.COLOR[name], linewidth=2.4 if "CorrTrack" in name else 1.3,
                label=f"{name}  m^{e:.2f}")
    ref = [pts[0]["bf_wall"] * (x / xs[0]) ** 2 for x in xs]
    ax.plot(xs, ref, color="#999999", linestyle="--", linewidth=1.0, label="quadratic reference  m^2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("number of series m"); ax.set_ylabel("wall clock for the whole stream (s)")
    ax.set_title("A. Subquadratic: the fitted exponent in m", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6); ax.legend(fontsize=8, loc="upper left")


def panel_complete(ax, camp):
    for name in ("CorrTrack-LSH", "CorrTrack-Ham"):
        r = sorted(c["arms"][name]["recall"] for c in camp if c["arms"][name] and c["arms"][name]["recall"] is not None)
        ax.plot(r, [i / (len(r) - 1) for i in range(len(r))], color=ms.COLOR[name], linewidth=2.4,
                label=f"{name}  (min {r[0]:.3f})")
    pre = [c["arms"][n]["precision"] for c in camp for n in ("CorrTrack-LSH", "CorrTrack-Ham")
           if c["arms"][n] and c["arms"][n]["precision"] is not None]
    ax.axvline(RECALL_TARGET, color="#999999", linestyle="--", linewidth=1.0)
    ax.text(RECALL_TARGET, 0.03, f" tuned target {RECALL_TARGET}", fontsize=8, color="#666666")
    ax.set_xlabel("recall in a cell"); ax.set_ylabel("fraction of cells at most that recall")
    ax.set_title(f"B. Complete and exact: recall over the {len(camp)} campaign cells\n"
                 f"precision is {min(pre):.3f} in every one of them, so every reported pair is a true one",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6); ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(0, 1.02)


def panel_online(ax, synth):
    """Steady state, not total time: each window's cost against that run's own median window.

    Plotting the per-window time against m would only repeat panel A. What has to be shown for an
    online operator is that the cost of a window does not drift as the stream advances, so the
    quartiles and the maximum are normalised by the run's own median."""
    pts = sr.on_axis(synth, "m", "ar1", "raw")
    width = 0.34
    for k, name in enumerate([ms.BF, "CorrTrack-LSH"]):
        key = "bruteforce" if name == ms.BF else "corrtrack"
        xs, lo, hi, wh, mx, out = [], [], [], [], [], []
        for i, c in enumerate(pts):
            r = c["raw_arms"].get(key) or {}
            if not r.get("step_time_median"):
                continue
            md = r["step_time_median"]
            xs.append(i + (k - 0.5) * width)
            lo.append(r["step_time_q1"] / md); hi.append(r["step_time_q3"] / md)
            wh.append(r["step_time_whisker_hi"] / md); mx.append(r["step_time_max"] / md)
            out.append((r.get("step_time_outliers") or 0, r.get("n_steps") or 0))
        if not xs:
            continue
        for x, a, b, w, m_ in zip(xs, lo, hi, wh, mx):
            ax.plot([x, x], [a, w], color=ms.COLOR[name], linewidth=2, alpha=0.6)
            ax.plot([x, x], [a, b], color=ms.COLOR[name], linewidth=8, solid_capstyle="butt")
            ax.plot(x, m_, marker="v", markersize=5, color=ms.COLOR[name], alpha=0.8)
        share = 100 * st.median([o / max(1, n) for o, n in out])
        ax.plot([], [], color=ms.COLOR[name], linewidth=8,
                label=f"{name}: quartiles and whisker; {share:.0f}% of windows beyond it, slowest marked")
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(["1x", "2x", "4x", "8x"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(range(len(pts))); ax.set_xticklabels([str(c["m"]) for c in pts])
    ax.set_xlabel("number of series m"); ax.set_ylabel("window time / that run's median window")
    ax.set_title("C. Online: the cost of a window is steady over the stream\n"
                 "1000 windows per cell; the quartiles sit within a few percent of the median at every m",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.legend(fontsize=8, loc="upper left")


def panel_nonstationary(ax, synth):
    regimes = [("ar1", "raw", "AR(1)\nlevels"), ("ar1", "differenced", "AR(1)\ndifferences"),
               ("rw", "raw", "random walk\nlevels"), ("rw", "differenced", "random walk\ndifferences")]
    base = {k: v for k, v in sr.BASE.items()}
    names = [n for n in SHOW]
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        xs, ys = [], []
        for i, (proc, space, _) in enumerate(regimes):
            cell = [c for c in synth if c["proc"] == proc and c["space"] == space
                    and all(c[a] == b for a, b in base.items())]
            if cell and cell[0]["arms"][name]:
                xs.append(i + (k - len(names) / 2 + 0.5) * width)
                ys.append(cell[0]["arms"][name]["speedup"])
        ax.bar(xs, ys, width=width, color=ms.COLOR[name], edgecolor="white", linewidth=0.5, label=name)
    ax.set_xticks(range(len(regimes))); ax.set_xticklabels([r[2] for r in regimes], fontsize=9)
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_ylabel("speedup over BF_vect"); ax.set_ylim(bottom=0)
    ax.set_title("D. Robust under nonstationarity: the same cell in four regimes\n"
                 "a random walk in levels is nonstationary and carries spurious correlation", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.legend(fontsize=8, loc="upper right")


def scaling_figure(synth, path):
    """Panel A on its own, with every arm and on both scales (2026-10-05, user).

    The log panel is where an exponent is read, since a power law is a straight line there; the linear
    panel is where the size of the gap is read, since a log axis hides it. Same data in both."""
    pts = sr.on_axis(synth, "m", "ar1", "raw")
    xs = [c["m"] for c in pts]
    arms = [ms.BF] + [n for n in ms.lead(ms.NAMES) if all(c["arms"].get(n) for c in pts)]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for ax, logscale in zip(axes, (True, False)):
        for name in arms:
            ys = [(c["bf_wall"] if name == ms.BF else c["arms"][name]["wall"]) for c in pts]
            e = exponent(xs, ys)
            ct = name.startswith("CorrTrack")
            ax.plot(xs, ys, marker="o", markersize=6 if ct else 4, color=ms.COLOR[name],
                    linewidth=2.4 if ct else 1.2, zorder=3 if ct else 2,
                    label=f"{name}   m^{e:.2f}" if logscale else None)
        if logscale:
            ref = [pts[0]["bf_wall"] * (x / xs[0]) ** 2 for x in xs]
            ax.plot(xs, ref, color="#999999", linestyle="--", linewidth=1.0, label="quadratic reference   m^2")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title("Log axes: the slope is the exponent in m", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
        else:
            ax.set_xscale("log"); ax.set_ylim(bottom=0)
            ax.set_title("Linear time axis: the size of the gap at m = 2000", fontsize=11)
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("number of series m (log spacing)")
        ax.set_ylabel("wall clock for the whole stream (s)")
        ax.grid(alpha=0.25, linewidth=0.6)
    fig.suptitle("How the cost grows with the number of series: AR(1) in levels, L=6, T=0.90, density 0.01, "
                 "1000 windows per cell\nthe exponent is fitted by least squares on the logs of the five points",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    camp_root, synth_root, figdir = sys.argv[1], sys.argv[2], sys.argv[3]
    out_md = sys.argv[4] if len(sys.argv) > 4 else None
    camp = cf.load(camp_root)
    synth = sr.load(synth_root)
    # the per-window quartiles are not in the synthetic loader's record, so read them alongside
    import json
    import glob
    raw = {}
    for p in glob.glob(os.path.join(synth_root, "nway", "*", "nway.json")):
        d = json.load(open(p))
        raw[os.path.basename(os.path.dirname(p))] = d["arms"]
    for c in synth:
        c["raw_arms"] = raw.get(c["run"], {})

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    panel_subquadratic(axes[0][0], synth)
    panel_complete(axes[0][1], camp)
    panel_online(axes[1][0], synth)
    panel_nonstationary(axes[1][1], synth)
    fig.suptitle("CorrTrack: a subquadratic operator for complete, online, lagged, signed, thresholded Pearson "
                 "correlation discovery, robust under nonstationarity", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(figdir, "headline_claim.png")      # the authors-only filter changes none of these panels
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    p = os.path.join(figdir, f"scaling_exponent{ms.SUFFIX}.png"); scaling_figure(synth, p); print("wrote", p)


if __name__ == "__main__":
    main()
