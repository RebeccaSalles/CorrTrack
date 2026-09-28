"""Figures for the final m=500 campaign (2026-09-28).

    python abaca/build_campaign_figures.py <results-root> <figures-dir>   # via campaign_figures.main

Two figures the tables cannot show:

  speedup_vs_density.png  speedup against the effective correlation density each cell actually has,
                          faceted by capability class and space. Density, not the threshold, is what
                          decides whether a filter can help, and it is the axis on which the methods
                          trade places.
  performance_profile.png a Dolan-More performance profile: the fraction of cells where a method is
                          within a factor tau of the best method on that cell. The right panel
                          counts a cell as solved only when the method also reached recall 0.95, so
                          the cost of an approximate arm's missed pairs is visible in the same plot.

Both encode where a capability comes from, because the classes are not all the authors' own: the
marker shape and the hatch come from the run's supports_lags / supports_neg_corr field, not from a
hardcoded list.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ORDER = [("bf_incremental", "bf_incr"), ("filcorr", "FilCorr"), ("tsubasa", "TSUBASA"), ("braid", "BRAID"),
         ("thinbraid", "ThinBRAID"), ("corrtrack", "CT-lsh"), ("corrtrack_hamming", "CT-ham"),
         ("parcorr", "ParCorr"), ("csz", "CSZ"), ("statstream", "StatStream"), ("corrjoin", "CorrJoin")]
NAMES = [n for _, n in ORDER]
COLOR = {"bf_incr": "#4c78a8", "FilCorr": "#f58518", "TSUBASA": "#8c6d31", "BRAID": "#7b4173",
         "ThinBRAID": "#d67ab1", "CT-lsh": "#c03d3e", "CT-ham": "#e8927c", "ParCorr": "#2f8a57",
         "CSZ": "#8fbf6b", "StatStream": "#57a3c7", "CorrJoin": "#c8b44a"}
# provenance of the capability the class asks for, as the runs themselves record it
SHAPE = {"native": "o", "enabled_by_us": "^", "specified": "s", "not_available": "X", None: "."}
TIER_LABEL = {"native": "evaluated by the authors",
              "enabled_by_us": "capability added by us",
              "specified": "specified by the authors, never evaluated there"}
CLASSES = [("S", "class S: synchronous, positive"), ("L", "class L: lagged, positive"),
           ("N", "class N: lagged, negative")]
SPACES = [("raw", "raw levels"), ("differenced", "first differences")]
RECALL_TARGET = 0.95
CELL = re.compile(r"^(?P<ds>.+?)_m\d+_W\d+_s\d+_L(?P<lags>\d+)_T(?P<T>[0-9.]+)(?P<diff>_diff)?_(?P<sign>pos|neg)$")


def specificity(arm, bf):
    U = bf.get("total_candidates") or bf.get("tested") or 0
    P = bf.get("correlated") or 0
    if arm.get("recall") is None or U <= P:
        return None
    return 1.0 - max((arm.get("correlated") or 0) - arm["recall"] * P, 0.0) / (U - P)


def load(root):
    cells = []
    for p in sorted(glob.glob(os.path.join(root, "nway", "*", "nway.json"))):
        d = json.load(open(p))
        m = CELL.match(d["cell"])
        if not m:
            continue
        bf = d["arms"].get("bruteforce") or {}
        if not bf.get("wall"):
            continue
        lagged = int(m["lags"]) > 0
        cls = "S" if not lagged else ("N" if m["sign"] == "neg" else "L")
        prof = d.get("dataset_profile") or {}
        cell = dict(run=os.path.basename(os.path.dirname(p)), dataset=m["ds"], cls=cls,
                    T=float(m["T"]), space="differenced" if m["diff"] else "raw",
                    density=prof.get("density_at_threshold"), bf_wall=bf["wall"], arms={})
        for key, label in ORDER:
            a = d["arms"].get(key)
            if not isinstance(a, dict) or a.get("status") != "ok" or not a.get("wall"):
                cell["arms"][label] = None
                continue
            tier = a.get("supports_neg_corr") if cls == "N" else (a.get("supports_lags") if lagged else "native")
            cell["arms"][label] = dict(wall=a["wall"], speedup=bf["wall"] / a["wall"], recall=a.get("recall"),
                                       precision=a.get("precision"), specificity=specificity(a, bf), tier=tier)
        cells.append(cell)
    return cells


def tier_of(cells, name):
    """The provenance to disclose for an arm over a slice: the least author-evaluated tier it uses,
    so a mixed slice is marked by the capability we had to add rather than by the one they shipped."""
    t = {c["arms"][name]["tier"] for c in cells if c["arms"][name]}
    for tier in ("enabled_by_us", "specified", "native"):
        if tier in t:
            return tier
    return "native"


def speedup_vs_density(cells, path):
    fig, axes = plt.subplots(len(SPACES), len(CLASSES), figsize=(16, 9), sharey=True)
    for i, (space, sdesc) in enumerate(SPACES):
        for j, (cls, cdesc) in enumerate(CLASSES):
            ax = axes[i][j]
            sl = [c for c in cells if c["space"] == space and c["cls"] == cls and c["density"]]
            for name in NAMES:
                by_T = {}
                for c in sl:
                    a = c["arms"][name]
                    if a:
                        by_T.setdefault(c["T"], []).append((c["density"], a["speedup"], a["recall"]))
                if not by_T:
                    continue
                xs = [st.median([v[0] for v in by_T[T]]) for T in sorted(by_T)]
                ys = [st.median([v[1] for v in by_T[T]]) for T in sorted(by_T)]
                rc = [st.median([v[2] for v in by_T[T] if v[2] is not None] or [1.0]) for T in sorted(by_T)]
                shape = SHAPE[tier_of(sl, name)]
                ct = name.startswith("CT-")
                ax.plot(xs, ys, color=COLOR[name], linewidth=2.4 if ct else 1.1, zorder=3 if ct else 2, label=name)
                for x, y, r in zip(xs, ys, rc):
                    ax.plot(x, y, marker=shape, markersize=9 if ct else 6, color=COLOR[name],
                            markerfacecolor=COLOR[name] if r >= RECALL_TARGET else "white",
                            markeredgecolor=COLOR[name], zorder=4 if ct else 2)
            ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_title(f"{cdesc}, {sdesc}", fontsize=10)
            if i == len(SPACES) - 1:      # one label per column, or the six collide
                ax.set_xlabel("effective correlation density\n(correlated pair-windows / tested)")
            if j == 0:
                ax.set_ylabel("median speedup over brute force")
    handles = [Line2D([], [], color=COLOR[n], marker="o", label=n) for n in NAMES]
    handles += [Line2D([], [], color="none", label=" ")]
    handles += [Line2D([], [], color="#555555", marker=SHAPE[t], linestyle="none", label=TIER_LABEL[t])
                for t in ("native", "enabled_by_us", "specified")]
    handles += [Line2D([], [], color="#555555", marker="o", linestyle="none", markerfacecolor="white",
                       label=f"hollow: median recall below {RECALL_TARGET}")]
    axes[0][-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Speedup against the density each cell actually has: median over the six datasets, one point per "
                 "threshold (0.70 to 0.95, density falling as the threshold rises)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.85, 0.95))
    fig.savefig(path, dpi=140); plt.close(fig)


def performance_profile(cells, path, taus=None):
    """rho_m(tau): the fraction of cells where method m is within tau of that cell's best speedup."""
    import numpy as np
    taus = taus if taus is not None else np.logspace(0, 2, 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for j, (title, gate) in enumerate([("every cell, recall ignored", False),
                                       (f"a cell counts only when the method also reached recall {RECALL_TARGET}", True)]):
        ax = axes[j]
        ratios = {n: [] for n in NAMES}
        n_cells = 0
        for c in cells:
            best = max((a["speedup"] for a in c["arms"].values() if a), default=None)
            if not best:
                continue
            n_cells += 1
            for name in NAMES:
                a = c["arms"][name]
                ok = a and (not gate or (a["recall"] is not None and a["recall"] >= RECALL_TARGET))
                ratios[name].append(best / a["speedup"] if ok else float("inf"))
        for name in NAMES:
            r = np.array(ratios[name])
            ys = [(r <= t).mean() for t in taus]
            ct = name.startswith("CT-")
            ax.step(taus, ys, where="post", color=COLOR[name], linewidth=2.4 if ct else 1.2,
                    zorder=3 if ct else 2, label=name)
            # the marker sits where the curve passes half the cells: the typical factor from the best
            half = next((k for k, y in enumerate(ys) if y >= 0.5), None)
            if half is not None:
                ax.plot(taus[half], ys[half], marker=SHAPE[tier_of(cells, name)], markersize=9 if ct else 7,
                        color=COLOR[name], markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.set_xscale("log"); ax.set_xlabel("tau: factor away from the best method on that cell")
        ax.grid(alpha=0.25, linewidth=0.6); ax.set_title(title, fontsize=10); ax.set_ylim(0, 1.02)
        if j == 0:
            ax.set_ylabel(f"fraction of the {n_cells} cells")
    handles = [Line2D([], [], color=COLOR[n], label=n) for n in NAMES]
    handles += [Line2D([], [], color="none", label=" ")]
    handles += [Line2D([], [], color="#555555", marker=SHAPE[t], linestyle="none", label=TIER_LABEL[t])
                for t in ("native", "enabled_by_us", "specified")]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Performance profile over every cell of the campaign: how often each method is close to the best "
                 "one, and how much of that survives the recall requirement", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.83, 0.94))
    fig.savefig(path, dpi=140); plt.close(fig)


def main(root, figdir):
    os.makedirs(figdir, exist_ok=True)
    cells = load(root)
    print(f"{len(cells)} cells")
    p = os.path.join(figdir, "speedup_vs_density.png"); speedup_vs_density(cells, p); print("wrote", p)
    p = os.path.join(figdir, "performance_profile.png"); performance_profile(cells, p); print("wrote", p)
    return cells


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
