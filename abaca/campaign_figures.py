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
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import method_style as ms

ORDER, NAMES, COLOR, SHAPE, TIER_LABEL = ms.ARMS, ms.NAMES, ms.COLOR, ms.SHAPE, ms.TIER_LABEL
CLASSES = [("S", "class S: synchronous, positive only"), ("L", "class L: lagged, positive only"),
           ("N", "class N: lagged, both signs")]
SPACES = [("raw", "raw levels"), ("differenced", "first differences")]
RECALL_TARGET = 0.95
# (2026-09-28, user) the profile is also read at the looser target, since a method that misses 0.95
# by a little is a different case from one that misses most of the correlated pairs
RECALL_LEVELS = (None, 0.95, 0.90)
CELL = re.compile(r"^(?P<ds>.+?)_m\d+_W(?P<W>\d+)_s(?P<step>\d+)_L(?P<lags>\d+)_T(?P<T>[0-9.]+)(?P<diff>_diff)?_(?P<sign>pos|neg)$")


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
                    density=prof.get("density_at_threshold"), bf_wall=bf["wall"], arms={},
                    m=prof.get("m"), n_obs=prof.get("n_obs"), n_windows=prof.get("n_windows"),
                    pair_windows=prof.get("pair_windows"), W=int(m["W"]), step=int(m["step"]),
                    n_lags=int(m["lags"]), lag_buckets=int(m["lags"]) // int(m["step"]) + 1,
                    best_params=(d.get("config") or {}).get("best_params"),
                    best_params_hamming=(d.get("config") or {}).get("best_params_hamming"))
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
            for name in ms.lead(NAMES):
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
                ct = name.startswith("CorrTrack")
                ax.plot(xs, ys, color=COLOR[name], linewidth=2.4 if ct else 1.1, zorder=3 if ct else 2, label=name)
                for x, y, r in zip(xs, ys, rc):
                    ax.plot(x, y, marker=shape, markersize=9 if ct else 6, color=COLOR[name],
                            markerfacecolor=COLOR[name] if r >= RECALL_TARGET else "white",
                            markeredgecolor=COLOR[name], zorder=4 if ct else 2)
            ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, linewidth=0.6)
            thr = sorted({c["T"] for c in sl})
            if i == 0:
                ax.set_title(cdesc, fontsize=11, pad=42)
            # a second axis on top names the threshold each point came from, since the x axis is density
            top = ax.twiny(); top.set_xscale("log"); top.set_xlim(ax.get_xlim())
            top.set_xticks([st.median([c["density"] for c in sl if c["T"] == T]) for T in thr])
            top.set_xticklabels([f"{T:g}" for T in thr], fontsize=8)
            if i == 0:
                top.set_xlabel("correlation threshold T", fontsize=9, labelpad=4)
            if i == len(SPACES) - 1:      # one label per column, or the six collide
                ax.set_xlabel("effective correlation density")
            if j == 0:
                ax.set_ylabel(f"{sdesc}\nmedian speedup over BF_vect")
    handles = [Line2D([], [], color=COLOR[n], marker="o", label=n) for n in ms.lead(NAMES)]
    handles += [Line2D([], [], color="none", label=" ")]
    handles += [Line2D([], [], color="#555555", marker=SHAPE[t], linestyle="none", label=TIER_LABEL[t])
                for t in ("native", "enabled_by_us", "specified")]
    handles += [Line2D([], [], color="#555555", marker="o", linestyle="none", markerfacecolor="white",
                       label=f"hollow: median recall below {RECALL_TARGET}")]
    axes[0][-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Speedup over BF_vect against the effective correlation density the data actually has "
                 "(correlated pair-windows / tested).\nEach point is the median of the six datasets at one threshold, "
                 "the upper axis naming that threshold; density falls as the threshold rises.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.85, 0.92))
    fig.savefig(path, dpi=140); plt.close(fig)


def performance_profile(cells, path, taus=None):
    """rho_m(tau): the fraction of cells where method m is within tau of that cell's best speedup."""
    import numpy as np
    taus = taus if taus is not None else np.logspace(0, 2, 200)
    fig, axes = plt.subplots(1, len(RECALL_LEVELS), figsize=(18, 5.8), sharey=True)
    for j, gate in enumerate(RECALL_LEVELS):
        title = "speed only, recall ignored" if gate is None else f"only cells where the method reached recall {gate}"
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
                ok = a and (gate is None or (a["recall"] is not None and a["recall"] >= gate))
                ratios[name].append(best / a["speedup"] if ok else float("inf"))
        for name in ms.lead(NAMES):
            r = np.array(ratios[name])
            ys = [(r <= t).mean() for t in taus]
            ct = name.startswith("CorrTrack")
            ax.step(taus, ys, where="post", color=COLOR[name], linewidth=2.4 if ct else 1.2,
                    zorder=3 if ct else 2, label=name)
            # the marker sits where the curve passes half the cells: the typical factor from the best
            half = next((k for k, y in enumerate(ys) if y >= 0.5), None)
            if half is not None:
                ax.plot(taus[half], ys[half], marker=SHAPE[tier_of(cells, name)], markersize=9 if ct else 7,
                        color=COLOR[name], markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.set_xscale("log")
        ax.grid(alpha=0.25, linewidth=0.6); ax.set_title(title, fontsize=11); ax.set_ylim(0, 1.02)
        if j == 0:
            ax.set_ylabel(f"fraction of the {n_cells} cells")
    handles = [Line2D([], [], color=COLOR[n], label=n) for n in ms.lead(NAMES)]
    handles += [Line2D([], [], color="none", label=" ")]
    handles += [Line2D([], [], color="#555555", marker=SHAPE[t], linestyle="none", label=TIER_LABEL[t])
                for t in ("native", "enabled_by_us", "specified")]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Performance profile over the 179 campaign cells (one dataset, one threshold, one capability class, "
                 "one space): how often\neach method runs within a factor tau of the fastest method on that same cell. "
                 "A cell the method does not solve under the\npanel's recall requirement never counts, at any tau; the "
                 "marker sits where a method reaches half the cells.", fontsize=11)
    fig.supxlabel("tau: factor away from the fastest method on the same cell", fontsize=10, x=0.44)
    fig.tight_layout(rect=(0, 0.03, 0.87, 0.90))
    fig.savefig(path, dpi=140); plt.close(fig)


TIE = 0.10          # as in abaca/build_ranking_tables.py: speedups within 10% count as tied
# What each arm guarantees and how its parameters were chosen in this campaign. These are design
# facts, not measurements; everything else in the capability table is read from the runs.
DESIGN = {
    "BF_vect": ("exact: the ground truth and the timing anchor of every cell", "none"),
    "BF_incr": ("exact: same set as BF_vect, incremental statistics", "none"),
    "FilCorr": ("exact at the full band used here, no recall knob", "full band, not tuned"),
    "TSUBASA": ("exact, no recall knob", "none"),
    "BRAID": ("approximate by design, reported the exact set on these lag grids", "published defaults"),
    "ThinBRAID": ("approximate: emits its own decision, no validation pass", "published defaults"),
    "CorrTrack-LSH": ("exact validation, probabilistic candidate search: precision 1, recall below 1",
                      "own hyperopt, sketch width fixed at 64"),
    "CorrTrack-Ham": ("exact validation, exact candidate search over the sketches",
                      "own hyperopt, hamming grid"),
    "ParCorr": ("exact validation, sketch pruning", "CSZ protocol"),
    "CSZ": ("exact validation, sketch pruning", "CSZ protocol"),
    "StatStream": ("approximate: emits its own decision from the sketch grid", "CSZ protocol"),
    "CorrJoin": ("exact validation, sketch pruning", "CSZ protocol"),
}


def leadership(cells, qualify=False):
    """Per method: the cells it leads and the cells it is among the two fastest in, exactly as the
    ranking tables count them (10% tie band). With qualify, only the arms reaching RECALL_TARGET in
    that cell are ranked, so an arm cannot lead a cell by skipping the pairs it should have found."""
    import collections
    first, top2, n_cells = collections.Counter(), collections.Counter(), 0
    for c in cells:
        vals = {n: a["speedup"] for n, a in c["arms"].items()
                if a and (not qualify or (a["recall"] is not None and a["recall"] >= RECALL_TARGET))}
        if not vals:
            continue
        n_cells += 1
        best = max(vals.values())
        ranked = sorted(vals.values(), reverse=True)
        cut = ranked[1] if len(ranked) > 1 else best
        for n, v in vals.items():
            if v >= best * (1 - TIE):
                first[n] += 1
            if v >= cut * (1 - TIE):
                top2[n] += 1
    return first, top2, n_cells


def capability_markdown(cells):
    """Table 1: what each method can do, where that capability comes from, and what it delivered.

    The capability tiers are read from the runs (supports_lags / supports_neg_corr), the recall and
    precision from the same runs; only the exactness and tuning columns are design facts."""
    import statistics as st

    def tier_in(name, cls):
        t = {c["arms"][name]["tier"] for c in cells if c["cls"] == cls and c["arms"][name]}
        if not t:
            return "not available"
        for k in ("enabled_by_us", "specified", "native"):
            if k in t:
                return {"enabled_by_us": "added by us", "specified": "specified, never evaluated",
                        "native": "authors"}[k]
        return "?"

    out = ["| method | what it guarantees | lagged search | negative correlation | parameters here | recall: median (10th percentile) | precision median |",
           "|---|---|---|---|---|---|---|"]
    guarantee, params = DESIGN["BF_vect"]
    out.append(f"| BF_vect | {guarantee} | authors | authors | {params} | 1.000 | 1.000 |")
    for name in ms.lead(NAMES):
        rec = sorted(c["arms"][name]["recall"] for c in cells if c["arms"][name] and c["arms"][name]["recall"] is not None)
        pre = [c["arms"][name]["precision"] for c in cells if c["arms"][name] and c["arms"][name]["precision"] is not None]
        guarantee, params = DESIGN[name]
        out.append(f"| {name} | {guarantee} | {tier_in(name, 'L')} | {tier_in(name, 'N')} | {params} | "
                   f"{st.median(rec):.3f} ({rec[len(rec) // 10]:.3f}) | {st.median(pre):.3f} |")
    return out


DATASET_LABEL = {"sp500_m500": "sp500", "streamflow_m500": "streamflow", "smartmeter_m500": "smartmeter",
                 "wikipedia_m500": "wikipedia", "global_weather_m500": "global_weather",
                 "acwi_capweighted_m500": "acwi_capweighted"}
ALWAYS_SHOW = ["BF_incr"]      # the exact incremental baseline is the reference, whether or not it leads


CT_ARMS = ["CorrTrack-LSH", "CorrTrack-Ham"]


def beats_corrtrack(sl, name):
    """(cells where this arm is faster than CorrTrack, cells where it reaches the recall target and
    CorrTrack does not). CorrTrack is taken as the method, i.e. the better of its two backends in
    that cell, since that is how the ranking tables treat it."""
    faster = reached = 0
    for c in sl:
        a = c["arms"][name]
        ct = [c["arms"][k] for k in CT_ARMS if c["arms"][k]]
        if not a or not ct:
            continue
        if a["speedup"] > max(v["speedup"] for v in ct):
            faster += 1
        if not any(v["recall"] is not None and v["recall"] >= RECALL_TARGET for v in ct) \
                and a["recall"] is not None and a["recall"] >= RECALL_TARGET:
            reached += 1
    return faster, reached


def recall_contest(sl):
    """The reviewer's question about the column rule: where CorrTrack's fastest backend misses the
    recall target, is some competitor the fastest arm that does meet it?

    Returns (cells where CorrTrack's fastest backend misses, [(cell, arm, speedup, corrtrack's best
    qualifying speedup) for the cells a competitor wins]). The column rule asks whether an arm
    reaches the target where CorrTrack cannot at all, which is a weaker question: CorrTrack always
    has a qualifying backend, so that clause never fires anywhere in this campaign."""
    missed, taken = 0, []
    for c in sl:
        ct = [(c["arms"][k]["speedup"], k) for k in CT_ARMS if c["arms"][k]]
        if not ct or max(ct)[1] and c["arms"][max(ct)[1]]["recall"] is None:
            continue
        if c["arms"][max(ct)[1]]["recall"] >= RECALL_TARGET:
            continue
        missed += 1
        ok = [(c["arms"][n]["speedup"], n) for n in NAMES
              if c["arms"][n] and c["arms"][n]["recall"] is not None and c["arms"][n]["recall"] >= RECALL_TARGET]
        if not ok:
            continue
        best_sp, best_n = max(ok)
        ct_ok = max([sp for sp, n in ok if n in CT_ARMS], default=None)
        if best_n not in CT_ARMS:
            taken.append((c, best_n, best_sp, ct_ok))
    return missed, taken


def columns_for(sl):
    """(2026-09-28, user) Which arms a per-cell table shows: CorrTrack's backends, the exact
    incremental baseline, and every arm that somewhere in this slice either runs faster than
    CorrTrack or reaches the recall target where CorrTrack does not.

    The rule compares each arm with the method the table is about, so an arm earns its column by
    being a real alternative in some cell rather than by being merely respectable. The earlier rules
    ranked an arm against its neighbours ("among the two fastest", which dropped CorrJoin because
    CorrTrack held both places in its best cell) or against the baseline, which let in arms that are
    never competitive with CorrTrack anywhere."""
    keep, dropped = [], []
    for n in ms.lead(NAMES):
        if n in ALWAYS_SHOW or n in CT_ARMS:
            keep.append(n)
            continue
        if not any(c["arms"][n] for c in sl):
            continue
        (keep if any(beats_corrtrack(sl, n)) else dropped).append(n)
    return keep, dropped


def _tuned(path, root):
    """The tuned CorrTrack parameters of a cell, read from the hyperopt best_params file."""
    if not path:
        return {}
    local = os.path.join(root, path.split("final_112ce48/", 1)[1]) if "final_112ce48/" in path else path
    try:
        return json.load(open(local))
    except (OSError, ValueError):
        return {}


def dataset_table_markdown(cells, cls, space, root):
    """One capability class in full, laid out per dataset and threshold: what the cell is, what brute
    force costs there, what each method saves, and what CorrTrack was tuned to.

    Speedups are against the cell's own BF_vect. The leading speedup of a row is bold. Density is
    brute force's own count over the tested pair-windows, which is already a per-window figure: it is
    the fraction of the (pair, lag, window) tuples tested that come back correlated, so it is the
    correlation density of an average window. The column beside it is that same number as a count
    rather than a fraction, density x (m - 1) x lag buckets, against the maximum a series could
    reach, (m - 1) x lag buckets, so the scale is on the page instead of in the reader's head."""
    import statistics as st
    sl = [c for c in cells if c["cls"] == cls and c["space"] == space]
    show, dropped = columns_for(sl)
    out = ["| dataset | m | L | pair-windows | obs | T | density % | corr. partners per series, of (m-1)xL | BF_vect s | "
           + " | ".join(show) + " | Ham recall | Ham gamma off | LSH recall | LSH gamma off | LSH occ |",
           "|---|---|---|---|---|---|---|---|---|" + "---|" * (len(show) + 5)]
    for stem in DATASET_LABEL:
        rows = sorted([c for c in sl if c["dataset"] == stem], key=lambda c: c["T"])
        for i, c in enumerate(rows):
            sp = {n: (c["arms"][n]["speedup"] if c["arms"][n] else None) for n in show}
            best = max([v for v in sp.values() if v], default=None)
            cells_txt = []
            for n, v in sp.items():
                if not v:
                    cells_txt.append(""); continue
                r = c["arms"][n]["recall"]
                flag = "" if r is None or r >= RECALL_TARGET else " †"      # bought speed by missing pairs
                cells_txt.append(("**%.2fx**" % v if v == best else "%.2fx" % v) + flag)
            ham, lsh = _tuned(c["best_params_hamming"], root), _tuned(c["best_params"], root)
            head = (f"| {DATASET_LABEL[stem]} | {c['m']} | {c['lag_buckets']} | {c['pair_windows']:,} | "
                    f"{c['n_obs']} " if i == 0 else "|  |  |  |  |  ")
            def g(d, k):
                v = d.get(k)
                return "" if v is None or v != v else (f"{v:g}")
            out.append(head + f"| {c['T']:g} | {c['density'] * 100:.3g} | "
                       f"{c['density'] * (c['m'] - 1) * c['lag_buckets']:.3g} of {(c['m'] - 1) * c['lag_buckets']:,} | "
                       f"{c['bf_wall']:.1f} | "
                       + " | ".join(cells_txt) +
                       f" | {c['arms']['CorrTrack-Ham']['recall']:.2f} | {g(ham, 'candidate_cosine_threshold_offset')} | "
                       f"{c['arms']['CorrTrack-LSH']['recall']:.2f} | {g(lsh, 'candidate_cosine_threshold_offset')} | "
                       f"{g(lsh, 'candidate_lsh_target_occupancy')} |")
    if dropped:
        summary = ", ".join(
            f"{n} {st.median([c['arms'][n]['speedup'] for c in sl if c['arms'][n]]):.2f}x median and "
            f"{max(c['arms'][n]['speedup'] for c in sl if c['arms'][n]):.2f}x at best" for n in dropped)
        missed, taken = recall_contest(sl)
        if missed:
            if taken:
                detail = "; ".join(f"{a} at {sp:.2f}x against CorrTrack's {ct:.2f}x on {c['dataset']} at T={c['T']:g}"
                                   for c, a, sp, ct in taken)
                note = (f" In {missed} of these cells CorrTrack's faster backend misses recall {RECALL_TARGET}, and in "
                        f"{len(taken)} of those a competitor is the fastest arm that does reach it: {detail}. In the "
                        f"other {missed - len(taken)} the second CorrTrack backend still is.")
            else:
                note = (f" In {missed} of these cells CorrTrack's faster backend misses recall {RECALL_TARGET}, and in "
                        f"every one of them its second backend is still the fastest arm that reaches the target.")
        else:
            note = f" CorrTrack's faster backend reaches recall {RECALL_TARGET} in every cell of this class."
        out += ["", f"Columns are CorrTrack's two backends, the exact incremental baseline, and every arm that in "
                    f"at least one of these {len(sl)} cells either runs faster than CorrTrack or reaches recall "
                    f"{RECALL_TARGET} where CorrTrack does not. The arms left out do neither anywhere in this "
                    f"class: {summary}. They are in the full table of this class in the appendix, where all eleven "
                    f"arms appear. A dagger marks a speedup whose recall in that cell is below {RECALL_TARGET}."
                    + note]
    return out


def leadership_markdown(cells):
    import statistics as st
    fa, ta, n = leadership(cells)
    fq, tq, nq = leadership(cells, qualify=True)
    med = {n_: st.median([c["arms"][n_]["speedup"] for c in cells if c["arms"][n_]]) for n_ in NAMES}
    out = [f"| method | leads | top two | leads (recall >= {RECALL_TARGET}) | top two (recall >= {RECALL_TARGET}) | median speedup |",
           "|---|---|---|---|---|---|"]
    for name in sorted(NAMES, key=lambda x: (-fa[x], -ta[x], -med[x])):
        out.append(f"| {name} | {fa[name]} | {ta[name]} | {fq[name]} | {tq[name]} | {med[name]:.2f}x |")
    return f"Over the {n} cells of the campaign, speedups within {int(TIE * 100)}% counting as tied.", out


def main(root, figdir, table_path=None):
    os.makedirs(figdir, exist_ok=True)
    cells = load(root)
    print(f"{len(cells)} cells")
    p = os.path.join(figdir, "real_speedup_vs_density.png"); speedup_vs_density(cells, p); print("wrote", p)
    p = os.path.join(figdir, "real_performance_profile.png"); performance_profile(cells, p); print("wrote", p)
    t3 = dataset_table_markdown(cells, "L", "differenced", root)
    print("\n" + "\n".join(t3))
    cap = capability_markdown(cells)
    print("\n" + "\n".join(cap))
    caption, table = leadership_markdown(cells)
    print("\n" + caption + "\n" + "\n".join(table))
    if table_path:
        open(table_path, "w").write(
            "# Campaign capabilities and leadership (2026-09-28)\n\n"
            "Generated by `abaca/campaign_figures.py` from the 179 cells of the final m=500 campaign. The "
            "capability columns are read from each run's own supports_lags / supports_neg_corr field and the "
            "recall and precision from the same runs; only the guarantee and parameter columns are design "
            "facts. Recall and precision are summarised over the cells where the method ran: the median, and "
            "in brackets the 10th percentile, meaning the method reached at least that recall in 90% of its "
            "cells. The percentile is there because a median alone hides a method that is fine in most cells "
            "and collapses in a few: ParCorr, CSZ and StatStream each fall below recall 0.5 in the same four "
            "smartmeter synchronous cells, where the correlation is so sparse that every sketch filter "
            "collapses.\n\n## Table 1: what each method can do, and who enabled it\n\n" + "\n".join(cap) +
            "\n\n## Table 2: the cells each method leads\n\n"
            "Generated by `abaca/campaign_figures.py`. " + caption + " A method leads a cell when its "
            "speedup is within 10% of the fastest arm of that cell, so several arms can lead one cell. The "
            f"qualified columns rank only the arms that reached recall {RECALL_TARGET} there, which is what "
            "stops an arm from leading by skipping the pairs it should have found. This count, over all 179 "
            "cells, is also the order the legends of every figure use.\n\n" + "\n".join(table) +
            "\n\n## Table 3: class L (lagged, positive only), first differences, per dataset and threshold\n\n"
            "Speedups are against the cell's own BF_vect, the leading one of each row in bold. Density is brute "
            "force's own count over the tested pair-windows, which is already a per-window figure: it is the "
            "fraction of tested (pair, lag, window) tuples that come back correlated, so it is the correlation "
            "density of an average window. The column beside it is the same number as a count, density x "
            "(m - 1) x lag buckets, shown against the maximum a series could reach so the scale is visible: "
            "sp500 at T=0.70 reads 13.4 of 2,658, a typical series correlated with about thirteen (partner, lag) "
            "combinations at once out of the 2,658 it is tested against; at T=0.95 it reads 0.023 of 2,658, so "
            "only about one series in forty has any correlated partner at all in a given window. The last five "
            "columns are what CorrTrack's own hyperopt chose on that "
            "cell, gamma off being the offset added to the calibrated cosine threshold and occ the LSH target "
            "occupancy.\n\n" + "\n".join(t3) + "\n")
        print("wrote", table_path)
    return cells


if __name__ == "__main__":
    main(*sys.argv[1:4])
