"""Memory and energy per arm, from the resources each run already recorded (2026-09-29, user).

    python abaca/resource_tables.py <campaign-root> <synth-root> <out.md> [<figures-dir>]

Memory comes from `resources` in every nway.json: the sampler's peak and mean RSS around the arm's
own isolated child process, so it is that arm's footprint and not the harness's. Energy is the node
power series Grid'5000's kwollect recorded for the job (`abaca/kwollect_power.py` writes
`power.json`), integrated over each arm's [t_start_epoch, t_end_epoch] and reported with the node's
own idle power subtracted. Two limits are stated rather than smoothed over: the series has one
sample per 15 s, so only arms running well over a minute are quotable, and the campaign's own jobs
are past kwollect's retention window, so energy exists for the synthetic campaign only.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import method_style as ms

ARM_KEY = {k: n for k, n in ms.ARMS}
MIN_WALL_FOR_ENERGY = 60.0      # 15 s sampling: shorter arms are not separable from their neighbours


def load(root, power_path=None):
    """[(cell name, profile, {label: arm record})] with the resource block kept."""
    power = json.load(open(power_path)) if power_path and os.path.exists(power_path) else None
    out = []
    for p in sorted(glob.glob(os.path.join(root, "nway", "*", "nway.json"))):
        d = json.load(open(p))
        cellname = d.get("cell") or ""
        # the synthetic run names carry their own T twice ("..._T0p9_..._T0.9_pos"), so take the last,
        # which is the one written in decimal by the harness
        mt = re.findall(r"_T([0-9]+\.[0-9]+)_", cellname)
        if mt and not ms.kept_T(mt[-1]):
            continue
        prof = d.get("dataset_profile") or {}
        host = ((d.get("node") or {}).get("hostname") or "").split(".")[0]
        arms = {}
        for key, label in [("bruteforce", ms.BF)] + ms.ARMS:
            a = d["arms"].get(key)
            if not isinstance(a, dict) or a.get("status") != "ok":
                continue
            r = a.get("resources") or {}
            lagged = "_L0_" not in (d.get("cell") or "")
            neg = (d.get("cell") or "").endswith("_neg")
            tier = a.get("supports_neg_corr") if neg else (a.get("supports_lags") if lagged else "native")
            if label != ms.BF and (not ms.keeps(tier) or not ms.kept_arm(label)):     # authors-only mode
                continue
            arms[label] = dict(wall=a.get("wall"), recall=a.get("recall"),
                               peak=r.get("peak_rss_mb"), peak_delta=r.get("peak_rss_delta_mb"),
                               mean=r.get("mean_rss_mb"), io_write=r.get("io_write_mb"),
                               energy=integrate(power, host, r.get("t_start_epoch"), r.get("t_end_epoch")),
                               supports_lags=a.get("supports_lags"), supports_neg_corr=a.get("supports_neg_corr"))
        out.append((os.path.basename(os.path.dirname(p)), prof, d.get("cell", ""), arms, host))
    return out, power


def integrate(power, host, t0, t1):
    """Energy in joules over [t0, t1] by trapezoid on the node's power series, None when unavailable."""
    if not power or not host or t0 is None or t1 is None:
        return None
    s = power.get(host)
    if not s:
        return None
    pts = [(t, w) for t, w in s if t0 - 30 <= t <= t1 + 30]
    if len(pts) < 2:
        return None
    e = 0.0
    for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
        lo, hi = max(ta, t0), min(tb, t1)
        if hi > lo:
            e += 0.5 * (wa + wb) * (hi - lo)
    return e or None


def idle_of(power, host):
    """The node's lowest recorded power during the job: the floor to subtract for dynamic energy."""
    s = (power or {}).get(host) or []
    return min((w for _, w in s), default=None)


def cls_of(cell_name):
    """(class, space) of a campaign cell, from its own name."""
    lagged = "_L0_" not in cell_name
    cls = "S" if not lagged else ("N" if cell_name.endswith("_neg") else "L")
    return cls, ("differenced" if "_diff_" in cell_name else "raw")


def per_class_table(cells, field, label, unit):
    """The same statistic per class and space, plus the cells every arm shares.

    A pooled median would compare arms over different cell sets as soon as one arm is missing from a
    class, and a missing class is never a random sample of the campaign."""
    COLS = [(c, sp) for c in ("S", "L", "N") for sp in ("raw", "differenced")]
    tagged = [(cls_of(c[2] or c[0]), c) for c in cells]
    present = [n for n in [ms.BF] + ms.lead(ms.NAMES) if any(n in c[3] for c in cells)]
    common = [c for c in cells if all(n in c[3] and c[3][n].get(field) for n in present)]
    out = ["| arm | " + " | ".join(f"{c} {sp[:4]}" for c, sp in COLS) + " | common cells |",
           "|---|" + "---|" * (len(COLS) + 1)]
    for n in present:
        vals = []
        for cls, sp in COLS:
            v = [c[3][n][field] for key, c in tagged if key == (cls, sp) and n in c[3] and c[3][n].get(field)]
            vals.append(f"{st.median(v):,.0f}" if v else "")
        cv = [c[3][n][field] for c in common if n in c[3] and c[3][n].get(field)]
        out.append(f"| {n} | " + " | ".join(vals) + (f" | {st.median(cv):,.0f} |" if cv else " |  |"))
    out += ["", f"Median {label} in {unit}, per capability class and space. The last column is the median over "
                f"the {len(common)} cells where every arm ran, the only column whose numbers all price the same "
                f"work; blank cells are classes the arm does not cover."]
    return out


def median_table(cells, field, label, unit, arms=None):
    names = arms or ([ms.BF] + ms.lead(ms.NAMES))
    rows = []
    for n in names:
        v = [a[n][field] for _, _, _, a, _ in cells if n in a and a[n].get(field)]
        if v:
            rows.append((n, st.median(v), min(v), max(v), len(v)))
    out = [f"| arm | median {label} ({unit}) | min | max | cells |", "|---|---|---|---|---|"]
    out += [f"| {n} | {m:,.0f} | {lo:,.0f} | {hi:,.0f} | {c} |" for n, m, lo, hi, c in rows]
    return out


def memory_figure(camp, synth, path):
    """Left: what each arm's process peaks at across the campaign. Right: how that grows with m."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    names = [n for n in [ms.BF] + ms.lead(ms.NAMES) if any(n in a for _, _, _, a, _ in camp)]
    ax = axes[0]
    for i, n in enumerate(names):
        v = sorted(a[n]["peak_delta"] for _, _, _, a, _ in camp if n in a and a[n].get("peak_delta"))
        if not v:
            continue
        y = len(names) - i
        ax.plot([v[len(v) // 10], v[9 * len(v) // 10]], [y, y], color=ms.COLOR[n], linewidth=2.5, alpha=0.5)
        ax.plot(st.median(v), y, marker="o", markersize=9, color=ms.COLOR[n])
    ax.set_yticks(range(len(names), 0, -1)); ax.set_yticklabels(names)
    ax.set_xscale("log"); ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("peak resident memory added by the arm (MB, log scale)")
    ax.set_title("Final campaign, 179 cells: median and 10th to 90th percentile", fontsize=11)
    ax = axes[1]
    for n in names:
        # raw levels only, so the curve is one series per m rather than two overlaid
        pts = sorted([(prof.get("m"), a[n]["peak"]) for name, prof, _, a, _ in synth
                      if n in a and a[n].get("peak") and "_diff" not in name and prof.get("m")], key=lambda t: t[0])
        agg = {}
        for m, v in pts:
            agg.setdefault(m, []).append(v)
        xs = sorted(agg)
        if xs:
            ax.plot(xs, [st.median(agg[x]) for x in xs], marker="o", markersize=5, color=ms.COLOR[n],
                    linewidth=2.2 if n.startswith("CorrTrack") else 1.2, label=n)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xticks([125, 250, 500, 1000, 2000]); ax.set_xticklabels(["125", "250", "500", "1000", "2000"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("number of series m"); ax.set_ylabel("peak resident memory (MB, log scale)")
    ax.set_title("Synthetic campaign: peak memory against m", fontsize=11)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Memory per arm, measured around each arm's own isolated process", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(path, dpi=140); plt.close(fig)


def energy_figure(eligible, power, path):
    """Energy per arm on the cells where every arm runs long enough for a 15 s series to separate them."""
    if not eligible:
        return None
    fig, ax = plt.subplots(figsize=(12, 5.6))
    bars = []
    for n in [ms.BF] + ms.lead(ms.NAMES):
        v = [a[n]["energy"] / 1000 for _, _, _, a, _ in eligible if n in a and a[n].get("energy")]
        if v:
            bars.append((n, st.median(v)))
    bars.sort(key=lambda b: b[1])
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[ms.COLOR[b[0]] for b in bars], edgecolor="white")
    for i, (n, v) in enumerate(bars):
        ax.text(i, v * 1.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log"); ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_ylabel("node energy over the arm's own interval (kJ, log scale)")
    ax.set_title(f"Energy per arm, the {len(eligible)} synthetic cells where every arm runs past "
                 f"{MIN_WALL_FOR_ENERGY:.0f} s (m = 1000 and 2000), lowest first", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def main() -> None:
    camp_root, synth_root, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    figdir = sys.argv[4] if len(sys.argv) > 4 else None
    camp, _ = load(camp_root)
    synth, power = load(synth_root, os.path.join(synth_root, "power.json"))
    out = ["# Memory and energy per arm (2026-09-29)", "",
           "Measured, not modelled: every arm of every cell ran as an isolated child process with an RSS sampler "
           "around it (`abaca/resource_probe.py`), so the numbers below are that arm's own footprint. Energy is "
           "the node power Grid'5000 recorded for the job, integrated over each arm's own start and end.", "",
           "## 1. Memory, final m=500 campaign (179 cells)", "",
           "Peak resident set of the arm's process, and the delta over what it had already allocated before the "
           "arm started, which is the part the method itself is responsible for.", ""]
    out += median_table(camp, "peak", "peak RSS", "MB") + [""]
    out += ["Same cells, the delta only:", ""] + median_table(camp, "peak_delta", "peak RSS delta", "MB") + [""]
    out += ["### Peak RSS delta per class and space", "",
            "Pooled medians compare arms over different cell sets whenever an arm is missing from a class, so "
            "the comparable reading is within a class:", ""]
    out += per_class_table(camp, "peak_delta", "peak RSS delta", "MB") + [""]
    out += ["## 2. Memory, synthetic campaign (64 cells, m from 125 to 2000)", "",
            "The same arms on generated data, where m is an axis rather than a constant:", ""]
    out += median_table(synth, "peak", "peak RSS", "MB") + [""]
    big = [c for c in synth if (c[1].get("m") or 0) >= 2000]
    if big:
        out += ["Peak RSS at m=2000 only, the largest cells of the study:", ""] + median_table(big, "peak", "peak RSS", "MB") + [""]

    # 3. energy: only the cells where every arm runs long enough for a 15 s series to separate them,
    # otherwise the per-arm medians compare different subsets (the fast arms would drop out and the
    # slow ones would look representative)
    eligible = [c for c in synth
                if all(a["energy"] and a["wall"] and a["wall"] >= MIN_WALL_FOR_ENERGY for a in c[3].values())]
    out += ["## 3. Energy, synthetic campaign", "",
            f"The power series has one sample per 15 s, so an arm is only separable from its neighbours when it "
            f"runs well past a minute. This section therefore uses the cells where EVERY arm clears "
            f"{MIN_WALL_FOR_ENERGY:.0f} s, which is {len(eligible)} of the {len(synth)} cells, all of them the "
            f"largest ones; taking a median per arm over whatever qualifies would compare different subsets. "
            f"Dynamic energy subtracts the node's lowest power during that job, so it is what the arm added over "
            f"the machine's baseline. The campaign's own jobs are past kwollect's retention window, so energy "
            f"exists for the synthetic campaign only.", ""]
    if eligible:
        out += [f"Cells used: " + ", ".join(sorted({f"m={c[1].get('m')}" for c in eligible})) + ".", "",
                "| arm | median energy (kJ) | dynamic (kJ) | mean power (W) | kJ per billion pair-windows | cells |",
                "|---|---|---|---|---|---|"]
        for n in [ms.BF] + ms.lead(ms.NAMES):
            v = []
            for name, prof, cell, arms, host in eligible:
                a = arms.get(n)
                if not a or not a["energy"]:
                    continue
                idle = idle_of(power, host) or 0.0
                pw = prof.get("pair_windows") or 0
                v.append((a["energy"], max(a["energy"] - idle * a["wall"], 0.0), a["energy"] / a["wall"],
                          a["energy"] / pw * 1e9 if pw else None))
            if v:
                per = [x[3] for x in v if x[3]]
                out.append(f"| {n} | {st.median([x[0] for x in v]) / 1000:,.1f} | {st.median([x[1] for x in v]) / 1000:,.1f} | "
                           f"{st.median([x[2] for x in v]):,.0f} | {st.median(per) / 1000:,.2f} | {len(v)} |")
        out += ["", "The mean power column is the reading to keep in mind: it is within a few watts for every arm, "
                    "so on this hardware energy is wall clock multiplied by a near-constant. The energy ranking is "
                    "therefore the speed ranking, and the honest claim is that CorrTrack saves energy in "
                    "proportion to the time it saves, not that it is more efficient per unit of work in some "
                    "further sense. A wattmeter node, which the mercantour cluster does not have, would be needed "
                    "to say more.", ""]
    else:
        out += ["No cell has every arm above the threshold, so no comparable per-arm energy is reported.", ""]
    if figdir:
        os.makedirs(figdir, exist_ok=True)
        p = os.path.join(figdir, f"resources_memory{ms.SUFFIX}.png"); memory_figure(camp, synth, p); print("wrote", p)
        p = os.path.join(figdir, f"resources_energy{ms.SUFFIX}.png")
        if energy_figure(eligible, power, p):
            print("wrote", p)
    out = ms.banner(out)
    open(out_path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
