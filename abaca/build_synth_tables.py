"""Tables for the synthetic OFAT campaign: generator validation, the four scaling curves and the
per-cell numbers (2026-09-28).

    python abaca/build_synth_tables.py <results-root> <out.md>

Every number comes from one job per cell: the twelve arms and the brute force that anchors their
speedups ran in that same job, monitoring off, on the whole stream. The generated data is what the
generator produced, so the density column is the one brute force measured (density_at_threshold in
the run's dataset profile), with the requested target beside it.
"""
from __future__ import annotations

import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_results as sr

PROCS = [("ar1", "AR(1), stationary in raw levels"), ("rw", "random walk, nonstationary in raw levels")]
AXIS_TITLE = {"m": "number of series m", "L": "lagged windows L", "T": "correlation threshold T",
              "density": "target correlation density"}
NAMES = [n for _, n in sr.ORDER]


def f(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def cell_of(cells, **kw):
    hit = [c for c in cells if all(c[k] == v for k, v in kw.items())]
    return hit[0] if hit else None


def main() -> None:
    root, out_path = sys.argv[1], sys.argv[2]
    cells = sr.load(root)
    out = ["# Synthetic scaling campaign: tables", "",
           f"{len(cells)} cells from `abaca/campaign_synth_overnight.py`: one factor at a time around the baseline "
           f"m={sr.BASE['m']}, L={sr.BASE['L']}, T={sr.BASE['T']}, target density={sr.BASE['density']}, in both "
           "spaces (raw levels and first differences) and for both base processes. W=60, step=6, 6000 observations, "
           "991 evaluated windows, positive correlation only. Each cell ran the twelve arms and its own brute force "
           "in one job with monitoring off, so every speedup is a wall-clock ratio inside one job on one node.", "",
           "In raw levels a random walk is nonstationary and carries spurious correlation the generator did not "
           "plant; differencing turns it into white noise, which is the uncooperative case for any filter built on "
           "energy concentration. The AR(1) process is stationary in raw levels already.", ""]

    # 1. does the generator deliver the density it was asked for?
    out += ["## 1. Generator validation: requested target against effective density", "",
            "Effective density is brute force's own count, correlated pair-windows over tested pair-windows, in the "
            "space the cell runs in. The ratio is effective/target: 1.00 means the request was met.", ""]
    ratios = {}
    for proc, desc in PROCS:
        out += [f"### {proc} ({desc})", "",
                "| axis | level | space | target | effective | ratio |", "|---|---|---|---|---|---|"]
        for axis in sr.AXES:
            for space in ("raw", "differenced"):
                for c in sr.on_axis(cells, axis, proc, space):
                    if c["effective"] is None:
                        continue
                    r = c["effective"] / c["density"]
                    ratios.setdefault((proc, space), []).append(r)
                    out.append(f"| {axis} | {c[axis]} | {space} | {c['density']:.3f} | {c['effective']:.4f} | {r:.2f} |")
        out.append("")
    out += ["### Median effective/target ratio", "", "| process | space | median ratio | min | max | cells |",
            "|---|---|---|---|---|---|"]
    for proc, _ in PROCS:
        for space in ("raw", "differenced"):
            v = ratios.get((proc, space)) or []
            if v:
                out.append(f"| {proc} | {space} | {st.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {len(v)} |")
    out.append("")

    # 2. the four scaling curves, raw and differenced side by side
    out += ["## 2. Scaling curves as tables", "",
            "Speedup over the cell's own brute force, with recall in parentheses. The exact arms (bf_incr, TSUBASA, "
            "full-band FilCorr) have no recall knob; StatForce-style arms that emit their own decision can fall "
            "below 1 on recall and precision.", ""]
    for proc, desc in PROCS:
        for axis in sr.AXES:
            out += [f"### {proc}, {AXIS_TITLE[axis]}", "",
                    "| " + axis + " | space | effective density | BF s | " + " | ".join(NAMES) + " |",
                    "|---|---|---|---|" + "---|" * len(NAMES)]
            for space in ("raw", "differenced"):
                for c in sr.on_axis(cells, axis, proc, space):
                    row = []
                    for n in NAMES:
                        a = c["arms"][n]
                        row.append(f"{a['speedup']:.2f}x ({f(a['recall'])})" if a else "")
                    out.append(f"| {c[axis]} | {space} | {c['effective']:.4f} | {c['bf_wall']:.0f} | " + " | ".join(row) + " |")
            out.append("")

    # 3. every cell, every metric
    out += ["## 3. Per-cell recall / precision / specificity / speedup", "",
            "| process | m | L | T | target d | effective d | space | BF s | " + " | ".join(NAMES) + " |",
            "|---|---|---|---|---|---|---|---|" + "---|" * len(NAMES)]
    for c in sorted(cells, key=lambda c: (c["proc"], c["m"], c["L"], c["T"], c["density"], c["diff"])):
        row = []
        for n in NAMES:
            a = c["arms"][n]
            row.append(f"{f(a['recall'])}/{f(a['precision'])}/{f(a['specificity'])}/{a['speedup']:.2f}x" if a else "")
        out.append(f"| {c['proc']} | {c['m']} | {c['L']} | {c['T']} | {c['density']} | {c['effective']:.4f} | "
                   f"{c['space']} | {c['bf_wall']:.0f} | " + " | ".join(row) + " |")
    out.append("")

    open(out_path, "w").write("\n".join(out) + "\n")
    print(f"{len(cells)} cells -> {out_path}")
    for proc, _ in PROCS:
        for space in ("raw", "differenced"):
            v = ratios.get((proc, space)) or []
            if v:
                print(f"  {proc:4s} {space:12s} effective/target median {st.median(v):.2f} (min {min(v):.2f}, max {max(v):.2f}, n={len(v)})")


if __name__ == "__main__":
    main()
