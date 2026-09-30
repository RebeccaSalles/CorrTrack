"""What the campaign's baselines cost per pair-window, against the naive tiers the competitor
papers measure themselves against (2026-09-28, user).

    python abaca/naive_cost_table.py <campaign-root> <probe.json> [<probe.json> ...] <out.md>

The campaign never runs a naive baseline: every speedup in this project is measured against
BF_vect, the vectorised exact brute force, which is a far stronger reference. This table makes that
choice auditable by pricing all four exact tiers per pair-window on one node, then reading the
ratio back onto a campaign cell: a paper reporting "N times faster than brute force" against an
interpreted per-pair loop is reporting a different N from ours.

The probe files come from abaca/naive_baseline.py (one job, m in {25..200}), the campaign costs from
the 179 cells of the final campaign.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_figures as cf
import method_style as ms
NAMES = cf.NAMES

TIERS = [("naive_python", "naive Python: np.corrcoef per pair, per window, per lag"),
         ("naive_numpy", "naive numpy: one correlation matrix per window (counts only)"),
         ("bruteforce", "BF_vect: this project's vectorised exact arm"),
         ("bf_incremental", "BF_incr: this project's incremental five-sum arm")]
LABEL = {"naive_python": "naive Python", "naive_numpy": "naive numpy", "bruteforce": "BF_vect",
         "bf_incremental": "BF_incr"}


def campaign_costs(root):
    cells = cf.load(root)
    out = {}
    for name, key in (("BF_vect", None), ("BF_incr", "BF_incr")):
        v = [1e9 * (c["bf_wall"] if key is None else c["arms"][key]["wall"]) / c["pair_windows"]
             for c in cells if c["pair_windows"] and (key is None or c["arms"][key])]
        out[name] = (st.median(v), len(v))
    return out


def main() -> None:
    root, out_path, probes = sys.argv[1], sys.argv[-1], sys.argv[2:-1]
    camp = campaign_costs(root)
    out = ["# Per pair-window cost of the exact tiers (2026-09-28)", "",
           "Every speedup in this project is measured against `BF_vect`, the vectorised exact brute force that "
           "reruns inside each cell. The competitor papers generally measure against a naive implementation "
           "instead, so this table prices the tiers against each other on one node and one dataset, to show what "
           "our choice of baseline costs us. No experiment is run on the naive tiers; this is a cost probe "
           "(`abaca/naive_baseline.py`, one job) plus the campaign's own measured cost.", "",
           "## Measured per pair-window, one node, generated data (W=60, step=6, T=0.90)", ""]
    for path in probes:
        d = json.load(open(path))
        cfgn = d["config"]["n_lags"]
        out += [f"### {'synchronous (n_lags=0)' if not cfgn else f'lagged (n_lags={cfgn}, 6 lag probes)'}", "",
                "| m | pair-windows | " + " | ".join(LABEL[k] for k, _ in TIERS) + " | naive Python / BF_vect | naive numpy / BF_vect | BF_vect / BF_incr |",
                "|---|---|" + "---|" * (len(TIERS) + 3)]
        for r in d["rows"]:
            cells = []
            for k, _ in TIERS:
                cells.append(f"{r[k]['ns_per_pair_window']:,.0f} ns" if k in r else "not run")
            py = r["naive_python"]["ns_per_pair_window"] / r["bruteforce"]["ns_per_pair_window"] if "naive_python" in r else None
            np_ = r["naive_numpy"]["ns_per_pair_window"] / r["bruteforce"]["ns_per_pair_window"]
            inc = r["bruteforce"]["ns_per_pair_window"] / r["bf_incremental"]["ns_per_pair_window"]
            out.append(f"| {r['m']} | {r['pair_windows']:,} | " + " | ".join(cells) +
                       f" | {f'{py:,.0f}x' if py else ''} | {np_:.2f}x | {inc:.2f}x |")
        out.append("")
    # One model, one set of numbers: a cell of the median size, priced at each tier's median rate.
    # Mixing medians does not compose (the cell with the median wall clock is not the cell with the
    # median size), and a reader who divides the printed numbers must land on the printed ratios.
    per_pair = st.median([r["naive_python"]["ns_per_pair_window"] for path in probes
                          for r in json.load(open(path))["rows"] if "naive_python" in r])
    cells = cf.load(root)
    size = st.median([c["pair_windows"] for c in cells if c["pair_windows"]])

    def rate(f):
        return st.median([1e9 * f(c) / c["pair_windows"] for c in cells if c["pair_windows"] and f(c)])

    def ct_wall(c):
        return min((c["arms"][k]["wall"] for k in ("CorrTrack-LSH", "CorrTrack-Ham") if c["arms"][k]), default=None)

    def hms(t):
        return f"{t / 3600:.1f} h" if t >= 3600 else f"{t:,.0f} s"

    r_bf, r_inc, r_ct = rate(lambda c: c["bf_wall"]), rate(lambda c: c["arms"]["BF_incr"]["wall"] if c["arms"]["BF_incr"] else None), rate(ct_wall)
    out += ["## What this means for the campaign", "",
            f"The naive per-pair loop costs {per_pair:,.0f} ns per pair-window and that number does not move: it is "
            f"flat across m (25 to 100) and across lag depth, because it is per-pair work that no amount of "
            f"vectorisation amortises. The campaign's own arms fall with m instead, as the kernels fill, which is "
            f"why the gap widens exactly where the experiments live.", "",
            f"The table below prices one cell of the median size of the campaign, {size:,.0f} pair-windows, at "
            f"each tier's median rate over the 179 cells, so every column divides exactly into the ratios beside "
            f"it. Quoting median wall clocks instead would not divide, because the cell with the median wall is "
            f"not the cell with the median size; for the record those measured medians are "
            f"{hms(st.median([c['bf_wall'] for c in cells]))} for BF_vect and "
            f"{hms(st.median([ct_wall(c) for c in cells if ct_wall(c)]))} for CorrTrack's faster backend.", "",
            "| tier | ns per pair-window | that cell would take | against BF_vect |", "|---|---|---|---|",
            f"| naive Python | {per_pair:,.0f} | {hms(size * per_pair * 1e-9)} | {per_pair / r_bf:,.0f}x slower |",
            f"| BF_vect | {r_bf:.1f} | {hms(size * r_bf * 1e-9)} | 1x |",
            f"| BF_incr | {r_inc:.1f} | {hms(size * r_inc * 1e-9)} | {r_bf / r_inc:.2f}x faster |",
            f"| CorrTrack, faster backend | {r_ct:.1f} | {hms(size * r_ct * 1e-9)} | {r_bf / r_ct:.2f}x faster |", "",
            f"So the same CorrTrack run on the same data reads {r_bf / r_ct:.1f}x against our baseline and about "
            f"{per_pair / r_ct:,.0f}x against the naive one, because the second denominator is "
            f"{per_pair / r_bf:,.0f} times weaker. Every number this project reports uses the first. Against the "
            f"stricter incremental baseline the naive tier is {per_pair / r_inc:,.0f}x.", "",
            "Two things this table is not. It is not a claim that a naive implementation is the fair comparison: "
            "it is the opposite, a measurement of how much a paper's headline number owes to its baseline. And "
            "the naive numpy row is not slow at all, at 11 to 39 ns per pair-window it is faster than BF_vect, "
            "because it is one BLAS product per window that counts matches and returns nothing: no pairs emitted, "
            "no lag grid, no near-constant or spike guards, no state carried between windows. It is a floor on the "
            "arithmetic, not a baseline anyone could use, and the lagged rows show its count already diverging "
            "from the exact arms (147,152 against 156,969 at m=200) because it does not reproduce the harness's "
            "early-window lag truncation.", "",
            "## Every arm of the campaign on the same standard", "",
            f"The same reading applied to every arm: its median rate over the cells it ran in, and the same "
            f"median-size cell ({size:,.0f} pair-windows) priced at that rate. Median recall sits beside it, "
            "because a rate only compares between arms that return the same answer.", "",
            "| arm | ns per pair-window | median-size cell | against BF_vect | median recall | cells |",
            "|---|---|---|---|---|---|",
            f"| naive Python (probe) | {per_pair:,.0f} | {hms(size * per_pair * 1e-9)} | "
            f"{per_pair / r_bf:,.0f}x slower | 1.000 | probe |",
            f"| naive numpy (probe, counts only) | 11.0 | {hms(size * 11.0 * 1e-9)} | "
            f"{r_bf / 11.0:.2f}x faster | n/a | probe |"]
    for name in ["BF_vect"] + ms.lead(NAMES):
        def wall(c, n=name):
            return c["bf_wall"] if n == "BF_vect" else (c["arms"][n]["wall"] if c["arms"][n] else None)
        rs = [1e9 * wall(c) / c["pair_windows"] for c in cells if c["pair_windows"] and wall(c)]
        if not rs:          # dropped in this variant
            continue
        rec = [c["arms"][name]["recall"] for c in cells
               if name != "BF_vect" and c["arms"][name] and c["arms"][name]["recall"] is not None]
        r = st.median(rs)
        ratio = "1x" if name == "BF_vect" else (f"{r / r_bf:.2f}x slower" if r > r_bf else f"{r_bf / r:.2f}x faster")
        out.append(f"| {name} | {r:,.1f} | {hms(size * r * 1e-9)} | {ratio} | "
                   f"{st.median(rec) if rec else 1.0:.3f} | {len(rs)} |")
    # (2026-09-29, user) a pooled median compares arms over different cell sets, which is wrong as soon
    # as an arm is missing from a class: the class an arm lacks is not a random sample of the campaign.
    # Two fixes, both printed: the same rate per class and space, where every surviving arm covers the
    # same cells, and one median over the cells where every displayed arm ran.
    COLS = [(c, sp) for c in ("S", "L", "N") for sp in ("raw", "differenced")]
    present = [n for n in [ms.BF] + ms.lead(NAMES)
               if n == ms.BF or any(c["arms"].get(n) for c in cells)]
    common = [c for c in cells if all(n == ms.BF or c["arms"].get(n) for n in present)]
    out += ["", "### The same rates per class and space", "",
            "A pooled median compares arms over different sets of cells as soon as one arm is missing from a "
            "class, and a missing class is never a random sample: it is the hardest or the easiest part of the "
            "campaign. Within one class and space every surviving arm covers the same cells, so these columns "
            "are the comparable ones.", "",
            "| arm | " + " | ".join(f"{c} {sp[:4]}" for c, sp in COLS) + " | common cells |",
            "|---|" + "---|" * (len(COLS) + 1)]
    for name in present:
        vals = []
        for cls, sp in COLS:
            sl = [c for c in cells if c["cls"] == cls and c["space"] == sp
                  and (name == ms.BF or c["arms"].get(name))]
            r = [1e9 * (c["bf_wall"] if name == ms.BF else c["arms"][name]["wall"]) / c["pair_windows"]
                 for c in sl if c["pair_windows"]]
            vals.append(f"{st.median(r):,.0f}" if r else "")
        rc = [1e9 * (c["bf_wall"] if name == ms.BF else c["arms"][name]["wall"]) / c["pair_windows"]
              for c in common if c["pair_windows"] and (name == ms.BF or c["arms"].get(name))]
        out.append(f"| {name} | " + " | ".join(vals) + f" | {st.median(rc):,.1f} |" if rc else
                   f"| {name} | " + " | ".join(vals) + " |  |")
    out += ["", f"The last column is the median over the {len(common)} cells where every arm in this table ran, "
                f"which is the only column where all the numbers price the same work. Blank cells are classes "
                f"the arm does not cover.", ""]
    out += ["", "The two probe rows are not campaign runs: naive numpy is its best value, the m=200 lagged one, "
                "and it counts matches without emitting pairs, so it has no recall to report. "
                + ("An arm's cell count is the cells left after the authors-only filter: the synchronous class "
                   "for ParCorr, CSZ and CorrJoin, everything but the negative class for FilCorr, everything but "
                   "the lagged positive class for TSUBASA."
                   if ms.AUTHORS_ONLY else
                   "ParCorr, CSZ and CorrJoin cover 119 cells rather than 179 because class N has no such arm.")
                + ("" if ms.AUTHORS_ONLY else
                   " ThinBRAID's rate is not comparable with the rest: at median recall 0.815 and precision 0.046 "
                   "it is pricing a different and much larger answer."), ""]
    out = ms.banner(out)
    open(out_path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
