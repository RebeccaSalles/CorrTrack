"""Rebuild the wide median table from the interim markdown and add per-dataset rank counts.

Ranking by the median over the six datasets can hide a method that wins on most datasets but is
dragged by one, and can crown a method that is merely middling everywhere. So each cell also
carries how many of that row's datasets the method ranks first in, and how many it is in the top
two of. Two rankings are produced: over every arm that ran, and over the arms that reach the
tuned recall target (0.95), since an arm that buys speed by missing pairs otherwise "wins" rows.
"""
import re, sys, statistics as st, collections

SRC, OUT = sys.argv[1], sys.argv[2]
TARGET = 0.95
ARMS_ORDER = ["STOMP", "FilCorr", "TSUBASA", "BRAID", "ThinBRAID", "CT-lsh", "CT-ham", "ParCorr", "CSZ", "StatStream", "CorrJoin"]

def parse(src):
    """(class, space) -> list of rows; each row = (dataset, T, density, bf_s, {arm: (rec, prec, spec, speedup)})"""
    out, cls, space, header = collections.defaultdict(list), None, None, None
    for line in open(src):
        m = re.match(r"^## Table ([SLN]), (raw|differenced):", line)
        if m:
            cls, space, header = m.group(1), m.group(2), None
            continue
        if cls and line.startswith("| dataset |"):
            header = [h.strip() for h in line.strip().strip("|").split("|")]
            continue
        if cls and header and line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != len(header) or cells[2] == "":
                continue
            ds, T = cells[0].rstrip("*"), cells[1]
            vals = {}
            for h, c in zip(header[4:-1], cells[4:-1]):
                p = c.split("/")
                if len(p) == 4 and p[3].endswith("x"):
                    vals[h] = tuple(float(x) for x in p[:3]) + (float(p[3][:-1]),)
            out[(cls, space)].append((ds, T, cells[2], cells[3], vals))
    return out

def ranks(rows, arm, T, qualified):
    """(n_first, n_top2, n_datasets) for one arm within one (class, space, T) group"""
    n1 = n2 = n = 0
    for ds, t, dens, bf, vals in rows:
        if t != T or arm not in vals:
            continue
        pool = {a: v for a, v in vals.items() if (not qualified or v[0] >= TARGET)}
        if arm not in pool:
            n += 1
            continue
        order = sorted(pool, key=lambda a: -pool[a][3])
        n += 1
        if order[0] == arm: n1 += 1
        if arm in order[:2]: n2 += 1
    return n1, n2, n

data = parse(SRC)
lines = ["# m=500 tables: medians with per-dataset rank counts", "",
         "Each cell: median speedup over the six datasets (median recall) followed by `[f/s]`, the number of "
         "datasets in that row where the method is the fastest and where it is among the two fastest. The "
         "median and the counts answer different questions: a method can hold the best median without winning "
         "most datasets, and a method that wins four of six can trail on the median because of one hard set.", "",
         "Two rankings are given. The first ranks every arm that ran. The second, marked `qualified`, ranks only "
         f"the arms whose recall reaches the tuned target of {TARGET:.2f} in that dataset, so an arm cannot win a "
         "row by skipping the work: ThinBRAID and StatStream miss most true pairs in most cells, and the exact "
         "arms and CorrTrack are the ones that consistently qualify.", ""]

for qualified in (False, True):
    lines += [f"## {'Qualified ranking (only arms reaching recall >= %.2f)' % TARGET if qualified else 'Ranking over every arm that ran'}", ""]
    lines += ["| table | space | T | density | " + " | ".join(ARMS_ORDER) + " |", "|---|---|---|---|" + "---|" * len(ARMS_ORDER)]
    for (cls, space), rows in sorted(data.items()):
        for T in ("0.7", "0.8", "0.9", "0.95"):
            grp = [r for r in rows if r[1] == T]
            if not grp: continue
            dens = st.median([float(r[2]) for r in grp])
            cells = []
            for arm in ARMS_ORDER:
                sp = [r[4][arm][3] for r in grp if arm in r[4]]
                rc = [r[4][arm][0] for r in grp if arm in r[4]]
                if not sp: cells.append(""); continue
                n1, n2, n = ranks(rows, arm, T, qualified)
                cells.append(f"{st.median(sp):.2f}x ({st.median(rc):.3f}) [{n1}/{n2}]")
            lines.append(f"| {cls} | {space} | {T} | {dens:.1e} | " + " | ".join(cells) + " |")
    lines.append("")
    # who wins how often overall
    tot = collections.Counter(); tot2 = collections.Counter(); seen = collections.Counter()
    for (cls, space), rows in data.items():
        for T in ("0.7", "0.8", "0.9", "0.95"):
            for arm in ARMS_ORDER:
                n1, n2, n = ranks(rows, arm, T, qualified)
                tot[arm] += n1; tot2[arm] += n2; seen[arm] += n
    lines += [f"Totals over all {sum(seen.values()) // max(1, len(ARMS_ORDER))} dataset-cells of every row:", "",
              "| arm | fastest | in the top two | cells ranked |", "|---|---|---|---|"]
    for arm in ARMS_ORDER:
        if seen[arm]: lines.append(f"| {arm} | {tot[arm]} | {tot2[arm]} | {seen[arm]} |")
    lines.append("")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:14]))
