"""Rankings-only view of the m=500 tables, with a 10% tie band on the speedups.

Per row (capability class, space, threshold) each arm gets [f/s]: the datasets it is fastest in
and the datasets it is among the two fastest in, out of the six. The winner is then compared with
the best OTHER arm on each dataset separately, which is the comparison that decides whether a win
is clear: margin is the geometric mean of those per-dataset ratios, spread is the geometric
standard deviation of the same ratios, and speedups within 10% of each other are treated as ties.
"""
import re, sys, math, statistics as st, collections

SRC, OUT = sys.argv[1], sys.argv[2]
TARGET = 0.95
TIE = 0.10          # (2026-09-23, user) arms within 10% of the best count as tied for that place
ARMS = ["STOMP", "FilCorr", "TSUBASA", "BRAID", "ThinBRAID", "CorrTrack", "ParCorr", "CSZ", "StatStream", "CorrJoin"]
# (2026-09-23, user) CT-lsh and CT-ham are two tuned backends of one method, so they are ranked as one
# arm whose speedup on a dataset is the better of the two (the better QUALIFYING one under the recall
# filter). Counting them separately would let one method occupy two places and split its own wins.
COLLAPSE = {"CT-lsh": "CorrTrack", "CT-ham": "CorrTrack"}

def parse(src):
    out, cls, space, header = collections.defaultdict(list), None, None, None
    for line in open(src):
        m = re.match(r"^## Table ([SLN]), (raw|differenced):", line)
        if m: cls, space, header = m.group(1), m.group(2), None; continue
        if cls and line.startswith("| dataset |"): header = [h.strip() for h in line.strip().strip("|").split("|")]; continue
        if cls and header and line.startswith("| ") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) != len(header) or c[2] == "": continue
            vals = {}
            for h, cell in zip(header[4:-1], c[4:-1]):
                p = cell.split("/")
                if len(p) == 4 and p[3].endswith("x"): vals[h] = (float(p[0]), float(p[3][:-1]))   # (recall, speedup)
            out[(cls, space)].append((c[0].rstrip("*"), c[1], vals))
    return out

def row_stats(rows, T, qualified):
    grp = [r for r in rows if r[1] == T]
    per = []           # per dataset: {arm: speedup} after the qualification filter
    for ds, t, vals in grp:
        pool = {}
        for a, (rec, sp) in vals.items():
            if qualified and rec < TARGET: continue
            name = COLLAPSE.get(a, a)
            if name not in pool or sp > pool[name]: pool[name] = sp
        if pool: per.append((ds, pool))
    counts = {a: [0, 0] for a in ARMS}
    for ds, pool in per:
        if not pool: continue
        best = max(pool.values())
        first = [a for a, v in pool.items() if v >= best * (1 - TIE)]      # tied for first
        for a in first: counts[a][0] += 1
        rest = {a: v for a, v in pool.items() if a not in first}
        second = []
        if rest:
            b2 = max(rest.values())
            second = [a for a, v in rest.items() if v >= b2 * (1 - TIE)]
        for a in set(first) | set(second): counts[a][1] += 1               # first or second place, ties included
    if not per: return counts, None
    # winner: most firsts, ties broken by the median speedup
    cand = max(counts, key=lambda a: (counts[a][0], st.median([p[a] for _, p in per if a in p]) if any(a in p for _, p in per) else 0))
    ratios, clear, tied = [], 0, 0
    for ds, pool in per:
        if cand not in pool or len(pool) < 2: continue
        best_other = max(v for a, v in pool.items() if a != cand)
        r = pool[cand] / best_other
        ratios.append(r)
        if r >= 1 + TIE: clear += 1                                        # ahead by more than the tie band
        elif r > 1 - TIE: tied += 1                                        # inside the band
    if len(ratios) < 3: return counts, None
    logs = [math.log(r) for r in ratios]
    gm, gsd = math.exp(st.mean(logs)), (math.exp(st.stdev(logs)) if len(logs) > 1 else 1.0)
    verdict = ("clear" if clear >= len(ratios) - tied and clear > len(ratios) / 2 else
               ("tied" if clear + tied == len(ratios) else "mixed"))
    return counts, dict(arm=cand, n=len(ratios), clear=clear, tied=tied, gm=gm, gsd=gsd,
                        lo=min(ratios), hi=max(ratios), verdict=verdict)

data = parse(SRC)
L = ["# m=500 tables: rankings, margins and paired tests", "",
     "`[f/s]` = datasets of that row where the arm is the fastest / among the two fastest, out of six. "
     "CorrTrack is one arm here: its value on a dataset is the better of its two tuned backends. "
     "The winner of a row is the arm with the most firsts, ties broken by the median speedup. It is then "
     "compared with the best other arm on each dataset separately: **margin** is the geometric mean of the six "
     "ratios, **spread** their geometric standard deviation (1.00 would mean the same ratio on every dataset), "
     "and **range** their smallest and largest value. A margin of 1.30 with a spread of 1.05 is a uniform win; "
     "the same margin with a spread of 1.60 means the win rests on one or two datasets.", "",
     "Speedups within 10% of each other count as tied, so `[f/s]` can exceed one arm per place and a row can "
     "have several arms tied for first. **ahead/tied/behind** counts the datasets where the leader is more than "
     "10% above the best other arm, within 10% of it, or more than 10% below. The verdict follows: *clear* when "
     "the leader is ahead on a majority of datasets and never behind, *tied* when every dataset is inside the "
     "band, *mixed* when it wins some and loses others.", ""]

for qualified in (False, True):
    L += [f"## {'Only arms reaching recall %.2f' % TARGET if qualified else 'Every arm that ran'}", "",
          "| table | space | T | " + " | ".join(ARMS) + " | leader | margin | spread | range | ahead/tied/behind | verdict |",
          "|---|---|---|" + "---|" * (len(ARMS) + 6)]
    for (cls, space), rows in sorted(data.items()):
        for T in ("0.7", "0.8", "0.9", "0.95"):
            counts, w = row_stats(rows, T, qualified)
            if not any(c[1] for c in counts.values()): continue
            present = {COLLAPSE.get(a, a) for _, _, v in rows for a in v}
            cells = [(f"{counts[a][0]}/{counts[a][1]}" if a in present else "") for a in ARMS]
            if w:
                tail = (f"| **{w['arm']}** | {w['gm']:.2f}x | {w['gsd']:.2f} | {w['lo']:.2f}-{w['hi']:.2f}x | "
                        f"{w['clear']}/{w['tied']}/{w['n'] - w['clear'] - w['tied']} | {w['verdict']} |")
            else:
                tail = "| | | | | | |"
            L.append(f"| {cls} | {space} | {T} | " + " | ".join(cells) + " " + tail)
    L.append("")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L[:8]))
