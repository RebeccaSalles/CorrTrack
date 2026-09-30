"""Rebuild the m=500 capability tables from one campaign root (2026-09-25).

Every cell holds every arm, measured in one job from one snapshot, so each row's speedups are
ratios against the brute force of that same job. Writes the per-cell tables, the wide median
table with the density column, and the per-threshold medians.
"""
import json, glob, os, sys, statistics as st, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import method_style as ms
ROOT, OUT = sys.argv[1], sys.argv[2]
DS = [("streamflow_m500_m500", "streamflow"), ("sp500_m500_m444", "sp500"), ("wikipedia_m500_m500", "wikipedia"),
      ("smartmeter_m500_m500", "smartmeter"), ("global_weather_m500_m500", "global_weather"), ("acwi_capweighted_m500_m500", "acwi")]
THR = ["0.7", "0.8", "0.85", "0.9", "0.95"]
TABLES = [("S", "synchronous, positive only (n_lags=0, neg_corr=False)", "L0", "pos"),
          ("L", "lagged, positive only (neg_corr=False)", "L15", "pos"),
          ("N", "lagged, both signs (neg_corr=True: negative correlation searched as well as positive)", "L15", "neg")]
ORDER = ms.ARMS          # arm keys and labels, shared with the figure builders
SHOWN = [(k, n) for k, n in ORDER if ms.kept_arm(n)]
SHOWN_ARMS = [n for _, n in SHOWN]
def elapsed(p):
    try:
        for l in open(p):
            if "Elapsed (wall clock)" in l:
                t = [float(x) for x in l.split(": ")[-1].strip().split(":")]
                return t[0]*3600+t[1]*60+t[2] if len(t) == 3 else t[0]*60+t[1]
    except OSError: return None
def spec(r, bf):
    U = bf.get("total_candidates") or bf.get("tested") or 0; P = bf.get("correlated") or 0
    if r.get("recall") is None or U <= P: return None
    return 1.0 - max((r.get("correlated") or 0) - r["recall"]*P, 0.0)/(U-P)
def f(x, d=3): return "" if x is None else f"{x:.{d}f}"
out = ["# m=500 capability tables (final campaign, 2026-09-25)", "",
       f"One code state for every arm and cell: snapshot `101ef2c` / `content=f7aa10f1814d9eee`, all "
       f"{len(SHOWN_ARMS) + 1} arms in "
       "one job per cell, brute force rerun in each as both ground truth and timing anchor, monitoring off, every "
       "arm evaluated on the whole stream while its parameters were chosen on the first 30%. CorrTrack is tuned by "
       "its own hyperopt (sketch width fixed at 64), the four pruning competitors by the CSZ protocol, "
       + ("BRAID at published defaults" if ms.AUTHORS_ONLY else "BRAID/ThinBRAID at published defaults")
       + ", and the exact arms have no recall knob.", "",
       "**Naming.** `BF_vect` is the vectorised exact brute force, which is also the ground truth and the "
       "timing anchor of its own cell; `BF_incr` is the exact incremental baseline; `CorrTrack-LSH` and "
       "`CorrTrack-Ham` are CorrTrack's two tuned backends. Every speedup is that cell's `BF_vect` wall clock "
       "divided by the arm's, measured in the same job.", "",
       "Cell entries are recall/precision/specificity/speedup."
       + ("" if ms.AUTHORS_ONLY else
          " One arm is absent from one cell: ThinBRAID exhausted the node's 192 GB in acwi lagged negative at "
          "T=0.70, the densest cell of the study (17.9% of pair-windows correlated, 224 million of them), where "
          "its reported set plus the metrics temporaries do not fit; the other eleven arms of that cell ran "
          "normally, brute force included."), "",
       "**Legend.** `†` published defaults, not tuned: BRAID" + ("" if ms.AUTHORS_ONLY else " and ThinBRAID") +
       " (the CSZ protocol does not apply to "
       "them; TSUBASA, bf_incremental and full-band FilCorr are exact and have no recall knob). `‡` the capability "
       "is ours, not the authors': ParCorr, CSZ and CorrJoin reach class L, and TSUBASA classes L and N, through a "
       "lag extension of ours, and FilCorr handles negative correlation in class N only because we enabled it (the "
       "runs record supports_lags / supports_neg_corr = enabled_by_us). `¶` specified in the paper but never "
       "evaluated there: StatStream's lags in classes L and N, and the negative correlation of BRAID"
       + ("" if ms.AUTHORS_ONLY else ", ThinBRAID") + " and StatStream in class N. `§` emits its own decision "
       "instead of a validated set, so precision and specificity can fall below 1: StatStream"
       + ("" if ms.AUTHORS_ONLY else " and ThinBRAID") + " does so in every cell, while BRAID, in the same "
       "family by design, reports exactly the brute-force set on these lag grids.", ""]
summ, dens_by = [], {}
for code, title, lag, tag in TABLES:
    for space, sp_tag in (("raw", ""), ("differenced", "_diff")):
        out += [f"## Table {code}, {space}: {title}", "",
                "| dataset | T | density | BF_vect s | " + " | ".join(n for _, n in ORDER if ms.kept_arm(n)) + " | tuning s (lsh/ham/CSZ) |",
                "|---|---|---|---|" + "---|"*sum(1 for _, n in ORDER if ms.kept_arm(n)) + "---|"]
        for stem_ds, label in DS:
            for T in [t for t in THR if ms.kept_T(t)]:
                W = ("W48_s8_L16" if lag == "L15" else "W48_s8_L0") if "smartmeter" in stem_ds else f"W30_s3_{lag}"
                run = f"{stem_ds}_{W}_T{T}{sp_tag}_{tag}"
                p = os.path.join(ROOT, "nway", run, "nway.json")
                if not os.path.exists(p):
                    out.append(f"| {label} | {T} | | *not available* |" + " |"*len(ORDER) + " |"); continue
                d = json.load(open(p)); arms = d["arms"]; bf = arms["bruteforce"]
                dn = (bf.get("correlated") or 0)/(bf.get("total_candidates") or bf.get("tested") or 1)
                dens_by.setdefault((code, space, T), []).append(dn)
                cells = []
                for key, name in [kv for kv in ORDER if ms.kept_arm(kv[1])]:
                    r = arms.get(key) or arms.get("exact_stomp" if key == "bf_incremental" else key)
                    if not isinstance(r, dict): cells.append(""); continue
                    if r.get("status") != "ok": cells.append("N/A" if r.get("status") == "N/A" else "ERR"); continue
                    tier = r.get("supports_neg_corr") if tag == "neg" else (r.get("supports_lags") if lag != "L0" else "native")
                    if not ms.keeps(tier): cells.append(""); continue     # authors-only mode
                    s = spec(r, bf); spd = bf["wall"]/r["wall"] if r.get("wall") else None
                    cells.append(f"{f(r.get('recall'))}/{f(r.get('precision'))}/{f(s)}/{f(spd,2)}x")
                    summ.append((code, space, T, label, name, r.get("recall"), r.get("precision"), s, spd))
                tw = tuple(elapsed(os.path.join(ROOT, k, run, "time_v.txt")) for k in ("hyperopt", "hyperopt_hamming", "tuned"))
                out.append(f"| {label} | {T} | {dn:.2e} | {bf['wall']:.0f} | " + " | ".join(cells) +
                           f" | {f(tw[0],0)}/{f(tw[1],0)}/{f(tw[2],0)} |")
        out.append("")

out += ["## Overall medians: one row per table, space and threshold, one column per arm", "",
        "Each entry is the MEDIAN over the six datasets of that row: median speedup over `BF_vect`, with the "
        "median recall in brackets. The density column is the median effective density of the same six cells.", "",
        "| table | space | T | density | " + " | ".join(n for _, n in SHOWN) + " |",
        "|---|---|---|---|" + "---|"*len(SHOWN)]
for code, _, _, _ in TABLES:
    for space in ("raw", "differenced"):
        for T in [t for t in THR if ms.kept_T(t)]:
            g0 = [x for x in summ if x[:3] == (code, space, T)]
            if not g0: continue
            cs = []
            for _, name in SHOWN:
                g = [x for x in g0 if x[4] == name]
                sp = [x[8] for x in g if x[8]]; rc = [x[5] for x in g if x[5] is not None]
                cs.append(f"{st.median(sp):.2f}x ({st.median(rc):.3f})" if sp and rc else "")
            dl = dens_by.get((code, space, T), [])
            out.append(f"| {code} | {space} | {T} | {st.median(dl):.1e} | " + " | ".join(cs) + " |")
out.append("")
open(OUT, "w").write("\n".join(ms.banner(out)) + "\n")
print("\n".join(out[-28:]))
