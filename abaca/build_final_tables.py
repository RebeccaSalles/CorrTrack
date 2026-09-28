"""Rebuild the m=500 capability tables from one campaign root (2026-09-25).

Every cell holds every arm, measured in one job from one snapshot, so each row's speedups are
ratios against the brute force of that same job. Writes the per-cell tables, the wide median
table with the density column, and the per-threshold medians.
"""
import json, glob, os, sys, statistics as st, collections
ROOT, OUT = sys.argv[1], sys.argv[2]
DS = [("streamflow_m500_m500", "streamflow"), ("sp500_m500_m444", "sp500"), ("wikipedia_m500_m500", "wikipedia"),
      ("smartmeter_m500_m500", "smartmeter"), ("global_weather_m500_m500", "global_weather"), ("acwi_capweighted_m500_m500", "acwi")]
THR = ["0.7", "0.8", "0.85", "0.9", "0.95"]
TABLES = [("S", "synchronous, positive (n_lags=0, neg_corr=False)", "L0", "pos"),
          ("L", "lagged, positive", "L15", "pos"), ("N", "lagged, negative", "L15", "neg")]
ORDER = [("bf_incremental", "bf_incr"), ("filcorr", "FilCorr"), ("tsubasa", "TSUBASA"), ("braid", "BRAID"),
         ("thinbraid", "ThinBRAID"), ("corrtrack", "CT-lsh"), ("corrtrack_hamming", "CT-ham"),
         ("parcorr", "ParCorr"), ("csz", "CSZ"), ("statstream", "StatStream"), ("corrjoin", "CorrJoin")]
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
       "One code state for every arm and cell: snapshot `101ef2c` / `content=f7aa10f1814d9eee`, all twelve arms in "
       "one job per cell, brute force rerun in each as both ground truth and timing anchor, monitoring off, every "
       "arm evaluated on the whole stream while its parameters were chosen on the first 30%. CorrTrack is tuned by "
       "its own hyperopt (sketch width fixed at 64), the four pruning competitors by the CSZ protocol, "
       "BRAID/ThinBRAID at published defaults, and the exact arms have no recall knob.", "",
       "Cell entries are recall/precision/specificity/speedup. One arm is absent from one cell: ThinBRAID "
       "exhausted the node's 192 GB in acwi lagged negative at T=0.70, the densest cell of the study (17.9% of "
       "pair-windows correlated, 224 million of them), where its reported set plus the metrics temporaries do not "
       "fit; the other eleven arms of that cell ran normally, brute force included.", "",
       "**Legend.** `†` published defaults, not tuned: BRAID and ThinBRAID (the CSZ protocol does not apply to "
       "them; TSUBASA, bf_incremental and full-band FilCorr are exact and have no recall knob). `‡` the capability "
       "is ours, not the authors': ParCorr, CSZ and CorrJoin reach class L, and TSUBASA classes L and N, through a "
       "lag extension of ours, and FilCorr handles negative correlation in class N only because we enabled it (the "
       "runs record supports_lags / supports_neg_corr = enabled_by_us). `¶` specified in the paper but never "
       "evaluated there: StatStream's lags in classes L and N, and the negative correlation of BRAID, ThinBRAID "
       "and StatStream in class N. `§` emits its own decision instead of a validated set, so precision and "
       "specificity can fall below 1: StatStream and ThinBRAID do so in every cell, while BRAID, in the same "
       "family by design, reports exactly the brute-force set on these lag grids.", ""]
summ, dens_by = [], {}
for code, title, lag, tag in TABLES:
    for space, sp_tag in (("raw", ""), ("differenced", "_diff")):
        out += [f"## Table {code}, {space}: {title}", "",
                "| dataset | T | density | BF s | " + " | ".join(n for _, n in ORDER) + " | tuning s (lsh/ham/CSZ) |",
                "|---|---|---|---|" + "---|"*len(ORDER) + "---|"]
        for stem_ds, label in DS:
            for T in THR:
                W = ("W48_s8_L16" if lag == "L15" else "W48_s8_L0") if "smartmeter" in stem_ds else f"W30_s3_{lag}"
                run = f"{stem_ds}_{W}_T{T}{sp_tag}_{tag}"
                p = os.path.join(ROOT, "nway", run, "nway.json")
                if not os.path.exists(p):
                    out.append(f"| {label} | {T} | | *not available* |" + " |"*len(ORDER) + " |"); continue
                d = json.load(open(p)); arms = d["arms"]; bf = arms["bruteforce"]
                dn = (bf.get("correlated") or 0)/(bf.get("total_candidates") or bf.get("tested") or 1)
                dens_by.setdefault((code, space, T), []).append(dn)
                cells = []
                for key, name in ORDER:
                    r = arms.get(key) or arms.get("exact_stomp" if key == "bf_incremental" else key)
                    if not isinstance(r, dict): cells.append(""); continue
                    if r.get("status") != "ok": cells.append("N/A" if r.get("status") == "N/A" else "ERR"); continue
                    s = spec(r, bf); spd = bf["wall"]/r["wall"] if r.get("wall") else None
                    cells.append(f"{f(r.get('recall'))}/{f(r.get('precision'))}/{f(s)}/{f(spd,2)}x")
                    summ.append((code, space, T, label, name, r.get("recall"), r.get("precision"), s, spd))
                tw = tuple(elapsed(os.path.join(ROOT, k, run, "time_v.txt")) for k in ("hyperopt", "hyperopt_hamming", "tuned"))
                out.append(f"| {label} | {T} | {dn:.2e} | {bf['wall']:.0f} | " + " | ".join(cells) +
                           f" | {f(tw[0],0)}/{f(tw[1],0)}/{f(tw[2],0)} |")
        out.append("")
out += ["## Overall medians: one row per table, space and threshold, one column per arm", "",
        "Median speedup over brute force (median recall) across the six datasets.", "",
        "| table | space | T | density | " + " | ".join(n for _, n in ORDER) + " |",
        "|---|---|---|---|" + "---|"*len(ORDER)]
for code, _, _, _ in TABLES:
    for space in ("raw", "differenced"):
        for T in THR:
            g0 = [x for x in summ if x[:3] == (code, space, T)]
            if not g0: continue
            cs = []
            for _, name in ORDER:
                g = [x for x in g0 if x[4] == name]
                sp = [x[8] for x in g if x[8]]; rc = [x[5] for x in g if x[5] is not None]
                cs.append(f"{st.median(sp):.2f}x ({st.median(rc):.3f})" if sp and rc else "")
            dl = dens_by.get((code, space, T), [])
            out.append(f"| {code} | {space} | {T} | {st.median(dl):.1e} | " + " | ".join(cs) + " |")
out.append("")
open(OUT, "w").write("\n".join(out) + "\n")
print("\n".join(out[-28:]))
