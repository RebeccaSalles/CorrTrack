"""Diagnostic of the CANDIDATE SELECTION downstream of the index.

Problem addressed: "after the index (bst, …), the candidate selection filters
far too little" — the index returns almost as many pairs as brute-force, so the
validation (Pearson) pays nearly the O(n²) cost *on top of* the index cost.

This script runs, SERIALLY (compute backend of your choice, never
sharded/parallel), the full corrtrack pipeline on ONE given dataset and compares
it to brute-force (ground truth) to quantify what the index actually filters:

    keep%(vs bf) = index_candidates / bf_candidates   ← close to 100% = filters nothing

Several indexes and/or several query radii can be swept to find the setting that
filters decisively WHILE keeping recall.

Examples
--------
    # bst on a CSV, vectorized backend, serial, radius sweep
    python -m v2.diagnose_select data.csv --index bst \
        --query-radius 2,4,6,8,10 --n-series 5 --n-years 1

    # compare several indexes at a fixed radius
    python -m v2.diagnose_select data.csv --index bst,bptree,kdtree,grid \
        --query-radius 6 --backend cython

    # smoke test without real data (synthetic correlated dataset)
    python -m v2.diagnose_select --synth --index bst --query-radius 2,4,8,16

    # WHY "almost everything passes": sketch separability on YOUR dataset
    python -m v2.diagnose_select data.csv --separability --n-series 25 --neg-corr false

    # reproduce the dense climate case (everything correlates) synthetically
    python -m v2.diagnose_select --synth --synth-shared 6 --separability

The table shows the FUNNEL: index candidates → keep% vs bf → cosine filtering
(cos%) → Pearson-validated pairs → val% (share passing the validation) →
recall/precision vs bf → select/validate time. A `val%` close to 100% means
"almost everything passes validation" (dense data).

`--separability` answers "why does it not filter": it measures whether the
sketch cosine separates the correlated pairs from the uncorrelated ones (AUC),
the base rate (share of truly correlated pairs) and the ideal threshold for the
target recall. AUC≈0.5 or an ideal keep ≥50% → the sketch/the data is the
bottleneck, not the index.
"""

import argparse
import os
import sys
from dataclasses import fields

from .backends import available, base_of, is_sharded
from .core import pipeline
from .core.config import Config, _coerce
from .core.log import get_logger, reset
from .core.metrics import compute_metrics

# Config fields exposed as passthrough options; these two are "sweepable" and
# driven by --index / --query-radius, so they are removed from the passthrough.
_SWEEP_FIELDS = {"index_backend", "query_radius"}


# --------------------------------------------------------------------------- #
# ANSI colors (like the rest of v2) — disabled outside a terminal
# --------------------------------------------------------------------------- #
_TTY = sys.stdout.isatty()


def _c(txt, code):
    return f"\033[{code}m{txt}\033[0m" if _TTY else str(txt)


def _keep_color(pct):
    """keep% (index/bf): green = filters well, red = filters almost nothing."""
    if pct != pct:                      # nan
        return "  -  "
    s = f"{pct:5.1f}%"
    if pct >= 50:
        return _c(s, "1;31")            # bold red — the index is useless
    if pct >= 20:
        return _c(s, "33")             # yellow
    return _c(s, "32")                 # vert


def _recall_color(r, target):
    if r != r:
        return "  -  "
    s = f"{r:5.3f}"
    return _c(s, "32") if r >= target else _c(s, "31")


# --------------------------------------------------------------------------- #
# generation of a small synthetic correlated dataset (for --synth)
# --------------------------------------------------------------------------- #
def _make_synth_csv(path, n_series=8, n_obs=24 * 90, n_templates=3, noise=0.35,
                    shared=0.0, seed=7):
    """Write a `time, s0, s1, …` CSV.

    Groups of series sharing a pattern (strong correlations) + noise. `shared`
    (>0) adds a COMMON SEASONALITY to every series — reproducing the climate case
    (ASOS) where every series shares the seasonal cycle, so almost every pair
    correlates ("almost everything passes validation")."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n_obs)
    # fast COMMON cycle (daily + weekly): it varies WITHIN a 7-day window, so it
    # pushes (almost) every pair to correlate — the ASOS climate case.
    season = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / (24 * 7))
    templates = [np.sin(2 * np.pi * t / (24 * (7 + 5 * k)) + k) for k in range(n_templates)]
    cols = {}
    for i in range(n_series):
        base = templates[i % n_templates]
        sign = 1.0 if (i // n_templates) % 2 == 0 else -1.0   # a few anti-correlated ones
        cols[f"s{i}"] = (shared * season + sign * base
                         + noise * rng.standard_normal(n_obs))
    import pandas as pd
    df = pd.DataFrame({"time": t, **cols})
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# extraction of a run's measurements
# --------------------------------------------------------------------------- #
def _phase_times(result):
    """Per-phase v2 times from result['timings'] (dict name -> seconds)."""
    tm = result.get("timings", {}) or {}
    return {
        "sketch": tm.get("sketch", 0.0) + tm.get("index", 0.0),
        "select": tm.get("select", 0.0),
        "valsk": tm.get("validate_sketches", 0.0),
        "validate": tm.get("validate", 0.0),
        "candidates": tm.get("candidates", 0.0),   # bf
        "total": result.get("runtime", 0.0),
    }


def _run(config, path, mode):
    """One full run (up to monitor); returns result + per-phase times."""
    res = pipeline.run(config, path, mode=mode, step="all")
    return res, _phase_times(res)


# --------------------------------------------------------------------------- #
# tableau
# --------------------------------------------------------------------------- #
_HDR = ("index", "radius", "idx_cands", "keep%(bf)", "aft_cos", "cos%",
        "pearson", "val%", "recall", "prec", "sel_t", "val_t", "tot_t", "×bf")
_FMT = ("{:<9} {:>7} {:>11} {:>10} {:>10} {:>6} {:>9} {:>5} {:>8} {:>6} "
        "{:>7} {:>7} {:>7} {:>6}")


def _print_header():
    print(_FMT.format(*_HDR))
    print(_c("-" * 118, "2"))


def _print_bf(bf_res, bf_t):
    n = bf_res.get("n_candidates", 0)
    corr = bf_res.get("n_correlated", 0)
    print(_c(f"[bruteforce] candidate universe = {n}  |  correlated (truth) = {corr}  "
             f"|  cand_t={bf_t['candidates']:.3f}s  val_t={bf_t['validate']:.3f}s  "
             f"total={bf_t['total']:.3f}s", "1;36"))


def _print_row(index, radius, ct, ct_t, bf_res, target_recall):
    idx_cands = ct.get("n_candidates", 0)
    aft_cos = ct.get("n_tested", 0)
    pearson = ct.get("n_correlated", 0)
    bf_cands = bf_res.get("n_candidates", 0) if bf_res else 0
    bf_total = bf_res.get("runtime", 0.0) if bf_res else 0.0

    keep = 100.0 * idx_cands / bf_cands if bf_cands else float("nan")
    cos_keep = 100.0 * aft_cos / idx_cands if idx_cands else float("nan")
    val_keep = 100.0 * pearson / aft_cos if aft_cos else float("nan")   # passe Pearson
    speedup = (bf_total / ct_t["total"] if (bf_res is not None and ct_t["total"])
               else float("nan"))

    recall = prec = float("nan")
    if bf_res is not None:
        m = compute_metrics(bf_res.get("correlated", []),
                            ct.get("correlated", []), bf_cands)
        recall, prec = m["recall"], m["precision"]

    print(_FMT.format(
        index, str(radius), idx_cands, _keep_color(keep), aft_cos,
        f"{cos_keep:4.0f}%" if cos_keep == cos_keep else "  -  ",
        pearson, f"{val_keep:3.0f}%" if val_keep == val_keep else "  -  ",
        _recall_color(recall, target_recall),
        f"{prec:5.3f}" if prec == prec else "  -  ",
        f"{ct_t['select']:.3f}", f"{ct_t['validate']:.3f}",
        f"{ct_t['total']:.3f}",
        f"{speedup:4.1f}" if speedup == speedup else "  -  "))
    return {"index": index, "radius": radius, "keep": keep,
            "recall": recall, "prec": prec, "total": ct_t["total"]}


# --------------------------------------------------------------------------- #
# sketch separability: does the sketch cosine tell correlated pairs apart from
# uncorrelated ones? (the REAL bottleneck when "everything passes")
# --------------------------------------------------------------------------- #
def _pct(a, qs=(5, 25, 50, 75, 95)):
    import numpy as np
    if a.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(a, q)) for q in qs}


def _auc(pos, neg):
    """AUC = P(cos of a correlated pair > cos of an uncorrelated one). Ties = 0.5.

    0.5 = the sketch separates NOTHING; 1.0 = perfect separation. Exact
    (searchsorted)."""
    import numpy as np
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ns = np.sort(neg)
    less = np.searchsorted(ns, pos, side="left")
    lesseq = np.searchsorted(ns, pos, side="right")
    return float((less + 0.5 * (lesseq - less)).sum() / (pos.size * neg.size))


def analyze_separability(config, path, cap_pairs=200_000, n_win=20, seed=12345):
    """Measure, over a sample of pairs, whether the sketch cosine separates the
    correlated pairs (Pearson≥threshold, ground truth) from the uncorrelated
    ones. Reuses EXACTLY the pipeline functions (same cosine/correlate
    backends)."""
    import numpy as np

    from .backends import get_backend
    from .steps.io import read_csv, subset
    from .steps.selection import enumerate_candidates
    from .steps.sketch import make_sketcher
    from .steps.windows import iter_windows

    sk_be = get_backend(config.backend_for("sketch"))
    cand_be = get_backend(config.backend_for("candidate"))
    val_be = get_backend(config.backend_for("validate"))

    ids, times, data = read_csv(path)
    if config.n_series or config.n_years or config.train_ratio < 1.0:
        ids, times, data = subset(ids, times, data, config.n_series,
                                  config.n_years, config.obs_mode, config.train_ratio)
    windows = list(iter_windows(data, times, config.window_size, config.window_step))
    if not windows:
        print("separability: no window"); return
    windows = windows[:min(len(windows), n_win)]

    sketcher = make_sketcher(config, sk_be)
    store = {}
    for win in windows:
        for i, sid in enumerate(ids):
            if config.std_threshold > 0.0 and float(win.block[i].std()) < config.std_threshold:
                continue
            key = (sid, win.start_time)
            store[key] = {"time": win.start_time, "raw": win.block[i],
                          "sketch": sketcher.sketch(sid, win.start_index, win.block[i])}

    pairs = enumerate_candidates(store, list(store.keys()), config.n_lags)
    n_all = len(pairs)
    if n_all == 0:
        print("separability: no pair (n_lags too small?)"); return
    rng = np.random.default_rng(seed)
    if n_all > cap_pairs:
        pairs = [pairs[i] for i in rng.choice(n_all, size=cap_pairs, replace=False).tolist()]

    cos = np.asarray(cand_be.cosine_batch(
        [(store[a]["sketch"], store[b]["sketch"]) for a, b in pairs]), dtype=float)
    corr = np.asarray([c if c is not None else np.nan
                       for c in val_be.correlate(
                           [(store[a]["raw"], store[b]["raw"]) for a, b in pairs])],
                      dtype=float)
    keep = np.isfinite(cos) & np.isfinite(corr)
    cos, corr = cos[keep], corr[keep]

    # effective filter metric = |cos| when neg_corr (like validate_sketches)
    fcos = np.abs(cos) if config.neg_corr else cos
    label = ((np.abs(corr) >= config.corr_threshold) if config.neg_corr
             else (corr >= config.corr_threshold))
    pos, neg = fcos[label], fcos[~label]
    n = fcos.size
    base_rate = 100.0 * pos.size / n if n else float("nan")
    auc = _auc(pos, neg)

    print()
    print(_c("=== SKETCH SEPARABILITY (cosine vs true correlation) ===", "1;36"))
    print(f"  sample              : {n} pairs ({n_all} in total in the sample "
          f"of {len(windows)} windows)")
    print(f"  base rate           : {_c(f'{base_rate:.1f}%', '1;33')} of the candidate pairs "
          f"are TRULY correlated (|corr|≥{config.corr_threshold})")
    if base_rate >= 40:
        print(_c("    → many pairs ARE correlated: little to filter out, "
                 "\"almost everything passes validation\" is EXPECTED on this data.", "33"))
    print(f"  AUC cos(corr>thr)   : {_c(f'{auc:.3f}', '1;33')}  "
          f"(0.5 = the sketch separates nothing; 1.0 = perfect separation)")

    pp, pn = _pct(pos), _pct(neg)
    print("  cos distribution    :            p05    p25    p50    p75    p95")
    print(f"    correlated   (n={pos.size:<7}) : "
          + "  ".join(f"{pp[q]:+.3f}" for q in (5, 25, 50, 75, 95)))
    print(f"    uncorrelated (n={neg.size:<7}) : "
          + "  ".join(f"{pn[q]:+.3f}" for q in (5, 25, 50, 75, 95)))

    # effect of the current sketch threshold + "ideal" threshold for the target recall
    thr = config.sketch_threshold
    rec_cur = float((pos >= thr).mean()) if pos.size else float("nan")
    pass_cur = 100.0 * float((fcos >= thr).mean()) if n else float("nan")
    print(f"  sketch thr={thr:<4}      : recall={rec_cur:.3f}  "
          f"(keeps {pass_cur:.1f}% of the candidate pairs)")
    if pos.size:
        target = config.target_recall
        thr_ideal = float(np.quantile(pos, max(0.0, 1.0 - target)))
        keep_ideal = 100.0 * float((fcos >= thr_ideal).mean())
        prec_ideal = (100.0 * float((pos >= thr_ideal).sum())
                      / max(1, int((fcos >= thr_ideal).sum())))
        print(f"  thr for recall {target:<4}   : cos≥{_c(f'{thr_ideal:.3f}', '1;32')}  "
              f"→ keeps {_c(f'{keep_ideal:.1f}%', '1;32')} of the pairs, "
              f"precision≈{prec_ideal:.1f}%")
        if keep_ideal >= 50:
            print(_c("    ⚠ even at the ideal threshold, ≥50% of the pairs pass: the "
                     "sketch does not discriminate enough → preprocessing "
                     "(deseasonalize/detrend) or another sketch (fft_lowpass, "
                     "median_*).", "33"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser():
    p = argparse.ArgumentParser(
        prog="python -m v2.diagnose_select",
        description="Diagnostic of the candidate-selection filtering downstream of "
                    "the index (serial, backend of your choice).")
    p.add_argument("csv", nargs="?", help="Dataset CSV (1st col = time). "
                                          "Omit it with --synth.")
    p.add_argument("--synth", action="store_true",
                   help="Generate a synthetic correlated dataset (no real CSV needed).")
    p.add_argument("--synth-shared", dest="synth_shared", type=float, default=0.0,
                   help="Strength of a COMMON seasonality in --synth (reproduces the "
                        "dense climate case where almost everything correlates). "
                        "E.g. 3.0. Default: 0.")
    p.add_argument("--index", dest="index_sweep", default="bst",
                   help="Indexes to test (comma-separated list). Default: bst.")
    p.add_argument("--query-radius", dest="radius_sweep", default=None,
                   help="Query radius (a value, or a list 'a,b,c' to sweep). "
                        "Default: 2,4,6,8,10.")
    p.add_argument("--no-baseline", dest="no_baseline", action="store_true",
                   help="Do not run brute-force (no keep%%/recall/precision).")
    p.add_argument("--separability", action="store_true",
                   help="Analysis mode: does the sketch cosine separate correlated "
                        "from uncorrelated pairs? (answers \"why does everything "
                        "pass\"). No sweep.")
    p.add_argument("--sep-pairs", dest="sep_pairs", type=int, default=200_000,
                   help="Cap on the pairs sampled for --separability.")
    p.add_argument("--sep-windows", dest="sep_windows", type=int, default=20,
                   help="Number of windows sampled for --separability.")
    p.add_argument("--verbose", action="store_true",
                   help="Full run logs (silent otherwise).")

    # every other Config field as a single-value passthrough
    for f in fields(Config):
        if f.name in _SWEEP_FIELDS:
            continue
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None,
                       help=argparse.SUPPRESS)
    return p


def _base_overrides(args):
    """Config overrides from the passthrough fields actually provided."""
    ov = {}
    for f in fields(Config):
        if f.name in _SWEEP_FIELDS:
            continue
        raw = getattr(args, f.name, None)
        if raw is not None:
            ov[f.name] = _coerce(raw, f.type)
    return ov


def main():
    args = _build_parser().parse_args()

    # --- dataset ---
    if args.synth:
        path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "corrtrack_synth_diag.csv")
        _make_synth_csv(path, shared=args.synth_shared)
        print(_c(f"[synth] dataset written -> {path} (shared={args.synth_shared})", "2"))
    else:
        path = args.csv
        if not path:
            _build_parser().error("provide a CSV, or --synth")
        if not os.path.exists(path):
            _build_parser().error(f"CSV not found: {path}")

    # --- sweeps ---
    indexes = [s.strip() for s in args.index_sweep.split(",") if s.strip()]
    if args.radius_sweep is not None:
        radii = [float(s) for s in args.radius_sweep.split(",") if s.strip()]
    else:
        radii = [2.0, 4.0, 6.0, 8.0, 10.0]

    overrides = _base_overrides(args)
    overrides.setdefault("output", "")   # no disk writing
    if "log_level" not in overrides:
        overrides["log_level"] = "info" if args.verbose else "error"

    # --- SERIAL: never a sharded backend (window-parallel partition) ---
    be = overrides.get("backend", Config().backend)
    if is_sharded(be):
        print(_c(f"[serial] backend '{be}' is sharded -> forced to serial "
                 f"'{base_of(be)}'", "33"))
        overrides["backend"] = base_of(be)
    overrides["workers"] = 0            # neutral in serial mode

    # single logger, configured once (silent by default)
    reset()
    get_logger(overrides["log_level"], "")

    target_recall = overrides.get("target_recall", Config().target_recall)

    # --- separability mode: no sweep, just the cosine vs corr analysis ---
    if args.separability:
        cfg = Config.build(**overrides, index_backend=indexes[0],
                           query_radius=radii[0])
        analyze_separability(cfg, path, cap_pairs=args.sep_pairs,
                             n_win=args.sep_windows)
        return

    # --- brute-force (ground truth) ---
    bf_res = None
    if not args.no_baseline:
        cfg_bf = Config.build(**overrides)
        print(_c("→ brute-force (ground truth)…", "2"))
        bf_res, bf_t = _run(cfg_bf, path, "bf")

    # --- corrtrack: every (index, radius) ---
    print()
    _print_header()
    if bf_res is not None:
        _print_bf(bf_res, bf_t)
        print(_c("-" * 118, "2"))

    rows = []
    for index in indexes:
        for radius in radii:
            cfg = Config.build(**overrides, index_backend=index, query_radius=radius)
            ct, ct_t = _run(cfg, path, "corrtrack")
            rows.append(_print_row(index, radius, ct, ct_t, bf_res, target_recall))

    # --- recommendation: smallest keep% (best filtering index) with an OK recall ---
    if bf_res is not None and rows:
        ok = [r for r in rows if r["recall"] == r["recall"]
              and r["recall"] >= target_recall]
        pool = ok or rows
        best = min(pool, key=lambda r: (r["keep"] if r["keep"] == r["keep"] else 1e9))
        tag = "recall≥target" if ok else _c("NO setting reaches the target recall", "31")
        print(_c("-" * 118, "2"))
        print(_c(f"→ best filtering ({tag}): index={best['index']} "
                 f"radius={best['radius']} → keep={best['keep']:.1f}% vs bf, "
                 f"recall={best['recall']:.3f}, prec={best['prec']:.3f}, "
                 f"total={best['total']:.3f}s", "1;32"))
        if best["keep"] == best["keep"] and best["keep"] >= 50:
            print(_c("  ⚠ even at best, the index lets ≥50% of the pairs through: "
                     "lower --query-radius, or revisit the sketch / key extraction.",
                     "33"))


if __name__ == "__main__":
    main()
