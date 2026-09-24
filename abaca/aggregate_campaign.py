#!/usr/bin/env python3
"""Aggregate the campaign's per-cell JSONs into one long table and the paper's summary tables (2026-09-19).

Inputs (under RESULTS_ROOT, the layout written by the OAR wrappers):
  nway/<stem>_<pos|neg>/nway.json          N-way comparison of a cell (abaca/nway_compare.py)
  hyperopt/<stem>_<tag>/**/best_params_corrtrack.json   CorrTrack's tuned parameters (for the sensitivity tables)
  tuned/<stem>_<tag>/best_params_<arm>.json, tuning_<arm>.json   CSZ-protocol tuning of the competitors
  hyperopt|tuned/<stem>_<tag>/time_v.txt   GNU time -v of the tuning jobs (their wall and peak RSS)

Outputs (in --out):
  runs.csv          one row per (cell, arm): design factors (dataset, m, W, step, L, T, space, neg_corr), the
                    dataset profile, status, counts, recall / precision / F1, candidate_precision,
                    candidate_specificity, phase times, runtime, wall, speedup vs bruteforce, step-latency
                    boxplot ticks, peak / mean RSS (absolute and delta), storage I/O, artifact size, CPU time,
                    energy (when a source exists), the arm's knobs, evidence tags
  cells.csv         one row per cell: node, idle power, universe, positives, density, missing / failed arms
  tuning.csv        one row per (cell, tuned arm): the chosen setting, calibration recall / precision,
                    bootstrap lower bound, tuning wall and peak RSS; CorrTrack's best_params as columns
  summary_by_T.md   median (IQR) of speedup, recall, candidate precision, specificity, step median per arm and T
  summary_by_m.md   the same by m rung; summary_by_space.md by raw / differenced; summary_by_dataset.md
  failures.md       cells with ERROR or N/A arms, with the reason
  repeatability.md  the cells measured several times (campaign_competitors.REPEATS), their spread, and the
                    dispersion the tables' differences must clear to be interpreted
  energy.md         when the kwollect power series was fetched (abaca/kwollect_power.py): energy per arm

Every table states the number of cells behind each number. The medians are over cells; a cell is one
(dataset, m, W, step, L, T, space, neg_corr) point, so a dataset with more rungs weighs more in the
pooled tables and the by-dataset table is the one to read for balance.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("bruteforce", "bf_incremental", "filcorr", "tsubasa", "braid", "thinbraid", "corrtrack", "corrtrack_hamming", "parcorr", "csz", "statstream", "corrjoin")
STEM_RE = re.compile(r"^(?P<dataset>.+?)_m(?P<m>\d+)_W(?P<W>\d+)_s(?P<step>\d+)_L(?P<nlags>\d+)_T(?P<T>[0-9.]+)(?P<diff>_diff)?_(?P<tag>pos|neg)$")
RUN_FIELDS = ("status", "reason", "correlated", "total_candidates", "tested", "recall", "precision", "f1", "candidate_precision",
              "candidate_specificity", "candidate_specificity_is_lower_bound", "candidate_fpr", "candidate_false_positives",
              "sk_time", "cand_time", "val_time", "monit_time", "other_time", "runtime", "artifact_time", "wall", "wall_outer",
              "n_steps", "step_time_min", "step_time_q1", "step_time_median", "step_time_q3", "step_time_max", "step_time_whisker_lo",
              "step_time_whisker_hi", "step_time_outliers", "step_time_mean", "candidate_time_per_pair_window_us",
              "supports_neg_corr", "pure_python_index", "n_vectors", "candidate_backend", "data_representation",
              "parcorr_k", "parcorr_f", "parcorr_c", "statstream_n_coeffs", "statstream_index_dims", "corrjoin_ks", "corrjoin_ke", "corrjoin_kb",
              "braid_b", "braid_gamma", "braid_thin", "filcorr_fs", "filcorr_ft")
RES_FIELDS = ("rss_before_mb", "peak_rss_mb", "peak_rss_delta_mb", "mean_rss_mb", "mean_rss_delta_mb", "io_read_mb", "io_write_mb",
              "artifact_mb", "cpu_user_s", "cpu_sys_s", "energy_j", "energy_dram_j", "energy_source", "mean_power_w", "t_start_epoch", "t_end_epoch", "isolated")
PROFILE_FIELDS = ("m", "n_obs", "n_windows", "density_at_threshold", "low_frequency_energy_share_mean", "white_noise_reference",
                  "lag1_autocorr_mean", "constant_window_fraction", "coefficient_of_variation_median", "nan_fraction_source", "regime_source",
                  "pair_windows", "correlated_pair_windows")


REPEAT_RE = re.compile(r"_r(\d+)$")


def split_repeat(name: str):
    """(base cell name, repeat index): the emitter suffixes a repeated measurement with _r2, _r3, ..."""
    m = REPEAT_RE.search(name)
    return (name[: m.start()], int(m.group(1))) if m else (name, 1)


def parse_stem(name: str) -> dict:
    m = STEM_RE.match(name)
    if not m:
        return {"dataset": name, "m": None, "W": None, "step": None, "n_lags": None, "L": None, "T": None, "space": None, "neg_corr": None}
    d = m.groupdict()
    step = int(d["step"]); nl = int(d["nlags"])
    return {"dataset": d["dataset"], "m": int(d["m"]), "W": int(d["W"]), "step": step, "n_lags": nl, "L": nl // step + 1 if step else None,
            "T": float(d["T"]), "space": "diff" if d["diff"] else "raw", "neg_corr": d["tag"] == "neg"}


def time_v(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "Maximum resident set size" in line:
            out["peak_rss_mb"] = float(line.split(":")[-1]) / 1024.0
        elif "Elapsed (wall clock) time" in line:
            hms = line.split("):")[-1].strip().split(":")
            out["wall_s"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(hms)))
        elif "User time (seconds)" in line:
            out["cpu_user_s"] = float(line.split(":")[-1])
    return out


def load(results_root: Path, power: dict | None = None):
    runs, cells, tuning = [], [], []
    for f in sorted((results_root / "nway").glob("*/nway.json")):
        d = json.load(open(f))
        name = d.get("cell") or f.parent.name
        name, repeat = split_repeat(name)
        fac = parse_stem(name)
        prof = d.get("dataset_profile", {})
        node = d.get("node", {})
        bf = d["arms"].get("bruteforce", {})
        cell = dict(cell=name, repeat=repeat, **fac, dataset_label=d.get("dataset"), hostname=node.get("hostname"), oar_job_id=node.get("oar_job_id"),
                    idle_power_w=node.get("idle_power_w"), energy_source=node.get("energy_source"),
                    corrtrack_params_source=d.get("corrtrack_params_source"), tuned_arms=",".join(sorted(d.get("competitor_params_tuned", {}))),
                    bf_wall=bf.get("wall"), universe=bf.get("total_candidates"), positives=bf.get("correlated"),
                    **{f"prof_{k}": prof.get(k) for k in PROFILE_FIELDS},
                    arms_ok=",".join(a for a, r in d["arms"].items() if r.get("status") == "ok"),
                    arms_na=",".join(a for a, r in d["arms"].items() if r.get("status") == "N/A"),
                    arms_error=",".join(a for a, r in d["arms"].items() if r.get("status") == "ERROR"))
        if power is not None:
            # idle proxy for the dynamic energy: the lowest power sample of this node during the cell's job
            cell["kw_idle_power_w"] = job_min_power(power, node.get("hostname"), d["arms"])
        cells.append(cell)
        for arm, r in d["arms"].items():
            # (2026-09-23) results written before the rename carry the old arm id; normalize so a mixed
            # results tree aggregates into one table
            arm = "bf_incremental" if arm == "exact_stomp" else arm
            row = dict(cell=name, repeat=repeat, **fac, arm=arm, **{k: r.get(k) for k in RUN_FIELDS}, **{f"res_{k}": (r.get("resources") or {}).get(k) for k in RES_FIELDS},
                       **{f"prof_{k}": prof.get(k) for k in ("density_at_threshold", "low_frequency_energy_share_mean", "lag1_autocorr_mean", "constant_window_fraction", "regime_source")})
            row["speedup_vs_bf"] = (bf["wall"] / r["wall"]) if (r.get("status") == "ok" and bf.get("wall") and r.get("wall")) else None
            row["cand_speedup_vs_bf"] = (bf["cand_time"] / r["cand_time"]) if (r.get("status") == "ok" and bf.get("cand_time") and r.get("cand_time")) else None
            pf = r.get("phase_fractions") or {}
            for k in ("sk_time", "cand_time", "val_time", "monit_time"):
                row[f"frac_{k}"] = pf.get(k)
            if power is not None and r.get("status") == "ok":
                rs = r.get("resources") or {}
                row["kw_energy_j"], row["kw_mean_power_w"], row["kw_samples"] = integrate_power(power, node.get("hostname"), rs.get("t_start_epoch"), rs.get("t_end_epoch"))
                row["kw_dynamic_energy_j"] = (row["kw_energy_j"] - (cell.get("kw_idle_power_w") or 0.0) * r["wall"]) if row["kw_energy_j"] is not None else None
            runs.append(row)
        # tuning outputs of this cell
        for arm in ("parcorr", "csz", "statstream", "corrjoin"):
            bp = results_root / "tuned" / name / f"best_params_{arm}.json"
            if bp.exists():
                b = json.load(open(bp)); t = b.get("_tuning", {})
                tuning.append(dict(cell=name, **fac, arm=arm, **{k: v for k, v in b.items() if not k.startswith("_")},
                                   calib_recall=t.get("recall"), calib_precision=t.get("precision"), calib_total_candidates=t.get("total_candidates"),
                                   tuning_status=t.get("status"), target_recall=t.get("target_recall"),
                                   **{f"job_{k}": v for k, v in time_v(results_root / "tuned" / name / "time_v.txt").items()}))
        for sub, arm in (("hyperopt", "corrtrack"), ("hyperopt_hamming", "corrtrack_hamming")):
            hp = list((results_root / sub / name).rglob("best_params_corrtrack.json"))
            if hp:
                b = json.load(open(hp[0]))
                tuning.append(dict(cell=name, **fac, arm=arm, **{k: v for k, v in b.items() if not isinstance(v, (dict, list))},
                                   **{f"job_{k}": v for k, v in time_v(results_root / sub / name / "time_v.txt").items()}))
    return runs, cells, tuning


def job_min_power(power: dict, host: str | None, arms: dict):
    if not host:
        return None
    series = power.get(host) or power.get(host.split(".")[0])
    ts = [(r.get("resources") or {}).get("t_start_epoch") for r in arms.values() if r.get("status") == "ok"]
    te = [(r.get("resources") or {}).get("t_end_epoch") for r in arms.values() if r.get("status") == "ok"]
    if not series or not ts or None in ts:
        return None
    lo, hi = min(ts) - 120, max(te) + 120
    vals = [w for t, w in series if lo <= t <= hi]
    return min(vals) if vals else None


def integrate_power(power: dict, host: str | None, t0, t1):
    """energy (J) over [t0, t1] from a node power series {host: [(epoch, watt), ...]} (trapezoid, edge-held)."""
    if not host or t0 is None or t1 is None or t1 <= t0:
        return None, None, 0
    series = power.get(host) or power.get(host.split(".")[0])
    if not series:
        return None, None, 0
    ts = np.array([p[0] for p in series], dtype=float); ws = np.array([p[1] for p in series], dtype=float)
    grid = np.concatenate(([t0], ts[(ts > t0) & (ts < t1)], [t1]))
    vals = np.interp(grid, ts, ws)
    e = float(np.trapz(vals, grid))
    return e, e / (t1 - t0), int(((ts >= t0) & (ts <= t1)).sum())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def _fmt(vals, pct=False, digits=2):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if v.size == 0:
        return "-"
    q1, med, q3 = np.quantile(v, (0.25, 0.5, 0.75))
    if pct:
        return f"{100 * med:.1f} ({100 * q1:.1f}-{100 * q3:.1f})"
    return f"{med:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def collapse_repeats(runs: list[dict]) -> tuple[list[dict], str]:
    """(2026-09-23) One row per (cell, arm): the median over the repeated measurements of that cell, with the
    observed spread of the speedup in `speedup_spread_pct` and the count in `n_repeats`. Cells measured once
    pass through unchanged. Also returns a one-line note on the dispersion actually seen, which is what the
    paper quotes instead of assuming a tolerance (log 2026-09-23 (f))."""
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["cell"], r["arm"])].append(r)
    out, spreads = [], []
    for rs in by_key.values():
        rs = sorted(rs, key=lambda r: r.get("repeat", 1))
        base = dict(rs[len(rs) // 2])
        base["n_repeats"] = len(rs)
        sp = [r["speedup_vs_bf"] for r in rs if r.get("speedup_vs_bf")]
        if len(sp) > 1:
            base["speedup_vs_bf"] = float(np.median(sp))
            base["speedup_spread_pct"] = 100.0 * (max(sp) - min(sp)) / float(np.mean(sp))
            spreads.append(base["speedup_spread_pct"])
        out.append(base)
    note = (f"{len(spreads)} (cell, arm) pairs were measured more than once: spread of the speedup, median "
            f"{np.median(spreads):.1f}%, 90th percentile {np.percentile(spreads, 90):.1f}%. Differences below "
            f"that are not interpreted." if spreads else
            "No repeated measurements in this results tree: every cell was run once, so no dispersion is available.")
    return out, note


def summary(runs: list[dict], by: str, title: str) -> str:
    runs, _note = collapse_repeats(runs)
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.get("status") != "ok" or r["arm"] == "bruteforce":
            continue
        groups[(r.get(by), r["arm"])]["speedup"].append(r.get("speedup_vs_bf"))
        groups[(r.get(by), r["arm"])]["recall"].append(r.get("recall"))
        groups[(r.get(by), r["arm"])]["cprec"].append(r.get("candidate_precision"))
        groups[(r.get(by), r["arm"])]["spec"].append(r.get("candidate_specificity"))
        groups[(r.get(by), r["arm"])]["step"].append(1e3 * r["step_time_median"] if r.get("step_time_median") is not None else None)
        groups[(r.get(by), r["arm"])]["peak"].append(r.get("res_peak_rss_delta_mb"))
        groups[(r.get(by), r["arm"])]["energy"].append(r.get("kw_energy_j") if r.get("kw_energy_j") is not None else r.get("res_energy_j"))
        groups[(r.get(by), r["arm"])]["n"].append(1)
    out = [f"## {title}", "", "Median (IQR) over cells; n = cells with the arm ok. Speedup = bruteforce wall / arm wall (same cell, same node).", "",
           f"| {by} | arm | n | speedup | recall % | cand. precision % | cand. specificity % | step median ms | peak RSS delta MB | energy J |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for (g, arm) in sorted(groups, key=lambda k: (str(k[0]), ARMS.index(k[1]) if k[1] in ARMS else 99)):
        d = groups[(g, arm)]
        out.append(f"| {g} | {arm} | {len(d['n'])} | {_fmt(d['speedup'])} | {_fmt(d['recall'], pct=True)} | {_fmt(d['cprec'], pct=True)} | "
                   f"{_fmt(d['spec'], pct=True)} | {_fmt(d['step'], digits=3)} | {_fmt(d['peak'], digits=0)} | {_fmt(d['energy'], digits=0)} |")
    return "\n".join(out) + "\n"


def failures(runs: list[dict]) -> str:
    out = ["## Failed and not-applicable arms", "", "| cell | arm | status | reason |", "|---|---|---|---|"]
    n = 0
    for r in runs:
        if r.get("status") in ("ERROR", "N/A"):
            out.append(f"| {r['cell']} | {r['arm']} | {r['status']} | {str(r.get('reason'))[:160]} |"); n += 1
    return "\n".join(out) + f"\n\n{n} rows.\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=os.path.expandvars("$HOME/corrtrack_abaca_results"))
    ap.add_argument("--out", default=None, help="output directory (default <results-root>/aggregate)")
    ap.add_argument("--power", default=None, help="node power series JSON from abaca/kwollect_power.py ({host: [[epoch, watt], ...]})")
    args = ap.parse_args()
    root = Path(args.results_root)
    out = Path(args.out) if args.out else root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    power = json.load(open(args.power)) if args.power else None
    if power is not None:
        power.pop("_meta", None)
    runs, cells, tuning = load(root, power)
    write_csv(out / "runs.csv", runs); write_csv(out / "cells.csv", cells); write_csv(out / "tuning.csv", tuning)
    (out / "summary_by_T.md").write_text(summary(runs, "T", "By correlation threshold"))
    (out / "summary_by_m.md").write_text(summary(runs, "m", "By number of series (design rung)"))
    (out / "summary_by_L.md").write_text(summary(runs, "L", "By number of lagged windows"))
    (out / "summary_by_space.md").write_text(summary(runs, "space", "By space (raw vs first differences)"))
    (out / "summary_by_dataset.md").write_text(summary(runs, "dataset", "By dataset"))
    (out / "summary_by_neg.md").write_text(summary(runs, "neg_corr", "By negative-correlation setting"))
    (out / "failures.md").write_text(failures(runs))
    collapsed, note = collapse_repeats(runs)
    rep_rows = [r for r in collapsed if r.get("n_repeats", 1) > 1]
    (out / "repeatability.md").write_text(
        "# Repeatability of the measured speedups\n\n" + note + "\n\n"
        "| cell | arm | repeats | median speedup | spread % |\n|---|---|---|---|---|\n"
        + "".join(f"| {r['cell']} | {r['arm']} | {r['n_repeats']} | {r['speedup_vs_bf']:.3f} | {r['speedup_spread_pct']:.1f} |\n"
                  for r in sorted(rep_rows, key=lambda r: -r.get("speedup_spread_pct", 0.0))))
    ok = sum(1 for r in runs if r.get("status") == "ok")
    print(f"{len(cells)} cells, {len(runs)} arm runs ({ok} ok), {len(tuning)} tuning rows -> {out}")


if __name__ == "__main__":
    main()
