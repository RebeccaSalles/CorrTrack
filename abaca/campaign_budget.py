#!/usr/bin/env python3
"""Time and memory projection of the competitor campaign, per design option (2026-09-20).

Three job kinds per cell and labelled run (pos = positive correlations, neg = the --neg-corr run):
  hyperopt  CorrTrack's proxy hyperopt (abaca/hyperopt_corrtrack.oar), packed core=2 jobs
  tune      CSZ protocol for the four grid competitors (abaca/tune_competitors.oar), packed core=2 jobs
  nway      the N-way comparison (abaca/nway_compare.oar), one whole host per job
The constants below are the pilot measurements on mercantour3 (Xeon Silver 4114, 2026-09-19/20) and are the
single place to update when a new measurement comes in. Outputs one markdown report with, per option:
node-hours and wall-clock days for each job kind, the memory profile (peak per job, cells above the budget),
and the cells dropped by the option.

    python abaca/campaign_budget.py                       # all options, markdown to stdout
    python abaca/campaign_budget.py --option C --cells    # one option with the per-cell table
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import abaca.campaign_competitors as camp  # noqa: E402

# ----------------------------------------------------------------------------- calibration (mercantour3)
K_BF = 1.2e-7                 # s per m^2 x L x n_steps: bruteforce wall (sp500 m492 L5: 34 s; ASOS m600 L1: 77 s)
ARM_FACTOR = {                # battery wall over bruteforce wall
    "all": {"pos": 8.0, "neg": 5.5},          # 11 arms (pilot: 4.8x without tsubasa/parcorr/csz/corrjoin)
    "light": {"pos": 3.2, "neg": 2.2},        # bruteforce + exact_stomp + the pruning arms
}
SET_BYTES = 28                # bytes per correlated pair-window held in memory (int32 x5 + float64)
# hyperopt with the NUMERIC proxy reference, campaign grid (80 settings), mercantour3 core=2 (probes 3124942-45,
# 2026-09-20): sp500 m492 L5 5.0 min 2.6 GB; ASOS m600 L1 (111 anchors) 9.9 min 3.0 GB; Berkeley m5000 L4 3.7 min
# 3.3 GB; corrjoin_gas m5000 L1 11.8 min 3.6 GB (the Python-object reference took 20 min and 6.1 GB at m492)
HYPEROPT_MIN = {8: 3.0, 80: 12.0, 150: 22.0}     # wall minutes by grid size, the worst probe
HYPEROPT_GB = 4.0
TUNE_MIN = {"pos": 16.0, "neg": 3.0}             # 300 series x 64 windows (probes 3123165-70); neg tunes statstream only
TUNE_GB = 3.0
PACK_CORES = 2
HOST_CORES, HOST_GB = 20, 192
DATA_GB_PER_CELL = lambda m, n: 3 * m * n * 8 / 2**30     # noqa: E731  raw + differenced + validation buffer copies

# density of correlated pair-windows at each T: measured where we have it (pilot, m=500 tables), assumed otherwise
DENSITY = {
    # dataset: {space: {T: density}}   (space "raw" or "diff")
    "sp500": {"raw": {0.7: 0.186, 0.8: 0.086, 0.9: 0.0161, 0.95: 0.0019}, "diff": {0.7: 5.1e-3, 0.8: 1.4e-3, 0.9: 1.1e-4, 0.95: 8.8e-6}},
    "acwi_capweighted": {"raw": {0.7: 0.10, 0.8: 0.04, 0.9: 0.01, 0.95: 0.002}, "diff": {0.7: 2.6e-3, 0.8: 7.2e-4, 0.9: 6.4e-5, 0.95: 5.6e-6}},
    "streamflow": {"raw": {0.7: 0.051, 0.8: 0.026, 0.9: 0.008, 0.95: 0.0023}, "diff": {0.7: 3.3e-3, 0.8: 1.5e-3, 0.9: 3.7e-4, 0.95: 9.0e-5}},
    "wikipedia": {"raw": {0.7: 3.9e-3, 0.8: 2.1e-3, 0.9: 7.1e-4, 0.95: 1.8e-4}, "diff": {0.7: 2.4e-3, 0.8: 1.1e-3, 0.9: 2.8e-4, 0.95: 4.8e-5}},
    "smartmeter": {"raw": {0.7: 1.9e-3, 0.8: 2.5e-4, 0.9: 8.6e-6, 0.95: 1.2e-6}, "diff": {0.7: 2.4e-5, 0.8: 8.8e-6, 0.9: 2.8e-6, 0.95: 1.0e-6}},
    "global_weather": {"raw": {0.7: 1.5e-4, 0.8: 5.1e-5, 0.9: 1.6e-5, 0.95: 5.7e-6}, "diff": {0.7: 1.5e-4, 0.8: 5.1e-5, 0.9: 1.6e-5, 0.95: 5.7e-6}},
    "global_asos_air_temperature": {"raw": {0.7: 0.05, 0.8: 0.025, 0.9: 0.008, 0.95: 0.002}, "diff": {0.7: 1e-3, 0.8: 2e-4, 0.9: 1e-5, 0.95: 1e-6}},
    "motes_temperature": {"raw": {0.7: 0.5, 0.8: 0.4, 0.9: 0.30, 0.95: 0.2}, "diff": {0.7: 0.02, 0.8: 0.005, 0.9: 0.0, 0.95: 0.0}},
    "corrjoin_gas": {"raw": {0.7: 0.45, 0.8: 0.35, 0.9: 0.21, 0.95: 0.1}, "diff": {0.7: 0.05, 0.8: 0.02, 0.9: 0.005, 0.95: 0.001}},
}
DEFAULT_DENSITY = {"raw": {0.7: 0.05, 0.8: 0.02, 0.9: 0.008, 0.95: 0.002}, "diff": {0.7: 5e-3, 0.8: 1e-3, 0.9: 1e-4, 0.95: 1e-5}}
ASSUMED_LIKE = {"motes_humidity": "motes_temperature", "uscrn2020_temperature": "global_asos_air_temperature",
                "global_asos_relative_humidity": "global_asos_air_temperature", "global_asos_wind_speed": "wikipedia",
                "global_asos_pressure": "motes_temperature", "corrjoin_chlorine": "corrjoin_gas", "corrjoin_stock": "sp500",
                "corrjoin_random": "smartmeter", "berkeley_tavg_anom_2010": "global_asos_air_temperature",
                "statstream_rw_m5000_T20000": "streamflow", "braid_sines_m5000_T32768": "wikipedia",
                "braid_spiketrains_m5000_T100000": "wikipedia", "yellowstone_bp3_7": "wikipedia", "yellowstone_raw": "wikipedia"}

OPTIONS = {
    "A": dict(desc="as emitted (synthetic n_obs 20,000)", synth_n=20000),
    "B": dict(desc="synthetic n_obs 5,000 (402 windows, the median of the real sets)", synth_n=5000),
    "C": dict(desc="B + at m >= 2,500 only bruteforce, exact_stomp and the pruning arms", synth_n=5000, light_above=2500),
    "D": dict(desc="C + neg_corr run only at L = 1 cells", synth_n=5000, light_above=2500, neg_l1_only=True),
    "E": dict(desc="B + neg_corr run only at L = 1 (all arms everywhere)", synth_n=5000, neg_l1_only=True),
    "C-mem": dict(desc="C + memory cut: no (m=5000, L=5) synthetic rung at density >= 0.05; Berkeley full-m at T >= 0.8 only",
                  synth_n=5000, light_above=2500, memory_cut=True),
    "D-mem": dict(desc="D + the same memory cut", synth_n=5000, light_above=2500, neg_l1_only=True, memory_cut=True),
}


def n_obs_of(cell, synth_n):
    if cell.dataset.startswith("synth_"):
        return synth_n
    d = next((x for x in camp.DATASETS if x.label == cell.dataset), None)
    if cell.n_obs:
        return cell.n_obs
    if d and d.n_obs:
        return d.n_obs
    cfg = REPO / cell.config
    if cfg.exists():
        m = re.search(r"N_OBS\s*=\s*\[(\d+)\]", cfg.read_text())
        if m:
            return int(m.group(1))
    return 8784


def density_of(cell):
    if cell.dataset.startswith("synth_"):
        return float("0." + re.search(r"_d0p(\d+)_", cell.dataset).group(1))
    space = "diff" if cell.preprocess else "raw"
    key = cell.dataset if cell.dataset in DENSITY else ASSUMED_LIKE.get(cell.dataset)
    table = DENSITY.get(key, DEFAULT_DENSITY)
    return table[space].get(cell.T, DEFAULT_DENSITY[space][cell.T])


def hyperopt_minutes(settings):
    keys = sorted(HYPEROPT_MIN)
    if settings in HYPEROPT_MIN:
        return HYPEROPT_MIN[settings]
    lo = max([k for k in keys if k <= settings], default=keys[0]); hi = min([k for k in keys if k >= settings], default=keys[-1])
    if lo == hi:
        return HYPEROPT_MIN[lo]
    return HYPEROPT_MIN[lo] + (HYPEROPT_MIN[hi] - HYPEROPT_MIN[lo]) * (settings - lo) / (hi - lo)


def project(option, settings=80):
    opt = OPTIONS[option]
    rows = []
    dropped = []
    for cell in camp.cells():
        m = cell.n_series; L = cell.n_lags // cell.step + 1
        n = n_obs_of(cell, opt["synth_n"])
        steps = max((n - cell.W) // cell.step, 1)
        universe = m * (m - 1) / 2 * L * steps
        dens = density_of(cell)
        if opt.get("memory_cut"):
            if cell.dataset.startswith("synth_") and m >= 5000 and L >= 5 and dens >= 0.05:
                dropped.append(cell.stem); continue
            if cell.dataset == "berkeley_tavg_anom_2010" and m > 5000 and cell.T < 0.8:
                dropped.append(cell.stem); continue
        light = bool(opt.get("light_above")) and m >= opt["light_above"]
        fac = ARM_FACTOR["light" if light else "all"]
        runs = ["pos"] + ([] if (opt.get("neg_l1_only") and L > 1) else ["neg"])
        bf_s = K_BF * m * m * L * steps
        set_gb = universe * dens * SET_BYTES / 2**30
        for run in runs:
            nway_h = fac[run] * bf_s / 3600
            nway_gb = DATA_GB_PER_CELL(m, n) + 2 * set_gb + 1.0          # bruteforce set + one arm set + partition temporaries
            tune_h = TUNE_MIN[run] / 60 * (0.8 if L > 1 else 1.0)         # corrjoin not tuned when n_lags > 0
            rows.append(dict(cell=cell.stem, run=run, dataset=cell.dataset, m=m, L=L, T=cell.T, space="diff" if cell.preprocess else "raw",
                             steps=steps, density=dens, set_gb=set_gb, nway_h=nway_h, nway_gb=nway_gb, nway_wall_h=nway_h,
                             hyper_h=hyperopt_minutes(settings) / 60, hyper_gb=HYPEROPT_GB, tune_h=tune_h, tune_gb=TUNE_GB, light=light))
    return rows, dropped


def summarize(rows, nway_hosts, pack_hosts, budget_gb):
    tot = defaultdict(float)
    for r in rows:
        tot["nway_h"] += r["nway_h"]; tot["hyper_h"] += r["hyper_h"]; tot["tune_h"] += r["tune_h"]
    n_jobs = len(rows)
    pack_slots = pack_hosts * HOST_CORES // PACK_CORES
    pack_mem_slots = pack_hosts * int(HOST_GB // max(HYPEROPT_GB, TUNE_GB))
    slots = min(pack_slots, pack_mem_slots)
    out = {
        "runs": n_jobs,
        "nway_node_h": tot["nway_h"], "nway_days": tot["nway_h"] / nway_hosts / 24,
        "hyper_core_h": tot["hyper_h"], "tune_core_h": tot["tune_h"],
        "pack_days": (tot["hyper_h"] + tot["tune_h"]) / slots / 24, "pack_slots": slots,
        "nway_peak_gb": max(r["nway_gb"] for r in rows), "nway_over_budget": sum(1 for r in rows if r["nway_gb"] > budget_gb),
        "nway_over_60": sum(1 for r in rows if r["nway_gb"] > 60),
        "longest_nway_h": max(r["nway_wall_h"] for r in rows), "nway_over_48h": sum(1 for r in rows if r["nway_wall_h"] > 48),
        "hyper_gb": HYPEROPT_GB, "tune_gb": TUNE_GB,
    }
    out["total_days"] = max(out["nway_days"], out["pack_days"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--option", default=None, help="one option (default: all)")
    ap.add_argument("--settings", type=int, default=80, help="hyperopt grid size (experiment_run_param_grid_campaign.py: 80)")
    ap.add_argument("--nway-hosts", type=int, default=11)
    ap.add_argument("--pack-hosts", type=int, default=4)
    ap.add_argument("--memory-budget-gb", type=float, default=150.0)
    ap.add_argument("--cells", action="store_true", help="per-cell table (largest first)")
    ap.add_argument("--out", default=None, help="write the markdown here too")
    args = ap.parse_args()
    lines = [f"# Campaign budget projection ({args.nway_hosts} N-way hosts, {args.pack_hosts} pack hosts, hyperopt grid {args.settings} settings)", "",
             f"Calibration: bruteforce = {K_BF:.1e} s x m^2 L n_steps; battery = {ARM_FACTOR['all']['pos']}x / {ARM_FACTOR['all']['neg']}x bruteforce (pos / neg), "
             f"light battery {ARM_FACTOR['light']['pos']}x / {ARM_FACTOR['light']['neg']}x; hyperopt {hyperopt_minutes(args.settings):.0f} min and {HYPEROPT_GB} GB per run; "
             f"tuning {TUNE_MIN['pos']:.0f} / {TUNE_MIN['neg']:.0f} min (pos / neg) and {TUNE_GB} GB; correlated set {SET_BYTES} B per pair-window, "
             f"N-way memory = data + bruteforce set + one arm set + 1 GB. Densities: measured for the m=500 sets and the pilot cells, assumed elsewhere.", "",
             "| option | runs | N-way node-h | N-way days | hyperopt core-h | tuning core-h | pack days | total days | N-way peak GB | runs > budget | runs > 60 GB | longest N-way h | runs > 48 h |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    chosen = [args.option] if args.option else list(OPTIONS)
    details = {}
    for name in chosen:
        rows, dropped = project(name, args.settings)
        sm = summarize(rows, args.nway_hosts, args.pack_hosts, args.memory_budget_gb)
        details[name] = (rows, dropped, sm)
        lines.append(f"| {name} | {sm['runs']} | {sm['nway_node_h']:,.0f} | {sm['nway_days']:.1f} | {sm['hyper_core_h']:,.0f} | {sm['tune_core_h']:,.0f} | "
                     f"{sm['pack_days']:.1f} | {sm['total_days']:.1f} | {sm['nway_peak_gb']:.0f} | {sm['nway_over_budget']} | {sm['nway_over_60']} | "
                     f"{sm['longest_nway_h']:.0f} | {sm['nway_over_48h']} |")
    lines += ["", "Options:"] + [f"- **{k}**: {v['desc']}" for k, v in OPTIONS.items() if k in chosen]
    lines += ["", f"Pack hosts: {args.pack_hosts} x {HOST_CORES} cores at core={PACK_CORES} = {args.pack_hosts * HOST_CORES // PACK_CORES} slots, "
              f"memory-bound to {args.pack_hosts * int(HOST_GB // max(HYPEROPT_GB, TUNE_GB))} slots at {max(HYPEROPT_GB, TUNE_GB)} GB per job. "
              "Total days = max(N-way days, pack days): the two pools run concurrently and the N-way jobs wait on their own hyperopt and tuning."]
    for name, (rows, dropped, sm) in details.items():
        if dropped:
            lines += ["", f"{name}: {len(dropped)} cells dropped by the memory cut (first 6): " + ", ".join(dropped[:6])]
        if args.cells:
            lines += ["", f"## {name}: largest runs", "", "| run | N-way h | N-way GB | set GB | density | steps |", "|---|---|---|---|---|---|"]
            for r in sorted(rows, key=lambda r: -r["nway_gb"])[:20]:
                lines.append(f"| {r['cell']}_{r['run']} | {r['nway_h']:.1f} | {r['nway_gb']:.1f} | {r['set_gb']:.1f} | {r['density']:.4f} | {r['steps']} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
