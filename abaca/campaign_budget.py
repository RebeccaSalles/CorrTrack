#!/usr/bin/env python3
"""Time and memory projection of the competitor campaign, per design option (2026-09-20).

Three job kinds per cell and labelled run (pos = positive correlations, neg = the --neg-corr run):
  hyperopt  CorrTrack's proxy hyperopt, one per backend (lsh_sign_dot grid and lsh_hamming_exact grid), packed core=2 jobs
  tune      CSZ protocol for the four grid competitors (abaca/tune_competitors.oar), packed core=2 jobs
  nway      the N-way comparison (abaca/nway_compare.oar), one whole host per job, repeated
            campaign_competitors.REPEATS times on the sampled cells (the measured-dispersion sample)
The constants below are the pilot measurements on mercantour3 (Xeon Silver 4114, 2026-09-19/20) and are the
single place to update when a new measurement comes in. Outputs one markdown report with, per option:
node-hours and wall-clock days for each job kind, the memory profile (peak per job, cells above the budget),
and the cells dropped by the option.

    python abaca/campaign_budget.py                       # all options, markdown to stdout
    python abaca/campaign_budget.py --option C --cells    # one option with the per-cell table
"""
from __future__ import annotations

import argparse
import math
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
    # 12 arms. Measured on sp500 m=492 W=60 step=5 L=5 T=0.9 raw (2026-09-19 pilot for the ten arms then
    # available, 2026-09-23 for the two whose lagged runs were enabled that day): bruteforce 13.1 s,
    # tsubasa 16.4 s (1.25x), corrjoin 3.7 s (0.28x). corrjoin is not_available on the neg run, tsubasa is
    # native there, so the neg factor gains only tsubasa.
    "all": {"pos": 9.8, "neg": 7.1},
    # bruteforce + bf_incremental + the pruning arms, both CorrTrack backends (m >= 2,500 cells); plain BRAID
    # and, on lagged cells, TSUBASA are out of those cells by the manifest's own memory rule
    "light": {"pos": 3.8, "neg": 2.5},
}
SET_BYTES = 28                # bytes per correlated pair-window held in memory (int32 x5 + float64)
# hyperopt with the NUMERIC proxy reference, campaign grid (80 settings), mercantour3 core=2 (probes 3124942-45,
# 2026-09-20): sp500 m492 L5 5.0 min 2.6 GB; ASOS m600 L1 (111 anchors) 9.9 min 3.0 GB; Berkeley m5000 L4 3.7 min
# 3.3 GB; corrjoin_gas m5000 L1 11.8 min 3.6 GB (the Python-object reference took 20 min and 6.1 GB at m492)
HYPEROPT_MIN = {8: 3.0, 80: 12.0, 150: 22.0}     # wall minutes by grid size, the worst probe
HYPEROPT_HAMMING_MIN = 4.0    # the lsh_hamming_exact grid (8 settings, experiment_run_param_grid_campaign_hamming.py), per run
HYPEROPT_GB = 4.0
TUNE_MIN = {"pos": 18.0, "neg": 3.5}             # 300 series x 64 windows (probes 3123165-70) + 25% on the statstream stage
                                                 # after its digest grid grew from 96 to 120 settings (2026-09-22)
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
    "C": dict(desc="B + at m >= 2,500 only bruteforce, bf_incremental and the pruning arms", synth_n=5000, light_above=2500),
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
            # (2026-09-23, user) the sampled cells are measured camp.REPEATS times so the tables can quote a
            # measured dispersion instead of an assumed tolerance; only the N-way job repeats
            reps = camp.repeat_runs(cell, run)
            nway_h = reps * fac[run] * bf_s / 3600
            # (2026-09-23) TSUBASA's lagged extension caches one m x m cross-segment sketch per (live segment,
            # probed lag); the manifest keeps it out of the m > 2,000 lagged cells, so this term stays small
            # (93 MB measured at m = 492, L = 5), but it is the largest per-arm structure after the sets
            tsubasa_gb = (0.0 if (L <= 1 or m > 2000 or light) else
                          (math.ceil((cell.W + cell.n_lags) / float(cell.step)) + 1) * (L - 1) * m * m * 8 / 2 ** 30)
            nway_gb = DATA_GB_PER_CELL(m, n) + 2 * set_gb + tsubasa_gb + 1.0   # bf set + one arm set + the arm + temporaries
            tune_h = TUNE_MIN[run] / 60 * (0.8 if L > 1 else 1.0)         # corrjoin not tuned when n_lags > 0
            rows.append(dict(cell=cell.stem, run=run, repeats=reps, dataset=cell.dataset, m=m, L=L, T=cell.T, space="diff" if cell.preprocess else "raw",
                             steps=steps, density=dens, set_gb=set_gb, nway_h=nway_h, nway_gb=nway_gb, nway_wall_h=nway_h,
                             hyper_h=(hyperopt_minutes(settings) + HYPEROPT_HAMMING_MIN) / 60, hyper_gb=HYPEROPT_GB, tune_h=tune_h, tune_gb=TUNE_GB, light=light))
    return rows, dropped


# ----------------------------------------------------------------------------- side experiments (own scripts, own jobs)
# Each returns (node_hours_nway_pool, core_hours_pack_pool, n_runs, note). Costs use the same k_bf and factors.
def _bf_hours(m, L, steps):
    return K_BF * m * m * L * steps / 3600


def _anchor_cells(synth_n, spaces=("raw", "diff"), thresholds=(0.7, 0.8, 0.9, 0.95)):
    """the dataset anchors of the manifest: (m_max, L=1) and (m_max, L_max) per dataset, at the requested T and spaces"""
    out = []
    for cell in camp.cells():
        d = next((x for x in camp.DATASETS if x.label == cell.dataset), camp.SYNTH_SPEC if cell.dataset.startswith("synth_") else None)
        if d is None or cell.T not in thresholds or (("diff" if cell.preprocess else "raw") not in spaces):
            continue
        L = cell.n_lags // cell.step + 1
        if cell.n_series == d.m_max and L in (1, d.L_max):
            out.append((cell, n_obs_of(cell, synth_n)))
    return out


def side_experiments(synth_n=5000, settings=80):
    rows = []
    # 1. Phase R: done (2.5 h on 2026-09-19); reruns of the four fixed reproductions at most
    rows.append(dict(name="Phase R (paper reproductions)", nway_h=2.5, pack_h=0.0, runs=8,
                     sweep="one run per paper on its own data / stand-in; already run, rerun only after the commit"))
    # 2. ablation ladder: bruteforce + 7 CorrTrack variants with the cell's tuned parameters, dataset anchors, pos run
    h = 0.0; n = 0
    for cell, nobs in _anchor_cells(synth_n):
        if cell.dataset.startswith("synth_") and "_d0p02_" not in cell.dataset:
            continue                                               # synthetic: one density (2%) is enough for the ladder
        m = cell.n_series; L = cell.n_lags // cell.step + 1; steps = max((nobs - cell.W) // cell.step, 1)
        h += 3.5 * _bf_hours(m, L, steps); n += 1
    rows.append(dict(name="CorrTrack ablation ladder", nway_h=h, pack_h=0.0, runs=n,
                     sweep=f"{n} anchor cells (real datasets + the 2% synthetic family; m_max at L=1 and L_max, 4 T, both spaces): plain BF -> sketch only -> +Hamming / +dot / both -> LSH +Hamming / +dot / both, tuned parameters from the campaign hyperopt"))
    # 3. parallel scaling: anchors with m >= 2000, T = 0.9, raw; threads 1..20 for bruteforce and CorrTrack, 3 repeats
    h = 0.0; n = 0
    for cell, nobs in _anchor_cells(synth_n, spaces=("raw",), thresholds=(0.9,)):
        if cell.n_series < 2000:
            continue
        m = cell.n_series; L = cell.n_lags // cell.step + 1; steps = max((nobs - cell.W) // cell.step, 1)
        h += 3 * (2.0 + 0.6) * _bf_hours(m, L, steps); n += 1     # sum over threads of bf(1 + 1/2 + ...) ~ 2 bf; CorrTrack ~0.6 bf
    rows.append(dict(name="Parallel scaling (threads)", nway_h=h, pack_h=0.0, runs=n,
                     sweep=f"{n} anchor cells with m >= 2000 at T=0.9 raw: sequential vs 2/4/8/16/20 threads, bruteforce and tuned CorrTrack, 3 repeats"))
    # 4. naive baseline
    rows.append(dict(name="Naive baseline (naive Python / numpy vs bruteforce)", nway_h=2.0, pack_h=0.0, runs=8,
                     sweep="m in {25, 50, 100, 200} on two datasets, T=0.9: naive per-pair Python, naive numpy, our bruteforce, bf_incremental"))
    # 5. monitoring / anomaly vs events
    rows.append(dict(name="Monitoring / anomalies vs real events", nway_h=30.0, pack_h=0.0, runs=20,
                     sweep="~5 datasets with event tables (Yellowstone M6.5, streamflow floods, USCRN 2020 events, Motes battery deaths, ASOS), 4 T, CorrTrack monitor on vs bruteforce monitor on; DESIGN PENDING (user's event tables)"))
    # 6. step sweep incl. step = 1
    h = 0.0; ph = 0.0; n = 0
    for label, m, L, nobs, W in (("sp500", 492, 5, 1255, 60), ("uscrn2020_temperature", 153, 3, 8784, 168), ("yellowstone_bp3_7", 28, 11, 30000, 2000)):
        for step in (1, 2, 3, 5, 10, W // 10):
            for _space in ("raw", "diff"):
                steps = max((nobs - W) // step, 1)
                h += 6.0 * _bf_hours(m, L, steps)                  # grid arms excluded at step != basic window
                ph += (hyperopt_minutes(settings) + TUNE_MIN["pos"]) / 60; n += 1
    rows.append(dict(name="Window step sweep, step = 1 included", nway_h=h, pack_h=ph, runs=n,
                     sweep=f"{n} cells: sp500 (W60), USCRN temperature (W168), Yellowstone (W2000) x step in {{1, 2, 3, 5, 10, W/10}} x both spaces at T=0.9; arms bruteforce, bf_incremental, filcorr, braid, thinbraid, corrtrack, statstream (grid arms need basic_window = step)"))
    # 7. dynamic window size / step change mid-stream
    h = 0.0; n = 0
    for label, m, L, nobs, W, step in (("sp500", 492, 5, 1255, 60, 5), ("uscrn2020_temperature", 153, 3, 8784, 168, 12), ("motes_temperature", 27, 3, 40000, 2880, 288)):
        for _sched in ("W shrink then grow", "step halve then restore", "W and step together"):
            for _space in ("raw", "diff"):
                steps = max((nobs - W) // step, 1)
                h += 4.0 * _bf_hours(m, L, steps); n += 1        # CorrTrack adaptive + CorrTrack restart + bruteforce with the schedule + reference runs
    rows.append(dict(name="Dynamic W / step change mid-stream", nway_h=h, pack_h=0.0, runs=n,
                     sweep=f"{n} cells: 3 datasets x 3 schedules x both spaces at T=0.9; CorrTrack update_window_size/update_window_step vs a restarted CorrTrack vs bruteforce following the same schedule; competitors as a capability table"))
    # 8. multiple window sizes
    h = 0.0; n = 0
    for label, m, L, nobs, W, step in (("sp500", 492, 5, 1255, 60, 5), ("uscrn2020_temperature", 153, 3, 8784, 168, 12), ("global_asos_air_temperature", 600, 5, 17520, 168, 12)):
        for sizes in (2, 3):
            for _space in ("raw", "diff"):
                steps = max((nobs - W) // step, 1)
                h += (sizes * 1.6 + 1.0) * _bf_hours(m, L, steps); n += 1   # bruteforce per size + separate CorrTracks + one shared-sketch run
    rows.append(dict(name="Multiple window sizes (shared sketches)", nway_h=h, pack_h=0.0, runs=n,
                     sweep=f"{n} cells: 3 datasets x size sets {{W/2, W}} and {{W/4, W/2, W}} x both spaces at T=0.9; CorrTrackMultiWindow vs one CorrTrack per size vs bruteforce per size"))
    return rows


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
    side = side_experiments(synth_n=5000, settings=args.settings)
    side_nway = sum(r["nway_h"] for r in side); side_pack = sum(r["pack_h"] for r in side)
    lines += ["", "## Side experiments (same for every option; own scripts and jobs)", "",
              "| experiment | runs | N-way-pool node-h | pack-pool core-h | sweep |", "|---|---|---|---|---|"]
    for r in side:
        lines.append(f"| {r['name']} | {r['runs']} | {r['nway_h']:,.0f} | {r['pack_h']:,.0f} | {r['sweep']} |")
    lines.append(f"| **total** | {sum(r['runs'] for r in side)} | **{side_nway:,.0f}** | **{side_pack:,.0f}** | |")
    lines += ["", "## Complete figure: campaign option + side experiments", "",
              "| option | campaign N-way node-h | side node-h | total N-way node-h | total days on the N-way hosts | pack core-h (hyperopt + tuning + side) | pack days |", "|---|---|---|---|---|---|---|"]
    for name, (rows_, dropped_, sm) in details.items():
        tot_nway = sm["nway_node_h"] + side_nway
        tot_pack = sm["hyper_core_h"] + sm["tune_core_h"] + side_pack
        lines.append(f"| {name} | {sm['nway_node_h']:,.0f} | {side_nway:,.0f} | {tot_nway:,.0f} | **{tot_nway / args.nway_hosts / 24:.1f}** | {tot_pack:,.0f} | {tot_pack / sm['pack_slots'] / 24:.1f} |")
    n_real = sum(1 for d in camp.DATASETS)
    lines += ["", "## What every option contains (the comparative campaign)", "",
              f"- Datasets: {n_real} real sets (each at its horizon W/step) and the synthetic family (2 processes x 5 densities), each as its own dataset in raw and differenced space.",
              "- Design per dataset (ladder): anchor (m_max, L=1) and rungs (m_max/8, 1), (m_max/4, ~L_max/3), (m_max/2, ~2L_max/3), (m_max, L_max); Berkeley adds the full 18,520-series anchor; plus the W-robustness cells (sp500, USCRN at half and double horizon) and the Yellowstone step=1 cell.",
              "- Every cell at T in {0.7, 0.8, 0.9, 0.95}, in both spaces, with a positive run and a negative-correlation run (neg_corr=True).",
              "- Per cell and run: two CorrTrack hyperopts (lsh_sign_dot grid, 80 settings; lsh_hamming_exact grid, 8 settings), CSZ-protocol tuning of ParCorr / CSZ / StatStream / CorrJoin, and the N-way comparison of the 12 arms (bruteforce, bf_incremental, filcorr, tsubasa, braid, thinbraid, corrtrack [lsh + dot gate], corrtrack_hamming [exact Hamming + dot gate], parcorr, csz, statstream, corrjoin) with the tracked metrics (recall, precision, F1, candidate precision, specificity, phase and total times, step-latency ticks, peak/mean RSS, I/O, energy).",
              "- The options differ only in: the synthetic stream length (A: 20,000 obs, others 5,000), which arms run at m >= 2,500 (C, D and -mem: bruteforce + bf_incremental + the pruning arms), whether the negative run exists at L > 1 (D, E: no), and the memory cut (-mem: no (m=5000, L=5) synthetic rung at density >= 0.05, Berkeley full-m at T >= 0.8 only). Nothing else is removed.",
              "", "Options:"] + [f"- **{k}**: {v['desc']}" for k, v in OPTIONS.items() if k in chosen]
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
