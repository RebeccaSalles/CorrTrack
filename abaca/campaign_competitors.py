"""Campaign manifest for the competitor evaluation (implementation plan sections 5 and 6;
comparison plan section 6) and the oarsub script that runs it on Abaca.

One *cell* = (dataset config, W, step, n_lags, T, m cap). For every cell and for each of the
two labelled runs (neg_corr=False primary, neg_corr=True second run) the campaign submits:
  1. hyperopt_corrtrack.oar    CorrTrack's own proxy-anchor hyperopt at that (W, step, lags, T, neg_corr)
  2. tune_competitors.oar      CSZ protocol for parcorr/csz/statstream/corrjoin (skips not_available arms under neg_corr)
  3. nway_compare.oar          all arms, CorrTrack from 1, competitors from 2 (OAR -a dependency on both)
CorrTrack is therefore tuned by its own procedure in every execution (decided 2026-09-17); a
cell whose hyperopt did not produce best_params runs UNTUNED and says so in the output.

Protocol choices per dataset are the 2026-09-17 proposals in cells() (awaiting the user's
decision): Motes W=240/24/L=480 epochs (covers BRAID's 224 min); Yellowstone W=2000/100/L=1000
(FilCorr's 20 s / 1 s / 10 s); USCRN W=168/12/L=48; Berkeley W=90/10; CorrJoin sets W=240/24 so
ks/ke = 15/30 hold. T sweep {0.7, 0.8, 0.9, 0.95} (0.9+ is where the grid methods prune). The lagged comparison includes one
step=1 cell (Yellowstone slice) because BRAID probes 70 lags where exact_stomp probes 15 at
step=12 (comparison plan section 5a.3). m is capped at 2000 (the battery target); Berkeley
Earth additionally runs the exact arms plus StatStream at the full 18,520 for scalability.

    python abaca/campaign_competitors.py --list
    python abaca/campaign_competitors.py --emit abaca/campaign_submit.sh   # then: bash abaca/campaign_submit.sh
    python abaca/campaign_competitors.py --emit abaca/pilot_submit.sh --select motes_temperature_W96_s12_L0_T0.9 \
        --select motes_temperature_W96_s12_L480_T0.9 --select yellowstone_bp3_7_W2000_s200_L1000_T0.9
"""
from __future__ import annotations

import argparse
import shlex

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Cell:
    dataset: str
    W: int
    step: int
    n_lags: int
    T: float
    n_series: int | None = None
    n_obs: int | None = None
    arms: str = "all"
    extra: dict = field(default_factory=dict)
    walltime: str = "12:00:00"
    calib_obs: int = 5000
    note: str = ""

    @property
    def config(self):
        return f"experiment_dataset_{self.dataset}.py"

    @property
    def stem(self):
        m = f"_m{self.n_series}" if self.n_series is not None else ""
        return f"{self.dataset}{m}_W{self.W}_s{self.step}_L{self.n_lags}_T{self.T}"

    def extra_args(self):
        parts = []
        if self.n_series is not None:
            parts += ["--n-series", str(self.n_series)]
        if self.n_obs is not None:
            parts += ["--n-obs", str(self.n_obs)]
        for k, v in self.extra.items():
            parts += ["--set", f"{k}={v}"]
        return " ".join(parts)


THRESHOLDS = (0.7, 0.8, 0.9, 0.95)          # swept in full at every design point (user, 2026-09-17)
CORRJOIN_KNOBS = {"corrjoin_ks": 14, "corrjoin_ke": 28}   # for W=168 (15, 30 do not divide it); W=240 cells keep the paper's 15/30


@dataclass
class DatasetSpec:
    """One dataset's protocol. W / step are fixed per dataset (proposal of 2026-09-17, log entry (f),
    awaiting the user's decision); m and L come from the Sobol design below."""
    label: str
    W: int
    step: int
    m_max: int                       # series available (capped at the 2k battery target)
    L_max: int                       # max lagged windows; n_lags = (L - 1) * step, L = 1 is synchronous
    m_min: int = 16
    n_obs: int | None = None
    arms: str = "all"
    extra: dict = field(default_factory=dict)
    walltime: str = "12:00:00"
    calib_obs: int = 5000
    note: str = ""
    full_m: int | None = None        # extra synchronous anchor at the full series count (exact arms + StatStream)


DATASETS = [
    # competitor papers' own data (registry datasets/competitor_sources.md)
    DatasetSpec("motes_temperature", 240, 24, 27, 21, m_min=8, n_obs=20000, note="BRAID Motes 31 s epochs; L_max=21 -> n_lags 480 (~4.1 h) covers the 224 min lag"),
    DatasetSpec("motes_humidity", 240, 24, 31, 21, m_min=8, n_obs=20000, note="BRAID Motes humidity"),
    DatasetSpec("yellowstone_bp3_7", 2000, 100, 28, 11, m_min=8, walltime="24:00:00", calib_obs=30000, extra={"corrjoin_ks": 20, "corrjoin_ke": 40}, note="FilCorr 100 Hz, W 20 s, step 1 s, L_max=11 -> lag 10 s"),
    DatasetSpec("yellowstone_raw", 2000, 100, 28, 11, m_min=8, walltime="24:00:00", calib_obs=30000, extra={"corrjoin_ks": 20, "corrjoin_ke": 40}, note="FilCorr raw counts"),
    DatasetSpec("uscrn2020_temperature", 168, 12, 153, 5, extra=dict(CORRJOIN_KNOBS), note="TSUBASA NOAA hourly; L_max=5 -> lag 48 h"),
    DatasetSpec("berkeley_tavg_anom_2010", 90, 10, 2000, 4, m_min=100, extra={"corrjoin_ks": 9, "corrjoin_ke": 18}, walltime="48:00:00", full_m=18520, note="TSUBASA Berkeley Earth daily; L_max=4 -> lag 30 d; full 18,520 anchor for the exact arms"),
    DatasetSpec("corrjoin_stock", 240, 24, 2000, 1, m_min=100, walltime="24:00:00", note="CorrJoin daily prices; synchronous method, L fixed at 1"),
    DatasetSpec("corrjoin_chlorine", 240, 24, 2000, 1, m_min=100, walltime="24:00:00", note="CorrJoin chlorine"),
    DatasetSpec("corrjoin_gas", 240, 24, 2000, 1, m_min=100, walltime="24:00:00", note="CorrJoin gas"),
    DatasetSpec("corrjoin_random", 240, 24, 2000, 1, m_min=100, walltime="24:00:00", note="CorrJoin i.i.d. uniform (uncooperative)"),
    DatasetSpec("statstream_rw_m2000_T20000", 256, 32, 2000, 5, m_min=100, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="24:00:00", note="StatStream random walks, CSZ's sw=256/bw=32 (generate first)"),
    DatasetSpec("braid_sines_m2000_T32768", 1024, 128, 2000, 3, m_min=100, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="24:00:00", note="BRAID Sines, planted lags <= 168 (generate first)"),
    DatasetSpec("braid_spiketrains_m2000_T100000", 1024, 128, 2000, 3, m_min=100, n_obs=40000, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="48:00:00", note="BRAID SpikeTrains (generate first)"),
    # the six real sets of the 2026-09 CorrTrack sweeps, at their historical W/step (added 2026-09-17)
    DatasetSpec("sp500", 60, 5, 492, 5, m_min=32, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, note="daily financial: W a quarter, step a week, L_max=5 -> lag a month"),
    DatasetSpec("acwi_capweighted", 60, 5, 263, 5, m_min=32, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, note="daily financial, same class as sp500"),
    DatasetSpec("streamflow", 30, 3, 538, 6, m_min=32, extra={"corrjoin_ks": 5, "corrjoin_ke": 10}, note="daily hydrology: W a month, L_max=6 -> lag 15 d"),
    DatasetSpec("wikipedia", 30, 3, 88, 6, m_min=16, extra={"corrjoin_ks": 5, "corrjoin_ke": 10}, note="daily page views"),
    DatasetSpec("global_weather", 30, 3, 100, 6, m_min=16, extra={"corrjoin_ks": 5, "corrjoin_ke": 10}, note="daily weather"),
    DatasetSpec("smartmeter", 48, 8, 510, 3, m_min=32, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, walltime="24:00:00", note="half-hourly: W a day, step 4 h, L_max=3 -> lag 16 h"),
]

STEP1_CELL = Cell("yellowstone_bp3_7", 2000, 1, 1000, 0.9, n_series=28, n_obs=30000, arms="bruteforce,exact_stomp,filcorr,braid,thinbraid,corrtrack,statstream",
                  walltime="48:00:00", calib_obs=9000, note="step=1 lagged cell (BRAID probes 70 lags vs exact_stomp 15 at step=12); step != basic_window so the grid arms are excluded")


def sobol_points(n_points: int, seed: int = 20260917):
    """The same scrambled Sobol sequence in the unit square for every dataset, mapped per dataset to
    (m log-uniform in [m_min, m_max], L uniform integer in [1, L_max]); mirrors the 2026-09-16 real-data
    sweep design (scipy.stats.qmc.Sobol, scrambled, fixed seed) with T taken out of the sequence and
    swept in full instead."""
    from scipy.stats import qmc
    if n_points <= 0:
        return np.zeros((0, 2))
    n_pow2 = 1 << max(0, int(np.ceil(np.log2(n_points))))
    u = qmc.Sobol(d=2, scramble=True, seed=seed).random(n_pow2)
    return u[:n_points]


def cells(points: int = 4, seed: int = 20260917) -> list[Cell]:
    """Per dataset: one synchronous anchor (m = m_max, L = 1), `points` Sobol (m, L) points, and for
    Berkeley Earth the full-m anchor; every cell at every T in THRESHOLDS. Rules checked
    programmatically: step passed as basic_window (ParCorr's step == basic_window), n_lags a
    multiple of step (StatStream's lag granularity), ks and ke divide W."""
    u = sobol_points(points, seed)
    out = []
    for d in DATASETS:
        designs = [(d.m_max, 1)]
        for row in u:
            m = int(round(np.exp(np.log(d.m_min) + row[0] * (np.log(d.m_max) - np.log(d.m_min)))))
            L = 1 + int(np.floor(row[1] * d.L_max)) if d.L_max > 1 else 1
            L = min(L, d.L_max)
            if (m, L) not in designs:
                designs.append((m, L))
        if d.full_m:
            designs.append((d.full_m, 1))
        for (m, L) in designs:
            for T in THRESHOLDS:
                arms = d.arms
                if d.full_m and m == d.full_m:
                    arms = "bruteforce,exact_stomp,filcorr,tsubasa,thinbraid,corrtrack,statstream"
                out.append(Cell(d.label, d.W, d.step, (L - 1) * d.step, T, n_series=m if m != d.m_max or d.full_m else m, n_obs=d.n_obs, arms=arms,
                                extra=dict(d.extra), walltime=("96:00:00" if (d.full_m and m == d.full_m) else d.walltime), calib_obs=d.calib_obs,
                                note=f"{d.note} | L={L}"))
    out.append(STEP1_CELL)
    return out


GENERATE = [
    "python datasets/fetch/gen_statstream_randomwalk.py --m 2000 --T 20000",
    "python datasets/fetch/gen_braid_synthetic.py --family sines --m 2000 --T 32768",
    "python datasets/fetch/gen_braid_synthetic.py --family spiketrains --m 2000 --T 100000 --period 6500",
]


def emit(path: str, results_root: str, select=None, points: int = 4, seed: int = 20260917) -> None:
    n_cells = 0
    lines = ["#!/bin/bash", "# generated by abaca/campaign_competitors.py; run from the CorrTrack working tree on the Sophia frontend",
             "set -uo pipefail", f"RESULTS_ROOT=${{RESULTS_ROOT:-{results_root}}}", "mkdir -p abaca/logs",
             "submit() { oarsub \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }", "",
             "# synthetic inputs (cheap, frontend-side)"] + GENERATE + [""]
    for c in cells(points, seed):
        if select and c.stem not in select:
            continue
        common = [f"DATASET_CONFIG={c.config}", f"WINDOW_SIZE={c.W}", f"WINDOW_STEP={c.step}", f"BASIC_WINDOW={c.step}", f"N_LAGS={c.n_lags}", f"THR={c.T}",
                  "EXTRA_ARGS=" + c.extra_args().replace(" ", "+")]   # one token; the .oar scripts decode "+"
        res = f"-l host=1,walltime={c.walltime}"
        lines.append(f"# --- {c.stem}: {c.note}")
        for tag, neg in (("pos", ""), ("neg", "NEG_CORR_FLAG=--neg-corr")):
            run = common + ([neg] if neg else [])
            hyper = f"$RESULTS_ROOT/hyperopt/{c.stem}_{tag}"
            tuned = f"$RESULTS_ROOT/tuned/{c.stem}_{tag}"
            h_cmd = " ".join(["./abaca/hyperopt_corrtrack.oar"] + run + [f"OUT_DIR={hyper}"])
            t_cmd = " ".join(["./abaca/tune_competitors.oar"] + run + [f"CALIB_OBS={c.calib_obs}", f"OUT_DIR={tuned}"])
            n_cmd = " ".join(["./abaca/nway_compare.oar"] + run + [f"ARMS={c.arms}", f"COMPETITOR_PARAMS={tuned}", f"HYPEROPT_DIR={hyper}"])
            lines += [f"H_JOB=$(submit {res} -S \"{h_cmd}\")",
                      f"T_JOB=$(submit {res} -S \"{t_cmd}\")",
                      f"submit -a \"$H_JOB\" -a \"$T_JOB\" {res} -S \"{n_cmd}\""]
        lines.append("")
        n_cells += 1
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}: {n_cells} cells, {6 * n_cells} jobs (hyperopt, tune, N-way; x2 for the neg_corr run)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emit", default=None)
    ap.add_argument("--results-root", default="$HOME/corrtrack_abaca_results")
    ap.add_argument("--select", action="append", default=None, help="only cells with exactly this stem (repeatable); e.g. a pilot")
    ap.add_argument("--points", type=int, default=4, help="Sobol (m, L) points per dataset, on top of the synchronous anchor at m_max")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    cs = cells(args.points, args.seed)
    if args.list or not args.emit:
        print(f"{'cell':60s} {'m':>6s} {'n_obs':>6s} {'wall':>9s} note")
        for c in cs:
            print(f"{c.stem:60s} {str(c.n_series or 'cfg'):>6s} {str(c.n_obs or 'cfg'):>6s} {c.walltime:>9s} {c.note}")
        print(f"{len(cs)} cells x 6 jobs")
    if args.emit:
        emit(args.emit, args.results_root, args.select, args.points, args.seed)


if __name__ == "__main__":
    main()
