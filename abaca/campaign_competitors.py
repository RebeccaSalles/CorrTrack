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

    python abaca/campaign_competitors.py --design          # the (m, L) points per dataset
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
    preprocess: bool = False          # run the cell on first differences (returns): truth, tuning and every arm alike
    config_path: str | None = None    # dataset config elsewhere than the repo root (generated synthetic sets)

    @property
    def config(self):
        return self.config_path or f"experiment_dataset_{self.dataset}.py"

    @property
    def stem(self):
        m = f"_m{self.n_series}" if self.n_series is not None else ""
        return f"{self.dataset}{m}_W{self.W}_s{self.step}_L{self.n_lags}_T{self.T}{'_diff' if self.preprocess else ''}"

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
    # m_max: 5,000 cap (user, 2026-09-17; was 2,000), datasets above it keep a full-m anchor (Berkeley 18,520; gas 5,120).
    # horizon rule (log entry (h)): one application horizon per sampling regime, shared by every dataset of that regime.
    # sub-daily sensors -> a day (Motes: 2,880 epochs of 31 s, like smartmeter's 48 half-hours); L_max=3 -> 576 epochs (~5 h) covers BRAID's 224 min
    DatasetSpec("motes_temperature", 2880, 288, 27, 3, m_min=8, n_obs=40000, walltime="24:00:00", note="BRAID Motes 31 s epochs; W a day, step 2.5 h; L_max=3 -> 576 epochs covers the 224 min lag"),
    DatasetSpec("motes_humidity", 2880, 288, 31, 3, m_min=8, n_obs=40000, walltime="24:00:00", note="BRAID Motes humidity, same regime"),
    DatasetSpec("yellowstone_bp3_7", 2000, 100, 28, 11, m_min=8, walltime="24:00:00", calib_obs=30000, extra={"corrjoin_ks": 20, "corrjoin_ke": 40}, note="FilCorr 100 Hz, W 20 s, step 1 s, L_max=11 -> lag 10 s"),
    DatasetSpec("yellowstone_raw", 2000, 100, 28, 11, m_min=8, walltime="24:00:00", calib_obs=30000, extra={"corrjoin_ks": 20, "corrjoin_ke": 40}, note="FilCorr raw counts"),
    DatasetSpec("uscrn2020_temperature", 168, 12, 153, 5, extra=dict(CORRJOIN_KNOBS), note="TSUBASA NOAA hourly; L_max=5 -> lag 48 h"),
    DatasetSpec("berkeley_tavg_anom_2010", 90, 10, 5000, 4, m_min=100, extra={"corrjoin_ks": 9, "corrjoin_ke": 18}, walltime="48:00:00", full_m=18520, note="TSUBASA Berkeley Earth daily; L_max=4 -> lag 30 d; full 18,520 anchor for the exact arms"),
    DatasetSpec("corrjoin_stock", 60, 5, 3878, 1, m_min=100, walltime="24:00:00", note="CorrJoin daily NASDAQ closes: daily-finance regime (a quarter / a week, as sp500); 15 and 30 divide 60"),
    DatasetSpec("corrjoin_chlorine", 240, 24, 4830, 1, m_min=100, walltime="24:00:00", note="CorrJoin chlorine"),
    DatasetSpec("corrjoin_gas", 240, 24, 5000, 1, m_min=100, walltime="24:00:00", note="CorrJoin gas"),
    DatasetSpec("corrjoin_random", 240, 24, 5000, 1, m_min=100, walltime="24:00:00", note="CorrJoin i.i.d. uniform (uncooperative)"),
    DatasetSpec("statstream_rw_m5000_T20000", 256, 32, 5000, 5, m_min=100, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="24:00:00", note="StatStream random walks, CSZ's sw=256/bw=32 (generate first)"),
    DatasetSpec("braid_sines_m5000_T32768", 1024, 128, 5000, 3, m_min=100, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="24:00:00", note="BRAID Sines, planted lags <= 168 (generate first)"),
    DatasetSpec("braid_spiketrains_m5000_T100000", 1024, 128, 5000, 3, m_min=100, n_obs=40000, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="48:00:00", note="BRAID SpikeTrains (generate first)"),
    # the six real sets of the 2026-09 CorrTrack sweeps, at their historical W/step (added 2026-09-17)
    DatasetSpec("sp500", 60, 5, 492, 5, m_min=32, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, note="daily financial: W a quarter, step a week, L_max=5 -> lag a month"),
    DatasetSpec("acwi_capweighted", 60, 5, 263, 5, m_min=32, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, note="daily financial, same class as sp500"),
    DatasetSpec("streamflow", 30, 3, 538, 6, m_min=32, extra={"corrjoin_ks": 5, "corrjoin_ke": 10}, note="daily hydrology: W a month, L_max=6 -> lag 15 d"),
    DatasetSpec("wikipedia", 30, 3, 88, 6, m_min=16, extra={"corrjoin_ks": 5, "corrjoin_ke": 10}, note="daily page views"),
    DatasetSpec("global_weather", 30, 3, 100, 6, m_min=16, extra={"corrjoin_ks": 5, "corrjoin_ke": 10}, note="daily weather"),
    DatasetSpec("smartmeter", 48, 8, 510, 3, m_min=32, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, walltime="24:00:00", note="half-hourly: W a day, step 4 h, L_max=3 -> lag 16 h"),
    # ASOS airports (hourly weather regime, a week / 12 h, L_max=5 -> 48 h), the project's original real data; last 1 year
    DatasetSpec("fr_air_temperature_121_1", 168, 12, 121, 5, m_min=16, extra=dict(CORRJOIN_KNOBS), note="ASOS France hourly air temperature (the 2026-09-12 four-way dataset)"),
    DatasetSpec("fr_wind_direction_121_1", 168, 12, 121, 5, m_min=16, extra=dict(CORRJOIN_KNOBS), note="ASOS France hourly wind direction (uncooperative)"),
    DatasetSpec("fr_wind_speed_121_1", 168, 12, 121, 5, m_min=16, extra=dict(CORRJOIN_KNOBS), note="ASOS France hourly wind speed"),
    DatasetSpec("br_air_temperature_146_1", 168, 12, 146, 5, m_min=16, extra=dict(CORRJOIN_KNOBS), note="ASOS Brazil hourly air temperature"),
    DatasetSpec("br_wind_direction_146_1", 168, 12, 146, 5, m_min=16, extra=dict(CORRJOIN_KNOBS), note="ASOS Brazil hourly wind direction (uncooperative)"),
    DatasetSpec("br_flights_109_1", 168, 12, 109, 5, m_min=16, extra=dict(CORRJOIN_KNOBS), note="ASOS Brazil hourly flight counts (airport operations, hourly regime)"),
    # global ASOS (181+ countries, companion session's fetch of 2026-09-17): last 2 years, stations with >= 90% coverage (~670), hourly regime
    DatasetSpec("global_asos_air_temperature", 168, 12, 600, 5, m_min=32, extra=dict(CORRJOIN_KNOBS), walltime="24:00:00", note="global ASOS hourly air temperature, best-covered 600 stations"),
    DatasetSpec("global_asos_wind_speed", 168, 12, 600, 5, m_min=32, extra=dict(CORRJOIN_KNOBS), walltime="24:00:00", note="global ASOS hourly wind speed (uncooperative)"),
    DatasetSpec("global_asos_relative_humidity", 168, 12, 600, 5, m_min=32, extra=dict(CORRJOIN_KNOBS), walltime="24:00:00", note="global ASOS hourly relative humidity"),
    DatasetSpec("global_asos_pressure", 168, 12, 600, 5, m_min=32, extra=dict(CORRJOIN_KNOBS), walltime="24:00:00", note="global ASOS hourly pressure"),
]

# density-controlled synthetic sets (datasets/fetch/gen_density_targeted.py, user 2026-09-18): one stationary
# process (ar1) and one nonstationary (random walk); densities of plan section 5 item 6; m up to the 5k cap with
# the same ladder as every other dataset (one file per rung, since the verified density is a property of the
# protocol (m, W, step, L, T)); hourly-regime protocol W=168/12, L_max=5, as the unitless synthetic sets have no
# horizon of their own. The effective degree of each file is recorded in its meta.
SYNTH_PROCS = ("ar1", "rw")
SYNTH_DENSITIES = (0.005, 0.02, 0.05)
SYNTH_N, SYNTH_W, SYNTH_STEP = 20000, 168, 12
SYNTH_SPEC = DatasetSpec("synth", SYNTH_W, SYNTH_STEP, 5000, 5, m_min=625)


def synth_name(proc, dens, T, m, L):
    return f"synth_{proc}_d{str(dens).replace('.', 'p')}_T{str(T).replace('.', 'p')}_m{m}_L{L}"


def synth_rungs():
    return design_points(SYNTH_SPEC, 4, 4, 2, "ladder")


STEP1_CELL = Cell("yellowstone_bp3_7", 2000, 1, 1000, 0.9, n_series=28, n_obs=30000, arms="bruteforce,exact_stomp,filcorr,braid,thinbraid,corrtrack,statstream",
                  walltime="48:00:00", calib_obs=9000, note="step=1 lagged cell (BRAID probes 70 lags vs exact_stomp 15 at step=12); step != basic_window so the grid arms are excluded")


def sobol_unit(n_points: int, d: int, seed: int):
    """Scrambled Sobol points in the unit cube, the same sequence for every dataset (mirrors the
    2026-09-16 real-data sweep: scipy.stats.qmc.Sobol, scrambled, fixed seed)."""
    from scipy.stats import qmc
    if n_points <= 0:
        return np.zeros((0, d))
    n_pow2 = 1 << max(0, int(np.ceil(np.log2(n_points))))
    return qmc.Sobol(d=d, scramble=True, seed=seed).random(n_pow2)[:n_points]


def _map_m(u, d):
    return int(round(np.exp(np.log(d.m_min) + u * (np.log(d.m_max) - np.log(d.m_min)))))


def design_points(d: DatasetSpec, m_levels: int = 4, l_levels: int = 4, replicates: int = 2, kind: str = "ladder", seed: int = 20260917):
    """(m, L) design per dataset. Default `kind="ladder"` (2026-09-17, user): m and L grow together
    along one work ladder plus the synchronous anchor, see below. `kind="latin"`: explicit marginal
    levels and a Latin-square pairing, so every level of m and every level of L appears exactly once
    per replicate.
      m levels: geometric from m_max downwards, m_max / 2^k for k = 0 .. m_levels-1, floored at m_min
      L levels: l_levels integers evenly spaced in [1, L_max] (1 always included; fewer if L_max is small)
      replicate r pairs m level i with L level (i + r * shift) mod n_L, shift = ceil(n_L / replicates)
    Replicate 0 is the diagonal and contains the synchronous anchor (m_max, L = 1). `kind="sobol"`
    keeps the earlier scrambled-Sobol draw (anchor + 1-D sync points + 2-D lagged points)."""
    if kind == "ladder":
        # (2026-09-17, user) L on its own mostly multiplies the pair-window count (the 2026-09-16 sweeps
        # showed little L effect beyond that), so m and L grow together along one work ladder:
        # level j = 0 .. m_levels-1: m_j = m_max / 2^(m_levels-1-j), L_j = 1 + round(j (L_max-1) / (m_levels-1)),
        # i.e. (m_max/8, 1) .. (m_max, L_max); plus the synchronous anchor (m_max, 1) for the primary
        # head-to-head. No replicates. Work ~ m^2 L grows ~ 4x to 8x per rung.
        pts = [(d.m_max, 1)]
        n = max(2, m_levels)
        for j in range(n):
            m = max(d.m_min, int(round(d.m_max / 2 ** (n - 1 - j))))
            L = 1 + int(round(j * (d.L_max - 1) / (n - 1))) if d.L_max > 1 else 1
            pts.append((m, L))
    elif kind == "sobol":
        pts = [(d.m_max, 1)]
        for u in sobol_unit(m_levels - 1, 1, seed)[:, 0]:
            pts.append((_map_m(u, d), 1))
        if d.L_max >= 2:
            for u in sobol_unit(l_levels, 2, seed + 1):
                L = 2 + int(np.floor(u[1] * (d.L_max - 1)))
                pts.append((_map_m(u[0], d), min(L, d.L_max)))
    else:
        ms = []
        for k in range(m_levels):
            m = max(d.m_min, int(round(d.m_max / 2 ** k)))
            if m not in ms:
                ms.append(m)
        Ls = sorted({int(round(v)) for v in np.linspace(1, d.L_max, num=min(l_levels, d.L_max))})
        n_L = len(Ls)
        shift = max(1, int(np.ceil(n_L / max(replicates, 1))))
        pts = []
        for r in range(replicates if n_L > 1 else 1):
            for i, m in enumerate(ms):
                pts.append((m, Ls[(i + r * shift) % n_L]))
    out = []
    for p in pts:
        if p not in out:
            out.append(p)
    return out


def cells(m_levels: int = 4, l_levels: int = 4, replicates: int = 2, kind: str = "ladder", seed: int = 20260917) -> list[Cell]:
    """Per dataset: design_points() cells, every one at every T in THRESHOLDS; datasets with more series
    than the 2k cap (Berkeley Earth, CorrJoin's files) add a full-m synchronous anchor for the exact
    arms + StatStream (the pure-Python pruning indexes are excluded there until the Cython ports land). Rules checked programmatically: step passed as
    basic_window (ParCorr's step == basic_window), n_lags a multiple of step (StatStream's lag
    granularity), ks and ke divide W."""
    out = []
    for d in DATASETS:
        designs = design_points(d, m_levels, l_levels, replicates, kind, seed)
        if d.full_m:
            designs.append((d.full_m, 1))
        for (m, L) in designs:
            for T in THRESHOLDS:
                arms = d.arms
                if arms == "all" and m > 2000 and L > 1:
                    # plain BRAID keeps an m x m matrix per (level, lag): ~200 MB each at m = 5,000, tens of GB
                    # over the probe set; ThinBRAID is its large-m form (their section 5), so it stands in
                    arms = "bruteforce,exact_stomp,filcorr,tsubasa,thinbraid,corrtrack,parcorr,csz,statstream,corrjoin"
                if d.full_m and m == d.full_m:
                    # above the 2k cap: plain BRAID excluded (O(m^2) per-pair state, ~2.2 GB at 2k); the pruning
                    # arms are included since their candidate loops run in competitor_kernels (2026-09-17)
                    arms = "bruteforce,exact_stomp,filcorr,tsubasa,thinbraid,corrtrack,parcorr,csz,statstream,corrjoin"
                out.append(Cell(d.label, d.W, d.step, (L - 1) * d.step, T, n_series=m, n_obs=d.n_obs, arms=arms,
                                extra=dict(d.extra), walltime=("96:00:00" if (d.full_m and m == d.full_m) else d.walltime), calib_obs=d.calib_obs,
                                note=f"{d.note} | L={L}"))
    # (2026-09-18) returns / differences run: every dataset's synchronous anchor also runs with preprocess=True
    # (the cooperative-vs-uncooperative axis of plan section 5 item 6: prices AND returns, one transform)
    for d in DATASETS:
        for T in THRESHOLDS:
            out.append(Cell(d.label, d.W, d.step, 0, T, n_series=d.m_max, n_obs=d.n_obs, arms=d.arms, extra=dict(d.extra),
                            walltime=d.walltime, calib_obs=d.calib_obs, preprocess=True, note=f"{d.note} | L=1, first differences"))
    # (2026-09-18) this project's own density-controlled generator (plan section 5 item 6, density {0.5%, 2%, 5%}):
    # one file per (process, density, T) because the verified density is a property of the protocol; m and L fixed
    for proc in SYNTH_PROCS:
        for dens in SYNTH_DENSITIES:
            for (m, L) in synth_rungs():
                for T in THRESHOLDS:
                    name = synth_name(proc, dens, T, m, L)
                    arms = "all" if not (m > 2000 and L > 1) else "bruteforce,exact_stomp,filcorr,tsubasa,thinbraid,corrtrack,parcorr,csz,statstream,corrjoin"
                    out.append(Cell(name, SYNTH_W, SYNTH_STEP, (L - 1) * SYNTH_STEP, T, n_series=m, n_obs=SYNTH_N, arms=arms,
                                    extra=dict(CORRJOIN_KNOBS), walltime="24:00:00" if m <= 2500 else "48:00:00",
                                    config_path=f"datasets/competitor/configs/experiment_dataset_{name}.py",
                                    note=f"synthetic {proc}, verified tuple density {dens} at T={T}, m={m}, L={L} (gen_density_targeted.py)"))
    out.append(STEP1_CELL)
    return out


GENERATE = [
    # density-controlled synthetic sets: one per (process, density, T), verified by bruteforce at generation
] + [f"python datasets/fetch/gen_density_targeted.py --proc {p} --density {d} --T {T} --m {m} --n {SYNTH_N} --W {SYNTH_W} --step {SYNTH_STEP} --L {L}"
     for p in SYNTH_PROCS for d in SYNTH_DENSITIES for (m, L) in synth_rungs() for T in THRESHOLDS] + [
    "python datasets/fetch/gen_statstream_randomwalk.py --m 5000 --T 20000",
    "python datasets/fetch/gen_braid_synthetic.py --family sines --m 5000 --T 32768 --copies 1",
    "python datasets/fetch/gen_braid_synthetic.py --family spiketrains --m 5000 --T 100000 --period 6500 --copies 1",
]


def emit(path: str, results_root: str, select=None, m_levels: int = 4, l_levels: int = 4, replicates: int = 2, kind: str = "ladder", seed: int = 20260917) -> None:
    n_cells = 0
    lines = ["#!/bin/bash", "# generated by abaca/campaign_competitors.py; run from the CorrTrack working tree on the Sophia frontend",
             "set -uo pipefail", f"RESULTS_ROOT=${{RESULTS_ROOT:-{results_root}}}", "mkdir -p abaca/logs",
             # -q abaca: the group's queue; a bare oarsub reports 'not enough resources' (2026-09-17)
             "submit() { oarsub -q abaca \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }", "",
             "# synthetic inputs (cheap, frontend-side)"] + GENERATE + [""]
    for c in cells(m_levels, l_levels, replicates, kind, seed):
        if select and c.stem not in select:
            continue
        common = [f"DATASET_CONFIG={c.config}", f"WINDOW_SIZE={c.W}", f"WINDOW_STEP={c.step}", f"BASIC_WINDOW={c.step}", f"N_LAGS={c.n_lags}", f"THR={c.T}",
                  f"PREPROCESS={1 if c.preprocess else 0}",
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
    ap.add_argument("--m-levels", type=int, default=4, help="m levels: m_max / 2^k, k < m_levels")
    ap.add_argument("--l-levels", type=int, default=4, help="L levels evenly spaced in [1, L_max]")
    ap.add_argument("--replicates", type=int, default=2, help="Latin-square replicates (each m and L level appears once per replicate)")
    ap.add_argument("--design-kind", choices=("ladder", "latin", "sobol"), default="ladder")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--design", action="store_true", help="print the (m, L) design per dataset and exit")
    args = ap.parse_args()
    if args.design:
        print(f"{'dataset':32s} {'W':>5s} {'step':>4s} {'L_max':>5s} {'m_full':>6s}  (m, L windows) cells x {len(THRESHOLDS)} T")
        for d in DATASETS:
            print(f"{d.label:32s} {d.W:5d} {d.step:4d} {d.L_max:5d} {str(d.full_m or '-'):>6s}  {design_points(d, args.m_levels, args.l_levels, args.replicates, args.design_kind, args.seed)}")
        return
    cs = cells(args.m_levels, args.l_levels, args.replicates, args.design_kind, args.seed)
    if args.list or not args.emit:
        print(f"{'cell':60s} {'m':>6s} {'n_obs':>6s} {'wall':>9s} note")
        for c in cs:
            print(f"{c.stem:60s} {str(c.n_series or 'cfg'):>6s} {str(c.n_obs or 'cfg'):>6s} {c.walltime:>9s} {c.note}")
        print(f"{len(cs)} cells x 6 jobs")
    if args.emit:
        emit(args.emit, args.results_root, args.select, args.m_levels, args.l_levels, args.replicates, args.design_kind, args.seed)


if __name__ == "__main__":
    main()
