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
        return f"{self.dataset}_W{self.W}_s{self.step}_L{self.n_lags}_T{self.T}"

    def extra_args(self):
        parts = []
        if self.n_series is not None:
            parts += ["--n-series", str(self.n_series)]
        if self.n_obs is not None:
            parts += ["--n-obs", str(self.n_obs)]
        for k, v in self.extra.items():
            parts += ["--set", f"{k}={v}"]
        return " ".join(parts)


THRESHOLDS = (0.7, 0.8, 0.9, 0.95)
CORRJOIN_KNOBS = {"corrjoin_ks": 14, "corrjoin_ke": 28}   # for W=168 (15, 30 do not divide it); W=240 cells keep the paper's 15/30


def cells() -> list[Cell]:
    """W / step / L per dataset: PROPOSALS of 2026-09-17 (log entry (f)), awaiting the user's decision.
    Rules applied throughout: step == basic_window (ParCorr's structural constraint; every step
    below divides W with W's parity so the library infers exactly that), W/step about 10:1, L a
    multiple of step (StatStream resolves lags at basic-window multiples), L set by the physics of
    the data where known, and W divisible by ks=15 and ke=30 where possible (W=240) so CorrJoin
    keeps its paper knobs (W=168 keeps the 2026-09-12 hourly convention and uses 14/28)."""
    out = []
    # BRAID's real lag set: Motes (31 s epochs). W=240 epochs (~2 h), step 24 (~12 min); L=480 (~4.1 h)
    # covers the 224 min (433 epoch) lag BRAID reports; the L=0 cell is the synchronous head-to-head.
    for var in ("temperature", "humidity"):
        for T in THRESHOLDS:
            out.append(Cell(f"motes_{var}", 240, 24, 0, T, n_obs=20000, note="BRAID Motes, synchronous"))
            out.append(Cell(f"motes_{var}", 240, 24, 480, T, n_obs=20000, walltime="24:00:00", note="BRAID Motes, lags cover 202/224 min"))
    # FilCorr's case study: 100 Hz, W=2000 (20 s), lag 1000 (10 s), band 3-7 Hz; step 100 = 1 s output rate.
    for T in THRESHOLDS:
        out.append(Cell("yellowstone_bp3_7", 2000, 100, 1000, T, extra={"corrjoin_ks": 20, "corrjoin_ke": 40}, walltime="24:00:00", calib_obs=30000, note="FilCorr Yellowstone, band-passed"))
        out.append(Cell("yellowstone_raw", 2000, 100, 1000, T, extra={"corrjoin_ks": 20, "corrjoin_ke": 40}, walltime="24:00:00", calib_obs=30000, note="FilCorr Yellowstone, raw (FilCorr applies its own band)"))
    out.append(Cell("yellowstone_bp3_7", 2000, 1, 1000, 0.9, n_obs=30000, arms="bruteforce,exact_stomp,filcorr,braid,thinbraid,corrtrack,statstream",
                    walltime="48:00:00", calib_obs=9000, note="step=1 lagged cell (BRAID probes 70 lags vs exact_stomp 15 at step=12); step != basic_window so the grid arms are excluded"))
    # TSUBASA's climate sets. USCRN hourly: W=168 (a week), step 12, L=48 (fronts cross the network in hours to two days).
    for T in THRESHOLDS:
        out.append(Cell("uscrn2020_temperature", 168, 12, 0, T, extra=dict(CORRJOIN_KNOBS), note="TSUBASA NOAA hourly, synchronous"))
        out.append(Cell("uscrn2020_temperature", 168, 12, 48, T, extra=dict(CORRJOIN_KNOBS), walltime="24:00:00", note="TSUBASA NOAA hourly, lagged 2 days"))
    # Berkeley Earth daily: W=90 (a season), step 10; synchronous m-sweep, one lagged cell (a month) at 2k.
    for T in (0.7, 0.9):
        for m in (500, 1000, 2000):
            out.append(Cell("berkeley_tavg_anom_2010", 90, 10, 0, T, n_series=m, extra={"corrjoin_ks": 9, "corrjoin_ke": 18}, walltime="24:00:00", note="TSUBASA Berkeley Earth, m sweep"))
        out.append(Cell("berkeley_tavg_anom_2010", 90, 10, 0, T, n_series=18520, arms="bruteforce,exact_stomp,filcorr,tsubasa,thinbraid,corrtrack,statstream",
                        walltime="96:00:00", note="scalability at full m (plain BRAID excluded: per-pair state)"))
    out.append(Cell("berkeley_tavg_anom_2010", 90, 10, 30, 0.9, n_series=2000, extra={"corrjoin_ks": 9, "corrjoin_ke": 18}, walltime="48:00:00", note="Berkeley Earth, lagged a month, 2k"))
    # CorrJoin's own benchmark files: W=240, step 24 keeps ks/ke = 15/30 (paper W=1020 is Phase R's setting).
    for ds in ("corrjoin_stock", "corrjoin_chlorine", "corrjoin_gas"):
        for T in THRESHOLDS:
            out.append(Cell(ds, 240, 24, 0, T, n_series=2000, walltime="24:00:00", note="CorrJoin real sets, m capped at 2000"))
    # Synthetic: StatStream random walks (cooperative, W=256/b=32 as in CSZ), CorrJoin random (uncooperative), BRAID families (lag anchors)
    for T in (0.7, 0.9):
        out.append(Cell("statstream_rw_m2000_T20000", 256, 32, 0, T, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, note="StatStream random walks (generate first)"))
        out.append(Cell("statstream_rw_m2000_T20000", 256, 32, 64, T, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="24:00:00", note="StatStream random walks, lags = 2 basic windows"))
        out.append(Cell("corrjoin_random", 240, 24, 0, T, n_series=2000, note="uncooperative i.i.d."))
        out.append(Cell("braid_sines_m2000_T32768", 1024, 128, 256, T, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="24:00:00", note="BRAID Sines with planted lags (generate first)"))
        out.append(Cell("braid_spiketrains_m2000_T100000", 1024, 128, 256, T, n_obs=40000, extra={"corrjoin_ks": 16, "corrjoin_ke": 32}, walltime="48:00:00", note="BRAID SpikeTrains (generate first)"))
    # licensed stock sets' stand-in (daily): W=60 (a quarter), step 10, L=20; step 5 of the 2026-09-16 sweep breaks ParCorr's step == basic_window
    for T in THRESHOLDS:
        out.append(Cell("sp500_sub263", 60, 10, 20, T, extra={"corrjoin_ks": 6, "corrjoin_ke": 12}, note="stand-in for TAQ / CRSP / Yahoo"))
    return out


GENERATE = [
    "python datasets/fetch/gen_statstream_randomwalk.py --m 2000 --T 20000",
    "python datasets/fetch/gen_braid_synthetic.py --family sines --m 2000 --T 32768",
    "python datasets/fetch/gen_braid_synthetic.py --family spiketrains --m 2000 --T 100000 --period 6500",
]


def emit(path: str, results_root: str, select=None) -> None:
    n_cells = 0
    lines = ["#!/bin/bash", "# generated by abaca/campaign_competitors.py; run from the CorrTrack working tree on the Sophia frontend",
             "set -uo pipefail", f"RESULTS_ROOT=${{RESULTS_ROOT:-{results_root}}}", "mkdir -p abaca/logs",
             "submit() { oarsub \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }", "",
             "# synthetic inputs (cheap, frontend-side)"] + GENERATE + [""]
    for c in cells():
        if select and c.stem not in select:
            continue
        common = [f"DATASET_CONFIG={c.config}", f"WINDOW_SIZE={c.W}", f"WINDOW_STEP={c.step}", f"N_LAGS={c.n_lags}", f"THR={c.T}",
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
    args = ap.parse_args()
    cs = cells()
    if args.list or not args.emit:
        print(f"{'cell':52s} {'m':>6s} {'n_obs':>6s} {'wall':>9s} note")
        for c in cs:
            print(f"{c.stem:52s} {str(c.n_series or 'cfg'):>6s} {str(c.n_obs or 'cfg'):>6s} {c.walltime:>9s} {c.note}")
        print(f"{len(cs)} cells x 6 jobs")
    if args.emit:
        emit(args.emit, args.results_root, args.select)


if __name__ == "__main__":
    main()
