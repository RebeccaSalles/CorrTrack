"""Campaign manifest for the competitor evaluation (implementation plan sections 5 and 6;
comparison plan section 6) and the oarsub script that runs it on Abaca.

One *cell* = (dataset config, W, step, n_lags, T, m cap). For every cell the campaign submits,
in dependency order:
  1. tune_competitors.oar      CSZ protocol for parcorr/csz/statstream/corrjoin at that (W, step, lags, T)
  2. nway_compare.oar          primary head-to-head, neg_corr=False, all arms, tuned knobs
  3. nway_compare.oar --neg-corr  second labelled run (not_available arms report N/A)
CorrTrack's own hyperopt (corrtrack_param_search.py) is a separate, unchanged stage whose
best_params path is passed through BEST_PARAMS when it exists; without it nway reports
"UNTUNED defaults" and the cell is not quotable for CorrTrack.

Protocol choices per dataset follow datasets/competitor_sources.md (Motes lags >= 433 epochs
cover BRAID's 224 min; Yellowstone W=2000/lag 1000 = FilCorr's 20 s/10 s; CorrJoin sets use
ks, ke = (14, 28) at W=168 because 15 and 30 do not divide 168). T sweep {0.7, 0.8, 0.9} on the
real sets (0.9 is where the grid methods start to prune). The lagged comparison includes one
step=1 cell (Yellowstone slice) because BRAID probes 70 lags where exact_stomp probes 15 at
step=12 (comparison plan section 5a.3). m is capped at 2000 (the battery target); Berkeley
Earth additionally runs the exact arms plus StatStream at the full 18,520 for scalability.

    python abaca/campaign_competitors.py --list
    python abaca/campaign_competitors.py --emit abaca/campaign_submit.sh   # then: bash abaca/campaign_submit.sh
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


THRESHOLDS = (0.7, 0.8, 0.9)
CORRJOIN_KNOBS = {"corrjoin_ks": 14, "corrjoin_ke": 28}   # 15, 30 do not divide W=168


def cells() -> list[Cell]:
    out = []
    # BRAID's real lag set: Motes (31 s epochs). n_lags=480 covers the 224 min lag (433 epochs).
    for var in ("temperature", "humidity"):
        for T in THRESHOLDS:
            out.append(Cell(f"motes_{var}", 96, 12, 0, T, n_obs=20000, note="BRAID Motes, synchronous"))
            out.append(Cell(f"motes_{var}", 96, 12, 480, T, n_obs=20000, walltime="24:00:00", note="BRAID Motes, lags cover 202/224 min"))
    # FilCorr's case study: 100 Hz, 20 s windows, 10 s lag. Plus the step=1 lagged cell on a 5 min slice.
    for T in (0.7, 0.9):
        out.append(Cell("yellowstone_bp3_7", 2000, 200, 1000, T, walltime="24:00:00", calib_obs=30000, note="FilCorr Yellowstone, band-passed"))
        out.append(Cell("yellowstone_raw", 2000, 200, 1000, T, walltime="24:00:00", calib_obs=30000, note="FilCorr Yellowstone, raw (FilCorr applies its own band)"))
    out.append(Cell("yellowstone_bp3_7", 2000, 1, 1000, 0.9, n_obs=30000, arms="bruteforce,exact_stomp,filcorr,braid,thinbraid,corrtrack,statstream",
                    walltime="48:00:00", calib_obs=9000, note="step=1 lagged cell (BRAID probes 70 lags vs exact_stomp 15 at step=12)"))
    # TSUBASA's climate sets
    for T in THRESHOLDS:
        out.append(Cell("uscrn2020_temperature", 168, 12, 0, T, note="TSUBASA NOAA hourly, synchronous"))
        out.append(Cell("uscrn2020_temperature", 168, 12, 168, T, walltime="24:00:00", note="TSUBASA NOAA hourly, lagged"))
    for T in (0.7, 0.9):
        for m in (500, 1000, 2000):
            out.append(Cell("berkeley_tavg_anom_2010", 90, 10, 0, T, n_series=m, walltime="24:00:00", note="TSUBASA Berkeley Earth, m sweep"))
        out.append(Cell("berkeley_tavg_anom_2010", 90, 10, 0, T, n_series=18520, arms="bruteforce,exact_stomp,filcorr,tsubasa,thinbraid,corrtrack,statstream",
                        walltime="96:00:00", note="scalability at full m (plain BRAID excluded: per-pair state)"))
    # CorrJoin's own benchmark files
    for ds in ("corrjoin_stock", "corrjoin_chlorine", "corrjoin_gas"):
        for T in THRESHOLDS:
            out.append(Cell(ds, 168, 12, 0, T, n_series=2000, extra=dict(CORRJOIN_KNOBS), walltime="24:00:00", note="CorrJoin real sets, m capped at 2000"))
    # Synthetic: StatStream random walks (cooperative), CorrJoin random (uncooperative), BRAID families (lag anchors)
    for T in (0.7, 0.9):
        out.append(Cell("statstream_rw_m2000_T20000", 168, 12, 0, T, note="StatStream random walks (generate first)"))
        out.append(Cell("corrjoin_random", 168, 12, 0, T, n_series=2000, extra=dict(CORRJOIN_KNOBS), note="uncooperative i.i.d."))
        out.append(Cell("braid_sines_m2000_T32768", 168, 12, 168, T, walltime="24:00:00", note="BRAID Sines with planted lags (generate first)"))
        out.append(Cell("braid_spiketrains_m2000_T100000", 168, 12, 168, T, n_obs=40000, walltime="48:00:00", note="BRAID SpikeTrains (generate first)"))
    # licensed stock sets' stand-in
    for T in THRESHOLDS:
        out.append(Cell("sp500_sub263", 60, 5, 20, T, note="stand-in for TAQ / CRSP / Yahoo"))
    return out


GENERATE = [
    "python datasets/fetch/gen_statstream_randomwalk.py --m 2000 --T 20000",
    "python datasets/fetch/gen_braid_synthetic.py --family sines --m 2000 --T 32768",
    "python datasets/fetch/gen_braid_synthetic.py --family spiketrains --m 2000 --T 100000 --period 6500",
]


def emit(path: str, results_root: str) -> None:
    lines = ["#!/bin/bash", "# generated by abaca/campaign_competitors.py; run from the CorrTrack working tree on the Sophia frontend",
             "set -uo pipefail", f"RESULTS_ROOT=${{RESULTS_ROOT:-{results_root}}}", "mkdir -p abaca/logs",
             "submit() { oarsub \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }", "",
             "# synthetic inputs (cheap, frontend-side)"] + GENERATE + [""]
    for c in cells():
        common = [f"DATASET_CONFIG={c.config}", f"WINDOW_SIZE={c.W}", f"WINDOW_STEP={c.step}", f"N_LAGS={c.n_lags}", f"THR={c.T}",
                  "EXTRA_ARGS=" + c.extra_args().replace(" ", "+")]   # one token; the .oar scripts decode "+"
        tuned = f"$RESULTS_ROOT/tuned/{c.stem}"
        best = f"$RESULTS_ROOT/corrtrack_hyperopt/{c.stem}/best_params_corrtrack.json"
        tune_cmd = " ".join(["./abaca/tune_competitors.oar"] + common + [f"CALIB_OBS={c.calib_obs}", f"OUT_DIR={tuned}"])
        nway = ["./abaca/nway_compare.oar"] + common + [f"ARMS={c.arms}", f"COMPETITOR_PARAMS={tuned}", f"BEST_PARAMS={best}"]
        nway_cmd = " ".join(nway)
        neg_cmd = " ".join(nway + ["NEG_CORR_FLAG=--neg-corr"])
        res = f"-l host=1,walltime={c.walltime}"
        lines += [f"# --- {c.stem}: {c.note}",
                  f"T_JOB=$(submit {res} -S \"{tune_cmd}\")",
                  f"submit -a \"$T_JOB\" {res} -S \"{nway_cmd}\"",
                  f"submit -a \"$T_JOB\" {res} -S \"{neg_cmd}\"",
                  ""]
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(cells())} cells, {3 * len(cells())} jobs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emit", default=None)
    ap.add_argument("--results-root", default="$HOME/corrtrack_abaca_results")
    args = ap.parse_args()
    cs = cells()
    if args.list or not args.emit:
        print(f"{'cell':52s} {'m':>6s} {'n_obs':>6s} {'wall':>9s} note")
        for c in cs:
            print(f"{c.stem:52s} {str(c.n_series or 'cfg'):>6s} {str(c.n_obs or 'cfg'):>6s} {c.walltime:>9s} {c.note}")
        print(f"{len(cs)} cells x 3 jobs")
    if args.emit:
        emit(args.emit, args.results_root)


if __name__ == "__main__":
    main()
