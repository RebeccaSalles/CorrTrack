"""Overnight synthetic campaign: the m=500 table protocol on generated data (2026-09-22, user).

Same chain and metrics as abaca/campaign_m500_tables.py (CorrTrack tuned by its own hyperopt on
two backends, competitors by the CSZ protocol, monitoring off, every arm evaluated on the whole
stream), but the data comes from this project's density-targeted generator instead of the six
real sets, so the dataset parameters become axes: base process (stationary ar1 / nonstationary
random walk), number of series m, lagged windows L, correlation threshold T and effective
correlation density.

Design: one factor at a time around a baseline, which is what the runtime-scaling curves need
(every point on a curve differs from the baseline in one axis only) and what keeps the campaign
inside one night. Per process: 4 + 4 + 4 + 5 levels minus the 3 repeated baselines = 14 cells,
28 cells over the two processes, 4 jobs each.

The generator plants only positive correlation here (--corr-sign pos) and every cell runs in raw
space with neg_corr=False, so L = 1 gives the synchronous comparison and L > 1 the lagged one;
negative correlation is not an axis of this campaign (it is one in the real-data tables).

    python abaca/campaign_synth_overnight.py --emit abaca/synth_overnight_submit.sh
    # then, on a node: bash abaca/synth_overnight_submit_generate.sh
    # then: SNAPSHOT=... python abaca/campaign_feeder.py abaca/synth_overnight_submit.sh
"""
from __future__ import annotations

import argparse
import re

import campaign_competitors as cc

W, STEP, N_OBS = 60, 6, 6000                 # 991 evaluated windows per cell
# (2026-09-22, user) the baseline matches the real-data tables so the two are readable together:
# m = 500 is their standardised cut, L = 6 is their modal lag setting (5 of the 6 datasets; smartmeter
# is the exception at L = 3), and the density axis keeps 0.02 while the baseline sits at 0.01.
BASE = {"m": 500, "L": 6, "T": 0.9, "density": 0.01}
AXES = {
    "m": [125, 250, 500, 1000],
    "L": [1, 3, 6, 11],
    "T": [0.7, 0.8, 0.9, 0.95],
    "density": [0.002, 0.01, 0.02, 0.05, 0.1],
}
PROCS = ["ar1", "rw"]                        # stationary / nonstationary
CORRJOIN_KNOBS = {"corrjoin_ks": 6, "corrjoin_ke": 12}    # 6 and 12 divide W = 60
# a dense or large cell costs the most; the cap keeps one runaway cell from eating the night
WALLTIME = {"pack": "3:00:00", "nway": "3:00:00"}


def points():
    """(proc, m, L, T, density) one factor at a time around BASE, the baseline kept once."""
    out = []
    for proc in PROCS:
        seen = set()
        for axis, levels in AXES.items():
            for v in levels:
                p = dict(BASE, **{axis: v})
                key = (proc, p["m"], p["L"], p["T"], p["density"])
                if key in seen:
                    continue
                seen.add(key); out.append(key)
    return out


def name_of(proc, m, L, T, density):
    return f"ovn_{proc}_m{m}_L{L}_T{str(T).replace('.', 'p')}_d{str(density).replace('.', 'p')}"


def gen_command(proc, m, L, T, density):
    # --allow-spurious: a raw random walk produces correlated tuples the generator did not plant;
    # the file is kept and the spurious fraction recorded, which is the point of the nonstationary arm
    return (f"python datasets/fetch/gen_density_targeted.py --proc {proc} --density {density} --T {T} "
            f"--m {m} --n {N_OBS} --W {W} --step {STEP} --L {L} --corr-sign pos --allow-spurious "
            f"--name {name_of(proc, m, L, T, density)}")


def synth_cells() -> list[cc.Cell]:
    out = []
    for (proc, m, L, T, density) in points():
        name = name_of(proc, m, L, T, density)
        out.append(cc.Cell(name, W, STEP, (L - 1) * STEP, T, n_series=m, n_obs=N_OBS, arms="all",
                           extra=dict(CORRJOIN_KNOBS), walltime=WALLTIME["nway"], calib_obs=N_OBS,
                           preprocess=False, config_path=f"datasets/competitor/configs/experiment_dataset_{name}.py",
                           note=f"{proc} ({'stationary' if proc == 'ar1' else 'nonstationary'}), m={m}, L={L}, T={T}, target density={density}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", required=True)
    ap.add_argument("--results-root", default="$HOME/corrtrack_abaca_results/synth_overnight")
    args = ap.parse_args()
    cells = synth_cells()
    cc.cells = lambda *a, **k: cells
    cc.emit(args.emit, args.results_root)

    lines = open(args.emit).read().split("\n")
    kept, dropped = [], 0
    for ln in lines:
        m = re.search(r"^(?:H_JOB|H2_JOB|T_JOB|N_JOB)=\$\(submit -n (?:hh|[htn])_(\S+)_(pos|neg) ", ln) or re.search(r'^echo "(\S+)_(pos|neg) H=', ln)
        if m and m.group(2) == "neg":          # positive correlation only in this campaign
            dropped += 1
            continue
        # campaign_competitors.emit already writes the hamming hyperopt job (hh_*), its HYPEROPT_HAMMING_DIR
        # and the N-way dependency on it (2026-09-21), so this driver only adds what is specific to it
        if ln.startswith("N_JOB=$(submit"):
            ln = ln.replace(" RUN_NAME=", " NWAY_LARGE_SET_GB=200 EVAL_SPAN=full RUN_NAME=")
            assert '"$H2_JOB"' in ln and "HYPEROPT_HAMMING_DIR" in ln, ln
        ln = ln.replace(f'walltime={cc.Cell("x", 1, 1, 0, 0.7).walltime}', f'walltime={WALLTIME["pack"]}')
        kept.append(ln)
    open(args.emit, "w").write("\n".join(kept))

    gen = ["#!/bin/bash",
           "# datasets of the overnight synthetic campaign; run on a node, not the frontend:",
           "#   oarsub -q abaca -p \"cluster='mercantour3'\" -l host=1,walltime=3:00:00 -S \"SNAPSHOT=... bash abaca/synth_overnight_submit_generate.sh\"",
           "#OAR -n synth_generate", "#OAR -q abaca",
           "#OAR -O /home/rpontess/corrtrack_abaca_results/synth_overnight/generate.%jobid%.out",
           "#OAR -E /home/rpontess/corrtrack_abaca_results/synth_overnight/generate.%jobid%.err",
           "set -uo pipefail",
           'for kv in "$@"; do export "${kv?}"; done',   # OAR passes parameters as KEY=VALUE, like the .oar wrappers
           "source ${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-corrtrack}",
           "cd ${SNAPSHOT:?set SNAPSHOT}", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
           "ok=0; fail=0"]
    for p in points():
        gen.append(f"{gen_command(*p)} && ok=$((ok+1)) || {{ fail=$((fail+1)); echo \"GEN FAILED: {name_of(*p)}\"; }}")
    gen += ['echo "generated $ok, failed $fail"', "echo SYNTH_GENERATE_DONE"]
    gpath = args.emit.replace(".sh", "_generate.sh")
    open(gpath, "w").write("\n".join(gen) + "\n")
    n_runs = sum(1 for ln in kept if ln.startswith("N_JOB="))
    print(f"{len(cells)} cells, {n_runs} runs x 4 jobs = {4 * n_runs} jobs (dropped {dropped} neg lines); datasets in {gpath}")


if __name__ == "__main__":
    main()
