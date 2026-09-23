"""The m=500 tables under the campaign protocol (2026-09-21, user).

The m=500 tables (docs/tables_m500_and_filcorr_sweep_2026-09-19.md) picked CorrTrack's and the
competitors' settings on the cell itself. This emits the same cells through the campaign chain
of abaca/campaign_competitors.py instead: CorrTrack tuned by its own hyperopt, the competitors by
the CSZ protocol (abaca/tune_competitors.py), every arm compared on the whole cut with monitoring
off, recall / precision / candidate specificity per arm, and the hyperopt / tuning wall times in
each job's time_v.txt. Cells: the six m=500 cuts (sp500_sub263 skipped) at the tables' W / step /
n_lags, T in {0.7, 0.8, 0.9, 0.95}, raw and differenced, neg_corr=True only (the tables' setting),
organised (2026-09-21, user) as three capability tables, each tuned at its own (n_lags, neg_corr):
  S  synchronous, positive  (n_lags=0, neg_corr=False): every arm available
  L  lagged, positive       (tables' n_lags, neg_corr=False): CorrTrack, FilCorr, StatStream, BRAID/ThinBRAID, exact arms
  N  lagged, negative       (tables' n_lags, neg_corr=True): the same arms, negative correlation included
i.e. 3 x 48 = 144 runs x 4 jobs (lsh hyperopt, hamming hyperopt, CSZ tuning, N-way). Every arm runs on the whole cut (EVAL_SPAN=full); tuning uses the first TRAIN_RATIO.

    python abaca/campaign_m500_tables.py --emit abaca/m500_tables_submit.sh
    SNAPSHOT=... python abaca/campaign_feeder.py abaca/m500_tables_submit.sh   # or bash it directly
"""
from __future__ import annotations

import argparse
import re

import campaign_competitors as cc

# (dataset config label, W, step, n_lags): the m=500 tables' protocol (hamming_exact_compare_m500.py CFG)
M500_SETS = [
    ("streamflow_m500", 30, 3, 15, 500, 2768),
    ("sp500_m500", 30, 3, 15, 444, 2768),
    ("wikipedia_m500", 30, 3, 15, 500, 2768),
    ("smartmeter_m500", 48, 8, 16, 500, 27649),
    ("global_weather_m500", 30, 3, 15, 500, 2768),
    ("acwi_capweighted_m500", 30, 3, 15, 500, 2768),
]
CORRJOIN_KNOBS_BY_W = {30: {"corrjoin_ks": 5, "corrjoin_ke": 10}, 48: {"corrjoin_ks": 6, "corrjoin_ke": 12}}   # as in campaign_competitors DATASETS


def m500_cells(tables=("S", "L", "N")) -> list[cc.Cell]:
    """One Cell per (set, T, space, lag setting); emit() writes a pos and a neg run for each, and
    main() keeps: pos for the L0 cells (table S), pos and neg for the lagged cells (tables L and N)."""
    out = []
    for label, W, step, n_lags, m, T_len in M500_SETS:
        for T in cc.THRESHOLDS:
            for diff in (False, True):
                for lags in sorted({0 if "S" in tables else None, n_lags if ("L" in tables or "N" in tables) else None} - {None}):
                    out.append(cc.Cell(label, W, step, lags, T, n_series=m, n_obs=T_len, arms="all",
                                       extra=dict(CORRJOIN_KNOBS_BY_W[W]), walltime="24:00:00", preprocess=diff,
                                       note=f"m=500 tables | {'differenced' if diff else 'raw'} | {'table S (sync)' if lags == 0 else 'tables L (pos) / N (neg)'}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", required=True)
    ap.add_argument("--results-root", default="$HOME/corrtrack_abaca_results")
    ap.add_argument("--tables", default="S,L,N", help="which capability tables to emit (S: sync pos, L: lagged pos, N: lagged neg)")
    ap.add_argument("--phase2", action="store_true", help="only the hamming hyperopt and the N-way job per run (the lsh hyperopt and the CSZ tuning already exist)")
    args = ap.parse_args()
    tables = tuple(t.strip().upper() for t in args.tables.split(",") if t.strip())
    cc.cells = lambda *a, **k: m500_cells(tables)    # reuse emit() unchanged on this cell list
    cc.emit(args.emit, args.results_root)
    lines = open(args.emit).read().split("\n")
    kept, dropped = [], 0
    for ln in lines:
        mrun = re.search(r"^(?:H_JOB|T_JOB|N_JOB)=\$\(submit -n [htn]_(\S+)_(pos|neg) ", ln) or re.search(r'^echo "(\S+)_(pos|neg) H=', ln)
        if mrun:
            stem, tag = mrun.group(1), mrun.group(2)
            sync = "_L0_" in stem
            want = (sync and tag == "pos" and "S" in tables) or (not sync and tag == "pos" and "L" in tables) or (not sync and tag == "neg" and "N" in tables)
            if not want:
                dropped += 1
                continue
        # every arm on the whole cut (tuning stays on the calibration span); CorrTrack's second backend
        # (lsh_hamming_exact) gets its own hyperopt job on the hamming grid, and the N-way job waits for it
        if ln.startswith("H_JOB=$(submit"):
            h2 = ln.replace("H_JOB=$(submit -n h_", "H2_JOB=$(submit -n hh_")
            h2 = re.sub(r' OUT_DIR=\$RESULTS_ROOT/hyperopt/', r' PARAM_GRID_CONFIG=experiment_run_param_grid_campaign_hamming.py OUT_DIR=$RESULTS_ROOT/hyperopt_hamming/', h2)
            assert "hyperopt_hamming" in h2, ln
            if args.phase2:
                kept.append(h2); continue
            kept.append(ln); kept.append(h2); continue
        if ln.startswith("T_JOB=$(submit") and args.phase2:
            continue
        if ln.startswith("N_JOB=$(submit"):
            # (2026-09-22) NWAY_LARGE_SET_GB=200: every correlated set travels child -> parent in memory. With the 1 GB
            # default, arms above it (ThinBRAID in 44 cells, every arm in the 3 densest) were written as .npy to NFS inside
            # the timed region, an I/O tax the other arms of the same cell did not pay, and the scratch of concurrent
            # dense cells hit the 100 GB home hard quota. The nodes have 192 GB; the largest set seen is 10 GB.
            ln = ln.replace(' RUN_NAME=', ' NWAY_LARGE_SET_GB=200 EVAL_SPAN=full RUN_NAME=')
            ln = re.sub(r' HYPEROPT_DIR=(\S+) ', r' HYPEROPT_DIR=\1 HYPEROPT_HAMMING_DIR=\1'.replace("hyperopt/", "hyperopt/") + " ", ln)
            ln = re.sub(r'HYPEROPT_HAMMING_DIR=\$RESULTS_ROOT/hyperopt/', 'HYPEROPT_HAMMING_DIR=$RESULTS_ROOT/hyperopt_hamming/', ln)
            assert "HYPEROPT_HAMMING_DIR=$RESULTS_ROOT/hyperopt_hamming/" in ln, ln
            if args.phase2:
                ln = ln.replace(' -a "$H_JOB" -a "$T_JOB" ', ' -a "$H2_JOB" ')
            else:
                ln = ln.replace(' -a "$H_JOB" -a "$T_JOB" ', ' -a "$H_JOB" -a "$T_JOB" -a "$H2_JOB" ')
            assert '"$H2_JOB"' in ln, ln
        if ln.startswith('echo "') and " H=$H_JOB " in ln:
            ln = ln.replace(" H=$H_JOB T=$T_JOB N=$N_JOB", " H2=$H2_JOB N=$N_JOB" if args.phase2 else " H=$H_JOB T=$T_JOB H2=$H2_JOB N=$N_JOB")
        kept.append(ln)
    open(args.emit, "w").write("\n".join(kept))
    n_runs = sum(1 for ln in kept if ln.startswith("N_JOB="))
    per = 2 if args.phase2 else 4
    print(f"tables {','.join(tables)}: {n_runs} runs x {per} jobs = {per * n_runs} jobs (dropped {dropped} lines of unwanted runs)")


if __name__ == "__main__":
    main()
