"""Fixed-degree synthetic m-sweep: CorrTrack's search exponent when the output grows only linearly (2026-10-16, user).

Why: the headline m-sweep (campaign_synth_overnight.py, m = 125 to 2000) holds the tuple DENSITY at 0.01, so the
correlated output grows as m^2 and the fitted wall exponent (1.64) mixes linear sketching, the subquadratic LSH search
and the quadratic output (candidates tested ~ m^2.00 there, within 0.05% of the correlated count). Classical LSH
sizing, which CorrTrack follows (band_width = ceil(log2(m L / occupancy)), n_bands from the recall target), predicts a
per-step search cost of m (m L)^rho with rho = log2(1 / (1 - arccos(gamma) / pi)) at the gate gamma: rho 0.42 / 0.33 /
0.22 / 0.15 at gamma 0.7 / 0.8 / 0.9 / 0.95. This sweep fixes the number of correlated partners per series (degree)
instead of the density, so the output is linear in m and the wall exponent should approach 1 + rho, falling as the
threshold rises. See docs/presentation_questions_2026-10-15.md, question 3.

Design:
- degree 55 (group size 56), every series grouped: at m = 500 this matches the overnight baseline cell (density about
  0.01), so the two sweeps cross there; the density then falls as 1/m (about 0.036 at m=125, 0.0012 at m=4000).
- m in {125, 250, 500, 1000, 2000, 4000}; T in {0.7, 0.8, 0.9, 0.95}; ar1 base process, raw levels, L = 6, W = 60,
  step 6, n = 6000 (the overnight protocol). Positive correlation only.
- ONE CorrTrack tuning per threshold (both backends, at m = 500), reused at every m: the gate stays fixed along a curve
  (per-cell tuning could move gamma with m and confound the exponent); band width and n_bands still resize with m at
  run time (n_bands_tolerance > 0), so the recall target is kept.
- arms: bruteforce (truth and the m^2 reference), bf_incremental, filcorr (full band, untuned), corrtrack,
  corrtrack_hamming. No competitor tuning job.

    python abaca/campaign_synth_degree.py --emit abaca/synth_degree_submit.sh
    # writes the submit script and one generate script per threshold next to it; nothing is submitted
"""
from __future__ import annotations

import argparse
from pathlib import Path

import campaign_competitors as cc

W, STEP, N_OBS, L, PROC = 60, 6, 6000, 6, "ar1"
DEGREE = 55
MS = [125, 250, 500, 1000, 2000, 4000]
TS = [0.7, 0.8, 0.9, 0.95]
TUNE_M = 500
ARMS = "bruteforce,bf_incremental,filcorr,corrtrack,corrtrack_hamming"
# (m: nway walltime) from the overnight m-sweep: the five arms take 0.6 / 1.9 / 29 min at m = 500 / 1000 / 2000 x 1.5
# headroom, and m = 4000 is about four times m = 2000 (brute force ~5.3 h)
NWAY_WALLTIME = {125: "1:00:00", 250: "1:00:00", 500: "2:00:00", 1000: "3:00:00", 2000: "6:00:00", 4000: "16:00:00"}
TUNE_WALLTIME = "6:00:00"
GEN_WALLTIME = "8:00:00"      # verification brute force over a quarter of the stream, all six m in sequence


def name_of(m: int, T: float) -> str:
    return f"deg_{PROC}_m{m}_L{L}_T{str(T).replace('.', 'p')}_k{DEGREE}"


def reachable_density(m: int) -> float:
    """The density with every series in a group of DEGREE + 1, as the generator computes it: planted
    (pair, lag) tuples per step over tested tuples per step (C(m,2) synchronous + m^2 per lagged bucket).
    Asked 2% below so the solver's own rounding never trips its reachability check."""
    g = DEGREE + 1
    planted = (m // g) * g * (g - 1) / 2.0
    tested = m * (m - 1) / 2.0 + m * m * (L - 1)
    return round(0.98 * planted / tested, 6)


def gen_command(m: int, T: float) -> str:
    return (f"python datasets/fetch/gen_density_targeted.py --proc {PROC} --density {reachable_density(m)} --T {T} "
            f"--m {m} --n {N_OBS} --W {W} --step {STEP} --L {L} --corr-sign pos --degree {DEGREE} --name {name_of(m, T)}")


# (2026-10-18, user) sketch-width ablation: at T = 0.8, the tuned 64-bit parameters copied with n_vectors 128 and 256,
# everything else unchanged, on regenerated inputs (same seed, same files as the main run). Tests whether the
# shared 64-bit band pool explains CorrTrack's exponent above the textbook 1 + rho: compute_lsh_n_bands predicts
# n_bands growing as N^0.56 / 0.45 / 0.39 at 64 / 128 / 256 bits (textbook 0.33). Arms: bruteforce (recall truth)
# and corrtrack; the 64-bit cells are the main run's.
ABLATION_T = 0.8
ABLATION_NV = (128, 256)


def emit_ablation(out: Path, results_root: str) -> int:
    tt = str(ABLATION_T).replace(".", "p")
    gen = out.with_name(f"{out.stem}_generate_T{tt}.sh")
    order = [TUNE_M] + [m for m in MS if m != TUNE_M]
    gen.write_text("\n".join(["#!/bin/bash", "set -uo pipefail",
                              "source ${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-corrtrack}",
                              "cd ${SNAPSHOT:?set SNAPSHOT}", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"]
                             + [gen_command(m, ABLATION_T) for m in order]) + "\n")
    common_t = [f"WINDOW_SIZE={W}", f"WINDOW_STEP={STEP}", f"BASIC_WINDOW={STEP}", f"N_LAGS={(L - 1) * STEP}", f"THR={ABLATION_T}",
                "PREPROCESS=0", "SNAPSHOT=$SNAPSHOT"]
    lines = ["#!/bin/bash", "# generated by abaca/campaign_synth_degree.py --ablation; run on the Sophia frontend with SNAPSHOT set",
             "set -uo pipefail", ": \"${SNAPSHOT:?set SNAPSHOT to the campaign snapshot}\"",
             f"RESULTS_ROOT=\"${{RESULTS_ROOT:-{results_root}}}\"", "SCRIPTS=\"${SCRIPTS:-$(dirname \"$0\")}\"",
             "mkdir -p abaca/logs",
             "submit() { oarsub -q abaca -p \"cluster='mercantour3'\" \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }",
             f"BP=$(find \"$RESULTS_ROOT/hyperopt/T{tt}\" -name best_params_corrtrack.json | head -1)",
             "[ -n \"$BP\" ] || { echo \"no tuned parameters under $RESULTS_ROOT/hyperopt\" >&2; exit 3; }"]
    for nv in ABLATION_NV:
        lines.append(f"mkdir -p \"$RESULTS_ROOT/ablation/nv{nv}\" && python3 -c \"import json,sys; b=json.load(open(sys.argv[1])); "
                     f"b.update(n_vectors={nv}, grid_dimension={nv}); json.dump(b, open(sys.argv[2], 'w'), indent=2)\" "
                     f"\"$BP\" \"$RESULTS_ROOT/ablation/nv{nv}/best_params_corrtrack.json\"")
    lines.append(f"G_JOB=$(submit -n g_abl_T{tt} -l core=10,walltime={GEN_WALLTIME} \"SNAPSHOT=$SNAPSHOT bash $SCRIPTS/{gen.name}\")")
    n_jobs, ids = 1, []
    for nv in ABLATION_NV:
        for m in MS:
            cfg = f"DATASET_CONFIG=datasets/competitor/configs/experiment_dataset_{name_of(m, ABLATION_T)}.py"
            run = f"{name_of(m, ABLATION_T)}_m{m}_W{W}_s{STEP}_L{(L - 1) * STEP}_T{ABLATION_T}_pos_nv{nv}"
            lines.append(f"N_{nv}_{m}=$(submit -n n_abl_nv{nv}_m{m} -a \"$G_JOB\" -l host=1,walltime={NWAY_WALLTIME[m]} "
                         f"-S \"$SNAPSHOT/abaca/nway_compare.oar {cfg} {' '.join(common_t)} EXTRA_ARGS=--n-series+{m}+--n-obs+{N_OBS} "
                         f"ARMS=bruteforce,corrtrack BEST_PARAMS=$RESULTS_ROOT/ablation/nv{nv}/best_params_corrtrack.json "
                         f"RESULTS_ROOT=$RESULTS_ROOT RUN_NAME={run}\")")
            ids.append(f"nv{nv}_m{m}=$N_{nv}_{m}")
            n_jobs += 1
    lines.append(f"echo \"G=$G_JOB {' '.join(ids)}\"")
    out.write_text("\n".join(lines) + "\n")
    return n_jobs


# (2026-10-19, corrtrack_release_indepbands only) independent-band-bits test: the same 24 fixed-degree cells, run from
# a snapshot of THIS copy, with the pooled (corrtrack) and the independent (corrtrack_indep) banding in one job per cell
# against one brute-force truth. Both arms use the main run's LSH tuning for that threshold (TUNE_ROOT), so the only
# difference between them is how band bits are drawn. The snapshot is built first (prepare_snapshot.oar of this copy).
def emit_indep(out: Path, results_root: str, tune_root: str, arms: str = "bruteforce,corrtrack,corrtrack_indep",
               skip_truth: bool = False, tune_dir: str = "hyperopt") -> int:
    lines = ["#!/bin/bash", "# generated by abaca/campaign_synth_degree.py --indep (corrtrack_release_indepbands); run on the frontend",
             "set -uo pipefail", ": \"${SNAPSHOT:?set SNAPSHOT to the snapshot to build}\" \"${REPO_DIR:?set REPO_DIR to the staged copy}\"",
             f"RESULTS_ROOT=\"${{RESULTS_ROOT:-{results_root}}}\"", f"TUNE_ROOT=\"${{TUNE_ROOT:-{tune_root}}}\"",
             "SCRIPTS=\"${SCRIPTS:-$(dirname \"$0\")}\"", "mkdir -p \"$RESULTS_ROOT\" abaca/logs",
             "submit() { oarsub -q abaca -p \"cluster='mercantour3'\" \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }",
             "S_JOB=$(submit -n snap_indep -l host=1,walltime=1:00:00 -S \"$REPO_DIR/abaca/prepare_snapshot.oar REPO_DIR=$REPO_DIR SNAPSHOT=$SNAPSHOT\")",
             "echo \"snapshot job $S_JOB\""]
    n_jobs = 1
    for T in TS:
        tt = str(T).replace(".", "p")
        gen = out.with_name(f"{out.stem}_generate_T{tt}.sh")
        order = [TUNE_M] + [m for m in MS if m != TUNE_M]
        gen.write_text("\n".join(["#!/bin/bash", "set -uo pipefail",
                                  "source ${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-corrtrack}",
                                  "cd ${SNAPSHOT:?set SNAPSHOT}", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"]
                                 + [gen_command(m, T) for m in order]) + "\n")
        common_t = [f"WINDOW_SIZE={W}", f"WINDOW_STEP={STEP}", f"BASIC_WINDOW={STEP}", f"N_LAGS={(L - 1) * STEP}", f"THR={T}",
                    "PREPROCESS=0", "SNAPSHOT=$SNAPSHOT"]
        lines.append(f"G_JOB=$(submit -n g_ind_T{tt} -a \"$S_JOB\" -l core=10,walltime={GEN_WALLTIME} \"SNAPSHOT=$SNAPSHOT bash $SCRIPTS/{gen.name}\")")
        n_jobs += 1
        ids = []
        for m in MS:
            cfg = f"DATASET_CONFIG=datasets/competitor/configs/experiment_dataset_{name_of(m, T)}.py"
            run = f"{name_of(m, T)}_m{m}_W{W}_s{STEP}_L{(L - 1) * STEP}_T{T}_pos"
            # (2026-10-19) -S is read at submission time and the snapshot does not exist yet: point it at the staged
            # copy's wrapper, which enters the snapshot at run time (_snapshot_enter.sh, refuses without SNAPSHOT_OK)
            lines.append(f"N_{m}=$(submit -n n_ind_{name_of(m, T)} -a \"$G_JOB\" -l host=1,walltime={NWAY_WALLTIME[m]} "
                         f"-S \"$REPO_DIR/abaca/nway_compare.oar {cfg} {' '.join(common_t)} EXTRA_ARGS=--n-series+{m}+--n-obs+{N_OBS}"
                         f"{'+--skip-truth' if skip_truth else ''} ARMS={arms} HYPEROPT_DIR=$TUNE_ROOT/{tune_dir}/T{tt} "
                         f"RESULTS_ROOT=$RESULTS_ROOT RUN_NAME={run}\")")
            ids.append(f"m{m}=$N_{m}")
            n_jobs += 1
        lines.append(f"echo \"T={T} G=$G_JOB {' '.join(ids)}\"")
    out.write_text("\n".join(lines) + "\n")
    return n_jobs


# (2026-10-23, corrtrack_release_indepbands only) tuning for the independent mode: CorrTrack's proxy-anchor hyperopt at
# m = 500 per threshold with experiment_run_param_grid_campaign_indep.py (independent bits on, occupancy widened), then
# the 24 fixed-degree cells with the corrtrack_indep arm only, that tuning, no truth (recall from stored brute-force
# counts). Inputs come from another run's generate jobs: pass their ids as GEN_T0p7=... etc. in the environment.
def emit_tune_indep(out: Path, results_root: str) -> int:
    lines = ["#!/bin/bash", "# generated by abaca/campaign_synth_degree.py --tune-indep (corrtrack_release_indepbands); run on the frontend",
             "set -uo pipefail", ": \"${SNAPSHOT:?set SNAPSHOT to the snapshot to build}\" \"${REPO_DIR:?set REPO_DIR to the staged copy}\"",
             f"RESULTS_ROOT=\"${{RESULTS_ROOT:-{results_root}}}\"", "mkdir -p \"$RESULTS_ROOT\" abaca/logs",
             "submit() { oarsub -q abaca -p \"cluster='mercantour3'\" \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }",
             "S_JOB=$(submit -n snap_tune -l host=1,walltime=1:00:00 -S \"$REPO_DIR/abaca/prepare_snapshot.oar REPO_DIR=$REPO_DIR SNAPSHOT=$SNAPSHOT\")",
             "echo \"snapshot job $S_JOB\""]
    n_jobs = 1
    for T in TS:
        tt = str(T).replace(".", "p")
        common_t = [f"WINDOW_SIZE={W}", f"WINDOW_STEP={STEP}", f"BASIC_WINDOW={STEP}", f"N_LAGS={(L - 1) * STEP}", f"THR={T}",
                    "PREPROCESS=0", "SNAPSHOT=$SNAPSHOT"]
        tune_cfg = f"DATASET_CONFIG=datasets/competitor/configs/experiment_dataset_{name_of(TUNE_M, T)}.py"
        hyper = f"$RESULTS_ROOT/hyperopt_indep/T{tt}"
        lines.append(f"DEP=\"-a $S_JOB${{GEN_T{tt}:+ -a $GEN_T{tt}}}\"")
        lines.append(f"H_JOB=$(submit -n hi_deg_T{tt} $DEP -l core={cc.pack_cores(TUNE_M)},walltime={TUNE_WALLTIME} "
                     f"-S \"$REPO_DIR/abaca/hyperopt_corrtrack.oar {tune_cfg} {' '.join(common_t)} EXTRA_ARGS=--n-series+{TUNE_M}+--n-obs+{N_OBS} "
                     f"PARAM_GRID_CONFIG=experiment_run_param_grid_campaign.py OUT_DIR={hyper}\")")
        n_jobs += 1
        ids = []
        for m in MS:
            cfg = f"DATASET_CONFIG=datasets/competitor/configs/experiment_dataset_{name_of(m, T)}.py"
            run = f"{name_of(m, T)}_m{m}_W{W}_s{STEP}_L{(L - 1) * STEP}_T{T}_pos"
            lines.append(f"N_{m}=$(submit -n nt_{name_of(m, T)} -a \"$H_JOB\" -l host=1,walltime={NWAY_WALLTIME[m]} "
                         f"-S \"$REPO_DIR/abaca/nway_compare.oar {cfg} {' '.join(common_t)} EXTRA_ARGS=--n-series+{m}+--n-obs+{N_OBS}+--skip-truth "
                         f"ARMS=corrtrack_indep HYPEROPT_DIR={hyper} RESULTS_ROOT=$RESULTS_ROOT RUN_NAME={run}\")")
            ids.append(f"m{m}=$N_{m}")
            n_jobs += 1
        lines.append(f"echo \"T={T} H=$H_JOB {' '.join(ids)}\"")
    out.write_text("\n".join(lines) + "\n")
    return n_jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", required=True)
    ap.add_argument("--results-root", default="$HOME/corrtrack_abaca_results/synth_degree")
    ap.add_argument("--tune-indep", action="store_true", help="emit the independent-mode tuning and its 24-cell run")
    ap.add_argument("--indep", action="store_true", help="emit the independent-band-bits test (this copy's snapshot, pooled vs independent arms)")
    ap.add_argument("--tune-root", default="$HOME/corrtrack_abaca_results/synth_degree", help="--indep: where the main run's tunings are")
    ap.add_argument("--arms", default="bruteforce,corrtrack,corrtrack_indep", help="--indep: N-way arms")
    ap.add_argument("--skip-truth", action="store_true", help="--indep: no bruteforce arm (recall from stored counts)")
    ap.add_argument("--tune-dir", default="hyperopt", help="--indep: tuning folder under --tune-root (hyperopt, or hyperopt_indep)")
    ap.add_argument("--ablation", action="store_true", help="emit the sketch-width ablation (T = 0.8, n_vectors 128 and 256) instead")
    ap.add_argument("--skip-generate", action="store_true", help="the inputs already exist on the cluster: no generate jobs, tuning starts at once")
    args = ap.parse_args()
    out = Path(args.emit)
    if args.tune_indep:
        n = emit_tune_indep(out, args.results_root)
        print(f"wrote {out}: independent-mode tuning, {len(TS)} thresholds, {n} jobs")
        return
    if args.indep:
        n = emit_indep(out, args.results_root, args.tune_root, args.arms, args.skip_truth, args.tune_dir)
        print(f"wrote {out}: independent-band-bits test, {len(TS)} thresholds x {len(MS)} m, {n} jobs")
        return
    if args.ablation:
        n = emit_ablation(out, args.results_root)
        print(f"wrote {out}: sketch-width ablation at T = {ABLATION_T}, n_vectors {ABLATION_NV}, {n} jobs")
        return
    lines = ["#!/bin/bash", "# generated by abaca/campaign_synth_degree.py; run on the Sophia frontend with SNAPSHOT set",
             "set -uo pipefail", ": \"${SNAPSHOT:?set SNAPSHOT to the campaign snapshot}\"",
             f"RESULTS_ROOT=\"${{RESULTS_ROOT:-{args.results_root}}}\"", "SCRIPTS=\"${SCRIPTS:-$(dirname \"$0\")}\"",
             "mkdir -p \"$RESULTS_ROOT\" abaca/logs",
             "submit() { oarsub -q abaca -p \"cluster='mercantour3'\" \"$@\" | sed -n 's/^OAR_JOB_ID=//p'; }", ""]
    n_jobs = 0
    for T in TS:
        tt = str(T).replace(".", "p")
        gen = out.with_name(f"{out.stem}_generate_T{tt}.sh")
        order = [TUNE_M] + [m for m in MS if m != TUNE_M]         # the tuning cell first, the largest last
        gen.write_text("\n".join(["#!/bin/bash", "set -uo pipefail",
                                  "source ${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-corrtrack}",
                                  "cd ${SNAPSHOT:?set SNAPSHOT}", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"]
                                 + [gen_command(m, T) for m in order]) + "\n")
        common_t = [f"WINDOW_SIZE={W}", f"WINDOW_STEP={STEP}", f"BASIC_WINDOW={STEP}", f"N_LAGS={(L - 1) * STEP}", f"THR={T}",
                    "PREPROCESS=0", "SNAPSHOT=$SNAPSHOT"]
        tune_cfg = f"DATASET_CONFIG=datasets/competitor/configs/experiment_dataset_{name_of(TUNE_M, T)}.py"
        hyper = f"$RESULTS_ROOT/hyperopt/T{tt}"
        hyper_h = f"$RESULTS_ROOT/hyperopt_hamming/T{tt}"
        # (2026-10-17) EXTRA_ARGS is required by hyperopt_corrtrack.oar (set -u) and carries --n-series/--n-obs, as
        # campaign_competitors.Cell.extra_args() passes it; the first submission omitted it and every tuning job failed
        tune_extra = f"EXTRA_ARGS=--n-series+{TUNE_M}+--n-obs+{N_OBS}"
        dep = "" if args.skip_generate else "-a \"$G_JOB\" "
        lines.append(f"# --- T = {T}: generate all m, tune both backends once at m = {TUNE_M}, then one N-way per m")
        if not args.skip_generate:
            lines.append(f"G_JOB=$(submit -n g_deg_T{tt} -l core=10,walltime={GEN_WALLTIME} \"SNAPSHOT=$SNAPSHOT bash $SCRIPTS/{gen.name}\")")
            n_jobs += 1
        lines += [f"H_JOB=$(submit -n h_deg_T{tt} {dep}-l core={cc.pack_cores(TUNE_M)},walltime={TUNE_WALLTIME} "
                  f"-S \"$SNAPSHOT/abaca/hyperopt_corrtrack.oar {tune_cfg} {' '.join(common_t)} {tune_extra} OUT_DIR={hyper}\")",
                  f"H2_JOB=$(submit -n hh_deg_T{tt} {dep}-l core={cc.pack_cores(TUNE_M)},walltime={TUNE_WALLTIME} "
                  f"-S \"$SNAPSHOT/abaca/hyperopt_corrtrack.oar {tune_cfg} {' '.join(common_t)} {tune_extra} "
                  f"PARAM_GRID_CONFIG=experiment_run_param_grid_campaign_hamming.py OUT_DIR={hyper_h}\")"]
        n_jobs += 2
        ids = []
        for m in MS:
            cfg = f"DATASET_CONFIG=datasets/competitor/configs/experiment_dataset_{name_of(m, T)}.py"
            run = f"{name_of(m, T)}_m{m}_W{W}_s{STEP}_L{(L - 1) * STEP}_T{T}_pos"
            lines.append(f"N_{m}=$(submit -n n_{name_of(m, T)} -a \"$H_JOB\" -a \"$H2_JOB\" -l host=1,walltime={NWAY_WALLTIME[m]} "
                         f"-S \"$SNAPSHOT/abaca/nway_compare.oar {cfg} {' '.join(common_t)} EXTRA_ARGS=--n-series+{m}+--n-obs+{N_OBS} ARMS={ARMS} "
                         f"HYPEROPT_DIR={hyper} HYPEROPT_HAMMING_DIR={hyper_h} RESULTS_ROOT=$RESULTS_ROOT RUN_NAME={run}\")")
            ids.append(f"m{m}=$N_{m}")
            n_jobs += 1
        lines += [f"echo \"T={T} {'' if args.skip_generate else 'G=$G_JOB '}H=$H_JOB H2=$H2_JOB {' '.join(ids)}\"", ""]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(TS)} thresholds x {len(MS)} m = {len(TS) * len(MS)} cells, {n_jobs} jobs; generators next to it")
    for m in MS:
        print(f"  m={m}: requested density {reachable_density(m)}")


if __name__ == "__main__":
    main()
