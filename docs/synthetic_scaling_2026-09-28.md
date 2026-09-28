# Synthetic scaling study (2026-09-28)

Supersedes `docs/synthetic_scaling_2026-09-26.md`, which covered raw levels only and labelled the
base process as if it were the space. The two are different axes: `ar1` and `rw` are base
processes, and every one of them is run twice, once on raw levels and once on first differences. A
random walk is nonstationary in levels and stationary after differencing, where it becomes white
noise, which is the uncooperative regime for any filter built on energy concentration.

Generated tables: `docs/synthetic_tables_2026-09-28.md` (builder `abaca/build_synth_tables.py`).
Figures: `docs/figures/scaling_ar1.png`, `docs/figures/scaling_rw.png`,
`docs/figures/threshold_bars_ar1.png`, `docs/figures/threshold_bars_rw.png`
(builder `abaca/build_synth_figures.py`). Both read the runs through `abaca/synth_results.py`.

## Design

One factor at a time around m=500, L=6, T=0.90, target density 0.01, for each of the two base
processes and each of the two spaces. W=60, step=6, 6000 observations, 991 evaluated windows,
positive correlation only, so L=1 is the synchronous case. Every cell ran the twelve arms and its
own brute force in a single job with monitoring off, on the whole stream, with CorrTrack tuned by
its own hyperopt on both backends and the four pruning competitors tuned by the CSZ protocol. Each
speedup is therefore a wall-clock ratio measured inside one job on one node.

60 of the 64 cells are in. The four m=2000 cells are running; three dense cells (target density
0.1) are being rerun because their CSZ tuning pass fell back to defaults, which is also why
StatStream reports recall 0.000 in exactly those three cells and nowhere else. Those four numbers
are not a result about StatStream; they are a tuning failure, and they are excluded from the claims
below.

## 1. Does the generator deliver the density it is asked for?

Yes, wherever the correlation is the generator's own. Effective density is brute force's own count,
correlated pair-windows over tested pair-windows, in the space the cell runs in.

| process | space | median effective/target | min | max | cells |
|---|---|---|---|---|---|
| ar1 | raw | 1.00 | 0.91 | 1.00 | 18 |
| ar1 | differenced | 1.00 | 0.91 | 1.00 | 18 |
| rw | raw | 1.22 | 0.90 | 4.83 | 18 |
| rw | differenced | 1.00 | 0.91 | 1.00 | 18 |

Three readings, all of them expected:

- The planted density is exact for the stationary process in both spaces, and for the random walk
  once it is differenced. The 0.91 floor appears only at target 0.1, where the correlated groups
  are large enough to saturate.
- A raw random walk overshoots, by 1.26x at T=0.90 and 4.83x at T=0.70. That is spurious
  correlation, the classic regression artefact of levels, and it is why the generator runs with
  `--allow-spurious` there: the cell is not a broken request but the nonstationary case.
- **Report effective density, not the requested target.** The target is an input to the generator;
  the density that governs the cost of every method is the one brute force measured, which is what
  the tables and figures use on their axes.

## 2. What the curves say

CorrTrack (both backends) leads every cell except the raw random walk at low thresholds, and the
lead grows with every axis that makes the problem harder.

- **Series count.** Stationary, raw: 8.30x at m=125, 13.66x at 250, 19.38x at 500, 24.57x at 1000,
  against 11.43x for the best competitor at m=1000 (ParCorr). Differenced, the gap widens, because
  the competitors lose ground where CorrTrack does not: 24.41x against 5.85x at m=1000.
- **Lag depth.** Raw, stationary: 5.58x at L=1 rising to 26.51x at L=11. The synchronous case is
  the one where CorrTrack's margin is smallest, which is the expected shape: the lag grid is what
  multiplies the work a filter can avoid.
- **Threshold.** Monotone in both spaces: 8.95x at T=0.70 to 25.18x at 0.95 (stationary, raw), and
  9.27x to 25.80x on the differenced random walk. The one place CorrTrack does not lead is the raw
  random walk at T=0.70 and 0.80, where FilCorr is ahead (6.00x against 4.27x, 6.28x against 6.91x
  at T=0.80 CorrTrack is already back in front), and that is the same crossover the real-data
  tables show around T=0.85.
- **Density.** Falling, steeply and as predicted: 27.4x at 0.002, 19.4x at 0.01, 15.1x at 0.02,
  8.7x at 0.05, 5.6x at 0.1 (stationary, raw), with the same shape in every other combination. At
  10% density the filter has little left to prune and every method converges toward brute force.

Recall holds throughout: CorrTrack's median recall is 0.998 (lsh) and 1.000 (hamming) with minimum
0.950 and 0.962, at precision 1.000 everywhere, since validation is exact. ThinBRAID is the only
arm that loses precision (median 0.967, minimum 0.733), which matches the real-data tables.

## 3. Reading the figures

`scaling_<proc>.png` is four rows, one per axis, and two columns, raw and differenced, with speedup
on a log scale and the brute-force line at 1.0 drawn in.

`threshold_bars_<proc>.png` is the wall-clock view at m=500, which is the largest size the
threshold axis covers in this design (m=1000 and m=2000 exist at T=0.90 only, because the design is
one factor at a time). Bars are hatched where the capability the cell exercises is not the
authors' own: `//` where the lagged search is an extension of ours (TSUBASA, ParCorr, CSZ,
CorrJoin) and `..` where the paper specifies it but never evaluated it (StatStream). The tier is
read from each run's `supports_lags` field, not from a list written by hand.
