# Synthetic scaling study (2026-09-28)

Supersedes `docs/synthetic_scaling_2026-09-26.md`, which covered raw levels only and labelled the
base process as if it were the space. The two are different axes: `ar1` and `rw` are base
processes, and every one of them is run twice, once on raw levels and once on first differences. A
random walk is nonstationary in levels and stationary after differencing, where it becomes white
noise, which is the uncooperative regime for any filter built on energy concentration.

Generated tables: `docs/synthetic_tables_2026-09-28.md` (builder `abaca/build_synth_tables.py`).
Figures: `docs/figures/synth_scaling.png` and `docs/figures/synth_threshold_bars.png` (builder
`abaca/build_synth_figures.py`). Both read the runs through `abaca/synth_results.py` and take their
colours and legend order from `abaca/method_style.py`, which is shared with the campaign figures.

## How to read the numbers

**Naming.** `BF_vect` is the vectorised exact brute force, each cell's own ground truth and timing
anchor. `BF_incr` is the exact incremental baseline. CorrTrack appears as its two tuned backends,
`CorrTrack-LSH` and `CorrTrack-Ham`. The competitors keep their published names.

**Aggregation.** Nothing in this study is averaged over datasets, because the design generates one
dataset per level: every speedup, recall and wall clock quoted below is that single cell's own
measurement against its own `BF_vect`, in the same job. The only medians here are in the generator
validation table, which says so in its heading. That is the opposite of the real-data campaign,
where each reported number is a median over the six datasets.

**Ordering.** In the bar charts the bars are sorted fastest first inside each threshold group, so
the ranking reads left to right; the methods are named in the legend. The legend itself is in the
same order in every figure of the project: the methods that lead the most cells of the real
campaign first, as the ranking tables count leadership.

**(2026-09-28, user) The figures show raw levels only.** Differencing makes both processes
stationary, and the differenced cells then track the raw AR(1) cells closely: CorrTrack-LSH over the
m axis reads 8.30 / 13.66 / 19.38 / 24.57 on raw AR(1), 8.02 / 13.60 / 20.47 / 24.41 on differenced
AR(1) and 7.97 / 14.05 / 20.64 / 24.40 on the differenced random walk. So the contrast worth
plotting is stationary against nonstationary, which is the raw pair; the differenced cells stay in
the tables, where the density validation still needs them.

## Design

One factor at a time around m=500, L=6, T=0.90, target density 0.01, for each of the two base
processes and each of the two spaces. W=60, step=6, 6000 observations, 991 evaluated windows,
positive correlation only, so L=1 is the synchronous case. Every cell ran the twelve arms and its
own brute force in a single job with monitoring off, on the whole stream, with CorrTrack tuned by
its own hyperopt on both backends and the four pruning competitors tuned by the CSZ protocol. Each
speedup is therefore a wall-clock ratio measured inside one job on one node.

All 64 cells are in and all 64 validate. The dense cells whose CSZ tuning pass had fallen back to
defaults were rerun, and StatStream now reports recall 1.000 at target density 0.1 instead of 0.000,
at 0.84x to 0.88x, so it is slower than brute force in exactly the regime where CorrTrack still
returns 5.6x to 5.8x. The m=2000 cells landed on 2026-09-28 evening: brute force alone costs 4,800 s
there, and CorrTrack-LSH returns 25.98x on the stationary process and 26.48x on the random walk, at
recall 1.000 and 0.997.

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

- **Series count.** Stationary, raw: 8.30x at m=125, 13.66x at 250, 19.38x at 500, 24.57x at 1000
  and 25.98x at 2000, against a best competitor that peaks at m=500 (ParCorr 16.08x) and then falls
  away to 11.43x at m=1000 and 9.82x at m=2000. The nonstationary process shows the same shape more
  sharply: 6.96x to 26.48x for CorrTrack across the same range, against 8.82x to 5.80x for the best
  competitor. That turn is the point of the axis: CorrTrack's advantage keeps growing with the
  number of series while the competitors' peaks and declines, so the gap at m=2000 is 2.6x to 4.6x
  rather than the 1.2x it was at m=500.
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
arm that loses precision (median 0.967, minimum 0.733), which matches the real-data tables, and no
arm is below recall 0.9 anywhere in the study.

## 3. Why the threshold still matters at a fixed density

A fair question the threshold axis answers, because it is the only axis where the generator holds
the density constant (effective density 0.00997 to 0.00998 at every level): if the speedup were a
function of density alone, that row would be flat. It is not. It runs 8.95x at T=0.70 to 25.18x at
T=0.95 on the stationary process in raw levels, a factor of 2.8 with the density unchanged.

The phase breakdown of those five cells says exactly where it comes from (CorrTrack-LSH, m=500,
L=6, target density 0.01, AR(1) raw):

| T | effective density | candidates tested | candidate precision | sketch s | candidate search s | validation s | total s | speedup |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 0.00997 | 13,575,096 | 0.995 | 2.5 | 27.0 | 2.9 | 33.3 | 8.95x |
| 0.80 | 0.00997 | 13,511,187 | 1.000 | 2.4 | 16.5 | 2.8 | 22.6 | 12.97x |
| 0.85 | 0.00997 | 13,545,798 | 0.999 | 2.4 | 12.1 | 2.8 | 18.2 | 16.12x |
| 0.90 | 0.00997 | 13,525,575 | 1.000 | 2.3 | 9.2 | 2.8 | 15.2 | 19.38x |
| 0.95 | 0.00998 | 13,536,195 | 1.000 | 2.2 | 5.9 | 2.7 | 11.7 | 25.18x |

Sketching costs the same everywhere, and so does validation, because the candidate set is the same
size in every cell and almost every candidate is a true correlation (candidate precision 0.995 to
1.000). The entire difference is the candidate search: 27.0 s at T=0.70 against 5.9 s at T=0.95, a
factor of 4.6.

So density and threshold govern two different costs. **Density fixes the validation work**, the
pairs that must be confirmed exactly and cannot be avoided by any filter. **The threshold fixes how
cheaply the index can find them**: the LSH bands are sized from the cosine threshold, so a high
threshold means a pair must agree over a long run of sign bits before it is even looked at, and the
scan touches few entries; a low threshold widens the neighbourhood the index must sweep, and the
cost of rejecting all those near misses lands in the candidate phase, before the dot-gamma gate
culls them back to the same final candidate set.

On real data the two move together, since raising the threshold also empties the result set, which
is why the real curves are steeper than this one. This axis is the clean measurement of the second
effect on its own.

## 4. Reading the figures

`synth_scaling.png` is four rows, one per axis, and two columns, AR(1) against random walk, both on
raw levels, with speedup on a log scale and the BF_vect line at 1.0 drawn in. The density row is
plotted at the density brute force measured, not the one the generator was asked for, which is why
the random-walk column reaches further right.

`synth_threshold_bars.png` is the wall-clock view at m=500, which is the largest size the threshold
axis covers in this design (m=1000 and m=2000 exist at T=0.90 only, because the design is one factor
at a time). Bars are sorted fastest first inside each threshold group and hatched where the capability the cell
exercises is not the authors' own: `//` where the lagged search is an extension of ours (TSUBASA, ParCorr, CSZ,
CorrJoin) and `..` where the paper specifies it but never evaluated it (StatStream). The tier is
read from each run's `supports_lags` field, not from a list written by hand.
