# Per pair-window cost of the exact tiers (2026-09-28)

**Authors-only variant.** Every result that exists only because we extended a method has been removed: an arm is dropped from a cell when the capability that cell exercises is tagged `enabled_by_us` in its own run. That takes TSUBASA, ParCorr, CSZ and CorrJoin out of the lagged classes, FilCorr out of the negative class, and leaves the synchronous class untouched, so the cell counts differ per arm. ThinBRAID is removed outright: its published algorithm does not reproduce here, at median recall 0.815 and precision 0.046, which is a gap between that paper and its implementation rather than a result about it. Thresholds below 0.8 are left out. Nothing else changes: same runs, same measurements, same code.

Every speedup in this project is measured against `BF_vect`, the vectorised exact brute force that reruns inside each cell. The competitor papers generally measure against a naive implementation instead, so this table prices the tiers against each other on one node and one dataset, to show what our choice of baseline costs us. No experiment is run on the naive tiers; this is a cost probe (`abaca/naive_baseline.py`, one job) plus the campaign's own measured cost.

## Measured per pair-window, one node, generated data (W=60, step=6, T=0.90)

### synchronous (n_lags=0)

| m | pair-windows | naive Python | naive numpy | BF_vect | BF_incr | naive Python / BF_vect | naive numpy / BF_vect | BF_vect / BF_incr |
|---|---|---|---|---|---|---|---|---|
| 25 | 18,300 | 55,753 ns | 287 ns | 1,689 ns | 3,648 ns | 33x | 0.17x | 0.46x |
| 50 | 74,725 | 55,992 ns | 86 ns | 333 ns | 885 ns | 168x | 0.26x | 0.38x |
| 100 | 301,950 | 55,561 ns | 39 ns | 239 ns | 280 ns | 233x | 0.16x | 0.85x |
| 200 | 1,213,900 | not run | 25 ns | 203 ns | 128 ns |  | 0.12x | 1.59x |

### lagged (n_lags=30, 6 lag probes)

| m | pair-windows | naive Python | naive numpy | BF_vect | BF_incr | naive Python / BF_vect | naive numpy / BF_vect | BF_vect / BF_incr |
|---|---|---|---|---|---|---|---|---|
| 25 | 191,800 | 54,236 ns | 119 ns | 262 ns | 782 ns | 207x | 0.45x | 0.34x |
| 50 | 768,600 | 54,499 ns | 38 ns | 220 ns | 229 ns | 248x | 0.17x | 0.96x |
| 100 | 3,077,200 | 54,086 ns | 16 ns | 192 ns | 87 ns | 281x | 0.08x | 2.20x |
| 200 | 12,314,400 | not run | 11 ns | 205 ns | 47 ns |  | 0.05x | 4.38x |

## What this means for the campaign

The naive per-pair loop costs 55,030 ns per pair-window and that number does not move: it is flat across m (25 to 100) and across lag depth, because it is per-pair work that no amount of vectorisation amortises. The campaign's own arms fall with m instead, as the kernels fill, which is why the gap widens exactly where the experiments live.

The table below prices one cell of the median size of the campaign, 1,251,083,598 pair-windows, at each tier's median rate over the 179 cells, so every column divides exactly into the ratios beside it. Quoting median wall clocks instead would not divide, because the cell with the median wall is not the cell with the median size; for the record those measured medians are 167 s for BF_vect and 16 s for CorrTrack's faster backend.

| tier | ns per pair-window | that cell would take | against BF_vect |
|---|---|---|---|
| naive Python | 55,030 | 19.1 h | 393x slower |
| BF_vect | 140.2 | 175 s | 1x |
| BF_incr | 37.0 | 46 s | 3.79x faster |
| CorrTrack, faster backend | 26.9 | 34 s | 5.21x faster |

So the same CorrTrack run on the same data reads 5.2x against our baseline and about 2,045x against the naive one, because the second denominator is 393 times weaker. Every number this project reports uses the first. Against the stricter incremental baseline the naive tier is 1,486x.

Two things this table is not. It is not a claim that a naive implementation is the fair comparison: it is the opposite, a measurement of how much a paper's headline number owes to its baseline. And the naive numpy row is not slow at all, at 11 to 39 ns per pair-window it is faster than BF_vect, because it is one BLAS product per window that counts matches and returns nothing: no pairs emitted, no lag grid, no near-constant or spike guards, no state carried between windows. It is a floor on the arithmetic, not a baseline anyone could use, and the lagged rows show its count already diverging from the exact arms (147,152 against 156,969 at m=200) because it does not reproduce the harness's early-window lag truncation.

## Every arm of the campaign on the same standard

The same reading applied to every arm: its median rate over the cells it ran in, and the same median-size cell (1,251,083,598 pair-windows) priced at that rate. Median recall sits beside it, because a rate only compares between arms that return the same answer.

| arm | ns per pair-window | median-size cell | against BF_vect | median recall | cells |
|---|---|---|---|---|---|
| naive Python (probe) | 55,030 | 19.1 h | 393x slower | 1.000 | probe |
| naive numpy (probe, counts only) | 11.0 | 14 s | 12.74x faster | n/a | probe |
| BF_vect | 140.2 | 175 s | 1x | 1.000 | 143 |
| CorrTrack-LSH | 29.3 | 37 s | 4.79x faster | 0.979 | 143 |
| CorrTrack-Ham | 31.3 | 39 s | 4.47x faster | 0.993 | 143 |
| FilCorr | 39.3 | 49 s | 3.57x faster | 1.000 | 95 |
| BF_incr | 37.0 | 46 s | 3.79x faster | 1.000 | 143 |
| CorrJoin | 97.0 | 121 s | 1.44x faster | 1.000 | 47 |
| StatStream | 55.0 | 69 s | 2.55x faster | 0.989 | 143 |
| BRAID | 87.2 | 109 s | 1.61x faster | 1.000 | 143 |
| TSUBASA | 114.4 | 143 s | 1.22x faster | 1.000 | 95 |
| ParCorr | 210.2 | 263 s | 1.50x slower | 0.965 | 47 |
| CSZ | 588.4 | 736 s | 4.20x slower | 0.968 | 47 |

### The same rates per class and space

A pooled median compares arms over different sets of cells as soon as one arm is missing from a class, and a missing class is never a random sample: it is the hardest or the easiest part of the campaign. Within one class and space every surviving arm covers the same cells, so these columns are the comparable ones.

| arm | S raw | S diff | L raw | L diff | N raw | N diff | common cells |
|---|---|---|---|---|---|---|---|
| BF_vect | 123 | 121 | 141 | 141 | 142 | 141 | 121.5 |
| CorrTrack-LSH | 59 | 45 | 21 | 11 | 33 | 18 | 49.9 |
| CorrTrack-Ham | 52 | 46 | 23 | 14 | 35 | 19 | 48.2 |
| FilCorr | 67 | 66 | 31 | 30 |  |  | 65.8 |
| BF_incr | 75 | 75 | 35 | 35 | 36 | 35 | 75.1 |
| CorrJoin | 89 | 98 |  |  |  |  | 97.0 |
| StatStream | 83 | 72 | 36 | 26 | 62 | 47 | 77.8 |
| BRAID | 91 | 89 | 84 | 85 | 86 | 85 | 89.9 |
| TSUBASA | 204 | 204 |  |  | 106 | 104 | 203.5 |
| ParCorr | 218 | 210 |  |  |  |  | 210.2 |
| CSZ | 554 | 660 |  |  |  |  | 588.4 |

The last column is the median over the 47 cells where every arm in this table ran, which is the only column where all the numbers price the same work. Blank cells are classes the arm does not cover.


The two probe rows are not campaign runs: naive numpy is its best value, the m=200 lagged one, and it counts matches without emitting pairs, so it has no recall to report. An arm's cell count is the cells left after the authors-only filter: the synchronous class for ParCorr, CSZ and CorrJoin, everything but the negative class for FilCorr, everything but the lagged positive class for TSUBASA.

