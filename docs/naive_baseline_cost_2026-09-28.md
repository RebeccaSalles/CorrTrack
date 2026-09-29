# Per pair-window cost of the exact tiers (2026-09-28)

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

The table below prices one cell of the median size of the campaign, 1,251,083,598 pair-windows, at each tier's median rate over the 179 cells, so every column divides exactly into the ratios beside it. Quoting median wall clocks instead would not divide, because the cell with the median wall is not the cell with the median size; for the record those measured medians are 168 s for BF_vect and 18 s for CorrTrack's faster backend.

| tier | ns per pair-window | that cell would take | against BF_vect |
|---|---|---|---|
| naive Python | 55,030 | 19.1 h | 393x slower |
| BF_vect | 140.2 | 175 s | 1x |
| BF_incr | 39.3 | 49 s | 3.57x faster |
| CorrTrack, faster backend | 32.7 | 41 s | 4.28x faster |

So the same CorrTrack run on the same data reads 4.3x against our baseline and about 1,682x against the naive one, because the second denominator is 393 times weaker. Every number this project reports uses the first. Against the stricter incremental baseline the naive tier is 1,401x.

Two things this table is not. It is not a claim that a naive implementation is the fair comparison: it is the opposite, a measurement of how much a paper's headline number owes to its baseline. And the naive numpy row is not slow at all, at 11 to 39 ns per pair-window it is faster than BF_vect, because it is one BLAS product per window that counts matches and returns nothing: no pairs emitted, no lag grid, no near-constant or spike guards, no state carried between windows. It is a floor on the arithmetic, not a baseline anyone could use, and the lagged rows show its count already diverging from the exact arms (147,152 against 156,969 at m=200) because it does not reproduce the harness's early-window lag truncation.

## The campaign's own cost, for scale

Median over the 179 cells of the final m=500 campaign, each arm's wall clock divided by the pair-windows it covers:

| arm | ns per pair-window | cells |
|---|---|---|
| BF_vect | 140.2 | 179 |
| BF_incr | 39.3 | 179 |

