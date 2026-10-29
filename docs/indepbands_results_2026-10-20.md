# Independent band bits: cluster results (2026-10-20)

Working copy `corrtrack_release_indepbands`, snapshot `indepbands_20261019` (built and tested on mercantour3-13),
OAR 3202627 to 3202655. The 24 fixed-degree cells of docs/synthetic_degree_sweep_2026-10-18.md (copied from dev):
AR(1) raw, W 60, step 6, 30 lags, 55 partners per series, m 125 to 4000, T 0.7 / 0.8 / 0.9 / 0.95. In each cell one
job ran BF_vect, CorrTrack-LSH with the pooled band bits (`corrtrack`) and with independent band bits
(`corrtrack_indep`), same node, same truth, same tuned parameters (the main run's LSH tuning per threshold).
Results: `~/results/synth_degree_indep/nway/`. Regenerated inputs deleted on the cluster afterwards.

## Exponents of m (least squares on logs)

| T | textbook 1 + rho | predicted pooled | predicted independent | pooled wall | independent wall | pooled, m <= 2000 | independent, m <= 2000 | BF_vect |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 1.42 | 1.69 | 1.43 | 1.85 | 1.60 | 1.77 | 1.53 | 2.02 |
| 0.80 | 1.33 | 1.56 | 1.34 | 1.68 | 1.46 | 1.58 | 1.38 | 2.02 |
| 0.90 | 1.22 | 1.40 | 1.25 | 1.45 | 1.31 | 1.34 | 1.23 | 2.02 |
| 0.95 | 1.15 | 1.29 | 1.19 | 1.31 | 1.23 | 1.21 | 1.15 | 2.01 |

Candidate search alone: pooled 1.99 / 1.85 / 1.64 / 1.48, independent 1.72 / 1.60 / 1.46 / 1.36.
The pooled column reproduces the final112 sweep (1.84 / 1.66 / 1.44 / 1.30), so the newer code base changes nothing
here.

## Speedup over BF_vect at m = 4000, and wall clock

| T | pooled | independent | independent / pooled wall |
|---|---|---|---|
| 0.70 | 9.2x | 20.7x | 0.44 |
| 0.80 | 19.0x | 37.7x | 0.50 |
| 0.90 | 47.1x | 71.8x | 0.66 |
| 0.95 | 80.9x | 104.6x | 0.77 |

At m = 125 the two are level (bands are few and narrow either way); the gap grows with m in every threshold.
Recall: pooled 0.9982 to 0.9999, independent 0.9988 to 0.9998; precision 1 everywhere. Sketching cost is identical
(8.4 to 9.0 s at m = 4000), but that says nothing about the projected bits: they are computed at insert, which is
timed inside the candidate phase (correction 2026-10-20, see below). The figures above are net of that cost.

## Insert cost of the projected bits (local profile, 2026-10-20)

`CORRTRACK_PROFILE=1`, same fixed-degree inputs and tuning, mean of two runs per mode:

| cell | insert pooled | insert independent | insert share of independent wall | scan pooled -> independent | wall pooled -> independent |
|---|---|---|---|---|---|
| m 500, T 0.8 | 0.5 s | 2.3 s | 21% | 10.1 -> 5.1 s | 14.6 -> 10.9 s |
| m 1000, T 0.8 | 1.5 s | 7.0 s | 21% | 43.0 -> 19.6 s | 51.7 -> 33.3 s |
| m 500, T 0.7 | 0.9 s | 3.9 s | 24% | 18.3 -> 8.5 s | 23.7 -> 16.1 s |
| m 1000, T 0.7 | 3.2 s | 12.5 s | 22% | 91.6 -> 36.4 s | 104.5 -> 56.6 s |

The projection costs m x n_bands x band_width x n_vectors multiply-adds per step, the same order as the search, so it
stays about a fifth of the run as m grows. It is a scalar per-window loop in `_insert_one`; projecting all windows
inserted in a step as one BLAS product (sketches m x 64 times the 64 x (n_bands * band_width) matrix) should cut it
by an order of magnitude. Implemented 2026-10-20 (see below).

## Batched projection on the cluster (OAR 3208060, 2026-10-23)

`insert_many` projects the whole batch with one float64 OpenBLAS product and `_insert_one` packs each band key from
the signs of its slice. Same node (mercantour3, OpenBLAS 0.3.25 single-threaded, 11.8 GFLOP/s on the product),
m = 1000, three alternating rounds, median (runs within 1 to 2%):

| T | variant | n_bands | insert | scan | candidate phase | wall | correlated rows |
|---|---|---|---|---|---|---|---|
| 0.8 | pooled | 59 | 2.2 s | 48.2 s | 53.3 s | 63.5 s | 25,878,054 |
| 0.8 | independent, per-window projection | 29 | 10.8 s | 24.9 s | 37.3 s | 47.9 s | 25,876,715 |
| 0.8 | independent, batched projection | 29 | 3.2 s | 25.5 s | 30.2 s | 40.5 s | 25,876,715 |
| 0.7 | pooled | 128 | 4.3 s | 87.3 s | 97.0 s | 108.1 s | 25,873,695 |
| 0.7 | independent, per-window projection | 51 | 18.4 s | 40.2 s | 61.2 s | 73.0 s | 25,871,296 |
| 0.7 | independent, batched projection | 51 | 5.0 s | 41.2 s | 48.5 s | 59.7 s | 25,871,296 |

Batching cuts the insert cost 3.4x (to 8% of the wall, from 23%), leaving it 1.2 to 1.5x the pooled insert. Against
pooled, the independent mode with batched projection is 1.57x faster at T = 0.8 and 1.81x at T = 0.7 at m = 1000.
Per-window and batched give the same correlated set (same row counts; locally identical fingerprints); the
independent sets differ from pooled by under 0.01% of rows (recall effect of the different banding).
The 24-cell cluster sweep above used the per-window projection, so its independent walls overstate the cost by
roughly the insert difference (about 7 to 13 s at m = 1000); its exponents are therefore slightly pessimistic.

## Reading

- Independent bits remove the excess that the shared 64-bit pool caused: n_bands at m = 4000 drops from
  354 / 132 / 41 / 19 to 93 / 46 / 20 / 11, and the measured exponent drops by 0.25 / 0.22 / 0.14 / 0.08.
- Up to m = 2000 the independent exponent equals textbook LSH at T = 0.9 (1.23 vs 1.22) and 0.95 (1.15 vs 1.15) and is
  within 0.05 to 0.11 of it at 0.8 and 0.7; with m = 4000 included every arm steepens (the machine effect seen
  in the main sweep, which also lifts BF_incr and FilCorr), adding 0.07 to 0.08.
- The remaining gap at low T is the low tuned gate (0.5 at T = 0.7): candidates tested grow faster than the output.
- The tuning was done for the pooled mode and reused; tuning the independent mode on its own could move the gate
  and occupancy and help further. Not tested.

## Status

Experimental, opt-in (`candidate_lsh_independent_bits`), only in this copy. Not merged into corrtrack_release_dev;
every paper number still comes from the pooled mode.

## Updated table: batched projection, and the independent mode's own tuning (2026-10-24)

OAR 3215158 to 3215186 (pooled and independent with the pooled tuning, batched projection, one job per cell) and
3215199 to 3215227 (independent with its own tuning). No brute force in these runs: recall = correlated / brute-force
correlated from the earlier runs (precision 1; brute-force counts identical across runs); speedup uses the median
brute-force wall of the two earlier runs.

| T | textbook 1 + rho | pooled | independent, pooled tuning | independent, own tuning | BF_vect |
|---|---|---|---|---|---|
| 0.70 | 1.42 | 1.86 | 1.56 | 1.61 | 2.02 |
| 0.80 | 1.33 | 1.68 | 1.42 | 1.49 | 2.02 |
| 0.90 | 1.22 | 1.44 | 1.27 | 1.35 | 2.02 |
| 0.95 | 1.15 | 1.31 | 1.19 | 1.22 | 2.02 |

Fits up to m = 2000: pooled 1.78 / 1.59 / 1.33 / 1.21, independent with pooled tuning 1.47 / 1.32 / 1.17 / 1.10,
own tuning 1.52 / 1.39 / 1.26 / 1.11.

| T | speedup at m = 4000: pooled | independent, pooled tuning | independent, own tuning |
|---|---|---|---|
| 0.70 | 9.0x | 24.5x | 19.7x |
| 0.80 | 18.8x | 43.8x | 31.6x |
| 0.90 | 47.0x | 82.3x | 56.1x |
| 0.95 | 80.0x | 114.5x | 95.9x |

Recall: pooled 0.9982 to 0.9999, independent with pooled tuning 0.9988 to 0.9998, own tuning 0.9967 to 0.9998 (lowest
at T = 0.7 and 0.8, where it chose gate offset 0).

**The own tuning is worse, at every m including the m = 500 it was tuned at.** It chose occupancy 48 / 48 / 48 / 24
(the top of the widened grid) and 21 / 14 / 8 / 6 bands at m = 500, against the pooled tuning's occupancy 12 and
79 / 40 / 18 / 10 bands. The hyperopt's selection ranks feasible configurations by fewest bands first, which favours
the largest occupancy (narrower bands, fewer of them) while ignoring that each bucket then holds about 48 unrelated
windows. The pooled campaign grid stopped at 12, so the same bias was capped there. Fixing it means ranking by the
measured proxy time (or by bands x occupancy) for this mode, or capping occupancy; not done.

## Version B: automatic occupancy and its own gate / Hamming tuning (2026-10-28)

OAR 3225818 to 3225850 (snapshot indepbands_auto_20261027, LSH_DEFAULT_COST_RATIO = 11, occupancy = c rho / (1 - rho),
tuning at m = 500 searches gate offset x Hamming only), results `~/results/synth_degree_auto/`. Pooled and
"independent, pooled tuning" columns from the batched run (same pooled code). Recall from stored brute-force counts.

| T | textbook 1 + rho | pooled | independent, pooled tuning | independent, auto occupancy + own tuning | BF_vect |
|---|---|---|---|---|---|
| 0.70 | 1.42 | 1.86 | 1.56 | 1.56 | 2.02 |
| 0.80 | 1.33 | 1.68 | 1.42 | 1.40 | 2.02 |
| 0.90 | 1.22 | 1.44 | 1.27 | 1.21 | 2.02 |
| 0.95 | 1.15 | 1.31 | 1.19 | 1.13 | 2.02 |

Fits to m = 2000 (version B): 1.47 / 1.30 / 1.12 / 1.04.

| T | speedup at m = 4000: pooled | independent, pooled tuning | version B |
|---|---|---|---|
| 0.70 | 9.0x | 24.5x | 24.8x |
| 0.80 | 18.8x | 43.8x | 47.9x |
| 0.90 | 47.0x | 82.3x | 101.9x |
| 0.95 | 80.0x | 114.5x | 145.0x |

Version B recall: 0.9995 to 0.9997 (T 0.7), 0.9997 to 0.9998 (0.8), 0.9979 to 0.9985 (0.9), 0.9988 to 0.9994 (0.95).
Tuned values (m = 500): occupancy 8.0 / 5.43 / 3.17 / 2.0, gate offset 0.1 / 0.1 / 0.0 / 0.0, Hamming 0.40 throughout.

m-dependence check (user): the same tuning at m = 2000 for T = 0.9 (OAR 3225961) chose offset 0.2 instead of 0.0, Hamming
0.40 both. The trial tables show every offset within timing noise at both sizes (repeat timings of the same configuration
differ by up to 10 to 17%), and the gate passes almost the same candidates whatever the offset (174,533 to 174,917 at
m = 500; 46,124 to 46,237 at m = 2000): a tie broken by noise, not a dependence on m. Offset 0.0 costs recall (0.998 vs
0.9998) for no speed, which is why version B's recall at T = 0.9 is 0.998. Suggested: when configurations are within
timing noise of the fastest, prefer the higher recall.

## Real data: 36 lagged positive cells, independent mode (version B) vs pooled mode, FilCorr, BF_incr (2026-10-28)

Cells: the 6 real datasets x T = 0.7 / 0.8 / 0.9 x raw / differenced, m = 500 (sp500 444), lagged, positive, the
campaign's window settings. Machine: esterel38 (2x AMD EPYC 7513, queue p1, CPU cores only, 8 cores per job),
snapshot `indepbands_sel_e38_20261027`. Per cell, one job runs the independent mode (its own version B tuning with the
work-then-recall selection rule, OAR 36 tunings in `real36/hyperopt_indep`), FilCorr and BF_incr; a second job on the same
node type runs the pooled mode with the campaign's tuning. EVAL_SPAN=full (results `~/corrtrack_abaca_results/real36_full`,
local `~/results/real36_full`). Recall = reported pairs / campaign brute-force count (CorrTrack precision is 1); the
pooled rerun reproduces the campaign's recall within 6e-5 on every cell. Recall target 0.95. A first run on the holdout
span (`real36`) is discarded for recall; its speed ratios agree (median new / pooled 1.45 / 1.55 / 1.77x).

| cell | T | space | new / pooled speed | new vs FilCorr | new vs BF_incr | recall new | recall pooled |
|---|---|---|---|---|---|---|---|
| acwi_capweighted | 0.7 | diff | 1.58x | 1.44x | 1.57x | 0.9741 | 0.9756 |
| acwi_capweighted | 0.7 | raw | 1.14x | 0.49x | 0.47x | 0.9705 | 0.9866 |
| global_weather | 0.7 | diff | 1.80x | 1.70x | 1.81x | 0.9773 | 0.9778 |
| global_weather | 0.7 | raw | 1.43x | 0.91x | 0.97x | 0.9602 | 0.9709 |
| smartmeter | 0.7 | diff | 1.45x | 1.73x | 1.76x | 0.9731 | 0.9569 |
| smartmeter | 0.7 | raw | 1.35x | 1.36x | 1.39x | 0.9622 | 0.9581 |
| sp500 | 0.7 | diff | 1.47x | 1.17x | 1.25x | 0.9839 | 0.9861 |
| sp500 | 0.7 | raw | 1.14x | 0.46x | 0.48x | 0.9719 | 0.9814 |
| streamflow | 0.7 | diff | 1.39x | 1.12x | 1.23x | 0.9807 | 0.9766 |
| streamflow | 0.7 | raw | 1.19x | 0.53x | 0.56x | 0.9777 | 0.9805 |
| wikipedia | 0.7 | diff | 1.61x | 1.53x | 1.63x | 0.9600 | 0.9606 |
| wikipedia | 0.7 | raw | 1.61x | 1.38x | 1.49x | 0.9554 | 0.9528 |
| acwi_capweighted | 0.8 | diff | 1.71x | 2.57x | 2.79x | 0.9514 | 0.9508 |
| acwi_capweighted (hard) | 0.8 | raw | 1.41x | 0.58x | 0.59x | 0.9777 | 0.9848 |
| global_weather | 0.8 | diff | 1.67x | 2.83x | 2.98x | 0.9730 | 0.9901 |
| global_weather | 0.8 | raw | 1.48x | 1.70x | 1.83x | 0.9798 | 0.9735 |
| smartmeter (hard) | 0.8 | diff | 1.43x | 2.63x | 2.71x | 0.9845 | 0.9310 |
| smartmeter | 0.8 | raw | 1.32x | 2.08x | 2.12x | 0.9773 | 0.9684 |
| sp500 | 0.8 | diff | 1.46x | 1.52x | 1.61x | 0.9791 | 0.9764 |
| sp500 (hard) | 0.8 | raw | 1.41x | 0.56x | 0.58x | 0.9755 | 0.9852 |
| streamflow | 0.8 | diff | 1.43x | 1.97x | 2.18x | 0.9730 | 0.9646 |
| streamflow | 0.8 | raw | 1.13x | 0.74x | 0.77x | 0.9863 | 0.9752 |
| wikipedia | 0.8 | diff | 1.74x | 2.62x | 2.76x | 0.9702 | 0.9705 |
| wikipedia | 0.8 | raw | 1.68x | 2.36x | 2.55x | 0.9682 | 0.9851 |
| acwi_capweighted | 0.9 | diff | 1.65x | 4.78x | 5.32x | 0.9787 | 0.9820 |
| acwi_capweighted (hard) | 0.9 | raw | 1.29x | 0.88x | 0.93x | 0.9753 | 0.9686 |
| global_weather | 0.9 | diff | 1.82x | 5.40x | 5.79x | 0.9892 | 0.9935 |
| global_weather | 0.9 | raw | 1.65x | 3.48x | 3.79x | 0.9858 | 0.9686 |
| smartmeter (hard) | 0.9 | diff | 1.25x | 4.18x | 4.38x | 0.9813 | 0.9295 |
| smartmeter | 0.9 | raw | 1.23x | 3.52x | 3.66x | 0.9794 | 0.9561 |
| sp500 | 0.9 | diff | 1.64x | 4.22x | 4.33x | 0.9746 | 0.9749 |
| sp500 (hard) | 0.9 | raw | 1.36x | 0.92x | 0.97x | 0.9746 | 0.9694 |
| streamflow | 0.9 | diff | 1.44x | 3.41x | 3.55x | 0.9886 | 0.9802 |
| streamflow (hard) | 0.9 | raw | 1.55x | 1.35x | 1.45x | 0.9926 | 0.9841 |
| wikipedia | 0.9 | diff | 1.73x | 4.95x | 5.19x | 0.9862 | 0.9863 |
| wikipedia | 0.9 | raw | 1.64x | 4.32x | 4.62x | 0.9837 | 0.9895 |

## Summary by threshold

| T | cells | new / pooled speed (median, range) | new faster than FilCorr / tie / slower | new faster than BF_incr / tie / slower | recall new (min) | recall pooled (min) | campaign: CorrTrack faster than FilCorr |
|---|---|---|---|---|---|---|---|
| 0.7 | 12 | 1.44x (1.14 to 1.80) | 8 / 1 / 3 | 8 / 1 / 3 | 0.9554 | 0.9528 | 6 of 12 |
| 0.8 | 12 | 1.45x (1.13 to 1.74) | 9 / 0 / 3 | 9 / 0 / 3 | 0.9514 | 0.9310 | 9 of 12 |
| 0.9 | 12 | 1.59x (1.23 to 1.82) | 10 / 1 / 1 | 10 / 2 / 0 | 0.9746 | 0.9295 | 9 of 12 |

## The 7 hard cells (no feasible pooled configuration on the tuning sample)

Speed new / pooled: median 1.41x; recall new 0.9746 to 0.9926, pooled 0.9295 to 0.9852; cells with recall >= 0.95: new 7 of 7, pooled 5 of 7.

Geometric-mean speed ratios (new mode faster when > 1):

| T | new / pooled | new vs FilCorr | new vs BF_incr | total wall, s: new / pooled / FilCorr / BF_incr |
|---|---|---|---|---|
| 0.7 | 1.42x | 1.05x | 1.10x | 304 / 398 / 275 / 287 |
| 0.8 | 1.48x | 1.61x | 1.70x | 192 / 271 / 260 / 273 |
| 0.9 | 1.51x | 2.95x | 3.14x | 99 / 141 / 249 / 264 |
| all 36 | 1.47x | 1.71x | 1.80x | 594 / 811 / 783 / 823 |

Reading. The independent mode is faster than the pooled mode on every one of the 36 cells (1.13x to 1.82x) and meets the
0.95 target on all 36 (minimum 0.9514); the pooled mode misses it on 2 (smartmeter differenced, T 0.8 and 0.9: 0.931,
0.930). On the 7 hard cells, recall rises from 5 of 7 to 7 of 7 above 0.95, at 1.41x the speed. Against FilCorr the
remaining losses are the raw acwi, sp500 and streamflow cells at T = 0.7 / 0.8 (FilCorr 1.35x to 2.2x faster) and
raw acwi / sp500 at T = 0.9 (within 12%); these are the dense raw cells where many pairs are correlated and the output
dominates. Same-node comparison only; the campaign numbers (mercantour) are not mixed in.
