# Occupancy of the independent-band-bits index: sweep, cost model, rule (2026-10-26)

## Why

Occupancy (expected unrelated windows per bucket) sets the band width K = log2(m L / occupancy); the band count then
follows from the recall target. Both hyperopt rankings tried so far push it to a grid edge: fewest bands picks the
largest occupancy, bands x occupancy the smallest, because each is monotone in occupancy. A defensible choice needs
the real cost, measured on the experiments' machine.

## Sweep

mercantour3 (Xeon Silver 4114, OpenBLAS single-threaded), snapshot indepbands_batched_20261023 (hashing at insertion,
batched), OAR 3217473 to 3217484. Fixed-degree AR(1) inputs (55 partners per series), W 60, step 6, 30 lags, n 6000,
independent band bits, gate offset and Hamming filter from each threshold's pooled tuning, occupancy 2 / 3 / 6 / 12 / 24 /
48, two alternating rounds per whole-node job; rounds agree within 3.2%. Results: `~/results/occ_sweep/sweep.*.out`.

Wall clock (mean of two rounds), bands in parentheses:

| T | m | occ 2 | occ 3 | occ 6 | occ 12 | occ 24 | occ 48 | within 10% of best |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 1000 | 78 s (125) | 64 s (93) | 59 s (69) | **58 s** (51) | 61 s (38) | 68 s (28) | 3 to 24 |
| 0.70 | 4000 | 923 s (226) | 836 s (168) | 735 s (125) | **720 s** (93) | 798 s (69) | 928 s (51) | 6 to 12 |
| 0.80 | 1000 | 42 s (58) | 38 s (46) | **38 s** (36) | 40 s (29) | 42 s (22) | 51 s (18) | 3 to 12 |
| 0.80 | 4000 | 379 s (93) | 354 s (74) | **346 s** (58) | 386 s (46) | 443 s (36) | 552 s (29) | 2 to 6 |
| 0.90 | 1000 | **24 s** (23) | 25 s (20) | 25 s (16) | 27 s (14) | 30 s (12) | 35 s (10) | 2 to 6 |
| 0.90 | 4000 | **154 s** (32) | 155 s (27) | 168 s (23) | 200 s (20) | 233 s (16) | 300 s (14) | 2 to 6 |
| 0.95 | 1000 | **19 s** (12) | 20 s (11) | 21 s (10) | 21 s (8) | 23 s (7) | 27 s (6) | 2 to 6 |
| 0.95 | 4000 | **99 s** (15) | 107 s (14) | 113 s (12) | 133 s (11) | 161 s (10) | 193 s (8) | 2 to 3 |

The correlated output varies by under 0.1% across occupancies (0.002 to 0.08%): occupancy is a speed knob only.

## Cost model and rule

Fitting the candidate-phase time as alpha x bands + beta x bands x occupancy per cell gives c = alpha / beta = 10 to 12
at m = 4000 and 13 to 17 at m = 1000: one band costs as much as about 12 unrelated collisions. This is the
decomposition of the E2LSH manual (Andoni, Indyk, E2LSH 0.1 user manual, 2005, Section 3.4): query time T_g + T_c,
T_g = O(d k L) for computing the L hash functions and retrieving their buckets, T_c = O(d #collisions); E2LSH fixes L
from the success probability and chooses k to minimise T_g + T_c, "to minimize the actual running time of query on
the host system". Here K plays E2LSH's k (set through occupancy) and the band count its L.

With independent bands, n_bands is proportional to (m L / occupancy)^rho with rho = log2(1 / (1 - arccos(T) / pi)),
so cost is proportional to occupancy^(-rho) (c + occupancy), minimised at

    occupancy* = c rho / (1 - rho).

With c = 12: 8.7 at T = 0.7, 5.9 at 0.8, 3.5 at 0.9, 2.2 at 0.95. Measured best: 12 (6 to 12 within 10%), 6, 2 (2 to 6
within 10%), 2. The rule lands inside the measured near-optimal range at every threshold, and the optimum is flat
(within 10% over a factor 2 to 4 of occupancy), so c need not be precise.

## Consequences

- The fixed occupancy 12 (the pooled tuning's choice, the top of its grid) is right at T = 0.7 but 10% slower than
  the rule at T = 0.8, 30% at T = 0.9 and 34% at T = 0.95 (m = 4000): 386 vs 346 s, 200 vs 154 s, 133 vs 99 s.
- For the paper: occupancy is set by minimising a calibrated query-cost model as E2LSH does (c calibrated once per
  implementation and machine), and the sweep shows the speed is insensitive to it near the optimum.

## How c is calibrated (2026-10-26)

Model: candidate-phase time t = beta x B(occupancy) x (c + occupancy), with B the band count, which the sizing formula
gives exactly for any occupancy, and beta a per-cell scale (number of queries, machine speed) that cancels out of the
choice of occupancy. c is the per-band cost (computing the band's hash code, looking up its bucket, the bookkeeping
around it) in units of one collision (one unrelated window found in a bucket: Hamming check and duplicate check).

Procedure, once per implementation and machine, not per dataset or per run: on the target machine, run CorrTrack in
the independent mode on a few representative cells at several occupancies (here 2 / 3 / 6 / 12 / 24 / 48, two
alternating rounds each, whole node), record the candidate-phase time and the band count, and fit c jointly: each
cell keeps its own beta (closed-form least squares), c is shared and minimises the summed squared relative error.
Here (8 cells, 4 thresholds x m = 1000 and 4000, 96 runs): c = 11.9; leaving any one cell out gives 11.4 to 12.5; the
m = 4000 cells alone give 10.2, the m = 1000 cells 14.1. The rule then uses only rho, which the band sizing already
computes from the threshold (p = 1 - arccos(T) / pi, rho = log2(1 / p)):

    occupancy* = c rho / (1 - rho)  ->  8.7 / 5.9 / 3.4 / 2.2 at T = 0.7 / 0.8 / 0.9 / 0.95 (c = 11.9).

Since the optimum is flat, a factor-2 error in c costs about 10% of speed at most in the measured cells.

Real-data check submitted: OAR 3222466, four final_112ce48 cells (sp500 T 0.9 and 0.8 raw, global_weather T 0.9
differenced, streamflow T 0.8 raw), each with its own tuned gate and Hamming filter, occupancy 2 to 48, two rounds.
Script `occ_real_run.py` (tested locally on sp500 and global_weather).

## Cross-CPU calibration and real-data check (2026-10-27)

Same cells on three CPU families (fixed degree, T 0.8 and 0.9, m 1000 and 4000, occupancy 2 to 48, two alternating
rounds, kernels rebuilt on each node): OAR 3222600 (mercantour5) and 3222630 (esterel38, 8 cores, no GPU, queue p1;
esterel33 / 34 / 35 / 40 were besteffort-only for the group). Joint fit of c on the common cells:

| CPU | c | m = 1000 | m = 4000 | best occupancy T 0.8 / 0.9 | round spread |
|---|---|---|---|---|---|
| Intel Xeon Silver 4114 (Skylake, mercantour3) | 11.0 | 13.1 | 9.4 | 6 / 2 | 2.1% |
| Intel Xeon Gold 6240 (Cascade Lake, mercantour5) | 10.8 | 12.2 | 9.7 | 6 / 2 | 3.5% |
| AMD EPYC 7513 (Zen 3, esterel38, shared node) | 10.3 | 12.1 | 8.9 | 6 (3 at m = 4000) / 2 | 5.6% |

Shipped default: `LSH_DEFAULT_COST_RATIO = 11.0` -> occupancy 8.0 / 5.4 / 3.2 / 2.0 at T = 0.7 / 0.8 / 0.9 / 0.95.

Real campaign cells (final_112ce48, own tuned gate and Hamming filter, OAR 3222466, mercantour3):

| cell | rule (c = 11) | best | within 10% of best |
|---|---|---|---|
| sp500 T 0.9 raw | 3.2 | 3 (37.4 s) | 2 to 6 |
| global_weather T 0.9 differenced | 3.2 | 3 (5.5 s) | 2 to 3 |
| sp500 T 0.8 raw | 5.4 | 3 (68.8 s) | 2 to 48 |
| streamflow T 0.8 raw | 5.4 | 2 (39.6 s) | 2 to 24 |

The rule is at the optimum at T = 0.9 and inside a flat range at T = 0.8 (these m ~ 500 cells are dominated by
validation). c fitted on the real cells: 15 (consistent with c rising at small m). The correlated output varies by
0.14 to 0.26% across occupancy 2 to 48 on these cells (a recall effect of the banding), less near the rule's value.
