# Memory and energy per arm (2026-09-29)

**Authors-only variant.** Every result that exists only because we extended a method has been removed: an arm is dropped from a cell when the capability that cell exercises is tagged `enabled_by_us` in its own run. That takes TSUBASA, ParCorr, CSZ and CorrJoin out of the lagged classes, FilCorr out of the negative class, and leaves the synchronous class untouched, so the cell counts differ per arm. ThinBRAID is removed outright: its published algorithm does not reproduce here, at median recall 0.815 and precision 0.046, which is a gap between that paper and its implementation rather than a result about it. Thresholds below 0.8 are left out. Nothing else changes: same runs, same measurements, same code.

Measured, not modelled: every arm of every cell ran as an isolated child process with an RSS sampler around it (`abaca/resource_probe.py`), so the numbers below are that arm's own footprint. Energy is the node power Grid'5000 recorded for the job, integrated over each arm's own start and end.

## 1. Memory, final m=500 campaign (179 cells)

Peak resident set of the arm's process, and the delta over what it had already allocated before the arm started, which is the part the method itself is responsible for.

| arm | median peak RSS (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 330 | 159 | 3,851 | 143 |
| CorrTrack-LSH | 284 | 163 | 6,595 | 143 |
| CorrTrack-Ham | 287 | 163 | 6,583 | 143 |
| FilCorr | 264 | 163 | 5,663 | 95 |
| BF_incr | 269 | 165 | 6,498 | 143 |
| CorrJoin | 360 | 253 | 1,086 | 47 |
| StatStream | 435 | 267 | 7,166 | 143 |
| BRAID | 333 | 163 | 6,572 | 143 |
| TSUBASA | 373 | 180 | 6,665 | 95 |
| ParCorr | 821 | 570 | 2,444 | 47 |
| CSZ | 792 | 467 | 2,413 | 47 |

Same cells, the delta only:

| arm | median peak RSS delta (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 159 | 5 | 3,697 | 143 |
| CorrTrack-LSH | 39 | 6 | 3,789 | 143 |
| CorrTrack-Ham | 34 | 6 | 3,770 | 143 |
| FilCorr | 27 | 8 | 3,627 | 95 |
| BF_incr | 49 | 14 | 3,648 | 143 |
| CorrJoin | 175 | 91 | 711 | 47 |
| StatStream | 229 | 13 | 4,271 | 143 |
| BRAID | 124 | 8 | 3,705 | 143 |
| TSUBASA | 151 | 8 | 3,797 | 95 |
| ParCorr | 637 | 385 | 2,092 | 47 |
| CSZ | 596 | 310 | 2,073 | 47 |

### Peak RSS delta per class and space

Pooled medians compare arms over different cell sets whenever an arm is missing from a class, so the comparable reading is within a class:

| arm | S raw | S diff | L raw | L diff | N raw | N diff | common cells |
|---|---|---|---|---|---|---|---|
| BF_vect | 71 | 12 | 210 | 160 | 228 | 159 | 29 |
| CorrTrack-LSH | 84 | 16 | 129 | 30 | 150 | 29 | 27 |
| CorrTrack-Ham | 88 | 17 | 160 | 27 | 167 | 23 | 22 |
| FilCorr | 85 | 14 | 87 | 19 |  |  | 25 |
| BF_incr | 81 | 23 | 103 | 37 | 123 | 36 | 34 |
| CorrJoin | 321 | 151 |  |  |  |  | 175 |
| StatStream | 366 | 146 | 589 | 151 | 576 | 150 | 216 |
| BRAID | 79 | 14 | 180 | 122 | 202 | 123 | 24 |
| TSUBASA | 87 | 35 |  |  | 277 | 192 | 44 |
| ParCorr | 689 | 556 |  |  |  |  | 637 |
| CSZ | 692 | 557 |  |  |  |  | 596 |

Median peak RSS delta in MB, per capability class and space. The last column is the median over the 47 cells where every arm ran, the only column whose numbers all price the same work; blank cells are classes the arm does not cover.

## 2. Memory, synthetic campaign (64 cells, m from 125 to 2000)

The same arms on generated data, where m is an axis rather than a constant:

| arm | median peak RSS (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 725 | 181 | 8,891 | 60 |
| CorrTrack-LSH | 1,058 | 206 | 13,590 | 60 |
| CorrTrack-Ham | 1,054 | 208 | 13,443 | 60 |
| FilCorr | 1,026 | 205 | 13,432 | 60 |
| BF_incr | 1,025 | 208 | 13,774 | 60 |
| CorrJoin | 474 | 433 | 497 | 4 |
| StatStream | 1,201 | 281 | 14,376 | 60 |
| BRAID | 1,141 | 211 | 15,288 | 60 |
| TSUBASA | 316 | 311 | 320 | 4 |
| ParCorr | 791 | 657 | 859 | 4 |
| CSZ | 711 | 590 | 885 | 4 |

Peak RSS at m=2000 only, the largest cells of the study:

| arm | median peak RSS (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 8,888 | 8,887 | 8,891 | 4 |
| CorrTrack-LSH | 13,564 | 13,557 | 13,590 | 4 |
| CorrTrack-Ham | 13,439 | 13,402 | 13,443 | 4 |
| FilCorr | 13,426 | 13,337 | 13,432 | 4 |
| BF_incr | 13,722 | 13,662 | 13,774 | 4 |
| StatStream | 14,361 | 14,326 | 14,376 | 4 |
| BRAID | 15,241 | 15,167 | 15,288 | 4 |

## 3. Energy, synthetic campaign

The power series has one sample per 15 s, so an arm is only separable from its neighbours when it runs well past a minute. This section therefore uses the cells where EVERY arm clears 60 s, which is 5 of the 60 cells, all of them the largest ones; taking a median per arm over whatever qualifies would compare different subsets. Dynamic energy subtracts the node's lowest power during that job, so it is what the arm added over the machine's baseline. The campaign's own jobs are past kwollect's retention window, so energy exists for the synthetic campaign only.

Cells used: m=1000, m=2000.

| arm | median energy (kJ) | dynamic (kJ) | mean power (W) | kJ per billion pair-windows | cells |
|---|---|---|---|---|---|
| BF_vect | 608.7 | 255.9 | 127 | 28.00 | 5 |
| CorrTrack-LSH | 23.9 | 10.4 | 128 | 1.10 | 5 |
| CorrTrack-Ham | 31.3 | 13.4 | 127 | 1.45 | 5 |
| FilCorr | 110.8 | 48.1 | 131 | 5.09 | 5 |
| BF_incr | 118.6 | 51.8 | 132 | 5.46 | 5 |
| StatStream | 246.9 | 104.4 | 128 | 11.36 | 5 |
| BRAID | 575.9 | 251.4 | 132 | 26.49 | 5 |

The mean power column is the reading to keep in mind: it is within a few watts for every arm, so on this hardware energy is wall clock multiplied by a near-constant. The energy ranking is therefore the speed ranking, and the honest claim is that CorrTrack saves energy in proportion to the time it saves, not that it is more efficient per unit of work in some further sense. A wattmeter node, which the mercantour cluster does not have, would be needed to say more.

