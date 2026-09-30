# Memory and energy per arm (2026-09-29)

Measured, not modelled: every arm of every cell ran as an isolated child process with an RSS sampler around it (`abaca/resource_probe.py`), so the numbers below are that arm's own footprint. Energy is the node power Grid'5000 recorded for the job, integrated over each arm's own start and end.

## 1. Memory, final m=500 campaign (179 cells)

Peak resident set of the arm's process, and the delta over what it had already allocated before the arm started, which is the part the method itself is responsible for.

| arm | median peak RSS (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 352 | 159 | 7,427 | 179 |
| CorrTrack-LSH | 350 | 163 | 13,494 | 179 |
| CorrTrack-Ham | 352 | 163 | 13,457 | 179 |
| FilCorr | 285 | 163 | 13,330 | 179 |
| BF_incr | 307 | 165 | 13,356 | 179 |
| CorrJoin | 422 | 253 | 11,479 | 119 |
| StatStream | 518 | 267 | 13,621 | 179 |
| BRAID | 366 | 163 | 13,432 | 179 |
| TSUBASA | 411 | 180 | 13,526 | 179 |
| ParCorr | 874 | 570 | 11,781 | 119 |
| ThinBRAID | 830 | 171 | 17,319 | 178 |
| CSZ | 862 | 467 | 11,729 | 119 |

Same cells, the delta only:

| arm | median peak RSS delta (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 161 | 5 | 7,273 | 179 |
| CorrTrack-LSH | 66 | 6 | 7,396 | 179 |
| CorrTrack-Ham | 61 | 6 | 7,371 | 179 |
| FilCorr | 55 | 8 | 7,217 | 179 |
| BF_incr | 73 | 14 | 7,239 | 179 |
| CorrJoin | 225 | 91 | 7,402 | 119 |
| StatStream | 286 | 13 | 7,824 | 179 |
| BRAID | 131 | 8 | 7,304 | 179 |
| TSUBASA | 197 | 8 | 7,399 | 179 |
| ParCorr | 690 | 385 | 7,704 | 119 |
| ThinBRAID | 491 | 13 | 14,451 | 178 |
| CSZ | 661 | 307 | 7,652 | 119 |

### Peak RSS delta per class and space

Pooled medians compare arms over different cell sets whenever an arm is missing from a class, so the comparable reading is within a class:

| arm | S raw | S diff | L raw | L diff | N raw | N diff | common cells |
|---|---|---|---|---|---|---|---|
| BF_vect | 104 | 14 | 230 | 160 | 245 | 162 | 153 |
| CorrTrack-LSH | 117 | 19 | 179 | 34 | 202 | 31 | 58 |
| CorrTrack-Ham | 116 | 20 | 199 | 30 | 199 | 33 | 58 |
| FilCorr | 115 | 22 | 157 | 24 | 165 | 27 | 52 |
| BF_incr | 117 | 27 | 159 | 41 | 159 | 40 | 65 |
| CorrJoin | 359 | 159 | 640 | 198 |  |  | 225 |
| StatStream | 402 | 192 | 624 | 188 | 594 | 205 | 290 |
| BRAID | 107 | 22 | 220 | 126 | 229 | 128 | 125 |
| TSUBASA | 119 | 43 | 307 | 197 | 316 | 195 | 189 |
| ParCorr | 693 | 604 | 1,103 | 645 |  |  | 690 |
| ThinBRAID | 475 | 254 | 3,662 | 1,106 | 7,226 | 2,303 | 475 |
| CSZ | 718 | 568 | 1,038 | 626 |  |  | 661 |

Median peak RSS delta in MB, per capability class and space. The last column is the median over the 119 cells where every arm ran, the only column whose numbers all price the same work; blank cells are classes the arm does not cover.

## 2. Memory, synthetic campaign (64 cells, m from 125 to 2000)

The same arms on generated data, where m is an axis rather than a constant:

| arm | median peak RSS (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 725 | 181 | 8,891 | 64 |
| CorrTrack-LSH | 1,060 | 206 | 13,590 | 64 |
| CorrTrack-Ham | 1,055 | 208 | 13,443 | 64 |
| FilCorr | 1,013 | 205 | 13,432 | 64 |
| BF_incr | 1,025 | 208 | 13,774 | 64 |
| CorrJoin | 1,208 | 261 | 14,004 | 64 |
| StatStream | 1,200 | 281 | 14,376 | 64 |
| BRAID | 1,142 | 211 | 15,288 | 64 |
| TSUBASA | 1,185 | 221 | 16,319 | 64 |
| ParCorr | 1,535 | 347 | 14,907 | 64 |
| ThinBRAID | 1,097 | 212 | 14,384 | 64 |
| CSZ | 1,510 | 295 | 14,968 | 64 |

Peak RSS at m=2000 only, the largest cells of the study:

| arm | median peak RSS (MB) | min | max | cells |
|---|---|---|---|---|
| BF_vect | 8,888 | 8,887 | 8,891 | 4 |
| CorrTrack-LSH | 13,564 | 13,557 | 13,590 | 4 |
| CorrTrack-Ham | 13,439 | 13,402 | 13,443 | 4 |
| FilCorr | 13,426 | 13,337 | 13,432 | 4 |
| BF_incr | 13,722 | 13,662 | 13,774 | 4 |
| CorrJoin | 13,892 | 13,749 | 14,004 | 4 |
| StatStream | 14,361 | 14,326 | 14,376 | 4 |
| BRAID | 15,241 | 15,167 | 15,288 | 4 |
| TSUBASA | 16,281 | 16,257 | 16,319 | 4 |
| ParCorr | 14,874 | 14,808 | 14,907 | 4 |
| ThinBRAID | 14,349 | 14,217 | 14,384 | 4 |
| CSZ | 14,463 | 14,224 | 14,968 | 4 |

## 3. Energy, synthetic campaign

The power series has one sample per 15 s, so an arm is only separable from its neighbours when it runs well past a minute. This section therefore uses the cells where EVERY arm clears 60 s, which is 5 of the 64 cells, all of them the largest ones; taking a median per arm over whatever qualifies would compare different subsets. Dynamic energy subtracts the node's lowest power during that job, so it is what the arm added over the machine's baseline. The campaign's own jobs are past kwollect's retention window, so energy exists for the synthetic campaign only.

Cells used: m=1000, m=2000.

| arm | median energy (kJ) | dynamic (kJ) | mean power (W) | kJ per billion pair-windows | cells |
|---|---|---|---|---|---|
| BF_vect | 608.7 | 255.9 | 127 | 28.00 | 5 |
| CorrTrack-LSH | 23.9 | 10.4 | 128 | 1.10 | 5 |
| CorrTrack-Ham | 31.3 | 13.4 | 127 | 1.45 | 5 |
| FilCorr | 110.8 | 48.1 | 131 | 5.09 | 5 |
| BF_incr | 118.6 | 51.8 | 132 | 5.46 | 5 |
| CorrJoin | 131.4 | 64.0 | 128 | 6.04 | 5 |
| StatStream | 246.9 | 104.4 | 128 | 11.36 | 5 |
| BRAID | 575.9 | 251.4 | 132 | 26.49 | 5 |
| TSUBASA | 402.4 | 180.1 | 133 | 18.51 | 5 |
| ParCorr | 49.1 | 20.1 | 127 | 2.86 | 5 |
| ThinBRAID | 1,197.1 | 505.2 | 128 | 55.06 | 5 |
| CSZ | 145.1 | 72.3 | 127 | 11.29 | 5 |

The mean power column is the reading to keep in mind: it is within a few watts for every arm, so on this hardware energy is wall clock multiplied by a near-constant. The energy ranking is therefore the speed ranking, and the honest claim is that CorrTrack saves energy in proportion to the time it saves, not that it is more efficient per unit of work in some further sense. A wattmeter node, which the mercantour cluster does not have, would be needed to say more.

