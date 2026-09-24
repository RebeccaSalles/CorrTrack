# m=500 tables: rankings, margins and paired tests

`[f/s]` = datasets of that row where the arm is the fastest / among the two fastest, out of six. CorrTrack is one arm here: its value on a dataset is the better of its two tuned backends. The winner of a row is the arm with the most firsts, ties broken by the median speedup. It is then compared with the best other arm on each dataset separately: **margin** is the geometric mean of the six ratios, **spread** their geometric standard deviation (1.00 would mean the same ratio on every dataset), and **range** their smallest and largest value. A margin of 1.30 with a spread of 1.05 is a uniform win; the same margin with a spread of 1.60 means the win rests on one or two datasets.

Speedups within 10% of each other count as tied, so `[f/s]` can exceed one arm per place and a row can have several arms tied for first. **ahead/tied/behind** counts the datasets where the leader is more than 10% above the best other arm, within 10% of it, or more than 10% below. The verdict follows: *clear* when the leader is ahead on a majority of datasets and never behind, *tied* when every dataset is inside the band, *mixed* when it wins some and loses others.

## Every arm that ran

| table | space | T | STOMP | FilCorr | TSUBASA | BRAID | ThinBRAID | CorrTrack | ParCorr | CSZ | StatStream | CorrJoin | leader | margin | spread | range | ahead/tied/behind | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | differenced | 0.7 | 0/5 | 5/6 |  | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 2/3 |  | **FilCorr** | 1.02x | 1.12 | 0.90-1.16x | 2/3/1 | mixed |
| L | differenced | 0.8 | 0/1 | 0/5 |  | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 |  | **CorrTrack** | 1.46x | 1.08 | 1.28-1.59x | 6/0/0 | clear |
| L | differenced | 0.9 | 0/1 | 0/3 |  | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/5 |  | **CorrTrack** | 2.04x | 1.05 | 1.91-2.15x | 6/0/0 | clear |
| L | differenced | 0.95 | 0/1 | 0/1 |  | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/5 |  | **CorrTrack** | 2.43x | 1.09 | 2.17-2.70x | 6/0/0 | clear |
| L | raw | 0.7 | 1/6 | 6/6 |  | 0/1 | 0/0 | 1/3 | 0/0 | 0/0 | 1/2 |  | **FilCorr** | 1.11x | 1.05 | 1.01-1.16x | 5/1/0 | clear |
| L | raw | 0.8 | 0/3 | 3/6 |  | 0/0 | 0/0 | 3/3 | 0/0 | 0/0 | 1/2 |  | **FilCorr** | 0.93x | 1.27 | 0.65-1.16x | 3/0/3 | mixed |
| L | raw | 0.9 | 0/4 | 3/4 |  | 0/0 | 0/0 | 4/4 | 0/0 | 0/0 | 0/5 |  | **CorrTrack** | 1.18x | 1.63 | 0.62-1.95x | 3/1/2 | mixed |
| L | raw | 0.95 | 0/1 | 0/4 |  | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 3/5 |  | **CorrTrack** | 1.22x | 1.83 | 0.62-2.35x | 3/1/2 | mixed |
| N | differenced | 0.7 | 1/6 | 6/6 |  | 0/0 | 0/0 | 0/1 |  |  | 0/0 |  | **FilCorr** | 1.16x | 1.03 | 1.10-1.18x | 5/1/0 | clear |
| N | differenced | 0.8 | 1/5 | 5/6 |  | 0/0 | 0/0 | 4/6 |  |  | 0/0 |  | **FilCorr** | 0.99x | 1.10 | 0.88-1.13x | 1/4/1 | mixed |
| N | differenced | 0.9 | 0/1 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.57x | 1.13 | 1.31-1.80x | 6/0/0 | clear |
| N | differenced | 0.95 | 0/1 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 2.23x | 1.13 | 1.83-2.61x | 6/0/0 | clear |
| N | raw | 0.7 | 2/6 | 6/6 |  | 0/2 | 0/0 | 0/0 |  |  | 0/0 |  | **FilCorr** | 1.12x | 1.03 | 1.07-1.15x | 4/2/0 | clear |
| N | raw | 0.8 | 0/6 | 6/6 |  | 0/0 | 0/0 | 2/2 |  |  | 0/0 |  | **FilCorr** | 1.11x | 1.06 | 1.00-1.17x | 5/1/0 | clear |
| N | raw | 0.9 | 0/4 | 3/6 |  | 0/0 | 0/0 | 3/4 |  |  | 0/0 |  | **CorrTrack** | 0.90x | 1.80 | 0.44-1.63x | 3/0/3 | mixed |
| N | raw | 0.95 | 0/4 | 3/6 |  | 0/0 | 0/0 | 4/5 |  |  | 2/4 |  | **CorrTrack** | 1.28x | 1.64 | 0.68-2.11x | 3/1/2 | mixed |
| S | differenced | 0.7 | 0/6 | 6/6 | 0/0 | 0/0 | 0/0 | 0/2 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 1.14x | 1.02 | 1.11-1.16x | 6/0/0 | clear |
| S | differenced | 0.8 | 0/6 | 6/6 | 0/0 | 0/0 | 0/0 | 2/3 | 0/0 | 0/0 | 1/1 | 0/0 | **FilCorr** | 1.11x | 1.03 | 1.05-1.16x | 4/2/0 | clear |
| S | differenced | 0.9 | 1/5 | 5/6 | 0/0 | 0/0 | 0/0 | 3/4 | 0/0 | 0/0 | 2/4 | 0/0 | **FilCorr** | 1.07x | 1.10 | 0.89-1.15x | 4/1/1 | mixed |
| S | differenced | 0.95 | 0/5 | 5/5 | 0/0 | 0/0 | 0/0 | 2/3 | 0/0 | 0/0 | 2/4 | 0/0 | **FilCorr** | 1.09x | 1.06 | 1.01-1.18x | 2/3/0 | tied |
| S | raw | 0.7 | 2/6 | 6/6 | 0/0 | 0/2 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.12x | 1.01 | 1.10-1.14x | 6/0/0 | clear |
| S | raw | 0.8 | 1/6 | 6/6 | 0/0 | 0/1 | 0/0 | 0/2 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 1.13x | 1.02 | 1.11-1.15x | 6/0/0 | clear |
| S | raw | 0.9 | 2/6 | 6/6 | 0/0 | 0/2 | 0/0 | 1/3 | 0/0 | 0/0 | 1/2 | 0/0 | **FilCorr** | 1.10x | 1.06 | 1.01-1.16x | 4/2/0 | clear |
| S | raw | 0.95 | 0/5 | 5/6 | 0/0 | 0/0 | 0/0 | 2/2 | 0/0 | 0/0 | 1/3 | 1/2 | **FilCorr** | 1.04x | 1.14 | 0.84-1.16x | 3/2/1 | mixed |

## Only arms reaching recall 0.95

| table | space | T | STOMP | FilCorr | TSUBASA | BRAID | ThinBRAID | CorrTrack | ParCorr | CSZ | StatStream | CorrJoin | leader | margin | spread | range | ahead/tied/behind | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | differenced | 0.7 | 0/6 | 6/6 |  | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 0/0 |  | **FilCorr** | 1.03x | 1.11 | 0.92-1.16x | 2/4/0 | tied |
| L | differenced | 0.8 | 1/1 | 1/6 |  | 0/0 | 0/0 | 5/6 | 0/0 | 0/0 | 0/0 |  | **CorrTrack** | 1.32x | 1.42 | 0.67-1.75x | 5/0/1 | mixed |
| L | differenced | 0.9 | 0/1 | 0/6 |  | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/0 |  | **CorrTrack** | 2.09x | 1.21 | 1.47-2.48x | 6/0/0 | clear |
| L | differenced | 0.95 | 1/1 | 1/6 |  | 0/0 | 0/0 | 5/5 | 0/1 | 0/0 | 0/0 |  | **CorrTrack** | 2.87x | 1.12 | 2.53-3.19x | 5/0/0 | clear |
| L | raw | 0.7 | 1/6 | 6/6 |  | 0/1 | 0/0 | 1/3 | 0/0 | 0/0 | 0/0 |  | **FilCorr** | 1.11x | 1.05 | 1.01-1.16x | 5/1/0 | clear |
| L | raw | 0.8 | 0/3 | 3/6 |  | 0/0 | 0/0 | 3/3 | 0/0 | 0/0 | 0/0 |  | **FilCorr** | 0.93x | 1.27 | 0.65-1.16x | 3/0/3 | mixed |
| L | raw | 0.9 | 0/4 | 3/6 |  | 0/0 | 0/0 | 4/4 | 0/0 | 0/0 | 0/0 |  | **CorrTrack** | 1.26x | 1.73 | 0.62-2.21x | 3/1/2 | mixed |
| L | raw | 0.95 | 1/3 | 3/6 |  | 0/1 | 0/0 | 5/5 | 0/0 | 0/0 | 0/0 |  | **CorrTrack** | 1.54x | 1.71 | 0.92-2.81x | 3/2/0 | clear |
| N | differenced | 0.7 | 1/6 | 6/6 |  | 0/0 | 0/0 | 0/1 |  |  | 0/0 |  | **FilCorr** | 1.16x | 1.03 | 1.10-1.18x | 5/1/0 | clear |
| N | differenced | 0.8 | 1/5 | 5/6 |  | 0/0 | 0/0 | 4/6 |  |  | 0/0 |  | **FilCorr** | 0.99x | 1.10 | 0.88-1.13x | 1/4/1 | mixed |
| N | differenced | 0.9 | 0/1 | 1/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.51x | 1.22 | 1.04-1.80x | 5/1/0 | clear |
| N | differenced | 0.95 | 1/1 | 1/6 |  | 0/1 | 0/0 | 5/5 |  |  | 0/0 |  | **CorrTrack** | 2.21x | 1.14 | 1.83-2.61x | 5/0/0 | clear |
| N | raw | 0.7 | 2/6 | 6/6 |  | 0/2 | 0/0 | 0/0 |  |  | 0/0 |  | **FilCorr** | 1.12x | 1.03 | 1.07-1.15x | 4/2/0 | clear |
| N | raw | 0.8 | 0/6 | 6/6 |  | 0/0 | 0/0 | 2/2 |  |  | 0/0 |  | **FilCorr** | 1.11x | 1.06 | 1.00-1.17x | 5/1/0 | clear |
| N | raw | 0.9 | 0/4 | 3/6 |  | 0/0 | 0/0 | 3/4 |  |  | 0/0 |  | **CorrTrack** | 0.90x | 1.80 | 0.44-1.63x | 3/0/3 | mixed |
| N | raw | 0.95 | 1/4 | 4/6 |  | 0/1 | 0/0 | 3/4 |  |  | 0/0 |  | **FilCorr** | 0.84x | 1.49 | 0.47-1.17x | 3/1/2 | mixed |
| S | differenced | 0.7 | 0/6 | 6/6 | 0/0 | 0/0 | 0/0 | 0/2 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.14x | 1.02 | 1.11-1.16x | 6/0/0 | clear |
| S | differenced | 0.8 | 0/6 | 6/6 | 0/0 | 0/0 | 0/0 | 2/3 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.11x | 1.03 | 1.05-1.16x | 4/2/0 | clear |
| S | differenced | 0.9 | 1/6 | 6/6 | 0/0 | 0/0 | 0/0 | 2/3 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.11x | 1.04 | 1.05-1.15x | 5/1/0 | clear |
| S | differenced | 0.95 | 0/5 | 5/5 | 0/0 | 0/1 | 0/0 | 2/4 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.11x | 1.07 | 1.01-1.19x | 3/2/0 | clear |
| S | raw | 0.7 | 2/6 | 6/6 | 0/0 | 0/2 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.12x | 1.01 | 1.10-1.14x | 6/0/0 | clear |
| S | raw | 0.8 | 1/6 | 6/6 | 0/0 | 0/1 | 0/0 | 0/2 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.13x | 1.02 | 1.11-1.15x | 6/0/0 | clear |
| S | raw | 0.9 | 2/6 | 6/6 | 0/0 | 0/2 | 0/0 | 1/3 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.11x | 1.05 | 1.01-1.16x | 5/1/0 | clear |
| S | raw | 0.95 | 0/5 | 5/6 | 0/0 | 0/0 | 0/0 | 2/2 | 0/0 | 0/0 | 0/0 | 1/2 | **FilCorr** | 1.04x | 1.15 | 0.84-1.18x | 4/1/1 | mixed |

