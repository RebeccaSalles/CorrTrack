# m=500 tables: rankings, margins and paired tests

`[f/s]` = datasets of that row where the arm is the fastest / among the two fastest, out of six. CorrTrack is one arm here: its value on a dataset is the better of its two tuned backends. The winner of a row is the arm with the most firsts, ties broken by the median speedup. It is then compared with the best other arm on each dataset separately: **margin** is the geometric mean of the six ratios, **spread** their geometric standard deviation (1.00 would mean the same ratio on every dataset), and **range** their smallest and largest value. A margin of 1.30 with a spread of 1.05 is a uniform win; the same margin with a spread of 1.60 means the win rests on one or two datasets.

Speedups within 10% of each other count as tied, so `[f/s]` can exceed one arm per place and a row can have several arms tied for first. **ahead/tied/behind** counts the datasets where the leader is more than 10% above the best other arm, within 10% of it, or more than 10% below. The verdict follows: *clear* when the leader is ahead on a majority of datasets and never behind, *tied* when every dataset is inside the band, *mixed* when it wins some and loses others.

## Every arm that ran

| table | space | T | BF_incr | FilCorr | TSUBASA | BRAID | ThinBRAID | CorrTrack | ParCorr | CSZ | StatStream | CorrJoin | leader | margin | spread | range | ahead/tied/behind | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | differenced | 0.7 | 0/3 | 2/6 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 1/2 | 0/0 | **CorrTrack** | 1.13x | 1.19 | 0.97-1.54x | 3/3/0 | tied |
| L | differenced | 0.8 | 0/1 | 0/4 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.78x | 1.16 | 1.57-2.39x | 6/0/0 | clear |
| L | differenced | 0.85 | 0/1 | 0/2 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 2.22x | 1.16 | 1.99-2.98x | 6/0/0 | clear |
| L | differenced | 0.9 | 0/1 | 0/2 | 0/0 | 0/0 | 0/0 | 6/6 | 0/1 | 0/0 | 0/6 | 0/4 | **CorrTrack** | 2.88x | 1.18 | 2.44-3.90x | 6/0/0 | clear |
| L | differenced | 0.95 | 0/1 | 0/1 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/1 | 0/6 | **CorrTrack** | 3.14x | 1.37 | 2.31-5.55x | 6/0/0 | clear |
| L | raw | 0.7 | 1/4 | 4/6 | 0/0 | 0/1 | 0/0 | 2/4 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.03x | 1.16 | 0.82-1.17x | 4/0/2 | mixed |
| L | raw | 0.8 | 0/3 | 3/5 | 0/0 | 0/0 | 0/0 | 3/4 | 0/0 | 0/0 | 0/1 | 0/1 | **CorrTrack** | 0.96x | 1.81 | 0.50-1.86x | 3/0/3 | mixed |
| L | raw | 0.85 | 0/3 | 3/5 | 0/0 | 0/0 | 0/0 | 4/4 | 0/0 | 0/0 | 0/1 | 1/4 | **CorrTrack** | 1.06x | 1.84 | 0.55-2.29x | 2/2/2 | mixed |
| L | raw | 0.9 | 0/2 | 2/3 | 0/0 | 0/0 | 0/0 | 4/5 | 0/0 | 0/0 | 0/0 | 3/6 | **CorrTrack** | 1.26x | 1.65 | 0.75-2.28x | 3/1/2 | mixed |
| L | raw | 0.95 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 4/4 | 0/0 | 0/0 | 0/3 | 3/6 | **CorrTrack** | 1.15x | 1.72 | 0.65-2.27x | 3/1/2 | mixed |
| N | differenced | 0.7 | 1/6 | 6/6 | 0/0 | 0/0 | 0/0 | 0/1 |  |  | 0/0 |  | **FilCorr** | 1.15x | 1.02 | 1.11-1.16x | 6/0/0 | clear |
| N | differenced | 0.8 | 0/2 | 1/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.21x | 1.09 | 1.03-1.33x | 5/1/0 | clear |
| N | differenced | 0.85 | 0/1 | 0/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.56x | 1.14 | 1.37-1.96x | 6/0/0 | clear |
| N | differenced | 0.9 | 0/1 | 0/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 2.19x | 1.17 | 1.85-2.89x | 6/0/0 | clear |
| N | differenced | 0.95 | 0/1 | 0/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 3.36x | 1.21 | 2.62-4.48x | 6/0/0 | clear |
| N | raw | 0.7 | 2/6 | 6/6 | 0/2 | 0/2 | 0/0 | 0/0 |  |  | 0/0 |  | **FilCorr** | 1.11x | 1.04 | 1.05-1.16x | 4/2/0 | clear |
| N | raw | 0.8 | 0/4 | 4/6 | 0/0 | 0/0 | 0/0 | 2/3 |  |  | 0/0 |  | **FilCorr** | 1.03x | 1.16 | 0.85-1.16x | 4/0/2 | mixed |
| N | raw | 0.85 | 0/4 | 4/6 | 0/0 | 0/0 | 0/0 | 3/4 |  |  | 0/0 |  | **FilCorr** | 0.91x | 1.37 | 0.54-1.14x | 3/1/2 | mixed |
| N | raw | 0.9 | 0/3 | 3/6 | 0/0 | 0/0 | 0/0 | 4/4 |  |  | 0/0 |  | **CorrTrack** | 1.15x | 1.91 | 0.56-2.58x | 3/1/2 | mixed |
| N | raw | 0.95 | 0/2 | 2/6 | 0/0 | 0/0 | 0/0 | 5/6 |  |  | 1/3 |  | **CorrTrack** | 1.84x | 1.84 | 0.83-4.06x | 5/0/1 | mixed |
| S | differenced | 0.7 | 0/5 | 4/5 | 0/0 | 0/0 | 0/0 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 0.93x | 1.27 | 0.68-1.16x | 3/1/2 | mixed |
| S | differenced | 0.8 | 0/2 | 2/5 | 0/0 | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.18x | 1.41 | 0.79-1.92x | 4/0/2 | mixed |
| S | differenced | 0.85 | 0/2 | 2/5 | 0/0 | 0/0 | 0/0 | 5/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.30x | 1.36 | 0.87-2.09x | 4/1/1 | mixed |
| S | differenced | 0.9 | 0/2 | 1/4 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.44x | 1.36 | 1.03-2.40x | 5/1/0 | clear |
| S | differenced | 0.95 | 0/0 | 0/2 | 0/0 | 0/0 | 0/0 | 5/5 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.45x | 1.23 | 1.21-2.00x | 5/0/0 | clear |
| S | raw | 0.7 | 2/6 | 6/6 | 0/0 | 0/2 | 0/0 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 1.03x | 1.09 | 0.92-1.14x | 2/4/0 | tied |
| S | raw | 0.8 | 1/3 | 3/6 | 0/0 | 0/1 | 0/0 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 0.91x | 1.25 | 0.70-1.12x | 3/0/3 | mixed |
| S | raw | 0.85 | 2/3 | 3/6 | 0/0 | 0/2 | 0/0 | 3/4 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.01x | 1.55 | 0.66-1.71x | 3/0/3 | mixed |
| S | raw | 0.9 | 1/3 | 3/5 | 0/0 | 0/1 | 0/0 | 3/5 | 0/0 | 0/0 | 0/3 | 0/0 | **CorrTrack** | 1.15x | 1.54 | 0.74-1.96x | 3/0/3 | mixed |
| S | raw | 0.95 | 0/2 | 1/4 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/3 | 0/2 | **CorrTrack** | 1.31x | 1.29 | 1.00-1.92x | 5/1/0 | clear |

## Only arms reaching recall 0.95

| table | space | T | BF_incr | FilCorr | TSUBASA | BRAID | ThinBRAID | CorrTrack | ParCorr | CSZ | StatStream | CorrJoin | leader | margin | spread | range | ahead/tied/behind | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | differenced | 0.7 | 0/3 | 2/6 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 1/2 | 0/0 | **CorrTrack** | 1.13x | 1.19 | 0.97-1.54x | 3/3/0 | tied |
| L | differenced | 0.8 | 0/1 | 0/4 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.63x | 1.09 | 1.39-1.79x | 6/0/0 | clear |
| L | differenced | 0.85 | 0/1 | 0/2 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 2.02x | 1.11 | 1.67-2.28x | 6/0/0 | clear |
| L | differenced | 0.9 | 0/1 | 0/2 | 0/0 | 0/0 | 0/0 | 6/6 | 0/1 | 0/0 | 0/6 | 0/4 | **CorrTrack** | 2.57x | 1.15 | 2.00-2.93x | 6/0/0 | clear |
| L | differenced | 0.95 | 0/1 | 0/1 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/1 | 0/6 | **CorrTrack** | 2.77x | 1.17 | 2.31-3.46x | 6/0/0 | clear |
| L | raw | 0.7 | 1/4 | 4/6 | 0/0 | 0/1 | 0/0 | 2/4 | 0/0 | 0/0 | 0/0 | 0/0 | **FilCorr** | 1.03x | 1.16 | 0.82-1.17x | 4/0/2 | mixed |
| L | raw | 0.8 | 0/3 | 3/5 | 0/0 | 0/0 | 0/0 | 3/4 | 0/0 | 0/0 | 0/1 | 0/1 | **CorrTrack** | 0.96x | 1.81 | 0.50-1.86x | 3/0/3 | mixed |
| L | raw | 0.85 | 0/3 | 3/5 | 0/0 | 0/0 | 0/0 | 4/4 | 0/0 | 0/0 | 0/1 | 1/4 | **CorrTrack** | 1.06x | 1.84 | 0.55-2.29x | 2/2/2 | mixed |
| L | raw | 0.9 | 0/2 | 2/3 | 0/0 | 0/0 | 0/0 | 4/5 | 0/0 | 0/0 | 0/0 | 3/6 | **CorrTrack** | 1.26x | 1.65 | 0.75-2.28x | 3/1/2 | mixed |
| L | raw | 0.95 | 0/0 | 0/1 | 0/0 | 0/0 | 0/0 | 4/4 | 0/0 | 0/0 | 0/3 | 4/6 | **CorrJoin** | 0.93x | 1.42 | 0.47-1.20x | 2/2/2 | mixed |
| N | differenced | 0.7 | 1/6 | 6/6 | 0/0 | 0/0 | 0/0 | 0/1 |  |  | 0/0 |  | **FilCorr** | 1.15x | 1.02 | 1.11-1.16x | 6/0/0 | clear |
| N | differenced | 0.8 | 1/2 | 2/6 | 0/0 | 0/0 | 0/0 | 5/6 |  |  | 0/0 |  | **CorrTrack** | 1.13x | 1.19 | 0.83-1.33x | 4/1/1 | mixed |
| N | differenced | 0.85 | 0/1 | 1/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.40x | 1.18 | 1.04-1.62x | 5/1/0 | clear |
| N | differenced | 0.9 | 0/1 | 0/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.93x | 1.22 | 1.32-2.30x | 6/0/0 | clear |
| N | differenced | 0.95 | 0/1 | 0/6 | 0/0 | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 2.93x | 1.27 | 1.94-3.95x | 6/0/0 | clear |
| N | raw | 0.7 | 2/6 | 6/6 | 0/2 | 0/2 | 0/0 | 0/0 |  |  | 0/0 |  | **FilCorr** | 1.11x | 1.04 | 1.05-1.16x | 4/2/0 | clear |
| N | raw | 0.8 | 0/4 | 4/6 | 0/0 | 0/0 | 0/0 | 2/3 |  |  | 0/0 |  | **FilCorr** | 1.03x | 1.16 | 0.85-1.16x | 4/0/2 | mixed |
| N | raw | 0.85 | 0/4 | 4/6 | 0/0 | 0/0 | 0/0 | 3/4 |  |  | 0/0 |  | **FilCorr** | 0.91x | 1.37 | 0.54-1.14x | 3/1/2 | mixed |
| N | raw | 0.9 | 0/3 | 3/6 | 0/0 | 0/0 | 0/0 | 4/4 |  |  | 0/0 |  | **CorrTrack** | 1.15x | 1.91 | 0.56-2.58x | 3/1/2 | mixed |
| N | raw | 0.95 | 0/2 | 2/6 | 0/0 | 0/0 | 0/0 | 5/6 |  |  | 1/3 |  | **CorrTrack** | 1.57x | 1.60 | 0.83-2.89x | 5/0/1 | mixed |
| S | differenced | 0.7 | 0/5 | 4/5 | 0/0 | 0/0 | 0/0 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 0.93x | 1.27 | 0.68-1.16x | 3/1/2 | mixed |
| S | differenced | 0.8 | 0/2 | 2/5 | 0/0 | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.17x | 1.39 | 0.79-1.78x | 4/0/2 | mixed |
| S | differenced | 0.85 | 0/2 | 2/5 | 0/0 | 0/0 | 0/0 | 5/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.29x | 1.35 | 0.87-2.03x | 4/1/1 | mixed |
| S | differenced | 0.9 | 0/2 | 1/4 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.43x | 1.34 | 1.03-2.28x | 5/1/0 | clear |
| S | differenced | 0.95 | 0/0 | 0/2 | 0/0 | 0/0 | 0/0 | 5/5 | 0/0 | 0/0 | 0/3 | 0/0 | **CorrTrack** | 1.48x | 1.29 | 1.21-2.22x | 5/0/0 | clear |
| S | raw | 0.7 | 2/6 | 6/6 | 0/0 | 0/2 | 0/0 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 1.03x | 1.09 | 0.92-1.14x | 2/4/0 | tied |
| S | raw | 0.8 | 1/3 | 3/6 | 0/0 | 0/1 | 0/0 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 0.91x | 1.25 | 0.70-1.12x | 3/0/3 | mixed |
| S | raw | 0.85 | 2/3 | 3/6 | 0/0 | 0/2 | 0/0 | 3/4 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.01x | 1.55 | 0.66-1.71x | 3/0/3 | mixed |
| S | raw | 0.9 | 1/3 | 3/5 | 0/0 | 0/1 | 0/0 | 3/5 | 0/0 | 0/0 | 0/2 | 0/0 | **CorrTrack** | 1.15x | 1.54 | 0.74-1.96x | 3/0/3 | mixed |
| S | raw | 0.95 | 0/2 | 1/4 | 0/0 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/2 | 0/2 | **CorrTrack** | 1.33x | 1.32 | 1.00-1.92x | 5/1/0 | clear |

