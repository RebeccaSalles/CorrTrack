# m=500 tables: rankings, margins and paired tests

`[f/s]` = datasets of that row where the arm is the fastest / among the two fastest, out of six. CorrTrack is one arm here: its value on a dataset is the better of its two tuned backends. The winner of a row is the arm with the most firsts, ties broken by the median speedup. It is then compared with the best other arm on each dataset separately: **margin** is the geometric mean of the six ratios, **spread** their geometric standard deviation (1.00 would mean the same ratio on every dataset), and **range** their smallest and largest value. A margin of 1.30 with a spread of 1.05 is a uniform win; the same margin with a spread of 1.60 means the win rests on one or two datasets.

Speedups within 10% of each other count as tied, so `[f/s]` can exceed one arm per place and a row can have several arms tied for first. **ahead/tied/behind** counts the datasets where the leader is more than 10% above the best other arm, within 10% of it, or more than 10% below. The verdict follows: *clear* when the leader is ahead on a majority of datasets and never behind, *tied* when every dataset is inside the band, *mixed* when it wins some and loses others.

## Every arm that ran

| table | space | T | BF_incr | FilCorr | TSUBASA | BRAID | CorrTrack | ParCorr | CSZ | StatStream | CorrJoin | leader | margin | spread | range | ahead/tied/behind | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | differenced | 0.8 | 0/1 | 0/4 |  | 0/0 | 6/6 |  |  | 0/4 |  | **CorrTrack** | 1.78x | 1.16 | 1.57-2.39x | 6/0/0 | clear |
| L | differenced | 0.85 | 0/1 | 0/2 |  | 0/0 | 6/6 |  |  | 0/4 |  | **CorrTrack** | 2.22x | 1.16 | 1.99-2.98x | 6/0/0 | clear |
| L | differenced | 0.9 | 0/1 | 0/2 |  | 0/0 | 6/6 |  |  | 0/6 |  | **CorrTrack** | 2.88x | 1.18 | 2.44-3.90x | 6/0/0 | clear |
| L | differenced | 0.95 | 0/1 | 0/2 |  | 0/0 | 6/6 |  |  | 0/5 |  | **CorrTrack** | 4.09x | 1.27 | 3.09-5.55x | 6/0/0 | clear |
| L | raw | 0.8 | 0/3 | 3/6 |  | 0/0 | 3/4 |  |  | 0/2 |  | **CorrTrack** | 0.99x | 1.83 | 0.50-1.86x | 3/0/3 | mixed |
| L | raw | 0.85 | 0/3 | 3/6 |  | 0/0 | 4/4 |  |  | 0/2 |  | **CorrTrack** | 1.17x | 1.95 | 0.55-2.39x | 3/1/2 | mixed |
| L | raw | 0.9 | 0/2 | 2/5 |  | 0/0 | 4/5 |  |  | 0/3 |  | **CorrTrack** | 1.61x | 1.92 | 0.77-3.18x | 4/0/2 | mixed |
| L | raw | 0.95 | 0/0 | 0/2 |  | 0/0 | 4/6 |  |  | 2/5 |  | **CorrTrack** | 1.91x | 2.21 | 0.78-4.98x | 4/0/2 | mixed |
| N | differenced | 0.8 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.39x | 1.09 | 1.20-1.54x | 6/0/0 | clear |
| N | differenced | 0.85 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.80x | 1.12 | 1.58-2.17x | 6/0/0 | clear |
| N | differenced | 0.9 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 2.52x | 1.14 | 2.14-3.15x | 6/0/0 | clear |
| N | differenced | 0.95 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 3.86x | 1.20 | 3.03-4.95x | 6/0/0 | clear |
| N | raw | 0.8 | 4/6 |  | 0/0 | 0/2 | 3/5 |  |  | 0/1 |  | **BF_incr** | 1.28x | 1.65 | 0.74-2.24x | 3/1/2 | mixed |
| N | raw | 0.85 | 3/6 |  | 0/0 | 0/3 | 4/6 |  |  | 0/1 |  | **CorrTrack** | 0.94x | 1.92 | 0.44-2.05x | 3/1/2 | mixed |
| N | raw | 0.9 | 3/6 |  | 0/0 | 0/0 | 4/6 |  |  | 0/1 |  | **CorrTrack** | 1.32x | 1.90 | 0.64-2.87x | 3/1/2 | mixed |
| N | raw | 0.95 | 0/4 |  | 0/0 | 0/0 | 5/6 |  |  | 1/3 |  | **CorrTrack** | 1.97x | 1.91 | 0.83-4.44x | 5/0/1 | mixed |
| S | differenced | 0.8 | 0/2 | 2/5 | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.18x | 1.41 | 0.79-1.92x | 4/0/2 | mixed |
| S | differenced | 0.85 | 0/2 | 2/5 | 0/0 | 0/0 | 5/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.30x | 1.36 | 0.87-2.09x | 4/1/1 | mixed |
| S | differenced | 0.9 | 0/2 | 1/4 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.44x | 1.36 | 1.03-2.40x | 5/1/0 | clear |
| S | differenced | 0.95 | 0/0 | 0/2 | 0/0 | 0/0 | 5/5 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.45x | 1.23 | 1.21-2.00x | 5/0/0 | clear |
| S | raw | 0.8 | 1/3 | 3/6 | 0/0 | 0/1 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 0.91x | 1.25 | 0.70-1.12x | 3/0/3 | mixed |
| S | raw | 0.85 | 2/3 | 3/6 | 0/0 | 0/2 | 3/4 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.01x | 1.55 | 0.66-1.71x | 3/0/3 | mixed |
| S | raw | 0.9 | 1/3 | 3/5 | 0/0 | 0/1 | 3/5 | 0/0 | 0/0 | 0/3 | 0/0 | **CorrTrack** | 1.15x | 1.54 | 0.74-1.96x | 3/0/3 | mixed |
| S | raw | 0.95 | 0/2 | 1/4 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/3 | 0/2 | **CorrTrack** | 1.31x | 1.29 | 1.00-1.92x | 5/1/0 | clear |

## Only arms reaching recall 0.95

| table | space | T | BF_incr | FilCorr | TSUBASA | BRAID | CorrTrack | ParCorr | CSZ | StatStream | CorrJoin | leader | margin | spread | range | ahead/tied/behind | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | differenced | 0.8 | 0/1 | 0/4 |  | 0/0 | 6/6 |  |  | 0/4 |  | **CorrTrack** | 1.63x | 1.09 | 1.39-1.79x | 6/0/0 | clear |
| L | differenced | 0.85 | 0/1 | 0/2 |  | 0/0 | 6/6 |  |  | 0/4 |  | **CorrTrack** | 2.02x | 1.11 | 1.67-2.28x | 6/0/0 | clear |
| L | differenced | 0.9 | 0/1 | 0/2 |  | 0/0 | 6/6 |  |  | 0/6 |  | **CorrTrack** | 2.57x | 1.15 | 2.00-2.93x | 6/0/0 | clear |
| L | differenced | 0.95 | 0/1 | 0/2 |  | 0/0 | 6/6 |  |  | 0/4 |  | **CorrTrack** | 3.62x | 1.27 | 2.65-5.27x | 6/0/0 | clear |
| L | raw | 0.8 | 0/3 | 3/6 |  | 0/0 | 3/4 |  |  | 0/2 |  | **CorrTrack** | 0.99x | 1.83 | 0.50-1.86x | 3/0/3 | mixed |
| L | raw | 0.85 | 0/3 | 3/6 |  | 0/0 | 4/4 |  |  | 0/2 |  | **CorrTrack** | 1.17x | 1.95 | 0.55-2.39x | 3/1/2 | mixed |
| L | raw | 0.9 | 0/2 | 2/5 |  | 0/0 | 4/5 |  |  | 0/3 |  | **CorrTrack** | 1.61x | 1.92 | 0.77-3.18x | 4/0/2 | mixed |
| L | raw | 0.95 | 0/0 | 0/2 |  | 0/0 | 4/6 |  |  | 2/4 |  | **CorrTrack** | 1.67x | 1.95 | 0.78-4.44x | 4/0/2 | mixed |
| N | differenced | 0.8 | 1/6 |  | 0/1 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.29x | 1.21 | 0.91-1.54x | 5/1/0 | clear |
| N | differenced | 0.85 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 1.62x | 1.20 | 1.15-1.86x | 6/0/0 | clear |
| N | differenced | 0.9 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 2.21x | 1.25 | 1.44-2.65x | 6/0/0 | clear |
| N | differenced | 0.95 | 0/6 |  | 0/0 | 0/0 | 6/6 |  |  | 0/0 |  | **CorrTrack** | 3.35x | 1.29 | 2.15-4.56x | 6/0/0 | clear |
| N | raw | 0.8 | 4/6 |  | 0/0 | 0/2 | 3/5 |  |  | 0/1 |  | **BF_incr** | 1.28x | 1.65 | 0.74-2.24x | 3/1/2 | mixed |
| N | raw | 0.85 | 3/6 |  | 0/0 | 0/3 | 4/6 |  |  | 0/1 |  | **CorrTrack** | 0.94x | 1.92 | 0.44-2.05x | 3/1/2 | mixed |
| N | raw | 0.9 | 3/6 |  | 0/0 | 0/0 | 4/6 |  |  | 0/1 |  | **CorrTrack** | 1.32x | 1.90 | 0.64-2.87x | 3/1/2 | mixed |
| N | raw | 0.95 | 0/4 |  | 0/0 | 0/0 | 5/6 |  |  | 1/3 |  | **CorrTrack** | 1.68x | 1.67 | 0.83-3.39x | 5/0/1 | mixed |
| S | differenced | 0.8 | 0/2 | 2/5 | 0/0 | 0/0 | 4/6 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.17x | 1.39 | 0.79-1.78x | 4/0/2 | mixed |
| S | differenced | 0.85 | 0/2 | 2/5 | 0/0 | 0/0 | 5/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.29x | 1.35 | 0.87-2.03x | 4/1/1 | mixed |
| S | differenced | 0.9 | 0/2 | 1/4 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/4 | 0/0 | **CorrTrack** | 1.43x | 1.34 | 1.03-2.28x | 5/1/0 | clear |
| S | differenced | 0.95 | 0/0 | 0/2 | 0/0 | 0/0 | 5/5 | 0/0 | 0/0 | 0/3 | 0/0 | **CorrTrack** | 1.48x | 1.29 | 1.21-2.22x | 5/0/0 | clear |
| S | raw | 0.8 | 1/3 | 3/6 | 0/0 | 0/1 | 3/3 | 0/0 | 0/0 | 0/1 | 0/0 | **FilCorr** | 0.91x | 1.25 | 0.70-1.12x | 3/0/3 | mixed |
| S | raw | 0.85 | 2/3 | 3/6 | 0/0 | 0/2 | 3/4 | 0/0 | 0/0 | 0/1 | 0/0 | **CorrTrack** | 1.01x | 1.55 | 0.66-1.71x | 3/0/3 | mixed |
| S | raw | 0.9 | 1/3 | 3/5 | 0/0 | 0/1 | 3/5 | 0/0 | 0/0 | 0/2 | 0/0 | **CorrTrack** | 1.15x | 1.54 | 0.74-1.96x | 3/0/3 | mixed |
| S | raw | 0.95 | 0/2 | 1/4 | 0/0 | 0/0 | 6/6 | 0/0 | 0/0 | 0/2 | 0/2 | **CorrTrack** | 1.33x | 1.32 | 1.00-1.92x | 5/1/0 | clear |

