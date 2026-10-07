# Evidence for the headline claim (2026-10-05)

Claim: *the first subquadratic operator for complete, online, lagged, signed, thresholded Pearson
correlation discovery that stays robust under nonstationarity.*

Figure: `docs/figures/headline_claim.png` (and `_authors.png`, the variant without the capabilities
we added to other methods). Builder: `abaca/headline_figure.py`. Four of the clauses can be
measured and each has a panel; lagged and signed are capability facts and stay in the capability
table, where their provenance is recorded per method.

## What each panel shows, and what it does not

**A. Subquadratic.** Wall clock against m on log axes, m from 125 to 2000 at fixed L, threshold and
density, with the exponent fitted by least squares on the logs: BF_vect 2.05, CorrTrack-LSH 1.64,
CorrTrack-Ham 1.74, FilCorr 1.84, BF_incr 1.85. The same fit on the other three regimes gives
CorrTrack-LSH 1.62 to 1.64 every time, so the exponent is not an artefact of one process or space.

Two honest limits, both of which a reviewer will raise. First, an exponent fitted over 125 to 2000
is a statement about the measured range, not an asymptotic proof; the theoretical argument belongs
in the method section and this panel is its evidence, not its substitute. Second, **FilCorr also
measures below 2** (1.84), so the word "subquadratic" alone does not separate CorrTrack from every
competitor empirically. What separates it is the conjunction: no other method in the study is
subquadratic *and* complete *and* lagged *and* signed *and* exact on the threshold. The claim should
be read as one sentence, not as a list of independent superlatives, and the figure is laid out that
way deliberately.

**B. Complete and thresholded.** The distribution of recall over the 179 cells, against the tuned
target of 0.95, with precision 1.000 in every cell for both backends. Precision is the part that
makes "thresholded Pearson" exact rather than approximate: every pair CorrTrack reports is above the
threshold, because the candidate set is validated exactly. Recall is where the approximation lives:
the hamming backend never falls below 0.954, the LSH backend reaches 0.868 at worst, in the
smartmeter cells. Completeness in the sense of the claim is about the search space, every pair,
every lag, every window, rather than recall equal to 1, and the panel states the recall honestly
rather than rounding it away.

**C. Online.** The cost of one window against that run's own median window, so the panel says
whether the per-window cost drifts as the stream advances rather than repeating panel A. The
interquartile band sits within a few percent of the median at every m, which is what an operator
over a stream has to deliver. The tail is disclosed in the same panel: CorrTrack's slowest window is
4 to 10 times its median, against 1.3 to 1.5 for brute force, and 4 to 8% of windows fall beyond the
whisker. Those are the windows where the index is maintained rather than merely queried. If a
reviewer asks for a latency guarantee rather than a throughput claim, this tail is the thing to
discuss, and the honest answer today is that the median and the quartiles are flat while the
maximum is not.

**C, continued: what the slow windows are.** A probe run (`CORRTRACK_STEP_TIMES`, job 3170213) kept
the per-window series instead of its quantiles, and the answer is neither warm-up nor index expiry:
the slow windows are the amortised doubling of the array that holds the correlated pairs. The cell
keeps 13,524 rows per window, so a doubling of that array falls at windows 78, 155, 310 and 620, and
the measured spikes are at 86, 163, 316 and 624, with magnitudes 2.0x, 2.7x, 4.9x and 7.9x of the
median, doubling each time as a copy of a doubling array must. The first five windows of the stream
are the fastest in the whole run (0.17x, 0.05x, 0.04x, 0.04x, 0.05x of the median), since the index
is still nearly empty, which rules out warm-up directly.

That is a fixable cost rather than an inherent one: preallocating the accumulator from the expected
density, or chunking it so growth never copies, would flatten the tail without touching the method.
Worth stating in the paper as an implementation note, because it turns a latency objection into a
known engineering item.

**D. Robust under nonstationarity.** The same cell under four regimes: AR(1) and a random walk, each
in levels and in first differences. A random walk in levels is nonstationary and carries spurious
correlation the generator never planted, which is the case that breaks filters built on energy
concentration. CorrTrack returns 19.4x, 20.5x, 13.6x and 20.6x across the four; the nonstationary
levels cell is its weakest and it still leads every competitor there, where FilCorr sits near 6.3x
in all four. Robustness here means the speedup does not collapse when the series stop being
stationary, not that it is unchanged.

## The numbers behind the panels

| regime | BF_vect s | CorrTrack-LSH | CorrTrack-Ham | FilCorr | BF_incr |
|---|---|---|---|---|---|
| AR(1), levels | 294 | 19.38x | 18.07x | 6.44x | 6.03x |
| AR(1), differences | 296 | 20.47x | 18.26x | 6.42x | 5.85x |
| random walk, levels | 298 | 13.59x | 11.52x | 6.28x | 5.92x |
| random walk, differences | 301 | 20.64x | 18.66x | 6.19x | 5.75x |

| arm | exponent in m, AR(1) levels | AR(1) differences | random walk levels | random walk differences |
|---|---|---|---|---|
| BF_vect | 2.05 | 2.05 | 2.05 | 2.05 |
| CorrTrack-LSH | 1.64 | 1.63 | 1.62 | 1.64 |
| CorrTrack-Ham | 1.74 | 1.75 | 1.72 | 1.74 |
| FilCorr | 1.84 | 1.83 | 1.82 | 1.82 |
| BF_incr | 1.85 | 1.85 | 1.84 | 1.84 |
