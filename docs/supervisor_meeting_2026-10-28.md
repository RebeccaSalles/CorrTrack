# CorrTrack: questions from the last presentation, and what they led to (supervisor meeting, 2026-10-28)

## Instructions for the slide generator

Make a slide deck from the content below for a 30 to 40 minute meeting with my PhD supervisors, who know the
CorrTrack method well.
- One numbered section = one slide. Sections 1 to 5 are Part A (the questions from my last presentation), and 6 to 14
  are Part B (what the review of the m exponent led to). Add a title slide and one divider slide per part.
- Write in English. Keep the bullets short. Show formulas with proper math notation, and keep every table as a table.
- Use only the numbers given here. Do not round them differently, invent values, or add claims. Where a slide is
  too dense, split it rather than cut numbers.
- Keep the tone factual and cautious. The slides report measured results and their limits, not "state of the art"
  claims.

Context: CorrTrack finds all pairs of time series whose lagged Pearson correlation over a sliding window exceeds a
threshold T, at every step of a stream (m series, L lags). It sketches each window incrementally (64-dimensional
random projection, normalized after sketching), searches candidates with an LSH band index on the sketch sign bits
plus a Hamming prefilter and a dot-product gate, and validates every candidate exactly (precision 1).

# Part A: questions from the last presentation

## 1. "Why not a PH-tree as the candidate-search backend?"

- The query is a cosine cap (cos >= gamma, gamma about 0.8) in 64 dimensions. Unit sketches have coordinates of
  about +-0.125. Any box containing the cap has radius sqrt(2 - 2 gamma), about 0.63, i.e. about five coordinate
  spreads, so it contains nearly every point.
- So an exact tree degrades to a linear scan: O(m) per query, O(m^2) per step. Exact space-partitioning indexes lose
  to a scan beyond about 10 to 20 dimensions (Weber, Schek, Blott, VLDB 1998). LSH is the standard escape.
- Indexing a few dimensions and bounding the rest fails: random-projection sketches spread their energy evenly, so a
  k-dimension prefix carries about k/64 of it. We tested this twice (prefix bound plus Cauchy-Schwarz residual,
  early-exit dot product): no gain.
- The PH-tree's strength is cheap updates, but insert plus expire is about 2.6% of CorrTrack's run at m = 2000. The
  scan is about 80%.

## 2. "Why not embeddings instead of sketches?"

- The sketch already is an embedding: random, linear, data-independent. Those three properties are the method.
- Guarantee: the sketch inner product is an unbiased estimate of the window Pearson correlation with known variance.
  So the LSH bands and the gate are set for a recall target before seeing the data. A learned embedding has no such
  theorem, and its recall must be measured per dataset.
- Drift: a random projection cannot go stale. A learned embedding is fitted to a distribution that a nonstationary
  stream moves.
- Incrementality and lags: linearity gives O(basic window) updates, and one sketch per window is reused at every lag.
- The one variant worth an ablation is a learned linear basis (sliding PCA). It is incremental and concentrates
  energy, but needs refreshing under drift and is weak on differenced data. Not run yet.

## 3. "The usual complexity for this kind of problem is m log m; why is CorrTrack at m^1.64?"

- At fixed correlation density d, the output is about d m^2 / 2 pairs per window and lag, and a complete method
  must report each one. So m log m is impossible for complete thresholded discovery. Such bounds come from problems
  with linear output (top-k, kNN, low-dimensional range search).
- Even without the output, high-dimensional inner-product joins resist subquadratic time (Ahle, Pagh, Razenshteyn,
  Silvestri, PODS 2016, under the Orthogonal Vectors conjecture).
- Theory for finding correlations, offline and for a planted outlier pair: Valiant (FOCS 2012 / J. ACM 2015),
  n^(3 omega / 4) < n^1.8; Karppa, Kaski, Kohonen (SODA 2016), about n^1.58 for strong outliers. These are theory
  context, not competitors: no stream, no lags, no complete thresholded output.
- LSH predicts per-step search time O(m (m L)^rho + |output|), with rho = log2(1 / p) and p = 1 - arccos(T) / pi.
  rho falls as T rises: 1 + rho = 1.42 / 1.33 / 1.22 / 1.15 at T = 0.7 / 0.8 / 0.9 / 0.95.
- The 1.64 in the talk was measured at fixed density 0.01 (T = 0.9), where the output itself grows as m^2. It mixes
  linear, search and quadratic output terms, so it is not the search exponent.

## 4. "Why is nonstationarity a problem if it does not change the number of pairs?"

Controlled synthetic cell, m = 500, W = 60, step 6, 30 lags, T = 0.9, planted density 0.01, same seed in every regime:

| regime | measured density | candidates tested | recall | speedup over brute force | FilCorr speedup |
|---|---|---|---|---|---|
| AR(1), levels | 0.0100 | 13.5 M | 0.998 | 19.4x | 6.4x |
| AR(1), differences | 0.0100 | 13.6 M | 1.000 | 20.5x | 6.4x |
| random walk, levels | 0.0126 | 21.5 M | 0.962 | 13.6x | 6.4x |
| random walk, differences | 0.0100 | 13.5 M | 0.998 | 20.6x | 6.5x |

- It does change the count: random walks in levels share trends and create spurious correlations the generator never
  planted (+26% pairs). These are real Pearson correlations, which a complete method must report.
- The cost of filter-and-verify is set by the mass of pairs near the threshold: candidates +59%, validation +64%.
- Drift also breaks global normalization, fitted filters and energy-concentration methods. CorrTrack normalizes per
  window and uses a data-independent projection.
- Caveat: random-walk levels is CorrTrack's weakest regime too (13.6x vs about 20x). The claim is graceful
  degradation, not immunity.

## 5. What question 3 led to: measuring the search exponent properly

- To isolate the search, fix the output to grow linearly: each series has a fixed number of correlated partners
  (55), so the theory predicts a wall exponent of 1 + rho.
- Measured exponent of m with the current (pooled) index, synthetic, m up to 4000:

| T | textbook 1 + rho | measured, pooled | exact methods |
|---|---|---|---|
| 0.70 | 1.42 | 1.86 | 2.0 |
| 0.80 | 1.33 | 1.68 | 2.0 |
| 0.90 | 1.22 | 1.44 | 2.0 |
| 0.95 | 1.15 | 1.31 | 2.0 |

- Good news: the exponent falls with T as predicted, and the exact methods stay at 2.
- But it is consistently above textbook LSH. Why?
- Ablation at T = 0.8: drawing band bits from a larger pool of sign bits gives 64 / 128 / 256 bits ->
  exponent 1.66 / 1.51 / 1.44 (sizing predicts 1.56 / 1.45 / 1.39), recall >= 0.9998.
  The excess comes from the shared 64-bit pool.

# Part B: independent band bits

## 6. Diagnosis: why CorrTrack's exponent of m was above textbook LSH

- Band index on sign bits: a pair at correlation T agrees on a bit with probability p = 1 - arccos(T) / pi.
  With b bands of K bits, recall = 1 - (1 - p^K)^b; textbook cost ~ m^(1 + rho), rho = log2(1 / p).
- Pooled mode (current default): every band takes its K bits from the same 64 sketch-coordinate signs.
  Bands overlap, so they are not independent; the sizing compensates with many more bands.
- Intuition: for a given pair, the 64 coordinate signs are fixed (say 50 agree, 14 disagree). Every band draws from
  the same 50 out of 64, so a pair with a few more disagreeing coordinates than average fails in all bands at once.
- Consequence at m = 4000: 354 bands instead of 93 (T = 0.7), 132 instead of 46 (T = 0.8). The overlap correction
  grows with K, which grows with m, hence the excess exponent.

## 7. The change: one fresh random hyperplane per band bit

- Draw H (b K x 64, Gaussian, fixed seed) once, when the index is sized.
- Each step: P = X H^T for the m new sketches (one BLAS product); key bit = sign(h . s).
- Negative correlation: key of -s is the bitwise complement (same lookup as before).
- Still one sketching algorithm: with s = R x, sign(h . s) = sign((h R) . x). The key is a hash of the sketch
  (Charikar 2002 sign random projection), computed in the candidate search, not a second sketch.
- Unchanged: sketch, normalization, Hamming prefilter, dot-product gate, exact Pearson validation (precision 1),
  lags, negative correlations, recall target.
- Extra cost: b K x 64 multiply-adds per window, about 8% of the run when batched.

## 8. Setting the bucket occupancy without a tuning sweep

- Cost model of the E2LSH manual (Andoni and Indyk 2005, Sec. 3.4): query time = T_g (hash and probe bands) + T_c (collisions).
  cost ~ bands x (c + occupancy), bands ~ (m L / occupancy)^rho.
- Minimizing gives occupancy* = c rho / (1 - rho), independent of m and L.
- c = per-band cost in collision units, calibrated once per implementation:

| CPU | c |
|---|---|
| Intel Xeon Silver 4114 | 11.0 |
| Intel Xeon Gold 6240 | 10.8 |
| AMD EPYC 7513 | 10.3 |

- Shipped default c = 11, giving occupancy 8.0 / 5.4 / 3.2 / 2.0 at T = 0.7 / 0.8 / 0.9 / 0.95.
  User-settable; a library function `calibrate_lsh_cost_ratio` refits it on another machine.
- The optimum is flat: a factor 2 error in c costs at most about 10%. On real cells the rule falls at or inside the
  near-optimal range.

## 9. Tuning and selection rule

- Hyperopt now searches only gate offset x Hamming threshold (occupancy comes from the rule).
- Selection: among feasible configurations (recall lower bound >= target), keep those within 10% of the fewest
  validated candidates, then take the best recall.
- Why: tuning timings are noisy (10 to 17% between repeats). Validation work is a stable proxy for runtime. On real
  data, gate offset 0.2 instead of 0.0 sends a median 6.6x more candidates to validation.

## 10. Synthetic results (fixed degree, m up to 4000, mercantour3)

| T | textbook 1 + rho | pooled | independent | speedup over BF_vect at m = 4000: pooled | independent |
|---|---|---|---|---|---|
| 0.70 | 1.42 | 1.86 | 1.56 | 9.0x | 24.8x |
| 0.80 | 1.33 | 1.68 | 1.40 | 18.8x | 47.9x |
| 0.90 | 1.22 | 1.44 | 1.21 | 47.0x | 101.9x |
| 0.95 | 1.15 | 1.31 | 1.13 | 80.0x | 145.0x |

Recall of the independent mode: 0.998 or more in every synthetic cell.

## 11. Real data: 36 cells (6 datasets x T 0.7 / 0.8 / 0.9 x raw / differenced, m = 500, lagged)

Same machine for every arm (AMD EPYC 7513, 8 cores). Recall against the campaign's brute force; target 0.95.

| T | new vs pooled (median, range) | geometric mean | recall >= 0.95: new | pooled | min recall: new | pooled |
|---|---|---|---|---|---|---|
| 0.7 | 1.44x (1.14 to 1.80) | 1.42x | 12 / 12 | 12 / 12 | 0.955 | 0.953 |
| 0.8 | 1.45x (1.13 to 1.74) | 1.48x | 12 / 12 | 11 / 12 | 0.951 | 0.931 |
| 0.9 | 1.59x (1.23 to 1.82) | 1.51x | 12 / 12 | 11 / 12 | 0.975 | 0.930 |
| all | | 1.47x | 36 / 36 | 34 / 36 | | |

- Faster than the pooled mode in 36 / 36 cells.
- The 7 hard cells (no feasible pooled configuration on the tuning sample): 1.41x faster, recall >= 0.95 in 7 / 7
  (pooled 5 / 7).

## 12. Real data: against FilCorr and BF_incr (same node)

Wins / ties / losses for the new mode, tie = within 10%:

| T | vs FilCorr | vs BF_incr | geometric mean vs FilCorr | campaign (pooled), faster than FilCorr |
|---|---|---|---|---|
| 0.7 | 8 / 1 / 3 | 8 / 1 / 3 | 1.05x | 6 of 12 |
| 0.8 | 9 / 0 / 3 | 9 / 0 / 3 | 1.61x | 9 of 12 |
| 0.9 | 10 / 1 / 1 | 10 / 2 / 0 | 2.95x | 9 of 12 |

- The remaining losses are on the raw acwi, sp500 and streamflow series at T <= 0.8, plus raw acwi at T = 0.9 (0.88x).
  These dense cells have many correlated pairs, so exact validation and output dominate the run.
- At T = 0.7, CorrTrack goes from beating FilCorr in half the cells to 8 of 12.

## 13. Limits

- Real cells are at m = 500 only. Larger m is covered by the synthetic sweep.
- 1 + rho is an idealized reference: it assumes unrelated pairs sit at correlation 0 and a constant cost per probe and
  per collision. In the synthetic data, unrelated sketches have a cosine spread of 0.21, which alone lifts the
  predicted exponent by about 0.1. Fitted exponents also depend on the m range (T = 0.7: 1.47 up to m = 2000, 1.56 over
  125 to 4000, 1.72 from 500 to 4000). The robust result is the pooled vs independent comparison, measured the
  same way on the same machine.
- Recall is measured, not guaranteed. The tuning checks a lower bound on a sample, as in the pooled mode.
- c fitted on the real m = 500 cells is about 15 (higher at small m). The rule still lands in the flat optimum.
- The new real-data runs are on one AMD node. The paper campaign ran on mercantour3 (Intel), so ratios are compared
  only within a node. The new mode, FilCorr and BF_incr ran in one job on the same cores. The pooled mode ran in a
  separate job on the same node, which was shared, so the new vs pooled ratios carry more timing noise.

## 14. Proposal

- Make the independent mode the default, with occupancy from the rule (c = 11) and the new selection rule.
  Keep the pooled mode as an option.
- Estimated time to update the paper results: about one week. That covers 1 day to merge and test, 2 to 3 days of
  cluster campaigns (hyperopt plus N-way on all cells), 1 day to rebuild the tables and figures, and 1 to 2 days of text.
