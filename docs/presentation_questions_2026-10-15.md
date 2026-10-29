# Questions raised at the CorrTrack presentation, and the answers (2026-10-07 to 2026-10-15)

Material for the paper's discussion and related-work sections. Every number below comes from the cited
doc or result file; regenerate it from there before it goes into the paper (paper numbers come from macros).

## 1. "Why not a PH-tree as the candidate-search backend?"

The PH-tree (Zaeschke, Zimmerli, Norrie, SIGMOD 2014) is a bit-prefix-sharing multidimensional index with very
cheap inserts and deletes and fast box and nearest-neighbour queries in low or moderate dimension.

- The query is a cosine cap (cos >= gamma, gamma about 0.8) in 64 dimensions. Unit sketches have coordinates of
  about +-0.125, and any box or ball containing the cap has radius sqrt(2 - 2 gamma), about 0.63, about five
  coordinate spreads. It contains nearly every point, so an exact tree degrades to a linear scan: O(m) per query,
  O(m^2) per step. Exact space-partitioning indexes lose to a scan beyond roughly 10 to 20 dimensions
  (Weber, Schek, Blott, VLDB 1998). LSH is the standard escape, and CorrTrack-LSH measures m^1.64.
- Indexing a few dimensions and bounding the rest fails for random-projection sketches: their energy is spread
  evenly over the 64 dimensions, so a k-dimension prefix carries about k/64 of it. Measured in July 2026
  (bound dims plus Cauchy-Schwarz residual, no benefit) and again on 2026-10-07 (early-exit dot product,
  implementation_log 2026-10-07).
- Its strength, cheap updates, targets a cost CorrTrack does not have: insert plus expire is about 2.6% of the run
  at m=2000; the scan is about 80% (bucket gathering, Hamming check, pair-dedup hash).
- Where it would fit: a representation that concentrates energy in a few dimensions (StatStream's leading DFT
  coefficients). That gives up robustness on differenced and nonstationary data.

Optional evidence: on one real cell, the fraction of stored windows inside the tightest box that contains the
query cap (expected near 100%).

## 2. "Why not embeddings instead of sketches?"

The sketch already is an embedding: random, linear, data-independent. Those three properties are the method.

- Guarantee: the sketch inner product is an unbiased estimate of the window Pearson correlation with known
  variance, so the LSH bands and the gate are set for a recall target before seeing the data. A learned
  embedding has no such theorem; its recall is measured per dataset.
- Drift: a random projection cannot go stale; a learned embedding is fitted to a distribution that a
  nonstationary stream moves.
- Incrementality and lags: linearity gives O(basic window) updates and one sketch per window reused at every lag.
  A neural encoder needs a forward pass per series, window and step.
- Dimension is not the bottleneck: the 64-dim dot products are about 20% of the scan (2026-10-07 measurement).
- The variant worth an ablation: a learned linear basis (sliding PCA or SVD). It stays incremental, concentrates
  energy, and has an exact residual-energy bound, but needs refreshing under drift and is only as tight as the
  data is compressible (StatStream's weakness on differenced data). Not run yet.
- Nonlinear embeddings belong to other similarities (shape, DTW, nonlinear dependence), not to complete
  thresholded Pearson discovery.

## 3. "The usual complexity for this kind of problem is m log m; why is CorrTrack at m^1.64?"

- At fixed correlation density d the output is quadratic: about d m^2 / 2 pairs per window and lag, and a complete
  method must report each. m log m is impossible for complete thresholded discovery. Such bounds come from
  problems with linear output: top-k or kNN per series, low-dimensional range search (log m + output), 1-d sweeps.
- Even without the output term, high-dimensional inner-product joins resist subquadratic time: Ahle, Pagh,
  Razenshteyn, Silvestri, "On the complexity of inner product similarity join", PODS 2016, show hardness of
  approximate IPS join under the Orthogonal Vectors conjecture (implied by SETH; dimension omega(log n)).
- Theory for finding correlations: G. Valiant, "Finding Correlations in Subquadratic Time, with Applications to
  Learning Parities and the Closest Pair Problem", FOCS 2012 / J. ACM 62(2) Art. 13, 2015, finds a planted
  rho-correlated pair among n random Boolean vectors in d n^(3 omega/4 + eps) poly(1/rho) < d n^1.8 time, with
  rho absent from the exponent (earlier methods, Paturi et al., LSH, are n^(2 - O(rho))). Method: aggregate
  random-signed groups of vectors, compute all group-sum inner products with fast matrix multiplication, search
  only inside flagged group pairs. Improved by Karppa, Kaski, Kohonen, "A faster subquadratic algorithm for
  finding outlier correlations", SODA 2016 / ACM TALG 2018 (arXiv:1510.03895): n^(2 omega / (3 - log_tau rho))
  plus a lower-order term, tending to about n^1.58 for strong outliers over a weak background (our reading of
  the formula). Derandomized by Karppa, Kaski, Kohonen, O Cathain, ESA 2016 / Algorithmica 2020
  (arXiv:1606.05608). All rely on fast matrix multiplication and are offline, single-window, outlier-detection
  results: no stream, no lags, no complete thresholded output. Cite as theory context, not as competitors.
- Where the bound comes from (corrected 2026-10-16): classical LSH (Indyk, Motwani STOC 1998; Gionis, Indyk,
  Motwani VLDB 1999) with sign-random-projection hashing for cosine (Charikar STOC 2002: a bit agrees with
  probability p(theta) = 1 - theta/pi). Over N = m L indexed windows, K = log2 N bits per band makes a background
  pair (cos about 0, p = 1/2) collide in a band with probability 1/N, and n_bands proportional to N^rho with
  rho = log2(1 / p(arccos gamma)) keeps the recall target at the gate gamma. CorrTrack's sizing is exactly this
  (band_width = ceil(log2(m L / occupancy)), n_bands from the recall target, candidate_kernels.pyx). Per step,
  m queries each probe N^rho buckets and meet about N^rho background collisions: O(m (m L)^rho + |output_step|),
  i.e. m^(1+rho) L^rho, not m^(1+rho) L as first written. rho falls as the gate rises: gamma 0.7 / 0.8 / 0.9 /
  0.95 give rho 0.42 / 0.33 / 0.22 / 0.15, search exponent 1.42 / 1.33 / 1.22 / 1.15. Pairs with correlation
  between 0 and gamma collide more often than background, which pushes the real search exponent above 1 + rho.
- Measured: 1.64 over m = 125 to 2000 at fixed density 0.01, T = 0.9 and fixed lag count (1.62 to 1.64 across
  four regimes), BF_vect 2.05 (docs/headline_claim_2026-10-05.md). Per phase in the AR(1) levels cells: sketch
  m^0.96, candidate search m^1.73, validation m^1.81, candidates tested m^2.00. Tested is within 0.05% of
  the correlated output (217.6 M vs 217.5 M at m=2000), so at fixed density the output-driven work dominates and
  the 1.64 is a mix of linear, search and quadratic output terms, not the search exponent.
- Concede in the paper: at fixed density the output is quadratic, so the exponent tends to 2 at very large m.
  This is a density statement, not a threshold one: a higher threshold lowers rho, and in real data it also lowers
  the density. The formal statement is output-sensitive, subquadratic search plus output. Under bounded
  partners per series (density proportional to 1/m) the output is linear and the predicted exponent is 1 + rho.
  Measured 2026-10-18 (docs/synthetic_degree_sweep_2026-10-18.md): at fixed 55 partners per series (output linear)
  CorrTrack-LSH's wall exponent is 1.84 / 1.66 / 1.44 / 1.30 at T 0.7 / 0.8 / 0.9 / 0.95, falling with the threshold
  as predicted; exact arms stay at 2. It tracks 1 + rho_eff, where rho_eff (0.69 / 0.56 / 0.40 / 0.29) is the growth
  of the n_bands CorrTrack sizes from its shared pool of 64 sign bits, about 1.6 to 1.9 times the textbook rho.
  Sketching and validation are linear; the candidate search carries the superlinear part. Speedup over BF_vect at
  m=4000: 9x / 19x / 44x / 78x.
- Framing limit (the user's first reading, corrected 2026-10-15): the theory results support that an exponent
  in this range is plausible and in line with what is achievable for simpler versions. They do not make
  CorrTrack "the new state of the art": an empirical slope over a range is not an asymptotic bound, and the
  problems and guarantees differ. The defensible claim stays the conjunction in the headline.

## 4. "Why is nonstationarity a problem if it does not change the number of pairs?"

Controlled synthetic cell, m=500, W=60, step 6, 30 lags, T=0.9, planted density 0.01, same seed in every regime
(`~/results/synth_overnight/nway/ovn_{ar1,rw}_m500_L6_T0p9_d0p01{,_diff}_m500_W60_s6_L30_T0.9_pos`):

| regime | measured density | CorrTrack candidates tested | recall | speedup | FilCorr |
|---|---|---|---|---|---|
| AR(1), levels | 0.0100 | 13.5 M | 0.998 | 19.4x | 6.4x |
| AR(1), differences | 0.0100 | 13.6 M | 1.000 | 20.5x | 6.4x |
| random walk, levels | 0.0126 | 21.5 M | 0.962 | 13.6x | 6.4x |
| random walk, differences | 0.0100 | 13.5 M | 0.998 | 20.6x | 6.5x |

- It does change the count: random walks in levels share trends and create spurious correlations the generator
  never planted (+26% pairs), real Pearson correlations a complete method must report.
- The cost of filter-and-verify is set by the mass of pairs near the threshold, not above it: true pairs +26%,
  candidates +59%, candidate time +51%, validation +64%, recall 0.998 to 0.962.
- Over time the correlated set churns and means and variances drift, which breaks global normalisation, fitted
  filters, embeddings or bases, and methods that rely on energy concentration (low-frequency energy share 0.83
  in random-walk levels, 0.44 in differences). CorrTrack normalises per window, uses a data-independent
  projection, and sets its gate from the sketch's known error.
- Caveat to state: random-walk levels is CorrTrack's weakest regime too (13.6x against about 20x). The claim is
  graceful degradation while staying ahead, not immunity.
