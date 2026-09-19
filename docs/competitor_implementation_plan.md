# Implementation plan: six competitor methods

Status: **plan only, nothing implemented** (rev. 2026-09-16).
Companion to `docs/competitor_comparison_plan.md`, which settles *what* each method is and *why*
we compare it. This document settles *how* to build it: files, classes, signatures, order, tests.

Scope: **TSUBASA, BRAID/ThinBRAID, ParCorr/Cole-Shasha-Zhao, StatStream, CorrJoin**, added to the
existing `bruteforce` / `exact_stomp` / `filcorr` / CorrTrack battery. All primary sources have
been read (comparison plan §10.4); there are no literature blockers left.

---

## 1. The architecture already splits the work for us

Every one of the six falls into one of **two integration patterns that already exist in this
codebase**. Nothing new has to be invented at the harness level, which is the single biggest
reason to expect this to go smoothly.

| pattern | axis | existing members | to add |
|---|---|---|---|
| **A. All-pairs baseline** | `baseline_mode` | `bruteforce`, `exact_stomp`, `filcorr` | **TSUBASA**, **BRAID** |
| **B. Pruning method** | `data_representation` x `candidate_backend` | `sketch_proj` x {`lsh_approx`, `hamming_exact`, `brute_force`} | **ParCorr/CSZ**, **StatStream**, **CorrJoin** |

**That split is exactly the "prunes pairs" column of the capability matrix**, which is a good
sign: the code's existing seams match the real structure of the literature.

### Pattern A: the FilCorr template, already debugged twice

Adding a `baseline_mode` arm is a known quantity, done once already:

1. a `Candidates_BF_X(Candidates_BF_ExactSTOMP)` class (inherits the window buffer, the
   constant-window and spike guards, `_newStream`, state dump/load) that overrides `run()`;
2. a `run_bf_X(...)` wrapper on `CorrTrack` mirroring `run_bf_filcorr`
   (`library_corrtrack_parallel.py:9060`, **61 lines**);
3. one branch in the `run_bf` dispatch (`:9129`);
4. one entry in `_resolve_baseline_mode` (`:1929`) plus its alias map and valid set;
5. CLI flags in `corrtrack_run_bruteforce.py` and defaults in `experiment_run_exec_param.py`.

Steps 2 to 5 are mechanical, roughly 120 lines per arm. **All the real work is step 1.**

### Pattern B: sketch plus index, with the grid to restore

A pruning arm needs a **representation** (produced by `Sketches`, `:9730`) and a **backend**
(consumed by `Candidates`, `:12183`). Both axes are already pluggable and validated:
`_VALID_DATA_REPRESENTATIONS` (`:1738`), `_VALID_CANDIDATE_BACKEND_AXIS` (`:1745`). They need
**widening back out**, not inventing, and the v1.0 snapshot holds a disabled grid implementation
to work from (comparison plan §2.2).

---

## 2. Phase 0: shared prerequisites (blocking, ~2-3 days code + ~3-5 days data)

Do these first. Every one has more than one consumer, and skipping them means writing the same
thing two or three times slightly differently, which is the main way this comparison could end
up non-comparable.

### 0a. Normalize the window before reducing (3 consumers)

**ParCorr, CorrJoin and StatStream all use the same centred unit-norm form:**

```
x_hat[i] = (x[i] - mean(x)) / sqrt( sum_i (x[i] - mean(x))^2 )
```

CorrTrack instead normalizes **the sketch** so the dot product is cosine: `sketch_norm` is
hardcoded to `"mean_l2"` at `library_corrtrack_parallel.py:4033`.

**Add a `sketch_norm="unit_l2_window"` path** that normalizes the raw window before projection,
leaving `mean_l2` untouched as CorrTrack's default. One code path, three consumers.
This is the identity every one of them rests on, so getting it wrong poisons three arms at once:

```
corr(x,y) = 1 - d^2(x_hat, y_hat) / 2
```

**Test**: for random windows, assert `2 - 2*corr == d^2` on the normalized pair, to 1e-12.

### 0b. Extract the five-sum sufficient statistics (3 consumers)

`sum x, sum y, sum x^2, sum y^2, sum xy` are used by `exact_stomp` (inline today), **CorrJoin**
(their Eq. 2 incremental update) and **BRAID** (their Eqs. 9 and 10, per level). Pull the
accumulate-and-combine logic into one helper rather than writing it a third time.

Keep it a plain function over arrays, not a class: BRAID needs it per level and per lag, CorrJoin
per stride, `exact_stomp` per window. A class would fit none of them well.

### 0c. Counter contract (this is what makes the comparison publishable)

Comparison plan §5b.1 needs a three-phase decomposition for **every arm**. Today the phase
counters are populated by the CorrTrack path; the `baseline_mode` arms need to fill them too,
and crucially **pruning counters must be explicitly `0`, never null**, or the summarize / prune /
verify table has holes exactly where the argument is.

Required from every arm, all already in `RUN_RESULT_COLUMNS` (`:155`):
`sk_time`, `cand_time`, `val_time`, `total_candidates`, `tested_candidates`,
`candidate_search_*_touched`.

For a Pattern A arm the candidate set IS the pair set, so the invariant is
`total_candidates == tested == all enumerated pairs` (prune ratio 1.0) and every
`candidate_search_*_touched == 0` (no index probes). `cand_time` is the arm's summarize cost
(e.g. TSUBASA's segment sketching), not zero. **Written as `_assert_competitor_contract` in the
test file (done 2026-09-16, entry (j)); all three existing Pattern-A arms pass it.** An earlier
draft of this paragraph said `total_candidates = 0`, which was wrong.

### 0d. Negative-correlation policy -- DECIDED 2026-09-16 (user's refinement)

This is a live problem, not a hypothetical: the capability matrix says **ParCorr has no negative
correlation and FilCorr has none either** (its Eq. 7 takes a `max` of signed correlations), yet
our shipped FilCorr port applies `|corr| >= threshold` because it inherits `exact_stomp`'s accept
mask. That deviation is already in published results (comparison plan §4b.2).

**The policy:**

1. **Primary head-to-head at `neg_corr=False`.** No arm is extended beyond its own paper.
   CorrTrack gives up a capability it has; that belongs in the capability table, not the speed
   table.
2. **A second, labelled run at `neg_corr=True`**, in which each arm lacking native support falls
   into one of two classes:
   - **Enable, report, and disclose.** Where enabling the capability does *not* require us to
     design any part of the method, we switch it on, report the results, and **state in the
     paper that we enabled it and that the original paper did not have it.** This is more
     informative than an N/A column and it is honest.
   - **N/A.** Where enabling it *would* require us to design part of their algorithm, we do not.
     A mechanism we invented for a competitor cannot be attributed to them.

**The criterion that separates the two classes** is not coding effort; it is whether enabling
negative correlation changes the algorithm under measurement:

| class | test | arms |
|---|---|---|
| **native** | the paper has it and exercises it | BF, STOMP, TSUBASA (`abs(c) > theta` in every experiment) |
| **specified by them, unevaluated** | their paper gives the mechanism, they never measured it. Enable it; disclose that the evaluation is ours | BRAID (Def. 1 `abs(R(l))`), StatStream (Lemma 3: probe the cell at `-c`), CorrJoin (Alg. 1 line 14) |
| **enabled by us** | the method already produces exact signed correlations; enabling is an acceptance filter on values it computed anyway. Candidate generation and timing untouched. Enable; disclose as **ours** | **FilCorr** (Parseval is sign-preserving; our port already does this -- see the flag below) |
| **N/A** | finding anti-correlated pairs would require a new candidate-generation mechanism (probing the negated sketch in every grid). We would be designing their method | **ParCorr / CSZ** |

**Implementation**: replace the boolean with a three-valued per-arm declaration,
`neg_corr_support in {"native", "specified", "enabled_by_us", "not_available"}`, that the harness
reads. On a `neg_corr=True` run it refuses `not_available` arms, runs the others, and **writes the
tag into `RUN_RESULT_COLUMNS`** next to every row, so the disclosure travels with the number
and cannot be lost between the run and the paper.

**FilCorr classification confirmed by the user (2026-09-16): `enabled_by_us`.** Our port already
enables it through the inherited accept mask, and the Parseval decomposition preserves sign, so
no part of FilCorr's algorithm is touched. This is what makes the existing 2026-09-12 numbers
legitimately disclosable rather than retracted: they stand, with the disclosure. ParCorr/CSZ is
the one arm in N/A. **§0d is fully decided; nothing here remains open.**

### 0e. One shared competitor contract test

Rather than remembering per-arm what to check, write `_assert_competitor_contract(arm_name)` and
have every arm call it. It asserts:

1. counters populated per 0c, with pruning counters exactly `0` for Pattern A;
2. the arm's declared capabilities match the matrix (so the code and the paper cannot drift);
3. **the degenerate-exact configuration reproduces the bruteforce pair set** (each arm names its
   own, see §3);
4. state dump/load round-trips.

This is the single highest-leverage item in phase 0. It is what caught both real FilCorr bugs.

### 0f. Competitor datasets -- **DONE 2026-09-17** (log entry (d)): every public set obtained, registry in `datasets/competitor_sources.md`

Status 2026-09-17: Motes (4 variables), Yellowstone (raw + 3-7 Hz, 28 of 29 stations), USCRN 2020 (4 variables), Berkeley Earth 2010-2019 (18,520 cells), CorrJoin's five files, five of CSZ's ten (DaISy; UCR TSDMA is offline), Sunspots, StatStream random walks and BRAID Sines/SpikeTrains generators. Licensed sets replaced by `sp500`/`corrjoin_stock`, stated. Loader `datasets/competitor_loader.py`, 22 `experiment_dataset_*.py` configs, N-way runner `abaca/nway_compare.py`. The original plan text follows.

### 0f (original). Competitor datasets -- DECIDED 2026-09-16: add every one that can be obtained

The user's call: try to add all of them. Two reasons this is worth its cost. It makes every
paper's published claim reproducible on that paper's own data (the strongest faithfulness check
we have, comparison plan §6.5), and several of these sets sit in regimes our current data does
not cover at all. Build a registry `datasets/competitor_sources.md` plus one fetch/align script
per set, following the existing `build_*_full.py` pattern. Each entry records: source paper,
access class, fetch method, series count and length, and **which axis it tests**.

| dataset | paper | access | what it gives us |
|---|---|---|---|
| Random-walk generator `s_i = 100 + sum (u_j - 0.5)` | StatStream §5 | **synthetic, formula given** | cooperative baseline; free |
| 5000 x 4080 random walk | CorrJoin | **synthetic**, in their Drive folder | cooperative, large `m` |
| Sines (mixture of sines, n=32,768), SpikeTrains (period 6500, n=100,000) | BRAID §6.1 | **synthetic**, reproducible approximately | periodic and bursty lag anchors; Sines is BRAID's zero-error case |
| **Intel Berkeley Lab / Motes**: 54 sensors, temperature, humidity, light, voltage, 30s | BRAID (2005 and 2010) | **public** (`db.csail.mit.edu/labdata`) | **the one real set that is multi-series and lag-correlated by construction** (202 and 224 min lags between nearby sensors). Our lagged comparison has nothing like it today |
| Kursk seismic, n=70,000 | BRAID | likely via IRIS | bursty, single-event lag |
| Sunspots, daily | BRAID | **public** (SILSO) | long-period, ~11 y |
| chlorine (4830 x 2040), gas (5120 x 3600), stock (3878 x 1259) | CorrJoin | **public**, Drive link in comparison plan §3.0 | real, already used at SIGMOD; reproduces their pruning/speedup claims |
| **Yellowstone seismic**, 29 stations, 100 Hz, event `us70008jr5`, 2020-03-31 23:52:30 UTC | FilCorr §VI | **public via IRIS**, fully specified | **uncooperative** (white-noise traces), lagged by wave propagation, band 3-7 Hz. Also reproduces FilCorr's own case study |
| NOAA USCRN hourly 2020, 157 stations | TSUBASA | **public**, URL in paper | **uncooperative** climate data; reproduces their §4.1 DFT-accuracy finding |
| Berkeley Earth 1x1 degree land grid, 18,638 series x 3,652 | TSUBASA | **public** (`berkeleyearth.org/data`) | **the largest `m` available**, well past our 2k target; scalability |
| UCR archive sets: spot_exrates, cstr, foetal_ecg, evaporator, steamgen, wind, winding, eeg, price, return | CSZ §6 | **public** (UCR TSDMA; check current archive for these 2002-era names) | 1,365 to 13,736 series each; the original cooperative-vs-uncooperative testbed |
| NYSE TAQ (300 stocks, 1s) | StatStream | **licensed** | stand-in: our `sp500` pools, **stated as such** |
| CRSP end-of-day, 7,861 stocks | CSZ | **licensed** | stand-in: `sp500`, stated |
| Yahoo Finance, ~40k symbols 2010-2018 | ParCorr | reproducible in principle, exact set unspecified | stand-in: `sp500`, stated |
| ParCorr seismic | ParCorr | unspecified in the paper | skip unless the authors clarify |

**Priority within 0f**: the free synthetic ones first (an afternoon), then **Motes and
Yellowstone** (the two that fill regimes we lack: real lag-by-construction, and real
uncooperative), then CorrJoin's Drive sets and the two climate sets, then UCR. Licensed sets
are replaced by `sp500` with the substitution named in the paper. **Estimate ~3-5 days** for
fetch and alignment across the public sets, given the existing pipeline.

Note what this does to the evaluation design: with Motes and Yellowstone in hand, the
lagged comparison (§6.1 of the comparison plan) can run on data where the lags are physical
facts rather than synthetic insertions, which is a much stronger position to argue from.

### 0g. Two small checks before starting

- **`n_lags=0` has no explicit guard** in the library. The primary head-to-head runs there
  (comparison plan §6.1), so confirm every existing arm handles it before adding five more.
- **scipy is available and already a dependency** (`scipy 1.11.4`, imported at `:19` for
  `scipy.stats.norm`). BRAID needs `CubicSpline` and `minimize_scalar`; both verified present.
  Confirm the same on the Abaca conda env before phase A2.

---

## 3. The six arms

### Track A: all-pairs baselines (Pattern A, low risk, independent of each other)

#### A1. TSUBASA (`baseline_mode="tsubasa"`) -- **DONE 2026-09-16** (entry (j)); was est. 2-3 days

**Build this first.** It is the simplest of the six, it is exact so it self-verifies, and it
validates the Pattern A template a second time before we rely on it for BRAID.

`class Candidates_BF_TSUBASA(Candidates_BF_ExactSTOMP)`, overriding `run()`.

Maintain per basic window `j`: for each series, `mean` and `std`; for each **pair**, the
correlation `c_j` on that basic window. Combine with their Lemma 1:

```
Corr(x,y) = sum_j B_j ( s_xj s_yj c_j + d_xj d_yj )
          / [ sqrt( sum_i B_i (s_xi^2 + d_xi^2) ) * sqrt( sum_i B_i (s_yi^2 + d_yi^2) ) ]
    d_xi  = mean(x_i) - (sum_k mean(x_k)) / ns
```

- **Memory, quantified now rather than discovered later.** The per-pair-per-basic-window store is
  `O(m^2 * k)` floats. At `m=500` with `window_size=168, basic_window=12` (`k=14`) that is
  `500*499/2 * 14 = 1.75M` floats, about **14 MB**: fine. At `m=2000` it is about **224 MB**:
  workable on Abaca, not on WSL. **Report it as a finding** (exact arbitrary-window queries
  bought with quadratic storage), and cap `m` on local runs.
- **Scope note**: TSUBASA's headline feature is *arbitrary* query windows, which our fixed-window
  harness never exercises. Say so rather than letting it look slower for no reason. Its honest
  comparison against `exact_stomp` is "per-pair precomputed sketches versus rolling statistics".
- **Anchor**: exact, so it must reproduce the bruteforce pair set precisely.
- Negative correlation: yes, their Algorithm 2 uses `|c| > theta` in every experiment.

#### A2. BRAID and ThinBRAID (`baseline_mode="braid"`, `thin` flag) -- **DONE 2026-09-16** (entry (k)); was est. 6-8 days

`class Candidates_BF_BRAID(Candidates_BF_ExactSTOMP)`, overriding `run()`.

Three pieces, in this order:

1. **Hierarchical window averages**: `Ax_h(t) = (Ax_{h-1}(2t-1) + Ax_{h-1}(2t)) / 2`, `Ax_0 = x`.
   One level per power of two, recomputed every `2^h` ticks.
2. **Five sums per level per lag** (phase 0b), probed at
   `l = {0,1,...,2b-1; 2b, 2(b+1), ...}` with `b = 16` (the paper's experimental value).
3. **Interpolate and locate**: `scipy.interpolate.CubicSpline` over the probed points, then
   `scipy.optimize.minimize_scalar(method="bounded")` for the maximum. The paper uses Brent and
   notes the interpolation choice is orthogonal, so this substitution is safe and should be noted.

Report the **earliest local maximum of `|R(l)|` above `gamma = 0.4`**.

- **One necessary adaptation, to be labelled**: BRAID's max lag is `m = n/2` and grows with the
  stream; our harness has a fixed `n_lags`. Cap at `n_lags`. That is a harness constraint, not
  an algorithm change, and it should be stated as such.
- **Anchor**: with `2b > n_lags`, level 0 covers every lag at raw resolution with no smoothing and
  no interpolation, so BRAID **must reproduce the bruteforce lagged pair set exactly**. Then a
  second anchor at `b=16, gamma=0.4` reproducing ~1% lag error on a periodic signal.
- **Output shape differs** (one lag per pair, not pairs per lag). Emit both: the chosen lag for
  the lag-agreement metric, and the pair set for the common-denominator comparison. See
  comparison plan §5a.3, which is the one place the uniform metric scheme genuinely breaks.
- **ThinBRAID is IN scope -- DECIDED 2026-09-16.** The battery will run **up to 2,000 series**,
  past the ~1,000 crossover in their Fig. 19, so ThinBRAID is the faster variant at the top of
  our range. **It is also the variant that fits in memory.** Plain BRAID stores `Sxy_h(l)` per
  pair per probed lag: with `b = 16` and `n_lags = 168` that is 5 levels, `32 + 4*16 = 96` values
  per pair; at `k = 2000` that is `1,999,000 * 96 * 8 B` = about **1.5 GB**. ThinBRAID stores
  `d = 400 / 2^h` projections per series per level instead: `(400+200+100+50+25) * 2000 * 8 B` =
  about **12 MB**. So at 2k, ThinBRAID is not an optimization, it is what runs.
  Build it as a mode of the same class: `Candidates_BF_BRAID(..., thin=True)`. It reuses the
  random-projection machinery CorrTrack already has; the only new piece is recovering
  `Sxy_h = (Sxx_h + Syy_h - Dp_h) / 2` from sketch distances (their Eq. 24) and the JL sizing
  `d0 = (4 + 2 delta) / (eps^2/2 - eps^3/3) * log n` (their Theorem 5). Anchor: at the same
  `b, gamma`, ThinBRAID's lag estimates must agree with BRAID's to within their Lemma 9 bound.
  **Adds ~2-3 days to Track A.** Run BRAID below ~1000 series and ThinBRAID above, and report
  the crossover we measure against theirs.

### Track B: pruning methods (Pattern B, where the engineering risk lives)

#### B1. ParCorr / Cole-Shasha-Zhao (`sketch_proj` x `parcorr_grid`) -- **DONE 2026-09-17** (log entry (a)); was est. 3-4 days. Built as a self-contained index behind the `Candidates._lsh_index` interface, not by reviving the disabled CorrTrack multi-grid path; v1.0 was not mined.

**One arm, not two** (comparison plan §4a.3): the candidate search is the same algorithm, so
build it once and treat the differences as ablations.

- Representation: `sketch_proj` **already exists**. Needs phase 0a's window normalization.
- Backend: new `candidate_backend="parcorr_grid"`. Partition the `r`-dim sketch into `r/k` groups
  of `k=2`, one grid per group, candidate if co-located in a fraction `f` of grids.
- **Mine `corrtrack_release_v1.0`, not `backup2`** (comparison plan §2.2). Three things there are
  present but disabled and must be corrected, not inherited:
  - `freq_threshold = 0` hardcoded at `v1.0:2233` ("disable frequency gating") -- the vote is off;
  - `full_vector_candidates` collapses to `n_grids=1` at `v1.0:2182-2185`;
  - `required_hits = n_grids` at `v1.0:4647` is effectively `f = 1.0`, not the paper's `0.7`.
- **Cell size**: do **not** use `_compute_base_cell_size`. Sweep CSZ's published distance
  multiplier `c` over `[0.1, 1.3]` with `f = 0.7`, `k = 2` fixed (comparison plan §2.3).
- **Same-cell only, no neighbour probing** -- ParCorr lists that as future work.
- **No negative correlation** (phase 0d refuses it rather than adding it).
- Constraint: `window_step == basic_window`. Assert it; exclude ParCorr from sweep cells that
  violate it rather than running a modified ParCorr.
- **Ablations within the arm**: structured random vectors plus convolution (CSZ's 30-40x cheaper
  update) versus plain vectors; and CSZ's combinatorial-design tuning versus ParCorr's tune-`f`.

#### B2. StatStream (`sketch_dft` x `statstream_grid`) -- **DONE 2026-09-17** (log entry (b)); was est. 5-7 days. Lemma 6 incremental digests not implemented (constant-factor; charged to sk_time).

Bigger than the earlier draft assumed, and **the arm a reviewer will press hardest on**, since it
is the closest structural peer to CorrTrack.

- **New representation `sketch_dft`**: keep the first `n` DFT coefficients of the normalized
  window. Incremental per basic window via their Lemma 6, which needs the `n` digests
  `zeta_m = sum_i e^{j 2 pi m (b-i)/w} x_i`. Normalized coefficients via Lemma 4
  (`X_hat_0 = 0`, `X_hat_i = X_i / sigma_x`).
  **Reuse FilCorr's caching pattern** (`_band_fft_at`, `_evict_band_fft_cache`) rather than
  inventing a second one; the earlier "no FFT machinery exists" objection is stale.
- **New backend `statstream_grid`**: grid over the bounded DFT cube (diameter `sqrt(2)`, their
  Lemma 7), cells of diameter `eps = sqrt(1 - T)`, **neighbouring cells probed** (unlike ParCorr).
- **Build the `[E]` core first, the `[S]` extras second and separately labelled.** Only the
  synchronous pruning path was ever evaluated by its authors. The lagged path (timestamped,
  never-cleared grid, lags in multiples of the basic window) and the negative-correlation path
  (their Lemma 3, probe the cell at `-c`) are specified but unmeasured, so **there is no published
  number to check our port against** and a poor result there is not evidence against StatStream.
- **Anchor, and it is the sharpest of any arm**: the paper separates two guarantees. The grid
  filter is provably false-negative-free (their Theorem 2) while the reported recall of
  0.9987-1.0 comes only from the DFT post-processing. So the port must show **grid recall exactly
  1.0** against bruteforce with post-processing disabled, and only then the published
  precision/recall with it enabled. A port that leaks false negatives in the grid is wrong
  however good its end-to-end numbers look.
- **Watch the lagged grid's memory**: it is never globally cleared, and eviction is driven by the
  `T_M` timestamp rule. Get that rule right or it grows without bound.

#### B3. CorrJoin (`sketch_paa_svd` x `corrjoin_double_filter`) -- **DONE 2026-09-17** (log entry (c)); was est. 5-7 days. Reproduced from the authors' R code: per-window SVD on the alive PAA(ks) block, bucket grid + 27-neighbourhood + exact eps_1-ball, eps_2 filter on PAA(ke). Negative correlation found unreachable through its filters: `supports_neg_corr="not_available"`.

Hardest, and last, but no longer gated on anything: the spec is complete (comparison plan §3.1).

- Representation: PAA to `ks=15` and to `ke=30`, then SVD to `kb=3`. Requires `n` divisible by
  both `ks` and `ke`; assert it.
- Backend, two stages: bucket grid of width `eps_1` on the `kb` dims with neighbours probed,
  then a Euclidean filter at `eps_2` on the `ke` dims.
  ```
  eps_1 = sqrt( 2 * ks * (1 - T) / n )
  eps_2 = sqrt( 2 * ke * (1 - T) / n )
  ```
  **Both divide by `n`, and `eps_1` uses `ks`, not `kb`.** Both derivative sources had this wrong
  and either error quietly corrupts recall and speed.
- Incremental correlation across strides via the five running sums (phase 0b) -- the same form
  `exact_stomp` already implements, so this is reuse.
- **SVD across windows -- SETTLED by the authors' own code (2026-09-16).** The user supplied
  the R scripts the paper made available (`docs/reference_code/corrjoin_authors_R/`, with a
  README of what was read). `2-CorrJoin.R` **recomputes the SVD from scratch every window**
  (`svdFunc(...)` inside `for (i in 2:numOfSW)`), and recomputes PAA per window too; only the
  per-series `sum x` and `sum x^2` are incremental, and `sum xy` is recomputed from raw for each
  *surviving* candidate. So: **per-window SVD is what they did**, and this arm is now a
  **reproduction**, not a spec-driven port. My earlier "incremental SVD is research-grade" was
  an overstatement either way: the SVD is over the `m x ks` matrix with `ks = 15`.
  **Mirror their structure exactly**: incremental `sum x, sum x^2` per series; PAA and SVD per
  window; `sum xy` only for survivors (which is what our `validate_corr_rows` path already does).
  **Stride note**: with PAA and SVD recomputed per window, the stride effect comes only from the
  per-window cost being amortized over fewer windows; state this next to any stride result.
- **The bucketing filter, from `3-BucketingFilter.R`**: 3D grid over the `kb = 3` SVD
  coordinates, cell width `eps_1`, **27-cell neighbourhood probed** (hand-unrolled in the R),
  and **an exact `eps_1`-ball test on every pair found in a neighbourhood**. Port that shape.
  **One thing not to copy**: a hardcoded `checkVal <- 9` scans only bins within +/-9 of the
  centre, i.e. coordinates in roughly `[-0.49, 0.49]` at their `eps_1`. It is an implementation
  shortcut absent from the paper and it can drop candidates at extreme coordinates. Implement
  the full grid and **measure whether the truncation ever matters**; report either way.
- **Normalization, from the code**: `paamN <- (paam - meanT) / tauT` with
  `tauT = sqrt(sum x^2 - n * mean^2)` -- normalize **after** the linear reduction using the
  window's mean and centred L2 norm. **This is exactly the phase 0a design** (mean-adjust the
  reduced vector, divide by the window norm), now confirmed by the authors rather than derived.
- **Experimental constants in their script**: `windowSize = 1020`, `theta = 0.1` (T = 0.9),
  `stride = 100`, `ks = 15`, `ke = 30`, `kb = 3`, `EuclThreshold = sqrt(2*theta/frameSize)`.
  Use these for the published-parameter anchor.
- **Bonus witness**: `6-TSUBASA.R` is the CorrJoin authors' reimplementation of TSUBASA, in
  the Lemma 1 form. It is the code behind their Fig. 10, and it is a second independent
  witness to check our own `Candidates_BF_TSUBASA` against (A1).
- Sanity ceiling while validating: their speedup is bounded above by `1 / r1`, where `r1` is the
  fraction surviving the first filter. A measured speedup above that means a bug.
- Needs `m > 100` before the reduction pays for itself; do not report it below that without
  saying so.

---

## 3a. Phase R: reproduce each paper's own result first (added 2026-09-17, user's request)

Before any head-to-head number is quoted, every port is run on its authors' own data (or the
registry's stand-in) in the regime its authors published, and our number is printed next to
theirs: `abaca/reproduce_papers.py {braid,corrjoin,statstream,parcorr,filcorr,tsubasa,all}`,
JSON under `tmp_artifacts/reproduce_papers/`. A port that cannot reproduce its own paper is not
a fair competitor yet; a documented discrepancy is a result.

| paper | data | paper's claim | our metric | result (Abaca job 3117155, 2026-09-18, and local pilots) |
|---|---|---|---|---|
| BRAID / ThinBRAID TKDD 2010 | Sines (same-spectrum pairs, Fig. 15), SpikeTrains (period 6,500 + white noise), Motes #1/#10 and #47/#48, Sunspots as 25,900-day chunks; the 55-sensor Humidity/Light set is not obtainable | Tables III/IV, E = 100 abs(l_b - l_n)/l_n over the whole sequence (max lag n/2): Sines 716 -> 716 (0.000) / 706 (1.397); SpikeTrains 2841 -> 2830 (0.387) / 2826 (0.528); Humidity 0.024 / 1.178; Light 0.529 / 0.176; Sunspots 1156 -> 1168 (1.038) / 1155 (0.086); Motes lags 202 and 224 min | same regime: W = n/2 vs history shifted up to n/2, Definition 1 with a +-16 neighbourhood on the naive CCF, Eq. 32 | **BRAID reproduces**: Sines, 32 pairs at n = 32,768: E = 0.000 to 1.34% (median 0.06%; paper 0.000%); SpikeTrains with the paper's lag scale (local, lags ~1,100 to 1,500): 0.135 / 0.450% (paper 0.387%); the Abaca SpikeTrains row was generated with lags 17 to 127 by mistake (planted lags of a few pulse widths on smoothed levels: 1 to 94%) and is rerun with --max-lag 3000. **ThinBRAID does not reproduce** (41 to 96% on Sines): Eq. 24 with d = 400/2^h has a JL error of sqrt(2/d)(1 - rho) per knot, independent projections per level put jumps at level boundaries, and on a sine CCF a 0.02 error in R moves the argmax by ~250 lags; d = 2,000 at every level does not fix it. Table IV cannot follow from the text as written; **open, disclosed**. Motes: in our epoch-aligned file the pairs peak at lags 1 to 5 epochs with R(0) ~ 0.9, not at 202/224 min (per-mote epoch counters in the raw file; the paper's lags look like an alignment artefact); Sunspots chunks likewise peak near 0. On such flat CCFs Definition 1 is ill-posed and E is not informative |
| CorrJoin PACMMOD 2023 | authors' stock/chlorine/gas/synthetic files, m = 1,000, W = 1,020, stride 10, ks/ke/kb = 15/30/3 | speedup <= 1/r1; gains only for m > 100; pruning vanishes near 20% correlated (Fig. 15) | mean r1, 1/r1 ceiling, correlated fraction, T in {0.7, 0.8, 0.9, 0.95} | **reproduces qualitatively**: r1 falls with T (stock 0.33 -> 0.06, ceiling 3.1x -> 17x; synthetic 0.28 -> 0.05, 3.6x -> 20x; chlorine 0.74 -> 0.13); gas, 26% correlated at T = 0.7, has r1 = 0.92 and a 1.1x ceiling, their Fig. 15 statement exactly. Speedup values not transcribed from the paper; the measured wall ratio corrjoin/bf (0.03 to 1.05) stays below 1/r1 in every cell, as it must |
| StatStream VLDB 2002 | random walks s = 100 + sum(u - 0.5), m = 500, sliding window 1 h = 3,600 at 1 s, basic window 60 | Fig. 5: grid pruning power 0.01 to 0.09, filter precision ~0.55 to 0.9 at 16 coefficients; Table 2 S0.85 t=0.0005: precision 0.9931, recall 1.0 (real cells 0.9765 to 0.9947 / 0.9987 to 1.0) | grid (16 sliding-window coefficients, Lemma 2 filter); post-processing = section 3.4 curve fitting with 2 DFT coefficients per basic window, exact means and sigmas, report if corr_approx > T - t | **reproduces**: pruning power 0.0093 to 0.0259, filter precision 0.68 to 0.76, S0.85 t=0.0005 -> 0.9936 / 0.9992, T=0.9 t=0.001 -> 0.9748 / 1.0 |
| ParCorr DMKD 2018 | Yahoo (unavailable) -> sp500_sub263 (m = 263), corrjoin_stock (m = 1,000), w = 500, b = 20 | recall > 90 / 96 / 95.7% at T = 0.7 / 0.8 / 0.9 with r = 60, k = 2, f = 0.7; precision 100% | recall vs bruteforce at exactly those settings; the cell size, which the paper does not state, at CSZ's c = 0.7 | **does not reproduce at the paper's stated settings**: recall 13 to 38% (precision 1.0). The unstated cell size is the whole story: the CSZ-protocol tuning of 2026-09-17 (c = 0.3 to 0.5, f = 0.6 to 1.0, N = 30 to 48) reached 0.94 to 0.97 on Motes. ParCorr's published recall is reachable only with a cell size the paper omits, which is what the protocol is for; disclosed |
| FilCorr ICDM 2020 | white noise at 100 Hz, 3-7 Hz band, W = 2,000, lag 100, m = 25 to 200; Yellowstone case study | up to 4x more sensors than naive (16x time at O(m^2)); beats ParCorr below ~700 streams | time ratio bruteforce/filcorr and its square root vs m; Yellowstone lagged pairs | **trend reproduces, magnitude below the paper's at our m**: time ratio 0.40x (m = 25) -> 1.27 -> 2.47 -> 4.82x (m = 200), sensor multiplier 2.2x at m = 200 and rising (paper "up to 4x", at larger m). Our naive baseline is the `bruteforce` arm: an all-pairs Pearson recompute per step, vectorized in Cython, a faster implementation of the same "naive" the paper used; `exact_stomp` (incremental) is the stricter comparison and runs in the campaign. Yellowstone: the job ran with lag 100 (1 s) instead of the paper's 10 s and found 0 pairs >= 0.5; rerun with lag 1,000 |
| TSUBASA SIGMOD 2022 | USCRN 2020 temperature, m = 153, W = 168, step 12, T = 0.7 and 0.9 | exact; >= 10x faster than raw Pearson recompute | exact-match flag; wall vs our `bruteforce` arm = Cython all-pairs Pearson **recomputed from raw values every step** (the papers' "naive", vectorized), not `exact_stomp` (the incremental five-sum arm) | **exactness reproduces** (1,707,473 and 207,565 pairs identical to bruteforce); wall 0.78x of the recompute baseline, i.e. 1.28x faster where the paper reports >= 10x over its naive. The gap is the baseline's tier (a per-pair recompute in their Go code vs our vectorized Cython recompute), consistent with CorrJoin's Fig. 10 placing TSUBASA at the naive's complexity. Against `exact_stomp` (incremental) the campaign will show the stricter ratio |

## 3b. Campaign (added 2026-09-17; redesigned the same day as a Sobol design)

`abaca/campaign_competitors.py`: a `DatasetSpec` table (19 datasets: 13 registry sets + sp500,
acwi_capweighted, streamflow, wikipedia, global_weather, smartmeter at their historical W/step)
and a design generator: per dataset one synchronous anchor (m_max, L=1) plus `--points` Sobol
points over (m log-uniform, L uniform in [1, L_max]) with `n_lags = (L-1) step`, **T swept in full**
{0.7, 0.8, 0.9, 0.95} at every point; Berkeley Earth adds a full-m (18,520) anchor for the exact
arms + StatStream; one step=1 Yellowstone cell. Per cell and per labelled run (neg_corr False /
True): `hyperopt_corrtrack.oar` (CorrTrack's own hyperopt, every execution), `tune_competitors.oar`
(CSZ protocol), `nway_compare.oar` (`-a` on both). `basic_window = step` passed to every arm.
4 points -> 385 cells, 2,310 jobs. Parameters travel as `KEY=VALUE` script arguments. Every
nway JSON carries the dataset profile (`abaca/dataset_profile.py`) and the metric set of log
entry (g). **Open for the user**: W/step policy (log (f) table), `--points`, pilot cells.
Nothing submitted; Abaca at 9e983d3 with datasets synced.

### 3b (ii). Execution mechanics on Abaca (2026-09-19)

Per-arm accounting (user's request): every arm of every battery runs in a forked child
(`abaca/resource_probe.py: run_isolated`) so its numbers are its own: phase times
(`sk_time`, `cand_time`, `val_time`, `monit_time`, `other_time`, `runtime` without artifact I/O,
`wall`), peak RSS (`ru_maxrss` of the child) and mean RSS (50 ms sampler), both also as deltas over
the shared baseline (dataset + interpreter inherited at fork), storage I/O (`/proc/self/io`),
artifact size, CPU user/sys, and the arm's absolute interval. Energy: RAPL is root-only on the
mercantour nodes, so the in-process reading is None; the N-way jobs are submitted with
`-t "monitor=prom_.*"` and `abaca/kwollect_power.py` pulls the node's ACPI power meter
series (15 s samples; idle 64 W, 20-core burn 245 W, one-sample response, probe 3122070) for `aggregate_campaign.py --power`, which integrates it over each
arm's interval and subtracts the node's idle power. To be stated: 15 s resolution, so per-arm energy
is quotable for arms running well over a minute, per-cell energy otherwise. The same accounting is
in `ablation_corrtrack.py` and `parallel_scaling.py`; the hyperopt and tuning jobs record their own
peak RSS and wall through GNU `time -v`.

One build per campaign: `abaca/prepare_snapshot.oar` copies the tree at the current commit to
`$RESULTS_ROOT/snapshots/<sha>[_tag]`, builds the kernels once (`-march=native`; the wrappers refuse a
node whose CPU model differs), runs the three test files, writes `SNAPSHOT_OK`. Every job runs from the
snapshot (`SNAPSHOT=` token; `abaca/_snapshot_enter.sh`); nothing rebuilds in the shared clone any more
(the old per-job `rm *.so; build; pytest` was a race between concurrent jobs and 5 min per job).
`datasets/competitor` inside the snapshot is a symlink to the clone's, so the generated synthetic sets
(`campaign_submit_generate.sh`, one OAR job run from the snapshot) are visible to all.

Node partition on mercantour3 (15 usable hosts, 20 cores / 192 GB each): the N-way jobs take a whole host
(`NWAY_HOSTS`, 11 hosts by default); the hyperopt and tuning jobs are packed by cores on `PACK_HOSTS`
(4 hosts; 2 / 4 / 10 cores per job for m <= 1250 / <= 2500 / larger, a memory share), so whole-host
jobs are never starved by core-level ones. OAR lesson: a `#OAR -l` in the script header plus a
command-line `-l` makes a moldable job (either may be chosen; the Phase R reruns got the header's 12 h);
resources and properties are now on the command line only. Results land in
`$RESULTS_ROOT/{hyperopt,tuned,nway}/<stem>_<pos|neg>/` (`RUN_NAME`), the N-way JSON carries `cell`.

Submission: `abaca/campaign_feeder.py` feeds the emitted script cell by cell while our Waiting jobs are
below `--max-waiting` (resumable through `<script>.state`, `--skip-done`). Aggregation:
`abaca/aggregate_campaign.py` -> `runs.csv`, `cells.csv`, `tuning.csv`, `summary_by_{T,m,L,space,dataset,neg}.md`,
`failures.md`, each number with its cell count.

## 4. Order of work

Track A and Track B touch **disjoint code** (`baseline_mode` dispatch versus the
representation/backend axes), so after phase 0 they can proceed independently. That matters for
risk: Track A is near-certain to land, Track B is where the schedule could slip.

```
Phase 0  shared prerequisites (0a-0g)              blocking, 2-3 d code + 3-5 d data
   |
   +--> Track A   A1 TSUBASA (2-3 d) --> A2 BRAID + ThinBRAID (6-8 d)
   |
   +--> Track B   B1 ParCorr/CSZ (3-4 d) --> B2 StatStream (5-7 d) --> B3 CorrJoin (5-7 d)
```

**Total: roughly 30-40 focused days** (was 25-32 before ThinBRAID and the dataset work), with the spread almost entirely in B2, B3 and 0f.

Rationale for this order:

- **TSUBASA first** because it is the cheapest way to prove the Pattern A template generalizes
  beyond FilCorr, and being exact it cannot quietly half-work.
- **BRAID second on Track A** because it is the only **evaluated** lag peer we do not already
  have. FilCorr is the other one (its Figs. 6-7 sweep `lag = 0, 100, 250`) and it is already
  built, so with BRAID the lagged comparison has both published peers; without it, CorrTrack's
  lag claim is tested against one method whose weakness (band-limiting) is a different one
  from BRAID's (approximating the lag value). StatStream's lags are specified-not-evaluated and
  Cole-Shasha-Zhao's are claimed-only, so neither substitutes. If anything gets cut, it must
  not be this.
- **ParCorr before StatStream** because it validates Pattern B on the arm where code already
  exists, and it builds the grid that StatStream then reuses.
- **CorrJoin last** because the incremental-SVD decision benefits from having the other two
  pruning arms measured first.

**Fallback if time runs short**: Track A complete plus B1 and B2. That gives **both**
evaluated lag peers (FilCorr already built, BRAID from Track A), a pruning peer from the same
lineage, the closest structural peer, and two exact baselines.
CorrJoin then goes in related work with the omission stated plainly. Cutting from the end of
Track B is much cheaper than cutting anything in Track A.

---

## 5. What makes the results comparable

The user's constraint. Six mechanisms, listed from strongest to weakest:

1. **One harness.** Same I/O, windowing, artifact writing, monitoring and metrics for every arm.
   That is what the `baseline_mode` and backend axes buy us and why nothing here runs as an
   external system (comparison plan §0).
2. **One exact-validation kernel.** Every pruning arm routes its survivors through the same
   `validate_corr_rows` Cython path, so **only candidate generation differs in implementation
   tier**. This is the specific answer to "your reimplementation of my method was worse than mine".
3. **Counters before seconds** (§0c). `total_candidates` and `tested_candidates` at equal recall
   are implementation-independent; wall-clock is not. Publish both, and where they disagree, that
   disagreement *is* the implementation-tier effect, made visible instead of hidden.
4. **Tuning -- DECIDED 2026-09-16, IMPLEMENTED 2026-09-17** (`abaca/tune_competitors.py`, log entry (e)): strength-2 covering array (130 rows over CSZ's 2,080-setting grid), coordinate refinement, block bootstrap with the 90% lower bound at TARGET_RECALL, on `prepare_training_data`'s span. Consumed by `abaca/nway_compare.py --competitor-params`. Campaign manifest: `abaca/campaign_competitors.py`. All **competitor** arms are tuned by Cole-Shasha-Zhao's
   published protocol: two-factor combinatorial design over the parameter space, local
   neighbourhood refinement, then bootstrapping for out-of-sample robustness, to the **same
   target recall of 0.95** on the **same held-out calibration span**. **CorrTrack keeps its own
   proxy-anchor hyperopt**, which is already a calibrate-to-target-recall procedure of comparable
   rigour and has been extensively optimized; re-tuning it under a different protocol would
   change a well-characterized system for no gain.
   **The asymmetry must be disclosed and defended, not hidden.** Two mitigations: (i) both
   protocols are described side by side in the methods section, with the shared target and
   shared calibration span made explicit; (ii) **run CorrTrack once under the CSZ protocol as a
   robustness check** and report that the two procedures land within noise of each other. That
   is cheap (this project already owns the OA and Sobol design-of-experiments machinery) and it
   pre-empts the obvious reviewer objection ("you tuned yours differently") with a measurement
   rather than an argument. If they do *not* agree, that is a finding about tuning sensitivity
   and must be reported.
5. **No arm gains a capability its paper lacks** (§0d), enforced in code rather than by
   convention.
7. **Every arm reports its DESIGNED output (user, 2026-09-18).** ParCorr, CSZ and CorrJoin verify every
   candidate exactly by design (ParCorr step 5, CorrJoin Alg. 1 line 14), so their survivors going
   through the shared exact kernel is their own method (only the kernel's speed is shared); TSUBASA and
   FilCorr are exact by construction; BRAID / ThinBRAID report their interpolated / sketched CCF values
   (approximate, no validation), which is what our Pattern-A port does. **StatStream** is the one method
   whose published output is approximate: pairs are reported when the correlation approximated from the
   per-basic-window digests exceeds T - t (section 3.4, Table 2 precision 0.9765 to 0.9947, recall
   0.9987 to 1.0). Until 2026-09-18 our StatStream arm validated its survivors exactly, a favour it does
   not have; now `statstream_report="approx"` (default) applies the paper's rule inside the harness
   (reproduced: S0.85 t=0.0005 -> precision 0.9925, recall 0.9991), `statstream_report="exact"` keeps
   the validated variant, labelled. The `candidate_precision` column stays the measure of what a
   monitor would inherit from an unvalidated candidate stream; StatStream's own reported precision
   (< 1) and BRAID's approximate values are the published-output evidence for the statement that
   CorrTrack's exact validation is what makes a monitoring stage meaningful.
6. **Both sides of every axis reported**: density {0.5%, 2%, 5%, 20%}, chosen to span the real sets
   (0.05% to 45% measured); **raw levels AND first differences for every cell** (raw: robustness to
   nonstationarity and its spurious correlation; differenced: robustness to uncooperative data); cooperative
   (prices) *and* uncooperative (returns), which is one transform on data already in hand;
   lags both at basic-window multiples (favours StatStream) and arbitrary (favours CorrTrack).
   Reporting one side of any of these would be cherry-picking of exactly the kind already
   forbidden for density.

---

## 6. Predictions recorded in advance

Writing these down before measuring is what makes the eventual numbers credible, and each one
also functions as a bug detector.

| arm | prediction | if it fails, suspect |
|---|---|---|
| TSUBASA | close to `exact_stomp`; both lose to CorrTrack as `m` grows | the per-pair sketch store, not the algorithm |
| BRAID | loses badly at `m=500+` (no pruning, `O(k^2)` pairs); competitive at small `k` | our port, if it wins at large `k` |
| FilCorr | competitive at `m=500`, losing as `m` grows -- their own crossover against ParCorr is ~700 streams | whether our port is band-limited enough to be their method |
| ParCorr/CSZ | wins on sparse data, ties on dense (the CorrJoin 20% finding, our own 17.6% tie) | the vote threshold, if recall is low |
| StatStream | **degrades sharply on returns while holding on prices** | nothing: this is the thesis of §5 item 6 |
| CorrJoin | needs `m > 100`; speedup capped at `1/r1` | a measured speedup above `1/r1` is a bug |

The cross-cutting one worth stating loudest: **StatStream, FilCorr and BRAID all bet that energy
concentrates at low frequencies, and CorrTrack does not.** Three independent papers
(Cole-Shasha-Zhao Figs. 2 and 4; FilCorr §I on white-noise seismic traces; TSUBASA §4.1 on
climate data needing every coefficient) support that this bet fails on uncooperative data. If
our returns-versus-prices runs do not show it, the port is wrong before the thesis is.

---

## 7. Files that will change

| file | change |
|---|---|
| `library_corrtrack_parallel.py` | 2 new `Candidates_BF_*` classes; 2 `run_bf_*` wrappers; dispatch at `:9129`; `_resolve_baseline_mode` at `:1929`; `_VALID_DATA_REPRESENTATIONS` at `:1738`; `_VALID_CANDIDATE_BACKEND_AXIS` at `:1745`; `sketch_norm` path at `:4033`; new representation and backend implementations |
| `corrtrack_run_bruteforce.py` | `--baseline-mode` choices; per-arm parameter flags |
| `experiment_run_exec_param.py` | documented defaults per arm |
| `test_stable_reproduced_changes.py` | `_assert_competitor_contract` plus 3 tests per arm (degenerate-exact, published-parameter, counter contract): **about 18 new tests**, 122 -> ~140 |
| `abaca/*.py`, `abaca/*.oar` | generalize `fourway_compare.py` to N-way (it already loops over modes) |
| `docs/`, `tasks/` | per the project handoff rule |

---

## 8. Open decisions to make before coding

1. ~~**Negative-correlation policy** (§0d)~~ -- **DECIDED 2026-09-16, refined by the user:**
   primary at `neg_corr=False`; second labelled run at `neg_corr=True` where arms that can be
   enabled without designing their algorithm are **enabled, reported and disclosed as ours**,
   and only arms that would need a new candidate mechanism report N/A. Four-class per-arm tag
   written into the result rows. FilCorr confirmed `enabled_by_us`; ParCorr/CSZ is the only
   N/A. **Fully decided.** See §0d.
2. ~~**CorrJoin SVD**~~ -- **DECIDED 2026-09-16: recompute the `m x ks` SVD per window.** It
   is a thin `O(m * ks^2)` decomposition, negligible per window; my "research-grade incremental
   SVD" framing was carried from the pre-paper draft and was an overstatement. Follow-ups: check
   the paper's `cSVD` term when the PDF is available; ask the authors for the R code. See B3.
3. ~~**ThinBRAID**~~ -- **DECIDED 2026-09-16: build it.** The battery runs to 2,000 series, past
   the ~1,000 crossover, and at 2k plain BRAID's per-pair store is ~1.5 GB versus ThinBRAID's
   ~12 MB. A `thin=True` mode of the BRAID class, +2-3 days on Track A. See A2.
4. ~~**CSZ tuning protocol**~~ -- **DECIDED 2026-09-16: adopt for all competitor arms; CorrTrack
   keeps its own optimized proxy-anchor hyperopt.** Same 0.95 target and calibration span for
   both. Asymmetry disclosed, and defused by running CorrTrack once under the CSZ protocol as a
   robustness check. See §5 item 4.
5. ~~**Competitors' datasets**~~ -- **DECIDED 2026-09-16: add every obtainable one.** Registry
   plus per-set fetch scripts in a new phase 0g; Motes and Yellowstone first among the real
   sets, since they fill regimes we currently lack. Licensed sets (TAQ, CRSP) replaced by
   `sp500` with the substitution stated. See §0g.

**All five §8 decisions are now made. Nothing blocks phase 0a.**

---

## 9. What running to 2,000 series changes

Recorded because three arms cross a threshold between 500 and 2,000:

| arm | at `m = 500` | at `m = 2000` |
|---|---|---|
| TSUBASA sketch store | ~14 MB | ~224 MB (Abaca yes, WSL no) |
| BRAID rolling per-pair store (70 probed lags x m^2) | ~22 MB (measured at m=200) | **~2.2 GB**; use ThinBRAID (~0 MB rolling state, measured) |
| FilCorr vs pruning | competitive (below their ~700 crossover) | **should lose clearly** to CorrTrack and ParCorr; if it does not, suspect the pruning arms |
| CorrJoin | above its `m > 100` floor | comfortably in its regime; the `1/r1` speedup ceiling matters here |
| ParCorr/CSZ sketch size | `r ~ 8 log(n)/eps^2` | grows only logarithmically; `r = 60` still reasonable |

**BRAID and `window_step` (entry (k)):** the harness lag grid is `window_step`-spaced, so at
step=12 exact_stomp evaluates 15 lags while BRAID probes 70. BRAID's published saving is against
an every-lag baseline. **The lagged comparison must include a step=1 configuration**, or BRAID
is asked to beat a baseline that already skips 90% of the lags.

The FilCorr row is a second prediction to record: its own paper places the crossover against
ParCorr at ~700 streams, so at 2k the unpruned methods should all be behind. **Nothing in the
plan needs restructuring for 2k; it changes which variant runs (ThinBRAID) and where (Abaca).**
