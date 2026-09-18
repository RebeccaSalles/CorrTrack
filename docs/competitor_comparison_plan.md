# Plan: comparing CorrTrack against the related-work methods

Status: **plan only, nothing implemented** (rev. 2026-09-16). §2 (ParCorr), §3 (CorrJoin),
§4 (StatStream), §4a (Cole-Shasha-Zhao) and §5a (BRAID/ThinBRAID) are all paper-authoritative;
§10.1 and §10.2 items 1, 4, 5 and 7 are closed.
**The capability matrix (§5b.5) is now graded by evidence tier (§5b.6)**: a capability a paper
specifies but never evaluates is recorded as such, not as a capability. This was the user's
correction and it changes several rows.
ParCorr's cell size is **decided** in §2.3: the v1 `_compute_base_cell_size` formula is not used.
**All primary sources are now read** (FilCorr, TSUBASA and the StatStream technical report
obtained 2026-09-16). No `[?]` cells remain in the capability matrix. See §10.4.
Follows on from the four-way comparison already done (bruteforce / exact_stomp / filcorr /
CorrTrack -- see `docs/implementation_log.md`'s 2026-09-12 entry).

**The engineering plan lives in `docs/competitor_implementation_plan.md`** (rev. 2026-09-16):
files, classes, signatures, ordering and tests. This document stays the *what and why*; that one
is the *how*.

Scope: CorrJoin, StatStream, ParCorr, TSUBASA, **BRAID** -- **all five to be reimplemented in
this codebase**, per the governing principle below.

---

## 0. Governing principle: uniform reimplementation

FilCorr, already in the benchmark, is a Claude reimplementation inside this codebase (ported
from the colleague's `feat/v2-engine` work, then brought to implementation-tier parity -- see
the 2026-09-11 (d) entry). **Every other competitor must be built the same way**: reimplemented
here, against the same I/O, windowing, exact-validation kernel, monitoring and metrics.

Rationale (this replaces the earlier draft's "run the authors' own code" recommendation, which
was wrong for this project):
- Running some arms as the authors' original systems (ParCorr on Spark, TSUBASA's Go/Python
  repo) and others as in-codebase ports makes the arms **non-comparable in the opposite
  direction** -- differences would reflect runtime, language and framework, not algorithms.
- ParCorr's published implementation is Spark-based and far heavier than the mechanism it
  demonstrates; running it would measure Spark, not ParCorr's candidate search.

**Consequence: reference implementations become reading references, not baselines.** We read
them to get the algorithm right; we never time against them.

**What this costs us, stated honestly:** we lose the "authors' own code" cross-check on
faithfulness. That makes the per-port verification anchors (§6.5) the load-bearing safeguard,
not an optional extra.

---

## 1. What each method actually is (verified against papers/repos, not memory)

| method | venue / authors | mechanism | exact? | lags? | reference implementation |
|---|---|---|---|---|---|
| **StatStream** | VLDB 2002, Zhu & Shasha (NYU) | DFT sketch (first `n` Fourier coefficients of the normalized series) + **grid** over the bounded DFT feature space; correlation reduced to Euclidean distance; incremental DFT; basic-window/sliding-window model **(the origin of CorrTrack's own model)** | approximate, but the **grid filter has no false negatives**; error comes only from the DFT approximation | **specified, not evaluated** (§5b.6): full algorithm + a named system parameter `T_M`, but **no lag experiment in the paper**. Lags in multiples of the basic window | none found |
| **FilCorr** | ICDM 2020, Zhong, Souza, Mueen (New Mexico) | **band-limited FFT coefficients maintained incrementally** over the sliding window; correlation recovered by Parseval in `O(B)`, `B` = band width. Filtering and correlation merged into one step | **exact** (Parseval identity) | **yes, evaluated.** Reports `max` correlation over the lag window per pair | code on the authors' site |
| **Cole-Shasha-Zhao** | KDD 2005, Cole, Shasha, Zhao (NYU) | **random-projection sketch + partition into groups + one grid per group + "close in a fraction `f` of groups"**. Structured random vectors + convolution for cheap incremental sketching | approximate; tuned to recall >= 0.99, precision >= 0.02 | **defined in the problem statement, not evaluated** (§5b.6) | none found; NYU TR referenced |
| **ParCorr** | DMKD 2018, Yagoubi, Akbarinia, Kolev, Levchenko, Masseglia, Valduriez, Shasha (Inria/LIRMM **Zenith** + NYU) | **random-projection sketch** + grid over sketch **subspaces**; candidate = co-occurrence across enough subspace cells. **This mechanism is Cole-Shasha-Zhao's (2005), parallelized** -- see §4a.0 | approximate (~95% recall / 100% precision reported) | **synchronous only** (paper's own future work) | Spark -- deliberately not used |
| **CorrJoin** | SIGMOD 2023 (PACMMOD 1(4):235), Nikoo, Böhlen, Helmer (UZH) | **PAA + SVD** complementary dimension reduction + **double filtering** (bucketing, then Euclidean) | approximate | **synchronous only** (paper verified, §3.1) | **original in R** (authors); third-party **Go** (Apache-2.0) and **Python** (UZH thesis) ports -- see §3 |
| **TSUBASA** | SIGMOD 2022, Xu, Liu, Nargesian (Rochester) | per-basic-window **mean, standard deviation and per-pair correlation**, combined by their Lemma 1 into **exact** Pearson over *arbitrary* query windows; embarrassingly parallel | **exact** | **none.** Same window for both series; pruning is their own future work | Go, paper only |
| **BRAID / ThinBRAID** | SIGMOD 2005 + **TKDD 2010** (Sakurai, Faloutsos, Papadimitriou; NTT + CMU + IBM) | hierarchical **window-average smoothing** + **geometric lag probing** + cubic-spline interpolation of the CCF. ThinBRAID adds random-projection sketches to cut *update* cost to `O(k)` | approximate (in the **lag value**, ~1% relative error) | **yes, evaluated** -- its entire contribution, measured on 6 datasets (§5a.2) | none found |

---

## 2. ParCorr -- cheapest, and partly already written

**The user implemented ParCorr's candidate-search strategy as the first version of CorrTrack.**
That machinery still exists in the sibling snapshots: `corrtrack_release_v1.0` and
`corrtrack_release_backup2` retain the `flat` / `bptree` / `sorted_arrays_bs` backends and the
`cell_size` / `grid_max` Euclidean-radius grid parameters that were stripped out of `dev` when
it was narrowed to `lsh_sign_dot` / `lsh_hamming_exact`. Starting point, not a blank page.

### Three faithfulness constraints (user-supplied; a port that ignores these is not ParCorr)

All three are now **confirmed against the DMKD 2018 paper** (obtained 2026-09-15 from the
HAL-LIRMM open mirror,
`https://hal-lirmm.ccsd.cnrs.fr/lirmm-01886794/file/ParCorr__DMKD2018.pdf`).

1. **Synchronous only -- no lag search.** ParCorr compares co-temporal windows. Extending it
   to lags would be our invention, not ParCorr, and must not be presented as ParCorr. This
   settles the lag-protocol question in §6.1.
   **The paper says so itself**, in its own future-work list: extending ParCorr "to cases where
   we want to discover delayed correlations" is listed as item 2, with the note that it "does
   require adjustments because the normalization transformation may change." That is a citable
   admission from the authors, and it is worth more than our assertion: it also describes
   precisely the difficulty CorrTrack solves, which strengthens the novelty claim.
2. **Normalization happens BEFORE sketching** -- z-normalize the raw window, then project.
   This is a real divergence from current CorrTrack, verified in the code: `sketch_norm` is
   hardcoded to `"mean_l2"` (`library_corrtrack_parallel.py:4033`, with the note that `"z"`
   was replaced project-wide in the 2026-07-10 entries), i.e. CorrTrack normalizes **the
   sketch** so that the dot product equals cosine. ParCorr normalizes **the window**. The port
   needs the raw-window normalization path restored for this arm specifically.
3. **`window_step` must equal `basic_window`.** A structural constraint of ParCorr's sliding
   model. The comparison grid must either respect it, or exclude ParCorr from configurations
   it cannot express -- and say so, rather than silently running a modified ParCorr.

Constraint 3 has a direct consequence for the benchmark design: the current default
(`window_step=12`, `basic_window=12` -- these already match, see the `ws168_step12` run
folders) is compatible, but any sweep that varies `window_step` independently of
`basic_window` puts ParCorr out of scope for those cells.

### 2.1 Paper-authoritative parameters (extracted 2026-09-15)

| symbol | meaning | paper's value | notes |
|---|---|---|---|
| `r` | sketch size = number of random ±1 vectors | **60** in the paper's worked example (§4.3 step 1) | theory: `r = 8·log(n)/ε²` (JL, Lemma 1), so it scales with series count, not window length |
| `k` | dimensions per subvector, i.e. per grid | **2** | "if r is 60 and k is 2, then the partition would be 0,1 2,3 4,5 ... 58,59". Number of grids = `r/k` = **30** |
| `f` | fraction of grids that must co-locate a pair | **0.7** (Table 4 default) | **this is the tuned knob**, see below |
| `w` | sliding window size | 500 | |
| `b` | basic window size | 20 | with `w=500` that is 76 windows over 2000 values |
| `T` | correlation threshold | 0.7 default; also 0.8 and 0.9 | |

**How `f` is calibrated (paper §5.3), and this matters for §6.2:** "We calibrate the fraction
`f` ... by using a small sample database. We increase `f` until reaching the desired recall
(e.g. 0.95) on the small sample, and then we use the found fraction (0.7 in the case) in our
experiments on big datasets." **This is the same calibrate-to-target-recall-on-a-sample
protocol CorrTrack uses** (`target_recall=0.95`, proxy-anchor hyperopt). So §6.2's equal-recall
protocol is not something we are imposing on ParCorr from outside; it is ParCorr's own
published tuning procedure, which removes the main fairness objection to it.

**Reported results to reproduce (§6.5 anchor):** 100% precision **by construction** (step 5
verifies every candidate explicitly), recall >90% for both Euclidean and Pearson at `T=0.7`,
>96% at `T=0.8`, >95.7% at `T=0.9`. The paper also states that candidate precision below 1%
is acceptable to them, which is a useful calibration for how aggressively ParCorr over-generates
relative to CorrTrack.

**ParCorr does NOT handle negative correlation** (user-confirmed, 2026-09-16). CorrTrack runs
`neg_corr=True`, so the comparison must either run ParCorr on positive correlation only and say
so, or report the asymmetry explicitly. **Do not silently give ParCorr an `abs()` it never had**
-- that would be inventing capability, the mirror of the error recorded in §10.3.

**Two things the paper does NOT do, both easy to get wrong:**
- **No neighbouring-cell search.** "Improving the recall perhaps by expanding the grid search
  to neighboring cells" is future-work item 1. A port that checks neighbours is a better
  algorithm than ParCorr, and must not be called ParCorr. **Note this is a genuine algorithmic
  difference from CorrJoin, which does check neighbouring buckets** (§3.1 step 4) -- worth a
  sentence in the paper, since the two methods otherwise look similar.
- **No grid cell size is specified anywhere in the paper.** It is absorbed into the calibration
  of `f`. See the audit below for why this matters.

### 2.2 Snapshot audit: what the v1 code actually still contains (checked, not assumed)

The starting-point claim above needed verifying rather than trusting. Result: **mine
`corrtrack_release_v1.0`, not `corrtrack_release_backup2`**, and expect to re-enable machinery
rather than merely restore a file. The ParCorr-defining parts are present but **disabled**:

| finding | location | consequence for the port |
|---|---|---|
| `grid_dimension` **is** ParCorr's `k`, and `n_grids = n_vectors // grid_dimension` **is** `r/k` | `v1.0/library_corrtrack_parallel.py:2188`, with the comment "divides n_vectors (sketch size) for 2D grid" at `:5368` | the subspace decomposition is real and matches the paper. Reusable as-is |
| `self.freq_threshold = 0`, hardcoded, with the comment "Retain the constructor argument for old callers, but **disable frequency gating**" | `v1.0:2233` | **the co-occurrence vote is switched off.** That vote is ParCorr's entire contribution. Must be re-enabled and wired to `f` |
| `full_vector_candidates` forces `grid_dimension = n_vectors` and `n_grids = 1` on the default path | `v1.0:2182-2185` | the default path collapses the grids to one full-dimensional space, which is not ParCorr. The port must take the `else` branch |
| surviving vote code uses `required_hits = max(1, int(self.n_grids))` | `v1.0:4647` | that is **`f = 1.0`** (hits in *every* grid), not the paper's `f = 0.7`. Reusing it as-is would systematically under-recall |
| `backup2` hardcodes `full_vector_candidates = True`, `n_grids = 1`, `freq_threshold = 0` | `backup2:3355-3357, 3402` | further from ParCorr than v1.0. **Do not use as the source** |
| `_compute_base_cell_size = sqrt(2·(1−T)) / sqrt(n_vectors)` and `grid_max = min(1, 3/sqrt(n_vectors))` | `v1.0:1315, 2230` | **these are the user's own derivations, not ParCorr.** **DECIDED: do not use them for the ParCorr arm** (see §2.3) |

### 2.3 DECIDED: how the ParCorr grid cell size is set

**The v1 `_compute_base_cell_size` formula is not used for this arm.** It is the user's own
derivation, the paper contains no such rule, and carrying it in would mean publishing "ParCorr"
numbers that depend on an unpublished design choice of ours. Ruling it out removes the single
largest faithfulness risk in this port.

**Instead, follow ParCorr's own published methodology**, which is a *calibration procedure*, not
a formula (§2.1): tune on a small sample until the target recall is met, then freeze. Concretely:

1. Fix `f = 0.7` and `k = 2` at the paper's values.
2. On a held-out calibration sample, search the cell size and take the **largest** value that
   still reaches **target recall 0.95** -- largest, because the cell size is the pruning knob and
   a larger cell means a weaker filter, so this is the setting that respects the recall
   constraint while giving ParCorr its best speed. Report the achieved recall beside the target,
   per §6.2.
3. Record the chosen cell size as a **calibrated hyperparameter**, stated in the paper as such.

This is defensible in a way a formula is not: every arm is then tuned by the procedure its own
authors published, ParCorr included, and the tuning is visible rather than baked into a
constant. It also keeps `f` at a published value instead of letting two knobs move at once.

**Better still, the gap is now filled by a published source.** ParCorr's candidate search is
Cole-Shasha-Zhao's (§4a.0), and **CSZ does publish the missing knob**: their **distance
multiplier `c`**, where a pair is a candidate if its sketch subvectors are within `c x d` in a
fraction `f` of groups, with `d` the correlation-derived distance. Their published search range
is `c` in {0.1, 0.2, ..., 1.3} and `f` in {0.1, ..., 1.0}, tuned by two-factor combinatorial
design plus local refinement (§4a.1). **So the calibration above should sweep `c`, not an
unnamed "cell size"**, over CSZ's published range. That makes the ParCorr arm's tuning entirely
traceable to published work by the same research line, with nothing of ours in it.

If the calibration turns out to be unstable (no cell size meets the target, or the choice swings
wildly across samples), that is itself a reportable property of ParCorr's grid and must be
reported rather than patched with a formula.

Note for contrast: **StatStream does specify a cell size** (diameter `ε = sqrt(1−T)` on a
`sqrt(2)`-diameter bounded cube, §4.1). That rule is StatStream's and is tied to the DFT
feature space's bound; it is not transferable to ParCorr's unbounded random-projection space,
and must not be borrowed across arms.

### Integration
`data_representation="sketch_proj"` (**already exists** -- random projection *is* ParCorr's
sketch) + a new `candidate_backend="parcorr_grid"` implementing the subspace-grid
co-occurrence voting. Only the voting rule is genuinely new code, and §2.2 shows a disabled
version of it is already in the v1.0 snapshot to work from.

---

## 3. CorrJoin

### 3.0 Source hierarchy -- the paper is the specification, everything else is a witness

**The authority is the original paper**: Alizade Nikoo, Böhlen & Helmer, *Correlation Joins
over Time Series Data Streams Utilizing Complementary Dimension Reduction and Transformation*,
PACMMOD 1(4):235, Dec 2023. It is **CC-BY licensed open access** (`10.1145/3626722`).
**Obtained and extracted 2026-09-15** -- §3.1 below is now the paper's specification, not a
sketch, and it **corrected two things the derivative sources had wrong** (§3.2).

Secondary sources, in descending trust:

| source | language | role | known defects |
|---|---|---|---|
| **the paper** (above) | -- | **specification** | none -- **in hand, §10.1 checklist closed** |
| **original implementation** (Alizade Nikoo et al.) | **R** | authors' own reference, **in hand** (`docs/reference_code/corrjoin_authors_R/`, six scripts incl. their TSUBASA and all baselines) | `checkVal <- 9` bucket-scan truncation not in the paper; PAA and SVD recomputed per window where the paper describes an `O(h+k)` PAA update |
| `codeberg.org/kpaschen/corrjoin` (was `github.com/kpaschen/corrjoin`) | Go, Apache-2.0 | independent implementation witness | third-party; framed as a Prometheus tool; fidelity to the paper unverified. `lib/` does contain `paa`, `svd`, `buckets`, `comparisons`, `settings` ("all the parameters for the corrjoin algorithm") |
| `github.com/felixmerz00/bachelor-thesis` + its thesis PDF | Python | **derivative, partial** -- useful for orientation and for its vectorization tricks | omits lags, omits negative correlation, drops the incremental correlation update, and is internally inconsistent on the ε thresholds (§3.2) |
| `drive.google.com/drive/folders/1skrE2x2DMgIms5lZR04kiOlLvvna2zNs` | data | **the original paper's own datasets** -- synthetic (5000x4080 random walk), chlorine (4830x2040), gas (5120x3600), stock (3878x1259) | none; this is the validation asset |

The dataset row is the most valuable: it makes CorrJoin the **one competitor where we can
reproduce the original paper's published claims on the original paper's own data** -- the
strongest faithfulness check available (§6.5).

### 3.1 Algorithm specification (from the paper -- authoritative)

Pipeline: `normalize -> PAA -> SVD -> bucketing filter -> Euclidean distance filter -> exact Pearson`

**The identity the whole method rests on** (paper §2.1): for centred unit-norm vectors,
`corr(x,y) = 1 − ½·d²(x̂,ŷ)` (Eq. 4), so `corr ≥ T  ⟺  d(x̂,ŷ) ≤ sqrt(2(1−T))`. Under a
`k`-dimensional PAA reduction, `d(X̂,Ŷ) ≤ sqrt(k/n)·d(x̂,ŷ)` -- the reduced distance is a lower
bound (PAA shrinks distances), which is what makes the filters false-negative-free.

1. **Normalize before reducing** (like ParCorr): `x̂[i] = (x[i] − x̄) / sqrt(Σ(x[i] − x̄)²)` --
   the centred **unit-norm** form. **Confirmed against the paper.** **Note this is the same
   normalize-the-window-before-reducing that ParCorr needs** (§2, constraint 2) -- so both ports
   share one new code path, not two.
2. **PAA twice** on the normalized window: to `ks` dims (feeding SVD) and to `ke` dims
   (feeding the second filter). Requires `n` divisible by both `ks` and `ke`.
3. **SVD** on the `ks`-dim representation -> `kb` dims, ordering dimensions by variability and
   keeping the most variable. Counteracts PAA's distance-shrinking side effect.
4. **Bucketing filter** (Alg. 2): `kb`-dimensional grid of bucket width `ε₁`; compare within
   each bucket and against neighbouring buckets -> candidate set `C₁`. No false negatives.
5. **Euclidean distance filter**: on the `ke`-dim PAA representation, keep pairs with
   `d ≤ ε₂` -> refined set `C₂`.
6. **Exact Pearson** on full-length raw windows for `C₂` only; report **`|corr| ≥ T`**
   (Alg. 1 line 14 -- negative correlation is included, as CorrTrack needs).

**The ε thresholds (Alg. 1 line 1 -- authoritative, and this settles the thesis's
self-contradiction):**

```
ε₁ = sqrt( 2·ks·(1 − T) / n )      # bucket width, on the SVD-reduced space
ε₂ = sqrt( 2·ke·(1 − T) / n )      # Euclidean filter radius, on the ke-dim PAA space
```

**Both divide by `n`, and ε₁ uses `ks` -- not `kb`.** Both thesis variants were wrong: §3.2.1's
`kb` in ε₁ and §3.2.2's missing `/n` in ε₂. Each error inflates or deflates the pruning radius,
so either would have produced CorrJoin numbers that are quietly wrong in *both* recall and
speed. This was the highest-priority item on the old §10.1 checklist and it is now closed.

**Incremental correlation across strides** (paper Eq. 2): maintained via the five running sums
`s1..s5` (`Σx, Σy, Σx², Σy², Σxy`) -- the same sufficient-statistics form `exact_stomp` already
uses, so the port reuses machinery rather than inventing it. PAA mean vectors update in
`O(h + k)` per stride. The paper reports (Fig. 16a) that stride has only a **slight** effect on
runtime -- note this is the opposite of the thesis's finding, which was an artefact of
recomputing correlation from scratch.

Parameters: `m` series, window `n`, stride `h` (`h < n` for sliding), threshold `T`, and
`ks`/`ke`/`kb`. **Paper-identified runtime optimum: `ks=15, ke=30, kb=3`** (confirmed). Gain
exceeds reduction/filtering overhead only for roughly **`m > 100`**, and speedup is
**upper-bounded by `1/r1`** where `r1` is the fraction of pairs surviving the first filter --
a useful sanity ceiling when validating our port's measured speedup.

### 3.2 What the paper corrected, and two findings that matter for CorrTrack's own paper

**Corrections to what was previously written here:**

1. **CorrJoin has no lag support.** The paper is entirely synchronous: Alg. 1 compares all
   series at the same window index `α`, exactly as the thesis pseudocode did. The earlier
   "time-delayed correlation over any time delay" claim traced to a search-engine summary of
   the abstract, not to the paper. **Consequence: ParCorr, StatStream *and* CorrJoin are all
   synchronous; only CorrTrack and FilCorr do lags.** This removes the possibility raised
   earlier that CorrJoin, rather than CorrTrack, would be the closest peer on the lag axis --
   **lag search remains a genuine CorrTrack capability differentiator against all three.**
2. **The ε formulas**, as above (§3.1).

**Two findings of direct value to our own paper** (both are independent corroboration from a
SIGMOD 2023 paper, which is worth more in the write-up than our own measurement alone):

- **Density dependence (paper §6.3.3, Fig. 15):** "once the rate reaches around 20%, the
  difference in performance between CorrJoin and IncP^PAA is not discernible anymore." This is
  independent confirmation of our own 2026-09-12 finding that CorrTrack ties brute force at
  ~17.6% density and wins 3.66x at ~2%. **Pruning-based methods stop paying off at high
  correlation density is a property of the class, not a weakness of CorrTrack** -- and we can
  now cite a competitor's paper saying so.
- **TSUBASA's position (paper Fig. 10):** the paper measures TSUBASA as having "a similar time
  complexity as the naive IncP since it does not reduce the number of pairwise comparisons,"
  an order of magnitude worse than its own baseline. That corroborates §5's assessment that
  TSUBASA occupies `exact_stomp`'s slot rather than being a pruning rival, and independently
  justifies keeping it ranked *secondary*.

**The paper's own baselines** (for the §6.5 reproduction check): ϵ-kdB tree, **TSUBASA**,
Quickjoin, and IncP^PAA (its main baseline). All implemented from scratch in **R** by the
authors, run on a 2.2GHz 6-core Mac with 16GB -- i.e. **the paper itself follows exactly the
uniform-reimplementation policy adopted in §0**, which is a citable precedent for our
methodology section.

Also of note: the paper's cost model (§5) decomposes runtime into `cnorm`, `cPAA`, `cSVD`,
`cbkt`, `cfilter`, `ctrue` against `cbase = O(m²n)` -- structurally the same three-phase
decomposition proposed in §5b.1, which is further evidence that the reporting scheme there is
what this literature expects.

**Correction 2026-09-16 on the SVD**: earlier revisions called incremental SVD across sliding
windows the "remaining genuine difficulty". That was carried from the pre-paper draft and is an
overstatement. The SVD acts on the `m x ks` matrix with `ks = 15`, a thin `O(m ks^2)`
decomposition that costs microseconds to milliseconds per window even at `m = 2000`.
Recomputing it per window is negligible and is the decided approach (implementation plan B3).
What the paper specifies as incremental is the correlation (Eq. 2) and the PAA means, which is
where stride behaviour comes from. Open follow-ups: confirm against the `cSVD` cost term, and
ask the authors for the R code (§10.2 item 3b).

### Integration
`data_representation="sketch_paa_svd"` (new) + `candidate_backend="corrjoin_double_filter"`
(new, two-stage: grid bucketing on `kb` dims, then Euclidean on `ke` dims).

---

## 4. StatStream -- the closest structural peer to CorrTrack, and the most demanding arm

Zhu & Shasha, *StatStream: Statistical Monitoring of Thousands of Data Streams in Real Time*,
VLDB 2002, 358-369. Paper obtained and extracted 2026-09-15; §4 is paper-authoritative.

### 4.0 Correction: StatStream does lags, and more besides

The earlier draft described StatStream as "primarily synchronous" and as a 2002 ancestor whose
comparison value was mostly historical. **Both were wrong, and the paper is unambiguous.** Its
abstract claims, and §3.5/§3.6 deliver, "the efficient computation of time-delayed correlation
over any size sliding window and any time delay." StatStream is a **lagged, pruning,
negative-correlation-aware, persistence-aware** method. It is not a primitive ancestor; it is
the nearest thing in the literature to CorrTrack's overall shape.

This is the **second** capability I asserted as CorrTrack-exclusive and the paper contradicted
(the first was lag support, where the error ran the other way and I wrongly credited CorrJoin).
The process lesson is recorded in §10.3.

### 4.1 Specification (paper-authoritative)

**Three-level time hierarchy** (their §1): timepoint < **basic window** < **sliding window**,
with `w = k·b`. **This is exactly CorrTrack's model, and it originates here** -- which is the
one part of the old "lineage" framing that survives intact.

Symbols (their Table 1): `w` sliding window, `b` basic window, `k = w/b` basic windows per
sliding window, `n` DFT coefficients kept as digests, `N_s` number of streams.

**Normalization** (their §3.6): `x̂_i = (x_i − x̄)/σ_x` with `σ_x = sqrt(Σ(x_i − x̄)²)` -- the
centred **unit-norm** form. **Identical to ParCorr's and CorrJoin's.** The phase-1 shared
prerequisite therefore has **three** consumers, not two.

**The reduction** (Lemma 1): `corr(x,y) = 1 − ½·d²(x̂,ŷ)`. The same identity as CorrJoin's
Eq. 4. All three pruning competitors rest on it.

**The filter radius** (Lemma 2): `corr(x,y) ≥ 1 − ε²  ⟹  d_n(X̂,Ŷ) ≤ ε`, where `d_n` is
Euclidean distance over the first `n` DFT coefficients of the normalized series. So

```
ε = sqrt(1 − T)
```

**No false negatives** (their Theorem 2). Note the contrast worth drawing in the paper: the
factor of 2 that CorrJoin's `ε` carries is absent here, because DFT conjugate symmetry gives
`2·d_n²(X̂,Ŷ) ≤ d²(X̂,Ŷ)` for free. Same identity, tighter bound, different transform.

**Negative correlation** (Lemma 3): `corr(x,y) ≤ −1 + ε² ⟹ d_n(−X̂,Ŷ) ≤ ε`, handled by also
probing the cell at `(−c_1, ..., −c_ĥ)`. **StatStream handles negative correlation**, matching
CorrTrack's `neg_corr=True`.

**Incremental DFT** (Lemmas 4-6): normalized coefficients come from raw ones as `X̂_0 = 0`,
`X̂_i = X_i/σ_x`. Per timepoint, `X_m^new = e^{j2πm/w}(X_m^old + (x_w − x_0)/sqrt(w))`. Per basic
window (Lemma 6), a batch update requiring the `n` per-basic-window digests
`ζ_m = Σ_{i=0}^{b−1} e^{j2πm(b−i)/w}·x_i`. **Nothing is recomputed from scratch and expiring
data is never revisited.**

**Bounded feature space** (Lemma 7): `|X̂_i| ≤ sqrt(2)/2`, so the `R^{2n}` DFT feature space is a
cube of diameter `sqrt(2)`.

**The grid**: index on the first `ĥ ≤ 2n` dimensions; partition the `sqrt(2)`-diameter cube into
cells of diameter `ε`, giving `(2·⌈sqrt(2)/(2ε)⌉)^ĥ` cells. **Adjacent cells are probed**
(like CorrJoin, unlike ParCorr -- §2.1).

**Lagged mode** (their §3.6, the part the earlier draft missed entirely): `T_M` is a
user-defined maximum lag. Cells *and* hashed streams carry timestamps; the grid is updated
every basic window and **never globally cleared** (in synchronous mode it *is* cleared every
basic window). Stale cells are cleared and stale streams deleted lazily on probe. **Constraint:
in the grid path "the time lag must be a multiple of the size of a basic window."** Arbitrary
sub-basic-window lags are available via §3.5's precomputed table
`W(m,p,d) = Σ_{i=1}^{d} f_m(i)·f_p(b−d+i)`, at `O(k·n²)` per pair versus `O(k·n)` when aligned
(their Theorem 1, Corollary 1).

**"Duration over Threshold"** (their §4): a user parameter requiring correlation to stay above
threshold for a minimum period -- "Has the one hour correlation between two stocks been over
0.95 during the last 10 minutes?" **This is persistence/episode monitoring**, and it invalidates
the old §5b.5 claim that such monitoring is CorrTrack-only. See §5b.5 for the corrected,
narrower claim.

### 4.2 Parameters and published results

| parameter | paper's value |
|---|---|
| `n`, DFT coefficients for the grid | **16** in the speed study; swept over **16, 24, 32, 40** in Fig. 5 |
| DFT coefficients per basic window for curve fitting | **2** (their §5.2; distinct from `n` above -- do not conflate) |
| `w`, sliding window | 1,800 to 7,200 timepoints (0.5 to 2 hours at 1s) |
| `b`, basic window | 0.5 to several minutes |
| `T` | 0.85 and 0.9 |
| post-processing tolerance `t` | 0.001 / 0.0005 (report pairs with approximate corr above `T − t`) |

**Results to reproduce (§6.5 anchor):** precision 0.9765-0.9947 and recall 0.9987-1.0 after
post-processing (their Table 2); grid pruning power (reported pairs / all pairs) roughly
0.01-0.09 depending on `n` and `T` (Fig. 5). **Recall below 1.0 comes from the DFT
approximation in post-processing, not from the grid** -- the grid itself is proven
false-negative-free. That distinction is a good faithfulness probe for our port.

Datasets: synthetic random walk `s_i = 100 + Σ_j (u_j − 0.5)`; NYSE TAQ, 300 heavily traded
stocks at 1-second timepoints. Hardware: 1.5GHz Pentium 4, 128MB.

Cost model: exact is `T = k_0·b·N_s²`; DFT-grid is `T_1 = k_1·b·N_s` (digests) plus
`T_2 = k_2·N_s²` (grid), giving `N_s ≈ sqrt(b/k_2)`. Again a three-phase decomposition,
matching §5b.1.

### 4.3 What this does to the CorrTrack-vs-StatStream framing

The old framing ("quantify what twenty years bought") was comfortable and is no longer
defensible as stated. On paper StatStream already has: DFT sketch, grid pruning with no false
negatives, lag search, negative correlation, incremental updates with no revisiting of expired
data, a duration/persistence parameter, and an embarrassingly parallel decomposition.

**A reviewer will ask what CorrTrack adds. The honest list, to be tested rather than asserted:**
- **Candidate backend**: LSH (`lsh_sign_dot`) and exact Hamming, versus a fixed regular grid on
  the first `ĥ` DFT dimensions. Grids degrade in higher dimensions; that is the mechanism claim,
  and §5b.1's counters can measure it directly at equal recall.
- **Lag granularity**: CorrTrack searches arbitrary integer lags; StatStream's grid path is
  restricted to **multiples of the basic window**, with finer lags costing `O(k·n²)` per pair.
- **Frequency-content dependence**: keeping the first `n` DFT coefficients assumes energy
  concentrates at low frequencies (the paper says so explicitly, citing random-walk and black-noise
  spectra). On bursty or high-frequency data that assumption fails. **This is a testable
  weakness and our density-targeted generator plus the real datasets can probe it.** Note the
  kinship with both FilCorr's band-pass and BRAID's smoothing: three different competitors all
  bet on low-frequency dominance, and CorrTrack does not. That is a coherent, defensible story
  for the paper, and stronger than a speed table.
- **Monitoring**: CorrTrack's episode/attention/anomaly tracking versus StatStream's single
  minimum-duration parameter. Narrower than previously claimed but not empty (§5b.5).

**This is now the highest-value single competitor**, and also the one most likely to embarrass
us. Both facts argue for building it early and reporting it honestly.

### Integration
`data_representation="sketch_dft"` (new) + the grid backend restored from the snapshots (§2.2),
sharing the ParCorr grid work. **The FFT machinery objection in the earlier draft is now stale**:
the FilCorr port added a band-limited FFT path with a persistent per-window cache
(`_band_fft_at`, `_evict_band_fft_cache`), so StatStream's digests can reuse that caching
pattern rather than introducing one.

Retain the **lineage** point, but restated: StatStream is the origin of the basic-window model
CorrTrack uses, so the comparison is a direct ancestor test, not a strawman. Frame it as
"what does a modern index buy over a 2002 grid, at equal recall" and let the counters answer.

---

## 4a. Cole-Shasha-Zhao 2005 -- the missing ancestor, and the source of ParCorr's mechanism

Cole, Shasha & Zhao, *Fast Window Correlations Over Uncooperative Time Series*, KDD 2005,
743-749. Paper obtained 2026-09-16; §4a is paper-authoritative.

### 4a.0 The finding: ParCorr's candidate search was published in 2005

Read CSZ §5.3 beside ParCorr §4.3 and they are the same algorithm:

> **CSZ (2005)**: "Partition each sketch vector `s` of size `N` into groups of some size `g`.
> The `i`th group of each sketch vector is placed in the `i`th grid structure (of dimension
> `g`). If two sketch vectors are within distance `c x d` in more than a fraction `f` of the
> groups, then the corresponding windows are candidate highly correlated windows."

> **ParCorr (2018)**: partition the sketch of length `r` into groups of size `k`, one grid per
> group, candidate if co-located in a fraction `f` of grids.

Same sketch (random +1/-1 projections), same partition-into-subvectors, same one-grid-per-group,
same fraction-of-groups vote, same `d^2 = 2(1 - corr)` reduction. **Dennis Shasha is an author of
StatStream (2002), CSZ (2005) and ParCorr (2018).** ParCorr's own contribution, stated in its
abstract, is the *incremental* sketch update with folded-in normalization and the *parallel*
(Spark) execution, not the candidate-search scheme.

**Three consequences, all of which improve the paper:**

1. **The lineage is real and citable**: StatStream (2002, DFT + grid) -> **CSZ (2005, sketch +
   group-grids + fraction vote)** -> ParCorr (2018, the same, incremental and distributed) ->
   CorrTrack v1 (the user's reimplementation of ParCorr's candidate search) -> CorrTrack.
   **CSZ, not ParCorr, is the true ancestor of CorrTrack's candidate search.** Citing ParCorr
   alone as the origin would be a genuine attribution error in our own related-work section.
2. **It fills the ParCorr cell-size gap** with a published parameter (§2.3).
3. **It is the clearest statement in the literature of the low-frequency assumption** that
   §4.3 identifies as the shared weakness of StatStream, FilCorr and BRAID -- see §4a.2, which
   is the most useful thing in this paper for us.

### 4a.1 Specification and parameters

- **Sketch**: dot products with `d` random vectors in {+1,-1}^m; `D^2(x_hat, y_hat) =
  2(1 - corr(x,y))` on normalized windows, the same identity as StatStream, ParCorr and CorrJoin.
- **Structured random vectors**: each random vector is built by concatenating `nb = sw/bw`
  copies of one random block `u` of length `bw`, each copy signed +/- by a random bit. Dot
  products with `u` are computed once per basic window **by convolution**, and each full sketch
  coordinate is then a sum of `nb` precomputed numbers. Cost: `O(sw/bw)` integer additions and
  `O(log bw)` floating-point operations per datum per random vector, a reported **30x to 40x**
  speedup over the naive sketch update, at negligible accuracy cost (their Fig. 3).
- **Four parameters with published search ranges** (their §5.4): sketch size `N` in
  {30, 36, 48, 60}; group size `g` in {1, 2, 3, 4}; distance multiplier `c` in {0.1 ... 1.3};
  fraction `f` in {0.1 ... 1.0}.
- **Tuning protocol**: two-factor **combinatorial design** (2,080 settings reduced to 130),
  then **local neighbourhood refinement**, then **bootstrapping** to check the setting is
  robust out of sample. Target: **recall >= 0.99** (threshold 0.95), **precision >= 0.02**.
  Accuracy stops improving past `N` around 60.
- Evaluation: CRSP end-of-day prices for 7,861 stocks, plus 10 UC Riverside datasets
  (1,365 to 13,736 series each); `sw=256`, `bw=32`. 1.6GHz, 512MB, language K.

**Their tuning protocol is more rigorous than ours**, and worth adopting rather than merely
citing: combinatorial design plus bootstrapping is a stronger answer to "did you tune the
competitors fairly?" than our current single calibration pass (§6.2). Note we already own the
machinery -- the OA(16,5,4,2) and Sobol sweep infrastructure in this project is exactly this
kind of design-of-experiments sampling.

### 4a.2 The cooperative/uncooperative distinction -- the most useful idea here

CSZ split time series into two classes:

- **Cooperative**: energy concentrates in the first few Fourier/wavelet coefficients (random
  walks, stock *prices*). DFT/DWT/SVD digests work well.
- **Uncooperative**: energy spread over all frequencies, white-noise-like (stock *returns*,
  i.e. `(p_t+1 - p_t)/p_t`). **DFT, DWT and even SVD approximate distances badly** (their
  Fig. 2 and Fig. 4); random-projection sketches remain accurate.

Their summary: "sketches are like B-trees (the default choice) and Fourier Transform approaches
are like bit vectors (better in some cases)." Their Fig. 5 shows neither dominates on speed.

**This is the organising axis for our evaluation, and it comes from the literature rather than
from us.** §4.3 had already identified that StatStream, FilCorr and BRAID all bet on
low-frequency dominance while CorrTrack does not; CSZ names that bet, gives it a vocabulary,
and demonstrates where it fails. Two direct consequences:

- **The evaluation must report cooperative and uncooperative data separately.** Our `sp500`
  pools are *prices* (cooperative, favourable to the DFT-based arms); **returns** of the same
  series are uncooperative and adversarial to them. Both are one transform apart and we already
  hold the data. Reporting only one would be cherry-picking, exactly as §6.4 forbids for density.
- It predicts the shape of the result in advance: **FilCorr, StatStream and BRAID should degrade
  on returns while CorrTrack and the sketch-based arms hold up.** Recording that prediction now
  (as §5a.4 does for BRAID) is what makes the eventual measurement honest.

### 4a.3 Should it be a separate arm?

**No -- fold it into the ParCorr arm rather than building a near-duplicate.** The candidate
search is the same algorithm (§4a.0), so two arms would measure one mechanism twice. Implement
the sketch + group-grid + fraction-vote mechanism **once**, parameterized, and report it as the
**ParCorr/CSZ family**, crediting CSZ as the origin. Then treat the genuine differences as
**ablations within that arm**, which is more informative than a separate row:

| difference | CSZ 2005 | ParCorr 2018 | how to report |
|---|---|---|---|
| random vectors | **structured** (blocks + convolution) | plain | ablation: does the 30-40x update speedup survive in our harness? |
| tuning | `c`, `f`, `N`, `g` by combinatorial design + bootstrap | `f` only, calibrated to target recall | adopt CSZ's protocol for both (§4a.1) |
| normalization | before sketching | before sketching, **folded into the incremental update** | ParCorr's actual contribution; keep it |
| execution | single node | Spark | out of scope for both (§0) |

---

## 4b. FilCorr -- the arm we shipped before reading the paper

Zhong, Souza & Mueen, *FilCorr: Filtered and Lagged Correlation on Streaming Time Series*,
ICDM 2020, 1436-1441. **Paper obtained and read 2026-09-16.** FilCorr has been implemented and
benchmarked in this project since 2026-09-11; §10.4 flagged reading it as the largest
outstanding risk. It was worth reading: the port is sound, but **two deviations from the paper
need to be documented in our results**.

### 4b.1 Specification (paper-authoritative)

Given `N` streams, a band `(fs, ft)` and a maximum lag `l`, compute Pearson correlation for all
pairs over a sliding window up to that lag. **Exact**, by Parseval:

```
Σ_j w_x[j]·w_y[j]  =  (1/2m)( Σ_k |W_x[k]|² + Σ_k |W_y[k]|² − Σ_k |W_x[k] − W_y[k]|² )
```

The contribution is **merging filtering and correlation into one step**. Rather than FFT,
zero the out-of-band bins, IFFT and correlate, FilCorr maintains only the in-band coefficients
and slides them forward directly:

```
W_x,t+1[q] = e^{i2πk/m} ( W_x,t[q] − d·e^{−i2π·0·k/m} + a·e^{−i2π·m·k/m} )
```

with `d` the outgoing and `a` the incoming sample. Cost `O(B)` per update, `B = 1 + m(ft−fs)/f`.

- **Complexity**: FilCorr `O(B + lB)`, worst case `O(m + lm)`; naive `O(m log m + lm)`. Space
  `O(lB)` versus `O(lm)`. Both carry a factor `O(N²)`: **FilCorr does no pruning**, by design.
  Their §I is explicit about why: "Seismic traces are mostly white noise ... stressing the
  algorithms to fall behind the stream quickly. The main reason for the failure of these methods
  is the **data-dependent** pruning, projection, or indexing technique." **FilCorr rejects
  pruning on purpose, to get data-independent throughput.**
- **Parameters**: `fs, ft` (band, set by the application), `f` (sampling rate), `m` (basic
  window), `l` (max lag), `step` (output rate). The paper stresses it is "devoid of sensitive
  parameters": the band is a domain choice, not a tuning knob.
- **Evaluation**: synthetic data (it is data-independent, so this is defensible). Up to 4x more
  sensors than naive. Case study: Yellowstone seismic network, 29 stations (628 pairs), 100Hz,
  20s window, band 3-7Hz, lag 10s.

### 4b.2 Two deviations in our port, both to be disclosed

1. **FilCorr has no negative-correlation handling. Our port adds it.** Their Equation 7 is
   `lcorr = Max(corr(t,ti), corr(ti,t))` over `ti ∈ [t−l, t]`: a maximum of signed correlations,
   never an absolute value. Our `Candidates_BF_FilCorr` inherits `Candidates_BF_ExactSTOMP`'s
   acceptance rule, which applies `|corr| >= threshold` when `neg_corr=True`.
   **This is defensible but must be stated.** As a `baseline_mode`, FilCorr's job in our harness
   is to produce the same ground-truth pair set as bruteforce so recall and precision mean
   anything; applying the harness's uniform acceptance rule is the right call. The correlation
   *values* are identical either way, since the Parseval decomposition is sign-preserving, so
   timings are essentially unaffected and the pair set is a superset. **Write it down as an
   extension of ours, not as FilCorr.**
2. **FilCorr reports one value per pair per timestamp** (the maximum over the lag window); our
   port reports per-lag pairs. The same output-shape mismatch as BRAID (§5a.3 item 1), and the
   same resolution: compare on the common denominator and put the rest in the capability table.

Neither invalidates the 2026-09-12 four-way numbers. Both belong in the paper's methods section.

### 4b.3 Their Table I is a precedent for our evidence tiers

FilCorr's §III contains its own capability matrix against ParCorr, BRAID, COLR-Tree and
StatStream, and its legend reads: **"✓ represents a claimed capability, – represents extendable
capability and × represents unknown."**

That is the same distinction §5b.6 draws, made by a competitor, in a published table. **It is a
direct citation for the methodology**, which is much better than presenting evidence grading as
our own invention. Two further details:

- They mark **StatStream's lagged correlation as `×` (unknown)** and write "None of these
  methods consider lagged correlation after filtering." A second group independently declining
  to credit StatStream with lag support supports our `[S]` grade (§5b.6).
- They mark several cells `–` (extendable but not demonstrated), which is precisely the
  "specified, not evaluated" tier. Our tiers refine their scheme rather than replace it.

### 4b.4 The crossover number, which lands inside our battery

FilCorr's §V-C benchmarks against **ParCorr** as the state-of-the-art baseline, running ParCorr
offline and subtracting Spark startup to favour it. Result: at `lag = 0`, **FilCorr beats
best-case ParCorr up to about 700 streams**, and they recommend FilCorr on a single desktop
below that. At 800 streams, FilCorr still wins when the output rate exceeds 6Hz.

**Our battery runs `m = 500`, just below their crossover.** So the published expectation is that
an unpruned exact method should be competitive with a pruning method at our scale, which is the
same phenomenon as CorrJoin's 20% density finding and our own 17.6% tie. Record the prediction
before running: **at `m = 500` on dense data, FilCorr should be competitive with CorrTrack and
should lose as `m` grows.** If CorrTrack wins easily at 500, check whether our FilCorr port is
band-limited enough to be the method they benchmarked.

---

## 5. TSUBASA -- exact, unpruned, and a direct attack on StatStream's assumption

Xu, Liu & Nargesian, *TSUBASA: Climate Network Construction on Historical and Real-Time Data*,
SIGMOD 2022, 286-295. **Paper obtained and read 2026-09-16**; §5 is paper-authoritative.

### 5.1 What it actually is

Not a correlation join. TSUBASA builds the **complete exact correlation matrix** for a climate
network over a user-chosen query window. Its contribution is a decomposition (their Lemma 1)
that recovers exact Pearson correlation from per-basic-window statistics:

```
Corr(x,y) = Σ_j B_j (σ_xj σ_yj c_j + δ_xj δ_yj)
            / [ sqrt(Σ_i B_i (σ_xi² + δ²_xi)) · sqrt(Σ_i B_i (σ_yi² + δ²_yi)) ]
      δ_xi = mean(x_i) − (Σ_k mean(x_k)) / ns
```

The sketch per basic window is **mean, standard deviation, and the correlation `c_j` of each
pair on that basic window**. Lemma 2 gives the incremental update when a new basic window
arrives. Because basic windows may be variable-length, it supports **arbitrary query windows**,
lifting the integral-multiple restriction that StatStream and Mueen et al. impose.

- **No pruning.** Confirmed, and their own conclusion lists "develop a pairwise correlation
  pruning algorithm based on a threshold" as **future work**. Everything is `O(N²)`.
- **Negative correlation: yes.** Their Algorithm 2 line 6 is `if |c| > θ`, and that edge rule is
  what every experiment runs, so it is exercised rather than merely stated.
- **No lags.** Both series are read on the same query window.
- Evaluation: NCEA (157 series) and Berkeley Earth (18,638 series). Go, PostgreSQL, 64 cores,
  512GB. At least an order of magnitude faster than a raw-Pearson baseline; **sketch time better
  than a DFT approximation, query time on par with it**.

### 5.2 The cost nobody mentions, and why it matters to us

**Space is `O(L·N²/B)`**, precisely `(L/B)(2 + N(N−1)/2)`, because TSUBASA stores **a correlation
value per pair per basic window**. That is quadratic in the number of series *in storage*, not
just in time. At our battery's `m = 500` and a short basic window this is a large artifact, and
this project has already lost three sessions to an accumulator that grew quadratically
(the 2026-09-11 int32 work). **Budget the sketch store before implementing, and record it as a
reportable property**: TSUBASA buys exact arbitrary-window queries with quadratic storage, which
is a real and publishable trade-off rather than an implementation detail.

### 5.3 Their §4.1 result is the most valuable thing here

TSUBASA measured the accuracy of DFT-based approximation on climate data and found that the
approximate network matched the exact one **only when all 200 coefficients of a 200-point basic
window were used**. They note explicitly that climate data are **uncooperative time-series**, and
that StatStream uses "two coefficients for any basic window size."

**This is independent corroboration, from a third paper, of the axis §4a.2 takes from
Cole-Shasha-Zhao**, and it is sharper than CSZ's own version because it shows the consequence in
the *output*: false-positive edges and a distorted network, not just a distorted distance. Three
separate papers now support the same claim, which is exactly the kind of support a reviewer
finds hard to dismiss:

| source | evidence |
|---|---|
| Cole-Shasha-Zhao 2005, Figs. 2 and 4 | DFT, DWT and SVD approximate distances badly on white-noise-like data; sketches hold up |
| FilCorr 2020, §I | "Seismic traces are mostly white noise ... stressing the algorithms to fall behind the stream quickly" -- their stated reason for rejecting pruning entirely |
| TSUBASA 2022, §4.1 | on climate data the DFT approximation needs *every* coefficient to reproduce the exact network |

### 5.4 Integration and anchor

`baseline_mode="tsubasa"` + `Candidates_BF_TSUBASA` beside `Candidates_BF_ExactSTOMP` and
`Candidates_BF_FilCorr`. **It is exact, so it must reproduce the bruteforce pair set precisely**;
that expectation is the faithfulness anchor, the same discipline that caught two real bugs in
the FilCorr port. Its distinct claim against `exact_stomp` is *arbitrary query windows from
precomputed per-pair sketches*, not speed, and the comparison should say so.

---

## 5a. BRAID -- the unpruned lag method, structurally unlike every other arm

Sakurai, Papadimitriou & Faloutsos, *BRAID: Stream Mining through Group Lag Correlations*,
SIGMOD 2005, 599-610. Journal extension: *Fast Discovery of Group Lag Correlations in Streams*,
TKDD 5(1):5, 2010 (Sakurai, Faloutsos & Papadimitriou), **obtained and read 2026-09-16**.
SIGMOD version from the authors' Osaka mirror,
`https://www.dm.sanken.osaka-u.ac.jp/~yasushi/publications/braid.pdf` (the CMU mirror is a
dvips Type-3 build whose text layer does not extract -- use the Osaka copy); §5a is
paper-authoritative across both versions.

### 5a.1 Why it belongs in the mix, and why it is not a fifth pruning competitor

**BRAID does not prune the pair space at all, and the TKDD 2010 extension confirms it rather
than fixing it.** Its top-level loop is literally `for each pair of sequence X and Y do
ProductKeeping(X, Y)`: statistics for every one of the `O(k²)` pairs, every tick.

The §10.2 open question was whether the journal version's emphasis on *group* lag correlations
added pruning. **It does not.** The extension adds **ThinBRAID**, which replaces the
`O(k² log n)` stored inner products with `O(k log n)` **random-projection sketches** per
sequence, recovering `Sxy` from sketch distances (their Eq. 24). That cuts the *per-tick update*
from `O(k²)` to `O(k)` -- but their own Table II is explicit that **computing the lag
correlations still costs `O(k² log n)`**, and the paper says so directly: "we still need
`O(k² log n)` time to estimate the lag correlations, but this will happen only when the user
requests us to do so."

So ThinBRAID makes the *maintenance* cheap while **still examining every pair** at output time.
It is an amortization, not a filter. **BRAID's row in §5b.5 stands: no pair-space pruning.**

This is worth a sentence in our paper, because it is a clean structural contrast: BRAID/ThinBRAID
and CorrTrack both use random projections, but for **opposite purposes** -- ThinBRAID to compress
what it stores about all pairs, CorrTrack to avoid looking at most pairs at all.

So BRAID is **orthogonal to the other four**, and that is exactly why it is worth adding:

| axis | ParCorr / CorrJoin | StatStream | BRAID | CorrTrack |
|---|---|---|---|---|
| pair-space pruning | yes -- their whole contribution | yes | **none** | yes |
| lag search | **none** (both verified synchronous) | yes, quantized to basic-window multiples | **yes -- its whole contribution, at any lag** | yes, arbitrary integer lags |
| what is approximated | which *pairs* are examined | which pairs, plus the DFT truncation | the *lag value*, and the correlation at high lags | which pairs |

**Correction to an earlier version of this section**: it claimed BRAID was the *only* peer on
the lag axis. That was wrong -- **FilCorr and StatStream both do lags too** (StatStream verified
in §4.1). BRAID's actual distinctiveness is narrower but still real, and it is worth stating
precisely rather than overselling:

- it is the **only lag method here that does no pair pruning at all**, which isolates the lag
  mechanism from the candidate-search mechanism;
- it is the **only arm whose approximation lands on the lag value itself** rather than on the
  pair set, so it probes a failure mode none of the others can;
- it searches **arbitrary lags**, where StatStream's grid path is restricted to multiples of the
  basic window.

So the lag axis now has three competitors, not zero and not one: **FilCorr** (band-limited),
**StatStream** (grid-indexed, quantized) and **BRAID** (unpruned, interpolated). That is a much
better-defended comparison than the plan had at any earlier revision, and it means CorrTrack's
lag claim will be genuinely contested rather than uncontested.

### 5a.2 Specification (paper-authoritative)

**Target**: per pair, the **earliest local maximum** of `score(l) = |R(l)|` above a threshold
`γ` (Definition 1; paper's `γ = 0.4`). Note `|R(l)|`, so **negative correlation is handled**,
matching CorrTrack's `neg_corr=True`. Max lag `m = n/2`, following standard time-series
practice, so `m` grows with the stream.

Three ideas, in the paper's own order:
1. **Sufficient statistics.** `R(l)` is algebraic, so five running numbers suffice
   (`Sx, Sxx, Sy, Syy, Sxy(l)`), with `Sxy(l) = Σ_{t>l} x_t · y_{t−l}`. **These are the same
   five sums `exact_stomp` and CorrJoin use** -- the third independent consumer, which argues
   for factoring that machinery out once rather than three times.
2. **Geometric probing.** Compute `R(l)` only at `l = 0, 1, 2, 4, ..., 2^i`, then **cubic-spline
   interpolate** between the probed lags and locate the maximum with **Brent's method**. The
   paper notes the interpolation choice is orthogonal to the method.
3. **Smoothing.** Hierarchical non-overlapping window averages, `Ax_h(t) = (Ax_{h−1}(2t−1) +
   Ax_{h−1}(2t)) / 2` with `Ax_0 ≡ x`, one level per power of two. The approximation is
   `R(l) ≈ R̂_h(l / 2^h)`: high lags are read off coarse levels. This is what buys `O(log n)`
   space instead of `O(n)`.

**Enhanced BRAID** keeps `b > 1` coefficients per level, giving a mixed arithmetic-geometric
lag set `l = {0, 1, ..., 2b−1; 2b, 2(b+1), ...}`. **The paper's experiments use `b = 16`.**

**Complexity**: `O(log n)` space per pair, `O(1)` amortized time per tick to update statistics,
`O(log n)` per interpolation. Naive is `O(n)` for both.

**Accuracy**: approximate, with a Nyquist-based bound. Lemma 3: BRAID resolves lags in
`0 ≤ l < l_R = 2b / f_R`, where `f_R` is the highest frequency present; error is **zero** when
the signal is sampled at or above Nyquist. Reported: correct lag "perfectly most of the time",
largest relative error ~1%, up to 40,000x faster than naive.

**ThinBRAID (TKDD 2010 §5)**: sketch each sequence with `d` random +1/-1 projections
(`d = 400/2^h` at level `h` in their experiments), recover the inner product from sketch
distances via `Sxy_h = (Sxx_h + Syy_h - Dp_h)/2`, then use the same Eq. 11. Complexity
(their Table II): update `O(k log n)` space and `O(k)` time; **output still `O(k² log n)`**.
Accuracy bound: Lemma 9, `Err = eps / (2 sqrt(Vx) sqrt(Vy))` with JL dimension
`d0 = (4 + 2*delta) / (eps²/2 - eps³/3) * log n`. Crossover where ThinBRAID overtakes BRAID is
at about **k = 1000 sequences** (their Fig. 19) -- relevant to us, since our battery runs
m = 500 and up.

**Exponential forgetting (TKDD §3.5)**: an optional decay factor `lambda` in (0,1] on all the
running sums, with `lambda = 1` used throughout their experiments. Worth noting because the
paper explicitly contrasts it with **StatStream's sliding-window model**, criticising the latter
for needing "a window size larger than the maximum lag" and hence "explicit buffering
requirements." That is a competitor criticising another of our arms on a point that also applies
to CorrTrack, and it belongs in the discussion.

**Measured accuracy (their Tables III and IV)** -- the anchor numbers to reproduce, lag error in
percent: Sines 0.000 / 1.397 (BRAID / ThinBRAID), SpikeTrains 0.387 / 0.528, Humidity
0.024 / 1.178, Light 0.529 / 0.176, Kursk 0.615 / 0.615, Sunspots 1.038 / 0.086.

**Evaluation in the paper**: the only baseline is the naive implementation. Datasets: Sines
(n=32,768), SpikeTrains (period 6500, n=100,000), Humidity / Light (55 sensors at 30s), Kursk
seismic (n=70,000), Sunspots (n=25,900), and **Motes** (54 Berkeley Mote sensors: temperature,
humidity, battery voltage, with a lab floor plan; measured lags of 202 and 224 minutes between
physically nearby sensors). Xeon 2.8GHz, 1GB RAM.

### 5a.3 Four comparability problems, and how to handle each

BRAID does not slot into the existing protocol cleanly. Naming the mismatches now is cheaper
than discovering them mid-implementation:

1. **Different output.** BRAID reports **one lag per pair** (the earliest qualifying local
   maximum); CorrTrack reports all pairs above threshold at every lag in its window. These are
   not the same query. **Resolution**: compare on the common denominator only -- "for pairs
   both methods report as lag-correlated, do they agree on the lag?" -- and put the rest in the
   §5b.5 capability table.
2. **Different accuracy axis.** BRAID's published error metric is **relative error in the lag
   value**, not recall/precision over a pair set. §6.2's equal-recall protocol therefore does
   **not** apply to it unmodified. **Resolution**: report lag-agreement separately, as its own
   metric, rather than forcing it into the recall column. This is the one place where the
   uniform metric scheme genuinely breaks, and saying so is better than fudging it.
3. **Unbounded vs bounded lag horizon.** BRAID's `m = n/2` grows with the stream; CorrTrack uses
   a fixed `n_lags`. **Resolution**: fix a common maximum lag for the head-to-head and state it.
4. **Smoothing bias.** Because high lags are read from coarse levels, BRAID is inherently biased
   toward low-frequency correlation -- the same family of concern as FilCorr's band-pass. It
   should do relatively well on the periodic/smooth data it was evaluated on and relatively
   badly on bursty high-frequency data. **That is a hypothesis worth testing rather than a flaw
   to hide**, and our density-targeted synthetic generator can produce both regimes.

### 5a.4 Expected result, stated in advance

BRAID maintains `O(k²)` pair states with no pruning, so at the series counts in our battery
(m = 500 and up) it should lose badly to CorrTrack on wall-clock while being competitive or
better at small `k`. **Predicting this before running it is the point**: if BRAID instead wins
at large `k`, our port or our understanding is wrong. Recording the prediction here makes that
check honest.

### 5a.5 Integration and verification anchor

Not a `candidate_backend` (there is no candidate generation). It follows the FilCorr /
TSUBASA pattern: `baseline_mode="braid"` + a `Candidates_BF_BRAID` class beside
`Candidates_BF_ExactSTOMP` and `Candidates_BF_FilCorr`.

**Anchor**: when `2b > n_lags`, level 0 alone covers every lag in the comparison range at raw
resolution, with no smoothing and no interpolation, so **BRAID must reproduce the bruteforce
lagged pair set exactly**. Same degenerate-configuration discipline that caught both FilCorr
bugs (the Nyquist weight and the odd-window off-by-one). Then a second anchor at the paper's
own `b = 16, γ = 0.4` reproducing its qualitative claim of ~1% lag error on a periodic signal.

---

## 5b. What to report in the paper -- resolving the "our reimplementation can be questioned" problem

The tension is real: reimplementing everything makes the arms comparable but invites "you
under-implemented my method"; *not* reimplementing leaves methods that don't even emit the same
outputs. It dissolves once three separate things are reported separately instead of collapsed
into one speed number.

### 5b.1 Three-phase work decomposition -- the implementation-independent backbone

Every method here, exact or approximate, decomposes the same way. Phases that are *zero* for a
method are informative, not gaps:

| phase | countable quantity | BF | exact_stomp / TSUBASA | FilCorr | BRAID | ParCorr / StatStream / CorrJoin / CorrTrack |
|---|---|---|---|---|---|---|
| **summarize** | coefficients produced per window; values read | 0 | sufficient statistics | `B` band coefficients | `O(log n)` levels x `b` coefficients | `k` sketch dims (+ SVD for CorrJoin) |
| **prune** | index probes, sketch-space distance/dot computations | **0** | **0** | **0** | **0** | the entire contribution |
| **verify** | pairs verified x **cost per verification** | all x O(w) | all x O(1) amortized | all x O(B) | all x O(log n) per tick, amortized O(1) | few x O(w) |

BRAID sits in the same "no pruning" column as the exact baselines even though it is
approximate, which is the clearest way to show that **approximate does not imply pruning**:
BRAID spends its approximation budget on the lag axis instead of the pair axis.

This answers the "exact methods have no candidate generation, so what do we compare?" problem
directly: **for exact methods the axis is `cost per verified pair`**, which is analytically
derivable (O(w) raw vs O(1) amortized incremental vs O(B) band-limited vs O(#segments)
sketch-based) and empirically checkable -- measured time-per-pair should scale with `w` exactly
as the model predicts, and if it doesn't, that is a finding about the implementation.

**All of this is already instrumented.** `sk_time`/`cand_time`/`val_time` are the seconds;
`total_candidates`/`tested_candidates`/`candidate_search_*_touched`/validated counts are the
implementation-independent quantities, all already in `RUN_RESULT_COLUMNS`. Publish both.
**Where the counts and the seconds tell the same story the timings are credible; where they
diverge, that divergence IS the implementation-tier effect, made visible instead of hidden.**

### 5b.2 Wall-clock: keep it, anchored on BF

Speedup-vs-brute-force is the currency the field expects and omitting it would read as evasive.
What makes it defensible here: BF runs **in the same job, same harness, same validation kernel**
as every arm, so the ratio is measured under identical conditions even where absolute
implementation quality differs. Report speedup-vs-BF as the headline, with §5b.1's counters as
the backing evidence.

### 5b.3 The line on "should we optimize other people's work?"

**Same engineering tier, unchanged algorithm.**
- ✅ Use the same vectorized primitives and the same shared exact-validation kernel -- that is
  not optimizing their method, it is *not handicapping* it.
- ❌ Never add algorithmic improvements they did not publish -- that is no longer their method.

This project already has a worked precedent to cite: the FilCorr port's real-decomposition fix
(a complex matmul whose imaginary half was discarded) and cross-step FFT memoization (the same
window's FFT recomputed ~15x) were handicap-removals, while a `prange` multi-core kernel was
**deliberately declined** because `exact_stomp` does not get one. Recorded in
`docs/implementation_log.md`'s 2026-09-11 (d) entry -- i.e. the policy is documented as it was
applied, not reconstructed afterwards.

### 5b.4 What actually defends the reimplementations, in priority order

1. **Reproduce each paper's own published claims on its own data.** ParCorr's reported ~95%
   recall / 100% precision; CorrJoin's reported pruning/speedup on the four original datasets
   (§3 -- we have the link). Demonstrated faithfulness beats asserted faithfulness.
2. **Release the code.**
3. **State the policy explicitly** in the evaluation section: all methods implemented in one
   framework, sharing the exact-validation kernel; no algorithmic modifications; hot paths
   vectorized uniformly.
4. **Cheap cross-check against existing implementations** (TSUBASA's repo; CorrJoin's Go/Python
   ports): report "our port is within Nx of implementation Y on the same input" as a footnote
   **sanity check, never as a comparison arm**. This is the refined position -- authors' code is
   a poor competitor arm but an excellent faithfulness witness.

### 5b.5 Methods that do not emit the same outputs -> capability matrix

Compare all arms only on the common denominator (find pairs above threshold in a window). Put
everything else in a **capability table** -- **graded by evidence tier (§5b.6)**, because
several competitors describe capabilities their own experiments never exercise.

Tiers: **[E]** evaluated (an experiment in the paper measures it) - **[S]** specified (algorithm
or lemma given, but no experiment) - **[C]** claimed (asserted in prose or a problem statement,
no algorithm) - **[N]** absent - **[?]** primary source not yet read.

| capability | CorrTrack | StatStream | ParCorr / CSZ | CorrJoin | BRAID | FilCorr | exact arms |
|---|---|---|---|---|---|---|---|
| **lag search** | [E] arbitrary integer | **[S]** algorithm + `T_M` parameter, **no lag experiment even in the technical report** (§10.2 item 3); basic-window multiples | **[C]** CSZ's problem statement names "asynchronous"; ParCorr lists it as future work | **[N]** synchronous | **[E]** whole contribution, 6 datasets | **[E]** its problem statement; `max` over the lag window | [N] |
| **pair-space pruning** | [E] | [E] pruning power 0.01-0.09 | [E] | [E] | **[N]** none, even in ThinBRAID (§5a.1) | **[N] by design** -- they reject pruning as data-dependent (§4b.1) | [N] |
| **negative correlation** | [E] `neg_corr=True` | **[S]** Lemma 3, no experiment | **[N]** user-confirmed absent | **[S] specified, unreachable: `not_available`.** Alg. 1 line 14 and `2-CorrJoin.R` test `abs(corr) >= T`, but that test only sees pairs that survived the eps_1 bucket grid and the eps_2 Euclidean filter, and both filters pass only *small distances*, i.e. positively correlated pairs; an anti-correlated pair sits at distance ~2 and is pruned before the `abs()` ever runs. Verified at the index level 2026-09-17 (log (c)). Cannot be enabled without replacing its filters with someone else's, so the neg_corr=True run reports N/A | [S] Definition 1 uses `abs(R(l))`; CCF plots do span negative values | **[N]** Eq. 7 takes `Max` of *signed* correlations. **Our port adds it: see §4b.2** | **[E]** TSUBASA Alg. 2 uses `abs(c) > θ` in every experiment |
| **persistence / min duration** | [E] | **[C]** a system-parameter description in §4, no algorithm, no experiment | [N] | [N] | [N] | [N] | [N] |
| **episode + attention + anomaly monitoring** | [E] | [N] | [N] | [N] | [N] | [N] | [N] |
| **band-pass filtered correlation** | [N] | [N] | [N] | [N] | [N] | **[E]** the contribution | [N] |
| **exactness** | [N] | [N] | [N] | [N] | [N] | **[E]** by Parseval | **[E]** TSUBASA by Lemma 1; BF and STOMP by construction |

**Capabilities the others lack belong in a feature table, never as a speed penalty charged
against them** -- and capabilities they *do* have must not be quietly left out of it.

### 5b.6 Evidence tiers: why "the paper says it can" is not "the method does it"

**This distinction came from the user and it corrects a real weakness in the two previous
revisions of this plan.** The pattern: a paper states a capability, sometimes with a lemma or
pseudocode, but **never runs an experiment on it**. Recording that as a plain capability is
wrong in both directions -- it overstates the competitor, and it understates CorrTrack, which
*does* evaluate the same capability.

The clearest case is **StatStream and lags**. It is not a bare claim: there is a derivation
(their Theorem 1, Corollary 1), a grid-maintenance algorithm with timestamps (§3.6), and a named
system parameter `T_M` (§4). But their §5 asks exactly three empirical questions -- speed,
approximation error versus window sizes, and pruning power/precision -- and **none of the
experiments involve a lag**. Same for their **Lemma 3** on negative correlation, and their
**"Duration over Threshold"** parameter is weaker still: a paragraph in the system description
with no algorithm and no measurement.

**How each tier is treated in the paper:**

- **[E]** Compare head-to-head on that capability. This is the only tier that supports "method
  X is faster/better at Y than CorrTrack".
- **[S]** Report as "specified but not evaluated by its authors". **We may still implement and
  measure it** -- that is a contribution, not a criticism -- but the result must be labelled as
  **our** evaluation of their specification, never as a reproduction of their published result.
  Crucially, **if our implementation of an [S] capability performs badly, that is not evidence
  the method is bad**: there is no published baseline to say whether we built it as intended.
  Say so explicitly rather than banking the win.
- **[C]** Do **not** implement it as theirs. Note the claim, note the absence of an algorithm,
  and exclude the method from that axis. Building it ourselves and attributing it to them is
  the §10.3 error in its most tempting form.
- **[N]** Capability table only.
- **[?]** **No claim of any kind until the primary source is read** (§10.4).

**Apply the same standard to CorrTrack.** Any capability we claim must be in the [E] column of
our own evaluation, not merely implemented. If persistence/episode monitoring or anomaly
attention is never measured in the paper, it belongs in [S] for us too. Holding competitors to
a standard we exempt ourselves from is the failure mode this whole section exists to prevent.

---

## 6. Fairness protocol -- decide and record before implementing

### 6.1 Lag handling
**Settled, after reading all four papers.** ParCorr is synchronous by construction and its own
paper lists lags as future work (§2). CorrJoin is synchronous, verified (§3.2). **StatStream
does lags** (§4.1), which an earlier revision of this plan got wrong. So the lagged arms are
**CorrTrack, FilCorr, StatStream and BRAID**.

- **Primary head-to-head runs at `n_lags=0`**, where every arm expresses the same problem
  honestly. This is the defensible mechanism comparison. **BRAID is degenerate here** (it is a
  lag method), so it enters this run only as a reference point, not as a rival. StatStream runs
  here in its synchronous mode, where the grid **is** cleared every basic window.
- **A second, lagged comparison** includes CorrTrack, FilCorr, **StatStream**, **BRAID** and the
  exact baselines. This is where CorrTrack's lag claim is actually contested.
- **Evidence tiers decide who counts as a rival here (§5b.6), and the answer is two, not one
  and not three.** Reading down the lag row of the capability matrix:
  - **FilCorr [E]**: lagged correlation *is* its problem statement, and its Figs. 6-7 sweep
    `lag = 0, 100, 250`. A genuine evaluated lag peer, **and it is already in the battery.**
  - **BRAID [E]**: lag detection is its whole contribution, measured on six datasets.
  - **StatStream [S]**: specified in full, never measured, in either the VLDB paper or the
    technical report. Can be implemented, but the result is *our* evaluation of *their* design.
  - **Cole-Shasha-Zhao [C]**: "asynchronous correlation" is defined in its problem statement
    and named in its summary, but §5-6 contain no lag algorithm and no lag experiment. Not a
    lag peer, and not to be built as one.
  So the lagged head-to-head has **two published peers, FilCorr and BRAID**. An earlier
  revision said "only BRAID"; that was written before the FilCorr paper was read and was not
  propagated when the matrix was updated. Corrected 2026-09-16.
- The lagged rivals probe **different weaknesses**, which is what makes this adversarial rather
  than decorative:
  - **FilCorr** keeps every pair (it rejects pruning by design, §4b.1) and restricts the
    frequency band, so it tests CorrTrack against an exact, unpruned, band-limited lag search;
  - **BRAID** keeps every pair and approximates the **lag value itself**, so it tests a
    different failure mode from FilCorr's;
  - **StatStream** prunes and indexes like CorrTrack but **quantizes lags to basic-window
    multiples**, so it is the fair test of "does a modern index beat a 2002 grid at equal
    recall", with the [S] caveat above.
- **Lag granularity must be reported, not just lag capability.** StatStream's grid path resolves
  only multiples of `b`; CorrTrack resolves every integer lag. A comparison run at lags that
  happen to be multiples of `b` flatters StatStream, and one at arbitrary lags flatters
  CorrTrack. **Run both, and say which is which.** Choosing only one would be cherry-picking of
  exactly the kind §6.4 forbids for density.
- Lag capability is reported as a **capability difference** (§5b.5), never folded into a speed
  ratio against methods that cannot do it.
- If a synchronous method is extended to lags by us (e.g. re-running candidate search per lag
  offset), that extension is **ours, not theirs**, and must be labelled as such.

### 6.2 Recall parity
All approximate arms have recall/speed knobs; comparing at each one's defaults measures
nothing. Tune every arm to the **same target recall (0.95**, CorrTrack's own `target_recall`)
on a held-out calibration span, then compare speed **at equal achieved recall**. Always report
achieved recall next to the target -- never assume it was met (the 2026-09-12 runs showed
CorrTrack landing at 0.90 untuned and 0.98 tuned on the same data).

### 6.3 Implementation tier -- uniformity helps between competitors, not against CorrTrack
Reimplementing all four here equalizes the **competitors with each other and with FilCorr**.
It does **not** equalize them with CorrTrack itself, which has months of Cython/`nogil`/`prange`
optimization behind it (`validate_corr_rows` etc.). A fresh numpy port of ParCorr losing to
CorrTrack on wall-clock would partly measure that gap, not the algorithms -- exactly the trap
the FilCorr work hit on 2026-09-11, where a "22% slower" result turned out to be tier mismatch.

Therefore:
1. **Headline the implementation-independent counters**: `total_candidates`,
   `tested_candidates`, `candidate_search_*_touched`, validated pairs -- all already recorded
   in `RUN_RESULT_COLUMNS`. "At equal recall, method X generated 3.2x more candidates" is a
   claim about algorithms that survives any implementation difference.
2. Route every arm's **exact validation through the same Cython kernel**, so only candidate
   generation differs in tier.
3. Report wall-clock too, with the tier caveat stated explicitly.

### 6.4 Density regimes
CorrTrack's standing **flipped** between the real dense dataset (~17.6% of tested pairs
correlated; CorrTrack 0.73x-1.03x) and sparse synthetic data (~2%; CorrTrack 3.66x, winning
outright). A single-density comparison is cherry-picked by construction. Sweep at minimum
{0.5%, 2%, 5%} via `synth_corr_gen.make_density_targeted_dataset`, plus the real dense dataset.

### 6.5 Per-port verification anchors (load-bearing, since we're not running authors' code)
Each port needs a configuration where its output is **provably determined**, made into an
automated test:
- **TSUBASA**: exact -- must reproduce the bruteforce pair set precisely.
- **ParCorr / StatStream / CorrJoin**: with the filter disabled (threshold admitting
  everything), the candidate set must equal the full enumerated pair set, and the surviving
  validated set must equal bruteforce's. That isolates "is the filter mathematically sound"
  from "is the plumbing right."
- **BRAID**: with `2b > n_lags`, level 0 covers every lag at raw resolution with no smoothing
  and no interpolation, so it must reproduce the bruteforce **lagged** pair set exactly (§5a.5).
- **StatStream** has the sharpest anchor of all, because the paper separates two guarantees:
  the **grid filter is provably false-negative-free** (their Theorem 2) while the reported recall
  of 0.9987-1.0 comes only from the **DFT post-processing approximation**. So the port must show
  **grid recall exactly 1.0** against bruteforce with post-processing disabled, and only then
  the published precision/recall with it enabled. A port that leaks false negatives in the grid
  is wrong no matter how good its end-to-end numbers look.
- Plus, for each, at least one published-parameter configuration reproducing a qualitative
  claim from its own paper: ParCorr's >90% recall at `T=0.7` / 100% precision with `r=60, k=2,
  f=0.7` (§2.1); StatStream's precision 0.9765-0.9947 at `n=16`, `T=0.85/0.9`, tolerance
  0.001/0.0005 and pruning power 0.01-0.09 (§4.2); BRAID's ~1% relative lag error at `b=16,
  γ=0.4` on a periodic signal (§5a.2); CorrJoin's pruning and speedup on its own four datasets
  (§3.0).

---

## 7. Sequencing

| phase | method | why this order | main risk |
|---|---|---|---|
| 0 | protocol + papers | record §6 decisions; **CorrJoin and ParCorr papers obtained and extracted (§10.1 and §10.2 item 1 closed)** -- remaining: StatStream (VLDB 2002) parameters and BRAID's TKDD 2010 extension, §10.2; clone the reference implementations and the original datasets | no blocker left |
| 1 | **shared prerequisites** | (a) the **normalize-window-before-reducing** path, needed by *both* ParCorr and CorrJoin (§2 constraint 2, §3.1 step 1); (b) factor out the **five-sum sufficient-statistics** helper now that `exact_stomp`, CorrJoin *and* BRAID all need it | small; both are consolidation, not invention |
| 2 | **ParCorr** | cheapest: sketch already exists, grid+vote code exists in the v1.0 snapshot (disabled -- see §2.2), user knows it intimately, and all parameters are now paper-confirmed (§2.1) | re-enabling the vote correctly (`f=0.7`, not the snapshot's effective `f=1.0`) and deciding the cell-size question (§2.2, last row) |
| 3 | **StatStream** | **the closest structural peer to CorrTrack** (§4.3) and the one a reviewer will press hardest on. Shares the grid work with phase 2 and the FFT caching with FilCorr. **Build the [E] core first** (synchronous pruning, the only part its authors evaluated); the [S] lag and negative-correlation paths are a second, separately-labelled step (§5b.6) | bigger than the earlier draft assumed: incremental DFT (Lemma 6 digests), the timestamped never-cleared lagged grid, and the `O(k·n²)` unaligned-window table -- **all of it unevaluated by its authors**, so there is no published number to check our port against |
| 4 | **BRAID** | the only lag method here that does **no pruning**, so it isolates the lag mechanism; and the only arm whose approximation lands on the lag value. No SVD, no index; hierarchical sums + spline + Brent | the comparability mismatches in §5a.3, especially the different accuracy axis -- design that reporting before coding |
| 5 | **TSUBASA** | exact, so it has a hard correctness anchor; structurally close to existing `exact_stomp`. Easy enough to slot anywhere | low |
| 6 | **CorrJoin** | **no longer gated** and no longer the hardest -- spec is in hand (§3.1), the SVD is a cheap per-window `m x ks` decomposition, and the original datasets are available for validation | the two-stage filter plumbing; ε and lag semantics resolved |

## 8. Infrastructure already in place

- `abaca/fourway_compare.py` generalizes directly to an N-way comparison (already loops over
  modes, computes recall/precision in memory via `CorrTrack.compute_metrics_bf`).
- `abaca/*.oar` + OAR workflow: reproducible cluster runs, node-local kernel builds, 192GB
  nodes (no local WSL memory limits).
- `synth_corr_gen.make_density_targeted_dataset`: controlled density regimes with exact ground
  truth.
- Per-phase counters for §6.3 already in `RUN_RESULT_COLUMNS`.
- Pluggable axes (`_VALID_CANDIDATE_BACKEND_AXIS:1745`, `_resolve_data_representation:1748`)
  are the insertion points -- they need widening back out, not inventing.

## 9. Scope estimate

ParCorr and StatStream are each realistically a few focused days given the snapshot code and
the existing axes -- ParCorr now slightly cheaper than previously estimated, since §2.1 closed
every parameter question and §2.2 located the exact code to re-enable. **BRAID is moderate**:
the mechanism is simple (hierarchical means, five sums, a cubic spline, Brent's method, all
standard and all available in scipy), and the real cost is in the reporting design of §5a.3
rather than the algorithm. TSUBASA is moderate and self-verifying. CorrJoin remains the largest
single
cost, but **now on one axis instead of two**: the **specification risk is eliminated** (§3.1 --
the paper resolved the ε formulas and the lag question, both of which the derivative sources
had wrong), leaving only the **engineering** cost of incremental SVD over sliding windows, which
is real but bounded. Its incremental correlation update turns out to reuse the same
sufficient-statistics form `exact_stomp` already implements, which reduces that cost further.

The fallback if time runs short: ParCorr + StatStream + **BRAID** implemented, TSUBASA and
CorrJoin discussed as related work with the omission stated plainly. Two asymmetries to weigh
against each other when deciding what to cut:
- CorrJoin is the **newest** competitor (SIGMOD 2023) and therefore the one a reviewer is most
  likely to ask about, which argues for paying its cost rather than deferring it.
- **BRAID is the only arm that tests CorrTrack's lag claim at all.** Cutting it leaves the
  central differentiator unmeasured, which is a worse hole than a missing pruning rival, since
  we already have four of those. On that reasoning BRAID should outrank TSUBASA, and arguably
  CorrJoin, in what actually gets built.

## 10. Phase 0 -- extraction checklist

### 10.1 CLOSED (2026-09-15): CorrJoin paper obtained and extracted

All six items resolved; the specification now lives in §3.1 and the corrections in §3.2.

| # | item | outcome |
|---|---|---|
| 1 | ε definitions | **Resolved, and both thesis variants were wrong.** `ε₁ = sqrt(2·ks·(1−T)/n)`, `ε₂ = sqrt(2·ke·(1−T)/n)` -- both divide by `n`; ε₁ uses `ks`, not `kb` |
| 2 | lag mechanism | **Resolved: there is none.** CorrJoin is synchronous. Propagated to §1, §5b.5, §6.1 |
| 3 | incremental update | **Resolved:** five running sums `s1..s5` (Eq. 2); PAA means in `O(h+k)`; stride has only slight runtime effect (Fig. 16a) |
| 4 | pseudocode / parameters | **Resolved:** Alg. 1 + Alg. 2; `ks=15, ke=30, kb=3`; gain needs `m > 100`; speedup ≤ `1/r1` |
| 5 | negative correlation | **Confirmed:** Alg. 1 line 14 uses `|corr| ≥ T` |
| 6 | paper's baselines / data | **Resolved:** ϵ-kdB tree, TSUBASA, Quickjoin, IncP^PAA; all reimplemented in R by the authors (a precedent for §0) |

### 10.2 Other Phase-0 questions

1. ~~**ParCorr's grid/subspace parameters**~~ -- **CLOSED 2026-09-15.** Paper obtained from the
   HAL-LIRMM open mirror (`lirmm-01886794`); all parameters in §2.1 (`r=60, k=2, f=0.7`,
   `w=500, b=20, T=0.7`), calibration procedure and reported recall/precision extracted. The
   v1/backup2 snapshot cross-check is in §2.2 and found three things the port must fix rather
   than inherit. **One genuine gap remains**: the paper specifies no grid cell size, so the
   v1 formula `sqrt(2(1−T))/sqrt(r)` is our addition and must be labelled as such, or replaced
   by the paper's tune-`f`-to-target-recall procedure.
2. **Whether to adopt the original CorrJoin datasets** (chlorine/gas/stock/synthetic) as an
   additional evaluation axis rather than only for faithfulness validation -- they are real,
   public, and already used by a SIGMOD paper, which makes them defensible shared ground.
   Same question now also applies to **ParCorr's** datasets (seismic; a Yahoo Finance set of
   2000 end-of-day returns for ~40,000 symbols) and **BRAID's** (the 55-sensor Humidity / Light
   / Temperature sets, Kursk seismic, Sunspots). BRAID's sensor data is the most interesting
   of these for us, since it is multi-series and lag-correlated by construction.
3. ~~**Zhu & Shasha TR2002-827** may hold the StatStream lag experiments~~ -- **CLOSED
   2026-09-16.** The technical report was obtained and read. It is the fuller version (it adds
   §3.6 on I/O performance and a basic-window-size figure), but **its §5 asks the same three
   empirical questions and still contains no lag experiment**. StatStream's lag support is
   confirmed `[S]`: specified in detail, never measured, in either version.
3b. ~~**Locate the CorrJoin authors' R implementation**~~ -- **CLOSED 2026-09-16.** The user
   supplied it: six scripts, in `docs/reference_code/corrjoin_authors_R/` with a README of
   what was extracted. CorrJoin is now a reproduction (implementation plan B3).
4. ~~**StatStream's DFT coefficient count and grid parameters**~~ -- **CLOSED 2026-09-15.**
   Paper obtained; full specification, parameters (`n=16`, swept 16/24/32/40; `T=0.85/0.9`;
   tolerance 0.001/0.0005) and published results are in §4. **It also overturned two capability
   claims** -- see §4.0 and §10.3. **All primary sources for the pruning competitors are now
   read.**
5. **BRAID's TKDD 2010 journal extension** (*Fast Discovery of Group Lag Correlations in
   Streams*, 4(1):5). The SIGMOD 2005 version has **no pair pruning** (§5a.1), and the journal
   title's emphasis on "group" suggests the extension may add some. **This changes BRAID's
   role if true** -- it would become a lag *and* pruning competitor rather than a pure lag peer.
   Worth resolving before phase 4, though not blocking: the SIGMOD version is a complete,
   citable method on its own.
6. **StatStream's own datasets**: synthetic random walk (trivially reproducible from the
   formula in §4.2) and NYSE TAQ 300 stocks. The synthetic one is free and should simply be
   added; TAQ is licensed, but our existing `sp500` pools are a defensible stand-in and the
   substitution should be stated.
7. ~~**Verify whether ParCorr handles negative correlation.**~~ -- **CLOSED 2026-09-16:
   it does not** (user-confirmed). The ParCorr/CSZ arm therefore runs positive-correlation only,
   and the asymmetry against CorrTrack's `neg_corr=True` is reported rather than papered over by
   adding an `abs()` the method never had (§2.1).
8. ~~**Does CorrJoin's paper actually *evaluate* negative correlation**~~ -- **CLOSED, revised
   2026-09-17: [S], and unreachable.** The 2026-09-16 reading ([E]) was wrong. `2-CorrJoin.R` does apply
   `abs(corr) >= corrThreshold`, but only to pairs that passed the eps_1 bucket grid and the eps_2
   Euclidean filter, and both admit only *small* distances between normalized windows, i.e. corr
   close to +1. An anti-correlated pair has distance close to 2 and never reaches the `abs()`.
   Verified on the implemented index (log 2026-09-17 (c)): with planted corr = -0.95 pairs and
   T = 0.9 the double filter returns zero of them at every eps. Tag: `supports_neg_corr =
   "not_available"` (the capability cannot be enabled without replacing its filters). ParCorr/CSZ
   is the same tier for the same reason. **No [S]-vs-[E] questions remain.**
9. **Adopt CSZ's tuning protocol** (two-factor combinatorial design + local refinement +
   bootstrapping, §4a.1) as the project-wide competitor-tuning method? It is more rigorous than
   the current single calibration pass, it is published by the competitors themselves, and this
   project already owns the design-of-experiments machinery from the OA and Sobol sweeps.
10. **Add stock *returns* alongside stock *prices*** as a cooperative/uncooperative pair
    (§4a.2). One transform on data we already hold, and it is the axis CSZ shows separates the
    DFT-based methods from the sketch-based ones.

### 10.3 Process note: two capability claims were wrong, and why

Worth recording, because it changes how the remaining claims should be handled.

Across this planning work, **two statements about competitor capabilities turned out to be
wrong once the actual paper was read**, and they failed in opposite directions:

1. **CorrJoin was credited with lag support it does not have.** Source of the error: a
   search-engine summary of the abstract, carried into the plan with an "unconfirmed" hedge that
   was too weak to stop it propagating into three sections.
2. **StatStream was denied lag support, persistence and negative-correlation handling that it
   does have.** Source of the error: an assumption that a 2002 paper must be a primitive
   ancestor, never checked against the paper. This one was worse, because it flattered
   CorrTrack, and flattering errors are the ones least likely to be caught internally.

Both were corrected only because the primary source was eventually read. The rule that follows:

> **No capability claim about a competitor enters the paper unless it was read in that
> competitor's own paper, with a section or lemma number recorded next to it.** Abstracts,
> search summaries, derivative theses and reasoning from a paper's publication date are all
> insufficient.

**Third refinement, 2026-09-16 (user's):** a section or lemma number is necessary but **not
sufficient**. Papers routinely specify capabilities their experiments never exercise -- 
StatStream derives lagged correlation, gives grid pseudocode for it and exposes a `T_M`
parameter, yet runs no lag experiment. So the rule gains a second clause:

> **Record the evidence tier alongside the pointer: evaluated, specified, or merely claimed
> (§5b.6).** Only *evaluated* capabilities support a head-to-head comparison. A *specified* one
> may be implemented, but the result is our evaluation of their design, not a reproduction of
> their result -- and a poor result is not evidence against the method. A merely *claimed* one
> is not implemented as theirs at all.

**And the same standard applies to CorrTrack.** It would be self-serving to grade competitors
on whether they evaluated a capability while claiming our own on the strength of implementation
alone.

Every capability row in §5b.5 now carries a pointer and a tier, and **as of 2026-09-16 no `[?]`
cells remain**: every primary source has been read. One `[S]`-versus-`[E]` question is still
open and marked rather than guessed: whether CorrJoin *evaluates* negative correlation
(§10.2 item 8).

**The scheme is not ours alone.** FilCorr's own Table I grades its competitors as "claimed",
"extendable" and "unknown" (§4b.3). Citing that precedent is stronger than presenting evidence
grading as an invention of ours, and it independently declines to credit StatStream with lag
support.

The same discipline applies to the **speed** claims once measurements start: the §5a.4
prediction-in-advance pattern (record what the result should be, then check) is the analogous
safeguard for numbers, and should be repeated for each arm.

### 10.4 Primary sources: all read as of 2026-09-16

**Read in full**: StatStream (VLDB 2002 **and** technical report TR2002-827),
Cole-Shasha-Zhao (KDD 2005), BRAID (SIGMOD 2005), BRAID/ThinBRAID (TKDD 2010), FilCorr
(ICDM 2020), TSUBASA (SIGMOD 2022), ParCorr (DMKD 2018), CorrJoin (PACMMOD 2023).

The risk flagged in the previous revision has been discharged: **the FilCorr paper, behind an
arm already implemented and already in our published results, has now been read** (§4b). The
port is sound; two deviations were found and are documented rather than quietly carried.

**Optional, not blocking:**
- **Cole, Shasha & Zhao, NYU Technical Report 2005** -- the convolution derivation for
  structured random vectors and the full bootstrapping results. Needed only if we adopt that
  optimization or their tuning protocol (§10.2 item 9).
- **Shasha & Zhu, *High Performance Discovery in Time Series*** (Springer 2003).
- **TSUBASA's extended arXiv version** -- carries the proof of their Lemma 2, omitted from the
  SIGMOD paper. Needed only when implementing the incremental update.

**Deliberately out of scope**, recorded so the decision is visible rather than an oversight:
**SPIRIT** (VLDB 2005, PCA and hidden variables, not a threshold join), **MUSCLES** (ICDE 2000,
multivariate regression with a small fixed lag limit), **COLR-tree** (ICDE 2008, cited by
FilCorr as a lagged-correlation option but an offline spatio-temporal index), and **iSAX**
(SIGKDD 2008, ParCorr's own baseline, an offline index). All belong in related work, not in the
benchmark.
