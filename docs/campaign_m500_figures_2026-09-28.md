# Final campaign: the two figures the tables cannot show (2026-09-28)

Built by `abaca/campaign_figures.py` from the same 179 cells as
`docs/campaign_m500_tables_final_2026-09-25.md` (snapshot `101ef2c`, `content=f7aa10f1814d9eee`,
twelve arms per cell, monitoring off, whole stream). Nothing here re-tunes or re-runs anything: the
figures read the result files.

## How to read the numbers

**Naming.** `BF_vect` is the vectorised exact brute force, which is both the ground truth and the
timing anchor inside each cell; `BF_incr` is the exact incremental baseline; CorrTrack appears as
its two tuned backends, `CorrTrack-LSH` and `CorrTrack-Ham`, which the ranking tables collapse into
one method called CorrTrack.

**The three capability classes.** S is synchronous and positive only (n_lags=0, neg_corr=False), L is
lagged and positive only, and N is lagged with both signs: neg_corr=True searches negative
correlation *as well as* positive, so class N is a superset of class L's work, not its opposite.

**What a cell is.** One dataset, one threshold, one capability class (S, L or N), one space (raw or
differenced). The campaign has 179 of them: 6 datasets x 5 thresholds x 3 classes x 2 spaces, minus
one cell that could not be tuned.

**Aggregation, stated per figure.** In `real_speedup_vs_density.png` every point is a MEDIAN over
the datasets of one threshold, and its x position is the median effective density of those same
cells; the top axis of each facet names the threshold the point came from. In
`real_performance_profile.png` nothing is averaged at all: each of the 179 cells is counted once,
and a method counts at tau when it ran within a factor tau of the fastest method on that same cell.

## Disclosure encoded in the marks, not only in the legend

Three of the capability classes are not equally the authors' own, and the figures say so per method
rather than in a footnote. The tier is read from each run's `supports_lags` / `supports_neg_corr`
field, so it cannot drift from what was actually run:

**Legend order, and the leadership count behind it.** Every figure lists the methods in one order:
the methods that lead the most cells first, counted as `abaca/build_ranking_tables.py` counts
leadership (a method leads a cell when its speedup is within 10% of that cell's fastest arm, so
several arms can lead one cell). The qualified columns rank only the arms that reached recall 0.95
in that cell, which is what stops an arm from leading by skipping the pairs it should have found.
Over the 179 cells:

| method | leads | top two | leads (recall >= 0.95) | top two (recall >= 0.95) | median speedup |
|---|---|---|---|---|---|
| CorrTrack-LSH | 116 | 129 | 103 | 116 | 4.25x |
| CorrTrack-Ham | 71 | 125 | 80 | 126 | 3.92x |
| FilCorr | 65 | 79 | 67 | 89 | 4.37x |
| BF_incr | 9 | 52 | 10 | 61 | 3.88x |
| CorrJoin | 7 | 11 | 8 | 12 | 1.64x |
| StatStream | 2 | 9 | 2 | 10 | 2.29x |
| BRAID | 0 | 0 | 0 | 0 | 1.60x |
| TSUBASA | 0 | 0 | 0 | 0 | 1.32x |
| ParCorr | 0 | 0 | 0 | 0 | 0.84x |
| ThinBRAID | 0 | 0 | 0 | 0 | 0.63x |
| CSZ | 0 | 0 | 0 | 0 | 0.32x |

Three things this table says that the median column alone does not. At least one CorrTrack backend
leads 126 of the 179 cells, 70% of the campaign, against FilCorr's 65, while FilCorr has the better
median speedup (4.37x against 4.25x): a method can carry the better typical number and still lead
far fewer cells, because CorrTrack's wins are large and concentrated in the sparse cells whereas
FilCorr is steady everywhere. Requiring recall 0.95 leaves the order unchanged and moves
CorrTrack-Ham up rather than down, which is the robustness check on the leadership claim. And five
of the eleven methods never lead a single cell.

The same table is generated on its own into `docs/campaign_m500_leadership_2026-09-28.md`, and the
order it produces is defined once in `abaca/method_style.py` together with the colours, so the
synthetic figures read the same way.

| mark | tier | meaning |
|---|---|---|
| circle | `native` | the authors implemented and evaluated this capability |
| triangle | `enabled_by_us` | the capability is an extension of ours (TSUBASA, ParCorr, CSZ and CorrJoin in class L; FilCorr's negative correlation in class N) |
| square | `specified` | the paper specifies it but never evaluates it (StatStream's lags; the negative correlation of BRAID, ThinBRAID and StatStream) |
| hollow | | the method's median recall in that cell group is below 0.95 |

Where a method shows several tiers across a slice, the figure marks the least author-evaluated one.
That is the conservative direction: it attributes less to the original papers, not more.

## `docs/figures/real_speedup_vs_density.png`

Speedup against the effective correlation density the cell actually has (correlated pair-windows
over tested), faceted by capability class and space, one point per threshold, median over the six
datasets. Density rather than threshold on the x axis, because density is what decides whether a
filter can help at all, and because the same threshold means a different problem on each dataset.

What it shows: CorrTrack's advantage is a function of density, falling smoothly from the sparse end
toward brute force as the data gets denser, and every crossover with FilCorr happens in the dense
corner of the sparse-to-dense range rather than at a particular threshold. In first differences,
where the competitors' energy-concentration assumptions weaken, the CorrTrack curves sit clearly
above the rest across the whole range.

## `docs/figures/real_performance_profile.png`

A Dolan-More performance profile over all 179 cells: the fraction of cells where a method is within
a factor tau of the fastest method on that same cell. Three panels: speed alone, then a cell
counted only when the method also reached recall 0.95, then the same at recall 0.90. The looser
level separates a method that misses the target by a little from one that misses most of the
correlated pairs, and the two are not the same story.

Headline numbers from the same data:

Each row below is over all 179 cells, not a median of medians: "fastest on" counts cells, "median
tau" is the median over cells of that method's factor away from the fastest method on the same cell.

| method | fastest on | median tau | cells at recall >= 0.95 | at recall >= 0.90 |
|---|---|---|---|---|
| CorrTrack-LSH | 83 of 179 | 1.02 | 92% | 98% |
| FilCorr | 53 | 1.38 | 100% | 100% |
| CorrTrack-Ham | 37 | 1.18 | 100% | 100% |
| CorrJoin | 5 | 2.29 | 66% | 66% |
| StatStream | 1 | 2.30 | 92% | 97% |
| BF_incr | 0 | 1.59 | 100% | 100% |
| BRAID | 0 | 2.87 | 100% | 100% |
| TSUBASA | 0 | 4.26 | 100% | 100% |
| ParCorr | 0 | 5.21 | 57% | 62% |
| ThinBRAID | 0 | 7.67 | 7% | 20% |
| CSZ | 0 | 12.79 | 49% | 59% |

The two recall columns separate three different situations. CorrTrack-LSH and StatStream sit just
under the strict target and recover almost entirely at 0.90 (92% to 98% and 92% to 97%). ThinBRAID
and CSZ are far below it and stay below it (7% to 20%, 49% to 59%), so their speed is bought with
pairs they never report. CorrJoin does not move at all (66% at both levels), because its shortfall
is not recall but capability: class N has no CorrJoin arm.

CorrTrack is the fastest method on 120 of the 179 cells counting both backends, and its typical
distance from the best method on a cell is 2% (CorrTrack-LSH). FilCorr is the fastest on 53, which is the
honest other half of the story and is exactly the dense and synchronous corner.

### When CorrTrack misses the recall target, who takes its place

The per-cell tables keep a column for any arm that beats CorrTrack in speed or reaches recall 0.95
where CorrTrack cannot. The second half of that rule never fires, because CorrTrack always has one
backend at or above the target, so the sharper question is the one a reviewer will ask: where
CorrTrack's *faster* backend misses the target, which arm is then the fastest that reaches it?

Across the campaign that happens in 14 cells, 13 of them smartmeter. In 11 the answer is CorrTrack's
own second backend; in 3 a competitor takes it, by a small margin: CorrJoin 9.71x against
CorrTrack-Ham's 9.23x (smartmeter, class L raw, T=0.95), FilCorr 4.50x against 2.86x (wikipedia,
class N raw, T=0.70) and FilCorr 4.29x against 3.55x (smartmeter, class N differenced, T=0.80). So
the honest version of the claim is not that recall is never a reason to choose another method, but
that it is a reason in 3 cells of 179, all of them at a margin under 1.6x. Counting every cell,
the fastest arm that meets recall 0.95 is a CorrTrack backend in 118 of 179, FilCorr in 54, CorrJoin
in 6 and StatStream in 1.

### Why CorrTrack-LSH stops at 0.92 in the recall-gated panel

It is not a tau effect: the curve is flat after about tau=3, so those cells are never counted at any
tau. They are the 14 cells of 179 where CorrTrack-LSH's own recall is below 0.95, and the profile
counts a cell as unsolved when the recall requirement is not met, whatever the speed. 13 of the 14
are smartmeter, the W=48 / step=8 dataset, where its recall lands between 0.868 and 0.948; the
fourteenth is wikipedia, class N, raw, T=0.70 at recall 0.947. CorrTrack-Ham reaches 0.955 to 0.997
on those same cells, which is why its curve does reach 1.0, and that difference is the point of
carrying two backends: the LSH candidate search is probabilistic, and smartmeter's correlations sit
at densities around 1e-6, where a probabilistic filter drops a handful of pairs that the exact
Hamming search keeps.

The third panel puts a size on the shortfall. At recall 0.90 CorrTrack-LSH reaches 98%: only three
cells remain, all smartmeter differenced, at recall 0.868, 0.878 and 0.882. So the misses are a
narrow band just under the target on one dataset, not a systematic loss, and they are covered by
the other backend.

One reading caveat that belongs in the caption: ParCorr, CSZ and CorrJoin plateau near 0.66 in both
panels because class N has no such arm at all, their negative-correlation capability being
`not_available`. That plateau is a missing capability, not slowness, and the profile counts it as
unsolved by design.
