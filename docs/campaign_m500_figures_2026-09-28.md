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

## `docs/figures/real_performance_profile.png` and `real_performance_profile_authors.png`

(2026-10-07, user) Replaces the Dolan-More curves, which pooled the classes and were hard to read. One panel
per capability class, one bar per method that can run that class. Each bar splits the class's cells by how far
the method runs from the fastest method on the same cell (classes stacked, each panel as tall as its
number of methods): fastest (within the 10% tie band of the ranking
tables, so several methods can be "fastest" on one cell and the first segments add up to more than 100% across
methods), up to 2x slower, 2x to 5x slower, over 5x slower. The two grey segments are the cells where the
method solved the problem below recall 0.95 (0.90 to 0.95, then below 0.90), whatever its speed. Speed is
always measured against the fastest method on the cell, whatever that method's recall. Methods are sorted by
how much of their bar lies close to the fastest.

Why per class: the classes admit different sets of methods (TSUBASA, ParCorr, CSZ and CorrJoin cannot run
class N in either version, and in the authors-only version they and FilCorr run fewer classes still). Pooled,
a method was counted as failing cells it could never attempt, and each cell's fastest method came from a
different field depending on the class. Raw and differenced cells are pooled inside a panel: every method in
a panel covers both spaces and each cell is measured against its own fastest method.

All-arms version (179 cells): a dagger marks a method whose capability for that class was added by us, a
double dagger one specified by its authors but never evaluated there. ThinBRAID's class N bar ends at 98%
because it has no result in one of the 60 cells.

| class S: synchronous, positive only (59 cells) | fastest (within 10%) | up to 2x | 2x to 5x | over 5x | recall 0.90 to 0.95 | below 0.90 |
|---|---|---|---|---|---|---|
| CorrTrack-Ham | 66% | 34% | 0% | 0% | 0% | 0% |
| CorrTrack-LSH | 61% | 34% | 0% | 0% | 3% | 2% |
| FilCorr | 42% | 47% | 10% | 0% | 0% | 0% |
| BF_incr | 8% | 69% | 22% | 0% | 0% | 0% |
| BRAID | 0% | 68% | 32% | 0% | 0% | 0% |
| StatStream | 0% | 59% | 25% | 3% | 3% | 8% |
| CorrJoin | 0% | 41% | 56% | 3% | 0% | 0% |
| TSUBASA | 0% | 0% | 80% | 20% | 0% | 0% |
| ParCorr | 0% | 0% | 51% | 32% | 5% | 12% |
| ThinBRAID | 0% | 0% | 3% | 5% | 14% | 78% |
| CSZ | 0% | 0% | 0% | 71% | 15% | 14% |

| class L: lagged, positive only (60 cells) | fastest (within 10%) | up to 2x | 2x to 5x | over 5x | recall 0.90 to 0.95 | below 0.90 |
|---|---|---|---|---|---|---|
| CorrTrack-LSH | 68% | 15% | 8% | 0% | 7% | 2% |
| FilCorr | 23% | 28% | 43% | 5% | 0% | 0% |
| CorrTrack-Ham | 20% | 70% | 10% | 0% | 0% | 0% |
| CorrJoin † | 12% | 15% | 58% | 15% | 0% | 0% |
| BF_incr | 2% | 40% | 47% | 12% | 0% | 0% |
| StatStream ‡ | 2% | 33% | 55% | 5% | 5% | 0% |
| BRAID | 0% | 0% | 40% | 60% | 0% | 0% |
| TSUBASA † | 0% | 0% | 38% | 62% | 0% | 0% |
| ParCorr † | 0% | 0% | 23% | 65% | 10% | 2% |
| CSZ † | 0% | 0% | 3% | 73% | 15% | 8% |
| ThinBRAID | 0% | 0% | 0% | 8% | 17% | 75% |

| class N: lagged, both signs (60 cells) | fastest (within 10%) | up to 2x | 2x to 5x | over 5x | recall 0.90 to 0.95 | below 0.90 |
|---|---|---|---|---|---|---|
| CorrTrack-LSH | 43% | 33% | 13% | 0% | 8% | 2% |
| FilCorr † | 43% | 30% | 27% | 0% | 0% | 0% |
| CorrTrack-Ham | 33% | 45% | 22% | 0% | 0% | 0% |
| BF_incr | 5% | 63% | 32% | 0% | 0% | 0% |
| StatStream ‡ | 2% | 15% | 65% | 10% | 8% | 0% |
| TSUBASA | 0% | 0% | 63% | 37% | 0% | 0% |
| BRAID ‡ | 0% | 0% | 63% | 37% | 0% | 0% |
| ThinBRAID ‡ | 0% | 0% | 0% | 5% | 7% | 87% |

Authors-only version (143 cells: ThinBRAID dropped, T below 0.8 dropped, and only capabilities the authors
evaluated):

| class S: synchronous, positive only (47 cells) | fastest (within 10%) | up to 2x | 2x to 5x | over 5x | recall 0.90 to 0.95 | below 0.90 |
|---|---|---|---|---|---|---|
| CorrTrack-Ham | 72% | 28% | 0% | 0% | 0% | 0% |
| CorrTrack-LSH | 64% | 30% | 0% | 0% | 4% | 2% |
| FilCorr | 32% | 55% | 13% | 0% | 0% | 0% |
| BF_incr | 6% | 66% | 28% | 0% | 0% | 0% |
| StatStream | 0% | 66% | 19% | 0% | 4% | 11% |
| BRAID | 0% | 60% | 40% | 0% | 0% | 0% |
| CorrJoin | 0% | 47% | 53% | 0% | 0% | 0% |
| TSUBASA | 0% | 0% | 74% | 26% | 0% | 0% |
| ParCorr | 0% | 0% | 55% | 30% | 2% | 13% |
| CSZ | 0% | 0% | 0% | 72% | 13% | 15% |

| class L: lagged, positive only (48 cells) | fastest (within 10%) | up to 2x | 2x to 5x | over 5x | recall 0.90 to 0.95 | below 0.90 |
|---|---|---|---|---|---|---|
| CorrTrack-LSH | 69% | 15% | 6% | 0% | 8% | 2% |
| CorrTrack-Ham | 19% | 75% | 6% | 0% | 0% | 0% |
| FilCorr | 17% | 25% | 52% | 6% | 0% | 0% |
| StatStream | 4% | 23% | 65% | 2% | 6% | 0% |
| BF_incr | 0% | 31% | 54% | 15% | 0% | 0% |
| BRAID | 0% | 0% | 33% | 67% | 0% | 0% |

| class N: lagged, both signs (48 cells) | fastest (within 10%) | up to 2x | 2x to 5x | over 5x | recall 0.90 to 0.95 | below 0.90 |
|---|---|---|---|---|---|---|
| CorrTrack-LSH | 56% | 25% | 8% | 0% | 8% | 2% |
| CorrTrack-Ham | 44% | 42% | 15% | 0% | 0% | 0% |
| BF_incr | 21% | 40% | 40% | 0% | 0% | 0% |
| StatStream | 2% | 21% | 60% | 6% | 10% | 0% |
| BRAID | 0% | 0% | 56% | 44% | 0% | 0% |
