# Final campaign: the two figures the tables cannot show (2026-09-28)

Built by `abaca/campaign_figures.py` from the same 179 cells as
`docs/campaign_m500_tables_final_2026-09-25.md` (snapshot `101ef2c`, `content=f7aa10f1814d9eee`,
twelve arms per cell, monitoring off, whole stream). Nothing here re-tunes or re-runs anything: the
figures read the result files.

## Disclosure encoded in the marks, not only in the legend

Three of the capability classes are not equally the authors' own, and the figures say so per method
rather than in a footnote. The tier is read from each run's `supports_lags` / `supports_neg_corr`
field, so it cannot drift from what was actually run:

| mark | tier | meaning |
|---|---|---|
| circle | `native` | the authors implemented and evaluated this capability |
| triangle | `enabled_by_us` | the capability is an extension of ours (TSUBASA, ParCorr, CSZ and CorrJoin in class L; FilCorr's negative correlation in class N) |
| square | `specified` | the paper specifies it but never evaluates it (StatStream's lags; the negative correlation of BRAID, ThinBRAID and StatStream) |
| hollow | | the method's median recall in that cell group is below 0.95 |

Where a method shows several tiers across a slice, the figure marks the least author-evaluated one.
That is the conservative direction: it attributes less to the original papers, not more.

## `docs/figures/speedup_vs_density.png`

Speedup against the effective correlation density the cell actually has (correlated pair-windows
over tested), faceted by capability class and space, one point per threshold, median over the six
datasets. Density rather than threshold on the x axis, because density is what decides whether a
filter can help at all, and because the same threshold means a different problem on each dataset.

What it shows: CorrTrack's advantage is a function of density, falling smoothly from the sparse end
toward brute force as the data gets denser, and every crossover with FilCorr happens in the dense
corner of the sparse-to-dense range rather than at a particular threshold. In first differences,
where the competitors' energy-concentration assumptions weaken, the CorrTrack curves sit clearly
above the rest across the whole range.

## `docs/figures/performance_profile.png`

A Dolan-More performance profile over all 179 cells: the fraction of cells where a method is within
a factor tau of the fastest method on that same cell. The left panel ignores recall, the right one
counts a cell only when the method also reached recall 0.95, so the price of an approximate arm's
missed pairs shows up in the same plot instead of a separate column.

Headline numbers from the same data:

| method | fastest on | median tau | cells at recall >= 0.95 |
|---|---|---|---|
| CT-lsh | 83 of 179 | 1.02 | 92% |
| FilCorr | 53 | 1.38 | 100% |
| CT-ham | 37 | 1.18 | 100% |
| CorrJoin | 5 | 2.29 | 66% |
| StatStream | 1 | 2.30 | 92% |
| bf_incr | 0 | 1.59 | 100% |
| BRAID | 0 | 2.87 | 100% |
| TSUBASA | 0 | 4.26 | 100% |
| ParCorr | 0 | 5.21 | 57% |
| ThinBRAID | 0 | 7.67 | 7% |
| CSZ | 0 | 12.79 | 49% |

CorrTrack is the fastest method on 120 of the 179 cells counting both backends, and its typical
distance from the best method on a cell is 2% (CT-lsh). FilCorr is the fastest on 53, which is the
honest other half of the story and is exactly the dense and synchronous corner.

One reading caveat that belongs in the caption: ParCorr, CSZ and CorrJoin plateau near 0.66 in both
panels because class N has no such arm at all, their negative-correlation capability being
`not_available`. That plateau is a missing capability, not slowness, and the profile counts it as
unsolved by design.
