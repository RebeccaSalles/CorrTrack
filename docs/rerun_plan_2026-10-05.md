# Does the accumulator fix require rerunning the campaigns? (2026-10-05, plan only)

## What the fix changed, measured

Same cell, same cluster, same tuned parameters, one probe each (jobs 3170213 and 3170283):

| | total wall | median window | slowest window | windows above 1.5x the median |
|---|---|---|---|---|
| doubling array (campaign code) | 15.00 s | 14.79 ms | 7.9x | 7 |
| segmented blocks (fixed code) | 15.21 s | 15.19 ms | 1.7x | 6 |

The tail is gone: the worst window drops from 7.9x the median to 1.7x, and the remaining bumps are
the allocation of a new 28 MB block rather than a copy of the whole set. **Throughput did not
change**: 15.00 s against 15.21 s is inside run-to-run variation, and the arithmetic agrees, since
the spikes were only about 1.5% of the stream's total time.

The fix is therefore a latency fix, not a speed fix, and it is not CorrTrack-specific: every arm
records through the same accumulator. Brute force pays the same copies, they are simply invisible
inside a window twenty times longer (0.08% of its total against 1.5% of CorrTrack's).

## What that means for the published numbers

Rerunning would raise CorrTrack's speedups by about 1.5% and every other arm's by less. The ranking
tables use a 10% tie band, the capability tables quote two decimals, and the leadership counts turn
on differences far larger than this. No row would change hands.

## The options, with cost

**A. Do not rerun the campaigns; measure the latency separately (recommended).** Keep the 179-cell
campaign and the 64-cell synthetic study as they are, one code state, and run the fixed code only
where the claim needs it: the per-window profile. Six probe jobs (m in {500, 2000} x three densities,
CorrTrack and brute force) give panel C its figure and the implementation note its numbers. Cost:
under an hour of cluster time, no retuning. Risk: the paper's figure C comes from a code state that
differs from the tables' by one commit, which has to be stated in the caption.

**B. Rerun everything on the fixed code.** 179 campaign cells at four jobs each (lsh hyperopt,
hamming hyperopt, CSZ tuning, N-way) plus 64 synthetic cells, so 972 jobs, about two days wall at
the concurrency the queue allowed last time, plus the validation pass and the rebuild of every table
and figure. Gains: one code state for everything, CorrTrack's numbers improve by about 1.5%. Risk:
two days of cluster time and a fresh chance of the partial-overwrite class of bug that cost us a
full campaign in September, for a change below the tie band.

**C. Rerun the synthetic campaign only.** 64 cells, 256 jobs, about half a day. The synthetic study
is where the per-window and scaling claims live, so this buys consistency exactly where latency is
discussed, and leaves the real-data tables on their own state. Middle cost, and the two campaigns
then differ by one commit, which is worse than either A or B for a reader.

## Recommendation

A, with the caption stating that the latency panel was measured on the fixed accumulator and the
throughput tables on the campaign's state, plus one sentence giving the measured difference (1.5% of
total time, no change to any ranking). If a reviewer asks for a single state, B is the fallback and
should then be run as one campaign with the same protocol, not as a patch of the existing results.

## Either way, before any rerun

The fix must be committed and a fresh snapshot built, since the frozen campaign snapshot was
restored to its committed state and the probe ran from a separate copy
(`snapshots/probe_blocks/`). Any rerun starts from a new snapshot whose fingerprint is recorded with
the results, as every campaign here does.
