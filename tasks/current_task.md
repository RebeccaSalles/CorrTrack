# Current Task

> `docs/` and `tasks/` did not exist in this checkout before this session — see the note at the
> top of `docs/implementation_log.md`. This file tracks only what's directly in front of this
> session; it is not a full project task tracker.

## Current branch
main. **Real uncommitted source changes, not committed yet (no request to commit was made):**
`candidate_kernels.pyx` (rebuilt .so repeatedly today: overlap-corrected b_min, the shared
sizing-function extraction, the recall safety margin, the configurable-margin parameter, int32
downsizing of `_band_keys`/`_neg_band_keys`/`_membership_pos`/`_post_capacity`/`_post_count`,
then the free-list slot-reuse fix for `SignLSHBandIndex`/`HammingExactIndex` plus the
entry_id-must-equal-slot-index fix it required), `library_corrtrack_parallel.py`,
`experiment_run_param_grid.py`, `experiment_run_exec_param.py`, `corrtrack_param_search.py`,
`test_stable_reproduced_changes.py` (also gained the `_win_free_idx`/`_release_window_idx`
window-idx-cache free-list fix). See `docs/implementation_log.md`'s 2026-09-03 and
2026-09-04 (a)/(b)/(c)/(d)/(e) entries.

## What this session did
Refined the OA(16,5,4,2) sweep dashboard through many rounds of user feedback (axis labels,
legends, sortable tables, box-and-whisker views, a real total-vs-brute-force projection
inconsistency bug fix, de-duplicating the Parameter Sensibility tab), then reproduced the same
refined structure for the companion Sobol sweep's 83 points in a new dashboard, adapting the
analysis method from OA's balanced-averaging to a proper multivariate regression since the Sobol
design has no discrete levels to average over. Went on to: rerun the m=296 outlier cell and
disclose its still-unresolved candidate_time anomaly; fix a real single-point-anchoring bug in
the extrapolation methodology (now median-based); add R²/noise-floor confidence metrics
throughout; diagnose and fix validation_time's weak fit with a 2-stage model
(tested_candidates-driven, R² 0.79→0.925); add a density-on-x Filtering chart, regression
trend-lines on Parameter Sensibility, 2-milestone Extrapolation charts, clearer Reliability
wording, and a visually separated pipeline-total card in Complexity. Built a second, cleaner
"highlight reel" dashboard (`corrtrack_story.html`) derived from the Sobol one, then iterated it
(pairs=m²L toggle, recall-floor/sign-detection explanation, a "how it works" supersection with a
complexity table, a live γ/occupancy calculator, and a measured phase-time-share visual). Most
recently, investigated (at the user's explicit request) WHY `tested_candidates` outgrows the
exact `m²L` brute-force space it's drawn from, tracing it to false-positive volume specifically,
then to the calibration's own `n_bands_tolerance`/`gamma` selection — and found that climb is
mostly a calibration-reliability artifact (small-m recall estimates are noisy and let
insufficient settings slip through) rather than a genuine "harder at scale" law. See
`docs/implementation_log.md`'s 2026-09-01 entry (17th–19th follow-ups) for full detail and exact
numbers.

**Then, escalated from dashboard work to a real production code fix:** the user asked whether
`b_min` could be corrected (holding `n_vectors` fixed) so achieved recall reliably reaches
target, not just gets measured more consistently. Derived an exact overlap-corrected LSH-band
recall formula (bands share a fixed bit pool, so the old formula's independent-bands assumption
provably overestimates recall — Jensen's inequality, not just an empirical fudge), implemented it
in `candidate_kernels.pyx`, validated it against the real 83-cell Sobol sweep's trial data (bias
+7.0→−0.3 points, R² −1.08→0.10), then — per the user's explicit follow-up asks — changed
`CorrTrack`/`Candidates`' default so `n_bands_tolerance=None` means AUTO (was: disabled) and
removed the now-unnecessary tolerance sweep from `experiment_run_param_grid.py`. Full test suite
green (113/113) throughout. See `docs/implementation_log.md`'s **2026-09-03 entry** for the full
derivation, validation numbers, a self-caught bug in an early version of the validation script,
and the real end-to-end smoke test confirmation (95.10% measured vs. 95.05% predicted).

**Then, two more rounds building on that fix:** (a) `OPTIM_PROXY_MAX_PAIR_ROWS` turned out to be
a single flat constant that silently strangled the real production hyperopt phase at m=150
(affording less than one anchor's own pair universe) — replaced with
`recommend_proxy_pair_row_budget`, computed per-dataset from the real `(m, L)`, bounded by a new
`OPTIM_PROXY_PAIR_ROW_HARD_CEILING`. (b) A full discussion-then-implementation round on the
hyperopt selection RULES themselves (`_apply_proxy_anchor_selection`) — `n_bands` (exact,
theoretical, free) replaces `candidate_rate` (shown directly to barely vary across a 3.6x real
cost swing) as the primary ranking signal; a data-driven recall safety margin (+0.028,
calibrated from the same Sobol residual data) is now baked into the sizing formula itself
instead of chased at selection time; bootstrap CI width is a new continuous tie-break sitting
alongside the binary underpowered flag; anchor-expansion is now bounded by the real pair-row
ceiling instead of an arbitrary round count. **Caught and fixed a real ordering bug in my own
first implementation** (CI width placed too early in the sort chain, silently overriding
n_bands entirely) via a real end-to-end rerun before considering it done — see
`docs/implementation_log.md`'s **2026-09-04 (a)/(b) entries**.

## Artifacts (not part of the git repo — published separately)
- OA(16,5,4,2) dashboard: `https://claude.ai/code/artifact/b7d77cd2-b60e-47a4-8892-83a8b30d77a0`
- Sobol sweep (83 points) technical dashboard: `https://claude.ai/code/artifact/0a673d9b-9b05-47e4-90d7-07bdebfdb38d`
- Sobol "highlight reel" story page: `https://claude.ai/code/artifact/3cd502b5-f51c-4dc4-bf87-89005358b85c`
- Local working copies: `/tmp/claude-1000/-home-rsalles/78a16dca-b5b8-4fc2-98e0-91fa6c37367d/scratchpad/oa_dashboard.html`,
  `sobol_dashboard.html`, and `corrtrack_story.html`, plus the exported data files feeding them
  (`oa_data_full.json`, `oa_trials.json`, `sobol_data_full.json`, `sobol_trials.json`) — these are
  scratch-space files, not committed anywhere.

## Commands run
Reading sweep CSVs/JSONs with `pandas`/`json` for dashboard data export; `esprima` (pip-installed
for this session) to syntax-check generated JavaScript before each publish. Later:
`python3 setup_cython.py build_ext --inplace` (rebuilt `candidate_kernels.pyx` twice — once for
the `_corrected_b_min` fix, again after adding `debug_*` test wrappers) and
`python3 -m pytest test_stable_reproduced_changes.py -q` (run 3 times across the session: after
the `_finalize_sizing` fix alone, after the default-behavior change, and again after the
`experiment_run_param_grid.py` edit — 113/113 green every time). An ad hoc real `CorrTrack`
smoke test (m=150, threshold=0.75, corr_prop=0.05, `n_steps=60`) was also run directly (not via
pytest) — see Known issues, its result is the one open concern.

## Results observed
All three dashboards publish and pass their integrity checks (JSON validity, JS syntax validity,
`getElementById` cross-checks, no duplicate ids). The Sobol dashboard's regression-fitted
brute-force exponent (2.04) matches the theoretical O(m²) sanity check, cross-validated against
an independent `numpy.linalg.lstsq` computation on the same data. The calibration-reliability
investigation's numbers were independently verified in Python against the exact JSON embedded in
the dashboard (byte-for-byte match confirmed) before publishing, so the in-page live-computed
numbers and this log's numbers are the same computation, not two independently-drifting ones.

## Known issues / open items
- **Resolved:** the full redesigned hyperopt selection (n_bands-primary + recall margin + CI
  tie-break + hard-ceiling expansion) was validated end-to-end (m=150, gamma=0.6, occupancy ∈
  {2,3,5,10}) and initially picked the WRONG config (219 bands, not the smallest feasible 88) —
  a real ordering bug (`_ci_width_rank` placed too early, a continuous value silently overriding
  `_n_bands_rank`). Fixed (moved to the end of the sort chain) and reconfirmed: `n_bands=88`
  (occupancy=10) now wins, matching both the design intent and the earlier manual
  investigation's own best-speedup finding (13.79x). See `docs/implementation_log.md`'s
  2026-09-04 (b) entry for the full before/after.
- `proxy_candidate_rate_close_tolerance`/`OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE` are now
  dead (unused by the new ranking) but were deliberately left in place rather than a further
  cross-file removal — flagged as an available follow-up if the user wants that cleaned up too.
- **Resolved:** the `n_steps=60` smoke test's ~62%-vs-95%-predicted recall gap was confirmed to
  be a too-short-run artifact, not a formula problem — a `n_steps=300` rerun of the identical
  config measured 95.10% recall at tolerance=1.0 (the new AUTO default), against the corrected
  formula's own 95.05% prediction. See `docs/implementation_log.md`'s 2026-09-03 entry for the
  full numbers. The corrected-formula default change is now confirmed end-to-end.
- `candidate_search_lsh_candidates_touched` (real pre-filter count) is not captured by either
  sweep script directly — but the 19th follow-up found a usable proxy already sitting in the
  existing trial-grid JSON (stage-1 trials run with gamma=0, so their `tested_candidates` IS a
  real measured pre-gate/touched count) — no rerun needed to get real touched numbers for the
  83 existing points, only for any NEW points.
- Gamma's stage-2 calibration trials aren't logged individually anywhere (only the summary
  fields) — disclosed, not fixed; this is what blocked doing the exact same fixed-slice
  diagnostic for gamma that was done for tolerance (had to use an indirect proxy — the
  gap-from-tightest-gamma correlation — instead).
- Whether the tolerance/gamma-driven share of tested_candidates' growth actually plateaus at
  larger m (as the calibration-reliability finding predicts) is untested — see next step.
- This checkout's `docs/`/`tasks/` directories were missing entirely before this session; if the
  "real" canonical history lives in a sibling checkout (per the project's own CLAUDE.md notes
  about shared git history across sibling directories), that history was not reconciled here.

## Then: memory/RSS reduction (user asked directly about OOM)
Found that this session's own 2026-09-03 fixes (overlap correction + recall margin) inflate
`n_bands` far more at realistic scale than the m=150 spot-check suggested — 8-11x at m=1000-3000
vs. ~3-4x at m=150 (compounds with scale, not flat). Measured REAL memory (with actual data
inserted, not lazy-allocated-and-never-touched pages, which understated the first attempt) at
m=1000: 973-band current-default config uses 1559MB. Implemented and verified two fixes: (1)
`RECALL_SAFETY_MARGIN` is now a real, overridable parameter
(`candidate_lsh_recall_safety_margin`, `None`=calibrated default 0.028, any explicit value
including 0.0 overrides), not a hardcoded constant; (2) `_band_keys`/`_neg_band_keys`/
`_membership_pos` AND `_post_capacity`/`_post_count` (found while measuring fix 1's impact --
comparably large, not originally scoped) downsized int64→int32, verified safe by value-range
analysis before touching anything. Combined real result: 1559MB → 1199MB automatically (int32,
no config change) → 693MB if also setting margin=0.0 (55% total reduction). Full details,
numbers, and what wasn't done (see below) in `docs/implementation_log.md`'s **2026-09-04 (c)
entry**.

## Then: real root cause of OOM found AND FIXED — `SignLSHBandIndex`/`HammingExactIndex` never reclaimed expired slots
The (c) memory fixes above only reduce the *bounded, one-shot* footprint. The user's actual
m=300 OOM was caused by something separate and more severe: a real streaming run's RSS grew
continuously and *without bound* (267MB→1638MB over just 500 steps at m=300, still climbing).
Root cause, found by reading `candidate_kernels.pyx` directly: `drop_before_time` correctly
marked expired entries dead and removed them from postings (query correctness/recall unaffected
— the class's own `_alive_count` stayed bounded at `m*L` as its docstring claims), but **no code
path ever reused a dead slot's array index** — `insert_many` always allocated the next
never-used slot at `self._count` (monotonically increasing, never decremented), and
`_ensure_capacity` grew every backing array to match, doubling reactively. So physical memory
tracked *total items ever inserted over the run's lifetime*, not the bounded logical alive count.
Pre-existing bug, not introduced this session; `HammingExactIndex` shares the exact same bug
(confirmed by reading it too, not assumed).

**Fixed, user confirmed and asked to proceed.** Implemented a free-list of dead slot indices for
both classes (`drop_before_time` pushes, `_insert_one` pops before ever growing `_count`/
`_capacity`). **Caught a real second bug while implementing, via 2 real test failures (not just
"it compiled"):** the query code (`_find_pair_rows_meta`) uses a returned `entry_id` directly as
an array subscript into every per-slot array — it was never an independent id, it only ever
*coincided* with the slot index because, pre-fix, slots were never reused. Making slots reusable
without also making `entry_id` track the slot broke that coincidence and caused real query
misindexing. Fixed by setting `entry_id = i` (the slot index) directly and removing the now-dead
`_next_entry_id` field. Full details, the exact tests that caught it, and why the fix is safe
(entry ids only ever used within the same step, and cleaned from `_reverse_entry_ids` in lockstep
with expiry) in `docs/implementation_log.md`'s **2026-09-04 (d) entry**.

**Verified end-to-end:** full suite green (115/115). Rerunning the same 500-step m=300 diagnostic
that found the leak: `lsh_count`/`lsh_capacity` now plateau exactly at the theoretical steady
state (4500 = m*L, capacity 8192) from step 50 onward — vs. the unbounded climb to 146,100/262,144
before the fix. RSS growth rate: ~0.136MB/step post-fix vs. ~2.83MB/step pre-fix (~21x reduction).
Disclosed honestly: RSS is not perfectly flat post-fix (309.7→371.1MB, steps 50→500) — a small
residual growth remains, not traced to a specific cause, over an order of magnitude smaller than
the bug just fixed. Debug `count`/`capacity` properties on `SignLSHBandIndex` kept permanently;
scratch investigation scripts (`memory_investigate.py`, `memory_investigate_nomonitor.py`,
`memory_trace.py`, `memory_investigate2.py`) deleted now that the fix is verified.

## Then: second leak found and fixed — `Candidates._window_idx`/`_win_sid*` never pruned; residual RSS traced to intended output, not a bug
User asked to investigate further since RSS "should be flat theoretically." Extended the
diagnostic to `Candidates`-level structures: `_win_sid`/`_window_idx` grew unboundedly
(11,100→146,100 over steps 50→500), the exact same trajectory `lsh_count` had before its own fix
— a second occurrence of the same "monotonic index, never reclaimed" bug, one layer up in Python.
`_clean_old_sketches` already pruned `sketches`/`_reverse_entry_ids` per expiring key but never
touched this window-idx cache. **Fixed:** a free-list (`_win_free_idx`) + `_release_window_idx`
helper, wired into `_clean_old_sketches`'s existing cleanup loop; `_get_or_create_window_idx`
reuses freed indices before growing. Verified: full suite green (115/115); `win_sid_len`/
`window_idx_dict` now plateau at 4,500 (= m*L) from step 50 onward, matching `lsh_count`.

**Residual RSS investigated further — real methodology bug found in how it was measured, not a
new leak.** The prior "~0.136MB/step residual" used `ru_maxrss`, a historical PEAK that can only
increase, never actual current memory. Redone with real current RSS (`/proc/self/status`): growth
decelerates (~0.10→0.02MB/step) rather than climbing linearly — the signature of a converging,
bounded process, not a leak. `gc.collect()`/`malloc_trim(0)` had zero effect (rules out
reclaimable garbage). `tracemalloc` traced the largest remaining live-object growth to
`self.correlated`/`self._maxlag_state` — this project's own **intended, by-design output
accumulators** (every genuine correlation found, kept for final CSV export), not a bug. Full
evidence and numbers in `docs/implementation_log.md`'s **2026-09-04 (e) entry**. **Conclusion:
RSS is now flat in the sense that matters** — both real structural leaks are fixed; what's left
is expected algorithm output plus ordinary NumPy warmup.

## Then: Sobol sweep rerun — widened, then real hyperopt calibration added, then rescoped under a hard 28-hour deadline
Long arc, three stages:
1. User asked to rerun the Sobol sweep with wider m/L now that memory is fixed (moderate
   `(50,500)/(8,96)` accuracy sweep + new `experiment_lsh_sobol_sweep_timing_only.py` at
   `(50,3000)/(8,128)`, no brute force). Launched; timing-only completed (58/64 cells; old
   83-point accuracy result preserved at `tmp_artifacts/lsh_sobol_sweep_v1_m300L64/`).
2. User then clarified neither sweep actually used `CorrTrack_optimize`'s real proxy-anchor
   hyperopt, and asked for it to be used directly (both sweeps), embracing its real limitations
   (a documented, pre-existing struggle at low corr_prop — user explicitly chose to let it
   struggle rather than work around it). Built `calibrate_via_proxy_hyperopt` +
   `theoretical_sizing_fallback` (`experiment_lsh_cost_sweep.py`). Found and fixed a real gap
   while doing this: `candidate_lsh_target_occupancy` was computed but never reached the CSV
   (`OPTIM_RESULT_COLUMNS` was missing it) — fixed.
3. **Real, measured cost wall found**: a single proxy-anchor's own pair universe is O(m²·L) in
   pure Python — 5.7M pairs at (m=300,L=64), 1.15B at (m=3000,L=129) — computationally
   impractical above a real, measured ~25µs/pair rate. Added a feasibility gate
   (`PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR=2,000,000`, ~50s worst case) with disclosed
   fallback (`hyperopt_used` column). Then the user added a **hard 28-hour deadline** plus:
   results comparable to the last (m≤300) sweep, some timing-only higher-m examples,
   `touched_candidates` tracked (`CorrTrack.candidate_search_lsh_candidates_touched`, already
   accumulated, just wasn't read), and a derived no-compute "speedup with validation (CorrTrack)
   + monitoring (both sides) disregarded" metric. Rescoped: accuracy sweep back to
   `(50,300)/(8,64)` (directly comparable, ~8.0h estimated); timing-only range left at
   `(50,3000)/(8,128)` (pushing higher risked single cells taking 5-6h per the fitted cost law —
   too risky for the deadline; re-running the same range with the new metrics **is** the "updated
   higher-m" ask). Both scripts now launched **concurrently** (not sequentially), each
   `--worker-memory-limit-gb 3.0` (was 6.0) so two concurrent workers fit this 7.8GB host.

Full derivation, every real number measured (not guessed), and the bug caught in end-to-end
smoke testing (a stale key name crash in the timing-only script's log line, fixed and reverified)
in `docs/implementation_log.md`'s **2026-09-08 entry**.

**Status: both running, launched 2026-09-08 08:27 CEST, confirmed processing real cells.** Old
timing-only result (no hyperopt, 58/64 cells, m up to 2470) kept at
`tmp_artifacts/lsh_sobol_sweep_timing_only_v1_nohyperopt_m3000/` for comparison.

## Then: user objected the feasibility threshold/fallback defeats CorrTrack's own scalability-and-accuracy promise -- fixed with internal series subsampling (real production change), not a benchmark workaround
Stopped both sweeps again. User's objection was correct: `theoretical_sizing_fallback` picks
occupancy/gamma with NO search once a cell is "infeasible" -- backwards for a method whose whole
point is scaling without giving up accuracy. Real fix implemented in `CorrTrack_optimize` itself
(not just these sweep scripts): `n_bands` already has a closed-form, provably-correct formula
(no search needed); what hyperopt actually searches for -- `target_occupancy`/`gamma` -- doesn't
depend on series count, only on `corr_threshold`/`n_vectors`/data characteristics. So when the
full series population would make a proxy anchor's ground-truth reference impractical,
`CorrTrack_optimize` now calibrates on a **stratified subsample** (variance/kurtosis/lag-1-
autocorrelation composite score, evenly-spaced systematic sample -- spans real, heterogeneous
data's diversity instead of a uniform random draw, per the user's own explicit concern) sized
adaptively from the same memory-safe budget (`proxy_series_subsample_max_pairs`, default
3,000,000). `n_bands` itself is always sized for the TRUE series count, so the recall guarantee
at real scale is unaffected by calibration-time subsampling.

**A second real bug caught by the SAME verification test**: the caller still sized `anchor_count`
off the true (pre-subsampling) series count, capping it to 1 at (m=296,L=55) even though the
actual subsampled cost could afford more -- needlessly starving the statistical power the whole
fix was meant to restore. Fixed (`estimate_proxy_series_subsample_cap` computed by the caller too,
consistently with what `CorrTrack_optimize` does internally).

**A third bug, found immediately after**: both sweep scripts passed the SAME constant
(`PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR`) as both the per-anchor subsample budget AND the
total `pair_row_hard_ceiling` -- since a subsampled anchor's own cost sits right at that budget
by construction, this left room for exactly 1 anchor regardless, silently reproducing the
pre-fix symptom. Fixed by decoupling: new `PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING`, separate from
the per-anchor budget.

**A fourth issue, this time a real cost/benefit finding rather than a bug**: the first decoupled
value tried (3x the per-anchor budget) is memory-safe (confirmed: 3.3GB peak, no swap) but caused
a severe, non-linear TIME regression at the exact pathological cell -- 3,042s (>50min) at
`anchor_count=3` vs. 220s at `anchor_count=1`, for ZERO statistical-power gain (`proxy_n_gt`
stayed at 1 either way). Lowered the multiplier to 1.5x, which reproduces the already-safe
1-anchor/220s behavior exactly at this pathological corner while still affording many more
anchors (23-244) at cells where a single anchor's cost is well below the budget -- a disclosed
limitation (very large m + very low corr_prop caps at 1 anchor), not a bug.

Removed the now-redundant external skip-gate in both sweep scripts and in
`calibrate_via_proxy_hyperopt` -- real hyperopt is now attempted at every scale; the fallback is
reserved for the genuinely rare "nothing selectable at all" case. Three new CSV columns
(`hyperopt_used`, `effective_n_series`, `series_subsampled`) make this transparent per cell.
Full detail, every verification performed (4-case exact-match re-check, a dedicated 4-type
synthetic diversity test, cap-sizing boundary check, two full end-to-end retests on the exact
incident cell -- one exposing the time blowup, one confirming the corrected multiplier's
arithmetic) in `docs/implementation_log.md`'s **2026-09-08 (c)/(d)/(e) entries**.

**DONE -- one-off validation exercise** on real data (`fr_air_temperature_121_1`, 121 stations),
forcing subsample caps of 30/60/90/121(full) via `max_feasible_pairs_per_anchor`: gamma came back
as exactly 0.45 at ALL FOUR cap levels -- direct empirical support for "gamma doesn't depend on
m". Caveat (checked, disclosed): recall was still climbing at the loosest gamma value actually
tested, so this is partly a grid-boundary effect, not proof of an interior optimum -- would need
`GAMMA_TRIAL_OFFSETS` widened past 0.25 to fully close this. Separately, a real, more concerning
finding surfaced: even at the FULL 121-series population with the whole occupancy/gamma grid,
recall topped out around 60% at corr_threshold=0.7 on this dataset -- nowhere near the 0.95
target. **User has explicitly deferred both the boundary-artifact follow-up and the 60%-ceiling
investigation to later** -- not blocking, not to be conflated with the current (sparser) sweep.
Full detail in `docs/implementation_log.md`'s **2026-09-08 (f) entry**.

## Next exact step
Both sweeps are relaunched (resumed) and progressing cleanly -- accuracy sweep at cell 48/85 as
of this update, timing-only queued to follow automatically once it finishes. **Immediate:**
monitor both to completion, report final `hyperopt_used`/`series_subsampled` splits (a real,
disclosable finding about where subsampling actually engages across each sweep's own range) plus
how many cells hit the 1.5x ceiling's 1-anchor floor and how many show `proxy_underpowered=True`
(observed so far: the large majority of low-corr_prop cells fall into one of these two buckets --
this is expected/disclosed, not a new problem, per the proxy-anchor method's own known
low-density limitation). Once both finish, rebuild the Sobol dashboard/story artifacts from the
new data (including `touched_candidates` and `speedup_no_val_no_monitor`), reporting recall
conditioned on calibration power rather than as one blended number, and compare against the
preserved old results.

Once the sweeps and dashboard are done, the two explicitly-deferred follow-ups are: (1) widen
`GAMMA_TRIAL_OFFSETS` past 0.25 and rerun the real-data cap=121 case to find gamma's actual
plateau; (2) investigate the ~60% real-data recall ceiling at corr_threshold=0.7 (try a smaller
threshold, larger `n_vectors`, or characterize whether the sketch/LSH approach has a fundamental
density regime limit).

Given each pathological-corner cell can still cost several minutes even at 1 anchor (220s
measured), and the sweep grid may contain more than one such cell, keep watching progress and
re-derive the time estimate from real observed per-cell costs rather than assuming uniform cost
across the grid -- the 28h deadline is real and this fix chain has already cost significant time
investigating.

No other blocking item remains from the OOM investigation — both real leaks (Cython slot reuse,
Python window_idx cache) diagnosed, fixed, and verified end-to-end; the residual was traced to
intended output accumulation, not a bug. Remaining, in priority order:
0. If the user later wants `self.correlated`/`self._maxlag_state` themselves bounded (e.g.
   streamed to disk incrementally for arbitrarily long production runs instead of held fully in
   memory), that's a deliberate output-semantics change needing explicit sign-off, not a leak fix
   — not started. `_get_or_create_window_idx_numeric` (bptree/blocked-index's tuple-partition
   path) does not yet get the free-list benefit — disclosed, not fixed, since it's unreachable
   for the LSH backend actually in use. Other memory follow-ups, disclosed but not done: revisit
   the sweep scripts' `_make_memory_limiter`/`RLIMIT_AS` worker memory limit now that real
   per-cell memory needs have shifted; consider whether `RECALL_SAFETY_MARGIN` should be
   scale-aware; `_post_members`'s own item-id storage was deliberately left at int64 (not audited
   for downsizing risk); `lsh_dead_node_ratio`/`hexact_dead_node_ratio` CSV columns changed
   meaning (no longer bounded to [0,1]) but are unused by any control flow, so left as-is.
1. Decide whether to also remove `proxy_candidate_rate_close_tolerance` and its exec_param/
   corrtrack_param_search chain now that it's fully dead (left in place this round, see Known
   issues) — a small, optional further cleanup, not blocking anything.
2. A handful of large-m cells at fixed (high) density — still useful to characterize the
   corrected formula's own m/L scaling (`m^0.50·L^0.51`, R²=0.97 at fixed occupancy) against real
   data at larger scale.
3. `candidate_search_lsh_candidates_touched`: still not directly logged, but the trial-grid's
   gamma=0 stage-1 `tested_candidates` remains a usable real proxy for every existing point.
4. Consider whether `experiment_lsh_nbands_occupancy_sweep.py`'s own
   `N_BANDS_TOLERANCE_GRID = (1.0, 1.5, 2.0)` should also be retired now that the underlying
   formula doesn't need a tolerance hedge — not touched so far (scoped to
   `experiment_run_param_grid.py`/`experiment_run_exec_param.py` specifically by the user).
5. If more dashboard changes are wanted instead, resume from the local working copies listed
   above and republish with the same artifact URLs (via the `url` parameter) to update in place.

## 2026-09-08 (g) — Presentation deadline: both dashboards/artifacts to be rebuilt from the new sweep, scope confirmed with user

User is presenting **tomorrow** and wants, once both sweeps finish:
1. Both artifacts updated in place (via `url`, not new artifacts):
   - **"CorrTrack at Scale"** (presentation story) -- `https://claude.ai/code/artifact/3cd502b5-f51c-4dc4-bf87-89005358b85c`
   - **"Sobol LSH Sweep"** (technical dashboard) -- `https://claude.ai/code/artifact/0a673d9b-9b05-47e4-90d7-07bdebfdb38d`
2. **Timing-only sweep results included in BOTH** -- not just the accuracy sweep. Real
   opportunity here: the story artifact's own "Where this goes" section currently extrapolates
   speedup all the way from m=296 (measured) to m=10,000 with an explicit "not measured, treat as
   a credible direction, not a promise" disclaimer -- the timing-only sweep reaches real measured
   m up to 3000, so the update should show a MEASURED curve out to 3000 and only extrapolate the
   shorter remaining stretch to 10,000, a materially stronger claim.
3. **Recall split by calibration power** in both -- well-powered vs. underpowered/fallback,
   not one blended median (per this session's own finding: many low-corr_prop cells fall into
   `hyperopt_used=False` or `proxy_underpowered=True`, and blending them into one number
   overstates what's actually guaranteed at low density).
4. New metrics (`touched_candidates`, `speedup_no_val_no_monitor`) added to both.
5. Comparison against the preserved old sweep results, and the two explicitly-deferred
   follow-ups (gamma-grid plateau, ~60% real-data recall ceiling) come after, not blocking.

**Real timeline risk, disclosed rather than assumed away**: since timing-only results are now
required for BOTH artifact updates, both updates are blocked on the timing-only sweep finishing
(61/64 cells remaining as of this entry, reaching m up to 3000 / L up to 128 -- durations at
that scale not yet observed, unlike the now-complete accuracy sweep's well-characterized
per-cell costs). Watching the first several large-m timing-only cells closely once the accuracy
sweep hands off, to give a real (not assumed) ETA against the user's "tomorrow" deadline.

**Next exact step**: monitor accuracy sweep to completion (3 cells left), then timing-only sweep
start and its first few large-m cells for real per-cell timing, then build the combined analysis
(regression refits including timing-only's extended m range, recall-by-calibration-power split,
new metric aggregates) before touching either artifact file.

## 2026-09-09 (l) — Deferred change: OPTIM_PROXY_ANCHOR_EXPAND_FACTOR default (2.0 -> ?)

User's supervisor suggested the default anchor-expansion factor (2.0, i.e. doubling anchor count
each round when calibration is underpowered -- `corrtrack_param_search.py:97-100`, passed into
`proxy_config["anchor_expand_factor"]` by `calibrate_via_proxy_hyperopt`,
`experiment_lsh_cost_sweep.py:604`) may be too aggressive/unnecessary. User explicitly wants this
deferred: "shouldn't affect the ongoing experiments in the queue."

**Confirmed why this matters mechanically, not just as a precaution**: each sweep cell runs as a
freshly-launched subprocess (`--worker-cell-spec`), re-importing everything from scratch -- so
editing this default WHILE either sweep (timing-only, or the cell-86/87 brute-force comparison)
is still running would take effect on the very next cell processed, silently splitting the
dataset between old- and new-factor calibrations mid-run. Not touched; will only be changed once
both are done.

**For whenever this is revisited**: 2.0 (doubling) can overshoot past "just enough" anchors,
wasting reference-construction cost per expansion round; a smaller factor (e.g. 1.5) grows more
gradually but needs more rounds when a large jump really is warranted. Worth testing empirically
against real calibration outcomes (e.g. how often expansion triggers, and by how much power
improves per round) rather than picking a new default by intuition alone.

**Next exact step once sweeps finish**: revisit this constant, run a small A/B comparison (2.0
vs. a candidate lower value, e.g. 1.5) on a handful of already-known underpowered cells from this
sweep, and only change the default if it shows a real, measured improvement -- not a blind swap.

## 2026-09-09 (m) — Second pinned topic: can target_occupancy be made automatic like n_bands?

User asked whether `target_occupancy` (currently grid-searched, `OCCUPANCY_GRID = (2.0, 3.0,
5.0, 10.0)`) could be derived automatically the same way `n_bands` is. Also explicitly deferred
-- same reason as the anchor-expand-factor above (fresh-subprocess-per-cell architecture means
any change would leak into the currently-running sweeps/comparison mid-run).

**Key distinction worth remembering when this is picked up**: `n_bands` has a real closed form
because it only has to satisfy a probability equation (hit target_recall given band_width) --
pure math, no measurement needed. `target_occupancy` trades off wall-clock SPEED (fewer bands vs.
more touched candidates per query), which depends on real per-candidate costs (hardware, memory
layout) that no recall-only formula captures -- there is no equivalent closed form, and framing
any occupancy default as "derived" the way n_bands is would misrepresent what's actually
happening. Any automatic-occupancy scheme is necessarily a heuristic, not an exact derivation --
must be labeled as such, not blurred with n_bands' own guarantee (matches CLAUDE.md's own rule
distinguishing exact vs. heuristic search).

**Real evidence already in hand from this sweep**: occupancy=3 measured a 14.6x median speedup
vs. 8.0x for occupancy=5-10 across the 85 accuracy-sweep points -- a real, usable pattern, but
from one sweep's data, not a proven law.

**Two candidate approaches, not yet built**:
1. Shrink `OCCUPANCY_GRID` (e.g. to `(3, 5)`) informed by this finding -- cheaper hyperopt,
   still a real (smaller) search, not a removal.
2. A genuine heuristic formula, e.g. occupancy as a function of corr_threshold (since the
   occupancy cost exponent `1-gamma` grows as threshold rises, occupancy should matter more
   there) -- needs its own validation pass before trusting it as a default.

**Next exact step once sweeps finish**: decide between the two approaches above (or both,
sequenced), and validate whichever is chosen against real cells (not assumed) before changing
any default -- same discipline as the anchor-expand-factor item.

## 2026-09-10 — All sweeps + brute-force comparison complete; state and remaining work

**Compute status: DONE.** No background jobs running. Full detail in
`docs/implementation_log.md`'s 2026-09-08 through 2026-09-10 entries.

**Final datasets:**
- Accuracy sweep: 85 valid points (cells 1-12 were re-run under the current codebase to replace
  stale pre-rewrite rows), + 2 anchor cells 86/87 (m=1021, m=1927) with real brute-force
  comparison, per supervisor request. `tmp_artifacts/lsh_sobol_sweep/sobol_sweep.csv`.
- Timing-only sweep: 42 succeeded, 6 failed at the 5GB cap (cells 4, 28, 29, 36, 44, 52 -- all
  large-m + low/moderate-threshold, memory blows up via large n_bands or touched-candidate
  volume; disclosed limitation). Real max m measured: 2,470.
  `tmp_artifacts/lsh_sobol_sweep_timing_only/sobol_timing_sweep.csv` (also holds 12 redundant
  m<300 rows from before the M_RANGE-overlap bug was caught -- harmless, ignore for the
  timing-only sweep's own unique-contribution analysis).

**Presentation artifact ("CorrTrack at Scale", `3cd502b5-...`): current and published**, folded
in all supervisor feedback across many rounds (restructure, boxplots, sensitivity charts, DOE
explainer, complexity split, limitations diagram, the two real brute-force overlay points, final
127-point combined-curve refit R²=0.78).

**Remaining, none blocking, in rough priority order:**
1. Rebuild the technical dashboard ("Sobol LSH Sweep", `0a673d9b-...`) -- needs real
   re-architecture (its calibration-analysis tabs assume the OLD tolerance x occupancy grid
   search; current method is proxy-anchor occupancy x gamma). Rebuild its embedded data from the
   final CSVs; compare against the preserved old sweep artifacts. **NOT STARTED.**
2. ~~The two pinned parameter investigations~~ -- **DONE, see 2026-09-10 (q) then CORRECTED by
   (t)**. Anchor-expand-factor is moot (never triggered in the sweep). Occupancy: the (q)/(s)
   conclusion ("higher occupancy faster, default to 10") was WRONG -- two bugs (n_vectors=32 not
   64; synthetic effective density ~1e-5). The corrected real-data experiment (entry (t)) shows
   the optimum is LOW occupancy (2 on sparse data, ~8 mid), and `touched/bf_tested` climbs to
   0.83-0.93 as occupancy rises. **Recommendation: do NOT change `OCCUPANCY_GRID` or the default
   (3.0). Keep it low.** Nothing to apply.
3. ~~Fix `M_RANGE`~~ -- **DONE**, 50 -> 300 in `experiment_lsh_sobol_sweep_timing_only.py:87`.
4. ~~Long-deferred: gamma-grid plateau check; the ~60% real-data recall ceiling~~ -- **DONE, see
   2026-09-10 (r).** Gamma grid needs no change (plateau is inside the searched range). "Ceiling"
   was not real -- it was the proxy calibration's recall ESTIMATE running low on dense data.
5. ~~Apply the occupancy-grid change~~ -- **DROPPED, see (t).** No change: keep occupancy low.
6. Gamma / fallback: `THEORETICAL_FALLBACK_GAMMA_OFFSET = 0.20` **DONE** (entry (r)). Gamma
   search space confirmed offset-based and locked with an assert (entry (u)). Still open: harden
   the proxy recall estimate for the dense regime (real algorithmic change, deferred); decide
   whether to rename `GAMMA_TRIAL_OFFSETS` -> `GAMMA_OFFSETS` (6-file rename, user's call).

**Only genuinely large remaining item: the technical dashboard rebuild (parked by user).**

**Changed files this session (all uncommitted, no commit requested):** `candidate_kernels.pyx`
(+`.so` rebuilds), `library_corrtrack_parallel.py`, `experiment_lsh_cost_sweep.py`
(`THEORETICAL_FALLBACK_GAMMA_OFFSET` + gamma-offset docstring/assert, entries (r)/(u)),
`experiment_lsh_sobol_sweep.py`, `experiment_lsh_sobol_sweep_timing_only.py`, plus the two
`tmp_artifacts/*/sobol_*.csv` and `sobol_design_points.json` design files (cells 86/87 appended).

**Occupancy correction (2026-09-10 (t)):** real-data experiment
`/tmp/.../scratchpad/occupancy_ceiling_realdata.py` +
`occupancy_ceiling_realdata_results.json`. Prior synthetic occupancy experiments
(`occupancy_experiment*.py`, `occupancy_density_experiment.py`) are compromised (wrong
n_vectors) -- superseded, keep only for the record.

**Numeric correlated-pair representation (2026-09-10 (bb)):** approved plan
`~/.claude/plans/misty-stargazing-liskov.md`. Retired the per-pair string-tuple key
(`_normalize_bf_key`) from the hot validate/monitor recording path -- `self.correlated` is now a
lazy `@property` over a persistent numeric `int64 (N,5)` accumulator (`_correlated_rows/_corrs`,
`_append_correlated_numeric`), canonicalized vectorized (`_canonicalize_rows`). `compute_metrics_bf`
gained a pure-numpy fast path (`compute_metrics_bf_numeric`) on `(rows, corrs)` / `CorrTrack` /
`NumericCorrelatedFlags` inputs. Verified byte-identical (key-set + all metrics, real dense data,
multiple thresholds). **m=121 4% density: CorrTrack wall 16.9s -> 6.8s (2.5x), speedup 0.75x ->
1.47x. `val.numeric_rows_postprocess` 7.28s -> 0.20s.** Suite 119/119. Changed:
`library_corrtrack_parallel.py` (big), 5 sweep scripts' `compute_metrics_bf` call sites,
`test_stable_reproduced_changes.py` (+3 tests). Steps 1-9 done; step 10 (Cython) not needed.

**Next exact step:** the technical dashboard rebuild (item 1) is the only large remaining item;
still parked. When built it must reflect entry (t) (occupancy optimum is low, not high), the
proxy-calibration dense-regime under-report (entry (r)), AND the numeric-representation phase-time
change (validation/monitor times in the OLD sweep CSVs were inflated by string-key bookkeeping;
a re-run would show cleaner phase splits and better dense-regime speedup).

## 2026-09-10 (cc) -- Density-deterministic synthetic generator: DONE and verified

**Branch:** main. Uncommitted (no commit requested).

**What was done:** implemented `synth_corr_gen.make_density_targeted_dataset` per the approved
plan (`~/.claude/plans/misty-stargazing-liskov.md`). It takes a target brute-force effective
tuple density (`bf_correlated / bf_tested`) plus the eval config and produces a dataset that
hits it deterministically, together with the exact ground-truth `(pair, lag, window, signed_r)`
set by construction (so a full per-cell BF is no longer needed). Correlated blocks driven by a
white shared signal, per-member loading -> strength distribution over `[thr, r_max]`, scheduled
OFF<->ON and `+<->-` transitions on a step-aligned epoch grid with an event log. Full
write-up + the 7 bugs found and fixed: `docs/implementation_log.md` entry **(cc)**.

**Changed files:** `synth_corr_gen.py`, `test_synth_density.py` (new).

**Commands run:**
- `python3 -m pytest test_synth_density.py test_stable_reproduced_changes.py -q` -> **128 passed**
- scratch grid (14 cells): 14/14 density within tolerance, `gt_precision == 1.0`,
  `gt_recall` 0.97..1.00, determinism byte-identical.

**Results:** density on target (`duty=1.0`: within ~8%; `duty<1`: within ~20% on
`analytic_density`, closer on `verified_density`). `gt_precision == 1.000` everywhere.
`m=600 @ 3e-2` reached (old generator capped ~1e-5).

**Known issues / limitations:**
- `lag_band > 1` is a lag-SPREAD knob (different pairs at different single lags), not a
  per-pair band -- a genuine band is geometrically incompatible with windowed decorrelation at
  `w ~ 8*step` (verified empirically, see (cc)).
- `duty<1` density accuracy ~0.20 rel (sharp-transition straddle windows); `analytic_density`
  and `verified_density` bracket the target.
- `_analytic_gt` allocates the full GT array (10M rows at m=600); fine cached, would need
  chunking for a very large fresh sweep.

**Next exact step:** wire the generator into the sweeps (all NOT started):
1. `datasets/synth_loader.py` -- `_stem_from_params` / `_REQUIRED_KEYS` new keys
   (`mode="density_targeted"`, `target_density`, `base_proc` type, `preprocess`, `lag_band`,
   `corr_sign`, `n_epochs`, `duty`), `load_dataset` mode dispatch to
   `make_density_targeted_dataset`.
2. `experiment_lsh_cost_sweep.py` -- `load_density_targeted_cell(...)` returning
   `(data, ids, gt_flags, analytic_density, verified_density)` with `gt_flags` a
   `NumericCorrelatedFlags`, and `synthetic_ground_truth(...)` shaped like
   `run_bruteforce_once`'s output (`record["tested"]` = analytic tuple count,
   `record["correlated"]` = `len(gt_flags)`, `record["runtime"]` = None). `--verify-bf-fraction`
   knob to still run the real full BF on a random subset.
3. `experiment_lsh_sobol_sweep.py` / `experiment_lsh_orthogonal_array.py` -- swap the
   `corr_prop` factor for `target_density` (`_generate_sobol_points` new `DENSITY_RANGE`
   log-uniform `(3e-4, 5e-2)`; OA `DENSITY_LEVELS = (5e-4, 3e-3, 1e-2, 3e-2)`), rename CSV
   columns, add `base_proc`/`preprocess`/`corr_sign`/`duty` columns, make the full-length BF
   phase conditional on `--verify-bf`.
4. Add a sweep smoke test.

**Still parked (unchanged):** technical dashboard rebuild (must reflect entries (t), (r), (bb)).

## 2026-09-10 (dd) -- staged accumulated work onto existing remote `dev`

**Branch:** `dev`, re-pointed to `origin/dev` (`7a486c3`, 2026-07-07). The remote `dev` branch
already existed -- this working copy was just a stale checkout at `83f43c6` that never pulled.
Full context in `docs/implementation_log.md` entry **(dd)**.

**Done:** staged the source work (all root `*.py`/`*.pyx`/`*.c`/`*.so` modified+new, `README.md`,
`datasets/synth_loader.py`, rewritten `.gitignore`, deletion of stray `header.png` +
`library_corrtrack_parallel copy.py`) for one commit on top of `origin/dev`. `docs/` and
`tasks/` kept PRIVATE (excluded + gitignored) per user instruction. Commit message: NO
`Co-Authored-By` trailer per user instruction.

**Not staged:** `__pycache__/*.pyc`, `build/`, `datasets/synth_outputs/*`, `*:Zone.Identifier`,
`xp/xp1.txt`/`xp2.txt`, `datasets/examples/synthetic/...correlated.csv` -- remote cruft / not
requested; follow-up `git rm --cached` cleanup available.

**Not done:** commit + push -- handed to the user as a ready command.

**Next exact step:** user runs the `git -c user.name=... commit -m "..."` + `git push origin dev`
command from the session response. After that: decide whether `dev` gets PR'd to `main`.

## 2026-09-11 -- Colleague's `feat/v2-engine` branch analyzed; FilCorr port scoped and ready

**Current branch: `dev`** (this checkout was on `dev` already at session start, per the prior
entry's `git push`; `docs/`/`tasks/` remain private/uncommitted as before).

User asked to analyze `origin/feat/v2-engine` (Benoit Lange's colleague branch) and plan reuse
of two contributions: a parallel-execution optimization, and a FilCorr competitor baseline.
Full analysis and reasoning in `docs/implementation_log.md`'s **2026-09-11 entry** -- summary:

- **Parallel optimization: not worth porting as-is.** `dev` already has a more advanced,
  differently-architected parallel story (`nogil` `prange` Cython kernels + thread/dask
  orchestration). v2's `ProcessPoolExecutor` wrapper was built for a much simpler non-LSH
  pipeline and would need unsafe redesign to fit `dev`'s stateful free-list LSH index.
- **FilCorr competitor: ready to port, high value, minimal diff.** `dev` already has the exact
  integration slot needed -- the existing `exact_stomp` baseline (`Candidates_BF_ExactSTOMP`,
  `baseline_mode`, `run_bf_exact_stomp` in `library_corrtrack_parallel.py`) is the precedent to
  mirror. Plan: port only `v2/fillcorr/algo.py`'s pure-numpy math (Parseval band-pass
  correlation), wrap it in a new `Candidates_BF_FilCorr` class with the same
  `.run() -> (accepted_rows, accepted_corrs, n_pairs, timing)` contract, add
  `baseline_mode="filcorr"` + a `run_bf_filcorr` dispatch, thread `filcorr_fs`/`filcorr_ft`/
  `filcorr_sampling_rate` through `base_config`. `corrtrack_compare_runs.py` needs no changes.
  **Caveat that must be respected during implementation:** only full-band FilCorr
  (`fs=0.0, ft=0.5`) is mathematically identical to the existing Pearson ground truth --
  that is the setting to use for the SOTA comparison; verify by reproducing bruteforce's exact
  pair set before trusting the port.

**Update -- both follow-ups done this same session (user approved both):** see
`docs/implementation_log.md`'s **2026-09-11 (b) entry** for full detail.

- **GIL investigation: real measurement done.** `parallel_validation`'s thread path calls into a
  compiled Cython kernel that releases the GIL (`with nogil:`) -- a direct micro-benchmark
  confirmed genuine multi-core speedup (1.83x/2.99x at 2/4 threads on this 8-vCPU WSL2 box, with
  diminishing returns/regression at 8 threads). Confirms: do not port v2's `ProcessPoolExecutor`
  wrapper.
- **FilCorr: implemented, verified, DONE.** `Candidates_BF_FilCorr` (in
  `library_corrtrack_parallel.py`) + `baseline_mode="filcorr"` + `run_bf_filcorr` + CLI flags on
  `corrtrack_run_bruteforce.py` (`--filcorr-fs/--filcorr-ft/--filcorr-sampling-rate`). **Two real
  bugs found and fixed via the verification test itself** (not assumed correct after "it
  compiled"): a missing-Nyquist-bin weighting bug (even window sizes) and an off-by-one at the
  odd-window upper band edge -- both only affect the full-band case, both now fixed and verified
  to floating-point precision (<1e-9) against `Candidates_BF_ExactSTOMP`'s raw Pearson, at window
  sizes 32/33/168/169. Also verified end-to-end via the real CLI (`corrtrack_run_bruteforce.py
  --baseline-mode filcorr` vs `bruteforce`): 411901 correlated pairs, identical, on a real run.

**Changed files this session (all uncommitted, no commit requested):**
`library_corrtrack_parallel.py`, `corrtrack_run_bruteforce.py`, `experiment_run_exec_param.py`,
`test_stable_reproduced_changes.py` (+2 new tests).

**Commands run:** `python3 -m pytest test_stable_reproduced_changes.py -q` -> 121 passed (119
pre-existing + 2 new, no regressions); `python3 -m pytest test_synth_density.py -q` -> 9 passed;
real CLI smoke run comparing `--baseline-mode bruteforce` vs `filcorr`.

**Known issues / not done (see implementation_log.md (b) entry for full list):**
- `filcorr_fs/ft/sampling_rate` not added as CSV columns (deliberate, minimal-diff, matches how
  `exact_stomp` itself adds none).
- No speed/timing benchmark of FilCorr vs bruteforce run yet -- only CORRECTNESS was verified
  this session. FilCorr's actual wall-clock speed advantage at real project scale (m=121+,
  window_size=168) is unmeasured.
- `CorrTrackMultiWindow.run_bf`'s propagation of `baseline_mode`/`filcorr_*` to its per-size
  tracker instances was not verified (pre-existing question, applies equally to `exact_stomp`).

**Update -- speed benchmark run (user approved), real and disclosed finding:** see
`docs/implementation_log.md`'s **2026-09-11 (c) entry**. Real dataset
(`fr_air_temperature_121_1`, 121 stations, 1yr), real project defaults (window_size=168,
window_step=12, n_lags=168, corr_threshold=0.7), sequential (not concurrent -- a first
concurrent-launch attempt was caught as contaminated by CPU contention and redone cleanly).

- **Correctness confirmed again at real scale:** identical correlated count (14,043,815) for
  bruteforce vs filcorr.
- **Speed: FilCorr is SLOWER here, not faster** -- bruteforce runtime 30.496s vs filcorr
  32.274s (~6% slower overall); validation phase specifically 11.419s vs 13.986s (~22% slower).
  Most likely cause: bruteforce already runs through the same compiled `nogil` Cython kernel the
  main pipeline uses, while `Candidates_BF_FilCorr` recomputes a fresh (uncompiled) numpy FFT per
  window per step with no incremental caching -- at window_size=168 the band-width reduction
  (~2x) doesn't cover that cost. Two concrete, NOT pursued follow-ups if the user wants FilCorr
  to actually win on speed: a Cython kernel for the Parseval correlation (colleague's
  `v2/fillcorr/_kernels.pyx`, not ported), or the `O(1)`-per-slide incremental band update
  (`algo.py`'s `incremental_update`, also not ported).

**Update -- user corrected the framing ("I do not want filcorr to win, but I want its results to
be comparable") and I made the implementation genuinely comparable, then re-measured:** see
`docs/implementation_log.md`'s **2026-09-11 (d) entry**. Two real, independently-justified
fixes: (1) real-decomposition instead of a wasted complex matmul (puts FilCorr on the same
BLAS-matmul tier `exact_stomp` itself uses, no `prange`/multi-core edge `exact_stomp` doesn't
get); (2) persistent cross-step FFT memoization (bounded to the `n_lags` window, found necessary
by measurement -- not assumed). Verified correct both times (121/121 tests). Profiled directly
(not guessed) to find the real remaining bottleneck: the per-lag MATMUL, not FFT recomputation.
**Principled, not-an-implementation-gap reason FilCorr can't match `exact_stomp`'s speed at full
band even when fair**: `exact_stomp`'s speed comes from incrementally updating a raw dot
product (simple additive sum over samples); FilCorr's Parseval correlation is a dot product of
BAND-FILTERED (FFT-mixed) representations, which has no valid closed-form incremental-update
analog -- forcing one would mean changing what the method computes.

**Final honest result**: fairness fixes closed part of the gap (FilCorr was ~22-25% slower than
bruteforce's val_time, now ~15-18% slower) without eliminating it -- exactly the expected outcome
at full band (no asymptotic FilCorr advantage there by design; its real O(B) edge only exists for
a genuinely narrower, non-comparable band). Correctness unaffected: identical correlated count
(14,043,815) throughout.

**Task status: DONE.** FilCorr is now verified correct AND on a fair, comparable implementation
footing with the baselines it's scored against. No commit was made or requested.

## 2026-09-11 (e) -- Density generator burst-mode + real-data scaling/recall investigation

**Branch:** dev (uncommitted; other unrelated work, e.g. the FilCorr entries above, has also
landed in this same working copy from elsewhere -- see git log, HEAD is behind at `da0f4f7`,
this file/`docs/implementation_log.md` accumulate across sessions independent of commits).

**What was done:** full write-up in `docs/implementation_log.md`'s **2026-09-11 (e) entry**.
Summary: added `burst_length_windows` to `synth_corr_gen.make_density_targeted_dataset`
(transient/short-lived correlations, not just persistent whole-epoch ones), found and fixed
2 real bugs in it (a `searchsorted` off-by-one, a burst-alignment-to-the-step-grid bug).
Investigated real French weather data: raw-space density is flat with m (bad/dense regime by
m=121), diff-space density genuinely decays with m out to ~65 (good/sparse regime). Found real
recall never reaches 0.95 target on real diff-space data (caps ~0.80) and traced it to
persistence (duration a correlation lasts), not strength -- **then built a controlled
synthetic test that contradicts that conclusion** (recall stayed ~1.0 across the whole
persistence range when strength is held high and independent of duration), so the real
driver is still open (likely strength-duration interaction, not persistence alone).

**Two controlled synthetic experiments run** (`tmp_artifacts/degree_vs_m_v2/`,
`tmp_artifacts/persistence_vs_recall_v2/`, both resumable/checkpointed to disk since the
machine restarted mid-session three times):
- **degree-vs-m: CONFIRMED.** Fixed low degree (1) -> speedup climbs cleanly with m (1.17 ->
  3.02, m=50->400). A "fixed high degree" arm was accidentally NOT fixed (hit the generator's
  own density ceiling at small m) but still showed the same pattern via a different path
  (degree ramped 3.2->20.3, speedup fell 1.38->1.20) -- second independent confirmation.
- **persistence-vs-recall: INCONCLUSIVE / open.** Real mechanism not yet found.

**Commands run:** `python3 -m pytest test_synth_density.py test_stable_reproduced_changes.py -q`
-> 130/130 passed.

**Known issues:** the density generator's ceiling (~6.7% at L_test=8) makes "fixed high
degree across a wide m range" hard to set up directly -- watch for this when designing degree
sweeps (verify achieved degree, don't trust the requested target at small m).

**Next exact step:** rerun the persistence sweep with strength ALSO varied (near-threshold /
`loading_skew="low"`) to isolate whether it's duration, strength, or their interaction driving
the real recall gap. Then decide whether to pursue task 2 (is the recall gap fixable in the
monitor/candidate-discovery code, or a property to disclose).

## 2026-09-12 -- Real four-way comparison (bruteforce/exact_stomp/filcorr/CorrTrack) run on
Abaca; a real memory bug fixed along the way; conclusion: dense real data disfavors CorrTrack,
sparse data (its actual regime) gives it a clear win

**See `docs/implementation_log.md`'s 2026-09-12 entry for full numbers and methodology.**
Direct follow-on from the (d) entry's FilCorr comparability work -- user asked for a real
timing/recall comparison of all four methods, which the local WSL machine could not run
safely (two real whole-machine OOM crashes, not just Python OOMs -- see below), so this ran
entirely on Abaca (`sophia.g5k`, OAR batch jobs, 192GB/node, no memory constraint there).

**Real memory bug found and fixed before any comparison numbers exist:**
`CorrTrack._append_correlated_numeric`'s int64 doubling-growth accumulator needed a 640MB+
single allocation for 14M+ correlated pairs (the real dense dataset). Fixed: internal storage
downsized to int32 (safe -- every external consumer already re-casts to int64 itself,
verified), plus two redundant defensive int64 upcasts removed elsewhere (`correlated_rows()`,
`NumericCorrelatedFlags.__init__`) that were each independently found -- via a SECOND and
THIRD real OOM -- to reintroduce the same peak one level up the call stack. Committed by the
user as `42b5b41` on `dev`, pushed, pulled+rebuilt on Abaca. 131 tests passing throughout.

**Four real comparisons run on Abaca** (all in `docs/implementation_log.md`'s entry, numbers
not repeated here):
1. Dense real data (`fr_air_temperature_121_1`), untuned CorrTrack: 1.03x speedup, recall 0.90.
2. Same data, hyperopt-tuned: recall improved to 0.98 but speed got WORSE (0.73x) -- a real,
   disclosed recall/speed tradeoff, not a bug.
3. `n_vectors` sweep (32/64/96/128) on the same data: 64 is the practical sweet spot
   (~1.0x speedup, recall~0.97) -- hyperopt's own pick (32) was NOT the sweet spot a plain
   manual override sweep found (a real, disclosed, not-yet-investigated gap).
4. SPARSE synthetic data (`synth_corr_gen.make_density_targeted_dataset`, target_density=2%,
   same m=121 series, n_vectors=64): **CorrTrack wins outright, 3.66x speedup**, beating even
   `exact_stomp` (3.33x) and `filcorr` (2.99x) -- the regime CorrTrack is actually built for.

**Task status: DONE for what was asked.** Coherent, honest conclusion carried forward: dense
real data (this project's own fr_air_temperature at corr_threshold=0.7) is architecturally
unfavorable for CorrTrack; realistic sparse data is a clear win. No commit was made for the
`abaca/fourway_compare.py`/`abaca/sparse_fourway_compare.py`/`abaca/*.oar` scripts themselves
(exist only on Abaca + this session's local scratchpad) -- flagged as worth committing if this
comparison needs to be reproducible/repeatable later.

## 2026-09-13 -- Three real-domain datasets, exact_stomp/filcorr preprocess bug fix, four-way
comparison across all three, hybrid_validation profiled/fixed, WSL crash investigated, moved to
Abaca

**Full write-up in `docs/implementation_log.md`'s 2026-09-13 entry.** Summary only here.

**Branch:** main, uncommitted (same working copy; `dev`'s `42b5b41` still HEAD).

**Real changes this session:**
- `library_corrtrack_parallel.py`: fixed `Candidates_BF_ExactSTOMP`/`Candidates_BF_FilCorr`
  silently ignoring `preprocess=True` (a real correctness bug, found by evidence + code read,
  had invalidated an earlier round of comparison numbers before they were reported). Verified
  byte-identical against bruteforce afterward.
- `candidate_kernels.pyx` + `library_corrtrack_parallel.py`: fixed `hybrid_validation=True`
  being SLOWER than `False` on all 3 real datasets despite 91.6-99.7% cache hit rates. Root
  cause (via cProfile): double Python-object construction (`HybridValidationCache.validate_pairs`
  built a list-of-tuples in Cython, `_get_validated_corr_numeric` re-unpacked it per-item in
  Python) -- same anti-pattern as the 2026-09-10 numeric-representation fix. Fixed via bulk
  numpy arrays across the Cython/Python boundary. `test_hybrid_validation_current_window_cache_
  matches_bruteforce` updated to the new return format (was asserting on a removed dict key,
  not skipped). **131/131 tests pass, locally and on Abaca.**

**Three curated real datasets** (finance/environmental/"other domain", per the user's own
framing): `sp500.npz` (492 tickers x 1255 days), `ca_streamflow.npz` (538 USGS gauges x 2192
days), `wikipedia.npz` (88 articles x 1827 days). Crypto fetched, not analyzed.

**Four-way comparison (bruteforce/exact_stomp/filcorr/CorrTrack), diff-space, hybrid_
validation=False, all on Abaca:** CorrTrack loses to `exact_stomp` on wikipedia (0.90x) and
streamflow (0.75x), wins clearly on sp500 (1.49x) -- vs `exact_stomp`, all vs bruteforce:
wikipedia 1.02x, sp500 2.31x, streamflow 1.49x. Recall 0.973-0.983, precision 1.0000 throughout.
Per-chunk instrumentation directly answered the user's "is speedup constant when scaling m"
question: no -- streamflow's own chunk-level speedup vs `exact_stomp` ranged 0.62x-0.91x within
a single run, since exact methods have fixed per-step cost but CorrTrack's tracks local density.

**hybrid_validation, before/after the fix, speedup vs bruteforce:**

| dataset | hybrid=False | hybrid=True (buggy) | hybrid=True (fixed) |
|---|---|---|---|
| wikipedia | 1.02x | 0.90x | 0.96x |
| sp500 | 2.31x | 1.85x | *(job running)* |
| streamflow | 1.49x | 1.18x | *(job running)* |

Recall/precision confirmed bit-identical hybrid=True vs False before trusting any of these
numbers (fix only changed performance). sp500/streamflow fixed numbers pending as of this
writing -- OAR jobs 3102700/3102701 in flight.

**WSL crashed twice mid-session**; user asked directly whether Claude's own work caused it --
investigated honestly (`free -h`, `.wslconfig` 8GB cap, baseline process memory, the specific
run that plausibly triggered it) rather than guessing. Result: moved all heavy computation to
Abaca (OAR jobs, `sophia.g5k`) per the user's own instruction, no memory ceiling there.

**Density-dependence brainstorm** (user's explicit ask, same message as the hybrid_validation
instruction): 5 options proposed in the doc entry, not yet implemented. Most actionable:
adaptive backend dispatch (fall back to exact search when the LSH's own touched-rate signals
it isn't filtering, capping the downside near parity with the best exact baseline).

**Known open items:** streamflow's hybrid_validation overhead only partially closed at ~12M
attempts (56.7s->66.0s locally, still 1.16x slower) -- worth a follow-up profile at that scale;
France+Brazil degree-rises-with-more-stations finding still unexplained; crypto dataset
unanalyzed; Abaca-only scripts (`fourway.py`, `fourway_hybrid.py`, job wrappers) not committed.

**Commands run:** see `docs/implementation_log.md` entry (pytest invocations, scp targets, OAR
job IDs 3102693/3102698/3102699-3102701).

**Next exact step:** collect the sp500/streamflow corrected hybrid=True numbers (jobs
3102700/3102701), decide whether `hybrid_validation=True` should become the default given the
residual streamflow gap, then either continue closing that gap or move to implementing the
adaptive-backend-dispatch brainstorm option if the user wants to keep reducing
density-dependence.

## 2026-09-13 (b) -- Options 4 (prange parallelism) and 5 (candidate-refresh cadence
decoupled from window_step) implemented, verified, real measured win

**Full write-up in `docs/implementation_log.md`'s 2026-09-13 (b) entry.** Summary only here.

**Corrected hybrid=True four-way numbers** (same-job, post-fix): wikipedia 0.96x, sp500 2.08x,
streamflow 1.39x (all vs bruteforce) -- still slightly slower than `hybrid_validation=False` in
every case even after the fix (0.96x/2.08x/1.39x vs False's 1.02x/2.31x/1.49x). **Recommend
keeping `hybrid_validation=False` as the default** until this closes further.

**Option 4 (prange, new `candidate_search_n_threads` param, default 1=unchanged):**
`SignLSHBandIndex._find_pair_rows_meta_parallel` (new, `candidate_kernels.pyx`) -- each thread
gets its own touched/visited-stamp scratch (sized to full capacity) and output buffer, merged
+ deduped after. Two real bugs found and fixed during bring-up (both disclosed in the doc
entry, not shipped silently): a Cython "reduction variable" compile error requiring the
per-candidate accumulations to move into standalone nogil helper functions; and a genuine
correctness bug (passed the LSH-internal entry id straight into the output instead of
translating via `window_idx[cand]` first) that was actually caught in testing --
recall/precision dropped from 0.9727/1.0000 to 0.8278/0.9869 with the bug present, exactly
0 (bit-identical) after the fix. **Verified**: row-set byte-identical vs sequential at
n_threads in {1,2,4,8}; recall/precision bit-identical end to end on all 3 real datasets.
**Real speedup**: sp500 1.85x (n=4), streamflow **2.24x (n=4)** -- the densest/largest dataset
gets the cleanest win, exactly the regime this investigation targets. wikipedia (m=88) actually
regresses at n_threads=8 (too little work per thread) -- n_threads needs to scale with m, not
be maxed blindly. 131/131 tests pass locally and on Abaca (job 3103191) after resync.

**Option 5 (candidate_refresh_interval, default 1=unchanged):** verified via code read + a
direct empirical check that `basic_window` is currently ONLY exploited for incremental sketch
computation, never for candidate-search cadence -- confirming the user's own hypothesis. New
param decouples them: real LSH search only every `interval`-th step, replayed from cached
`(sid1,sid2,delta,w)` templates otherwise. One real bug found+fixed (a mask wrongly excluded
all zero-lag cross-series pairs, the common case for financial data -- caught via a debug
script showing replayed rows were 5-8% of cached templates instead of ~100%). Precision stays
exactly 1.0000 by construction; recall degrades gracefully with interval (sp500: 0.9727 -> 0.9678
(2) -> 0.9592 (3) -> 0.9290 (5)) -- **a real recall-for-speed trade, not a free win**, unlike
option 4. Real speedup up to 1.36x (interval=5) before diminishing returns.

**Combining both**: composes correctly (independent effects on recall) but not multiplicatively
-- understood, not just observed: `candidate_search_n_threads` only speeds up real searches,
and `candidate_refresh_interval` shrinks how much real-search time there is left to speed up.

**Task status: both options implemented, verified, real measured win reported.** No commit
made; both files synced to Abaca (md5-matched) and rebuilt there (131/131 pass).

**Known open items:** neither option's cached-template/parallel path is wired beyond the LSH
numeric-rows backend (disclosed scope limit); not yet combined with `hybrid_validation`; no
auto-tuning of `n_threads` by dataset size; the three-domain four-way comparison has NOT been
rerun on Abaca with either option enabled yet.

**Next exact step:** if continuing, rerun the three-domain four-way comparison on Abaca with
`candidate_search_n_threads=4` to see how much of the exact_stomp gap on wikipedia/streamflow
this closes in the full four-way context.

## 2026-09-14 -- Option 4 gated behind parallel_candidates; "option 6" (symmetric dot-gate
score cache) implemented, verified exact, empirically a net loss -- root cause identified

**Full write-up in `docs/implementation_log.md`'s 2026-09-14 entries.** Summary only here.

**Gating fix (user feedback):** `candidate_search_n_threads` now silently clamps to 1 whenever
`self.parallel_candidates` (this project's existing parallel-execution toggle, defaults False
regardless of `exec=`) is falsy -- verified directly: `exec="sequential"` +
`candidate_search_n_threads=4` alone now resolves to 1 (no OpenMP threads spun up); only
`parallel_candidates=True` explicitly set lets it take effect, and the real 1.85x speedup
still shows up when set correctly. Also hit and fixed a real Abaca infra gotcha along the way:
rerunning tests there WITHOUT rebuilding first crashed with `Illegal instruction` (the `.so`
is `-march=native`, node-specific -- a job landed on a different node than the one that built
it). Lesson applied: always rebuild inside the same OAR job that runs tests on Abaca.

**New feature ("option 6"), from the user's own proposal in chat + its symmetric extension to
persistently-far pairs**: a persistent, EXACT dot-gate score cache (new `candidate_score_cache`
param, default False) keyed by `(query_series, cand_series, lag)`, using a Cauchy-Schwarz bound
on each series' own per-step sketch movement to decide when a cached score's gate verdict
provably cannot have flipped -- skipping the O(n_vectors) dot product for both persistently-
close AND persistently-far pairs, without ever skipping the bucket scan itself (so, unlike
`candidate_refresh_interval`, this carries no recall risk by construction).

**Verified EXACT**: `ct.correlated` identical (625,539 windows, same set) on/off, on real
sp500 data -- recall/precision bit-identical. The bound math and cache design are correct.

**But empirically useless as built**: 0 cache hits across ~22M lookups, 2.4x SLOWER
(37.0s vs 15.3s) than the baseline. Root cause investigated and found, not guessed: directly
inspected consecutive-step sketch vectors and found `||delta||` between a series' own sketch
one step apart is consistently ~1.2-1.6 (out of a max of 2.0) -- nowhere near "small" -- and
this holds even in the branch where no `basic_window`-boundary toggle-reweighting occurs at
all (ruled out that hypothesis directly), and across n_basic_windows = 4/12/20/30 (ruled out
the "too coarse a basic_window" hypothesis too). Even strongly autocorrelated synthetic AR(1)
data only gets the delta down to ~0.82 (raw) -- still too loose for the bound to ever fire
near a realistic gamma. Likely mechanism: the sketch is a sum of several basic-window
projections with no single dominant term (by the estimator's own design), so unit-normalizing
it is inherently unstable to small numerator changes. Same "correct but not beneficial"
pattern as this project's earlier row-level Cauchy-Schwarz bound -- now with the actual
mechanism identified.

**Status: implemented, correct, default OFF, left in the codebase as a documented negative
result** (not reverted -- harmless, opt-in, 131/131 tests green either way). Not recommended
for use. Open, untested idea: a representation with a more stable (higher signal-to-norm)
sketch construction might show a genuinely small per-step delta where `sketch_proj` does not.

**Commands run**: full local test suite (131 passed) after each change; Abaca resync + OAR
rebuild+test jobs 3103742 (gating fix) and 3103945 (score cache) -- both confirmed green.

**Next exact step:** either (a) accept the density-dependence investigation as complete for
now (options 4 and 5 land real, verified, disclosed wins; option 6 is a documented dead end)
and move to rerunning the three-domain four-way comparison on Abaca with the working options
enabled, or (b) if pursuing option 6 further, investigate whether a different sketch/
normalization scheme could produce a genuinely small per-step delta before trying the bound
idea again.

## 2026-09-14 (b) -- Options 5 and 6 rolled back per user request; density-dependence
investigation closed for now; pivoting to sourcing sparse+scalable real-world datasets

**Full write-up in `docs/implementation_log.md`'s 2026-09-14 (b) entry.** Summary only here.

Tested one more idea before the rollback (user's own question): does a different sketch
REPRESENTATION make option 6's bound useful, without touching `_incremental_sketches`'s
update algorithm? Checked the obvious candidate (raw pre-normalization delta vs normalized
delta) directly -- essentially identical (~1.3-1.5 either way, at every basic_window
granularity tested). Not a normalization artifact; the toggle-randomized per-basic-window
projection scheme's own construction (needed for it to be an unbiased cross-series estimator)
is what produces the large step-to-step movement. No fix available without either changing
the randomization (changes what's estimated) or falling back to a heuristic proxy (trades
away the exactness that was the point). Reported; user then said "roll it back."

**Rollback done, precisely**: options 5 (`candidate_refresh_interval`, cached-template
replay) and 6 (`candidate_score_cache`, the score-cache hash map + bound) are fully removed
from both `library_corrtrack_parallel.py` and `candidate_kernels.pyx` -- not just disabled.
**Option 4 (`candidate_search_n_threads` + `parallel_candidates` gate) is untouched and still
verified working** (byte-identical row sets at n_threads in {1,2,4,8}, gate behaves exactly
as before). 131/131 tests pass locally and on Abaca (job 3104019) after resync.

**Task status: density-dependence investigation closed for now.** Surviving deliverable:
option 4, a real ~2.24x exact speedup on the densest tested dataset (streamflow). The
underlying finding is unchanged by the rollback: every real dataset curated this session
lands in the high-true-degree regime where CorrTrack's LSH advantage is structurally limited.

**User's new direction**: stop fighting the density limitation for now; search for real-world
datasets that are genuinely correlation-SPARSE (low true degree, not just low density/(m-1) --
that metric was already found misleading this session, see the 2026-09-11(e) "degree metric
bug" entry) AND scalable (large m) -- the regime CorrTrack already demonstrably wins in
(2026-09-12's sparse-synthetic 3.66x result), to build the paper's real-data story around.

**Next exact step:** search for and screen candidate real-world datasets across domains
(finance/environmental/one more, per the original brief; France/Brazil or worldwide
preferred, US acceptable fallback), explicitly checking TRUE correlated-degree at real scale
(via the distinct-partners-per-series count, not density) before investing in any fetch/
curation work.

## 2026-09-14 (c) -- Redundant same-time enumeration fixed in HammingExactIndex AND
SignLSHBandIndex (default lsh_sign_dot); five-way comparison table; validation-cost
profiling

**Full write-up in `docs/implementation_log.md`'s 2026-09-14 (c) entry.** Summary only here.

**Branch:** `dev`. **Changed file:** `candidate_kernels.pyx` only (uncommitted -- user runs
commits).

**What changed:** both `HammingExactIndex._find_pair_rows_meta` and
`SignLSHBandIndex._find_pair_rows_meta` now skip a touched-candidate when it is a genuinely
same-time "recent vs recent" pair AND a `sid_idx` tie-break says the other direction already
claims it (`is_recent[node] and time_idx[node] == q_time and sid_idx[node] <= q_sid`) --
eliminates redundant duplicate enumeration/dot-product work without changing the final
candidate set (`pair_seen` already deduped output before this fix). Two real bugs caught via
an actual failing test (`test_lsh_sign_dot_index_flat_posting_list_survives_repeated_insert_
drop_churn`) before landing correctly: `sid_rank` is not a reliable tie-breaker (no
uniqueness guarantee in the general `insert_many` API -- that test passes the round number
for it); the time check must be explicit, not inferred from `is_recent` alone (that test's
query batch spans multiple time-rounds, not just one synchronized batch).

**Commands run:**
- `python3 setup_cython.py build_ext --inplace` (rebuild after each edit; one forced clean
  rebuild via `rm -f candidate_kernels.c candidate_kernels*.so` to rule out stale compilation)
- `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q` -> **131/131
  passed**, confirmed with the fix in its final, active state in both classes.
- `verify_parallel_lsh.py`, `verify_hamming_fix_correctness.py`, `debug_churn_test.py`,
  `hamming_enum_check.py`, `lsh_approx_enum_check.py`, `hamming_time_breakdown.py`,
  `hamming_exact_compare.py` -- all in scratchpad, not part of the repo.

**Results observed:**
- HammingExactIndex, sp500: `candidate_time` 12.899s->7.814s (39% down), `wall` 11.89s->9.44s
  (21% faster); `total_touched` 111,701,886->94,743,387.
- SignLSHBandIndex (`lsh_sign_dot`, the project default), sp500: `total_touched`
  121,648,955->105,453,691 (13.3% down), reproducible across 6 post-fix / 4 pre-fix runs.
  Wall-clock for this backend was NOT cleanly separable from run-to-run noise (candidate_time
  ranges overlapped substantially between pre/post-fix on this machine) -- reported honestly
  as an open point, not papered over.
- Recall/precision unchanged to the decimal on real data (HammingExactIndex sp500:
  0.966717/1.000000 before and after); byte-identical row sets vs. the unmodified parallel
  kernel for SignLSHBandIndex.

**Known issues:**
- `SignLSHBandIndex._find_pair_rows_meta_parallel` (option-4 prange kernel) not given this
  fix -- sequential kernels only.
- Five-way `hamming_exact_compare.py` table (wikipedia/sp500/smartmeter/streamflow) predates
  this fix for HammingExactIndex and is stale.
- `lsh_sign_dot` before/after only measured on sp500.
- Not yet synced to Abaca / rebuilt there.

**Next exact step:** sync `candidate_kernels.pyx` to Abaca and rebuild; then decide whether
to re-run the five-way comparison across all four datasets with the fix active, and get a
cleaner (more repeats / less-loaded machine / median-of-N) wall-time read for `lsh_sign_dot`
before drawing a real-world speedup conclusion for that backend from this fix.

## 2026-09-14 (d) -- lsh_sign_dot cleanup done; Python-loop audited and fixed; idea 4
built and tested -- decisive negative result at series-pair granularity

**Full write-up in `docs/implementation_log.md`'s 2026-09-14 (d) entry.** Summary only here.

**Branch:** `dev`. **Changed files:** `candidate_kernels.pyx` (idea-4 skip mechanism, new
and currently unused by any config path), `library_corrtrack_parallel.py`
(`_get_or_create_window_idx`'s return-value fix). Both uncommitted (user runs commits).

**Corrections made to earlier claims in this same session** (both caught before being
relied on further): idea 3 (incremental sketch update) is already implemented
(`_incremental_sketches`) -- nothing to build there. The first idea-4 persistence number
(99.9%) was a measurement bug (compared against the cumulative `ct.correlated` instead of
the per-step validated buffer) -- corrected to 71.6%/80.5% (wikipedia/sp500).

**"No Python in hamming_exact" audit**: candidate search confirmed fully Cython (one
batched call per step). Candidate INSERTION has a real per-series Python loop
(`_get_or_create_window_idx`), measured at 0.69% of wall time on sp500 -- real but not
material. Fixed anyway (user's call, hygiene): the function now returns `(idx, sid_idx,
sid_rank)` instead of just `idx`, removing a redundant list-index lookup at all 5 call
sites. 131/131 tests pass; not yet synced to Abaca.

**`lsh_sign_dot` cleanup (completing the prior entry's pending item)**: clean five-way
rerun across all 4 datasets confirms modest, real, consistent gains from the redundant-
enumeration fix (dot-gate speedup vs stomp: wikipedia 0.59->0.66x, sp500 0.72->0.74x,
smartmeter 0.82->0.86x, streamflow 0.62->0.64x) -- none beat stomp. Touched-candidate
reduction confirmed on all 4 datasets (8.8%-17.0%). One earlier concurrent-load run had
falsely shown smartmeter at 1.05x (beating stomp) -- redone cleanly, it's 0.86x; flagged
explicitly so that number is never mistaken for real.

**Idea 4 built and tested -- clear negative result, precisely diagnosed**: added a
`set_active_pairs` method + touched-loop skip check to `HammingExactIndex` (true no-op by
default, 131/131 pass unchanged), then a Python-side prototype that tracks validated
`(s1,s2,lag)` triples and injects their rows directly into the exact-Pearson validation
kernel each step, bypassing search for them. Result on wikipedia: recall collapsed 0.9808
-> 0.5743 (precision held at 1.0000 -- the injection/validation plumbing itself is
correct). Root cause isolated precisely (confirmed via a controlled fix-and-rerun, not
guessed): the search-side skip is granular at the SERIES-PAIR level, but real correlations
commonly span multiple simultaneous lags -- once a pair is marked active from one lag, the
search stops looking for it at ANY lag, permanently losing coverage of the pair's other
lags. **A viable version needs lag-aware tracking (per-pair lag sets), not attempted yet.**

**Commands run**: `python3 setup_cython.py build_ext --inplace`; `python3 -m pytest
test_stable_reproduced_changes.py test_synth_density.py -q` -> 131/131 passed (checked
after each change). Abaca: `ssh sophia.g5k`, `oarsub` job 3104885 (rebuild+test on a real
compute node after the frontend's `-march=native` SIGILL was diagnosed and avoided).

**Known issues**: idea 4's `set_active_pairs` mechanism exists in `candidate_kernels.pyx`
but must NOT be enabled/wired into any production config as-is -- it causes the recall
collapse documented above. `SignLSHBandIndex` doesn't have the idea-4 mechanism (only
`HammingExactIndex` does, for this first prototype). `_get_or_create_window_idx`'s fix not
yet synced to Abaca.

**Next exact step**: decide whether to build the lag-aware version of idea 4 (bigger lift:
per-pair lag sets checked against each candidate's actual time relationship, not a single
bit per pair) or close idea 4 out as a documented negative result given the more modest
71-80% persistence ceiling (vs the originally misreported 99%+). Sync
`_get_or_create_window_idx`'s fix to Abaca.

## 2026-09-14 (e/f/g) -- Idea 4 blind-spot fixed (two real bugs), but honest final
result: no reliable net speedup once measured with repetition

**Full write-up in `docs/implementation_log.md`'s 2026-09-14 (e), (f), (g) entries.**
Summary only here.

**User asked**: "can you fix the temporary blind-spot problem?" -- yes, fully fixed, via
two real bugs found through direct tracing (not theorizing):
1. Lag-aware tracked-pair mechanism built entirely in Cython (`candidate_kernels.pyx`
   `HammingExactIndex`: `update_tracked_pairs`, `get_tracked_rows`, a lag-aware skip in
   `_find_pair_rows_meta`, reusing the existing `_pair_seen_*` hash-set primitives + one
   new `_pair_seen_contains`) -- per the user's explicit "avoid Python loops altogether"
   instruction. Verified true no-op by default (131/131 pass).
2. First bug found (series-pair-only skip, wrong): superseded by the lag-aware version.
3. Lag-aware version STILL showed the same recall collapse (0.57) -- a targeted
   step-by-step trace (not the "flicker" theory, which was checked and logically ruled
   out first) found the real cause: `T_now` read before `ct.run()` is one `window_step`
   stale relative to what validation uses during that same call. Fixed:
   `T_now = window_index[-curr_window_size] + window_step`.
4. **Result: recall fully recovered** -- now matches or slightly EXCEEDS baseline on
   every dataset tested (wikipedia 0.9846 vs 0.9808 baseline; sp500 0.9783 vs 0.9667;
   smartmeter 0.9688 vs 0.9621; streamflow 0.9898 vs 0.9862). Precision 1.0000
   throughout. **The blind-spot problem the user asked about is fully resolved.**

**But fixing correctness revealed a bigger, separate problem**: per-phase profiling
found the search-skip mechanism itself is free (`get_tracked_rows`/`update_tracked_pairs`
= 0.03s of a 19.3s sp500 run), but a correctly-functioning tracked set is large
(mean ~2,600 pairs/step on sp500) and pushing all of them through the FULL
validation+monitor+record pipeline every step costs more than the search-skip saves.
Repeated measurement (sp500: 4 runs, all 0.76-0.85x; smartmeter: one run 1.031x, a repeat
0.867x with IDENTICAL `total_touched` both times, proving the swing is pure noise) shows
**no reliable net speedup on any dataset** -- isolated positive readings do not replicate.

**Commands run**: `python3 setup_cython.py build_ext --inplace`; `python3 -m pytest
test_stable_reproduced_changes.py test_synth_density.py -q` -> 131/131 (after each
Cython change). Scratchpad scripts: `idea4_prototype.py` (main driver, now
Cython-backed), `idea4_trace_pair.py` (the targeted trace that found the real bug),
`idea4_overhead_profile.py` (the per-phase profile that found the downstream-cost
issue), `idea4_missing_diag.py` (the diagnostic that first characterized the gap).

**Known issues**: `update_tracked_pairs`/`get_tracked_rows`/the lag-aware skip exist in
`candidate_kernels.pyx` as tested, correct, true-no-op-by-default infrastructure --
NOT wired into any production config path, and not recommended for that as-is (net
wall-time benefit is not established as real). Only `HammingExactIndex` has this;
`SignLSHBandIndex` doesn't.

**Next exact step**: decide whether to pursue the one remaining path with a plausible
chance of a genuine win -- true O(1) incremental sufficient-statistics tracking for
tracked pairs that bypasses the full validation+monitor+record pipeline too (not just
the search), entering the heavier path only when a pair's status actually changes. This
is a materially larger redesign than what exists now, not a small patch. Otherwise,
close idea 4 out here: correctness fully understood and fixed, but no demonstrated net
benefit from this design.

## 2026-09-14 (h) -- Checked the incremental-sufficient-statistics redesign before
building it: not promising, evidence-based. Idea 4 closed for HammingExactIndex.

**User asked**: "If it is promising, pursue it." Checked first via direct phase-time
decomposition (`ct.validation_time`/`ct.monitor_time`/`ct.candidate_time`) rather than
assuming the earlier size-correlation diagnosis was right. It wasn't: the extra cost is
almost entirely in `candidate_time` (+2.515s of ~2.6s total, sp500), not
validation/monitor (+0.014s/+0.066s -- noise-sized). The redesign targets
validation/monitoring, which was never the bottleneck -- it would not have helped, so it
was NOT built.

**Real mechanism**: `HammingExactIndex` is an exhaustive, unbucketed O(alive_count) scan;
its per-candidate cost (one XOR+popcount on a packed 64-bit word) is already near the
floor. The lag-aware skip check's hash lookup, run for every enumerated candidate, costs
more than the single-word comparison it decides whether to skip. This is a structural
mismatch between the mechanism and this specific backend's already-minimal per-candidate
cost, not a fixable implementation detail -- no redesign of what happens to a tracked pair
AFTER identification can fix a cost that lives in the identification step itself.

**Status: idea 4 is closed for `HammingExactIndex`.** Correctness (recall/precision) is
fully fixed and verified (2026-09-14(f)/(g)) -- the mechanism just doesn't pay for itself
on this backend. Not evaluated for `SignLSHBandIndex` (the default `lsh_sign_dot`
backend) -- real bucket/posting-list traversal there is plausibly expensive enough that a
skip-check could still win, but this is speculation, not tested.

**Next exact step (if continuing this direction)**: test the lag-aware skip mechanism on
`SignLSHBandIndex` specifically, where the per-candidate cost being skipped is a real
posting-list bucket traversal, not a single word compare -- the one remaining place this
approach could plausibly still pay off. Otherwise, idea 4 investigation is complete.

## 2026-09-14 (i) -- Idea 4 tested on SignLSHBandIndex per user request: unpromising,
for a different precise reason than HammingExactIndex. Idea 4 investigation complete.

**Full write-up in `docs/implementation_log.md`'s 2026-09-14 (i) entry.** Summary here.

**User asked**: "go ahead and test it" (SignLSHBandIndex). Ported the identical
lag-aware mechanism there (same fields/methods, skip check inside the band-scanning
loop, plus a `visited_stamp` marking addition needed because a correlated pair typically
shares many bands). Verified true no-op by default (131/131 pass,
`verify_parallel_lsh.py` byte-identical vs the untouched parallel kernel). Correctness
confirmed (recall slightly above baseline on wikipedia, precision perfect).

**Wall-time was noise-dominated** (three sp500 runs: 1.023x, 0.858x, 0.881x; a phase
breakdown across three more runs showed `candidate_time` deltas flipping sign entirely:
+1.235s, -2.350s, +4.028s). Rather than keep fighting noise, measured the
noise-immune metric directly: touched-candidate count, baseline vs idea4 on the same
data. **Result: 105,453,691 -> 104,833,122, a 0.59% reduction -- conclusive on its own.**

**Why so small**: `SignLSHBandIndex` already deduplicates a candidate to one touch per
query via `visited_stamp` regardless of how many bands find it. The mechanism can only
save the fraction of touches that are already-tracked pairs being rediscovered -- a tiny
fraction, since most of this backend's touched volume is one-off spurious bucket
co-occurrence (unrelated series sharing a band key by chance), which this mechanism
cannot help with at all.

**Final conclusion, both backends now tested**: idea 4 does not pay off, for two
distinct, precisely understood, structural reasons -- `HammingExactIndex`'s skip-check
costs more than the near-minimal thing it replaces; `SignLSHBandIndex` has almost
nothing for the skip-check to find (most touches are one-off, not repeat). Both are
properties of how these backends already work, not fixable by iterating on this
mechanism further.

**Status: idea 4 investigation is complete.** All infrastructure remains in
`candidate_kernels.pyx` on both index classes as tested, correct, true-no-op-by-default
code -- not wired into production, not recommended for it. Correctness work
(2026-09-14(f)/(g)) stands as a real, verified fix regardless of the speed conclusion.

**Next exact step**: none pending for idea 4. Available for a new direction, or return
to earlier open items (Abaca sync of `_get_or_create_window_idx`'s fix; the density-
targeted synthetic generator plan; re-running the five-way comparison with all of this
session's fixes, if still wanted).

## 2026-09-14 (j-o) -- Idea 4 rolled back + Abaca synced; dataset search found a real,
usable low-degree candidate (m=94) and the exact recipe to scale it further

**Full write-up across `docs/implementation_log.md`'s 2026-09-14 (j) through (o)
entries.** Summary here.

**Rollback + Abaca sync (complete)**: all idea-4 code removed from `candidate_kernels.pyx`
(both index classes -- fields, methods, skip checks, the orphaned `_pair_seen_contains`
helper); verified via grep (zero references left), a forced clean rebuild, 131/131 tests,
and `verify_parallel_lsh.py` byte-identical cross-check. Synced to Abaca (`sophia.g5k`),
rebuilt via OAR job 3106372 on a real compute node, confirmed clean (131 passed, 245
subtests, 36.54s). The tree now reflects only this session's kept fixes: the
redundant-enumeration (`is_recent`) fix in both index classes, and
`_get_or_create_window_idx`'s return-value fix.

**Dataset search (in progress, real finding, not yet finalized)**: tested the hypothesis
that spreading series across genuinely independent contexts (geography/market/sector)
keeps true correlated-degree bounded as m grows, unlike every dataset curated earlier
this session (all drawn from one compact population).
- Global temperature (100 stations worldwide via Open-Meteo): **failed decisively** --
  raw-space avg_degree=100/100 (every series correlated with every other, seasonal-cycle
  confound), diff-space avg_degree=40.23/100 (still high -- daily temperature carries
  globally-correlated fluctuation structure beyond the removable seasonal trend).
- Global equities v1 (64 tickers, Yahoo Finance): promising signal, avg_degree=13.81,
  clear regional gradient (US high, Asia-Pacific low) -- but m too small, T truncated by
  an IPO-alignment bug.
- Global equities v2 (168 tickers, fixed T-truncation, T=2824): overall average degree
  rose to 32.40 -- diagnosed as a ticker-list-composition mistake (grew US/EU roughly as
  much as the low-correlation group, so its SHARE of the sample didn't shrink) plus a
  real, expected effect of the longer 11-year span including real crisis-correlation
  spikes (2020, 2022). LatAm and Canada/South Africa turned out NOT to be low-correlation
  (23-36 avg degree) -- only specifically Asia-Pacific + smaller/less-globally-integrated
  markets are.
- Global equities v3 (206 tickers, deepened the low-correlation regions): overall average
  barely improved (30.05) despite doubling that group's ticker count -- diagnosed
  precisely: adding MORE tickers WITHIN already-included markets recreates small
  "mini-sp500" clusters per country, so depth-per-market drives degree up, not region
  choice.
- **Isolated the real lever and confirmed it directly**: extracted the 94-series
  Asia-Pacific/smaller-market subset from v3's already-fetched data and screened it as
  its OWN universe (no new fetch) -- **avg_degree=6.43, median 7.0, 10 series at exactly
  zero**, with Israel's 5 tickers at avg_degree=0.00 (zero correlation with anything,
  even each other) and New Zealand at 0.33. Per-exact-market breakdown confirms: degree
  scales with HOW MANY TICKERS PER MARKET are in the sample (Australia/Taiwan/Indonesia/
  Thailand at 9-12 tickers each show degree 5-8; Israel/NZ at 5-6 tickers show near-zero)
  -- NOT with region identity.

**Current state**: a real, immediately usable candidate exists at m=94, T=2613
(~10.4 years), avg_degree=6.43 -- comparable in scale to wikipedia (this session's
smallest real dataset) but with much longer history and a far more favorable (low-degree)
correlation structure than every other real dataset curated this session. The exact
recipe to scale this to sp500-size m (~500) is now known and validated: add MANY more
DISTINCT markets (aim 2-4 tickers each, not 8-12) rather than deepening existing ones.

**Datasets saved**: `tmp_artifacts/global_weather/global_weather.npz` (m=100, T=5113 --
negative result, not useful), `tmp_artifacts/global_stocks/global_stocks.npz` (v1, m=64,
T=1052 -- superseded), `tmp_artifacts/global_stocks_v2/global_stocks_v2.npz` (m=168,
T=2824), `tmp_artifacts/global_stocks_v3/global_stocks_v3.npz` (m=206, T=2613 -- contains
the promising 94-series Asia-Pacific/smaller-market subset).

**Next exact step**: decide whether to (a) use the m=94 subset as-is for the five-way
comparison re-run on Abaca now, or (b) invest in a v4 fetch (many more distinct markets,
2-4 tickers each) to reach a larger m before running the comparison. Either way, the
comparison itself has not been run yet on any of this new data -- that is the concrete
remaining step once a dataset is finalized.

## 2026-09-15 (a) -- BREAKTHROUGH: CorrTrack beats exact_stomp on two real, honestly-
sourced datasets -- the first time this entire session's investigation has found this

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (a) entry.** Summary here.

**Context**: user challenged the dataset-sourcing methodology directly ("Is this
realistic? I am skeptical") after the 2026-09-14 hand-tuned global-equity iterations
(v1-v4) kept adjusting ticker composition toward the target metric. Correctly caught a
real methodological problem. Response: fetched the REAL, live iShares MSCI ACWI ETF's
actual disclosed holdings (2,251 real positions from ishares.com) and built two datasets
using selection rules fixed BEFORE seeing any correlation result:
- `acwi_real` (m=125, T=2425): top 4 by weight per country (diversification-capped).
  avg_degree=22.21.
- `acwi_capweighted` (m=263, T=2452): top 300 by raw global weight, no country cap
  (necessarily US-mega-cap-heavy). avg_degree=119.95, comparable to/above sp500's own
  density.

**Clean, single-OAR-job Abaca measurements (dedicated compute node, not noisy local
machine)**:
- `acwi_real`: lsh_hamming_exact+dotgate beats exact_stomp **1.05x** (2.35s vs 2.5s),
  beats filcorr 1.90x. Precision 1.0000.
- `acwi_capweighted`: lsh_hamming_exact+dotgate beats exact_stomp **1.48x** (8.52s vs
  12.6s). Precision 1.0000.

**This is the first time this session has found ANY real dataset where CorrTrack beats
STOMP outright** -- and it happened on two independently-built real datasets. Every prior
real dataset (wikipedia/sp500/smartmeter/streamflow, and the earlier hand-tuned global-
equity variants) showed 0.49x-0.86x, never beating STOMP.

**Open question, not yet resolved**: `acwi_capweighted` has avg_degree comparable to
sp500's own high-degree regime (previously established as WHY LSH speedup collapses),
yet it beats STOMP here. Candidate explanations not yet isolated: longer T (~10y vs
sp500's shorter span), smaller m (263 vs 492), or the SHAPE of the degree distribution
rather than its average.

**Commands run**: `scp` datasets + `hamming_exact_compare.py` to `sophia.g5k:~/
corrtrack_release_dev/` and `~/corrtrack_abaca_results/`; `oarsub` jobs 3107130
(acwi_real) and 3107131 (acwi_capweighted), both completed cleanly on p2 queue. One real
bug caught and fixed along the way: the recreated `hamming_exact_compare.py` (rebuilt
after a `/tmp` scratchpad cleanup wiped the original) hardcoded the LOCAL username's repo
path (`/home/rsalles/...`), which doesn't exist on Abaca (`rpontess`) -- fixed to resolve
the repo path from the working directory instead (`os.getcwd()`), since the OAR job
script already `cd`s into the repo before invoking the script.

**Datasets**: `tmp_artifacts/acwi_real/acwi_real.npz`, `tmp_artifacts/acwi_capweighted/
acwi_capweighted.npz` (both local and synced to Abaca). Result JSONs on Abaca at
`~/corrtrack_release_dev/tmp_artifacts/hamming_exact_compare_acwi_{real,capweighted}.json`.

**Next exact step**: investigate the open question above (why acwi_capweighted beats
STOMP despite high average degree) -- isolate T vs m vs degree-distribution-shape as the
driver, likely via a controlled re-run of sp500 at matched T or a synthetic degree-
distribution-shape sweep.

## 2026-09-15 (b/c) -- Corrected five-way methodology (lsh_sign_dot tuned replaces
no-dotgate ablation); complete seven-dataset Abaca table; major reframing finding

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (b) and (c) entries.**
Summary here.

**Methodology change (user request)**: `hamming_exact_compare.py`'s fifth comparison arm
changed from "no dot-gate" (internal ablation) to "lsh_sign_dot, tuned" (the actual
default production `candidate_backend`, gamma-calibrated the same way as the dotgate
arm, `target_occupancy=5.0`).

**Negative control**: subsampled sp500 to m=263 (`sp500_sub263`, matching
acwi_capweighted's scale exactly) -- does NOT beat STOMP (0.93x/0.75x). Rules out `m`
alone as the reason acwi_capweighted beats STOMP.

**Complete seven-dataset table, clean single-OAR-job Abaca measurements**:

| dataset | m | T | hamming+dotgate vs STOMP | lsh_sign_dot(tuned) vs STOMP |
|---|---|---|---|---|
| wikipedia | 88 | 1827 | 1.08x | 0.92x |
| sp500 | 492 | 1255 | 0.90x | 0.72x |
| smartmeter | 510 | 15000 | 0.97x | 0.73x |
| streamflow | 538 | 2192 | 1.30x | 1.09x |
| acwi_real | 125 | 2425 | 1.05x | 0.85x |
| acwi_capweighted | 263 | 2452 | 1.51x | 1.17x |
| sp500_sub263 (control) | 263 | 1255 | 0.93x | 0.75x |

**Major finding, bigger than any single dataset result**: these Abaca numbers are
dramatically better than the SAME datasets measured locally earlier this session
(wikipedia 0.66x->1.08x, streamflow 0.64x->1.30x, sp500 0.74x->0.90x, smartmeter
0.86x->0.97x) -- far beyond noise. Two real, disclosed, unconfounded-from-each-other
candidate causes: (1) dedicated Abaca compute node vs. this session's own noisy, shared
local machine; (2) `-march=native` compiled specifically for Abaca's CPU, which could
have wider SIMD (e.g. AVX-512) that disproportionately helps CorrTrack's own
Hamming/dot-product kernels relative to STOMP's simpler scalar updates. Not yet isolated
which cause explains how much of the gap.

**Practical implication**: the premise behind the whole 2026-09-14 idea-4 investigation
and dataset search ("CorrTrack never beats STOMP on real data") does not hold once
measured on dedicated, representative hardware -- 3 of 4 original real datasets now
show parity or better. The correctness fixes from idea 4 (off-by-one T_now bug,
lag-granularity fix) remain real and independently verified regardless of this reframing.

**Commands/artifacts**: all 7 jobs run via `oarsub` on Abaca (`sophia.g5k`), datasets and
`hamming_exact_compare.py` synced to `~/corrtrack_release_dev/tmp_artifacts/` and
`~/corrtrack_abaca_results/` respectively. Result JSONs saved per-dataset on Abaca.

**Next exact step**: decide whether to isolate the dedicated-hardware-vs-SIMD-width
confound (recompile with a portable `-march` target on both machines, re-measure) or
accept Abaca's numbers as the authoritative, representative measurement going forward.

## 2026-09-15 (d) -- Confound resolved: contention, not CPU capability

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (d) entry.**

Checked Abaca's actual CPU (Xeon E5-2650 v2, 2013, AVX-only, no AVX2/AVX-512) vs local
(i7-1165G7, 2020, full AVX-512) -- opposite of the SIMD-width hypothesis; Abaca's CPU is
older/weaker. Rebuilt locally with `-march=haswell` (no AVX-512) and re-measured wikipedia
-- result (0.69x) barely moved from the original AVX-512 build's historical number
(0.66x), ruling out AVX-512 downclocking. Checked `ps aux` on the local machine directly:
multiple OTHER active Claude Code sessions and VS Code processes running concurrently
RIGHT NOW, confirmed as a genuinely shared, multi-tenant machine, unlike Abaca's
exclusively-allocated OAR compute nodes. **Conclusion: contention on the shared local
machine, not CPU capability, explains the gap.** Abaca's numbers are the trustworthy,
representative measurement going forward. Local build reverted to `-march=native` (its
normal, correct state) after the test; `setup_cython.py` has no uncommitted diff.

## 2026-09-15 (e) -- Threshold sweep (0.70/0.80/0.90/0.95) complete across all 8 datasets

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (e) entry.** Five-way
comparison (bruteforce/exact_stomp/filcorr/lsh_hamming_exact+dotgate/lsh_sign_dot tuned)
re-run at corr_threshold in {0.70, 0.80, 0.90, 0.95} across wikipedia, sp500, smartmeter,
streamflow, acwi_real, acwi_capweighted, sp500_sub263, global_weather, all on Abaca (24
OAR jobs, 3107291-3107314). Headline: `exact_stomp`/`filcorr` speedups stay flat across
threshold (fixed O(1)-per-pair exact cost regardless of density); `hamming_exact+dotgate`
and `lsh_sign_dot` speedups grow dramatically as threshold tightens (sp500 hamming+dotgate
2.91x->9.62x, lsh_sign_dot 2.33x->12.41x, going 0.70->0.95) because sparser true-positive
signal lets the sketch/bucket filter calibrate tighter. Every cell met recall>=0.95 except
smartmeter's lsh_sign_dot at thr=0.90 (0.924) and thr=0.95 (0.883) -- disclosed, not
investigated further. Confirmed corr_threshold=0.70 was used identically across the
original (non-swept) battery, and confirmed both hamming_exact+dotgate AND lsh_sign_dot
use the identical gamma-offset calibration loop (both genuinely tuned, not one-sided).

## 2026-09-15 (f) -- m/L-scaling investigation; n_vectors sweep launched; W/L audit

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (f)/(g) entries.**
m-scaling: using `sp500` (m=492) vs `sp500_sub263` (m=263, same underlying data,
negative control) at the already-collected threshold-sweep data, CorrTrack's relative
speedup advantage over the smaller-m version grows with threshold (1.19x at thr=0.80,
1.54x at 0.90, 1.88x at 0.95) while degree-as-fraction-of-(m-1) stays flat between the
pair -- mechanistically explained by bruteforce/STOMP being exactly O(m^2) vs
CorrTrack's sub-quadratic bucket search. L-scaling: no controlled test run (deferred,
not requested); theoretical expectation only, disclosed as unverified.

n_vectors investigation: user flagged that n_vectors=64 exceeds window_size in every
dataset this session. Verified in code that `n_words=ceil(n_vectors/64)` stays 1 for any
n_vectors<=64 (Hamming pre-filter cost unaffected), and that the sketch construction is a
toggle-sign-summation ensemble over a small `basic_window`-sized basis, not a one-shot JL
projection -- meaning the classical "target dim <= source dim" heuristic does not apply.
Ran `nvectors_sweep.py` (fixed corr_threshold=0.70) across all 6 non-degenerate datasets
at n_vectors in {4,8,16,32,64} (powers of 2 per user's explicit instruction) --
**n_vectors=64 wins unambiguously in every dataset, often by a wide margin** (sp500:
0.24x->0.90x speedup_vs_stomp). User decision: keep n_vectors=64 throughout, confirmed
by evidence, no rework needed on existing tables.

W/L audit (no new sweep, per user's explicit "don't sweep W/L for now" instruction):
existing configs fell into two internally-consistent classes -- daily financial
(W=60/STEP=5/N_LAGS=20, L=5: sp500/acwi_real/acwi_capweighted/sp500_sub263) and daily
social/environmental (W=30/STEP=3/N_LAGS=15, L=6: wikipedia/streamflow/global_weather) --
plus smartmeter alone on its own sub-daily config. Flagged that wikipedia/streamflow
sharing an identical config despite unrelated domains is likely a copy-paste artifact,
not independent derivation. Conclusion at the time: within-class comparisons already
clean, cross-class comparisons confounded but disclosed, not resolved.

## 2026-09-15 (h) -- W/L unified to W=30/STEP=3/N_LAGS=15 (L=6) across all daily datasets;
financial datasets re-running on Abaca

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (h) entry.** User asked to
resolve the cross-class confound from (f)/(g) rather than merely disclose it -- offered
the choice of unifying onto 30 days, 60 days, or keeping the split; user chose **30
days**. Checked first that this is sound: wikipedia, streamflow, sp500, acwi_real,
acwi_capweighted, sp500_sub263, global_weather are all daily-resolution (one column = one
day), so a shared day-count is a genuine like-for-like span across all seven. smartmeter
is native 30-minute resolution (current W=48 = exactly 1 day) and was kept on its own,
separately-justified config (W=48/STEP=8/N_LAGS=16) as a disclosed exception -- forcing
it onto 30 calendar days would misrepresent its actual target (daily consumption-cycle
correlation, not monthly trend). n_vectors stays fixed at 64 (already the case, no change
needed).

Changed `sp500`/`acwi_real`/`acwi_capweighted`/`sp500_sub263` in
`hamming_exact_compare.py`'s CFG from W=60/STEP=5/N_LAGS=20 to W=30/STEP=3/N_LAGS=15,
matching wikipedia/streamflow/global_weather exactly. This invalidates those four
datasets' existing five-way comparison results at every threshold level, so re-submitted
all 16 cells (4 datasets x {0.70, 0.80, 0.90, 0.95}) as OAR jobs on Abaca: **job IDs
3107693-3107707,3107709**, queue `abaca`, using the updated script synced to
`~/corrtrack_abaca_results/hamming_exact_compare.py` on sophia.g5k.
wikipedia/streamflow/global_weather results are unaffected (already at W=30) and do not
need re-running.

### Known issues / open items
- 16 OAR jobs (3107693-3107707,3107709) pending on Abaca as of this entry -- financial datasets'
  five-way comparison at the new unified W=30/L=6 config, across all 4 threshold levels.
- Once complete, the (e) threshold-sweep tables need republishing with the new W=30
  financial numbers replacing the old W=60 ones.
- L-scaling remains theoretically reasoned but empirically untested (deliberately
  deferred, not requested).
- smartmeter's lsh_sign_dot recall shortfall at thr=0.90/0.95 (0.924/0.883) remains
  disclosed but not investigated (calibration grid may be too coarse).

### Next exact step (superseded -- see 2026-09-15 (i)/(j) below)

## 2026-09-15 (i) -- Two more real tuning bugs fixed (n_bands was already auto,
occupancy genuinely wasn't tuned); final 32-job corrected battery complete

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (i) entry -- has the
complete 32-cell final table (8 datasets x 4 thresholds).** User asked directly
whether hamming_exact/stomp/filcorr need tuning beyond gamma (answer: no -- gamma is
hamming_exact's only knob, verified in `HammingExactIndex._finalize_threshold`;
stomp/filcorr have zero tunables, confirmed from their signatures), then flagged
that `candidate_lsh_n_bands` "should not be fixed" and occupancy "should also be
tuned." Verified: n_bands was ALREADY auto-computed in every prior run (the script's
explicit `n_bands=64, n_bands_tolerance=1.0` was a no-op matching the constructor's
own default -- `_finalize_sizing` always overrides it), so nothing was actually
broken there, just misleadingly written; removed for clarity. Occupancy, however,
genuinely WAS fixed at 5.0 and never swept -- a real gap. Fixed by sweeping occupancy
in {2,3,5,8,12} x the existing gamma-offset grid in `run_lsh_sign_dot`'s calibration.

This invalidated every `lsh_sign_dot_tuned` number collected this session (not
hamming_exact/stomp/filcorr/bruteforce, which don't touch occupancy). Re-ran the
full battery: all 8 datasets x all 4 thresholds = 32 OAR jobs (IDs 3107734-3107765),
all completed within minutes. This is now the final, authoritative table -- see the
implementation log entry for the full 32-row table. Tuned occupancy now genuinely
varies (2-12) by dataset/threshold instead of sitting at the old fixed 5.0, and this
materially changed results (e.g. sp500 at thr=0.95 lsh_sign_dot: 11.21x at occ=2).

## 2026-09-15 (j) -- Started building maximal-pool versions of every dataset (for
later Sobol-sweep reproduction); m=500/big-T standardization still pending

**Full write-up in `docs/implementation_log.md`'s 2026-09-15 (j) entry.** User wants
"all available data" for every dataset (a flexible large pool for later reproducing
the Sobol sweep across bf/stomp/filcorr/both CorrTrack backends), and separately
asked about fixing m=500 with a common big T (bigger for smartmeter) for the FIRST
five-way experiment specifically.

Feasibility-checked first: wikipedia's pageviews API has a hard floor at 2015-07-01
(verified via direct 404 test) -- cannot reach 2013 like the other datasets.
smartmeter needed real Kaggle credentials to expand past m=510 (none were configured
in this environment). User chose: shared window = 2013-2023 for everyone except
wikipedia (max possible 2015-07-01 to 2023-12-31, disclosed exception), and provided
a Kaggle API token (new bearer-token format, works directly, no legacy kaggle.json
needed) -- **handled carefully: not written to any repo file/log, used only via an
in-session env var; user was told to regenerate it since it was pasted in chat**.

Built so far, all in the scratchpad directory (not yet copied to `tmp_artifacts/`):
- `sp500_full.npz`: m=444, T=2768 (2013-2023) -- down from 492 as expected (index
  reconstitution loss going back a decade, matches the tradeoff flagged in advance).
- `streamflow_full.npz`: m=608, T=4017 (2013-2023) -- UP from 538, fetched from the
  full ~2419-site USGS CA candidate pool (985 returned data, 608 survived alignment).
- `smartmeter_full.npz`: m=2953, T=27649 (2012-08-01 to 2014-02-28) -- built from
  ALL 112 halfhourly blocks of the Kaggle "Smart meters in London" dataset (~5561
  households total, 7.4GB downloaded). `2012-06-01` start exactly reproduced the
  ORIGINAL m=510 dataset's T=30577 (nice independent consistency check);
  `2012-08-01` chosen as the best m*T trade-off point.
- `weather_full_raw.pkl` and `wikipedia_full_raw.pkl`: both STILL FETCHING in the
  background as of this entry (candidate pools: 856 stations via a pre-registered
  5-degree grid-binning rule over a public world-cities list; 866 articles from 5
  large-direct-membership categories). Both APIs rate-limit hard; fixed with
  exponential backoff + checkpointing.

### Known issues / open items
- weather/wikipedia background fetches not yet finished -- check
  `/tmp/.../scratchpad/{weather,wikipedia}_fetch.log` for progress, then run the
  same align-and-build pattern as `build_sp500_full.py`/`build_streamflow_full.py`.
- The actual m=500-capped, common-T standardized subsample (for re-running the FIRST
  five-way experiment on directly-comparable data) has NOT been built yet -- still
  needs a deterministic-prefix-slice design (matching `prepare_training_data`'s
  nested-subsample convention) once all `_full.npz` pools exist, plus a decision on
  the exact T to standardize on for smartmeter given its different resolution.
- L-scaling remains theoretically reasoned but empirically untested (deferred).
- smartmeter's lsh_sign_dot recall shortfall at thr=0.90/0.95 (0.952/0.941 in the
  corrected battery) remains disclosed but not investigated.

### Next exact step
Check `tail /tmp/claude-1000/-home-rsalles/*/scratchpad/{weather,wikipedia}_fetch.log`
for completion. Once both are done, build `weather_full.npz`/`wikipedia_full.npz` via
the same densest-calendar + ffill/bfill(limit=3) + drop-unfillable pattern used for
sp500/streamflow. Then design the m=500/common-T standardized subsample and confirm
the exact numbers with the user before re-running the five-way battery on it.

---

## 2026-09-15 (n) -- CorrJoin paper extraction (planning only, no implementation)

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

The user supplied the CorrJoin paper (PACMMOD 1(4):235, CC-BY) and instructed that
it, not the derivative UZH thesis, is the specification. Worked the §10.1 extraction
checklist against it; all six items resolved and §10.1 is now closed. §3.1 of the
plan is the paper-authoritative specification; §3.2 records what changed.

**Two claims previously in the plan were wrong and are now corrected:**
1. **CorrJoin has no lag support** -- it is entirely synchronous. So ParCorr,
   StatStream and CorrJoin are *all* synchronous; only CorrTrack and FilCorr do
   lags. Propagated to §1, §5b.5 and §6.1.
2. **Epsilon formulas**: `e1 = sqrt(2*ks*(1-T)/n)`, `e2 = sqrt(2*ke*(1-T)/n)` --
   both divide by n, and e1 uses ks (not kb). Both thesis variants were wrong.

Also extracted: incremental update via five running sums (reuses `exact_stomp`'s
sufficient-statistics form); speedup bounded by `1/r1`; the paper's own baselines
were all reimplemented in R by its authors (precedent for the plan's §0 policy);
and two results worth citing in our own paper -- their Fig. 15 density finding
(corroborates our 17.6%-vs-2% result) and their Fig. 10 TSUBASA measurement
(corroborates ranking TSUBASA secondary).

### Known issues / open items
- "Plan it but do not do it yet" still stands -- no competitor implementation begun.
- §10.2 still open: ParCorr grid/subspace parameters (DMKD 2018), StatStream DFT
  coefficient count + grid parameters (VLDB 2002), locating the authors' R code,
  and whether to adopt CorrJoin's four datasets as an evaluation axis.
- The 7 `abaca/` comparison scripts remain **uncommitted**; commit command was
  handed over earlier, no confirmation it was run.

### Next exact step
Extract ParCorr's grid/subspace parameters from the DMKD 2018 paper and cross-check
them against the `corrtrack_release_v1.0` / `corrtrack_release_backup2` snapshot
implementations -- or, if the user lifts the implementation hold, start phase 1: the
shared normalize-window-before-reducing path needed by both ParCorr and CorrJoin.

---

## 2026-09-15 (o) -- ParCorr extracted + snapshot audit; BRAID added (planning only)

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

Three pieces of work, all planning:

1. **ParCorr DMKD 2018 obtained** (HAL-LIRMM `lirmm-01886794`). Parameters closed:
   `r=60, k=2` so `n_grids=30`, `f=0.7`, window 500, basic window 20, T=0.7.
   `f` is calibrated to a 0.95 target recall on a sample -- the same protocol
   CorrTrack uses, which removes the main fairness objection to §6.2. ParCorr does
   **not** check neighbouring cells (CorrJoin does), and its own future-work list
   confirms it has no lag support. Gap: the paper specifies **no grid cell size**.

2. **v1/backup2 snapshot audited.** Mine `corrtrack_release_v1.0`, not `backup2`.
   The ParCorr machinery is present but disabled: `freq_threshold=0` hardcoded
   ("disable frequency gating", v1.0:2233), `full_vector_candidates` collapses to
   `n_grids=1` on the default path (v1.0:2182-2185), and the surviving vote uses
   `required_hits = n_grids`, i.e. effective `f=1.0` not 0.7 (v1.0:4647). The
   cell-size formula `sqrt(2(1-T))/sqrt(r)` is the user's own derivation, not
   ParCorr's, and must be labelled as such or replaced by the paper's procedure.

3. **BRAID added to the competitor set** (SIGMOD 2005; Osaka mirror -- the CMU copy
   has no extractable text layer). It is the **only peer on the lag axis**, which
   after entry (n) is CorrTrack's sole differentiator against the whole pruning
   family. It does **no pair pruning** (all O(k^2) pairs kept), so it is orthogonal
   to the other four rather than a fifth pruning rival. Sequenced at phase 4, ahead
   of TSUBASA and CorrJoin, on scientific-value-per-unit-work grounds.

### Known issues / open items
- "Plan it but do not do it yet" still stands. Nothing implemented.
- BRAID's accuracy axis (**relative lag error**, not pair recall/precision) does not
  fit §6.2's equal-recall protocol. Recorded in plan §5a.3; needs its own metric.
- **BRAID's TKDD 2010 extension is unread and may add pair pruning** -- would change
  its role from pure lag peer to lag+pruning competitor.
- StatStream (VLDB 2002) is now the only unread primary source among the pruning
  competitors.
- The ParCorr cell-size decision must be made before phase 2.
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step
Read StatStream (VLDB 2002) for its DFT coefficient count and grid parameters, and
check whether BRAID's TKDD 2010 extension adds pair pruning. Neither blocks;
implementation stays gated on the user lifting the hold.

---

## 2026-09-15 (p) -- StatStream read; two capability claims corrected; ParCorr cell
size decided (planning only)

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

1. **StatStream (VLDB 2002) does lags** -- and negative correlation (Lemma 3), and
   a minimum-duration/persistence parameter, and grid pruning with **no false
   negatives** (Theorem 2). Entry (o)'s claim that BRAID is the only lag peer was
   wrong, as was the plan's "primarily synchronous" description. The lagged arms are
   **CorrTrack, FilCorr, StatStream and BRAID**. StatStream is now the **closest
   structural peer to CorrTrack** and the most demanding arm, not a historical
   ancestor.
   Key numbers: `eps = sqrt(1-T)` filter radius, `n=16` DFT coefficients (swept
   16/24/32/40), `T=0.85/0.9`, precision 0.9765-0.9947, recall 0.9987-1.0, pruning
   power 0.01-0.09. Lags in the grid path are **quantized to basic-window
   multiples**.

2. **Process rule added (plan §10.3)**: two capability claims have now been
   overturned by reading primary sources, in opposite directions (CorrJoin credited
   with lags it lacks; StatStream denied lags it has). The second flattered
   CorrTrack, which is the dangerous direction. No capability claim goes in the
   paper without a section or lemma number from that competitor's own paper.

3. **ParCorr cell size DECIDED (plan §2.3)**: `_compute_base_cell_size` is **not
   used** for the ParCorr arm, per the user's instruction. Instead follow ParCorr's
   own calibration methodology -- fix `f=0.7`, `k=2`, then take the largest cell size
   on a held-out sample that still reaches target recall 0.95, and report it as a
   calibrated hyperparameter. StatStream's cell-size rule is not transferable here.

### Known issues / open items
- "Plan it but do not do it yet" still stands. Nothing implemented.
- **Open, deliberately not guessed**: does ParCorr handle negative correlation? The
  only blank cell in the §5b.5 capability table.
- BRAID's TKDD 2010 extension unread; may add pair pruning.
- §6.1 now requires the lagged comparison to be run **twice** (basic-window-multiple
  lags, which favour StatStream; and arbitrary lags, which favour CorrTrack), with
  both reported.
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step
No literature blockers remain for the pruning family. Either close the two small
open questions (ParCorr negative correlation; BRAID's TKDD extension), or start
phase 1 if the hold is lifted: the shared normalize-window-before-reducing path,
now with **three** consumers (ParCorr, CorrJoin, StatStream), plus factoring out the
five-sum sufficient-statistics helper shared by `exact_stomp`, CorrJoin and BRAID.

---

## 2026-09-16 (a) -- Evidence tiers added; CSZ 2005 found to be ParCorr's source;
BRAID TKDD and ParCorr neg-corr closed (planning only)

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

1. **Evidence tiers (plan §5b.6)**, from the user's observation that papers claim
   capabilities they never implement or evaluate. Capability matrix now graded
   **[E] evaluated / [S] specified / [C] claimed / [N] absent / [?] unread**.
   StatStream's lags are **[S]**: full derivation, grid pseudocode and a `T_M`
   parameter, but **no lag experiment in the paper**. Same for its Lemma 3 on
   negative correlation. So the lagged comparison has **one** genuine published peer
   (BRAID), not three. The same standard is applied to CorrTrack.

2. **Cole-Shasha-Zhao, KDD 2005 (plan §4a)**: **ParCorr's candidate search was
   published in 2005** -- same sketch, same partition into groups, same one-grid-per-
   group, same fraction-`f` vote. Shasha authored StatStream, CSZ and ParCorr.
   So **CSZ, not ParCorr, is the true ancestor of CorrTrack's candidate search**.
   It also **fills the ParCorr cell-size gap** with a published knob (distance
   multiplier `c`, swept 0.1-1.3), which is better than yesterday's
   calibrated-hyperparameter compromise. Decision: fold CSZ into the ParCorr arm as
   one arm with ablations, not a second near-duplicate.
   Its **cooperative/uncooperative** distinction is the organising axis for the
   evaluation: stock *prices* are cooperative (favour DFT-based arms), stock
   *returns* are uncooperative (break them). Both one transform apart on data we
   already hold, and both must be reported.

3. **BRAID TKDD 2010**: adds ThinBRAID (`O(k)` updates via random projections) but
   **no pair pruning** -- their Table II keeps output at `O(k^2 log n)`. BRAID's
   framing as the unpruned lag method stands.

4. **ParCorr does not handle negative correlation** (user-confirmed). The arm runs
   positive-only and the asymmetry is reported, not patched with an `abs()`.

### Known issues / open items
- "Plan it but do not do it yet" still stands. Nothing implemented.
- **Biggest open risk: the FilCorr paper (ICDM 2020) has never been read**, yet
  FilCorr is already implemented and already in our published benchmark results.
  Its capability column is entirely `[?]`. See plan §10.4.
- TSUBASA (SIGMOD 2022) also unread, though only planned.
- Zhu & Shasha TR2002-827 may hold the missing StatStream lag experiments; if so,
  StatStream's lag row moves [S] to [E].
- Does CorrJoin evaluate negative correlation, or only specify it?
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step
Ask the user for the **FilCorr (ICDM 2020)** paper, then **TSUBASA (SIGMOD 2022)**.
Read FilCorr against the existing `Candidates_BF_FilCorr` implementation to verify
the port and fill its capability column -- that is a check on already-published
results, so it outranks further planning.

---

## 2026-09-16 (b) -- All primary sources read; two deviations found in the shipped
FilCorr port (planning only)

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

1. **FilCorr (ICDM 2020) read.** The port is sound, but two deviations must be
   disclosed: (a) **FilCorr has no negative-correlation handling** (Eq. 7 takes `Max`
   of signed correlations); our port inherits exact_stomp's `|corr| >= threshold` rule
   when `neg_corr=True`. Defensible as a harness choice, values identical, pair set a
   superset, 2026-09-12 numbers unaffected, but it is **our extension, not FilCorr**.
   (b) FilCorr reports one value per pair per timestamp (max over the lag window); our
   port reports per-lag pairs.
   Also: FilCorr **rejects pruning by design** (data-dependence); its Table I grades
   competitors as claimed/extendable/unknown, a **published precedent for our evidence
   tiers**, and marks StatStream's lags as unknown; and it beats best-case ParCorr up
   to **~700 streams**, just above our `m=500` battery.

2. **TSUBASA (SIGMOD 2022) read.** Exact correlation matrix from per-basic-window
   mean, standard deviation and per-pair correlation; arbitrary query windows; **no
   pruning** (their own future work); **negative correlation yes and exercised**; no
   lags. **Space is `O(L*N^2/B)`** because it stores a correlation per pair per basic
   window: budget before implementing. Their §4.1 independently corroborates the
   uncooperative-data finding, and more sharply than CSZ.

3. **StatStream TR2002-827 read.** The fuller report still has **no lag experiment**.
   StatStream's lag support is confirmed `[S]` in both versions.

**No `[?]` cells remain in the capability matrix. All primary sources are read.**

### Known issues / open items
- "Plan it but do not do it yet" still stands. Nothing implemented.
- **The two FilCorr deviations are not yet in the 2026-09-12 four-way writeup.**
- Does CorrJoin evaluate negative correlation, or only specify it?
- TSUBASA's sketch store is quadratic in series count.
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step
Either start phase 1 if the hold is lifted, or add the FilCorr deviation note to the
2026-09-12 four-way writeup -- a cheap correction to already-published results.

---

## 2026-09-16 (c) -- Implementation plan written for all six competitor arms

**Branch**: dev. **New file**: `docs/competitor_implementation_plan.md`. Comparison plan
now links to it. **No code touched.**

The six methods fall into **two integration patterns that already exist in the code**:
- **Pattern A, `baseline_mode`** (all-pairs, no pruning): add **TSUBASA** and **BRAID**
  beside `bruteforce` / `exact_stomp` / `filcorr`. FilCorr is a debugged 61-line
  template (`library_corrtrack_parallel.py:9060`); ~120 mechanical lines per arm plus
  the class.
- **Pattern B, `data_representation` x `candidate_backend`** (pruning): add
  **ParCorr/CSZ**, **StatStream**, **CorrJoin**. Both axes are already pluggable
  (`:1738`, `:1745`) and need widening, not inventing; v1.0 holds a disabled grid.

That split is exactly the "prunes pairs" column of the capability matrix.

**Phase 0 (blocking, 2-3 d)**: window-normalization path shared by 3 arms; five-sum
helper shared by 3; counter contract (pruning counters explicitly 0 for Pattern A);
**negative-correlation policy**; one shared `_assert_competitor_contract` test; check
`n_lags=0` has no guard. scipy 1.11.4 verified present for BRAID.

**Two independent tracks after phase 0**, since they touch disjoint code:
A: TSUBASA (2-3 d) then BRAID (4-5 d). B: ParCorr/CSZ (3-4 d), StatStream (5-7 d),
CorrJoin (7-10 d). **Total ~25-32 focused days**, spread almost all in B2/B3.
Fallback: Track A + B1 + B2; cut CorrJoin, never BRAID.

### Known issues / open items
- Still no implementation; awaiting the go-ahead.
- **5 open decisions in §8**, the first of which also settles the outstanding FilCorr
  negative-correlation deviation: run primary at `neg_corr=False` so no arm is extended
  beyond its paper, with arms *refusing* rather than silently extending.
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step
Decide §8 item 1 (negative-correlation policy), then start phase 0a.

---

## 2026-09-16 (d) -- Correction: FilCorr is also an evaluated lag peer. §8 item 1 decided.

User caught a stale claim: BRAID is **not** the only lag peer with published evidence.
**FilCorr is [E]** (lag is its problem statement; Figs. 6-7 sweep lag 0/100/250) and is
already built. **CSZ is [C]** (asynchronous correlation defined in the problem statement,
no algorithm, no experiment). StatStream stays [S]. So: **two** evaluated lag peers.

While fixing it, found a worse stale line in comparison plan §6.1: FilCorr described as
"shares CorrTrack's pruning structure". **FilCorr does not prune.** Both fixed; §6.1 now
grades the lag row explicitly. Root cause: prose written before the FilCorr paper was
read, not re-derived after the matrix was updated; plus an aborted patch script whose
§6.1 edit never landed.

BRAID's priority is unchanged but the reason is now correct: it is the only evaluated lag
peer **we do not already have**, and it probes a different failure mode from FilCorr.

**§8 item 1 DECIDED** (user agreed): primary at `neg_corr=False`; second labelled run
where arms lacking the capability report N/A; enforced by a per-arm `supports_neg_corr`
flag the harness refuses on. Also settles the FilCorr deviation.

### Next exact step
Phase 0a: `sketch_norm="unit_l2_window"` path with the `2 - 2*corr == d^2` identity test.

---

## 2026-09-16 (e-f) -- §8 item 1 refined and fully decided

Policy: primary at `neg_corr=False`. Second labelled run at `neg_corr=True` with a
four-class per-arm tag written into result rows: `native` (BF, STOMP, TSUBASA),
`specified` (BRAID, StatStream, CorrJoin -- their spec, our evaluation),
`enabled_by_us` (**FilCorr**, confirmed by user), `not_available` (**ParCorr/CSZ**, the
only N/A). Criterion: whether enabling changes the algorithm under measurement, not
coding effort. The 2026-09-12 FilCorr numbers stand with disclosure; no re-run.

### Next exact step
Phase 0a: `sketch_norm="unit_l2_window"` with the `2 - 2*corr == d^2` identity test.

---

## 2026-09-16 (g) -- §8 items 2-5 decided; all five now closed

2. **CorrJoin SVD: per-window.** The SVD is on the `m x ks` (ks=15) matrix, `O(m ks^2)`,
   negligible. My "research-grade incremental SVD" was an overstatement carried from the
   pre-paper draft. Follow-ups: check `cSVD` term; ask authors for R code.
3. **ThinBRAID: build.** Battery to 2k series; at 2k plain BRAID needs ~1.5 GB vs ~12 MB.
4. **Tuning: CSZ protocol for competitors, CorrTrack keeps its own.** Same 0.95 target;
   robustness check runs CorrTrack once under CSZ.
5. **Datasets: add all obtainable.** New phase 0f, 14 sets catalogued; Motes and
   Yellowstone first among the real ones.

Estimate now ~30-40 focused days. **Nothing blocks phase 0a.**

---

## 2026-09-16 (h-i) -- Implementation started. Phase 0a + 0b done, 126/126.

CorrJoin authors' R code received (`docs/reference_code/corrjoin_authors_R/`): SVD and
PAA recomputed per window, only per-series sums incremental, normalization after PAA
with window stats (= the 0a design), `abs(corr)` live -> CorrJoin neg-corr is [E].
All literature items closed.

**0a**: `sketch_norm="unit_l2_window"` in `Sketches`, rides the mean_l2 kernel path and
rescales by `sqrt(var_sum)` afterwards; kernels untouched; `mean_l2` byte-identical.
Exact test `_sketch_matrix == R_eff @ x_hat` (independent reconstruction) passes at
1e-10. Not yet a `CorrTrack.__init__` parameter (B1 does that).
**0b**: `five_sums`, `pearson_from_five_sums` module-level helpers.
Standalone patch: `docs/patches/2026-09-16_phase0a_0b_unit_l2_window_five_sums.library.patch`.

### Known issues
- **Working tree has another session's uncommitted library changes** (230 diff lines
  in `CorrTrack`/`Candidates_BF_ExactSTOMP`/`Candidates`) intermingled with mine (122).
  Commit needs `git add -p` or the patch file. Not reviewed by me.
- 7 `abaca/` scripts still uncommitted.

### Next exact step
Phase 0c/0e (counter contract + `_assert_competitor_contract`), then A1 TSUBASA.

---

## 2026-09-16 (j) to 2026-09-17 (c) -- All six competitor arms implemented. 145/145.

See `docs/implementation_log.md` entries (j) TSUBASA, (k) BRAID/ThinBRAID, 2026-09-17 (a)
ParCorr/CSZ, (b) StatStream, (c) CorrJoin, and `docs/competitor_implementation_plan.md`
(A1, A2, B1, B2, B3 marked DONE).

**Branch**: dev (nothing committed by me; user commits).
**Changed files**: `library_corrtrack_parallel.py`, `test_stable_reproduced_changes.py`
(122 -> 145 tests), `corrtrack_run_bruteforce.py` (tsubasa/braid modes, `--braid-*`),
`corrtrack_run_corrtrack.py` (`--parcorr-*`, `--statstream-*`, `--corrjoin-*`, new
representations/backends in help), `experiment_run_exec_param.py` (BRAID_*, PARCORR_*,
STATSTREAM_*, CORRJOIN_*), docs (comparison plan, implementation plan, log),
`docs/patches/2026-09-16_competitors_cumulative.library.patch`,
`docs/reference_code/corrjoin_authors_R/`.

**Arms and how to run them**
- `corrtrack_run_bruteforce.py --baseline-mode tsubasa|braid [--braid-thin]` (Pattern A,
  all-pairs; counters `total_candidates == tested`).
- `corrtrack_run_corrtrack.py --data-representation sketch_proj --candidate-backend parcorr_grid`
  (`--parcorr-k/-f/-c`, `--parcorr-neighbor-probe` for CSZ), `--data-representation sketch_dft
  --candidate-backend statstream_grid`, `--data-representation sketch_paa_svd
  --candidate-backend corrjoin_double_filter` (Pattern B; `total >= tested >= correlated`).
- Every record carries `supports_neg_corr` in {native, specified, enabled_by_us, not_available}.

**Commands run**: `python3 -m pytest test_stable_reproduced_changes.py -q` -> 145 passed.
**Known issues**: other session's 230 uncommitted library lines coexist (commit via the patch
file or `git add -p`); `abaca/` 7 scripts uncommitted; Pattern-B competitor indexes are pure
Python (a tier below the Cython `SignLSHBandIndex`; counters are primary).
**Corrections recorded**: CorrJoin neg-corr [E] -> [S]/unreachable, `not_available`.

### Next exact step
Phase 0f: `datasets/registry.py` + fetch scripts for the catalogued sets (implementation plan
§0f), then generalize `abaca/fourway_compare.py` to N arms.

---

## 2026-09-17 (d) -- Phase 0f done; N-way runner; ThinBRAID fixed on real data. 147/147.

See log entry 2026-09-17 (d). **Branch**: dev, nothing committed by me.
**New**: `datasets/competitor_loader.py`, `datasets/competitor_sources.md`, `datasets/fetch/*`
(11 scripts), 22 `experiment_dataset_<name>.py`, `abaca/nway_compare.py`, `abaca/nway_compare.oar`.
**Modified**: `library_corrtrack_parallel.py` (ThinBRAID cache key + centred Eq. 24;
`_BASELINE_SUPPORTS_NEG_CORR`), `test_stable_reproduced_changes.py` (+2), `.gitignore`.
**Data**: `datasets/competitor/*.npz` (1.1 GB, git-ignored, regenerable; raw/ cached).

**Commands run**
- `python3 datasets/fetch/fetch_motes.py`, `fetch_yellowstone_iris.py`, `fetch_uscrn_hourly.py --year 2020`,
  `fetch_corrjoin_drive.py`, `fetch_sunspots.py`, `fetch_csz_daisy.py`,
  `<venv with h5py> fetch_berkeley_earth.py --decade 2010 --absolute`
- `python3 abaca/nway_compare.py --dataset-config experiment_dataset_motes_temperature.py --arms all
  --window-size 96 --window-step 12 --n-lags 0 --corr-threshold 0.9 --n-obs 4000 --set corrjoin_ks=12 --set corrjoin_ke=24`
  and the same with `--n-lags 24 --neg-corr --n-obs 2500`
- `python3 -m pytest test_stable_reproduced_changes.py -q` -> 147 passed

**Results**: table in the log entry; all exact arms 1.0; StatStream/CorrJoin/CSZ 1.0;
ThinBRAID 0.967 (was 0.12 before the two fixes); ParCorr 0.40 at defaults (needs tuning).

**Known issues**: ParCorr untuned; h5py needed for Berkeley Earth; docs/ and tasks/ untracked.

### Next exact step
Write `abaca/tune_competitors.py` implementing CSZ's protocol (comparison plan section 4a.1)
for parcorr/csz/statstream/corrjoin knobs at a 0.95 recall target, per dataset; CorrTrack keeps
its own hyperopt. Then per-dataset campaign configs (W/step/N_LAGS from the registry rows).

---

## 2026-09-17 (e) -- CSZ tuning protocol + campaign manifest. 148/148.

See log entry 2026-09-17 (e). **Branch** dev, nothing committed by me.
**New**: `abaca/tune_competitors.py`, `abaca/tune_competitors.oar`, `abaca/campaign_competitors.py`.
**Modified**: `abaca/nway_compare.py` (`--competitor-params`), `abaca/nway_compare.oar`,
`test_stable_reproduced_changes.py` (+1), implementation plan, 2026-09-12 log entry (FilCorr note).
**Commands**: `python3 abaca/tune_competitors.py --dataset-config experiment_dataset_motes_temperature.py
--arms parcorr,csz,statstream,corrjoin --window-size 96 --window-step 12 --n-lags 0 --corr-threshold 0.9
--n-obs 10000 --calib-obs 2000 --out-dir <dir>` (16 min, CSZ dominates) then
`python3 abaca/nway_compare.py ... --competitor-params <dir>`; `python3 abaca/campaign_competitors.py --list|--emit`;
pytest -> 148 passed.
**Results**: held-out recall parcorr 0.944, csz 0.938, statstream 1.0, corrjoin 1.0 (ParCorr was 0.40 untuned).
**Known issues**: CorrTrack hyperopt stage not yet in the manifest; nothing submitted to Abaca;
datasets must be synced to Abaca (git-ignored).

### Next exact command
After the user commits and pushes: on `sophia.g5k`, `git pull`, rsync `datasets/competitor/`, then
`python abaca/campaign_competitors.py --emit abaca/campaign_submit.sh` and submit a 3-cell pilot
(edit the script down to Motes T=0.9 sync, Motes L480 T=0.9, Yellowstone bp T=0.9) before the full run.

---

## 2026-09-17 (f) -- hyperopt per cell, Phase R runner, Abaca synced. 148/148.

See log (f). **Branch** dev; local unpushed: da417b0 + uncommitted (this entry's files).
**New**: `abaca/hyperopt_corrtrack.oar`, `abaca/reproduce_papers.py`. **Modified**:
`corrtrack_param_search.py`, `abaca/campaign_competitors.py`, `abaca/nway_compare.oar`,
`abaca/tune_competitors.oar`, `datasets/fetch/gen_braid_synthetic.py`, implementation plan.
**Abaca**: repo at 9e983d3, datasets synced, abaca/ backup + stash made.
**Open decisions for the user**: W/step/L table in log (f); pilot cells.

### Next exact step
User pushes; on `sophia.g5k`: `cd ~/corrtrack_release_dev && git pull --ff-only origin dev &&
python abaca/campaign_competitors.py --emit abaca/pilot_submit.sh --select motes_temperature_W240_s24_L0_T0.9
--select motes_temperature_W240_s24_L480_T0.9 --select yellowstone_bp3_7_W2000_s100_L1000_T0.9 && bash abaca/pilot_submit.sh`;
in parallel `oarsub -l host=1,walltime=24:00:00 -S "./abaca/reproduce_papers.oar"` once written (Phase R at scale).

---

## 2026-09-17 (g) -- Sobol (m, L) campaign design, T sweep, six legacy sets, run metrics. 148/148.
See log (g). New: `abaca/dataset_profile.py`, six `experiment_dataset_*.py`. Modified: campaign,
nway, three .oar (BASIC_WINDOW), loader glob, reproduce_papers. Awaiting user: W/step policy,
`--points`, pilot cells; then push + `git pull` on sophia.g5k + `bash abaca/pilot_submit.sh`.

---

## 2026-09-17 (h) -- Phase R: BRAID and StatStream reproduce; ThinBRAID open; horizon rule. 148/148.
See log (h). Modified: reproduce_papers, gen_braid_synthetic, library (thin_d_min), campaign (Motes
2880/288, corrjoin_stock 60/5), plan 3a. Decisions: horizon rule, Sobol semantics, Cython ports.
### Next exact step
`competitor_kernels.pyx`: Cython hot loops for StatStreamGridIndex (probe + filter), then ParCorr, CorrJoin;
parity tests; setup_cython.py entry; measure per-candidate time on the Motes N-way.

## 2026-09-17 (i) -- design split (sync/lag Sobol), ASOS sets added: 25 datasets, 557 cells. See log (i).

## 2026-09-17 (k) -- competitor_kernels.pyx: Cython candidate loops for StatStream, ParCorr/CSZ, CorrJoin. 149/149.
Parity-tested; 46x to 149x on the candidate stage; competitor arms now within 1.2x to 2.4x of CorrTrack's
candidate time on Motes. See log (k). Next: user push, `git pull` + pilot on sophia.g5k, W-robustness cells, Phase R at scale.
## 2026-09-17 (l) -- ladder design (m, L together), 505 cells / 3,030 jobs, cap 2k + full-m anchors. See log (l).

---

## 2026-09-17 (m) -- m cap 5k, step-time quantiles, ablation runner, global ASOS, Phase R job 3117155. 149/149.
See log (m). **Abaca**: repo at 53f9600; Phase R running (job 3117155, results under
`~/corrtrack_abaca_results/reproduce_papers_3117155/`). **Next exact step**: after the user's push,
`ssh sophia.g5k`, `cd ~/corrtrack_release_dev && git pull --ff-only origin dev && python abaca/campaign_competitors.py
--emit abaca/pilot_submit.sh --select motes_temperature_m27_W2880_s288_L0_T0.9 --select sp500_m492_W60_s5_L20_T0.9
--select global_asos_air_temperature_m600_W168_s12_L0_T0.9 && bash abaca/pilot_submit.sh`; then the W-robustness
cells, the ablation cells, the full campaign.

## 2026-09-18 (a) -- Phase R results (job 3117155), boxplot ticks, ablation ladder + all_pairs backend. 150/150.
See log (a). Next: push; on sophia.g5k pull, rerun `reproduce_papers.oar EXPERIMENT=braid` and `EXPERIMENT=filcorr`,
submit the pilot (`campaign_competitors.py --emit ... --select` 3 cells), write `abaca/ablation.oar`, W-robustness cells.

---

## 2026-09-18 (g)+(i) [this thread] -- Monitor kernel: hoisted views (g, ~4x) then AoS layout + prefetch + pointer appenders (i, ~10x more on the kernel, 6.3x on library monit_time), bit-identical. 150/150.
Branch: main (local), Abaca repo `dev`. **Uncommitted**: `monitor_kernels.pyx` (both (g) and (i); regenerated `.c`/`.so`),
`docs/implementation_log.md`, `tasks/current_task.md`, `.gitignore` (docs/tasks unignored). User runs the commit.
See log 2026-09-18 (g) and (i). Verification: `scratchpad/mon_equiv/{equiv_test,equiv_repo_vs_cur,e2e_check,bench4}.py`.
**Abaca**: new kernel synced to `~/corrtrack_release_dev/monitor_kernels.pyx`; running jobs hold their own isolated
copies. Job 3120771 (`aos_verify_job.sh`) = test suite + full-T e2e equivalence + `monitor_profile_diag.py` on a node.
Batches: `m5v2_*` 51/56 and `fcv2_*` 19/20 on the (g) kernel; `fcnm_*` (FilCorr sweep, monitoring off) 10/20.
ASOS 18-country re-fetch on the frontend: 15/18 done.
### Next exact step
Cluster verification 3120771 done: E2E_OK on full T, monit_time 30.1s -> 3.8s; 148/149 tests (path assertion only).
When `m5v2_*` = 56, `fcv2_*` = 20, `fcnm_*` = 20: pull the JSONs from `~/corrtrack_release_dev/tmp_artifacts/`
(`hamming_exact_compare_m500_*.json`, `filcorr_band_thr_sweep_*{,_nomon}.json`), build the m=500 diff, m=500 raw,
FilCorr-sweep (monitored) and FilCorr-sweep (no monitor) tables, diff against `_pre_monitor_opt_2026-09-18/`.
User chose the third rerun on the (i) kernel: `m5v3_*`/`fcv3_*` (70 submitted, 6 deferred via
`~/corrtrack_abaca_results/rerun_v3_jobs/defer_launch.sh` on the frontend; log `defer_launch.log`). (g) results
archived in `tmp_artifacts/_g_kernel_2026-09-18/`. Final tables come from the v3 JSONs (76 monitored) + 20 `_nomon`. -> v3 complete 2026-09-19 21:15 (76/76 + 20 nomon); final tables in docs/tables_m500_and_filcorr_sweep_2026-09-19.md. ASOS gap fill resumes 01:30 (gaps_fill_night.sh), then pivot+screen.
Sweden ASOS: `global_asos/se_retry_watch.sh` on the frontend retries SE with a 900s curl timeout one hour after
`MISSING_COUNTRIES_REFETCH_DONE`; check `se_retry_watch.log` / `fetch_missing.log` for `SE_RETRY_DONE`, then rerun
`pivot_asos_wide.py` and `screen_global_asos_degree.py` (`screen_asos_job.sh`).

## 2026-09-19 (c) [competitor campaign thread] -- W-robustness cells; generator split; Phase R reruns queued; pilot emitted, not run
Branch: `dev` (repo `corrtrack_release_dev`), Abaca frontend at d0e3378. **Uncommitted (this thread)**: `abaca/campaign_competitors.py`,
`docs/implementation_log.md` (c), this file. See log 2026-09-19 (c).
Abaca: Phase R reruns 3122029 (braid) 3122030 (parcorr) 3122031 (csz) 3122032 (filcorr). `abaca/pilot_submit.sh` (36 jobs) and
`abaca/campaign_submit.sh` + `abaca/campaign_submit_generate.sh` (9,606 jobs, 403 generator commands) emitted on the frontend.
### Next exact step
1. `ssh sophia.g5k; cd ~/corrtrack_release_dev; bash abaca/pilot_submit.sh` (after the user's go); watch `oarstat -u`, then
   inspect `$RESULTS_ROOT/<stem>/{hyperopt,tune,nway}` JSONs and fit walltimes.
2. Read `tmp_artifacts/reproduce_{braid,parcorr,csz,filcorr}*.json` when 3122029..32 finish; update the Phase R table.
3. Write the aggregator `abaca/aggregate_campaign.py` (per-cell JSON -> long table -> recall/precision/specificity/speedup/
   step-latency figures by T, m, L, space, dataset profile).
4. Wait for the complete ASOS pivot (companion thread, job 3122021 then the 2026-09-20 gap retry) and set `global_asos_*` m_max.
5. Full campaign: `oarsub -q abaca -l host=1,walltime=24:00:00 "bash abaca/campaign_submit_generate.sh"` then
   `bash abaca/campaign_submit.sh` in per-dataset batches.

## 2026-09-19 (d) [competitor campaign thread] -- per-arm resources/energy, snapshot build, node partition, feeder, aggregator
Branch `dev`. **Uncommitted (this thread)**: `abaca/{resource_probe,campaign_feeder,aggregate_campaign,kwollect_power}.py` (new),
`abaca/prepare_snapshot.oar`, `abaca/_snapshot_enter.sh` (new), `abaca/{nway_compare,campaign_competitors,ablation_corrtrack,parallel_scaling}.py`,
`abaca/{nway_compare,hyperopt_corrtrack,tune_competitors,reproduce_papers,experiment}.oar`, `test_abaca_tools.py` (new),
`docs/competitor_implementation_plan.md` s3b(ii), `docs/implementation_log.md` (c)(d), this file. See log 2026-09-19 (d).
Abaca: same files scp'd to `~/corrtrack_release_dev` (uncommitted there too); snapshot job 3122067 (TAG=pilot) running;
probes 3122061 (RAPL: root-only), 3122062/64/65/66/68/69/70 (kwollect monitor types).
### Runbook (in order)
1. `oarsub -q abaca -p "cluster='mercantour3'" -l host=1,walltime=1:00:00 -S "./abaca/prepare_snapshot.oar TAG=<tag>"` -> SNAPSHOT dir
2. `export SNAPSHOT=$HOME/corrtrack_abaca_results/snapshots/<sha>_<tag>/corrtrack_release_dev`
3. `python abaca/campaign_competitors.py --emit abaca/campaign_submit.sh` (also writes `_generate.sh`)
4. `oarsub -q abaca -p "cluster='mercantour3'" -l host=1,walltime=24:00:00 "SNAPSHOT=$SNAPSHOT bash abaca/campaign_submit_generate.sh"`
5. `nohup python3 abaca/campaign_feeder.py abaca/campaign_submit.sh --max-waiting 60 --poll 120 > abaca/logs/feeder.log 2>&1 &`
6. `python abaca/kwollect_power.py --results-root ~/corrtrack_abaca_results --out ~/corrtrack_abaca_results/power.json`
7. `python abaca/aggregate_campaign.py --power ~/corrtrack_abaca_results/power.json`
### Pilot state at hand-over (2026-09-19 ~15:00)
motes 4 cells complete (all arms ok, tuned files loaded). sp500 neg runs complete (see log); sp500 pos N-way waits on
the uncapped tune jobs 3122221/3122233 (parcorr 30 min, csz running). ASOS cells resubmitted with CALIB_WINDOWS=64
CALIB_SERIES=500 (pilot3: 3122288..3122299). Pilot snapshot `snapshots/d0e337877e3d_pilot/corrtrack_release_dev`
carries two post-build patches (tune_competitors.py), listed in its SNAPSHOT_OK; the campaign snapshot must be
built from the committed tree. Open decisions for the user: synthetic n_obs (20000 -> 5000?), heavy all-pairs arms
above m = 2,500, neg_corr run at L > 1, PACK/NWAY host counts, feeder limits.
All pilot jobs finished; results under `~/corrtrack_abaca_results/{hyperopt,tuned,nway}/<stem>_<tag>/`. Last verification
running: pilot4 (one motes cell) through the snapshot's own wrappers after the third snapshot rebuild (job 3122383).
### 2026-09-19 (e) hyperopt preprocess fix
`library_corrtrack_parallel.py` (proxy reference in the sketched space), test added; **uncommitted**. Not yet on Abaca:
sync after the user's commit and rebuild the campaign snapshot. Probes queued on the pilot snapshot for the hyperopt
cost at large m (raw space, unaffected by the fix): 3122888 (corrjoin_gas m5000 L1), 3122889 (berkeley m5000 L4),
3122890 (corrjoin_gas m2500 L1), core=10 on PACK_HOSTS, out `~/corrtrack_abaca_results/probes/hyperopt_*`.
