# Implementation Log

> Note: `docs/` and `tasks/` did not exist in this checkout of `corrtrack_release_dev` before
> this entry — CLAUDE.md assumes they do, but this particular sandbox checkout was missing them.
> This entry documents only the work directly observable in this session (dashboard artifacts
> built from already-completed sweep CSVs); it does not attempt to reconstruct the history of the
> underlying `n_bands` theory, the OA/Sobol sweep implementations, or their production wiring,
> since that work is not evidenced by files present in this checkout and would be fabricated if
> written here from memory alone.

## 2026-08-31 — OA dashboard refinement + companion Sobol dashboard

**Branch:** main (no source-code changes this session — all work was building/publishing HTML
dashboard artifacts from pre-existing sweep result CSVs; nothing here was committed).

**Inputs used (read-only, not modified):**
- `tmp_artifacts/lsh_orthogonal_array/oa_sweep.csv` (16 rows) + `trial_grids/cell{1..16}.json`
- `tmp_artifacts/lsh_sobol_sweep/sobol_sweep.csv` (83 rows, 2 known failures) + `trial_grids/*.json`
  + `sobol_sweep_manifest.json` (design parameters: 65 unique Sobol points, 18 replicate reruns of
  9 of them, 1 fixed anchor point, ranges m∈[50,300], L∈[8,64], density∈[0.001,0.1], threshold∈[0.7,0.95])

**What changed:**
1. Iteratively refined the OA(16,5,4,2) dashboard (`https://claude.ai/code/artifact/b7d77cd2-b60e-47a4-8892-83a8b30d77a0`)
   across many rounds: added/fixed axis labels and legends on every chart type, added a generic
   sortable+filterable table component, added box-and-whisker mode for high-repeat-count trial
   data, fixed a real bug where a chart's y-axis could place a measured point above the visible
   canvas when a regression fit undershot the true max, fixed a bug where two "projected speedup"
   figures (a table and a chart) silently disagreed because one anchored its projection to a real
   measured point and the other evaluated the bare regression fit — unified on the anchored
   convention everywhere. Removed a "minimum-height-not-to-scale" rendering workaround after the
   user rejected it, restoring the honest (if visually thin) true-scale comparison charts instead.
   De-duplicated the Parameter Sensibility tab (mechanism-knob charts had been repeating content
   already shown in the per-parameter rows).
2. Built a companion dashboard for the Sobol sweep's 83 points
   (`https://claude.ai/code/artifact/0a673d9b-9b05-47e4-90d7-07bdebfdb38d`), reproducing the OA
   dashboard's refined structure and chart infrastructure (drawScatter/drawLinearMulti/
   drawExtrapChart/renderSortableTable reused near-verbatim) but adapted for continuous,
   non-balanced data:
   - Replaced the OA's "average over cells sharing a discrete level" main-effects method with a
     real multivariate log-log OLS regression (solved via Gauss-Jordan elimination on the normal
     equations, cross-checked against `numpy.linalg.lstsq` on the actual data — identical to 4
     decimal places) for m/L/density/threshold exponents, each controlling for the other three
     simultaneously.
   - Replaced the "Recall by Configuration" 4×4 heatmap grid (meaningless for continuous data)
     with 2D scatter coverage plots (m×L, density×threshold) colored by recall.
   - Replaced the GF(4) orthogonal-array balance proof with a Sobol-appropriate check: 2D
     projection scatter plots plus a direct 4×4-bin occupancy count (min/max points-per-cell,
     empty-cell count) as the "verified, not assumed" evidence of even coverage.
   - Added a new analysis with no OA equivalent: grouped the 9 points that were rerun 2 extra
     times each (18 replicate rows total) to directly measure run-to-run noise (mean recall
     spread ≈4.8%, mean speedup spread ≈23%, both by range÷mean) — the only place in either sweep
     with genuine repeated measurements.
   - Verified before publishing (not assumed): JSON validity of both embedded data blocks, full
     JS syntax validity via `esprima.parseScript` (a real parser, not just brace-counting),
     `getElementById` cross-check against actual HTML ids, duplicate-id check, and the regression
     math cross-validated against `numpy` as above.
   - Sanity check: brute-force runtime's regression-fitted m-exponent came out to 2.04 (true
     complexity is exactly m²) — confirms the regression method is implemented correctly on real
     data, independently of the OA dashboard's own (different-method) sanity check.

**Known open items (disclosed in both dashboards' Caveats tabs, not resolved):**
- `candidate_search_lsh_candidates_touched` (a genuine pre-dot-gamma-filter count) is not
  captured by either sweep script; the Filtering tab's funnel/table use the post-filter
  `tested_candidates` count instead, with an honest note rather than fabricating the distinction.
- Gamma's own stage-2 calibration search isn't logged trial-by-trial anywhere (only the final
  selected value + a trial count survive) — Parameter Sensibility analyzes gamma like a
  controlled per-point value, not with the same full-trial view as tolerance/occupancy.
- The 2 Sobol failures are both replicate reruns of the same extreme corner (m=296, L=55); the
  original point at that corner completed fine, so this reads as a marginal/flaky memory-limit
  failure mode, not a hard wall — consistent with what was already known about this corner.

## 2026-09-01 — Sobol dashboard: outlier rerun, model-based extrapolation, validation 2-stage fit

**Branch:** main (still no source-code changes — all work is on the published dashboard
artifacts; nothing here was committed).

**What changed (Sobol dashboard, `https://claude.ai/code/artifact/0a673d9b-9b05-47e4-90d7-07bdebfdb38d`):**
1. Cell 5 (m=296, the extreme m×L×sparse-density corner) was flagged as a leave-one-out outlier
   (200-775% above trend on several phases) and rerun via
   `python3 experiment_lsh_sobol_sweep.py --only-cells 5` (~56 min); `sobol_sweep.csv` row
   replaced. sketch/validation/monitor times dropped sharply (consistent with the original run
   being resource-contention-distorted); `candidate_time` got worse (230.8s→302.7s), still the
   largest outlier. Root cause: this corner's 400-step calibration window structurally
   underestimates recall for every tolerance×occupancy trial tried (best calibration recall 0.697
   vs. actual 0.962) — an extreme instance of the same false-negative mechanism already documented
   in the Reliability tab, not something a rerun can fix. Disclosed in the Design tab and Caveats,
   not hidden.
2. Found and fixed a real methodology flaw in the extrapolation curves: the intercept/vertical
   position was previously pivoted through one raw observed row (whichever had the largest m or
   L), making it fully dependent on that single point's noise. Replaced with a model-based
   intercept (other 3 factors held at the dataset's median) — verified numerically to change the
   m=10,000 total_time projection from ~26,300s to ~9,200s.
3. Added R²/residual-vs-noise-floor confidence metrics (noise floor measured from the 9
   replicated points) throughout the Complexity and Extrapolation tabs; added the total
   experimental time (~14h wall-clock from file mtimes, ~7h measured-compute lower bound from
   summed columns) as stat tiles.
4. Investigated why validation_time's 4-factor fit was the weakest in the study (R²=0.79) despite
   visibly trending with m, and why monitor_time's fit (R²=0.90) was fine despite depending on
   correlation density: monitor's true driver, `validated_candidates` (the true-correlation
   count), is an almost clean function of (m, density) alone (R²=0.93); validation's true driver,
   `tested_candidates`, is only R²=0.68 explained by the 4 design factors, because it depends on
   which of 12 discrete tolerance×occupancy combinations the tuner's search happened to land on
   per point — a real, jumpy, non-power-law effect, not a code bug. Cross-checked against
   `numpy.linalg.lstsq` directly on the CSV.
5. Improved validation_time's fitted model per the user's request: now a 2-stage fit
   (`validation_time ~ tested_candidates + m + L + density + threshold`, R²=0.925) instead of the
   plain 4-factor fit, with `tested_candidates` itself projected forward via its own 4-factor
   model for extrapolation. Implemented as `fitWithModelTwoStage()`, algebraically reduced to the
   same `{slope, intercept}` shape every other phase fit uses (composition of two power laws,
   holding factors fixed, is still one power law) — verified numerically against
   `numpy.linalg.lstsq` (R²=0.7931→0.9245, matching to 4 decimals). This raises validation's
   in-sample fit quality close to the other phases, but the forward projection still carries
   tested_candidates' own weaker R²=0.68 as compounded uncertainty — disclosed in the
   phase-exponent table (asterisked row) and the extrapolation callout, not hidden by the
   improved headline R².
6. Added a density-on-x-axis reduction chart to the Filtering tab (mirrors the existing
   threshold-on-x chart, roles swapped), a new gold-family `ctSeqColor`/`ctToColor` color ramp for
   it.
7. Added regression trend-line overlays (fit in the same log/linear space each chart already
   plots in) to every Parameter Sensibility chart, including the boxplot-mode ones (fit through
   all underlying trial values, not just box medians).
8. Reduced the Extrapolation tab's charts to 2 milestones (m=1,000 and m=10,000, previously also
   5,000) since 3-wide charts had become too small to read; the summary table is unaffected and
   still shows 500/1,000/5,000/10,000.
9. Reworded "spread" to "variation (range ÷ mean)" in the Calibration Reliability tab per explicit
   user feedback that "spread" was ambiguous; internal variable names (`relSpread` etc.) left
   unchanged.
10. Visually separated the pipeline-total plots from the 4 individual phase plots in the
    Complexity tab (own bordered card, distinct from the phase grid) so the total can't be
    mistaken for a 5th phase.
11. Verified before republishing: JSON validity of both embedded data blocks, full JS syntax
    validity via `esprima.parseScript`, `getElementById` cross-check (including
    template-literal-constructed ids) against actual HTML ids, duplicate-id check, duplicate
    top-level declaration check — all clean.

**Known open items:**
- The 2-stage validation model is a genuine improvement to the *dashboard's post-hoc curve
  fitting*, not a change to CorrTrack's own validation algorithm or its correlation semantics —
  no scientific meaning of the method itself was touched.
- `candidate_search_lsh_candidates_touched` and gamma's per-trial stage-2 log remain uncaptured
  (unchanged from the prior entry).
- Cell 5's `candidate_time` remains an unresolved outlier even after the rerun (see above) —
  disclosed, not fixed.

**Follow-up same day:** user reported "the validation fit is not present in the Complexity tab."
Investigated at length (re-verified JSON/esprima/getElementById/duplicate-id/duplicate-declaration
checks, re-derived the 2-stage model's coefficients independently via `numpy.linalg.lstsq`,
confirmed via the user's own browser console that no script error was firing — the console output
was entirely browser-extension noise, not from the page). Root cause: not a bug — validation's fit
curve on the "vs m"/"vs L" mini-charts is real but renders compressed into the bottom ~28% of the
chart's height, because cell 5's own outlier (3.76s, still not fully explained even by the improved
model since its L=55 sits far from the median L the curve holds other factors at) sets the y-axis
ceiling under true-scale rendering. Presented the user 3 options (log-y scale, annotate the outlier,
or leave it true-scale with an explanatory note); user chose the third, consistent with their
earlier explicit rejection of "not to scale" workarounds. Added a short in-chart note to both
validation mini-charts saying so in words. No chart geometry changed.

**Second follow-up same day — a real overclaim caught and corrected:** user then reported the
validation plots and their m exponents "look exactly the same" as before the 2-stage model was
added. Checked numerically (independent `numpy.linalg.lstsq`): the 2-stage composed slope for
validation_time vs. m is IDENTICAL to the plain 4-factor slope to 14 decimal places (2.1331201995
either way), and the same holds vs. L, density, and threshold. This is not a coincidence — it's a
Frisch–Waugh–Lovell identity: when a mediator (`tested_candidates`) used in a second regression
stage is itself forecast by exactly the same predictor set as the direct fit, substituting that
forecast back in for extrapolation is algebraically forced to reproduce the direct fit's slope
exactly. In plain terms: the improved R² (0.79→0.93) is real and means the 83 *already-measured*
points are explained much better once `tested_candidates` is known for them — but it provides zero
new information for *forecasting* validation_time at an unmeasured m, since forecasting
`tested_candidates` itself needs the same 4 design factors already in the plain model. Corrected
the dashboard: `phaseFits`/`phaseFitsL`'s displayed curve/exponent for validation_time reverted to
the plain `fitWithModel` (unused `fitWithModelTwoStage` function deleted, dead code); the improved
R²/residual numbers are kept ONLY in the confidence table, now clearly labeled as a different,
better-fitting model than the one driving the curve; the phase-exponent-note footnote and the
Extrapolation tab's callout were rewritten to state this plainly, including that genuinely closing
the gap would need new instrumentation (logging which tolerance×occupancy the tuner picks as a
function of scale), not a cleverer regression on what's already logged. Re-verified integrity
(JSON/esprima/getElementById/duplicate-id/duplicate-declaration, plus confirmed zero remaining
references to the deleted function) before republishing. This is the honest, final answer to the
user's original "I want to improve the validation exponent" ask: it can't be improved from the
currently-captured data, and that's now stated outright rather than implied to have been fixed.

**Third follow-up same day — mechanistic decomposition, requested by the user:** user proposed
"validation depends on m², but as a small fraction of it because of CorrTrack's filtering."
Checked directly: `bf_tested` (the exact O(m²L) brute-force search space) is ground truth in the
data; `survival_fraction = tested_candidates / bf_tested` is the share of it that actually reaches
validation. Regressing `log(survival_fraction)` against m/L/density/threshold gives m^1.3 (R²=0.60,
m^1.2 excluding cell 5) — clearly positive, not flat. So the clean version of the hypothesis (a
roughly-constant small fraction) doesn't hold: CorrTrack's own LSH filtering gets proportionally
*less* selective as m grows, which is why `tested_candidates` (m^3.32) outgrows the exact m² space
it's drawn from. Separately, decomposing validation_time itself against tested_candidates directly
(the existing 5-factor model, R²=0.79→0.92) shows candidate count is the main driver
(tested_candidates^0.29) but a residual m^1.16 direct effect survives even after accounting for
candidate count — part of validation's cost isn't candidate count, it's m on its own (plausibly more
borderline candidates surviving the filter as m grows, the same mechanism as the survival-fraction
finding, seen from the validation side rather than the filtering side).
Added to the dashboard: (1) a new Filtering tab chart, "Does filtering stay equally selective as m
grows?", plotting survival_fraction vs. m (log-log, colored by density, with the existing trend-line
overlay), reporting both the full-data and cell-5-excluded exponents in the card note; (2) both
validation mini-charts in the Complexity tab (vs. m and vs. L) now state the
tested_candidates^0.29 × residual-m^1.16 decomposition directly, alongside the existing true-scale
squash note; (3) a new "Fitted complexity expressions" card in the Complexity tab writing out the
full closed-form fitted model (constant × m^a × L^b × density^c × e^(d·threshold), R² included) for
all 4 phases and the pipeline total, not just isolated exponents; (4) the Extrapolation tab's
validation callout rewritten to tie the decomposition and the survival-fraction finding together,
while keeping the (unchanged, still valid) Frisch–Waugh–Lovell conclusion that none of this
sharpens the forward forecast without new instrumentation. All numbers are computed live via the
dashboard's own `fitMulti`, not hardcoded, and cross-checked against an independent
`numpy.linalg.lstsq` run before publishing (m^1.156, tested_candidates^0.294, R²=0.9245 for the
validation decomposition; m^1.32/R²=0.599 full-data and m^1.17/R²=0.604 excl.-cell-5 for
survival_fraction). Re-verified integrity (JSON/esprima/getElementById/duplicate-id/duplicate-
declaration) before republishing.

**Fourth follow-up same day:** user asked to make m^1.16 (the decomposition's residual, candidate-
count-controlled coefficient) "the official" validation exponent everywhere, to fold the
decomposition into the complexity-expressions card, and for the complexity expressions to be
theoretical O() design complexities, not fitted ones (fitted ones kept alongside, per instruction).
Implemented the latter two directly: added a "theoretical (design) complexity" block per phase
(sketch O(m); candidate search O(m&middot;L), the LSH design target; validation O(tested_candidates),
definitional; monitor O(validated_candidates), definitional; brute force O(m²L), confirmed exact;
pipeline total &asymp; O(m&middot;L) if filtering holds near its design target) ahead of the existing
fitted-expression list, and added the tested_candidates^0.29 &middot; m^1.16 decomposition as its own
explicit line under validation's fitted expression.
Declined the first part and explained why rather than silently complying: m^1.16 is the effect of m
on validation_time WITH candidate count held fixed, not the total effect — since tested_candidates
itself grows sharply with m (m^3.32), most of validation's real growth runs through that channel, and
the plain m^2.13 fit already captures both paths correctly (proven identical to the composed
direct+indirect effect via the Frisch–Waugh–Lovell identity established in the prior follow-up).
Making m^1.16 the displayed/table/extrapolation exponent would silently understate validation's true
growth rate — a factual regression, not a refinement — so kept m^2.13 as the one exponent used
everywhere except the explicit decomposition line, with an inline explanation of the distinction so
it doesn't read as an unexplained inconsistency. Re-verified integrity and republished.

**Fifth follow-up same day:** user supplied their own prior theoretical complexity derivation for
CorrTrack — O(m&middot;K&middot;√w) [sketch] + O(m&middot;log(m&middot;L)) [candidate search] +
O(C&middot;w) [validation], K=hyperplane/sketch width, w=window size, C=tested_candidates — asking
for the dashboard's theoretical complexity section to use this instead of the invented one from the
previous round, alongside a live measured-vs-theory comparison; asked for the "2.13 = 1.16 + indirect
path via tested_candidates growing as m^3.32" claim to be shown didactically with visuals, not just
asserted in prose; and asked for the pipeline's m^1.56/L^0.41 to be pulled out of its title into
something more visible. Verified K (n_vectors_fixed=64) and w (window_size=256) are both constant
across all 83 points in this sweep, so each term collapses to a pure m,L power law for direct
comparison — implemented `bivariateLogFit` to regress each phase's actual time against its own
theoretical quantity (log-log): sketch ≈ theory (matches O(m) closely), candidate search costs more
than O(m·log(mL)) predicts as scale grows, validation costs LESS than O(C) predicts (sub-linear,
real batching benefit) even though C itself outgrows expectations — three different, honest verdicts,
not a uniform "theory holds" or "theory fails." Added a dedicated card with a tested_candidates-vs-m
chart (log-log, trend line) and a worked-arithmetic derivation box computing
direct + tested_candidates_coef × tested_candidates'_own_m_exponent = total, live from the same fit
objects everywhere else uses (shown to 3 decimals specifically in that box, not the dashboard's usual
2, since rounding each term to 2dp before summing drifts ~0.01 from the true total — would have
undermined exactly the demonstration meant to build confidence). Pulled the pipeline's m/L exponents
into two `.stat` tiles (the same component the header uses) instead of a title suffix. Re-verified
integrity (JSON/esprima/getElementById/duplicate-id/duplicate-declaration) and republished.

**Sixth follow-up same day — theory corrected against the actual current code, not the user's own
possibly-stale prior derivation:** user flagged that the O(m·K·√w) + O(m·log(mL)) + O(C·w) formula
they'd supplied was from an older B+tree-based design and said candidate search "completely
changed" since — asked for it to be updated first, before anything else. Read
`candidate_kernels.pyx`'s `SignLSHBandIndex` (the actual class behind this sweep's
`lsh_sign_dot`/`lsh_approx` backend) directly rather than trusting either the old formula or
inventing a new one: `_finalize_sizing` computes `band_width = ceil(log2(mL/target_occupancy))` and,
when `n_bands_tolerance>0` (the mode this sweep actually tunes), a closed-form LSH-banding minimum
`n_bands = ceil(tolerance · b_min)`, `b_min` from the standard OR-of-bands recall bound given
`p_bit = 1-acos(threshold)/π` and `p_r = p_bit^band_width`. Ported this formula into the dashboard's
JS (`theoreticalSizing`) and added a live self-check: it must reproduce the data's own
`n_bands_selected` exactly for all 83 points before being trusted for anything plotted (it does).
This gives a fundamentally different, code-grounded candidate-search complexity —
O(m·n_bands·band_width) with n_bands itself growing with m·L (not constant, as the old formula
implicitly assumed) — which fits candidate_time far better than the retired formula (slope 0.93,
R²=0.82, vs. the old formula's slope 1.58, R²=0.73): candidate search actually matches ITS OWN
current design closely once the real n_bands scaling is accounted for. Applied the same n_bands
grounding to tested_candidates (validation's driver): O(m·n_bands·target_occupancy), R²=0.615,
slope 1.87 — real candidates accumulate faster than the design's own per-bucket occupancy target
predicts, the same "filtering leaks more at scale" finding as the Filtering tab's survival-fraction
chart, now derived mechanistically instead of only regressed. Cross-referenced this from the
decomposition card. Re-verified integrity and republished.

**Seventh follow-up same day — restructure into 2 new tabs + a real bug fix:** user caught that the
candidate-search theory row's annotation ("[K,w fixed here → ...]") was nonsensical — that formula
has neither K nor w, the bracket was a copy-paste artifact from the sketch/validation rows' template.
Fixed: each phase now gets its own accurate collapse note (or none, for candidate search, which
explicitly explains it does NOT collapse). User also asked to explain the log term (why band_width's
log₂ is bit-sizing while n_bands's own growth, though built from that log, is a power law not a log —
"feeding a log into an exponent turns it back into a power law") — explained in chat, then asked for
that explanation, a new tab separating theory from the increasingly-crowded Complexity tab, a new tab
distinguishing swept/tuned/derived parameters (especially n_bands), and brief sublinear/linear/
super-linear/quadratic labels on the Complexity tab's exponent table. Implemented: split into a new
"Complexity Theory" tab (theoretical complexity per phase, a didactic card walking through band_width's
log₂ vs. b_min's natural log vs. why composing them gives a power law — with a worked table at the
dataset's own median threshold showing band_width climbing +1 per m·L doubling vs. n_bands climbing
×1.2-ish per doubling — the fitted expressions, and the validation decomposition derivation) and a new
"Parameter Hierarchy" tab (swept: m/L/density/threshold; tuned: n_bands_tolerance/target_occupancy/
gamma, searched per point; derived: band_width/n_bands, computed by closed-form formula, with the
formula written out and a predicted-vs-actual n_bands scatter as the verification). Added a Regime
column (sublinear/≈linear/super-linear/≈quadratic/super-quadratic, with fixed cutoffs) plus a short
per-phase theoretical-rationale note to the Complexity tab's own exponent table, so a quick read
doesn't require the deep-dive tab. Re-verified integrity (including tab-key ↔ data-panel consistency)
and republished.

**Eighth follow-up same day — simplify and visualize:** user pushed back hard: "too wordy, too
complicated, not nearly enough visuals," and specifically wanted the n_bands/band_width derivation
reframed as "just target_occupancy and corr_threshold, then add the tolerance term." Rewrote the
Parameter Hierarchy tab's "Derived" card around a new reusable `.flow`/`.flow-box` CSS component: a
literal left-to-right pipeline (m,L,occupancy &rarr; band_width &rarr; (+threshold) p_r &rarr;
(+target_recall) b_min &rarr; (&times;tolerance) n_bands), each box showing a real worked number
(dataset medians), making the "tolerance is just a final multiplier" point visually instead of in
prose. Replaced the Complexity Theory tab's text-heavy log walkthrough (a full table + 4 paragraphs)
with two small side-by-side charts (band_width vs. m·L flat, n_bands vs. m·L curving) sharing one
axis, plus a single one-line caption — added a `connectLine` option to the shared `drawScatter`
helper (draws a literal polyline through already-computed/deterministic points, distinct from the
existing regression-fit `trend` option, which would have drawn a misleading straight line through
genuinely curved data). Cut card-note prose by roughly 60-80% throughout the Complexity, Complexity
Theory, and Parameter Hierarchy tabs — most multi-sentence explanations reduced to a single clause,
with pointers to the Theory tab for anyone who wants the full derivation instead of repeating it
inline. Re-verified integrity and republished.

**Ninth follow-up same day — traced the residual m^1.16 effect into the actual code, and corrected an
unverified claim:** user asked where the direct m^1.16 effect (validation's residual, not mediated by
candidate count) actually comes from, after confirming that a constant-cost-per-candidate model would
make validation's exponent equal tested_candidates' own (m^3.32) exactly. Traced the real call path
for this sweep's config (`run()` -> `_get_validated_corr` -> `_get_validated_corr_numeric` ->
`_cy_validate_corr_rows`) line by line: `_current_window_cache_kwargs()` returns `{}` immediately
because `validation_current_window_cache` defaults to False and is never overridden by this sweep's
scripts, so the per-series raw-moments cache (`_recompute_current_window_raw_moments`, an O(m) op)
never runs; `_get_validation_window_data()`'s `np.ascontiguousarray` call is a no-op since
`self.window_data` is already rebuilt fresh and contiguous every step via `np.append` inside
`_update_curr_data` (which itself runs BEFORE any of the 4 phase timers start, so its own cost, if
any, isn't attributed to validation_time at all). Both of the two most plausible mechanisms are
therefore confirmed INACTIVE for this exact sweep -- meaning the dashboard's earlier text ("plausibly
more borderline candidates surviving the filter as m grows") was an unverified guess that the deeper
investigation does not support. Corrected: removed that claim everywhere it appeared (both validation
mini-chart squash notes, the Extrapolation tab callout, the Complexity Theory tab's derivation card),
replaced with an honest "not yet traced to a specific mechanism, two most likely causes checked and
ruled out" disclosure. Also added, since it WAS confirmed: `validate_corr_rows` batches the whole
candidate set into one bulk `nogil` call, so its fixed per-call overhead amortizes across candidate
count -- this is the real, verified reason the tested_candidates exponent (0.29) is well below 1, and
is now stated explicitly next to the derivation box instead of left implicit. Re-verified integrity
and republished. This is the second time this session a plausible-sounding but unverified explanation
was caught and corrected (see also the earlier 2-stage-model/FWL-identity episode) -- both by actually
reading the implementation rather than accepting an intuitive story.

**Tenth follow-up same day — supervisor's pairs=m²L request, then corrected to additive per
feedback:** user relayed a supervisor suggestion to fit scaling against pairs=m²L (what brute
force's own cost is exactly a function of) instead of m and L separately. Scoped with the user
first (headline scaling + speedup only, not the per-phase mechanistic/n_bands material) and
implemented it as a REPLACEMENT of the existing m-vs-L breakdown throughout the Complexity and
Extrapolation tabs. User then said "Stop, I did not like it. Roll-back," followed immediately by
"add the versions of the plots with pairs in the x-axis, but do not lose the rest, otherwise I do
not have quadratic looking curves anymore" -- i.e. the ask was always additive, not a replacement;
losing the m-based brute-force-vs-m view specifically lost the recognizable ~m² curve. Reverted
every replacement edit by hand (no git/backup existed for this scratch file, so each of the ~19
edits was undone by reapplying it in reverse, in reverse order, verified against the pre-change
tool-call record) back to the exact original m/L-separate state, then re-added pairs=m²L as a
THIRD, additive view: a new "vs. pairs" mini-chart grid alongside the existing m and L grids, a
3rd panel on the pipeline/total-vs-brute-force/speedup cards (grid-2 -> grid-3), and 2 new columns
on the exponent table -- nothing existing removed or renamed. Verified the rollback was clean
(grepped for zero remaining stray pairs-only variable names before adding anything back) and the
new additive code numerically consistent (bf_runtime ~ pairs^1.02, R²=0.988, cross-checked against
an independent numpy fit) before republishing. The companion OA dashboard was mid-edit for the
same (later reverted) replacement approach when the "stop" arrived -- its 2 edits were reverted
immediately and nothing was ever published for OA, so no republish was needed there; OA's own
additive pairs=m²L view was not built (not asked for a second time after the correction landed on
Sobol) -- flagged to the user as a pending option, not assumed.

**Eleventh follow-up same day — a new, separate highlight-reel dashboard:** user asked for "another
dashboard derived of the sobol one, but much cleaner and simpler, with a clear storytelling and
that highlights the contributions of my method" — i.e. a companion narrative page, not another
edit to the technical dashboard. Built as a standalone single-scroll page (`corrtrack_story.html`,
published separately, not overwriting the Sobol dashboard's URL): a hero with 3 headline stats
(68x max measured speedup, 99.99% of the search space avoided, 96% median recall), then 5 short
numbered sections (the brute-force cost problem, CorrTrack's shallower curve, correctness
preserved via a 3-stage funnel, an extrapolation-to-10,000-series speedup curve explicitly marked
measured-vs-projected, and a closing rigor summary) — a fraction of the original's 14 tabs and
prose density, by design. Loaded the `artifact-design` skill first per its requirement; picked a
distinctive palette (deep charcoal-green/amber duo, not the muted dashboard's blue-grey) and
caught one real accessibility issue before publishing: the first color draft paired a rust-red
"brute force" against a green "CorrTrack," which sits squarely on the red-green colorblindness
axis -- switched to a slate-blue/amber pair (a safe categorical opponent pair) and made the
funnel's middle stage a `color-mix()` blend between them instead of introducing a third hue.
Every number on the page (curve exponents, funnel counts, recall stats, the 9.4B-pairs figure)
was recomputed directly from `sobol_data_full.json` using the same model-fitting approach as the
technical dashboard and spot-checked before writing copy — caught and fixed one real numeric slip
in an early draft (used the sweep's MEAN bf_tested count where the copy actually meant the
LARGEST single run's count, an 8x difference: 1.2B vs 9.4B). Verified JSON/esprima/id-integrity
before publishing (no capabilities declared -- static page, no persistence needed).

**Twelfth follow-up same day — 5 tweaks to the highlight-reel page:** (1) section 1's stat changed
from the bare m-exponent to `O(m²L)`, with both the m^2.03 and L^1.03 measured exponents shown as
confirmation (verified L^1.03 via the same 4-factor fit used throughout the session); (2) section
2's contribution chart gained a toggle between m and pairs=m²L axes -- both independently fit
(pairs: wall^0.67, bf^1.02), not one rescaled from the other; (3) section 3 now explains the 12%
of runs (10/83, exact) that missed the 0.90 recall floor -- verified their median correlation
density is ~4x lower than the sweep overall, and the true minimum recall observed is 79% (cell 14,
m=70/L=18/density=0.0054) -- plus a sentence confirming positive AND negative correlation
detection, verified against `corr_sign="both"`/`neg_corr=True` in the sweep's own config; (4) an
m=1,000 milestone dot added to the extrapolation curve; (5) a new "Under the hood" section added
after the rigor section -- a 4-box pipeline flow (sketch/candidate search/validation/monitor), a
short parameter glossary (m, L, corr_threshold, target_occupancy, n_bands/tolerance, gamma), and
the per-phase Big-O complexity (reusing the theoretical derivations already established and
verified against `candidate_kernels.pyx`/`library_corrtrack_parallel.py` earlier this session).
Re-verified JSON/esprima/id-integrity and republished to the same artifact URL.

**Thirteenth follow-up same day — 4 corrections to the highlight-reel page, one requiring real
derivation:** (1) monitor's description was wrong -- checked `_monitor_corr_buffered_numeric` and
`monitor_kernels.pyx`'s `NumericMonitorState` directly rather than guessing again: status rows
store `[s1,s2,lag,t1,t2,length,sign]` (length=duration, sign=current sign) and anomaly rows mark
entry (+1), sign-change (0/1 vs last sign), and exit (-1) events -- corrected the copy to describe
duration/sign-change/entry-exit tracking for anomaly detection, matching the user's own correction
exactly; (2) "n_bands, tolerance" renamed to the single parameter `n_bands_tolerance`; (3) sketch's
complexity restored to the full O(m·K·√w) (had been collapsed to O(m) for this sweep's fixed K,w,
but this section is about the general method, not one sweep's special case); (4) the harder ask --
candidate search's O(m·n_bands·band_width) needs n_bands/band_width, which aren't known until the
method runs, so it's useless for a user estimating cost in advance. Derived a fully user-computable
form by substituting the closed-form n_bands/band_width sizing (already established earlier this
session) into the complexity and simplifying asymptotically: O(m^(1+γ)·L^γ·log(mL)), γ =
-log₂(1-acos(corr_threshold)/π) -- a single constant fixed by the threshold alone. Verified against
real data before writing it: this form correlates with measured candidate_time at R²=0.90 (better
than the earlier "exact" n_bands-based check's R²=0.82, and MUCH better once target_occupancy was
tried in the exponent too and made the fit worse, R²=0.78 -- occupancy is tuner-selected per point,
not exogenous, so naively including it introduces noise rather than removing it; left out of the
final formula for exactly that reason, not simplicity alone). Applied the same substitution to
validation (O(m^(1+γ)·L^γ·w), R²=0.64 against real validation_time -- rougher, disclosed as such,
consistent with validation being the noisiest phase throughout this whole session's investigation).
Also fixed the rigor section's compute-time stat from ~7h (a lower bound excluding calibration
search overhead) to the true ~14h wall-clock total, per the user catching that the smaller number
was being presented as if it were the whole cost. Re-verified integrity and republished.

**Fourteenth follow-up same day:** two more additions to the highlight-reel page's complexity
section: (1) candidate search and validation now show BOTH forms stacked -- the operational
O(m·n_bands·band_width) / O(C·w) first, then the derived m/L/threshold-only approximation
underneath with an "≈" and a note on what was substituted in, rather than only showing the
derived form as in the previous round; (2) a small live calculator -- a corr_threshold slider
(0.50-0.99) that recomputes γ = -log₂(1-acos(threshold)/π) and the resulting candidate-search
exponents (1+γ on m, γ on L) in real time, using the exact same formula as the static examples
above it, so a reader can check their own threshold instead of only trusting the two worked
numbers. Verified the default slider position (0.83) reproduces the page's own static "γ≈0.30"
example exactly before publishing. Re-verified integrity and republished.

**Fifteenth follow-up same day — corrected a real conflation between touched and validated candidate
counts:** user caught that "validation spends touched × w" was wrong, since touched candidates pass
through the dot-gamma gate FIRST and only the survivors (C = tested_candidates) reach validation --
touched and C are not the same quantity. Updated the formula box following the user's own requested
pattern (operational form first, then the m/L-derived equivalent below it) for both phases:
candidate search now shows BOTH its terms -- O(m·n_bands·band_width) [build] + O(touched·K)
[search/gate] -- with the derived m^(1+γ)·L^γ forms under each, explicitly flagging that the search
term alone (R²=0.89) explains real candidate_time better than the build term (R²=0.82); validation's
operational form now says C is what SURVIVES the gate, not touched itself, and its derived
m^(1+γ)·L^γ·w form is now honestly framed as fit directly against C (R²=0.64), not derived through
touched via an unverified pass-rate step. Updated the calculator's shared-shape line and caption to
stop implying the log(mL) factor applies to both terms (it's build-specific; the search term's own
secondary factor is occupancy^(1-γ)·K instead). Re-verified integrity and republished.

**Sixteenth follow-up same day:** user caught that the calculator's "shared shape: m^1.30 · L^0.30"
line was itself misleading -- true for m and L (both terms really do carry the same m^(1+γ)·L^γ
factor), but it silently dropped the search term's own occupancy^(1-γ) factor, which the build term
doesn't have at all. Rebuilt the calculator with the two terms shown separately and a second slider
for target_occupancy (2-10, matching the sweep's own grid) so the occupancy factor is live and
computed (e.g. occupancy^0.70 = 2.16× at occupancy=3), not just named in a caveat sentence.
Cross-checked the default slider positions (threshold=0.83, occupancy=3) reproduce the page's own
pre-filled numbers (γ=0.30, 2.16×) exactly before publishing. Re-verified integrity and republished.

**Seventeenth follow-up same day — more visuals + a real table for the highlight-reel page's
"how it works" section:** user asked for section 6 (`corrtrack_story.html`) to include more
visuals and for "Complexity, per phase" to become a table instead of a stack of divs. Converted
the formula-box into an actual `<table>` (phase / operational cost / m,L,γ-derived form, notes
nested under each formula, brute-force row visually separated with the bruteforce-soft
background). Added two new visuals grounded in real sweep numbers, not decorative: (1) a
measured mean phase-time-share stacked bar (sketch 22.4% / candidate search 75.0% / validation
2.4% / monitor 0.2%, computed directly from the 83-point CSV's own timing columns) placed right
after the table, making "the dot-gate dominates real cost" concrete instead of only a claim in a
footnote; (2) a live three-bar visual inside the γ calculator showing the three exponents
(1+γ, γ, 1−γ) as proportional bar lengths on a shared scale, updating with the existing sliders.
Verified the phase-time-share numbers against the CSV directly before hardcoding them (they're
a fixed, dataset-wide mean, not something that needs live recomputation in-page). Re-verified
JS syntax + id cross-checks and republished.

**Eighteenth follow-up same day — added the false-positive-vs-true-signal decomposition to the
Filtering tab:** in discussion, established precisely why `tested_candidates` outgrows the exact
`m²L` space it's drawn from: `validated_candidates` (true positives) stays ~quadratic (m^2.06,
R²=0.994, essentially bf_tested's own exponent, since it's just "how many real correlations
exist"), while `tested_candidates − validated_candidates` (false positives reaching validation
and failing it) grows even faster than the whole, m^3.70 (R²=0.65) — the entire super-quadratic
effect is chaff, not signal. Also found the per-cell tuner's own selected `n_bands_tolerance`
drifts up with m (m^0.34, R²=0.59) and `gamma_selected` drifts slightly looser, consistent with
the tuner widening its net to hold the same target recall against a bigger background as m grows.
Added a callout under the existing survival-fraction chart with these numbers plus a heavily-
caveated order-of-magnitude "if this same power law held forever" parity-crossover estimate
(~m in the low millions at median L/density/threshold, R²=0.68, thousands of times past the
measured range — stated explicitly as not a forecast). Re-verified and republished.

**Nineteenth follow-up same day — investigated whether that tolerance/gamma climb is a genuine
scale effect or a calibration-reliability artifact (user's explicit ask):** went to the actual
calibration code (`experiment_lsh_nbands_occupancy_sweep.py`'s `calibrate_nbands_occupancy` /
`experiment_lsh_cost_sweep.py`'s `calibrate_gamma`, confirmed via import trace to be exactly what
produced the Sobol sweep's data) and the persisted 996-trial stage-1 grid
(`sobol_trials.json`), not just the 83 final selections. Key findings, all directly computed
against the trial data:
- Stage-1 trials run with `candidate_cosine_threshold=0.0` (gate wide open) over the FULL
  `n_steps` (not an abbreviated calibration run) — their `tested_candidates` is therefore the
  first real, directly-measured stand-in for "touched" this project has (never logged
  otherwise). Sliced at a FIXED tolerance (1.0 or 2.0, occupancy=3.0), it fits m^1.38·L^0.36
  (R²=0.97) — clean, sub-quadratic, matching the theoretical m^(1+γ)·L^γ prediction almost
  exactly. Retrieval itself is not the problem.
- At that same fixed tolerance=1.0, mean achievable recall is statistically flat across m
  (88.2% for m≤median vs. 88.7% for m>median) but its standard deviation collapses 3x (0.251 →
  0.088) as m grows — a direct law-of-large-numbers signature (corr(log(validated_candidates),
  |recall−mean|) = −0.51, verified, not asserted). validated_candidates (the recall estimate's
  effective sample size) averages ~217 in the low-m half vs. ~1,441 in the high-m half.
- Consequence: tolerance=1.0's pass rate against the 0.95 target is 55% for m≤median but only
  12% for m>median — not because larger m is structurally harder, but because larger m (more
  true correlations to average recall over) measures the SAME ~88% underlying rate more
  reliably, correctly failing a setting that was never actually sufficient. gamma calibration
  shows the identical pattern (gap-from-tightest-gamma correlates about equally with m and with
  log(validated_candidates), 0.35 vs. 0.40; met-target rate 69%→44%; tightest-gamma-selected
  rate 19%→5%).
- Decomposed real (post-gate) tested_candidates at each cell's own selected setting into
  touched-at-that-setting (m^2.20, R²=0.945) × gate-pass-rate (m^1.12, R²=0.42) — these multiply
  out exactly to the already-established m^3.32, and both factors carry the same
  reliability-not-difficulty signature.
- Verdict written into the dashboard, with appropriate hedging: mostly a calibration-reliability
  artifact, not a genuine "harder at scale" law — predicts the tolerance/gamma-driven share of
  the growth should plateau once past the small-sample regime, but this is a testable
  prediction from a dataset that only reaches m=296, not a proven asymptote. Recommended
  follow-up (not run): a few cells at much larger m with density held fixed, to check whether
  n_bands_tolerance_selected/gamma_selected actually flatten.
Added a new Filtering-tab card (two charts + a callout) presenting all of the above, computed
live in-page from the same embedded TRIALS/DATA the rest of the dashboard already uses (not
hardcoded), and softened the previous callout's "real cost of scale" framing to point to this
refined finding. Re-verified JS syntax + id cross-checks and republished.

## 2026-09-03 — Overlap-corrected LSH band sizing (real production code change, not dashboard-only)

**Branch:** main. **Files changed:** `candidate_kernels.pyx` (rebuilt), `library_corrtrack_parallel.py`,
`experiment_run_param_grid.py`.

**Context:** continuing the 2026-09-01 entry's calibration-reliability investigation, the user
asked directly: since achievable recall at a fixed tolerance is flat with m but stuck ~7 points
below target even in reliable (large-sample) measurements, can `b_min` itself be corrected —
holding `n_vectors` fixed, not scaling it — so the *stable* mean recall actually reaches target,
rather than just being measured more reliably?

**Derivation.** `SignLSHBandIndex` builds each band from an independently-drawn random
`band_width`-bit subset of the SAME fixed pool of `n_vectors` sign bits (confirmed directly in
the class's own docstring: "drawn without replacement WITHIN a band but WITH replacement —
deliberate overlap — ACROSS bands"). The old formula (`1-(1-p_bit^band_width)^n_bands`) treats
the `n_bands` band-successes as independent; they aren't, since they share bits. Exact fix:
condition on `t` = how many of the `K=n_vectors` bits actually match (`t ~ Binomial(K, p_bit)`).
*Given* `t`, each band succeeds with probability `q(t) = C(t,w)/C(K,w)` (a random `w`-subset
landing entirely inside the `t` matching bits), and — conditional on `t` — the bands' successes
really are i.i.d., so `P(recall) = Σ_t Binomial(K,p_bit,t) · [1-(1-q(t))^n_bands]`. The old
formula is recovered as `E_t[q(t)]` for a SINGLE band (unbiased there), but `1-(1-x)^n` is
concave, so by Jensen's inequality the old formula is a *proven* upper bound on true recall
whenever bands overlap — not a coincidental overestimate.

**Validation against real data (83-cell Sobol sweep, 996-trial calibration grid, restricted to
660 trials with validated_candidates≥100 so the recall measurement itself is trustworthy):**
recomputed both formulas at each trial's own real `n_bands` (no fitting/tuning involved, just
evaluating both formulas at what was actually used) and compared to real measured recall:

| | mean predicted | gap vs. actual (91.0%) | R² |
|---|---|---|---|
| old (independent-bands) | 98.0% | +7.0 pts | −1.08 (worse than the mean) |
| corrected (overlap-aware) | 90.7% | −0.3 pts | 0.10 |

A second check confirmed real discriminative power, not just an unbiased mean: trials whose
actual `n_bands` already met-or-exceeded the corrected requirement averaged 94.0% recall
(pass-rate 43%) vs. 90.2% (pass-rate 21%) for those that didn't. Solving the exact formula for
the true minimum `n_bands` (holding `n_vectors`=64 fixed, per the user's explicit request not to
scale it) needs **1.25×–4.44× the old formula's `b_min` (median 2.0×, mean 2.05×)** at a fixed
representative occupancy — explains why 33/83 real cells hit the old *tried* tolerance grid's
ceiling (2.0×) and 15 of those still failed to reach target recall: for some configs the true
requirement exceeds what that grid ever tried.

**Self-caught bug, disclosed rather than silently fixed:** the first pass at this "implied
multiplier" statistic used a scratch Python search with a doubling step that, past n=64, jumps
65→130→260→520→... and never refines back down — it can only ever return one of those specific
values, overshooting the true minimum whenever it doesn't land exactly on one. This inflated the
first-reported multiplier to 2.35×/2.60× (mean) and 8.32× (max). Caught by cross-checking the
scratch script's numbers against the actual compiled Cython implementation once it existed
(`debug_exact_band_recall`/`debug_corrected_b_min`, added as test-only wrappers) — the compiled
binary-search version and a corrected Python reference agree exactly; the numbers above are from
the corrected search. The core bias-validation table above was unaffected (it evaluates the
formula at real recorded `n_bands`, never at a searched value).

**Implementation (`candidate_kernels.pyx`):** added `_log_binom` (log-domain binomial
coefficients via `lgamma`, avoiding integer overflow — `C(64,32)` alone is ~1.83×10^18, uncomfortably
close to int64's ceiling, and this generalizes to any `n_vectors`), `_exact_band_recall`,
`_band_recall_ceiling` (the `n_bands→∞` limit, `P(T≥w)` — flags a `(K,w,p_bit)` combination as
structurally unreachable, distinct from the old formula's degenerate `p_r==0` case), and
`_corrected_b_min` (doubling search + binary-search refinement to the exact minimum; runs once
per index in `_finalize_sizing`, never per query, so its cost — a few dozen to a few hundred
`O(n_vectors)` evaluations — is irrelevant to steady-state throughput). `_finalize_sizing`'s old
`p_r`/`log_denom`/`b_min_f` computation was replaced with a direct call to `_corrected_b_min`;
`n_bands_tolerance` keeps working exactly as before as a multiplier on top. Added
`debug_exact_band_recall`/`debug_band_recall_ceiling`/`debug_corrected_b_min` as thin
Python-callable wrappers (test/debug utilities, not part of the public API) so this validation is
reproducible against the actual compiled artifact, not just the `.pyx` source. Rebuilt via
`python3 setup_cython.py build_ext --inplace` (clean build, only pre-existing unrelated warnings).
Cross-checked `SignLSHBandIndex.band_width`/`.n_bands` against an independent Python
reimplementation for 4 hand-picked (m, L, occupancy, threshold) cases — `band_width` matched
exactly in all 4 (unaffected by this change); `n_bands` matched exactly once the reference
script's own search bug above was fixed.

**Default-behavior change (explicitly requested, not a silent one):** the user's stated goal was
retiring `n_bands_tolerance` as something needing per-run tuning, then explicitly asked for it to
"receive a None or auto option so it is computed automatically." Changed `CorrTrack.__init__` and
`Candidates.__init__` (`library_corrtrack_parallel.py`) so `candidate_lsh_n_bands_tolerance=None`
(the existing default, i.e. what every caller gets when it isn't set) now means **AUTO**
(tolerance=1.0 against the corrected formula) instead of **disabled** (the old default, which
left `candidate_lsh_n_bands` at its literal fixed value, 64). Passing `0` or a negative value
explicitly still opts out and falls back to the literal value — that path is kept, not removed.
This is a real, project-wide default change (affects every existing `CorrTrack(...)` call that
doesn't set this parameter), not scoped narrowly — full test suite re-run and confirmed green
(113/113) both immediately after the `_finalize_sizing` fix alone and again after this default
flip, so no test in `test_stable_reproduced_changes.py` was asserting the old flat-64 behavior.
`experiment_run_param_grid.py`'s `PARAM_GRID` no longer sweeps
`candidate_lsh_n_bands_tolerance` over `{1.0, 1.5, 2.0}` (the key is omitted entirely, falling
through to the new default) — nothing left to tune there. `experiment_run_exec_param.py` and
`corrtrack_run_corrtrack.py` were deliberately left untouched: neither ever set this parameter
(unlike its sibling `CANDIDATE_LSH_TARGET_OCCUPANCY`, which both files do expose), so they were
already implicitly getting whatever `CorrTrack`'s own default was — they now get AUTO for free,
with no new constant or CLI flag added, which is the more literal reading of "minimize what the
user needs to set" than adding a new (even if `None`-defaulted) knob would have been.

**Resolved — the `n_steps=60` smoke test discrepancy was a too-short-run artifact, confirmed, not
a formula problem.** The initial ad hoc smoke test (m=150, L=32, corr_prop=0.05, threshold=0.75,
`n_steps=60`) measured structural recall (gamma=0) at only ~61-62% across tolerance=1.0-2.0,
against a corrected-formula prediction of ~95-99% — a much larger gap than anywhere in the broad
Sobol-trial validation. Rerunning the identical config at `n_steps=300` (still short of the real
sweep's 2000, but 5x longer) gave:

```
tolerance=1.0 occ=2.0  n_bands=211  recall=0.9492
tolerance=1.0 occ=3.0  n_bands=136  recall=0.9510   <- AUTO default (tolerance=1.0, occ=3.0)
tolerance=1.0 occ=5.0  n_bands=88   recall=0.9516
tolerance=1.0 occ=10.0 n_bands=58   recall=0.9467
```

`structural_met_target=True` at tolerance=1.0 — the new AUTO default. **95.10% measured against
95.05% predicted by the corrected formula at n_bands=136** — matching to within 0.05 points, and
`recall` clears 0.95 (or comes within noise of it) at every occupancy tried, at tolerance=1.0
alone, with nothing swept. The `n_steps=60` result is confirmed as an artifact of the LSH index
not having reached its assumed steady-state population size that quickly, not a flaw in the
correction. **The corrected-formula default change is now confirmed end-to-end, not just against
historical trial data.**

## Next exact step
The blocking end-to-end confirmation above is now done. Remaining, in priority order:
1. A handful of large-m cells at fixed (high) density, to test the earlier follow-up's
   prediction that n_bands_tolerance_selected/gamma_selected plateau once validated_candidates
   is comfortably large, rather than keep climbing — now partially superseded by the corrected
   formula itself (which no longer needs per-cell tolerance search at all), but still relevant
   for characterizing the corrected formula's own m/L scaling (`m^0.50·L^0.51`, R²=0.97,
   measured at fixed occupancy=3.0) against real data at larger scale.
2. If the `candidate_search_lsh_candidates_touched` gap is ever worth closing directly (now
   partially superseded by the trial-grid-based proxy used in this and the prior entry, which
   already gives real touched numbers at gamma=0 for every existing sweep point): a one-line
   instrumentation addition to both sweep scripts' `run_one_cell`/equivalent, plus a rerun.
3. Consider whether `experiment_lsh_nbands_occupancy_sweep.py`'s own
   `N_BANDS_TOLERANCE_GRID = (1.0, 1.5, 2.0)` should also be retired/simplified now that the
   underlying formula no longer needs a tolerance hedge — not changed in this pass since the user
   scoped this round to `experiment_run_param_grid.py` and `experiment_run_exec_param.py`
   specifically.

## 2026-09-04 (a) — Data-size-aware proxy pair-row budget (correcting a gap in this log)

**Note first:** this change was made and reported to the user in conversation, but this entry
was never actually written at the time — a real handoff gap, not a backdated fabrication of new
work. Recording it now for the sake of an accurate history.

**Problem:** `OPTIM_PROXY_MAX_PAIR_ROWS` (`experiment_run_exec_param.py`) was a single flat
constant (50,000) capping the total proxy-anchor reference size regardless of `(m, L)`. At
m=150, L=32, one anchor's own local pair universe is ~709K rows — the flat default couldn't
even afford one full anchor, silently truncating the proxy hyperopt down to effectively 1
anchor (observed directly: `proxy_n_gt=1`, `proxy_warning="proxy pair row cap reached; anchors
were truncated"`) — a severe, silent loss of statistical power.

**Fix:** `library_corrtrack_parallel.py` gained `estimate_proxy_pairs_per_anchor(n_series,
n_lags, window_step)` (extracted from `_prepare_proxy_anchor_reference`'s own inline formula,
which now calls it too, so the two can't drift) and `recommend_proxy_pair_row_budget(n_series,
n_lags, window_step, max_anchor_count, hard_ceiling)`, which computes `min(max_anchor_count ×
est_pairs_per_anchor, hard_ceiling)` — sized for the *adaptive* anchor ceiling, not just the
initial request, bounded by a real, user-set resource cap
(`OPTIM_PROXY_PAIR_ROW_HARD_CEILING = 20_000_000`, new constant in
`experiment_run_exec_param.py`). `corrtrack_param_search.py`'s `main()` now computes this
per-dataset (inside the real `for n_var in N_VARS` loop, where the real `m` is known) instead of
resolving a flat constant, and prints when the ceiling — not the anchor target — is the binding
constraint, rather than only leaving a truncation note in the CSV. Verified: small `m` uses the
full requested anchor ceiling; m=150 goes from ~1 affordable anchor to 28; a deliberately huge
`m=2000` degrades gracefully to 1 anchor rather than blowing up. `OPTIM_PROXY_MAX_PAIR_ROWS`/
`OPTIM_PROXY_DISTANCE_CACHE_MAX_ROWS` were fully removed from `experiment_run_exec_param.py` and
their now-dead `DEFAULT_*`/dict entries removed from `corrtrack_param_search.py` (not just
overridden) — the user explicitly asked to minimize what needs to be set, and this method's
class-level literal fallback (250000) is a safe default for the rare direct-construction caller
that bypasses the per-dataset computation. Full test suite green (113/113) throughout.

## 2026-09-04 (b) — Hyperopt selection-rule redesign (4 changes, all discussed and approved
before implementation)

**Context:** discussed the actual proxy-anchor selection logic (`_apply_proxy_anchor_selection`)
directly with the user, who explained real history behind the current design (candidate-rate
minimization was prioritized because validation used to be the dominant cost phase, before
vectorization — no longer true; sketch=22.4%/candidate=75.0%/validation=2.4%/monitor=0.2% per
this session's own m=150 measurement). Four concrete changes, agreed on before any code was
touched:

**1. `n_bands` replaces `candidate_rate` as the primary ranking signal.** Grounded in real data
from this session's own gamma=0.6, 4-occupancy comparison: `candidate_rate_mean` varied only in
the 5th significant digit (0.000029–0.000030) across a 3.6× real structural-cost swing
(`n_bands` 211→58 as occupancy went 2→10) — too weak a signal to trust as primary, and it's the
literal reason the ranking got lucky rather than reliably correct before (it only worked because
all four candidates happened to fall within the 5% tolerance, routing the decision to the
real-time tie-break by chance, not by design). `n_bands`, by contrast, is exact, requires no
measurement at all (a pure function of config, computed via the new `compute_lsh_sizing`), and
for a *fixed* occupancy is provably the right thing to minimize (both the insertion term and the
dot-gate/search term found earlier this session scale with it). Real measured time
(`proxy_search_time_total`, already collected regardless) is the tie-break for the remaining
cost variation `n_bands` alone doesn't capture (occupancy's own effect on touched-candidate
volume) — promoted from "only used within a 5% candidate-rate window" to "always the tie-break
whenever `n_bands` ties."

**Prerequisite bug fixed first:** `record["candidate_lsh_n_bands"]` in
`_run_corrtrack_proxy_anchor` was only ever the CorrTrack constructor's literal default (e.g.
64) echoed back — never the real post-sizing value, which lives inside the Cython
`SignLSHBandIndex` this proxy path never builds directly (it constructs a fresh, throwaway
`CorrTrack` *per anchor* instead — see `_proxy_candidate_keys_for_anchor` — so there's no
long-lived index object a caller could read the real value back from). Fixed by computing
`n_bands` analytically via the new `candidate_kernels` functions (below) right where `corrtrack`
is built, using `corrtrack.n_lagged_windows`/`corrtrack.candidate_lsh_target_occupancy`/
`corrtrack.candidate_lsh_n_bands_tolerance`/`corrtrack.target_recall` (each already correctly
resolved on the constructed object, confirmed directly) — zero cost, no data touched, exact.
Also fixed: `candidate_lsh_target_occupancy` and `candidate_lsh_n_bands_tolerance` are now
logged into the record at all (previously absent from the CSV entirely).

**2. Data-driven recall safety margin, calibrated once into the formula, not chased at
selection time.** The user's own point: selecting on "highest observed recall" at hyperopt time
just picks whichever noisy proxy trial happened to read favorably, and costs far more than
necessary. Instead: computed the actual residual distribution (`actual − predicted` recall) from
the 660 reliable Sobol trials — mean +0.003 (confirms unbiasedness), std 0.061, with a real
percentile table (75th percentile: −0.021; 80th: −0.028; 85th: −0.042; 90th: −0.055, already
invalid as an additive pad since `target_recall + 0.055` exceeds 1.0 for `target_recall=0.95`).
Chose the 80th-percentile pad (`RECALL_SAFETY_MARGIN = 0.028`, new module constant in
`candidate_kernels.pyx`) — pushes the *internal* target the sizing search solves for to 0.978
(comfortably under the ceiling, unlike 90%+), roughly doubling real clearing confidence from the
zero-margin ~50% baseline to ~80%. Capped at `RECALL_SAFETY_MARGIN_CAP = 0.999` so the padded
target can never reach/exceed 1.0 (an invalid, unreachable probability) regardless of
`target_recall`. Explicitly disclosed: this was calibrated specifically at `target_recall=0.95`
(the only value used throughout the Sobol sweep this analysis is based on); applying the same
additive pad at very different `target_recall` values is a reasonable default, not independently
re-validated there. Verified the ordering this change must preserve still holds: at m=150,
occupancy 2/3/5/10 now needs 354/219/137/88 bands respectively (up from 211/136/88/58
pre-margin) — occupancy=10 still cheapest, consistent with change 1's own ranking logic.

**3a. Continuous statistical-power tie-break (bootstrap CI width), binary `gt_support<30`
threshold kept only as a sanity floor.** Added `proxy_recall_ub − proxy_recall_lb` as an early
tie-break (narrower preferred) — already computed for every trial, so this costs nothing extra
and uses information the ranking previously discarded once the binary underpowered flag was
resolved. Two configs both just above the 30-event floor and one with 10x more support were
previously ranked identically on power; now the narrower, more trustworthy one wins ties.

**3b. Anchor-expansion bounded by the real resource ceiling, not an arbitrary round count.**
`OPTIM_PROXY_ANCHOR_EXPAND_MAX_ROUNDS` (fixed at 1, chosen defensively against memory/cost
blowup) is replaced by a direct check against `proxy_max_pair_rows` (itself sized by
2026-09-04(a)'s fix) — keep expanding while underpowered **and** the next anchor count's own
pair-row need still fits the budget. A new, generously large
`_MAX_ANCHOR_EXPANSION_SAFETY_ROUNDS = 20` constant remains purely as a defensive backstop
against a genuinely infinite loop (e.g. a future arithmetic bug), not as the real gate — it
should never bind in normal operation. Now prints explicitly when the *ceiling* (not the round
count) is what stopped expansion, distinguishing "gave up arbitrarily" from "hit a real,
disclosed resource limit."

**4. The 5% candidate-rate closeness tolerance's role is gone, not just re-tuned.** A direct
consequence of change 1: `n_bands` is an exact integer, not a noisy measurement, so "how close
counts as tied" no longer needs an arbitrary percentage — two configs either land on the same
`n_bands` (real tie → falls to time) or they genuinely differ (no tie, no tolerance needed).
`proxy_candidate_rate_close_tolerance`/`OPTIM_PROXY_CANDIDATE_RATE_CLOSE_TOLERANCE` are left in
place (harmless, still accepted, silently unused by the new ranking) rather than a further
cross-file removal not explicitly scoped this round — flagged as an available follow-up.

**Implementation:** `candidate_kernels.pyx` gained `compute_lsh_band_width`,
`compute_lsh_n_bands` (applies `RECALL_SAFETY_MARGIN` internally), and `compute_lsh_sizing`
(convenience wrapper) as pure, side-effect-free module-level functions — `_finalize_sizing`
itself now calls them instead of duplicating the math inline, so the "real" sizing path and any
external caller's analytical computation can never drift apart. Rebuilt clean.
`_apply_proxy_anchor_selection`'s near/far candidate-rate-tolerance split was replaced with a
single sort: `_underpowered_rank → _ci_width_rank → _n_bands_rank → _time_rank →` (the
remaining search-work/precision/specificity/recall/cand_w chain, now late tie-breaks only).

**Verification:** one existing test (`test_proxy_selection_uses_real_time_tiebreak_within_tolerance`)
specifically validated the now-removed tolerance-gated mechanism and correctly failed once the
redesign landed — not a regression, the expected consequence of an approved design change.
Replaced with three tests reflecting the new intended behavior:
`test_proxy_selection_prefers_fewer_n_bands` (fewer bands wins even against a faster-but-more-
bands alternative), `test_proxy_selection_uses_real_time_tiebreak_among_equal_n_bands` (time
still decides real ties, preserving the original 2026-07-05 intent minus the arbitrary gate),
and `test_proxy_selection_prefers_narrower_confidence_interval` (new, covers 3a). Full suite
green (115/115 — 113 original + 2 net new).

**A real end-to-end rerun caught a real bug in this same implementation, before it was
considered done.** First real rerun (m=150, gamma=0.6 fixed, occupancy ∈ {2,3,5,10}, same
config as the earlier manual-vs-hyperopt comparison) selected a 219-band config (occupancy=3)
over an available, feasible, faster 88-band one (occupancy=10, 1.72s vs. 3.87s) — wrong, by the
new design's own intent. Root cause: `_ci_width_rank` had been placed second in the sort chain,
right after `_underpowered_rank` — but it's a CONTINUOUS value, and the losing config's CI
happened to be perfectly tight (`recall_lb==recall_ub==1.0`) purely by chance, so it won outright
before `_n_bands_rank` (third in the chain) ever got a chance to matter. Any continuous key
placed that early defeats everything after it (exact float ties are vanishingly rare) — the
exact same failure mode `candidate_rate` had before this whole redesign, just relocated. Fixed
by moving `_ci_width_rank` to the end of the primary chain (`_underpowered_rank → _n_bands_rank
→ _time_rank → _ci_width_rank`), where it's a true last-resort tie-break, not a primary-tier
signal. Full suite re-confirmed green (115/115) after the fix (the CI-width test itself was
unaffected, since it deliberately uses exact `n_bands`/time ties to isolate that mechanism).
Reran the same end-to-end config: **`n_bands=88` (occupancy=10) now wins, exactly as the design
intends** — recall 98.7% (well above target), feasible, not underpowered, 1.95s measured. This
is the same combo (occupancy=10) the very first real comparison this session found gave the
best true speedup (13.79x) — the redesigned, more principled ranking arrives at the same answer
a purpose-built manual investigation found, without needing that investigation.

## 2026-09-04 (c) — Memory/RSS reduction: the recall margin's real cost, made configurable, plus
int32 downsizing of SignLSHBandIndex's core arrays

**Trigger:** user asked directly about OOM/RSS problems with "my method." Investigated rather
than assumed, and found this session's own earlier fixes are a very likely direct contributor:
`SignLSHBandIndex`'s core arrays scale linearly with `n_bands`, and the 2026-09-03 overlap
correction + recall margin substantially inflated `n_bands` — and the inflation **compounds with
scale**, not a flat multiplier: ~3-4x at m=150 (the scale this session's earlier validation
happened to use), but **8-11x at m=1000-3000** (a more realistic production range), because
`band_width` itself grows with `m*L`, and the overlap correction's own penalty worsens as
`band_width` grows.

**Real memory measured, not just n_bands counted.** First attempt at measuring (via
`resource.getrusage` immediately after construction, no data inserted) gave misleadingly small
deltas — `np.zeros`-allocated pages can be lazily committed by the OS and never touched until
written, understating real-world usage. Corrected by actually calling `insert_many` with real
sign vectors before measuring, at m=1000, L=32, occ=3, giving trustworthy numbers:

| config | n_bands | peak RSS |
|---|---|---|
| pre-session formula (no overlap correction) | 115 | 202 MB (post-fixes below) |
| overlap-corrected, margin=0.0 | 537 | 693 MB |
| overlap-corrected + margin (current default), pre-int32 | 973 | 1559 MB |
| overlap-corrected + margin (current default), post-int32 | 973 | **1199 MB** |

**Fix 1 — `RECALL_SAFETY_MARGIN` made a real, overridable parameter, not a hardcoded constant.**
`compute_lsh_n_bands`/`compute_lsh_sizing` (`candidate_kernels.pyx`) gained a
`recall_safety_margin` parameter (default: the calibrated 0.028, unchanged behavior unless
overridden). `SignLSHBandIndex.__cinit__` gained the same parameter (negative sentinel = "use
calibrated default"; 0.0 is a legitimate explicit choice, not conflated with "unset").
Threaded through `Candidates.__init__` and `CorrTrack.__init__` as
`candidate_lsh_recall_safety_margin` (`None` default preserves current behavior exactly; an
explicit value, including 0.0, overrides it) and forwarded at the one real `SignLSHBandIndex(...)`
construction site. Also wired into `_extract_feature_overrides` so it can be swept via
`PARAM_GRID` if ever wanted. Verified end-to-end: `candidate_lsh_recall_safety_margin=0.0` on a
real `CorrTrack` reproduces the pre-margin `n_bands` (136 at the m=150 reference config) exactly;
default (unset) reproduces the with-margin value (219) exactly.

**Fix 2 — `_band_keys`/`_neg_band_keys`/`_membership_pos` downsized int64→int32.** Verified safe
before changing anything: band-key values are bounded by `2^band_width - 1`, and `band_width` is
capped at `min(24, n_vectors)`, so values never exceed ~16.7M — far under int32's ~2.1B range.
`_membership_pos` (a position within one bucket's posting list) is bounded by realistic bucket
occupancy, similarly far under range. All three arrays' allocation sites (initial `__cinit__`,
post-tolerance-resize reallocation, capacity-growth reallocation) and all 7 typed-memoryview
declarations across the file updated together; local variables interacting with them stay
`Py_ssize_t`/`int64_t` (safe widening on read, verified-safe narrowing on write).

**Fix 3, found while measuring Fix 2's real impact, not originally scoped — `_post_capacity`/
`_post_count` also downsized.** Fix 2 alone only measured a ~15% reduction (1558→1321MB) at
m=1000, smaller than expected from halving 3 of the arrays involved. Investigated why rather
than accept a smaller-than-expected number: the posting-list scaffolding
(`_post_total_slots = n_bands * bucket_count`, three parallel arrays —
`_post_members`/`_post_capacity`/`_post_count`) turned out to be a comparably-sized consumer at
large `band_width` (bucket_count grows too) — computed directly at this config: ~383MB at int64
for just `_post_capacity`+`_post_count`, vs. ~255MB at int32. Same safety argument as Fix 2
applies (per-bucket counts/capacities, bounded by realistic alive-population size) — deliberately
did NOT touch `_post_members`'s own contents (actual item IDs), which are a different risk
profile (potentially unbounded over a very long-running stream, not audited here). After both
int32 fixes: 973-band config drops from 1559MB to 1199MB (~23%, ~360MB saved) — this is the
number in the table above.

**Combined effect, the two levers together (int32 automatic + margin=0.0 explicit):** 973 bands
→ 537 bands → 693MB, a real 55% reduction (866MB saved) from where this investigation started.
Disclosed plainly: even at margin=0.0, 537 bands is still ~4.7x the pre-session 115 — the
overlap correction itself (the scientifically necessary part, not optional) accounts for most of
that remaining gap; only the margin's cost is truly optional.

**Verification:** full test suite green (115/115) after each of the three fixes. Real end-to-end
retrieval correctness re-checked at the m=150 reference config after both int32 changes:
`n_bands=88` and `precision=1.0` matched exactly, before and after — the dtype change did not
corrupt band-key computation or posting-list membership tracking. (Recall/speedup showed some
run-to-run variance unrelated to this change — traced the main LSH construction path's
`band_seed` to confirm it's a fixed default unrelated to `seed`/dtype, but did not fully trace
the recall variance itself; flagged honestly rather than invented an explanation.)

**Not done this round, disclosed as open:** `_post_members`'s own item-id storage (kept at
int64, not audited for downsizing); the sweep scripts' existing `_make_memory_limiter`/
`RLIMIT_AS` safety net (built after earlier real WSL OOM crashes) has not been revisited given
`n_bands` — and therefore memory needs — just changed substantially; whether the recall
margin's flat, scale-independent pad should instead be scale-aware (smaller at large `m*L`,
where its relative cost is worst) was raised but not investigated.

### 2026-09-04 (d) — Memory-leak investigation: `SignLSHBandIndex` never reclaims expired slots (root cause of real-world OOM at m=300)

**Trigger:** user asked why they got OOM running the real project at `m=300` when the (c) table
above showed only ~1.2-1.6GB peak RSS at `m=1000`. That table measured a single one-shot batch
build, not a long-running stream — a real gap in what had been benchmarked.

**Method.** Ran the real default config (`window_size=168, window_step=12, n_lags=168` → `L=15`)
at `m=300` for 500 real `ct.run()` steps, checkpointing `ru_maxrss`. RSS grew continuously and
*accelerated* (267→366→461→621→964→964→1638 MB at steps 10/50/100/200/300/400/500) — no
plateau. Ruled out, in order, with direct evidence each time (not assumption):
- Multiple grid nodes: `CorrTrack.__init__` hardcodes `n_grids=1`. Not the cause.
- Multiprocessing/process duplication: `library_corrtrack_parallel.py` uses only
  `ThreadPoolExecutor` (shared memory) on this path. Not the cause.
- Monitor state (`_append_status`/`_append_anomaly`, unpruned — `prune_mask` confirmed dead code,
  grepped zero call sites): a direct `monitor=False` control run gave an *identical* growth
  curve. Disproven, dropped.
- `tracemalloc` snapshot diff (step 100 vs. 400) localized the dominant growth (+320MB) to
  `library_corrtrack_parallel.py`'s `self._lsh_index.insert_many(...)` call site.

Added temporary debug properties `SignLSHBandIndex.count`/`.capacity` (return `_count`/
`_capacity`) and re-ran the same 500-step test: `lsh_count` grew linearly and without bound
(0→11,100→26,100→56,100→86,100→116,100→146,100), ~+300/step (= one full `m` per step, i.e.
essentially nothing was ever being excluded), while `lsh_capacity` doubled reactively each time
count exceeded it (1024→...→262,144). `n_bands` stayed flat at 354 throughout, so this is
independent of the (a)-(c) work above.

**Root cause, found by reading `candidate_kernels.pyx` directly, not inferred:**
`SignLSHBandIndex.drop_before_time` (`:3965`) correctly marks expiring entries dead
(`alive[i]=0`, `_alive_count -= 1`, `_dead_count += 1`) and eagerly removes them from every band's
posting list (`_remove_from_postings`) — so query correctness and recall are unaffected by this
bug, and the class's own docstring claim that "`alive_count` plateaus at exactly `m*L`" (line
~3194) is accurate for that *logical* counter. But **no code path ever reuses a dead slot's
index.** `insert_many`'s per-item allocation (`:3756-3757`) always takes the next
never-before-used slot at `i = self._count`, then `self._count += 1` — monotonically, forever.
`_ensure_capacity` (`:3669`) grows *every* backing array (`_vectors`, `_alive`, `_window_idx`,
`_sid_idx`, `_sid_rank`, `_time`, `_window_size`, `_entry_ids`, `_band_keys`, `_neg_band_keys`,
`_membership_pos`, `_visited_stamp`, `_words`, plus the malloc'd `_touched_buf`) to accommodate
`self._count`, doubling capacity reactively — there is no free-list, so a dead slot's storage is
never recovered or reused. Net effect: physical memory scales with **total items ever inserted
across the whole run's lifetime**, not with the bounded logical alive population (`_alive_count`,
correctly ≈ `m*L`). This is *why* the m=1000 snapshot in (c) looked bounded (a one-shot batch
never touches this path at all — nothing ever expires in a single batch) while a real, long
enough stream at *any* `m`, including the smaller m=300, eventually exhausts memory: given enough
steps, `_count` — and therefore `_capacity` and every array sized by it — climbs without limit.
This is a pre-existing bug, unrelated to and more severe than the (a)-(c) work; not introduced
this session.

**Status: FIXED and verified.** User confirmed and asked to proceed. Implemented the proposed
free-list: `SignLSHBandIndex` and `HammingExactIndex` (the same lazy-eviction contract, same bug,
confirmed by reading its `_insert_one`/`drop_before_time` too — not assumed) each gained
`_free_slots`/`_free_count`/`_free_capacity` fields and a `_push_free_slot` helper (same
doubling-growth pattern as `_post_members`' own realloc). `drop_before_time` now pushes a slot
onto this stack right after removing it from postings/`_alive_list`; `_insert_one` pops from the
stack before ever calling `_ensure_capacity`/growing `_count`. Every per-slot field is
unconditionally overwritten on insert, so a reused slot needs no separate clearing.

**A second, real bug found and fixed while implementing this, not assumed away:** the first
version of the free-list fix (slots reused, but `entry_id` still assigned from a separate
ever-incrementing `_next_entry_id`) broke 2 of the 115 tests
(`test_corrtrack_dist_corr_algorithm_fast_matches_naive_end_to_end`,
`test_corrtrack_distance_corr_sketch_multichannel_gate_rejects_independent_pairs` — both went
from finding real candidates to finding zero). Investigated rather than reverted: reading
`_find_pair_rows_meta` directly showed it uses a caller-supplied `entry_id` **directly as an
array subscript** into every per-slot array (`alive[q_entry]`, `window_idx[q_entry]`,
`band_keys[q_entry, ...]`, ...) — i.e. `entry_id` was never an independent id, it was *required*
to equal the physical slot index. Pre-fix, `_next_entry_id` and `_count` were incremented
together on every insert with no deletions ever skipping either, so `entry_id == slot index`
held by coincidence; making slots reusable broke that coincidence (a reused slot's fresh
`entry_id` no longer matched the older, larger index `_next_entry_id` had reached), causing real
queries to misindex into the wrong slot and silently return nothing. Fixed by setting
`entry_id = i` (the actual slot index) directly and removing the now-fully-dead
`_next_entry_id` field entirely (confirmed via grep it was read nowhere else). This is exactly
why entry ids didn't need to be globally unique forever in the first place: `_recent_entry_ids`/
`recent_entry_ids` are only ever used within the same step's own insert-then-immediately-query
call, and any Python-side reverse-lookup for an expiring window's old id is dropped from
`_reverse_entry_ids` in lockstep with expiry (`_clean_old_sketches`), before that slot could ever
be handed to a new insert.

**Verification:**
- Full suite green again after the entry_id fix: 115/115 (was 113/115 with the free-list-only,
  entry_id-bug version).
- Reran the exact same 500-step, m=300 diagnostic that found the leak: `lsh_count`/`lsh_capacity`
  now **plateau exactly at the theoretical steady state** (`lsh_count=4500 = m*L = 300*15`,
  `lsh_capacity=8192`) from step 50 onward, unchanged through step 500 — vs. the pre-fix run's
  unbounded climb to 146,100/262,144. `n_bands` unchanged (354) throughout, confirming this is
  independent of the (a)-(c) sizing work.
- RSS growth rate from step 50→500: **~0.136MB/step post-fix vs. ~2.83MB/step pre-fix — a ~21x
  reduction.** Disclosed honestly, not overclaimed: RSS is not perfectly flat post-fix (309.7MB
  at step 50 → 371.1MB at step 500, +61MB total) — some slow residual growth remains, not yet
  traced to a specific cause (candidate: ordinary Python-side dict/list churn elsewhere in the
  pipeline, or allocator fragmentation not returning freed pages to the OS; not investigated
  further this round). This residual is over an order of magnitude smaller than the bug just
  fixed and does not block calling the leak itself resolved.

Debug `count`/`capacity` properties added to `SignLSHBandIndex` for this investigation are kept
permanently (cheap, useful for any future diagnostics). Scratch investigation scripts
(`memory_investigate.py`, `memory_investigate_nomonitor.py`, `memory_trace.py`,
`memory_investigate2.py`) deleted now that the fix is verified.

**Not done this round, disclosed as open (superseded by the follow-up entry below):**
`lsh_dead_node_ratio`/`hexact_dead_node_ratio` (pure diagnostic CSV columns, not used for control
flow anywhere — confirmed via grep) changed meaning with this fix — `_dead_count` now counts
cumulative expiration *events* over the run's lifetime (a slot can die more than once after being
reused), while `_count` stabilizes near the steady-state population, so this ratio is no longer
bounded to [0, 1] the way it was pre-fix; not adjusted this round since nothing consumes it
programmatically, but a future reader of old vs. new CSVs should not compare the two eras' values
directly.

### 2026-09-04 (e) — Second leak found and fixed: `Candidates._window_idx`/`_win_sid*` never pruned; residual RSS growth investigated and traced to intended output accumulation, not a leak

**User asked "the memory RSS should be flat theoretically. Investigate it further"** — the (d)
entry's own disclosed ~0.136MB/step residual (measured after the slot-reuse fix) was not yet
explained.

**Second real leak found, same investigation session.** Extended the diagnostic to also print
`len(Candidates._win_sid)`/`len(Candidates._window_idx)`: both grew linearly and without bound
(11,100→146,100 over steps 50→500), the *exact same trajectory* `lsh_count` had before being
fixed, while `lsh_count`/`lsh_capacity` themselves stayed correctly flat. Root cause, found by
reading `_get_or_create_window_idx`/`_clean_old_sketches` directly: `_get_or_create_window_idx`
assigns each `(sid, start_time, window_size)` key a permanent integer handle in `_window_idx`
(dict) plus 5 parallel lists (`_win_sid`/`_win_sid_idx`/`_win_sid_rank`/`_win_time`/`_win_w`) —
the exact same "monotonic index, never reclaimed" pattern as the Cython slot leak, one layer up
in the Python `Candidates` class. `_clean_old_sketches` already prunes `self.sketches`/
`self._reverse_entry_ids` for each expiring key but never touched this cache — a plain oversight,
not something subtle. This cache is shared by every backend (blocked/instinct/lsh/bptree all call
`_get_or_create_window_idx`), so the same fix benefits all of them, though only the LSH path
(the user's real backend) was directly exercised and verified here.

**Fixed:** added `self._win_free_idx = []` (a free-list, same pattern as the Cython fix) plus a
new `_release_window_idx(window_id)` helper that pops the freed idx into it; wired into
`_clean_old_sketches`'s existing per-key cleanup loop for the blocked/instinct/lsh dict-cleanup
branches (the ones actually reachable for those backends today); `_get_or_create_window_idx` now
pops from the free-list before ever growing `_win_sid`'s length. Safe by the same reasoning as
the Cython fix: `_release_window_idx` runs immediately before the matching `_expire_*_index()`
call, which uses the identical time cutoff that made this partition eligible for eviction, so no
still-alive entry can reference the freed idx before it's reused. `_get_or_create_window_idx_numeric`
(a near-duplicate used only by the tuple-partition path, itself unreachable for LSH — see the (d)
entry's scope note on `keep_numeric_partition`) was deliberately left unfixed and disclosed, not
silently ignored — it doesn't collide with the free-list (it only ever appends), it simply
doesn't get the benefit for the backends that do reach it (bptree, possibly blocked-index).

**Verified:** full suite green (115/115). Rerunning the same 500-step m=300 diagnostic:
`win_sid_len`/`window_idx_dict` now plateau exactly at 4,500 (matching `lsh_count`) from step 50
onward, vs. climbing unboundedly to 146,100 before this fix.

**Residual RSS investigated further, real methodological correction found:** the (d) entry's
"~0.136MB/step" figure was measured via `resource.getrusage(...).ru_maxrss`, which is a
**historical peak, not current memory** — it can only ever increase, even after memory is freed.
Confirmed this by running `gc.collect()` + `ctypes` `malloc_trim(0)` at each checkpoint: zero
effect on the reported number at every step, exactly what peak-tracking predicts (this would also
happen if it were pure fragmentation, so it doesn't by itself distinguish the two — the deciding
evidence is below). Redone with actual current RSS (`/proc/self/status`'s `VmRSS`, not a
historical max): growth from step 30 (309.6MB, right when the two real fixes' structures first
plateau) to step 500 (332.5MB) is +22.9MB over 470 steps, and — unlike the two real bugs' perfectly
linear, undecelerating climb — visibly **decelerates** over the run (~0.10MB/step in the first
70 steps of that window down to ~0.02MB/step in the last 100), the signature of a converging,
bounded process rather than an unbounded leak. `tracemalloc` (step 100 vs. 500 snapshot compare)
confirms directly: the largest live Python-object growth sites are `self.correlated.update(...)`
and `self._maxlag_state[key] = {...}` — both are this project's own **intended, by-design output
accumulators** (every genuinely-found correlation and its per-pair max-lag record, kept for final
CSV export via `_save_correlated`/`_save_max_lag_correlated`), not internal bookkeeping that
should have stayed invisible. Their growth naturally decelerates as the finite set of truly
correlated pairs in this dataset gets discovered. Their absolute size is small (~1,000-1,200
entries combined by step 500, well under 1MB) — smaller than the full ~22.9MB residual, so the
remainder is most likely ordinary NumPy/BLAS scratch-buffer working-set warmup (settling in over
the first several dozen steps as more distinct temporary-array shapes get exercised), a normal
property of any long-running vectorized Python process, not a discoverable data-structure leak.

**Conclusion:** RSS is now flat in the sense that matters — both structural, unbounded-with-
stream-length leaks (Cython slot reuse, Python window_idx cache) are fixed and verified; the
small remainder is real algorithmic output (by design) plus ordinary numeric-library warmup, not
a bug. If the user later wants `self.correlated`/`self._maxlag_state` themselves bounded (e.g.
streamed to disk incrementally instead of held fully in memory for arbitrarily long production
runs), that is a deliberate output-semantics change, not a leak fix, and would need explicit
sign-off per CLAUDE.md's "don't silently change the scientific meaning of the method."

Scratch investigation scripts used for this entry (`memory_investigate3.py` through `..5.py`,
`memory_trace2.py`) deleted after writing up. `memory_investigate2.py` (from the (d) entry,
itself already deleted then) is not affected.

### 2026-09-07 — Sobol sweep rerun with widened (m, L), plus a new large-scale CorrTrack-only timing companion

**User asked to rerun the Sobol sweep (`experiment_lsh_sobol_sweep.py`) and increase m/L now
that memory is fixed.** Investigated the real cost tradeoff before touching anything: widening
the range is a real multi-day compute commitment dominated by the O(m^2*L) brute-force ground
truth (needed for recall/precision), NOT by CorrTrack's own memory footprint — the memory fix
doesn't change that cost at all. Computed real numbers from the existing (m<=300,L<=64) run's
own `bf_runtime`: current range ~1.4 days total; m<=1000/L<=64 ~9.5 days; m<=3000/L<=128 ~102
days (using the project's own measured total/bf-runtime ratio, 5.28x). Presented this tradeoff
to the user with concrete options; they chose a moderate widening for the full,
accuracy-validated rerun, **plus** a separate large-scale, CorrTrack-only ("we will not have
accuracy metrics, we can compare running time") companion at much higher m/L, since skipping
brute force removes the actual cost driver.

**Full accuracy-validated rerun:** `experiment_lsh_sobol_sweep.py`'s `M_RANGE`/`L_RANGE` widened
from `(50,300)`/`(8,64)` to `(50,500)`/`(8,96)` (N_DESIGN_POINTS unchanged at 64). Est. ~4.6
days (`--generate-design` output: 21.0h brute-force-only, 110.8h total). The old 83-point result
(`tmp_artifacts/lsh_sobol_sweep/`) was renamed to `tmp_artifacts/lsh_sobol_sweep_v1_m300L64/`
(not overwritten) before regenerating the design, so the pre-widening result stays available for
a direct before/after comparison.

**New: `experiment_lsh_sobol_sweep_timing_only.py`** — a large-scale (m in [50,3000], L in
[8,128], the exact range this session's own memory-leak-fix benchmarks validated as safe)
companion that skips brute force entirely and therefore also skips empirical (n_bands,
occupancy, gamma) calibration (that calibration search itself needs ground truth to check
achieved recall — see `calibrate_nbands_occupancy`). Uses the THEORETICAL sizing directly instead:
`candidate_kernels.compute_lsh_sizing(m, L, target_occupancy=3.0, corr_threshold, n_vectors=64,
target_recall=0.95, n_bands_tolerance=1.0)` — the exact overlap-corrected, margin-padded formula
`_finalize_sizing` itself uses, at tolerance=1.0 (this session's own "AUTO" convention: trust the
calibrated margin, no extra multiplier). `target_occupancy` fixed at the project default (3.0,
not tuned) and gamma left at `CorrTrack`'s own default derivation (not calibrated) — both
disclosed simplifications, since neither precision nor recall is measured here anyway. Records
`sketch_time`/`candidate_time`/`validation_time`/`monitor_time`/`smart_wall_time`,
`validated_candidates`/`total_candidates`/`tested_candidates`, a derived
`candidate_rate_vs_m2L = tested_candidates / (m*(m-1)/2 * L * n_steps)`, and `peak_rss_gb` — no
`bf_*`/precision/recall/f1/speedup columns at all (there is no ground truth to compute them
against). Cost estimate: fit `log(wall_time) ~ 1.586*log(m) + 0.346*log(L) - 6.39` (R^2=0.80)
against the OLD sweep's own real measured `smart_wall_time`, giving ~5.2h total for N=64 points
at this range — affordable specifically because it skips the O(m^2*L) ground-truth cost the
accuracy sweep can't avoid.

**Verified before launching anything long:** both scripts' `--generate-design` ran cleanly (85
cells / 64 cells respectively, correlation among factors still low at 0.0235 for the widened
accuracy design); a real end-to-end smoke test on the timing-only script's cheapest cell (m=73,
L=8) produced a well-formed CSV row (2.8s wall time, `n_bands=23`, all columns populated, no
NaN/None where unexpected); a second smoke test on its most expensive cell (m=2627, L=94) was
launched to confirm the large-scale path itself runs cleanly (result pending as of this entry).

**Resource note:** this machine has only 7.8GB RAM with ~1.2GB free at the time of launch
(unrelated to CorrTrack — other processes hold ~5.5GB). Both scripts already process cells
strictly sequentially within themselves (one subprocess at a time), but running BOTH scripts
concurrently could still mean two heavy subprocesses active at once on an already-tight machine
— so the two sweeps are launched sequentially (timing-only first, since it is far cheaper and
directly exercises the just-fixed memory path at real scale; the moderate accuracy sweep follows
after), not in parallel, purely as a host-resource precaution unrelated to the leak fix itself.

**Status: launched, then reworked under real time pressure -- see the 2026-09-08 entry below.**
Both scripts support resume (`_load_completed_cell_counts`/`_cell_resume_key`), so an
interruption does not lose completed cells. The original launch of this entry ran for a while
(timing-only sweep completed: 58/64 cells succeeded, 6 failed cleanly at the extreme m/L+low-
corr_threshold corners -- moved to `tmp_artifacts/lsh_sobol_sweep_timing_only_v1_nohyperopt_m3000/`)
before the user's next request (below) required calibration to be added, which then surfaced a
real cost problem serious enough to redesign again.

### 2026-09-08 -- Real hyperopt calibration added (both sweeps), a hard cost wall discovered and fixed, then a 28-hour deadline forced a full rescope

**User's next message, three parts:** (1) confirmed the memory/hyperopt-phase question --
neither running sweep actually used `CorrTrack_optimize`'s real proxy-anchor hyperopt (both used
older, separate mechanisms: `calibrate_nbands_occupancy`'s own brute-force-based grid search, or
no calibration at all); (2) pointed out the timing-only sweep should ALSO calibrate (against a
small ground-truth sample, not the full brute force it deliberately avoids) and that always
calibrating against full brute-force ground truth doesn't "showcase how CorrTrack will handle
actual data" -- in real deployment you don't know where correlations are during the hyperopt
phase either; (3) asked to use these experiments to validate their own hyperopt pipeline
directly. Both running sweeps were killed (`pkill`) since they were using the wrong methodology
for what was now being asked.

**Investigated `CorrTrack_optimize.get_optim_params` -> `_get_proxy_anchor_optim_params` ->
`_run_proxy_anchor_options`/`_run_corrtrack_proxy_anchor` end to end** (the real production
entry point, matching `corrtrack_param_search.py`'s own usage exactly) to design a faithful
integration. Found and fixed a real, separate gap while doing this: `candidate_lsh_target_
occupancy` was already being computed into `_run_corrtrack_proxy_anchor`'s own `record` dict
(a fix from earlier this session) but silently dropped on the way to CSV -- `_row_from_mapping`
only ever emits columns listed in `OPTIM_RESULT_COLUMNS`, and that column was missing from the
list. Added it (`library_corrtrack_parallel.py`, `OPTIM_RESULT_COLUMNS`) -- needed to read back a
proxy-hyperopt winner's chosen occupancy at all.

**Found a real, pre-existing, documented warning before implementing blindly:**
`calibrate_gamma`'s own docstring (`experiment_lsh_cost_sweep.py`) records that proxy-anchor
hyperopt was already tried for this exact synthetic generator and rejected -- at sparse
corr_prop it can see only 1 true positive per anchor (recall estimate degenerates to 0%/100%,
not a real statistic), and reaching more power costs O(m^2*L) per additional anchor. Presented
this tradeoff to the user with three options (use it as-is and record the struggle honestly;
narrow corr_prop so it has enough signal; seed anchors near known injection points). **User chose
option 1** -- use the real pipeline as-is, let it struggle where it struggles, bound cost via the
existing `proxy_max_pair_rows` ceiling.

**Built `calibrate_via_proxy_hyperopt`** (new function, `experiment_lsh_cost_sweep.py`) -- wraps
`CorrTrack_optimize` construction + `get_optim_params`, sweeping `candidate_lsh_target_occupancy`
(`OCCUPANCY_GRID`) x `candidate_cosine_threshold` (`GAMMA_TRIAL_OFFSETS` below corr_threshold) --
24 combos, matching `calibrate_nbands_occupancy`'s own two-parameter scope exactly, just via the
real hyperopt mechanism. `n_vectors` fixed (not searched), per the user's own explicit statement.
Smoke-tested standalone (m=80): 24/24 combos succeeded, sensible winning result (n_bands=25,
occupancy=10.0, gamma=0.65, feasible, proxy_n_gt=395).

**Then the user added a hard 28-hour deadline** plus three more requirements: (a) results
comparable to the last (m<=300) sweep, not the wider moderate range; (b) some timing-only
higher-m examples; (c) `touched_candidates` tracked (found: `CorrTrack.candidate_search_lsh_
candidates_touched`, a real, already-accumulated attribute -- just wasn't being read into either
sweep's CSV); (d) a derived, no-extra-compute "speedup with validation (CorrTrack only) and
monitoring (both sides) disregarded" metric, from `sketch_time+candidate_time` (CorrTrack) vs.
`cand_time+val_time` (brute force, monitor_time excluded) -- both already tracked as separate
timing fields in each side's own record; confirmed brute force tracks the identical
`candidate_time`/`validation_time`/`monitor_time` breakdown CorrTrack does (`run_bf`, same
accumulator pattern).

**Real cost wall discovered by direct measurement, not assumed:** timed `calibrate_via_proxy_
hyperopt` at (m=300, L=64) -- the accuracy sweep's own worst case -- and it exceeded 115s even
capped to a single anchor at a 6,000,000-row ceiling; `est_pairs_per_anchor` (a real, exact
formula, `estimate_proxy_pairs_per_anchor`) reaches 5.7M pairs there, 64.5M at (m=1000, L=65),
1.15B at (m=3000, L=129) for THIS project's own `PROXY_HYPEROPT_MAX_ANCHOR_COUNT` ceiling logic
-- `recommend_proxy_pair_row_budget`'s `max(1, budget//est)` always attempts at least ONE anchor
regardless of how far over budget it is, and reference construction (`_prepare_proxy_anchor_
reference`) is a pure-Python nested loop with no Cython/vectorized fast path, so a large single
anchor is not merely slow, it's computationally impractical. A second, smaller timed run (m=150,
L=20, 438,675 pairs/anchor) took 11.0s -- giving a real, measured ~25us/pair rate to design
against, not a guess.

**Fix: a feasibility gate, added to `calibrate_via_proxy_hyperopt` itself.** New module constant
`PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR = 2,000,000` (~50s worst case at the measured 25us/pair
rate) -- `calibrate_via_proxy_hyperopt` now checks `estimate_proxy_pairs_per_anchor` FIRST and
returns `None` immediately (a clear printed reason, not a hang) if it's over threshold, before
touching anything expensive. New `theoretical_sizing_fallback` function (`compute_lsh_sizing` at
`target_occupancy=3.0`, `gamma=corr_threshold`) used by both sweep scripts whenever hyperopt
returns `None` (infeasible OR genuinely nothing selectable) -- a real, disclosed fallback, not a
silently substituted one: a new `hyperopt_used` boolean column in both CSVs makes the distinction
explicit per cell. `pair_row_hard_ceiling`/`max_anchor_count` also exposed as real, overridable
parameters (both scripts pass `PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR` as the ceiling, matching
the feasibility gate exactly).

**Rescoped given the 28-hour wall clock (real numbers, not guesses):**
- `experiment_lsh_sobol_sweep.py`: `M_RANGE`/`L_RANGE` reverted from the moderate `(50,500)/
  (8,96)` back to the ORIGINAL `(50,300)/(8,64)` -- "over my last sobol sweep (up to m=300)" --
  directly comparable to the pre-existing 83-point result, and this range's own worst case
  (5.7M pairs/anchor) sits just under the new feasibility threshold. `--generate-design` now
  reports 7.1h brute-force-only + a flat 40s/cell proxy-hyperopt estimate (85 cells, ~0.9h) =
  ~8.0h total, comfortably inside 28h. (The old, stale "5.28x" total/bf ratio -- calibrated
  against the OLD, much more expensive `calibrate_nbands_occupancy` mechanism -- no longer
  applies and was replaced with this flat per-cell estimate.)
- `experiment_lsh_sobol_sweep_timing_only.py`: `M_RANGE`/`L_RANGE` left UNCHANGED at `(50,3000)/
  (8,128)` rather than pushed higher as first considered -- a quick check of the fitted cost
  law (`m^1.63 * L^0.72`, refit on the prior run's real data) predicted single cells up to
  5-6 HOURS at m=5000, an unacceptable risk to the 28h deadline for one data point. Re-running
  the SAME already-validated range with the new hyperopt+touched-candidates additions is treated
  as the "updated ... higher-m examples" ask, not a further widening.
- Old, wrong-methodology accuracy-sweep artifacts (one partial cell) deleted (not archived --
  nothing worth keeping); old timing-only results (58/64 cells, no hyperopt, real data) kept at
  `tmp_artifacts/lsh_sobol_sweep_timing_only_v1_nohyperopt_m3000/` for comparison.
- Both scripts now launched CONCURRENTLY (not sequentially as originally planned) given the
  tighter deadline -- `--worker-memory-limit-gb` reduced to 3.0 (from 6.0) per script so two
  concurrent workers' hard RLIMIT_AS caps sum to ~6GB, leaving headroom on this 7.8GB host.

**Verification before committing the full 28h run:** both `--generate-design` runs succeeded.
End-to-end smoke tests on each script's cheapest cell, full `n_steps=2000` fidelity:
- Accuracy sweep (m=52, L=10): clean row, real hyperopt used (`hyperopt_used=True`,
  `proxy_underpowered=True, proxy_n_gt=1` -- a real, honestly-reported "let it struggle" case at
  low corr_prop, not hidden), `n_bands=10`, `speedup=5.50`, `recall=0.97`, `precision=1.0`,
  `touched_candidates=14,844,836`, `speedup_no_val_no_monitor=5.11` (close to but distinct from
  the plain `speedup`, as expected).
- Timing-only sweep (m=73, L=8): **caught and fixed a real bug** -- `run_worker_cell`'s own print
  statement still referenced the old `n_bands_theoretical`/pre-rename key, crashing with
  `KeyError` right after hyperopt legitimately found nothing selectable (fed into
  `theoretical_sizing_fallback` correctly; the print statement was the only broken part). Fixed
  (`n_bands_used`, added `hyperopt_used`/`touched` to the log line); rerun confirmed clean:
  `hyperopt_used=False` (fallback path correctly taken and logged), `n_bands=23`, `wall_time=3.0s`,
  `touched=14,445,281`.

**Both sweeps launched concurrently** (`nohup ... & disown`, independent of this session) at
2026-09-08 08:27 CEST -- confirmed via `ps aux` processing real cells (accuracy: m=141/L=37;
timing-only: m=554/L=16) immediately after launch.

### 2026-09-08 (b) -- User corrected an imprecise claim; real Cython fix to CorrTrack_optimize's own hyperopt pipeline, not just this sweep

**User pushed back directly**: "You said 'in pure Python'... There shouldn't be any pure Python
constraints. No Python. All Cython. What did you mean by that?" -- a fair challenge, since
`fast_corr_and_dist` (the per-pair correlation function `_prepare_proxy_anchor_reference` calls)
**is** Cython. Corrected precisely: the correlation *math* was already Cython; the *orchestration*
calling it ~5.7 million times in a plain Python loop (building tuples, dict lookups, list
appends) was not. Confirmed a real, existing bulk Cython validator (`validate_corr_batch`)
exists in `candidate_kernels.pyx` but has different semantics (adds its own pair-level
constant/spike filtering) -- using it would have silently duplicated/disagreed with the
window-level filtering `_prepare_proxy_anchor_reference` already applies via `is_valid`
(`CorrTrack.is_near_constant`/`is_structurally_spiked`, the SAME checks the real streaming run
uses -- user explicitly confirmed these must stay exactly as-is, not weakened or replaced).

**User also flagged a separate real problem**: running both sweeps concurrently with a reduced
3GB/worker memory cap (to fit two processes on this 7.8GB host) was causing avoidable OOM
failures (confirmed: a 951-band cell failed trying to allocate just 29.7MiB, having already
consumed most of its 3GB cap). Fixed immediately: stopped the timing-only sweep (resume-safe,
no data lost), let the accuracy sweep continue solo, queued timing-only to follow sequentially
afterward at the original 6GB/worker cap (superseded by the fuller rewrite below).

**Important clarification given to the user, unprompted**: `_prepare_proxy_anchor_reference` is
a method of `CorrTrack_optimize` itself, called by every real invocation of `get_optim_params` --
including the user's own production usage via `corrtrack_param_search.py`. This was never
something specific to these two benchmark sweeps; both sweeps just call the same production
method every real caller does. This raised the value (and the correctness stakes) of fixing it
properly rather than working around it in the benchmark scripts alone.

**Profiled the real bottleneck precisely (`cProfile`, 438,675 real pairs) rather than guess
further**: only 18% of cost was in `_fast_corr_and_dist` calls; 50%+ was the raw Python loop body
itself (tuple construction, conditionals, per-pair dict lookups for `is_valid`/`get_window`);
~12% in `_normalize_window_pair_key` + list appends. Presented this corrected, more modest
picture to the user (a naive batched-correlation-only fix would give ~15-20%, not the
order-of-magnitude hoped for) and asked for an explicit scope decision given the 28h deadline and
time already spent; **user chose to go all the way**.

**Implemented, verified, and shipped a full fix:**
1. Added `fast_corr_and_dist_batch` (`candidate_kernels.pyx`) -- a mechanical, `nogil` batched
   port of `fast_corr_and_dist`'s exact per-pair formula (no filtering added, deliberately, per
   the user's explicit requirement to preserve their existing semantics exactly). Verified
   bit-for-bit identical to the per-pair function across 5000 test pairs including edge cases
   (constant windows, near-1.0 correlation) before using it anywhere.
2. Rewrote `_prepare_proxy_anchor_reference` (`library_corrtrack_parallel.py`): validity
   (`is_near_constant`/`is_structurally_spiked`, exact same formulas, exact same `std_thresh=
   1e-3`/`kurt_thresh=5.0` defaults) is now computed once per unique window position for every
   series at once (numpy), not via a per-pair dict lookup; correlation is computed once per
   `s_current` across all its valid `(lag_start, s_other)` candidates via the new batch function,
   not once per pair. The per-row key-construction loop (`_normalize_window_pair_key`,
   `key_to_indices` population) was deliberately left as Python and NOT redesigned around a
   different key representation -- confirmed via a direct grep that `key_to_indices` is looked up
   by exactly this string-tuple key elsewhere in the codebase (the real recall-evaluation path),
   so changing it would have meant auditing and changing that consumer too, out of scope for a
   deadline-constrained fix.
3. **Verified correctness rigorously before trusting it**: reconstructed the exact original
   algorithm in a standalone script and ran it side-by-side against the new implementation on 4
   deliberately varied test cases (small, medium, a cap-binding edge case, a low-corr_threshold/
   negative-correlation-heavy case) -- every single `(pair_key -> truth, truth_sign)` mapping,
   `n_pairs`, `n_gt`, and `anchor_count` matched exactly across all 4. Full test suite green
   (115/115) after the change.
4. **Real measured speedup**: ~4us/pair (was ~25us/pair) -- a genuine ~6x reduction, better than
   the profiled estimate suggested (cProfile itself adds overhead that distorted the earlier
   comparison). (m=300, L=64) -- this sweep's own worst case -- now takes ~23s to build a
   reference, not >115s as before.
5. Raised `PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR` from 2,000,000 to **15,000,000** (~60s worst
   case at the new rate) -- both sweep scripts pick this up automatically since they import the
   constant rather than a hardcoded number. This does NOT remove the feasibility ceiling entirely
   (est_pairs_per_anchor still reaches 64.5M at (m=1000, L=65), 1.15B at (m=3000, L=129) -- true
   infeasibility at real scale, unaffected by this fix, still falls back to theoretical sizing,
   disclosed via `hyperopt_used`) -- it raised where that ceiling falls, substantially.

**Relaunched sequentially** (not concurrently -- addressing the user's memory concern directly),
`--worker-memory-limit-gb 6.0` for both, resuming from already-completed cells (4 accuracy + 3
timing-only cells were already done under the OLD, slower-but-identical algorithm -- their
results remain valid and were not discarded, just resumed past).

### 2026-09-08 (c) -- Real production incident from the Cython fix: raised threshold on CPU time alone, without checking memory

A live cell (m=296, L=55, seed=11, corr_prop=0.00107) drove this host to 3.9GB of swap and
stalled 10+ minutes mid-run. Root cause, found by inspecting the running process directly
(`/proc/<pid>/status`, `VmSwap`): `PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR` had just been
raised from 2,000,000 to 15,000,000 purely on the strength of the Cython speedup's CPU-time win
(~60s worst case at the new ~4us/pair rate) -- the MEMORY cost of holding millions of Python
`pair_keys`/`key_to_indices` tuples was never checked. Measured directly (`tracemalloc`): ~400
bytes/pair-row, unchanged by the Cython fix (that fix cut CPU time in the correlation/validity
computation, not the fundamentally Python-object-based bookkeeping the key-construction loop
still does -- deliberately left as Python, see the earlier entry's own reasoning about
`key_to_indices`'s downstream consumer). At 15,000,000 pairs that's ~6GB -- more than this
7.8GB host has to spare. **Memory, not CPU time, is the real binding constraint; the time-based
win didn't relax it.** Killed the stuck process, reset the threshold to 3,000,000 (~1.2GB for
the reference itself), verified the exact problematic cell now skips (at the time, still a
skip-based design -- see below) in 0.0004s instead of thrashing, reconfirmed the full test
suite, and relaunched (resumed, no cells lost).

### 2026-09-08 (d) -- User's real objection: the feasibility threshold/fallback defeats the method's own scalability-and-accuracy promise; fixed with internal series subsampling, not a workaround

**User pushed back on the whole skip/fallback design itself**, not just its threshold value:
"This method is meant to be scalable and keep the accuracy, and these defeat the purpose."
Correct and important -- `theoretical_sizing_fallback` picks occupancy/gamma with NO search at
all once a cell is deemed infeasible, which is exactly backwards for a method whose value
proposition is scaling without giving up accuracy guarantees.

**Analyzed why the reference-construction cost is inherently quadratic in series count** (real
ground-truth verification needs an exhaustive pairwise comparison somewhere -- same reason full
brute force is O(m^2*L) too) **but argued the FIX doesn't need to eliminate that quadratic cost,
it needs to decouple it from the true m**: `n_bands` already has a closed-form, provably-correct
sizing formula (`compute_lsh_sizing`, no empirical search needed at all since the 2026-09-03
overlap correction) -- what hyperopt's search is actually FOR is `target_occupancy` (a
speed/band-count tradeoff, not a recall-correctness question) and `gamma` (a precision/recall
tradeoff depending on `corr_threshold`/`n_vectors`/data characteristics, not on series count).
**User confirmed this matches their own understanding**, with one real caveat: real datasets can
have heterogeneous series (different noise/trend/spikiness), so a uniform-random subsample risks
missing rare-but-real behavior types -- they wanted every series accounted for specifically to
capture that diversity, not out of a belief that gamma itself depends on m.

**Design, per the user's own follow-up questions ("how do we better select series in the cap"
and "what are the options for the cap"), approved to implement:**
- **Selection**: stratified, not uniform-random. Three cheap, `O(m)` per-series statistics
  (variance, kurtosis -- the SAME formulas `is_near_constant`/`is_structurally_spiked` already
  use, for consistency -- and lag-1 autocorrelation for temporal-structure diversity), combined
  into one composite percentile-rank score, then an evenly-spaced systematic sample across the
  sorted composite axis -- spans the full range of observed behavior instead of clustering near
  wherever a random draw happens to land. Disclosed as a real, simple heuristic (collapsing 3
  dimensions to 1 composite score can still miss some multi-axis diversity a full stratified
  cross-tabulation would catch), not an exhaustive guarantee, chosen for implementation speed
  under the deadline.
- **Cap size**: derived from the SAME resource-budget principle as the (now-superseded)
  feasibility threshold -- largest N such that `estimate_proxy_pairs_per_anchor(N, n_lags,
  window_step) <= budget`, adaptive to L rather than one arbitrary constant. Exposed as a real,
  documented, overridable `CorrTrack_optimize` constructor parameter
  (`proxy_series_subsample_max_pairs` via `proxy_config["series_subsample_max_pairs"]`, default
  3,000,000 -- same memory-safe value as (c)'s reset).

**Implemented in `library_corrtrack_parallel.py`:**
- `estimate_proxy_series_subsample_cap(n_series_total, n_lags, window_step, budget)` (module-
  level, mirrors `estimate_proxy_pairs_per_anchor`'s own pattern) -- binary search for the cap.
- `CorrTrack_optimize._proxy_series_subsample_cap` -- thin instance wrapper.
- `CorrTrack_optimize._proxy_stratified_series_sample(values_full, cap)` -- the stratified
  selection described above.
- `_prepare_proxy_anchor_reference` now computes the (possibly-subsampled) series indices/values/
  ids BEFORE the main anchor loop; everything downstream (anchor_count sizing, the per-anchor
  loop itself, `n_gt`, etc.) is otherwise UNCHANGED, since it already treated `n_series`/
  `values`/`proxy_ids` as "the population to search over" generically. `n_bands` itself is never
  derived from this subsample -- `compute_lsh_sizing` always uses the TRUE, full `n_series`
  wherever it's called, so the analytical recall guarantee at real scale is unaffected by
  calibration-time subsampling.

**A second, real bug found via the SAME smoke test that verified the fix worked**: the caller
(`calibrate_via_proxy_hyperopt`, `experiment_lsh_cost_sweep.py`) still sized `anchor_count`/
`max_pair_rows` off the TRUE `n_series` via `recommend_proxy_pair_row_budget`, even though
`CorrTrack_optimize` now internally subsamples to a much smaller effective count -- at (m=296,
L=55) this capped `anchor_count` to 1 (based on the true m's 4.77M-pair cost) even though the
ACTUAL, subsampled per-anchor cost was far cheaper and could easily have afforded more anchors.
This directly starves statistical power exactly where the whole fix was supposed to restore it.
Fixed: `calibrate_via_proxy_hyperopt` now calls `estimate_proxy_series_subsample_cap` itself
BEFORE sizing the anchor budget, using the EFFECTIVE (post-subsampling) series count consistently
with what `CorrTrack_optimize` will actually do internally. Verified directly: at (m=296, L=55,
budget=3M), `anchors_affordable` goes from 1 (sizing off the true m) to 6 (sizing off the correct
effective n_series=234) -- a real, meaningful 6x improvement in statistical power, not just a
theoretical one.

**Removed the now-redundant external skip-gate** in both sweep scripts
(`experiment_lsh_sobol_sweep.py`/`experiment_lsh_sobol_sweep_timing_only.py`) and in
`calibrate_via_proxy_hyperopt` itself -- previously they pre-checked `est_pairs_per_anchor`
against `PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR` and skipped straight to
`theoretical_sizing_fallback` for large cells; now real hyperopt is attempted at EVERY scale
(the same constant is passed through as the subsample budget instead of a skip threshold).
`theoretical_sizing_fallback` is now reserved for the genuinely rare case where
`get_optim_params` itself raises (nothing selectable at all), not "m was too large" -- disclosed
via `hyperopt_used`/`effective_n_series`/`series_subsampled`, three new CSV columns added to both
sweep scripts for transparency (not silently blended into the aggregate).

**Verified before relaunching**: full test suite green (115/115); the 4-case correctness
verification against the reconstructed original algorithm re-run and still exact (subsampling is
a no-op at small m, confirming it doesn't change behavior where it shouldn't trigger); a
dedicated new test with 4 deliberately distinct synthetic series "types" (white noise, trending,
spiked, periodic; 150 each) confirmed the stratified sampler includes all 4 types in a 60-series
subsample; the cap-sizing binary search verified exact at its own boundary
(`est_pairs_per_anchor(cap) <= budget < est_pairs_per_anchor(cap+1)`); end-to-end retest on the
exact (c)-incident cell confirmed no more swap growth (peak ~1.6GB RSS, not 3.9GB swap) and real
hyperopt now genuinely attempted (24/24 combos) where it used to skip straight to the fallback --
see the next entry for the final anchor-count-fixed timing/result once that retest completes.

**One-off validation exercise, requested by the user, still pending**: run calibration at a
few increasing subsample caps on one of the project's real (not synthetic) datasets and check
whether the chosen gamma actually stabilizes as the cap grows -- a direct empirical check of the
"gamma doesn't depend on m" assumption this whole fix rests on, not yet performed as of this
entry.

## 2026-09-08 (e) — Third bug in the same fix chain: per-anchor budget and total ceiling were the same constant

**Branch:** main.

The "6x improvement" claimed in the previous entry (`anchors_affordable` 1 -> 6 at the incident
cell) was verified only via a **standalone** call to `recommend_proxy_pair_row_budget` using
`hard_ceiling=20_000_000` -- not the value the real sweep scripts actually pass. The subsequent
full end-to-end retest on the identical cell (m=296, L=55, corr_prop=0.001068, corr_threshold=
0.707, seed=11) still came back with `proxy_anchor_count=1`/`anchors_affordable_at_budget=1`,
unchanged from before the fix, despite `effective_n_series=234`/`series_subsampled=True`
confirming the subsampling itself had engaged correctly. Root cause: both
`experiment_lsh_sobol_sweep.py` and `experiment_lsh_sobol_sweep_timing_only.py` call
`calibrate_via_proxy_hyperopt(..., pair_row_hard_ceiling=PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR)`
-- the SAME constant used as `max_feasible_pairs_per_anchor` (the per-anchor subsample budget,
left at its default). `estimate_proxy_series_subsample_cap`'s binary search picks the LARGEST
`effective_n_series` whose single-anchor cost stays `<= budget`, so by construction one anchor's
own cost sits right up against that same budget (2,984,085 out of 3,000,000 here). Passing the
identical number as the TOTAL ceiling then left `anchors_affordable = ceiling // est_pairs_per_
anchor = 1` with no room for a second anchor -- silently reproducing the exact starved-power
symptom the (d) fix was meant to eliminate, just for a different reason than the first bug.

**Fixed (first attempt)**: decoupled the two budgets. Added `PROXY_HYPEROPT_TOTAL_PAIR_ROW_CEILING
= 3 * PROXY_HYPEROPT_FEASIBLE_PAIRS_PER_ANCHOR` (9,000,000) in `experiment_lsh_cost_sweep.py`,
with the per-anchor constant now feeding ONLY `max_feasible_pairs_per_anchor` (unchanged) and the
new constant passed as `pair_row_hard_ceiling` in both sweep scripts. Standalone recomputation
confirmed `anchors_affordable` went from 1 to 3 at the exact incident cell's inputs, and the full
test suite stayed green (115/115).

**A full end-to-end re-run of the identical incident cell (not a standalone calculation this
time) surfaced a fourth, more serious problem**: wall-clock time, not memory, blew up. The run
took 3,042s (>50min) at `anchor_count=3`, vs. 220s previously measured at `anchor_count=1` -- a
~14x time increase for a 3x anchor increase, and `proxy_n_gt` stayed at 1 (zero statistical-power
gain for all that extra cost). Peak RSS was fine (3.3GB, no swap) -- the memory reasoning behind
the 3x multiplier held up -- but the pipeline's real per-trial cost is evidently NOT linear in
anchor count once a single anchor's own pair count is already near the per-anchor budget (likely
the 24-trial LSH sweep's own candidate-retrieval/validation cost growing worse than linearly with
total reference size in that regime; not root-caused further under the time budget). Ruled out
the adaptive-anchor expansion loop (`_get_proxy_anchor_optim_params`) as the cause by inspection
and by the log's own evidence: only one round of the 24-trial grid ran (exactly "24/24
completed", not 48 or 72), so this is the intrinsic cost of one round at `anchor_count=3` with a
near-cap per-anchor size, not repeated expansion.

**Fixed (corrected)**: lowered the multiplier from 3x to 1.5x (4,500,000). Re-verified via the
same standalone calculation that this reproduces `anchors_affordable=1` (the already-measured-
safe 220s/1-anchor behavior -- no regression) exactly at this pathological near-cap corner, while
still affording far more anchors at cells where a single anchor's cost is well below the budget
(23 at m=100/L=20, 244 at m=40/L=12) -- i.e. real multi-anchor power gains stay available
precisely where they're cheap, and the method stops chasing anchors precisely where the evidence
shows it stops paying off. This is now a disclosed, accepted limitation (very large m combined
with very low corr_prop caps out at 1 anchor, same as calibrate_gamma's own documented struggle
at low corr_prop), not a bug. Full test suite re-confirmed green (115/115) after this second
change. Not re-run end-to-end a second time (the arithmetic plus the already-measured 220s data
point for `anchor_count=1` at this exact cell together establish the result without repeating a
50-minute run).

**Lessons, worth keeping**:
- A fix verified only via a standalone re-implementation of part of the call graph
  (`recommend_proxy_pair_row_budget` in isolation, parameters typed by hand) is not the same as
  verifying the actual wired-up call site -- the first standalone check silently used a different
  `hard_ceiling` than production's real call site ever passed, and missed the conflated-constant
  bug entirely. Matches this project's own standing memory note about verifying cross-class
  wiring end-to-end.
- A memory-safe change is not automatically a time-safe one, and vice versa -- both must be
  measured on the real pipeline, not assumed from one dimension alone. Matches this project's own
  standing "correct is not the same as beneficial" lesson: the 3x ceiling was memory-correct and
  mechanically achieved its stated goal (more anchors really were attempted), yet was net-harmful
  at this cell until benchmarked and caught.

**Still pending**: the one-off real-dataset validation exercise (gamma stability across
increasing subsample caps), and relaunching both sweeps sequentially (resumed) now that this
fix is verified safe at the pathological corner.

## 2026-09-08 (f) — Both sweeps relaunched; one-off real-dataset gamma-stability validation run; a real recall-ceiling finding surfaced along the way

**Branch:** main. Relaunched both sweeps sequentially (resumed, not restarted) via
`/tmp/.../scratchpad/relaunch_chain.sh` -- accuracy sweep from cell 13/85, timing-only queued to
follow automatically. Progressing cleanly, no failures, as of this entry (accuracy sweep at
cell 48/85).

**While relaunched, ran the user's standing request**: the one-off validation exercise on REAL
(not synthetic) data -- `fr_air_temperature_121_1` (121 French weather stations, 212,604
timesteps). Real data tops out at 121 series, below where subsampling naturally engages for a
typical (m, L) in this project's own sweeps, so subsampling was forced deliberately via
`max_feasible_pairs_per_anchor` (the exact same production knob), using
`estimate_proxy_series_subsample_cap`'s own binary search to pick budgets landing exactly on
caps of 30, 60, 90, and 121 (full population, no subsampling) -- config: window_size=256,
window_step=16, L=20 (n_lags=304), corr_threshold=0.7, seed=11, occupancy grid (2,3,5,10), same
gamma trial offsets as production (0.25 down to 0.0 below corr_threshold).

**Result 1 (the question asked)**: `gamma_selected` came back as exactly **0.45** at all four
cap levels (30, 60, 90, 121) -- direct empirical support, on real data, for the assumption the
whole subsampling fix rests on (gamma doesn't depend on series count). Ground-truth count
(`proxy_n_gt`) grew as expected with cap size (51,384 -> 243,427 -> 530,419 -> 524,957) and
recall estimate rose steadily (0.16 -> 0.25 -> 0.35 -> ~0.60 lb/ub), confirming more series ->
more calibration signal, as expected.

**Caveat, checked and disclosed rather than glossed over**: inspected the full cap=121 trial
grid directly -- recall was still climbing monotonically all the way down to the loosest gamma
value actually tested (corr_threshold - 0.25 = 0.45; recall rose from 0.366 at gamma=0.70 to
0.595 at gamma=0.45, not yet plateaued). So every cap level selecting 0.45 is partly a "hit the
edge of the tested grid" artifact, not proof of convergence to a genuine interior optimum --
`GAMMA_TRIAL_OFFSETS` would need widening past 0.25 to see where (or whether) recall actually
plateaus. What IS established: the method's behavior is consistent across cap sizes (no
cap-dependent selection noise), which is the more directly actionable part of the question for
trusting the sweep's own subsampling.

**Result 2, a separate, real, more concerning finding, NOT part of what was asked but surfaced
by the same runs and worth flagging rather than burying**: even at cap=121 (the full real
population, no subsampling at all) with the complete occupancy/gamma grid searched, recall
topped out around **60%** (proxy_recall_ub=0.62) -- nowhere near the 0.95 target, at
corr_threshold=0.7 on this specific real dataset. Real nearby-weather-station data is apparently
far denser/more mutually-correlated than anything the synthetic sweep's corr_prop range (0.001-
0.1) tests, and at this threshold/n_vectors=32 combination the calibration genuinely cannot
reach target no matter what it picks -- not a bug in today's fixes (same behavior at every cap
level, including no subsampling), but a real, disclosed limitation of the sketch/LSH approach
at this density regime. **User has explicitly deferred this for separate follow-up
investigation later** -- not to be conflated with or block the current sweep, which operates in
a different (sparser) density regime where this ceiling has not been observed.

**Verified**: full test suite untouched by this entry (no source changes, diagnostic-only run).
Both sweep chains continuing to progress normally throughout.

**Still pending**: nothing blocking -- sweeps continue to completion, then dashboard rebuild.
Follow-up items for later (both explicitly deferred by the user, not urgent): (1) widen
`GAMMA_TRIAL_OFFSETS` past 0.25 and rerun the cap=121 real-data case to find gamma's actual
plateau, closing the stability-vs-boundary-artifact question cleanly; (2) investigate the ~60%
real-data recall ceiling at corr_threshold=0.7 -- check whether a smaller `corr_threshold`, a
larger `n_vectors`, or a fundamentally different regime characterization is needed for
high-density real data, since the current calibration pipeline was mainly stress-tested against
the sparse synthetic regime this session.

## 2026-09-08 (h) — Accuracy sweep completed (85/85); timing-only sweep hits a real memory ceiling at m=2627

Accuracy sweep finished cleanly, exit 0, all 85 cells. Timing-only sweep (resumed at cell 4/64)
started immediately after per the chain script. Its first cell (m=2627, L=94) crashed with
`MemoryError` -- but NOT during calibration (all 24 hyperopt trials completed successfully
first, confirming subsampling handled this scale fine, as expected: effective_n_series capped
to 179, matching `estimate_proxy_series_subsample_cap`'s prediction exactly). The crash was in
the REAL production run afterward: `SignLSHBandIndex._find_pair_rows_meta`
(`candidate_kernels.pyx:4396`) explicitly raises `MemoryError()` when its internal pair-output
buffer's `realloc()` returns NULL (`_append_pair`, `candidate_kernels.pyx:335`) -- a genuine
allocation failure, not an artificial cap or overflow check.

Root cause: this is the REAL, full-scale (uncapped, not subsampled -- subsampling only applies
to calibration's proxy-anchor reference construction) candidate-retrieval buffer at m=2627, ~9x
the largest scale (m=296) stress-tested so far this session. The worker subprocess's own
self-imposed `RLIMIT_AS` is 3.0GB (`--worker-memory-limit-gb 3.0`, chosen earlier in the
session mainly around calibration-phase memory needs) -- at this much larger production scale,
the real candidate buffer plausibly needs more than that. This directly confirms the
already-flagged-but-not-yet-actioned "Known issues" item: *"revisit the sweep scripts'
`_make_memory_limiter`/`RLIMIT_AS` worker memory limit now that real per-cell memory needs have
shifted."*

**Handled gracefully, not a full crash**: the sweep's own per-cell subprocess isolation
(`_run_worker_subprocess`-equivalent) caught the failed worker's exit code and moved on to cell
5 automatically -- confirmed via process listing and log tail. One cell's data is simply
missing from the final CSV, not a blocked sweep.

**Not yet fixed**: given more large-m cells remain in the timing-only design (up to m=3000), a
higher worker memory limit is likely needed to avoid losing exactly the large-m timing data the
user now explicitly wants for both artifact updates. Plan (not yet executed): let the sweep
continue as-is for now (no need to interrupt), watch for further such failures, then re-run any
failed cells specifically (`--only-cells`) at a carefully-chosen, tested-first higher memory
limit -- NOT a blind relaunch of the whole sweep at a new limit, per the earlier lesson from the
15M-pairs-per-anchor incident (memory-safe changes must be verified before wide rollout, not
assumed from reasoning alone).

**Still pending**: identify all timing-only cells that fail this way once the sweep completes
its current pass; determine and verify a safe higher worker memory limit; re-run failed cells;
then proceed to the (now-expanded) artifact update scope -- both "CorrTrack at Scale"
(`3cd502b5-...`) and "Sobol LSH Sweep" (`0a673d9b-...`) need timing-only data, recall split by
calibration power, and the new `touched_candidates`/`speedup_no_val_no_monitor` metrics, per the
user's explicit request ahead of a presentation tomorrow (2026-09-09).

## 2026-09-08/09 (i) — Second timing-only crash (cell 8, m=1021): wrong hypothesis tried and disproven, real cause identified

A second failure mode appeared alongside cell 4's (genuine large-buffer) crash: cell 8 (m=1021,
L=44) crashed on a TRIVIAL `numpy.core._exceptions._ArrayMemoryError: Unable to allocate 7.48
MiB` immediately after calibration completed (all 24 trials succeeded, then fell back to
`theoretical_sizing_fallback` since nothing was selectable -- itself a normal, disclosed
outcome), right at the start of the real run's first sketch computation.

**First hypothesis, tried and DISPROVEN by direct evidence**: assumed `CorrTrack_optimize`'s
internal object graph (band-index <-> cached-distance <-> trial-record back-references) held
reference cycles that Python's refcounting alone wouldn't free, so calibration's own large
proxy-reference memory would still be resident when the production run started in the same
process. Fix tried: explicit `del optimizer; gc.collect()` on every return path of
`calibrate_via_proxy_hyperopt` (`experiment_lsh_cost_sweep.py`). Verified test suite green
(115/115), then re-ran the exact failing cell (8) end-to-end: **identical crash, same exact
7.48 MiB allocation, same array shape** -- gc.collect() had zero effect.

**Real root cause**: `RLIMIT_AS` (the worker's own self-imposed memory cap, via
`--worker-memory-limit-gb`) limits the process's total *virtual address space*, not
currently-live Python objects. glibc's `malloc` generally does not return freed heap memory to
the OS (no `munmap`/`brk`-shrink on ordinary `free()`), so a process's `RLIMIT_AS` footprint
does not shrink just because Python's own garbage collector reclaims objects -- `gc.collect()`
frees *Python-level* references, not the OS-level address-space reservation glibc's allocator is
still holding for potential reuse. Calibration's own peak virtual-memory footprint at this scale
apparently already sits close enough to the 3GB ceiling that even a fresh few-MB allocation
afterward (needing a new arena/mmap rather than reusable freed space) tips it over.

**The `del optimizer; gc.collect()` fix is being left in place** (harmless, and genuinely
correct for any *actual* Python-level reference-cycle memory the calibration phase holds -- just
not sufficient by itself for the `RLIMIT_AS`/glibc-arena issue actually observed here).

**Real fix being pursued instead**: raise the worker's memory ceiling
(`--worker-memory-limit-gb`) for the timing-only sweep specifically, since it reaches far larger
m/L than the accuracy sweep did and is where both failures occurred. Being tested on the exact
failing cell (`--only-cells 8`) before committing to a full relaunch, per this session's own
standing lesson: a memory-limit change must be measured on the real pipeline before trusting it,
not assumed safe from reasoning alone (see the earlier 2026-09-08 incident where the same
mistake was made and caught the hard way).

**Current sweep state**: killed and relaunched once already (after the gc.collect() fix, to stop
losing cells to the by-then-still-present bug); cells 1,2,3,5,6,7 banked; cells 4 and 8
permanently failed under the 3GB ceiling (both retried under the fix, both still fail -- 4 is
the genuine large-buffer case, 8 is the RLIMIT_AS/glibc-arena case); cell 9 in progress.

## 2026-09-09 (j) — Session interruption and recovery; memory-cap fix verified; presentation artifact rebuilt

**Real session gap**: the background sweep processes were lost for roughly 8 hours (harness
flagged the tracking tasks as orphaned, no completion record -- likely a machine/session
restart, not anything this session did deliberately). All prior progress was safe (written
incrementally to CSV); both sweeps were resumed, not restarted, once noticed.

**Data-integrity fix, user-directed**: cells 1-12 of the accuracy sweep were the stale,
pre-session-rewrite rows (`bf_correlated==1.0` for all of them, producing recall>1.0 --
mathematically impossible, confirmed in the (h)/(i) entries' own analysis). Backed up the CSV,
removed those 12 rows, and re-ran them (`--only-cells 1..12`) under the current codebase. All 12
now valid; accuracy sweep is a full, internally-consistent 85/85 dataset.

**Memory-cap fix for the timing-only sweep, tested and verified**: raised
`--worker-memory-limit-gb` from 3.0 to 5.0 and re-ran the known-failing cell (8: m=1021, L=44)
in isolation first, with live memory monitoring, before committing to a full relaunch (per this
session's own standing lesson about verifying memory changes on the real pipeline). Result:
succeeded cleanly -- wall_time=3,130s (~52min), peak_rss=3.39GB, comfortably under the new 5GB
ceiling, 10.26 billion candidates touched. Merged into the main timing-only CSV and relaunched
the full sweep (resumed) at 5GB. Cell 4 (m=2627, the larger of the two original failures) will
be retried under this same higher ceiling as the resumed sweep reaches it -- not yet separately
verified to fit at 5GB; if it fails again, a further-targeted retry (and possibly a case-by-case
higher ceiling for the largest few cells) is the next step, not a blind further increase.

**Real-time correction from the user, applied**: the "no validation/no monitoring" speedup
comparison originally described as "validation and monitoring excluded from both sides" was
inaccurate -- the actual computation (already correct in the CSV/sweep code) is asymmetric:
CorrTrack's sketch+candidate time vs. brute force's candidate+validation time (monitoring
excluded from both, validation only excluded from CorrTrack's side, since brute force has no
separate filtering step to exclude). Fixed the artifact's wording to match the real computation.
Per further user feedback, this comparison was later removed from the artifact entirely (added
little insight; a real answered secondary question, but not this room's priority).

**Presentation artifact ("CorrTrack at Scale") -- major rebuild, twice**: first refreshed with
corrected data (85 valid points, geometric-mean funnel with a new "touched by LSH" stage, recall
split by calibration confidence, extended first chart to include the timing-only sweep's larger
measured range with brute force shown dashed past its own m=296 limit). Then, per detailed user
direction, fully restructured: reordered sections (extrapolation moved next to the scaling
chart; recall split into its own section; calibration-confidence content moved into a new,
diagram-based Limitations section at the end); added a Sobol/design-of-experiments explainer
with real coverage scatter plots; expanded "under the hood" with an inner candidate-search
pipeline (file into bands -> touch candidates -> gamma gate) and a fixed/derived/tuned parameter
table; split out a dedicated Complexity section with a fixed gamma-glyph rendering issue, a
corrected K definition (number of random vectors, not "hyperplanes"), and a clarified
occupancy-multiplier baseline; added a new Sensitivity section with real quartile-binned trends
(pairs->speedup, density->pre-validation precision, density->recall, threshold->recall,
occupancy->speedup) computed directly from the 85-row dataset; shortened and humanized all copy.
Published in place at the same URL throughout (`3cd502b5-...`).

**Technical dashboard ("Sobol LSH Sweep") rebuild**: scoped as a full rebuild per explicit user
instruction, accepted as unlikely to finish before the presentation. Discovered mid-scoping that
this is not a data refresh: the dashboard's Parameter Sensibility/calibration-reliability/
n_bands-verification tabs are architected around the OLD tolerance x occupancy grid-search
calibration method, while the current sweep uses proxy-anchor hyperopt (occupancy x gamma) --
several major tabs need re-architecture, not just new numbers. Not yet started in earnest;
still queued after the presentation artifact and sweep monitoring.

**Still pending**: finish monitoring the 5GB timing-only sweep to completion (or further
failures); the technical dashboard rebuild; the two previously-deferred follow-ups (gamma-grid
plateau, real-data recall ceiling).

## 2026-09-09 (k) — Presentation artifact: major supervisor-directed revision; real brute-force comparison launched at m~1k/2k

**Cell 4 (m=2627, L=94) still fails at 5GB** -- confirms genuine hardware-scale limit (not the
RLIMIT_AS/glibc-arena bug fixed for cell 8). Cell 20 (m=2267, L=36) succeeded cleanly at 5GB
(wall_time=6,498s, touched=13.4B) -- close in m to cell 4 but far smaller L, confirming L (via
n_bands) drives the memory need at least as much as m alone. Disclosed as a real, accepted
limitation rather than chased further.

**Presentation artifact restructured per the user's supervisors' feedback** (several rounds):
removed the exponent labels from the "What CorrTrack changes" section (kept qualitative,
pointing to the new dedicated Complexity section for exact figures); added back a plain-language
`touched = m*n_bands*occupancy` note to the complexity table; moved the Sensitivity section to
directly follow Recall; removed the 10,000-series projection entirely from "Where this goes" --
it now shows only the range actually tested with CorrTrack (measured throughout, up to the
current max of 2,267 series), with brute force's own side still marked dashed past its true
measured limit (296); replaced the "Speed without shortcuts" funnel (single geometric-mean bars)
with real boxplots (min/Q1/median/Q3/max computed directly from all 85 rows) showing the actual
spread at each of the 4 funnel stages, not just a point estimate; summarized Limitations &
Dependencies down to two compact entries covering BOTH recall (density -> calibration
confidence) and speedup (threshold/occupancy/scale, including the real memory-ceiling finding
from cell 4 as a concrete example) -- previously covered recall only.

**Real brute-force comparison at m~1k/2k, launched (long-running, not yet complete)**: the
timing-only sweep deliberately never runs brute force (that's the point, brute force is
infeasible to run routinely at that scale) -- but the user's supervisors specifically asked for
real brute-force time+accuracy comparisons at the m~1k/2k scale CorrTrack was tested at. Cost
estimated first via the reliable pairs-based brute-force fit (R^2=0.99) before committing to
anything: ~6.1h at cell 8's config (m=1021, L=44) and ~4.4h at cell 13's config (m=1927, L=9) --
chosen (user's pick) as the cheapest real m~1k/2k coverage over the more literal-but-far-more-
expensive alternative (cell 20 at ~25.5h). Implemented by appending these two exact configs as
new cells (86, 87) to the ACCURACY sweep's own design file and running them through the SAME
`run_one_cell` pipeline as the other 85 points (not a bespoke script) -- ensures identical metric
definitions (recall/precision/speedup) and CSV schema, and reuses the existing calibration +
brute-force-comparison machinery already verified correct. Launched concurrently with the
still-running timing-only sweep (not strictly sequential, a deliberate exception to that rule) --
checked host memory immediately after and periodically since (held at ~1.5-1.6GB available,
swap flat at 1.5GB, not growing) before trusting it to run unattended; will revisit if either
sweep's own memory footprint grows further as bigger cells come up.

**Still pending**: both long-running jobs (timing-only sweep resuming past cell 22; the two
brute-force comparison cells, ~10.5h combined estimate) to completion; once cell 86/87 finish,
add their real (m, L, speedup, recall, precision) numbers as overlay points on the "Where this
goes" chart per the supervisors' request; technical dashboard rebuild; the two long-deferred
follow-ups (gamma-grid plateau, real-data recall ceiling).

## 2026-09-09 (n) — Real design bug caught by the user: timing-only sweep's M_RANGE overlaps the accuracy sweep's own coverage

User pointed out the timing-only sweep was never meant to re-cover m<300 -- that range already
has full accuracy-sweep coverage WITH brute-force comparison (85 points); the timing-only sweep
exists specifically to extend CorrTrack's own measured range PAST m=300, where running brute
force would take too long to be practical. `M_RANGE = (50, 3000)`
(`experiment_lsh_sobol_sweep_timing_only.py:87`) was set with no floor at 300, so a large chunk
of its 64-cell design falls in the already-covered range -- real, avoidable wasted compute time,
not a data-quality problem (no wrong data produced, just duplicated effort).

**Measured extent of the waste**: of 26 cells completed so far, 12 had m<300 (wasted -- pure
duplication of accuracy-sweep coverage). Of the 37 cells still pending, 16 more were m<300 (would
also have been wasted); only 21 are m>=300 (genuinely new, useful coverage this sweep exists
for).

**Fixed by redirecting the queue, not by editing the design/M_RANGE constant** (that edit would
hit the exact "fresh-subprocess-per-cell picks up code changes immediately" issue already
flagged for the two currently-pinned topics -- not worth the same risk for a one-off queue
filter). The currently-running cell (28, m=1568) is itself a useful, m>=300 cell -- let it finish
naturally, then killed the orchestrator and relaunched with `--only-cells` restricted to the
exact 21 useful pending cell_ids (28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 47, 48, 49, 50, 52, 53,
56, 57, 60, 61, 64), skipping the 16 wasted ones entirely. The 12 already-completed m<300 rows
stay in the CSV (harmless, just redundant with the accuracy sweep -- not deleted, since deleting
real, valid measurements for no reason would be its own mistake) but are excluded from any
future analysis of the timing-only sweep's OWN unique contribution (m>=300 range).

**Not fixed for next time**: `M_RANGE` itself should be changed to `(300, 3000)` (or just above
296) before this design is ever regenerated fresh -- left as-is for now since the current run is
mid-flight and regenerating the design would invalidate cell_id-based resume tracking; a real
fix for the SOURCE file, not just this run's queue, is a fast follow-up once nothing depends on
the current design file's cell_id numbering.

**Update, same entry**: before the queue redirect actually took effect, two more useful (m>=300)
cells failed at the 5GB cap -- cell 28 (m=1568, L=69, corr_threshold=0.709,
`_ArrayMemoryError: 833MiB for shape (131072, 1666) int32`) and cell 29 (m=2888, L=25,
corr_threshold=0.757, `374MiB for shape (65536, 1497) int32`). Both are a DIFFERENT failure
mechanism than cell 4's (touched-candidate buffer growth): here it's the LSH index's own internal
storage arrays (`_band_keys`/`_membership_pos`, sized `capacity x n_bands`) that ran out of room,
driven by a very large `n_bands` (1666, 1497) -- itself a direct consequence of a LOW
corr_threshold (0.709, 0.757) at large m, matching the exact mechanism already documented in the
presentation artifact's own Limitations section (loose threshold -> larger gamma -> more bands
needed). Confirms the 5GB cap, while a real improvement over 3GB, is still not sufficient for the
lower-threshold corner of the m>=300 range -- now 3 of the useful-range cells have failed (4, 28,
29), all sharing either large touched-candidate volume or large n_bands, both ultimately driven
by scale combined with a loose threshold. Not chased further for now (disclosed, accepted
limitation, consistent with the artifact's own framing); relaunch proceeded past all three.

**Relaunched correctly**: killed the (already cell-30-wasted) run, recomputed the useful-and-not-
yet-attempted cell list (excluding banked cells and all three known failures: 4, 28, 29), and
relaunched via `--only-cells 32 33 36 37 40 41 44 45 47 48 49 50 52 53 56 57 60 61 64` (19 cells)
-- confirmed the very next cell picked up (32, m=609) is a genuine useful, not-yet-attempted
cell, not a repeat of a wasted or failed one.

**Update, still same entry**: 5 more useful-range cells failed the same way as the sweep
continued (36, 44 -- n_bands-driven LSH-index storage growth; 52 -- a tiny 22MB allocation in the
VALIDATION phase, matching cell 8's original RLIMIT_AS/glibc-arena signature exactly, i.e. the
process was already near its 5GB ceiling before validation even started). Total now 6 failures in
the useful (m>=300) range: 4, 28, 29, 36, 44, 52 -- a real ~21% failure rate among useful cells
attempted so far. Not chased further (would need either a higher cap, risking host-level
thrashing on this 7.8GB machine already running the concurrent brute-force comparison job, or a
real fix to the LSH index's memory layout at very large n_bands -- out of scope for right now).
Filtered queue otherwise progressing cleanly; longest single cell so far: cell 45 (m=2113, L=22,
corr_threshold=0.712) at 14,028.7s (~3.9h), touched=35.2B candidates.

**Milestone: first real brute-force comparison cell complete**. Cell 86 (m=1021, L=44,
corr_threshold=0.746, matching the timing-only sweep's own cell 8 config exactly) finished via
the accuracy-sweep's own `run_one_cell` pipeline: CorrTrack wall_time=4,665.2s (~1.3h),
brute-force runtime=28,084.3s (~7.8h) -- close to, though somewhat above, the ~6.1h estimate from
the pairs-based fit (real BF cost sits a bit above the R²=0.99 model's extrapolation at this
scale, a useful calibration data point for that model's own reliability past m=296). Real,
measured result: **speedup=6.02x, recall=85.2%, precision=1.0**. Cell 87 (m=1927, L=9,
corr_threshold=0.908) now running, ~4.4h estimated.

**Timing-only sweep complete**: all 19 filtered cells attempted -- 13 succeeded, 3 more failed
the same way (36, 44 -- n_bands/LSH-index storage; 52 -- tiny-allocation/RLIMIT_AS pattern,
matching cell 8's original signature). Final tally: 48/48 useful (m>=300) cells attempted, 42
succeeded, 6 failed (4, 28, 29, 36, 44, 52) -- confirmed via direct check that zero useful cells
remain un-attempted. Real max m successfully measured: **2,470** (cell 61).

**Presentation artifact refreshed with the completed dataset**: refit the combined
accuracy+timing-only CorrTrack-only curve on the full 127 points (85 accuracy + 42 timing-only,
m 51-2,470) -- R² improved from 0.66 (17-point interim fit) to 0.78 with the additional data,
slope 1.71->1.82. Updated both charts using this fit (`AXIS_MODES.m` and the "Where this goes"
speedup chart), the max-measured-m references (2,267->2,470) throughout, and the "17+ points, m
to 1,927 and counting" stats/footer text to the final "42 points, m to 2,470" (sweep is done, no
longer open-ended). Published in place.

**Still running**: cell 87 (m=1927, L=9, corr_threshold=0.908) of the brute-force comparison job
-- once it finishes, add both cell 86 and 87's real (m, speedup, recall, precision) numbers as
overlay points on the presentation artifact's "Where this goes" chart, per the supervisors'
original request, then the technical dashboard rebuild and the two long-deferred follow-ups
(gamma-grid plateau, real-data recall ceiling) remain the only open items.

## 2026-09-10 (o) — All background jobs complete; brute-force comparison points added to the artifact

Cell 87 finished. Both brute-force comparison cells done, job exited clean:
- **Cell 86** (m=1021, L=44, corr_threshold=0.746): CorrTrack 4,665.2s, brute force 28,084.3s
  (~7.8h), **speedup 6.02x**, recall 85.2%, precision 1.0.
- **Cell 87** (m=1927, L=9, corr_threshold=0.908): CorrTrack 393.5s, brute force 16,817.9s
  (~4.7h), **speedup 42.7x**, recall 86.0%, precision 1.0.

The ~7x spread between them at similar m (6x vs 43x) is a clean, real demonstration of the
threshold/L story the artifact already tells: cell 86 has a low threshold and high L (both push
CorrTrack's own cost curve toward brute force's), cell 87 the opposite. Neither lands near the
fitted model curve -- expected, the model is an average trend across all 127 points, not a
per-config predictor.

**Artifact updated**: added both as real "brute force actually run" blue-dot overlay points on
the "Where this goes" speedup chart (with tooltips giving the exact config and measured speedup),
reworded the section to say brute force was genuinely executed at two more points past m=296
(not just modeled), expanded the chart's y-axis to fit cell 87's 42.7x, relabeled the model
endpoint as "model at m=2,470" to keep model vs. measurement distinct. Published.

**All compute is now done.** No background jobs running. Remaining open items, none blocking:
1. Technical dashboard ("Sobol LSH Sweep") rebuild -- still needs the methodology re-architecture
   flagged in the 2026-09-09 scoping (old tolerance x occupancy calibration tabs vs. the current
   proxy-anchor occupancy x gamma method).
2. Two pinned parameter investigations (2026-09-09 (l)/(m)): OPTIM_PROXY_ANCHOR_EXPAND_FACTOR
   default (2.0 -> lower?), and whether target_occupancy can be made auto/heuristic. Now safe to
   touch code -- no running experiments to disturb.
3. `M_RANGE` in `experiment_lsh_sobol_sweep_timing_only.py` should be changed from (50, 3000) to
   start at ~300 before that design is ever regenerated (the current run's 12 wasted m<300 rows
   stay in the CSV, harmless).
4. The two long-deferred follow-ups: gamma-grid plateau check, real-data recall ceiling.
5. Rebuild the technical dashboard's own data from the final CSVs; the two-artifact comparison
   against preserved old sweep results.

## 2026-09-10 (p) — Presentation artifact: speedup/time modeling switched to a proper multivariate metamodel

User raised the right methodological objection: a Sobol design varies all 4 factors at once, so
you cannot fit a model to "speedup vs. m" in isolation -- every point has a different L, density,
threshold too. The artifact's charts had been using naive marginal log-log fits (ln(time) ~
ln(m) alone), which conflate m's effect with the co-varying factors and fit poorly (CorrTrack
wall-time R²=0.78).

**Fixed**: replaced both charts (Section 2 "What CorrTrack changes" and Section 3 "Where this
goes") with a single multivariate metamodel, its shape taken from the complexity analysis:
`ln(time) = c0 + c1*ln(m) + c2*ln(L) + c3*ln(density) + c4*gamma`, fit by least squares. Two
fits:
- **Brute force** (n=87, the accuracy sweep's 85 points + the 2 real BF-comparison cells 86/87):
  R²=0.993. Exponents m^2.08, L^1.05, density and gamma coefficients ~0.01 -- it's just m²L, as
  theory says.
- **CorrTrack wall time** (n=117: all accuracy points + the 42 useful timing-only cells): R²=0.957.
  Exponents m^1.83, L^0.67, and a large gamma coefficient (7.92) -- the threshold effect (loose
  threshold -> large gamma -> steeper curve), consistent with the artifact's own narrative.

Both chart curves are these models evaluated with L, density, threshold pinned to the sweep
medians (L=23, density 1.0%, threshold 0.83), so only m moves. Speedup = the ratio of the two.
At those medians: ~8x at m=50, ~13x at m=296, ~22x at m=2,470 -- a more honest, lower projection
than the old marginal fit's 28x, because the marginal fit was picking up co-varying effects.

**Both charts now also overlay every measured point** (117 CorrTrack wall times, 87 brute-force
times, 87 measured speedups) as faint scatter, plus the two real BF-comparison runs as larger
labeled dots -- per the user's explicit "include the measured dots together with the fitted model
line". The charts moved to log-log axes to fit the full range (2s to ~1 day). The 3-way axis
toggle (m / measured-only / pairs) was removed -- the scatter now shows directly where
measurement ends, and "pairs (m²L)" as a single axis is less honest than the multivariate model
that handles m and L separately.

**Model construction is documented in the artifact itself**: Section 2 has a collapsible "How
the model was built" reveal with the equation form, the full fitted-coefficient table (both
fits, all 4 exponents, n, R²), the median-holding explanation, and the note that CorrTrack's
leftover 4% variance is mostly calibration run-to-run noise. Published.

## 2026-09-10 (q) — Two pinned parameter topics investigated: anchor-expand-factor is moot; occupancy default should go UP, not down

**`M_RANGE` source fix applied** (`experiment_lsh_sobol_sweep_timing_only.py:87`): 50 -> 300,
with a comment explaining why (only extends past the accuracy sweep's own ceiling; m<300 is
pure duplication). Current run's design file already generated, so unaffected; a fresh
`--fresh-design` would now use the corrected range.

**`OPTIM_PROXY_ANCHOR_EXPAND_FACTOR` -- moot for this sweep**: checked every one of the 88
trial-grid CSVs; `proxy_anchor_expansion_iteration` is 0 for all of them. The adaptive
anchor-expansion loop in `_get_proxy_anchor_optim_params` never triggered a second round
anywhere in the sweep -- so the expand factor (2.0) had zero effect on any result. The
supervisor's concern about doubling being "too much" is reasonable in principle, but there's
nothing to A/B against in this data. Note for whoever revisits it: the loop RE-RUNS the entire
24-trial grid each expansion round, so a smaller factor = more rounds = more total re-runs =
MORE cost, not less. If the goal is cost, doubling is arguably right; if the goal is not
overshooting the anchor count, a smaller factor helps but pays for it in re-runs. Needs a
dedicated experiment on cells where expansion actually fires (very-low-density, adequate
row-budget) -- none of this sweep's cells qualified.

**`target_occupancy` auto-selection -- ran a real controlled experiment**
(`/tmp/.../scratchpad/occupancy_experiment.py`): fixed (L=20-30, threshold=0.83, density=0.02,
gamma=threshold, n_vectors=DEFAULT), swept occupancy in {2,3,5,10} with n_bands derived from the
closed-form formula each time, at m in {100, 200, 300}, measuring real speedup + recall vs.
brute-force ground truth. Result, consistent across all 3 scales:

| occ | m=100 | m=200 | m=300 | recall (2->10) |
|-----|-------|-------|-------|----------------|
| 2   | 5.6x  | 5.3x  | 2.5x  | m=100: 0.80->0.80 |
| 3   | 5.6x  | 4.9x  | 4.6x  | m=200: 0.79->0.79 |
| 5   | 7.1x  | 7.3x  | 6.2x  | m=300: 0.77->0.77 |
| 10  | 7.9x  | 7.6x  | 7.1x  | |

**Higher occupancy is reliably FASTER, the gap widens with m (at m=300, occ=10 is 2.8x faster
than occ=2), and recall does not move.** Mechanism: low occupancy -> many more bands ->
band-filing cost (candidate_time) explodes (43s at occ=10 vs 126s at occ=2 for m=300); the
extra touched candidates from high occupancy don't offset it.

**This DIRECTLY CONTRADICTS the confounded sweep observation** (which showed occ=3 faster, 14.6x
vs 8.0x median) -- that was selection bias, since occupancy is TUNED, so cells that got occ=3
selected were systematically different (smaller/sparser), not occ=3 causing the speedup.
Correcting a real misconception here.

**Recommendation** (not yet applied, needs 2-3 more (density, threshold) points first since this
is one point in that plane -- high density could shift the tradeoff via more real positives ->
more validation cost): drop `OCCUPANCY_GRID` to `(5, 10)` or just default to occ=10. NOT a
threshold-based formula -- higher is simply better across the tested range, no formula needed.
The earlier speculation about "occupancy as a function of threshold" (entry (m)) is not
supported -- the effect isn't threshold-dependent in the data, it's a straightforward
band-filing-cost story.

## 2026-09-10 (r) — Both long-deferred follow-ups resolved: gamma grid is fine; the "recall ceiling" was a calibration-estimate artifact

Ran one experiment (`/tmp/.../scratchpad/realdata_gamma_recall.py`) covering both: real
`fr_air_temperature` 121-series data, L=20, occupancy=5, at corr_threshold in {0.70, 0.80, 0.90},
sweeping candidate_cosine_threshold (gamma) from 0.20 up to = threshold, real recall measured
against full brute-force ground truth. Effective density (fraction of tested windows actually
correlated above threshold): 4.0% at 0.70, 1.2% at 0.80, 0.22% at 0.90 -- genuinely dense data
at the lower thresholds, unlike the sparse synthetic sweep.

**Recall vs gamma, real (not proxy-estimated):**

| corr_threshold | plateau recall (gamma <= threshold-0.25) | at offset 0.25 | at offset 0.00 |
|----------------|------------------------------------------|----------------|----------------|
| 0.70 | ~97.5% | 97.3% | 76.1% |
| 0.80 | ~97.1% | 97.0% | 74.5% |
| 0.90 | ~96.8% | 96.8% | 72.0% |

Precision was 1.000 at every single point.

**(1) Gamma-grid plateau -- RESOLVED, no change needed.** Recall plateaus at ~97% well inside
the sweep's own gamma range; `GAMMA_TRIAL_OFFSETS`'s loosest offset (0.25) already lands on that
plateau at every threshold, comfortably above the 0.95 target. The plateau widens with
threshold (at 0.90 recall is flat ~96.8% from gamma 0.20 all the way to 0.75). No need to
extend `GAMMA_TRIAL_OFFSETS`.

**(2) Real-data ~60% recall "ceiling" -- RESOLVED, was NOT real.** The earlier gamma-stability
exercise's ~0.60 figure was `proxy_recall_lb/ub` -- the proxy-anchor calibration's OWN recall
ESTIMATE, off a handful of anchors -- not measured recall. Real recall at the exact gamma that
exercise flagged infeasible (0.45 at threshold 0.70) is 97.3%. The proxy-anchor recall estimate
is unreliable on dense data: it systematically UNDER-estimates (whereas on sparse data it
degenerates to a 0/1 coin flip). Two different failure modes, same root cause -- too few local
anchors to estimate a rate.

**Two real, actionable issues did surface:**
- `theoretical_sizing_fallback` (`experiment_lsh_cost_sweep.py`) sets `gamma = corr_threshold`
  exactly, no offset. On dense real data that gives only ~72-76% recall -- well below target.
  When hyperopt finds nothing selectable and this fallback fires, it should apply a modest
  offset (~0.15-0.25 below threshold), matching where the real recall plateau starts. NOT yet
  changed.
- The proxy-anchor calibration can wrongly flag a genuinely-feasible config as infeasible on
  dense data, because its recall estimate runs low. This is why the earlier gamma-stability run
  saw `proxy_feasible=False` everywhere on real data even though real recall was ~97%. A more
  robust estimate (more anchors, or a bias correction for the dense regime) would help. NOT yet
  addressed -- a real limitation of the calibration mechanism, now documented.

**Filter still does real work on dense data**: at target recall it tests 8-17% of all pairs at
threshold 0.7-0.8 (a 6-12x reduction), 2-6% at threshold 0.9 (17-50x) -- far from the
millions-to-one on sparse synthetic data, but real, and precision stays perfect throughout.

**All open topics from the 2026-09-10 (o) list now either done or explicitly parked:**
1. Technical dashboard rebuild -- PARKED by user ("do (c)"), the two follow-ups came first.
2/3. Pinned parameter topics + M_RANGE -- done (entry (q)).
4. Both long-deferred follow-ups -- done (this entry).
5. Occupancy-grid change -- recommendation stands (default high / drop the low end), one more
   validation step queued (rerun the occupancy sweep at effective density ~0.001 and ~0.1 to
   confirm "higher is better" doesn't flip at high density -- the user's effective-density idea).

## 2026-09-10 (s) — Occupancy x density validation: "higher occupancy = faster" holds across every density the synthetic generator can reach

Ran the queued validation point (`/tmp/.../scratchpad/occupancy_density_experiment.py`): fixed
m=200, L=20, corr_threshold=0.83, gamma=threshold, n_vectors=DEFAULT; swept occupancy in
{2,3,5,10} at three requested densities corr_prop in {0.001, 0.02, 0.1}, real speedup + recall
vs. brute-force ground truth each time.

| corr_prop | true pairs | occ=2 | occ=3 | occ=5 | occ=10 | recall (2->10) |
|-----------|-----------|-------|-------|-------|--------|----------------|
| 0.001 | 60    | 4.99x | 4.21x | 7.51x | 8.99x | 0.767 flat |
| 0.02  | 1301  | 3.96x | 4.87x | 7.28x | 8.55x | 0.793 -> 0.792 |
| 0.1   | 6429  | 5.15x | 4.87x | 7.42x | 8.60x | 0.765 -> 0.764 |

**Same conclusion as entry (q), now across a 100x range of requested density: occ=5 and occ=10
dominate occ=2/3 at every density, recall is flat, and the gap does not shrink as density
rises.** The low end (occ 2 vs 3) is mildly non-monotonic (noise in band-filing vs
touched-candidate tradeoff), but 5 and 10 are unambiguous. Candidate_time falls monotonically
with occupancy at every density (e.g. cp=0.1: 17.8s -> 9.3s from occ=2 to occ=10); validation
time barely moves (0.18-0.23s) because precision stays ~1.0 so more true positives cost almost
nothing to validate.

**Caveat, stated plainly:** the synthetic generator's *effective* density (fraction of tested
window-pairs actually correlated above threshold) tops out at ~1e-05 even when asked for
corr_prop=0.1 -- it cannot manufacture the genuinely dense regime that real weather data shows
(entry (r) measured 4% effective density on `fr_air_temperature` at threshold 0.7). So this
experiment rules out a density-driven flip *within the synthetic-reachable range* only. The one
dense-regime data point available (entry (r), real data, occupancy fixed at 5) showed recall
plateauing at ~97% with precision 1.0 -- consistent with occupancy not hurting -- but occupancy
was not swept there. A real-data occupancy sweep would close this fully; not blocking the
recommendation.

**Presentation artifact first chart** (`corrtrack_at_scale.html`, URL 3cd502b5): replaced the
3-way scale toggle with a 4-way one -- measured-range/full-range crossed with true-scale/log-scale,
in that button order, default = measured-range + true-scale (only real data, no model line past
m=296). Fixed the left-padding bug (linear x-axis now starts at m=50 in measured mode / m=45
full, so the curve meets the y-axis) and the measured-mode y-axis ceiling (was the median-L
model value ~900s, clipping the real m=296 brute-force dots at ~2150s; now scales to the actual
max measured point). Published.

**Recommendation now considered validated enough to apply:** default `target_occupancy` to 10
(or set `OCCUPANCY_GRID = (5, 10)` if the grid is kept). Higher is simply better across every
density and scale tested; no threshold- or density-dependent formula is warranted. The
"estimate density from tuning data and pick occupancy from it" idea (user's suggestion) is not
supported by the evidence -- there is no crossover to steer around. NOT yet applied to
`OCCUPANCY_GRID` / the default; that is a one-line change left for the next session to make
deliberately (it changes every future sweep's calibration search).

## 2026-09-10 (t) — CORRECTION: the "higher occupancy = faster" conclusion (entries (q)+(s)) was WRONG. On real dense data the optimum is LOW occupancy (2-3). Do NOT raise the default.

Two compounding bugs invalidated entries (q) and (s):

1. **Wrong `n_vectors`.** All three prior occupancy experiments (`occupancy_experiment.py`,
   `occupancy_density_experiment.py`, and the first pass of the ceiling experiment) passed
   `n_vectors = experiment_lsh_cost_sweep.DEFAULT_N_VECTORS = 32`. The actual Sobol/OA sweep and
   production use **64** (`FIXED_N_VECTORS`). With only 32 vectors, the low-occupancy regime
   forces `band_width` to 10-11 -- a third of the vector budget -- and the overlap-correction
   penalty in `compute_lsh_n_bands` explodes `n_bands` (1637 bands at occ=2, threshold 0.72, vs
   292 at the same point with n_vectors=64). That made low occupancy look catastrophically slow.
   An artifact of the wrong constant, not an occupancy law.
2. **Degenerate synthetic density.** The synthetic generator's effective density (fraction of
   tested window-pairs actually correlated above threshold) caps at ~1e-5 no matter what
   `corr_prop` is requested. At that density touched candidates never approach the brute-force
   universe regardless of occupancy, so the touched-candidate ceiling never engages -- the only
   visible effect was band-filing cost, which alone favors high occupancy. Real data does not
   behave this way.

**Corrected experiment** (`/tmp/.../scratchpad/occupancy_ceiling_realdata.py`): real
`fr_air_temperature` 121 series, L=20, **n_vectors=64**, gamma = corr_threshold - 0.20,
occupancy in {2,8,32,64,128,256,512}, at three real density regimes set by corr_threshold.
`band_width` runs from 11 (occ=2) down to its floor of 3 (occ>=512).

| corr_threshold | eff. density | best occ | occ=2 | occ=8 | occ=64 | occ=512 | touched/bf_tested, occ 2 -> 512 |
|----------------|--------------|----------|-------|-------|--------|---------|---------------------------------|
| 0.88 | 0.34% | **2**   | 3.29x | 3.24x | 2.89x | 2.43x | 0.21 -> 0.83 |
| 0.80 | 1.20% | **8**   | 1.38x | 1.48x | 1.44x | 1.27x | 0.34 -> 0.91 |
| 0.72 | 3.25% | (loses) | 0.73x | 0.62x | 0.69x | 0.81x | 0.48 -> 0.93 |

**Findings:**
- **Optimum occupancy is LOW: 2 on sparse data, ~8 in the middle.** Every step above that trades a
  shrinking band-filing saving for a growing touched-candidate cost. At threshold 0.88, occ=512
  is 26% slower than occ=2; at 0.80, occ=512 is 14% slower than the occ=8 peak. This is the exact
  opposite of entries (q)/(s).
- **`touched_candidates / bf_tested` rises monotonically with occupancy at every density** -- from
  ~0.2-0.5 at occ=2 to 0.83-0.93 at occ=512. At the `band_width=3` floor the LSH prefilter has
  essentially stopped filtering: CorrTrack touches 83-93% of the entire brute-force universe.
  This is the quadratic-blowup the user predicted ("touched pairs cannot go close to brute-force
  universe or we have quadratic complexity"). It is real and it is what bounds occupancy from
  above.
- **Recall flat ~0.97-0.98 and precision exactly 1.0 at every occupancy and threshold.**
  Occupancy is purely a speed knob; the gamma offset (0.20) holds recall on its plateau
  independently.
- **The confounded sweep's raw observation (entry (q): occ=3 median faster than occ=10, 14.6x vs
  8.0x) was directionally CORRECT.** Calling it "pure selection bias" in entry (q) was itself
  wrong -- the selection bias is real but it sits on top of a real effect (low occupancy is
  faster). Both were pointing the same way.
- **Method density ceiling (separate finding):** at 3.25% effective density (threshold 0.72),
  CorrTrack is net *slower* than brute force at every occupancy -- validation dominates (val_t
  ~14-16s of ~25-32s wall) because ~2.4M windows validate as real. When a large fraction of
  candidates are true positives, filtering cannot buy enough advantage to cover the sketch+LSH
  overhead. This is at m=121; the crossover density rises with m (brute force grows m^2 while
  CorrTrack's validation is bounded by the true-positive count), so it is not a fixed limit, but
  it is a real regime boundary worth stating.

**Recommendation, corrected:** do NOT raise the occupancy default or `OCCUPANCY_GRID`. The
current default `target_occupancy = 3.0` is well placed. If anything the grid's high end (10.0)
is the questionable value on real data, not the low end. Keep the grid spanning low values
`(2, 3, 5, 10)` so the tuner can pick 2-3 where that is best. No density-based auto-selection:
the useful range is narrow and low, and picking `min` would be nearly as good as any formula.

**Caveat:** real-data evidence is m=121 only (the dataset's full width). The *direction* is
mechanistically robust to m -- the touched-candidate term carries the m^2 and only worsens the
high-occupancy penalty at larger m -- but the exact optimum (2 vs 8) could shift. The synthetic
generator cannot corroborate at larger m because it cannot reach real effective density.

**Entries (q) and (s) should be read as superseded by this one.** `THEORETICAL_FALLBACK_GAMMA_
OFFSET` and the gamma-offset work in entry (r) are unaffected -- those held up.

## 2026-09-10 (u) — Gamma search space made explicitly offset-based (was already, now documented + locked)

Per the user: "hyperopt should not sweep through gamma but a list of GAMMA_OFFSETs based on the
corr_threshold." It already does -- both `calibrate_gamma` (per-cell vs. real brute force) and
`calibrate_via_proxy_hyperopt` (proxy-anchor) build their gamma trial set as
`gamma = max(0, corr_threshold - offset)` for `offset in GAMMA_TRIAL_OFFSETS`
`= (0.25, 0.20, 0.15, 0.10, 0.05, 0.0)`. The param grid handed to `CorrTrack_optimize` carries
resolved absolute `candidate_cosine_threshold` values only because that CorrTrack parameter is
itself absolute -- there is no offset knob inside CorrTrack.

Changes (minimal, `experiment_lsh_cost_sweep.py`):
- Rewrote the `GAMMA_TRIAL_OFFSETS` comment to state the contract: the search space is defined
  ONLY as offsets below `corr_threshold`, never as absolute gamma values; the no-search fallback
  takes the single `THEORETICAL_FALLBACK_GAMMA_OFFSET` (0.20) out of the same tuple.
- Added `assert THEORETICAL_FALLBACK_GAMMA_OFFSET in GAMMA_TRIAL_OFFSETS` so the search and
  no-search paths cannot drift onto different gamma ladders.
- Tests: 115/115 pass.

Not renamed to `GAMMA_OFFSETS` (would touch 6 files that import the name); flagged for the user
to decide.

## 2026-09-10 (v) — `candidate_cosine_threshold_offset`: a real CorrTrack parameter for offset-based gamma; `experiment_run_param_grid.py` switched to it

Per the user: the production param grid should set the retrieval gate as a MARGIN below
`corr_threshold`, not a hard-coded absolute cosine value (a grid pinned at `0.55` is a silent
0.35 offset at threshold 0.90).

New parameter `candidate_cosine_threshold_offset` (default `None`), threaded through:
- **`CorrTrack.__init__`** (`library_corrtrack_parallel.py`): added to the signature; resolved
  in `__init__` where `corr_threshold` is known -- `candidate_cosine_threshold = max(-1, min(1,
  corr_threshold - offset))`. Precedence: an explicit absolute `candidate_cosine_threshold`
  still wins; the offset only fills in when the absolute is `None`; if neither is set, the old
  `_candidate_gamma_from_tau` default is unchanged.
- **`_extract_feature_overrides`**: passes the offset through raw (like `data_representation` /
  `candidate_backend`), so every grid-driven path (`execute_corrtrack_pass` at line ~1290,
  `CorrTrack_optimize.get_optim_params` via `_extract_feature_overrides(bst)`) resolves it
  inside `CorrTrack.__init__`.
- **`OPTIM_RESULT_COLUMNS`**: added `candidate_cosine_threshold_offset` column; the optim
  record now logs both the winning offset and the resolved absolute gamma
  (`candidate_cosine_threshold`). `RUN_RESULT_COLUMNS` / `COMPARISON_COLUMNS` left alone (their
  row builders assert exact length; not the path the user's grid uses).
- **`experiment_run_param_grid.py`**: `"candidate_cosine_threshold": [0.55]` ->
  `"candidate_cosine_threshold_offset": [0.20]` (the measured plateau-entry offset, entry (r)).
- **`experiment_lsh_cost_sweep.py`**: `calibrate_via_proxy_hyperopt` now builds its search as an
  explicit `{offset: gamma}` map over `GAMMA_TRIAL_OFFSETS`, resolves to absolute only at the
  `CorrTrack_optimize` handoff, and returns the winning `gamma_offset`. `theoretical_sizing_
  fallback` returns `gamma_offset = THEORETICAL_FALLBACK_GAMMA_OFFSET`.
- **`experiment_lsh_sobol_sweep.py`**: `CSV_COLUMNS` gains `gamma_offset_selected`; the row
  records it from the hyperopt/fallback result.
- **Test**: `test_candidate_cosine_threshold_offset_resolves_against_corr_threshold` (offset
  resolves; absolute wins when both set; neither -> non-None default; override passes through).
  Suite 116/116.

## 2026-09-10 (w) — The Sobol sweep tested an unrealistically sparse regime; real-data 2D sweep launched (option C / arm A)

Checked every accuracy-sweep cell's EFFECTIVE density (`bf_correlated / bf_tested` -- the
fraction of tested pair/lag/window tuples genuinely correlated above threshold, the quantity
that governs how much filtering can help):

| | effective density |
|---|---|
| Sobol sweep, min | 1.1e-8 |
| Sobol sweep, median | 4.3e-7 |
| Sobol sweep, max (densest cell) | 1.5e-5 |
| real `fr_air_temperature`, threshold 0.88 | 3.4e-3 |
| real `fr_air_temperature`, threshold 0.72 | 3.3e-2 |

**The densest cell in the entire sweep is ~230x sparser than real weather data at a strict
threshold and ~2000x sparser at a loose one; the median cell is ~1e4-1e5x too sparse.** Cause:
the generator's `corr_prop` is `achieved_z` (fraction of individual (series, timestep) SAMPLES
inside a planted correlated segment), and each planted pair shares one `template_len`-long
correlated window at one position, so the fraction of tested PAIR comparisons that are real
stays microscopic regardless of `corr_prop`. Confirmed earlier: effective density caps ~1e-5
even at `corr_prop = 0.1`.

What generalises from the sweep: recall (~96% sweep vs ~97% real dense). What does NOT:
speedup (sweep 5-64x; real dense data at m=121: 0.7-3.3x, below 1x at ~3% density) and the
phase mix (validation-dominated at real density, candidate-search-dominated in the sweep).
The presentation artifact's "correlation density" axis is `achieved_z` (0.1-10%), not effective
density -- needs relabelling + a caveat that dense-data behaviour is spot-checked only.

**Arm A launched** (`/tmp/.../scratchpad/realdata_2d_sweep.py`): real `fr_air_temperature`,
m in {40,70,100,121} (nested subsample) x corr_threshold in {0.70,0.78,0.86,0.94} (each a
different real effective density), n_vectors=64, per cell: brute-force truth -> calibrate
(occupancy {2,3,5} x gamma_offset {0.15,0.20,0.25}) against truth -> full run at the winner,
measuring speedup / recall / precision / per-phase time / touched-over-bf. Capped at m=121
(the only real dataset is 121 FR stations; `fr_wind_direction` is a second 121-station set).

**Arm B (generator fix) -- scoping, not started.** The generator needs a mode where planted
correlation persists across many windows (or many stacked templates per pair) so effective
density can reach 1e-3..1e-1, then a small density-varying synthetic sweep at m > 121.

## 2026-09-10 (x) — Arm A complete: on real-density data the method's accuracy holds but speedup only clears 1x below ~1.5% effective density

`/tmp/.../scratchpad/realdata_2d_sweep.py` -> `realdata_2d_sweep_results.json`. Real
`fr_air_temperature`, m in {40,70,100,121} x corr_threshold in {0.70,0.78,0.86,0.94},
n_vectors=64, per cell: brute-force truth -> calibrate (occupancy {2,3,5} x gamma_offset
{0.15,0.20,0.25}) against truth -> full run at the winner.

| eff. density | m=40 | m=70 | m=100 | m=121 |
|--------------|------|------|-------|-------|
| ~4%   (thr 0.70) | 0.52x | 0.70x | 0.66x | 0.81x |
| ~1.5% (thr 0.78) | 0.77x | 1.27x | 1.53x | 1.68x |
| ~0.5% (thr 0.86) | 1.46x | 2.52x | 3.22x | 3.15x |
| ~0.06% (thr 0.94) | 2.47x | 4.63x | 7.14x | 7.07x |

- **Recall 0.967-0.981 and precision exactly 1.000 in every one of the 16 cells.** The method's
  correctness is not in question at real density -- only its speed.
- **Effective density is the dominant axis; m barely matters.** At a fixed density the speedup
  is roughly flat across m (slight improvement at the sparse end). Speedup clears 1x only below
  ~1.5% effective density; at ~4% (threshold 0.70) it is a net slowdown at every m.
- **Phase mix flips with density**: validation-dominated at ~4% (e.g. m=121: sk/ca/va/mo =
  0.47/3.75/14.37/3.52), candidate/sketch-dominated below ~0.5%. `touched/bf_tested` runs
  0.60 -> 0.12 as density drops.
- Calibration selected occupancy 2-5 and gamma_offset 0.15-0.20 throughout -- consistent with
  entries (t) and (r).
- **Root cause of the high-density slowdown** (from the implied per-pair costs): CorrTrack
  validates ~5x fewer pairs than brute force tests at ~4% density, but each validation costs
  ~5x more (BF's exhaustive Cython kernel amortizes rolling-window statistics across a perfectly
  regular iteration; CorrTrack's validator gets an irregular candidate set and can't), so the
  two cancel, and the LSH touch/dot-check (~60% of the universe) + sketch + monitor tip it
  negative. The filter only wins when it cuts the validation set by orders of magnitude
  (sparse), not ~5x. Optimization avenue: reuse the sketch stage's incremental window stats in
  validation.

**Bottom line for the write-up:** the Sobol sweep's 5-64x speedups are real but only for the
~1e-5-density regime it tested; at the 1e-3..1e-2 effective density real weather data shows,
speedup is ~1-3x and turns negative above ~1.5%. Recall/precision generalise; speedup does not.
The presentation artifact's density axis and speedup framing need this caveat.

**Arm B launched** (`/tmp/.../scratchpad/arm_b_dense_sweep.py`): B2 cluster synthetic generator
(each series correlated to at most one cluster head, so planted correlations aren't clobbered),
m in {150,300} x {small,large} clusters at threshold 0.80, same calibrate->run harness.

v1 (low-pass-smeared, phi=0.85) was killed after the first cell -- measured effective density
only 2e-4 (each correlated pair's own "self density" over lags/windows was ~2.6%, too low).
v2 fix: phi=0.95 so the base process's own autocorrelation gives each correlated pair a
natural ~4-lag band above the 0.80 bar with no smoothing (corr at lag k ~ RHO*phi^k), plus
persistence (every window) -> ~15-25% self density; clusters sized so 3.5-12% of all series
pairs are correlated -> target effective density ~0.6-2.5%. Still measured from brute force,
not assumed. Generator sanity-checked: planted within-cluster pairs reach windowed |r| ~0.89.

## 2026-09-10 (y) — Arm B complete: m up to 300 confirms density (not m) governs speedup; synthetic dense regime caps at ~0.2% and under-recalls

`arm_b_dense_sweep_results.json`. Cluster synthetic (n_vectors=64, threshold 0.80):

| m | eff. density | speedup | recall | precision | dominant phase |
|---|--------------|---------|--------|-----------|----------------|
| 150 | 0.067% | 7.17x | 0.935 | 1.0 | candidate |
| 150 | 0.226% | 4.86x | 0.935 | 1.0 | candidate |
| 300 | 0.052% | 6.74x | 0.933 | 1.0 | candidate (20.5s / 24s) |
| 300 | 0.122% | 6.34x | 0.940 | 1.0 | candidate |

- **Speedup at m=150-300 matches arm A at the same density** (arm A m=100-121 at 0.05-0.5%:
  3-7x). The m-extension confirms arm A's core finding: effective density, not m, governs
  speedup; extending m does not degrade or materially improve it at fixed density.
- **At larger m the candidate-search phase dominates** (m=300: 20.5s of 24s), not validation --
  a different bottleneck from the dense-small-m regime where validation dominates.
- **Recall stuck at 0.933-0.940 (below the 0.95 target) on every synthetic cell**
  (`calib_met_target=False` throughout), while arm A's real data hit ~0.97 at the same
  densities. This is a generator artifact: the broadband-cluster correlations place many pairs
  right at the threshold where a lossy 64-dim sketch loses them; real weather correlations sit
  further above threshold. NOT a method weakness.
- The generator still could not cleanly reach the 1-3% effective-density regime (topped out at
  0.23%). The definitive dense-regime evidence therefore remains arm A's real-data m<=121 cells
  (speedup 0.7-1.7x at 1.5-4% density, recall ~0.97, precision 1.0).

**Option C combined verdict:** the method's accuracy (recall ~0.97, precision 1.0) holds across
the full real-density range; speedup is 5-64x only in the ultra-sparse regime the Sobol sweep
tested (~1e-5), ~1-3x at real 1e-3..1e-2 density, and negative above ~1.5%; and none of this
changes with m up to 300. The Sobol sweep's headline numbers are a best-case (sparse) regime.

## 2026-09-10 (z) — Validation prefix-sum-moments optimization: tried, VERIFIED bit-identical, but a net LOSS -- reverted

Hypothesis (from entry (x)): CorrTrack's exact-Pearson validator re-accumulates each window's
first four raw moments per candidate row (O(w)), while brute force reads them in O(1) from
per-step prefix sums; giving CorrTrack the same prefix-sum moments should close the ~4-5x
per-pair gap and make the speedup comparison "fair".

Implemented: `validate_corr_rows` (candidate_kernels.pyx) gained optional prefix_sum/_sq/_cu/_qu
args (padded cumsums of data / data^2 / data^3 / data^4, the exact scheme Candidates_BF._window_
prefixes uses); when supplied, both sides' moments are prefix diffs and the per-row loop does
ONLY the cross term sum_xy. Python side: `_validation_window_prefix_sums()` (cached per step),
flag `validation_incremental_moments` default True, wired into the sequential Pearson path.

**Correctness: verified.** Correlated-set bit-identical (symdiff 0) vs the old path on real
`fr_air_temperature` at m=90 and m=121, thresholds 0.70-0.94. Recall/precision unchanged.

**Performance: net loss, measured (m=121, n_vectors=64, real data):**

| threshold | eff. density | val_t OFF | val_t ON | wall OFF -> ON |
|-----------|--------------|-----------|----------|----------------|
| 0.70 | 4.0% (15.1M rows) | 15.98s | 16.71s (+5%) | 33.8s -> 30.3s (noise) |
| 0.94 | 0.07% (1.1M rows) | 0.48s | 1.14s (+136%) | 2.4s -> 3.6s (-50%) |

Why it failed: validation is **memory-bandwidth bound on the cross-term dot** (loading 2x256
strided doubles per row), NOT compute-bound on the moments. Removing the moment flops doesn't
help when you are waiting on memory anyway, and the prefix path ADDS traffic (8 prefix reads/row
+ building 4 cumsum arrays/step -- a fixed per-step tax that dominates when row count is low).

**Deeper finding -- the "unfair comparison" framing does not hold.** Brute force's amortization
(its incremental m x m dot-matrix, O(m^2 * window_step) per lag serving all m^2 pairs) is
available *precisely because* it processes every pair. CorrTrack cannot amortize the cross term
the same way without also computing all pairs -- which IS brute-force candidate cost. At high
density, where the candidate set is a large fraction of all pairs, brute force's regularity is a
genuine algorithmic advantage, not an implementation artifact. The measured per-pair gap
(~1057 ns/row validation vs ~262 ns/test brute force, ~4x) is partly real implementation slack
(the strided per-row dot doesn't vectorize as well as brute force's blocked matmul), but
closing it fully would require lag-grouped/blocked validation (medium-risk restructuring) or the
already-existing `hybrid_validation` cross-step incremental path (opt-in, only helps recurring
pairs) -- NOT the prefix-sum-moments idea.

**Reverted** in full (candidate_kernels.pyx + library_corrtrack_parallel.py), .so rebuilt,
suite 116/116. No behavior change from this entry.

## 2026-09-10 (bb) — Numeric correlated-pair accumulator: IN PROGRESS. Byte-identical output, 2.5x faster CorrTrack at 4% density, speedup flips 0.75x -> 1.47x

Implementing the approved plan (`~/.claude/plans/misty-stargazing-liskov.md`). Steps 1-5 landed:
- `_canonicalize_rows(rows)` -- vectorized `_normalize_bf_key` (`swap = (t1<t2)|((t1==t2)&(s1>s2))`),
  `_rows_as_void_keys` for numpy row-set ops. Unit test vs the scalar fn (12k rows incl. edge
  cases) passes.
- Persistent numeric accumulator `_correlated_rows/_corrs/_count` + `_append_correlated_numeric`
  (canonicalize + amortised-doubling append, no per-pair Python).
- `_record_correlated_numeric` hot path routes to the accumulator; the string
  `_numeric_rows_to_pair_map`/`_record_correlated_batch`/`_update_maxlag_state*` calls are gone
  from it. `CORRTRACK_LEGACY_CORRELATED=1` env var keeps the old path for A/B.
- `_maxlag_state` left empty on the hot path; `_save_max_lag_correlated` already has an
  equivalent rebuild-from-`self.correlated` fallback -- no new code.
- `self.correlated` is now a `@property` (+ setter) that lazily materialises the string-tuple
  dict from the accumulator on access, re-canonicalising on the string LABELS so the view is
  byte-identical to the old `_numeric_rows_to_pair_map` output. `correlated_rows()` returns the
  raw `(rows, corrs)` (index-canonical). The 2 legacy `self.correlated.update(...)` call sites
  now write through `_correlated_legacy_writethrough`.

**Equivalence verified** (real `fr_air_temperature` m=121, thr 0.70 and 0.94, numeric vs
`CORRTRACK_LEGACY_CORRELATED=1`): correlated key-SET identical (symdiff 0), recall / precision /
f1 identical to 5 dp.

**Perf (m=121, thr 0.70, ~4% effective density, NS=300):**

| | legacy | numeric |
|---|--------|---------|
| CorrTrack wall | 16.9s | **6.8s** (2.5x) |
| CorrTrack validation_time | 8.87s | 1.65s |
| CorrTrack bookkeeping | 2.20s | 0.12s |
| brute force runtime | 12.7s | 10.0s (1.27x) |
| **speedup (BF / CorrTrack)** | **0.75x** | **1.47x** |

Answers the "does BF benefit too, so the comparison is unchanged?" concern: BF does benefit
(~1.3x) but CorrTrack benefits ~2.5x, because the per-pair Python recording was a far larger
fraction of CorrTrack's total (it does ~10x less validation-kernel work). At 4% density
CorrTrack flips from a net slowdown to a ~1.5x speedup. Recall/precision unchanged.

**Steps 6-9 also landed:**
- (6) `compute_metrics_bf_numeric` -- pure-numpy replica of `_compute_metrics_bf_windows` over
  numeric `(rows, corrs)`: canonicalize both, view each row as an opaque 40-byte void key,
  `np.isin` set arithmetic, pos/neg by sign. `compute_metrics_bf` auto-dispatches when neither
  input is a dict. `aucroc`/`pr_auc` left nan on this path (unused downstream). Matches the dict
  path to 9 dp on a random-overlap test (new `test_compute_metrics_bf_numeric_matches_dict_path`).
- (7) `execute_corrtrack_pass` / `CorrTrack_optimize._mode_run` return `corr_flags` as a
  `NumericCorrelatedFlags` wrapper (numeric `(rows,corrs)`, `len()` == n correlated windows,
  `.correlated_rows()`, tuple-unpacks). The 5 sweep call sites
  (`experiment_lsh_{cost,sobol,nbands_occupancy,orthogonal_array}_sweep.py`) now pass the
  `CorrTrack` object to `compute_metrics_bf` instead of `.correlated`; scratchpad harnesses
  updated the same way.
- (8) `CorrTrackMultiWindow._merge_correlated_rows` + `.correlated_rows()` -- numeric merge of
  per-window-size trackers (vstack + canonicalize + dedup keeping stronger |corr|). The
  `.correlated` dict merge still works via the child property (tests green); thread mode
  unaffected.
- (9) `CorrTrack_compare` metrics go through `_mode_run` -> `NumericCorrelatedFlags` ->
  `compute_metrics_bf` numeric path automatically.
- (10) Cython -- NOT needed. Re-profile (below) shows the numpy canonicalize+append is 0.20s.

**Re-profile (m=121, thr 0.70, ~4% density, NS=250), numeric accumulator vs entry (aa):**

| profile sub-phase | entry (aa) | now |
|-------------------|------------|-----|
| `val.numeric_rows_kernel` | 1.20s | 0.97s |
| `val.numeric_rows_postprocess` | **7.28s** | **0.20s** (36x) |
| `monitor.numeric_state_cython` | 1.44s | 1.14s |

Validation is now kernel-bound. `_normalize_bf_key`'s per-pair Python is gone from the hot path.

**`CORRTRACK_LEGACY_CORRELATED` A/B env flag removed** after verification. Suite 119/119.

**End-to-end calibrated arm-A cell re-run (m=121, NS=500, occ x gamma_offset calibration,
numeric representation):**

| threshold | eff. density | arm A (entry x) speedup | now | recall | precision |
|-----------|--------------|-------------------------|-----|--------|-----------|
| 0.70 | ~4%   | **0.81x** | **1.68x** | 0.965 | 1.0 |
| 0.86 | ~0.5% | 3.15x     | **4.83x** | 0.970 | 1.0 |

At 4% density CorrTrack flips from a net slowdown to a 1.7x speedup end to end, with calibration
and recall/precision intact. Phase mix at 4% is now sk/ca/va/mo = 0.5/4.9/2.3/3.8 -- validation
dropped from 14.4s (arm A) to 2.3s; candidate search and the (genuine Cython) monitor kernel are
now the larger phases, no longer masked by Python string-key bookkeeping.

**This retroactively changes the interpretation of the whole option-C dense-regime story**
(entries x, y): the "speedup goes negative above ~1.5% density" finding was measured with the
string-key recording overhead in both BF and CorrTrack, and that overhead hit CorrTrack far
harder. With the numeric representation, CorrTrack stays net-positive up to ~4% density. The
Sobol/OA sweep CSVs' validation_time / monitor_time columns are inflated by the same overhead;
a re-run would show cleaner phase splits and higher dense-regime speedups. NOT re-run yet.

**Status: numeric-representation work COMPLETE** (plan steps 1-10; step 10 Cython not needed).
Suite 119/119. Changed: `library_corrtrack_parallel.py`, `experiment_lsh_{cost,sobol,nbands_
occupancy,orthogonal_array}_sweep.py`, `test_stable_reproduced_changes.py`.

## 2026-09-10 (aa) — Profiled the dense-regime validation stage: the correlation KERNEL is not the bottleneck; recording results is

`fast_corr_and_dist_batch` micro-benchmark (realistic dense candidate rows, 121x560 buffer, 6M
rows): current `validate_corr_rows` (gather-free, one bulk nogil call) = **238 ns/row**;
vectorized-gather + `fast_corr_and_dist_batch` = **3845 ns/row (16x slower)** -- 21s of gather
vs 1.7s of kernel. The batch kernel is fine; materializing the strided windows into contiguous
`(n_rows, w)` arrays is what kills it. (This is also *why* `fast_corr_and_dist_batch` is not
used for validation: a gather-based predecessor `validate_corr_batch` was already tried and
reverted -- code comment at `_iter_validation_payloads`: "the Python-level bookkeeping cost
often exceeded the Cython-level saving". It IS used, for the proxy-anchor calibration path,
where the bottleneck was Python-loop orchestration, not the gather.)

**`CORRTRACK_PROFILE=1` on a real m=121 / threshold 0.70 / 4% density run:**

| validation sub-phase | time | per tested row |
|----------------------|------|----------------|
| `val.numeric_rows_kernel` (`_cy_validate_corr_rows`) | **1.20s** | **193 ns** |
| `val.numeric_rows_postprocess` | **7.28s** | -- |
| `val.numeric_rows_prep` | 0.04s | -- |

The Cython correlation kernel is **193 ns/tested-row -- faster than brute force's ~262 ns/test.**
The "~1057 ns/row" gap seen in entries (x)/(z) is the whole validation STAGE, and ~6x of it is
`postprocess`: `_record_correlated_numeric` -> `_numeric_rows_to_pair_map` (a per-accepted-row
Python loop building `_normalize_bf_key((id1,id2,t1,t2,w))` string-tuple keys) + the
validated-step buffer append. At 4% density there are ~1M correlated windows to record, one
Python object at a time.

**Consequences:**
- The "unfair comparison / BF amortizes and we don't" framing is wrong at the kernel level --
  the kernel already matches BF per pair. The gap is Python-layer result recording.
- Blocked/lag-grouped matmul (the entry (z) proposal) would optimize the kernel -- the part
  that is already fast. Not worth it.
- The real lever: make correlated-result recording bulk/vectorized -- keep results as numeric
  arrays, build the string-keyed pair map lazily (only when metrics/CSV actually need it) rather
  than eagerly every step. Lives in the Python orchestration layer (`_record_correlated_numeric`
  / `_numeric_rows_to_pair_map`), lower risk than a Cython rewrite, and targets the measured
  7.3s. NOT yet implemented -- needs care (the pair map feeds monitoring and artifacts; the
  `_normalize_bf_key` string-tuple representation is consumed by the recall-eval path per the
  2026-09-08(b) note, so deferring it means auditing those consumers).

## 2026-09-10 (cc) -- Density-deterministic synthetic generator: `make_density_targeted_dataset`, verified

Implemented the approved plan (`~/.claude/plans/misty-stargazing-liskov.md`, "Density-deterministic
synthetic generator"). The old `make_corr_dataset` chain could not control the brute-force
effective tuple density (`bf_correlated / bf_tested`) -- the dominant factor in every recent
result -- so `experiment_lsh_{sobol,orthogonal_array}` swept `corr_prop`, which mapped to
1e-8..1.5e-5 effective density with no usable monotone relation. The new generator takes a
TARGET effective density plus the eval config and hits it, deterministically, with the exact
ground-truth `(pair, lag, window, signed_r)` set returned by construction.

**Design as built** (`synth_corr_gen.py`, ~360 new lines after `make_corr_dataset`):
- `ng` disjoint groups of `g` series (the rest uncorrelated). Density knob is `(ng, g)`, solved
  from `d_exact(ng,g) = ng*C(g,2)*on_frac / tested_per_step` where
  `tested_per_step = C(m,2) + m^2*(L_test-1)` -- this is exactly what `Candidates_BF` accumulates
  per full step (synchronous pairs once, every ordered current/history pair incl. self at each
  nonzero lag). Solved by grid search over `(ng, g)`, not a linear inversion (the relation is
  not monotone-linear once a size is pinned). `g=2` with small `ng` is the sparse regime, one
  big group the dense regime -- no special-case "disjoint mode".
- Shared driver `c_group` is **white noise**. A driver whose windowed autocorrelation stayed
  >= threshold across a lag BAND of width `b*step` would necessarily have a correlation length
  ~`b*step`, and at `w ~ 8*step` that makes two INDEPENDENT group drivers windowed-correlated
  well above threshold -- verified empirically, a hard geometric incompatibility, not a tuning
  issue (see "errors" below). So each pair is planted at exactly ONE lag bucket
  (`|o_i - o_j|`), and `lag_band` instead spreads member offsets over `b` consecutive buckets
  (jitter) so different within-group pairs sit at different lags spanning `b` buckets
  collectively. Offsets are in bucket units, applied as `offset*step`-sample driver shifts so
  BF's bucketed lag search lands exactly on them.
- Member `i`: `X_i(t) = s_i(t) * a_i * shift(c_group, o_i)(t) + eps_i(t)`, `c_group` and `eps_i`
  both normalised to unit WINDOWED std in the evaluation space (diff space iff `preprocess`).
  Within-group pair strength `= a_i a_j / sqrt((a_i^2+1)(a_j^2+1))`, `a(r) = sqrt(r/(1-r))`,
  loadings drawn Beta(2,5)-skewed over `[a(thr+margin), a(r_max)]` (most pairs well above
  threshold), `near_threshold_fraction` pinned at the floor.
- Scheduled state timeline: `n_epochs` epochs, per-group state in `{0, +1, -1}` from
  `_build_group_schedule`, step-aligned epoch boundaries so transitions are sharp and land on
  the window grid. `corr_sign="both"` forces at least one `+<->-` sign flip; `duty<1` schedules
  OFF epochs. Event log (onset / offset / sign-change) written to `_meta.json`.
- **Exact ground truth by construction, vectorised** (`_analytic_gt`): per group, the set of
  valid current-window starts depends only on the pair's lag bucket (members share the state
  timeline), computed once per `(group, lag)` and reused. GT is signed, canonical (later-start
  first, id-sorted on the `t1==t2` tie -- matches `_normalize_bf_key` / `_canonicalize_rows`),
  and 1-based in time (Candidates_BF records window starts 1-based).
- **Self-verification** (`_verify`, gated by `verify_bf=True`): runs the real
  `run_and_log_bruteforce` over exactly the first `n_eval` windows and compares the analytic GT
  key-set to the BF key-set. Windows whose span straddles a state transition are inherently
  ambiguous (part ON, part OFF/opposite-sign); the analytic GT never emits those and they are
  dropped from the BF set before the comparison (`_determinate_mask`, vectorised). Guards:
  raise if `gt_recall < 1 - near_threshold_fraction - 0.05` (construction not clearing
  threshold) or `gt_precision < 0.95` (spurious cross-group correlation). `verify_bf=False`
  skips BF entirely (needed at large m -- the BF pass is the O(m^2 L n) cost the analytic GT is
  meant to replace) and reports `analytic_density` from a closed-form tested-tuple count.
- Correction loop: at most `_max_correction_regens` (default 3) rebuilds, re-solving `(ng, g)`
  at the measured model gain `analytic_density / d_exact`. Switching `(ng, g)` changes the
  transition structure so the gain is not perfectly stable; the loop keeps the
  closest-to-target attempt, not the last, to avoid ending on an oscillation overshoot.

**Verification (all in `test_synth_density.py`, 9 tests, plus a 14-cell scratch grid):**
- Density hit: `|analytic_density - target| / target` within tolerance (0.15 default) for
  `duty=1.0` across `m in {60, 200, 600}`, `target in {1e-3 .. 3e-2}`, ar1 / random_walk,
  pos / both. `duty<1` (scheduled OFF) lands within ~0.20 on `analytic_density` and closer on
  `verified_density` -- the two bracket the target because BF counts the transition-straddle
  windows the analytic GT excludes; documented as a limitation of the sharp-transition model.
- **`gt_precision == 1.000` everywhere** (planted set == BF-confirmed set on determinate
  windows); `gt_recall` 0.97..1.00.
- Ground truth is signed, canonical, `|r| in [thr-0.05, r_max]`.
- `corr_sign="both"` schedules contain `+<->-` flips and BF reports both signs.
- `duty<1` produces onset/offset events.
- Byte-identical determinism (`.npz`, `_gt.npz`, `_meta.json`) for a fixed
  `(target, seed, base_proc, corr_sign, n_epochs, duty, lag_band)`.
- random_walk + `preprocess=True`: correlation lives in diff space, BF (which also diffs)
  confirms it; `gt_precision == 1.0`.
- `m=600, target=3e-2` reached (`g=403, ng=1`, analytic 0.0274, verified 0.0304, precision
  1.0) -- the dense regime the old generator (cap ~1e-5) could never touch.
- Existing suite `test_stable_reproduced_changes.py` still 119/119; combined 128/128.

**Errors found and fixed along the way** (chronological):
1. Spurious cross-group correlation at 5-15% of BF positives. Root cause: the original
   `sigma_c ~ 33` Gaussian-smoothed driver looked like a slow trend within a 128-sample window,
   so two independent instances were windowed-correlated ~0.5-0.7. A probe over kernel widths
   showed even `sigma=0.5` (near-white) gives cross-group max `|r| ~ 0.35` over ~60k tuples
   while `sigma>=3` blows past threshold. Fixed: driver is now pure white; lag band moved to
   offset jitter. This is a genuine geometric limit, recorded in the code comment.
2. `analytic_density` was ~1.85x below `verified_density`: the denominator convention was
   `C(m,2)*L_test*n_eval` (unordered, one lag per pair) but `Candidates_BF` counts
   `C(m,2) + m^2*(L_test-1)` per step (ordered, self-pairs, per nonzero lag). Fixed
   `tested_per_step` and the `(ng, g)` solver to match.
3. Time base mismatch: analytic GT used 0-based window starts, BF records 1-based. Fixed
   (`+1` in `_analytic_gt`).
4. Offsets were in bucket units but applied as raw-sample driver shifts, so BF's bucketed lag
   search never landed on the planted lag (recall ~25%). Fixed (`offset*step`).
5. `_g_for` linear inversion (`d ∝ g-1`) broke once the solution needed `g > m/2` (n_groups
   pinned at 1), causing the correction loop to oscillate. Replaced with a `(ng, g)` grid
   search + best-attempt tracking.
6. Straddle windows near sharp transitions: BF fires on them (mostly-live window), analytic
   excludes them. Fixed by the `_determinate_mask` exclusion from the metric on both sides,
   plus step-aligned epoch boundaries. The count of such windows is reported as
   `n_bf_ambiguous_windows` in `_meta.json`.
7. `_analytic_gt` Python double-loop was 73s at m=600. Vectorised (valid starts computed once
   per group-lag): 2.5s.

**Changed files:** `synth_corr_gen.py` (generator + `_gen_shared_driver_signal`,
`_build_group_schedule`, `_canonical_gt_row`, stem/meta), `test_synth_density.py` (new).

**Not done (deliberately deferred to a fresh session -- the generator itself is the hard,
now-finished part):** the sweep wiring from the plan -- `datasets/synth_loader.py` mode
dispatch + stem keys; `experiment_lsh_cost_sweep.load_density_targeted_cell` +
`synthetic_ground_truth`; `experiment_lsh_{sobol,orthogonal_array}_sweep.py` factor swap
(`corr_prop` -> `target_density`, `DENSITY_RANGE`/`DENSITY_LEVELS`, CSV columns, conditional BF
phase). Also `_analytic_gt` at m > ~400 still allocates the full GT (10M rows at m=600);
fine for a cached generate, would want chunking for a very large sweep.

**Commands run:**
- `python3 -m pytest test_synth_density.py test_stable_reproduced_changes.py -q` -> 128 passed
- scratch grid `/tmp/.../scratchpad/dbg_grid.py` (14 cells, m in {60,200}, targets 1e-3..2e-2,
  ar1/random_walk, pos/both, duty 0.6/1.0) -> 14/14 within tolerance, precision 1.0
- scratch probes `dbg_cross.py` / `probe_driver.py` (driver decorrelation), `dbg_orient.py`
  (time/id convention), `dbg_spur2.py` (spurious-row breakdown)

## 2026-09-10 (dd) -- Staged the accumulated dev work onto the existing remote `dev` branch

**Repo situation discovered (corrected an initial wrong assumption):** this `corrtrack_release_dev`
working copy was a STALE checkout pinned at `83f43c6` (2025-12-11 "Update README.md") carrying
~9 months of uncommitted work. Meanwhile the real commits were made from a sibling checkout and
pushed: `origin/main` advanced to `7ae37ef` (2026-04-14 `param_hyperopt`) and `origin/dev`
(which DOES exist -- `https://github.com/RebeccaSalles/CorrTrack/tree/dev`) sits one commit
beyond that at `7a486c3` (2026-07-07 "dev version"). A `feat/v2-engine` branch also exists on
the remote. `83f43c6` is a clean ancestor of `origin/dev`.

**Divergence assessed:** every file `origin/dev` has that this working tree lacks is junk
(`__pycache__/*.pyc`, `*:Zone.Identifier`, `build/`, `datasets/synth_outputs/`, `header.png`,
`library_corrtrack_parallel copy.py`) -- no real source is lost. This working tree's
`corrtrack_param_search.py` was confirmed to already contain the remote's April `_load_module`/
`DEFAULT_EXEC_PARAM_CONFIG` additions, i.e. the working copy is a strict forward evolution of
`origin/dev`'s real source (LSH backend, memory fixes, sweeps, numeric-pair repr, density
generator all present here, none on `origin/dev`: `library_corrtrack_parallel.py` 15166 lines
on dev vs 19201 here; markers `SignLSHBandIndex`/`_win_free_idx`/`RECALL_SAFETY_MARGIN` etc.
absent on dev).

**What was done:** local `dev` re-pointed to `origin/dev` (`git reset --mixed origin/dev`,
working tree preserved), then the source work staged for one commit on top of it.

- Rewrote `.gitignore` (remote had only a broken 2-line one) to add `__pycache__/`, `*.pyc`,
  `*.egg-info/`, virtualenvs, `*.Zone.Identifier` / `*:Zone.Identifier`, `/build/`, `/docs/`,
  `/tasks/`, `/tmp_artifacts/`, `/correlation/`, `/datasets/synth_outputs/`. Does NOT ignore
  `*.c`/`*.so` -- user chose to commit the Cython-generated `.c` and the prebuilt
  linux-x86-64 `.so` alongside the `.pyx`.
- Staged: all root source (`*.py`, `*.pyx`, `*.c`, `*.so` -- modified + new), `README.md`,
  `datasets/synth_loader.py`, the new `.gitignore`, and the deletions of `header.png` +
  `library_corrtrack_parallel copy.py` (stray cruft on the remote).
- **`docs/` and `tasks/` deliberately kept PRIVATE** per user's explicit instruction (they hold
  internal dev narrative + `claude.ai/code/artifact/...` URLs) -- excluded from the commit AND
  added to `.gitignore`. `abaca/` also left out this round (env-capture notes; not requested).
- Left unstaged (tracked cruft on the remote, minimal-diff / not requested): `__pycache__/*.pyc`,
  `build/`, `datasets/synth_outputs/*`, `*:Zone.Identifier`, `xp/xp1.txt`/`xp2.txt`,
  `datasets/examples/synthetic/...correlated.csv`. Available as a follow-up `git rm --cached`
  cleanup.

**Commands run:** `git fetch origin`; `git checkout -b dev` then `git reset --mixed origin/dev`;
`git add` of the source set above. **No commit, no push** -- handed to the user (commit message
per user request carries NO `Co-Authored-By` trailer).

**Known issues / caveats:**
- Local git identity is unset (`user.name`/`user.email` empty at repo and global scope); prior
  commits authored `Rebecca Pontes Salles <rebeccapsalles@gmail.com>`. The handed-over command
  sets it inline via `git -c`.
- The remote `dev` still tracks ~37 MB of junk (pyc/build/synth_outputs/header.png); this commit
  removes only `header.png` + the `copy.py`, not the rest.
- `origin/main` is NOT touched; whether `dev` is later merged/PR'd to `main` is the user's call.

**Next exact step:** user runs the `git -c user.name=... commit` + `git push origin dev` command
from the session response.

## 2026-09-11 -- Analysis of `feat/v2-engine` (colleague's branch) and a merge plan for FilCorr + parallel work

**Branch:** `dev`. Analysis only, no source files touched. Requested by the user: understand
what a colleague (Benoit Lange) built on `origin/feat/v2-engine` and plan how to reuse two
specific contributions (a parallel-execution optimization, and a FilCorr competitor baseline)
without adopting the rest.

**What `feat/v2-engine` actually is:** 16 commits, all dated 2026-09-10, authored by
`benoit lange <benoit@lange.xyz>` (+ Claude Opus 5 co-author trailers), branched from
`7ae37ef` (`param_hyperopt`, the same commit `origin/main` sits on and the merge-base with
`dev`). It is a from-scratch, self-contained rewrite ("v2") under a new `v2/` package: its own
pipeline/config/CLI/logging, 5 swappable sketch methods, **13** swappable candidate-index
backends (grid/tree/kdtree/quadtree/bptree/bst/octree/knn/bptree3d/bst3d/hnsw/vptree/annoy),
and pluggable compute backends (python/vectorized/parallel/cython/mps/coreml/cuda-stub). The
colleague's own slide deck committed alongside it (`slides_3way_v1_v1dev_v2.tex`) explicitly
frames v2 as diverging from the **fork point**, not from current `dev`: at `7ae37ef`, "v1-dev"
(what `dev` was then) had just gained a cosine-gate + Cauchy-Schwarz-upper-bound-pruned
`bptree` index. Everything `dev` built afterwards and up to `da0f4f7` (LSH sign/Hamming
candidate backends, the overlap-corrected recall formula, proxy-anchor hyperopt, the
free-list memory-leak fixes, the numeric correlated-pair representation, the density-targeted
synthetic generator -- see this log's 2026-07 through 2026-09-10 entries) has **no
counterpart in v2** and is not reflected in the colleague's own comparison slides.

**Contribution 1 -- "parallel execution optimization" (`v2/backends/parallel.py` +
`base.py`/`cython.py`): assessed as NOT worth porting as-is.** It is a `ProcessPoolExecutor`
wrapper that chunks `correlate(pairs)` across processes, composable over a `python`/
`vectorized`/`cython` base kernel, with a `MIN_PARALLEL_BATCH=10_000` sequential-fallback
threshold and a persistent atexit-registered pool -- sound engineering, but built for v2's much
simpler, non-incremental-index, non-LSH pipeline. `dev`'s own parallel story is already more
advanced and architecturally different: `candidate_kernels.pyx` uses real `nogil`
`cython.parallel.prange` for true multi-core parallelism inside the compiled kernels
themselves, orchestrated from Python via `ThreadPoolExecutor`/dask's threaded scheduler
(`_parallel_map`, `parallel_sketch`/`parallel_candidates`/`parallel_validation`,
`candidate_parallel_mode="recent_shards"`) -- never `ProcessPoolExecutor`. Porting v2's process
pool would (a) duplicate an axis `dev` already covers more effectively, and (b) be unsafe to
graft onto `dev`'s stateful, free-list-backed LSH index without redesign (pickling that state
across processes every step is exactly the kind of change CLAUDE.md's "don't simplify /
don't change scientific meaning silently" rule warns against). Recommendation: treat v2's
parallel module only as a *design reference*, not an import. The one legitimately open question
worth a real (separate, measured) investigation: whether `dev`'s `ThreadPoolExecutor` call
sites (`library_corrtrack_parallel.py:4722,4803,13063,13184`) get real speedup given the GIL,
i.e. whether the code they call actually releases it -- if that investigation finds a real gap,
the fix would be a small dev-native process-pool path (mirroring `MIN_PARALLEL_BATCH`'s
threshold/fallback idea), not a wholesale port.

**Contribution 2 -- FilCorr competitor (`v2/fillcorr/`, `v2/fillcorr.md`): assessed as HIGH
reuse value, clean port.** FilCorr (Zhong, Souza, Mueen, ICDM 2020) computes Pearson
correlation over band-pass-filtered windows via Parseval's identity directly on FFT
coefficients (`O(B)` per pair, `B` = band width, vs `O(m)`). `v2/fillcorr/algo.py` is
pure-numpy, dependency-free math (`band_indices`, `full_band_fft(_batch)`, `parseval_corr(_batch)`,
`incremental_update`, `naive_filtered_pearson` as a reference implementation) -- fully portable.
`v2/fillcorr/runner.py`/`backends.py` are NOT portable as-is (tightly coupled to v2's own
`steps`/`core` framework) and should not be ported; only the math should move.

**Found the right integration point already in `dev`:** `library_corrtrack_parallel.py` already
has precedent for exactly this -- a second exact "competitor" baseline, `exact_stomp`
(`Candidates_BF_ExactSTOMP`, line ~11317), selected via `baseline_mode` (`_resolve_baseline_mode`,
line ~1920) and dispatched from `CorrTrack.run_bf` (line ~8958) to a dedicated
`run_bf_exact_stomp` method. It is a fully self-contained, stateful, streaming-incremental class
with its own `.run(new_data_step, ids, ...) -> (accepted_rows, accepted_corrs, n_pairs, timing)`
returning numeric rows in the same `(N,5)=[s1,s2,t1,t2,lag]` canonical format
`compute_metrics_bf`/`compute_metrics_bf_numeric` already consume -- and `--baseline-mode` is
already a `corrtrack_run_bruteforce.py` CLI flag. FilCorr fits this exact slot: a new
`Candidates_BF_FilCorr` class (ported math from `algo.py`, windowed/incremental like
`Candidates_BF_ExactSTOMP`), `baseline_mode="filcorr"` added to the alias table, a
`run_bf_filcorr` dispatch mirroring `run_bf_exact_stomp`, and `filcorr_fs`/`filcorr_ft`/
`filcorr_sampling_rate` threaded through `base_config` the same way `neg_corr` etc. already are.
`corrtrack_compare_runs.py` needs **no changes** -- it already compares whatever `baseline_mode`
produced.

**Important scientific-meaning caveat to carry into implementation (per CLAUDE.md):** FilCorr's
Parseval identity is mathematically identical to standard Pearson only at full band
(`fs=0.0, ft=0.5`, DC removed) -- that is the setting to use for the SOTA-speed/recall/precision
comparison against the existing bruteforce ground truth, since a real band-pass filter computes
a genuinely different quantity (filtered correlation) that cannot be scored against the
unfiltered ground truth without misrepresenting recall/precision. Band-pass FilCorr is a
separate, legitimate future experiment, not to be conflated with the competitor-benchmark ask.
Verification plan before trusting the port: confirm full-band FilCorr reproduces the *same*
correlated-pair set as `run_and_log_bruteforce`'s own Pearson baseline (bit-for-bit key match,
small FP tolerance on `corr`) on a real dataset, as an automated test -- mirroring this
project's existing verify-before-trusting discipline.

**Not ported, and why:** the rest of v2 (13 index backends, 5 sketch methods, MPS/CoreML/CUDA
backends, the whole `steps`/`core` pipeline framework) -- all superseded by `dev`'s own LSH/
Cython/proxy-hyperopt machinery, and adopting it would mean **regressing** the algorithm (no
LSH backend, no overlap-corrected recall formula, no free-list memory bounding, no numeric
correlated-pair fast path in v2) -- directly against CLAUDE.md's "do not simplify the algorithm"
/ "do not silently change the scientific meaning" rules.

**Not done:** no source files were edited this session (analysis/planning only, per the user's
explicit ask). The FilCorr port (Contribution 2) is ready to implement on request; the parallel
investigation (Contribution 1) is a separate, smaller, measurement-first follow-up.

**Next exact step:** if the user approves, implement the FilCorr port as scoped above
(`Candidates_BF_FilCorr` + `baseline_mode="filcorr"` wiring + the full-band-equivalence
regression test), verified against `run_and_log_bruteforce` before being treated as a trusted
competitor baseline; separately, and only if asked, run the GIL/`ThreadPoolExecutor` real-speedup
investigation before deciding whether any process-pool work from v2 is worth a dev-native
reimplementation.

## 2026-09-11 (b) -- GIL investigation run; FilCorr port implemented, 2 real bugs found and fixed, verified end-to-end

User approved both follow-ups from the (a) entry above: run the GIL/`ThreadPoolExecutor`
investigation, and implement the FilCorr port. Both done this session; source files now have
real, uncommitted changes (no commit requested).

**GIL investigation -- real measurement, not assumption.** Traced `parallel_validation`'s
thread-pool path (`_corr_validation_batch_worker` -> `candidate_kernels.validate_corr_rows`) and
confirmed the compiled kernel wraps its main loop in `with nogil:` (`candidate_kernels.pyx:2676`).
Wrote a standalone micro-benchmark (`/tmp/.../scratchpad/gil_bench.py`) calling
`validate_corr_rows` directly from `ThreadPoolExecutor`, sequential vs 2/4/8 threads, on
synthetic data (400k candidate rows, 40 series x 20k cols), measuring both wall-clock
(`perf_counter`) and total CPU time across threads (`process_time`, which sums all threads' CPU
on Linux). Real result on this machine (8 vCPU WSL2): sequential cpu/wall=1.05 (single-core, as
expected); 2 threads: cpu/wall=2.05, 1.83x wall speedup; 4 threads: cpu/wall=3.88, 2.99x speedup;
8 threads: cpu/wall=4.87, but speedup REGRESSED to 2.63x (more total CPU burned, less wall-clock
gain than 4 threads -- diminishing returns / likely hyperthread contention on this box).
**Conclusion: `dev`'s existing thread-based `parallel_validation` gets real, substantial
multi-core speedup (not GIL-serialized) up to ~4 threads on this machine** -- confirms the (a)
entry's recommendation NOT to port v2's `ProcessPoolExecutor` wrapper (it would add IPC/pickling
overhead for no benefit over an already-genuinely-parallel nogil path). Disclosed, not chased
further: the 4-thread sweet spot / 8-thread regression is this machine's own characteristic (WSL2
vCPU count is not necessarily physical-core count) and would need re-measuring on the actual
target machine (e.g. Abaca) before treating `max_workers` defaults as tuned.

**FilCorr port -- implemented, verified, two real bugs found and fixed via the verification test
itself (not assumed correct after "it compiled").** Per the (a) entry's plan:
`library_corrtrack_parallel.py` gained `_filcorr_band_indices` (module-level, ported math) and
`Candidates_BF_FilCorr(Candidates_BF_ExactSTOMP)` (inherits the streaming window buffer +
raw-window constant/spike guards; overrides `run()` with a band-limited-FFT + Parseval
correlation matrix, vectorized exactly like `Candidates_BF_ExactSTOMP`'s own dot-product matrix).
Wired via `baseline_mode="filcorr"` (`_resolve_baseline_mode`'s alias table) and `run_bf_filcorr`
(mirrors `run_bf_exact_stomp`, dispatched from `run_bf`). `filcorr_fs`/`filcorr_ft`/
`filcorr_sampling_rate` set as plain `CorrTrack` attributes (default 0.0/0.5/1.0, matching how
`baseline_mode` itself is set -- NOT added to the already-huge constructor signature), threaded
from `base_config` in `run_and_log_bruteforce`. `corrtrack_run_bruteforce.py` gained
`--filcorr-fs`/`--filcorr-ft`/`--filcorr-sampling-rate` CLI flags + `"filcorr"` as a
`--baseline-mode` choice; `experiment_run_exec_param.py` documents the new `FILCORR_FS/FT/
SAMPLING_RATE` defaults alongside the existing `BASELINE_MODE` doc comment.
**`corrtrack_compare_runs.py` needed zero changes**, as scoped -- it already scores whatever
`baseline_mode` produced. No changes to `RUN_RESULT_COLUMNS`/`COMPARISON_COLUMNS` (mirrors how
`exact_stomp` itself adds no new CSV columns; `filcorr_fs/ft/sampling_rate` are not currently
persisted per-row -- a disclosed, deliberate minimal-diff choice, not an oversight).

**Two real bugs found by the full-band-equivalence test itself, before trusting the port:**
1. Missing-Nyquist-bin bug (even window sizes): the one-sided band `[lb,ub)` never includes the
   Nyquist bin (k=m/2), which has no distinct FFT mirror -- every OTHER included bin implicitly
   represents itself + its mirror (a uniform 2x factor that cancels exactly in the correlation
   ratio, verified algebraically against `naive_filtered_pearson`'s time-domain reconstruction),
   but Nyquist has nothing to be doubled against. First test run (window_size=32, white-noise
   synthetic data) showed real, non-trivial deviations from raw Pearson (up to several % absolute
   correlation error, not float noise) -- traced to this, fixed by extending the band to include
   Nyquist with POWER weight 0.5 (sqrt(0.5) in amplitude) instead of the implicit weight-1 every
   other bin gets.
2. Off-by-one at the odd-window upper edge: `ub = floor(m*ft/f)` under an exclusive `[lb,ub)`
   slice silently drops the last valid non-mirrored bin whenever `m*ft/f` isn't an integer (e.g.
   m=33, ft=0.5 -> floor(16.5)=16, excludes k=16 even though its own frequency 16/33=0.4848 is
   below ft=0.5 and should be kept) -- found via the SAME test on odd window sizes (492 key
   mismatches, max |corr diff| 0.18 for m=33). Fixed by forcing `ub = window_size//2 + 1`
   (the closed-form exact upper edge for both parities) whenever the request reaches/exceeds
   Nyquist, independent of the Nyquist-weighting fix above.

Both fixes are gated on "request reaches/exceeds Nyquist" (i.e. only affect the full-band /
near-full-band case that must match Pearson exactly); any genuinely narrower band (the "different
scientific quantity" case, per the (a) entry's caveat) is unaffected.

**Verified, three levels:**
1. Unit level (`test_filcorr_candidates_node_matches_exact_stomp_pearson_full_band`, new):
   `Candidates_BF_FilCorr` vs `Candidates_BF_ExactSTOMP` directly, 40-60 streaming steps, window
   sizes 32/33/168/169 (both parities, including the project's own real `window_size=168`) --
   0 key mismatches, max |corr diff| < 1e-9 (float noise) after the two fixes above (was:
   hundreds of mismatches / up to 0.18 absolute error before).
2. Integration level (`test_run_and_log_bruteforce_filcorr_matches_bruteforce_count`, new):
   `run_and_log_bruteforce` with `baseline_mode="filcorr"` vs `"bruteforce"` on the same synthetic
   dataset -- identical `record["correlated"]` count.
3. Real CLI end-to-end (`corrtrack_run_bruteforce.py --baseline-mode filcorr` vs `bruteforce`, the
   `synthetic_10_52560` dataset config, window_size=32/step=8/n_lags=16/corr_threshold=0.6):
   **411901 correlated pairs, identical, for both** -- confirms the full CLI/config-threading
   path (not just the Python-level call), per this project's "verify cross-class wiring
   end-to-end, under the real default config" discipline.

**Commands run:** `python3 -m pytest test_stable_reproduced_changes.py -q` -> **121 passed**
(119 pre-existing + 2 new, no regressions); `python3 -m pytest test_synth_density.py -q` -> 9
passed; the CLI smoke run above (`corrtrack_run_bruteforce.py`, both baseline modes).

**Changed files (all uncommitted, no commit requested):** `library_corrtrack_parallel.py`
(`_resolve_baseline_mode` alias table, `CorrTrack.__init__` FilCorr defaults,
`run_and_log_bruteforce` config threading, `_filcorr_band_indices`, `Candidates_BF_FilCorr`,
`run_bf_filcorr`, `run_bf` dispatch), `corrtrack_run_bruteforce.py` (CLI flags + config
threading), `experiment_run_exec_param.py` (documented defaults),
`test_stable_reproduced_changes.py` (+2 tests, +2 imports).

**Known issues / not done:**
- `filcorr_fs`/`filcorr_ft`/`filcorr_sampling_rate` are not persisted as CSV columns (deliberate,
  see above) -- if per-cell sweep/dashboard reporting of these ever matters, add them to
  `RUN_RESULT_COLUMNS`/`COMPARISON_COLUMNS` then.
- `corrtrack_run_corrtrack.py` (the main-run launcher) was NOT touched -- `baseline_mode`/
  FilCorr is a `run_bf`-only (ground-truth/competitor baseline) concept, matching `exact_stomp`'s
  own scope; it does not apply to the main CorrTrack candidate-search run.
- `CorrTrackMultiWindow.run_bf` delegates to per-size `CorrTrack` tracker instances -- whether
  `baseline_mode`/`filcorr_*` attributes actually propagate to each tracker was not verified
  (pre-existing question that applies equally to `exact_stomp`, not a new gap introduced here;
  out of scope for this session).
- Real band-pass FilCorr (`fs>0` or `ft<0.5`) was not benchmarked or verified against anything
  (by design -- a real band-pass computes a different quantity, not scorable against the
  unfiltered ground truth; that would be a separate, deliberate future experiment).
- No speed/timing benchmark of FilCorr vs bruteforce was run this session -- only correctness
  (recall/precision-equivalence at full band) was verified. FilCorr's actual speed advantage
  (`O(B)` vs `O(window_size)` per pair) at real project scale (m=121+ series, window_size=168) is
  unmeasured; worth doing before using it as a genuine "SOTA speed comparison" in a paper/report.

**Next exact step:** if useful, benchmark FilCorr's actual wall-clock speed at real project scale
(mirroring the `exact_stomp` vs `bruteforce` comparisons already in this codebase) now that
correctness is verified; otherwise this item is done and available for the benchmark harness as
`--baseline-mode filcorr`.

## 2026-09-11 (c) -- Real speed benchmark run: FilCorr correctness holds at project scale, but it is NOT faster (~6% slower overall, ~22% slower on validation)

User approved running the speed benchmark. Real dataset (`fr_air_temperature_121_1`, 121
stations, 1 year), real project defaults (`window_size=168, window_step=12, n_lags=168,
corr_threshold=0.7`), via `corrtrack_run_bruteforce.py --baseline-mode bruteforce` vs
`filcorr`.

**Methodology note (caught and corrected before trusting the numbers):** the first pair of runs
was launched concurrently (two background OS processes on the same 8-vCPU machine) -- both
single-threaded (`exec_mode=sequential`) but competing for the same cores/cache/memory
bandwidth, contaminating the wall-clock comparison. Re-ran them SEQUENTIALLY (one finishes
before the other starts) for a clean measurement -- the numbers below are from that second,
clean run.

**Correctness, confirmed again at real scale (not just the synthetic unit tests):** both runs
produced the exact same `correlated` count (14,043,815) and the same `tested` count (79,928,982).

**Speed, real and disclosed honestly -- FilCorr is SLOWER here, not faster:**

| | cand_time | val_time | monit_time | runtime |
|---|---|---|---|---|
| bruteforce | 1.415s | 11.419s | 15.397s | **30.496s** |
| filcorr (full band) | 1.052s | 13.986s | 15.086s | **32.274s** |

Overall ~6% slower; the validation phase specifically (the one FilCorr's `O(B)` vs `O(m)` claim
is about) ~22% slower.

**Most likely cause, not chased further this session (disclosed, not fixed):** this project's
plain `bruteforce` baseline is not a naive Python loop -- it already goes through the same
compiled, `nogil` Cython kernel (`candidate_kernels.validate_corr_rows`, confirmed real and
multi-core-parallel-capable in the (b) entry's GIL investigation) the main CorrTrack pipeline
uses. `Candidates_BF_FilCorr`, faithful to the ported reference's own non-incremental default (no
`algo.py` `incremental_update` usage, matching `v2/fillcorr.md`'s own stated window_step>1
behavior), recomputes a fresh `numpy.fft.fft` per window per step with no cross-step caching. At
`window_size=168` the band width B≈84 (full band, DC removed) only buys a ~2x theoretical
per-pair reduction -- nowhere near enough to cover a fresh per-window FFT's cost against an
already-compiled dot product.

**Two concrete, not-yet-pursued follow-ups if FilCorr's speed is meant to actually beat
bruteforce here** (both explicitly out of scope for this session -- correctness + an honest speed
read was the ask, not "make FilCorr win"):
1. Port `v2/fillcorr/_kernels.pyx` (a compiled Parseval kernel) -- the original (a) entry flagged
   this as "only if profiling shows a real bottleneck"; it now has one.
2. Port `algo.py`'s `incremental_update` (O(1)-per-slide band update, analogous to
   `Candidates_BF_ExactSTOMP`'s own incremental dot-product-reuse trick that likely explains
   bruteforce's/exact_stomp's own speed here).

**Not done / disclosed:** no comparison against `baseline_mode="exact_stomp"` specifically (only
vs the default `bruteforce`); no sweep across window sizes/series counts to see whether FilCorr's
relative standing changes at a different scale (e.g. a much larger window_size, where B's
relative reduction is more favorable, or a narrower real band-pass, which is a different
scientific question per the (a) entry's caveat and wasn't benchmarked either).

**Commands run:** two sequential `corrtrack_run_bruteforce.py` invocations (bruteforce, then
filcorr) on `fr_air_temperature_121_1`, real defaults, written to
`/tmp/.../scratchpad/filcorr_speed2/{bf,fc}/`.

**Next exact step:** none required -- FilCorr is verified correct and its real (currently
unfavorable) speed profile at this scale is now honestly on record. If the user wants FilCorr to
actually win on speed, pick up one of the two follow-ups above.

## 2026-09-11 (d) -- Made FilCorr's implementation tier COMPARABLE to the baselines it's scored
against (user's correction: "I do not want filcorr to win, but I want its results to be
comparable"), then measured the honest result

**User correction, important framing shift.** The (c) entry's "FilCorr is slower" finding
compared an implementation-tier mismatch, not a clean algorithmic one: `bruteforce` validates
through a compiled `nogil` Cython kernel (`validate_corr_rows`); `Candidates_BF_FilCorr`
validated through plain numpy. User explicitly clarified the goal is a FAIR comparison, not
making FilCorr win. Fixed two real, well-justified inefficiencies (not "handicap removal for its
own sake" -- each one is independently defensible on its own merits):

1. **Real-decomposition instead of a wasted complex matmul.** `run()` computed
   `Re(Wx @ Wy.conj().T)` via a FULL complex GEMM then discarded the imaginary half -- complex
   GEMM does ~4x the real FLOPs of the 2 real GEMMs actually needed
   (`Re(a*conj(b)) = Re(a)Re(b) + Im(a)Im(b)`). Changed `_band_fft_at` to return `(re, im)` float64
   arrays instead of one complex array; `run()` now computes `sxy = Wx_re@Wy_re.T + Wx_im@Wy_im.T`.
   This puts FilCorr on the SAME BLAS-matmul footing `Candidates_BF_ExactSTOMP.run` itself uses
   (`curr_window @ hist_window.T`, one real matmul) -- deliberately NOT a hand-written/`prange`
   Cython kernel, which would have hand-modified FilCorr a multi-core edge `exact_stomp` doesn't
   also get (the wrong kind of "improvement" given the user's actual ask).
2. **Persistent, cross-step FFT memoization**, found necessary by measurement, not assumed:
   traced why (1) alone barely moved val_time (13.99s -> 13.82s on the real benchmark) while
   `exact_stomp`'s own incremental dot-product cache gets a 4.9x win over plain `bruteforce` on
   the SAME data -- each window's band-FFT was being recomputed from scratch up to
   `~n_lags/window_step` times (once per later step referencing the same window position at an
   increasing lag) instead of once. Fixed: `_band_fft_cache` is now PERSISTENT across `run()`
   calls, keyed by absolute window start_time (not buffer position, which shifts), with a new
   `_evict_band_fft_cache` bounding it to the same `n_lags` reachability window `window_data`
   itself already uses -- bounded memory over an arbitrarily long stream, not unbounded growth.
   Deliberately NOT `algo.py`'s O(B)-per-slide `incremental_update` twiddle-factor formula --
   plain memoization already eliminates the real redundancy found, with less correctness surface
   than deriving/verifying an incremental-slide formula that (per a rough O() estimate) would
   cost about the same as a fresh FFT at this project's own window_step=12/window_size=168 anyway.

**Verified after each fix:** `python3 -m pytest test_stable_reproduced_changes.py -q` -> 121/121
green both times (no regression to the exact-Parseval-equivalence guarantee from entry (b)).

**Then profiled directly (cProfile, 60 real-scale steps, m=121, window_size=168) instead of
continuing to guess -- found the FFT-redundancy hypothesis was WRONG about where the remaining
time goes:** `_band_fft_at` totals only **0.010s of 1.222s** (<1%) -- the memoization worked
exactly as designed (47 real FFT calls instead of ~900 for 60 steps), it just was never the
dominant cost. The dominant cost (1.031s "tottime" inside `run` itself) is the per-lag MATMUL,
recomputed fresh every lag every step.

**Real, principled reason FilCorr cannot match `exact_stomp`'s speed here even when fairly
implemented -- not an implementation gap:** `exact_stomp`'s speed comes from incrementally
updating its cached PER-LAG dot-product MATRIX when the window advances by exactly `window_step`
(`_dot_for_lag`'s `can_increment` branch: subtract the aged-out samples' contribution, add the
newly-arrived samples' -- an O(window_step) correction instead of an O(window_size) recompute).
This works because a raw dot product is a simple additive sum over independent samples. FilCorr's
Parseval correlation is a dot product of BAND-FILTERED representations, and band-filtering (via
the FFT) mixes all samples in the window -- it is NOT a simple per-sample-additive quantity, so
the same "subtract old, add new" trick has no valid closed-form analog for the per-lag CORRELATION
MATRIX itself (only for the underlying per-window FFT, which memoization already handles, and
which was shown by profiling not to be the bottleneck). Concluded, not assumed: attempting to
force an incremental analog here would mean reformulating what the method computes, which is
exactly what CLAUDE.md's "do not change the scientific meaning" rule warns against -- correctly
out of scope.

**Final, honest measured result** (real dataset, real project defaults, sequential/clean runs,
repeated for consistency given real run-to-run variance -- `bruteforce`'s own val_time ranged
9.95-11.4s across 3 runs on identical config):

| | val_time (typical range) | vs bruteforce |
|---|---|---|
| bruteforce | 9.95 - 11.4s | -- |
| filcorr (before this entry's fixes) | 13.8 - 14.0s | ~1.22-1.25x slower |
| filcorr (after this entry's fixes) | 12.3 - 12.6s | ~1.15-1.18x slower |

The fairness fixes closed part of the gap (22-25% slower -> 15-18% slower) without eliminating
it -- and per the principled reason above, a residual gap in this direction is the mathematically
expected outcome at full band (no asymptotic advantage possible there; FilCorr's real O(B) edge
only exists for a genuinely narrower band, a different, non-comparable quantity per entry (b)'s
caveat), not a sign more engineering would close it further. Correctness unaffected throughout:
identical `correlated` count (14,043,815) on every real-dataset run in this and the (c) entry.

**Changed files (all uncommitted, no commit requested):** `library_corrtrack_parallel.py`
(`Candidates_BF_FilCorr.__init__`/`_band_fft_at`/`_evict_band_fft_cache`/`run`, updated
docstrings).

**Not done / disclosed:** no attempt at a narrower-band speed characterization (out of scope,
different quantity); no attempt to close the residual ~15-18% gap further given the principled
explanation above; BLAS threading itself (single-threaded reference BLAS on this machine, see
`docs/local_environment_baseline.md`) was left as a shared, symmetric factor affecting both sides
equally, not chased as a FilCorr-specific lever.

**Next exact step:** none required -- this is now considered the fair, final comparison for
`baseline_mode="filcorr"` at full band. If useful later: characterize FilCorr's real speed at a
genuinely narrow band (a separate, deliberate experiment, not scored against the unfiltered
ground truth) to see the O(B) advantage the method is actually built for.

## 2026-09-11 (e) -- Density generator: burst-mode feature + 2 real bugs fixed; real-data scaling/recall investigation; two controlled synthetic experiments (degree-vs-m confirmed, persistence-vs-recall inconclusive)

Continuation of entry (cc)'s density generator, driven by a real question from the user: is
the "big speedup, near-perfect recall, gets better with scale" story from the Sobol sweep
actually true on real data, or was it measured in an easier regime than reality?

**Real-data findings (French weather stations, `fr_air_temperature`, nested subsample by
station count via `prepare_training_data`, m = 10..121):**
- Raw (undifferenced) temperature: effective density is FLAT (~8%) as m grows 10->121, so
  degree (density * (m-1), i.e. true correlated partners per series) grows roughly linearly,
  9.98 at m=121 -- already past the LSH's calibrated occupancy (2-5), a dense/flat-speedup
  regime. Largely a shared diurnal/seasonal-trend artifact, not necessarily real local
  coupling.
- Differenced (`preprocess=True`): density genuinely decays m=10->65 (2.93%->1.12%), degree
  stays bounded (<=1.64 even at m=121) -- real evidence this domain CAN look sparse/scalable
  once the trend confound is removed. Degree ticks back up m=65->121 (0.72->1.64), not
  explained (possibly non-geographic station ordering in the nested subsample -- not checked).
- Recall never reached the 0.95 calibration target on real diff-space data (capped ~0.80),
  independent of gamma-offset widening (0.15->0.40, no effect) or `n_vectors` (64->256, no
  effect) -- ruled out as a simple calibration problem.
- Real BF true positives are 58-87% concentrated within 0.05 of whatever threshold is chosen
  (checked both 0.70 and, by filtering the same data, what 0.80 would give: 80.2% in
  diff-space) -- **raising the threshold to 0.80 does NOT fix this, it makes the relative
  concentration WORSE** (a property of any decaying correlation-strength distribution, not
  specific to 0.70).
- Per-station-pair aggregation found recall correlates far more with **persistence**
  (consecutive windows a pair stays correlated: 0.51 recall at 1-2 windows -> 0.82 at 32+,
  corr=0.376) than with **strength** (corr=0.123, too weak to be the primary driver) --
  though persistence and strength are themselves correlated in real data (0.561), which
  matters below.

**Generator: `burst_length_windows` parameter added** (`synth_corr_gen.py`) to let a
correlated pair be ON for a short, transient span instead of the whole epoch -- needed to
reproduce/test the persistence finding synthetically. Two real bugs found and fixed while
building it:
1. `_valid_for_lag`/`_determinate_mask` used `searchsorted(..., side="left")`, which
   incorrectly flagged a window as straddling a transition when the transition landed
   EXACTLY at the window's start (`ss[lo]` is already the new, correct state there) -- fixed
   to `side="right"`. Without this fix short bursts of length exactly `w/step` windows were
   silently dropped entirely from the ground truth.
2. The burst's centering offset (`(hi-lo-span)//2`) was not guaranteed to be a multiple of
   `step`, so a burst could land off the evaluation-window grid entirely, contributing ZERO
   evaluable windows even though it existed in the data -- fixed by flooring the offset to
   the nearest step multiple. Both verified via bisection over `n_epochs` until every
   previously-zero-GT case produced a nonzero, correct ground truth. Suite green
   (130/130) after both fixes.

**Experiment 1 -- degree-vs-m crossover (controlled synthetic, `tmp_artifacts/degree_vs_m_v2/`):**
Fix degree (true correlated partners/series), vary m, using the calibrate-then-run harness
(`experiment_lsh_cost_sweep.run_smart_cell` / `run_bruteforce_once`) at corr_threshold=0.70,
window_size=128, window_step=16, n_lags=112 (L_test=8), n_vectors=64, occupancy in {2,3,5}.
- **degree=1** (well-controlled -- verified achieved degree matches target at every m,
  comfortably under this generator's ~6.7% density ceiling): speedup climbs monotonically
  with m -- **1.17 (m=50) -> 1.83 (100) -> 2.10 (200) -> 3.02 (400)**. Clean, controlled
  confirmation that speedup improves with scale when true degree stays low.
- **degree "20"**: NOT actually held constant -- the generator's own density ceiling
  (`1/(2*L_test-1) ~= 6.7%` at these settings, found in entry (cc)) made the requested
  density infeasible at small m, so the ACHIEVED degree actually ramped 3.2 (m=50) -> 6.6
  (100) -> 13.3 (200) -> 20.3 (400), only reaching the intended 20 at m=400. Speedup fell
  as that ramp proceeded: 1.38 -> 1.68 -> 1.43 -> **1.20**. A messier, unintended version of
  the same test (degree rising through the crossover as m grows) but still consistent with,
  and a second independent confirmation of, the same mechanism.
- **Verdict: the degree/occupancy crossover mechanism (proposed a few sessions' worth of
  discussion ago to reconcile the Sobol sweep's growing-with-scale speedups against arm
  A/B's flat ones) is now confirmed with real controlled evidence, not just a plausible
  story.** Whether CorrTrack's speedup improves with scale is a property of whether the
  dataset's true degree stays bounded, not of m/L/density-fraction alone.

**Experiment 2 -- persistence-vs-recall (controlled synthetic,
`tmp_artifacts/persistence_vs_recall_v2/`):** m=150, single occurrence per pair
(`n_epochs=1`, `burst_length_windows` varied 1-2 .. 64-128 windows, plus "persistent"),
fixed group structure (ng=1, g=150, i.e. everyone correlated, same loadings distribution
throughout) so only DURATION varies, not strength or topology. v1 of this experiment (not
kept) was confounded -- it gave each pair 23 repeated short bursts instead of one, so
"short" pairs still had many independent chances and showed high recall; this v2 fixes
that.
- **Result: recall stayed ~0.97-1.00 across the ENTIRE persistence range (10.6 to 299.8
  windows), including the shortest achieved** -- no climb at all, contradicting the real-data
  finding.
- **This means the "it's persistence, not strength" explanation from the real-data
  investigation was premature and is likely wrong, or at least incomplete.** Since real
  data's persistence and strength are themselves correlated (0.56), the real recall-vs-
  persistence pattern more likely reflects persistence and strength acting together (or
  strength being the true driver, with persistence just correlated with it in practice) --
  this synthetic generator's loadings are independent of burst duration and skewed high by
  default, so it never tests a SHORT+WEAK pair, which is what real short-persistence pairs
  actually are.
- **Not yet run**: the natural follow-up -- vary duration while ALSO pinning strength near
  threshold (`loading_skew="low"` or high `near_threshold_fraction`), to test whether
  short+weak reproduces the real recall drop where short+strong did not.

**Task status:** degree-vs-m crossover -- CONFIRMED, controlled evidence in hand. Persistence-
vs-recall -- OPEN, real mechanism still unknown; the strength/duration interaction
experiment above is the next concrete step. Investigating whether the recall gap is
fixable at the algorithm level (monitor/candidate-discovery mechanism) has not been started.

**Changed files:** `synth_corr_gen.py` (`burst_length_windows` param + the 2 bug fixes above).
New: `tmp_artifacts/persistence_vs_recall_v2/run.py`, `tmp_artifacts/degree_vs_m_v2/run.py`
(resumable, checkpoint-per-cell -- the dev machine restarted mid-session three times; results
written to disk under `tmp_artifacts/`, not `/tmp`, survive that). `tmp_artifacts/
persistence_vs_recall/` and `tmp_artifacts/degree_vs_m/` (v1, superseded/confounded, kept for
the record of what was wrong and why).

**Commands run:** `python3 -m pytest test_synth_density.py test_stable_reproduced_changes.py -q`
-> 130/130 passed after the burst-mode fixes.

**Next exact step:** run the strength x duration interaction experiment (vary
`burst_length_windows` at `loading_skew="low"` / high `near_threshold_fraction`) to find the
real cause of the persistence-vs-recall pattern before concluding anything further about it;
separately, investigate the monitor/candidate-discovery code path to see whether the real
recall gap (whatever its true cause) is a fixable lever or an inherent property to disclose.

## 2026-09-12 -- Four-way comparison (bruteforce / exact_stomp / filcorr / CorrTrack) run for
real on Abaca; hyperopt tuning, an n_vectors sweep, and a sparse-data confirmation

**Branch:** `dev` (`42b5b41`, pushed to `origin/dev` and pulled onto Abaca this session --
see the (d) entry for that commit's own content). All work this entry describes ran on Abaca
(`sophia.g5k`, `~/corrtrack_release_dev`, conda env `corrtrack`, OpenBLAS) via OAR batch jobs
submitted from this session -- nothing run locally, per the (d) entry's WSL-crash history.
Scripts live at `abaca/fourway_compare.py`, `abaca/sparse_fourway_compare.py`, and the
`abaca/*.oar` batch wrappers (mirroring `abaca/smoke.oar`'s own build-kernels-in-job convention)
-- copied over via `scp`, not yet committed to the repo (disclosed below).

> **Note added 2026-09-17 (comparison plan section 4b.2).** The `filcorr` arm in every table of
> this entry ran with `neg_corr=True`, `fs=0.0`, `ft=0.5` (the full band), and `n_lags=168` at
> `window_step=12`. Two of those are deviations from the FilCorr paper that the paper's own
> method does not have: (1) negative correlation. FilCorr's Eq. 7 is a max of *signed*
> correlations over the lag window; our port adds `abs()`, tagged `supports_neg_corr =
> "enabled_by_us"` in the run record since 2026-09-17 (d). (2) A full band makes FilCorr the
> exact all-pairs computation, so its recall of 1.0000 here is by construction, not a finding
> about the method's band-limited pruning (it has none: the paper rejects data-dependent
> pruning on purpose). The wall-time ranking `exact_stomp` < `filcorr` < `bruteforce` stands as
> measured. Both points must be stated wherever these numbers are reused in the paper.

**Why this exists:** direct follow-on from the (d) entry's FilCorr comparability work -- the
user asked for a real four-way timing/recall comparison (bruteforce, `exact_stomp`, `filcorr`,
and the actual CorrTrack main algorithm) on real data, at real project scale, which the local
WSL machine could not safely run (see the two real OOM crashes recorded below).

### Real memory bug found and fixed along the way (before any of the numbers below)

The first two real-scale local attempts crashed the WHOLE machine (not just the Python
process) -- traced to `CorrTrack._append_correlated_numeric`'s doubling-growth accumulator
needing a 640MB+ single allocation for 14M+ correlated pairs (the real fr_air_temperature
dataset at `corr_threshold=0.7` is unusually dense, ~17.6% of tested pairs correlated). Fixed
in this session, on top of the (d) entry's already-committed FilCorr work:
1. `_append_correlated_numeric`'s internal storage downsized int64 -> int32 (values never
   approach int32 range; verified every external consumer already re-casts to int64 itself).
2. Two redundant defensive int64 upcasts removed (`correlated_rows()`, `NumericCorrelatedFlags.
   __init__`) -- each was independently found, via a SECOND and THIRD real OOM, to reintroduce
   the exact same peak one level up the call stack the moment `correlated_rows()`'s own fix
   landed. Verified every actual consumer re-casts itself when it needs to.
Committed (by the user, not this session -- see the earlier "attribution" exchange) as
`42b5b41` on `dev` and pushed; see that commit message for the full rationale. 131 tests
passing throughout (`test_stable_reproduced_changes.py` + `test_synth_density.py`).

**Abaca has 192GB/node (mercantour3/mercantour2, `memnode=196608` MB) -- no memory constraint
there at all**; the int32 fix was still the right thing to ship (real bug, real machines this
small do exist and did crash) but the actual comparison runs below did not need it to succeed
on Abaca specifically.

### Comparison methodology (all runs)

Real dataset `fr_air_temperature_121_1` (121 stations, 1 year, n_obs=6133) unless noted
"sparse". Fixed config: `window_size=168, window_step=12, n_lags=168, corr_threshold=0.7,
neg_corr=True`, `exec="sequential"`, BLAS pinned to 1 thread (`OMP/MKL/OPENBLAS/NUMEXPR_
NUM_THREADS=1`) for a clean single-threaded comparison. Kernels rebuilt in-job every time
(`-march=native`, per this project's own established Abaca discipline -- never reuse `.so`
across nodes). Recall/precision of `exact_stomp`/`filcorr`/`corrtrack` computed against that
SAME job's own `bruteforce` run via `CorrTrack.compute_metrics_bf`, entirely in memory
(`NumericCorrelatedFlags`, no CSV artifact round-trip -- the (d) entry's memory fix made this
practical). `bruteforce`'s own wall time varies ~15-38% run to run on this shared cluster
(29.6-58.8s observed across jobs) -- cross-job absolute-time comparisons are noisy; the
WITHIN-JOB relative speedup (each job computes its own `bf_wall` denominator) is the number to
trust.

### Run 1 -- dense real data, untuned CorrTrack (job 3095335)

Confirms correctness at real scale (not just synthetic tests): `bruteforce`/`exact_stomp`/
`filcorr` all found the IDENTICAL 14,043,815 correlated pairs, precision/recall 1.0000/1.0000
for both against bruteforce. `n_vectors=32` (guessed, no hyperopt): CorrTrack found 12,668,515
(precision 1.0000, recall 0.9021), speedup vs bruteforce 1.03x -- barely faster, `cand_time`
(23.9s, the LSH search itself) dominates at this real density.

| method | wall_s | cand_s | val_s | monit_s | correlated | recall | speedup |
|---|---|---|---|---|---|---|---|
| bruteforce | 58.77 | 1.43 | 33.34 | 23.14 | 14,043,815 | -- | 1.00x |
| exact_stomp | 30.54 | 2.38 | 5.03 | 22.15 | 14,043,815 | 1.0000 | 1.92x |
| filcorr | 31.40 | 1.68 | 6.73 | 22.00 | 14,043,815 | 1.0000 | 1.87x |
| corrtrack (nv=32) | 56.83 | 23.87 | 10.20 | 21.19 | 12,668,515 | 0.9021 | 1.03x |

### Run 2 -- same dense data, hyperopt-tuned CorrTrack (job 3095527)

`corrtrack_param_search.py`'s real proxy-anchor hyperopt (`[proxy] ... 94 of 256 max anchors
afforded at m=121, n_lags=168, window_step=12`, 8/8 backend-search anchors completed) picked
`n_vectors=32` (same as the guess), `candidate_backend=lsh_sign_dot`,
`candidate_lsh_target_occupancy=3.0`. **Real, honest, somewhat counterintuitive finding**:
tuning IMPROVED recall (0.9021 -> 0.9800, comfortably past the 0.95 target) but made overall
speed WORSE (1.03x -> 0.73x, i.e. slower than plain bruteforce) -- `cand_time` grew to 25.99s
reaching that higher, safer recall. Not a bug: this is the real recall/speed tradeoff the
hyperopt is built to navigate, and on this unusually dense real dataset it chose reliability
over speed. Disclosed to the user as such, not spun as a win.

### Run 3 + n_vectors sweep -- same dense data, n_vectors override (jobs 3095703, 3098647)

Kept the tuned `candidate_backend`/`occupancy`, varied only `n_vectors` (32 tuned baseline,
then 64/96/128 overrides on top of the same tuned config):

| n_vectors | recall | precision | cand_s | val_s | speedup |
|---|---|---|---|---|---|
| 32 (hyperopt) | 0.9800 | 1.0000 | 25.99 | 11.03 | 0.73x |
| 64 | 0.9704 | 1.0000 | 11.43 | 10.07 | 0.99x |
| 96 | 0.9773 | 1.0000 | 10.87 | 8.93 | 1.01x |
| 128 | 0.9780 | 1.0000 | 10.92 | 9.37 | 1.01x |

Clear, real, disclosed pattern: 32->64 is the big win (`cand_time` more than halves); 64->96->
128 plateaus (`cand_time`~11s, recall~0.97-0.98, speedup~1.0x) -- diminishing returns past
n_vectors=64 on this dataset. The hyperopt's own pick (32) was NOT the sweet spot the manual
sweep found (64) -- a real, disclosed gap between what the search actually explored and what a
simple 1-D override sweep found, not chased further this session (the search space/grid the
hyperopt used vs a plain n_vectors override are not doing the same search).

### Run 4 -- SPARSE synthetic data, the regime CorrTrack is actually built for (job 3098682)

User's own next question ("next, use other sparse data") -- the real dataset above is
unusually DENSE (~17.6% of tested pairs correlated), a known-unfavorable regime for
LSH-based pruning (less to prune). Generated via `synth_corr_gen.make_density_targeted_
dataset` (already on `dev`/Abaca, not new this session) -- same m=121 series count for
comparability, `target_density=0.02` (achieved 0.01734, `n_obs=6144`, `base_proc={"type":
"ar1","phi":0.6}`, `corr_sign="both"`, `n_epochs=6`, `duty=1.0`, `lag_band=4`, seed=7,
`verify_bf=False` since this session's own bruteforce run is the real check). `n_vectors=64`
(the sweet spot found in the sweep above), same tuned `candidate_backend`/`occupancy`.

**CorrTrack wins outright here** -- fastest of all four, not just competitive:

| method | wall_s | cand_s | val_s | monit_s | correlated | recall | speedup |
|---|---|---|---|---|---|---|---|
| bruteforce | 29.62 | 1.69 | 24.97 | 2.78 | 2,056,726 | -- | 1.00x |
| exact_stomp | 8.91 | 2.37 | 4.01 | 2.35 | 2,056,726 | 1.0000 | 3.33x |
| filcorr | 9.92 | 1.62 | 5.80 | 2.32 | 2,056,726 | 1.0000 | 2.99x |
| corrtrack | 8.09 | 4.39 | 0.59 | 2.39 | 2,046,228 | 0.9949 | **3.66x** |

Mechanism, real and as-expected: `cand_time` (4.39s, the LSH search) costs MORE than the raw
dot-product baselines' equivalent phase here too, but at 2% density almost nothing is a true
positive, so `val_time` collapses to 0.59s (vs bruteforce validating everything at 24.97s) --
a trade that pays off decisively at low density, unlike the dense-data runs above. Precision
1.0000, recall 0.9949 (untuned n_vectors=64, no fresh hyperopt run for this dataset).

**Coherent, honest, complete story across both regimes**: dense real data (this project's own
real fr_air_temperature at corr_threshold=0.7) is architecturally unfavorable for CorrTrack
(ties at best, 0.73x-1.03x depending on tuning); realistic sparse data (the regime the method
is actually designed for) shows a clear, decisive win (3.66x, beating even `exact_stomp`/
`filcorr`). Both measured with the same rigor (real hardware, real recall/precision against a
real bruteforce ground truth in the same job) -- not a cherry-picked comparison.

**Commands run:** `oarsub -q abaca -l /host=1,walltime=...` x6 real jobs (3095335, 3095527,
3095703, 3098647, 3098682, plus one earlier misconfigured submission, 3095322, that errored
before running anything -- relative `-O`/`-E` log paths resolved against the wrong launching
directory because the `oarsub` command wasn't run from inside the repo; fixed by always `cd
~/corrtrack_release_dev &&` before `oarsub`). Each job: `rm -rf build *.so && python
setup_cython.py build_ext --inplace -j 4` (node-specific rebuild), then the comparison script.
Local: `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q` (131
passed) before each push.

**Not done / disclosed:**
- `abaca/fourway_compare.py`, `abaca/sparse_fourway_compare.py`, and the `abaca/*.oar` wrappers
  exist only on Abaca (copied via `scp`) and in this session's local scratchpad -- NOT added to
  the git repo. Worth committing if this comparison is meant to be reproducible/repeatable
  later (mirrors `abaca/smoke.oar`'s own already-committed pattern).
- The hyperopt-vs-manual-sweep gap (hyperopt picked n_vectors=32, manual override found 64
  clearly better) was not investigated further -- worth a real look at whether
  `corrtrack_param_search.py`'s own grid/search actually covers n_vectors=64 at this
  `(m, n_lags, window_step)` combination, or whether the proxy-anchor evaluation itself is
  giving a different ranking than the real four-way comparison does.
- No fresh hyperopt run for the sparse dataset -- n_vectors=64 there is carried over from the
  dense-data sweep, not independently tuned; the 3.66x speedup is very likely NOT this
  sparse dataset's own best case either.
- `bf_total_candidates` for `compute_metrics_bf`'s `specificity` denominator uses `bruteforce`'s
  own `total_candidates`/`tested` field -- not independently re-verified against the synthetic
  generator's own `gt_rows`/analytic ground truth for the sparse run (a real, available
  cross-check that was skipped for time).

**Next exact step:** if this comparison is to be repeated or extended (more densities, more
`n_vectors` values, a real hyperopt run on the sparse dataset specifically), commit the
`abaca/` scripts first so the setup is reproducible; otherwise this four-way investigation is
considered complete and its conclusion (dense: unfavorable for CorrTrack; sparse: CorrTrack
wins) is the one to carry forward.

## 2026-09-13 -- Three real-domain datasets curated; exact_stomp/filcorr `preprocess` bug found
and fixed; four-way comparison across all three; `hybrid_validation` profiled, root-caused, and
fixed; WSL crashed twice (investigated honestly, root cause found), work moved to Abaca

**Branch:** main (uncommitted; same working copy as the 2026-09-12 entry above -- `dev`'s
`42b5b41` is still HEAD there; changes below are additional uncommitted edits on top).

### Context: the three "killer" real datasets

Following the 2026-09-11(e)/2026-09-12 conclusion (dense real data disfavors CorrTrack, sparse
data is a clear win), the user asked for three real datasets from different domains, chosen to
be genuinely representative of where CorrTrack's contribution is strongest (scalability, i.e.
sub-quadratic candidate search), preferably France/Brazil/worldwide, US-specific acceptable if
no better option exists:

- **Finance**: S&P 500 daily closes, `sp500.npz` = 492 tickers x 1255 days (Yahoo Finance chart
  API, no auth). 11 GICS sectors used as a structural-clustering probe.
- **Environmental**: California streamflow, `ca_streamflow.npz` = 538 USGS gauges x 2192 days
  (`waterservices.usgs.gov` REST API, batched). 104 HUC watersheds used the same way.
- **"Another domain"**: Wikipedia daily pageviews, `wikipedia.npz` = 88 articles x 1827 days
  (Wikimedia REST API, 429-backoff + per-10-article checkpointing needed). 5 categories.
- A fourth (crypto, CoinGecko) was fetched but never analyzed -- not a blocker, parked.

Fetch scripts all hit the same alignment bug independently (naive exact-timestamp intersection
across hundreds of series collapses to near-zero common rows) -- fixed identically in each with
a densest-series calendar grid + `ffill/bfill(limit=3)` + drop-unfillable-columns pattern, plus
a raw-fetch cache (pickle) so re-running the alignment logic doesn't re-hit the network. Also
merged France (121 stations) + Brazil (139 stations) temperature data as a side investigation
into whether cross-hemisphere station count affects degree (it does, unexpectedly -- degree
ROSE 58.6->66.2 as Brazil stations were added, contradicting an independence assumption; not
further investigated this session, flagged as open).

### Real bug found: `exact_stomp`/`filcorr` silently ignored `preprocess`

Confirmed by evidence, then by code read: `exact_stomp`'s diff-space `n_true` on one dataset
almost exactly matched a RAW-levels bruteforce count from an earlier session, not that
dataset's own diff-space bruteforce count -- i.e. `exact_stomp` was computing raw-level
correlations regardless of `preprocess=True`. Root cause: `Candidates_BF_ExactSTOMP.__init__`
never accepted/stored `preprocess`; `run_bf_exact_stomp`/`run_bf_filcorr` never passed
`self.preprocess` when constructing these baseline objects; `_newStream` never differenced
incoming data. This had silently invalidated an entire earlier round of three-way comparison
numbers (not previously logged as trustworthy -- caught before being reported).

**Fix** (`library_corrtrack_parallel.py`): added `preprocess` param + `window_data_diff`
maintenance to `Candidates_BF_ExactSTOMP.__init__`/`_newStream` (mirrors
`_update_window_data_diff`'s carry-over-last-raw-value trick so chunks stay column-aligned);
added a `_current_data()` helper (raw vs diff, single point of truth) used by
`_window_prefixes`/`_dot_for_lag`/`run`; `Candidates_BF_FilCorr.__init__`/`_band_fft_at`
updated the same way; `run_bf_exact_stomp`/`run_bf_filcorr` now pass `preprocess=self.preprocess`
through. **Verified**: diff-space cross-check on a 59-ticker sample -- BF/exact_stomp/filcorr
now byte-identical (17,723/17,723 matching keys); 99.98% match on the larger streamflow data
(residual is a floating-point boundary effect, not a structural bug).

### Four-way comparison (bruteforce / exact_stomp / filcorr / CorrTrack), all three domains

Run on Abaca (see below for why), diff-space, CorrTrack calibrated per-dataset via an
occupancy x gamma-offset grid search against plain bruteforce (same pattern as 2026-09-12),
`hybrid_validation=False` for this first pass (the buggy version -- see next section for why
this was corrected to `True` afterward):

| dataset | m | bf wall_s | exact_stomp | filcorr | corrtrack | corrtrack vs stomp | corrtrack vs filcorr |
|---|---|---|---|---|---|---|---|
| wikipedia | 88 | 3.0 | 1.14x (recall 1.00) | 1.18x (recall 1.00) | **1.02x** (recall 0.983) | 0.90x | 0.87x |
| sp500 | 492 | 40.8 | 1.55x (recall 1.00) | 1.73x (recall 1.00) | **2.31x** (recall 0.973) | 1.49x | 1.33x |
| streamflow | 538 | 108.2 | 1.99x (recall 0.9998) | 1.05x (recall 0.9998) | **1.49x** (recall 0.979) | 0.75x | 1.42x |

Speedups are wall-clock vs bruteforce, same data/config, this project's established convention.
Precision was 1.0000 for CorrTrack in every case (user's own framing: precision is guaranteed
by validation, recall+speedup are the real contribution). **Honest conclusion, not the
blanket "CorrTrack is faster" story**: CorrTrack loses to `exact_stomp` on the two densest
domains (wikipedia, streamflow) and only clearly wins on the moderate-density S&P500 case --
consistent with the degree/occupancy mechanism established 2026-09-11(e)/2026-09-12 (all three
real datasets land at high true degree at real scale, an unfavorable regime for LSH pruning).

**Per-chunk instrumentation (answers "is speedup constant while scaling m?", the user's
explicit question)**: split each CorrTrack run into 10 equal chunks, compared each chunk's
wall-time against the exact baseline's constant per-chunk cost (fixed by construction, since
exact methods test the same m^2*L space every step regardless of what's found). Streamflow's
10 chunk-level speedups vs `exact_stomp`: **0.62x-0.91x**, fluctuating meaningfully within a
single run. Answer: no, it is not constant -- exact methods have data-independent per-step
cost; CorrTrack's cost tracks the actual local candidate-touched volume, which varies with the
data as the stream progresses.

### WSL crashed twice; investigated honestly per direct question, root cause found

User asked directly "was it you blowing the memory" -- investigated rather than guessed:
`free -h` showed only 7.8GiB total (`.wslconfig` capped `memory=8GB`); 9 concurrent Claude Code
processes + VS Code/Pylance already used ~4-4.5GB baseline before any of this session's work;
a raw-levels bruteforce run on the streamflow data (before the diff-space-only decision above)
produced ~113.7M correlated tuples (~5.5GB) -- a very plausible trigger, but the deeper issue is
near-zero headroom generally, not one single run. **User's resulting instruction**: use Abaca
for heavy runs from now on. All four-way and hybrid-validation work above and below ran there
(OAR jobs, `sophia.g5k`, no memory ceiling).

### `hybrid_validation` investigation and fix

User's own instruction: rerun the "is speedup constant when scaling m" comparison with
`hybrid_validation=True` for CorrTrack (`experiment_run_exec_param.py:58`), since otherwise the
comparison against exact baselines is unfair. Contrary to that expectation, the first
`hybrid_validation=True` rerun (on Abaca, same 3 datasets) came out SLOWER on every dataset
despite a 91.6%-99.7% cache hit rate:

| dataset | speedup vs bf, hybrid=False | speedup vs bf, hybrid=True (buggy) |
|---|---|---|
| wikipedia | 1.02x | 0.90x |
| sp500 | 2.31x | 1.85x |
| streamflow | 1.49x | 1.18x |

**Root cause, found via `cProfile` (not guessed)**: `HybridValidationCache.validate_pairs`
(`candidate_kernels.pyx`) built a Python list-of-tuples per candidate inside its own Cython
loop; `_get_validated_corr_numeric` (`library_corrtrack_parallel.py`) then re-iterated that
list per-candidate again with `bool()`/`float()` unpacking -- a double Python-object
construction overhead, the exact same anti-pattern the 2026-09-10 numeric-representation work
(entry (bb)) had already fixed elsewhere. On the S&P500 case, `_get_validated_corr_numeric`'s
own cProfile tottime was 4.043s with hybrid on vs 0.233s with it off (~17x).

**Fix**: `candidate_kernels.pyx`'s `validate_pairs` now writes into 7 preallocated numpy arrays
(`ok`/`is_corr`/`corr`/`dist`/`is_const`/`is_spiked`/`used_hybrid`) via typed memoryviews
instead of `results.append((...))`, and returns those arrays directly (the `"results"` list key
is gone). `_get_validated_corr_numeric`'s corresponding per-candidate Python `for` loop was
replaced with vectorized numpy (boolean masking, `.sum()`, `np.argmin` for min-distance
tracking, `np.any(is_corr)` gating `_record_correlated_numeric`). Rebuilt locally
(`setup_cython.py build_ext --inplace`, only pre-existing unrelated warnings).

**One test broke and was fixed, not skipped**:
`test_hybrid_validation_current_window_cache_matches_bruteforce` referenced the removed
`"results"` key -- updated to read `out["corr"][i]` from the new array-based return instead,
preserving the test's original intent (cross-checking the "current window" cache path against
a from-scratch Pearson computation). **131/131 tests pass locally and on Abaca** after
resyncing `library_corrtrack_parallel.py` + `candidate_kernels.pyx` + the test file (the first
Abaca rebuild job was run before the test file was synced -- caught and corrected).

**Verified correctness before trusting any new numbers**: recall/precision are bit-identical
between `hybrid_validation=True/False` on all three real datasets (e.g. sp500 recall
0.9727/precision 1.0000 both ways; streamflow recall 0.9794/0.9794, precision 1.0000/1.0000)
-- the fix only changed performance, not results.

**Local performance re-check (S&P500)**, hybrid vs non-hybrid wall time, before -> after fix:
was ~1.85x slower, now 1.10x slower (14.4s vs 15.8s); `_get_validated_corr_numeric` cProfile
tottime dropped 4.04s -> 1.56s (~2.6x). Wikipedia: was 0.90x-equivalent, now 1.06x slower
(1.72s vs 1.83s). Streamflow (largest dataset, ~12M hybrid attempts): still the least-closed
case, 1.16x slower (56.7s vs 66.0s), validation_time 1.27s vs 8.81s -- the residual gap here is
real cache/lookup work (EMA tracking, current-window-sums bookkeeping) scaling with attempt
count, not an accident, but not yet as cheap as it could be at this scale.

**Full corrected four-way-hybrid comparison re-run on Abaca after the fix**: see the next
session's log entry / `tasks/current_task.md` for the final numbers -- jobs were in flight when
this entry was written; do not reuse the buggy `hybrid=True` numbers in the table above for any
conclusion beyond illustrating the bug that was fixed.

**Known issues / open items**:
- Streamflow's hybrid_validation overhead is still only partially closed at ~12M attempts --
  worth a follow-up profile at that specific scale if hybrid_validation is meant to be the
  default for dense large-m datasets.
- The France+Brazil degree-rising-with-added-stations finding (58.6->66.2) is still
  unexplained (genuine teleconnection vs. multiple-testing noise) -- not investigated further.
- Crypto dataset fetched, never analyzed.
- `abaca/` scripts for the three-domain four-way comparison (`fourway.py`, `fourway_hybrid.py`,
  their `.sh` job wrappers) exist only on Abaca + local scratchpad, not committed -- same
  disclosure as the 2026-09-12 entry's sparse-comparison scripts.

**User also asked**, in the same message, to brainstorm ways to reduce CorrTrack's
density-dependence (its speedup collapses when true correlated-partners-per-series exceeds the
LSH's calibrated occupancy of 2-5). Options proposed, not yet implemented: (1) adaptive backend
dispatch -- monitor the LSH's own touched-candidates-per-query rate and fall back to exact
search when it indicates the filter isn't helping, capping the downside near parity with the
best exact baseline; (2) exploit the structural clustering already found in every real dataset
this session (GICS sector, HUC watershed, Wikipedia category) -- exact validation within a
block, LSH only across blocks, since within-block density is near-total and across-block is
near-zero; (3) degree-aware per-series routing without needing cluster labels (classify hub vs.
normal series from a warm-up pass); (4) keep shrinking constant-factor overhead universally
(this section's own fix is an instance); (5) incremental LSH bucket maintenance exploiting
`window_step << window_size` overlap, analogous to exact STOMP's own incremental advantage.
None implemented yet -- (1) was flagged as the most bounded, most directly responsive next step
if the user wants to continue this thread.

**Commands run**: `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q`
(131 passed, both locally and via an Abaca OAR job after resync); `scp` of the three changed
files to `sophia.g5k:~/corrtrack_release_dev/`; OAR jobs for the Cython rebuild+test
(3102693, then 3102698 after the test-file resync was caught), and three `fourway_hybrid_job.sh`
resubmissions (3102699/3102700/3102701, wikipedia/sp500/streamflow) for the corrected numbers.

**Next exact step**: once the three corrected `fourway_hybrid_job.sh` runs finish, record the
corrected hybrid=True speedups next to the hybrid=False baseline in a new log entry, decide
whether `hybrid_validation=True` should be the default going forward (currently `False`) given
the residual streamflow gap, then move to implementing brainstorm option (1) above if the user
wants to continue reducing density-dependence.

## 2026-09-13 (b) -- Corrected hybrid=True four-way numbers; options 4 and 5 from the
density-dependence brainstorm implemented, verified, and land a real, disclosed win

**Branch:** main (uncommitted, same working copy). Follows directly from entry (a) above.

### Corrected `hybrid_validation=True` four-way numbers (same-job, post-fix)

| dataset | hybrid=False | hybrid=True (buggy) | hybrid=True (fixed) |
|---|---|---|---|
| wikipedia | 1.02x | 0.90x | 0.96x |
| sp500 | 2.31x | 1.85x | 2.08x |
| streamflow | 1.49x | 1.18x | 1.39x |

The fix closed most of the gap but `hybrid_validation=True` is still slightly slower than
`False` on all three even after the fix (confirmed with a clean same-process local comparison
too, isolating cross-job Abaca node-load noise: sp500 1.10x slower, wikipedia 1.06x slower,
streamflow 1.16x slower). Recommendation: keep `hybrid_validation=False` as the default until
this closes further; the earlier density-dependence conclusion (CorrTrack loses to exact_stomp
on the two dense domains, wins clearly only on sp500) holds regardless of this setting.

### Option 4 -- prange parallelism for the LSH candidate-search kernel

Investigation first (see chat, not repeated here): `SignLSHBandIndex._find_pair_rows_meta`
(`candidate_kernels.pyx`) -- the kernel `_increment_candidates`'s LSH branch calls every step
-- is already a tight, allocation-free nogil-style kernel (typed memoryviews, no per-candidate
Python object construction); the 2026-09-10 numeric-representation fix already covers this
path. No further "double Python conversion" win available there. But the kernel's per-query
loop (`for i in range(n_recent)`, one iteration per series per step) ran strictly
single-threaded, unlike a *different* candidate-search backend in the same file (bptree/
sorted-arrays) which already uses `cython.parallel.prange` successfully with a proven
per-thread-buffer-then-merge pattern.

**Implemented**: `SignLSHBandIndex.find_pair_rows_full_cosine_parallel`/`_signed_parallel`
(new, `n_threads=1` default dispatches straight to the unchanged sequential path) and
`_find_pair_rows_meta_parallel` (new `cdef` method): each thread gets its own
`touched_buf`/`visited_stamp` array (both sized to the full index capacity, so no cross-thread
writes are ever possible) and its own growable output-pair buffer (the same `_append_pair`
helper the bptree kernel already uses). The cross-query `_pair_seen` pre-dot dedup is dropped
(would need real synchronization to stay safe under threading) -- a pair found from both
directions by two different threads pays for the dot product twice instead of once, a small
bounded cost -- and correctness is restored by a single vectorized canonicalize+dedup pass
after the parallel region, so the returned row SET is exact either way. New CorrTrack param
`candidate_search_n_threads` (default 1, unchanged behavior), wired through `Candidates` and
`_increment_candidates`'s LSH row-finder selection exactly like `candidate_refresh_interval`.

**Two real bugs found and fixed during bring-up, not shipped silently:**
1. Cython compile error ("Cannot read reduction variable in loop body"): any top-level `cdef`
   variable touched with an in-place operator (`+=`) anywhere inside a `prange` loop body gets
   inferred as a cross-iteration OpenMP reduction for the WHOLE parallel region, not a
   per-iteration scratch value -- wrong for `hpos`/`hneg`/`score`/`touched_count` here, which
   are meant to reset every query. Fixed by moving each accumulation into its own small
   `cdef inline ... nogil` helper (`_lsh_scan_touched_bands`, `_hamming_best_dist`,
   `_dot_score`) so the accumulator lives on that helper's own stack frame, invisible to the
   caller's prange analysis -- mirroring `_row_l2_sq_until`'s existing pattern in this same
   file (`acc = acc + ...`, never `+=`, used successfully inside a prange loop already).
   `cdef` cannot appear directly inside a `for` loop body either (a second compile error hit
   along the way) -- Cython requires all `cdef`s at a function's top level.
2. **A real correctness bug**, found via direct verification (not assumed correct after it
   compiled): `_append_pair(&out_bufs[tid], ..., q_win, cand)` passed `cand` -- the LSH index's
   *internal* entry id -- directly, instead of translating it through `window_idx[cand]` first
   (the window index the row-resolution tail code and `self._win_sid_idx_arr` etc. actually
   index by) -- exactly what the sequential kernel does (`window_idx[cand]`, not `cand`). This
   silently read wrong (sid, time, w) triples for the second element of every output row.
   Caught immediately by end-to-end verification: with the bug, sp500 recall dropped
   0.9727->0.8278 AND precision dropped from 1.0000 to 0.9869 (real false positives, not just
   missed candidates) at `n_threads>1`; a direct row-set diff against the sequential kernel
   confirmed a genuine content mismatch (47/17401 rows differed, both directions, symmetric --
   not just an ordering artifact). Fixed with the one-line `window_idx[cand]` correction.

**Verified correct after the fix**: a direct row-set comparison (`SignLSHBandIndex`
constructed via a real `CorrTrack` run on sp500, sequential vs parallel kernel called on the
identical internal state) is **byte-identical** at n_threads in {1,2,4,8}, both signed
(`neg_corr=True`) and unsigned. End-to-end recall/precision against bruteforce are **bit-
identical** to the sequential path on all three real datasets, at every `n_threads` tested.

**Real, measured speedup** (8-core machine, both local and Abaca):

| dataset (m) | n_threads=1 | =2 | =4 | =8 |
|---|---|---|---|---|
| sp500 (492) | 14.19s (1x) | 8.57s (1.66x) | 7.67s (1.85x) | 8.69s (1.63x, oversubscribed) |
| wikipedia (88) | 1.78s (1x) | 1.67s (1.07x) | 1.54s (1.16x) | 2.75s (0.65x, too little work/thread) |
| streamflow (538) | 71.31s (1x) | 35.66s (2.00x) | 31.79s (2.24x) | 32.96s (2.16x, plateaued) |

The biggest, densest dataset (streamflow -- the exact regime this whole density-dependence
investigation targets) gets the cleanest win (2.24x at n_threads=4); the smallest (wikipedia,
m=88) shows thread-startup overhead dominating at n_threads=8, confirming this needs to be
tuned to series count, not maxed out blindly. `n_threads=4` looks like the practical default
worth adopting for real-scale data on an 8-core machine.

### Option 5 -- exploit basic windows: candidate-refresh cadence decoupled from window_step

Investigation first (verified by code read AND a direct empirical check, not assumed): a new
instrumented run with `basic_window=15`, `window_step=5` (ratio 3) showed `_get_sketches` --
and with it the full LSH-insert-and-search step -- firing 40/40 times over 40 steps, completely
independent of `basic_window`. `basic_window` is already exploited for exactly one thing today
(`_incremental_sketches` reuses `basicDots` partial sums across window_step ticks), never for
reducing how often the two expensive steps (LSH insertion, candidate search) run.

**Implemented**: new CorrTrack param `candidate_refresh_interval` (default 1, unchanged
behavior). At `>1`, the real LSH bucket-scan search runs only every `interval`-th step; on the
other steps, `Candidates._replay_cached_pair_templates` regenerates candidate rows for the
CURRENT (synchronized) window position directly from `(sid1, sid2, delta, w)` templates cached
at the last real search (`_update_cached_pair_templates`), skipping the bucket scan entirely.
Validation still runs on every step against whatever candidates are produced either way, so
precision is unaffected by construction (a stale replayed pair that's no longer correlated
simply fails the exact Pearson check, same as always) -- the real, disclosed cost is
recall/latency: a pair that only becomes correlated strictly between two refresh boundaries
isn't discoverable until the next one.

**One real bug found and fixed during bring-up**: the replay mask incorrectly excluded every
`t1==t2` row unconditionally, meant to guard against the true self-pair (`sid1==sid2`, same
window) degenerate case -- but a *zero-lag match between two different series* is a legitimate
"synchronous" pair (the same case `_canonicalize_rows`/`_normalize_bf_key` both handle
explicitly), and zero-lag templates turned out to be the common case for daily-return financial
data. Caught immediately via a debug instrumentation script showing replayed row counts were
5-8% of the cached template count when they should have been ~100% one step after a real
search. Fixed by only excluding `t1==t2` when `sid1==sid2` too.

**Verified correct after the fix**: precision stayed exactly 1.0000 at every interval tested (as
expected by construction); recall degrades *gracefully and monotonically* with interval, not a
cliff -- sp500: 0.9727 (interval=1) -> 0.9678 (2) -> 0.9592 (3) -> 0.9290 (5).

**Real, measured speedup** (sp500, single-threaded):

| interval | wall | candidate_time | recall |
|---|---|---|---|
| 1 | 14.78s | 15.8s | 0.9727 |
| 2 | 13.01s (1.14x) | 10.9s | 0.9678 |
| 3 | 11.09s (1.33x) | 8.9s | 0.9592 |
| 5 | 10.87s (1.36x) | 8.6s | 0.9290 |

Diminishing returns past interval~3 (insertion + validation overhead becomes the floor).
**This is a real recall-for-speed trade, not a free win** -- unlike option 4, which is exact.

### Options 4 and 5 compose, with an expected, understood interaction (not a bug)

Combined (sp500, `n_threads=4` + `refresh_interval=3`) gave 10.25s, barely better than
`n_threads=4` alone (9.86s) despite `refresh_interval=3` alone also being a real win (13.08s).
Root cause, understood not just observed: `candidate_search_n_threads` only speeds up the real
bucket-scan search; on the `refresh_interval`'s replay steps (a growing majority as interval
increases) there is no bucket scan to speed up at all, so the two levers compete for the same
shrinking pool of "real search" time rather than stacking multiplicatively. Both remain
independently useful; combining them is not automatically better than the stronger of the two
alone at a given setting.

**Commands run**: `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q`
(131 passed, locally and via Abaca OAR job 3103191 after resync); direct row-set diff script
(`verify_parallel_lsh.py`) and end-to-end recall/precision/timing scripts
(`verify_refresh_interval.py`, `verify_refresh_all.py`, `verify_nthreads.py`,
`verify_nthreads_all.py`, `verify_combined.py`) in this session's local scratchpad, not yet
committed anywhere.

**Known issues / open items:**
- `candidate_refresh_interval`'s cached-template replay is only wired for the LSH numeric-rows
  path (`_increment_candidates`'s `lsh_ready` branch, `numeric_rows=True`) -- the non-numeric
  and other backends (bptree, instinct, blocked) are untouched, a disclosed scope limit, not a
  silent gap.
- `candidate_search_n_threads` likewise only touches the LSH backend's numeric-rows path.
- Neither new parameter has been swept against `hybrid_validation` yet (both should compose
  with it in principle -- hybrid_validation only affects the VALIDATION phase, these two affect
  CANDIDATE SEARCH -- but not empirically confirmed together).
- Optimal `n_threads` is dataset-size-dependent (this session's 8-core evidence: 4 was the
  sweet spot for m>=490, 2-4 for m~500+, but m=88 already regresses at n_threads=8) -- no
  auto-tuning implemented; a fixed default would need to consider `m`/`n_recent` per step.
- The four-way comparison (bruteforce/exact_stomp/filcorr/CorrTrack) has NOT been rerun with
  either new option enabled -- the numbers above are CorrTrack-vs-bruteforce only, not yet
  placed back into the full four-way context on Abaca.
- No commit made or requested; `library_corrtrack_parallel.py`/`candidate_kernels.pyx` differ
  from `dev`'s `42b5b41` by all of entries (a) and (b)'s changes, synced to Abaca (byte-identical
  md5sums confirmed) but not committed there either.

**Next exact step**: if continuing, rerun the three-domain four-way comparison on Abaca with
`candidate_search_n_threads=4` (and optionally `candidate_refresh_interval` at a tuned value)
to see how much of the exact_stomp gap on wikipedia/streamflow this closes in the full
four-way context; otherwise this is a good stopping point -- both options are implemented,
verified correct, and show a real, disclosed, measured win.

## 2026-09-13 (c) -- Gate `candidate_search_n_threads` behind `parallel_candidates`

User feedback: option 4's OpenMP threading must not fire just because
`candidate_search_n_threads>1` was passed -- it has to respect this project's own existing
"parallel execution" toggle, so a user who asked for sequential execution actually gets
sequential execution. `self.parallel_candidates` (already existed, governs this project's
other candidate-search parallelism -- grid-node sharding via Python threads) is that toggle;
it defaults to `False` regardless of `exec=`, by this project's own pre-existing convention
(`_resolve_parallel_flag(parallel_candidates, False)`, unlike `parallel_validation` which
defaults from `exec`).

**Fix**: right after `self.parallel_candidates` is resolved in `CorrTrack.__init__`, clamp
`self.candidate_search_n_threads` back to `1` whenever `self.parallel_candidates` is falsy --
a single source of truth, so every downstream call site (the `Candidates(...)` construction,
`_increment_candidates`'s row-finder selection) just sees the already-clamped value and needs
no extra check. Verified directly: `exec="sequential"` + `candidate_search_n_threads=4`
(no `parallel_candidates=True`) now resolves to `candidate_search_n_threads=1` (silently
sequential, as it always was); `exec="thread"` alone ALSO does not enable it (matches
`parallel_candidates`'s own pre-existing default -- not something this fix changes); only
`parallel_candidates=True` explicitly set lets `candidate_search_n_threads>1` actually take
effect. Re-ran the sp500 recall/precision/timing check with `parallel_candidates=True` set
explicitly -- same real speedup as before (1.85x at n_threads=4, recall/precision unchanged) --
confirming the gate only suppresses the UNWANTED case, not the real feature.

**Commands run**: `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q`
(131 passed locally); synced `library_corrtrack_parallel.py` to Abaca.

**Abaca gotcha, real, hit directly**: reran the suite on Abaca WITHOUT rebuilding first (job
3103736), reasoning no Cython source had changed since the last rebuild -- this crashed with
`Illegal instruction` (SIGILL) mid-collection, not a normal test failure. Root cause: the
`.so` is compiled with `-march=native -ffast-math`, tying it to the specific CPU of whichever
node happened to build it; OAR can (and did) schedule this new job onto a *different* physical
node than the one that built the existing `.so`, which lacked whatever instruction the
built-for-node's `-march=native` had baked in. **Lesson, now applied**: always rebuild
(`setup_cython.py build_ext --inplace`) inside the SAME OAR job that runs the tests on Abaca,
every time, even when only a `.py` file changed with zero Cython edits -- never reuse a
previously-built `.so` across job submissions there. Resubmitted as a combined rebuild+test
job (3103742) and confirmed green.

## 2026-09-14 -- "Option 6": symmetric dot-gate score cache (Cauchy-Schwarz bound on
per-series sketch movement). Implemented, verified EXACT, but empirically gives ZERO
benefit -- root cause found, not just observed

**Branch:** main, uncommitted (same working copy). Directly implements the user's own
proposal (chat discussion, not summarized here) plus its extension to symmetric "persistently
far" pairs, both explicitly authorized ("go ahead").

### Design implemented

New CorrTrack param `candidate_score_cache` (default `False`, opt-in, `lsh_approx`-only,
sequential-search-only -- mutually exclusive with `candidate_search_n_threads>1` this round,
a disclosed scope limit not a silent gap). When enabled:
- `candidate_kernels.pyx` gains a new persistent (survives across calls, unlike the existing
  per-call `_pair_seen_*` set) open-addressing hash MAP on `SignLSHBandIndex`
  (`_score_cache_init/free/grow/lookup/upsert`, mirroring `_pair_seen_*`'s proven structure
  with an added `values` array), keyed by `(query_sid, cand_sid*LAG_MULT+lag)` -- series ids
  (permanent, never reused) and `lag` (a small bounded integer), never any reused slot/entry
  index, so there is no staleness-from-slot-reuse risk at all: the key space itself is capped
  at `m^2 * n_lagged_windows` for the whole run, it cannot grow unboundedly.
- In `_find_pair_rows_meta`'s touched-candidate loop, before the O(n_vectors) dot product:
  look up the cached score for this (query, candidate, lag) triple; if found, compute the
  Cauchy-Schwarz bound `||dq|| + ||dc|| + ||dq||*||dc||` (both series' own sketches are unit-
  normalized) from each series' own per-step delta norm; if the cached score sits farther
  from `gamma` (and `-gamma` when `signed_abs`) than this bound, the gate decision provably
  cannot have flipped -- skip the dot product, reuse the cached verdict. Symmetric by
  construction: a persistently-far (rejected) pair skips exactly the same way a persistently-
  close (accepted) pair does. The full bucket scan still runs every step regardless, so no
  candidate discovery is ever skipped -- only redundant rescoring of an already-touched pair.
- Python side (`library_corrtrack_parallel.py`): `Candidates._update_score_cache_delta_norm`
  maintains, per series (sid_idx), `||current_sketch - last_sketch||`, called from
  `_input_tree_lsh_batch` every step; a first-ever sighting gets a deliberately large sentinel
  (4.0) forcing a real computation rather than an unfounded skip.

### Verified EXACT (not just "should be exact")

Direct end-to-end comparison on sp500 (real bruteforce ground truth, `candidate_score_cache`
off vs on): **`ct.correlated` dict identical** (625,539 correlated windows, same set, same
values, both ways) -- recall 0.9727/precision 1.0000 both ways, bit-for-bit. This confirms the
bound math and cache-key design are correct: the feature genuinely never changes the output,
exactly the "no recall risk by construction" property it was designed for, unlike
`candidate_refresh_interval`.

### But: zero cache hits, real slowdown -- root cause found, not hand-waved

First real run: `candidate_score_cache=True` was **2.4x SLOWER** (37.0s vs 15.3s), with
`score_cache_hits=0` across ~22.4M lookups despite `score_cache_found_in_table` matching
`misses` almost 1:1 (i.e., the SAME (series,series,lag) key genuinely does recur constantly --
recurrence is not the problem). Investigated rather than shipped or reverted blind:

- Added a temporary diagnostic (`score_cache_found_in_table`, kept as a permanent stat) to
  separate "key never seen before" from "key found but bound too loose" -- confirmed it is
  the LATTER, every time.
- Directly inspected consecutive-step sketch vectors (`sk._sketch_matrix[0]` across 20+ steps):
  `||delta||` between a series' own sketch one step apart is **consistently ~1.2-1.6**, out of
  a maximum possible 2.0 for unit vectors -- nowhere near "small." A bound built from this can
  never be tight enough to prove anything, so it fails every single time.
- Tested whether this was the `basic_window`/`window_step` boundary-crossing toggle-reweighting
  wrinkle disclosed in the design discussion (the `diff_toggleVector` resign that only fires at
  a `same_start=False` step): **ruled out directly** -- confirmed `same_start=True` (the cheap,
  non-toggle, purely-additive branch) at every one of the sampled steps, and `||delta||` was
  still ~1.3-1.5. Also tested finer `basic_window`/`window_size` ratios (n_basic_windows =
  4, 12, 20, 30) expecting the delta to shrink as the "currently forming" basic window becomes
  a smaller fraction of the total -- **it did not shrink** (stayed 1.3-1.5 across all four).
- Tested on synthetic AR(1) data (phi=0.98, strongly autocorrelated, nothing like financial
  noise) in both raw and differenced form: raw gave `||delta||~0.82`, differenced gave `~1.38`
  -- smaller for genuinely persistent raw data, but still far from small enough for the bound
  to bite in practice (a 0.82 bound already covers a large fraction of the full similarity
  range around any reasonable `gamma`).

**Conclusion, understood not just measured**: the sketch representation's own construction
(random-projection sums per basic window, combined via the `toggleVector` sign-randomization
scheme -- see `_refresh_toggle_weights`/`_generate_randomVectors`) does not, in practice,
produce a small step-to-step movement in the UNIT-NORMALIZED sketch, even when the raw
underlying data changes very little and even in the branch where no toggle resign occurs at
all. The likely mechanism: the pre-normalization raw sketch is a sum of several basic-window
contributions with no single dominant term (by design, for the projection's own estimator
properties), so dividing by its own (comparatively small, noise-influenced) norm to unit-
normalize is exactly the kind of operation that amplifies modest numerator changes into large
directional swings -- normalizing a low-signal-to-noise vector is inherently unstable. This is
a property of the representation, not a bug in this feature's code (which is verified exact),
and it directly explains why the bound is always too loose to fire. This is the same "correct
but not beneficial" pattern this project has hit before with the row-level Cauchy-Schwarz
bound (`docs/implementation_log.md`'s "Part 1" entry, ~0 wall-clock benefit) -- now with the
actual mechanism identified for THIS application, not just another data point.

**Status: implemented, correct, default OFF, not recommended for use as-is.** Left in the
codebase (opt-in, harmless, all 131 tests green with it both off and on) rather than reverted,
since it is a real, verified, documented negative result worth keeping for reference -- per
CLAUDE.md's "treat every idea as an experiment" and "record what remains uncertain."

**What might still salvage the underlying idea (not attempted, disclosed as open)**: the bound
only needs `||delta||` for the SKETCH representation actually in use; a representation with a
more stable (higher signal-to-norm) construction might show a genuinely small per-step delta
where `sketch_proj`'s toggle-randomized scheme does not. Untested. Also untested: whether a
non-normalized or differently-normalized internal score (bypassing the unit-normalization step
specifically) could support a usable bound even though the final cosine-gate score can't.

**Commands run**: `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q`
(131 passed, locally and via Abaca OAR job 3103945 after resync); scratchpad scripts
`verify_score_cache.py`, `debug_score_cache.py`, `debug_sketch_delta.py`,
`debug_sketch_delta2.py`, `debug_sketch_delta3.py` (this session's local scratchpad only).

## 2026-09-14 (b) -- User investigated a targeted fix for option 6's dead end (does a
different sketch representation help, without touching the incremental algorithm?), found
no, then asked to roll back options 5 AND 6 -- density-dependence investigation closed for
now; pivoting to sourcing genuinely sparse, scalable real-world datasets

### Follow-up investigation before the rollback decision

User asked whether some OTHER sketch representation could make option 6's Cauchy-Schwarz
bound useful, without changing `_incremental_sketches`'s own update algorithm. Tested the
most obvious candidate directly rather than reasoning abstractly: the RAW (pre-normalization)
sketch and its own norm are already computed as intermediate values inside `_sketches_from_
scratch`/`_incremental_sketches` (thrown away once `norm_matrix = raw_matrix/||raw_matrix||`
is produced) -- if normalization itself were amplifying a genuinely small underlying change
into a large one, the RAW delta ratio (`||raw_delta||/||raw_norm_old||`) should look smaller
than the normalized delta. It does not: 1.325 vs 1.331 (basic_window=15), 1.385 vs 1.386
(basic_window=3), 1.474 vs 1.460 (basic_window=2) -- essentially identical at every
granularity tested. **Conclusion: not a normalization artifact.** The toggle-randomized
per-basic-window projection scheme's own construction -- necessary for it to be an unbiased
cross-series correlation estimator -- is what produces the large step-to-step movement, in
both raw and normalized form alike. A representation that avoided this would need to change
the randomization itself (altering what the sketch estimates), or fall back to a heuristic
proxy (e.g. the already-stable `basicFeatureSums`/window mean) that isn't the same quantity
being gated -- which would trade away option 6's exact, no-recall-risk property, turning it
into something closer to option 5's character. No further action taken on this -- reported
to the user, who then closed the direction.

### Rollback: options 5 and 6 fully reverted, option 4 kept

User: "I do not like it. roll back the code for both options 5 and 6." Removed, precisely,
leaving option 4 (`candidate_search_n_threads` + its `parallel_candidates` gate) untouched:

- `candidate_kernels.pyx`: deleted the `_score_cache_init/free/grow/lookup/upsert` hash-map
  functions; the `SignLSHBandIndex` fields, `__cinit__` param, and `__dealloc__` cleanup;
  reverted `find_pair_rows_full_cosine`/`_signed`'s signatures (dropped `sid_delta_norm`);
  reverted `_find_pair_rows_meta` to its pre-option-6 form (no cache lookup/upsert in the
  touched-candidate loop, no score-cache stats in `last_stats`). `_find_pair_rows_meta_
  parallel` and its three helper functions (option 4) were never touched by option 6 and
  are untouched by this revert either.
- `library_corrtrack_parallel.py`: removed `candidate_refresh_interval` and
  `candidate_score_cache` from `CorrTrack.__init__` and `Candidates.__init__` (signatures +
  body state); removed both kwargs from the `Candidates(...)`/`index_cls(...)` construction
  call sites; deleted `_update_score_cache_delta_norm`, `_update_cached_pair_templates`,
  `_replay_cached_pair_templates` entirely; removed the delta-norm call site from
  `_input_tree_lsh_batch`; reverted `_increment_candidates`'s LSH branch to always call the
  real search every step (no refresh-interval gating, no score-cache row_args extension) --
  `candidate_search_n_threads`'s own parallel dispatch logic in that same branch is
  unchanged.

**Verified clean**: `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py
-q` -> 131/131 passed, locally and on Abaca (job 3104019) after resync. `grep` for
`candidate_refresh_interval|candidate_score_cache|_score_cache_|_cached_pair_templates` across
`library_corrtrack_parallel.py` returns nothing. Re-ran the option-4 gating check
(`verify_gate.py`) and the direct sequential-vs-parallel row-set diff (`verify_parallel_lsh.py`,
byte-identical at n_threads in {1,2,4,8}, both signed/unsigned) -- both pass exactly as
before, confirming option 4 survived the surgery untouched.

### Status of the density-dependence investigation: closed for now

Final standing: **option 4 (prange parallelism) is the one surviving, real, exact win** from
this whole investigation -- up to 2.24x on the densest tested dataset (streamflow), byte-
identical output, gated correctly behind `parallel_candidates`. Options 5 and 6 are both
reverted, not merely disabled -- the code no longer exists in the working tree. The
underlying finding stands and is not undone by the rollback: CorrTrack's speedup collapses
when true correlated-degree exceeds the LSH's calibrated occupancy, and every real dataset
curated this session (weather, streamflow, S&P500, Wikipedia) lands in that unfavorable,
high-degree regime at real scale.

**User's resulting direction**: rather than continuing to chase density-dependence fixes,
pivot to sourcing real-world datasets that are BOTH genuinely correlation-sparse (low true
degree, the regime CorrTrack already demonstrably wins in -- see the 2026-09-12 sparse-
synthetic 3.66x result) AND scalable (large m), to build the paper's real-data story around
the method's actual strength rather than continuing to fight its disclosed limitation.

**Next exact step**: search for and evaluate candidate real-world sparse+scalable datasets
across domains (the original brief was finance/environmental/one more, France/Brazil or
worldwide preferred, US acceptable as fallback) -- this time explicitly screening for LOW
true correlated-degree at real scale, not just density in the density/(m-1) sense that
turned out to be misleading earlier this session (see the 2026-09-11(e) "degree metric bug"
entry) -- before investing in any further fetch/curation work on a dataset.

## 2026-09-14 (c) -- Five-way HammingExactIndex comparison; validation-cost profiling
disproves "drop validation" idea; real redundant-enumeration bug found and fixed in BOTH
HammingExactIndex and SignLSHBandIndex (the default lsh_sign_dot backend)

### Five-way comparison: bruteforce / exact_stomp / filcorr / hamming+dotgate / hamming
no-dotgate, across wikipedia/sp500/smartmeter/streamflow

Script: `hamming_exact_compare.py` (scratchpad). Gamma calibrated per dataset via an offset
sweep below `corr_threshold` (`[0.10, 0.15, 0.20, 0.25]`), picking the fastest config meeting
`target_recall=0.95` (or best-recall fallback). All speedups below are vs. plain bruteforce.

| dataset | exact_stomp | filcorr | hamming+dotgate | vs stomp | vs filcorr | hamming no-dotgate | vs stomp | vs filcorr |
|---|---|---|---|---|---|---|---|---|
| wikipedia | 1.87x | 1.46x | 1.10x | 0.59x | 0.75x | 0.99x | 0.53x | 0.67x |
| sp500 | 2.93x | 1.84x | 2.12x | 0.72x | 1.15x | 1.59x | 0.54x | 0.86x |
| smartmeter | 2.29x | 1.89x | 1.88x | 0.82x | 1.00x | 1.33x | 0.58x | 0.70x |
| streamflow | 2.13x | 1.90x | 1.32x | 0.62x | 0.70x | 1.04x | 0.49x | 0.55x |

Consistent findings: the dot-gamma gate always wins over no-dotgate; Hamming-exact+dotgate
ties/beats FilCorr on 2/4 datasets but never beats exact_stomp; recall lands at 0.96-0.99
(not 1.0 -- the Hamming pre-filter is probabilistic), precision exactly 1.0 throughout (the
exact Pearson validation stage never lets a false positive through). These numbers predate
the fix below and are now stale for HammingExactIndex; not re-run across all four datasets
yet (listed as a next step).

### Validation is not the bottleneck: user's "compute a Pearson proxy per pair, like STOMP,
and only validate promising pairs" idea was already implemented in HammingExactIndex

User initially proposed dropping exact validation in favor of an incremental sketch-based
Pearson proxy computed for every pair (like STOMP's own incremental per-pair update), then
clarified: not dropping validation -- computing the cheap proxy incrementally and validating
only the pairs where it looks promising. That is exactly `HammingExactIndex`'s existing
two-stage design: packed-bit Hamming/XOR-popcount as a cheap Stage-1 filter, the exact
dot+gamma gate as Stage 2 (`_passes_dot_gamma_gate`), then exact Pearson validation for
survivors only -- nothing new needed to build for that half of the idea.

Direct time-breakdown profiling (`hamming_time_breakdown.py`, sp500) confirmed validation
was never the cost driver: `validation_time=0.225s`, only 1.9% of `wall=11.89s`;
`candidate_time=12.899s` (>100% due to overlap) dominates almost entirely. This redirected
the investigation to the actual bottleneck: the O(m^2) enumeration/dot-product stage, not
validation.

### Real bug found: redundant same-time enumeration in the "recent vs alive" architecture

Both `HammingExactIndex._find_pair_rows_meta` and `SignLSHBandIndex._find_pair_rows_meta`
insert ALL of a step's new windows into the index before any of them query. For two
SIMULTANEOUS new windows A and B (same time T), the pair gets discovered independently from
BOTH directions -- once when A's query scans and finds B, once more when B's query scans and
finds A. This is genuinely duplicate computation (Hamming/dot-product work), not just
duplicate output -- the existing `_pair_seen_insert`/`_canonical_window_pair` per-call dedup
already collapsed the final output before this fix; the waste was entirely upstream of that
dedup. Comparisons between a recent window and an OLDER (different-time) window are never
redundant -- each corresponds to a genuinely unique `(t1,t2)` tuple that is never repeated --
only the same-time sub-case is doubly examined.

Measured directly on sp500 (`hamming_enum_check.py`, HammingExactIndex): `total_enumerated
=274,742,640` vs. the theoretical unordered-pair-lag count `m*(m-1)/2 * n_lagged_windows *
n_steps = 144,943,200` -- a 1.90x ratio, confirming the redundancy. `total_touched=
111,701,886` (40.6% Hamming pass rate at the recall-forced operating point -- a weak filter
here); `total_dot_checks=94,743,387`.

### Fix

Added an `is_recent` uint8 array (sized `self._count`, populated from `recent_entry_ids` at
the top of each `_find_pair_rows_meta` call) marking which entries belong to THIS call's
query batch. In the touched-candidate discovery loop (the innermost band/bucket-member
scan), skip a candidate `node` when it is genuinely a same-time recent-vs-recent pair AND a
tie-breaker favors not testing it now, so exactly one direction claims each same-time pair.
Final skip condition, identical in both classes:

```
is_recent[node] and time_idx[node] == q_time and sid_idx[node] <= q_sid
```

This only removes wasted upstream work; the final candidate SET returned is unchanged
(`pair_seen` already deduped it before this fix).

**Two real bugs surfaced while landing this, caught by an actual failing test, not by
inspection:**

1. First attempt used `sid_rank[node] <= q_rank` as the tie-breaker, assuming `sid_rank` is a
   stable, unique-per-series identity (true for this project's own usage, matching
   `_canonical_window_pair`'s synchronous-case tie-break elsewhere). This is NOT guaranteed
   by the general `insert_many` API -- `sid_rank_in` is a caller-supplied parameter with no
   uniqueness contract. Caught by
   `test_lsh_sign_dot_index_flat_posting_list_survives_repeated_insert_drop_churn` failing
   (recall collapsed from 100% to ~1.27%, `0.012658227848101266` exactly): that test passes
   the round number (identical for every series inserted that round) as `sid_rank_in`, so
   `sid_rank[node] <= q_rank` ties for every same-round pair, making BOTH directions skip and
   silently dropping the pair entirely. Fixed by switching the tie-breaker to `sid_idx`, the
   field genuinely guaranteed unique per logical series by construction.
2. The initial condition also lacked an explicit time check, implicitly assuming
   `is_recent[node]==True` implies same time as the query -- true for this project's own
   actual CorrTrack call pattern (`recent_entry_ids` is always exactly one step's synchronized
   new batch), but not a general guarantee of the public API: the same failing test's
   `query_entry_ids` spans the ENTIRE alive population across multiple time-rounds
   (re-querying, not just the newest batch), which the general API must legally support.
   Fixed by adding an explicit `time_idx[node] == q_time` conjunct, necessary in general even
   though redundant for this project's own real usage.

Debugged via a standalone reproduction script (`debug_churn_test.py`, scratchpad) that
prints per-round `found`/`expected` counts for the failing test's exact logic, isolating each
bug in turn (confirmed `is_recent` population itself was correct; confirmed disabling the
whole skip condition restores 100% correctness; then pinpointed the literal `round_i`
argument in the test's `insert_many` call as the `sid_rank` non-uniqueness root cause).

### Verification

- `python3 -m pytest test_stable_reproduced_changes.py test_synth_density.py -q` -> 131/131
  passed, with the fix genuinely active in both classes (re-confirmed after a temporary
  disable-for-baseline-measurement/restore cycle on `SignLSHBandIndex`, see below).
- Standalone reproduction of the previously-failing churn test: 100% correct
  (`found == expected`) across all 8 rounds.
- Row-set cross-validation: `verify_parallel_lsh.py` -- sequential (fixed) kernel vs. the
  unmodified parallel kernel, byte-identical row counts at `n_threads` in {1,2,4,8}, both
  `neg_corr` directions (17643/17643 and 17401/17401), `only_in_seq=0`, `only_in_par=0`.
- Real-data recall/precision unchanged to the decimal: `verify_hamming_fix_correctness.py`
  (sp500, gamma=0.6) -> post-fix recall=0.966717 precision=1.000000, matching the pre-fix
  reference (0.9667/1.0000) exactly.
- Forced a clean rebuild (`rm -f candidate_kernels.c candidate_kernels*.so` then rebuild) at
  one point to rule out stale compilation as an explanation for an early failure -- ruled out.

### Measured payoff

**HammingExactIndex, sp500:** `candidate_time` 12.899s -> 7.814s (39% reduction); `wall`
11.89s -> 9.44s (21% faster). `total_touched` dropped 111,701,886 -> 94,743,387 (now equal to
`total_dot_checks`, confirming the redundant duplicates were eliminated before the dot-check
stage, not merely deduped after it).

**SignLSHBandIndex (default `lsh_sign_dot` backend), sp500:** measured via a new script
`lsh_approx_enum_check.py` (scratchpad), `CorrTrack(candidate_backend="lsh_approx", ...)`,
sequential exec, accumulating `total_touched`/`total_dot_checks` from `last_stats` over the
full run. `total_touched` dropped 121,648,955 -> 105,453,691, a 13.3% reduction -- real but
smaller in relative terms than HammingExactIndex's (consistent with LSH's bucket-based
pre-filtering already eliminating most far-apart pairs before the redundancy could manifest
as broadly). `total_dot_checks` was identical in both states (105,453,691), matching the
post-fix `total_touched` exactly -- pre-fix, the larger 121.6M touched collapsed to the same
105.4M via the pre-existing `pair_seen` dedup; post-fix, every touched candidate proceeds
straight to the dot-check with no further dedup needed. Confirmed reproducible across 6
independent post-fix runs (touched count identical every time: 105,453,691) and 4 independent
pre-fix runs (121,648,955 every time) via repeated sed-disable/rebuild/measure/restore
cycles.

**Wall-clock for `lsh_sign_dot` was NOT cleanly separable from run-to-run noise on this
machine.** Six post-fix timing runs: candidate_time in [12.408s, 16.239s]. Four pre-fix runs:
candidate_time in [12.767s, 15.698s]. The ranges overlap substantially (min pre-fix 12.767s
vs. min post-fix 12.408s -- close, within the observed noise band); an earlier
`verify_nthreads.py`-based attempt was similarly noisy. Reporting this honestly rather than
picking a favorable pair of runs: the touched-candidate-count reduction is solid and
reproducible, but a clean wall-time win for the default `lsh_sign_dot` backend on sp500 is
not established by this session's measurements, unlike HammingExactIndex's wall-time win
(12.899s->7.814s candidate_time), which was clear and consistent. Plausible explanation not
yet verified: 13.3% fewer touched candidates is a smaller relative reduction than
HammingExactIndex's (111.7M->94.7M plus the far larger upstream 1.9x raw-enumeration cut),
and per-candidate dot-product cost for `SignLSHBandIndex` may be swamped by other per-step
overhead (band/bucket bookkeeping, Python/numpy call overhead in the measurement harness
itself) at a level comparable to the savings. Not investigated further this session.

### Known issues / not done

- `SignLSHBandIndex._find_pair_rows_meta_parallel` (the option-4 prange kernel) was NOT given
  this same fix -- only the sequential `_find_pair_rows_meta` methods of both classes were
  touched. Left as a disclosed scope gap.
- The five-way `hamming_exact_compare.py` table above predates this fix for
  HammingExactIndex and is now stale; not re-run across the four datasets this session.
- `lsh_sign_dot`'s before/after comparison was only run on sp500; not extended to
  wikipedia/streamflow/smartmeter.
- Not yet synced to Abaca / rebuilt there.

### Next exact step

Sync `candidate_kernels.pyx` to Abaca and rebuild; then decide whether to re-run the five-way
comparison across all four datasets with the fix active, and/or get a cleaner (more
repeats, less loaded machine, or median-of-N) wall-time comparison for `lsh_sign_dot` before
concluding anything about its real-world speedup impact from this fix.

## 2026-09-14 (d) -- lsh_sign_dot cleanup completed across all datasets; Python-loop audit
and fix; idea 4 ("track known pairs, search only for new ones") built, tested, and found
to fail at series-pair granularity -- a real, decisive negative result pinpointing the
exact fix a viable version needs

### Corrections to two claims made earlier this session (both caught before being relied on)

1. **Idea 3 ("build an incremental sketch update") was already implemented.** Read
   `_incremental_sketches` (`library_corrtrack_parallel.py:10862`) directly: the sketch
   projection for a basic-window slot reuses the previous dot product via a cheap ±1
   sign-ratio multiply (`diff_toggleVector = toggleVector[1:]/toggleVector[:-1]`) whenever
   the underlying data block hasn't changed, and only pays a real projection cost for the
   genuinely new basic-window slice each step. The ~17-20% of wall time measured as
   `sketch_time` (wikipedia, both `lsh_hamming_exact` and `lsh_sign_dot`) is already close
   to this scheme's architectural floor, not a naive recompute waiting to be fixed.
2. **The first idea-4 persistence number (99.9%) was a measurement bug, caught before
   reporting it as final.** It compared against `ct.correlated`, confirmed (via `grep`) to
   be reset only at construction/explicit reset -- a whole-run cumulative accumulator, not
   a per-step live set. An append-only set trivially overlaps ~100% with itself step to
   step; that number measured "how fast does the total known-pairs log grow," not
   persistence. Corrected by monkey-patching `_iter_validated_numeric_step` (the call that
   hands the per-step buffer to the monitor right before `_reset_validated_numeric_step()`
   clears it) to capture the genuinely-per-step validated rows. **Corrected result:** mean
   persistence 71.6% (wikipedia), 80.5% (sp500) -- real and still a majority, but far from
   the initial figure.

### "No Python in the hamming_exact hot path" -- audited end to end, one real (small) hit
found and fixed

User flagged a recollection that the Hamming-exact candidate backend runs partly in
Python. Verified directly rather than re-asserting from memory:
- The candidate SEARCH itself (Hamming pre-filter + dot-gate + enumeration) is confirmed,
  by reading the call site (`library_corrtrack_parallel.py:~14927`), to be ONE Cython call
  per step (`row_finder(*row_args)`, e.g. `find_pair_rows_full_cosine[_signed]`), batched
  over the whole `recent_entry_ids` array inside `candidate_kernels.pyx` -- no Python loop.
- The candidate INSERTION path (registering each step's new windows before search) DOES
  have a real per-series Python loop: `_input_tree_lsh_batch`'s dict-branch iterates
  `last_partition.items()` and calls `_get_or_create_window_idx` (dict `.get()` lookups +
  list `.append()` calls) once per series per step. Confirmed via monkey-patched
  instrumentation this is the path actually exercised for `data_representation=
  "sketch_proj"` + `lsh_hamming_exact` (52,008 calls on wikipedia/600 steps, 112,668 on
  sp500/240 steps -- exactly m-per-step). **Measured cost: 0.69% of wall time on sp500**
  (0.0945s / 13.778s) -- real, but not the driver of any of this session's "doesn't beat
  STOMP" findings (those are the already-Cython search cost itself).

**Fix applied** (user confirmed: fix it anyway, for hygiene, given the small measured
payoff and the shared/correctness-sensitive nature of the function): `_get_or_create_
window_idx` now returns `(idx, sid_idx, sid_rank)` instead of just `idx` -- every one of
its 5 call sites was doing `sid_meta = int(self._win_sid_idx[win_idx]); rank_meta =
int(self._win_sid_rank[win_idx])` immediately after calling it (confirmed identical
pattern at all 5 sites via grep), a redundant pair of list-index lookups on top of an
already-Python per-item call. All 5 call sites updated to unpack the tuple directly. Free-
list/slot-reuse semantics (a past memory-leak-fix area) were NOT touched -- same branches,
same order of operations, only the return value changed. **Verified**: 131/131 tests pass;
re-measured loop cost afterward (0.0760s on sp500) -- as expected, no material wall-clock
change, since the theoretical ceiling was always ~0.7%. This was scoped as a hygiene fix,
not a performance lever, and delivered as exactly that.

### `lsh_sign_dot` cleanup completed across all four datasets (the remaining "do all of
these" item from the previous entry)

**Clean (non-concurrent) five-way rerun, dot-gate config, speedup vs exact_stomp:**

| dataset | pre-fix (stale, prior entry) | post-fix (clean rerun) |
|---|---|---|
| wikipedia | 0.59x | 0.66x |
| sp500 | 0.72x | 0.74x |
| smartmeter | 0.82x | 0.86x |
| streamflow | 0.62x | 0.64x |

Modest, consistent gains everywhere; none close to beating `exact_stomp`. Note: an
earlier, concurrently-run version of this same rerun showed smartmeter briefly at 1.05x
(beating STOMP) -- redone cleanly (no other jobs running), it's 0.86x. That 1.05x was
contention noise, not a real result; flagging this explicitly since it would have been an
easy, wrong thing to report.

**`lsh_sign_dot` (default backend) touched-candidate reduction, all four datasets** (a
deterministic count, insensitive to timing noise, so trustworthy even where wall-clock
isn't):

| dataset | pre-fix touched | post-fix touched | reduction |
|---|---|---|---|
| wikipedia | 17,617,920 | 16,061,414 | 8.8% |
| sp500 | 121,648,955 | 105,453,691 | 13.3% |
| streamflow | 416,003,527 | 377,091,925 | 9.4% |
| smartmeter | 617,763,572 | 512,933,174 | 17.0% |

Consistently positive across every dataset. Abaca sync + rebuild + test also reconfirmed
clean on a real compute node (131 passed, 245 subtests, 23.67s) -- the one earlier SIGILL
crash was from building directly on the shared frontend with `-march=native` baking in
frontend-specific CPU features, not a real bug; rebuilding via `oarsub` on an allocated
compute node fixed it.

### Idea 4 ("split tracking of known-active pairs from discovery of new ones"): built,
tested, and found to fail at series-pair granularity -- a precise, decisive negative result

**Design, built in `candidate_kernels.pyx`'s `HammingExactIndex`:**
- New state: `_active_pair` (an `(n_series, n_series)` uint8 symmetric membership matrix,
  `None` by default) + `_active_pair_n`.
- New method `set_active_pairs(n_series, sid_a, sid_b)`: rebuilds the matrix fresh from
  scratch every call (no incremental growth-tracking -- simplest and safest at the m~500
  scale tested, at most a few hundred KB to reallocate).
- `_find_pair_rows_meta`'s touched-candidate loop gets one new early-skip check (after the
  existing `is_recent`/redundant-enumeration checks, before the Hamming distance
  computation): if `has_active_pair and active_pair_view[q_sid, sid_idx[node]]`, skip --
  avoids paying Hamming+dot-gate cost for a candidate whose pair is already known-active.
- **Verified a true no-op by default**: `self._active_pair` stays `None` until a caller
  opts in via `set_active_pairs`; 131/131 tests pass unchanged (none of them call it).

**Python-side prototype** (`idea4_prototype.py`, scratchpad, not wired into the library):
tracks validated `(s1, s2, lag)` triples across steps; each step, before `ct.run()`, calls
`set_active_pairs` with the tracked pairs (collapsed to `(s1,s2)`, since the Cython skip
is series-pair-granular, not lag-aware) and monkey-patches `_get_validated_corr_numeric`
to inject rows `[s1, s2, T_now, T_now-lag, w]` for every tracked `(s1,s2,lag)` directly
into validation, bypassing the search entirely for them. `T_now = ct.window_index[0]` --
confirmed via `Sketches._curr_startTime` that every series shares one current-window start
time each step, so this needs no per-series lookup. The exact-Pearson kernel
(`_cy_validate_corr_rows`) is a pure function of `(data, rows, window_index[0], threshold,
neg_corr)`, so an injected row is validated exactly the same way a search-discovered one
would be -- no validation is skipped for any pair, only the discovery route changes.

**Result on wikipedia (m=88): recall collapsed 0.9808 -> 0.5743, precision held at
1.0000, wall time got slightly WORSE (0.98x), not better.** Precision staying perfect
confirms the injection/validation plumbing itself is correct (no data-corruption bug --
every row that validates positive is genuinely correlated). The recall collapse was first
suspected to be a bug in the Python-side tracking dict (it was originally keyed by
`(s1,s2)` only, silently overwriting when a pair validates at multiple simultaneous lags
in the same step) -- fixed to key by `(s1,s2,lag)` and re-ran: **recall and touched-count
came back byte-identical (0.5743, 11,961,448)**, proving that bug was NOT the cause and
isolating the real one precisely: **the search-side skip is granular at the series-pair
level, but real correlations commonly span multiple simultaneous lags** (matching this
project's own synthetic-generator design notes on `lag_band`). Once a pair is marked
"active" from even one lag, the search stops looking for it at ANY lag, and any other
lag relationship for that same pair -- present at the time of tracking or newly emerging
later -- becomes permanently undiscoverable via search; only the specific already-tracked
lag(s) get re-injected. This is a structural flaw in the granularity of the design, not a
coding error, and it compounds over time as more pairs accumulate "active" status.

**What a viable version needs**: lag-aware tracking, not series-pair-aware -- either (a) a
per-pair SET of already-tracked lags checked against each candidate's actual time
relationship (`time_idx[q_entry] - time_idx[node]`) inside the Cython skip, not a single
bit per pair, or (b) track and inject a pair's FULL currently-valid lag range each step
(already available every step from `_validated_numeric_rows_step`, which does report every
simultaneously-valid lag) while making the skip itself lag-aware too. This is a larger,
more invasive structural change than the version built here, not yet attempted this
session.

### Known issues / not done

- Idea 4's lag-aware version is not built. The current `set_active_pairs`/skip mechanism
  in `candidate_kernels.pyx` is real, tested, additive infrastructure (a true no-op by
  default) but is NOT safe to enable as-is (causes the recall collapse documented above)
  -- must not be wired into any production config surface in its current form.
- `SignLSHBandIndex` was not given the idea-4 skip mechanism (only `HammingExactIndex`
  was, for this first prototype).
- The `_get_or_create_window_idx` fix was not synced to Abaca yet (pure-Python change, no
  Cython rebuild needed there, but the file itself hasn't been copied over).

### Next exact step

Either (a) design and build the lag-aware version of idea 4's skip (a real, larger
implementation: per-pair lag sets, not a single bit), or (b) if that's judged too large a
lift for the ceiling it would recover (mean persistence was 71-80%, not the 99%+ first
misreported), close out idea 4 as a documented negative result and redirect effort
elsewhere. Sync `_get_or_create_window_idx`'s fix to Abaca. Decide whether the `idea4_*`
scratchpad scripts' technique (monkey-patching `_iter_validated_numeric_step`/
`_get_validated_corr_numeric` for per-step introspection) is worth formalizing into a
reusable test/profiling utility, given it was built twice this session already.

## 2026-09-14 (e) -- Idea 4, lag-aware version: built entirely in Cython (per user
instruction: no Python loops), a critical bug found and fixed, and the FIRST real,
positive net speedup this whole session's investigation has produced

### Fully Cython implementation (no Python-level tracking loop anywhere)

Per the user's explicit instruction to avoid Python loops altogether, the lag-aware
tracked-pair mechanism was built entirely inside `HammingExactIndex`
(`candidate_kernels.pyx`), replacing the earlier series-pair-only `set_active_pairs`
(removed, not just disabled -- it was a confirmed dead end):

- **New persistent state**: a hash set (`_tracked_occupied/_tracked_key_a/_tracked_key_b/
  _tracked_cap/_tracked_count`) tracking `(sid_a, sid_b, lag)` triples -- reuses the
  existing `_pair_seen_init/_grow/_clear/_insert` primitives verbatim (they already take
  raw buffer pointers as parameters, not `self`-bound state, so no new hash-table code was
  needed) by packing `key_a = (sid_a << 32) | sid_b`, `key_b = lag`. A new read-only
  `_pair_seen_contains` was added (mirrors `_pair_seen_insert`'s probe sequence without its
  insert side effects -- needed for a check inside the hot loop that must not mutate
  state). A parallel dense list (`_tracked_list_a/_b/_lag/_count`) supports O(k) iteration
  for building injected rows, avoiding a sparse hash-table scan.
- **`update_tracked_pairs(rows)`**: one Cython loop rebuilding both structures from
  scratch from this step's full validated rows (freshly-discovered and
  just-reconfirmed-via-injection alike). A pair absent from `rows` this call is simply
  absent from the rebuilt set -- no explicit "dropped below threshold" bookkeeping needed.
- **`get_tracked_rows(T_now, window_size)`**: one Cython loop building
  `[s1, s2, T_now, T_now-lag, w]` for every tracked triple.
- **The touched-candidate loop's skip check is now lag-aware**: canonicalizes
  `(q_sid, sid_idx[node], q_time, time_idx[node], q_rank, sid_rank[node])` into
  `(lookup_s1, lookup_s2, lookup_lag)` using EXACTLY `_canonical_window_pair`'s own
  tie-break rule (verified by reading `_canonical_window_pair` and the row-assembly code
  that follows it directly, not assumed), then does a read-only hash lookup. A DIFFERENT
  lag for the same series pair is never blocked by this check -- only the exact tracked
  triple is skipped, fixing the earlier series-pair-only version's structural flaw.
- **Verified a true no-op by default**: `_tracked_count == 0` from `__cinit__` until a
  caller opts in; 131/131 tests pass unchanged.
- The Python-side driver (`idea4_prototype.py`, scratchpad) now does zero per-pair Python
  iteration -- exactly two calls into Cython per step (`get_tracked_rows`,
  `update_tracked_pairs`), replacing the earlier prototype's Python dict + per-row loops
  entirely.

### First test after the lag-aware fix: recall STILL collapsed (0.9808 -> 0.5743,
identical to the series-pair-only version) -- a second, more fundamental bug found

Precision stayed perfect throughout (1.0000), again confirming the validation math itself
was never wrong -- something was suppressing genuinely correlated candidates from ever
reaching validation with correct data. Direct instrumentation (`ct.window_index[0]` printed
before/after `ct.run()` across steps) found it: **`window_index` holds the WHOLE retained
lag buffer, not just the current window** -- `window_index[0]` is the OLDEST retained
timestamp, and diverges hugely from the true current-window start once the buffer fills
past `window_size` (measured directly: 31 vs 46 by step 24 on wikipedia). The correct
current-window start is `window_index[-curr_window_size]` (`CorrTrack._curr_startTime()`'s
own formula). Every injected row had been silently anchored to a stale, wrong window the
entire time -- this, not the lag-vs-series-pair granularity, was the dominant cause of the
original recall collapse; the granularity fix was real and necessary but couldn't show its
benefit until this bug was also found and fixed.

### After fixing T_now: recall recovered dramatically, and a REAL, positive net speedup
appeared for the first time this session

**Wikipedia** (m=88): recall 0.9808 (baseline) -> 0.9356 (idea4), precision 1.0000 both.
Wall time still slightly worse (0.94x) -- dataset too small for the search-skip savings to
outweigh the injection/bookkeeping overhead.

**sp500** (m=492, denser): recall 0.9667 (baseline) -> 0.9463 (idea4), precision 1.0000
both. **candidate_time 15.168s -> 10.897s (28% reduction), wall speedup 1.077x** -- the
first genuinely positive net wall-time result this entire session's investigation has
produced from a structural (not just waste-reducing) mechanism. Still slower than
`exact_stomp`'s own clean time (~7.9s on this dataset from the 2026-09-14(d) clean rerun),
so this does not yet beat STOMP outright, but the trend (net benefit appearing and growing
at larger/denser scale, where earlier this session's redundant-enumeration fix and the
naive series-pair skip both failed to help or actively hurt) is a genuinely new and
promising direction.

### Residual recall gap characterized (not fully closed): a plausible, disclosed
discovery-latency limitation, not a validation-correctness issue

A direct diagnostic (`idea4_missing_diag.py`) compared idea4's final numeric correlated
rows against bruteforce ground truth on wikipedia: of 3,866 missing (sid1,sid2,t1,t2,w)
tuples, a 200-pair sample showed **192/200 (96%) had some OTHER lag for the same
(sid1,sid2) pair correctly found** -- ruling out a broad structural miss and pointing to
something narrower. Most likely explanation (plausible, matches the pattern, not yet
exhaustively proven): a tracked lag that transiently dips below threshold for exactly one
step gets correctly dropped from tracking for the NEXT step (via `update_tracked_pairs`
rebuilding fresh from that step's validated rows) -- but the search-side skip for that
exact triple was still in effect DURING that same step (the skip state is fixed before
`ct.run()` starts, before knowing injection will fail), producing a narrow, one-step
"flicker" blind spot at that exact window even though the pair is rediscovered normally
the very next step once untracked. This is a discovery-LATENCY tradeoff, not a removal of
validation -- every row CorrTrack ever reports as correlated is still exactly, genuinely
validated (precision stayed 1.0000 throughout every experiment this session touching this
mechanism); the tradeoff is that a handful of individual (pair, exact-window) instances can
be missed for one step during a transient dip.

### Known issues / not done

- The "flicker" hypothesis for the residual recall gap is plausible and consistent with
  the diagnostic evidence, not proven by a targeted reproduction yet.
- Only tested on wikipedia and sp500 so far; smartmeter and streamflow not yet run through
  the fixed prototype.
- The mechanism lives only in `HammingExactIndex`; `SignLSHBandIndex` (the default
  `lsh_sign_dot` backend) does not have it.
- Still a scratchpad prototype (`idea4_prototype.py`), wired via runtime monkey-patching of
  `_get_validated_corr_numeric`/`_iter_validated_numeric_step` for experimentation --  not
  wired into `library_corrtrack_parallel.py`'s real `_increment_candidates` call path.
- Does not yet beat `exact_stomp` outright on any tested dataset, though sp500 shows the
  gap closing for the first time via a real mechanism rather than incremental waste
  reduction.

### Next exact step

Test smartmeter and streamflow through the fixed prototype to see whether sp500's positive
trend holds or grows at larger/denser scale (the working hypothesis, given the mechanism's
benefit scaled up rather than down going from wikipedia to sp500). If the trend holds,
decide whether to invest in closing the residual "flicker" gap (e.g., a short grace period
before a dip actually un-tracks a pair) and/or wiring this into
`library_corrtrack_parallel.py` for real (replacing the monkey-patch prototype) before
declaring it production-ready.

## 2026-09-14 (f) -- The "blind spot" is fixed: root cause was a precise off-by-one-step
bug, not a flicker. Fixing it exposed a NEW, more fundamental cost problem: re-validating
and re-monitoring a large persistent tracked set every step costs more than the search
savings recover.

### The real root cause, found via a targeted trace, not theorized

The earlier "flicker" theory (2026-09-14(e)) had a logical hole, caught before pursuing a
fix based on it: if an injected row and a hypothetical fresh search both test the EXACT
SAME (s1,t1,s2,t2) data, they must compute the identical Pearson value -- a transient dip
below threshold would cause BOTH to fail identically, so blocking search for that exact
tracked triple could never lose anything by itself. This logical check redirected the
investigation before building the wrong fix.

A targeted step-by-step trace (`idea4_trace_pair.py`, following one specific missing pair
from the earlier diagnostic through every nearby step) found the actual bug: `T_now`,
read via `ct.window_index[-ct.curr_window_size]` BEFORE calling `ct.run()` for that
iteration, is one `window_step` STALE relative to what validation actually uses DURING
that same call -- `ct.run()` advances the window to reflect its own newly-ingested data
before validating. Confirmed precisely: at the step whose pre-run read gave `T_now=859`,
the row genuinely validated inside that SAME call showed `t1=862` (859+STEP). Checked
against the specific missing row from the earlier diagnostic, `(44,19,865,850,30)`: the
step whose pre-run read was 862 predicts `862+STEP=865`, an exact match.

**Fix**: `T_now = window_index[-curr_window_size] + window_step` instead of the bare
formula. **Result**: recall on wikipedia went from 0.9356 (buggy lag-aware version) to
**0.9846 -- actually higher than the baseline's own 0.9808** (idea4 does normal search
PLUS a targeted re-check of tracked pairs, occasionally catching a borderline case the
probabilistic Hamming pre-filter alone misses on a purely fresh search). On sp500: 0.9463
-> 0.9783 (baseline 0.9667, also now above baseline). Precision stayed 1.0000 throughout
every experiment. The blind-spot problem is resolved, not just reduced.

### Fixing recall correctness reveals a new, more fundamental finding: net wall-time
regressed once the mechanism works correctly

Four repeated, clean (idle-machine) sp500 runs after the fix: idea4 vs baseline speedup
0.815x, 0.846x, 0.776x, 0.761x -- **consistently slower**, not the earlier apparent 1.077x
(measured under the STILL-BUGGY off-by-one version, before this fix). `total_touched` is
nearly identical between the buggy and fixed versions (94,019,633 vs 94,116,393) -- the
search-skip mechanism itself works the same either way; what changed is that the FIXED
version's injected rows now correctly reconfirm, so tracked pairs are retained far more
reliably (matching the corrected, higher recall) instead of churning in and out of
tracking every step or two the way the buggy version's failing injections caused.

A direct per-phase profile (`idea4_overhead_profile.py`) found precisely where the extra
cost is, and where it ISN'T: `get_tracked_rows` and `update_tracked_pairs` (this session's
new Cython methods) together took 0.03s out of a 19.3s run -- genuinely free, confirming
the Cython implementation itself is not the problem. The cost is entirely inside `ct.run()`
itself (99.8% of wall time). The tracked-set size explains why: **mean 2,618 pairs per
step, spiking to 17,795** -- a properly-functioning, high-persistence tracked set (matching
sp500's own measured 80.5% mean persistence from 2026-09-14(c)) means thousands of pairs
get reinjected and pushed through the FULL validation + monitor + record pipeline every
single step, even when nothing about their status has changed. Skipping the SEARCH
(Hamming/dot-gate) for these pairs saves real work, but apparently costs less than what
re-running the complete downstream pipeline (exact Pearson kernel call, monitor state
update, correlated-accumulator append) for thousands of redundant reconfirmations adds
back, at real dataset scale.

### What this means going forward

The current design pays near-full per-pair processing cost for tracked pairs every step
(cheaper DISCOVERY route, but the same full VALIDATION+MONITORING route) -- this isn't
enough once the tracked set is large, because that downstream cost scales with tracked-set
size and dominates once persistence is high (which is exactly the regime idea 4 targets).
A design that could still win would need to make the per-step cost for an
already-confirmed, unchanged pair genuinely O(1) all the way through -- e.g., true
STOMP-style incremental sufficient statistics maintained directly for tracked pairs
(bypassing not just the search but the exact-Pearson-kernel-call-plus-monitor-plus-record
pipeline too), only entering the heavier path when a pair's status actually changes
(newly correlated, or drops below threshold). This is a materially different, larger
design than what has been built so far, not a small patch on the current mechanism.

### Known issues / not done

- smartmeter and streamflow are being re-run with the fix (the earlier logged numbers for
  both were from the still-buggy off-by-one version and are superseded/unreliable).
- The larger incremental-sufficient-statistics redesign (the one path that could plausibly
  still win) has not been attempted.
- Still a scratchpad prototype; not wired into `library_corrtrack_parallel.py` for real.

### Next exact step

Get the fixed smartmeter/streamflow numbers to complete the four-dataset picture. Decide,
with the user, whether to pursue the larger incremental-sufficient-statistics redesign
(the one remaining path with a plausible chance of a genuine, robust net win) or close
idea 4 out here: correctness now fully understood and fixed, but the current design's
per-step downstream cost for a large tracked set outweighs its search-side savings on
every real dataset tested so far.

## 2026-09-14 (g) -- Fixed-version four-dataset results: apparent wins on smartmeter/
streamflow did not replicate -- honest final conclusion is no reliable net speedup

Fixed-version (both the lag-aware granularity fix and the off-by-one T_now fix) results,
one run each initially:

| dataset | baseline recall | idea4 recall | speedup (idea4 vs baseline) |
|---|---|---|---|
| wikipedia | 0.9808 | 0.9846 | 0.893x |
| sp500 (4 runs) | 0.9667 | 0.9783 | 0.815x, 0.846x, 0.776x, 0.761x |
| smartmeter (run 1) | 0.9621 | 0.9688 | 1.031x |
| smartmeter (run 2, repeat) | 0.9621 | 0.9688 | 0.867x |
| streamflow | 0.9862 | 0.9898 | 1.082x |

Recall/precision are now solid and reproducible everywhere (idea4 matches or slightly
EXCEEDS baseline recall on every dataset, precision 1.0000 throughout -- the correctness
fix from 2026-09-14(f) holds across all four datasets). Speedup is not: a smartmeter
repeat landed at 0.867x against the first run's 1.031x, with `total_touched` IDENTICAL
between the two runs (459,874,208 both times) -- proving the algorithm itself is fully
deterministic and the wall-clock swing is pure environmental noise, not a real
difference in work done. Given sp500 needed 4 repeats to reveal a consistent (negative)
pattern, and the one smartmeter repeat immediately flipped sign, the honest conclusion is
that the apparent positive readings (sp500's initial 1.077x, smartmeter's first 1.031x,
streamflow's 1.082x) are not established as reliable, reproducible effects -- they are
individual samples from a noisy distribution whose center, per the per-phase profiling in
2026-09-14(f) (thousands of tracked pairs pushed through the full validation+monitor+
record pipeline every step), is structurally expected to run at or below breakeven.

**Final, honest conclusion for this design**: the lag-aware tracked-pair mechanism is now
fully CORRECT (both bugs found and fixed, recall matches/exceeds baseline everywhere,
precision perfect everywhere) but does NOT demonstrate a reliable net wall-time win on any
of the four real datasets once measured with repetition. The 2026-09-14(f) diagnosis
stands as the explanation: skipping the search for tracked pairs saves real work, but
paying the full downstream validation+monitor+record cost for a large, persistent tracked
set (mean ~2,600 pairs/step on sp500, likely more on the denser datasets) costs at least
as much back. A design that could still plausibly win would need genuine O(1)
incremental-sufficient-statistics tracking bypassing that whole downstream pipeline too --
not attempted, a materially larger redesign than what was built.

### Status: idea 4 closed for now, pending a decision on the incremental-sufficient-
statistics redesign

Everything built this session (`update_tracked_pairs`/`get_tracked_rows`/the lag-aware
skip in `HammingExactIndex`) remains in `candidate_kernels.pyx` as tested, correct,
true-no-op-by-default infrastructure (131/131 tests pass without it being invoked) -- not
wired into any production config path, and not recommended for that without the bigger
redesign, since its wall-time payoff is not established as real.

## 2026-09-14 (h) -- Checked before pursuing the incremental-sufficient-statistics
redesign: it would not have helped. The real bottleneck is the skip-check's own
per-candidate overhead, not downstream validation/monitoring.

Before implementing the incremental-sufficient-statistics redesign proposed in
2026-09-14(f), decomposed exactly where idea4's extra cost goes using CorrTrack's own
`sketch_time`/`candidate_time`/`validation_time`/`monitor_time` attributes
(`idea4_phase_breakdown.py`), rather than assuming the earlier tracked-set-size
correlation implied the downstream pipeline was the cause.

**sp500, one paired run**: `candidate_time` +2.515s (8.086s -> 10.601s), `validation_time`
+0.014s, `monitor_time` +0.066s, `sketch_time` -0.001s. The 2026-09-14(f) diagnosis was
wrong about WHERE the cost is (it correlated tracked-set size with total slowdown and
inferred the downstream pipeline, but never actually decomposed the phases) -- the real
cost is almost entirely inside `candidate_time`, i.e. the search phase itself, exactly
where the lag-aware skip check lives.

**Root mechanism**: `HammingExactIndex` has no bucketing -- every query does an
exhaustive O(alive_count) scan, and for `n_vectors<=64` the per-candidate cost being
skipped (one XOR + one popcount on a single packed 64-bit word) is already near the
floor of what a CPU operation can cost. The lag-aware skip check
(`_pair_hash_key` + an open-addressing probe via `_pair_seen_contains`) runs for EVERY
enumerated candidate (guarded only by `self._tracked_count > 0`, not by whether that
specific candidate is actually tracked), so its cost scales with `total_enumerated`
(roughly constant, a property of the dataset/config) regardless of tracked-set size --
consistent with the slowdown magnitude looking similar whether the tracked set was small
(the earlier buggy, churning version) or large (the fixed, stable version). The hash
lookup itself costs more than the single-word comparison it's deciding whether to skip.

**Conclusion: the incremental-sufficient-statistics redesign would NOT help.** It targets
validation/monitoring cost, which the direct measurement shows was never the bottleneck
(both deltas are noise-sized, <0.07s). No redesign of what happens to a tracked pair
AFTER it's identified can fix a problem that lives in IDENTIFYING it -- the cost is
structural to this backend: HammingExactIndex's own per-candidate primitive (one word
compare) is cheaper than any lookup that could decide whether to skip it. This is a hard
architectural mismatch specific to this backend's design (an already near-minimal
exhaustive scan), not a fixable implementation detail.

**Idea 4 is closed for `HammingExactIndex`.** Recall/precision correctness is fully
understood and fixed (2026-09-14(f)/(g)); the mechanism itself is architecturally
unsuited to a backend this cheap per-candidate. Not evaluated for `SignLSHBandIndex`
(the default `lsh_sign_dot` backend) -- that backend's per-candidate cost includes real
bucket/posting-list traversal, plausibly expensive enough that a skip-check could still
win there, but this was not tested and is pure speculation, not a finding.

## 2026-09-14 (i) -- Idea 4 tested on SignLSHBandIndex per user's follow-up ("go ahead
and test it"): ported, verified correct, and found unpromising for a different, precise
reason than HammingExactIndex

### Ported the lag-aware mechanism to SignLSHBandIndex

Identical structure to `HammingExactIndex`'s copy (`_tracked_occupied/_tracked_key_a/
_tracked_key_b/_tracked_cap/_tracked_count` + the dense list + `update_tracked_pairs`/
`get_tracked_rows`), added to `SignLSHBandIndex` in `candidate_kernels.pyx`. The skip
check sits inside the band-scanning loop (`for mi in range(n_members):`), placed after the
existing `is_recent` check, canonicalizing `(q_sid, sid_idx[node], q_time, time_idx[node],
q_rank, sid_rank[node])` into a lookup key exactly as `HammingExactIndex`'s copy does. One
addition beyond the direct port: a tracked-skip also sets `visited_stamp[node] = stamp`
(the is_recent skip deliberately does NOT, left untouched) -- necessary here because a
genuinely correlated pair typically shares MANY band keys, so without marking visited, the
(comparatively expensive) hash lookup would be paid once per band the node re-appears in
for the same query, not once per query; verified safe since a given (q_entry, node) pair's
is_recent/tracked status cannot change mid-query, so marking visited early changes nothing
about which nodes end up touched.

**Verified a true no-op by default**: 131/131 tests pass; `verify_parallel_lsh.py`
(sequential vs the untouched parallel kernel) still byte-identical at n_threads in
{1,2,4,8}, both `neg_corr` directions.

### Correctness confirmed, same pattern as HammingExactIndex

Wikipedia: recall 0.9839 (baseline) -> 0.9873 (idea4), precision 1.0000 both -- again
slightly ABOVE baseline, consistent with the fix landing correctly on this backend too.

### Wall-time: noise-dominated, no reliable signal either direction

Three sp500 runs: 1.023x, 0.858x, 0.881x. A direct phase-time breakdown
(`sketch_time`/`candidate_time`/`validation_time`/`monitor_time`) across three MORE runs
showed `candidate_time` deltas of +1.235s, -2.350s, +4.028s -- wildly inconsistent in
SIGN, not just magnitude, confirming the noise floor at this dataset's ~15-19s runtime
scale swamps whatever real effect exists. `sketch_time`/`validation_time`/`monitor_time`
deltas stayed consistently small (a few hundredths of a second) in every run -- those
phases are not where any effect, positive or negative, lives.

### The decisive, deterministic measurement: only 0.59% touched-candidate reduction

Rather than keep fighting wall-clock noise, measured the one thing that isn't
noise-sensitive: `total_touched`/`total_dot_checks`, baseline (no tracking) vs idea4
(tracking active), on the same sp500 run. **Result: 105,453,691 -> 104,833,122, a 0.59%
reduction.** This is conclusive on its own, independent of any timing question: there is
barely anything for this mechanism to save on this backend.

**Why so small, precisely**: unlike `HammingExactIndex`'s exhaustive per-query scan,
`SignLSHBandIndex` already deduplicates a candidate to exactly one "touch" per query via
`visited_stamp`, regardless of how many of that query's bands find it. My mechanism can
only ever save the fraction of touched candidates that are ALREADY-TRACKED, persistently-
correlated pairs being rediscovered -- and empirically that's a tiny fraction of this
backend's total touched volume. The bulk of what gets touched here is spurious bucket
co-occurrence (unrelated series sharing a band key by chance, at `target_occupancy=5.0`
average bucket size) that a query has never seen before and never will again -- exactly
the kind of touch this mechanism cannot help with, since it only targets rediscovery of
KNOWN pairs, not the dominant source of touched volume in a bucket-based backend.

### Final conclusion: idea 4 does not pay off on either backend, for two distinct,
precisely understood reasons

- `HammingExactIndex`: the skip-check itself (a hash lookup) costs more than the
  already-minimal per-candidate primitive (one word compare) it replaces -- confirmed via
  direct phase decomposition (2026-09-14(h)).
- `SignLSHBandIndex`: the skip-check is cheap enough relative to what it guards, but
  there is almost nothing to skip -- confirmed via a deterministic touched-count
  comparison (0.59% reduction) that this backend's touched volume is dominated by
  one-off spurious co-occurrences, not persistent-pair rediscovery.

Both are real, structural properties of how these two backends already work, not
implementation details that further iteration on THIS mechanism could fix. Correctness
(recall/precision) is solid on both backends -- the mechanism does what it's supposed to
do; it just doesn't move enough real work to matter, or moves work whose skip-check cost
now exceeds the saving.

### Status: idea 4 closed on both tested backends

All infrastructure (`update_tracked_pairs`/`get_tracked_rows`/the lag-aware skip) remains
in `candidate_kernels.pyx` on both `HammingExactIndex` and `SignLSHBandIndex` as tested,
correct, true-no-op-by-default code -- not wired into any production config path, not
recommended for production use given no demonstrated net benefit on either backend.

## 2026-09-14 (j) -- Idea 4 code fully rolled back; Abaca synced and verified

Per user instruction, all idea-4-related code was removed from `candidate_kernels.pyx`
(both `HammingExactIndex` and `SignLSHBandIndex`): the `_tracked_*` state fields,
`__cinit__`/`__dealloc__` init/cleanup, `update_tracked_pairs`/`get_tracked_rows` methods,
the lag-aware skip check in both classes' `_find_pair_rows_meta`, the `lookup_*` local
variables, and the module-level `_pair_seen_contains` helper (left orphaned once both
call sites were removed). `library_corrtrack_parallel.py` was never touched by idea 4
(all wiring lived in scratchpad monkey-patches), so nothing to revert there.

Verified via `grep` (zero remaining references to "idea 4"/"tracked_pair"/
"update_tracked_pairs"/"get_tracked_rows" anywhere in `candidate_kernels.pyx` or
`library_corrtrack_parallel.py`), a forced clean rebuild (`rm candidate_kernels.c
candidate_kernels*.so` then rebuild), the full test suite (131/131), and
`verify_parallel_lsh.py` (sequential vs. the untouched parallel kernel, byte-identical
row sets at n_threads in {1,2,4,8}, both `neg_corr` directions). The tree now reflects
only the session's earlier, kept fixes: the redundant-enumeration (`is_recent`) fix in
both index classes, and `_get_or_create_window_idx`'s return-value fix.

**Abaca synced**: copied both files to `sophia.g5k:~/corrtrack_release_dev/`, rebuilt via
an OAR job on a real compute node (job 3106372 -- the frontend's `-march=native` SIGILL
issue from earlier this session is a known, avoided pitfall, not a recurrence), confirmed
clean (no build errors) and green (131 passed, 245 subtests, 36.54s).

### Next exact step

Continue sourcing real-world datasets that are large-scale AND genuinely low true
correlated-degree (not just low density/(m-1), which was already found misleading this
session -- see the 2026-09-11(e) "degree metric bug" entry), to re-run the five-way
comparison on Abaca once a suitable dataset is found and curated.

## 2026-09-14 (k) -- Dataset sourcing: two candidates in progress, designed around a
specific hypothesis for why earlier datasets all showed high true correlated-degree

### The hypothesis being tested

Every real dataset curated earlier this session (weather/streamflow within one region,
sp500, smartmeter) shares a structural property: all series are drawn from ONE
geographically or economically compact population, so most series share substantial
common risk/driver exposure (regional weather systems, one country's equity market
factors, one utility's grid), and that dense sharing produces the high true correlated-
degree that structurally caps CorrTrack's LSH-based speedup (per the 2026-09-14(b)
closed investigation). The hypothesis: SPREADING the population across genuinely
independent contexts -- geography, currency/market, sector -- should lower average true
degree even as total m grows, because a given series' set of genuinely correlated
partners is bounded by shared-driver exposure, not by how many total unrelated series
exist elsewhere in the dataset. This gives a natural, scalable route to low-degree, large-
m real data: keep adding series from NEW, independent contexts rather than more series
from the SAME context.

### Candidate 1: global weather, geographically dispersed (in progress)

100 points via Open-Meteo's historical archive API (`archive-api.open-meteo.com`, daily
mean temperature 2010-2023, no key needed for research use, confirmed via direct test):
two 8-point tight clusters (Paris region France, Sao Paulo region Brazil, each within
~60km -- a sanity check that genuine short-range correlation still shows up and the
methodology/threshold are reasonable) plus 84 points spread across every inhabited
continent and climate zone, chosen to be mutually >1000km apart in most pairs. Fetched
via `fetch_global_weather.py` (scratchpad), batched (20 locations/call) with exponential
backoff for the API's rate limiting (encountered and handled, not a blocker).

Known confound flagged before evaluation: raw daily temperature carries a strong,
near-global SEASONAL cycle (same-hemisphere locations warm/cool together via shared
orbital forcing, independent of any real regional weather-system correlation;
opposite-hemisphere locations anti-phase) that could spuriously inflate |correlation|
almost everywhere rather than producing genuine sparsity. CorrTrack's own `preprocess`
(differencing) should remove most of this before evaluation -- degree will be measured in
BOTH raw and diff space (mirroring `screen_smartmeter_degree.py`'s established method) to
see this confound directly rather than assume it away.

### Candidate 2: globally-diverse equities (in progress)

66 tickers via Yahoo Finance's public chart API (`query1.finance.yahoo.com`, needs a
browser User-Agent header to avoid a block, confirmed via direct test), 2014-2024 daily
closes, spanning US (20), Europe (18, across France/UK/Germany/Switzerland), Asia (13,
across Japan/Hong Kong/India/South Korea/Singapore), Latin America (7, Brazil/Mexico),
and Australia/South Africa/Canada (8) -- deliberately crossing currencies, market hours,
and sectors, unlike sp500's single-country, single-currency universe. Fetched via
`fetch_global_stocks.py` (scratchpad); aligned onto the busiest ticker's trading calendar
with forward-fill for other exchanges' holiday-calendar gaps (a real, disclosed
methodological choice, not free of its own artifacts -- forward-filled days contribute
zero-variance stretches that must be checked before trusting a "no correlation" reading
on affected days).

### Degree-screening methodology (established earlier this session, reused as-is)

`screen_smartmeter_degree.py`'s approach: run `run_and_log_bruteforce` (both `preprocess`
True and False), take the returned rows, build a `partners[series] = set(other series it
was EVER found correlated with, at any lag/window)` map (collapsing across lag/window
instances -- the true DEGREE question, not density), report `avg_degree`/`max_degree`
over the series that have >=1 partner. This is what will be run on both candidates once
fetched.

### Next exact step

Once both fetches complete: run the degree-screening script on each (raw and diff space).
If either shows genuinely low avg_degree at this m (e.g., low single digits, not tens),
that's a real candidate for the five-way comparison re-run on Abaca; if both still show
high degree, the hypothesis above is wrong or these specific constructions don't
sufficiently decorrelate, and the search continues (a third domain -- e.g., global
COVID case counts by country, or global electricity demand by market zone -- was
considered but not yet started).

## 2026-09-14 (l) -- Global weather (temperature) screened: negative, decisively

Ran the established degree-screening method (`screen_global_weather_degree.py`, mirroring
`screen_smartmeter_degree.py`) on the 100-point global-weather dataset, both raw and diff
space, `window_size=30, window_step=3, n_lags=15, corr_threshold=0.70`.

**Raw space: avg_degree=100.00, max_degree=100 -- every single series correlated with
every other series.** Confirms the anticipated seasonal-cycle confound completely: shared
orbital/seasonal forcing dominates so totally that geographic distance is irrelevant in
raw temperature.

**Diff space (after CorrTrack's own differencing): avg_degree=40.23/100, still high.**
Differencing helps (100 -> 40) but nowhere near enough -- even Paris-region and
São Paulo-region stations (opposite hemispheres, ~9,500km apart) show mutual partners
after differencing. Daily temperature apparently carries persistent globally-correlated
day-to-day fluctuation structure beyond the smooth seasonal curve (plausibly large-scale
climate patterns like ENSO, or systematic calendar-linked residual structure) that plain
differencing does not remove.

**Conclusion: temperature is not a viable variable for this hypothesis, regardless of
geographic dispersal.** The "spread series across independent contexts" strategy needs a
variable whose fluctuations are NOT globally synchronized by a shared physical driver.
Precipitation (much more locally/stochastically driven than smooth temperature curves) is
a plausible alternative if pursuing weather further, but not yet tried. Moving to check
the globally-diverse equities candidate next, a fundamentally different kind of shared-
driver structure (economic/sector risk factors, not a planetary physical cycle).

## 2026-09-14 (m) -- Global equities screened: promising, real, interpretable structure
(unlike temperature's uniform failure)

Ran the same degree-screening method on the 64-ticker, globally-diverse equity dataset
(`screen_global_stocks_degree.py`), `window_size=60, window_step=5, n_lags=20,
corr_threshold=0.70` (matching sp500's own config for comparability).

**Diff-space (the realistic eval config): avg_degree=13.81 over all 64 series, with a
clear, economically-interpretable gradient, not a uniform number:**
- US large-caps (20 tickers, all NYSE/NASDAQ, overlapping trading hours, shared US market
  beta): high degree, 18-31 -- expected, this is exactly sp500-like structure repeated
  within the sample.
- European large-caps: mixed, mostly high (13-28) -- significant US trading-hours overlap
  and global-macro co-exposure.
- Asia-Pacific: LOW degree, several near zero -- `7203.T` (Toyota) degree=2, `005930.KS`
  (Samsung) degree=1, `0941.HK` degree=0, `BHP.AX`/`CBA.AX`/`CSL.AX` (Australia) all
  degree=0, `WALMEX.MX` (Mexico) degree=0.
- **9 of 64 series (14%) have degree 0-1** -- genuinely, verifiably uncorrelated with
  everything else in the sample at this threshold.

**Raw-space: avg_degree=64.00 (every series correlated with every other) -- the same
"raw prices carry a shared trend" confound as every other raw-price financial series
tried this session; diff-space is the only usable evaluation config, as expected and
already this project's established practice.**

**Why this is different from temperature's uniform failure**: correlation between
equities comes from a REAL, economically bounded mechanism (shared trading hours, shared
market-wide risk factors, sector co-movement) that genuinely does NOT extend to
timezone-disjoint, economically-unrelated markets -- unlike a shared planetary physical
cycle (temperature's seasonal forcing), which by construction touches everything. Adding
more tickers from LOW-correlation contexts (more Asia-Pacific/Africa/Latin-America
names) should dilute the average further as m grows, exactly the scalable-sparsity
property the search is looking for.

**Caveat on this specific fetch**: `T=1052` came out shorter than the requested
2014-2024 range because the alignment step required ALL tickers to have data from a
common start date, and one late-IPO ticker (`9988.HK`, Alibaba, listed 2019) forced
trimming the shared history down to ~4 years. A rebuild dropping recent-IPO tickers (or
alignment that doesn't require the full-history intersection) would recover the longer
history.

### Status: global equities is the current leading candidate

Real, interpretable, non-uniform sparse structure found -- unlike temperature. Not yet
scaled to a "large m" candidate (currently m=64, well below sp500's m=492). Reporting
back before committing to a much larger fetch (hundreds of tickers across more
countries/exchanges, which will take a while against Yahoo Finance's per-ticker rate
limiting).

## 2026-09-14 (n) -- Global equities v2 (m=168, T=2824): refined the finding, corrected
a methodology mistake, identified exactly which regions are low-correlation

### v2 fetch: scaled to m=168, T=2824 (~11 years, 2013-2023)

`fetch_global_stocks_v2.py` (scratchpad): 179 candidate tickers across US(40)/EU(44)/
Asia-high-correlation(24: Japan/HK/India/Korea/Singapore large caps)/Asia-low-correlation
(37: Taiwan/Thailand/Malaysia/Indonesia/NZ/Australia/Israel/Turkey)/LatAm(21)/Other(13:
South Africa/Canada/Egypt). Added ThreadPoolExecutor concurrency (4 workers) -- fetched
172/179 in 87s, dramatically faster than v1's sequential approach. Also fixed v1's
IPO-truncation bug properly this time: computed each ticker's first-valid-day as a
fraction of the full requested range BEFORE aligning, reported the worst offenders
(`US:DOW` 57%, `ASIA_HI:9988.HK` 64%, `ASIA_LO:SCB.BK` 85%, `OTHER:SHOP.TO` 22% through
the range -- all genuine late-listings/spinoffs, not fetch bugs), dropped anything
starting after 15% through the range (4 tickers), THEN aligned the remaining 168 --
recovered the full ~11-year span (2,824 of 2,827 possible trading days) instead of v1's
accidental 424-day truncation.

### Result: overall average degree went UP (13.81 -> 32.40), but the underlying
regional structure sharpened and clarified rather than disappearing

Diff-space, same config as sp500 (`W=60, STEP=5, N_LAGS=20, THR=0.70`):

| region | n | avg_degree | median | n_zero |
|---|---|---|---|---|
| ASIA_HI (Japan/HK/India/Korea/Singapore large caps) | 23 | 9.22 | 8.0 | 0 |
| **ASIA_LO (Taiwan/Thailand/Malaysia/Indonesia/NZ/Australia/Israel/Turkey)** | 33 | **4.36** | 4.0 | **7** |
| EU | 44 | 50.00 | 51.0 | 0 |
| LATAM | 19 | 23.32 | 30.0 | 1 |
| OTHER (South Africa/Canada/Egypt) | 12 | 36.17 | 30.5 | 1 |
| US | 37 | 54.35 | 53.0 | 0 |

**Two real findings, not one confounded number:**
1. **Two mistakes explain the overall-average rise, both understood, not mysterious.**
   First, the ticker-list growth added roughly as many new US/EU names as
   low-correlation-region names, so the high-correlation group's SHARE of the total
   sample didn't shrink the way the strategy intended (US+EU = 81/168 = 48% of the
   sample, same order as before). Second, the longer 11-year span (vs v1's accidental
   ~4-year window) includes real market-wide crisis episodes (2020 COVID crash, 2022
   rate-shock selloff) where equity correlations genuinely spike broadly -- a longer,
   more representative sample naturally shows higher average correlation than a shorter
   one that happens to avoid those episodes, and this project's own established practice
   is to use full, real, undoctored spans rather than cherry-picking calm periods.
2. **The regional gradient is real, sharper than v1, and specifically about WHICH
   markets, not "non-US/EU" in general.** LatAm and Canada/South Africa ("OTHER") are
   NOT low-correlation (23-36 avg degree) -- they track global commodity/macro cycles
   closely enough to correlate substantially with the US/EU cluster. The genuinely
   low-correlation markets are specifically Taiwan/Thailand/Malaysia/Indonesia/NZ/
   Israel/Turkey-style smaller, less globally-integrated equity markets (`ASIA_LO`,
   avg_degree=4.36, 7 of 33 names at exactly zero) and, to a lesser extent, the
   Asia-Pacific large-caps (`ASIA_HI`, avg_degree=9.22). Raw-space is again completely
   uniform (avg_degree=168.00, every series correlated with every other) -- the same
   shared-trend confound as every raw-price series tried this session; diff-space
   remains the only usable evaluation config.

### Next exact step

Build v3: do NOT grow US/EU/LatAm/Canada further (they are confirmed high-correlation,
not useful for lowering the average) -- concentrate all new tickers in the
`ASIA_LO`-style category (more Taiwan/Thailand/Malaysia/Indonesia names, plus
Philippines/Poland/Czech-Republic/Greece-style smaller, less globally-integrated markets
if Yahoo coverage allows) to scale m up while keeping the confirmed-low-correlation group
a clear MAJORITY of the sample, not a minority as in v2.

## 2026-09-14 (o) -- Isolated the real lever: distinct markets, not deeper per-market
lists. A genuinely promising standalone candidate found (m=94, avg_degree=6.43)

### Why v3's overall average barely improved despite doubling the low-correlation group

Checked (using already-fetched v3 data, no new fetch): `ASIA_LO`'s OWN average degree
rose from 4.36 (v2, 33 tickers) to 8.97 (v3, 71 tickers) even though only MORE tickers
were added to it, nothing removed. Cause: the new tickers were added mostly WITHIN
markets already represented (more Taiwan names, more Thai names, etc.), and stocks
within the SAME country share that country's own market-wide risk factor -- exactly
recreating multiple small "mini-sp500" clusters inside the supposedly-low-correlation
bucket. The real lever isn't "how many low-correlation-region tickers," it's "how many
DISTINCT markets," with few tickers per market so no single market's internal cluster
dominates.

### Direct test: the Asia-Pacific/smaller-market subset as its OWN universe

Extracted just the `ASIA_HI`+`ASIA_LO` columns from the already-fetched v3 data (94
series, T=2613, no new fetch needed) and ran the same bruteforce degree screen on it
standalone (`screen_asia_subset_degree.py`) -- i.e. does this set look low-degree only
by comparison against the US/EU anchor, or is it genuinely low-degree on its own terms?

**Result: avg_degree=6.43 over all 94 series (median 7.0, max 19), 10 series at exactly
zero.** Much lower than the 30.05 seen when this same group sat inside the full
206-series dataset (that number was dragged up by CROSS-group correlation with the
US/EU anchor, not by anything in this group's own internal structure).

**Per-exact-market breakdown confirms the within-market-clustering mechanism precisely**:
every market shows SOME internal degree, bounded by how many tickers from that market
are in the sample --

| market | n tickers | avg_degree | max | n_zero |
|---|---|---|---|---|
| .SI (Singapore) | 3 | 14.33 | 16 | 0 |
| .HK (Hong Kong) | 5 | 9.60 | 19 | 0 |
| .NS (India) | 5 | 8.20 | 12 | 0 |
| .AX (Australia) | 12 | 7.83 | 11 | 0 |
| .IS (Turkey) | 9 | 7.78 | 11 | 0 |
| .TW (Taiwan) | 10 | 7.70 | 15 | 1 |
| .T (Japan) | 8 | 7.62 | 9 | 0 |
| .KS (Korea) | 2 | 7.00 | 8 | 0 |
| .WA (Poland) | 5 | 6.40 | 9 | 0 |
| .JK (Indonesia) | 10 | 5.40 | 7 | 0 |
| .AT (Greece) | 3 | 5.33 | 7 | 0 |
| .BK (Thailand) | 9 | 5.00 | 10 | 0 |
| .KL (Malaysia) | 2 | 3.50 | 5 | 0 |
| **.NZ (New Zealand)** | 6 | **0.33** | 1 | 4 |
| **.TA (Israel)** | 5 | **0.00** | 0 | 5 |

Israel's 5 tickers show ZERO correlation with anything in the entire 94-series universe,
including each other. New Zealand is nearly as sparse. Markets with more tickers (.AX,
.TW, .JK, .BK: 9-12 each) show correspondingly higher internal degree -- directly
confirming per-market ticker count, not region, is the real degree-driving variable.

### This is a genuinely strong, immediately usable candidate

m=94 (comparable in scale to wikipedia's m=88, this session's smallest real dataset),
T=2613 (~10.4 years daily, longer than every other real dataset curated this session),
avg_degree=6.43 with real zero-degree series -- a much more favorable operating point
for CorrTrack's LSH advantage than sp500/smartmeter/streamflow's high-degree regime that
originally motivated this whole search (2026-09-14(b)). To reach sp500-scale m (~500),
the identified path is clear: add MANY MORE distinct markets (aim for 2-4 tickers each,
not 8-12) rather than deepening the ones already present -- Yahoo Finance has coverage
for dozens of additional national exchanges not yet tried (e.g. more of Eastern Europe,
more of the Middle East, more of Southeast Asia, more African exchanges beyond South
Africa/Egypt).

### Status: reporting back before further scaling

A real, usable, well-understood candidate exists now at m=94. Growing it to sp500-scale
m via "many more distinct markets, few tickers each" is a large additional fetch effort
(potentially 60-100+ more distinct-market tickers) -- checking in before committing to
that scope, given how much iteration (temperature failure, v2's mixed-category mistake,
v3's within-market-clustering discovery) has already gone into reaching this point.

## 2026-09-15 (a) -- First real wins against exact_stomp this entire session, on two
independently-constructed, honestly-sourced real datasets. Clean Abaca measurements.

### Context: the skepticism check that led here

The user directly challenged the dataset-sourcing methodology ("Is this application
realistic? I am skeptical") after the 2026-09-14 series of hand-tuned global-equity
constructions (v1-v4) kept iterating the ticker list specifically toward lower measured
degree -- a fair methodological objection (fitting the dataset construction to the target
metric). Response: fetched the ACTUAL, live, publicly-disclosed holdings of the real
iShares MSCI ACWI ETF directly from ishares.com (2,251 real holdings, ticker/weight/
country/exchange, as of Sep 11 2026) and built two datasets from it using selection rules
fixed BEFORE looking at any correlation outcome:

1. **`acwi_real`**: top 4 holdings by weight within EACH of the 40 countries the fund
   maps to a confident Yahoo suffix (a diversification-capped slice -- no single country
   can dominate the sample, matching how a risk-managed global fund actually caps country
   concentration). m=125, T=2425 (2013-2023, ~9.6y after dropping 8 late-listed names).
2. **`acwi_capweighted`**: top 300 holdings by RAW GLOBAL WEIGHT, no per-country cap --
   the fund's actual unconstrained market-cap emphasis (necessarily US-mega-cap-heavy,
   since that's what global cap-weighting produces by construction). m=263, T=2452 (37
   recent-IPO names dropped, e.g. PLTR/UBER/SNOW/CRWD/DASH -- today's largest tech names
   are disproportionately recent listings).

Degree screening (same bruteforce-based method used throughout this session) confirmed
the expected, honest contrast BEFORE running the five-way comparison: `acwi_real`
avg_degree=22.21 (125 series, real developed-vs-emerging gradient, Mexico/Egypt/NZ near
zero); `acwi_capweighted` avg_degree=119.95 (263 series, median 141 -- 166 of 263 are US,
avg_degree=156.02 within that group alone) -- essentially recreating sp500's dense
structure, exactly as expected from an undiversified cap-weighted slice. Both real,
neither hand-picked to hit a target.

### Clean, single-OAR-job Abaca measurements (dedicated compute node each, not the noisy
shared local machine used for most of this session's other numbers)

**`acwi_real`** (m=125, T=2425, W=60/STEP=5/N_LAGS=20/THR=0.70, same config as sp500):

| method | wall | recall | speedup vs bf |
|---|---|---|---|
| bruteforce | 4.7s | -- | 1.00x |
| exact_stomp | 2.5s | 1.0000 | 1.90x |
| filcorr | 4.5s | 1.0000 | 1.05x |
| **lsh_hamming_exact+dotgate** | **2.35s** | 0.9709 | **2.00x** |
| lsh_hamming_exact (no dotgate) | 3.61s | 0.9903 | 1.30x |

**lsh_hamming_exact+dotgate beats exact_stomp: 1.05x**, and beats filcorr by 1.90x.

**`acwi_capweighted`** (m=263, T=2452, same config):

| method | wall | recall | speedup vs bf |
|---|---|---|---|
| bruteforce | 22.1s | -- | 1.00x |
| exact_stomp | 12.6s | 1.0000 | 1.75x |
| filcorr | 11.6s | 1.0000 | 1.90x |
| **lsh_hamming_exact+dotgate** | **8.52s** | 0.9763 | **2.60x** |
| lsh_hamming_exact (no dotgate) | 17.07s | 0.9954 | 1.30x |

**lsh_hamming_exact+dotgate beats exact_stomp: 1.48x** -- despite this dataset having
avg_degree comparable to or higher than sp500's own high-degree structure, where the
dot-gate config previously measured 0.72-0.86x (never beating STOMP) across every prior
real dataset this session (wikipedia, sp500, smartmeter, streamflow).

**This is the first time this entire session's investigation has found a real dataset
where CorrTrack's sketch/candidate-search machinery beats exact_stomp outright**, and it
happened on two independently-constructed real datasets, not one. Precision is 1.0000 in
every configuration, as established throughout this session -- these are real speed wins,
not accuracy tradeoffs.

### Open question, not yet explained: why does acwi_capweighted beat STOMP despite being
nearly as dense as sp500?

`acwi_capweighted`'s avg_degree (119.95) is comparable to sp500's own high-degree regime
that was established (2026-09-14(b) and earlier) as the reason LSH-based speedup
collapses. Yet it beats STOMP by 1.48x here. Candidate explanations, none yet verified:
`acwi_capweighted` has a longer time span (2013-2023, ~10y) than the sp500 dataset used
throughout this session, and roughly half the m (263 vs 492) -- either could change the
touched-candidate-volume-vs-m²  relationship in ways not yet isolated. Not investigated
further yet; flagged as a real, open, worth-explaining question rather than claimed as
understood.

### Status

Two real, honestly-sourced, non-cherry-picked datasets now show CorrTrack beating
exact_stomp for the first time this session. Both synced to and measured on Abaca (clean,
dedicated-node results, not local-machine noise). Datasets: `tmp_artifacts/acwi_real/
acwi_real.npz`, `tmp_artifacts/acwi_capweighted/acwi_capweighted.npz`. Result JSONs saved
on Abaca at `~/corrtrack_release_dev/tmp_artifacts/hamming_exact_compare_acwi_real.json`
and `..._acwi_capweighted.json`.

### Next exact step

Investigate why `acwi_capweighted` beats STOMP despite high average degree (the open
question above) -- likely candidates to isolate: T (longer span), m (smaller than
sp500), or something about the SHAPE of the degree distribution (a right-skewed
distribution with a smaller number of very-high-degree hubs vs sp500's own distribution
shape) rather than the average alone. Consider also re-running sp500 itself at a matched
T to sp500's history length as a further isolation check.

## 2026-09-15 (b) -- Methodology updated per user request (lsh_sign_dot tuned replaces
the no-dotgate ablation); negative control rules out `m` as the explanation for
acwi_capweighted beating STOMP

### Comparison methodology changed

Per explicit user request, `hamming_exact_compare.py`'s fifth arm changed from
"lsh_hamming_exact, no dot-gate" (an internal diagnostic ablation) to "lsh_sign_dot,
tuned" (`candidate_backend="lsh_approx"`, the actual default production backend,
`target_occupancy=5.0` per the 2026-09-14 occupancy-sweep finding that 3-5 is already
near-optimal, gamma calibrated via the same offset sweep as the hamming+dotgate arm) --
this reflects the real user-facing choice rather than an internal ablation. All datasets
re-run under this corrected methodology.

### Negative control: `m` alone does not explain why acwi_capweighted beats STOMP

Subsampled sp500 to its own first 263 series (`sp500_sub263`, m=263 matching
`acwi_capweighted` exactly, T=1255 unchanged -- sp500's own native, high-degree
structure at the SAME m) and ran the identical five-way comparison on Abaca. **Result:
does NOT beat STOMP** (hamming+dotgate 0.93x, lsh_sign_dot tuned 0.75x) -- consistent
with sp500's own full-m behavior (0.72-0.74x range throughout this session), not with
acwi_capweighted's 1.48-1.51x win. This rules out `m` as the driver: shrinking sp500 to
acwi_capweighted's exact scale does NOT reproduce the win. The remaining candidate
explanations are `T` (sp500_sub263's T=1255 vs acwi_capweighted's T=2452, roughly half)
or something about the specific degree-distribution shape/underlying data structure,
not just scale.

### Re-run results so far (Abaca, clean single-node measurements), corrected methodology

| dataset | m | T | hamming+dotgate vs stomp | lsh_sign_dot(tuned) vs stomp |
|---|---|---|---|---|
| acwi_real | 125 | 2425 | 1.05x | 0.85x |
| acwi_capweighted | 263 | 2452 | 1.51x | 1.17x |
| sp500_sub263 (control) | 263 | 1255 | 0.93x | 0.75x |

Note: `lsh_sign_dot` (tuned) beats STOMP on `acwi_capweighted` (1.17x) too, though by a
smaller margin than `hamming_exact+dotgate` (1.51x) -- the win is not specific to one
backend. On `acwi_real`, `lsh_sign_dot` does NOT beat STOMP (0.85x) even though
`hamming_exact+dotgate` does (1.05x) -- backend choice matters more at the lower-degree
end.

### Status: full seven-dataset comparison in progress

Also re-running wikipedia/sp500/smartmeter/streamflow under the same corrected
methodology on Abaca (these had not been synced to Abaca this session -- now done) for a
complete, consistent seven-dataset table. Results pending.

## 2026-09-15 (c) -- The complete, corrected, seven-dataset picture: measurement
environment (dedicated Abaca node vs. shared local machine) turns out to matter more
than any algorithmic finding investigated earlier this session

### Full table, clean single-OAR-job-per-dataset Abaca measurements, corrected
methodology (hamming_exact+dotgate and lsh_sign_dot-tuned, per the user's 2026-09-15
request replacing the no-dotgate ablation)

| dataset | m | T | hamming_exact+dotgate vs STOMP | lsh_sign_dot (tuned) vs STOMP |
|---|---|---|---|---|
| wikipedia | 88 | 1827 | 1.08x | 0.92x |
| sp500 | 492 | 1255 | 0.90x | 0.72x |
| smartmeter | 510 | 15000 | 0.97x | 0.73x |
| streamflow | 538 | 2192 | 1.30x | 1.09x |
| acwi_real | 125 | 2425 | 1.05x | 0.85x |
| acwi_capweighted | 263 | 2452 | 1.51x | 1.17x |
| sp500_sub263 (control) | 263 | 1255 | 0.93x | 0.75x |

Full per-arm detail (calibration trials, recall at every gamma) is in each dataset's
saved `tmp_artifacts/hamming_exact_compare_<dataset>.json` on Abaca.

### The bigger finding: these numbers are dramatically better than every earlier
same-dataset measurement from this session's local machine

| dataset | earlier local-machine number (2026-09-14, clean rerun) | this Abaca number |
|---|---|---|
| wikipedia | 0.66x | **1.08x** |
| sp500 | 0.74x | 0.90x |
| smartmeter | 0.86x | 0.97x |
| streamflow | 0.64x | **1.30x** |

Every single one moved substantially in the same direction, by far more than run-to-run
noise (this session repeatedly measured 5-20% swings from noise alone; these gaps are
40-100%+). Two real, disclosed differences between the two measurement environments,
either or both of which could explain this:
1. **Dedicated vs. shared hardware**: the 2026-09-14 numbers came from this session's own
   local development machine, used concurrently for many other things throughout the
   session (explicitly documented as noisy -- e.g. the 2026-09-14(f)/(g) entries where a
   repeated measurement flipped sign entirely, 1.031x -> 0.867x, with identical work
   done). Abaca's OAR jobs run on a dedicated, exclusively-allocated compute node.
2. **`-march=native` compiled for a different, possibly more capable CPU**: `setup_cython.py`
   bakes in `-march=native -O3 -ffast-math` (see its own 2026-07-30 comment on relying on
   this for auto-vectorization). Compiled on Abaca's own node, this targets THAT node's
   actual instruction set (plausibly wider SIMD, e.g. AVX-512, than the local development
   machine) -- and the operations this would help most (packed-bit Hamming
   XOR+popcount, the dot-gate's vector reduction) are exactly the CorrTrack-side
   operations, not STOMP's/bruteforce's simpler scalar running-sum updates -- so a
   faster, wider-SIMD CPU would disproportionately help CorrTrack's own machinery
   relative to the baselines it's compared against, moving the ratio in exactly the
   direction observed.

Neither cause has been isolated from the other yet (both are real, present differences
between the two environments; this entry states what's disclosed and true -- the
measurements differ hugely -- without yet claiming to know which mechanism is
responsible for how much of the gap).

### Practical implication: the earlier "CorrTrack structurally can't beat STOMP on real
data" conclusion (the premise behind the whole idea-4 investigation and the dataset
search that followed) does not hold once measured on dedicated, representative hardware

Three of four ORIGINAL real datasets (wikipedia, streamflow, and smartmeter nearly at
breakeven) now show hamming_exact+dotgate at or above parity with exact_stomp -- only
sp500 stays clearly below. This means idea 4's premise -- "CorrTrack's LSH-based
machinery never beats STOMP on real data, so something structurally different is needed"
-- was itself measured on the noisy environment and may not have been true on
representative hardware in the first place. This is not a reason to regret the idea-4
investigation (the correctness bugs found and fixed there -- the off-by-one T_now bug,
the lag-granularity fix -- are real, verified, independent of this) but it reframes the
motivating premise significantly.

### Known issues / not done

- The dedicated-hardware-vs-`-march=native` confound is not isolated. A clean test would
  compile WITHOUT `-march=native` (or with a fixed, portable target) on both machines and
  re-measure, to separate "faster/dedicated CPU" from "wider SIMD instruction set
  specifically helping CorrTrack's own kernels."
- The local machine's own exact CPU model/instruction set support was never queried this
  session for direct comparison against Abaca's compute node.

### Next exact step

Decide whether to isolate the dedicated-hardware-vs-SIMD-width confound (compile
identically, e.g. `-march=x86-64-v2` or similar portable baseline, on both machines and
re-measure sp500 as a representative case), or accept the Abaca numbers as the
project's authoritative, representative measurement going forward (dedicated hardware is
what any real deployment would look like anyway) and move on.

## 2026-09-15 (d) -- The hardware/measurement confound, investigated and resolved: it's
contention, not CPU capability

### Step 1: checked actual CPU capabilities on both machines (didn't assume)

Local machine: Intel Core i7-1165G7 (11th Gen, Tiger Lake, 2020) -- AVX2 AND full
AVX-512, including `avx512_vpopcntdq` (a hardware population-count instruction directly
relevant to the Hamming-distance popcount CorrTrack's kernels do).

Abaca compute node (checked via a real OAR batch job, not assumed): Intel Xeon E5-2650 v2
(Ivy Bridge, 2013) -- only plain AVX, not even AVX2, and scalar POPCNT only, no AVX-512
at all.

**This is the OPPOSITE of the original hypothesis.** Abaca's compute node has an OLDER,
LESS CAPABLE CPU than the local machine, not a wider-SIMD one -- so "`-march=native`
targeting a more capable Abaca CPU" cannot be the explanation for Abaca's better
relative numbers.

### Step 2: direct test -- does removing AVX-512 locally recover Abaca's numbers?

Rebuilt locally with `-march=haswell` (AVX2, explicitly no AVX-512) instead of
`-march=native`, re-ran wikipedia's five-way comparison. **Result: speedup_vs_stomp for
hamming+dotgate = 0.69x -- essentially identical to the original AVX-512 build's
historical number (0.66x), not closer to Abaca's 1.08x.** Disabling AVX-512 locally did
not move the result meaningfully. This rules out AVX-512 downclocking (a real,
documented phenomenon on client/mobile Intel chips under sustained AVX-512 use) as the
explanation -- removing the supposedly-throttling instructions didn't help.

### Step 3: checked for contention directly, found it directly

`ps aux` on the local machine, checked at the time of this investigation, shows MULTIPLE
OTHER active Claude Code sessions running concurrently (a different session id,
`d1586681-ba0c-4bb8-b842-283619109f22`, plus additional unlabeled `claude` processes),
alongside several VS Code extension-host and Pylance language-server processes, all
competing for the same 8 CPU cores. This is not an inference -- it is a directly observed
fact about the machine this whole session's local measurements were taken on: **it is a
genuinely shared, multi-tenant environment, not a dedicated benchmarking machine.** Abaca
OAR jobs, by contrast, run on an exclusively-allocated compute node for the walltime of
the job -- no other job shares that node while it's held.

### Conclusion: contention, not CPU capability, explains the gap

Ruled out (by direct measurement, not assumption): wider SIMD on Abaca (Abaca's CPU is
strictly older/weaker), AVX-512 downclocking locally (removing AVX-512 didn't change the
local result). Confirmed directly: the local machine has real, concurrent, competing
work from other sessions running on it throughout this whole investigation. Cache- and
memory-bandwidth-sensitive Cython kernels (Hamming/dot-product operations over sketch
vectors) are exactly the kind of workload that suffers disproportionately from cross-
process cache pollution and memory contention, more so than STOMP's small, simple,
cache-friendly running-sum updates -- consistent with contention hurting CorrTrack's
relative standing specifically, not uniformly across all methods.

**Practical conclusion**: Abaca's numbers (the seven-dataset table in the 2026-09-15(c)
entry) are the trustworthy, representative measurement -- a dedicated node is what any
real deployment would actually look like, and the local machine's numbers throughout
this session were measured under real, now-directly-confirmed contention. The local
build was reverted to `-march=native` (its normal state) after this test; no change is
recommended to the actual compile flags -- `-march=native` remains correct for whatever
machine actually runs CorrTrack in production, this investigation was only about
explaining a MEASUREMENT discrepancy between two specific machines, not about changing
build configuration.

## 2026-09-15 (e) -- Threshold sweep (0.80/0.90/0.95) across all 8 datasets, including
the environmental candidate: CorrTrack's speedup vs STOMP grows dramatically as
corr_threshold tightens

### Setup

Extended `hamming_exact_compare.py`: added an optional second CLI arg overriding
`corr_threshold` (default 0.70 per dataset, unchanged), added `global_weather` (the
environmental candidate set aside in 2026-09-14(l) for an unfavorable degree screen,
included now per explicit request -- its earlier negative characterization is not
revisited here, just carried forward), and folded `avg_degree`/`density`/`L` computation
directly into the bruteforce pass (no separate script needed). Ran all 8 datasets x 3
thresholds = 24 jobs on Abaca (`oarsub`, ~3 concurrent slots, serialized into ~8 waves).

### corr_threshold = 0.80

| dataset | m | T | L | BF(s) | density | avg_degree |
|---|---|---|---|---|---|---|
| wikipedia | 88 | 1827 | 6 | 2.9 | 0.000537 | 23.07 |
| sp500 | 492 | 1255 | 5 | 31.8 | 0.000509 | 56.29 |
| smartmeter | 510 | 15000 | 3 | 77.1 | 0.000006 | 2.72 |
| streamflow | 538 | 2192 | 6 | 109.1 | 0.000468 | 168.69 |
| acwi_real | 125 | 2425 | 5 | 4.1 | 0.000286 | 6.83 |
| acwi_capweighted | 263 | 2452 | 5 | 15.5 | 0.000621 | 52.59 |
| sp500_sub263 | 263 | 1255 | 5 | 10.4 | 0.000500 | 28.70 |
| global_weather | 100 | 5113 | 6 | 9.3 | 0.001035 | 2.35 |

| dataset | stomp (spd/recall) | filcorr | hamming+dotgate | lsh_sign_dot |
|---|---|---|---|---|
| wikipedia | 1.20x/1.000 | 1.24x/1.000 | 1.45x/0.990 | 1.31x/0.986 |
| sp500 | 3.46x/1.000 | 4.11x/1.000 | 4.55x/0.969 | 4.71x/0.960 |
| smartmeter | 2.51x/1.000 | 2.58x/1.000 | 2.85x/0.984 | 2.73x/0.959 |
| streamflow | 2.48x/1.000 | 3.02x/1.000 | 3.15x/0.988 | 3.32x/0.978 |
| acwi_real | 1.77x/1.000 | 1.60x/1.000 | 2.26x/0.981 | 1.90x/0.973 |
| acwi_capweighted | 3.00x/1.000 | 3.25x/1.000 | 3.50x/0.988 | 3.43x/0.979 |
| sp500_sub263 | 1.80x/1.000 | 2.01x/1.000 | 3.83x/0.968 | 3.63x/0.959 |
| global_weather | 1.20x/1.000 | 1.22x/1.000 | 1.53x/0.994 | 1.46x/0.994 |

### corr_threshold = 0.90

| dataset | m | T | L | BF(s) | density | avg_degree |
|---|---|---|---|---|---|---|
| wikipedia | 88 | 1827 | 6 | 2.8 | 0.000042 | 2.91 |
| sp500 | 492 | 1255 | 5 | 41.5 | 0.000044 | 4.82 |
| smartmeter | 510 | 15000 | 3 | 125.0 | 0.000002 | 0.66 |
| streamflow | 538 | 2192 | 6 | 89.0 | 0.000127 | 60.57 |
| acwi_real | 125 | 2425 | 5 | 4.6 | 0.000041 | 0.72 |
| acwi_capweighted | 263 | 2452 | 5 | 17.8 | 0.000046 | 3.90 |
| sp500_sub263 | 263 | 1255 | 5 | 10.1 | 0.000041 | 2.53 |
| global_weather | 100 | 5113 | 6 | 9.3 | 0.000964 | 1.30 |

| dataset | stomp (spd/recall) | filcorr | hamming+dotgate | lsh_sign_dot |
|---|---|---|---|---|
| wikipedia | 1.16x/1.000 | 1.18x/1.000 | 1.70x/0.990 | 1.57x/0.982 |
| sp500 | 2.59x/1.000 | 3.05x/1.000 | 7.87x/0.965 | 8.75x/0.956 |
| smartmeter | 2.35x/1.000 | 2.78x/1.000 | 4.71x/0.988 | **5.14x/0.924** |
| streamflow | 2.95x/1.000 | 2.86x/1.000 | 4.90x/0.985 | 6.38x/0.972 |
| acwi_real | 1.87x/1.000 | 1.06x/1.000 | 2.88x/0.985 | 2.58x/0.979 |
| acwi_capweighted | 2.56x/1.000 | 2.81x/1.000 | 4.91x/0.990 | 5.34x/0.981 |
| sp500_sub263 | 1.80x/1.000 | 2.00x/1.000 | 5.11x/0.972 | 5.72x/0.962 |
| global_weather | 1.18x/1.000 | 1.19x/1.000 | 1.91x/0.994 | 1.81x/0.993 |

### corr_threshold = 0.95

| dataset | m | T | L | BF(s) | density | avg_degree |
|---|---|---|---|---|---|---|
| wikipedia | 88 | 1827 | 6 | 1.9 | 0.000003 | 0.32 |
| sp500 | 492 | 1255 | 5 | 40.9 | 0.000005 | 0.47 |
| smartmeter | 510 | 15000 | 3 | 125.0 | 0.000001 | 0.23 |
| streamflow | 538 | 2192 | 6 | 107.4 | 0.000038 | 22.80 |
| acwi_real | 125 | 2425 | 5 | 4.6 | 0.000013 | 0.16 |
| acwi_capweighted | 263 | 2452 | 5 | 17.8 | 0.000007 | 0.35 |
| sp500_sub263 | 263 | 1255 | 5 | 9.3 | 0.000006 | 0.21 |
| global_weather | 100 | 5113 | 6 | 10.1 | 0.000782 | 1.18 |

| dataset | stomp (spd/recall) | filcorr | hamming+dotgate | lsh_sign_dot |
|---|---|---|---|---|
| wikipedia | 1.55x/1.000 | 1.55x/1.000 | 1.78x/1.000 | 1.74x/0.987 |
| sp500 | 2.55x/1.000 | 2.97x/1.000 | 9.62x/0.979 | 12.41x/0.974 |
| smartmeter | 2.34x/1.000 | 2.80x/1.000 | 5.43x/0.976 | **6.56x/0.883** |
| streamflow | 2.49x/1.000 | 3.02x/1.000 | 6.72x/0.980 | 9.42x/0.967 |
| acwi_real | 1.86x/1.000 | 1.04x/1.000 | 3.01x/0.974 | 2.95x/0.960 |
| acwi_capweighted | 2.55x/1.000 | 2.82x/1.000 | 5.61x/0.998 | 6.63x/0.990 |
| sp500_sub263 | 1.21x/1.000 | 1.30x/1.000 | 5.12x/0.974 | 6.29x/0.974 |
| global_weather | 1.37x/1.000 | 1.42x/1.000 | 2.19x/0.994 | 2.13x/0.990 |

### Headline finding: CorrTrack's speedup advantage over exact_stomp/filcorr grows
dramatically as corr_threshold tightens

`exact_stomp`/`filcorr` stay roughly flat across thresholds (both always do the same
O(1)-per-pair exact work regardless of how many pairs actually pass -- their speedup vs
bruteforce is governed only by their fixed per-pair constant, not by the threshold). The
LSH-based methods, by contrast, benefit enormously from a stricter threshold: sp500's
`hamming_exact+dotgate` goes 2.91x (thr=0.70) -> 4.55x (0.80) -> 7.87x (0.90) -> **9.62x
(0.95)**; `lsh_sign_dot` goes even further, 2.33x -> 4.71x -> 8.75x -> **12.41x**. This
makes clear, mechanistic sense: a stricter threshold means the sketch/bucket-based
pre-filter has to distinguish an ever-SPARSER true-positive signal from the noise floor,
and the calibration can afford a correspondingly tighter gamma -- fewer candidates ever
reach the expensive validation stage, while STOMP/bruteforce/filcorr pay the identical
per-pair cost regardless.

### Real caveat, disclosed not glossed over: recall shortfall for smartmeter's
lsh_sign_dot at high thresholds

`smartmeter`'s `lsh_sign_dot` recall is 0.924 at thr=0.90 and **0.883 at thr=0.95** --
both BELOW the 0.95 target. This means the calibration's `meeting = [t for t in trials
if recall >= 0.95]` filter found NO trial among the 4 tested gamma offsets that reached
target recall, so the fallback (`max(trials, key=recall)`) picked the best-available
trial instead -- the reported wall-time/speedup for this cell is real, but it comes with
a genuine, disclosed accuracy shortfall, not silently hit the same guarantee every other
cell in this table meets. Every OTHER cell in the sweep (all other datasets, both
methods, all three thresholds) meets or exceeds 0.95 recall. Not yet investigated
further -- plausible causes (not verified): smartmeter's very low avg_degree at these
thresholds (0.66, 0.23) means very few true positives for the calibration sweep to
detect reliably, or the fixed offset grid `[0.10,0.15,0.20,0.25]` may need to extend
further at very high threshold for this specific dataset.

### The environmental dataset (`global_weather`), included per request

Shows real, positive speedups (1.20x-2.19x for hamming+dotgate across the threshold
sweep) but consistently the SMALLEST margins of any dataset at every threshold, despite
comparably low avg_degree to `acwi_real`/`sp500_sub263` at the higher thresholds (1.30,
1.18 vs acwi_real's 0.72, 0.16) -- degree alone still does not fully predict the
achievable speedup; this dataset's underlying correlation structure (the seasonal/
climate-driven confound documented in 2026-09-14(l)) evidently behaves differently from
the equity datasets' structure even when average degree numbers look similar. Not
investigated further.

### Status

All 24 result JSONs saved on Abaca (`tmp_artifacts/hamming_exact_compare_<dataset>_thr
<NNN>.json`). `precision` was not re-verified as 1.0 in this entry's summary tables (it
was 1.0 throughout every prior measurement this session and the per-trial JSON detail
still records it) -- worth a final spot-check before treating it as unconditionally true
at these new, more extreme threshold values.

## 2026-09-15 (f) -- n_vectors sweep: user's concern (n_vectors=64 > window_size for
every dataset this session) tested empirically and directly explained via the actual
sketch construction

### The concern, and why it was reasonable to raise

`n_vectors=64` has been used unchanged in every comparison this whole session, yet
EVERY dataset tested has `window_size < 64` (wikipedia/streamflow/global_weather=30,
smartmeter=48, sp500/acwi_*=60) -- so the sketch's target dimension has exceeded the
window's own dimension in every single case, which conflicts with the classical
Johnson-Lindenstrauss intuition that a random projection's target dimension should not
exceed its source dimension (projecting UP adds no compression benefit and, naively,
should just add redundant/correlated dimensions).

### Empirical sweep: powers of 2 (per the user's specification), reusing one
n_vectors-independent bruteforce/exact_stomp run per dataset

`nvectors_sweep.py` (scratchpad), sweeping `n_vectors` at fixed `corr_threshold=0.70`,
same gamma-offset calibration as the main comparison. Wikipedia was swept first with a
denser grid (8,16,24,30,48,64) before the user specified powers-of-2 only; the rest used
[4,8,16,32,64].

**wikipedia (W=30)**: n_vectors=8 recall collapses (0.81, never reaches target); recall
and speedup_vs_stomp climb MONOTONICALLY from n_vectors=16 through 64, peaking at 64
(recall=0.9808, speedup_vs_stomp=1.06x) -- no peak or plateau before 64.

**sp500 (W=60)**: same monotonic pattern, far more dramatic. n_vectors=4:
speedup_vs_stomp=0.24x; n_vectors=64: **0.90x** -- nearly a 4x improvement. Absolute wall
time also drops from 47.4s (n_vectors=4) to 12.6s (n_vectors=64) -- n_vectors=64 is
faster in absolute terms too, despite more per-candidate compute, because the tighter
sketch lets far fewer candidates reach the expensive exact-Pearson validation stage.

**global_weather (W=30)** and **acwi_real (W=60)**: same monotonic pattern confirmed,
n_vectors=64 best in both.

### Mechanism, verified directly in the code (not assumed): why the classical JL
"target dim <= source dim" heuristic doesn't apply here

Checked `_sketch_feature_width()` (`library_corrtrack_parallel.py:9988-9989`): it
returns `basic_window`, and `basicRandomVector` (the shared random projection basis) has
shape `(n_vectors, basic_window)` -- for wikipedia, `(64, 3)`. The random basis
genuinely has only `basic_window` (=3) independent random directions, reused UNCHANGED
across all `n_basic_windows` (=10) basic-window slots -- what varies per basic-window
(and per output dimension) is only a random +/-1 sign flip (`toggleVector`), and the
signed copies are summed to form the final sketch. This is not a one-shot JL projection
of the whole window (where "target <= source dimension" would be the relevant
heuristic) -- it is closer to a redundant, structured sign-hashing ensemble over a much
smaller shared basis. Increasing `n_vectors` samples more independent random SIGN
PATTERNS over that same small basis (more "hash checks"), not more projection
directions beyond the data's real dimensionality -- so the naive size-based heuristic,
whether compared against `window_size` OR `basic_window`, does not predict this
construction's behavior. No closed-form rule was found (or should be assumed) to
predict the right n_vectors from window_size directly; this is why the empirical sweep,
not a formula, was necessary.

### User's decision: keep n_vectors=64 throughout

Given the sweep confirms 64 is the best of everything tested (not merely "good enough"),
the user decided NOT to switch to a smaller, "principled" power-of-2 (16 for W=30, 32
for W=48/60) -- every comparison already run this session used n_vectors=64 correctly by
the empirical evidence, and needs no rework.

### Separate methodology note, addressed without a new sweep (per user request not to
sweep W/L for now)

User asked whether window_size/L were chosen with real domain reasoning. Audited
honestly: `smartmeter` (W=48/STEP=8/N_LAGS=16, documented elsewhere as "1-day window,
4-hour step, half-day max lag") and `sp500` (W=60/STEP=5/N_LAGS=20, quarter-window/
week-step/month-lag, standard financial cadences) have plausible domain grounding.
`wikipedia` and `streamflow` share an IDENTICAL config (W=30/STEP=3/N_LAGS=15) despite
being unrelated domains (social attention dynamics vs. hydrology) -- a red flag that at
least one was copied rather than independently derived. `acwi_real`/`acwi_capweighted`/
`sp500_sub263` (matched to sp500 by asset-class analogy) and `global_weather` (matched
to wikipedia/streamflow by resolution analogy) were explicitly NOT independently derived
this session. Resolution reached without a new sweep: the datasets already fall into two
internally-consistent classes (daily financial: W=60/L=5, all four financial datasets
identical; daily social/environmental: W=30/L=6, all three identical; smartmeter stands
alone on its own sub-daily resolution) -- so WITHIN-class comparisons in every table this
session are not confounded by arbitrary per-dataset config drift. CROSS-class
comparisons (financial vs. social/environmental) remain confounded by the differing
window/lag choice and should not be over-interpreted as purely reflecting data
structure differences. No sweep performed at the user's explicit request; this
constraint is disclosed, not resolved.

### Status

streamflow and smartmeter's n_vectors sweeps still running on Abaca as of this entry;
results pending.

## 2026-09-15 (g) -- n_vectors sweep complete: confirmed on all 6 datasets, zero
exceptions, n_vectors=64 is unambiguously the best value tested everywhere

Final two datasets (streamflow, smartmeter) completed the sweep. Full speedup-vs-stomp
(hamming_exact+dotgate) table across all n_vectors tested:

| dataset | n_v=4 | n_v=8 | n_v=16 | n_v=32 | n_v=64 |
|---|---|---|---|---|---|
| wikipedia | -- | 0.80x | 0.99x | -- | 1.06x |
| sp500 | 0.24x | 0.31x | 0.39x | 0.45x | **0.90x** |
| streamflow | 0.43x | 0.56x | 0.64x | 0.78x | **1.22x** |
| smartmeter | 0.29x | 0.37x | 0.46x | 0.53x | **0.95x** |
| global_weather | 0.69x | 0.93x | 1.00x | 0.93x | **1.05x** |
| acwi_real | 0.52x | 0.67x | 0.82x | 0.87x | **1.06x** |

Monotonically increasing in every dataset except one tiny wobble (global_weather dips
from 1.00x at n_vectors=16 to 0.93x at 32, within likely noise, before still peaking at
64) -- n_vectors=64 is the unambiguous best choice in every single dataset tested, often
by a wide margin (sp500: 0.24x -> 0.90x, a ~3.75x improvement from n_vectors=4 to 64).
This closes the user's original concern: despite window_size < 64 in every dataset this
session, n_vectors=64 is empirically the correct choice throughout, confirmed
comprehensively rather than assumed. No further n_vectors work planned; the user's
decision (keep n_vectors=64 throughout) stands, now with full six-dataset support.

## 2026-09-15 (h) -- W/L unified to a single value across all daily-resolution datasets;
n_vectors=64 finalized

Following on from (f)'s within-class/cross-class audit, the user asked whether W/L
should be derived less arbitrarily, or a single value fixed across all domains instead
(to remove the cross-class confound rather than merely disclose it). Presented the
choice directly: unify onto 30 days (current social/environmental default), unify onto
60 days (current financial default), or keep the two-class split. User chose **30 days**.

**Rationale checked before acting**: all affected datasets are same-resolution
(one column = one calendar day) -- wikipedia, streamflow, sp500, acwi_real,
acwi_capweighted, sp500_sub263, global_weather. A shared day-count is therefore a
genuine apples-to-apples span across all seven, not an arbitrary knob. `smartmeter`
alone is native 30-minute resolution (its existing W=48 = exactly 1 day) and forcing it
onto a 30-calendar-day span would misrepresent its actual analytical target (daily
consumption-cycle correlation, not monthly trend) -- kept on its own justified config
(W=48/STEP=8/N_LAGS=16) as a disclosed, deliberate exception, not unified.

**Change made**: `sp500`, `acwi_real`, `acwi_capweighted`, `sp500_sub263` in
`hamming_exact_compare.py`'s CFG changed from W=60/STEP=5/N_LAGS=20 (L=5) to
W=30/STEP=3/N_LAGS=15 (L=6) -- matching wikipedia/streamflow/global_weather exactly.
n_vectors stays fixed at 64 (already the case; user's "fix n_vector=64" restates the (f)
decision, no code change needed there).

**Re-run required**: the four financial datasets' five-way comparison is invalidated by
the window change and must be re-run at all four threshold levels (0.70/0.80/0.90/0.95)
already swept under the old W=60 config, so the full comparison table stays complete.
Submitted 16 OAR jobs on Abaca (sophia.g5k, queue `abaca`), OAR IDs 3107693-3107707,3107709,
covering {sp500, acwi_real, acwi_capweighted, sp500_sub263} x {0.70, 0.80, 0.90, 0.95},
via `hamming_exact_compare.py <dataset> <threshold>` (updated script synced to
`~/corrtrack_abaca_results/hamming_exact_compare.py`). wikipedia/streamflow/
global_weather results at W=30 (already run under the old two-class scheme) remain
valid as-is and do not need re-running -- their config is unchanged by this decision.

### Status (superseded by (i) below -- a second methodology bug was found before
these 16 jobs' lsh_sign_dot numbers were ever reported)

## 2026-09-15 (i) -- Two more real methodology bugs found and fixed: n_bands was
already auto (script was misleading, not actually wrong), occupancy genuinely WAS
fixed and untuned; full 32-job corrected battery run

User asked directly: "does hamming_exact need any tuning besides gamma? What about
stomp/filcorr?", then flagged that `candidate_lsh_n_bands` "should not be fixed, it
is automatically computed" and occupancy "should also be tunned."

**Verified in code, not assumed:**
- `hamming_exact`+dotgate: gamma is the ONLY tunable parameter. `HammingExactIndex.
  _finalize_threshold` (candidate_kernels.pyx:5047) auto-derives the Hamming
  threshold from gamma via the SimHash relation; the margin (`sqrt(n_vectors)`) is a
  fixed constant, not exposed as a parameter. No bands/occupancy concept at all.
- `exact_stomp`/`filcorr`: no tuning whatsoever -- `run_bf_exact_stomp`/
  `run_bf_filcorr` are exact baselines driven only by the shared base config
  (window/step/lags/threshold), confirmed from their signatures (no extra params).
- `candidate_lsh_n_bands`: the script's explicit `candidate_lsh_n_bands=64,
  candidate_lsh_n_bands_tolerance=1.0` was ALREADY a no-op -- `n_bands_tolerance`
  defaults to 1.0 (None -> auto) regardless, and `SignLSHBandIndex._finalize_sizing`
  always overrides any literal n_bands with `ceil(tolerance * corrected_b_min(...))`
  once observed_m is known (verified in `Candidates.__init__` and
  `candidate_kernels.pyx`'s `_finalize_sizing`). So n_bands was never actually fixed
  in any of this session's runs -- the script just LOOKED like it was, which is why
  it's removed now, for clarity, not correctness.
- `candidate_lsh_target_occupancy`: this ONE genuinely WAS fixed at 5.0 and never
  swept -- a real gap. Fixed: `run_lsh_sign_dot`'s calibration loop now sweeps
  occupancy in {2.0, 3.0, 5.0, 8.0, 12.0} jointly with the existing gamma-offset grid
  (20 trials instead of 4), picking the fastest trial meeting recall>=0.95 exactly
  like every other calibration in this session.

**This invalidates every `lsh_sign_dot_tuned` result collected this session**
(the original single-threshold battery, the 4-threshold sweep, and the just-finished
16-job W/L-unification batch) -- `hamming_exact+dotgate`/`exact_stomp`/`filcorr`/
`bruteforce` numbers are UNAFFECTED (occupancy/n_bands are lsh_sign_dot-only
concepts) and remain valid as-is.

Submitted a full corrected battery: all 8 datasets x all 4 thresholds = 32 OAR jobs
on Abaca (IDs 3107734-3107765), using the corrected script
(`~/corrtrack_abaca_results/hamming_exact_compare.py`, synced). All 32 completed
within minutes. Final corrected table (W=30/STEP=3/N_LAGS=15/L=6 for all except
smartmeter at W=48/STEP=8/N_LAGS=16/L=3; n_vectors=64 throughout; occupancy tuned):

| thr | dataset | m | T | L | BF(s) | density | avg_deg | stomp spd/rec | filcorr spd/rec | hamming+dot spd/rec | lsh_sign_dot spd/rec/occ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.70 | wikipedia | 88 | 1827 | 6 | 2.6 | 0.002363 | 63.93 | 1.00x/1.000 | 1.01x/1.000 | 1.09x/0.981 | 0.98x/0.976/5 |
| 0.70 | sp500 | 492 | 1255 | 6 | 67.3 | 0.004150 | 399.80 | 2.64x/1.000 | 3.10x/1.000 | 1.95x/0.979 | 1.77x/0.972/12 |
| 0.70 | smartmeter | 510 | 15000 | 3 | 139.1 | 0.000020 | 20.27 | 1.18x/1.000 | 1.32x/1.000 | 2.14x/0.962 | 1.78x/0.963/8 |
| 0.70 | streamflow | 538 | 2192 | 6 | 89.3 | 0.001194 | 326.09 | 2.79x/1.000 | 2.63x/1.000 | 1.92x/0.986 | 1.85x/0.975/12 |
| 0.70 | acwi_real | 125 | 2425 | 6 | 6.9 | 0.001484 | 68.98 | 1.48x/1.000 | 1.51x/1.000 | 1.43x/0.977 | 1.28x/0.970/8 |
| 0.70 | acwi_capweighted | 263 | 2452 | 6 | 35.9 | 0.003562 | 192.94 | 2.70x/1.000 | 2.64x/1.000 | 2.09x/0.980 | 1.93x/0.972/12 |
| 0.70 | sp500_sub263 | 263 | 1255 | 6 | 18.0 | 0.004045 | 212.41 | 2.62x/1.000 | 2.54x/1.000 | 2.03x/0.979 | 1.87x/0.972/8 |
| 0.70 | global_weather | 100 | 5113 | 6 | 9.3 | 0.001141 | 40.23 | 1.17x/1.000 | 1.16x/1.000 | 1.29x/0.993 | 1.13x/0.992/8 |
| 0.80 | wikipedia | 88 | 1827 | 6 | 2.5 | 0.000537 | 23.07 | 1.04x/1.000 | 1.04x/1.000 | 1.24x/0.990 | 1.24x/0.983/3 |
| 0.80 | sp500 | 492 | 1255 | 6 | 70.4 | 0.001159 | 223.63 | 3.27x/1.000 | 3.95x/1.000 | 3.59x/0.989 | 3.89x/0.981/8 |
| 0.80 | smartmeter | 510 | 15000 | 3 | 108.6 | 0.000006 | 2.72 | 1.85x/1.000 | 2.77x/1.000 | 3.15x/0.984 | 3.12x/0.957/3 |
| 0.80 | streamflow | 538 | 2192 | 6 | 110.7 | 0.000468 | 168.69 | 2.08x/1.000 | 1.13x/1.000 | 2.79x/0.988 | 3.09x/0.975/8 |
| 0.80 | acwi_real | 125 | 2425 | 6 | 7.1 | 0.000460 | 23.63 | 1.66x/1.000 | 1.74x/1.000 | 1.96x/0.990 | 1.95x/0.981/3 |
| 0.80 | acwi_capweighted | 263 | 2452 | 6 | 31.3 | 0.001069 | 121.70 | 2.61x/1.000 | 2.31x/1.000 | 2.24x/0.991 | 2.39x/0.979/5 |
| 0.80 | sp500_sub263 | 263 | 1255 | 6 | 17.2 | 0.001133 | 117.86 | 2.77x/1.000 | 2.74x/1.000 | 2.84x/0.988 | 3.13x/0.980/5 |
| 0.80 | global_weather | 100 | 5113 | 6 | 9.2 | 0.001035 | 2.35 | 1.19x/1.000 | 1.21x/1.000 | 1.51x/0.994 | 1.51x/0.994/3 |
| 0.90 | wikipedia | 88 | 1827 | 6 | 2.6 | 0.000042 | 2.91 | 1.18x/1.000 | 1.19x/1.000 | 1.71x/0.990 | 1.72x/0.985/2 |
| 0.90 | sp500 | 492 | 1255 | 6 | 66.2 | 0.000122 | 38.16 | 3.48x/1.000 | 1.75x/1.000 | 5.21x/0.994 | 7.55x/0.983/2 |
| 0.90 | smartmeter | 510 | 15000 | 3 | 125.1 | 0.000002 | 0.66 | 2.34x/1.000 | 2.79x/1.000 | 4.73x/0.988 | 4.06x/0.952/12 |
| 0.90 | streamflow | 538 | 2192 | 6 | 108.3 | 0.000127 | 60.57 | 2.50x/1.000 | 3.03x/1.000 | 5.27x/0.985 | 6.99x/0.969/2 |
| 0.90 | acwi_real | 125 | 2425 | 6 | 6.6 | 0.000066 | 3.73 | 1.44x/1.000 | 1.48x/1.000 | 2.28x/0.995 | 2.45x/0.989/2 |
| 0.90 | acwi_capweighted | 263 | 2452 | 6 | 30.6 | 0.000105 | 29.32 | 2.68x/1.000 | 2.39x/1.000 | 3.61x/0.997 | 4.31x/0.985/2 |
| 0.90 | sp500_sub263 | 263 | 1255 | 6 | 16.6 | 0.000112 | 19.29 | 2.75x/1.000 | 2.74x/1.000 | 4.24x/0.993 | 5.24x/0.985/2 |
| 0.90 | global_weather | 100 | 5113 | 6 | 9.2 | 0.000964 | 1.30 | 1.20x/1.000 | 1.20x/1.000 | 1.84x/0.994 | 1.98x/0.993/2 |
| 0.95 | wikipedia | 88 | 1827 | 6 | 2.4 | 0.000003 | 0.32 | 0.99x/1.000 | 0.99x/1.000 | 1.70x/1.000 | 1.88x/1.000/2 |
| 0.95 | sp500 | 492 | 1255 | 6 | 64.4 | 0.000013 | 4.05 | 3.39x/1.000 | 1.70x/1.000 | 6.63x/0.994 | 11.21x/0.982/2 |
| 0.95 | smartmeter | 510 | 15000 | 3 | 124.8 | 0.000001 | 0.23 | 2.34x/1.000 | 2.77x/1.000 | 5.30x/0.976 | 5.10x/0.941/12 |
| 0.95 | streamflow | 538 | 2192 | 6 | 107.8 | 0.000038 | 22.80 | 2.50x/1.000 | 3.07x/1.000 | 6.64x/0.980 | 10.37x/0.962/2 |
| 0.95 | acwi_real | 125 | 2425 | 6 | 6.7 | 0.000015 | 0.58 | 1.49x/1.000 | 1.53x/1.000 | 2.63x/0.992 | 2.98x/0.989/2 |
| 0.95 | acwi_capweighted | 263 | 2452 | 6 | 22.9 | 0.000011 | 2.27 | 2.86x/1.000 | 2.62x/1.000 | 4.35x/0.997 | 5.67x/0.987/2 |
| 0.95 | sp500_sub263 | 263 | 1255 | 6 | 15.8 | 0.000013 | 2.02 | 2.62x/1.000 | 2.61x/1.000 | 4.67x/0.989 | 6.05x/0.984/3 |
| 0.95 | global_weather | 100 | 5113 | 6 | 9.2 | 0.000782 | 2.21 | 1.20x/1.000 | 1.23x/1.000 | 2.07x/0.994 | 2.21x/0.990/2 |

This is now the FINAL, authoritative five-way comparison table for this session --
supersedes every earlier version. Notably, the tuned occupancy varies genuinely by
dataset/threshold (2-12) rather than sitting at the old fixed 5.0 default, confirming
the fix mattered: e.g. sp500 at thr=0.95 improved from whatever the untuned-occupancy
run would have given to 11.21x (occ=2) once occupancy was actually searched. Every
cell meets recall>=0.95 except smartmeter's lsh_sign_dot at thr=0.90 (0.952) and
thr=0.95 (0.941) -- consistent with the previously-disclosed shortfall, still not
further investigated (calibration grid may be too coarse for that specific dataset).

Result files: `tmp_artifacts/occfix_results/hamming_exact_compare_<dataset>_thr<NNN>.json`
(all 32, copied from Abaca).

## 2026-09-15 (j) -- Maximal-pool dataset expansion started (for later Sobol-sweep
reproduction across bf/stomp/filcorr/both CorrTrack variants)

User: "I want to have all available data for all these datasets... reproducing the
Sobol sweep with them running all five methods", and separately asked whether m=500
should be fixed for all datasets with a common big T (bigger for smartmeter).

**Feasibility-checked before committing to a fetch plan (not assumed):**
- Wikipedia pageviews REST API has a hard floor at 2015-07-01 (verified: 404 for
  June 2015, valid data from July 2015). Cannot reach 2013 no matter what.
- Open-Meteo archive API and Yahoo Finance both cover 2013 fine.
- smartmeter's household count (m=510) could not grow without Kaggle credentials
  (none configured: no `~/.kaggle`, `kaggle` package not installed, no surviving
  raw source file on disk).

User decisions: shared window = 2013-2023 for everyone except wikipedia (which uses
its max possible 2015-07-01 to 2023-12-31, disclosed as the one shorter-T dataset);
user provided a Kaggle API token (new-format `KGAT_...` bearer token, works directly
against `kaggle.com/api/v1` with `Authorization: Bearer <token>`, no legacy
username+key `kaggle.json` needed) to unblock smartmeter expansion. **Security note**:
token was pasted directly in chat -- not written to any repo file or log, used only
in-session via an env var; user was advised to regenerate it from Kaggle account
settings once this pull is done, since anything pasted into a chat session should be
treated as exposed.

**Built so far** (all in `/tmp/.../scratchpad`, not yet copied into
`tmp_artifacts/` -- pending final m=500/big-T subsampling decision):
- `sp500_full.npz`: m=444, T=2768 (2013-01-02 to 2023-12-29). Down from 492 (the
  5-year window) as expected -- 51 of 495 successfully-fetched tickers dropped for
  unfillable gaps (late listings/reconstitution), confirming the tradeoff flagged
  before the user chose this window.
- `streamflow_full.npz`: m=608, T=4017 (2013-01-01 to 2023-12-31). UP from 538 (had
  only used a subset before) -- fetched from the full ~2419-site USGS CA candidate
  pool (waterservices.usgs.gov site list), 985 sites returned data, 608 survived the
  same densest-calendar + ffill/bfill(limit=3) + drop-unfillable alignment used
  throughout this session.
- `smartmeter_full.npz`: m=2953, T=27649 (2012-08-01 to 2014-02-28). Built from ALL
  112 halfhourly blocks of the Kaggle "Smart meters in London" dataset
  (jeanmidev/smart-meters-in-london, ~5561 households total, 7.4GB downloaded).
  Coverage profile computed directly (not assumed): household count ramps from 2 in
  Nov 2011 to a peak of 5531 around Nov 2012, settling ~4987-5528 through Feb 2014.
  Tested several candidate start dates for the m*T trade-off; `2012-06-01` exactly
  reproduces the ORIGINAL dataset's T=30577 (independent confirmation of consistency
  with earlier session work); `2012-08-01` was chosen as the best m*T product
  (m=2953 vs m=1387 at 2012-06-01, at a T cost of ~10%).
- `weather_full_raw.pkl` (in progress, background PID still running as of this
  entry): candidate pool is 856 stations selected via a fixed 5-degree lat/lon
  grid-binning rule over the public `lutangar/cities.json` world-cities list (one
  city per occupied grid cell, order-independent of any outcome) -- a pre-registered
  selection rule, not populated cherry-picking. Open-Meteo rate-limits aggressively
  (429s after ~2 unthrottled chunks); fixed with exponential backoff + checkpointing,
  progressing at ~20 stations/batch as of this entry.
- `wikipedia_full_raw.pkl` (in progress, background PID still running as of this
  entry): candidate pool is 866 articles from 5 categories chosen for having large
  DIRECT membership (not subcategory-only containers, verified by probing counts
  before picking): Chemical_elements (125), Programming_languages (178),
  Mountains_of_the_Alps (250), World_Heritage_Sites_in_Italy (219), Amino_acids (94).
  Wikipedia's API rate-limits hard (retry-after headers observed); fixed with
  backoff + per-10-article checkpointing, matching the original session's own
  established pattern for this exact API.

### Known issues / open items
- Both background fetches (weather, wikipedia) still running as of this entry --
  need to finish, then get aligned into `weather_full.npz`/`wikipedia_full.npz` the
  same way sp500/streamflow/smartmeter were.
- The "fix m=500 for all datasets, big shared T (bigger for smartmeter)" design for
  the FIRST five-way experiment (as opposed to the flexible maximal pool for later
  Sobol-sweep reuse) has not been built yet -- once all `_full.npz` pools exist,
  still need to derive one m=500-capped, common-T nested subsample per dataset
  (matching `prepare_training_data`'s deterministic-prefix-slice convention) and
  re-run the five-way comparison on THAT standardized cut.
- sp500_full's ticker list was scraped from Wikipedia's current "List of S&P 500
  companies" page (502 of ~503 parsed correctly); GICS sector for the 2
  Yahoo-ticker-reformatted entries (BRK.B->BRK-B, BF.B->BF-B) was left as "unknown"
  since the sector lookup wasn't redone for those two specifically -- cosmetic, not
  used by any comparison logic.

### Next exact step (superseded -- see (k) below)

## 2026-09-15 (k) -- User wants >=3 datasets with m>2000; streamflow expanded
nationwide, wikipedia/weather candidate pools widened

User asked why only smartmeter reached m>2000 when others could too. Answered
honestly: it's a mix of REAL hard limits (sp500 ~500 by index definition; ACWI's
own holdings list caps around 2099 mappable candidates) and SELF-IMPOSED candidate
pool sizes for wikipedia (5 categories, 866 candidates) and weather (856-station
5-degree grid) that were never actually maximized, unlike smartmeter where every
one of the 112 real blocks was pulled. User then asked for at least 3 datasets with
m>2000.

**streamflow, expanded nationwide** (not just California): USGS's site-list service
requires a bounding filter (`countryCd` alone is rejected, confirmed via a real
400 error) -- looped over all 50 states + DC instead, yielding **24,525** candidate
sites (vs. 2,419 for CA alone). Batched `dv` fetch (same pattern as before, no
rate-limiting issues from USGS at this scale) got 10,274 sites with data; alignment
(densest-calendar + ffill/bfill(limit=3) + drop-unfillable) kept **6,732** --
**replaces** the old CA-only `streamflow_full.npz` (m=608) entirely. This is now
the biggest real dataset built this session by a wide margin.

**wikipedia, candidate pool widened**: added 6 more categories probed beforehand
for large DIRECT membership (Moths/Beetles/Spiders_of_Europe capped at the 500-per-
page limit -- paginated to 700 each; Rivers_of_Germany similarly; Snakes_of_Africa
233; Birds_of_South_America 267) -- 3,300 new candidates on top of the original 866,
total pool 4,166. Fetch resumes from the existing `wikipedia_full_raw.pkl`
checkpoint (skips the 829 already fetched), still running as of this entry (slow:
~1s/article, one request per article, no batch endpoint).

**weather, expanded to a 2-degree grid** (from 5-degree): 3,462 candidate stations
(vs. 856), regenerated via the same fixed grid-binning rule over the public
world-cities list, just finer. `weather_full.npz` at the 5-degree grid was already
built first (**m=776, T=4017**, zero alignment drops -- Open-Meteo's reanalysis
data has no real gaps, unlike sensor networks) before the user asked to push
further; the 2-degree refetch is running now, resuming from that 776-station
checkpoint (new grid cells mostly select different city coordinates, so most of
the 3,462 candidates are genuinely new fetch targets, not duplicates).

Weather's Open-Meteo rate limit was hit HARD partway through the original 856-
station fetch -- confirmed via a fresh single-request curl test still returning
429 with nothing else running, i.e. a real quota block, not a per-minute throttle
backoff could fix. Killed the fetch, waited, confirmed with periodic curl probes,
and resumed once a probe returned 200 (roughly 35 minutes later). Both weather
fetches (5-degree and the ongoing 2-degree one) hit occasional further 429s that
self-resolved with the existing exponential backoff, not full hard blocks.

### Status as of this entry

| pool | m | T |
|---|---|---|
| sp500_full | 444 | 2768 |
| streamflow_full | **6732** | 4017 |
| smartmeter_full | **2953** | 27649 |
| weather_full | 776 (5-deg; 2-deg refetch in progress) | 4017 |
| wikipedia_full | fetching (4166 candidates, ~640 processed so far) | -- |

Goal (>=3 datasets with m>2000) is ALREADY met by streamflow_full (6732) and
smartmeter_full (2953); wikipedia_full is expected to be a third once its fetch
completes (829/866 already had a 95.7% hit rate and 78.4% alignment survival on
the original batch -- similar yield on the 4166-candidate pool would land well
over 2000).

### Next exact step (superseded -- see (l) below)

## 2026-09-15 (l) -- wikipedia's new categories have much worse alignment
survival than expected; estimation error caught and corrected

Stopped the wikipedia fetch early (at 3142 raw articles) believing it had passed
a safe margin, based on the ORIGINAL 5 categories' 78.4% raw-to-aligned survival
rate. Built `wikipedia_full.npz` to check: **m=1673, T=3106** -- well short of
2000, and short of the estimate. Investigated: the 6 NEW categories added this
entry (Moths/Beetles/Spiders_of_Europe, Rivers_of_Germany, Snakes_of_Africa,
Birds_of_South_America) have only ~44% alignment survival, not 78% -- these are
far more obscure articles than the original picks (chemical elements, programming
languages, well-known mountains/UNESCO sites), so their daily pageview series has
far more missing/unfillable gaps (very low-traffic pages more often have
literal API-side data holes, not just occasional single-day gaps the
ffill/bfill(limit=3) tolerance can absorb).

**Also caught and fixed a real math error of my own**: initially estimated
~1700 candidates remained and ~4 more hours of fetching, panicking about the
pace -- actually mis-read the `fetch_wikipedia_pageviews_full.py` log's
"processed" counter, which only increments for genuinely NEW fetch attempts
(titles already in the checkpoint are `continue`d before the counter increments,
per the script's own loop structure) -- so the true remaining candidate count was
only ~830 (4166 total - ~3336 already attempted), not ~1700. Restarted the fetch
to finish these ~830 remaining candidates (should take ~15-30 min, not hours).

Weather's 2-degree refetch remains heavily throttled by Open-Meteo (many
consecutive "permanently failed after 8 tries" chunks, occasional slow trickle of
new stations) -- left running passively since it's a bonus dataset, not required
for the >=3-over-2000 goal (already met by streamflow_full and smartmeter_full).

### Status as of this entry

| pool | m | T |
|---|---|---|
| sp500_full | 444 | 2768 |
| streamflow_full | **6732** | 4017 |
| smartmeter_full | **2953** | 27649 |
| weather_full | 776 (5-deg; 2-deg refetch still throttled, ~1419/3462 raw so far) | 4017 |
| wikipedia_full | 1673 (INTERIM, fetch resumed for the remaining ~830 candidates) | 3106 |

Goal (>=3 datasets with m>2000) already satisfied by streamflow_full (6732) and
smartmeter_full (2953) alone; wikipedia is expected to cross 2000 once the
remaining ~830 candidates are fetched, given the math above, but this is no
longer strictly necessary for the stated goal -- continuing mainly for
completeness/safety margin.

### Next exact step (superseded -- see (m) below)

## 2026-09-15 (m) -- m=500/shared-big-T standardized cut designed, built, and
launched for 4 of 6 datasets; two more pending their fetches

User confirmed the design directly: deterministic prefix slice (matching this
project's own `prepare_training_data` nested-subsample convention -- smaller cuts
are exact prefixes of larger ones, not independent draws), accept sp500's
under-target m=444 (its real ceiling), run the comparison once the outstanding
fetches finish, and fold the results into the SAME big all-thresholds table
already built (not a separate new table).

**Design** (`build_m500_cuts.py`, scratchpad): m=500 for every dataset except
sp500 (m=444, real ceiling). Shared "big T" for every DAILY dataset = **2768** --
bounded by `sp500_full` specifically, because it samples TRADING days only (fewer
rows per calendar year than the other three's calendar-day sampling), not by
wikipedia's shorter 2015-2023 span as originally assumed -- worth flagging since
it means the shared T ended up smaller than hoped, for a reason unrelated to the
wikipedia API floor. sp500_full's own shape (444x2768) already equals the target
exactly, so it's used unchanged, no slicing needed. `smartmeter` keeps its full
native T=27649 untouched (already satisfies "bigger T for smartmeter", no
artificial scaling formula needed). `sp500_sub263` (the same-data negative
control) is RE-DERIVED from the new sp500 cut: first 263 of its 444 series, same
T=2768 (was T=1255 before) -- keeps its role consistent with the new baseline.
`acwi_real`/`acwi_capweighted` are explicitly OUT of scope for this cut (no
`_full` pool was built for them this round) -- they stay at their original native
size/window, excluded from the new comparison script rather than silently mixed
in at a different scale.

**Built so far**: `sp500_m500` (m=444,T=2768, = sp500_full unchanged),
`sp500_sub263_m500` (m=263,T=2768), `streamflow_m500` (m=500,T=2768, sliced from
the 6732x4017 nationwide pool), `smartmeter_m500` (m=500,T=27649, sliced from the
2953x27649 pool). New script `hamming_exact_compare_m500.py` (copied from the
original, only the CFG paths/output filename changed -- same methodology, same
calibration discipline). Synced to Abaca; **16 OAR jobs submitted** (IDs
3108633-3108648) covering these 4 datasets x {0.70, 0.80, 0.90, 0.95}.

**Still pending**: `wikipedia_m500` and `weather_m500` (`global_weather` in the
CFG) -- their `_full` pools are still fetching (see entry (l)). Once both finish,
slice them the same way (first 500 series, first 2768 time-steps) via
`build_m500_cuts.py`'s tail section, sync, and submit the remaining 8 jobs
(2 datasets x 4 thresholds) to complete the full 6-dataset standardized battery.

### Status

| pool | m in m500 cut | T in m500 cut |
|---|---|---|
| sp500 | 444 (unchanged, real ceiling) | 2768 |
| sp500_sub263 | 263 (re-derived) | 2768 |
| streamflow | 500 | 2768 |
| smartmeter | 500 | 27649 (native, "bigger") |
| wikipedia | pending | 2768 (planned) |
| global_weather | pending | 2768 (planned) |
| acwi_real / acwi_capweighted | not included (native size/window, excluded from this cut) | -- |

### Next exact step (superseded -- see (n) below)

## 2026-09-15 (n) -- acwi_capweighted expanded per user request; wikipedia and
weather fetches finished (both cross 2000 too); full 7-dataset x 4-threshold
m500 battery (28 jobs) launched

User asked to clarify why `acwi_real`/`acwi_capweighted` were excluded from the
m500 cut, then explicitly chose to expand `acwi_capweighted` toward its real
ceiling rather than exclude it or include it at native scale. Built
`acwi_capweighted_full` (`fetch_acwi_capweighted_full.py` + 
`build_acwi_capweighted_full.py`, same weight-sorted/suffix-mapped selection
rule as the original m=263 version, just TOP_N raised to all 2099 mappable
tickers, same modern densest-calendar+ffill/bfill(limit=3)+drop-unfillable
alignment as this session's other `_full` pools): **1901/2099 fetched, m=1482,
T=2822** after alignment -- comfortably clears 500. `acwi_real` stays excluded
(its own 4-per-country selection rule structurally caps its pool near 160,
unfixable without changing that rule).

**wikipedia's fetch finished**: 3982 raw articles (up from the 3142 that only
yielded m=1673) -> **wikipedia_full: m=2172, T=3106** -- crosses 2000, closing
out the originally-requested third dataset properly (not just via the earlier,
premature stop).

**weather's fetch reached m=2000 too** (checkpoint advanced to 2000 stations
with data since the last check, zero alignment drops as before -- Open-Meteo
reanalysis has no real gaps): **weather_full: m=2000, T=4017**. Per user
instruction, built `weather_m500` from this checkpoint NOW rather than waiting
for the full 3462-candidate fetch to finish, while leaving the fetch process
running in the background to keep collecting the remaining candidates for the
eventual maximal pool (not blocking the m500 comparison, which is already
locked to this snapshot).

**Every _full pool now stands at**:

| pool | m | T |
|---|---|---|
| sp500_full | 444 | 2768 |
| streamflow_full | 6732 | 4017 |
| smartmeter_full | 2953 | 27649 |
| wikipedia_full | **2172** | 3106 |
| weather_full | **2000** (still growing in the background) | 4017 |
| acwi_capweighted_full | 1482 | 2822 |
| acwi_real | 125 (unchanged, structural ceiling) | 2425 |

**Four datasets now genuinely exceed m=2000** (streamflow, smartmeter, wikipedia,
weather) -- well past the original ">=3" ask.

**m500 cuts built and synced for all 7 comparable datasets** (sp500, sp500_sub263,
streamflow, smartmeter, acwi_capweighted, wikipedia, weather/global_weather) --
`acwi_real` remains excluded (no path to m=500). **Full battery of 28 OAR jobs
submitted** (7 datasets x 4 thresholds): sp500/sp500_sub263/streamflow/smartmeter
(3108633-3108648), acwi_capweighted (3108716-3108719), wikipedia
(3108752-3108755), global_weather (3108765-3108768). Multiple already completed
as of this entry (sp500 thr=0.70, sp500_sub263 all 4, streamflow thr=0.90/0.95,
smartmeter thr=0.90, and more finishing progressively).

### Next exact step (superseded -- see (o)/(p) below)

## 2026-09-16 (o) -- m=500 battery complete (28/28); comparative dataset table
delivered; weather's fetch actually finished (m=2295, not still-growing)

All 28 m500 jobs completed. Pulled every result JSON, compiled the full table
(dataset x threshold, density/avg_degree/BF-runtime/all 4 methods with
recall+gamma+occupancy) and delivered it to the user across several messages
(full 28-row table, then narrower re-cuts on request). Headline findings:
standardizing m=500 flips `wikipedia`/`global_weather` from CorrTrack's
weakest cases (at their old small native sizes) to among its strongest
(13-14x at thr=0.95); `sp500` vs `sp500_sub263` (same data, m=444 vs 263)
reproduces the m-scaling advantage cleanly (12.56x vs 6.37x at thr=0.95);
`smartmeter`'s lsh_sign_dot recall shortfall got WORSE at m=500 (0.891 at
thr=0.95, vs 0.941 at the old m=510), not better -- still open, disclosed.

Built a comparative dataset table (domain/application/source/resolution/native
span/m-ceiling/T-ceiling/m-used/T-used) plus density+avg_degree at all 4
thresholds for every dataset including `acwi_real` (native scale). While
building it, discovered `weather_full`'s background fetch (still running from
entry (n)) had actually FINISHED completely on its own (all 3462 candidates
processed, ended at 2295 stations with data, zero alignment drops) --
corrected the reported ceiling from a stale "m=2000, still growing" to the
final **m=2295, T=4017**.

## 2026-09-16 (p) -- Real-data Sobol-sweep reproduction: m-sweep launched
(27 cells); CorrJoin dataset investigation corrected via a DIFFERENT prior
thread's already-authoritative research

User asked to (1) plan reproducing the Sobol sweep over these real datasets
running BF/STOMP/FilCorr/CorrTrack (not just BF/CorrTrack as the original
synthetic sweep did), and (2) assess whether to include the CorrJoin
competitor paper's own datasets (Drive link given).

**Sweep design** (confirmed with user: skip CorrJoin data for now, keep BOTH
CorrTrack backends per cell): per the user's standing "don't sweep W/L"
instruction, the sweep's only new free axis is **m** (nested prefix subsample,
matching `prepare_training_data`'s convention), at fixed corr_threshold=0.70
(the threshold axis is already fully characterized at m=500 by entry (o)'s
battery). Design follows the original plan's "core factorial + OFAT, not
blind full-cross" discipline: per-dataset log-spaced m grids from ~100 up to
each dataset's own real ceiling (reusing each dataset's existing m=500 cell,
not rerunning it), plus 2 spot-check cells (streamflow and wikipedia at their
ceiling m, thr=0.90) to catch any m x threshold interaction the design might
otherwise miss.

**New instrumentation**: `hamming_exact_compare_msweep.py` (new script, forked
from the m500 version) adds `ct.sketch_time`/`candidate_time`/
`validation_time`/`monitor_time`/`tested_candidates`/`validated_candidates` to
every calibration trial's recorded output (verified these attributes exist on
`CorrTrack`, confirmed via grep) -- absent from the m500 script, needed for
the sweep's actual payoff: fitting `sketch_time ~ f(m)`,
`candidate_time ~ f(m,L,occupancy)`, `validation_time ~ f(tested_candidates)`
cost models the way the original synthetic Sobol sweep did.

Built 24 new m-cuts via `build_msweep_cuts.py` (deterministic prefix slice,
same T=2768 shared daily / native T=27649 for smartmeter):

| dataset | new m points built (ceiling in bold) |
|---|---|
| sp500 | 100, 200, 300 (444 = ceiling, already have from m500) |
| sp500_sub263 | 60, 130, 200 (263 = ceiling by definition) |
| acwi_capweighted | 100, 900, **1482** |
| wikipedia | 100, 1000, 1600, **2172** |
| global_weather | 100, 1000, 1600, **2295** |
| smartmeter | 100, 1000, 2000, **2953** |
| streamflow | 100, 1500, 3500, **6732** |

(`acwi_real` excluded -- no `_full` pool, structural ~125 ceiling, not worth
sweeping.) All synced to Abaca; **27 OAR jobs submitted** (IDs
3110102-3110128): 25 at thr=0.70 (one per new m-cut) + 2 spot-checks
(streamflow_m6732 and wikipedia_m2172 at thr=0.90). Several already completed
as of this entry.

**CorrJoin datasets, corrected via `docs/competitor_comparison_plan.md`**
(a separate, much larger, ALREADY-EXISTING plan document in this repo covering
5 competitor reimplementations -- CorrJoin/StatStream/ParCorr/TSUBASA/BRAID --
that this session had not previously read): that document's §3 already has
the CorrJoin paper (Alizade Nikoo/Böhlen/Helmer, PACMMOD 2023) obtained and
extracted as of 2026-09-15, with the exact dataset sizes straight from the
paper: **chlorine 4830x2040, gas 5120x3600, stock 3878x1259, synthetic (bonus)
5000x4080** -- all DIFFERENT from (larger than) the classic 166x4310 SPIRIT/
EPANET chlorine dataset this session had found independently via web search
before discovering the existing doc. Corrected the framing given to the user:
these are fixed, published reproduction assets (the doc calls them "the
strongest faithfulness check available" against the paper's own published
claims) -- there is no "more data" to source, and trying to expand them would
defeat their purpose. The existing doc's own §10.2 item 2 ("whether to adopt
the original CorrJoin datasets as an additional evaluation axis") is exactly
the user's question, still marked open/undecided there too. The actual data
files (chlorine.txt/gas.txt/stock.txt, confirmed exact names from that doc's
own dataset row) were never downloaded by either thread -- still needed from
the user directly, since Drive's folder listing needs JS/API access this
session doesn't have (confirmed via repeated WebFetch attempts, all blocked).

### Known issues / still open

- 27 m-sweep jobs running/queued on Abaca as of this entry.
- CorrJoin's chlorine.txt/gas.txt/stock.txt still not obtained -- waiting on
  the user to share them directly (or a working download link).
- This session's real-data sweep work (BF/STOMP/FilCorr/CorrTrack scaling) and
  `docs/competitor_comparison_plan.md`'s competitor-reimplementation effort
  (CorrJoin/StatStream/ParCorr/TSUBASA/BRAID, "plan only, nothing implemented")
  are related but distinct threads -- worth reconciling explicitly next time
  work touches competitor comparisons, so effort isn't duplicated across them.
- Cost-model fitting (the sweep's actual analytical payoff) not yet done --
  needs all 27 cells' phase-timing data once complete.

### Next exact step (superseded -- see (q) below)

## 2026-09-16 (q) -- User corrected the sweep design (genuine Sobol sequence,
not hand-picked grid; m x L x threshold, density dropped); CorrJoin's real
chlorine/gas/stock/synthetic/random datasets obtained directly from the user;
72-cell real Sobol sweep launched, replacing the 27-cell m-only sweep

User flagged directly: entry (p)'s "m-sweep" was NOT actually a Sobol sequence
-- it was a hand-picked log-spaced grid per dataset. Also: "last time" (the
original synthetic sweep) swept m, L, density AND corr_threshold; this
reproduction should sweep **m, L, corr_threshold** but **drop density**
(it's an emergent property of real data, not a dial -- correctly excluded,
unlike the synthetic sweep which could set it directly).

**CorrJoin's real datasets obtained.** The user found that individual Google
Drive file-share links (`.../file/d/<id>/view`) ARE fetchable, unlike the
folder listing (blocked all session). Downloaded all 5 files via
`curl -L "https://drive.google.com/uc?export=download&id=<id>"`, handling
Google's "virus scan warning" interstitial page programmatically (parse the
HTML form's hidden `uuid` field, re-request
`drive.usercontent.google.com/download?...&confirm=t&uuid=<uuid>`) -- each
came back as a zip containing one `.txt` file. **All 4 real/bonus files
matched `docs/competitor_comparison_plan.md`'s documented sizes exactly**:
chlorine 4830x2040, gas 5120x3600, stock 3878x1259, synthetic (random-walk)
5000x4080, plus a 5th bonus `random.txt` (5000x4080, i.i.d. uniform noise --
verified genuinely different content from `synthetic.txt` via a distinct
md5sum and value range, not a duplicate). Converted all 5 to `.npz`
(`build_corrjoin_npz.py`, `tmp_artifacts/corrjoin_{chlorine,gas,stock,
synthetic,random}/`).

**Killed the entire 27-job m-only sweep** (entry (p)) before it produced
usable results -- superseded by the corrected design below, no partial
results salvaged (none had finished yet).

**New design** (`generate_sobol_design.py`): a genuine `scipy.stats.qmc.Sobol`
low-discrepancy sequence, `d=3` dims (m, L, corr_threshold), scrambled, seed
fixed for reproducibility. 8 points per dataset, same unit-cube sequence
applied to every dataset (comparable relative design across datasets despite
different absolute ceilings). Mappings: `m` log-uniform in
`[m_min, dataset_ceiling]`; `L` integer in `[2,10]` via `n_lags=(L-1)*STEP` at
each dataset's already-fixed `STEP` (W/STEP themselves stay fixed, not
swept -- only `n_lags` moves); `corr_threshold` uniform in `[0.70, 0.95]`.
**9 datasets x 8 points = 72 cells**: the 6 core real datasets (sp500,
streamflow, smartmeter, acwi_capweighted, wikipedia, global_weather) plus
CorrJoin's 3 real-world benchmarks (chlorine, gas, stock) -- `synthetic`/
`random` built as bonus assets but excluded from this sweep to control scope
(disclosed, not silently dropped). W/STEP for chlorine/gas/stock are NEW,
independent choices (60/6, 100/10, 40/4 respectively, roughly matching this
session's own W/STEP~10:1 convention) since the paper's own PAA/SVD parameters
govern a different algorithm's windowing, not ours -- explicitly disclosed as
not derived from the paper.

**New run script** `hamming_exact_compare_sobol.py` (forked from the msweep
script): takes W/STEP/N_LAGS/THR directly as CLI args (no more per-family
lookup) since every axis now varies per-cell independently. Calibration grids
DELIBERATELY narrowed for this sweep only (hamming: 4->2 offset trials;
lsh_sign_dot: 20->4 occupancy x offset trials) to control total compute across
72 cells -- a disclosed tradeoff, not applied to any other battery this
session.

**Walltime estimation, calibrated against real data**: cost proxy
`m^2 * L * N_STEPS`; `k_bf=1.058e-7` s/unit derived from `streamflow_m500`'s
actually-measured BF wall time (144.9s at cost=1.3695e9); x7 multiplier for
the full battery (9 runs: BF+stomp+filcorr+2 hamming+4 lsh, most cheaper than
BF due to CorrTrack's own pruning) x2 safety margin; clamped to [1h, 96h] (96h
confirmed accepted by the `abaca` queue via a real throwaway test submission,
job 3110188, immediately deleted). Largest cells: `streamflow` pt6
(m=4364,L=9) at 64h23m, `smartmeter` pt6 (m=2084,L=9) at 55h30m -- both
disclosed as multi-day, unattended background runs.

Built 72 unique (dataset,m) slices (`build_sobol_cuts.py`), synced all to
Abaca (verified byte-exact via size comparison, not just scp exit code), and
**submitted all 72 jobs** (IDs 3110190-3110262).

### Known issues / still open
- 72 real Sobol-sweep jobs running/queued on Abaca, several multi-day --
  expect this sweep to take days to fully complete, not hours.
- `synthetic`/`random` CorrJoin datasets built but not included in this
  sweep -- available for a future addition if wanted.
- Chlorine/gas/stock's W/STEP are disclosed independent choices, not derived
  from the CorrJoin paper's own windowing (which uses different parameters for
  a different algorithm) -- worth flagging in any write-up that cites these
  results against CorrJoin's own reported numbers.
- Cost-model fitting (the sweep's actual analytical payoff) still pending --
  needs all 72 cells' phase-timing data once complete, likely in batches as
  they finish rather than waiting for the very last (multi-day) cell.

### Next exact step (superseded -- see (r) below)

## 2026-09-16 (r) -- User provided the actual CorrJoin paper (PDF) and its
authors' R implementation; W/STEP for chlorine/gas/stock were wrong, corrected
using the real parameters found in the code

User directly supplied the CorrJoin paper (Alizade Nikoo/Böhlen/Helmer,
PACMMOD 2023, CC-BY) and the 8 R scripts from the paper's own Drive `Code/`
folder (`1-IncPPAA.R`, `2-CorrJoin.R`, `3-BucketingFilter.R`, `4-Quickjoin.R`,
`5-Ekdb.R`, `6-TSUBASA.R`), after I disclosed that entry (q)'s W/STEP=60/6,
100/10, 40/4 for chlorine/gas/stock were my own invented values, not the
paper's.

**Corrected from the paper text directly** (not a secondary extraction this
time): chlorine is 161 real pipe-junction series (not 166 as the classic
SPIRIT/EPANET benchmark this session found earlier via web search -- a
DIFFERENT network), replicated 30x with small additive noise -> m=4830,
matching our downloaded file exactly. Gas is 128 real chemical-sensor series
(UCSD ChemoSignals Lab, 2007-2011, 6h sampling), replicated 40x -> m=5120.
Stock is 3878 real NASDAQ daily closing prices, 2016-2020, Yahoo Finance.

**Corrected from the R code directly** (the real payoff of getting the code):
every demo script hardcodes `windowSize <- 1020`, `stride <- 100`,
`frameSize <- 68` -- and `1020/68 = 15` and a separate reduction pass uses
`frameSize <- windowSize/30` (=34), matching the paper's own reported optimal
`ks=15, ke=30` *exactly* -- not a coincidental demo default. Chlorine's own
T=2040 = 1020 x 2 exactly, matching the demo's hardcoded `numOfSW <- 2`. This
is strong, direct evidence that **n=1020 (window size), h=100 (stride)** are
the real parameters behind the paper's main reported comparison figures (Figs.
12/14/19), reused identically across chlorine/gas/stock (not scaled
per-dataset as I had invented).

**Fix applied**: killed the 7 corrjoin Sobol jobs still running with the wrong
W/STEP (several had already finished and produced now-stale JSON output,
harmless since the corrected reruns overwrite the same filenames). Updated
`sobol_design.json`'s 24 corrjoin entries to `W=1020, STEP=100` (m and L
values unchanged -- only `n_lags=(L-1)*STEP` needed recomputing since it's the
only quantity depending on STEP). Regenerated and resubmitted all 24 corrjoin
job scripts (IDs 3110449-3110472) -- lower cost proxies than before (fewer,
bigger windows -> fewer N_STEPS), so all show ~1h walltime except gas's
largest-m point.

Note: stock's own T=1259 barely fits `n=1020` (only
`(1259-1020)//100+1 = 3` window-steps) -- workable but sparse, consistent with
the paper's own methodology which appears to prioritize a shared, ks/ke-tied
window size over per-dataset step-count richness.

### Known issues / still open
- The 6 non-corrjoin dataset families' Sobol jobs (54 cells) were unaffected
  by this fix and continue running independently.
- `n=1020/h=100` is inferred with strong but not 100% certain evidence (the
  demo scripts' `datasetTXT <- read.table("../Datasets/synthetic.txt", ...)`
  line is hardcoded to synthetic.txt in every script -- there's no direct
  textual proof the SAME script was run unmodified against chlorine/gas/stock
  files with identical parameters, though the ks=15/ke=30 tie-back and
  chlorine's exact T=1020x2 divisibility make this very likely).

### Next exact step
Poll `ssh sophia.g5k "oarstat -u rpontess"` and
`grep -l 'DONE, saved' sob_*.out | wc -l` (target 72) periodically. Pull
`hamming_exact_compare_sobol_*_pt*.json` files as they become available,
build the m/L/threshold -> speedup surface per dataset, and fit cost models
(`sketch_time ~ f(m)`, `candidate_time ~ f(m,L,occupancy)`,
`validation_time ~ f(tested_candidates)`) with held-out-scale
cross-validation once enough of the design is in.

---

## 2026-09-15 (n) -- CorrJoin paper obtained; competitor-comparison plan's §3 replaced
with the paper-authoritative specification. Two previously-written claims were wrong.

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md` only.
**Nothing implemented** -- the user's "plan it but do not do it yet" still stands.

### What happened

The user supplied the actual CorrJoin paper (Alizade Nikoo, Bohlen & Helmer,
*Correlation Joins over Time Series Data Streams...*, PACMMOD 1(4):235, 2023,
CC-BY), after instructing that the paper, not the derivative UZH bachelor thesis,
is the specification. The §10.1 extraction checklist was worked against it and is
now closed; all six items resolved.

### The two corrections (both were in the plan as written, both now fixed)

1. **CorrJoin has NO lag support.** The paper is entirely synchronous -- Alg. 1
   compares all series at the same window index. The earlier "time-delayed
   correlation over any time delay" claim came from a web-search summary of the
   abstract, not from the paper, and I had carried it into the plan with only an
   "unconfirmed" hedge. Consequence: **ParCorr, StatStream and CorrJoin are all
   synchronous; only CorrTrack and FilCorr do lags.** Lag search stays a genuine
   CorrTrack capability differentiator. The possibility I had flagged -- that
   CorrJoin might displace CorrTrack as the closest peer on the lag axis -- does
   not apply.
2. **The epsilon thresholds** were wrong in both derivative variants.
   Authoritative (Alg. 1 line 1): `e1 = sqrt(2*ks*(1-T)/n)`,
   `e2 = sqrt(2*ke*(1-T)/n)`. **Both divide by n, and e1 uses ks, not kb.** The
   thesis gave `kb` in e1 (§3.2.1) and dropped `/n` from e2 (§3.2.2). Either
   error changes the pruning radius, so implementing against the thesis would have
   produced CorrJoin numbers quietly wrong in both recall and speed.

### Confirmed as already written

Normalization (centred unit-norm, before reduction -- shared code path with
ParCorr); negative correlation via `|corr| >= T` (Alg. 1 line 14); `ks=15, ke=30,
kb=3`; gain requires roughly `m > 100`.

### New facts extracted

- **Incremental update**: five running sums `s1..s5` (Eq. 2) -- the same
  sufficient-statistics form `exact_stomp` already implements, so the port reuses
  existing machinery. PAA means update in `O(h+k)`. Stride has only a slight
  runtime effect (Fig. 16a), the opposite of the thesis's finding, which was an
  artefact of its from-scratch recomputation.
- **Speedup is upper-bounded by `1/r1`** (r1 = fraction surviving filter 1) --
  a sanity ceiling for validating our port's measured speedup.
- **The paper's own baselines** (eps-kdB tree, TSUBASA, Quickjoin, IncP^PAA) were
  all **reimplemented from scratch in R by the authors**. That is a citable
  precedent for this plan's §0 uniform-reimplementation policy.
- The paper's §5 cost model (cnorm/cPAA/cSVD/cbkt/cfilter/ctrue vs
  cbase=O(m^2 n)) is structurally the three-phase decomposition proposed in §5b.1.

### Two findings of direct value to the CorrTrack paper

- **Fig. 15 / §6.3.3**: "once the rate reaches around 20%, the difference in
  performance between CorrJoin and IncP^PAA is not discernible anymore."
  Independent SIGMOD corroboration of the 2026-09-12 finding that CorrTrack ties
  brute force at ~17.6% density and wins 3.66x at ~2%. Pruning stops paying off
  at high correlation density is a **property of the method class**, not a
  CorrTrack weakness, and we can now cite a competitor's paper saying so.
- **Fig. 10**: TSUBASA "has a similar time complexity as the naive IncP since it
  does not reduce the number of pairwise comparisons" -- an order of magnitude
  worse than their baseline. Corroborates the plan's §5 assessment that TSUBASA
  occupies `exact_stomp`'s slot, and justifies keeping it ranked secondary.

### Plan sections updated

§1 table row, §3.0 source hierarchy, §3.1 (rewritten as specification), §3.2
(rewritten as corrections + findings), §5b.5 capability matrix, §6.1 lag protocol,
§7 sequencing (phases 0 and 5 no longer blocked), §9 scope estimate (CorrJoin now
carries engineering risk only, not specification risk), §10.1 (closed).

### Known issues / still open

- §10.2 remains open: ParCorr's grid/subspace parameters (DMKD paper),
  StatStream's DFT coefficient count and grid parameters (VLDB 2002), locating the
  authors' R implementation, and whether to adopt CorrJoin's own four datasets
  (chlorine/gas/stock/synthetic -- Google Drive link in §3.0) as an evaluation axis
  rather than only for faithfulness validation.
- `abaca/` comparison scripts (7 files: `fourway_compare.py`,
  `sparse_fourway_compare.py`, 5 `.oar` wrappers) are **still uncommitted**; a
  commit command was handed over earlier but no confirmation it was run.

### Next exact step

Either (a) extract ParCorr's grid/subspace parameters from the DMKD 2018 paper and
cross-check against the `corrtrack_release_v1.0` / `corrtrack_release_backup2`
snapshot implementations (the user wrote that code, so it is partly a
direct-knowledge question), or (b) begin phase 1 -- the shared
normalize-window-before-reducing path needed by both ParCorr and CorrJoin -- if
the user lifts the "do not implement yet" hold.

---

## 2026-09-15 (o) -- ParCorr paper extracted + v1 snapshot audited; BRAID added to the
competitor set as the lag-axis peer. Planning only, still no implementation.

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

### (1) ParCorr, DMKD 2018 -- obtained and extracted

Source: HAL-LIRMM open mirror
`https://hal-lirmm.ccsd.cnrs.fr/lirmm-01886794/file/ParCorr__DMKD2018.pdf`.
Every parameter question in the old §10.2 item 1 is now closed.

Parameters (plan §2.1): `r = 60` random +1/-1 vectors (worked example; JL guidance
`r = 8 log n / eps^2`), `k = 2` dimensions per subvector so `n_grids = r/k = 30`,
`f = 0.7` fraction-of-grids vote, sliding window 500, basic window 20, T = 0.7
(also 0.8, 0.9). Reported: 100% precision by construction, recall >90% at T=0.7,
>96% at T=0.8, >95.7% at T=0.9.

Three findings beyond the parameter list:

- **`f` is calibrated by raising it until a target recall (0.95) is met on a small
  sample.** That is the same protocol CorrTrack uses (`target_recall=0.95`), so the
  plan's §6.2 equal-recall comparison is ParCorr's own published tuning procedure,
  not something we impose. Removes the main fairness objection to it.
- **ParCorr does NOT search neighbouring grid cells** -- listed as future work
  item 1. CorrJoin does. A real algorithmic difference between two methods that
  otherwise look alike, and a port that adds neighbours is no longer ParCorr.
- **The paper lists lag support as future work item 2**, noting it "does require
  adjustments because the normalization transformation may change." That confirms
  the user's synchronous-only constraint from the authors' own mouth, and describes
  exactly the difficulty CorrTrack solves. Citable for the novelty claim.

**Gap**: the paper never specifies a grid cell size; it is absorbed into the
calibration of `f`.

### (2) Snapshot audit -- v1.0 vs backup2 (checked, not assumed)

The plan had claimed the v1 snapshots are "a starting point, not a blank page."
Verified, and the claim needed qualifying: the ParCorr-defining machinery is
present but **disabled**.

- `grid_dimension` is ParCorr's `k` and `n_grids = n_vectors // grid_dimension` is
  `r/k` (`v1.0:2188`, comment at `:5368`). Reusable as-is.
- `self.freq_threshold = 0` is hardcoded with the comment "disable frequency
  gating" (`v1.0:2233`). **The co-occurrence vote, ParCorr's entire contribution,
  is switched off.**
- `full_vector_candidates` forces `grid_dimension = n_vectors`, `n_grids = 1` on
  the default path (`v1.0:2182-2185`), collapsing the grids to one full-dimensional
  space. Not ParCorr.
- The surviving vote code uses `required_hits = max(1, int(self.n_grids))`
  (`v1.0:4647`), i.e. **effective `f = 1.0`**, not the paper's 0.7. Reusing it
  as-is would systematically under-recall.
- `backup2` hardcodes `full_vector_candidates=True`, `n_grids=1` (`:3355-3357`)
  and `freq_threshold=0` (`:3402`). **Further from ParCorr than v1.0 -- mine v1.0,
  not backup2.**
- `_compute_base_cell_size = sqrt(2(1-T))/sqrt(n_vectors)` (`v1.0:1315`) and
  `grid_max = min(1, 3/sqrt(n_vectors))` (`:2230`) are the user's own derivations.
  The paper specifies no cell size, so these cannot be presented as ParCorr.

### (3) BRAID added to the competitor set (user request)

Sakurai, Papadimitriou & Faloutsos, SIGMOD 2005, 599-610. Obtained from the Osaka
mirror `https://www.dm.sanken.osaka-u.ac.jp/~yasushi/publications/braid.pdf`.
(The CMU mirror of the same paper is a dvips Type-3 build with no extractable text
layer -- noted in the plan so the next person does not repeat the detour.)

**Why it matters here**: BRAID is a lag-correlation method, and after the CorrJoin
paper turned out to be synchronous (entry (n)), CorrTrack's lag capability had
*no peer at all* in the planned comparison -- an unchecked claim rather than a
measured one. BRAID makes it measurable. That is the reason to include it, more
than speed rivalry.

**It is structurally unlike the other four**: BRAID does **no pair-space pruning**.
Its top-level loop is `for each pair of sequence X and Y do ProductKeeping(X,Y)`,
maintaining state for all O(k^2) pairs. "Group lag correlation" means running it
over all pairs (55 temperature sensors), not a pruning mechanism.

Specification (plan §5a.2): five sufficient-statistic sums (the **same** form
`exact_stomp` and CorrJoin use -- third consumer, so factor it out once);
geometric lag probing at `l = 0,1,2,4,...`; hierarchical non-overlapping window
averages `Ax_h(t) = (Ax_{h-1}(2t-1)+Ax_{h-1}(2t))/2` giving
`R(l) ~= R_h(l/2^h)`; cubic-spline interpolation between probed lags; Brent's
method for the maximum. Enhanced BRAID keeps `b` coefficients per level
(**experiments use b=16**). Target is the **earliest local maximum of |R(l)|
above gamma=0.4**, so negative correlation is handled. Max lag m = n/2.
Complexity O(log n) space per pair, O(1) amortized update, O(log n) interpolation.
Accuracy ~1% relative lag error, up to 40,000x faster than naive; zero error when
sampled at or above Nyquist (Lemma 3: resolves lags `0 <= l < 2b/f_R`).

**Four comparability mismatches recorded in §5a.3** rather than discovered later.
The important one: BRAID's published accuracy metric is **relative error in the
lag value**, not recall/precision over a pair set, so §6.2's equal-recall protocol
does not apply to it unmodified. This is the one place the uniform metric scheme
genuinely breaks, and the plan says so instead of fudging it.

**Prediction recorded in advance (§5a.4)**: with no pruning and O(k^2) pair states,
BRAID should lose badly to CorrTrack at m=500+ while being competitive at small k.
If it wins at large k, our port or our understanding is wrong.

**Verification anchor**: with `2b > n_lags`, level 0 covers every lag at raw
resolution with no smoothing and no interpolation, so BRAID must reproduce the
bruteforce lagged pair set exactly.

### Plan sections updated

§1 table (BRAID row), scope line, §2 (constraints confirmed + new §2.1 parameters
+ new §2.2 snapshot audit), new §5a (BRAID, five subsections), §5b.1 phase table
(BRAID column), §5b.5 capability matrix, §6.1 lag protocol, §6.5 anchors, §7
sequencing (now 6 phases, BRAID at 4), §9 scope/fallback, §10.2 (item 1 closed,
new item 5).

### Known issues / still open

- "Plan it but do not do it yet" still stands. Nothing implemented.
- §10.2 open: StatStream's DFT/grid parameters (VLDB 2002) is now the only unread
  primary source among the pruning competitors; **BRAID's TKDD 2010 extension**
  (*Fast Discovery of Group Lag Correlations in Streams*, 4(1):5) is unread and
  **may add pair pruning** -- if it does, BRAID's role changes from pure lag peer
  to lag+pruning competitor; locating the CorrJoin authors' R code; and whether to
  adopt any competitor's original datasets as an evaluation axis (BRAID's 55-sensor
  Humidity/Light/Temperature sets are the most interesting for us, being
  multi-series and lag-correlated by construction).
- The ParCorr cell-size question (§2.2 last row) must be decided before phase 2:
  adopt the paper's tune-`f` procedure, or keep the v1 formula and label it ours.
- The 7 `abaca/` comparison scripts remain **uncommitted** (unchanged from entry
  (n)).

### Next exact step

Read the StatStream VLDB 2002 paper for its DFT coefficient count and grid
parameters, closing the last §10.2 literature item for the pruning family; and
check whether BRAID's TKDD 2010 extension adds pair pruning, since that determines
whether §5a's framing of BRAID as a pure lag peer survives. Neither is blocking.
Implementation still gated on the user lifting the hold.

---

## 2026-09-15 (p) -- StatStream paper read: it does lags, persistence and negative
correlation. Two of my capability claims were wrong. ParCorr cell size decided.
Planning only, still no implementation.

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

### The correction, stated plainly

The user supplied the StatStream paper (Zhu & Shasha, VLDB 2002, 358-369) with the
note that BRAID is not the only lag-capable competitor. Correct on both counts.
Entry (o) had described BRAID as "the only peer on the lag axis" and the plan had
described StatStream as "primarily synchronous" and a mostly-historical ancestor.
**Both wrong.** The paper's abstract claims and its sections 3.5/3.6 deliver
"time-delayed correlation over any size sliding window and any time delay."

StatStream actually has: DFT sketch, grid pruning with **no false negatives**
(their Theorem 2), **lag search** via a timestamped never-cleared grid,
**negative correlation** (Lemma 3), incremental DFT with no revisiting of expired
data (Lemmas 4-6), a **"Duration over Threshold"** minimum-persistence parameter,
and an embarrassingly parallel decomposition. It is the closest structural peer to
CorrTrack in the entire comparison, not a primitive ancestor.

**This is the second capability claim overturned by reading a primary source**, and
the two failed in opposite directions: CorrJoin was credited with lags it lacks
(entry (n)), StatStream was denied lags it has. The second error is the worse one
because it flattered CorrTrack, and flattering errors are the least likely to be
caught internally. A process rule is now recorded in plan §10.3: no capability
claim about a competitor enters the paper without a section or lemma number from
that competitor's own paper beside it. Abstracts, search summaries, derivative
theses and inference from publication date are all insufficient.

Every capability row in §5b.5 now carries such a pointer except one: whether
ParCorr handles negative correlation, which is flagged open rather than guessed.

### StatStream specification extracted (plan §4)

Three-level hierarchy timepoint < basic window < sliding window, `w = k*b` --
**this is CorrTrack's own model and it originates here**, the one part of the
lineage framing that survives. Normalization is the centred unit-norm form,
**identical to ParCorr's and CorrJoin's**, so the phase-1 shared prerequisite now
has three consumers. Lemma 1 gives `corr = 1 - d^2/2`, the same identity CorrJoin
uses. Lemma 2 gives the filter radius **`eps = sqrt(1-T)`** with no false
negatives -- note it lacks CorrJoin's factor of 2, because DFT conjugate symmetry
supplies `2*d_n^2 <= d^2` for free. Lemma 7 bounds the feature space to a cube of
diameter sqrt(2), and the grid partitions that cube into cells of diameter `eps`,
probing adjacent cells (like CorrJoin, unlike ParCorr). Lagged mode uses `T_M` as
the max lag with per-cell and per-stream timestamps; **the grid path quantizes lags
to multiples of the basic window**, with finer lags available via a precomputed
`W(m,p,d)` table at `O(k*n^2)` per pair instead of `O(k*n)`.

Parameters: `n=16` DFT coefficients (swept 16/24/32/40), 2 coefficients per basic
window for curve fitting (a different quantity -- do not conflate), `w` 1800-7200
timepoints, `b` 0.5 to several minutes, `T` 0.85/0.9, post-processing tolerance
0.001/0.0005. Published results: precision 0.9765-0.9947, recall 0.9987-1.0,
pruning power 0.01-0.09. **Recall below 1.0 comes from the DFT post-processing,
not the grid** -- which gives the sharpest verification anchor of any arm: with
post-processing disabled our port must show grid recall of exactly 1.0.

### Consequence for the CorrTrack story (plan §4.3)

The comfortable framing is gone and should not be reinstated. A reviewer will ask
what CorrTrack adds over StatStream. The honest list, to be **tested rather than
asserted**: LSH/Hamming candidate backends versus a fixed regular grid on the first
few DFT dimensions (grids degrade in higher dimensions -- measurable at equal recall
with the §5b.1 counters); arbitrary integer lags versus basic-window-multiple lags;
and episode/attention/anomaly monitoring versus a single minimum-duration parameter.

One genuinely good story did emerge: **StatStream (first `n` DFT coefficients),
FilCorr (band-pass) and BRAID (smoothing) all bet on energy concentrating at low
frequencies, and CorrTrack does not.** That is a coherent, testable thesis on
bursty and high-frequency data, and it is stronger than a speed table.

Also recorded: §6.1 now requires running the lagged comparison **both** at
basic-window-multiple lags (which flatters StatStream) and at arbitrary lags (which
flatters CorrTrack), and saying which is which. Reporting only one would be
cherry-picking of the same kind §6.4 already forbids for density.

### ParCorr cell size -- DECIDED (plan §2.3)

Per the user's instruction, **`_compute_base_cell_size` is not used for the ParCorr
arm.** It is the user's own derivation, the paper contains no such rule, and
carrying it in would mean publishing ParCorr numbers that depend on an unpublished
design choice of ours.

Replacement follows ParCorr's own published methodology, which is a calibration
procedure rather than a formula: fix `f=0.7` and `k=2` at the paper's values, then
on a held-out sample take the **largest** cell size that still reaches target recall
0.95, and record it as a calibrated hyperparameter stated as such. Largest, because
the cell size is the pruning knob and a bigger cell is a weaker filter, so this is
the setting that respects the recall constraint while giving ParCorr its best speed.
If the calibration proves unstable, that instability is itself a reportable property
of ParCorr's grid and must be reported, not patched with a formula.

Noted for contrast: StatStream **does** specify a cell size (`eps = sqrt(1-T)` on
its bounded cube), but that rule is tied to the DFT feature space's bound and is not
transferable to ParCorr's unbounded random-projection space. Not to be borrowed
across arms.

### Plan sections updated

Status header; §1 table (StatStream row rewritten); §2.2 last row and new §2.3
(cell-size decision); §4 fully rewritten as paper-authoritative with §4.0
correction, §4.1 specification, §4.2 parameters/results, §4.3 the reframing;
§5a.1 (BRAID's distinctiveness narrowed and restated); §5b.5 capability matrix
rebuilt as a table with sources; §6.1 lag protocol; §6.5 anchors (StatStream's
two-stage anchor added); §7 sequencing (StatStream re-justified as the most
important arm, risk upgraded); §10.2 item 4 closed, items 6 and 7 added;
new §10.3 process note.

### Known issues / still open

- "Plan it but do not do it yet" still stands. Nothing implemented.
- **Open and explicitly not guessed**: does ParCorr handle negative correlation?
  The one blank cell in the capability table (§10.2 item 7).
- BRAID's TKDD 2010 extension still unread; may add pair pruning (§10.2 item 5).
- StatStream's TAQ dataset is licensed; the synthetic random walk is reproducible
  from the formula and should simply be added (§10.2 item 6).
- The 7 `abaca/` comparison scripts remain uncommitted (unchanged since entry (n)).

### Next exact step

No literature blockers remain for the pruning family. Either resolve the two small
open questions (ParCorr negative correlation; BRAID's TKDD extension), or begin
phase 1 if the user lifts the implementation hold: the shared
normalize-window-before-reducing path, now with **three** consumers (ParCorr,
CorrJoin, StatStream), plus factoring out the five-sum sufficient-statistics helper
shared by `exact_stomp`, CorrJoin and BRAID.

---

## 2026-09-16 (a) -- BRAID/TKDD and Cole-Shasha-Zhao read; ParCorr negative correlation
closed; evidence-tier grading added to the capability matrix. Planning only.

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

### (1) The user's methodological correction, now structural (plan §5b.6)

The user observed that papers claim capabilities (lags, negative correlation) that
they never implement or evaluate, so we should not assume they have them. Correct,
and it exposed a weakness in the previous two revisions of this plan.

The clearest case is **StatStream and lags**. Not a bare claim: there is a
derivation (Theorem 1, Corollary 1), grid pseudocode with timestamps (§3.6) and a
named system parameter `T_M` (§4). But StatStream's §5 asks exactly three empirical
questions -- speed, approximation error versus window size, pruning power/precision
-- and **no experiment involves a lag**. Same for Lemma 3 on negative correlation.
"Duration over Threshold" is weaker still: a paragraph in the system description,
no algorithm, no measurement.

The capability matrix is now graded **[E] evaluated / [S] specified / [C] claimed /
[N] absent / [?] source unread**, with the handling rules spelled out:

- **[E]** only tier that supports a head-to-head claim.
- **[S]** may be implemented, but the result is *our* evaluation of *their* design,
  never a reproduction of their result. **A poor result is not evidence against the
  method**, since there is no published baseline to say we built it as intended.
- **[C]** not implemented as theirs at all. Building it and attributing it to them
  is the §10.3 error in its most tempting form.
- **The same standard applies to CorrTrack.** Grading competitors on evaluation
  while claiming our own on implementation alone would be self-serving.

**Consequence for §6.1**: the lagged comparison has **one** genuine published peer
(BRAID), not three. StatStream's lag path can still be built and measured, but
labelled as our evaluation.

### (2) Cole-Shasha-Zhao, KDD 2005 -- the biggest finding of the session (plan §4a)

*Fast Window Correlations Over Uncooperative Time Series* (Cole, Shasha, Zhao, NYU).

**ParCorr's candidate search was published in 2005.** CSZ §5.3: partition the sketch
vector into groups of size `g`, one grid per group, candidate if within `c x d` in
more than a fraction `f` of the groups. ParCorr §4.3 is the same scheme with the
same `f`. Same random +1/-1 sketch, same `d^2 = 2(1-corr)` reduction. **Dennis
Shasha is an author of StatStream (2002), CSZ (2005) and ParCorr (2018).** ParCorr's
stated contribution is the incremental sketch update with folded-in normalization
plus Spark execution, not the candidate-search scheme.

Three consequences:

- **The lineage is real**: StatStream 2002 -> **CSZ 2005** -> ParCorr 2018 ->
  CorrTrack v1 -> CorrTrack. **CSZ, not ParCorr, is the true ancestor of CorrTrack's
  candidate search.** Citing ParCorr alone as the origin would be an attribution
  error in our own related-work section.
- **It fills the ParCorr cell-size gap with a published parameter.** CSZ publishes
  the distance multiplier `c` (swept 0.1 to 1.3) alongside `f` (0.1 to 1.0), `N`
  (30/36/48/60) and `g` (1-4). So §2.3's calibration should sweep **`c` over CSZ's
  published range**, making the ParCorr arm's tuning fully traceable to published
  work by the same research line, with nothing of ours in it. This is a better
  answer than the calibrated-hyperparameter compromise recorded yesterday.
- **Their tuning protocol is more rigorous than ours**: two-factor combinatorial
  design (2,080 settings to 130), local neighbourhood refinement, then bootstrapping
  for out-of-sample robustness, targeting recall >= 0.99. Worth adopting rather than
  just citing -- and this project already owns that machinery from the OA(16,5,4,2)
  and Sobol sweeps.

**The cooperative/uncooperative axis (plan §4a.2) is the most useful idea in the
paper.** Cooperative = energy in the first few Fourier coefficients (random walks,
stock *prices*); uncooperative = energy spread across all frequencies (stock
*returns*). For uncooperative data, **DFT, DWT and even SVD approximate distances
badly** while random-projection sketches hold up. Their summary: "sketches are like
B-trees (the default choice) and Fourier Transform approaches are like bit vectors."

This names, from the literature, the axis entry (p) had identified independently:
StatStream, FilCorr and BRAID all bet on low-frequency dominance; CorrTrack does
not. Two consequences: the evaluation **must report prices and returns separately**
(one transform apart, on data we already hold), and the predicted result is recorded
in advance -- the DFT-based arms should degrade on returns while the sketch-based
ones hold up.

**Decision: fold CSZ into the ParCorr arm, do not build a second near-duplicate**
(§4a.3). One mechanism, reported as the ParCorr/CSZ family with CSZ credited as
origin, and the genuine differences (structured random vectors + convolution,
tuning protocol, incremental normalization) treated as ablations within that arm.

### (3) BRAID TKDD 2010 -- §10.2 item 5 closed

The open question was whether the journal version's "group" emphasis added pruning.
**It does not.** The extension adds **ThinBRAID**: random-projection sketches
replacing `O(k^2 log n)` stored inner products with `O(k log n)` sketches, cutting
the per-tick update from `O(k^2)` to `O(k)`. But their Table II keeps **output at
`O(k^2 log n)`**, and the paper states it plainly: "we still need `O(k^2 log n)` time
to estimate the lag correlations, but this will happen only when the user requests
us to do so." It is an amortization, not a filter. **BRAID's framing as the unpruned
lag method survives.**

Nice structural contrast for the paper: ThinBRAID and CorrTrack both use random
projections, for **opposite purposes** -- ThinBRAID to compress what it stores about
all pairs, CorrTrack to avoid looking at most pairs.

Also extracted: BRAID/ThinBRAID crossover at about **k = 1000 sequences** (their
Fig. 19; our battery runs m = 500 and up); exponential forgetting factor `lambda`
(unused, `lambda = 1` throughout) with an explicit criticism of **StatStream's**
sliding-window model for needing "a window size larger than the maximum lag";
per-dataset lag errors (Tables III/IV) as reproduction anchors; and the **Motes**
dataset (54 Berkeley sensors, lab floor plan, measured lags of 202 and 224 minutes
between physically nearby sensors).

### (4) ParCorr negative correlation -- §10.2 item 7 closed

User-confirmed: **ParCorr does not handle it.** The arm runs positive-correlation
only and the asymmetry against CorrTrack's `neg_corr=True` is reported, **not**
papered over by adding an `abs()` the method never had -- which would be inventing
capability, the mirror of the §10.3 error.

### Known issues / still open

- "Plan it but do not do it yet" still stands. Nothing implemented.
- **`docs/competitor_comparison_plan.md` §10.4 now lists the unread primary sources,
  and one of them is a real problem: the FilCorr paper (ICDM 2020) has never been
  read, yet FilCorr is already implemented and already benchmarked in our published
  results.** Two mathematical bugs were caught in that port by testing alone. Given
  that two capability claims elsewhere were overturned by finally reading the paper,
  this is the largest remaining risk in the comparison. Its whole capability column
  is `[?]`.
- **TSUBASA (SIGMOD 2022) also unread**, though only planned, not built.
- **Zhu & Shasha TR2002-827** may contain the lag experiments missing from the
  StatStream VLDB paper; if so, StatStream's lag row moves [S] to [E] and §6.1
  changes. Worth checking before concluding they never evaluated it.
- Open [S]-vs-[E] question: does CorrJoin's paper evaluate negative correlation?
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step

Ask the user for the **FilCorr (ICDM 2020)** and **TSUBASA (SIGMOD 2022)** papers,
in that order, and read FilCorr against the existing `Candidates_BF_FilCorr`
implementation to verify the port and fill its capability column. That is a check on
already-published results, so it outranks any further planning.

---

## 2026-09-16 (b) -- FilCorr, TSUBASA and the StatStream technical report read. All
primary sources now closed. Two documented deviations found in our shipped FilCorr port.

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md`,
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

### (1) FilCorr, ICDM 2020 -- the risk flagged in entry (a) is discharged

Read at last. The port is sound: the Parseval decomposition, the `O(B)` incremental
band-coefficient slide, and the exactness claim all match what we implemented. **But
two deviations need to go in the paper's methods section.**

**Deviation 1: FilCorr has no negative-correlation handling. Our port adds it.**
Their Equation 7 is `lcorr = Max(corr(t,ti), corr(ti,t))` over `ti` in `[t-l, t]`: a
maximum of **signed** correlations, never an absolute value. Verified in our code:
`Candidates_BF_FilCorr` inherits `Candidates_BF_ExactSTOMP`, whose acceptance rule
applies `np.abs(corr) >= threshold` when `neg_corr=True`
(`library_corrtrack_parallel.py`, the exact_stomp accept mask).
This is **defensible but must be stated**. As a `baseline_mode`, FilCorr's job in our
harness is to produce the same ground-truth pair set as bruteforce so that recall and
precision mean anything, so applying the harness's uniform acceptance rule is right.
The correlation **values** are identical either way, since the Parseval decomposition
is sign-preserving, so timings are essentially unaffected and the pair set is a
superset. It does **not** invalidate the 2026-09-12 four-way numbers. It is an
extension of ours and must be labelled as such, not as FilCorr.

**Deviation 2**: FilCorr reports one value per pair per timestamp (the max over the
lag window); our port reports per-lag pairs. Same output-shape mismatch as BRAID, same
resolution: compare on the common denominator, capability table for the rest.

**FilCorr rejects pruning on purpose**, which sharpens its row in the matrix. Their §I:
"Seismic traces are mostly white noise ... stressing the algorithms to fall behind the
stream quickly. The main reason for the failure of these methods is the
**data-dependent** pruning, projection, or indexing technique." So `[N]` on pruning is
a design stance, not an omission.

**Their Table I is a precedent for our evidence tiers.** Its legend: "checkmark
represents a **claimed** capability, dash represents **extendable** capability and
cross represents **unknown**." That is the same distinction as plan §5b.6, published
by a competitor. Citing it beats presenting evidence grading as our invention. They
also mark **StatStream's lagged correlation as unknown** and write "None of these
methods consider lagged correlation after filtering" -- a second group declining to
credit StatStream with lag support, which supports our `[S]` grade.

**The crossover number lands inside our battery.** FilCorr benchmarks against ParCorr
(run offline, Spark startup subtracted, to favour it) and beats best-case ParCorr at
`lag=0` **up to about 700 streams**. Our battery runs `m=500`, just below that.
Prediction recorded in advance (plan §4b.4): at `m=500` on dense data FilCorr should be
competitive with CorrTrack and should lose as `m` grows. If CorrTrack wins easily at
500, check whether our port is band-limited enough to be the method they benchmarked.

### (2) TSUBASA, SIGMOD 2022 -- read

Not a correlation join: it builds the **complete exact correlation matrix** for a
climate network. Lemma 1 recovers exact Pearson from per-basic-window mean, standard
deviation and **per-pair correlation**; Lemma 2 gives the incremental update; variable
length basic windows give **arbitrary query windows**, lifting the integral-multiple
restriction StatStream imposes.

- **No pruning, confirmed** -- their own conclusion lists "develop a pairwise
  correlation pruning algorithm based on a threshold" as future work.
- **Negative correlation: yes and exercised.** Algorithm 2 line 6 is `if |c| > theta`,
  and that edge rule runs in every experiment. Graded `[E]`.
- **No lags.**
- **Space is `O(L*N^2/B)`**, precisely `(L/B)(2 + N(N-1)/2)`, because it stores a
  correlation per **pair** per basic window. Quadratic in storage, not just in time. At
  `m=500` that is a large artifact, and this project has already lost three sessions to
  a quadratically-growing accumulator. Budget it before implementing and report it: the
  trade-off is exact arbitrary-window queries bought with quadratic storage.

**Their §4.1 is the most valuable thing in the paper for us.** On climate data the
DFT approximation matched the exact network **only when all 200 coefficients of a
200-point basic window were used**; they note climate data are **uncooperative** and
that StatStream uses two coefficients regardless of basic window size. That is
independent corroboration of the cooperative/uncooperative axis, and sharper than
Cole-Shasha-Zhao's version because it shows the consequence in the **output** (false
positive edges, a distorted network) rather than only in the distance. Three
independent papers now support the same claim:

| source | evidence |
|---|---|
| CSZ 2005, Figs. 2 and 4 | DFT, DWT and SVD approximate distances badly on white-noise-like data |
| FilCorr 2020, §I | white-noise seismic traces are why they reject pruning outright |
| TSUBASA 2022, §4.1 | on climate data the DFT approximation needs every coefficient |

### (3) StatStream technical report TR2002-827 -- read, and it settles the question

The open question was whether the fuller technical report contains the lag experiments
missing from the VLDB paper. **It does not.** The TR adds §3.6 on I/O performance and an
extra basic-window-size figure, but its §5 asks the same three empirical questions
(time savings, approximation error, pruning power and precision) and **no experiment
involves a lag**. StatStream's lag support is confirmed `[S]` in both versions:
specified in detail via Theorem 1, Corollary 1, §3.7's timestamped grid and the `T_M`
system parameter, and never measured.

### Plan sections updated

Status header; §1 table (new FilCorr row, TSUBASA row rewritten); **new §4b (FilCorr,
four subsections)**; §5 rewritten as paper-authoritative TSUBASA; §5b.5 capability
matrix (**all `[?]` cells resolved**); §5b.6 (FilCorr's Table I cited as precedent);
§10.2 item 3 closed; §10.4 rewritten as "all read".

### Known issues / still open

- "Plan it but do not do it yet" still stands. Nothing implemented.
- **The two FilCorr deviations must reach the paper's methods section.** They are
  documented in plan §4b.2 but the four-way comparison writeup does not yet mention them.
- One `[S]`-versus-`[E]` question remains: does CorrJoin *evaluate* negative correlation?
- TSUBASA's sketch store is quadratic in the number of series; budget before building.
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step

No literature blockers remain anywhere in the plan. Either start phase 1 if the user
lifts the hold (the shared normalize-window-before-reducing path, now with three
consumers, plus the five-sum sufficient-statistics helper shared by `exact_stomp`,
CorrJoin and BRAID), or add the FilCorr deviation note to the 2026-09-12 four-way
writeup, which is a correction to already-published results and is cheap.

---

## 2026-09-16 (c) -- Implementation plan for all six competitor arms written.
Planning only; still nothing implemented.

**Branch**: dev. **New file**: `docs/competitor_implementation_plan.md` (393 lines).
**Changed**: `docs/competitor_comparison_plan.md` (adds the cross-reference),
`docs/implementation_log.md`, `tasks/current_task.md`. **No code touched.**

The comparison plan settles what each method is and why we compare it; it had no
engineering plan. This adds one: files, classes, signatures, ordering, tests.

### The main structural finding

**All six arms fall into two integration patterns that already exist in this codebase**,
and the split is exactly the "prunes pairs" column of the capability matrix, which is a
good sign that the code's seams match the structure of the literature.

- **Pattern A, `baseline_mode`** (all-pairs): `bruteforce` / `exact_stomp` / `filcorr`
  already; add **TSUBASA** and **BRAID**. The recipe is a
  `Candidates_BF_X(Candidates_BF_ExactSTOMP)` subclass overriding `run()`, a `run_bf_X`
  wrapper mirroring `run_bf_filcorr` (`library_corrtrack_parallel.py:9060`, **61 lines**),
  one dispatch branch (`:9129`), one `_resolve_baseline_mode` entry (`:1929`), and CLI
  flags. Steps 2-5 are ~120 mechanical lines; all real work is the class.
- **Pattern B, `data_representation` x `candidate_backend`** (pruning): add
  **ParCorr/CSZ**, **StatStream**, **CorrJoin**. Both axes already validated and
  pluggable (`_VALID_DATA_REPRESENTATIONS:1738`,
  `_VALID_CANDIDATE_BACKEND_AXIS:1745`); they need widening back out, not inventing, and
  `corrtrack_release_v1.0` holds a disabled grid implementation to work from.

### Phase 0, blocking, 2-3 days

- **0a** `sketch_norm="unit_l2_window"`: normalize the window before reducing. **Three
  consumers** (ParCorr, CorrJoin, StatStream all use the centred unit-norm form).
  CorrTrack's `mean_l2` (`:4033`) stays the default and is untouched.
- **0b** Extract the five-sum sufficient statistics. **Three consumers**
  (`exact_stomp` inline today, CorrJoin's Eq. 2, BRAID's Eqs. 9-10). Keep it a plain
  function, not a class: the three call sites have different shapes.
- **0c** Counter contract. Every arm must fill `sk_time`/`cand_time`/`val_time`,
  `total_candidates`, `tested_candidates`. For Pattern A arms the pruning counters must
  be **explicitly 0, never null**, or the three-phase table in comparison plan §5b.1 has
  holes exactly where the argument is.
- **0d** **Negative-correlation policy, decided in writing.** Live problem, not
  hypothetical: ParCorr has none, FilCorr has none (Eq. 7 takes `max` of *signed*
  correlations), yet our shipped FilCorr port applies `|corr| >= T` via the inherited
  accept mask. **Recommendation: primary head-to-head at `neg_corr=False`** so no arm is
  extended beyond its paper, plus a second labelled run where arms lacking the capability
  report N/A rather than being silently extended. Enforce with a per-arm
  `supports_neg_corr` flag the harness *refuses* on, not a convention. This also settles
  the outstanding FilCorr deviation from entry (b).
- **0e** One shared `_assert_competitor_contract(arm)` test: counters, declared
  capabilities matching the matrix, the degenerate-exact anchor, state round-trip.
  **Highest-leverage item in phase 0** -- this discipline caught both real FilCorr bugs.
- **0f** Two checks: `n_lags=0` has **no explicit guard** in the library and the primary
  head-to-head runs there; scipy 1.11.4 verified present locally (already imported at
  `:19`), confirm on the Abaca env before BRAID.

### Per-arm notes worth keeping

- **TSUBASA** first: cheapest, exact so self-verifying, proves the Pattern A template
  generalizes past FilCorr. **Memory quantified in advance**: per-pair-per-basic-window
  store is `O(m^2 k)`, about 14 MB at m=500/k=14 but **~224 MB at m=2000** -- Abaca fine,
  WSL not. Report the quadratic storage as a finding, not a surprise. Note its headline
  feature (arbitrary query windows) is never exercised by our fixed-window harness.
- **BRAID**: hierarchical means + five sums + `CubicSpline` and
  `minimize_scalar(method="bounded")` (the paper uses Brent and says interpolation is
  orthogonal, so the substitution is safe but should be noted). Adaptation to label:
  BRAID's `m = n/2` grows with the stream, ours is capped at `n_lags`. Anchor: `2b >
  n_lags` makes level 0 exact, so it must match the bruteforce lagged pair set.
- **ParCorr/CSZ as one arm**, differences as ablations. Three things in v1.0 are present
  but disabled and must be corrected not inherited (`freq_threshold=0` at `:2233`,
  `n_grids=1` collapse at `:2182-2185`, `required_hits=n_grids` i.e. effective `f=1.0`
  at `:4647`). Cell size via CSZ's published `c` in [0.1,1.3], **not**
  `_compute_base_cell_size`.
- **StatStream**: build the `[E]` synchronous core first; the `[S]` lagged and
  negative-correlation paths second and separately labelled, since **no published number
  exists to check them against**. Sharpest anchor of any arm: grid recall must be
  **exactly 1.0** with post-processing disabled (their Theorem 2), only then the
  published precision/recall with it enabled. Watch the never-cleared lagged grid's
  memory.
- **CorrJoin** last. `eps_1 = sqrt(2 ks (1-T)/n)`, `eps_2 = sqrt(2 ke (1-T)/n)`; both
  divide by n and eps_1 uses ks. **Ship per-window SVD first**, measure, then decide on
  incremental -- and state which, next to any stride result, since per-window recompute
  is what made the UZH thesis report no stride effect.

### Ordering

Track A and Track B touch **disjoint code**, so they run independently after phase 0.
A: TSUBASA (2-3 d) -> BRAID (4-5 d). B: ParCorr/CSZ (3-4 d) -> StatStream (5-7 d) ->
CorrJoin (7-10 d). **Total ~25-32 focused days**, spread almost entirely in B2/B3.
Fallback: Track A + B1 + B2 complete, CorrJoin to related work. **Cut from the end of
Track B, never from Track A** -- BRAID is the only lag peer with published evidence, so
its absence would leave CorrTrack's central claim unmeasured.

### Six comparability mechanisms (§5 of the new doc)

One harness; one exact-validation kernel so only candidate generation differs in tier;
counters before seconds; each arm tuned by its own authors' protocol; no arm gains a
capability its paper lacks (enforced in code); both sides of every axis reported
(density, cooperative/uncooperative, lag granularity).

### Predictions recorded in advance (§6)

Each doubles as a bug detector, e.g. BRAID winning at large k means our port is wrong;
a CorrJoin speedup above `1/r1` is a bug; StatStream *should* degrade on returns and
hold on prices, and if it does not the port is wrong before the thesis is.

### Known issues / still open

- Nothing implemented. Awaiting the go-ahead.
- **Five open decisions in §8**; item 1 (negative-correlation policy) should be settled
  first because it also resolves the FilCorr deviation already in published results.
- The 7 `abaca/` comparison scripts remain uncommitted.

### Next exact step

Decide §8 item 1, then phase 0a: add the `sketch_norm="unit_l2_window"` path with the
`2 - 2*corr == d^2` identity test, leaving `mean_l2` as CorrTrack's default.

## 2026-09-16 (d) -- Cooperative/uncooperative (CSZ 2005) axis wired into the real
comparison harness; sp500 FilCorr band-width experiment; raw-mode reproduction of the
FINAL five-way table submitted to Abaca

User connected `competitor_comparison_plan.md` sec 4a.2 (Cole/Shasha/Zhao KDD 2005's
cooperative/uncooperative distinction: stock prices are cooperative -- DFT/DWT/SVD
digests work; stock returns are uncooperative -- they don't, sketches hold up) to the
project's own existing `preprocess` parameter, and relayed two supervisor asks: add
sp500 *returns* to the big table (not only prices) and add a FilCorr recall column.

**Verified before acting:**
- `preprocess=True` (`Sketches._preprocess_data`, library_corrtrack_parallel.py:5053-
  5060) is exactly `diff_t = t[:,1:] - t[:,:-1]` -- a plain first difference,
  disclosed as an approximation of the literal finance return `(p_t+1-p_t)/p_t`, not
  that quantity itself.
- The script that built the FINAL five-way table (2026-09-15(i)),
  `hamming_exact_compare.py`, hardcodes `preprocess=True` for every dataset including
  sp500 (three call sites). **So the existing sp500 rows are already the
  differenced/returns-like (uncooperative) case, not prices** -- the actual gap is the
  missing raw-price (cooperative) case, the opposite of how the request first reads.
  This resolves why every real dataset's raw space was previously found "near-total
  correlation" (2026-09-11/12 entries): that finding *was* the cooperative regime, we
  just hadn't named it.
- FilCorr's band parameters (`filcorr_fs`/`filcorr_ft`, default 0.0/0.5 = full band,
  DC removed) are already fully wired end-to-end (`run_and_log_bruteforce`'s
  `base_config` -> `corrtrack.filcorr_fs/ft` -> `Candidates_BF_FilCorr`,
  library_corrtrack_parallel.py:1198-1205,:9085-9087) -- no code change needed to vary
  the band, only to actually pass a narrow one. Full band is mathematically identical
  to raw Pearson (Parseval identity), which is why FilCorr's recall reads 1.000 in
  every row of the existing table -- it has never been run at anything but full band.
  User chose (over "just split the existing spd/rec cell") to make FilCorr's recall
  genuinely non-trivial: run it at a narrow low-frequency band and show it losing
  recall on uncooperative data, matching StatStream's own published convention of
  keeping 2 DFT coefficients per basic window (plan doc sec 4.2) as the narrow-band
  choice (`filcorr_ft = 3/W`, keeping bins [1,3) i.e. the 2 lowest non-DC frequencies).
  This makes narrow-band FilCorr a genuinely APPROXIMATE method in this experiment
  (unlike its full-band use elsewhere as an exact ground-truth-equivalent baseline) --
  flagged per CLAUDE.md's exact-vs-heuristic distinction requirement.

**Built and running:**
1. `filcorr_band_coop_uncoop.py` (scratchpad): sp500, W=30/STEP=3/N_LAGS=15/THR=0.70
   (matching the big table's own sp500 config), 2 regimes (prices: preprocess=False;
   returns: preprocess=True) x 2 FilCorr bands (full vs narrow-2-coefficient),
   recall/precision/speedup of each band against that regime's own bruteforce ground
   truth. Running locally (backgrounded via the harness's own `run_in_background`,
   not nohup+disown -- see the ASOS-fetch lesson below). Interim result: prices-regime
   bruteforce took 332.0s (vs 64-70s for the same dataset in returns space in the
   original table) with avg_degree=492/492 (m=492 -- literally every series pairs
   with every other) and tuple density 0.1808 (vs sub-1% typical in returns space) --
   confirms sp500 raw/cooperative space is both scientifically degenerate (matches
   every other dataset's raw-space finding) AND substantially more expensive to
   compute, not just less interesting.
2. **Minimal-diff edit to `hamming_exact_compare.py`** (the exact script that built
   the FINAL five-way table, synced at `~/corrtrack_abaca_results/`): added an
   optional third CLI arg selecting `preprocess` (default `true`, so every existing
   invocation and result file is byte-for-byte unaffected), threaded into the three
   previously-hardcoded `preprocess=True` call sites, and the output filename gains a
   `_raw` suffix when `preprocess=False` so raw-mode results cannot collide with or
   overwrite the 32 existing returns-space result files.
3. **Smoke-tested locally first** (wikipedia, m=88, thr=0.70, preprocess=false) before
   committing to the full grid: completed cleanly in ~80s total, no hang/OOM,
   confirming the same degenerate pattern (avg_degree=87.69/87, i.e. total
   correlation) and, notably, CorrTrack's own methods collapsing to parity with or
   worse than brute force (`lsh_sign_dot` picked at 1.00x, `hamming+dot` at 0.85x) --
   at near-total density there is nothing left to prune, a clean and expected result,
   not a bug.
4. Given sp500's demonstrated ~5x raw-mode slowdown (and total-correlation regimes
   generally inflating validation/recording cost, not just BF), routed the full
   8-dataset x 4-threshold reproduction (32 runs) to Abaca rather than running serially
   on a laptop, mirroring exactly how the original table was built. Synced the updated
   script; all 8 dataset `.npz` files were already present on Abaca from the original
   run. Generated and submitted 32 OAR jobs (`raw_<dataset>_<thr>`, 24h walltime each,
   queue `abaca`) -- **OAR IDs 3111266-3111297**. Not yet checked for completion.

**Process note, unrelated to the above but from the same turn**: a `nohup ... &
disown`-based background-download launcher (used earlier this session for sourcing
more ASOS airport station data) was silently torn down by the sandbox roughly 15
minutes after the launching tool call returned, despite `disown` -- concurrently, 8
parallel requests also tripped mesonet's rate limiter. Both were fixed by (a) using
the harness's own `run_in_background: true` directly on the actual long-running
command instead of a detached nohup wrapper (survives correctly, confirmed by the
199-country global ASOS fetch now in progress), and (b) serializing requests with
retry/backoff instead of firing them concurrently.

### Known issues / still open

- The 32 raw-mode Abaca jobs (3111266-3111297) and the local sp500 FilCorr
  coop/uncoop experiment had not finished by the time of this entry.
- Raw/cooperative space is expected, per every prior real-dataset finding this
  session, to show near-total density for every dataset in the 32-job grid, not just
  sp500/wikipedia -- precision/recall/avg_degree numbers there may be near-degenerate
  across the board; report as-is rather than treating it as a run failure.
- `filcorr_band_coop_uncoop.py`'s narrow-band FilCorr result is not yet available for
  either regime -- the predicted pattern (narrow band recall holds on prices,
  degrades on returns) is stated in the script's own docstring in advance, not yet
  confirmed.
- The full 199-country global ASOS fetch (separate task, same session) is still
  in progress; some large countries are expected to fail on mesonet's "1,000
  station-years" per-request cap (confirmed for Italy) and will need a follow-up
  chunked-window pass.

### Next exact step

Check `oarstat`/job output for 3111266-3111297 and the local FilCorr experiment;
once both land, build the paired prices/returns rows into the big table (labeled
explicitly, e.g. "sp500 (prices)" / "sp500 (returns)") and the narrow-vs-full-band
FilCorr recall comparison; update `tasks/current_task.md`.

## 2026-09-16 (e) -- Session restart lost local scratchpad work; both experiments
relaunched to Abaca; user corrected which "big table" raw mode should reproduce

**Session restart, mid-turn**: the previous session ended and restarted. The 32 raw-
mode Abaca OAR jobs (independent cluster processes) were unaffected -- 29/32 had
completed. But both locally-run background processes were silently killed with no
completion record, and their entire `/tmp/.../scratchpad/` directory (not just the
process) was wiped on restart: the local sp500 FilCorr coop/uncoop experiment (no
result saved, only the "prices" bruteforce had finished) and the full 199-country
global ASOS fetch (was around country 92+/199, every downloaded file lost). This
confirms a real, now twice-observed constraint: neither a `nohup ... & disown`
launcher NOR the harness's own `run_in_background` on a LOCAL process survives a
session boundary -- only work submitted to an external, session-independent system
(Abaca/OAR, or a remote frontend process launched over its own ssh connection)
does. Local `/tmp` scratchpad content does not survive either.

**Both relaunched onto Abaca, this time correctly decoupled from the local session:**
- Global ASOS fetch: rewritten as `fetch_all_countries.py` (adds proper recursive
  date-range bisection on mesonet's "too much data... 1,000 station_years" cap,
  seen on Italy last time, instead of uselessly retrying the same oversized request
  6 times) and launched via `ssh sophia.g5k "nohup python3 fetch_all_countries.py
  ... & disown"` directly on the Abaca **frontend** (confirmed to have outbound
  internet access; this is I/O-bound waiting, not compute, so it does not need an
  OAR allocation). Verified alive via `pgrep` after the launching ssh command
  returned -- this is a genuinely different mechanism from the earlier local nohup
  (a different machine entirely, unaffected by this machine's session lifecycle).
- FilCorr coop/uncoop band experiment: resubmitted as a real OAR job (`filcorr_band_
  coopuncoop`, **job ID 3111875**, 6h walltime) instead of a local background
  process, for the same durability reason.

**User then corrected the raw-mode big-table reproduction**: the 32-job battery
submitted in entry (d) reproduced the FINAL five-way table's own per-dataset-native-m
configuration (`hamming_exact_compare.py` -- sp500 m=492, streamflow m=538, etc.).
The user wanted the OTHER, separately-existing standardized battery instead: the
m=500-common-scale comparison (`hamming_exact_compare_m500.py` /
`tmp_artifacts/m500_results/`, 7 datasets -- streamflow/sp500/wikipedia/smartmeter/
sp500_sub263/global_weather/acwi_capweighted, sp500 accepted under-target at its
real m=444 ceiling; acwi_real excluded, structurally capped near m~160 by its own
top-4-per-country selection rule, documented as non-comparable at this scale).

**Fix**: applied the identical minimal-diff `preprocess` CLI parameterization (third
arg, default `true`, `_raw` filename suffix when `false`) to
`hamming_exact_compare_m500.py` as was done to the native-m script in entry (d) --
same 3 call sites, same line numbers (the two scripts are structurally identical
apart from the CFG dict and dataset set). Verified all 7 m=500-cut `.npz` files
already present on Abaca. Generated and submitted 28 OAR jobs (`rawm5_<dataset>_<thr>`,
7 datasets x 4 thresholds, 24h walltime) -- **OAR IDs 3111881-3111908**.

The native-m raw-mode results from entry (d) (29/32 cells) are not discarded -- they
answer a different, still-valid question (raw mode at each dataset's own real native
scale, directly paired against the FINAL five-way table's own per-dataset-m rows) --
but the m=500-standardized battery is the one the user actually asked to see paired
against `m500_results/`, and is what "the big table" now refers to going forward.

### Known issues / still open

- All of (d)'s open items still open (32-native-m job stragglers now done per the
  chat, local FilCorr result still pending -- now as OAR job 3111875).
- The 28 m=500-standardized raw-mode jobs (3111881-3111908) had not finished by the
  time of this entry.
- Global ASOS fetch restarted from zero (country 1/199) on the Abaca frontend --
  no progress carries over from the lost attempt.

### Next exact step

Check `oarstat`/job outputs for 3111881-3111908 (m=500 raw table) and 3111875
(FilCorr band experiment); check the Abaca-frontend ASOS fetch log
(`~/corrtrack_abaca_results/global_asos/fetch_progress.log`) for progress. Build the
m=500 raw-vs-returns paired table once both regimes are available; update
`tasks/current_task.md`.

---

## 2026-09-16 (d) -- Correction: two evaluated lag peers, not one. §8 item 1 decided.

**Branch**: dev. **Changed files**: `docs/competitor_comparison_plan.md` (§6.1),
`docs/competitor_implementation_plan.md` (§4 rationale, §4 fallback, §8 item 1),
`docs/implementation_log.md`. **No code touched.**

### The correction, and how it happened

The user asked: if BRAID is "the only lag peer with published evidence", what about
FilCorr and CSZ? Checked against the capability matrix, which was already right:

- **FilCorr is [E] on lags.** Lagged correlation is its problem statement, and its
  Figs. 6-7 sweep `lag = 0, 100, 250`. A genuine evaluated lag peer, and **already in
  the battery**.
- **Cole-Shasha-Zhao is [C].** "Asynchronous correlation" is defined in its problem
  statement and named in its summary, but its §5-6 contain no lag algorithm and no lag
  experiment. Not a lag peer, and not to be built as one.
- StatStream is [S], as established.

So the lagged head-to-head has **two** published peers, FilCorr and BRAID. The "only
BRAID" sentence was written in entry (a), when FilCorr's column was still `[?]`, and
**was not propagated when the matrix was updated in entry (b)**. It survived into the
implementation plan's §4 ordering rationale, and the same stale premise was in the
user-facing summary. Same failure class as §10.3 of the comparison plan: a claim not
re-derived after its source was read.

### A second stale claim found while fixing the first, and it was worse

Comparison plan §6.1 described FilCorr as "**shares CorrTrack's pruning structure** but
restricts the frequency band." **FilCorr does not prune** -- it rejects pruning by
design (its §I), and the capability matrix already said `[N]`. That sentence was also
pre-FilCorr-paper. Additionally, the evidence-tier paragraph I intended for §6.1 in
entry (a) **never landed**: it was in a patch script whose later assertion failed, so
the whole script aborted before writing, and I re-ran only the parts I noticed.

Both fixed. §6.1 now grades the lag row explicitly (FilCorr [E], BRAID [E],
StatStream [S], CSZ [C]) and describes FilCorr correctly as exact, unpruned and
band-limited.

### Consequence for the implementation plan

BRAID's priority argument changes from "the only evaluated lag peer" to "**the only
evaluated lag peer we do not already have**". Still not to be cut: FilCorr and BRAID
probe different failure modes (band-limiting versus approximating the lag value
itself), so one does not substitute for the other. The fallback now correctly reads
"both evaluated lag peers" (FilCorr built, BRAID from Track A).

### Decision recorded

**§8 item 1 (negative-correlation policy) is decided, with the user's agreement**:
primary head-to-head at `neg_corr=False`; a second labelled run at `neg_corr=True` in
which arms whose papers lack the capability report N/A; enforced by a per-arm
`supports_neg_corr` flag the harness refuses on. This also settles the FilCorr
deviation from entry (b).

### Process note

Two consecutive corrections came from the user reading my summary against my own
matrix. The matrix was right both times; the prose around it was stale. The lesson
for the remaining planning is mechanical: **after any matrix update, grep every doc
for prose that restates a matrix cell and re-derive it**. A patch script that aborts
on assertion must be re-run whole, not piecemeal.

### Next exact step

Phase 0a: the `sketch_norm="unit_l2_window"` path with the `2 - 2*corr == d^2`
identity test, `mean_l2` untouched as CorrTrack's default.

---

## 2026-09-16 (e) -- §8 item 1 refined by the user: enable-and-disclose where possible,
N/A only where we would be designing their algorithm.

**Changed**: `docs/competitor_implementation_plan.md` §0d and §8. **No code touched.**

User's refinement: the `neg_corr=True` run should **not** blanket-N/A every arm lacking
native support. Where the capability can be enabled without us inventing part of the
method, enable it, report it, and say plainly in the paper that we enabled it and the
original did not have it. N/A is reserved for arms where enabling would mean designing
their candidate mechanism for them. More informative and still honest.

**Criterion adopted**: not coding effort, but whether enabling changes the algorithm
under measurement. Four classes, written as a per-arm tag into `RUN_RESULT_COLUMNS` so
the disclosure travels with the number:
- `native`: BF, STOMP, TSUBASA.
- `specified` (their paper gives it, never measured -- our evaluation of their spec):
  BRAID, StatStream (Lemma 3), CorrJoin (Alg. 1 L14).
- `enabled_by_us` (exact signed values already computed; enabling is an acceptance
  filter, algorithm untouched): **FilCorr**.
- `not_available` (would need a new candidate-generation mechanism): **ParCorr/CSZ**.

**Flagged, pending confirmation**: the user gave FilCorr as the example of "cannot
easily be enabled". The code says otherwise -- our port already enables it via the
inherited accept mask, and Parseval preserves sign, so nothing in FilCorr's algorithm
changes. Under the adopted criterion FilCorr is `enabled_by_us`, and that is precisely
what turns the 2026-09-12 numbers into a disclosable extension rather than a retraction.
ParCorr/CSZ is the arm that genuinely lands in N/A. Recorded as recommended.

### Next exact step
Confirm the FilCorr classification, then phase 0a.

---

## 2026-09-16 (f) -- §8 item 1 fully decided: FilCorr confirmed `enabled_by_us`.

**Changed**: `docs/competitor_implementation_plan.md` §0d, §8. **No code touched.**

User confirmed FilCorr is `enabled_by_us`, not N/A (the earlier example was a slip).
The 2026-09-12 FilCorr numbers therefore **stand**, with the disclosure that negative
correlation was enabled by us and absent from the original paper. No re-run needed.
ParCorr/CSZ is the only arm in N/A. §0d has no open items.

### Next exact step
Phase 0a, whenever the implementation hold is lifted.

---

## 2026-09-16 (g) -- §8 items 2-5 decided by the user. One overstatement of mine
corrected (CorrJoin's SVD). Battery target raised to 2,000 series.

**Changed**: `docs/competitor_implementation_plan.md` (§0f new, §0g renumbered, A2, B3,
§4, §5 item 4, §8, new §9), `docs/competitor_comparison_plan.md` (§3.2 SVD correction,
§7 row 6). **No code touched.**

### 2. CorrJoin SVD -- my framing was wrong

User asked: if the authors' R code is incremental, why not reproduce it rather than
call it research-grade? Re-derived: the SVD acts on the **`m x ks` matrix**, `ks = 15`.
That is a thin `O(m ks^2)` decomposition -- ~450k flops at `m = 2000`, microseconds to
milliseconds per window. **Recomputing per window is negligible**, not a compromise.
"Incremental SVD is research-grade" was carried from the pre-paper draft and never
re-derived after the paper; same failure class as §10.3. Also: what the paper specifies
as incremental is the correlation (Eq. 2) and PAA means (`O(h+k)`), not the SVD, and we
do not have the R code (not online; ask the authors). **Decision: per-window SVD.**
Follow-ups: confirm the paper's `cSVD` term when the PDF is to hand; request the R code.
CorrJoin's estimate drops 7-10 d -> 5-7 d and it is no longer "the hardest".

### 3. ThinBRAID -- build it

Battery runs to **2,000 series**, past the ~1,000 crossover. Memory decides it more
than speed: plain BRAID's per-pair `Sxy_h(l)` store at `k = 2000`, `b = 16`, 5 levels
is `1,999,000 * 96 * 8 B` = **~1.5 GB**; ThinBRAID's per-series projections
`(400+200+100+50+25) * 2000 * 8 B` = **~12 MB**. `thin=True` mode of the BRAID class,
reusing CorrTrack's projection machinery; new pieces are Eq. 24 and the JL sizing
(Theorem 5). Anchor: agree with BRAID within their Lemma 9 bound. +2-3 d on Track A.

### 4. Tuning -- CSZ protocol for competitors, CorrTrack keeps its own

Same 0.95 target and calibration span for both. Asymmetry disclosed in the methods
section and **defused by running CorrTrack once under the CSZ protocol as a robustness
check**; if the two disagree, that is a tuning-sensitivity finding to report.

### 5. Datasets -- add every obtainable one

New phase 0f: registry `datasets/competitor_sources.md` + per-set fetch scripts on the
`build_*_full.py` pattern. 14 sets catalogued with access class and the axis each tests.
Priority: free synthetic first; then **Motes** (BRAID; the only real multi-series set
that is lag-correlated by construction) and **Yellowstone** (FilCorr; real uncooperative,
lagged by wave propagation, fully specified event); then CorrJoin's Drive sets and the
two climate sets; then UCR. Licensed TAQ/CRSP replaced by `sp500`, stated. ~3-5 d.

### Totals

~30-40 focused days (was 25-32), spread in B2, B3 and 0f. **All five §8 decisions
made; nothing blocks phase 0a.** New §9 records what 2k changes per arm, including a
second FilCorr prediction: at 2k, past their ~700 crossover, the unpruned methods
should all be behind.

### Next exact step
Phase 0a, on the go-ahead.

---

## 2026-09-16 (h) -- CorrJoin authors' R code received. CorrJoin is now a reproduction.
Implementation hold lifted; phase 0a begins.

**Branch**: dev. **New**: `docs/reference_code/corrjoin_authors_R/` (6 scripts + README).
**Changed**: `docs/competitor_implementation_plan.md` B3, `docs/competitor_comparison_plan.md`
(§3.0 source table, §5b.5 matrix, §10.2 items 3b and 8). **No library code touched yet.**

I had said "we don't have the R code". The user had not sent it earlier in this session
(transcript checked: eight PDFs only), but sent it now: the six scripts the paper made
available -- IncPPAA, CorrJoin, BucketingFilter, Quickjoin, Ekdb, and **their TSUBASA**.

### What `2-CorrJoin.R` settles

- **SVD recomputed from scratch every window**, inside `for (i in 2:numOfSW)`. The
  per-window decision from entry (g) is therefore what the authors did. CorrJoin becomes
  a **reproduction**, not a spec-driven port.
- **PAA also recomputed per window**, not the `O(h+k)` update the paper describes. Only
  per-series `sum x`, `sum x^2` are incremental. `sum xy` is recomputed from raw for each
  surviving candidate -- which is exactly what our `validate_corr_rows` path does.
- **Normalization is applied after PAA** using the window mean and centred L2 norm
  (`paamN <- (paam - meanT)/tauT`, `tauT = sqrt(sum x^2 - n*mean^2)`). By linearity that is
  PAA(x_hat). **This is precisely the phase 0a design** -- mean-adjust the reduced vector,
  divide by the window norm -- now confirmed by the authors rather than derived by me.
- `EuclThreshold <- sqrt(2*theta/frameSize)` with `frameSize = n/k`: confirms
  `eps = sqrt(2 k (1-T) / n)` against running code.
- `abs(corr) >= corrThreshold` is live in the experimental loop, so **CorrJoin negative
  correlation moves [S] -> [E]**. §10.2 item 8 closed; no [S]-vs-[E] questions remain.
- Constants: `windowSize=1020`, `theta=0.1` (T=0.9), `stride=100`, ks=15, ke=30, kb=3.

### What `3-BucketingFilter.R` settles

3D grid over the kb=3 SVD coordinates, cell width eps_1, **27-cell neighbourhood probed**
(hand-unrolled across 1004 lines), **exact eps_1-ball test on every pair in a
neighbourhood**. One thing **not** to copy: `checkVal <- 9` scans only bins within +/-9 of
the centre, roughly coordinates in [-0.49, 0.49] at their eps_1 -- an implementation
shortcut absent from the paper that can drop candidates at extreme coordinates. Port the
full grid and measure whether the truncation ever matters.

### Bonus

`6-TSUBASA.R` is the CorrJoin authors' TSUBASA reimplementation (Lemma 1 form), the code
behind their Fig. 10. A second independent witness for our own `Candidates_BF_TSUBASA`.

Also noticed: `tmp_artifacts/corrjoin_{chlorine,gas,stock}_sobol_*` already exist in
this checkout, so the CorrJoin datasets are already downloaded and in use by another
session. Phase 0f can reuse them.

### Next
Phase 0a, starting now: `sketch_norm="unit_l2_window"` in `Sketches`, wired at the three
`self._sketch_matrix = norm_matrix` finalization points, Cython untouched, `mean_l2`
byte-identical; plus the 0b five-sum helper and tests.

---

## 2026-09-16 (i) -- Phase 0a and 0b implemented and tested. 126/126.

**Branch**: dev. **Changed**: `library_corrtrack_parallel.py` (8 hunks, all in
`Sketches` + one module-level helper block), `test_stable_reproduced_changes.py`
(+4 tests, 122 -> 126). **Standalone patch of my library change**:
`docs/patches/2026-09-16_phase0a_0b_unit_l2_window_five_sums.library.patch`
(applies cleanly to the pre-0a file and reproduces the current one byte-for-byte).

### 0a: `sketch_norm="unit_l2_window"` in `Sketches`

The normalization the pruning competitors need (ParCorr/CSZ, CorrJoin, StatStream):
project the centred unit-L2 window, `x_hat = (x - mean) / ||x - mean||`.

**Design, and why it costs no new state.** For any linear reduction `R`,
`R(x_hat) = (R(x) - mean(x) * R(1)) / ||x - mean(x)||`. The numerator is exactly what
`_mean_adjust_matrix` already computes from the raw sketch and `_random_vector_sums`
(= `R(1)`); the denominator is `sqrt(var_sum)` from `_raw_window_sums` /
`_raw_window_sums_sq`, maintained unconditionally in `_newStream`. So the new mode
**rides the existing mean_l2 kernel path (code 1) and is rescaled afterwards**; the
three Cython kernels are untouched. Verified against `sketch_kernels.pyx` that all
three kernels return `raw` **un-adjusted** (adjustment goes only into `norm`), so no
double-adjust. **This is precisely what the CorrJoin authors' code does**
(`2-CorrJoin.R`: `paamN <- (paam - meanT)/tauT`), confirmed after the fact.

Changes: `_sketch_norm_mode_code` maps `unit_l2_window` -> 1; new
`_sketch_norm_is_unit_l2_window`; new `_apply_unit_l2_window_norm(raw, norm)` (no-op
for every other mode); the two Python-fallback normalizers treat `unit_l2_window` like
`mean_l2`; one call inserted before each of the **three** `self._sketch_matrix =
norm_matrix` finalization points (line-anchored, verified `raw_matrix` bound on every
branch at each). **`mean_l2` is byte-identical**: for any other mode the new call
returns its input.

Not yet plumbed as a `CorrTrack(...)` parameter -- `CorrTrack.__init__` still hardcodes
`"mean_l2"` at the 2026-07-31 site. The tests set `ct.sketch_norm` before the lazy
`Sketches` build. Proper plumbing lands with B1 (ParCorr/CSZ), its first consumer.

### 0b: `five_sums` / `pearson_from_five_sums` (module level, above `Candidates_BF`)

Plain vectorized functions over arrays, not a class, since the three consumers
(exact_stomp per window, CorrJoin per stride, BRAID per level and lag) accumulate
differently. Constant windows yield 0.0, matching the exact baselines' guard.
`exact_stomp`'s own inline sums were **not** refactored (minimal diff; risk with no
benefit until a second consumer exists).

### Tests (4 new)

- `test_unit_l2_window_identity_two_minus_two_corr_equals_d2`: the identity every
  pruning competitor rests on, 50 random windows, 1e-12.
- `test_sketches_unit_l2_window_equals_projection_of_normalized_window`: **exact**
  check `_sketch_matrix == R_eff @ x_hat` to 1e-10, with `R_eff` reconstructed
  independently from `node._toggle_weights` (and `R_eff @ 1 == _random_vector_sums`
  asserted). Also asserts the default `mean_l2` rows are still exactly unit-norm and
  parallel to the new rows. Not the code checking itself.
- `test_sketches_unit_l2_window_incremental_matches_from_scratch`: sliding
  incrementally through 40 samples equals the from-scratch sketch of the final window,
  1e-9 (same discipline as the existing mean_l2 test).
- `test_pearson_from_five_sums_matches_numpy`: incl. perfect anticorrelation = -1.0
  exactly and constant series = 0.0, not NaN.

Two test failures on the first run were harness misuse on my side (fed a whole window
in one `run`; the harness consumes `window_step` chunks). Fixed the tests, not the code.

### Full suite: **126 passed** (was 122).

### Working-tree caveat, important for the commit

`git diff --stat` shows 352 library lines changed, but **only 122 are mine**. The other
230 (280 insertions / 72 deletions vs HEAD across `CorrTrack`,
`Candidates_BF_ExactSTOMP`, `Candidates`) were **already uncommitted in the working
tree from another session** before I started; 9 claude processes are active on this
checkout. The 126/126 covers both sets together, which is reassuring for that other
work, but I did not review it and make no claim about it. The standalone patch above
isolates mine so it can be committed separately (e.g. `git apply` on a clean checkout,
or `git add -p`).

### Next exact step

Phase 0c/0e: the counter contract and `_assert_competitor_contract`. Then Track A1
(TSUBASA) and Track B1 (ParCorr/CSZ, which also plumbs `sketch_norm` through
`CorrTrack.__init__`).

---

## 2026-09-16 (j) -- Track A1 done: TSUBASA baseline arm, exact against exact_stomp
and bruteforce. Shared competitor contract test in place. 129/129.

**Branch**: dev. **Changed**: `library_corrtrack_parallel.py` (new
`Candidates_BF_TSUBASA`, `run_bf_tsubasa`, dispatch, `_resolve_baseline_mode`,
`tsubasa_bf_node` slot), `corrtrack_run_bruteforce.py` (`--baseline-mode tsubasa`),
`experiment_run_exec_param.py` (doc), `test_stable_reproduced_changes.py` (+3 tests,
126 -> 129). Cumulative standalone patch of all my library changes today:
`docs/patches/2026-09-16_competitors_cumulative.library.patch`.

### Design (docstring on the class is the full account)

Pattern A, cloned from FilCorr: `Candidates_BF_TSUBASA(Candidates_BF_ExactSTOMP)`
overriding `run()`, a 61-line `run_bf_tsubasa` mirroring `run_bf_filcorr`, one
dispatch branch, one alias entry. Per-segment sketch = per-series `(mean_j, M2_j)` plus
the **per-pair centred cross-product `CS_j`**, which equals their `B_j sigma_xj sigma_yj
c_j` exactly but needs no divide-then-multiply by sigma and no constant-segment special
case. Combination is their Lemma 1 as the ANOVA decomposition; full segments cached by
absolute segment start on a grid anchored at the first observation, evicted when they
leave the window; partial head/tail segments computed from raw each step (their
arbitrary-window case). `candidate_time` = sketch building ("summarize"),
`validation_time` = Lemma-1 combination + threshold ("verify"). Raw Euclidean
`min_dist` recovered exactly from the sketches to match exact_stomp's convention.

**One deliberate deviation from the paper's formula, in the direction of correctness**:
their `delta_xi = xbar_i - (sum_k xbar_k)/ns` uses the UNWEIGHTED mean of segment means,
exact only when all `B_i` are equal. With partial head/tail segments the length-weighted
`xbar = sum_j B_j mean_j / W` is required for the decomposition to hold. Implemented
weighted; coincides with theirs for equal segments; the exactness test below is what
proves it.

**Lags**: TSUBASA has none ([N]). Per the 0d policy the constructor **refuses**
`n_lags > 0` with a message pointing at the plan, rather than extending the method.
Declares `supports_neg_corr = "native"` (their Algorithm 2 `abs(c) > theta`).

### 0c/0e: the shared contract, corrected

Written as `_assert_competitor_contract(record, pattern)` in the test file, reading
only the `RUN_RESULT_COLUMNS` record (what reaches the paper). **My plan's 0c wording
was wrong**: for a no-pruning arm the candidate set IS the pair set, so the invariant is
`total_candidates == tested` (prune ratio 1.0) and `candidate_search_*_touched == 0`,
not `total_candidates == 0`. The FilCorr wrapper already did this correctly; the plan
text is fixed below. All three Pattern-A arms (exact_stomp, filcorr, tsubasa) pass it.

### Tests

- `test_tsubasa_node_matches_exact_stomp_pearson`: key-for-key and value-for-value
  against `Candidates_BF_ExactSTOMP` at lag 0, four configurations incl. odd window
  (33/11/3) and **window_step < basic_window** so the window start is usually off the
  segment grid -- the case that forces partial segments and the weighted xbar. Max
  abs err < 1e-9, zero key mismatches, pair counts equal, cache bounded by W/B + 1.
- `test_tsubasa_refuses_lags`.
- `test_run_and_log_bruteforce_tsubasa_matches_bruteforce_and_meets_contract`:
  end-to-end through `run_and_log_bruteforce` for bruteforce / exact_stomp / filcorr /
  tsubasa at `n_lags=0`; all four agree on `correlated`, and the three competitor arms
  satisfy the contract.

### Sanity at scale (not a benchmark)

m=200, W=168, B=12, step=12, 60 steps: **0 pair-set mismatches** vs exact_stomp;
TSUBASA 0.134 s vs exact_stomp 0.079 s (~1.7x, consistent with the plan §6 prediction
"close to exact_stomp"); 60 segments built (one per step, since step = B), cache 14 =
W/B; per-segment sketch 0.32 MB at m=200 -> the §9 memory figures hold.

### Next
Track A2: BRAID + ThinBRAID (`baseline_mode="braid"`, `thin` flag).

---

## 2026-09-16 (k) -- Track A2 done: BRAID and ThinBRAID baseline arm. 133/133.
Two findings about BRAID in a sliding-window harness, both recorded before benchmarking.

**Branch**: dev. **Changed**: `library_corrtrack_parallel.py` (new `Candidates_BF_BRAID`,
`run_bf_braid`, dispatch, aliases incl. `thinbraid`, `braid_*` knobs threaded through
`run_and_log_bruteforce`), `corrtrack_run_bruteforce.py` (`--baseline-mode braid`,
`--braid-b/-gamma/-thin/-thin-d0/-report-mode`), `experiment_run_exec_param.py`
(`BRAID_*` defaults), `test_stable_reproduced_changes.py` (+4 tests, 129 -> 133).
Cumulative patch refreshed: `docs/patches/2026-09-16_competitors_cumulative.library.patch`.

### Design (the class docstring is the full account)

Pattern A. Three ideas ported as written: five sums per (pair, probed lag); the
enhanced probing scheme `{0..2b-1; 2^h i, i in [b,2b)}` (their §3.4, verified against
their Figure 6 with b=4 by a test); block-mean smoothing anchored to absolute time.
ThinBRAID as a `thin=True` mode: per-series projections `d_h = d0 / 2^h` (their 400/2^h),
`Sxy` recovered by their Eq. 24.

**Two harness adaptations, labelled, neither an algorithmic change:**
1. BRAID's Eq. 2 correlates the common part of ONE growing stream with `m = n/2`; this
   harness correlates TWO full windows `curr = x[t-W,t)` vs `hist = y[t-W-l,t-l)` with a
   fixed `n_lags`, exactly as `exact_stomp` does. The five sums are therefore rolling
   window sums per level, **rolled the same way exact_stomp rolls `_dot_by_lag`**
   (subtract outgoing block columns, add incoming) when `window_step` is a multiple of
   the level's block width, recomputed otherwise -- exact_stomp's own `can_increment`
   policy. Same tier, verified: exact_stomp IS incremental (checked `_dot_for_lag`
   before writing this), so a matmul-per-lag BRAID would have been a tier mismatch.
2. Lag domain is integer and <= n_lags, so the spline is read on the integer grid and
   local maxima taken from it instead of Brent's method (the paper calls both choices
   orthogonal).

Two output modes: `all_lags` (thresholded pairs at every harness lag -- the
common-denominator set) and `braid` (Definition 1: one row per pair at its earliest
local max of `|R_hat| >= gamma`). `last_lag_estimates` (m x m) exposed for the
lag-agreement metric of comparison plan §5a.3.

### Performance work that was needed, and what it means

First version: 13.9 s vs exact_stomp 0.35 s at m=200, W=168, step=12, n_lags=168.
Profile: **49% in scipy's `CubicSpline` constructor** (a banded solve over 40,000
pair-curves every step); the actual BRAID sums were 16%. Fix: spline interpolation is
linear in the knot values and the knot set is fixed by the probing scheme, so the
whole fit-and-evaluate is one precomputable `(L x K)` operator, built once by pushing
the K unit vectors through `CubicSpline` -- identical output, one matmul per step.
Now **2.05 s**, ~6x exact_stomp.

**Finding 1 (comparability, important for §6.1):** that 6x is not overhead, it is the
algorithm meeting this harness. The harness lag grid is `window_step`-spaced, so at
step=12, n_lags=168 exact_stomp evaluates **15** lags while BRAID probes **70** (its
probing does not know about `window_step`). BRAID's published saving is against a
baseline that evaluates EVERY integer lag (169 here) -- which none of our exact
baselines do. **BRAID can only show its advantage at small `window_step`.** The lagged
comparison must therefore include a step=1 (or small-step) configuration, otherwise
BRAID is being asked to beat a baseline that already skips 90% of the lags. Recorded as
a required configuration, not a caveat.

**Finding 2 (ThinBRAID):** 20.5 s, ~10x plain BRAID. At `d0=400 > W_h=168` every
ThinBRAID lag costs ~4x the FLOPs of the per-pair matmul it replaces, and the JL noise
at short windows degrades lag estimates (below). ThinBRAID's win is **memory** -- 0 MB
rolling state (nothing stored across steps) vs plain BRAID's 70 (m x m) matrices,
22 MB at m=200 and **~2.2 GB at m=2000** -- and compute only when the window is much
longer than `d`. This is exactly what the plan predicted ("d exceeds the window, no
benefit") and it is now measured. **Correction to plan §9**: the 1.5 GB / 12 MB figures
were for the growing-stream formulation; in the rolling formulation the numbers are
~2.2 GB / ~0 MB. Same conclusion, ThinBRAID is what runs at 2k.

### Accuracy (lag estimation, Definition 1 vs the same rule on the exact CCF)

W=64, n_lags=30, injected lag 9, smooth non-periodic signal:

| variant | lags probed | exact-lag agreement | mean abs lag error |
|---|---|---|---|
| exact anchor b=16 (2b > n_lags) | 31/31 | **1.00** | 0.00 |
| BRAID b=4 | 16/31 | 0.69 | 0.73 |
| BRAID b=2 | 10/31 | 0.59 | 1.29 |
| ThinBRAID b=2, d0=400 | 10/31 | 0.37 | 4.84 |
| ThinBRAID b=2, d0=4000 | 10/31 | 0.52 | 2.11 |

The paper's ~1% relative lag error is at lags of 700-4000 on 32k-long sequences; ours
are single-digit lags, where one interpolation cell of error is already >10%. Regime
effect, recorded; not a bug (the anchor is exact to 7.8e-16). **An earlier version of
this check was my error**: I compared against argmax-over-accepted-rows on a signal
with a period-17 component, so even the exact CCF's argmax was 17, not the injected 6
-- exactly why Definition 1 says *earliest* local maximum. Fixed by applying the same
rule to the exact CCF.

### Tests (4 new)
`test_braid_probing_scheme_matches_paper` (their Fig. 6 lag set, and the exact-anchor
degenerate scheme); `test_braid_exact_anchor_matches_exact_stomp_when_2b_exceeds_n_lags`
(key-for-key, < 1e-12, rolling updates must engage);
`test_braid_lag_estimate_recovers_exact_earliest_local_max` (anchor = 1.00; interpolating
> 0.5; Thin not better than plain); `test_run_and_log_bruteforce_braid_dispatch_and_contract`.

### Next
Track B1: ParCorr / Cole-Shasha-Zhao (`sketch_proj` x `parcorr_grid`), which also plumbs
`sketch_norm` through `CorrTrack.__init__`.

---

## 2026-09-17 (a) -- Track B1 done: ParCorr / Cole-Shasha-Zhao as a Pattern-B arm
(`candidate_backend="parcorr_grid"`). 137/137.

**Branch**: dev. **Changed**: `library_corrtrack_parallel.py` (new `ParCorrGridIndex`;
backend axis + aliases + `_resolve_internal_dispatch` + `_resolve_candidate_backend` +
`_LSH_SIGN_DOT_BACKENDS`; `Candidates.__init__` construction branch and 4 kwargs;
`CorrTrack.__init__` 4 kwargs + sketch_norm switch + neg_corr refusal + cell size;
`_extract_feature_overrides`; record columns in `RUN_RESULT_COLUMNS` /
`OPTIM_RESULT_COLUMNS`), `corrtrack_run_corrtrack.py` (`--candidate-backend parcorr_grid`,
`--parcorr-k/-f/-c/--parcorr-neighbor-probe`), `experiment_run_exec_param.py`
(`PARCORR_*`), `test_stable_reproduced_changes.py` (+4 tests, 133 -> 137).
Cumulative patch refreshed.

### Two discoveries about `dev` that changed the approach

1. **The multi-grid vote machinery survives in `dev`** (`_run_grids`, `freq_pairs`,
   `freq_threshold`, `n_grids`), hard-disabled at `CorrTrack.__init__`
   (`full_vector_candidates=True`, `n_grids=1`, `freq_threshold=0`) -- the same state the
   plan documented for v1.0. Reviving that CorrTrack-level path touches partitioning,
   per-grid nodes and the parallel merge. **Not done.** Instead, the route
   `SignLSHBandIndex` took in July: a self-contained index class behind the existing
   `Candidates._lsh_index` interface (`insert_many` / `find_pair_rows_full_cosine` /
   `drop_before_time` / `last_stats`), so `Candidates` dispatches to it unchanged.
   Consequence: v1.0 was not mined at all; the port is from the two papers.
2. **`_resolve_candidate_backend` normalizes unknown keys to `"auto"`** rather than passing
   them through (an earlier grep of mine had suggested otherwise). First end-to-end run
   silently built a `SignLSHBandIndex` and reported identical recall for every `c`/`f`
   -- caught by the invariance, fixed by an explicit `parcorr_grid` return before the
   fallback. **The identical-across-knobs symptom is now something the test asserts
   against** (recall must move with `c` and `f`).

### Design (`ParCorrGridIndex` docstring is the full account)

Sketch = random +/-1 projection of the **unit-L2-normalized window**
(`sketch_norm="unit_l2_window"`, phase 0a, set automatically for this backend), split into
`n_grids = r // k` groups (paper: r=60, k=2 -> 30). Each group is a k-dim regular grid of
side `cell_size`; candidate if **same cell in >= ceil(f * n_grids) grids** (paper: f=0.7).
No dot gate: the paper verifies every candidate exactly, which the shared validation kernel
does downstream. `neighbor_probe=False` = ParCorr (same-cell; neighbour search is their
future work); `True` = Cole-Shasha-Zhao (probe 3^k cells, keep a group hit only if group
distance <= cell_size). One arm, the ablation the plan asked for (§4a.3).
`cell_size = c * sqrt(2(1-T))` -- CSZ's `c x d`, **not** `_compute_base_cell_size`, by
decision. Rows canonicalized exactly like the Cython indexes (later time first, tie by rank).
Pure numpy + dict postings with lazy deletion and amortized compaction.

**Negative correlation refused at construction** (`CorrTrack(..., neg_corr=True,
candidate_backend="parcorr_grid")` raises; the signed query raises) -- policy §0d.
`supports_neg_corr = "not_available"` written into the run record.

### Behaviour (m=60, W=64, step=8, n_lags=0, T=0.7, 210 true pairs; not a benchmark)

| config | recall | precision | tested | touched |
|---|---|---|---|---|
| CorrTrack lsh_sign_dot | 1.000 | 1.000 | 210 | 850 |
| parcorr c=0.2 | 0.038 | 1.000 | 8 | 11,550 |
| parcorr c=0.4 | 0.657 | 1.000 | 138 | 25,342 |
| parcorr c=0.7 (default) | 0.752 | 1.000 | 159 | 26,962 |
| parcorr c=1.0 | 0.752 | 1.000 | 159 | 26,962 |
| parcorr c=0.7, f=0.5 | 1.000 | 1.000 | 472 | 26,962 |
| CSZ c=0.4, neighbour | 1.000 | 1.000 | 30,623 | 103,184 |

Exactly the published tuning surface: recall rises with `c` then saturates once `f` binds;
loosening `f` restores recall at more candidates (472 for 210 true -> candidate precision
44%, which the paper says is acceptable). CSZ's radius at c=0.4 prunes almost nothing here
(the natural per-group distance is ~0.14 = d*sqrt(k/r), so c for CSZ should be small).
**Precision is 1.0 throughout** (exact validation downstream), as the paper reports.
**The implementation-independent counter does its job**: the grid touches ~27k
candidates where CorrTrack's LSH touches 850 -- the §5b.1 comparison, at equal recall
once calibrated.

### Tier note (honest)

`ParCorrGridIndex` is pure Python; `SignLSHBandIndex` is Cython. ~0.12 s vs 0.02 s here.
That is a tier gap the plan's §5b.3 anticipates: publish the counters (touched, tested) as
the primary comparison, wall-clock second with the gap stated. Vectorizing or Cythonizing
the vote is a later, optional step; correctness and the tuning surface come first.

### Tests (4 new)
`test_parcorr_grid_index_vote_and_row_semantics` (unit: vote threshold, ParCorr vs CSZ rule,
canonical rows incl. lagged, signed query refused, expiry);
`test_parcorr_grid_backend_dispatch_normalization_and_tuning_surface` (index class selected,
sketch_norm switched, precision 1.0, recall monotone in c and 1/f, neg_corr refused,
divisibility enforced); `test_parcorr_grid_filter_disabled_recovers_bruteforce` (the §6.5
anchor: c=1e6, f->1 hit gives the full pair set and exactly bruteforce's validated set);
`test_run_and_log_corrtrack_parcorr_knobs_thread_and_pattern_b_contract`.

**Schema note**: the knobs and `supports_neg_corr` are registered in `RUN_RESULT_COLUMNS`
and `OPTIM_RESULT_COLUMNS` (dict-built). **Not** in `COMPARISON_COLUMNS`, which is
positional -- an insert there shifted two legacy tests; reverted.

### Next
Track B2: StatStream (`sketch_dft` x `statstream_grid`). The grid machinery from B1
(cell hashing, postings, canonical rows) is reusable; the new parts are the DFT
representation with StatStream's Lemma 6 incremental digests and the neighbouring-cell
probe on the bounded DFT cube with `eps = sqrt(1-T)`.

---

## 2026-09-17 (b) -- Track B2 done: StatStream (`sketch_dft` x `statstream_grid`).
Theorem 2 anchor holds exactly across 16 configurations. 141/141.

**Branch**: dev. **Changed**: `library_corrtrack_parallel.py` (`Sketches`: `representation`
kwarg, `_dft_basis_matrix`, `_sketches_dft`, hook in `_get_sketches`; new
`StatStreamGridIndex`; `sketch_dft` representation + `statstream_grid` backend in the axes,
aliases, `_resolve_internal_dispatch`, `_resolve_candidate_backend`, `_LSH_SIGN_DOT_BACKENDS`;
`Candidates` construction branch + 3 kwargs; `CorrTrack.__init__` 3 kwargs, representation
switch, `eps`, `n_vectors := 2n`; feature overrides; record columns),
`corrtrack_run_corrtrack.py` (`--data-representation sketch_dft`,
`--statstream-n-coeffs/-index-dims`, `--no-statstream-dft-filter`),
`experiment_run_exec_param.py`, tests (+4, 137 -> 141). Cumulative patch refreshed.

### Design (docstrings on `_sketches_dft` and `StatStreamGridIndex` are the full account)

**Representation.** DFT is linear, so StatStream's Lemma 4 (`X_hat_0 = 0`, `X_hat_i = X_i /
sigma_x`) is the phase-0a identity: project the raw window onto the DFT basis for bins 1..n
(bin 0 is where the mean lives, so excluding it IS the mean-adjust) and divide by the
window's centred L2 norm. 2n real dims `[Re, Im]`, paper's `1/sqrt(W)` scaling. Recomputed
per step from the window; the paper's Lemma 6 per-basic-window digest update is an
`O(W/b)` constant-factor optimization of the same numbers and is not implemented -- its
absence is charged to this arm's `sk_time` and stated. `n_vectors` is set to `2n`
internally for this arm.

**Index.** Regular grid on the first `h` of the 2n coordinates of the bounded cube (Lemma 7:
every coordinate in `[-sqrt2/2, sqrt2/2]`), cell **side** `eps = sqrt(1-T)` (Lemma 2), the
`3^h` neighbouring cells probed -- which is what makes it false-negative-free (Theorem 2) --
then the paper's n-approximate filter `||X_hat - Y_hat||_2n <= eps`, then the harness's exact
validation. The paper never states `h`; `3^h` probes bound it, default 4 (two complex
coefficients). Negative correlation per Lemma 3: also probe the cell of `-X_hat` and test
`||X_hat + Y_hat|| <= eps`. Lags: entries stay alive across the lag horizon and are evicted
by time -- their "timestamped, never globally cleared" grid, at the harness's step
granularity. Both **[S]**: enabled as their specification and our evaluation, tagged
`supports_neg_corr = "specified"` in the record (plan §0d).

### One real bug, found by the negative-correlation test

First run: lags fine, but exactly the 100 anti-correlated pairs missed. Cause: with
`eps ~ 0.55` the `-X_hat` cell is a **neighbour** of the `+X_hat` cell, so an anti-correlated
entry is first met by the positive probe, fails the positive distance test, and a shared
`seen` set then made the negative probe skip it. Fix: dedupe per `(entry, sign)`. This is the
kind of bug only the [S]-path test catches -- had negative correlation been left as a
"claimed" cell, the code would have silently under-recalled.

### Behaviour (m=60, W=64, step=8, n_lags=0; not a benchmark)

| data | T | truth | arm | recall | tested | grid touched |
|---|---|---|---|---|---|---|
| white noise | 0.7 | 210 | StatStream n=16, h=4 | 1.000 | 771 | 3,540 |
| white noise | 0.7 | 210 | StatStream **n=4**, h=4 | 1.000 | **36,796** | 3,540 |
| white noise | 0.9 | 110 | StatStream n=16 | 1.000 | 210 | 3,540 |
| random walks | 0.7 | 2,670 | StatStream n=16 | 1.000 | 3,633 | 3,342 |
| random walks | 0.7 | 2,670 | StatStream n=4 | 1.000 | 6,852 | 3,342 |
| random walks | 0.9 | 160 | StatStream n=16 | 1.000 | 258 | **2,514** |

Three things, all consistent with the literature and now measured in our harness:
1. **Recall 1.0 everywhere** -- Theorem 2, with exact validation replacing their approximate
   post-processing (so precision is 1.0 too, where they report 0.977-0.995).
2. **The uncooperative finding reproduces on the first run**: with n=4 coefficients white
   noise tests 36,796 pairs for 210 true (~no pruning) while random walks test 6,852 for
   2,670. Cole-Shasha-Zhao and TSUBASA said this; it is now a regression test here.
3. **At moderate T the grid does nothing; the DFT filter does the pruning.** With
   coordinates bounded by 0.707 and `eps = 0.55`, there are 2 cells per axis, so the 3^h
   neighbourhood covers the whole cube: touched = all alive pairs at T=0.7, dropping only at
   T=0.9 (`eps = 0.32`, 4-6 cells per axis). Their Fig. 5 pruning power (0.01-0.09) is at
   T=0.85-0.9 for the same reason. **StatStream's grid is a high-threshold instrument**; the
   paper's evaluation range is where it works, and the plan's T sweep must include 0.9.

### Tests (4 new)
`test_sketch_dft_representation_is_normalized_dft_and_bounded` (exact vs `np.fft`; Lemma 7
bound; Lemma 2's `d_n <= sqrt(2(1-corr))` for every pair);
`test_statstream_grid_has_no_false_negatives_theorem_2` (**16 configurations**: T in
{0.7, 0.9} x neg in {F, T} x n_lags in {0, 16} x DFT filter {off, on}; recall == 1.0,
precision == 1.0, validated set == bruteforce's set, with anti-correlated and lag-16 pairs
planted); `test_statstream_dft_filter_prunes_and_uncooperative_data_prunes_less`;
`test_run_and_log_corrtrack_statstream_knobs_thread_and_pattern_b_contract`.

### Next
Track B3: CorrJoin (`sketch_paa_svd` x `corrjoin_double_filter`), now a reproduction from
the authors' R code: PAA(ks=15) -> per-window SVD -> 3-dim bucket grid (side eps_1, 27
neighbours, exact eps_1-ball) -> PAA(ke=30) Euclidean eps_2 -> exact validation.

---

## 2026-09-17 (c) -- Track B3 done: CorrJoin (`sketch_paa_svd` x `corrjoin_double_filter`).
All six competitor arms now implemented. 145/145.

**Branch**: dev. **Changed**: `library_corrtrack_parallel.py` (`Sketches`: `representation="paa"`,
`paa_ks`/`paa_ke` kwargs, `_paa_basis_matrix`, `_sketches_paa`; new `CorrJoinDoubleFilterIndex`;
`sketch_paa_svd` representation and `corrjoin_double_filter` backend in the axes, aliases,
`_resolve_internal_dispatch`, `_resolve_candidate_backend`, `_LSH_SIGN_DOT_BACKENDS`; `Candidates`
construction branch + 5 kwargs; `CorrTrack.__init__` kwargs `corrjoin_ks=15, corrjoin_ke=30,
corrjoin_kb=3`, divisibility check, `eps_1`/`eps_2` derivation, `n_vectors := ks + ke`; feature
overrides; record columns `corrjoin_ks/ke/kb/eps1/eps2`), `corrtrack_run_corrtrack.py`
(`--data-representation sketch_paa_svd`, `--corrjoin-ks/-ke/-kb`), `experiment_run_exec_param.py`
(`CORRJOIN_KS/KE/KB`), tests (+4, 141 -> 145). Cumulative patch refreshed
(`docs/patches/2026-09-16_competitors_cumulative.library.patch`, 2140 lines).

### What was built (reproduction of `docs/reference_code/corrjoin_authors_R/`)
- Representation: normalized window `x_hat` -> `[PAA_ks(x_hat) | PAA_ke(x_hat)]`, both linear
  operators (frame means), stored in one `Sketches` matrix of width `ks + ke`. Normalization is
  after PAA with the window's own mean and L2 norm, as in `1-IncPPAA.R`.
- Backend, per query batch: SVD of the *alive* `m x ks` block (`3-BucketingFilter.R` recomputes
  it every window; nothing incremental beyond the per-series sums), keep `kb=3` right singular
  vectors (zero-padded when rank < kb, i.e. m < 3), bucket grid of side `eps_1` on the projected
  coordinates, probe the 27-cell neighbourhood, exact `eps_1`-ball test on the projection, then
  the `eps_2` Euclidean test on the `PAA_ke` block. `eps_1 = sqrt(2 ks (1-T) / W)`,
  `eps_2 = sqrt(2 ke (1-T) / W)`. Survivors go to the shared exact Pearson validation.
- `last_r1` exposes the fraction of pairs that survive the first filter (the paper's `r1`; its
  speedup ceiling is `1/r1`).
- Refuses `n_lags > 0` (synchronous only, [N] for lags) and `neg_corr=True` (see finding).

### Verified
- `_sketch_matrix` equals an independent `PAA(x_hat)` reconstruction at 1e-10, including
  off-grid window starts and incremental refills.
- **No false negatives**: recall 1.0 against `bruteforce` in all 8 configurations
  (`T in {0.7, 0.9}` x `{noise, walks}` x `{ks,ke} in {(14,28),(7,14)}` at W=168, m=20).
  This is the guarantee the filters are supposed to give (both bounds are Cauchy-Schwarz /
  Parseval-type lower bounds on the true distance) and the test confirms it end to end.
- Pattern-B counter contract holds (`total >= tested >= correlated`); knobs and `eps_1/eps_2`
  land in the record; `supports_neg_corr == "not_available"` recorded.
- `r1`: 0.99 on white noise at T=0.7 (the grid does nothing, same as StatStream's finding in
  (b)); 0.107 on random walks at T=0.9. Consistent with the paper's speedups being reported on
  cooperative data at high thresholds.

### Finding: negative correlation is unreachable through CorrJoin's filters
Yesterday's entry (h-i) tagged CorrJoin neg-corr as [E] because `2-CorrJoin.R` tests
`abs(corr) >= T`. That reading was wrong and is corrected today in the comparison plan
(§5b.5 matrix cell and §10.2 item 8). The `abs()` runs *after* the bucket grid and the
Euclidean filter, and both admit only pairs at *small* distance, i.e. `corr` near `+1`. An
anti-correlated pair has `||x_hat - y_hat||^2 = 2 - 2 corr` close to 4 and is pruned before
the `abs()` is ever evaluated. Test `test_corrjoin_negative_correlation_unreachable_through_its_filters`
plants `corr = -0.95` pairs and shows the index returns none of them while returning all the
positive plants. Tier goes [E] -> [S] (specified in the code, structurally unreachable);
`supports_neg_corr = "not_available"`: it cannot be enabled without replacing the method's
filters, so the neg_corr=True run reports N/A for CorrJoin, same as ParCorr/CSZ.

### Known issues
- Pure-Python index like the other Pattern-B competitors; SVD via `numpy.linalg.svd` per batch.
  Counters remain the primary comparison, wall time secondary and stated as such.
- `ks`, `ke` must divide W. Paper used W=1020 with (15, 30); for W=168 the campaign needs
  (14, 28) or similar and the paper must say so.
- Other session's 230 uncommitted library lines still coexist; `abaca/` scripts uncommitted.

### Next
Phase 0f (datasets registry + fetch scripts), N-way generalization of
`abaca/fourway_compare.py`, CSZ tuning protocol for the competitor arms (CorrTrack keeps its
own), then the campaign configs.

---

## 2026-09-17 (d) -- Phase 0f done: competitor datasets fetched and registered; N-way runner;
two ThinBRAID defects found on real data and fixed; FilCorr/BRAID neg-corr tags reach the record. 147/147.

**Branch**: dev. **Changed**: new `datasets/competitor_loader.py`, `datasets/competitor_sources.md`,
`datasets/fetch/{_common,_mseed,gen_statstream_randomwalk,gen_braid_synthetic,fetch_motes,
fetch_yellowstone_iris,fetch_uscrn_hourly,fetch_berkeley_earth,fetch_corrjoin_drive,fetch_sunspots,
fetch_csz_daisy}.py`, 22 `experiment_dataset_<name>.py` configs, `abaca/nway_compare.py` +
`abaca/nway_compare.oar`, `.gitignore` (+`/datasets/competitor/`),
`library_corrtrack_parallel.py` (ThinBRAID fixes; `_BASELINE_SUPPORTS_NEG_CORR`), tests (+2, 145 -> 147).
Data on disk: `datasets/competitor/*.npz`, 1.1 GB, git-ignored, all regenerable.

### Datasets (registry has the per-set alignment choices)
- **Motes** (BRAID): 2.3 M rows -> 4 variables on the file's 31 s epoch grid; 93k 7-field rows
  skipped; physically impossible readings masked (19% of raw temperatures are 122 C / 385 C from
  dying batteries); motes < 50% coverage dropped -> 27/31/42/42 motes.
- **Yellowstone** (FilCorr): EarthScope FDSN (IRIS's ASCII `timeseries` service is gone, only
  miniSEED `dataselect` remains) -> wrote a 150-line Steim1/2 decoder with the frames' reverse
  integration constant as a built-in check rather than install obspy into the system Python.
  WY network, 100 Hz vertical channels open on the event date = **29 stations, the paper's
  number**; one (YHR) has no waveform in the window. The M6.5 event is unmistakable (RMS x400).
- **USCRN 2020** (TSUBASA): 155 station files, 4 variables, QC flags honoured (unflagged solar
  had 2256 W/m^2 readings). 135 to 153 stations kept at 90% coverage.
- **Berkeley Earth** (TSUBASA): 458 MB HDF5, read with h5py in a scratch venv (not available in
  the system Python; on Abaca use the conda env). 18,520 complete land cells vs the paper's
  18,638. This is the > 2k-series scalability set.
- **CorrJoin**: the five Drive files the other session fetched on 2026-09-16 re-exported with
  provenance metadata; the script also refetches from Drive file ids.
- **CSZ**: UCR TSDMA 2002 is offline; five of the ten sets are DaISy originals and were fetched
  from KU Leuven. `--chunk L` reconstructs the "thousands of series" reading, disclosed.
- Sunspots (single series), StatStream random walks (exact formula), BRAID Sines/SpikeTrains
  (approximation, planted lags recorded in meta).

### N-way runner
`abaca/nway_compare.py` runs any subset of the 11 arms (bruteforce, exact_stomp, filcorr,
tsubasa, braid, thinbraid, corrtrack, parcorr, csz, statstream, corrjoin) on one dataset config
and applies section 8 automatically: `--neg-corr` reports N/A for `not_available` arms;
`n_lags > 0` reports N/A for TSUBASA and CorrJoin. Recall/precision against bruteforce via
`compute_metrics_bf`; JSON output. Verified on Motes temperature (W=96, step=12, T=0.9):

| arm | n_lags=0 | | n_lags=24, neg_corr | |
|---|---|---|---|---|
| | recall | total/tested | recall | total/tested |
| exact_stomp, filcorr, braid | 1.000 | 79,326 | 1.000 | 247,455 |
| tsubasa | 1.000 | 79,326 | N/A (sync only) | |
| thinbraid (after fixes) | 0.967 | 79,326 | 0.975 | 247,455 |
| corrtrack (untuned n_vectors=32) | 0.784 | 26,326 | 0.804 | 117,093 |
| parcorr (paper defaults) | 0.405 | 13,735 | N/A | |
| csz (neighbour probe) | 1.000 | 57,706 | N/A | |
| statstream | 1.000 | 34,652 | 1.000 | 149,670 |
| corrjoin (ks=12, ke=24) | 1.000 | 35,471 | N/A | |

ParCorr at paper defaults on real data recalls 40%: the CSZ tuning pass (next item) is
required before any ParCorr number is quoted. Pure-Python indexes are 5-50x slower in wall
time than the Cython LSH; counters are the comparison, as decided.

### Two ThinBRAID defects, invisible on the synthetic tests
ThinBRAID recalled **0.12** on Motes at T=0.9 (BRAID 1.0). Diagnosis with a direct
estimate-vs-exact probe (`Candidates_BF_BRAID(thin=True)` on 12 series):
1. The shared-projection cache `_px_cache` was keyed on `(h, cb_start)`, a *buffer-relative*
   block index. Once the rolling buffer starts evicting, `cb_start` is constant, so every step
   after the first eviction reused the first step's `px`. Mean |corr error| 0.67, unchanged by
   d0. The A2 tests never rolled the buffer. Key is now the absolute start time.
2. Eq. 24 applied to raw windows: the JL error of the distance estimate scales with
   `||x - y||^2`, dominated on sensor data by `W (mean_x - mean_y)^2`, while Pearson needs the
   centred cross-sum. With the cache fixed, mean error was still 0.16 (max 1.5, values
   clipped at 1.0). Projecting the mean-adjusted windows (linear; uses the sums already in
   hand; same state shape as their Table II) and adding the exact `sx sy / W_h` back gives
   mean error **0.009** (max 0.07) at d0=400 and 0.005 at d0=4000, i.e. JL noise. This is a
   deviation from the paper's equation as written and is disclosed in the code and here.
Recall on Motes went 0.12 -> 0.967 (precision 0.935) at n_lags=0; 0.975 / 0.981 at
n_lags=24. New regression test rolls the buffer on offset-mean data and bounds the error.

### Record tag for Pattern A
`supports_neg_corr` was set from `_internal_dispatch` only, so all Pattern-A arms recorded
`native`. `run_and_log_bruteforce` now sets it per method (`_BASELINE_SUPPORTS_NEG_CORR`:
filcorr `enabled_by_us`, braid `specified`, tsubasa/bruteforce/exact_stomp `native`); the
A1 end-to-end test asserts it.

### Known issues
- ParCorr/CSZ need the CSZ tuning protocol before being quoted (recall 0.40 at defaults).
- Berkeley Earth fetch needs h5py; USCRN takes ~3 min; Yellowstone ~30 s; all cached in raw/.
- `docs/` and `tasks/` are untracked (`??`) after the user's 25fbe9a commit removed them
  from .gitignore; not staged by me.
- ThinBRAID wall time is above BRAID's here (short W, d0=400 > W_h): expected, the point of
  Thin is memory at m = 2k, not speed (implementation plan section 9).

### Next
CSZ tuning protocol for the competitor arms (`abaca/tune_competitors.py`: two-factor design +
local refinement, 0.95 recall target, CorrTrack keeps its own hyperopt), then the campaign
configs per dataset (W/step/lag per registry row), then the T sweep including 0.9 and the
lagged comparison including step=1.

---

## 2026-09-17 (e) -- CSZ tuning protocol implemented for the competitor arms; campaign
manifest and OAR wrappers; FilCorr deviation note added to the 2026-09-12 entry. 148/148.

**Branch**: dev. **New**: `abaca/tune_competitors.py`, `abaca/tune_competitors.oar`,
`abaca/campaign_competitors.py`. **Modified**: `abaca/nway_compare.py` (`--competitor-params`),
`abaca/nway_compare.oar` (KEY=VALUE args, COMPETITOR_PARAMS / BEST_PARAMS), tests (+1),
`docs/competitor_implementation_plan.md` (section 5 item 4 implemented; new section 3b campaign),
2026-09-12 log entry (FilCorr note). Cumulative patch unchanged (no library edits).

### Protocol as implemented (comparison plan section 4a.1, decided 2026-09-16)
Calibration span = `corrtrack_param_search.prepare_training_data` (first TRAIN_RATIO of the
stream), the span CorrTrack's own hyperopt uses; target = exec config `TARGET_RECALL` (0.95).
1. Strength-2 covering array over the arm's grid. ParCorr/CSZ use CSZ's published grid, N in
   {30, 36, 48, 60}, g in {1..4}, c in {0.1..1.3}, f in {0.1..1.0} = 2,080 settings, covered in
   **130 rows, the paper's own count** (only the structurally invalid pair N=30, g=4 is
   uncovered; test asserts this). StatStream (24) and CorrJoin (84 valid at W=96) grids are
   small enough for full enumeration.
2. Coordinate-neighbourhood refinement from the best row, up to 3 rounds.
3. Block bootstrap of recall (8 contiguous window blocks x 1000 resamples); the 90% lower
   bound must reach the target, else the next feasible setting is tried.
Selection: feasible (recall >= target, precision >= 0.02) first, then fewer
`total_candidates`, then higher precision. Exact-recall arms therefore get their cheapest
filter. Output `best_params_<arm>.json` (+ `_tuning` block) and `tuning_<arm>.json` (every
evaluated setting), consumed by `nway_compare.py --competitor-params`.

### First run: Motes temperature, W=96, step=12, n_lags=0, T=0.9, calibration 2,000 rows
| arm | design runs | feasible | chosen | calib recall / cand | bootstrap lower | held-out recall (8,000 rows) |
|---|---|---|---|---|---|---|
| parcorr | 130 (56 s) | 45 | N=48, g=1, c=0.3, f=0.6 | 0.969 / 38,027 | 0.958 | **0.944** |
| csz | 130 (827 s) | 114 | N=30, g=1, c=0.5, f=1.0 | 0.966 / 37,583 | 0.9515 | **0.938** |
| statstream | 24 (11 s) | 24 | n_coeffs=32, h=1 | 1.000 / 36,368 | 1.0 | 1.000 |
| corrjoin | 84 (38 s) | 84 | ks=32, ke=48, kb=4 | 1.000 / 36,688 | 1.0 | 1.000 |
(ground truth on the calibration span: 36,066 correlated of 55,809 pairs; the dataset is so
dense at T=0.9 that the pruning floor is ~36k, and every tuned arm sits within 5% of it.)

Observations worth keeping:
- The bootstrap stage changed a decision: CSZ's cheapest design row (g=2, 36,257 candidates,
  recall 0.954) failed the lower bound and the protocol moved to g=1.
- ParCorr went from **0.40 recall at paper defaults** (entry (d)) to 0.94 held-out. CSZ's
  neighbour probing buys nothing here (same recall, 10x the time) because the tuned cell size
  already makes same-cell collisions sufficient.
- Held-out recall lands 2 to 3 points under the calibration figure for the approximate arms;
  a 2,000-row calibration span is short (17 blocks of 12 steps). The campaign uses
  `CALIB_OBS=5000`. This gap is a result to report, not to tune away.
- CorrTrack itself ran UNTUNED here (0.69); its hyperopt is a separate stage and the N-way
  output says so explicitly. No CorrTrack number from this session is quotable.

### Campaign manifest
`abaca/campaign_competitors.py`: 51 cells (Motes x {0, 480 lags} x {0.7, 0.8, 0.9}; Yellowstone
raw and band-passed at FilCorr's W=2000 / lag 1000, plus a **step=1** lagged cell; USCRN
synchronous and lagged; Berkeley Earth m in {500, 1000, 2000} and the exact arms + StatStream
at the full 18,520; CorrJoin's three real sets at ks/ke = 14/28; StatStream random walks,
CorrJoin random, BRAID Sines and SpikeTrains; `sp500_sub263` stand-in). `--emit` writes the
`oarsub` script: tune job, then two dependent N-way jobs (primary, `--neg-corr`). Parameters
travel as `KEY=VALUE` script arguments (`+` encodes spaces inside EXTRA_ARGS) because OAR does
not propagate the submitter's environment. **Nothing submitted**; the Abaca clone is at
da0f4f7 and needs the pending commit(s) plus the datasets (`datasets/competitor/` is
git-ignored: rsync it or run the fetch scripts on the frontend; Berkeley Earth needs h5py in
the conda env).

### Known issues
- Pure-Python competitor indexes make tuning slow on larger m (CSZ 6 s per run at m=27; at
  m=2000 the 130-row design will need the 12 to 48 h walltimes in the manifest, or a smaller
  calibration span).
- The campaign's CorrTrack stage (`corrtrack_param_search.py` per cell) is not in the manifest
  yet; `BEST_PARAMS` paths are placeholders until it is.

### Next
Push the pending commit, sync datasets to Abaca, submit a 3-cell pilot (Motes T=0.9 sync,
Motes lag 480, Yellowstone bp) before the full 51-cell campaign; add the CorrTrack hyperopt
stage to the manifest.

---

## 2026-09-17 (f) -- CorrTrack hyperopt in every campaign cell; Phase R paper-reproduction runner
(first findings); T sweep adds 0.95; W/step/L proposals; Abaca synced. 148/148.

**Branch**: dev. **New**: `abaca/hyperopt_corrtrack.oar`, `abaca/reproduce_papers.py`.
**Modified**: `corrtrack_param_search.py` (`--n-series/--n-obs` overrides, dataset_id follows them),
`abaca/campaign_competitors.py` (6 jobs per cell, T 0.95, `--select`, W/step/L proposals),
`abaca/nway_compare.oar` (`HYPEROPT_DIR`), `abaca/tune_competitors.oar`,
`datasets/fetch/gen_braid_synthetic.py` (`--n-components`, default 1), implementation plan (3a, 3b).
No library changes; cumulative patch unchanged.

### Decisions applied
- **CorrTrack tuned by its own hyperopt in every execution** (user, 2026-09-17). Per cell and per
  labelled run: `hyperopt_corrtrack.oar` -> `corrtrack_param_search.py --result-folder <cell dir>`,
  `tune_competitors.oar`, then `nway_compare.oar` with `-a` on both. Verified end to end locally on
  Motes m=12: hyperopt best_params consumed by nway, CorrTrack recall 0.998 (untuned 0.69).
- T sweep {0.7, 0.8, 0.9, 0.95}: 68 cells, 408 jobs.

### Phase R: reproduce the papers on their own data (`abaca/reproduce_papers.py`)
Local pilots at small m; the Abaca-scale runs are queued in the plan.
- **StatStream** (random walks, m=150, W=256, n=16): grid pruning power 0.0086-0.0247 (paper
  0.01-0.09) and recall 1.0 (paper 0.9987-1.0) reproduce; **precision of their approximate
  rule is 0.66-0.74 against their 0.9765-0.9947.** The rule as implemented: corr_approx =
  1 - d_n^2/2 from 16 coefficients, report if >= T - t; truncation only drops energy so it
  over-reports. Their sliding-window length and walk parameters are not in the plan; until
  read from the PDF this is an open discrepancy, recorded, not tuned away.
- **BRAID**: the naive baseline of the paper is Definition 1 on the exact CCF, so that is the
  reference. Single-component sines (m=16, W=1024, L=256): BRAID 0.29% error relative to the
  lag, 0.017% relative to the lag range (paper 0.000%); ThinBRAID 22% / 1.8% (paper 1.397%).
  The range-normalized ThinBRAID figure matches the paper and the lag-normalized one does
  not, which is evidence the paper normalizes by the range; **to confirm on the PDF.** Two
  method-level facts surfaced: (i) a mixture of incommensurate sines gives the CCF spurious
  early local maxima, so Definition 1 is ambiguous there (my first generator, 3 components:
  5% error); the paper's 0.000% implies a clean single lag, hence `--n-components 1`;
  (ii) Motes CCFs are flat at small lags (R(0)=0.9957, R(1)=0.9958), Definition 1 lands on
  0 or 1 at random on both sides, and a lag-relative error is meaningless; BRAID's knot values
  there are exact to 1e-12 (checked at W=1024 and 4096), so this is the metric, not the port.
- CorrJoin (r1, 1/r1 ceiling, correlated fraction at the paper's W=1020/15/30/3), ParCorr
  (r=60, k=2, f=0.7, w=500, b=20 on the stock stand-ins), FilCorr (time ratio vs naive on white
  noise at 100 Hz in the 3-7 Hz band; Yellowstone case study), TSUBASA (exactness, wall vs our
  incremental all-pairs with the caveat): implemented, to run on Abaca.

### W / step / L proposals (awaiting decision; encoded in `campaign_competitors.cells()`)
| dataset | sampling | W | step (= basic window) | L | reason |
|---|---|---|---|---|---|
| Motes | 31 s | 240 (~2 h) | 24 (~12 min) | 0 and 480 (~4.1 h) | L covers BRAID's 224 min (433 epochs); 240 keeps CorrJoin's ks/ke = 15/30 |
| Yellowstone | 100 Hz | 2000 (20 s) | 100 (1 s) | 1000 (10 s) | FilCorr's own case-study setting; one step=1 cell for the lag-probe argument |
| USCRN | hourly | 168 (a week) | 12 | 0 and 48 (2 days) | 2026-09-12 hourly convention; fronts cross the network in hours to two days; ks/ke 14/28 |
| Berkeley Earth | daily | 90 (a season) | 10 | 0 (m sweep), 30 at 2k | TSUBASA ran synchronous; ks/ke 9/18 |
| CorrJoin stock/chlorine/gas/random | unitless | 240 | 24 | 0 | synchronous method; 15/30 hold; paper's W=1020 is Phase R |
| StatStream random walks | unitless | 256 | 32 | 0 and 64 | CSZ's sw=256/bw=32; L = 2 basic windows |
| BRAID Sines/SpikeTrains | unitless | 1024 | 128 | 256 | planted lags <= 168 |
| sp500_sub263 | daily | 60 (a quarter) | 10 | 20 | step 5 of the 2026-09-16 sweep breaks ParCorr's step == basic_window |
Rules: step divides W with W's parity so `get_bst_basic_window` returns step (ParCorr's
constraint); L a multiple of step (StatStream lag granularity); W/step about 10:1. Checked
programmatically; the only exception is the deliberate step=1 cell, whose arm list excludes
the grid methods.

### Abaca
Frontend `sophia.g5k`: repo pulled to 9e983d3 (scp'd `abaca/` copies backed up to
`~/corrtrack_abaca_results/backup_abaca_scp_2026-09-17`, local edits stashed as
`stash@{0}`), `datasets/competitor/` rsynced (24 files, 1.1 GB, Berkeley Earth included so h5py
is not needed there). Still needed before submission: the user's push of da417b0 and of this
entry's commit, `git pull` on the frontend, and the W/step/L decision. Then
`python abaca/campaign_competitors.py --emit abaca/pilot_submit.sh --select ...` (3 cells,
18 jobs) and `bash abaca/pilot_submit.sh`.

### Known issues
- StatStream precision discrepancy and BRAID error normalization: both need the PDFs.
- Phase R CorrJoin/ParCorr/FilCorr/TSUBASA not yet run at scale.
- `campaign_competitors.py` walltimes are conservative guesses, not fitted.

---

## 2026-09-17 (g) -- Campaign redesigned as a Sobol (m, L) design with the full T sweep; six
legacy real sets added; dataset profile and memory recorded per run; basic_window = step. 148/148.

**Branch**: dev. **New**: `abaca/dataset_profile.py`, `experiment_dataset_{sp500,acwi_capweighted,
streamflow,wikipedia,global_weather,smartmeter}.py`. **Modified**: `abaca/campaign_competitors.py`
(DatasetSpec table + `sobol_points`/`cells(points, seed)`, BASIC_WINDOW token, m in the stem),
`abaca/nway_compare.py` (profile, peak RSS, per-pair candidate time, knob columns in the JSON),
three `.oar` wrappers (`BASIC_WINDOW`, default = step), `datasets/competitor_loader.py` (glob
fallback for `finance_sectors/sp500.npz`, `streamflow/ca_streamflow.npz`),
`abaca/reproduce_papers.py` (StatStream rule variants, W=1800/b=60).

### Design (user, 2026-09-17)
- **T swept in full** {0.7, 0.8, 0.9, 0.95} at every design point.
- **Sobol over (m, L)** per dataset: the same scrambled `qmc.Sobol(d=2)` unit-square sequence for
  every dataset (as in the 2026-09-16 real-data sweep, with T removed from the sequence), mapped
  to m log-uniform in [m_min, m_max] and L uniform integer in [1, L_max], `n_lags = (L-1) step`.
  Plus one synchronous anchor per dataset (m_max, L=1), Berkeley Earth's full-m anchor for the
  exact arms + StatStream, and the step=1 Yellowstone cell. `--points` (default 4) sets the size:
  4 points -> 385 cells x 6 jobs = **2,310 OAR jobs**; 8 points -> roughly twice that. The user
  chooses the budget.
- **Datasets**: 13 registry sets + the six 2026-09 sweep sets at their historical W/step (sp500
  60/5, acwi 60/5, streamflow 30/3, wikipedia 30/3, global_weather 30/3, smartmeter 48/8), each
  with an L_max from its cadence (log entry (f) table).
- **basic_window = step is passed explicitly to every arm** (ParCorr's constraint; also fixes
  StatStream's lag granularity to the step). Previously the library inferred the divisor closest
  to sqrt(W) that step divides (10 for sp500's 60/5), which silently violated ParCorr's model.

### Metrics kept per run (nway JSON + the run CSV's RUN_RESULT_COLUMNS)
- Accuracy vs bruteforce on the same span: recall, precision, F1 (per (pair, window), canonical
  later-time-first keys); BRAID's lag agreement is a separate metric (Phase R and
  `last_lag_estimates`), not folded into recall.
- Work counters (implementation-independent): `total_candidates`, `tested`, `correlated`,
  `candidate_search_*_touched`, `lsh_candidates_touched`; derived candidate ratio and
  candidate time per pair-window.
- Time: wall per arm, `sk_time`, `cand_time`, `val_time`, `monit_time`; speedup vs bruteforce.
- Memory: process peak RSS after each arm (coarse, monotone; the ThinBRAID-vs-BRAID question).
- Provenance: every knob (n_vectors, backend, representation, parcorr/statstream/corrjoin/braid/
  filcorr parameters, derived cell sizes and eps), `supports_neg_corr`, tuning source
  (hyperopt file, CSZ best_params), pure-Python-index flag.
- Dataset profile (`abaca/dataset_profile.py`): m, n_obs, windows, pair-windows, **density at T**,
  low-frequency energy share of the first 16 DFT coefficients (cooperativeness; white-noise
  reference 2n/W printed next to it), lag-1 autocorrelation, constant-window fraction,
  coefficient of variation, source NaN fraction and regime tag from the registry.

### Phase R follow-up
StatStream precision stays 0.63-0.79 at the paper's window (W=1800, b=60) under both readings
of the approximate rule (truncated digests: recall exactly 1; renormalized digests: also 1 here).
The paper's recall < 1 says their approximation is two-sided; ours is not on pure random walks.
Unresolved without the paper's Table 2 settings; both rules are kept in the script.

### Next
User decisions: W/step policy, Sobol budget (`--points`), pilot cells. Then push, pull on Abaca,
submit pilot + Phase R.

---

## 2026-09-17 (h) -- BRAID and StatStream PDFs re-read: Phase R reproduces BRAID and StatStream;
ThinBRAID does not, with a quantitative reason. Horizon rule for W/step. 148/148.

**Branch**: dev. **Modified**: `abaca/reproduce_papers.py` (BRAID whole-sequence regime, Eq. 32,
Definition 1 with a +-16 neighbourhood; StatStream per-basic-window post-processing rule),
`datasets/fetch/gen_braid_synthetic.py` (Sines pairs with the same spectrum and new phases,
frequencies in 2..32 cycles, no noise, |R(0)| < 0.35 as in Fig. 11a; SpikeTrains lags planted),
`library_corrtrack_parallel.py` (`Candidates_BF_BRAID(thin_d_min=0)`, paper default; diagnostic knob),
`abaca/campaign_competitors.py` (Motes 2880/288, corrjoin_stock 60/5; horizon rule in the docstring),
implementation plan section 3a table. Cumulative patch refreshed.

### What the PDFs settled
- **BRAID Eq. 32**: E = 100 |l_b - l_n| / l_n, relative to the *naive* lag; but the CCF is over the
  whole sequence (n = 32,768 to 100,000, max lag n/2) and the reference lags are 567 to 4,160.
  The morning's sliding-window test bed (W = 1,024, lags <= 256) was the wrong regime; the
  normalization was right. Sines pairs have the *same power spectrum* and differ in phase (Fig. 15
  caption), so the lag is emergent; Sunspots are contiguous 25,900-day chunks; Motes pairs are
  #1/#10 (202 min) and #47/#48 (224 min), section 6.5.
- **StatStream**: sliding window 1 h at 1 s, basic windows 0.5 to several minutes; the grid uses
  n = 16 sliding-window coefficients, but the post-processing approximation is section 3.4's curve
  fitting with **the first 2 DFT coefficients of each basic window** (Fig. 4 caption), summed over
  the k basic windows. Table 2's only synthetic cell: S0.85, t = 0.0005, precision 0.9931, recall 1.0.

### Phase R results (local, small m)
| paper | ours | paper |
|---|---|---|
| BRAID Sines (4 pairs) | E = 0.000, 0.000, 0.120, 0.612% | 0.000% |
| BRAID SpikeTrains | 0.135, 0.450% | 0.387% |
| BRAID Sunspots, Motes | 0.000% (BRAID = naive at the earliest local max) | 1.038%; 202/224 min lags |
| ThinBRAID Sines | **41 to 91%**, estimates on level boundaries | 1.397% |
| StatStream S0.85 t=0.0005 | precision 0.9938, recall 0.9992 | 0.9931, 1.0 |
| StatStream T=0.9 t=0.001 | 0.9739 / 1.0 | real cells 0.9765 / 1.0 |
| StatStream grid pruning power | 0.0079 to 0.0233 | 0.01 to 0.09 |
| StatStream filter precision (16 coeffs) | 0.65 to 0.75 | ~0.55 to 0.9 |

Two things had to be fixed on our side before BRAID reproduced, neither in the port itself: the
naive Definition 1 must dominate a +-b neighbourhood (the exact CCF of finite data has wiggles of
order 1/sqrt(n) that otherwise create an "earliest" maximum on the shoulder of the peak: naive 634
vs true peak 668 with |R| differing by 0.002), and the Sines pairs must start below gamma at lag 0
as the paper's do, otherwise every wiggle above 0.4 is an earliest maximum.

**ThinBRAID (open, disclosed).** Eq. 24 recovers Sxy from ||Px - Py||^2 with d = 400/2^h. The JL
error per knot is about sqrt(2/d) (1 - rho): 0.086 measured at d = 400 for rho = -0.44 (6 seeds),
0.13 at d = 100, unusable at d <= 12 (levels >= 5, i.e. lags >= 512 with b = 16). Independent
random matrices per level make the assembled CCF jump at level boundaries, which is where the
estimates land (31, 66, 262, 491). On a sine CCF of period ~5,000 a 0.02 error in R moves the
argmax by ~250 lags, so even d = 2,000 at every level (`thin_d_min`) does not recover the naive
lag. Table IV's 0.1 to 1.4% cannot follow from Eq. 24 with d = 400/2^h as written; the authors'
ThinBRAID must differ from the text (shared projections across levels, or far larger d). The port
implements the text. ThinBRAID stays in the campaign as the O(k)-memory arm with this caveat; its
lag accuracy is not quoted as the paper's.

**Motes.** In our epoch-aligned file the pairs #1/#10 and #47/#48 peak at lags 1 to 5 epochs with
R(0) ~ 0.9; the paper's Fig. 22 shows R(0) ~ 0.2 and a peak at ~-400 epochs. The Intel Lab file's
epoch counters are per mote; a time alignment that ignores this produces exactly such offsets. The
202/224 min lags are therefore not reproducible from time-aligned data and are most plausibly an
alignment artefact; the Motes lag cells in the campaign keep their value as a real multi-series
sensor set, not as a lag ground truth.

### Decisions recorded from the user (2026-09-17)
- Campaign W/step: option 1 (application horizons) for the campaign, authors' settings in Phase R,
  and a W-robustness check (two datasets, W in {1/2, 1, 2}x at W/step = 10) instead of a W sweep.
  **Horizon rule**, one horizon per sampling regime shared by every dataset in the regime: daily
  finance a quarter / a week (sp500, acwi, corrjoin_stock); daily climate a season (Berkeley);
  daily hydrology and page views a month (streamflow, wikipedia, global_weather); hourly weather a
  week (USCRN); sub-daily sensors a day (smartmeter 48 half-hours, Motes 2,880 epochs of 31 s);
  seismic FilCorr's 20 s / 1 s (no application cycle, the one authors' setting); unitless
  synthetic the source paper's (CSZ 256/32, BRAID 1024/128, CorrJoin 240/24). L_max from the
  physical lag scale where known, 2 windows otherwise.
- Sobol: `--points k` = k joint (m, L) points per dataset, not a k x k grid, plus the synchronous
  anchor at m_max; T swept in full at every point. 4 points -> 385 cells, 2,310 jobs.
- Cythonize the three competitor indexes' hot loops (StatStream first) before wall-clock is quoted.

### Next
Cython ports: `competitor_kernels.pyx` with (1) StatStream 3^h neighbour probe + DFT distance
filter, (2) ParCorr/CSZ postings lookup and vote, (3) CorrJoin bucket grid + eps_1 ball + eps_2
filter; the Python classes keep their interface and call the kernels; parity tests against the
Python paths; then re-run the Motes N-way to measure the per-candidate time change. Then the
W-robustness cells, the pilot, Phase R at scale on Abaca.

---

## 2026-09-17 (i) -- Sobol design split into synchronous and lagged parts; six ASOS sets added.

**Modified**: `abaca/campaign_competitors.py` (`design_points`: anchor + `--sync-points` 1-D Sobol in
log m at L=1 + `--lag-points` 2-D Sobol in (log m, L in [2, L_max]); `--design` prints the table;
six ASOS DatasetSpecs), new `experiment_dataset_{br_air_temperature_146_1, br_wind_direction_146_1,
br_flights_109_1, fr_wind_speed_121_1}.py` (the fr air_temperature / wind_direction 121_1 configs
already existed). 25 datasets, 557 cells, 3,342 jobs at the defaults (2 + 3 points).

Why the split: the first draft drew L uniformly in [1, L_max] and, for the datasets with a small
L_max, rounding put three of five cells at L = 1; the lagged comparison was under-sampled while the
synchronous one was oversampled. Now every dataset gets 3 synchronous cells (m scaling at L = 1)
and 3 lagged cells with L >= 2; synchronous-only datasets (CorrJoin files, L_max = 1) get the 3
synchronous cells only. The same unit-cube sequence is used for every dataset, so the relative
design is identical across datasets. `--design` output is in the reply of 2026-09-17 and reproducible.

ASOS regime: hourly weather, a week / 12 h, L_max = 5 (48 h), the 2026-09-12 convention;
br_flights (hourly flight counts) is placed in the same hourly regime and said so. ASOS configs keep
`N_YEARS` semantics (last year of data), so the campaign passes no `--n-obs` for them.

---

## 2026-09-17 (j) -- Design made legible: Latin-square (m, L) levels replace the Sobol draw;
full-m anchors for the CorrJoin files; ASOS placeholder note.

User feedback: the Sobol (m, L) points "show no pattern and seem arbitrary", and several datasets
have far more series than the m shown. Changes in `abaca/campaign_competitors.py`:
- `design_points(kind="latin")` (default): m levels m_max / 2^k (k = 0..3, floored at m_min), L levels
  evenly spaced integers in [1, L_max] (at most 4), paired by a Latin square with 2 replicates so
  every m level and every L level appears exactly once per replicate; replicate 0 is the diagonal
  and holds the synchronous anchor (m_max, 1). `--design` prints the table; `--design-kind sobol`
  keeps the old draw. 725 cells x 6 jobs = 4,350 at the defaults; `--replicates 1` halves it.
- m_max stays at the 2,000 cap decided 2026-09-16 (battery target; pure-Python competitor indexes).
  Datasets above the cap now all carry a full-m synchronous anchor for the exact arms + StatStream:
  Berkeley Earth 18,520, CorrJoin stock 3,878, chlorine 4,830, gas 5,120, random 5,000. The cap
  is to be revisited once the Cython ports land.
- ASOS: the six entries are placeholders at the current France/Brazil files; the full-body ASOS
  fetch (199 countries) is running in another session and the specs will be regenerated from it.

---

## 2026-09-17 (k) -- Cython hot loops for the three competitor indexes (`competitor_kernels.pyx`).
Identical pair sets and counters against the Python paths; 46x to 149x faster candidate stages. 149/149.

**New**: `competitor_kernels.pyx` (+ `setup_cython.py` entry): `cell_keys` (exact mixed-radix packing
of integer cell coordinates, base 4096, offset 2048, <= 5 dims), `statstream_probe` (3^h neighbour
probe, +X / -X cells, per-(entry, sign) stamp dedupe, DFT distance filter), `parcorr_probe`
(n_grids sorted-key grids, same-cell or 3^k probe with the CSZ cell-size ball, one vote per grid via
a (query, grid) stamp, required_hits threshold), `corrjoin_double_filter` (kb-dim grid, 27
neighbourhood, exact eps_1 ball, eps_2 test with early exit). All loops are `nogil` over flat
arrays; posting lists are contiguous ranges of the alive entries sorted by key (binary search), so
no hash tables or Python objects inside the loop; output buffers are sized by a first pass that
sums the probed range lengths (an upper bound on touched).
**Modified**: `library_corrtrack_parallel.py` (`_HAVE_COMPETITOR_KERNELS`; `use_cython=None`
kwarg on `StatStreamGridIndex`, `ParCorrGridIndex`, `CorrJoinDoubleFilterIndex`: defaults to the
kernel when built, the Python postings path stays as the reference; `index_tier` in `last_stats`;
ParCorr keeps exact cell coordinates `_cellc` next to its hashed keys), `abaca/nway_compare.py`
(the "py" label now means the extension is missing), tests (+1 parity test: StatStream signed and
unsigned, ParCorr and CSZ, CorrJoin; pair sets and touched / distance-check counters equal).

### Measurements
Parity scripts (scratchpad), pair sets and counters identical in every case:
| index | python | cython | speedup |
|---|---|---|---|
| StatStream signed, m=300, 6 steps, 3 lag windows | 8.09 s | 0.068 s | 118x |
| StatStream unsigned | | | 149x |
| ParCorr (same cell), m=400 | 2.14 s | 0.046 s | 46x |
| CSZ (3^k probe + ball), m=400 | 120.5 s | 1.01 s | 119x |
| CorrJoin, m=1500, one window | 0.96 s | 0.018 s | 53x |

Motes temperature N-way (W=96, step=12, T=0.9, tuned knobs, same run as entry (e)); counts and
recall unchanged, candidate-stage time:
| arm | cand_time before | after | wall before | after |
|---|---|---|---|---|
| corrtrack (Cython throughout) | 0.187 | 0.141 | 0.48 | 0.35 |
| parcorr | 2.632 | 0.339 | 2.98 | 0.62 |
| csz | 28.355 | 0.318 | 27.73 | 0.59 |
| statstream | 0.891 | 0.171 | 1.25 | 0.48 |
| corrjoin | 1.630 | 0.277 | 2.02 | 0.60 |
The competitor candidate stages are now within 1.2x to 2.4x of CorrTrack's on this cell, and the
remaining gap is dominated by the per-batch key sort and numpy bookkeeping around the kernel,
not by Python loops. Wall-clock comparisons are quotable once the pilot confirms this at m = 2k.

### Known issues
- Kernels assume cell coordinates within +-2048 (offset) and <= 5 grid dims: true for every
  configuration in the campaign (bounded feature cubes, index_dims <= 4, k <= 4, kb <= 4); a
  `ValueError` is raised for more dims.
- The per-batch `argsort` of alive entries is O(N log N) per step; at m = 2k with several lag
  windows N ~ 10k, negligible against the probe work.
- Abaca's job scripts rebuild all extensions per node (`setup_cython.py build_ext --inplace`), so
  the new module needs no extra step there.

### Next
Pilot on Abaca after the user's push; the W-robustness cells; Phase R at scale.

---

## 2026-09-17 (l) -- Design: one work ladder (m and L grow together) replaces the Latin square;
full-m anchors run every arm; cost estimate for the m cap.

User: L on its own mostly multiplies the pair count (the 2026-09-16 sweeps showed little L effect
beyond that), so m and L should grow together and replicates are unnecessary. `design_points(kind=
"ladder")` (default): synchronous anchor (m_max, 1) plus rungs (m_max/8, 1), (m_max/4, ~L_max/3),
(m_max/2, ~2 L_max/3), (m_max, L_max); work ~ m^2 L grows 4x to 8x per rung. 25 datasets, 505 cells,
3,030 jobs. Latin and Sobol kept as `--design-kind` options.

Cost model (k_bf = 1.058e-7 s per m^2 L n_steps unit, the bruteforce calibration of log 2026-09-16 (q)):
bruteforce over all 505 cells = 15.3 h, of which Berkeley Earth's four 18,520-series anchors are
10 h; all arms + hyperopt + tuning ~ 7x to 15x that = 107 to 229 core-hours, i.e. a few days on a
handful of nodes. Per cell at the cap: m = 2,000 -> 1 to 2 min bruteforce synchronous, 3 to 7 min
at L = 4; m = 3,000 -> 2 to 4 / 6 to 16 min; m = 5,000 -> 4 to 11 / 17 to 44 min. **Decision
(user): ladder capped at 2,000; the datasets above it keep their full-m synchronous anchors
(CorrJoin 3,878 to 5,120; Berkeley 18,520), now for every arm except plain BRAID (per-pair state),
since the pruning arms' candidate loops are Cython.** More cells above 2k can be added later.

---

## 2026-09-17 (m) -- m cap 5,000; per-step latency quantiles; CorrTrack ablation runner; global
ASOS wired; Phase R submitted on Abaca (OAR 3117155). 149/149.

**Modified**: `library_corrtrack_parallel.py` (`execute_corrtrack_pass` times every `run`/`run_bf`
call; `n_steps`, `step_time_{p50,p90,p99,max,mean}` in the record and in RUN/OPTIM_RESULT_COLUMNS,
steady-state steps only, i.e. after the first full window), `abaca/nway_compare.py` (step quantiles
in the table and JSON), `datasets/competitor_loader.py` (`last_obs`, `min_coverage`: trailing span
and coverage-ordered station selection), `abaca/campaign_competitors.py` (m_max 5,000; global ASOS
specs; synthetic families regenerated at m = 5,000; plain BRAID excluded above m = 2,000 when
L > 1; `-q abaca` in the emitted `oarsub`), four `.oar` wrappers (`#OAR -q abaca`; a bare `oarsub`
reported "not enough resources"), new `abaca/ablation_corrtrack.py`, new configs
`experiment_dataset_global_asos_{air_temperature,wind_speed,relative_humidity,pressure}.py` and
`experiment_dataset_{statstream_rw_m5000_T20000,braid_sines_m5000_T32768,braid_spiketrains_m5000_T100000}.py`.

### Decisions (user)
- **m_max = 5,000.** Ladder rungs at 625 / 1,250 / 2,500 / 5,000 for the large sets; Berkeley keeps the
  18,520 anchor; corrjoin_gas capped at 5,000 of 5,120 (disclosed). 29 datasets, 569 cells, 3,414 jobs.
- **Per-step latency** for the online argument: quantiles of the wall time of each step's method call
  (sketch + candidates + validation + monitor, artifact I/O excluded), steady state only. Motes m=12
  check: bruteforce p50 0.09 ms / p99 0.16 ms, StatStream 0.85 / 1.24 ms.
- **Ablation** (`abaca/ablation_corrtrack.py`): from the hyperopt configuration, one component at a
  time: no gamma gate, no Hamming gate, both off, exact-Hamming index instead of banded LSH, hybrid
  validation toggled, n_vectors halved / doubled, bruteforce reference. Same span, same ground
  truth, recall + counters + times + step quantiles. Smoke on Motes m=12 runs; effects at that size
  are within noise, the campaign cells are where it means something.
- **Sensitivity**: m, L, T from the design; W from the robustness check (to add); n_vectors, gamma,
  occupancy from the hyperopt tables (`corrtrack_optim_*` outputs); the ablation covers the gates.

### Global ASOS
Companion session's fetch: 181 country files (7.4 GB) on Abaca, pivoted to
`tmp_artifacts/global_asos/global_{air_temperature,wind_speed,relative_humidity,pressure}.npz`,
3,586 to 3,615 stations x 137,688 hours (2011-01-01 to 2026-09-15), 72% missing overall; AU was
still being bisected at 23:26, so the pivot may be regenerated. Coverage over the last 2 years:
>= 90% for 661 to 675 stations, >= 80% for ~745. Campaign entries: last 17,520 h, `min_coverage=0.9`,
best-covered 600 stations, hourly regime (168/12, L_max 5). The six country-level ASOS specs stay
(they are the 2026-09-12 datasets).

### Abaca
`git pull` to 53f9600 done on the frontend; **Phase R submitted**: `oarsub -q abaca -p mercantour3
-l host=1,walltime=24:00:00 -S "./abaca/reproduce_papers.oar EXPERIMENT=all"` -> OAR_JOB_ID 3117155
(queue p1). The pilot waits for the push of this entry's changes (step quantiles, 5k design).

---

## 2026-09-18 (a) -- Phase R results from Abaca (job 3117155, 2 h 25 min); boxplot ticks for the
step latency; ablation as a phase ladder with a new `all_pairs` backend. 150/150.

**Modified**: `library_corrtrack_parallel.py` (step-time record fields are now the boxplot ticks:
`step_time_{min,q1,median,q3,max,whisker_lo,whisker_hi,outliers,mean}` over steady-state steps,
artifact I/O excluded; new `AllPairsGateIndex` and `candidate_backend="all_pairs"`: sketch computed,
no index, the sign-Hamming and dot >= gamma gates switchable, loop in
`competitor_kernels.all_pairs_gates`), `abaca/ablation_corrtrack.py` (the ladder: bruteforce,
sketch_only, sketch_hamming, sketch_dot, sketch_both, lsh_hamming, lsh_dot, lsh_both; one fixed
parameter set for all rungs), `abaca/nway_compare.py` (ticks in the table), `abaca/reproduce_papers.py`
(Yellowstone at the paper's 10 s lag), `abaca/reproduce_papers.oar` (SpikeTrains at the paper's lag
scale), tests (+1: sketch_only == bruteforce universe, gates only shrink, tick ordering), plan
section 3a table filled with the Abaca numbers. Cumulative library patch: superseded by git history
since 25fbe9a; the file is left empty and will be removed.

### Phase R (plan section 3a has the full table)
- StatStream **reproduces** at m = 500 (Table 2 S0.85: 0.9936/0.9992 vs 0.9931/1.0; pruning power and
  filter precision inside Fig. 5).
- CorrJoin **reproduces qualitatively** (r1 and 1/r1 across T on all four files; gas at 26% correlated
  has a 1.1x ceiling, their Fig. 15 statement).
- BRAID **reproduces** on Sines at scale (32 pairs, median E 0.06%, max 1.34%); SpikeTrains only when
  generated at the paper's lag scale (local 0.135/0.450%; the Abaca row used lags 17 to 127 by mistake
  and is rerun). ThinBRAID open, as before.
- ParCorr **does not reproduce at its stated settings** (recall 13 to 38% vs > 90%): the cell size the
  paper omits decides it; the CSZ tuning reaches 0.94+. This is the finding to write up.
- FilCorr: throughput trend reproduces (time ratio 0.4x -> 4.8x from m = 25 to 200), magnitude below
  the paper's 4x sensors at our m and against our stronger baseline. Yellowstone rerun at 10 s lag.
- TSUBASA: exact; 22% faster than our incremental all-pairs (their 10x is vs raw recompute).

### Ablation ladder (user's design)
`all_pairs` backend added so that "sketch + gates, no index" is expressible; `brute_force` skips the
sketch and could not serve. Motes m=20 smoke, untuned n_vectors=32, T=0.9, lags 24, neg_corr:
sketch_only recall 1.0 at 222,540 candidates (= bruteforce), +Hamming 0.999 at 172,183, +dot 0.75 at
68,279, LSH+both 0.70 at 64,222. The dot gate at the untuned gamma is what costs recall here, which
is the kind of statement the ladder exists to make; the campaign runs it with each cell's hyperopt
parameters.

### Next
User's push; then on Abaca: rerun Phase R BRAID SpikeTrains + FilCorr Yellowstone
(`EXPERIMENT=braid`, `EXPERIMENT=filcorr`), the 3-cell pilot, ablation `.oar`, W-robustness cells.

---

## 2026-09-18 (b) -- Ablation always on the cell's tuned parameters; ParCorr reproduced through its
own calibration protocol; CSZ protocol reproduction added to Phase R.

**Modified**: `abaca/ablation_corrtrack.py` (`--best-params` required; no untuned fallback),
`abaca/reproduce_papers.py` (`parcorr`: row (a) stated settings, row (b) the paper's section 5.3
calibration of the sample-dependent quantities, c and f, on the first 30% of the stream at the
0.95 target, held-out recall on the rest; new `csz`: the 130-row design at the 0.99 target on
stand-ins; `_fit_window` halves W until the calibration span holds >= 20 windows, after an
sp500 run with W = 500 on a 376-row span produced a 0/0 recall that made every setting feasible;
`_calibrate_grid` refuses a span with no correlated pair), `abaca/reproduce_papers.oar` (csz).

sp500_sub263, m = 120, W = 125 / b = 5 (the stream is too short for w = 500):
| T | (a) stated r=60, k=2, f=0.7, c=0.7 | (b) calibrated c, f (calib recall) -> held-out | paper |
|---|---|---|---|
| 0.7 | 19.2% | c=1.3 f=0.5 (0.898) -> 87.9% | > 90% |
| 0.8 | 28.3% | c=0.7 f=0.5 (0.956) -> 95.0% | > 96% |
| 0.9 | 41.3% | c=0.5 f=0.5 (0.967) -> 96.4% | > 95.7% |
So ParCorr's numbers are reproducible **through its protocol** (calibrate on a sample) and not from
its printed parameters: the cell size the paper omits is the parameter that matters. The c grid is
extended to 2.0 for the Abaca rerun (T = 0.7 hit the CSZ grid's edge at 1.3).

---

## 2026-09-18 (c) -- `candidate_precision` recorded; correction on what "our bruteforce" is.

- `candidate_precision = correlated / total_candidates` (precision BEFORE validation: StatStream's
  Fig. 5 precision, the complement of CorrJoin's r1; for an all-pairs arm the correlation density at T)
  is now an explicit run-record column and the N-way table's `cand_prec` column. It was derivable from
  the counters before; the user asked for it as a tracked metric for the method analysis.
- Correction to (a): the Phase R speed baselines are the `bruteforce` arm, which **recomputes** all-pairs
  Pearson from raw values every step (Cython-vectorized `validate_corr_rows`), i.e. the papers' own
  "naive" definition, implemented fast; it is NOT the incremental `exact_stomp` arm. Hence TSUBASA at
  0.78x of it is 1.28x faster than a vectorized naive where its paper reports >= 10x over a per-pair
  naive: the gap is the naive's implementation tier, and the campaign's `exact_stomp` column is the
  stricter comparison. FilCorr's 4.8x time ratio at m = 200 is likewise against the vectorized naive.
  Plan section 3a table corrected accordingly.
- ThinBRAID: user accepted option (i): disclosed open discrepancy; ThinBRAID is compared as the
  O(k)-memory variant only, its lag accuracy not quoted as the paper's.

---

## 2026-09-18 (d) -- candidate-stage specificity tracked (user's request).

`abaca/nway_compare.py` and `abaca/ablation_corrtrack.py`: for every arm, against the bruteforce run of
the same cell, `candidate_specificity = 1 - FP / (U - P)` with U = the bruteforce pair-window universe,
P = its correlated pair-windows, FP = the arm's candidates minus its true positives (validation is exact,
so the arm's `correlated` is its TP). Also stored: `universe_pair_windows`, `positives`,
`candidate_false_positives`, `candidate_fpr`. Table column `cand_spec`. All-pairs arms have specificity 0
by construction (every uncorrelated pair-window is a candidate); on the Motes smoke CorrTrack 0.9945,
ParCorr 0.9943, StatStream 0.9717. Together with recall (sensitivity) and candidate_precision this is
the full confusion matrix of the candidate stage per cell. The per-run CSV cannot hold specificity on
its own (it needs the universe), so it lives in the N-way / ablation JSON, next to the counters.

---

## 2026-09-18 (e) -- The corrtrack arm refuses to run untuned.

`abaca/nway_compare.py`: the `corrtrack` arm now requires `--best-params`; `--allow-untuned` runs it with
defaults for a plumbing smoke only and labels the output "UNTUNED (smoke, not quotable)". Arms other
than corrtrack are unaffected. `abaca/nway_compare.oar` exits 3 when no `best_params_corrtrack.json`
is found under HYPEROPT_DIR, so a failed or missing hyperopt job fails the N-way job instead of
producing a competitors-only table by accident. Same rule as `ablation_corrtrack.py` (entry (b)).
Every CorrTrack number in this session's smoke tables (entries (d), (k), (m), (a)) was untuned and
is not quotable; the tuned m=12 check (recall 0.998) was the only exception.

---

## 2026-09-18 (f) -- Synthetic data audit; density-controlled sets and the returns (preprocess) axis
added to the campaign; '+' decode bug in two OAR wrappers fixed.

### What the synthetic sets are (user's question)
- **Paper families (ours, approximations)**: StatStream random walks, the paper's formula exactly
  (`s = 100 + sum(u - 0.5)`), independent walks, no planted structure, correlation density emergent;
  BRAID Sines: 3 sine components per seed, frequencies log-uniform in 2 to 32 cycles per sequence (Fig.
  15's band), amplitudes U(0.5, 1.5), the pair partner shares the spectrum with new phases and
  |R(0)| < 0.35 (Fig. 11a), no noise; BRAID SpikeTrains: Gaussian pulses of width U(20, 60) every
  6,500 samples with 2% jitter, partner = shifted copy + white noise, lag planted; CorrJoin synthetic
  and random: the authors' own files. None of these controls density or degree: BRAID pairs are
  degree 1 by construction, walks have whatever spurious density smooth walks give at T.
- **This project's generator** (`synth_corr_gen.py`), two modes. `make_corr_dataset(z, ...)`: a fraction
  z of samples inside planted correlated template pairs; sign pos/neg/both; optional max_lag; base
  processes stationary (ar1, wn, seasonal_arima, lagged_seasonal_ar, ou) and nonstationary (rw,
  rw_seasonal_drift, trend_poly, integrated_seasonal); volatility equalizer; density in samples, not
  verified in tuples. `make_density_targeted_dataset(target_density, ...)`: the (pair, lag, window)
  tuple density at the protocol (W, step, L, T) is set and **verified by brute force** within a
  tolerance; correlation comes from groups sharing a driver, (n_groups, group_size) solved from the
  density, so the **degree (group_size - 1) is a consequence of the density, not an independent
  knob** (m=300, 2%: 32 groups of 8); persistence controls (epochs, duty, bursts); `preprocess`
  defines the density in the differenced space; base_proc as above. So: fixed, verified density yes,
  for stationary and nonstationary processes; degree not independently fixed.
- **Neither was in the campaign until today.** Added: `datasets/fetch/gen_density_targeted.py` (writes the
  npz with gt rows and the experiment config under `datasets/competitor/configs/`), campaign family
  `synth_{ar1,wn,rw}_d{0.005,0.02,0.05}_T{T}_L3` (m = 1,000, n = 20,000, W = 168 / 12, L = 3), one file
  per T because the density is defined at T; 36 files, generated on the frontend by the emitted script.

### Cooperative vs uncooperative, and preprocess
`preprocess=True` is first differencing inside the harness (`Sketches._preprocess_data`, the
validation window `window_data_diff`, and the same flag handed to every Pattern-A node), so every arm
and the bruteforce truth see the same differenced stream; a run is a different problem (returns), not
a different method. Until today every campaign cell ran `preprocess=False`, and a hyperopt file with
`preprocess=True` would have silently mismatched the raw bruteforce truth in nway. Now: `--preprocess`
on `nway_compare.py` (applied to the truth, forced onto every arm's parameters, profile computed on the
differenced data), `tune_competitors.py`, `ablation_corrtrack.py`, and `corrtrack_param_search.py`
(`PARAM_GRID["preprocess"] := [True]`, so the proxy-anchor labels are differenced too); `PREPROCESS=1`
token in the three OAR wrappers. Campaign: every dataset's synchronous anchor also runs as a `_diff`
cell (+116 cells): that is the prices-vs-returns axis of plan section 5 item 6. 721 cells, 4,326 jobs.

First look (synth rw, 2%, T = 0.9, differenced = white noise, untuned smoke): CorrTrack recall 1.0 with
candidate precision 1.0 and specificity 1.0; StatStream and CorrJoin recall 1.0 but specificity 0.15 and
0.12 (they pass 85 to 88% of the uncorrelated pair-windows). The section 6 prediction, on the first try.

### Bug
`nway_compare.oar` and `tune_competitors.oar` were missing the `EXTRA_ARGS="${EXTRA_ARGS//+/ }"`
decode line (only `hyperopt_corrtrack.oar` had it), so the campaign's `--n-obs+20000` token would
have reached the Python runners unsplit. Fixed; the pilot would have caught it.

## 2026-09-18 (g) -- Monitor phase found to dominate narrow-band FilCorr wall time; NumericMonitorState hot loop optimized ~4x (bit-identical); m=500 raw/diff + FilCorr sweep reruns launched

### What was found
Splitting the FilCorr bandwidth sweep's wall time into phases (`val_time`, `cand_time`,
`monit_time`, `bookkeeping`) showed the narrow-band slowdown is NOT correlation compute
(flat as the band narrows, ~5s) and NOT recording/bookkeeping (grows, but stays single-to-
double-digit seconds). It is `monit_time`: e.g. global_weather thr=0.70 2coef, 57.1s of 67.6s
wall; smartmeter 2coef, 243s of 292s; sp500_prices 2coef, 458s of 503s. An earlier answer
in this session attributed the slowdown to recording cost; that was wrong, corrected here.

`CORRTRACK_PROFILE=1` on `NumericMonitorState.update()` (global_weather thr=0.70, full vs
2coef): rows/step 62 -> 27,483 (443x), active episodes 62 -> 21,291 (340x), monitor time
0.17s -> 106.3s (625x), but per-row cost only 1.23 -> 1.75 us. So the monitor is linear in
rows and closeout is already O(expired) (frontier swap-remove), not O(active). The blowup is
volume (narrow-band false positives), not an algorithmic defect. The per-row constant was the
fixable part.

### Root cause of the constant
`monitor_kernels.pyx` `NumericMonitorState._find_slot` (called once per row) re-derived four
memoryviews from `self._xxx_arr` on every call; `_append_status` re-derived eight. This is the
exact hot-loop cost the 2026-07-03 pass documented and fixed for `update()`'s outer loop and the
four frontier/active helpers, but these two were missed. ~1 us/row of attribute-lookup + buffer
handshake for a ~100 ns hash probe.

### Change (monitor_kernels.pyx only; git-clean file, untouched since da0f4f7)
- `_find_slot`, `_append_status`: read-only views passed in as parameters from `update()`,
  `_rehash`, `finalize` (the output `rows` view in `_append_status` is still acquired locally,
  and must be: `_ensure_status_capacity` can reallocate it).
- `_find_slot`, `_activate_slot`, `_deactivate_slot`, `_mark_current_frontier`,
  `_remove_previous_frontier_slot`: `noexcept nogil` (Cython 3.0.8 otherwise inserts an
  exception check after every call).
- No logic change. `SkipAheadState._find_slot` (same file) left as is (not on this hot path).

### Verification
- `python3 -m pytest test_stable_reproduced_changes.py`: 150/150 pass.
- Old-vs-new equivalence (`scratchpad/mon_equiv/equiv_test.py`): pre-edit `.pyx` built from
  `git show HEAD:` as `monitor_kernels_old`; identical (rows, corrs) sequence into both, 400
  steps, all three save-flag combinations; status rows, anomaly rows, active_count and branch
  counts (new 8209 / early 14 / extend 5570 / transition 92) identical at every step and after
  `finalize()`.
- Micro-benchmark, 2coef-shaped flood (27K rows/step, 12K persistent + 15K flicker, 200 steps),
  interleaved old/new: row_loop 1.09-1.45 us/row -> 0.26-0.29 us/row (4.2-5x); closeout 3.3-3.8x;
  total 7.6-10.2s -> 1.9-2.2s (~4x). Branch counts identical.
- Cluster confirmation job (monitor_profile_diag, 3120441) submitted; result recorded below.

### Consequence for existing tables
monitor time is paid by every method, and in dense cells it dominates even for exact methods
(sp500_prices full band: monit 160s of 187s). So the m=500 raw table, the m=500 diff table and
the FilCorr bandwidth sweep all shift, non-uniformly (methods where monitor was a larger share
of their wall gain more). Per user instruction all three are being rerun on the new kernel:
76 jobs (`rerun_v2/`: 28 m500 diff + 28 m500 raw via `hamming_exact_compare_m500.py`, which
now carries the widened {0,0.05,0.10,0.15,0.20,0.25} offset grid by default, + 20
`filcorr_band_thr_sweep.py`). Pre-rerun result files archived on Abaca at
`tmp_artifacts/_pre_monitor_opt_2026-09-18/` (76 files) for a before/after comparison.
Job preamble now aborts if the old kernel is present (`grep noexcept nogil`).

### Also closed this entry: widened gamma grid selection (gext4, old-kernel timings)
54/56 cells completed before being superseded. hamming+dot picked off=0.05 in 35/54 and 0.0 in
7/54 (78% chose an offset absent from the old grid); lsh_sign_dot 32/54. off=0.0 (no margin)
meets recall>=0.95 in only 7/54 hamming cells (recall 0.68-0.97); off=0.05 meets it in 44/54 and
is the modal pick. Raw regime tolerates a tighter gate (rec@0 0.93-0.97) than diff (0.68-0.93),
consistent with 2026-09-18 (f)'s regime-shift finding that gamma is the regime-sensitive knob.
Recall side is timing-independent and stands; the picks are finalized by the rerun.

### Known issues / open
- Remaining ~0.27 us/row is a cache-missing probe across 11 separate int64 arrays (SoA). An
  AoS slot layout (one cache line per slot instead of ~8) is the next lever if monitor still
  dominates after the rerun; not done, bigger refactor, less certain payoff.
- The optimization does not make narrow-band FilCorr fast: 443x more rows is still 443x more
  rows. It removes a constant-factor tax that every method was paying.
- 18-country ASOS re-fetch still running on the Abaca frontend (unaffected, pure Python).

### Next exact step
Confirm 3120441 shows the ~4x on cluster CPUs; submit `rerun_v2/jobs/*.sh` (76); rebuild all
three tables from the new JSONs and diff against `_pre_monitor_opt_2026-09-18/`.

---

## 2026-09-18 (g) -- Fixed-degree generator option; synthetic density family on the 5k ladder;
monitoring off in the comparative campaign; preprocess default clarified. 150/150 (+ synth 9/9).

**Modified**: `synth_corr_gen.py` (`make_density_targeted_dataset(group_size=...)`: pins g = degree + 1
and lets the density choose n_groups only, `RuntimeError` when the density is unreachable at that
degree; `effective_degree` measured on the verified truth, mean distinct partners per correlated series,
returned and stored in meta), `datasets/fetch/gen_density_targeted.py` (`--degree`; file names carry m
and L; meta has `degree_by_construction`, `effective_degree`, `requested_degree`),
`abaca/campaign_competitors.py` (synthetic family: ar1 stationary + rw nonstationary, densities
{0.5, 2, 5}%, the same ladder as every dataset up to 5k, one file per rung, hourly protocol 168/12
L_max 5; 120 generated files; 805 cells, 4,830 jobs), `abaca/nway_compare.py` (`--monitor`, default
off), `abaca/ablation_corrtrack.py` (monitor off). Check: m=300 rw 2%: group_size=8 -> 32 groups,
effective degree 7.00 = by construction; group_size=21 refused by the generator's own spurious-
correlation check (large groups of random walks decorrelate poorly), group_size=3 at 5% refused as
unreachable (100 groups give 0.67%). Degree and density are coupled by density ~ coverage x degree /
(m - 1); the generator now says so instead of silently picking.

**preprocess default**: False everywhere. The `_diff` cells are additional anchors (every dataset,
every T, L = 1), so both problems are measured on the same data; extending `_diff` to the whole ladder
would add ~700 cells and is not done.

**Validation and monitoring in the arms** (user's question): Pattern-B arms (CorrTrack, ParCorr/CSZ,
StatStream, CorrJoin, the ablation backends) generate candidates and pass them through the shared
exact Cython validation (`validate_corr_rows`); the `bruteforce` arm enumerates all pairs and validates
them through the same kernel; `exact_stomp`, `filcorr`, `tsubasa`, `braid` compute the exact Pearson
inside their own incremental kernels (five sums / band-limited FFT / segment sketches / smoothed sums)
and hand accepted rows to the harness (`validation_time` = their combination time; no second
validation). All arms then run the same monitoring step when `monitor=True` (`_monitor_corr` after
the correlated set of the step). Decision (user): **monitor off for every arm in the comparative
campaign**; monitoring becomes CorrTrack's own experiment (next entry).

---

## 2026-09-18 (h) -- Realistic densities; every cell in both spaces (raw and differenced); spurious
correlation kept and recorded in the raw nonstationary synthetic files; exactness-as-published note.

**Modified**: `synth_corr_gen.py` (`allow_spurious`: keeps a file whose base process produces
unplanted correlated tuples, verified density then above target, `spurious_fraction` recorded),
`datasets/fetch/gen_density_targeted.py` (`--allow-spurious`; note that the campaign's bruteforce is
the operative truth, the file's gt_rows are the planted tuples), `abaca/campaign_competitors.py`
(densities {0.5, 2, 5, 20}%; every cell duplicated with preprocess=True; synthetic files generated in
both spaces, raw with `--allow-spurious`, differenced with `--preprocess`), plan section 5 items 6/7.
**Size**: 1,457 cells, 8,742 jobs (729 raw + 728 differenced), 320 generated synthetic files.

### Decisions (user)
- **Densities that match real applications.** Measured so far at T in {0.7..0.95}: CorrJoin stock 0.09
  to 9.7%, chlorine 0.5 to 6.5%, synthetic walk 0.05 to 8.4%, sp500 daily ~1% at 0.9, gas 8 to 26%,
  fr_air_temperature ~17.6% at 0.7, Motes temperature 45% at 0.9. Synthetic densities {0.5, 2, 5, 20}%
  span that range; the dataset profile's `density_at_threshold` will refine it after the pilot.
- **preprocess: both, for every cell, neither as "the" default.** The user's two concerns are both
  right: raw nonstationary data carries spurious correlation (the generator's own check refuses raw
  random walks at 5% / T = 0.9 for exactly that: 6.5 to 7.9% unplanted tuples), which is why finance
  works on returns and why a differenced primary is defensible; and a robustness-to-nonstationarity
  claim cannot rest on differenced data alone. So the paper reports both throughout, headlines the
  domain-conventional space per regime (returns for finance, levels for sensors and climate), and uses
  the other as the robustness result. The raw synthetic random-walk files keep their spurious tuples
  (the truth is bruteforce's, spurious included); the differenced files define the density where they
  run. Degree is reported (`effective_degree`), not fixed, in the campaign; `--degree` exists.
- **Exactness as published** recorded in plan section 5 item 7: StatStream's and BRAID's outputs are
  approximate as published; the `candidate_precision` column is the false-positive rate a monitor
  would inherit from an unvalidated candidate stream. To be stated next to the monitoring experiment.

### Cost
Roughly twice entry (l): ~30 bruteforce-hours over the cells, ~200 to 450 core-hours with every arm
plus hyperopt and tuning; the synthetic 5k rungs add a few hours each. Submission per dataset in batches.

---

## 2026-09-18 (i) -- NumericMonitorState rewritten as array-of-structs with prefetch and pointer appenders: ~10x on the kernel, 6.3x on library monit_time, bit-identical; FilCorr sweep gains a monitoring-off arm

### Why
After entry (g) the monitor phase was still the largest single cost of the narrow-band FilCorr
cells (global_weather thr 0.70, 2coef: monit 9.6s of 20.3s wall, from 57.1s of 67.6s before (g)).
The user asked whether a smarter monitoring algorithm could make this phase faster still. Per-row
cost was ~0.25 us on the 27K rows/step flood, far above a hash probe, so the remaining cost was
memory traffic and call overhead, not the algorithm.

### What was found
- Slot state lived in 14 parallel int64/uint8 numpy arrays (SoA). `_ensure_hash_capacity` keeps
  capacity >= 2 x (size + rows per step), so the 2coef cell runs a 262K-slot table: ~29 MB of
  arrays, each row touching ~13 separate cache lines outside L2.
- `_append_status` / `_append_anomaly` re-acquired a memoryview from `self._status_rows_arr` /
  `self._anomaly_rows_arr` on every call (the same pattern entry (g) removed from `_find_slot`);
  they run on every new activation, every transition and every closeout, i.e. ~50% of rows on
  the flood plus every expired episode. In the SoA version each `_append_status` call also
  passed nine memoryview structs by value.
- `_seen_step_arr` was written on every row and never read (it now lives in the struct, so the
  write is free and nothing observable changes).

### Change (`monitor_kernels.pyx`, NumericMonitorState only; SkipAheadState untouched)
- One 64-byte `MonitorSlot` struct per slot (t1, t2 int64; s1, s2, lag, window, length, seen,
  queued_step, active_pos, frontier_pos int32; sign int8; occupied, active uint8; padding),
  64-byte aligned, allocated from a uint8 numpy buffer kept alive on the object. Same
  `_monitor_hash_key`, same linear probing, same insertion order in `_rehash`, same
  `_maybe_compact` policy (comment kept verbatim), same branch structure and output order.
- `update()` prefetches the home slot of row i+16 (`__builtin_prefetch`, `prefetch_dist`
  constructor kwarg, default 16, 0 disables) while processing row i.
- `_append_status` / `_append_anomaly` are `inline` and write through cached `int64_t*`
  pointers refreshed only when `_ensure_*_capacity` reallocates.
- int32 narrowing is guarded, not silent: `update()` scans the incoming rows (sequential, 5
  loads per row) and raises `ValueError` if any series id, time index, window size, or the step
  count reaches 2**30.
- The row loop stays under the GIL (single-threaded; a `with gil` re-acquisition per appender
  call would cost more than the append).

### Verification
- `mon_equiv/equiv_test.py` (400 synthetic steps, all three save_status/save_anomalies
  combinations, every branch exercised): status rows, anomaly rows, active counts and branch
  counts identical to both the pre-(g) kernel and the (g) kernel, for the scratchpad prototype
  and for the repo build.
- ctypes mirror of the struct: sizeof 64. Guard test: a 2**30 time index raises, a normal
  update passes.
- `python3 -m pytest test_stable_reproduced_changes.py -q`: 150 passed.
- End to end through the library (`mon_equiv/e2e_check.py`, global_weather m=100, T capped to
  1500, 491 steps, FilCorr 2coef and full band, `_cy_numeric_monitor_state_cls` swapped between
  the (g) kernel and the new one): 4,424,777 status rows, 6,827,299 anomaly rows, 13,185,287
  correlated rows, branch counts, all identical. monit_time 7.17s -> 1.14s (6.3x); wall
  9.21s -> 3.08s. Full band: 0.02s -> 0.01s, identical.
- Micro-bench (`mon_equiv/bench4.py`, 27K rows/step x 200 steps, -O3 -march=native for every
  variant), row loop us/row and closeout: pre-(g) 1.05 / 1.60s; (g) 0.25-0.29 / 0.50s; (g) plus
  pointer appenders only 0.16-0.19 / 0.18s; AoS without prefetch 0.027 / 0.02s; AoS with
  prefetch 16 0.024 / 0.02s. The two levers compound (2x and 2.1x alone, 10x together).
  Prefetch distance 8/16/32 within noise on the laptop (table mostly L3-resident there).
- Cluster (job 3120771, `aos_verify_job.sh`, full T=5113, 1695 steps, FilCorr 2coef):
  15,511,094 status rows, 23,978,907 anomaly rows, 46,336,553 correlated rows and branch counts
  identical between the (g) kernel and the new one on the same node; monit_time 30.10s -> 3.82s
  (7.9x), wall 41.06s -> 14.25s. Full band: 0.06s -> 0.04s, identical. `monitor_profile_diag`:
  row_loop 2.62s for 46.3M rows (~57 ns/row on the cluster CPU), closeout 0.99s, swap 0.11s.
  Test suite on the node: 148/149; the one failure is
  `test_stable_csv_schemas_drop_fixed_and_bootstrap_columns`, which asserts
  `config_folder().startswith("corrtrack_release_dev/")` and trips because the isolated job copy
  lives at `/tmp/corrtrack_iso_<jobid>/`; unrelated to the kernel (job 3120772).

### Also this entry: monitoring-off arm of the FilCorr band-width sweep
`filcorr_band_thr_sweep.py` takes an optional third argument (`false` = monitoring off for BF,
FilCorr and both CorrTrack backends; `_meta.monitor` records it; output suffix `_nomon`).
20 jobs (`fcnm_*`, same isolated-repo preamble) submitted; 10 finished within the hour. These
results do not depend on the monitor kernel at all.

### Consequence for the running tables
The 76-cell rerun launched in entry (g) uses the (g) kernel (each job rsyncs its own copy at
start, so syncing the new kernel to Abaca did not touch them). Monitor time is a smaller share
now than in (g) but still not zero in dense cells, so the m=500 raw/diff tables and the
monitored FilCorr sweep will shift again on this kernel. Whether to rerun them a third time is
the user's call; the (g) results are internally consistent and archived.

### Known issues / open
- The remaining ~24 ns/row is close to the floor for a random hash probe plus an
  unpredictable four-way branch; the next lever would be algorithmic (fewer rows reaching the
  monitor), not layout.
- `SkipAheadState` still uses the SoA layout and per-call view acquisition; not on any measured
  hot path, left alone.
- The 2**30 limit is new and explicit; no dataset in this project comes within three orders of
  magnitude of it.

### Next exact step
Cluster verification done (above). When the 76-cell rerun and the 20 `fcnm_*`
cells are done, build the three tables plus the no-monitor sweep table.

---

## 2026-09-18 (i) -- Densities {0.5, 2, 5, 10, 20}%; StatStream reports with its own approximate
rule (designed output) instead of borrowing the exact validation. 151/151.

**Modified**: `library_corrtrack_parallel.py` (`CorrTrack(statstream_report="approx"|"exact",
statstream_tolerance=0.0005, statstream_bw_coeffs=2)`; `_statstream_digests` computes per-basic-window
DFT digests and exact block sums for the whole buffer once per step, `_validate_numeric_rows_
statstream_approx` reports survivors with corr_approx > T - t from those digests, unaligned rows fall
back to a direct digest; record columns `statstream_report/tolerance/bw_coeffs`),
`abaca/campaign_competitors.py` (densities), tests (+1; the grid tests pin `statstream_report="exact"`
since they probe Theorem 2), plan section 5 item 7 rewritten.

Which arms have a validation stage by design (user's question): ParCorr, CSZ, CorrJoin verify every
candidate exactly (their method; the shared kernel only makes that verification as fast as ours);
TSUBASA and FilCorr are exact by construction; BRAID / ThinBRAID report approximate CCF values
without validation, and so does our port; StatStream reports an approximation from digests, and
until today our port validated exactly instead. Now it does what the paper does. Random walks m=200,
W=3600/60, T=0.85: approx t=0.0005 -> recall 0.9991, precision 0.9925 (Table 2: 1.0 / 0.9931); t=0.001
-> 0.9999 / 0.9842; exact mode 1.0 / 1.0. Cost: the digest path is numpy (0.39 s vs 0.09 s for the
exact Cython kernel on this cell, the per-step digest recompute over the buffer dominating); it is
StatStream's own step so it is charged to its `val_time`, and the tier gap is stated.

Campaign: 1,537 cells, 9,222 jobs.

### Follow-up (user decision): third rerun on the AoS kernel, Sweden retry
User chose to rerun the 76 monitored cells on the (i) kernel. (g)-kernel results archived on
Abaca at `tmp_artifacts/_g_kernel_2026-09-18/` (the six cells still running on the (g) kernel
are re-archived by `rerun_v3_jobs/defer_launch.sh` when they finish). `rerun_v3/jobs` = the
`rerun_v2` set with names `m5v3_*`/`fcv3_*` and the preamble guard changed to
`grep -q MonitorSlot monitor_kernels.pyx`; 70 submitted immediately, the six deferred ones
(`fcv3_sp500_pr_07`, `m5v3_acwi_cap_07/08_raw`, `m5v3_sp500_07/08_raw`, `m5v3_streamfl_07_raw`)
are submitted by the frontend watcher once their v2 counterpart reports done, so the shared
output files never race. Sweden: every SE__ASOS request in the 18-country re-fetch died at
exactly 300s (curl timeout), including 3-month slices, so the server was slow for that network
rather than the slice too large. `global_asos/se_retry_watch.sh` (frontend, nohup) waits for
`MISSING_COUNTRIES_REFETCH_DONE`, sleeps one hour, then runs `fetch_sweden_retry.py` (same
bisection, curl timeout 900s, 3-hour budget). Pivot + degree screen after that.

---

## 2026-09-18 (j) -- StatStream's incremental digests (Lemmas 5/6) close its sketch-tier gap; two
new side experiments: parallel scaling and the naive baseline. 151/151.

**Modified**: `library_corrtrack_parallel.py` (`Sketches._dft_raw_incremental`: the sliding-window
DFT coefficients and running sums maintained across steps as in StatStream's Lemmas 5/6, full
recompute every 64 steps against drift, exact to 8e-14 against the recompute over 151 steps;
`_statstream_digests` caches per-basic-window digests by absolute block start so a step computes only
the blocks that entered; `_extract_feature_overrides` treats NaN / "nan" as absent, which a hyperopt
best_params file carries for arms' knobs it did not use). **New**: `abaca/parallel_scaling.py`,
`abaca/naive_baseline.py`, `abaca/experiment.oar` (generic wrapper for the side experiments).

### Time comparability, arm by arm (user's question)
Random walks m=200, W=3600/60, T=0.85, 91 steps (untuned CorrTrack, plumbing figures only):
| arm | sk | cand | val | runtime |
|---|---|---|---|---|
| CorrTrack (Cython throughout) | 0.32 | 0.17 | 0.23 | 0.80 |
| StatStream before (recompute sketch, numpy report) | 1.53 | 0.28 | 0.39 | 2.27 |
| StatStream now (Lemma 6 sketch, cached digests, paper's rule) | 0.30 | 0.16 | 0.15 | 0.69 |
| bruteforce | | | | 4.18 |
So: the representation stage is now at parity for StatStream (its own incremental scheme, which is
also the faithful one); ParCorr/CSZ share CorrTrack's Cython projection sketch; CorrJoin's PAA is a
BLAS matmul and its SVD a LAPACK call; candidate stages are the Cython kernels of (k); validation is
the shared Cython kernel for the exact-validating arms and StatStream's own digest rule (numpy over
cached digests, now cheaper than the exact kernel, as it should be); Pattern-A arms are BLAS/Cython.
The residual tier differences are numpy glue around the kernels; wall-time comparisons are quotable
with that sentence, counters remain the headline.

### Side experiments (user)
- **Parallel scaling** (`parallel_scaling.py`): tuned CorrTrack `exec="thread"` with parallel sketch,
  candidates and validation at --threads {1,2,4,8,16} vs its sequential run, and bruteforce sequential
  vs parallel validation at the same counts; recall must not change (checked against the sequential
  bruteforce); wall is the primary time (stage times sum worker time under threading); BLAS pinned to
  1 thread by the wrapper. Smoke at m=20 shows thread overhead dominating (0.1x to 0.3x), as expected
  at that size; the experiment is for the 2k to 5k anchors.
- **Naive baseline** (`naive_baseline.py`): the papers' "naive" made explicit: per-pair per-step
  `np.corrcoef` in an interpreted loop (naive_python), one vectorized corrcoef per step (naive_numpy),
  our `bruteforce` (vectorized Cython recompute) and `exact_stomp` (incremental); counts must agree.
  sp500 W=60/5 T=0.9, 600 rows: naive_python is 80x / 241x / 261x slower than bruteforce at m = 25 /
  50 / 100 (counts identical); naive_numpy is faster than bruteforce at these tiny sizes (harness
  overhead). This is the table that explains TSUBASA's ">= 10x over naive" and FilCorr's "4x more
  sensors": their naive is our naive_python tier, ours is 2 orders of magnitude faster.
