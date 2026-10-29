# Fixed-degree m-sweep: CorrTrack's search exponent (2026-10-18)

Question (presentation question 3, docs/presentation_questions_2026-10-15.md): the headline m-sweep holds the
density at 0.01, so the correlated output grows as m^2 and its fitted 1.64 mixes linear, search and quadratic
output terms. Here each series keeps exactly 55 correlated partners at every m (density falls as 1/m: 0.036 at
m=125 to 0.0012 at m=4000), so the output grows linearly and the wall exponent exposes the search.

Protocol: `abaca/campaign_synth_degree.py`, frozen final112 snapshot (same code as every paper number), OAR jobs
3198069 onward (first submission 3197393 to 3197428 generated the inputs; its tuning failed, see
implementation_log 2026-10-17). AR(1) raw levels, W=60, step 6, L=6 (30 lags), n=6000, positive correlation;
m in 125..4000; T in 0.7, 0.8, 0.9, 0.95. CorrTrack tuned once per threshold at m=500 (both backends) and reused at
every m; band width and n_bands resize with m at run time. Results: `~/results/synth_degree/nway/*/nway.json`.
Recall of both CorrTrack backends is at least 0.997 in all 24 cells, precision 1.

## Exponents in m (least squares on logs, m = 125 to 4000)

rho is the textbook LSH exponent for sign hashing, log2(1/(1 - arccos(T)/pi)); rho_eff is the growth exponent in
N = m L of the n_bands CorrTrack actually sizes (`compute_lsh_n_bands`, sized at T, from a pool of 64 sign bits).

| T | textbook 1+rho | 1+rho_eff (64-bit pool) | CorrTrack-LSH wall | LSH candidate search | LSH validation | CorrTrack-Ham wall | BF_incr wall | FilCorr wall | BF_vect wall |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 1.42 | 1.69 | 1.84 | 2.00 | 1.20 | 1.88 | 1.99 | 1.95 | 2.02 |
| 0.8 | 1.33 | 1.56 | 1.66 | 1.85 | 1.05 | 1.74 | 1.99 | 1.95 | 2.02 |
| 0.9 | 1.22 | 1.40 | 1.44 | 1.64 | 1.03 | 1.59 | 1.99 | 1.95 | 2.02 |
| 0.95 | 1.15 | 1.29 | 1.30 | 1.49 | 1.02 | 1.56 | 2.00 | 1.95 | 2.02 |

## CorrTrack-LSH speedup over BF_vect

| T | m=125 | m=250 | m=500 | m=1000 | m=2000 | m=4000 |
|---|---|---|---|---|---|---|
| 0.7 | 4.6x | 7.0x | 9.2x | 10.3x | 9.6x | 9.2x |
| 0.8 | 5.5x | 8.9x | 14.1x | 17.8x | 19.6x | 19.0x |
| 0.9 | 6.3x | 11.6x | 20.6x | 32.5x | 43.1x | 44.4x |
| 0.95 | 6.6x | 13.9x | 26.5x | 42.7x | 66.9x | 78.0x |

## Reading

- The prediction holds in direction and roughly in size: the higher the threshold, the lower the exponent
  (wall 1.84 / 1.66 / 1.44 / 1.30). Every exact arm sits at 2 (BF_vect 2.02, BF_incr 1.99, FilCorr 1.95).
- Sketching (about m^0.96) and validation (m^1.02 to 1.20) are linear: with linear output the whole superlinear
  cost is the candidate search.
- CorrTrack sits above the textbook 1 + rho because its bands share one pool of 64 sign bits: as the band width
  grows with m (6 to 11 bits here), bands overlap more and the recall target needs more of them, so n_bands grows as
  N^rho_eff with rho_eff about 1.6 to 1.9 times rho (0.69 / 0.56 / 0.40 / 0.29). The wall exponent tracks
  1 + rho_eff closely at high T (1.44 vs 1.40, 1.30 vs 1.29) and exceeds it at T = 0.7 (1.84 vs 1.69), where the
  tuned gate is low (0.5) and the candidates tested grow faster than the output (local slope up to 1.43).
- The candidate search alone grows faster than the wall (2.00 / 1.85 / 1.64 / 1.49). Part of that is a machine
  effect at the last step: from m = 2000 to 4000 every memory-bound arm steepens (BF_incr 2.7, FilCorr 2.6, which
  their algorithms cannot produce; BF_vect, BLAS-blocked, stays at 1.97), and CorrTrack's random-access scan with
  them. Without m = 4000 the candidate search fits 1.97 / 1.80 / 1.57 / 1.43.
- Speedup grows with m at every threshold above 0.7 (to 44x at T=0.9 and 78x at T=0.95 at m=4000), and flattens at
  the last step for the same machine reason. At T = 0.7 it peaks near 10x.
- Lever, not tested: more sketch bits shrink the pool overlap. At 128 / 256 bits rho_eff would fall to 0.45 / 0.39
  at T=0.8 (textbook 0.33), at the cost of a longer sketch and dot product.

## For the paper

State the time as subquadratic search plus output, and give the measured search exponent under linear output:
1.30 to 1.84 depending on the threshold, explained by LSH theory with CorrTrack's shared-bit-pool sizing. Do not
quote one exponent without its threshold, and say the fit range is m = 125 to 4000 on one node type.

## Figure

`docs/figures/synth_degree_scaling.png`, builder `abaca/synth_degree_figure.py ~/results/synth_degree <png>`.
A: CorrTrack-LSH wall against m per threshold (one red ramp, darker = higher T), BF_vect for reference, dashed
lines with the slope 1 + rho_eff through m = 500. B: exponent against T, measured (wall, and candidate search alone)
against the textbook 1 + rho and CorrTrack's 1 + rho_eff at 64 bits; the exact arms sit at 2. The authors-only
filter changes nothing here (class L, positive correlation: every arm shown is in the authors-only set), so there
is one version. rho_eff is a staircase in T (integer band width and n_bands); B draws it on a coarse T grid.

## Sketch-width ablation (submitted 2026-10-18, OAR 3200796 to 3200808)

At T = 0.8, the tuned 64-bit parameters copied with n_vectors 128 and 256 (everything else unchanged,
`~/corrtrack_abaca_results/synth_degree/ablation/nv*/best_params_corrtrack.json`), inputs regenerated with the same
seed, arms bruteforce and corrtrack, m = 125 to 4000; run names end in `_nv128` / `_nv256`. Prediction from the band
sizing: rho_eff 0.56 / 0.45 / 0.39 at 64 / 128 / 256 bits (textbook 0.33). The figure builder adds the measured
points and the 128 / 256-bit curves to panel B once the results are copied back.

### Ablation results (2026-10-19)

All 12 cells completed; recall >= 0.9998 everywhere; regenerated inputs deleted on the cluster.

| sketch bits | n_bands, m = 125 to 4000 | predicted 1 + rho_eff | measured wall exponent | without m = 4000 | candidate search | sketching | wall at m = 4000 | speedup at m = 4000 |
|---|---|---|---|---|---|---|---|---|
| 64 | 19 to 132 | 1.56 | 1.66 | 1.57 | 1.85 | 0.96 | 676 s | 19.0x |
| 128 | 16 to 76 | 1.45 | 1.51 | 1.42 | 1.71 | 1.10 | 440 s | 29.2x |
| 256 | 15 to 59 | 1.39 | 1.44 | 1.38 | 1.61 | 1.19 | 399 s | 32.2x |

The exponent falls with the pool size as the band sizing predicts (measured minus predicted: +0.10 / +0.06 /
+0.05), confirming that band overlap in the 64-bit pool is the main reason CorrTrack sits above textbook LSH.
Wider sketches cost more per window (sketching 13 / 31 / 54 s at m = 4000) and do not pay at small m (m = 125:
2.2 / 2.3 / 2.8 s); 128 bits is faster from m = 500 on, 256 bits only from m = 4000. Decomposition of the 64-bit
growth exponent at T = 0.8 (from compute_lsh_n_bands): textbook 0.33, independent bands at the actual widths 0.36,
shared-pool overlap 0.51, plus the +0.028 recall margin 0.56. Fix that keeps the 64-dim sketch: draw each band bit
as the sign of a fresh random projection of the sketch (independent bits, no pool), expected to bring the growth
exponent back near the textbook rho; not implemented.
