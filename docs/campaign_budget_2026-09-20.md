# Campaign budget projection (11 N-way hosts, 4 pack hosts, hyperopt grid 80 settings)

Calibration: bruteforce = 1.2e-07 s x m^2 L n_steps; battery = 9.8x / 7.1x bruteforce (pos / neg), light battery 3.8x / 2.5x; hyperopt 12 min and 4.0 GB per run; tuning 18 / 4 min (pos / neg) and 3.0 GB; correlated set 28 B per pair-window, N-way memory = data + bruteforce set + one arm set + 1 GB. Densities: measured for the m=500 sets and the pilot cells, assumed elsewhere.

| option | runs | N-way node-h | N-way days | hyperopt core-h | tuning core-h | pack days | total days | N-way peak GB | runs > budget | runs > 60 GB | longest N-way h | runs > 48 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3202 | 15,387 | 58.3 | 854 | 511 | 1.4 | 58.3 | 1080 | 162 | 262 | 67 | 160 |
| B | 3202 | 5,167 | 19.6 | 854 | 511 | 1.4 | 19.6 | 264 | 34 | 102 | 40 | 0 |
| C | 3202 | 2,070 | 7.8 | 854 | 511 | 1.4 | 7.8 | 264 | 34 | 102 | 15 | 0 |
| D | 2329 | 1,470 | 5.6 | 621 | 470 | 1.1 | 5.6 | 264 | 18 | 53 | 15 | 0 |
| E | 2329 | 3,603 | 13.6 | 621 | 470 | 1.1 | 13.6 | 264 | 18 | 53 | 40 | 0 |
| C-mem | 3102 | 1,512 | 5.7 | 827 | 497 | 1.4 | 5.7 | 106 | 0 | 4 | 15 | 0 |
| D-mem | 2277 | 1,114 | 4.2 | 607 | 458 | 1.1 | 4.2 | 106 | 0 | 3 | 15 | 0 |

## Side experiments (same for every option; own scripts and jobs)

| experiment | runs | N-way-pool node-h | pack-pool core-h | sweep |
|---|---|---|---|---|
| Phase R (paper reproductions) | 8 | 2 | 0 | one run per paper on its own data / stand-in; already run, rerun only after the commit |
| CorrTrack ablation ladder | 528 | 323 | 0 | 528 anchor cells (real datasets + the 2% synthetic family; m_max at L=1 and L_max, 4 T, both spaces): plain BF -> sketch only -> +Hamming / +dot / both -> LSH +Hamming / +dot / both, tuned parameters from the campaign hyperopt |
| Parallel scaling (threads) | 32 | 210 | 0 | 32 anchor cells with m >= 2000 at T=0.9 raw: sequential vs 2/4/8/16/20 threads, bruteforce and tuned CorrTrack, 3 repeats |
| Naive baseline (naive Python / numpy vs bruteforce) | 8 | 2 | 0 | m in {25, 50, 100, 200} on two datasets, T=0.9: naive per-pair Python, naive numpy, our bruteforce, bf_incremental |
| Monitoring / anomalies vs real events | 20 | 30 | 0 | ~5 datasets with event tables (Yellowstone M6.5, streamflow floods, USCRN 2020 events, Motes battery deaths, ASOS), 4 T, CorrTrack monitor on vs bruteforce monitor on; DESIGN PENDING (user's event tables) |
| Window step sweep, step = 1 included | 36 | 2 | 18 | 36 cells: sp500 (W60), USCRN temperature (W168), Yellowstone (W2000) x step in {1, 2, 3, 5, 10, W/10} x both spaces at T=0.9; arms bruteforce, bf_incremental, filcorr, braid, thinbraid, corrtrack, statstream (grid arms need basic_window = step) |
| Dynamic W / step change mid-stream | 18 | 0 | 0 | 18 cells: 3 datasets x 3 schedules x both spaces at T=0.9; CorrTrack update_window_size/update_window_step vs a restarted CorrTrack vs bruteforce following the same schedule; competitors as a capability table |
| Multiple window sizes (shared sketches) | 12 | 2 | 0 | 12 cells: 3 datasets x size sets {W/2, W} and {W/4, W/2, W} x both spaces at T=0.9; CorrTrackMultiWindow vs one CorrTrack per size vs bruteforce per size |
| **total** | 662 | **572** | **18** | |

## Complete figure: campaign option + side experiments

| option | campaign N-way node-h | side node-h | total N-way node-h | total days on the N-way hosts | pack core-h (hyperopt + tuning + side) | pack days |
|---|---|---|---|---|---|---|
| A | 15,387 | 572 | 15,959 | **60.4** | 1,383 | 1.4 |
| B | 5,167 | 572 | 5,739 | **21.7** | 1,383 | 1.4 |
| C | 2,070 | 572 | 2,642 | **10.0** | 1,383 | 1.4 |
| D | 1,470 | 572 | 2,042 | **7.7** | 1,109 | 1.2 |
| E | 3,603 | 572 | 4,175 | **15.8** | 1,109 | 1.2 |
| C-mem | 1,512 | 572 | 2,084 | **7.9** | 1,342 | 1.4 |
| D-mem | 1,114 | 572 | 1,685 | **6.4** | 1,083 | 1.1 |

## What every option contains (the comparative campaign)

- Datasets: 29 real sets (each at its horizon W/step) and the synthetic family (2 processes x 5 densities), each as its own dataset in raw and differenced space.
- Design per dataset (ladder): anchor (m_max, L=1) and rungs (m_max/8, 1), (m_max/4, ~L_max/3), (m_max/2, ~2L_max/3), (m_max, L_max); Berkeley adds the full 18,520-series anchor; plus the W-robustness cells (sp500, USCRN at half and double horizon) and the Yellowstone step=1 cell.
- Every cell at T in {0.7, 0.8, 0.9, 0.95}, in both spaces, with a positive run and a negative-correlation run (neg_corr=True).
- Per cell and run: two CorrTrack hyperopts (lsh_sign_dot grid, 80 settings; lsh_hamming_exact grid, 8 settings), CSZ-protocol tuning of ParCorr / CSZ / StatStream / CorrJoin, and the N-way comparison of the 12 arms (bruteforce, bf_incremental, filcorr, tsubasa, braid, thinbraid, corrtrack [lsh + dot gate], corrtrack_hamming [exact Hamming + dot gate], parcorr, csz, statstream, corrjoin) with the tracked metrics (recall, precision, F1, candidate precision, specificity, phase and total times, step-latency ticks, peak/mean RSS, I/O, energy).
- The options differ only in: the synthetic stream length (A: 20,000 obs, others 5,000), which arms run at m >= 2,500 (C, D and -mem: bruteforce + bf_incremental + the pruning arms), whether the negative run exists at L > 1 (D, E: no), and the memory cut (-mem: no (m=5000, L=5) synthetic rung at density >= 0.05, Berkeley full-m at T >= 0.8 only). Nothing else is removed.

Options:
- **A**: as emitted (synthetic n_obs 20,000)
- **B**: synthetic n_obs 5,000 (402 windows, the median of the real sets)
- **C**: B + at m >= 2,500 only bruteforce, bf_incremental and the pruning arms
- **D**: C + neg_corr run only at L = 1 cells
- **E**: B + neg_corr run only at L = 1 (all arms everywhere)
- **C-mem**: C + memory cut: no (m=5000, L=5) synthetic rung at density >= 0.05; Berkeley full-m at T >= 0.8 only
- **D-mem**: D + the same memory cut

Pack hosts: 4 x 20 cores at core=2 = 40 slots, memory-bound to 192 slots at 4.0 GB per job. Total days = max(N-way days, pack days): the two pools run concurrently and the N-way jobs wait on their own hyperopt and tuning.

C-mem: 50 cells dropped by the memory cut (first 6): berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7, berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7_diff, synth_ar1_d0p05_T0p7_m5000_L5_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p7_m5000_L5_diff_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p8_m5000_L5_m5000_W168_s12_L48_T0.8, synth_ar1_d0p05_T0p8_m5000_L5_diff_m5000_W168_s12_L48_T0.8

D-mem: 50 cells dropped by the memory cut (first 6): berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7, berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7_diff, synth_ar1_d0p05_T0p7_m5000_L5_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p7_m5000_L5_diff_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p8_m5000_L5_m5000_W168_s12_L48_T0.8, synth_ar1_d0p05_T0p8_m5000_L5_diff_m5000_W168_s12_L48_T0.8
