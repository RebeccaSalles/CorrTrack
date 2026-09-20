# Campaign budget projection (11 N-way hosts, 4 pack hosts, hyperopt grid 80 settings)

Calibration: bruteforce = 1.2e-07 s x m^2 L n_steps; battery = 8.0x / 5.5x bruteforce (pos / neg), light battery 3.2x / 2.2x; hyperopt 12 min and 4.0 GB per run; tuning 16 / 3 min (pos / neg) and 3.0 GB; correlated set 28 B per pair-window, N-way memory = data + bruteforce set + one arm set + 1 GB. Densities: measured for the m=500 sets and the pilot cells, assumed elsewhere.

| option | runs | N-way node-h | N-way days | hyperopt core-h | tuning core-h | pack days | total days | N-way peak GB | runs > budget | runs > 60 GB | longest N-way h | runs > 48 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3202 | 12,028 | 45.6 | 640 | 452 | 1.1 | 45.6 | 1080 | 162 | 262 | 55 | 80 |
| B | 3202 | 3,995 | 15.1 | 640 | 452 | 1.1 | 15.1 | 264 | 34 | 102 | 33 | 0 |
| C | 3202 | 1,702 | 6.4 | 640 | 452 | 1.1 | 6.4 | 264 | 34 | 102 | 13 | 0 |
| D | 2329 | 1,182 | 4.5 | 466 | 417 | 0.9 | 4.5 | 264 | 18 | 53 | 13 | 0 |
| E | 2329 | 2,784 | 10.5 | 466 | 417 | 0.9 | 10.5 | 264 | 18 | 53 | 33 | 0 |
| C-mem | 3102 | 1,224 | 4.6 | 620 | 439 | 1.1 | 4.6 | 106 | 0 | 4 | 13 | 0 |
| D-mem | 2277 | 881 | 3.3 | 455 | 406 | 0.9 | 3.3 | 106 | 0 | 3 | 13 | 0 |

Options:
- **A**: as emitted (synthetic n_obs 20,000)
- **B**: synthetic n_obs 5,000 (402 windows, the median of the real sets)
- **C**: B + at m >= 2,500 only bruteforce, exact_stomp and the pruning arms
- **D**: C + neg_corr run only at L = 1 cells
- **E**: B + neg_corr run only at L = 1 (all arms everywhere)
- **C-mem**: C + memory cut: no (m=5000, L=5) synthetic rung at density >= 0.05; Berkeley full-m at T >= 0.8 only
- **D-mem**: D + the same memory cut

Pack hosts: 4 x 20 cores at core=2 = 40 slots, memory-bound to 192 slots at 4.0 GB per job. Total days = max(N-way days, pack days): the two pools run concurrently and the N-way jobs wait on their own hyperopt and tuning.

C-mem: 50 cells dropped by the memory cut (first 6): berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7, berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7_diff, synth_ar1_d0p05_T0p7_m5000_L5_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p7_m5000_L5_diff_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p8_m5000_L5_m5000_W168_s12_L48_T0.8, synth_ar1_d0p05_T0p8_m5000_L5_diff_m5000_W168_s12_L48_T0.8

D-mem: 50 cells dropped by the memory cut (first 6): berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7, berkeley_tavg_anom_2010_m18520_W90_s10_L0_T0.7_diff, synth_ar1_d0p05_T0p7_m5000_L5_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p7_m5000_L5_diff_m5000_W168_s12_L48_T0.7, synth_ar1_d0p05_T0p8_m5000_L5_m5000_W168_s12_L48_T0.8, synth_ar1_d0p05_T0p8_m5000_L5_diff_m5000_W168_s12_L48_T0.8
