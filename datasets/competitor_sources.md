# Competitor-paper datasets (implementation plan phase 0f)

Registry of every dataset used by the six competitor papers, what we obtained, how, and which
axis of the evaluation it tests. Built 2026-09-17. Files live in `datasets/competitor/<name>.npz`
(git-ignored, ~1.1 GB total) in the `(m, T)` layout of `datasets/fetch/_common.py`; the raw
downloads are kept in `datasets/competitor/raw/`. Every file is regenerable by the script named
in its row; `meta` inside each npz records the exact alignment choices and per-series coverage.

Loader: `datasets.competitor_loader.load_dataset(name=...)`, wired through the
`experiment_dataset_<name>.py` configs (`OBS_MODE="count"`). N-way runner:
`abaca/nway_compare.py --dataset-config experiment_dataset_<name>.py`.

## Obtained

| name (npz) | paper | m x T | script | regime / what it tests | notes |
|---|---|---|---|---|---|
| `motes_temperature` | BRAID 2005/2010 | 27 x 65,516 | `fetch_motes.py` | **real, lag-by-construction** (BRAID: 202 and 224 min lags = ~390 and ~433 epochs of 31 s) | 54 motes in the file; 7-field rows (4%) skipped; readings outside [-10, 50] C masked (dying batteries emit 122 C); motes < 50% coverage dropped; gaps <= 20 epochs forward-filled |
| `motes_humidity` | BRAID | 31 x 65,516 | same | same | mask [0, 100] % |
| `motes_light` | BRAID | 42 x 65,516 | same | same | mask >= 0 |
| `motes_voltage` | BRAID | 42 x 65,516 | same | same | mask [1.5, 3.5] V |
| `yellowstone_raw` | FilCorr ICDM 2020 section VI | 28 x 120,000 | `fetch_yellowstone_iris.py` | **real, uncooperative**, lagged by wave propagation | WY network, all 100 Hz vertical channels open on 2020-03-31 = 29 stations, exactly the paper's count; YHR had no waveform in the window -> 28. 23:45 to 00:05 UTC around event us70008jr5 (M6.5 Stanley, Idaho; RMS x400 at minute 8). miniSEED decoded by `_mseed.py` with reverse-integration checks |
| `yellowstone_bp3_7` | FilCorr | 28 x 120,000 | same | the band FilCorr correlates | zero-phase Butterworth order 4, 3-7 Hz |
| `uscrn2020_temperature` | TSUBASA SIGMOD 2022 | 153 x 8,784 | `fetch_uscrn_hourly.py` | **real, uncooperative** hourly climate | NOAA USCRN hourly02, 155 station files, T_HR_AVG; QC flags honoured for solar/humidity; gaps <= 6 h filled; < 90% coverage dropped |
| `uscrn2020_precipitation` | TSUBASA | 145 x 8,784 | same | | P_CALC |
| `uscrn2020_solar` | TSUBASA | 137 x 8,784 | same | | SOLARAD |
| `uscrn2020_humidity` | TSUBASA | 135 x 8,784 | same | | RH_HR_AVG |
| `berkeley_tavg_anom_2010` | TSUBASA | **18,520 x 3,652** | `fetch_berkeley_earth.py` (needs h5py) | **largest m**, scalability past the 2k target | 1x1 degree daily TAVG anomalies 2010-2019, land_mask >= 0.5 and no missing day (paper: 18,638) |
| `berkeley_tavg_abs_2010` | TSUBASA | 18,520 x 3,652 | same | strongly seasonal, cooperative | anomaly + climatology |
| `corrjoin_stock` | CorrJoin PACMMOD 2023 | 3,878 x 1,259 | `fetch_corrjoin_drive.py` | real, cooperative; reproduces their speedup claims | authors' Drive files, obtained 2026-09-16 (log (q)); paper W=1020, ks=15, ke=30 |
| `corrjoin_chlorine` | CorrJoin | 4,830 x 2,040 | same | real (EPANET) | |
| `corrjoin_gas` | CorrJoin | 5,120 x 3,600 | same | real (gas sensor array) | |
| `corrjoin_synthetic` | CorrJoin | 5,000 x 4,080 | same | cooperative random walk | |
| `corrjoin_random` | CorrJoin | 5,000 x 4,080 | same | uncooperative i.i.d. uniform (bonus file) | |
| `csz_cstr`, `csz_evaporator`, `csz_steamgen`, `csz_winding`, `csz_foetal_ecg` | CSZ KDD 2005 (via UCR TSDMA 2002) | 3 to 8 channels x 2,500 to 9,600 | `fetch_csz_daisy.py` | cooperative process / biomedical | UCR TSDMA is offline (404, 2026-09-17); these five originate in KU Leuven DaISy and are served there. CSZ's 1,365 to 13,736 "series" per set imply a cut of the channels they do not describe; `--chunk L` gives a documented reconstruction |
| `sunspots_daily` | BRAID | 1 x 76,214 | `fetch_sunspots.py` | single long univariate; lag sanity check only | SILSO V2.0 daily total |
| `statstream_rw_m<m>_T<T>` | StatStream VLDB 2002 section 5 | any | `gen_statstream_randomwalk.py` | cooperative synthetic; formula reproduced exactly | `s(t) = 100 + sum(u - 0.5)` |
| `braid_sines_m<m>_T<T>` | BRAID section 6.1 | any | `gen_braid_synthetic.py --family sines` | cooperative, planted lags known | **approximation**: the paper gives no generator; mixture of 3 sines, periods log-uniform in [64, 4096], lagged copies with noise |
| `braid_spiketrains_m<m>_T<T>` | BRAID section 6.1 | any | `gen_braid_synthetic.py --family spiketrains` | bursty, planted lags known | approximation; Gaussian pulses every `period` (6,500) samples with jitter |

Also usable through the same loader (built by the 2026-09-16 Sobol sweep, `tmp_artifacts/`):
`sp500_sub263` (263 x 1,255), `acwi_capweighted`, `streamflow`, `smartmeter`, `wikipedia`,
`global_weather`.

## Not obtainable, and the stand-in

| paper's set | paper | why | stand-in (to be stated in the paper) |
|---|---|---|---|
| NYSE TAQ, 300 stocks at 1 s | StatStream | licensed | `sp500` pools; `corrjoin_stock` (daily) |
| CRSP end-of-day, 7,861 stocks | CSZ | licensed | `sp500_sub263`, `corrjoin_stock` |
| Yahoo Finance, ~40k symbols 2010-2018 | ParCorr | reproducible in principle, exact symbol set unspecified | `sp500`, `corrjoin_stock` |
| spot_exrates, wind, eeg, price, return | CSZ | UCR TSDMA 2002 offline; not in DaISy | price/return -> `sp500`; the rest have no stand-in |
| Kursk seismic (n = 70,000) | BRAID | not located (single event, source not named beyond "Kursk") | `yellowstone_raw` covers the bursty single-event regime |
| ParCorr seismic | ParCorr | unspecified in the paper | `yellowstone_*` |

## Reproduction targets these enable (comparison plan section 6.5)

- BRAID's Motes lags (202 and 224 min): run `braid` with `N_LAGS >= 433` epochs on
  `motes_temperature` / `motes_humidity`, compare `last_lag_estimates` with exact CCF.
- FilCorr's Yellowstone case study: `yellowstone_bp3_7`, W=2000 (20 s), N_LAGS=1000 (10 s).
- TSUBASA's DFT-accuracy point: `uscrn2020_*` vs a DFT approximation; Berkeley Earth for scale.
- CorrJoin's `r1` and speedup on `corrjoin_stock/chlorine/gas` at T in {0.7, 0.8, 0.9}
  (W must be divisible by ks and ke: W=1020 as in the paper, or (14, 28) at W=168).
- StatStream's cooperative-vs-uncooperative gap: `statstream_rw_*` vs `corrjoin_random` /
  `yellowstone_raw`.
