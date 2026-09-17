"""Intel Berkeley Research Lab sensor data ("Motes"; BRAID 2005/2010, also StatStream-era
work). Source: https://db.csail.mit.edu/labdata/labdata.html, file data.txt.gz (34 MB,
2.3 M rows: date time epoch moteid temperature humidity light voltage; 54 motes, one epoch
every 31 s, 2004-02-28 to 2004-04-05).

Alignment (all choices recorded in meta):
- pivot on the file's own ``epoch`` counter (0..65535), one column per epoch;
- keep motes 1..54 (rows with moteid > 54 are known artefacts of the release);
- restrict to the epoch range in which at least ``--min-active`` motes report;
- forward-fill gaps up to ``--max-gap`` epochs, longer gaps stay NaN;
- drop motes whose observed coverage inside that range is below ``--min-coverage``;
- rows with 7 fields (one reading missing, position not recoverable; ~4% of rows) are skipped;
- readings outside a physical plausibility range are treated as missing before the fill
  (``--no-mask`` disables): temperature [-10, 50] C, humidity [0, 100] %, light >= 0,
  voltage [1.5, 3.5] V. Dying batteries produce runs of 122 C / 385 C in the raw file.
Writes one file per variable: motes_temperature, motes_humidity, motes_light, motes_voltage.
BRAID reports physical lags of 202 and 224 minutes (~390 and ~433 epochs) between nearby
sensors, so this is the one real set where the lag axis is a measured fact.

    python datasets/fetch/fetch_motes.py
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import RAW_DIR, coverage, download, forward_fill, open_maybe_gzip, save_competitor_npz

URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
VARS = ("temperature", "humidity", "light", "voltage")
PLAUSIBLE = {"temperature": (-10.0, 50.0), "humidity": (0.0, 100.0), "light": (0.0, np.inf), "voltage": (1.5, 3.5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-active", type=int, default=30, help="epochs with fewer reporting motes are trimmed from both ends")
    ap.add_argument("--max-gap", type=int, default=20, help="forward-fill limit in epochs (~10 min)")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--no-mask", action="store_true", help="keep physically implausible readings")
    args = ap.parse_args()

    raw = download(URL, RAW_DIR / "motes_data.txt.gz")
    n_epochs, n_motes = 65536, 54
    sums = np.zeros((len(VARS), n_motes, n_epochs))
    cnt = np.zeros((n_motes, n_epochs))
    bad = 0
    with open_maybe_gzip(raw) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 8:
                bad += 1
                continue
            try:
                epoch, mote = int(parts[2]), int(parts[3])
                vals = [float(v) for v in parts[4:8]]
            except ValueError:
                bad += 1
                continue
            if not (1 <= mote <= n_motes) or not (0 <= epoch < n_epochs):
                bad += 1
                continue
            for k, v in enumerate(vals):
                sums[k, mote - 1, epoch] += v
            cnt[mote - 1, epoch] += 1
    observed = cnt > 0
    active = observed.sum(axis=0)
    ok = np.where(active >= args.min_active)[0]
    lo, hi = int(ok[0]), int(ok[-1]) + 1
    print(f"rows skipped: {bad}; active epoch range [{lo}, {hi}) = {hi - lo} epochs", flush=True)
    for k, var in enumerate(VARS):
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.where(observed, sums[k] / np.maximum(cnt, 1), np.nan)[:, lo:hi]
        masked = 0
        if not args.no_mask:
            lo_v, hi_v = PLAUSIBLE[var]
            implausible = ~np.isnan(mat) & ((mat < lo_v) | (mat > hi_v))
            masked = int(implausible.sum())
            mat = np.where(implausible, np.nan, mat)
        filled, obs = forward_fill(mat, max_gap=args.max_gap)
        cov = coverage(obs)
        keep = cov >= args.min_coverage
        ids = [f"mote{i + 1:02d}" for i in range(n_motes)]
        save_competitor_npz(
            f"motes_{var}", filled[keep], [s for s, kp in zip(ids, keep) if kp],
            {"source": URL, "paper": "BRAID (Sakurai et al. 2005/2010), Motes/Humidity/Light", "variable": var,
             "epoch_range": [lo, hi], "epoch_seconds": 31, "min_active": args.min_active, "max_gap": args.max_gap,
             "min_coverage": args.min_coverage, "rows_skipped_malformed": bad,
             "implausible_masked": masked, "plausible_range": None if args.no_mask else list(PLAUSIBLE[var]), "coverage_kept": {ids[i]: round(float(cov[i]), 4) for i in range(n_motes) if keep[i]},
             "dropped": [ids[i] for i in range(n_motes) if not keep[i]],
             "regime": "real, lag-by-construction (BRAID reports 202 and 224 min lags between nearby motes)"},
        )


if __name__ == "__main__":
    main()
