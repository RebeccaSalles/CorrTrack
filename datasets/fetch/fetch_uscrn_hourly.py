"""NOAA U.S. Climate Reference Network, hourly product (TSUBASA, SIGMOD 2022, section 4:
"NCEA" / NOAA climate data, 157 hourly series for 2020). Source directory
https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/<year>/ , one fixed-column text
file per station (columns documented in ``headers.txt`` next to it). Missing values are
-9999.0 / -99.000 depending on the field; the hour stamp is UTC end-of-hour.

Writes one ``(m, 8784)`` file per variable on the full UTC hourly grid of the year:
    uscrn<year>_temperature   T_HR_AVG   (deg C, hourly mean)
    uscrn<year>_precipitation P_CALC     (mm)
    uscrn<year>_solar         SOLARAD    (W/m^2)
    uscrn<year>_humidity      RH_HR_AVG  (%)
Values whose QC flag is non-zero (solar, humidity) are treated as missing. Gaps up to ``--max-gap`` hours are forward-filled; stations below ``--min-coverage`` dropped.
Uncooperative in CSZ's sense (weather at hourly resolution has little smooth structure), the
regime TSUBASA used to show its DFT-accuracy point.

    python datasets/fetch/fetch_uscrn_hourly.py --year 2020
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from _common import RAW_DIR, coverage, download, forward_fill, save_competitor_npz

BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
# variable -> (value column, missing sentinel, QC-flag column or None; flag != 0 means the value failed QC)
FIELDS = {"temperature": (9, -9999.0, None), "precipitation": (12, -9999.0, None), "solar": (13, -99999.0, 14), "humidity": (26, -9999.0, 27)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--min-coverage", type=float, default=0.9)
    args = ap.parse_args()
    y = args.year
    listing = download(f"{BASE}/{y}/", RAW_DIR / f"uscrn_{y}_index.html")
    names = sorted(set(re.findall(rf"CRNH0203-{y}-[A-Za-z0-9_]+\.txt", open(listing, encoding="utf-8", errors="replace").read())))
    print(f"{len(names)} station files for {y}", flush=True)
    hours = np.arange(np.datetime64(f"{y}-01-01T00"), np.datetime64(f"{y + 1}-01-01T00"), np.timedelta64(1, "h"))
    T = hours.size
    mats = {v: np.full((len(names), T), np.nan) for v in FIELDS}
    ids = []
    for i, fn in enumerate(names):
        path = download(f"{BASE}/{y}/{fn}", RAW_DIR / f"uscrn_{y}" / fn, quiet=True)
        ids.append(fn[len(f"CRNH0203-{y}-"):-4])
        for line in open(path, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) < 27:
                continue
            # UTC_TIME 0000 is the last hour of the previous day: stamp = end of hour
            stamp = np.datetime64(f"{p[1][:4]}-{p[1][4:6]}-{p[1][6:8]}T00") + np.timedelta64(int(p[2][:2]), "h")
            k = int((stamp - hours[0]) / np.timedelta64(1, "h")) - 1  # bucket by hour start
            if not (0 <= k < T):
                continue
            for v, (col, miss, flag) in FIELDS.items():
                try:
                    val = float(p[col])
                except (ValueError, IndexError):
                    continue
                if val <= -99.0 and val in (miss, -9999.0, -99999.0, -99.0):
                    continue
                if flag is not None and p[flag] != "0":
                    continue
                mats[v][i, k] = val
        if (i + 1) % 25 == 0:
            print(f"  parsed {i + 1}/{len(names)}", flush=True)
    for v in FIELDS:
        filled, obs = forward_fill(mats[v], max_gap=args.max_gap)
        cov = coverage(obs)
        keep = cov >= args.min_coverage
        save_competitor_npz(
            f"uscrn{y}_{v}", filled[keep], [s for s, k in zip(ids, keep) if k],
            {"source": f"{BASE}/{y}/", "paper": "TSUBASA SIGMOD 2022 section 4 (NOAA hourly climate, 157 series)", "variable": v,
             "field": ["WBANNO", "UTC_DATE", "UTC_TIME", "LST_DATE", "LST_TIME", "CRX_VN", "LONGITUDE", "LATITUDE", "T_CALC", "T_HR_AVG", "T_MAX", "T_MIN", "P_CALC", "SOLARAD"][FIELDS[v][0]] if FIELDS[v][0] < 14 else "RH_HR_AVG",
             "qc_flag_column": FIELDS[v][2], "grid": f"UTC hourly {y}", "max_gap": args.max_gap, "min_coverage": args.min_coverage,
             "coverage_kept": {ids[i]: round(float(cov[i]), 4) for i in range(len(ids)) if keep[i]}, "dropped": [ids[i] for i in range(len(ids)) if not keep[i]],
             "regime": "real, uncooperative (hourly weather)"},
        )


if __name__ == "__main__":
    main()
