"""Berkeley Earth daily land temperature on the 1x1 degree grid (TSUBASA, SIGMOD 2022:
"Berkeley Earth", 18,638 series x 3,652 days, the largest m among the competitor papers).

Source: https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/
Complete_TAVG_Daily_LatLong1_<decade>.nc (NetCDF4/HDF5, ~460 MB per decade). Variables:
``temperature`` (day, lat, lon) daily anomaly in deg C, ``climatology`` (day_of_year, lat,
lon), ``land_mask`` (lat, lon) fraction of land. Reading the file needs ``h5py`` (or
``netCDF4``); neither is in the system Python here, so run this with a venv/conda that has
h5py, e.g. on Abaca's ``corrtrack`` env (``conda install h5py``).

One ``(m, T)`` file per requested variant:
    berkeley_tavg_anom_<decade>  : the anomaly series as stored (TSUBASA's input)
    berkeley_tavg_abs_<decade>   : anomaly + climatology (absolute temperature; strongly
                                   seasonal, hence cooperative), with ``--absolute``
Series = grid cells with ``land_mask >= --land-min`` (default 0.5 gives ~18.6k cells at 1x1)
and no missing days; cells with any missing value are dropped and counted in meta. To keep a
smoke-sized version, ``--max-series`` takes the first N land cells in row-major order.

    python datasets/fetch/fetch_berkeley_earth.py --decade 2010
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import RAW_DIR, download, save_competitor_npz

URL = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Complete_TAVG_Daily_LatLong1_{decade}.nc"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decade", type=int, default=2010)
    ap.add_argument("--land-min", type=float, default=0.5)
    ap.add_argument("--absolute", action="store_true", help="also write anomaly + climatology")
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("fetch_berkeley_earth.py needs h5py (pip install h5py in a venv, or conda install h5py)") from exc

    path = download(URL.format(decade=args.decade), RAW_DIR / f"berkeley_TAVG_Daily_LatLong1_{args.decade}.nc", timeout=3600)
    with h5py.File(path, "r") as f:
        land = f["land_mask"][...]
        lat, lon = f["latitude"][...], f["longitude"][...]
        year, month, day = f["year"][...], f["month"][...], f["day"][...]
        temp = f["temperature"]
        n_days = temp.shape[0] if args.max_days is None else min(temp.shape[0], args.max_days)
        li, lj = np.where(land >= args.land_min)
        if args.max_series is not None:
            li, lj = li[: args.max_series], lj[: args.max_series]
        print(f"{len(li)} land cells (land_mask >= {args.land_min}), {n_days} days", flush=True)
        anom = np.empty((len(li), n_days))
        chunk = 64
        for d0 in range(0, n_days, chunk):
            block = temp[d0: min(n_days, d0 + chunk)]  # (days, lat, lon)
            anom[:, d0: d0 + block.shape[0]] = block[:, li, lj].T
        clim = None
        if args.absolute:
            cl = f["climatology"][...]  # (366 or 365, lat, lon)
            doy = np.array([(np.datetime64(f"{int(y)}-{int(m):02d}-{int(dd):02d}") - np.datetime64(f"{int(y)}-01-01")).astype(int) for y, m, dd in zip(year[:n_days], month[:n_days], day[:n_days])])
            doy = np.minimum(doy, cl.shape[0] - 1)
            clim = cl[:, li, lj][doy].T
    ok = ~np.isnan(anom).any(axis=1)
    ids = [f"lat{lat[i]:+.1f}_lon{lon[j]:+.1f}" for i, j in zip(li, lj)]
    meta = {"source": URL.format(decade=args.decade), "paper": "TSUBASA SIGMOD 2022 (Berkeley Earth, 18,638 x 3,652)", "decade": args.decade,
            "land_min": args.land_min, "first_day": f"{int(year[0])}-{int(month[0]):02d}-{int(day[0]):02d}",
            "last_day": f"{int(year[n_days - 1])}-{int(month[n_days - 1]):02d}-{int(day[n_days - 1]):02d}",
            "cells_dropped_missing": int((~ok).sum()), "regime": "real, climate grid; anomalies weakly cooperative, absolute strongly seasonal"}
    save_competitor_npz(f"berkeley_tavg_anom_{args.decade}", anom[ok], [s for s, k in zip(ids, ok) if k], {**meta, "units": "deg C anomaly"})
    if clim is not None:
        save_competitor_npz(f"berkeley_tavg_abs_{args.decade}", (anom + clim)[ok], [s for s, k in zip(ids, ok) if k], {**meta, "units": "deg C absolute (anomaly + climatology)"})


if __name__ == "__main__":
    main()
