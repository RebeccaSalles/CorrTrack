"""SILSO daily total sunspot number (BRAID's ``Sunspots``, n = 25,900 in the paper). Source:
https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv (semicolon separated: year; month; day;
decimal year; SN; std; n_obs; definitive flag; SN = -1 when missing).

A single series, so this is not a multi-series benchmark: BRAID used it as one lagged pair.
Kept because the user asked for every competitor dataset; it serves as a long univariate
lag-estimation sanity check (pair the series with a shifted copy) and nothing more.

    python datasets/fetch/fetch_sunspots.py
"""
from __future__ import annotations

import numpy as np

from _common import RAW_DIR, download, forward_fill, save_competitor_npz

URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"


def main() -> None:
    path = download(URL, RAW_DIR / "SN_d_tot_V2.0.csv")
    rows = [l.split(";") for l in open(path, encoding="utf-8") if l.strip()]
    sn = np.array([float(r[4]) for r in rows])
    dates = [f"{int(r[0]):04d}-{int(r[1]):02d}-{int(r[2]):02d}" for r in rows]
    sn = np.where(sn < 0, np.nan, sn)
    filled, obs = forward_fill(sn[None, :], max_gap=None)
    save_competitor_npz(
        "sunspots_daily", filled, ["SN_daily_total"],
        {"source": URL, "paper": "BRAID SIGMOD 2005 / TKDD 2010 (Sunspots)", "first_date": dates[0], "last_date": dates[-1],
         "missing_filled": int((~obs).sum()), "regime": "single long univariate series; lag sanity check only"},
    )


if __name__ == "__main__":
    main()
