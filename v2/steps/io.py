"""Step 1 — read the CSV.

Two layouts are auto-detected from the first column:

* **single time column** (numeric 1st column): `time, s0, s1, …`
  → `times` = that column, series = columns 1+.
* **ASOS date+time** (non-numeric 1st column, e.g. "2000-01-01"):
  `date, time(hour), s0, s1, …` → `times` = integer hours since the first
  timestamp (numeric axis, lags in hours), series = columns **2+**.
  See `datasets/asos_loader.py` for the original convention.
"""

import numpy as np
import pandas as pd


def read_csv(path):
    """Return (ids, times, data).

    ids   : list of series names.
    times : np.ndarray (T,), numeric (lags are expressed in this unit).
    data  : np.ndarray (n_series, T); data[i] = series ids[i].
    """
    df = pd.read_csv(path)

    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        # 1 numeric time column, then the series.
        times = df.iloc[:, 0].to_numpy()
        series = df.iloc[:, 1:]
    else:
        # ASOS format: date (str) + time (hour) -> datetime -> integer hours.
        dt = pd.to_datetime(
            df.iloc[:, 0].astype(str) + "T" + df.iloc[:, 1].astype(str),
            format="%Y-%m-%dT%H",
        )
        times = ((dt - dt.iloc[0]) // pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
        series = df.iloc[:, 2:]

    ids = list(series.columns)
    data = series.to_numpy(dtype=float).T
    return ids, times, data


HOURS_PER_YEAR = 365 * 24  # v1 convention (leap years ignored)


def subset(ids, times, data, n_series=0, n_years=0, obs_mode="years", train_ratio=1.0):
    """Filter the data that was read (same rules as v1).

    n_series    : keep only the first `n_series` series (0 = all).
    n_years     : keep only some observations (0 = all):
        * obs_mode="years": the LAST `n_years × 365×24` (the most recent ones);
        * obs_mode="count": the FIRST `n_years` rows.
    train_ratio : `<1` -> keep only the FIRST `round(train_ratio × T)` obs
                  (time prefix for the hyperparameter sweep).
    """
    if n_series and n_series < len(ids):
        ids = ids[:n_series]
        data = data[:n_series]

    if n_years:
        total = data.shape[1]
        if obs_mode == "count":
            keep = min(int(n_years), total)
            times, data = times[:keep], data[:, :keep]
        else:  # "years"
            keep = min(int(n_years) * HOURS_PER_YEAR, total)
            times, data = times[-keep:], data[:, -keep:]

    if train_ratio and train_ratio < 1.0:
        keep = max(1, round(train_ratio * data.shape[1]))
        times, data = times[:keep], data[:, :keep]

    return ids, times, data
