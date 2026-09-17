"""FilCorr's case study (Zhong, Souza, Mueen, ICDM 2020, section VI): the Yellowstone
seismic network, 29 stations at 100 Hz, around USGS event us70008jr5 (2020-03-31 23:52:30 UTC,
the M6.5 Stanley, Idaho earthquake), 20 s windows, 3-7 Hz band, 10 s max lag.

Data come from EarthScope (the former IRIS DMC; ``service.iris.edu`` now redirects there) via
FDSN ``station`` and ``dataselect``. The station query for network WY, vertical channels at
100 Hz, open on the event date, returns exactly 29 stations. miniSEED is decoded by
``_mseed.py`` (Steim1/2 with reverse-integration checks). Two files are written:

    yellowstone_raw      : counts, 100 Hz, [--start, --end), NaN where a station has a gap
    yellowstone_bp3_7    : the same after a zero-phase 4th-order Butterworth 3-7 Hz band-pass,
                           i.e. the signal FilCorr correlates; gaps filled with 0 before filtering

The default window is 23:45:00 to 00:05:00 (20 min = 120,000 samples), the event at 7.5 min.

    python datasets/fetch/fetch_yellowstone_iris.py
"""
from __future__ import annotations

import argparse
import urllib.parse

import numpy as np

from _common import RAW_DIR, download, save_competitor_npz
from _mseed import read_mseed, trace_to_regular

STATION = "https://service.earthscope.org/fdsnws/station/1/query"
DATASELECT = "https://service.earthscope.org/fdsnws/dataselect/1/query"


def station_list(net, channels, t0, t1):
    q = urllib.parse.urlencode({"network": net, "channel": channels, "starttime": t0, "endtime": t1, "level": "channel", "format": "text"})
    path = download(f"{STATION}?{q}", RAW_DIR / f"yellowstone_stations_{net}.txt")
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("|")
        rows.append({"net": f[0], "sta": f[1], "loc": f[2], "cha": f[3], "lat": float(f[4]), "lon": float(f[5]), "rate": float(f[14])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="WY")
    ap.add_argument("--channels", default="HHZ,EHZ", help="vertical components; HHZ preferred when a station has both")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--start", default="2020-03-31T23:45:00")
    ap.add_argument("--end", default="2020-04-01T00:05:00")
    ap.add_argument("--band", default="3,7")
    args = ap.parse_args()
    t0, t1 = np.datetime64(args.start), np.datetime64(args.end)

    rows = [r for r in station_list(args.network, args.channels, args.start, args.end) if r["rate"] == args.rate]
    by_sta = {}
    for r in rows:  # prefer HHZ over EHZ when both exist
        cur = by_sta.get(r["sta"])
        if cur is None or (cur["cha"] != "HHZ" and r["cha"] == "HHZ"):
            by_sta[r["sta"]] = r
    stations = [by_sta[k] for k in sorted(by_sta)]
    print(f"{len(stations)} stations at {args.rate:g} Hz in {args.network}", flush=True)

    n = int(round((t1 - t0) / np.timedelta64(1, "s") * args.rate))
    data = np.full((len(stations), n), np.nan)
    ids, meta_st = [], []
    for i, r in enumerate(stations):
        q = urllib.parse.urlencode({"net": r["net"], "sta": r["sta"], "loc": r["loc"] or "--", "cha": r["cha"], "starttime": args.start, "endtime": args.end})
        path = download(f"{DATASELECT}?{q}", RAW_DIR / "yellowstone_mseed" / f"{r['net']}.{r['sta']}.{r['loc']}.{r['cha']}.mseed", quiet=True)
        traces = read_mseed(open(path, "rb").read())
        if not traces:
            print(f"  {r['sta']}: no data", flush=True)
        for tr in traces:
            x, rate = trace_to_regular(tr, t0, t1)
            if abs(rate - args.rate) > 1e-6:
                raise ValueError(f"{r['sta']} rate {rate} != {args.rate}")
            data[i] = x
        ids.append(f"{r['net']}.{r['sta']}.{r['loc']}.{r['cha']}")
        meta_st.append({**r, "gap_fraction": float(np.isnan(data[i]).mean())})
        print(f"  {ids[-1]:20s} gaps={meta_st[-1]['gap_fraction']:.4f}", flush=True)

    have = ~np.all(np.isnan(data), axis=1)
    dropped = [ids[i] for i in range(len(ids)) if not have[i]]
    data = data[have]
    ids = [s for s, h in zip(ids, have) if h]
    meta_st = [m for m, h in zip(meta_st, have) if h]
    if dropped:
        print(f"dropped (no waveform in window): {dropped}", flush=True)

    base_meta = {"dropped_no_data": dropped, "source": f"{DATASELECT} (EarthScope / IRIS DMC)", "paper": "FilCorr ICDM 2020 section VI", "event": "us70008jr5 2020-03-31T23:52:30Z M6.5 Stanley, Idaho",
                 "network": args.network, "rate_hz": args.rate, "start": args.start, "end": args.end, "stations": meta_st,
                 "regime": "real, uncooperative (white-noise-like traces), lagged by wave propagation"}
    save_competitor_npz("yellowstone_raw", data, ids, {**base_meta, "units": "raw counts"})

    from scipy.signal import butter, sosfiltfilt
    lo, hi = (float(v) for v in args.band.split(","))
    sos = butter(4, [lo, hi], btype="bandpass", fs=args.rate, output="sos")
    filled = np.where(np.isnan(data), 0.0, data - np.nanmean(data, axis=1, keepdims=True))
    bp = sosfiltfilt(sos, filled, axis=1)
    save_competitor_npz(f"yellowstone_bp{int(lo)}_{int(hi)}", bp, ids, {**base_meta, "units": "counts, band-passed", "band_hz": [lo, hi], "filter": "butter order 4, sosfiltfilt (zero phase), gaps zero-filled after demeaning"})


if __name__ == "__main__":
    main()
