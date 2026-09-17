"""Cole, Shasha, Zhao (KDD 2005) evaluated on ten sets from the UCR Time Series Data Mining
Archive of 2002: spot_exrates, cstr, foetal_ecg, evaporator, steamgen, wind, winding, eeg,
price, return. That archive is offline (cs.ucr.edu/~eamonn/TSDMA returns 404, checked
2026-09-17). Five of the ten are system-identification benchmarks that originated in KU
Leuven's DaISy collection, which is still served:

    cstr        7500 x 3   continuous stirred tank reactor
    evaporator  6305 x 5   four-stage evaporator
    steamgen    9600 x 8   steam generator
    winding     2500 x 6   web winding process
    foetal_ecg  2500 x 8   cutaneous foetal ECG (8 leads)

(first column of each .dat is the time stamp and is dropped). These are *few long channels*;
CSZ report 1,365 to 13,736 series per set, so they must have cut the channels into pieces,
without saying how. ``--chunk L`` reproduces the only reasonable reading: every channel is cut
into non-overlapping length-L segments and each segment becomes one series. With the default
``--chunk 0`` the channels are stored as they are (``csz_<name>``); with a chunk length the
file is ``csz_<name>_chunk<L>``. spot_exrates, wind, eeg, price and return remain unobtained;
our ``sp500`` pools stand in for price/return, stated as such in the paper.

    python datasets/fetch/fetch_csz_daisy.py            # raw channels
    python datasets/fetch/fetch_csz_daisy.py --chunk 512
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import RAW_DIR, download, open_maybe_gzip, save_competitor_npz

BASE = "https://ftp.esat.kuleuven.be/pub/SISTA/data"
SETS = {"cstr": "process_industry/cstr", "evaporator": "process_industry/evaporator", "steamgen": "process_industry/steamgen",
        "winding": "process_industry/winding", "foetal_ecg": "biomedical/foetal_ecg"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--names", default=",".join(SETS))
    args = ap.parse_args()
    for name in args.names.split(","):
        rel = SETS[name]
        path = download(f"{BASE}/{rel}.dat.gz", RAW_DIR / "daisy" / f"{name}.dat.gz")
        download(f"{BASE}/{rel}.txt", RAW_DIR / "daisy" / f"{name}.txt", quiet=True)
        with open_maybe_gzip(path) as fh:
            mat = np.loadtxt(fh)
        channels = mat[:, 1:].T  # drop the time column -> (channels, T)
        ids = [f"{name}_ch{k + 1}" for k in range(channels.shape[0])]
        meta = {"source": f"{BASE}/{rel}.dat.gz", "paper": "Cole, Shasha, Zhao KDD 2005 section 6 (via UCR TSDMA 2002, now offline)",
                "description_file": str(RAW_DIR / "daisy" / f"{name}.txt"), "regime": "real, process/biomedical; cooperative (smooth)"}
        if args.chunk > 0:
            L = args.chunk
            n_seg = channels.shape[1] // L
            segs = channels[:, : n_seg * L].reshape(channels.shape[0], n_seg, L).reshape(-1, L)
            seg_ids = [f"{ids[c]}_seg{s:03d}" for c in range(channels.shape[0]) for s in range(n_seg)]
            save_competitor_npz(f"csz_{name}_chunk{L}", segs, seg_ids, {**meta, "chunk": L, "note": "channels cut into non-overlapping segments; CSZ's own procedure is unspecified"})
        else:
            save_competitor_npz(f"csz_{name}", channels, ids, meta)


if __name__ == "__main__":
    main()
