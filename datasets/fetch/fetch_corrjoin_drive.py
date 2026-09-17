"""CorrJoin's own datasets (Wang et al., PACMMOD 2023): synthetic random walk 5000x4080,
random (i.i.d. uniform) 5000x4080, chlorine 4830x2040, gas 5120x3600, stock 3878x1259, shared
by the authors at drive.google.com/drive/folders/1skrE2x2DMgIms5lZR04kiOlLvvna2zNs .

The folder listing is not fetchable programmatically, but the per-file share links are; each
downloads as a zip holding one whitespace-separated ``<name>.txt`` with one series per row.
All five were obtained on 2026-09-16 (implementation log entry (q)) and converted to
``tmp_artifacts/corrjoin_<name>/corrjoin_<name>.npz`` in the same ``(m, T)`` layout this
folder uses, so by default this script only re-exports those with metadata into
``datasets/competitor/``. To refetch from scratch pass the file ids from the folder listing:

    python datasets/fetch/fetch_corrjoin_drive.py --file-id stock=<id> --file-id gas=<id> ...

or drop the raw ``<name>.txt`` / zip files into ``datasets/competitor/raw/corrjoin/``.
"""
from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from _common import RAW_DIR, REPO, save_competitor_npz

NAMES = ("synthetic", "random", "chlorine", "gas", "stock")
EXPECTED = {"synthetic": (5000, 4080), "random": (5000, 4080), "chlorine": (4830, 2040), "gas": (5120, 3600), "stock": (3878, 1259)}
SOURCE = "https://drive.google.com/drive/folders/1skrE2x2DMgIms5lZR04kiOlLvvna2zNs"


def drive_download(file_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with urllib.request.urlopen(url, timeout=600) as r:
        body = r.read()
    if body[:2] != b"PK" and b"uuid" in body:  # virus-scan interstitial for large files
        uuid = re.search(rb'name="uuid" value="([^"]+)"', body).group(1).decode()
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid}"
        with urllib.request.urlopen(url, timeout=1800) as r:
            body = r.read()
    dest.write_bytes(body)
    return dest


def load_txt(path: Path) -> np.ndarray:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            inner = [n for n in z.namelist() if n.endswith(".txt")][0]
            return np.loadtxt(io.BytesIO(z.read(inner)))
    return np.loadtxt(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file-id", action="append", default=[], help="name=<drive file id>")
    ap.add_argument("--names", default=",".join(NAMES))
    args = ap.parse_args()
    ids = dict(kv.split("=", 1) for kv in args.file_id)
    raw_dir = RAW_DIR / "corrjoin"
    for name in args.names.split(","):
        data = None
        provenance = None
        if name in ids:
            path = drive_download(ids[name], raw_dir / f"{name}.zip")
            data, provenance = load_txt(path), f"drive file id {ids[name]}"
        else:
            for cand in (raw_dir / f"{name}.txt", raw_dir / f"{name}.zip"):
                if cand.exists():
                    data, provenance = load_txt(cand), str(cand)
                    break
        if data is None:
            prev = REPO / "tmp_artifacts" / f"corrjoin_{name}" / f"corrjoin_{name}.npz"
            if not prev.exists():
                print(f"{name}: no raw file and no {prev}; pass --file-id {name}=<id>", flush=True)
                continue
            with np.load(prev, allow_pickle=True) as z:
                data = np.asarray(z["data"], dtype=np.float64)
            provenance = f"re-export of {prev.relative_to(REPO)} (built 2026-09-16 from the authors' Drive files)"
        if tuple(data.shape) != EXPECTED[name]:
            raise ValueError(f"{name}: shape {data.shape} != paper's {EXPECTED[name]}")
        save_competitor_npz(
            f"corrjoin_{name}", data, [f"{name}_{i}" for i in range(data.shape[0])],
            {"source": SOURCE, "provenance": provenance, "paper": "CorrJoin PACMMOD 2023 (authors' own benchmark files)",
             "regime": {"synthetic": "cooperative (random walk)", "random": "uncooperative (i.i.d. uniform)", "chlorine": "real, EPANET chlorine, cooperative",
                        "gas": "real, gas sensor array, cooperative", "stock": "real, daily stock prices, cooperative"}[name]},
        )


if __name__ == "__main__":
    main()
