"""Minimal miniSEED (SEED 2.4 data records) reader: enough for FDSN ``dataselect`` output
with Steim1 / Steim2 / plain integer encodings, so the Yellowstone fetch does not need obspy.

Each Steim frame carries the forward (x0) and reverse (xn) integration constants; the decoder
checks the last reconstructed sample against xn and raises on mismatch, so a decoding error
cannot pass silently. Returns per-channel sample arrays with their start times and rates.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    net: str
    sta: str
    loc: str
    cha: str
    rate: float
    start: np.datetime64
    samples: list = field(default_factory=list)


def _btime(buf, off, bo):
    year, day, hour, minute, sec, _u, frac = struct.unpack(bo + "HHBBBBH", buf[off:off + 10])
    base = np.datetime64(f"{year:04d}-01-01T00:00:00.000000") + np.timedelta64(day - 1, "D")
    return base + np.timedelta64(hour, "h") + np.timedelta64(minute, "m") + np.timedelta64(sec, "s") + np.timedelta64(frac * 100, "us")


def _rate(factor, mult):
    if factor > 0 and mult > 0:
        return factor * mult
    if factor > 0 and mult < 0:
        return factor / -mult
    if factor < 0 and mult > 0:
        return mult / -factor
    return 1.0 / (factor * mult)


def _u32s(chunk, bo):
    return struct.unpack(bo + "I" * (len(chunk) // 4), chunk)


def _signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def _steim(data, bo, n, version):
    out = []
    x0 = xn = None
    for f0 in range(0, len(data) - 63, 64):
        w = _u32s(data[f0:f0 + 64], bo)
        ctrl = w[0]
        for k in range(1, 16):
            nib = (ctrl >> (2 * (15 - k))) & 3
            if f0 == 0 and k == 1:
                x0 = _signed(w[k], 32)
                continue
            if f0 == 0 and k == 2:
                xn = _signed(w[k], 32)
                continue
            if nib == 0:
                continue
            v = w[k]
            if nib == 1:
                out.extend(_signed(v >> s, 8) for s in (24, 16, 8, 0))
            elif version == 1:
                if nib == 2:
                    out.extend(_signed(v >> s, 16) for s in (16, 0))
                else:
                    out.append(_signed(v, 32))
            else:
                dnib = v >> 30
                if nib == 2:
                    if dnib == 1:
                        out.append(_signed(v, 30))
                    elif dnib == 2:
                        out.extend(_signed(v >> s, 15) for s in (15, 0))
                    elif dnib == 3:
                        out.extend(_signed(v >> s, 10) for s in (20, 10, 0))
                    else:
                        raise ValueError("Steim2: bad dnib for nibble 2")
                else:
                    if dnib == 0:
                        out.extend(_signed(v >> s, 6) for s in (24, 18, 12, 6, 0))
                    elif dnib == 1:
                        out.extend(_signed(v >> s, 5) for s in (25, 20, 15, 10, 5, 0))
                    elif dnib == 2:
                        out.extend(_signed(v >> s, 4) for s in (24, 20, 16, 12, 8, 4, 0))
                    else:
                        raise ValueError("Steim2: bad dnib for nibble 3")
    if x0 is None:
        raise ValueError("Steim frame without integration constants")
    diffs = np.asarray(out[:n], dtype=np.int64)
    if diffs.size < n:
        raise ValueError(f"Steim: decoded {diffs.size} < {n} samples")
    x = np.empty(n, dtype=np.int64)
    x[0] = x0
    if n > 1:
        x[1:] = x0 + np.cumsum(diffs[1:])
    if int(x[-1]) != xn:
        raise ValueError(f"Steim reverse-integration check failed: {x[-1]} != {xn}")
    return x.astype(np.float64)


def read_mseed(buf: bytes) -> list[Trace]:
    traces: dict[tuple, Trace] = {}
    pos = 0
    while pos + 48 <= len(buf):
        hdr = buf[pos:pos + 48]
        if hdr[6:7] not in (b"D", b"R", b"Q", b"M"):
            raise ValueError(f"not a data record at offset {pos}")
        year_be = struct.unpack(">H", hdr[20:22])[0]
        bo = ">" if 1900 <= year_be <= 2100 else "<"
        sta, loc, cha, net = (hdr[8:13].decode().strip(), hdr[13:15].decode().strip(), hdr[15:18].decode().strip(), hdr[18:20].decode().strip())
        start = _btime(hdr, 20, bo)
        nsamp, factor, mult = struct.unpack(bo + "Hhh", hdr[30:36])
        nblk = hdr[39]
        tcorr, data_off, blk_off = struct.unpack(bo + "iHH", hdr[40:48])
        if not (hdr[36] & 0x02) and tcorr:
            start = start + np.timedelta64(int(tcorr) * 100, "us")
        enc, reclen = None, None
        off = blk_off
        for _ in range(nblk):
            btype, nxt = struct.unpack(bo + "HH", buf[pos + off:pos + off + 4])
            if btype == 1000:
                enc = buf[pos + off + 4]
                reclen = 1 << buf[pos + off + 6]
            if nxt == 0:
                break
            off = nxt
        if reclen is None:
            raise ValueError("record without blockette 1000; record length unknown")
        data = buf[pos + data_off:pos + reclen]
        if nsamp > 0:
            if enc in (10, 11):
                x = _steim(data, bo, nsamp, 1 if enc == 10 else 2)
            elif enc == 3:
                x = np.frombuffer(data[: 4 * nsamp], dtype=bo + "i4").astype(np.float64)
            elif enc == 1:
                x = np.frombuffer(data[: 2 * nsamp], dtype=bo + "i2").astype(np.float64)
            elif enc == 4:
                x = np.frombuffer(data[: 4 * nsamp], dtype=bo + "f4").astype(np.float64)
            elif enc == 5:
                x = np.frombuffer(data[: 8 * nsamp], dtype=bo + "f8")
            else:
                raise NotImplementedError(f"miniSEED encoding {enc}")
            key = (net, sta, loc, cha)
            tr = traces.get(key)
            rate = _rate(factor, mult)
            if tr is None:
                tr = traces[key] = Trace(net, sta, loc, cha, rate, start)
            tr.samples.append((start, x))
        pos += reclen
    return list(traces.values())


def trace_to_regular(tr: Trace, t0: np.datetime64, t1: np.datetime64) -> tuple[np.ndarray, float]:
    """Place a trace's records on a regular grid [t0, t1) at its own rate; NaN where no data."""
    dt_us = 1e6 / tr.rate
    n = int(round((t1 - t0) / np.timedelta64(1, "us") / dt_us))
    out = np.full(n, np.nan)
    for start, x in tr.samples:
        i0 = int(round((start - t0) / np.timedelta64(1, "us") / dt_us))
        a, b = max(i0, 0), min(i0 + x.size, n)
        if b > a:
            out[a:b] = x[a - i0:b - i0]
    return out, tr.rate
