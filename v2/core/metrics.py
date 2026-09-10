"""Quality metrics: CorrTrack (prediction) vs brute-force (ground truth).

The classified unit is a **correlated window-pair**: the key `(a, b)` where
`a=(series, time)`, `b=(series, time)` is already canonical (order imposed
upstream by `selection._order`). The sign of the correlation separates positive
from negative.

Reproduces the semantics of the v1 harness (`library_corrtrack_parallel.py`):
  * global precision/recall/f1 + pos/neg breakdowns;
  * specificity over the universe of the pairs tested by bf;
  * recall_min: was the most strongly correlated bf pair recovered;
  * aucroc / pr_auc: NaN (as in the v1 streaming path, no continuous score).
"""

import math
import os

NAN = float("nan")


def _safe_div(num, den):
    if num is None or den is None or den == 0:
        return NAN
    return num / den


def _pr_f1(tp, pred_total, gt_total):
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _signed_set(correlated):
    """{(a, b): sign} ; sign = +1 si corr > 0, -1 sinon."""
    return {(a, b): (1 if corr > 0 else -1) for a, b, corr, _ in correlated}


def compute_metrics(gt_correlated, pred_correlated, total_pairs_bf):
    """Return the block of metric columns (dict).

    gt_correlated   : list of (a,b,corr,lag) from brute-force (ground truth).
    pred_correlated : list of (a,b,corr,lag) from corrtrack.
    total_pairs_bf  : total number of pairs tested by bf (for specificity).
    """
    gt = _signed_set(gt_correlated)
    pred = _signed_set(pred_correlated)

    gt_keys, pred_keys = set(gt), set(pred)
    tp_keys = gt_keys & pred_keys
    tp = len(tp_keys)
    gt_total, pred_total = len(gt_keys), len(pred_keys)
    fp = max(pred_total - tp, 0)

    precision, recall, f1 = _pr_f1(tp, pred_total, gt_total)

    # sign split (positive / negative)
    gt_pos = sum(1 for s in gt.values() if s > 0)
    gt_neg = gt_total - gt_pos
    pred_pos = sum(1 for s in pred.values() if s > 0)
    pred_neg = pred_total - pred_pos
    tp_pos = sum(1 for k in tp_keys if gt[k] > 0 and pred[k] > 0)
    tp_neg = sum(1 for k in tp_keys if gt[k] <= 0 and pred[k] <= 0)
    precision_pos, recall_pos, f1_pos = _pr_f1(tp_pos, pred_pos, gt_pos)
    precision_neg, recall_neg, f1_neg = _pr_f1(tp_neg, pred_neg, gt_neg)

    # specificity: negative universe = pairs tested by bf that are not correlated
    negatives = max((total_pairs_bf or 0) - gt_total, 0)
    tn = max(negatives - fp, 0)
    specificity = tn / negatives if negatives else 0.0

    # recall_min: was the most strongly correlated bf pair recovered?
    recall_min = NAN
    if gt_correlated:
        hardest = max(gt_correlated, key=lambda e: abs(e[2]))
        recall_min = float((hardest[0], hardest[1]) in pred_keys)

    # "classic" recall: per distinct RELATION (id1, id2, lag), not per window
    # instance. This is v1's `recall_by_window=False` mode, equivalent to
    # `_normalize_pair_lag_key(id1, id2, lag)` = `tuple(sorted(ids)) + (lag,)`.
    # It answers: "did we discover that a relation A↔B exists at lag L
    # (regardless of how many windows materialize it)?"
    def _rel_keys(corr_list):
        return {(tuple(sorted([a[0], b[0]])) + (int(lag),))
                for a, b, _corr, lag in corr_list}
    gt_rel = _rel_keys(gt_correlated)
    pred_rel = _rel_keys(pred_correlated)
    tp_rel = len(gt_rel & pred_rel)
    recall_relation = tp_rel / len(gt_rel) if gt_rel else 0.0

    return {
        "precision_pos": precision_pos, "recall_pos": recall_pos, "f1_pos": f1_pos,
        "precision_neg": precision_neg, "recall_neg": recall_neg, "f1_neg": f1_neg,
        "precision": precision, "recall": recall, "specificity": specificity,
        "recall_min": recall_min, "recall_relation": recall_relation, "f1": f1,
        "aucroc": NAN, "pr_auc": NAN,
    }


# ----------------------------------------------------------------------------
# STREAMING variant (memory-frugal): same block of metrics, but reads the
# `correlated.csv` files from DISK and keeps only a compact representation
# (pair key packed into an int64). On dense datasets (ASOS 25 series / 10 years
# → 66M pairs), materializing two Python `_signed_set` (~10 GB each) overflows a
# 32 GB node. Here each pair = 1 int64 (key) + 1 int8 (sign) → ~600 MB for 66M
# pairs, independently of the number of runs compared.
#
# Key packing (id1, t1, id2, t2) over 62 bits (canonical order preserved,
# strictly identical to `_signed_set`, which keeps the key `(a, b)` as-is):
#   [ i1 : ID_BITS | t1 : T_BITS | i2 : ID_BITS | t2 : T_BITS ]
# The identifiers (airport codes, etc.) are interned str -> int (shared between
# the two files → same pair == same key). Capacity overflow = explicit
# ValueError (never a silent collision).

_ID_BITS = 11          # 2048 distinct series
_T_BITS = 20           # ~1M windows
_SH_I2 = _T_BITS
_SH_T1 = _T_BITS + _ID_BITS
_SH_I1 = 2 * _T_BITS + _ID_BITS
_ID_MAX = (1 << _ID_BITS) - 1
_T_MAX = (1 << _T_BITS) - 1
# RELATION key (id_min, id_max, lag) for recall_relation
_LAG_BITS = 22
_LAG_OFFSET = 1 << (_LAG_BITS - 1)     # signed lags ±2^21
_SH_RMAX = _LAG_BITS
_SH_RMIN = _LAG_BITS + _ID_BITS

_CORR_COLS = ["id1", "t1", "id2", "t2", "lag", "corr"]
_MISSED_SORT_CAP = 2_000_000   # beyond that, missed.csv is written UNSORTED (memory)


def _intern_ids(values, intern):
    """Map an array of identifiers (str) -> int indices through `intern` (a
    shared dict, filled as we go). Return an int64 ndarray."""
    import numpy as np
    uniq = [v for v in dict.fromkeys(values.tolist()) if v not in intern]
    for v in uniq:
        intern[v] = len(intern)
    idx = values.map(intern).to_numpy(dtype=np.int64)
    if idx.size and int(idx.max()) > _ID_MAX:
        raise ValueError(
            f"too many distinct series for the packing ({len(intern)} > "
            f"{_ID_MAX + 1}) — increase _ID_BITS in v2/core/metrics.py")
    return idx


def _load_compact(path, intern):
    """Read a correlated.csv in chunks -> compact representation.

    Returns a dict: keys (unique sorted int64), signs (aligned int8, sign of the
    first occurrence — like the `_signed_set` dict), rel (unique sorted int64),
    hardest (int64 key of the max-|corr| pair, or None when empty), n_pos/n_neg.
    """
    import numpy as np
    import pandas as pd

    key_parts, sign_parts, rel_parts = [], [], []
    hardest_key = None
    hardest_abs = -1.0
    if os.path.getsize(path) == 0:
        return {"keys": np.empty(0, np.int64), "signs": np.empty(0, np.int8),
                "rel": np.empty(0, np.int64), "hardest": None,
                "n_pos": 0, "n_neg": 0}
    for chunk in pd.read_csv(path, usecols=_CORR_COLS,
                             dtype={"t1": np.int64, "t2": np.int64,
                                    "lag": np.int64, "corr": np.float64},
                             chunksize=4_000_000):
        if chunk.empty:
            continue
        i1 = _intern_ids(chunk["id1"].astype(str), intern)
        i2 = _intern_ids(chunk["id2"].astype(str), intern)
        t1 = chunk["t1"].to_numpy(dtype=np.int64)
        t2 = chunk["t2"].to_numpy(dtype=np.int64)
        if t1.size and (int(t1.max()) > _T_MAX or int(t2.max()) > _T_MAX):
            raise ValueError(
                f"window index too large for the packing (>{_T_MAX}) — "
                f"increase _T_BITS in v2/core/metrics.py")
        keys = (i1 << _SH_I1) | (t1 << _SH_T1) | (i2 << _SH_I2) | t2
        corr = chunk["corr"].to_numpy(dtype=np.float64)
        signs = np.where(corr > 0, np.int8(1), np.int8(-1))
        # relation: (min(id), max(id), lag) — symmetric over the ids
        rmin = np.minimum(i1, i2)
        rmax = np.maximum(i1, i2)
        lag = chunk["lag"].to_numpy(dtype=np.int64) + _LAG_OFFSET
        if lag.size and (int(lag.min()) < 0 or int(lag.max()) >> _LAG_BITS):
            raise ValueError("lag out of range for the relation packing — "
                             "increase _LAG_BITS in v2/core/metrics.py")
        rel = (rmin << _SH_RMIN) | (rmax << _SH_RMAX) | lag

        key_parts.append(keys)
        sign_parts.append(signs)
        rel_parts.append(np.unique(rel))
        # most strongly correlated pair (recall_min)
        a = np.abs(corr)
        j = int(np.nanargmax(a)) if a.size else -1
        if j >= 0 and float(a[j]) > hardest_abs:
            hardest_abs = float(a[j])
            hardest_key = int(keys[j])

    if not key_parts:        # header-only file (0 pair)
        return {"keys": np.empty(0, np.int64), "signs": np.empty(0, np.int8),
                "rel": np.empty(0, np.int64), "hardest": None,
                "n_pos": 0, "n_neg": 0}
    keys = np.concatenate(key_parts)
    signs = np.concatenate(sign_parts)
    rel = np.unique(np.concatenate(rel_parts))
    # key uniqueness keeping the sign of the FIRST occurrence (semantics of the
    # `_signed_set` dict). Stable argsort -> the 1st of a group = 1st occurrence.
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    signs = signs[order]
    first = np.ones(keys.shape, dtype=bool)
    first[1:] = keys[1:] != keys[:-1]
    ukeys = keys[first]
    usigns = signs[first]
    n_pos = int((usigns > 0).sum())
    return {"keys": ukeys, "signs": usigns, "rel": rel, "hardest": hardest_key,
            "n_pos": n_pos, "n_neg": int(ukeys.size - n_pos)}


def _write_missed_streaming(gt_path, missed_keys, missed_path, intern, log=None):
    """Rewrite the missed pairs (keys `missed_keys`, sorted int64) by streaming
    gt_path. `intern` = the str-id -> int table used to pack `missed_keys` (every
    gt id is already in it). Sorts by |corr| desc when the total fits under
    _MISSED_SORT_CAP, otherwise writes UNSORTED (and says so) to preserve
    RAM."""
    import numpy as np
    import pandas as pd

    n = int(missed_keys.size)
    if n == 0:
        return 0
    os.makedirs(os.path.dirname(missed_path), exist_ok=True)
    sort_ok = n <= _MISSED_SORT_CAP
    if not sort_ok and log is not None:
        log.warning("missed.csv: %d missed pairs > %d → written UNSORTED "
                    "(preserves memory)", n, _MISSED_SORT_CAP)
    collected = []
    header_written = False
    for chunk in pd.read_csv(gt_path, usecols=_CORR_COLS,
                             dtype={"t1": np.int64, "t2": np.int64,
                                    "lag": np.int64, "corr": np.float64},
                             chunksize=4_000_000):
        if chunk.empty:
            continue
        # same packing as _load_compact through the shared intern (every gt id
        # is already interned → -1 should never show up).
        i1 = chunk["id1"].astype(str).map(intern).to_numpy(dtype=np.int64)
        i2 = chunk["id2"].astype(str).map(intern).to_numpy(dtype=np.int64)
        t1 = chunk["t1"].to_numpy(dtype=np.int64)
        t2 = chunk["t2"].to_numpy(dtype=np.int64)
        keys = (i1 << _SH_I1) | (t1 << _SH_T1) | (i2 << _SH_I2) | t2
        pos = np.searchsorted(missed_keys, keys)
        pos_c = np.clip(pos, 0, n - 1)
        hit = missed_keys[pos_c] == keys
        sub = chunk.loc[hit, _CORR_COLS]
        if sub.empty:
            continue
        if sort_ok:
            collected.append(sub)
        else:
            sub.to_csv(missed_path, mode="a", header=not header_written,
                       index=False)
            header_written = True
    if sort_ok and collected:
        out = pd.concat(collected, ignore_index=True)
        # |corr| desc sort through numpy (pandas' `sort_values` regresses on
        # some versions: IndexError inside nargsort). NaN -> last.
        absv = np.abs(out["corr"].to_numpy(dtype=np.float64))
        absv = np.where(np.isnan(absv), -np.inf, absv)
        order = np.argsort(absv, kind="stable")[::-1]
        out.iloc[order].to_csv(missed_path, columns=_CORR_COLS, index=False)
    return n


def compare_streaming(gt_path, pred_path, total_pairs_bf,
                      missed_path=None, log=None):
    """Disk/streaming equivalent of `compute_metrics` (+ writes missed.csv).

    Reads both `correlated.csv` from disk, keeping only an int64 key + an int8
    sign per pair. Returns `(metrics_dict, missed_count)` where metrics_dict has
    EXACTLY the same keys as `compute_metrics`.
    """
    import numpy as np

    intern = {}
    gt = _load_compact(gt_path, intern)
    pred = _load_compact(pred_path, intern)

    gk, gs = gt["keys"], gt["signs"]
    pk, ps = pred["keys"], pred["signs"]
    gt_total, pred_total = int(gk.size), int(pk.size)

    # intersection (unsigned keys) + signs aligned on the TPs
    if gt_total and pred_total:
        pos = np.searchsorted(gk, pk)
        pos_c = np.clip(pos, 0, gt_total - 1)
        match = gk[pos_c] == pk
        tp = int(match.sum())
        gt_sign_tp = gs[pos_c][match]
        pred_sign_tp = ps[match]
        tp_pos = int(((gt_sign_tp > 0) & (pred_sign_tp > 0)).sum())
        tp_neg = int(((gt_sign_tp <= 0) & (pred_sign_tp <= 0)).sum())
    else:
        tp = tp_pos = tp_neg = 0

    fp = max(pred_total - tp, 0)
    precision, recall, f1 = _pr_f1(tp, pred_total, gt_total)

    gt_pos, gt_neg = gt["n_pos"], gt["n_neg"]
    pred_pos, pred_neg = pred["n_pos"], pred["n_neg"]
    precision_pos, recall_pos, f1_pos = _pr_f1(tp_pos, pred_pos, gt_pos)
    precision_neg, recall_neg, f1_neg = _pr_f1(tp_neg, pred_neg, gt_neg)

    negatives = max((total_pairs_bf or 0) - gt_total, 0)
    tn = max(negatives - fp, 0)
    specificity = tn / negatives if negatives else 0.0

    # recall_min: was the most strongly correlated bf pair recovered?
    recall_min = NAN
    if gt["hardest"] is not None:
        hk = gt["hardest"]
        if pred_total:
            j = int(np.searchsorted(pk, hk))
            recall_min = float(j < pred_total and int(pk[j]) == hk)
        else:
            recall_min = 0.0

    # "per relation" recall (distinct id1,id2,lag)
    gr, pr_ = gt["rel"], pred["rel"]
    if gr.size:
        tp_rel = int(np.intersect1d(gr, pr_, assume_unique=True).size)
        recall_relation = tp_rel / gr.size
    else:
        recall_relation = 0.0

    metrics_dict = {
        "precision_pos": precision_pos, "recall_pos": recall_pos, "f1_pos": f1_pos,
        "precision_neg": precision_neg, "recall_neg": recall_neg, "f1_neg": f1_neg,
        "precision": precision, "recall": recall, "specificity": specificity,
        "recall_min": recall_min, "recall_relation": recall_relation, "f1": f1,
        "aucroc": NAN, "pr_auc": NAN,
    }

    missed_count = max(gt_total - tp, 0)
    if missed_path is not None and missed_count:
        # gt keys missing from pred (false negatives)
        if pred_total:
            pos = np.searchsorted(pk, gk)
            pos_c = np.clip(pos, 0, pred_total - 1)
            present = pk[pos_c] == gk
            missed_keys = gk[~present]
        else:
            missed_keys = gk
        _write_missed_streaming(gt_path, missed_keys, missed_path,
                                intern=intern, log=log)
    return metrics_dict, missed_count
