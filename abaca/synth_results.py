"""Loader for the synthetic OFAT campaign (abaca/campaign_synth_overnight.py), shared by the
table and figure builders (2026-09-28).

One cell directory per (base process, m, L, T, target density, space); the run name carries the
generator request and the dataset profile inside the JSON carries what the data actually is, so
the effective density is read from the run, never from the request.
"""
from __future__ import annotations

import glob
import json
import os
import re

NAME = re.compile(r"^ovn_(?P<proc>[a-z0-9]+)_m(?P<m>\d+)_L(?P<L>\d+)_T(?P<T>[0-9p]+)_d(?P<d>[0-9p]+)(?P<diff>_diff)?_m\d+_W")
import method_style as ms

ORDER = ms.ARMS          # the arms and their labels, shared with the campaign figures
AXES = ("m", "L", "T", "density")
BASE = {"m": 500, "L": 6, "T": 0.9, "density": 0.01}


def _num(s):
    return float(s.replace("p", "."))


def specificity(arm, bf):
    """Reported-set specificity, 1 - FP/(U-P), with FP = reported - recall*P (exact, not a bound)."""
    U = bf.get("total_candidates") or bf.get("tested") or 0
    P = bf.get("correlated") or 0
    if arm.get("recall") is None or U <= P:
        return None
    return 1.0 - max((arm.get("correlated") or 0) - arm["recall"] * P, 0.0) / (U - P)


def load(root):
    """[cell] for every finished run under <root>/nway, newest metrics as recorded in the job."""
    cells = []
    for p in sorted(glob.glob(os.path.join(root, "nway", "*", "nway.json"))):
        name = os.path.basename(os.path.dirname(p))
        m = NAME.match(name)
        if not m:
            continue
        if not ms.kept_T(_num(m["T"])):
            continue
        d = json.load(open(p))
        prof, bf = d.get("dataset_profile") or {}, d["arms"].get("bruteforce") or {}
        if not bf.get("wall"):
            continue
        L = int(m["L"])
        cell = dict(run=name, proc=m["proc"], m=int(m["m"]), L=L, T=_num(m["T"]),
                    density=_num(m["d"]), diff=bool(m["diff"]),
                    space="differenced" if m["diff"] else "raw",
                    effective=prof.get("density_at_threshold"),
                    pair_windows=prof.get("pair_windows"), n_windows=prof.get("n_windows"),
                    bf_wall=bf["wall"], arms={})
        for key, label in ORDER:
            a = d["arms"].get(key)
            if not isinstance(a, dict) or a.get("status") != "ok" or not a.get("wall"):
                cell["arms"][label] = None
                continue
            if not ms.keeps(a.get("supports_lags") if L > 1 else "native") or not ms.kept_arm(label):
                cell["arms"][label] = None
                continue
            cell["arms"][label] = dict(wall=a["wall"], speedup=bf["wall"] / a["wall"],
                                       recall=a.get("recall"), precision=a.get("precision"),
                                       specificity=specificity(a, bf),
                                       supports_lags=a.get("supports_lags"),
                                       supports_neg_corr=a.get("supports_neg_corr"))
        cells.append(cell)
    return cells


def on_axis(cells, axis, proc, space):
    """The OFAT slice: every cell that differs from BASE in <axis> only, sorted by the axis."""
    out = [c for c in cells if c["proc"] == proc and c["space"] == space
           and all(c[k] == v for k, v in BASE.items() if k != axis)]
    return sorted(out, key=lambda c: c[axis])
