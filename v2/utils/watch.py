"""Live watch of a pipeline run — table refreshed every N seconds.

Displays the SAME table as `v2.pipeline` does at the end, but live while a
pipeline is running. Reads the `result.json` of each run as it appears in
`results/<name>/<run>/`.

For RUNNING runs (✓/⏳ not finished yet), parses the last `CP: N/M ... ETA T`
line of `run.log` to show progress / throughput / ETA live — no need to wait for
the end of the run to see it advance.

Per-run statuses:
  ✓  : finished (`result.json` present → full metrics row)
  ⏳ : running (run.log parsed → progress bar + win/s + ETA)
  ·  : pending (not started yet)

Usage:
    python3 -m v2.utils.watch path/to/pipeline.json
    python3 -m v2.utils.watch path/to/pipeline.json --interval 5

Ctrl-C to quit.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time

from ..core import benchmark, metrics
from ..core.config import Config
from ..pipeline import (
    _ansi, _color_score, _color_speedup, _norm_params, _safe_float,
    load_jsonc, load_result,
)


# Regex of the checkpoint line in run.log (see core/pipeline.py:180):
# [INFO] [HH:MM:SS]   CP: 142/1000 (t=...) | back=python | cand=12345 corr=67 \
#         | 28.3 win/s 12450 cand/s | cum: sk=… | Δ: … | ETA 1m23s
_CP_RE = re.compile(
    r"CP:\s*(\d+)/(\d+)\s.*?\|\s*cand=(\d+)\s+corr=(\d+)\s*\|\s*"
    r"([\d.]+)\s*win/s\s+([\d.]+)\s*cand/s.*?\|\s*ETA\s+(\S+)"
)


def _parse_progress(run_dir):
    """Find the LAST 'CP: ...' line in run.log and parse it.

    Returns a dict {w_idx, w_total, n_cand, n_corr, win_per_s, cand_per_s, eta}
    or None when there is no checkpoint yet.
    """
    log_path = os.path.join(run_dir, "run.log")
    if not os.path.exists(log_path):
        return None
    try:
        # Only the last ~16 KB are read (the recent checkpoints). Far faster
        # than scanning the whole file on long runs.
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    matches = list(_CP_RE.finditer(tail))
    if not matches:
        return None
    m = matches[-1]
    return {
        "w_idx": int(m.group(1)),
        "w_total": int(m.group(2)),
        "n_cand": int(m.group(3)),
        "n_corr": int(m.group(4)),
        "win_per_s": float(m.group(5)),
        "cand_per_s": float(m.group(6)),
        "eta": m.group(7),
    }


def _f(v, fmt="{:.3f}"):
    try:
        x = float(v)
        return "nan" if x != x else fmt.format(x)
    except (TypeError, ValueError):
        return str(v)


def _fphase(v):
    x = _safe_float(v)
    return "    —  " if x is None else f"{x:.2f}x"


def _build_row(run_spec, base_params, baseline_res, base_dir, opt_time_default=0.0):
    """Load the run's `result.json` and build the row (dict) matching the
    pipeline format. Returns (row, status) — row=None when there is no result
    yet."""
    name = run_spec.get("name")
    if name is None:
        return None, "·"
    run_dir = os.path.join(base_dir, name)
    if not os.path.isdir(run_dir):
        return None, "·"
    try:
        res = load_result(run_dir)             # ignores signature mismatch
    except (json.JSONDecodeError, ValueError):
        res = None
    if res is None:
        # running run: try to parse run.log for the progress
        prog = _parse_progress(run_dir)
        return prog, "⏳"
    if baseline_res is None:
        # no baseline to compare against → the raw fields are shown anyway
        cls = run_spec.get("mode", "corrtrack")
        params = _norm_params({**(base_params or {}), **(run_spec.get("params") or {})})
        cfg = Config.build(**params)
        # build_record requires a reference dict; a minimal baseline is
        # simulated by borrowing this one → speedup=1, self-consistent
        # recall/prec.
        row = benchmark.build_record(cfg, name, res, res, alg=cls)
    else:
        cls = run_spec.get("mode", "corrtrack")
        params = _norm_params({**(base_params or {}), **(run_spec.get("params") or {})})
        cfg = Config.build(**params)
        row = benchmark.build_record(cfg, name, baseline_res, res, alg=cls)

    # opt_time: read back from result.json when possible (written by save_result)
    try:
        with open(os.path.join(run_dir, "result.json")) as f:
            rj = json.load(f)
        row["opt_time"] = float(rj.get("opt_time", opt_time_default) or 0.0)
    except Exception:
        row["opt_time"] = opt_time_default

    # per-phase speedups (idem pipeline.py)
    sk_bf = float(row.get("sk_time_bf", 0) or 0)
    row["sk_speedup"] = (metrics._safe_div(sk_bf, row.get("sk_time", 0))
                         if sk_bf > 0 else float("nan"))
    row["cand_speedup"] = metrics._safe_div(row.get("cand_time_bf", 0),
                                             row.get("cand_time", 0))
    row["val_speedup"] = metrics._safe_div(row.get("val_time_bf", 0),
                                            row.get("val_time", 0))
    row["monit_speedup"] = metrics._safe_div(row.get("monit_time_bf", 0),
                                              row.get("monit_time", 0))
    cwp = metrics._safe_div(row.get("cand_w", 0), row.get("cand_w_bf", 0))
    cop = metrics._safe_div(row.get("corr_w", 0), row.get("corr_w_bf", 0))
    row["cand_w_pct"] = 100.0 * cwp if cwp == cwp else cwp
    row["corr_w_pct"] = 100.0 * cop if cop == cop else cop
    try:
        row["total_time"] = float(row["runtime"]) + float(row.get("opt_time", 0))
    except (TypeError, ValueError):
        row["total_time"] = row.get("runtime", 0)
    row.setdefault("missed", "")
    row.setdefault("recall_relation", "")
    return row, "✓"


# ============================ GRID view =====================================

def _cell_label(run_name):
    """Compact cell name: strips `corrtrack_` and the backend suffix (both
    redundant, since cells are grouped per section)."""
    s = run_name
    for prefix in ("corrtrack_", "filcorr_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # strip the longest backend suffixes first
    for be in ("vectorized_parallel", "cython_parallel", "mps", "cython",
               "vectorized", "parallel"):
        if s.endswith("_" + be):
            s = s[:-len(be) - 1]
            break
    return s


def _cell_classify(run_spec):
    """Retourne (backend, sketch_method) pour regrouper en sections."""
    p = run_spec.get("params", {}) or {}
    mode = run_spec.get("mode", "corrtrack")
    be = p.get("backend") or p.get("filcorr_backend") or "python"
    if mode == "bf":
        return ("__ref__", "bf") if run_spec.get("name") == "bf_python" else (be, "bf")
    if mode == "filcorr":
        return (be, "filcorr")
    sk = p.get("sketch_method", "random_projection")
    return (be, sk)


def _cell_format(run_name, row, status, width):
    """Rend une cellule (2 lignes) de largeur `width`. Renvoie list[str]."""
    label = _cell_label(run_name)
    label_w = max(8, width - 3)
    label = (label[:label_w - 1] + "…") if len(label) > label_w else label

    if status == "·":
        ico = _ansi(" ·", "2;37")
        l1 = f"{ico} {label:<{label_w}}"
        l2 = _ansi(f"   {'(pending)':<{label_w}}", "2;37")
        return [l1, l2]

    if status == "⏳" and row is None:
        ico = _ansi(" ⏳", "1;33")
        l1 = f"{ico} {label:<{label_w}}"
        l2 = _ansi(f"   {'(starting…)':<{label_w}}", "33")
        return [l1, l2]

    if status == "⏳":
        # row is a progress dict {w_idx, w_total, ...}
        p = row
        pct = (p["w_idx"] / p["w_total"]) if p["w_total"] else 0
        mini_w = min(10, width - 18)
        if mini_w < 4:
            mini_w = 4
        fill = int(mini_w * pct)
        bar = "█" * fill + "░" * (mini_w - fill)
        ico = _ansi(" ⏳", "1;33")
        l1 = f"{ico} {label:<{label_w}}"
        info = f"  {bar} {pct*100:4.1f}% ETA{p['eta']}"
        # truncate when too long
        info = info[:width]
        l2 = _ansi(f"{info:<{width}}", "33")
        return [l1, l2]

    # ✓ finished
    ico = _ansi(" ✓", "1;32")
    l1 = f"{ico} {label:<{label_w}}"
    # line 2: recall + speedup + runtime
    rec = row.get("recall", 0)
    spd = row.get("speedup", 0)
    rt = row.get("runtime", 0)
    rec_str = _ansi(f"r={_f(rec,'{:.2f}')}", _color_score(rec))
    spd_str = _ansi(f"{_f(spd,'{:.1f}')}x", _color_speedup(spd))
    rt_str = f"{_f(rt,'{:.1f}')}s"
    raw = f"   {rec_str}  {spd_str}  ⏱ {rt_str}"
    # padding (en tenant compte des codes ANSI invisibles)
    visible = f"   r={_f(rec,'{:.2f}')}  {_f(spd,'{:.1f}')}x  ⏱ {rt_str}"
    pad = max(0, width - len(visible))
    return [l1, raw + " " * pad]


def _render_grid(rows_status, runs_spec, name, base_dir, interval):
    """Grid view, grouped by (backend, sketch_method)."""
    term_w = shutil.get_terminal_size((140, 40)).columns
    cell_w = 32                                 # width of a cell
    n_cols = max(1, term_w // (cell_w + 1))     # +1 for the separator

    done = sum(1 for _, _, st in rows_status if st == "✓")
    running = sum(1 for _, _, st in rows_status if st == "⏳")
    pending = sum(1 for _, _, st in rows_status if st == "·")
    total = len(rows_status)
    pct = (done / total * 100) if total else 0.0
    bar_w = 30
    fill = int(bar_w * done / total) if total else 0
    bar = "█" * fill + "░" * (bar_w - fill)

    out = []
    ts = time.strftime("%H:%M:%S")
    out.append(f"[watch {ts}] {name}: \x1b[1;32m{done}\x1b[0m/{total} done "
               f"(⏳{running} ·{pending}) [{bar}] {pct:5.1f}%  "
               f"→ {base_dir}  ({n_cols} cols × {cell_w} chars, refresh {interval}s)")
    out.append("")

    # grouping by (backend, sketch_method) while preserving the JSON order
    sections = []  # list of (header, [(run_name, row, status), ...])
    sec_index = {}
    for (run_name, row, status), spec in zip(rows_status, runs_spec):
        sec = _cell_classify(spec)
        if sec not in sec_index:
            sec_index[sec] = len(sections)
            sections.append((sec, []))
        sections[sec_index[sec]][1].append((run_name, row, status))

    # stable order: reference first, filcorr last, the rest in between
    def _sec_rank(sec_be_sk):
        be, sk = sec_be_sk
        if be == "__ref__": return (0, "", "")
        if sk == "filcorr": return (3, be, "")
        if sk == "bf":      return (1, be, "")
        return (2, be, sk)
    sections.sort(key=lambda s: _sec_rank(s[0]))

    for (be, sk), cells in sections:
        # section header
        if be == "__ref__":
            header = "  ─── REFERENCE (baseline) ───"
        elif sk == "bf":
            header = f"  ─── bf · backend={be} ───"
        elif sk == "filcorr":
            header = f"  ─── filcorr · backend={be} ───"
        else:
            n_done = sum(1 for _, _, s in cells if s == "✓")
            header = (f"  ─── backend={be} · sketch={sk}  "
                      f"({n_done}/{len(cells)}) ───")
        out.append(_ansi(header, "1;36"))

        # cellules par ligne
        for i in range(0, len(cells), n_cols):
            row_cells = cells[i:i + n_cols]
            formatted = [_cell_format(rn, r, s, cell_w) for (rn, r, s) in row_cells]
            # line-by-line composition (2 lines per cell)
            for li in range(2):
                line = "  ".join(c[li] for c in formatted)
                out.append("  " + line)
        out.append("")

    return "\n".join(out)


# ============================ SYNTH view =====================================
# Optimized for hundreds of runs. 4 panels: heatmap matrix per
# (backend × sketch_method), list of running runs (5 most advanced), top 10
# Pareto best (recall × speedup), global stats.

def _render_synth(rows_status, runs_spec, name, base_dir, interval):
    done = sum(1 for _, _, st in rows_status if st == "✓")
    running = sum(1 for _, _, st in rows_status if st == "⏳")
    pending = sum(1 for _, _, st in rows_status if st == "·")
    total = len(rows_status)
    pct = (done / total * 100) if total else 0.0

    # construit l'index : (backend, sketch_method) -> list of (run_name, row, status)
    matrix = {}
    for (run_name, row, status), spec in zip(rows_status, runs_spec):
        be, sk = _cell_classify(spec)
        if be == "__ref__" or sk in ("bf", "filcorr"):
            continue              # excluded from the main matrix
        matrix.setdefault((be, sk), []).append((run_name, row, status))
    backends = sorted({be for (be, _) in matrix})
    sketches = ["random_projection", "fft_lowpass", "fft_topk",
                "median_blocks", "median_phase"]
    sk_short = {"random_projection": "RP    ", "fft_lowpass": "FFT_lp",
                "fft_topk": "FFT_tk", "median_blocks": "MED_bl",
                "median_phase": "MED_ph"}

    out = []
    ts = time.strftime("%H:%M:%S")
    bar_w = 40
    fill = int(bar_w * done / total) if total else 0
    bar = "█" * fill + "░" * (bar_w - fill)
    out.append(f"[watch {ts}] \x1b[1m{name}\x1b[0m  →  {base_dir}")
    out.append(f"  \x1b[1;32m✓ {done}\x1b[0m  "
               f"\x1b[1;33m⏳ {running}\x1b[0m  "
               f"\x1b[2;37m· {pending}\x1b[0m  /  {total} total   "
               f"[{bar}] {pct:5.1f}%  (refresh {interval}s)")
    out.append("")

    # ─────── 1. Matrix heatmap (backend × sketch_method) ───────
    # Cellule = "████░░ N/T" ; largeur visible 11 chars.
    out.append(_ansi("  ─── MATRICE  backend × sketch_method  ───", "1;36"))
    CELL_W = 11
    header_cells = "  ".join(f"{sk_short[s]:<{CELL_W}}" for s in sketches)
    out.append(f"  {'backend':<22} | {header_cells}")
    out.append("  " + "─" * 22 + "─┼─" + "─" * len(header_cells))
    for be in backends:
        cells = []
        for sk in sketches:
            cell = matrix.get((be, sk), [])
            if not cell:
                cells.append(_ansi(f"{'—':<{CELL_W}}", "2;37"))
                continue
            n_done = sum(1 for _, _, s in cell if s == "✓")
            n_run = sum(1 for _, _, s in cell if s == "⏳")
            n_tot = len(cell)
            mini_w = 5
            n_fill = int(mini_w * n_done / n_tot)
            mini = "█" * n_fill + "░" * (mini_w - n_fill)
            if n_done == n_tot:    col = "1;32"
            elif n_run > 0:         col = "1;33"
            elif n_done > 0:        col = "32"
            else:                   col = "2;37"
            # texte 11 chars visibles : "█████ 25/25" (5+1+5)
            plain = f"{mini} {n_done:>2}/{n_tot:<2}"
            plain = plain.ljust(CELL_W)[:CELL_W]
            cells.append(_ansi(plain, col))
        out.append(f"  {be:<22} | " + "  ".join(cells))
    out.append("")

    # ─────── 2. Bf / filcorr (reference + alternative) ───────
    refs = [(rn, r, st) for (rn, r, st), spec in zip(rows_status, runs_spec)
            if spec.get("mode") in ("bf", "filcorr")]
    if refs:
        out.append(_ansi("  ─── REFERENCE & FILCORR ───", "1;36"))
        for rn, row, st in refs[:6]:
            ico = (_ansi(" ✓", "1;32") if st == "✓" else
                   _ansi(" ⏳", "1;33") if st == "⏳" else
                   _ansi(" ·", "2;37"))
            if st == "✓" and row is not None:
                info = (f"  r={_f(row.get('recall',0),'{:.2f}')}  "
                        f"spd={_f(row.get('speedup',0),'{:.1f}')}x  "
                        f"⏱ {_f(row.get('runtime',0),'{:.1f}')}s")
            else:
                info = "  (en attente/cours)"
            out.append(f"  {ico} {rn:<40}{info}")
        out.append("")

    # ─────── 3. En cours (top 5 par avancement) ───────
    in_progress = [(rn, r, st) for (rn, r, st) in rows_status
                   if st == "⏳" and r is not None]
    in_progress.sort(key=lambda x: -(x[1].get("w_idx", 0) / max(1, x[1].get("w_total", 1))))
    if in_progress:
        out.append(_ansi(f"  ─── EN COURS ({len(in_progress)}) ───", "1;33"))
        for rn, p, _ in in_progress[:8]:
            short = _cell_label(rn)
            pct_run = (p["w_idx"] / p["w_total"]) if p["w_total"] else 0
            mini_w = 14
            fr = int(mini_w * pct_run)
            mb = "█" * fr + "░" * (mini_w - fr)
            out.append(f"  ⏳ {short:<32} [{mb}] {pct_run*100:5.1f}%  "
                       f"{p['win_per_s']:>5.1f} win/s  cand={p['n_cand']:>6}  "
                       f"ETA {p['eta']}")
        if len(in_progress) > 8:
            out.append(_ansi(f"     ... +{len(in_progress) - 8} autres en cours",
                             "2;37"))
        out.append("")

    # ─────── 4. Top 10 done (par recall × speedup, exclut bf) ───────
    done_rows = [(rn, r) for (rn, r, st) in rows_status
                 if st == "✓" and r is not None and not rn.startswith("bf_")]
    def _score(rn_r):
        r = rn_r[1]
        rec = r.get("recall", 0) or 0
        spd = r.get("speedup", 0) or 0
        return rec * max(0.0, spd)            # product; no penalty when spd<1
    done_rows.sort(key=_score, reverse=True)
    if done_rows:
        out.append(_ansi(f"  ─── TOP 10 (recall × speedup) ───", "1;32"))
        out.append(f"  {'#':<2} {'run':<40}{'recall':>8}{'spd':>8}{'⏱':>9}")
        for i, (rn, r) in enumerate(done_rows[:10], 1):
            rec = r.get("recall", 0)
            spd = r.get("speedup", 0)
            rt = r.get("runtime", 0)
            rec_c = _ansi(f"{_f(rec,'{:.3f}'):>8}", _color_score(rec))
            spd_c = _ansi(f"{_f(spd,'{:.2f}')+'x':>8}", _color_speedup(spd))
            out.append(f"  {i:<2} {rn:<40}{rec_c}{spd_c}"
                       f"{_f(rt,'{:.1f}')+'s':>9}")
        out.append("")

    # ─────── 5. Stats globales ───────
    if done_rows:
        recalls = [r.get("recall", 0) for _, r in done_rows
                   if r.get("recall") is not None]
        speedups = [r.get("speedup", 0) for _, r in done_rows
                    if r.get("speedup") is not None]
        runtimes = [r.get("runtime", 0) for _, r in done_rows
                    if r.get("runtime") is not None]
        out.append(_ansi("  ─── STATS (over the ✓ done) ───", "1;36"))
        if recalls:
            out.append(f"  recall   : moy={sum(recalls)/len(recalls):.3f}  "
                       f"min={min(recalls):.3f}  max={max(recalls):.3f}")
        if speedups:
            out.append(f"  speedup  : moy={sum(speedups)/len(speedups):.2f}x  "
                       f"min={min(speedups):.2f}x  max={max(speedups):.2f}x")
        if runtimes:
            out.append(f"  runtime  : mean={sum(runtimes)/len(runtimes):.1f}s  "
                       f"min={min(runtimes):.1f}s  max={max(runtimes):.1f}s")

    return "\n".join(out)


def _render(rows_status, name, base_dir, interval):
    """Build the complete string to print (with colors)."""
    done = sum(1 for _, _, st in rows_status if st == "✓")
    running = sum(1 for _, _, st in rows_status if st == "⏳")
    pending = sum(1 for _, _, st in rows_status if st == "·")
    total = len(rows_status)
    pct = (done / total * 100) if total else 0.0

    # progress bar 30 chars
    bar_w = 30
    filled = int(bar_w * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_w - filled)

    out = []
    ts = time.strftime("%H:%M:%S")
    out.append(f"[watch {ts}] {name}: \x1b[1;32m{done}\x1b[0m/{total} done "
               f"(⏳{running} ·{pending}) [{bar}] {pct:5.1f}%  "
               f"→ {base_dir}  (refresh {interval}s, Ctrl-C pour quitter)")

    # largeur dynamique du nom
    w = max([len(rn) for rn, _, _ in rows_status] + [len("run")])
    head = (f"{'':<3}{'run':<{w}}{'runtime':>10}{'speedup':>9}{'opt_t':>8}"
            f"{'total':>10}{'sk_t':>7}{'sk_x':>7}{'cand_t':>8}{'cand_x':>7}"
            f"{'val_t':>9}{'val_x':>7}{'monit_t':>9}{'monit_x':>7}"
            f"{'cand_w':>11}{'cand%':>7}{'tested_w':>10}{'corr_w':>9}{'corr%':>7}"
            f"{'missed':>8}{'recall':>8}{'rec_rel':>8}{'prec':>7}{'spec':>7}")
    out.append(head)

    for run_name, row, status in rows_status:
        # colored status icon
        if status == "✓":
            ico = _ansi(" ✓ ", "1;32")
        elif status == "⏳":
            ico = _ansi(" ⏳", "1;33")
        else:
            ico = _ansi(" · ", "2;37")

        if status == "·" or (status == "⏳" and row is None):
            tag = "(pending…)" if status == "·" else "(starting…)"
            out.append(f"{ico}{run_name:<{w}}    {tag}")
            continue
        if status == "⏳":
            # row holds a progress dict {w_idx, w_total, ...}
            p = row
            pct = (p["w_idx"] / p["w_total"]) if p["w_total"] else 0
            mini_w = 16
            mini_bar = "█" * int(mini_w * pct) + "░" * (mini_w - int(mini_w * pct))
            out.append(
                f"{ico}{run_name:<{w}}  "
                f"\x1b[33m{p['w_idx']:>6}/{p['w_total']:<6}\x1b[0m "
                f"[{mini_bar}] {pct*100:5.1f}%  "
                f"\x1b[36m{p['win_per_s']:>6.1f} win/s\x1b[0m  "
                f"cand={p['n_cand']:>7}  corr={p['n_corr']:>5}  "
                f"\x1b[35mETA {p['eta']}\x1b[0m")
            continue

        spd  = _ansi(f"{_f(row['speedup'],'{:.2f}')+'x':>9}",  _color_speedup(row['speedup']))
        sk_x = _ansi(f"{_fphase(row['sk_speedup']):>7}",       _color_speedup(row['sk_speedup']))
        cd_x = _ansi(f"{_fphase(row['cand_speedup']):>7}",     _color_speedup(row['cand_speedup']))
        vl_x = _ansi(f"{_fphase(row['val_speedup']):>7}",      _color_speedup(row['val_speedup']))
        mn_x = _ansi(f"{_fphase(row['monit_speedup']):>7}",    _color_speedup(row['monit_speedup']))
        rec  = _ansi(f"{_f(row['recall']):>8}",                _color_score(row['recall']))
        rrel = _ansi(f"{_f(row.get('recall_relation','')):>8}", _color_score(row.get('recall_relation','')))
        prec = _ansi(f"{_f(row['precision']):>7}",             _color_score(row['precision']))
        spc  = _ansi(f"{_f(row['specificity']):>7}",           _color_score(row['specificity']))
        out.append(
            f"{ico}{run_name:<{w}}{_f(row['runtime'],'{:.2f}')+'s':>10}"
            f"{spd}{_f(row['opt_time'],'{:.1f}')+'s':>8}"
            f"{_f(row['total_time'],'{:.2f}')+'s':>10}"
            f"{_f(row['sk_time'],'{:.2f}'):>7}{sk_x}"
            f"{_f(row['cand_time'],'{:.2f}'):>8}{cd_x}"
            f"{_f(row['val_time'],'{:.2f}'):>9}{vl_x}"
            f"{_f(row['monit_time'],'{:.3f}'):>9}{mn_x}"
            f"{_f(row['cand_w'],'{:.0f}'):>11}{_f(row['cand_w_pct'],'{:.1f}')+'%':>7}"
            f"{_f(row['tested_w'],'{:.0f}'):>10}"
            f"{_f(row['corr_w'],'{:.0f}'):>9}{_f(row['corr_w_pct'],'{:.1f}')+'%':>7}"
            f"{_f(row['missed'],'{:.0f}'):>8}"
            f"{rec}{rrel}{prec}{spc}")

    return "\n".join(out)


def watch(config_path, interval=3.0, once=False, view="auto"):
    spec = load_jsonc(config_path)
    name = spec.get("name", "pipeline")
    base_dir = os.path.join(spec.get("output", "results"), name)
    runs = spec.get("runs", [])
    base_params = spec.get("params", {})
    bidx = spec.get("baseline", 0)
    if not runs:
        print(f"[watch] config {config_path} has no 'runs' — nothing to watch.")
        return

    # baseline name (tries spec.baseline, then bf_python, then the first bf)
    if 0 <= bidx < len(runs):
        baseline_name = runs[bidx].get("name")
    else:
        baseline_name = next((r.get("name") for r in runs
                              if r.get("mode") == "bf"), runs[0].get("name"))

    # Alternate screen buffer (like htop/less): no scrollback pollution, and on
    # exit the terminal goes back to its previous state.
    # \x1b[?1049h : enter alt buffer
    # \x1b[?25l   : hide cursor
    # \x1b[H      : home (no clear; we clear "on the fly" at each refresh)
    # \x1b[J      : clear from cursor to end of screen
    is_tty = sys.stdout.isatty()
    use_alt = is_tty and not once
    if use_alt:
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()

    try:
        while True:
            # baseline result (may not be ready at the beginning)
            baseline_dir = os.path.join(base_dir, baseline_name)
            try:
                baseline_res = load_result(baseline_dir)
            except Exception:
                baseline_res = None

            rows_status = []
            for r in runs:
                row, st = _build_row(r, base_params, baseline_res, base_dir)
                rows_status.append((r.get("name") or "?", row, st))

            # view: synth (default, optimized for large JSON, keeps the live
            # progress in the RUNNING section), grid (detailed grid, 1 cell per
            # run), table (1 row per run, the former view).
            actual_view = view
            if actual_view == "auto":
                actual_view = "synth"          # default: synthetic view
            if actual_view == "synth":
                text = _render_synth(rows_status, runs, name, base_dir, interval)
            elif actual_view == "grid":
                text = _render_grid(rows_status, runs, name, base_dir, interval)
            else:
                text = _render(rows_status, name, base_dir, interval)
            if use_alt:
                # home, write, then clear-to-end to erase whatever sticks out
                # of the previous render (otherwise a tail-of-frame could
                # remain when the new frame is shorter).
                sys.stdout.write("\x1b[H" + text + "\x1b[J\n")
            else:
                sys.stdout.write(text + "\n")
            sys.stdout.flush()

            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if use_alt:
            # show cursor + leave alt buffer (restore previous terminal content)
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
        sys.stdout.write("[watch] stopped.\n")
        sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(
        description="Live watch of a pipeline (table refreshed every N seconds).")
    p.add_argument("config", help="Path of the pipeline JSON file.")
    p.add_argument("--interval", type=float, default=3.0,
                   help="Refresh interval in seconds (default 3).")
    p.add_argument("--once", action="store_true",
                   help="Print a single snapshot then exit.")
    p.add_argument("--view", choices=["auto", "synth", "grid", "table"],
                   default="auto",
                   help="Layout: `synth` (default, synthetic view: matrix + "
                        "running + top + stats), `grid` (1 cell per run, "
                        "grouped by backend×sketch), `table` (1 row per run, "
                        "the former view), `auto` = synth.")
    args = p.parse_args()
    watch(args.config, interval=args.interval, once=args.once, view=args.view)


if __name__ == "__main__":
    main()
