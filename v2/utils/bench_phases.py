"""PER-STEP comparison bench: plain vectorized vs sharded (w2/w4/w8).

Logs, for each dataset size, the wall time of every phase (sketch+insert /
select / validate / monitor / setup) and the **per-step speedup** against the
serial vectorized baseline — to see exactly where sharding wins (or does not).

    python -m v2.utils.bench_phases data.csv
    python -m v2.utils.bench_phases data.csv --series 5,25,100,200 --workers 2,4,8
    python -m v2.utils.bench_phases data.csv --index bst --window-size 48 --reps 3

Without `--series`, runs once on the whole dataset. With `--series`, it
sub-samples (`n_series`) to sweep the sizes.
"""

import argparse
import time

from ..core import pipeline
from ..core.config import Config
from ..core.log import get_logger

# grouping of the fine-grained v2 phases -> readable steps (same as the analysis)
PHASES = [
    ("sketch+insert", ("sketch", "index", "ingest")),
    ("select",        ("select",)),
    ("validate",      ("validate", "validate_sketches")),
    ("monitor",       ("monitor",)),
    ("setup",         ("read", "windows", "evict")),
]


def _group(timings):
    return {name: sum(timings.get(k, 0.0) for k in keys) for name, keys in PHASES}


def _run(csv, backend, workers, n_series, index, window_size, reps):
    """Meilleur-de-`reps` ; renvoie (wall, phases_dict, n_correlated)."""
    cfg = Config(window_size=window_size, index_backend=index, backend=backend,
                 workers=workers, n_series=n_series or 0, key_mode="truncate",
                 sketch_method="random_projection", output="", log_level="error")
    best, phases, nc = float("inf"), None, 0
    for _ in range(reps):
        t0 = time.perf_counter()
        r = pipeline.run(cfg, csv, mode="corrtrack", step="monitor")
        wl = time.perf_counter() - t0
        if wl < best:
            best, phases, nc = wl, _group(r["timings"]), r["n_correlated"]
    return best, phases, nc


def _log_compare(log, label, base_wall, base_ph, wall, ph, nc, ref_nc):
    """Log one line per phase: base -> variant (speedup), + total."""
    flag = "" if nc == ref_nc else f"  !! corr={nc} (ref {ref_nc})"
    log.info("  %-12s  total %7.3fs -> %7.3fs  (%.2fx)%s",
             label, base_wall, wall, base_wall / wall if wall else 0.0, flag)
    for name, _ in PHASES:
        b, v = base_ph[name], ph[name]
        sp = b / v if v > 1e-9 else float("inf")
        bar = "speedup" if sp >= 1.05 else ("~" if sp >= 0.95 else "SLOWER")
        log.info("       %-14s %7.3fs -> %7.3fs   %5.2fx  %s", name, b, v, sp, bar)


def main():
    p = argparse.ArgumentParser(description="Per-step bench: vectorized vs sharded.")
    p.add_argument("csv")
    p.add_argument("--series", default="", help="n_series list, e.g. 5,25,100,200 (def: whole dataset)")
    p.add_argument("--workers", default="2,4,8", help="list of sharded workers (def: 2,4,8)")
    p.add_argument("--index", default="bptree")
    p.add_argument("--backend", default="vectorized", help="compute base (def: vectorized)")
    p.add_argument("--window-size", type=int, default=48)
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()

    log = get_logger("info", "")
    sizes = [int(s) for s in args.series.split(",") if s] or [0]
    workers = [int(w) for w in args.workers.split(",") if w]

    for ns in sizes:
        tag = f"{ns} series" if ns else "whole dataset"
        log.info("=== %s | index=%s backend=%s window=%d reps=%d ===",
                 tag, args.index, args.backend, args.window_size, args.reps)
        # baseline: plain vectorized (serial)
        b_wall, b_ph, ref_nc = _run(args.csv, args.backend, 0, ns,
                                    args.index, args.window_size, args.reps)
        log.info("  baseline (%s serial): %.3fs | %s | corr=%d", args.backend, b_wall,
                 " ".join(f"{n}={b_ph[n]:.3f}" for n, _ in PHASES), ref_nc)
        for w in workers:
            wall, ph, nc = _run(args.csv, f"sharded_{args.backend}", w, ns,
                                args.index, args.window_size, args.reps)
            _log_compare(log, f"sharded w{w}", b_wall, b_ph, wall, ph, nc, ref_nc)


if __name__ == "__main__":
    main()
