"""Benchmark launcher: compares CorrTrack to brute-force and emits stats rows.

Runs both modes on the same CSV and produces one row per config (~70 columns,
`;`-separated, v1 harness format): per-phase timings (bf + corrtrack), speedup,
validation waste, and precision/recall/f1 vs brute-force (ground truth).

Every flag accepts a comma-separated list → sweep (cartesian product); the
brute-force is recomputed only when a parameter that affects it changes.

    python -m v2.utils.bench data.csv --dataset-id fr_air_5_6 --stats-output stats.csv
    python -m v2.utils.bench data.csv --cell-size 0.25,0.5,1.0 --grid-dimension 1,2
    python -m v2.utils.bench data.csv          # header + row(s) on stdout
"""

import os

from ..core import benchmark
from ..core.cli import clean_output_dir, parse_bench
from ..core.config import Config


def main():
    csv, dataset_id, stats_output, axes = parse_bench()
    out_dir = axes["output"][0] if "output" in axes else Config().output
    clean = axes["clean"][0] if "clean" in axes else Config().clean
    clean_output_dir(out_dir, clean)
    if stats_output is None and out_dir:
        stats_output = os.path.join(out_dir, "stats.csv")
    rows = benchmark.sweep(axes, csv, dataset_id=dataset_id)
    if stats_output:
        benchmark.write_rows(stats_output, rows)
        n_err = sum(r["status"] == "error" for r in rows)
        print(f"[bench] {len(rows)} row(s) written to {stats_output}"
              + (f" ({n_err} with errors)" if n_err else ""))
    else:
        print(";".join(benchmark.COLUMNS))
        for r in rows:
            print(";".join(benchmark._fmt(r.get(c)) for c in benchmark.COLUMNS))


if __name__ == "__main__":
    main()
