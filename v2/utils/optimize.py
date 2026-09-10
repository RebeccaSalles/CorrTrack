"""Hyperparameter search launcher (port of the v1 optimizer).

Sweeps a parameter grid against the brute-force ground truth (on the first
`--train-ratio` fraction of observations) and selects the best config:
recall >= target, then near-best speedup, then fewest candidates.

    python -m v2.utils.optimize data.csv --n-vectors 8,16,32,64 \
        --cell-size 0.25,0.5,1.0 --train-ratio 0.3333 --target-recall 0.95

Pass `--sweep-validate false` to skip the Pearson validation phase during the
sweep (faster, but precision/val_time/speedup then reflect candidate generation
only — the prediction set is the post-sketch-cosine pairs).

Writes (in --output, default 'results/'): stats.csv (every config) and
best_params.json (the chosen config + its recall/speedup).
"""

import json
import os

from ..core import benchmark, optimize
from ..core.cli import clean_output_dir, parse_bench
from ..core.config import Config


def main():
    csv, dataset_id, stats_output, axes = parse_bench()
    defaults = Config()
    out = axes["output"][0] if "output" in axes else defaults.output
    clean = axes["clean"][0] if "clean" in axes else defaults.clean
    clean_output_dir(out, clean)
    if stats_output is None and out:
        stats_output = os.path.join(out, "stats.csv")

    rows = benchmark.sweep(axes, csv, dataset_id=dataset_id)
    if stats_output:
        benchmark.write_rows(stats_output, rows)

    target = axes.get("target_recall", [defaults.target_recall])[0]
    rfnr = axes.get("recall_fallback_near_ratio", [defaults.recall_fallback_near_ratio])[0]
    snr = axes.get("speedup_near_ratio", [defaults.speedup_near_ratio])[0]
    tprec = axes.get("target_precision", [defaults.target_precision])[0]
    best = optimize.select_best(rows, target, rfnr, snr, target_precision=tprec)

    if best is None:
        print(f"[optimize] {len(rows)} configs, none successful — no selection.")
        return

    params = optimize.best_params(best)
    chosen = {"params": params,
              "recall": best.get("recall"), "speedup": best.get("speedup"),
              "cand_w": best.get("cand_w"), "corr_w": best.get("corr_w"),
              "dataset_id": best.get("dataset_id")}
    if out:
        with open(os.path.join(out, "best_params.json"), "w") as f:
            json.dump(chosen, f, indent=2)

    print(f"[optimize] {len(rows)} configs swept "
          f"(target_recall={target}, speedup_near={snr}).")
    print(f"[optimize] best: recall={best.get('recall')} speedup={best.get('speedup')} "
          f"cand_w={best.get('cand_w')}")
    print(f"[optimize] params: {params}")
    if out:
        print(f"[optimize] written: {stats_output}, {os.path.join(out, 'best_params.json')}")


if __name__ == "__main__":
    main()
