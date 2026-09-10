"""Brute-force launcher: enumerates every pair -> exact validation.

No sketch and no index: serves as the reference (ground truth). Reuses the same
CSV reading, the same windowing and the same validation as corrtrack.

    python -m v2.bf data.csv
    python -m v2.bf data.csv --step candidates
    CORRTRACK_CORR_THRESHOLD=0.8 python -m v2.bf data.csv
"""

from .core import pipeline
from .core.cli import apply_run_dir, clean_output_dir, parse, report


def main():
    config, csv, step = parse("bf")
    apply_run_dir(config, "bf")
    clean_output_dir(config.output, config.clean)
    report(pipeline.run(config, csv, mode="bf", step=step), config)


if __name__ == "__main__":
    main()
