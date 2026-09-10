"""Lanceur CorrTrack : sketch -> index (grid|tree) -> select -> validation.

    python -m v2.dans data.csv
    python -m v2.corrtrack data.csv --index-backend tree --step select
    python -m v2.corrtrack data.csv --backend cython
    CORRTRACK_WINDOW_SIZE=48 python -m v2.corrtrack data.csv
"""

from .core import pipeline
from .core.cli import apply_run_dir, clean_output_dir, parse, report


def main():
    config, csv, step = parse("corrtrack")
    apply_run_dir(config, "corrtrack")
    clean_output_dir(config.output, config.clean)
    report(pipeline.run(config, csv, mode="corrtrack", step=step), config)


if __name__ == "__main__":
    main()
