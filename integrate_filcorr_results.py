#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    _resolved = Path(__file__).resolve()
    candidate_roots = list(_resolved.parents[:4])
    for candidate in candidate_roots:
        if (candidate / "correlation" / "load_data_asos.py").exists():
            sys.path.append(str(candidate))
            break
    else:
        sys.path.append(str(_resolved.parents[-1]))

from correlation.load_data_asos import load_csvs_asos

ID_CODE_PATTERN = re.compile(r"^[A-Z0-9]{4}$")
DEFAULT_RECENT_YEARS = (1, 2, 5, 10)


def _consume_station_id(tokens: List[str]) -> str:
    """
    Consume tokens until the trailing ICAO-like code (e.g. LFOI) is reached.
    """
    if not tokens:
        raise ValueError("No tokens available to parse station id")

    parts: List[str] = []
    while tokens:
        part = tokens.pop(0)
        parts.append(part)
        if ID_CODE_PATTERN.fullmatch(part):
            return "_".join(parts)

    raise ValueError("Could not detect trailing station code in filename")


def parse_station_pair(path: Path) -> Tuple[str, str, str]:
    tokens = path.stem.split("_")
    mutable_tokens = list(tokens)
    try:
        id1 = _consume_station_id(mutable_tokens)
        id2 = _consume_station_id(mutable_tokens)
    except ValueError as exc:
        raise ValueError(f"Unable to parse station ids from '{path.name}': {exc}") from exc
    if not mutable_tokens:
        raise ValueError(f"No configuration suffix detected in '{path.name}'")
    config_suffix = "_".join(mutable_tokens)
    return id1, id2, config_suffix


def load_datetime_lookup(country: str, variable: str) -> Tuple[np.ndarray, Sequence[str]]:
    data, ids = load_csvs_asos(country, variable)
    datetimes = np.asarray(data[:, 0], dtype="datetime64[ns]")
    return datetimes, ids


def iter_filcorr_rows(
    files: Iterable[Tuple[Path, str, str]], datetimes: np.ndarray
) -> Iterable[Tuple[str, str, int, np.datetime64, float]]:
    for file_path, id1, id2 in sorted(files, key=lambda item: item[0].name):
        with file_path.open("r", newline="") as fh:
            reader = csv.reader(fh)
            for idx, row in enumerate(reader, start=1):
                if len(row) < 2:
                    continue
                try:
                    t1_index = int(float(row[0]))
                except ValueError:
                    raise ValueError(
                        f"Invalid t1_index value '{row[0]}' in file '{file_path.name}' at line {idx}"
                    ) from None

                if t1_index < 0 or t1_index >= len(datetimes):
                    raise IndexError(
                        f"t1_index {t1_index} out of bounds for dataset (0..{len(datetimes) - 1})"
                    )

                try:
                    max_corr = float(row[1])
                except ValueError:
                    raise ValueError(
                        f"Invalid max_corr value '{row[1]}' in file '{file_path.name}' at line {idx}"
                    ) from None

                time1 = np.datetime64(datetimes[t1_index], "ns")
                yield id1, id2, t1_index, time1, max_corr


def write_max_lag_csv(
    rows: Iterable[Tuple[str, str, int, np.datetime64, float]],
    output_path: Path,
    *,
    already_sorted: bool = False,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not already_sorted:
        rows = sorted(rows, key=lambda item: (item[0], item[1], item[2]))
    count = 0
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id1", "id2", "t1_index", "time1", "max_corr", "lag"])
        for id1, id2, t1_index, time1, max_corr in rows:
            count += 1
            writer.writerow([id1, id2, t1_index, time1, max_corr, ""])
    return count


def write_recent_windows(
    rows: Sequence[Tuple[str, str, int, np.datetime64, float]],
    output_path: Path,
    windows: Sequence[int],
) -> List[Tuple[int, Path, int]]:
    if not rows or not windows:
        return []

    base_output = Path(output_path)
    max_time = max(np.datetime64(row[3], "ns") for row in rows)
    results: List[Tuple[int, Path, int]] = []

    for years in windows:
        if years is None or years <= 0:
            continue

        year_end = int(str(max_time)[:4])
        year_start = max(1, year_end - years + 1)
        start_time = np.datetime64(f"{year_start:04d}-01-01T00:00:00", "ns")

        filtered = [row for row in rows if np.datetime64(row[3], "ns") >= start_time]
        if not filtered:
            continue

        suffix = f"_last_{years}y"
        derived_path = base_output.with_name(f"{base_output.stem}{suffix}{base_output.suffix}")

        count = write_max_lag_csv(filtered, derived_path, already_sorted=True)
        results.append((years, derived_path, count))

    return results


def deduplicate_rows(
    rows: Sequence[Tuple[str, str, int, np.datetime64, float]]
) -> Tuple[List[Tuple[str, str, int, np.datetime64, float]], int, int]:
    collapsed: dict[Tuple[str, str, int], Tuple[str, str, int, np.datetime64, float]] = {}
    duplicates = 0
    improved = 0
    for id1, id2, t1_index, time1, max_corr in rows:
        key = (id1, id2, t1_index)
        existing = collapsed.get(key)
        if existing is None:
            collapsed[key] = (id1, id2, t1_index, time1, max_corr)
            continue
        duplicates += 1
        prev_corr = existing[4]
        if abs(max_corr) > abs(prev_corr) or (abs(max_corr) == abs(prev_corr) and max_corr > prev_corr):
            collapsed[key] = (id1, id2, t1_index, time1, max_corr)
            improved += 1
    return list(collapsed.values()), duplicates, improved


def derive_output_path(base_output: Path, raw_output: str, suffix: str, multi_suffix: bool) -> Path:
    if "{suffix}" in raw_output:
        return Path(raw_output.format(suffix=suffix))
    if multi_suffix:
        return base_output.with_name(f"{base_output.stem}_{suffix}{base_output.suffix}")
    return base_output


def process_suffix(
    suffix: str,
    file_records: Sequence[Tuple[Path, str, str]],
    datetimes: np.ndarray,
    known_ids: set[str],
    output_path: Path,
    recent_windows: Sequence[int],
) -> None:
    print(f"Processing {len(file_records)} FilCorr files with configuration suffix '{suffix}'.")

    rows: List[Tuple[str, str, int, np.datetime64, float]] = []
    for id1, id2, t1_index, time1, max_corr in iter_filcorr_rows(file_records, datetimes):
        if id1 not in known_ids or id2 not in known_ids:
            raise ValueError(f"Station ids '{id1}', '{id2}' not present in dataset")
        rows.append((id1, id2, t1_index, time1, max_corr))

    if not rows:
        raise ValueError(f"No rows were parsed for configuration suffix '{suffix}'.")

    dedup_rows, duplicates, improved = deduplicate_rows(rows)
    if duplicates:
        print(
            f"Deduplicated {duplicates} overlapping entries "
            f"({improved} replaced with higher |max_corr| values)"
        )

    sorted_rows = sorted(dedup_rows, key=lambda item: (item[0], item[1], item[2]))
    total_rows = write_max_lag_csv(sorted_rows, output_path, already_sorted=True)
    print(f"Wrote {total_rows} rows to {output_path}")

    recent_results = write_recent_windows(sorted_rows, output_path, recent_windows)
    for years, path, count in recent_results:
        print(f"Wrote {count} rows to {path} (last {years} year{'s' if years != 1 else ''})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge FilCorr outputs into a max_lag-style CSV."
    )
    parser.add_argument(
        "--results-dir",
        default="correlation/asos_exp/tests/filcorr_res",
        help="Directory containing FilCorr result CSV files.",
    )
    parser.add_argument(
        "--output",
        default="correlation/asos_exp/tests/filcorr_res/filcorr_max_lag_correlated.csv",
        help="Path for the merged CSV output.",
    )
    parser.add_argument(
        "--country",
        default="fr",
        help="Country code used to load the ASOS dataset.",
    )
    parser.add_argument(
        "--variable",
        default="air_temperature",
        help="Variable name used to load the ASOS dataset.",
    )
    parser.add_argument(
        "--config-suffix",
        default=None,
        help=(
            "Only merge FilCorr files whose names end with this configuration suffix "
            "(e.g. '48_168_1_100_0_50'). If omitted, all files must share the same suffix."
        ),
    )
    parser.add_argument(
        "--all-suffixes",
        action="store_true",
        help=(
            "Process every configuration suffix discovered in the results directory. "
            "When multiple suffixes are processed, the output filename automatically "
            "includes the suffix unless the --output path contains '{suffix}'."
        ),
    )
    parser.add_argument(
        "--recent-years",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate additional outputs restricted to the last N years "
            "(default: 1 2 5 10). Pass the flag with no values to skip."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files = list(results_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files were found in {results_dir}")

    datetimes, ids = load_datetime_lookup(args.country, args.variable)
    known_ids = set(str(x) for x in ids)

    suffix_map: dict[str, List[Tuple[Path, str, str]]] = {}
    for file_path in sorted(files):
        try:
            id1, id2, suffix = parse_station_pair(file_path)
        except ValueError:
            print(
                f"[integrate_filcorr_results] Skipping unrecognized file name: {file_path.name}",
                file=sys.stderr,
            )
            continue
        suffix_map.setdefault(suffix, []).append((file_path, id1, id2))

    if not suffix_map:
        raise FileNotFoundError("No FilCorr result files matched the expected naming pattern.")

    if args.recent_years is None:
        recent_windows = list(DEFAULT_RECENT_YEARS)
    else:
        recent_windows = [years for years in args.recent_years if years and years > 0]
    recent_windows = sorted(set(recent_windows))

    all_suffixes = sorted(suffix_map)
    if args.config_suffix:
        if args.config_suffix not in suffix_map:
            raise ValueError(
                f"No FilCorr result files end with the configuration suffix '{args.config_suffix}'."
            )
        target_suffixes = [args.config_suffix]
    else:
        target_suffixes = all_suffixes
        if len(all_suffixes) > 1 and not args.all_suffixes and args.config_suffix is None:
            formatted = ", ".join(all_suffixes)
            print(
                f"Detected multiple configuration suffixes ({formatted}); "
                "processing each one sequentially."
            )

    base_output = Path(args.output)
    multi_suffix = len(target_suffixes) > 1

    for suffix in target_suffixes:
        file_records = suffix_map[suffix]
        output_path = derive_output_path(base_output, args.output, suffix, multi_suffix)
        process_suffix(suffix, file_records, datetimes, known_ids, output_path, recent_windows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
