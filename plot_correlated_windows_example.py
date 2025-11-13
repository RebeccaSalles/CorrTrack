#!/usr/bin/env python3
"""
Reproduce the colored correlated-window plot for the synthetic dataset.

Default arguments point to the `synt_stat_corr0p20_m8_w96_s12_signboth_thr0p8_lag48`
artifacts under `corrtrack_release/tmp_artifacts`, and highlight the first 500
samples of series S1, S4, and S5. Override paths or plotting scope via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot correlated windows with pair-specific colors."
    )
    parser.add_argument(
        "--data-npz",
        type=Path,
        default=Path(
            "corrtrack_release/tmp_artifacts/"
            "synt_stat_corr0p20_m8_w96_s12_signboth_thr0p8_lag48.npz"
        ),
        help="Path to the synthetic dataset NPZ (default: current tmp artifact).",
    )
    parser.add_argument(
        "--correlated-csv",
        type=Path,
        default=Path(
            "corrtrack_release/tmp_artifacts/"
            "synt_stat_corr0p20_m8_w96_s12_signboth_thr0p8_lag48_correlated.csv"
        ),
        help="CSV listing injected correlated windows (default: current tmp artifact).",
    )
    parser.add_argument(
        "--series",
        type=str,
        default="s1,s4,s5",
        help="Comma-separated series ids to plot (e.g., 's1,s4,s5').",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=96,
        help="Injected window length (default: 96).",
    )
    parser.add_argument(
        "--time-min",
        type=int,
        default=0,
        help="Minimum time index to display (default: 0).",
    )
    parser.add_argument(
        "--time-max",
        type=int,
        default=500,
        help="Maximum time index to display (default: 500).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "corrtrack_release/tmp_artifacts/"
            "correlated_windows_example_colored.png"
        ),
        help="Where to save the plot (default: tmp_artifacts/...colored.png).",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Figure DPI (default: 200)."
    )
    return parser.parse_args()


def load_series(npz_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with np.load(npz_path) as data:
        arr = data[data.files[0]]
    time_index = arr[:, 0]
    series = {f"s{i+1}": arr[:, i + 1] for i in range(arr.shape[1] - 1)}
    return time_index, series


def load_windows(
    csv_path: Path,
) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def windows_for_series(
    rows: Iterable[Dict[str, str]],
    series_ids: Iterable[str],
    window_size: int,
    time_min: int,
    time_max: int,
) -> Tuple[Dict[str, List[Tuple[int, int, str, float, str]]], Dict[str, str]]:
    series_ids = tuple(series_ids)
    windows = {sid: [] for sid in series_ids}
    pair_colors: Dict[Tuple[str, str], str] = {}
    legend_labels: Dict[str, str] = {}
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    def in_scope(start: int) -> bool:
        return not (start > time_max or start + window_size < time_min)

    for row in rows:
        id1, id2 = row["id1"], row["id2"]
        start1, start2 = int(row["time1"]), int(row["time2"])
        corr = float(row["corr"])
        key = tuple(sorted((id1, id2)))
        both_selected = id1 in windows and id2 in windows

        highlight_color: Optional[str] = None
        highlight_pair = False
        if both_selected:
            if key not in pair_colors:
                pair_colors[key] = next(color_cycle)
                legend_labels[key] = f"{id1.upper()}-{id2.upper()} ({corr:+.2f})"
            highlight_color = pair_colors[key]
            highlight_pair = True
        else:
            highlight_color = "#dddddd"

        for sid, start, partner in ((id1, start1, id2), (id2, start2, id1)):
            if sid not in windows or not in_scope(start):
                continue
            windows[sid].append(
                (
                    start,
                    start + window_size,
                    partner,
                    corr,
                    highlight_color,
                    highlight_pair,
                )
            )

    for spans in windows.values():
        spans.sort()

    legend = {
        legend_labels[key]: color for key, color in pair_colors.items() if key in legend_labels
    }
    return windows, legend


def plot_windows(
    time_index: np.ndarray,
    series_data: Dict[str, np.ndarray],
    windows: Dict[str, List[Tuple[int, int, str, float, str]]],
    legend: Dict[str, str],
    series_ids: Iterable[str],
    time_min: int,
    time_max: int,
    window_size: int,
    output_path: Path,
    dpi: int,
) -> None:
    series_ids = list(series_ids)
    fig, axes = plt.subplots(len(series_ids), 1, sharex=True, figsize=(10, 6))

    for ax, sid in zip(axes, series_ids):
        mask = (time_index >= time_min) & (time_index <= time_max)
        ax.plot(time_index[mask], series_data[sid][mask], color="black", linewidth=1)
        ax.set_ylabel(sid.upper())
        ymin, ymax = ax.get_ylim()
        span_offsets: Dict[Tuple[str, str], int] = {}

        for start, end, partner, corr, color, highlight in windows[sid]:
            alpha = 0.35 if highlight else 0.12
            ax.axvspan(start, end, color=color, alpha=alpha)
            if not highlight:
                continue
            key = (partner, color)
            offset = span_offsets.get(key, 0)
            span_offsets[key] = offset + 1
            ytext = ymax - 0.08 * (1 + offset) * (ymax - ymin)
            ax.text(
                start + 2,
                ytext,
                f"{partner.upper()}\n{corr:+.2f}",
                fontsize=8,
                color="black",
                bbox=dict(facecolor=color, alpha=0.65, edgecolor="none"),
                verticalalignment="top",
            )

        ax.set_xlim(time_min, time_max + window_size)

    axes[-1].set_xlabel("Time index")
    fig.suptitle(
        f"Correlated windows across {', '.join(s.upper() for s in series_ids)} "
        f"({time_min}–{time_max})"
    )

    if legend:
        handles = [
            plt.Line2D([0], [0], color=color, lw=4, label=label)
            for label, color in legend.items()
        ]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    series_ids = [sid.strip().lower() for sid in args.series.split(",") if sid.strip()]
    time_index, series_data = load_series(args.data_npz)
    rows = load_windows(args.correlated_csv)
    windows, legend = windows_for_series(
        rows,
        series_ids,
        args.window_size,
        args.time_min,
        args.time_max,
    )
    plot_windows(
        time_index,
        series_data,
        windows,
        legend,
        series_ids,
        args.time_min,
        args.time_max,
        args.window_size,
        args.output,
        args.dpi,
    )


if __name__ == "__main__":
    main()
