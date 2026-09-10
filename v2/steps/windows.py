"""Step 2 — split the stream into sliding sub-windows."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    start_index: int      # start position (column)
    start_time: object    # corresponding time value
    block: np.ndarray     # (n_series, window_size)


def iter_windows(data, times, window_size, window_step):
    """Generate the sub-windows in time order (all series included)."""
    n_cols = data.shape[1]
    for start in range(0, n_cols - window_size + 1, window_step):
        yield Window(start, times[start], data[:, start:start + window_size])
