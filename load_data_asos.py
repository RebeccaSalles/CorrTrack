"""Backward-compatible ASOS loader shim."""

from typing import Optional, Tuple

import numpy as np

from datasets.asos_loader import load_dataset, DEFAULT_ROOT


def load_csvs_asos(country: str, variable: str, folder: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    root = folder or DEFAULT_ROOT
    return load_dataset(country, variable, root=root)
