import os
import pandas as pd
import numpy as np

DEFAULT_ROOT = "datasets/asos-airports"
_LEGACY_ROOT = "correlation/asos-airports"


def load_dataset(country: str, variable: str, root: str = DEFAULT_ROOT):
    filename = f"{country}-{variable}.csv"
    path = os.path.join(root, filename)
    if not os.path.exists(path) and os.path.exists(os.path.join(_LEGACY_ROOT, filename)):
        path = os.path.join(_LEGACY_ROOT, filename)

    data, names = _load_csv(path)
    return data, names


def _load_csv(csv_path: str):
    npz_path = csv_path.replace(".csv", ".npz")

    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as f:
            return f["data"], f["names"]

    df = pd.read_csv(csv_path, dtype=str)
    datetime_col = pd.to_datetime(
        df.iloc[:, 0] + "T" + df.iloc[:, 1], format="%Y-%m-%dT%H"
    )
    other_data = df.iloc[:, 2:].astype(float).to_numpy()

    full_data = np.hstack([
        datetime_col.to_numpy(dtype="datetime64[m]")[:, None].astype(object),
        other_data.astype(object),
    ])
    column_names = df.columns[2:].to_numpy()

    np.savez_compressed(npz_path, data=full_data, names=column_names)
    return full_data, column_names
