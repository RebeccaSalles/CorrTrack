from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from synth_corr_gen import _build_stem, make_corr_dataset

DEFAULT_CACHE_ROOT = Path("datasets/synth_outputs")
_DEFAULT_BASE_PROC: Dict[str, Any] = {"type": "ar1", "phi": 0.6, "sigma": 1.0}


def _stem_from_params(params: Dict[str, Any]) -> str:
    required = ("m", "n", "w", "z")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"generator_params is missing required keys: {', '.join(missing)}")

    return _build_stem(
        m=int(params["m"]),
        n=int(params["n"]),
        w=int(params["w"]),
        s=int(params.get("s", 1)),
        z=float(params["z"]),
        corr_sign=str(params.get("corr_sign", "pos")),
        threshold=float(params.get("threshold", 0.7)),
        max_lag=int(params.get("max_lag", 0)),
        base_proc=params.get("base_proc"),
    )


def _load_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as npz:
        key = npz.files[0]
        data = npz[key]
    return data


def load_dataset(
    country: str,
    variable: str,
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    generator_params: Dict[str, Any] | None = None,
    refresh: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (or synthesize) a dataset suitable for CorrTrack experiments.

    Parameters
    ----------
    country, variable : str
        Identifiers forwarded by the experiment config. They are used only to
        namespace cached outputs (allowing multiple synthetic datasets).
    cache_root : str | Path
        Folder used to persist generated artifacts. Defaults to
        ``datasets/synth_outputs``.
    generator_params : Dict[str, Any]
        Keyword arguments forwarded to :func:`make_corr_dataset`. Must include
        ``m``, ``n``, ``w``, and ``z``.
    refresh : bool
        When True, forces regeneration even if cached outputs exist.
    """

    if generator_params is None:
        raise ValueError("generator_params must be provided to synth_loader.load_dataset")

    params = dict(generator_params)
    params.setdefault("s", 1)
    params.setdefault("threshold", 0.7)
    params.setdefault("corr_sign", "pos")
    params.setdefault("max_lag", 0)
    params.setdefault("lag_step", None)
    params.setdefault("nonoverlap", True)
    params.setdefault("seed", 7)
    params["base_proc"] = dict(params.get("base_proc") or _DEFAULT_BASE_PROC)

    cache_root = Path(cache_root)
    slug = country if country == variable else f"{country}_{variable}"
    dataset_dir = cache_root / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)

    params["save_dir"] = str(dataset_dir)
    stem = _stem_from_params(params)
    npz_path = dataset_dir / f"{stem}.npz"

    if refresh or not npz_path.exists():
        make_corr_dataset(**params)

    array = _load_npz(npz_path).astype(np.float32, copy=False)
    ids = np.asarray([f"s{i+1}" for i in range(array.shape[1] - 1)], dtype=object)
    return array, ids
