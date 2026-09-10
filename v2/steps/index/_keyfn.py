"""1D key extraction for the sorted indexes (bst, bptree).

5 modes:
  * `truncate` (default): `key = sketch[0]` (1st coord, like v1).
  * `median` : `key = median(sketch)`.
  * `sum`    : `key = sum(sketch)`.
  * `abs_sum`: `key = sum(|sketch|)` — sum of the absolute values. Captures the
               L1 energy of the sketch, robust to sign (useful when the sketch
               can flip sign between similar windows).
  * `random` : `key = mean(R · sketch)` where R is a JL random projection of
               shape `(dim, n_vectors)` ~ N(0, 1/dim). Averaging `dim` random
               projections yields a more stable scalar than a single one
               (Johnson-Lindenstrauss in 1D averaged over `dim` draws).

⚠️ The 1D bisect is only guaranteed free of false negatives (Lipschitz 1) for
`truncate` and `random` (unit projections). For `median`, `sum` and `abs_sum`
the pre-filter is heuristic → precision is preserved by the downstream Pearson
validation, but recall may suffer if `tree_radius` is not adapted.
"""

import numpy as np

from ._projection import make_random_projection


def make_key_fn(mode, dim, n_vectors, seed):
    """Return a `vec -> float` function (1D scalar for bisect)."""
    mode = (mode or "truncate").lower()
    if mode == "truncate":
        return lambda v: float(np.asarray(v, dtype=float).ravel()[0])
    if mode == "median":
        return lambda v: float(np.median(np.asarray(v, dtype=float).ravel()))
    if mode == "sum":
        return lambda v: float(np.sum(np.asarray(v, dtype=float).ravel()))
    if mode == "abs_sum":
        return lambda v: float(np.sum(np.abs(np.asarray(v, dtype=float).ravel())))
    if mode == "random":
        d = max(1, int(dim or 1))
        R = make_random_projection(d, max(1, int(n_vectors or 1)), seed)
        return lambda v, _R=R: float(np.mean(_R @ np.asarray(v, dtype=float).ravel()))
    raise ValueError(f"unknown key_mode: {mode!r} "
                     "(truncate | median | sum | abs_sum | random)")


def _chunk_apply(v, d, fn):
    """Split `v` into `d` (nearly) equal chunks and apply `fn` to each one
    → array of length d. Used for median/sum in multi-D mode."""
    arr = np.asarray(v, dtype=float).ravel()
    chunks = np.array_split(arr, d)
    return np.array([float(fn(c)) for c in chunks])


def make_proj_fn(mode, dim, n_vectors, seed):
    """Return a `vec -> np.ndarray(dim,)` function — N-D projection of the sketch.

    Like `make_key_fn` but returns a vector of length `dim` instead of a
    scalar. Used by bst3d/bptree3d (dim=3) to generate the 3D coords.

      * `truncate` : `vec[:dim]` — the first d coords of the sketch.
      * `median`   : median over d chunks of the sketch.
      * `sum`      : sum over d chunks of the sketch.
      * `abs_sum`  : sum of the absolute values over d chunks of the sketch.
      * `random`   : JL random projection `R(dim, n_vectors) @ vec` (default).
    """
    mode = (mode or "random").lower()
    dim = max(1, int(dim or 3))
    if mode == "truncate":
        return lambda v, _d=dim: np.asarray(v, dtype=float).ravel()[:_d]
    if mode == "median":
        return lambda v, _d=dim: _chunk_apply(v, _d, np.median)
    if mode == "sum":
        return lambda v, _d=dim: _chunk_apply(v, _d, np.sum)
    if mode == "abs_sum":
        return lambda v, _d=dim: _chunk_apply(v, _d, lambda c: np.sum(np.abs(c)))
    if mode == "random":
        R = make_random_projection(dim, max(1, int(n_vectors or 1)), seed)
        return lambda v, _R=R: _R @ np.asarray(v, dtype=float).ravel()
    raise ValueError(f"unknown key_mode: {mode!r} "
                     "(truncate | median | sum | abs_sum | random)")
