"""Random "sketch of sketch" projection for the low-dimension indexes.

The pipeline sketches are high-dimensional (n_vectors, e.g. 64). The 2D/3D
indexes (quadtree, octree) must work in a fixed dimension (2 or 3). Two options
are available:

  * **truncation**: take `sketch[:d]` (the first d coords) — throws away
    n_vectors−d coords (a lot for d=2 out of 64).
  * **random projection**: `R · sketch` with `R` of shape `(d, n_vectors)`,
    coefficients ~ N(0, 1/d) → preserves the Euclidean distances within a
    (1±ε) deviation guaranteed by Johnson-Lindenstrauss (at the price of a
    variance inversely proportional to d).

Benefit: the random projection **averages over every coord** instead of keeping
only a fraction of them → far better distance preservation for d≪n_vectors.
This is the "double projection": 1) random projection of the windows → sketch;
2) random projection of the sketch → low-dim coords of the index.
"""

import numpy as np


def make_random_projection(d, n_vectors, seed):
    """Matrix (d, n_vectors), N(0, 1/d) → preserves distances on average."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(d), int(n_vectors))) / np.sqrt(d)
