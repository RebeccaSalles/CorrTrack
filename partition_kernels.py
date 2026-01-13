import numpy as np


def build_partitions(matrix, grid_dimension, const_flags):
    matrix = np.asarray(matrix, dtype=np.float64)
    const_flags = np.asarray(const_flags, dtype=np.uint8)
    if grid_dimension <= 0:
        raise ValueError("grid_dimension must be positive")
    if matrix.size == 0:
        return (
            np.empty((0, 0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.uint8),
        )
    n_rows, n_dim = matrix.shape
    n_grids = n_dim // grid_dimension
    if n_grids <= 0:
        return (
            np.empty((0, 0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.uint8),
        )
    chunks = np.empty((n_grids, n_rows, grid_dimension), dtype=np.float64)
    norms = np.empty((n_grids, n_rows), dtype=np.float64)
    is_const = np.empty((n_grids, n_rows), dtype=np.uint8)
    for grid in range(n_grids):
        start = grid * grid_dimension
        end = start + grid_dimension
        chunk = matrix[:, start:end]
        chunks[grid] = chunk
        grid_norms = np.linalg.norm(chunk, axis=1)
        grid_norms = np.where(np.isfinite(grid_norms), grid_norms, 0.0)
        norms[grid] = grid_norms
        is_const[grid] = np.logical_or(const_flags != 0, grid_norms == 0.0).astype(np.uint8)
    return chunks, norms, is_const
