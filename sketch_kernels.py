import numpy as np


def compute_series_dots(window_blocks, weights):
    return np.einsum("sbw,bvw->sbv", window_blocks, weights, optimize=True)


def apply_orth_and_normalize(raw_matrix, perm, signs, norm_mode):
    raw_matrix = np.asarray(raw_matrix, dtype=np.float64)
    raw = raw_matrix.copy()
    if perm is not None and signs is not None:
        perm = np.asarray(perm, dtype=np.int64)
        signs = np.asarray(signs, dtype=np.float64)
        if perm.size == raw.shape[1] and signs.size == raw.shape[1]:
            raw = raw[:, perm] * signs

    mean = np.mean(raw, axis=1, keepdims=True)
    centered = raw - mean
    if norm_mode == 1:
        denom = np.linalg.norm(centered, axis=1, keepdims=True)
    else:
        denom = np.std(centered, axis=1, keepdims=True)
    norm = np.zeros_like(centered)
    valid = np.isfinite(denom[:, 0]) & (denom[:, 0] > 0)
    if np.any(valid):
        norm[valid] = centered[valid] / denom[valid]
    return raw, norm


def build_sketch_matrix(window_blocks, weights, perm, signs, norm_mode):
    window_blocks = np.asarray(window_blocks, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    series_dots = compute_series_dots(window_blocks, weights)
    raw = np.sum(series_dots, axis=1)
    raw, norm = apply_orth_and_normalize(raw, perm, signs, norm_mode)
    return series_dots, raw, norm


def compute_constant_flags(sum1, sum2, sum3, sum4, n, std_thresh=1e-3, kurt_thresh=5.0):
    sum1 = np.asarray(sum1, dtype=np.float64)
    sum2 = np.asarray(sum2, dtype=np.float64)
    sum3 = np.asarray(sum3, dtype=np.float64)
    sum4 = np.asarray(sum4, dtype=np.float64)
    if n <= 0:
        const_flags = np.ones(sum1.shape[0], dtype=np.uint8)
        spiked_flags = np.zeros(sum1.shape[0], dtype=np.uint8)
        return const_flags, spiked_flags

    n_d = float(n)
    mean = sum1 / n_d
    var_sum = sum2 - (sum1 * sum1) / n_d
    var_sum = np.maximum(var_sum, 0.0)
    const_flags = (var_sum <= (std_thresh ** 2) * n_d).astype(np.uint8)

    mu4_sum = (
        sum4
        - 4.0 * mean * sum3
        + 6.0 * (mean ** 2) * sum2
        - 4.0 * (mean ** 3) * sum1
        + n_d * (mean ** 4)
    )
    var = var_sum / n_d
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = (mu4_sum / n_d) / (var * var) - 3.0
    spiked_flags = (kurt > kurt_thresh).astype(np.uint8)
    spiked_flags = np.where(np.isfinite(kurt) & (var > 0), spiked_flags, 0).astype(np.uint8)
    return const_flags, spiked_flags
