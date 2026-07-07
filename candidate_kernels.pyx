# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import time
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, floor
from libc.stdlib cimport malloc, realloc, free
from libc.stdint cimport int64_t, uint64_t, uint8_t
from cython.parallel cimport prange, threadid

np.import_array()


cdef inline Py_ssize_t _bisect_left(double[:] values, double x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    cdef double v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_right(double[:] values, double x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    cdef double v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline void _projection_interval(double value,
                                      double gamma,
                                      double *lower,
                                      double *upper) noexcept nogil:
    cdef double v = value
    cdef double g = gamma
    cdef double one_minus_v2
    cdef double one_minus_g2
    cdef double spread
    cdef double center
    if v < -1.0:
        v = -1.0
    elif v > 1.0:
        v = 1.0
    if g < -1.0:
        g = -1.0
    elif g > 1.0:
        g = 1.0
    if g <= -1.0:
        lower[0] = -1.0
        upper[0] = 1.0
        return
    one_minus_v2 = 1.0 - v * v
    if one_minus_v2 < 0.0:
        one_minus_v2 = 0.0
    one_minus_g2 = 1.0 - g * g
    if one_minus_g2 < 0.0:
        one_minus_g2 = 0.0
    center = v * g
    spread = sqrt(one_minus_v2) * sqrt(one_minus_g2)
    if v <= -g:
        lower[0] = -1.0
    else:
        lower[0] = center - spread
    if v >= g:
        upper[0] = 1.0
    else:
        upper[0] = center + spread
    if lower[0] < -1.0:
        lower[0] = -1.0
    if upper[0] > 1.0:
        upper[0] = 1.0


# (2026-07-06) Part 1: Cauchy-Schwarz block/row upper-bound pruning.
#
# For a unit vector x split into "bound dims" x_I (a configurable subset,
# `candidate_bound_dims`/`candidate_bound_dim_selection`, independent of the
# indexing dims already used for the sorted-range candidate search) and a
# residual x_R over the remaining dims, ||x_R|| = sqrt(1 - ||x_I||^2) since
# ||x||=1. For any unit query q split the same way (q_I, q_R):
#
#   q.x = q_I.x_I + q_R.x_R
#       <= q_I.x_I + ||q_R|| * ||x_R||        (Cauchy-Schwarz on the residual)
#
# so if x_I is only known to lie in a per-coordinate box [min_k, max_k]
# (e.g. the min/max over every row in a block) rather than being a specific
# vector, the worst case of q_I.x_I over that box is
# sum_k max(q_I[k]*min_k, q_I[k]*max_k), giving a safe upper bound on q.x for
# *any* vector matching the box constraints -- used to reject a whole block
# without visiting any of its rows. The same identity with x_I evaluated
# exactly (not just boxed) and ||x_R|| read from a per-row cached value
# gives a cheaper, tighter bound to reject a single candidate row before
# paying for the full O(n_vectors) dot product. Both bounds are true
# (proven) inequalities, not heuristics -- enabling this pruning cannot
# produce a false negative (a row that would have passed gamma can never be
# rejected by a bound derived this way), so it preserves exact recall
# whenever disabled by default and only prunes usable, verified-safe cases.
# See docs/implementation_log.md, "Part 1: Cauchy-Schwarz block/row
# upper-bound pruning".
cdef inline double _query_residual_norm(double[:] q_vec, int64_t[:] bound_dims) noexcept nogil:
    cdef Py_ssize_t k, n_bound = bound_dims.shape[0]
    cdef int64_t d
    cdef double qk, q_i_norm2 = 0.0
    for k in range(n_bound):
        d = bound_dims[k]
        qk = q_vec[d]
        q_i_norm2 += qk * qk
    if q_i_norm2 >= 1.0:
        return 0.0
    return sqrt(1.0 - q_i_norm2)


cdef inline bint _block_ub_prune_check(double[:] q_vec,
                                       int64_t[:] bound_dims,
                                       double[:] bound_min,
                                       double[:] bound_max,
                                       double max_residual_norm,
                                       double q_r_norm,
                                       double gamma,
                                       bint signed_abs) noexcept nogil:
    """True iff no row matching the block's box constraints could reach gamma."""
    cdef Py_ssize_t k, n_bound = bound_dims.shape[0]
    cdef int64_t d
    cdef double qk, a, b, sum_max = 0.0, sum_min = 0.0, ub, lb
    for k in range(n_bound):
        d = bound_dims[k]
        qk = q_vec[d]
        a = qk * bound_min[k]
        b = qk * bound_max[k]
        if a > b:
            sum_max += a
            sum_min += b
        else:
            sum_max += b
            sum_min += a
    ub = sum_max + q_r_norm * max_residual_norm
    if signed_abs:
        lb = sum_min - q_r_norm * max_residual_norm
        return ub < gamma and lb > -gamma
    return ub < gamma


cdef inline bint _cone_block_prune_check(double[:] q_vec,
                                         double[:] ref_direction,
                                         double min_cos_to_ref,
                                         double gamma,
                                         bint signed_abs) noexcept nogil:
    """True iff no unit vector within the block's angular extent (a
    spherical cap of radius arccos(min_cos_to_ref) around ref_direction)
    could reach gamma. Tighter than _block_ub_prune_check's per-coordinate
    box for correlated/clustered rows, since it captures the block's actual
    angular spread directly instead of bounding each dimension
    independently. See docs/implementation_log.md, "Part 1 follow-up:
    cone-based block bound"."""
    cdef Py_ssize_t d, n_dim = q_vec.shape[0]
    cdef double v = 0.0
    cdef double lower, upper
    for d in range(n_dim):
        v += q_vec[d] * ref_direction[d]
    _projection_interval(v, min_cos_to_ref, &lower, &upper)
    if signed_abs:
        return upper < gamma and lower > -gamma
    return upper < gamma


cdef inline bint _row_ub_prune_check(double[:] q_vec,
                                     double[:, :] vectors,
                                     Py_ssize_t row,
                                     int64_t[:] bound_dims,
                                     double residual_norm_row,
                                     double q_r_norm,
                                     double gamma,
                                     bint signed_abs) noexcept nogil:
    """True iff this specific row cannot reach gamma (safe to skip the full dot)."""
    cdef Py_ssize_t k, n_bound = bound_dims.shape[0]
    cdef int64_t d
    cdef double dot_bound = 0.0, ub, lb
    for k in range(n_bound):
        d = bound_dims[k]
        dot_bound += q_vec[d] * vectors[row, d]
    ub = dot_bound + q_r_norm * residual_norm_row
    if signed_abs:
        lb = dot_bound - q_r_norm * residual_norm_row
        return ub < gamma and lb > -gamma
    return ub < gamma


cdef inline Py_ssize_t _bisect_left_range(double[:] values, Py_ssize_t lo, Py_ssize_t hi, double x) nogil:
    cdef Py_ssize_t mid
    cdef double v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_right_range(double[:] values, Py_ssize_t lo, Py_ssize_t hi, double x) nogil:
    cdef Py_ssize_t mid
    cdef double v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_left_long(long[:] values, long x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    cdef long v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_left_i64_n(int64_t[:] values, Py_ssize_t hi, int64_t x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t mid
    cdef int64_t v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_left_u64_n(uint64_t[:] values, Py_ssize_t hi, uint64_t x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t mid
    cdef uint64_t v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _bisect_right_u64_n(uint64_t[:] values, Py_ssize_t hi, uint64_t x) nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t mid
    cdef uint64_t v
    while lo < hi:
        mid = (lo + hi) // 2
        v = values[mid]
        if v <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline uint64_t _hybrid_hash_key(long s1, long s2, long lag, long w) nogil:
    cdef uint64_t h = <uint64_t>1469598103934665603
    h ^= <uint64_t>s1
    h *= <uint64_t>1099511628211
    h ^= <uint64_t>s2
    h *= <uint64_t>1099511628211
    h ^= <uint64_t>lag
    h *= <uint64_t>1099511628211
    h ^= <uint64_t>w
    h *= <uint64_t>1099511628211
    h ^= h >> 33
    h *= <uint64_t>0xff51afd7ed558ccd
    h ^= h >> 33
    return h


cdef inline Py_ssize_t _hybrid_find_slot(uint8_t[:] occupied,
                                         long[:] key_s1,
                                         long[:] key_s2,
                                         long[:] key_lag,
                                         long[:] key_w,
                                         Py_ssize_t capacity,
                                         long s1,
                                         long s2,
                                         long lag,
                                         long w,
                                         bint *found) nogil:
    cdef Py_ssize_t mask = capacity - 1
    cdef Py_ssize_t idx = <Py_ssize_t>(_hybrid_hash_key(s1, s2, lag, w) & <uint64_t>mask)
    while occupied[idx] != 0:
        if key_s1[idx] == s1 and key_s2[idx] == s2 and key_lag[idx] == lag and key_w[idx] == w:
            found[0] = True
            return idx
        idx = (idx + 1) & mask
    found[0] = False
    return idx


cdef inline bint _key_lt(double av, int64_t ai, double bv, int64_t bi) nogil:
    if av < bv:
        return True
    if av > bv:
        return False
    return ai < bi


cdef inline bint _append_pair(int64_t **buf,
                              Py_ssize_t *count,
                              Py_ssize_t *cap,
                              int64_t a,
                              int64_t b) nogil:
    if count[0] >= cap[0]:
        cap[0] = cap[0] * 2
        buf[0] = <int64_t *>realloc(buf[0], cap[0] * 2 * sizeof(int64_t))
        if buf[0] == NULL:
            return False
    buf[0][2 * count[0]] = a
    buf[0][2 * count[0] + 1] = b
    count[0] += 1
    return True


cdef inline bint _append_pair_dist(int64_t **buf,
                                   double **dist_buf,
                                   Py_ssize_t *count,
                                   Py_ssize_t *cap,
                                   int64_t a,
                                   int64_t b,
                                   double dist_sq) nogil:
    if count[0] >= cap[0]:
        cap[0] = cap[0] * 2
        buf[0] = <int64_t *>realloc(buf[0], cap[0] * 2 * sizeof(int64_t))
        dist_buf[0] = <double *>realloc(dist_buf[0], cap[0] * sizeof(double))
        if buf[0] == NULL or dist_buf[0] == NULL:
            return False
    buf[0][2 * count[0]] = a
    buf[0][2 * count[0] + 1] = b
    dist_buf[0][count[0]] = dist_sq
    count[0] += 1
    return True


cdef inline uint64_t _pair_hash_key(int64_t a, int64_t b) nogil:
    cdef uint64_t x = <uint64_t>(a + 1)
    cdef uint64_t y = <uint64_t>(b + 1)
    cdef uint64_t h = x * <uint64_t>0x9E3779B185EBCA87
    h ^= y + <uint64_t>0xC2B2AE3D27D4EB4F + (h << 6) + (h >> 2)
    h ^= h >> 33
    h *= <uint64_t>0xff51afd7ed558ccd
    h ^= h >> 33
    return h


cdef inline void _canonical_window_pair(int64_t win_a,
                                        int64_t sid_a,
                                        int64_t rank_a,
                                        int64_t time_a,
                                        int64_t win_b,
                                        int64_t sid_b,
                                        int64_t rank_b,
                                        int64_t time_b,
                                        int64_t *out_a,
                                        int64_t *out_b) noexcept nogil:
    if sid_a == sid_b:
        if time_a < time_b:
            out_a[0] = win_b
            out_b[0] = win_a
        else:
            out_a[0] = win_a
            out_b[0] = win_b
        return
    if time_a == time_b:
        if rank_a <= rank_b:
            out_a[0] = win_a
            out_b[0] = win_b
        else:
            out_a[0] = win_b
            out_b[0] = win_a
        return
    if time_a < time_b:
        out_a[0] = win_b
        out_b[0] = win_a
    else:
        out_a[0] = win_a
        out_b[0] = win_b


cdef bint _pair_seen_init(uint8_t **occupied,
                          int64_t **keys_a,
                          int64_t **keys_b,
                          Py_ssize_t *capacity,
                          Py_ssize_t requested) noexcept nogil:
    cdef Py_ssize_t cap = 16
    cdef Py_ssize_t i
    if requested < 16:
        requested = 16
    while cap < requested:
        cap <<= 1
    occupied[0] = <uint8_t *>malloc(cap * sizeof(uint8_t))
    keys_a[0] = <int64_t *>malloc(cap * sizeof(int64_t))
    keys_b[0] = <int64_t *>malloc(cap * sizeof(int64_t))
    if occupied[0] == NULL or keys_a[0] == NULL or keys_b[0] == NULL:
        if occupied[0] != NULL:
            free(occupied[0])
        if keys_a[0] != NULL:
            free(keys_a[0])
        if keys_b[0] != NULL:
            free(keys_b[0])
        occupied[0] = NULL
        keys_a[0] = NULL
        keys_b[0] = NULL
        capacity[0] = 0
        return False
    for i in range(cap):
        occupied[0][i] = <uint8_t>0
    capacity[0] = cap
    return True


cdef void _pair_seen_free(uint8_t *occupied,
                          int64_t *keys_a,
                          int64_t *keys_b) noexcept nogil:
    if occupied != NULL:
        free(occupied)
    if keys_a != NULL:
        free(keys_a)
    if keys_b != NULL:
        free(keys_b)


cdef inline void _pair_seen_clear(uint8_t *occupied, Py_ssize_t capacity) noexcept nogil:
    # (2026-07-07) Tier 1 perf fix for InstinctIndex -- reset an existing
    # pair-seen hash set in place (O(capacity) memset) instead of the
    # free-then-malloc-then-zero cycle _pair_seen_init does, so a
    # persistent buffer can be reused across calls without reallocating.
    cdef Py_ssize_t i
    for i in range(capacity):
        occupied[i] = <uint8_t>0


cdef bint _pair_seen_grow(uint8_t **occupied,
                          int64_t **keys_a,
                          int64_t **keys_b,
                          Py_ssize_t *capacity) noexcept nogil:
    cdef Py_ssize_t old_cap = capacity[0]
    cdef Py_ssize_t new_cap = old_cap * 2
    cdef Py_ssize_t i, slot, mask
    cdef uint8_t *new_occupied
    cdef int64_t *new_keys_a
    cdef int64_t *new_keys_b
    cdef uint64_t h
    if old_cap <= 0:
        return _pair_seen_init(occupied, keys_a, keys_b, capacity, 16)
    new_occupied = <uint8_t *>malloc(new_cap * sizeof(uint8_t))
    new_keys_a = <int64_t *>malloc(new_cap * sizeof(int64_t))
    new_keys_b = <int64_t *>malloc(new_cap * sizeof(int64_t))
    if new_occupied == NULL or new_keys_a == NULL or new_keys_b == NULL:
        if new_occupied != NULL:
            free(new_occupied)
        if new_keys_a != NULL:
            free(new_keys_a)
        if new_keys_b != NULL:
            free(new_keys_b)
        return False
    for i in range(new_cap):
        new_occupied[i] = <uint8_t>0
    mask = new_cap - 1
    for i in range(old_cap):
        if occupied[0][i] == 0:
            continue
        h = _pair_hash_key(keys_a[0][i], keys_b[0][i])
        slot = <Py_ssize_t>(h & <uint64_t>mask)
        while new_occupied[slot] != 0:
            slot = (slot + 1) & mask
        new_occupied[slot] = <uint8_t>1
        new_keys_a[slot] = keys_a[0][i]
        new_keys_b[slot] = keys_b[0][i]
    free(occupied[0])
    free(keys_a[0])
    free(keys_b[0])
    occupied[0] = new_occupied
    keys_a[0] = new_keys_a
    keys_b[0] = new_keys_b
    capacity[0] = new_cap
    return True


cdef bint _pair_seen_insert(uint8_t **occupied,
                            int64_t **keys_a,
                            int64_t **keys_b,
                            Py_ssize_t *capacity,
                            Py_ssize_t *count,
                            int64_t key_a,
                            int64_t key_b,
                            bint *failed) noexcept nogil:
    cdef Py_ssize_t slot, mask
    cdef uint64_t h
    if capacity[0] <= 0:
        if not _pair_seen_init(occupied, keys_a, keys_b, capacity, 16):
            failed[0] = True
            return False
    if (count[0] + 1) * 2 >= capacity[0]:
        if not _pair_seen_grow(occupied, keys_a, keys_b, capacity):
            failed[0] = True
            return False
    mask = capacity[0] - 1
    h = _pair_hash_key(key_a, key_b)
    slot = <Py_ssize_t>(h & <uint64_t>mask)
    while occupied[0][slot] != 0:
        if keys_a[0][slot] == key_a and keys_b[0][slot] == key_b:
            return False
        slot = (slot + 1) & mask
    occupied[0][slot] = <uint8_t>1
    keys_a[0][slot] = key_a
    keys_b[0][slot] = key_b
    count[0] += 1
    return True


cdef inline int _popcount64(uint64_t x) noexcept nogil:
    cdef int count = 0
    while x != 0:
        x &= x - <uint64_t>1
        count += 1
    return count


cdef Py_ssize_t _dedupe_pair_buffer(int64_t *buf, Py_ssize_t count):
    cdef Py_ssize_t cap = 16
    cdef Py_ssize_t i, slot, mask, unique_count = 0
    cdef int64_t a, b, ka, kb
    cdef uint8_t *occupied
    cdef int64_t *keys_a
    cdef int64_t *keys_b
    cdef uint64_t h

    if count <= 1:
        return count
    while cap < count * 2:
        cap <<= 1
    mask = cap - 1
    occupied = <uint8_t *>malloc(cap * sizeof(uint8_t))
    keys_a = <int64_t *>malloc(cap * sizeof(int64_t))
    keys_b = <int64_t *>malloc(cap * sizeof(int64_t))
    if occupied == NULL or keys_a == NULL or keys_b == NULL:
        if occupied != NULL:
            free(occupied)
        if keys_a != NULL:
            free(keys_a)
        if keys_b != NULL:
            free(keys_b)
        return count
    for i in range(cap):
        occupied[i] = <uint8_t>0

    for i in range(count):
        a = buf[2 * i]
        b = buf[2 * i + 1]
        if a <= b:
            ka = a
            kb = b
        else:
            ka = b
            kb = a
        h = _pair_hash_key(ka, kb)
        slot = <Py_ssize_t>(h & <uint64_t>mask)
        while occupied[slot] != 0:
            if keys_a[slot] == ka and keys_b[slot] == kb:
                break
            slot = (slot + 1) & mask
        if occupied[slot] != 0:
            continue
        occupied[slot] = <uint8_t>1
        keys_a[slot] = ka
        keys_b[slot] = kb
        buf[2 * unique_count] = a
        buf[2 * unique_count + 1] = b
        unique_count += 1

    free(occupied)
    free(keys_a)
    free(keys_b)
    return unique_count


cdef inline double _row_l2_sq_until(double[:, :] a,
                                    double[:, :] b,
                                    Py_ssize_t ai,
                                    Py_ssize_t bi,
                                    Py_ssize_t n_dim,
                                    double tau_sq) nogil:
    cdef Py_ssize_t d
    cdef double diff
    cdef double acc = 0.0
    for d in range(n_dim):
        diff = a[ai, d] - b[bi, d]
        acc = acc + diff * diff
        if acc > tau_sq:
            return acc
    return acc


cdef void _scan_tree_scalar(int64_t node,
                            int64_t[:] left,
                            int64_t[:] right,
                            double[:] values,
                            int64_t[:] window_idx,
                            uint8_t[:] active,
                            double lower,
                            double upper,
                            int64_t ridx,
                            long sid_r,
                            long time_r,
                            long[:] win_sid_idx,
                            long[:] win_time,
                            int64_t **buf,
                            Py_ssize_t *count,
                            Py_ssize_t *cap,
                            bint *failed) noexcept nogil:
    cdef double val
    cdef int64_t other_idx
    cdef long sid_o, time_o

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_scalar(left[node], left, right, values, window_idx, active, lower, upper,
                          ridx, sid_r, time_r, win_sid_idx, win_time, buf, count, cap, failed)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0:
        other_idx = window_idx[node]
        if other_idx != ridx:
            sid_o = win_sid_idx[other_idx]
            time_o = win_time[other_idx]
            if not (sid_o == sid_r and time_o == time_r):
                if not _append_pair(buf, count, cap, ridx, other_idx):
                    failed[0] = True
                    return

    if val <= upper:
        _scan_tree_scalar(right[node], left, right, values, window_idx, active, lower, upper,
                          ridx, sid_r, time_r, win_sid_idx, win_time, buf, count, cap, failed)


cdef void _scan_tree_scalar_dist(int64_t node,
                                 int64_t[:] left,
                                 int64_t[:] right,
                                 double[:] values,
                                 int64_t[:] window_idx,
                                 uint8_t[:] active,
                                 double lower,
                                 double upper,
                                 double rvalue,
                                 int64_t ridx,
                                 long sid_r,
                                 long time_r,
                                 long[:] win_sid_idx,
                                 long[:] win_time,
                                 int64_t **buf,
                                 double **dist_buf,
                                 Py_ssize_t *count,
                                 Py_ssize_t *cap,
                                 bint *failed) noexcept nogil:
    cdef double val
    cdef double diff
    cdef int64_t other_idx
    cdef long sid_o, time_o

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_scalar_dist(left[node], left, right, values, window_idx, active, lower, upper,
                               rvalue, ridx, sid_r, time_r, win_sid_idx, win_time,
                               buf, dist_buf, count, cap, failed)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0:
        other_idx = window_idx[node]
        if other_idx != ridx:
            sid_o = win_sid_idx[other_idx]
            time_o = win_time[other_idx]
            if not (sid_o == sid_r and time_o == time_r):
                diff = rvalue - val
                if not _append_pair_dist(buf, dist_buf, count, cap, ridx, other_idx, diff * diff):
                    failed[0] = True
                    return

    if val <= upper:
        _scan_tree_scalar_dist(right[node], left, right, values, window_idx, active, lower, upper,
                               rvalue, ridx, sid_r, time_r, win_sid_idx, win_time,
                               buf, dist_buf, count, cap, failed)


cdef void _scan_tree_full(int64_t node,
                          int64_t[:] left,
                          int64_t[:] right,
                          double[:] values,
                          int64_t[:] window_idx,
                          uint8_t[:] active,
                          uint8_t[:] has_vector,
                          double[:, :] vectors,
                          double lower,
                          double upper,
                          int64_t rnode,
                          int64_t ridx,
                          long sid_r,
                          long time_r,
                          long[:] win_sid_idx,
                          long[:] win_time,
                          Py_ssize_t n_dim,
                          double tau_sq,
                          int64_t **buf,
                          Py_ssize_t *count,
                          Py_ssize_t *cap,
                          bint *failed) noexcept nogil:
    cdef double val
    cdef int64_t other_idx
    cdef long sid_o, time_o
    cdef Py_ssize_t d
    cdef double diff, acc

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_full(left[node], left, right, values, window_idx, active, has_vector, vectors,
                        lower, upper, rnode, ridx, sid_r, time_r, win_sid_idx, win_time,
                        n_dim, tau_sq, buf, count, cap, failed)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0 and has_vector[node] != 0:
        other_idx = window_idx[node]
        if other_idx != ridx:
            sid_o = win_sid_idx[other_idx]
            time_o = win_time[other_idx]
            if not (sid_o == sid_r and time_o == time_r):
                acc = 0.0
                for d in range(n_dim):
                    diff = vectors[rnode, d] - vectors[node, d]
                    acc += diff * diff
                    if acc > tau_sq:
                        break
                if acc <= tau_sq:
                    if not _append_pair(buf, count, cap, ridx, other_idx):
                        failed[0] = True
                        return

    if val <= upper:
        _scan_tree_full(right[node], left, right, values, window_idx, active, has_vector, vectors,
                        lower, upper, rnode, ridx, sid_r, time_r, win_sid_idx, win_time,
                        n_dim, tau_sq, buf, count, cap, failed)


cdef void _scan_tree_full_dist(int64_t node,
                               int64_t[:] left,
                               int64_t[:] right,
                               double[:] values,
                               int64_t[:] window_idx,
                               uint8_t[:] active,
                               uint8_t[:] has_vector,
                               double[:, :] vectors,
                               double lower,
                               double upper,
                               int64_t rnode,
                               int64_t ridx,
                               long sid_r,
                               long time_r,
                               long[:] win_sid_idx,
                               long[:] win_time,
                               Py_ssize_t n_dim,
                               double tau_sq,
                               int64_t **buf,
                               double **dist_buf,
                               Py_ssize_t *count,
                               Py_ssize_t *cap,
                               bint *failed,
                               Py_ssize_t *range_hits,
                               Py_ssize_t *distance_checks,
                               Py_ssize_t *after_similarity) noexcept nogil:
    cdef double val
    cdef int64_t other_idx
    cdef long sid_o, time_o
    cdef Py_ssize_t d
    cdef double diff, acc

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_full_dist(left[node], left, right, values, window_idx, active, has_vector, vectors,
                             lower, upper, rnode, ridx, sid_r, time_r, win_sid_idx, win_time,
                             n_dim, tau_sq, buf, dist_buf, count, cap, failed,
                             range_hits, distance_checks, after_similarity)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0 and has_vector[node] != 0:
        range_hits[0] += 1
        other_idx = window_idx[node]
        if other_idx != ridx:
            sid_o = win_sid_idx[other_idx]
            time_o = win_time[other_idx]
            if not (sid_o == sid_r and time_o == time_r):
                distance_checks[0] += 1
                acc = 0.0
                for d in range(n_dim):
                    diff = vectors[rnode, d] - vectors[node, d]
                    acc += diff * diff
                    if acc > tau_sq:
                        break
                if acc <= tau_sq:
                    after_similarity[0] += 1
                    if not _append_pair_dist(buf, dist_buf, count, cap, ridx, other_idx, acc):
                        failed[0] = True
                        return

    if val <= upper:
        _scan_tree_full_dist(right[node], left, right, values, window_idx, active, has_vector, vectors,
                             lower, upper, rnode, ridx, sid_r, time_r, win_sid_idx, win_time,
                             n_dim, tau_sq, buf, dist_buf, count, cap, failed,
                             range_hits, distance_checks, after_similarity)


cdef void _scan_tree_scalar_meta(int64_t node,
                                 int64_t[:] left,
                                 int64_t[:] right,
                                 double[:] values,
                                 int64_t[:] window_idx,
                                 int64_t[:] sid_idx,
                                 int64_t[:] time_idx,
                                 int64_t[:] win_size,
                                 uint8_t[:] active,
                                 double lower,
                                 double upper,
                                 double rvalue,
                                 int64_t ridx,
                                 int64_t sid_r,
                                 int64_t time_r,
                                 int64_t w_r,
                                 bint want_dist,
                                 int64_t **buf,
                                 double **dist_buf,
                                 Py_ssize_t *count,
                                 Py_ssize_t *cap,
                                 bint *failed) noexcept nogil:
    cdef double val
    cdef double diff
    cdef int64_t other_idx

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_scalar_meta(left[node], left, right, values, window_idx, sid_idx, time_idx, win_size, active,
                               lower, upper, rvalue, ridx, sid_r, time_r, w_r, want_dist,
                               buf, dist_buf, count, cap, failed)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0:
        other_idx = window_idx[node]
        if (
            other_idx != ridx
            and win_size[node] == w_r
            and not (sid_idx[node] == sid_r and time_idx[node] == time_r)
        ):
            if want_dist:
                diff = rvalue - val
                if not _append_pair_dist(buf, dist_buf, count, cap, ridx, other_idx, diff * diff):
                    failed[0] = True
                    return
            else:
                if not _append_pair(buf, count, cap, ridx, other_idx):
                    failed[0] = True
                    return

    if val <= upper:
        _scan_tree_scalar_meta(right[node], left, right, values, window_idx, sid_idx, time_idx, win_size, active,
                               lower, upper, rvalue, ridx, sid_r, time_r, w_r, want_dist,
                               buf, dist_buf, count, cap, failed)


cdef void _scan_tree_full_meta(int64_t node,
                               int64_t[:] left,
                               int64_t[:] right,
                               double[:] values,
                               int64_t[:] window_idx,
                               int64_t[:] sid_idx,
                               int64_t[:] time_idx,
                               int64_t[:] win_size,
                               uint8_t[:] active,
                               uint8_t[:] has_vector,
                               double[:, :] vectors,
                               int64_t[:] dim_order,
                               double lower,
                               double upper,
                               int64_t rnode,
                               int64_t ridx,
                               int64_t sid_r,
                               int64_t time_r,
                               int64_t w_r,
                               Py_ssize_t n_dim,
                               double tau_sq,
                               bint want_dist,
                               int64_t **buf,
                               double **dist_buf,
                               Py_ssize_t *count,
                               Py_ssize_t *cap,
                               bint *failed,
                               Py_ssize_t *range_hits,
                               Py_ssize_t *distance_checks,
                               Py_ssize_t *after_similarity) noexcept nogil:
    cdef double val
    cdef int64_t other_idx
    cdef Py_ssize_t dpos
    cdef int64_t d
    cdef double diff, acc

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_full_meta(left[node], left, right, values, window_idx, sid_idx, time_idx, win_size,
                             active, has_vector, vectors, dim_order, lower, upper, rnode, ridx, sid_r,
                             time_r, w_r, n_dim, tau_sq, want_dist, buf, dist_buf, count, cap, failed,
                             range_hits, distance_checks, after_similarity)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0 and has_vector[node] != 0:
        range_hits[0] += 1
        other_idx = window_idx[node]
        if (
            other_idx != ridx
            and win_size[node] == w_r
            and not (sid_idx[node] == sid_r and time_idx[node] == time_r)
        ):
            distance_checks[0] += 1
            acc = 0.0
            for dpos in range(n_dim):
                d = dim_order[dpos]
                diff = vectors[rnode, d] - vectors[node, d]
                acc += diff * diff
                if acc > tau_sq:
                    break
            if acc <= tau_sq:
                after_similarity[0] += 1
                if want_dist:
                    if not _append_pair_dist(buf, dist_buf, count, cap, ridx, other_idx, acc):
                        failed[0] = True
                        return
                else:
                    if not _append_pair(buf, count, cap, ridx, other_idx):
                        failed[0] = True
                        return

    if val <= upper:
        _scan_tree_full_meta(right[node], left, right, values, window_idx, sid_idx, time_idx, win_size,
                             active, has_vector, vectors, dim_order, lower, upper, rnode, ridx, sid_r,
                             time_r, w_r, n_dim, tau_sq, want_dist, buf, dist_buf, count, cap, failed,
                             range_hits, distance_checks, after_similarity)


cdef void _scan_tree_full_meta_cosine(int64_t node,
                                      int64_t[:] left,
                                      int64_t[:] right,
                                      double[:] values,
                                      int64_t[:] window_idx,
                                      int64_t[:] sid_idx,
                                      int64_t[:] win_sid_rank,
                                      int64_t[:] time_idx,
                                      int64_t[:] win_size,
                                      uint8_t[:] active,
                                      uint8_t[:] has_vector,
                                      double[:, :] vectors,
                                      int64_t[:] dim_order,
                                      double lower,
                                      double upper,
                                      double lower_neg,
                                      double upper_neg,
                                      int64_t rnode,
                                      int64_t ridx,
                                      int64_t sid_r,
                                      int64_t rank_r,
                                      int64_t time_r,
                                      int64_t w_r,
                                      Py_ssize_t n_dim,
                                      double gamma,
                                      bint signed_abs,
                                      int64_t **buf,
                                      Py_ssize_t *count,
                                      Py_ssize_t *cap,
                                      bint *failed,
                                      Py_ssize_t *range_hits,
                                      Py_ssize_t *unique_pre_dot_pairs,
                                      Py_ssize_t *duplicate_pre_dot_pairs,
                                      Py_ssize_t *dot_checks,
                                      Py_ssize_t *after_similarity,
                                      uint8_t **pair_seen_occupied,
                                      int64_t **pair_seen_a,
                                      int64_t **pair_seen_b,
                                      Py_ssize_t *pair_seen_capacity,
                                      Py_ssize_t *pair_seen_count) noexcept nogil:
    # (2026-07-06) Fused single traversal over +q and -q for signed_abs mode,
    # replacing two full separate tree walks. `lower`/`upper` and
    # `lower_neg`/`upper_neg` are each independently correct (necessary)
    # bounds from _projection_interval; `env_lower`/`env_upper` is only used
    # to decide which subtrees are worth descending into (a superset of
    # either range), while leaf acceptance still tests the two exact ranges
    # separately, so this changes no accept/reject decision. This is a real
    # win specifically when gamma is loose (the two ranges overlap heavily
    # or coincide -- e.g. at query value 0, they're identical -- so the old
    # two-pass code walked nearly the same nodes twice); for tight gamma
    # with disjoint ranges it costs a bounded number of extra internal-node
    # visits in the gap, which are rejected cheaply at the leaf check. See
    # docs/implementation_log.md, "bptree two-pass query fusion".
    cdef double val
    cdef double env_lower
    cdef double env_upper
    cdef bint in_pos
    cdef bint in_neg
    cdef int64_t other_idx
    cdef Py_ssize_t dpos
    cdef int64_t d
    cdef int64_t pair_a, pair_b
    cdef double dot

    if node < 0 or failed[0]:
        return

    val = values[node]
    env_lower = lower_neg if (signed_abs and lower_neg < lower) else lower
    env_upper = upper_neg if (signed_abs and upper_neg > upper) else upper

    if val >= env_lower:
        _scan_tree_full_meta_cosine(left[node], left, right, values, window_idx, sid_idx, win_sid_rank, time_idx, win_size,
                                    active, has_vector, vectors, dim_order, lower, upper, lower_neg, upper_neg, rnode, ridx, sid_r,
                                    rank_r, time_r, w_r, n_dim, gamma, signed_abs, buf, count, cap, failed,
                                    range_hits, unique_pre_dot_pairs, duplicate_pre_dot_pairs, dot_checks, after_similarity,
                                    pair_seen_occupied, pair_seen_a, pair_seen_b, pair_seen_capacity, pair_seen_count)
        if failed[0]:
            return

    if val >= env_lower and val <= env_upper and active[node] != 0 and has_vector[node] != 0:
        in_pos = val >= lower and val <= upper
        in_neg = signed_abs and val >= lower_neg and val <= upper_neg
        if in_pos or in_neg:
            range_hits[0] += 1
            other_idx = window_idx[node]
            if (
                other_idx != ridx
                and win_size[node] == w_r
                and not (sid_idx[node] == sid_r and time_idx[node] == time_r)
            ):
                _canonical_window_pair(
                    ridx,
                    sid_r,
                    rank_r,
                    time_r,
                    other_idx,
                    sid_idx[node],
                    win_sid_rank[other_idx],
                    time_idx[node],
                    &pair_a,
                    &pair_b,
                )
                # (2026-07-06) Fixed a correctness bug: this used to `return`
                # on a duplicate/failed pair_seen_insert, which -- unlike the
                # equivalent loop-based `continue` in BlockedLazyIndex's row
                # scans -- aborted this ENTIRE recursive call, silently
                # skipping the right-subtree traversal below (and everything
                # under it) for the current node. Since pair_seen accumulates
                # across every query in a batched call, a later query whose
                # traversal revisits a node already marked "seen" by an
                # earlier query would truncate its own search there, losing
                # true-positive candidates with a probability that grew with
                # batch size. Now we only skip the dot-check/append for this
                # one candidate and still fall through to the right-subtree
                # recursion; only a genuine allocation failure (failed[0])
                # aborts further recursion, via the check below. See
                # docs/implementation_log.md, "bptree duplicate-pair early
                # return correctness bug".
                if _pair_seen_insert(pair_seen_occupied, pair_seen_a, pair_seen_b, pair_seen_capacity, pair_seen_count, pair_a, pair_b, failed):
                    unique_pre_dot_pairs[0] += 1
                    dot_checks[0] += 1
                    dot = 0.0
                    for dpos in range(n_dim):
                        d = dim_order[dpos]
                        dot += vectors[rnode, d] * vectors[node, d]
                    if (signed_abs and fabs(dot) >= gamma) or ((not signed_abs) and dot >= gamma):
                        after_similarity[0] += 1
                        if not _append_pair(buf, count, cap, ridx, other_idx):
                            failed[0] = True
                elif not failed[0]:
                    duplicate_pre_dot_pairs[0] += 1

    if failed[0]:
        return

    if val <= env_upper:
        _scan_tree_full_meta_cosine(right[node], left, right, values, window_idx, sid_idx, win_sid_rank, time_idx, win_size,
                                    active, has_vector, vectors, dim_order, lower, upper, lower_neg, upper_neg, rnode, ridx, sid_r,
                                    rank_r, time_r, w_r, n_dim, gamma, signed_abs, buf, count, cap, failed,
                                    range_hits, unique_pre_dot_pairs, duplicate_pre_dot_pairs, dot_checks, after_similarity,
                                    pair_seen_occupied, pair_seen_a, pair_seen_b, pair_seen_capacity, pair_seen_count)


cdef void _scan_tree_full_meta_neg(int64_t node,
                                   int64_t[:] left,
                                   int64_t[:] right,
                                   double[:] values,
                                   int64_t[:] window_idx,
                                   int64_t[:] sid_idx,
                                   int64_t[:] time_idx,
                                   int64_t[:] win_size,
                                   uint8_t[:] active,
                                   uint8_t[:] has_vector,
                                   double[:, :] vectors,
                                   int64_t[:] dim_order,
                                   double lower,
                                   double upper,
                                   int64_t rnode,
                                   int64_t ridx,
                                   int64_t sid_r,
                                   int64_t time_r,
                                   int64_t w_r,
                                   Py_ssize_t n_dim,
                                   double tau_sq,
                                   int64_t **buf,
                                   Py_ssize_t *count,
                                   Py_ssize_t *cap,
                                   bint *failed,
                                   Py_ssize_t *range_hits,
                                   Py_ssize_t *distance_checks,
                                   Py_ssize_t *after_similarity) noexcept nogil:
    cdef double val
    cdef int64_t other_idx
    cdef Py_ssize_t dpos
    cdef int64_t d
    cdef double diff, acc

    if node < 0 or failed[0]:
        return

    val = values[node]
    if val >= lower:
        _scan_tree_full_meta_neg(left[node], left, right, values, window_idx, sid_idx, time_idx, win_size,
                                 active, has_vector, vectors, dim_order, lower, upper, rnode, ridx, sid_r,
                                 time_r, w_r, n_dim, tau_sq, buf, count, cap, failed,
                                 range_hits, distance_checks, after_similarity)
        if failed[0]:
            return

    if val >= lower and val <= upper and active[node] != 0 and has_vector[node] != 0:
        range_hits[0] += 1
        other_idx = window_idx[node]
        if (
            other_idx != ridx
            and win_size[node] == w_r
            and not (sid_idx[node] == sid_r and time_idx[node] == time_r)
        ):
            distance_checks[0] += 1
            acc = 0.0
            for dpos in range(n_dim):
                d = dim_order[dpos]
                diff = vectors[rnode, d] + vectors[node, d]
                acc += diff * diff
                if acc > tau_sq:
                    break
            if acc <= tau_sq:
                after_similarity[0] += 1
                if not _append_pair(buf, count, cap, ridx, other_idx):
                    failed[0] = True
                    return

    if val <= upper:
        _scan_tree_full_meta_neg(right[node], left, right, values, window_idx, sid_idx, time_idx, win_size,
                                 active, has_vector, vectors, dim_order, lower, upper, rnode, ridx, sid_r,
                                 time_r, w_r, n_dim, tau_sq, buf, count, cap, failed,
                                 range_hits, distance_checks, after_similarity)


cdef class BlockedLazyIndex:
    cdef public object last_stats
    cdef Py_ssize_t _n_vectors
    cdef Py_ssize_t _block_size
    cdef Py_ssize_t _index_dims
    cdef Py_ssize_t _current_capacity
    cdef Py_ssize_t _current_count
    cdef int64_t _next_entry_id
    cdef int64_t _min_valid_time
    # (2026-07-06) Part 1 follow-up: flat typed-array block table, replacing
    # a Python list of per-block dicts -- see docs/implementation_log.md,
    # "block-level cone pruning: flat typed-array block storage". Per-row
    # fields for every closed block live contiguously in the `_bt_*` arrays
    # below; a block's own rows are the slice [offset, offset+length) of
    # those arrays. Live blocks span metadata index range
    # [_block_head, _block_head + _block_count). `_block_head` is nonzero
    # only transiently inside drop_before_time, which eagerly compacts
    # (shifts the live range back to 0) immediately after any eviction --
    # so every other reader can assume _block_head == 0 in practice, but
    # code still uses it explicitly for correctness/clarity.
    cdef Py_ssize_t _row_capacity
    cdef Py_ssize_t _bt_next_row_offset
    cdef Py_ssize_t _block_capacity
    cdef Py_ssize_t _block_head
    cdef Py_ssize_t _block_count
    cdef object _bt_vectors
    cdef object _bt_keys
    cdef object _bt_entry_ids
    cdef object _bt_window_idx
    cdef object _bt_sid_idx
    cdef object _bt_sid_rank
    cdef object _bt_time
    cdef object _bt_window_size
    cdef object _bt_residual_norms
    cdef object _bt_orders_by_dim
    cdef object _bt_sorted_keys_by_dim
    cdef object _bt_block_offset
    cdef object _bt_block_length
    cdef object _bt_block_end_time
    cdef object _bt_block_max_residual_norm
    cdef object _bt_block_min_cos_to_ref
    cdef object _bt_block_ref_direction
    cdef object _bt_block_dims
    cdef object _bt_block_dim_order
    cdef object _bt_block_bound_dims
    cdef object _bt_block_bound_min
    cdef object _bt_block_bound_max
    cdef object _current_vectors_arr
    cdef object _current_keys_arr
    cdef object _current_entry_ids_arr
    cdef object _current_window_idx_arr
    cdef object _current_sid_idx_arr
    cdef object _current_sid_rank_arr
    cdef object _current_time_arr
    cdef object _current_window_size_arr
    cdef object _current_dims_arr
    cdef object _current_dim_order_arr
    cdef object _last_entry_ids_arr
    cdef object _last_vectors_arr
    cdef object _last_keys_arr
    cdef object _last_window_idx_arr
    cdef object _last_sid_idx_arr
    cdef object _last_sid_rank_arr
    cdef object _last_time_arr
    cdef object _last_window_size_arr
    cdef object _win_sid_idx_arr
    cdef object _win_sid_rank_arr
    cdef object _win_time_arr
    cdef object _win_size_arr
    cdef Py_ssize_t _win_capacity
    cdef bint _bucketed_mode
    cdef double _bucket_width
    cdef Py_ssize_t _bound_dims
    cdef bint _bound_dim_selection_variance
    cdef bint _enable_block_ub_pruning
    cdef bint _enable_row_ub_pruning
    cdef bint _block_similarity_assignment
    cdef Py_ssize_t _max_open_blocks

    def __cinit__(self,
                  Py_ssize_t n_vectors=1,
                  Py_ssize_t block_size=1024,
                  Py_ssize_t index_dims=1,
                  Py_ssize_t initial_capacity=1024,
                  Py_ssize_t bound_dims=0,
                  bint bound_dim_selection_variance=True,
                  bint enable_block_ub_pruning=False,
                  bint enable_row_ub_pruning=False,
                  bint block_similarity_assignment=False,
                  Py_ssize_t max_open_blocks=4):
        if n_vectors <= 0:
            n_vectors = 1
        if block_size <= 0:
            block_size = 1024
        if initial_capacity < 16:
            initial_capacity = 16
        self._n_vectors = n_vectors
        self._block_size = block_size
        self._index_dims = max(1, min(index_dims, n_vectors))
        # (2026-07-06) Part 1: bound_dims=0 (default) disables Cauchy-Schwarz
        # block/row upper-bound pruning entirely regardless of the enable
        # flags -- opt-in, not opt-out, per this session's "point 1" lesson
        # about defaults. See docs/implementation_log.md.
        self._bound_dims = max(0, min(bound_dims, n_vectors))
        self._bound_dim_selection_variance = bound_dim_selection_variance
        # (2026-07-06) Block-level pruning uses a cone/angular bound over the
        # *full* vector (see _compute_cone_metadata) -- independent of
        # candidate_bound_dims, unlike row-level pruning which still uses
        # the per-coordinate-box + Cauchy-Schwarz-residual bound over just
        # the bound dims.
        self._enable_block_ub_pruning = enable_block_ub_pruning
        self._enable_row_ub_pruning = enable_row_ub_pruning and self._bound_dims > 0
        # (2026-07-06) Part 1 follow-up: block-level cone pruning is only
        # tight if a block's rows are already angularly close -- but blocks
        # were previously just chronological insertion chunks (whatever
        # arrived consecutively), unrelated to similarity. On real
        # correlation-search workloads (unlike a synthetic benchmark that
        # happens to insert each cluster's members consecutively), this
        # meant min_cos_to_ref was essentially always loose and
        # num_blocks_pruned_by_ub stayed at 0. block_similarity_assignment
        # groups each closing block's rows by online "leader" clustering
        # (see _leader_split_groups) instead, at block-close time only --
        # opt-in, off by default. See docs/implementation_log.md.
        self._block_similarity_assignment = block_similarity_assignment
        self._max_open_blocks = max(1, max_open_blocks)
        self._current_capacity = initial_capacity
        self._current_count = 0
        self._next_entry_id = 0
        self._min_valid_time = -9223372036854775807
        self._row_capacity = initial_capacity
        self._bt_next_row_offset = 0
        self._block_capacity = 16
        self._block_head = 0
        self._block_count = 0
        self._bt_vectors = np.empty((self._row_capacity, self._n_vectors), dtype=np.float64)
        self._bt_keys = np.empty((self._row_capacity, self._index_dims), dtype=np.float64)
        self._bt_entry_ids = np.empty(self._row_capacity, dtype=np.int64)
        self._bt_window_idx = np.empty(self._row_capacity, dtype=np.int64)
        self._bt_sid_idx = np.empty(self._row_capacity, dtype=np.int64)
        self._bt_sid_rank = np.empty(self._row_capacity, dtype=np.int64)
        self._bt_time = np.empty(self._row_capacity, dtype=np.int64)
        self._bt_window_size = np.empty(self._row_capacity, dtype=np.int64)
        self._bt_residual_norms = np.empty(self._row_capacity, dtype=np.float64)
        self._bt_orders_by_dim = np.empty((self._index_dims, self._row_capacity), dtype=np.int64)
        self._bt_sorted_keys_by_dim = np.empty((self._index_dims, self._row_capacity), dtype=np.float64)
        self._bt_block_offset = np.empty(self._block_capacity, dtype=np.int64)
        self._bt_block_length = np.empty(self._block_capacity, dtype=np.int64)
        self._bt_block_end_time = np.empty(self._block_capacity, dtype=np.int64)
        self._bt_block_max_residual_norm = np.empty(self._block_capacity, dtype=np.float64)
        self._bt_block_min_cos_to_ref = np.empty(self._block_capacity, dtype=np.float64)
        self._bt_block_ref_direction = np.empty((self._block_capacity, self._n_vectors), dtype=np.float64)
        self._bt_block_dims = np.empty((self._block_capacity, self._index_dims), dtype=np.int64)
        self._bt_block_dim_order = np.empty((self._block_capacity, self._n_vectors), dtype=np.int64)
        self._bt_block_bound_dims = np.empty((self._block_capacity, self._bound_dims), dtype=np.int64)
        self._bt_block_bound_min = np.empty((self._block_capacity, self._bound_dims), dtype=np.float64)
        self._bt_block_bound_max = np.empty((self._block_capacity, self._bound_dims), dtype=np.float64)
        self._current_vectors_arr = np.empty((self._current_capacity, self._n_vectors), dtype=np.float64)
        self._current_keys_arr = np.empty((self._current_capacity, self._index_dims), dtype=np.float64)
        self._current_entry_ids_arr = np.empty(self._current_capacity, dtype=np.int64)
        self._current_window_idx_arr = np.empty(self._current_capacity, dtype=np.int64)
        self._current_sid_idx_arr = np.empty(self._current_capacity, dtype=np.int64)
        self._current_sid_rank_arr = np.empty(self._current_capacity, dtype=np.int64)
        self._current_time_arr = np.empty(self._current_capacity, dtype=np.int64)
        self._current_window_size_arr = np.empty(self._current_capacity, dtype=np.int64)
        self._current_dims_arr = np.arange(self._index_dims, dtype=np.int64)
        self._current_dim_order_arr = np.arange(self._n_vectors, dtype=np.int64)
        self._last_entry_ids_arr = np.empty(0, dtype=np.int64)
        self._last_vectors_arr = np.empty((0, self._n_vectors), dtype=np.float64)
        self._last_keys_arr = np.empty((0, self._index_dims), dtype=np.float64)
        self._last_window_idx_arr = np.empty(0, dtype=np.int64)
        self._last_sid_idx_arr = np.empty(0, dtype=np.int64)
        self._last_sid_rank_arr = np.empty(0, dtype=np.int64)
        self._last_time_arr = np.empty(0, dtype=np.int64)
        self._last_window_size_arr = np.empty(0, dtype=np.int64)
        self._win_capacity = initial_capacity
        self._win_sid_idx_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_sid_rank_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_time_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._win_size_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._bucketed_mode = False
        self._bucket_width = 0.0
        self.last_stats = {}

    def configure_bucketed(self, double bucket_width=0.0):
        self._bucketed_mode = True
        self._bucket_width = bucket_width
        return self

    cdef void _ensure_current_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        if need <= self._current_capacity:
            return
        new_cap = self._current_capacity
        while new_cap < need:
            new_cap *= 2
        new_vectors = np.empty((new_cap, self._n_vectors), dtype=np.float64)
        new_keys = np.empty((new_cap, self._index_dims), dtype=np.float64)
        new_entry = np.empty(new_cap, dtype=np.int64)
        new_win = np.empty(new_cap, dtype=np.int64)
        new_sid = np.empty(new_cap, dtype=np.int64)
        new_rank = np.empty(new_cap, dtype=np.int64)
        new_time = np.empty(new_cap, dtype=np.int64)
        new_w = np.empty(new_cap, dtype=np.int64)
        if self._current_count > 0:
            n = self._current_count
            new_vectors[:n, :] = self._current_vectors_arr[:n, :]
            new_keys[:n, :] = self._current_keys_arr[:n, :]
            new_entry[:n] = self._current_entry_ids_arr[:n]
            new_win[:n] = self._current_window_idx_arr[:n]
            new_sid[:n] = self._current_sid_idx_arr[:n]
            new_rank[:n] = self._current_sid_rank_arr[:n]
            new_time[:n] = self._current_time_arr[:n]
            new_w[:n] = self._current_window_size_arr[:n]
        self._current_vectors_arr = new_vectors
        self._current_keys_arr = new_keys
        self._current_entry_ids_arr = new_entry
        self._current_window_idx_arr = new_win
        self._current_sid_idx_arr = new_sid
        self._current_sid_rank_arr = new_rank
        self._current_time_arr = new_time
        self._current_window_size_arr = new_w
        self._current_capacity = new_cap

    cdef void _ensure_window_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        if need <= self._win_capacity:
            return
        new_cap = self._win_capacity
        while new_cap < need:
            new_cap *= 2
        new_sid = np.full(new_cap, -1, dtype=np.int64)
        new_rank = np.full(new_cap, -1, dtype=np.int64)
        new_time = np.zeros(new_cap, dtype=np.int64)
        new_w = np.zeros(new_cap, dtype=np.int64)
        if self._win_capacity > 0:
            new_sid[:self._win_capacity] = self._win_sid_idx_arr[:self._win_capacity]
            new_rank[:self._win_capacity] = self._win_sid_rank_arr[:self._win_capacity]
            new_time[:self._win_capacity] = self._win_time_arr[:self._win_capacity]
            new_w[:self._win_capacity] = self._win_size_arr[:self._win_capacity]
        self._win_sid_idx_arr = new_sid
        self._win_sid_rank_arr = new_rank
        self._win_time_arr = new_time
        self._win_size_arr = new_w
        self._win_capacity = new_cap

    cdef void _ensure_row_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap, n
        if need <= self._row_capacity:
            return
        new_cap = self._row_capacity
        if new_cap <= 0:
            new_cap = 16
        while new_cap < need:
            new_cap *= 2
        new_vectors = np.empty((new_cap, self._n_vectors), dtype=np.float64)
        new_keys = np.empty((new_cap, self._index_dims), dtype=np.float64)
        new_entry = np.empty(new_cap, dtype=np.int64)
        new_win = np.empty(new_cap, dtype=np.int64)
        new_sid = np.empty(new_cap, dtype=np.int64)
        new_rank = np.empty(new_cap, dtype=np.int64)
        new_time = np.empty(new_cap, dtype=np.int64)
        new_w = np.empty(new_cap, dtype=np.int64)
        new_residual = np.empty(new_cap, dtype=np.float64)
        new_orders_by_dim = np.empty((self._index_dims, new_cap), dtype=np.int64)
        new_sorted_keys_by_dim = np.empty((self._index_dims, new_cap), dtype=np.float64)
        n = self._bt_next_row_offset
        if n > 0:
            new_vectors[:n, :] = self._bt_vectors[:n, :]
            new_keys[:n, :] = self._bt_keys[:n, :]
            new_entry[:n] = self._bt_entry_ids[:n]
            new_win[:n] = self._bt_window_idx[:n]
            new_sid[:n] = self._bt_sid_idx[:n]
            new_rank[:n] = self._bt_sid_rank[:n]
            new_time[:n] = self._bt_time[:n]
            new_w[:n] = self._bt_window_size[:n]
            new_residual[:n] = self._bt_residual_norms[:n]
            new_orders_by_dim[:, :n] = self._bt_orders_by_dim[:, :n]
            new_sorted_keys_by_dim[:, :n] = self._bt_sorted_keys_by_dim[:, :n]
        self._bt_vectors = new_vectors
        self._bt_keys = new_keys
        self._bt_entry_ids = new_entry
        self._bt_window_idx = new_win
        self._bt_sid_idx = new_sid
        self._bt_sid_rank = new_rank
        self._bt_time = new_time
        self._bt_window_size = new_w
        self._bt_residual_norms = new_residual
        self._bt_orders_by_dim = new_orders_by_dim
        self._bt_sorted_keys_by_dim = new_sorted_keys_by_dim
        self._row_capacity = new_cap

    cdef void _ensure_block_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap, n
        if need <= self._block_capacity:
            return
        new_cap = self._block_capacity
        if new_cap <= 0:
            new_cap = 16
        while new_cap < need:
            new_cap *= 2
        new_offset = np.empty(new_cap, dtype=np.int64)
        new_length = np.empty(new_cap, dtype=np.int64)
        new_end_time = np.empty(new_cap, dtype=np.int64)
        new_max_residual = np.empty(new_cap, dtype=np.float64)
        new_min_cos = np.empty(new_cap, dtype=np.float64)
        new_ref_direction = np.empty((new_cap, self._n_vectors), dtype=np.float64)
        new_dims = np.empty((new_cap, self._index_dims), dtype=np.int64)
        new_dim_order = np.empty((new_cap, self._n_vectors), dtype=np.int64)
        new_bound_dims = np.empty((new_cap, self._bound_dims), dtype=np.int64)
        new_bound_min = np.empty((new_cap, self._bound_dims), dtype=np.float64)
        new_bound_max = np.empty((new_cap, self._bound_dims), dtype=np.float64)
        n = self._block_head + self._block_count
        if n > 0:
            new_offset[:n] = self._bt_block_offset[:n]
            new_length[:n] = self._bt_block_length[:n]
            new_end_time[:n] = self._bt_block_end_time[:n]
            new_max_residual[:n] = self._bt_block_max_residual_norm[:n]
            new_min_cos[:n] = self._bt_block_min_cos_to_ref[:n]
            new_ref_direction[:n, :] = self._bt_block_ref_direction[:n, :]
            new_dims[:n, :] = self._bt_block_dims[:n, :]
            new_dim_order[:n, :] = self._bt_block_dim_order[:n, :]
            new_bound_dims[:n, :] = self._bt_block_bound_dims[:n, :]
            new_bound_min[:n, :] = self._bt_block_bound_min[:n, :]
            new_bound_max[:n, :] = self._bt_block_bound_max[:n, :]
        self._bt_block_offset = new_offset
        self._bt_block_length = new_length
        self._bt_block_end_time = new_end_time
        self._bt_block_max_residual_norm = new_max_residual
        self._bt_block_min_cos_to_ref = new_min_cos
        self._bt_block_ref_direction = new_ref_direction
        self._bt_block_dims = new_dims
        self._bt_block_dim_order = new_dim_order
        self._bt_block_bound_dims = new_bound_dims
        self._bt_block_bound_min = new_bound_min
        self._bt_block_bound_max = new_bound_max
        self._block_capacity = new_cap

    cdef Py_ssize_t _total_block_rows(self):
        if self._block_count <= 0:
            return 0
        return int(np.sum(self._bt_block_length[self._block_head:self._block_head + self._block_count]))

    cdef object _select_dims(self, np.ndarray[np.float64_t, ndim=2] vectors):
        cdef Py_ssize_t n = vectors.shape[0]
        cdef Py_ssize_t d = vectors.shape[1]
        cdef Py_ssize_t keep = self._index_dims
        if keep >= d:
            return np.arange(d, dtype=np.int64)
        if n <= 1:
            return np.arange(keep, dtype=np.int64)
        variances = np.var(vectors, axis=0)
        return np.ascontiguousarray(np.argsort(variances)[::-1][:keep], dtype=np.int64)

    cdef object _select_bound_dims(self, np.ndarray[np.float64_t, ndim=2] vectors):
        # (2026-07-06) Part 1: independent of _select_dims (the indexing
        # dims used for the sorted-range candidate search) -- a block's
        # bound dims can be a different count and/or selection criterion,
        # controlled by candidate_bound_dims/candidate_bound_dim_selection.
        cdef Py_ssize_t n = vectors.shape[0]
        cdef Py_ssize_t d = vectors.shape[1]
        cdef Py_ssize_t keep = self._bound_dims
        if keep <= 0:
            return np.empty(0, dtype=np.int64)
        if keep >= d:
            return np.arange(d, dtype=np.int64)
        if not self._bound_dim_selection_variance or n <= 1:
            return np.arange(keep, dtype=np.int64)
        variances = np.var(vectors, axis=0)
        return np.ascontiguousarray(np.argsort(variances)[::-1][:keep], dtype=np.int64)

    cdef object _compute_bound_metadata(self, np.ndarray[np.float64_t, ndim=2] vectors):
        # (2026-07-06) Part 1: returns (bound_dims, bound_min, bound_max,
        # residual_norms, max_residual_norm) for a block's rows -- computed
        # once when a block is built (closed, or the current-block view is
        # constructed fresh each query batch), reused across every query
        # against that block. See the module-level bound helper functions
        # for how these are used, and docs/implementation_log.md.
        cdef Py_ssize_t n = vectors.shape[0]
        bound_dims = np.ascontiguousarray(self._select_bound_dims(vectors), dtype=np.int64)
        if bound_dims.shape[0] == 0 or n == 0:
            return (
                bound_dims,
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
                0.0,
            )
        bound_vectors = vectors[:, bound_dims]
        bound_min = np.ascontiguousarray(np.min(bound_vectors, axis=0), dtype=np.float64)
        bound_max = np.ascontiguousarray(np.max(bound_vectors, axis=0), dtype=np.float64)
        norm_i_sq = np.sum(bound_vectors * bound_vectors, axis=1)
        residual_norms = np.ascontiguousarray(np.sqrt(np.maximum(0.0, 1.0 - norm_i_sq)), dtype=np.float64)
        max_residual_norm = float(np.max(residual_norms))
        return (bound_dims, bound_min, bound_max, residual_norms, max_residual_norm)

    cdef object _compute_cone_metadata(self, np.ndarray[np.float64_t, ndim=2] vectors):
        # (2026-07-06) Part 1 follow-up: block-level pruning via an
        # axis-aligned per-coordinate box (bound_min/bound_max above) was
        # measured to essentially never fire on clustered/correlated data --
        # a box treats every dimension as independently worst-cased, which
        # is far looser than a real cluster's shape (all rows near a common
        # direction, i.e. correlated across dimensions, not independent) --
        # and this gap widens with more dimensions. Replaced for
        # block-level pruning specifically with a tighter, genuinely
        # angular bound: a block's rows all lie in a spherical cap around a
        # reference direction `ref` (the block's own mean direction) with
        # cap radius `min_cos_to_ref` = the smallest cosine similarity any
        # row in the block has to `ref`. Given a query q, the maximum (and,
        # for signed_abs, minimum) achievable q.x for *any* unit vector x in
        # that cap is exactly what _projection_interval already computes --
        # the same "max/min cosine to q over a spherical cap" identity this
        # codebase already relies on for the sorted-range candidate search
        # itself, just applied here to a whole block's angular extent
        # instead of a single indexed coordinate. Uses the *full* vector
        # (not just candidate_bound_dims), since the point of block-level
        # pruning is to avoid O(rows_in_block * n_vectors) work at the cost
        # of one O(n_vectors) check per (query, block) pair -- unlike
        # row-level pruning, there is no benefit to shrinking the dimension
        # count here. See docs/implementation_log.md, "Part 1 follow-up:
        # cone-based block bound".
        cdef Py_ssize_t n = vectors.shape[0]
        if n == 0:
            return (np.zeros(self._n_vectors, dtype=np.float64), -1.0)
        ref = np.sum(vectors, axis=0)
        norm = float(np.linalg.norm(ref))
        if norm <= 0.0 or not np.isfinite(norm):
            # Degenerate (e.g. rows exactly cancel out) -- min_cos_to_ref=-1.0
            # makes _projection_interval return the widest possible interval
            # ([-1, 1]), which safely disables pruning for this block only
            # (never a false negative, just no pruning opportunity here).
            return (np.zeros(self._n_vectors, dtype=np.float64), -1.0)
        ref = ref / norm
        cos_to_ref = vectors @ ref
        min_cos_to_ref = float(np.min(cos_to_ref))
        return (np.ascontiguousarray(ref, dtype=np.float64), min_cos_to_ref)

    cdef object _leader_split_groups(self, double[:, :] vectors, Py_ssize_t max_groups):
        # (2026-07-06) Part 1 follow-up: online "leader" clustering (a
        # classic single-pass approximate clustering method), run once per
        # block-close event, not on the hot per-row insert path. Each row is
        # assigned to whichever of up to `max_groups` running leader sums it
        # is most cosine-similar to; while under the cap, a row instead
        # always starts a fresh leader rather than being forced into a
        # possibly-unrelated existing one. Returns (group_id, num_groups):
        # group_id[i] in [0, num_groups) for each input row. See
        # docs/implementation_log.md, "block-level cone pruning:
        # similarity-aware block assignment".
        cdef Py_ssize_t n = vectors.shape[0]
        cdef Py_ssize_t n_dim = vectors.shape[1]
        cdef Py_ssize_t i, d, k, best_k
        cdef Py_ssize_t num_active = 0
        cdef double dot, leader_norm_sq, cos, best_cos
        cdef np.ndarray[np.int64_t, ndim=1] group_id_arr = np.empty(n, dtype=np.int64)
        cdef int64_t[:] group_id = group_id_arr
        cdef np.ndarray[np.float64_t, ndim=2] leader_sum_arr = np.zeros((max_groups, n_dim), dtype=np.float64)
        cdef double[:, :] leader_sum = leader_sum_arr
        for i in range(n):
            if num_active < max_groups:
                best_k = num_active
                num_active += 1
            else:
                best_k = 0
                best_cos = -1e18
                for k in range(num_active):
                    dot = 0.0
                    leader_norm_sq = 0.0
                    for d in range(n_dim):
                        dot += vectors[i, d] * leader_sum[k, d]
                        leader_norm_sq += leader_sum[k, d] * leader_sum[k, d]
                    if leader_norm_sq > 0.0:
                        cos = dot / sqrt(leader_norm_sq)
                    else:
                        cos = -1.0
                    if cos > best_cos:
                        best_cos = cos
                        best_k = k
            group_id[i] = best_k
            for d in range(n_dim):
                leader_sum[best_k, d] += vectors[i, d]
        return (group_id_arr, int(num_active))

    cdef object _dim_order_from_dims(self, object dims_obj):
        cdef np.ndarray[np.int64_t, ndim=1] dims = np.asarray(dims_obj, dtype=np.int64).ravel()
        cdef np.ndarray[np.uint8_t, ndim=1] seen = np.zeros(self._n_vectors, dtype=np.uint8)
        cdef np.ndarray[np.int64_t, ndim=1] order = np.empty(self._n_vectors, dtype=np.int64)
        cdef Py_ssize_t i, fill = 0
        cdef int64_t dim
        for i in range(dims.shape[0]):
            dim = dims[i]
            if dim < 0 or dim >= self._n_vectors:
                continue
            if seen[dim] != 0:
                continue
            order[fill] = dim
            seen[dim] = 1
            fill += 1
        for dim in range(self._n_vectors):
            if seen[dim] == 0:
                order[fill] = dim
                fill += 1
        if fill <= 0:
            return np.arange(self._n_vectors, dtype=np.int64)
        return np.ascontiguousarray(order[:fill], dtype=np.int64)

    cdef void _refresh_current_dim_order(self):
        cdef np.ndarray[np.float64_t, ndim=2] vectors
        cdef object dims
        if self._current_count <= 0:
            dims = np.arange(self._index_dims, dtype=np.int64)
        else:
            vectors = np.ascontiguousarray(
                self._current_vectors_arr[:self._current_count, :],
                dtype=np.float64,
            )
            dims = self._select_dims(vectors)
        self._current_dims_arr = np.ascontiguousarray(dims, dtype=np.int64)
        self._current_dim_order_arr = self._dim_order_from_dims(self._current_dims_arr)

    cdef void _append_block_to_table(self, np.ndarray[np.float64_t, ndim=2] vectors,
                                      np.ndarray[np.float64_t, ndim=2] key_values,
                                      np.ndarray[np.int64_t, ndim=1] entry_ids,
                                      np.ndarray[np.int64_t, ndim=1] win_idx,
                                      np.ndarray[np.int64_t, ndim=1] sid_idx,
                                      np.ndarray[np.int64_t, ndim=1] sid_rank,
                                      np.ndarray[np.int64_t, ndim=1] time_idx,
                                      np.ndarray[np.int64_t, ndim=1] win_size):
        cdef Py_ssize_t n = vectors.shape[0]
        cdef Py_ssize_t dim_count = int(key_values.shape[1])
        cdef Py_ssize_t dim_i, row_offset, block_idx, bw
        if n <= 0:
            return
        dims = self._select_dims(vectors)
        dim_order = self._dim_order_from_dims(dims)
        orders_by_dim = np.empty((dim_count, n), dtype=np.int64)
        sorted_keys_by_dim = np.empty((dim_count, n), dtype=np.float64)
        for dim_i in range(dim_count):
            keys_i = np.ascontiguousarray(key_values[:, dim_i], dtype=np.float64)
            order_i = np.ascontiguousarray(np.argsort(keys_i, kind="mergesort"), dtype=np.int64)
            orders_by_dim[dim_i, :] = order_i
            sorted_keys_by_dim[dim_i, :] = keys_i[order_i]
        bound_dims, bound_min, bound_max, residual_norms, max_residual_norm = self._compute_bound_metadata(vectors)
        ref_direction, min_cos_to_ref = self._compute_cone_metadata(vectors)

        self._ensure_row_capacity(self._bt_next_row_offset + n)
        row_offset = self._bt_next_row_offset
        self._bt_vectors[row_offset:row_offset + n, :] = vectors
        self._bt_keys[row_offset:row_offset + n, :] = key_values
        self._bt_entry_ids[row_offset:row_offset + n] = entry_ids
        self._bt_window_idx[row_offset:row_offset + n] = win_idx
        self._bt_sid_idx[row_offset:row_offset + n] = sid_idx
        self._bt_sid_rank[row_offset:row_offset + n] = sid_rank
        self._bt_time[row_offset:row_offset + n] = time_idx
        self._bt_window_size[row_offset:row_offset + n] = win_size
        self._bt_residual_norms[row_offset:row_offset + n] = residual_norms
        self._bt_orders_by_dim[:, row_offset:row_offset + n] = orders_by_dim
        self._bt_sorted_keys_by_dim[:, row_offset:row_offset + n] = sorted_keys_by_dim
        self._bt_next_row_offset = row_offset + n

        block_idx = self._block_head + self._block_count
        self._ensure_block_capacity(block_idx + 1)
        self._bt_block_offset[block_idx] = row_offset
        self._bt_block_length[block_idx] = n
        self._bt_block_end_time[block_idx] = int(np.max(time_idx))
        self._bt_block_max_residual_norm[block_idx] = max_residual_norm
        self._bt_block_min_cos_to_ref[block_idx] = min_cos_to_ref
        self._bt_block_ref_direction[block_idx, :] = ref_direction
        self._bt_block_dims[block_idx, :] = dims
        self._bt_block_dim_order[block_idx, :] = dim_order
        bw = bound_dims.shape[0]
        if bw > 0:
            self._bt_block_bound_dims[block_idx, :bw] = bound_dims
            self._bt_block_bound_min[block_idx, :bw] = bound_min
            self._bt_block_bound_max[block_idx, :bw] = bound_max
        self._block_count += 1

    cdef void _close_current_block(self):
        cdef Py_ssize_t n = self._current_count
        cdef Py_ssize_t num_groups, k
        if n <= 0:
            return
        vectors = np.ascontiguousarray(self._current_vectors_arr[:n, :], dtype=np.float64).copy()
        key_values = np.ascontiguousarray(self._current_keys_arr[:n, :], dtype=np.float64).copy()
        entry_ids = np.ascontiguousarray(self._current_entry_ids_arr[:n], dtype=np.int64).copy()
        win_idx = np.ascontiguousarray(self._current_window_idx_arr[:n], dtype=np.int64).copy()
        sid_idx = np.ascontiguousarray(self._current_sid_idx_arr[:n], dtype=np.int64).copy()
        sid_rank = np.ascontiguousarray(self._current_sid_rank_arr[:n], dtype=np.int64).copy()
        time_idx = np.ascontiguousarray(self._current_time_arr[:n], dtype=np.int64).copy()
        win_size = np.ascontiguousarray(self._current_window_size_arr[:n], dtype=np.int64).copy()
        # (2026-07-06) Part 1 follow-up: when enabled, split this closing
        # batch into up to _max_open_blocks groups by online leader
        # clustering (_leader_split_groups) before sealing, so block-level
        # cone pruning has a real chance of firing on real (interleaved,
        # not pre-sorted-by-cluster) streaming data. Coverage is identical
        # either way (every row still ends up in exactly one sealed block) --
        # this only changes which rows share a block, never which rows get
        # searched, so it cannot affect recall.
        if self._block_similarity_assignment and n > 1:
            group_id, num_groups = self._leader_split_groups(vectors, self._max_open_blocks)
            group_id_np = np.asarray(group_id)
            for k in range(num_groups):
                mask = group_id_np == k
                if not np.any(mask):
                    continue
                self._append_block_to_table(
                    np.ascontiguousarray(vectors[mask], dtype=np.float64),
                    np.ascontiguousarray(key_values[mask], dtype=np.float64),
                    np.ascontiguousarray(entry_ids[mask], dtype=np.int64),
                    np.ascontiguousarray(win_idx[mask], dtype=np.int64),
                    np.ascontiguousarray(sid_idx[mask], dtype=np.int64),
                    np.ascontiguousarray(sid_rank[mask], dtype=np.int64),
                    np.ascontiguousarray(time_idx[mask], dtype=np.int64),
                    np.ascontiguousarray(win_size[mask], dtype=np.int64),
                )
        else:
            self._append_block_to_table(
                vectors, key_values, entry_ids, win_idx, sid_idx, sid_rank, time_idx, win_size,
            )
        self._current_count = 0
        self._refresh_current_dim_order()

    cdef object _compute_current_view_metadata(self):
        """Compute the same per-query metadata the old current-block "view"
        dict wrapper used to carry, as a bare tuple instead -- no dict, no
        list, just the raw arrays a scan call needs. Returns None if the
        current (still-open) block is empty.

        Built fresh once per query batch (not cached/stored), since the
        current block mutates every step -- the per-dimension argsort cost
        here is bounded by _current_count, which candidate_block_size_steps
        keeps small (same asymptotic cost _append_block_to_table always
        pays when a block closes, just paid every step instead of every
        ~N steps).

        (2026-07-06) Part 1 follow-up: block_similarity_assignment
        deliberately does NOT split this still-open snapshot, only *closed*
        blocks (_close_current_block) -- an earlier attempt did split it
        too, since on a typical run most data can sit in the still-open
        current block for a long time and closed-block-only splitting has
        no effect there. But this snapshot is rebuilt from scratch on
        *every single query batch* (it mutates continuously), so splitting
        it re-paid the full leader-clustering + per-group metadata
        computation on every query instead of once per block_size
        insertions -- measured 1.2x-3x wall-clock regression (scaling with
        max_open_blocks) despite genuinely lower dot_checks, on a realistic
        interleaved-series benchmark. Reverted to unsplit; see
        docs/implementation_log.md, "block-level cone pruning:
        similarity-aware block assignment".
        """
        cdef Py_ssize_t n = self._current_count
        cdef Py_ssize_t dim_i, dim_count
        if n <= 0:
            return None
        vectors = np.ascontiguousarray(self._current_vectors_arr[:n, :], dtype=np.float64)
        key_values = np.ascontiguousarray(self._current_keys_arr[:n, :], dtype=np.float64)
        dim_count = int(key_values.shape[1])
        orders_by_dim = np.empty((dim_count, n), dtype=np.int64)
        sorted_keys_by_dim = np.empty((dim_count, n), dtype=np.float64)
        for dim_i in range(dim_count):
            keys_i = np.ascontiguousarray(key_values[:, dim_i], dtype=np.float64)
            order_i = np.ascontiguousarray(np.argsort(keys_i, kind="mergesort"), dtype=np.int64)
            orders_by_dim[dim_i, :] = order_i
            sorted_keys_by_dim[dim_i, :] = keys_i[order_i]
        bound_dims, bound_min, bound_max, residual_norms, max_residual_norm = self._compute_bound_metadata(vectors)
        ref_direction, min_cos_to_ref = self._compute_cone_metadata(vectors)
        return (
            np.ascontiguousarray(orders_by_dim, dtype=np.int64),
            np.ascontiguousarray(sorted_keys_by_dim, dtype=np.float64),
            bound_dims,
            bound_min,
            bound_max,
            residual_norms,
            max_residual_norm,
            ref_direction,
            min_cos_to_ref,
        )

    cpdef clear_recent(self):
        self._last_entry_ids_arr = np.empty(0, dtype=np.int64)
        self._last_vectors_arr = np.empty((0, self._n_vectors), dtype=np.float64)
        self._last_keys_arr = np.empty((0, self._index_dims), dtype=np.float64)
        self._last_window_idx_arr = np.empty(0, dtype=np.int64)
        self._last_sid_idx_arr = np.empty(0, dtype=np.int64)
        self._last_sid_rank_arr = np.empty(0, dtype=np.int64)
        self._last_time_arr = np.empty(0, dtype=np.int64)
        self._last_window_size_arr = np.empty(0, dtype=np.int64)

    cdef void _compact_current_after_min_time(self):
        cdef Py_ssize_t read, write = 0, d
        cdef double[:, :] vectors = self._current_vectors_arr
        cdef double[:, :] keys = self._current_keys_arr
        cdef int64_t[:] entry = self._current_entry_ids_arr
        cdef int64_t[:] win = self._current_window_idx_arr
        cdef int64_t[:] sid = self._current_sid_idx_arr
        cdef int64_t[:] rank = self._current_sid_rank_arr
        cdef int64_t[:] time_idx = self._current_time_arr
        cdef int64_t[:] win_size = self._current_window_size_arr
        if self._current_count <= 0:
            self._refresh_current_dim_order()
            return
        for read in range(self._current_count):
            if time_idx[read] < self._min_valid_time:
                continue
            if write != read:
                entry[write] = entry[read]
                win[write] = win[read]
                sid[write] = sid[read]
                rank[write] = rank[read]
                time_idx[write] = time_idx[read]
                win_size[write] = win_size[read]
                for d in range(self._n_vectors):
                    vectors[write, d] = vectors[read, d]
                for d in range(self._index_dims):
                    keys[write, d] = keys[read, d]
            write += 1
        self._current_count = write
        self._refresh_current_dim_order()

    cdef void _compact_block_table(self):
        # (2026-07-06) Part 1 follow-up: eagerly shift the live block range
        # back to offset 0 whenever drop_before_time evicts at least one
        # block, instead of a head-pointer/ring-buffer that only compacts
        # periodically. Called at most once per drop_before_time invocation
        # (typically once per step, not once per row), so this is not
        # expected to be a hot loop -- if benchmarking ever shows otherwise,
        # the gated/periodic compaction pattern already proven in
        # monitor_kernels.pyx's _maybe_compact is the natural next step, but
        # that's not warranted without measuring a real cost first. See
        # docs/implementation_log.md, "block-level cone pruning: flat
        # typed-array block storage".
        cdef Py_ssize_t n = self._block_count
        cdef Py_ssize_t row_start, row_end, row_count
        if self._block_head == 0:
            return
        if n <= 0:
            self._block_head = 0
            self._bt_next_row_offset = 0
            return
        row_start = int(self._bt_block_offset[self._block_head])
        row_end = int(self._bt_block_offset[self._block_head + n - 1] + self._bt_block_length[self._block_head + n - 1])
        row_count = row_end - row_start
        self._bt_vectors[:row_count, :] = self._bt_vectors[row_start:row_end, :].copy()
        self._bt_keys[:row_count, :] = self._bt_keys[row_start:row_end, :].copy()
        self._bt_entry_ids[:row_count] = self._bt_entry_ids[row_start:row_end].copy()
        self._bt_window_idx[:row_count] = self._bt_window_idx[row_start:row_end].copy()
        self._bt_sid_idx[:row_count] = self._bt_sid_idx[row_start:row_end].copy()
        self._bt_sid_rank[:row_count] = self._bt_sid_rank[row_start:row_end].copy()
        self._bt_time[:row_count] = self._bt_time[row_start:row_end].copy()
        self._bt_window_size[:row_count] = self._bt_window_size[row_start:row_end].copy()
        self._bt_residual_norms[:row_count] = self._bt_residual_norms[row_start:row_end].copy()
        self._bt_orders_by_dim[:, :row_count] = self._bt_orders_by_dim[:, row_start:row_end].copy()
        self._bt_sorted_keys_by_dim[:, :row_count] = self._bt_sorted_keys_by_dim[:, row_start:row_end].copy()
        self._bt_block_offset[:n] = self._bt_block_offset[self._block_head:self._block_head + n].copy() - row_start
        self._bt_block_length[:n] = self._bt_block_length[self._block_head:self._block_head + n].copy()
        self._bt_block_end_time[:n] = self._bt_block_end_time[self._block_head:self._block_head + n].copy()
        self._bt_block_max_residual_norm[:n] = self._bt_block_max_residual_norm[self._block_head:self._block_head + n].copy()
        self._bt_block_min_cos_to_ref[:n] = self._bt_block_min_cos_to_ref[self._block_head:self._block_head + n].copy()
        self._bt_block_ref_direction[:n, :] = self._bt_block_ref_direction[self._block_head:self._block_head + n, :].copy()
        self._bt_block_dims[:n, :] = self._bt_block_dims[self._block_head:self._block_head + n, :].copy()
        self._bt_block_dim_order[:n, :] = self._bt_block_dim_order[self._block_head:self._block_head + n, :].copy()
        self._bt_block_bound_dims[:n, :] = self._bt_block_bound_dims[self._block_head:self._block_head + n, :].copy()
        self._bt_block_bound_min[:n, :] = self._bt_block_bound_min[self._block_head:self._block_head + n, :].copy()
        self._bt_block_bound_max[:n, :] = self._bt_block_bound_max[self._block_head:self._block_head + n, :].copy()
        self._block_head = 0
        self._bt_next_row_offset = row_count

    cpdef drop_before_time(self, long min_valid_time):
        cdef Py_ssize_t evicted = 0
        self._min_valid_time = <int64_t>min_valid_time
        while self._block_count > 0 and int(self._bt_block_end_time[self._block_head]) < self._min_valid_time:
            self._block_head += 1
            self._block_count -= 1
            evicted += 1
        if evicted > 0:
            self._compact_block_table()
        self._compact_current_after_min_time()

    def insert_many(self,
                    values_in,
                    window_idx_in,
                    vectors_in=None,
                    sid_idx_in=None,
                    time_in=None,
                    window_size_in=None,
                    sid_rank_in=None):
        cdef np.ndarray[np.int64_t, ndim=1] window_idx_np = np.asarray(window_idx_in, dtype=np.int64).ravel()
        cdef Py_ssize_t n = window_idx_np.shape[0]
        cdef np.ndarray[np.float64_t, ndim=2] keys_np
        cdef object values_obj
        cdef np.ndarray[np.float64_t, ndim=2] vectors_np
        cdef np.ndarray[np.int64_t, ndim=1] sid_idx_np
        cdef np.ndarray[np.int64_t, ndim=1] time_np
        cdef np.ndarray[np.int64_t, ndim=1] win_size_np
        cdef np.ndarray[np.int64_t, ndim=1] sid_rank_np
        cdef np.ndarray[np.int64_t, ndim=1] entry_ids
        cdef Py_ssize_t i, d, kdim, pos
        cdef int64_t win_id

        if n == 0:
            self.clear_recent()
            return np.empty(0, dtype=np.int64)
        values_obj = np.asarray(values_in, dtype=np.float64)
        if values_obj.ndim == 1:
            if values_obj.shape[0] != n:
                raise ValueError("values size does not match window_idx size")
            keys_np = np.empty((n, self._index_dims), dtype=np.float64)
            keys_np[:, 0] = np.asarray(values_obj, dtype=np.float64).ravel()
            for kdim in range(1, self._index_dims):
                keys_np[:, kdim] = 0.0
        elif values_obj.ndim == 2:
            if values_obj.shape[0] != n:
                raise ValueError("values rows must match window_idx size")
            keys_np = np.empty((n, self._index_dims), dtype=np.float64)
            for kdim in range(self._index_dims):
                if kdim < values_obj.shape[1]:
                    keys_np[:, kdim] = values_obj[:, kdim]
                else:
                    keys_np[:, kdim] = 0.0
        else:
            raise ValueError("values must be a 1D or 2D array")
        if vectors_in is None:
            raise ValueError("BlockedLazyIndex requires full sketch vectors")
        vectors_np = np.ascontiguousarray(np.asarray(vectors_in, dtype=np.float64))
        if vectors_np.ndim != 2 or vectors_np.shape[0] != n:
            raise ValueError("vectors must be a 2D array with one row per value")
        if vectors_np.shape[1] != self._n_vectors:
            raise ValueError("vector size does not match BlockedLazyIndex dimension")
        if sid_idx_in is None:
            sid_idx_np = np.full(n, -1, dtype=np.int64)
        else:
            sid_idx_np = np.asarray(sid_idx_in, dtype=np.int64).ravel()
        if time_in is None:
            time_np = np.zeros(n, dtype=np.int64)
        else:
            time_np = np.asarray(time_in, dtype=np.int64).ravel()
        if window_size_in is None:
            win_size_np = np.zeros(n, dtype=np.int64)
        else:
            win_size_np = np.asarray(window_size_in, dtype=np.int64).ravel()
        if sid_rank_in is None:
            sid_rank_np = np.full(n, -1, dtype=np.int64)
        else:
            sid_rank_np = np.asarray(sid_rank_in, dtype=np.int64).ravel()
        if sid_idx_np.shape[0] != n or time_np.shape[0] != n or win_size_np.shape[0] != n or sid_rank_np.shape[0] != n:
            raise ValueError("metadata array sizes must match values size")

        entry_ids = np.empty(n, dtype=np.int64)
        self._last_entry_ids_arr = entry_ids
        self._last_vectors_arr = np.ascontiguousarray(vectors_np, dtype=np.float64)
        self._last_keys_arr = np.ascontiguousarray(keys_np, dtype=np.float64)
        self._last_window_idx_arr = np.ascontiguousarray(window_idx_np, dtype=np.int64)
        self._last_sid_idx_arr = np.ascontiguousarray(sid_idx_np, dtype=np.int64)
        self._last_sid_rank_arr = np.ascontiguousarray(sid_rank_np, dtype=np.int64)
        self._last_time_arr = np.ascontiguousarray(time_np, dtype=np.int64)
        self._last_window_size_arr = np.ascontiguousarray(win_size_np, dtype=np.int64)

        self._ensure_current_capacity(self._current_count + n)
        # Hoisted (2026-07-04): these attributes were previously indexed
        # directly (self._current_xxx_arr[pos] = ..., self._win_xxx_arr[win_id]
        # = ...) on every row/dimension of this loop -- since they are
        # declared as plain `cdef object` (not typed memoryviews), each such
        # write went through full Python/NumPy __setitem__, not a fast typed
        # indexed store. Fixed the same way BalancedIndex.insert_many already
        # did it correctly: pre-scan for the max window_idx needed, call
        # _ensure_window_capacity once with that bound (instead of once per
        # row), then hoist every memoryview before the loop. This is safe --
        # _close_current_block() (which can run mid-loop when a block fills
        # up) only copies data out and resets a counter, it never reallocates
        # self._current_*_arr, so those memoryviews stay valid across it.
        cdef int64_t max_window_idx = -1
        for i in range(n):
            if window_idx_np[i] > max_window_idx:
                max_window_idx = <int64_t>window_idx_np[i]
        if max_window_idx >= 0:
            self._ensure_window_capacity(<Py_ssize_t>max_window_idx + 1)
        cdef int64_t[:] cur_entry_ids_mv = self._current_entry_ids_arr
        cdef int64_t[:] cur_window_idx_mv = self._current_window_idx_arr
        cdef int64_t[:] cur_sid_idx_mv = self._current_sid_idx_arr
        cdef int64_t[:] cur_sid_rank_mv = self._current_sid_rank_arr
        cdef int64_t[:] cur_time_mv = self._current_time_arr
        cdef int64_t[:] cur_window_size_mv = self._current_window_size_arr
        cdef double[:, :] cur_vectors_mv = self._current_vectors_arr
        cdef double[:, :] cur_keys_mv = self._current_keys_arr
        cdef int64_t[:] win_sid_idx_mv = self._win_sid_idx_arr
        cdef int64_t[:] win_sid_rank_mv = self._win_sid_rank_arr
        cdef int64_t[:] win_time_mv = self._win_time_arr
        cdef int64_t[:] win_size_mv = self._win_size_arr
        for i in range(n):
            pos = self._current_count
            entry_ids[i] = self._next_entry_id
            self._next_entry_id += 1
            cur_entry_ids_mv[pos] = entry_ids[i]
            cur_window_idx_mv[pos] = window_idx_np[i]
            cur_sid_idx_mv[pos] = sid_idx_np[i]
            cur_sid_rank_mv[pos] = sid_rank_np[i]
            cur_time_mv[pos] = time_np[i]
            cur_window_size_mv[pos] = win_size_np[i]
            for d in range(self._n_vectors):
                cur_vectors_mv[pos, d] = vectors_np[i, d]
            for d in range(self._index_dims):
                cur_keys_mv[pos, d] = keys_np[i, d]
            win_id = window_idx_np[i]
            if win_id >= 0:
                win_sid_idx_mv[win_id] = sid_idx_np[i]
                win_sid_rank_mv[win_id] = sid_rank_np[i]
                win_time_mv[win_id] = time_np[i]
                win_size_mv[win_id] = win_size_np[i]
            self._current_count += 1
            if self._current_count >= self._block_size:
                self._close_current_block()
        self._refresh_current_dim_order()
        return entry_ids

    cdef inline bint _entry_valid_for_query(self,
                                            int64_t q_win,
                                            int64_t q_sid,
                                            int64_t q_time,
                                            int64_t q_w,
                                            int64_t o_win,
                                            int64_t o_sid,
                                            int64_t o_time,
                                            int64_t o_w) nogil:
        if o_time < self._min_valid_time:
            return False
        if o_win == q_win:
            return False
        if o_w != q_w:
            return False
        if o_sid == q_sid and o_time == q_time:
            return False
        return True

    cdef inline bint _coord_filter_row(self,
                                       double[:] q_vec,
                                       double[:, :] vectors,
                                       Py_ssize_t row,
                                       int64_t[:] dims,
                                       Py_ssize_t skip_dim_i,
                                       double tau,
                                       bint neg_mode):
        cdef Py_ssize_t dim_i
        cdef int64_t dim
        cdef double diff
        for dim_i in range(dims.shape[0]):
            if dim_i == skip_dim_i:
                continue
            dim = dims[dim_i]
            if dim < 0 or dim >= self._n_vectors:
                continue
            diff = q_vec[dim] + vectors[row, dim] if neg_mode else q_vec[dim] - vectors[row, dim]
            if diff > tau or diff < -tau:
                return False
        return True

    cdef inline bint _partial_l2_bound_row(self,
                                          double[:] q_vec,
                                          double[:, :] vectors,
                                          Py_ssize_t row,
                                          int64_t[:] dim_order,
                                          double tau_sq,
                                          bint neg_mode):
        cdef Py_ssize_t order_i
        cdef int64_t dim
        cdef double diff
        cdef double acc = 0.0
        for order_i in range(dim_order.shape[0]):
            dim = dim_order[order_i]
            if dim < 0 or dim >= self._n_vectors:
                continue
            diff = q_vec[dim] + vectors[row, dim] if neg_mode else q_vec[dim] - vectors[row, dim]
            acc += diff * diff
            if acc > tau_sq:
                return False
        return True

    cdef inline bint _partial_l2_bound_row_signed(self,
                                                 double[:] q_vec,
                                                 double[:, :] vectors,
                                                 Py_ssize_t row,
                                                 int64_t[:] dim_order,
                                                 double tau_sq):
        cdef Py_ssize_t order_i
        cdef int64_t dim
        cdef double diff_pos
        cdef double diff_neg
        cdef double acc_pos = 0.0
        cdef double acc_neg = 0.0
        cdef bint pos_alive = True
        cdef bint neg_alive = True
        for order_i in range(dim_order.shape[0]):
            dim = dim_order[order_i]
            if dim < 0 or dim >= self._n_vectors:
                continue
            if pos_alive:
                diff_pos = q_vec[dim] - vectors[row, dim]
                acc_pos += diff_pos * diff_pos
                if acc_pos > tau_sq:
                    pos_alive = False
            if neg_alive:
                diff_neg = q_vec[dim] + vectors[row, dim]
                acc_neg += diff_neg * diff_neg
                if acc_neg > tau_sq:
                    neg_alive = False
            if not pos_alive and not neg_alive:
                return False
        return True

    cdef inline bint _key_filter_row_l2(self,
                                        double[:] q_keys,
                                        double[:, :] keys,
                                        Py_ssize_t row,
                                        Py_ssize_t skip_key_i,
                                        double tau,
                                        bint neg_mode):
        cdef Py_ssize_t key_i
        cdef double diff
        for key_i in range(q_keys.shape[0]):
            if key_i == skip_key_i:
                continue
            diff = q_keys[key_i] + keys[row, key_i] if neg_mode else q_keys[key_i] - keys[row, key_i]
            if diff > tau or diff < -tau:
                return False
        return True

    cdef inline bint _key_filter_row_cosine(self,
                                           double[:] q_keys,
                                           double[:, :] keys,
                                           Py_ssize_t row,
                                           Py_ssize_t skip_key_i,
                                           double gamma,
                                           bint neg_mode):
        cdef Py_ssize_t key_i
        cdef double q_key
        cdef double lower
        cdef double upper
        cdef double value
        for key_i in range(q_keys.shape[0]):
            if key_i == skip_key_i:
                continue
            q_key = -q_keys[key_i] if neg_mode else q_keys[key_i]
            _projection_interval(q_key, gamma, &lower, &upper)
            value = keys[row, key_i]
            if value < lower or value > upper:
                return False
        return True

    cdef void _scan_block(self,
                          double[:] q_vec,
                          double[:] q_keys,
                          int64_t q_win,
                          int64_t q_sid,
                          int64_t q_time,
                          int64_t q_w,
                          double[:, :] vectors,
                          double[:, :] keys,
                          int64_t[:] win_idx,
                          int64_t[:] sid_idx,
                          int64_t[:] time_idx,
                          int64_t[:] win_size,
                          int64_t[:] dims,
                          int64_t[:] dim_order,
                          double[:, :] sorted_keys_by_dim,
                          int64_t[:, :] orders_by_dim,
                          double tau,
                          double tau_sq,
                          bint neg_mode,
                          int64_t **buf,
                          Py_ssize_t *count,
                          Py_ssize_t *cap,
                          bint *failed,
                          Py_ssize_t *range_hits,
                          Py_ssize_t *valid_index_hits,
                          Py_ssize_t *after_coord_filter,
                          Py_ssize_t *distance_checks,
                          Py_ssize_t *after_similarity):
        cdef double[:] sorted_keys
        cdef Py_ssize_t n_rows = vectors.shape[0]
        cdef Py_ssize_t dim, dim_i, best_dim_i, pos, left, right, row, key_dims
        cdef Py_ssize_t best_left = 0
        cdef Py_ssize_t best_right = 0
        cdef Py_ssize_t width, best_width
        cdef int64_t primary = 0
        cdef double diff, acc, q_key, lower, upper

        if n_rows == 0:
            return
        key_dims = sorted_keys_by_dim.shape[0]
        best_width = n_rows + 1
        best_dim_i = 0
        for dim_i in range(key_dims):
            sorted_keys = sorted_keys_by_dim[dim_i]
            q_key = -q_keys[dim_i] if neg_mode else q_keys[dim_i]
            lower = q_key - tau
            upper = q_key + tau
            left = _bisect_left(sorted_keys, lower)
            right = _bisect_right(sorted_keys, upper)
            width = right - left
            if width < best_width:
                best_width = width
                best_left = left
                best_right = right
                best_dim_i = dim_i
                if width == 0:
                    break
        if best_width > n_rows:
            return
        range_hits[0] += best_width
        for pos in range(best_left, best_right):
            row = orders_by_dim[best_dim_i, pos]
            if not self._entry_valid_for_query(q_win, q_sid, q_time, q_w, win_idx[row], sid_idx[row], time_idx[row], win_size[row]):
                continue
            valid_index_hits[0] += 1
            after_coord_filter[0] += 1
            if not self._key_filter_row_l2(q_keys, keys, row, best_dim_i, tau, neg_mode):
                continue
            distance_checks[0] += 1
            acc = 0.0
            for dim_i in range(dim_order.shape[0]):
                dim = dim_order[dim_i]
                if dim < 0 or dim >= self._n_vectors:
                    continue
                diff = q_vec[dim] + vectors[row, dim] if neg_mode else q_vec[dim] - vectors[row, dim]
                acc += diff * diff
                if acc > tau_sq:
                    break
            if acc <= tau_sq:
                after_similarity[0] += 1
                if not _append_pair(buf, count, cap, q_win, win_idx[row]):
                    failed[0] = True
                    return

    cdef void _scan_current(self,
                            double[:] q_vec,
                            double[:] q_keys,
                            int64_t q_win,
                            int64_t q_sid,
                            int64_t q_time,
                            int64_t q_w,
                            double tau,
                            double tau_sq,
                            bint neg_mode,
                            int64_t **buf,
                            Py_ssize_t *count,
                            Py_ssize_t *cap,
                            bint *failed,
                            Py_ssize_t *range_hits,
                            Py_ssize_t *valid_index_hits,
                            Py_ssize_t *after_coord_filter,
                            Py_ssize_t *distance_checks,
                            Py_ssize_t *after_similarity):
        cdef double[:, :] vectors = self._current_vectors_arr
        cdef double[:, :] keys = self._current_keys_arr
        cdef int64_t[:] win_idx = self._current_window_idx_arr
        cdef int64_t[:] sid_idx = self._current_sid_idx_arr
        cdef int64_t[:] time_idx = self._current_time_arr
        cdef int64_t[:] win_size = self._current_window_size_arr
        cdef int64_t[:] dims = self._current_dims_arr
        cdef int64_t[:] dim_order = self._current_dim_order_arr
        cdef Py_ssize_t row, dim_i
        cdef int64_t primary = 0
        cdef int64_t dim
        cdef double diff, acc
        if dims.shape[0] > 0:
            primary = dims[0]
        if primary < 0 or primary >= self._n_vectors:
            primary = 0
        for row in range(self._current_count):
            diff = q_keys[0] + keys[row, 0] if neg_mode else q_keys[0] - keys[row, 0]
            if diff > tau or diff < -tau:
                continue
            range_hits[0] += 1
            if not self._entry_valid_for_query(q_win, q_sid, q_time, q_w, win_idx[row], sid_idx[row], time_idx[row], win_size[row]):
                continue
            valid_index_hits[0] += 1
            after_coord_filter[0] += 1
            if not self._key_filter_row_l2(q_keys, keys, row, 0, tau, neg_mode):
                continue
            distance_checks[0] += 1
            acc = 0.0
            for dim_i in range(dim_order.shape[0]):
                dim = dim_order[dim_i]
                if dim < 0 or dim >= self._n_vectors:
                    continue
                diff = q_vec[dim] + vectors[row, dim] if neg_mode else q_vec[dim] - vectors[row, dim]
                acc += diff * diff
                if acc > tau_sq:
                    break
            if acc <= tau_sq:
                after_similarity[0] += 1
                if not _append_pair(buf, count, cap, q_win, win_idx[row]):
                    failed[0] = True
                    return

    cdef void _scan_block_cosine(self,
                                 double[:] q_vec,
                                 double[:] q_keys,
                                 int64_t q_win,
                                 int64_t q_sid,
                                 int64_t q_rank,
                                 int64_t q_time,
                                 int64_t q_w,
                                 double[:, :] vectors,
                                 double[:, :] keys,
                                 int64_t[:] win_idx,
                                 int64_t[:] sid_idx,
                                 int64_t[:] sid_rank,
                                 int64_t[:] time_idx,
                                 int64_t[:] win_size,
                                 double[:, :] sorted_keys_by_dim,
                                 int64_t[:, :] orders_by_dim,
                                 double[:] residual_norms,
                                 Py_ssize_t row_offset,
                                 Py_ssize_t n_rows,
                                 int64_t[:] bound_dims,
                                 double[:] bound_min,
                                 double[:] bound_max,
                                 double max_residual_norm,
                                 double[:] ref_direction,
                                 double min_cos_to_ref,
                                 double gamma,
                                 double tau,
                                 bint signed_abs,
                                 int64_t **buf,
                                 Py_ssize_t *count,
                                 Py_ssize_t *cap,
                                 bint *failed,
                                 Py_ssize_t *range_hits,
                                 Py_ssize_t *valid_index_hits,
                                 Py_ssize_t *unique_index_hits,
                                 Py_ssize_t *duplicate_index_hits,
                                 Py_ssize_t *after_coord_filter,
                                 Py_ssize_t *partial_bound_checks,
                                 Py_ssize_t *after_partial_bound,
                                 Py_ssize_t *unique_pre_dot_pairs,
                                 Py_ssize_t *duplicate_pre_dot_pairs,
                                 Py_ssize_t *dot_checks,
                                 Py_ssize_t *after_similarity,
                                 uint8_t **pair_seen_occupied,
                                 int64_t **pair_seen_a,
                                 int64_t **pair_seen_b,
                                 Py_ssize_t *pair_seen_capacity,
                                 Py_ssize_t *pair_seen_count,
                                 Py_ssize_t *blocks_visited,
                                 Py_ssize_t *blocks_pruned_by_ub,
                                 Py_ssize_t *rows_in_surviving_blocks,
                                 Py_ssize_t *dot_checks_saved_by_row_ub):
        # (2026-07-06) Part 1 follow-up: row-level fields take the FULL flat
        # table + row_offset/n_rows -- see the identical comment in
        # _scan_block_bucketed_cosine. This function's own call site is
        # currently unreached in production (only self._bucketed_mode=False
        # would use it, and every live backend leaves that True), but is
        # kept correct/consistent rather than left on a stale signature.
        cdef double[:] sorted_keys
        cdef Py_ssize_t row, dim, dim_i, best_dim_i, pos, left, right, pass_i, key_dims
        cdef Py_ssize_t best_left = 0
        cdef Py_ssize_t best_right = 0
        cdef Py_ssize_t width, best_width
        cdef int64_t primary = 0
        cdef int64_t pair_a, pair_b
        cdef double dot, q_key, lower, upper
        cdef bint neg_mode
        cdef uint8_t[:] seen
        cdef object seen_obj = None
        cdef uint8_t mode_bit
        # (2026-07-06) Part 1: Cauchy-Schwarz block/row upper-bound pruning,
        # opt-in (self._enable_block_ub_pruning/_enable_row_ub_pruning both
        # default False; bound_dims.shape[0]==0 whenever candidate_bound_dims
        # isn't set, in which case both checks below are unconditionally
        # skipped). See the module-level helper functions and
        # docs/implementation_log.md.
        cdef bint have_bound_dims = bound_dims.shape[0] > 0
        cdef double q_r_norm = 0.0

        if n_rows == 0:
            return
        if have_bound_dims and self._enable_row_ub_pruning:
            q_r_norm = _query_residual_norm(q_vec, bound_dims)
        if self._enable_block_ub_pruning:
            if _cone_block_prune_check(q_vec, ref_direction, min_cos_to_ref, gamma, signed_abs):
                blocks_pruned_by_ub[0] += 1
                return
        blocks_visited[0] += 1
        rows_in_surviving_blocks[0] += n_rows
        key_dims = sorted_keys_by_dim.shape[0]
        if signed_abs:
            seen_obj = np.zeros(n_rows, dtype=np.uint8)
            seen = seen_obj
        for pass_i in range(2):
            if pass_i == 1 and not signed_abs:
                break
            neg_mode = pass_i == 1
            if neg_mode:
                mode_bit = <uint8_t>2
            else:
                mode_bit = <uint8_t>1
            best_width = n_rows + 1
            best_dim_i = 0
            best_left = row_offset
            best_right = row_offset
            for dim_i in range(key_dims):
                sorted_keys = sorted_keys_by_dim[dim_i]
                q_key = -q_keys[dim_i] if neg_mode else q_keys[dim_i]
                _projection_interval(q_key, gamma, &lower, &upper)
                left = _bisect_left_range(sorted_keys, row_offset, row_offset + n_rows, lower)
                right = _bisect_right_range(sorted_keys, row_offset, row_offset + n_rows, upper)
                width = right - left
                if width < best_width:
                    best_width = width
                    best_left = left
                    best_right = right
                    best_dim_i = dim_i
                    if width == 0:
                        break
            if best_width > n_rows:
                continue
            range_hits[0] += best_width
            for pos in range(best_left, best_right):
                row = orders_by_dim[best_dim_i, pos]
                if not self._entry_valid_for_query(q_win, q_sid, q_time, q_w, win_idx[row_offset + row], sid_idx[row_offset + row], time_idx[row_offset + row], win_size[row_offset + row]):
                    continue
                valid_index_hits[0] += 1
                if signed_abs:
                    if (seen[row] & <uint8_t>4) != 0:
                        duplicate_index_hits[0] += 1
                        continue
                    if (seen[row] & mode_bit) != 0:
                        duplicate_index_hits[0] += 1
                        continue
                    if seen[row] == 0:
                        unique_index_hits[0] += 1
                    else:
                        duplicate_index_hits[0] += 1
                    seen[row] = seen[row] | mode_bit
                else:
                    unique_index_hits[0] += 1
                after_coord_filter[0] += 1
                partial_bound_checks[0] += 1
                if not self._key_filter_row_cosine(q_keys, keys, row_offset + row, best_dim_i, gamma, neg_mode):
                    continue
                # after_partial_bound (2026-07-04): moved here, i.e. right after
                # the real secondary/key filter passes and before pair dedup,
                # instead of being counted post-dedup (where it was previously
                # always identical to unique_pre_dot_pairs). See
                # docs/implementation_log.md, "candidate-search proxy metrics
                # redesign", for why this was misleading.
                after_partial_bound[0] += 1
                _canonical_window_pair(
                    q_win,
                    q_sid,
                    q_rank,
                    q_time,
                    win_idx[row_offset + row],
                    sid_idx[row_offset + row],
                    sid_rank[row_offset + row],
                    time_idx[row_offset + row],
                    &pair_a,
                    &pair_b,
                )
                if not _pair_seen_insert(pair_seen_occupied, pair_seen_a, pair_seen_b, pair_seen_capacity, pair_seen_count, pair_a, pair_b, failed):
                    if failed[0]:
                        return
                    duplicate_pre_dot_pairs[0] += 1
                    continue
                unique_pre_dot_pairs[0] += 1
                if have_bound_dims and self._enable_row_ub_pruning:
                    if _row_ub_prune_check(q_vec, vectors, row_offset + row, bound_dims, residual_norms[row_offset + row], q_r_norm, gamma, signed_abs):
                        dot_checks_saved_by_row_ub[0] += 1
                        if signed_abs:
                            seen[row] = seen[row] | <uint8_t>4
                        continue
                dot_checks[0] += 1
                dot = 0.0
                for dim in range(self._n_vectors):
                    dot += q_vec[dim] * vectors[row_offset + row, dim]
                if signed_abs:
                    seen[row] = seen[row] | <uint8_t>4
                    if fabs(dot) < gamma:
                        continue
                elif dot < gamma:
                    continue
                after_similarity[0] += 1
                if not _append_pair(buf, count, cap, q_win, win_idx[row_offset + row]):
                    failed[0] = True
                    return

    cdef void _scan_block_bucketed_cosine(self,
                                          double[:] q_vec,
                                          double[:] q_keys,
                                          int64_t q_win,
                                          int64_t q_sid,
                                          int64_t q_rank,
                                          int64_t q_time,
                                          int64_t q_w,
                                          double[:, :] vectors,
                                          double[:, :] keys,
                                          int64_t[:] win_idx,
                                          int64_t[:] sid_idx,
                                          int64_t[:] sid_rank,
                                          int64_t[:] time_idx,
                                          int64_t[:] win_size,
                                          double[:, :] sorted_keys_by_dim,
                                          int64_t[:, :] orders_by_dim,
                                          double[:] residual_norms,
                                          Py_ssize_t row_offset,
                                          Py_ssize_t n_rows,
                                          int64_t[:] bound_dims,
                                          double[:] bound_min,
                                          double[:] bound_max,
                                          double max_residual_norm,
                                          double[:] ref_direction,
                                          double min_cos_to_ref,
                                          double gamma,
                                          bint signed_abs,
                                          int64_t **buf,
                                          Py_ssize_t *count,
                                          Py_ssize_t *cap,
                                          bint *failed,
                                          Py_ssize_t *range_hits,
                                          Py_ssize_t *valid_index_hits,
                                          Py_ssize_t *unique_index_hits,
                                          Py_ssize_t *duplicate_index_hits,
                                          Py_ssize_t *after_coord_filter,
                                          Py_ssize_t *partial_bound_checks,
                                          Py_ssize_t *after_partial_bound,
                                          Py_ssize_t *unique_pre_dot_pairs,
                                          Py_ssize_t *duplicate_pre_dot_pairs,
                                          Py_ssize_t *dot_checks,
                                          Py_ssize_t *after_similarity,
                                          uint8_t **pair_seen_occupied,
                                          int64_t **pair_seen_a,
                                          int64_t **pair_seen_b,
                                          Py_ssize_t *pair_seen_capacity,
                                          Py_ssize_t *pair_seen_count,
                                          Py_ssize_t *blocks_visited,
                                          Py_ssize_t *blocks_pruned_by_ub,
                                          Py_ssize_t *rows_in_surviving_blocks,
                                          Py_ssize_t *dot_checks_saved_by_row_ub):
        # (2026-07-06) Part 1 follow-up: row-level fields (proportional to
        # block size) take the FULL flat table + row_offset/n_rows instead
        # of a pre-sliced per-block sub-array -- even a hoisted-memoryview
        # slice is a fresh object per (query, block) visit, and that cost
        # scales with how many rows the slice covers. Per-BLOCK metadata
        # (bound_dims, ref_direction, etc.) stays a plain small array/scalar
        # passed by the caller -- O(1) size regardless of block length, so
        # slicing it once per call was never the expensive part. See
        # docs/implementation_log.md, "block-level cone pruning: flat
        # typed-array block storage".
        cdef double[:] sorted_keys
        cdef Py_ssize_t key_dims = sorted_keys_by_dim.shape[0]
        cdef Py_ssize_t dim_i, pos, left, right, width, row, pass_i
        cdef Py_ssize_t best_dim_i, best_left, best_right, best_width
        cdef double q_key, lower, upper, dot
        cdef uint8_t *processed = NULL
        cdef int64_t pair_a, pair_b
        cdef Py_ssize_t dim
        cdef bint neg_mode
        cdef bint have_bound_dims = bound_dims.shape[0] > 0
        cdef double q_r_norm = 0.0

        if n_rows == 0 or key_dims == 0:
            return
        if have_bound_dims and self._enable_row_ub_pruning:
            q_r_norm = _query_residual_norm(q_vec, bound_dims)
        if self._enable_block_ub_pruning:
            if _cone_block_prune_check(q_vec, ref_direction, min_cos_to_ref, gamma, signed_abs):
                blocks_pruned_by_ub[0] += 1
                return
        blocks_visited[0] += 1
        rows_in_surviving_blocks[0] += n_rows
        processed = <uint8_t *>malloc(n_rows * sizeof(uint8_t))
        if processed == NULL:
            failed[0] = True
            return
        for row in range(n_rows):
            processed[row] = <uint8_t>0
        # (2026-07-05) Narrowest-dimension + per-row cascade, replacing the
        # previous approach of fully scanning postings across every one of
        # the key_dims dimensions: pick the single tightest dimension via
        # bisect (cheap, O(log n) per dimension), fully enumerate only that
        # dimension's postings, and verify the remaining dimensions per
        # candidate via _key_filter_row_cosine -- an O(key_dims) per-row
        # check against the exact same bound, the same mechanism
        # _scan_block_cosine (blocked_lazy's own closed-block scan) already
        # uses. Mathematically equivalent to full multi-dimension
        # intersection (identical final accept/reject decision), but touches
        # far fewer array entries when key_dims is large. `processed` plays
        # the same fused-single-pass role as marks_pos/marks_neg did before:
        # a row that qualifies under both +q and -q only gets one
        # entry-validity/pair-dedup/dot-check attempt. See
        # docs/implementation_log.md, "bucketed_bst/current-block narrowest-dim rewrite".
        for pass_i in range(2):
            if pass_i == 1 and not signed_abs:
                break
            neg_mode = pass_i == 1
            best_width = n_rows + 1
            best_dim_i = 0
            best_left = row_offset
            best_right = row_offset
            for dim_i in range(key_dims):
                sorted_keys = sorted_keys_by_dim[dim_i]
                q_key = -q_keys[dim_i] if neg_mode else q_keys[dim_i]
                _projection_interval(q_key, gamma, &lower, &upper)
                left = _bisect_left_range(sorted_keys, row_offset, row_offset + n_rows, lower)
                right = _bisect_right_range(sorted_keys, row_offset, row_offset + n_rows, upper)
                width = right - left
                if width < best_width:
                    best_width = width
                    best_left = left
                    best_right = right
                    best_dim_i = dim_i
                    if width == 0:
                        break
            if best_width > n_rows:
                continue
            range_hits[0] += best_width
            for pos in range(best_left, best_right):
                row = orders_by_dim[best_dim_i, pos]
                if processed[row]:
                    duplicate_index_hits[0] += 1
                    continue
                if not self._entry_valid_for_query(q_win, q_sid, q_time, q_w, win_idx[row_offset + row], sid_idx[row_offset + row], time_idx[row_offset + row], win_size[row_offset + row]):
                    continue
                valid_index_hits[0] += 1
                unique_index_hits[0] += 1
                after_coord_filter[0] += 1
                partial_bound_checks[0] += 1
                if not self._key_filter_row_cosine(q_keys, keys, row_offset + row, best_dim_i, gamma, neg_mode):
                    continue
                after_partial_bound[0] += 1
                processed[row] = <uint8_t>1
                _canonical_window_pair(
                    q_win,
                    q_sid,
                    q_rank,
                    q_time,
                    win_idx[row_offset + row],
                    sid_idx[row_offset + row],
                    sid_rank[row_offset + row],
                    time_idx[row_offset + row],
                    &pair_a,
                    &pair_b,
                )
                if not _pair_seen_insert(pair_seen_occupied, pair_seen_a, pair_seen_b, pair_seen_capacity, pair_seen_count, pair_a, pair_b, failed):
                    if failed[0]:
                        free(processed)
                        return
                    duplicate_pre_dot_pairs[0] += 1
                    continue
                unique_pre_dot_pairs[0] += 1
                if have_bound_dims and self._enable_row_ub_pruning:
                    if _row_ub_prune_check(q_vec, vectors, row_offset + row, bound_dims, residual_norms[row_offset + row], q_r_norm, gamma, signed_abs):
                        dot_checks_saved_by_row_ub[0] += 1
                        continue
                dot_checks[0] += 1
                dot = 0.0
                for dim in range(self._n_vectors):
                    dot += q_vec[dim] * vectors[row_offset + row, dim]
                if signed_abs:
                    if fabs(dot) < gamma:
                        continue
                elif dot < gamma:
                    continue
                after_similarity[0] += 1
                if not _append_pair(buf, count, cap, q_win, win_idx[row_offset + row]):
                    failed[0] = True
                    free(processed)
                    return
        free(processed)

    cpdef object find_pair_rows_full_cosine_signed(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_full_cosine_meta(recent_entry_ids, gamma, tau, True)

    cpdef object find_pair_rows_full_cosine(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_full_cosine_meta(recent_entry_ids, gamma, tau, False)

    cdef object _find_pair_rows_full_cosine_meta(self, long[:] recent_entry_ids, double gamma, double tau, bint signed_abs):
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef double[:] q_vec
        cdef double[:] q_keys
        cdef Py_ssize_t i, b_i, count = 0, cap = 1024, out_i
        cdef Py_ssize_t q_row
        cdef Py_ssize_t current_count
        cdef Py_ssize_t offset, length
        cdef int64_t q_entry
        cdef int64_t q_win
        cdef int64_t q_sid
        cdef int64_t q_time
        cdef int64_t q_w
        cdef bint found
        cdef object current_meta
        cdef bint have_current_meta
        cdef int64_t[:, :] cur_orders_by_dim
        cdef double[:, :] cur_sorted_keys_by_dim
        cdef int64_t[:] cur_bound_dims_meta
        cdef double[:] cur_bound_min_meta
        cdef double[:] cur_bound_max_meta
        cdef double[:] cur_residual_norms_meta
        cdef double cur_max_residual_norm_meta
        cdef double[:] cur_ref_direction_meta
        cdef double cur_min_cos_to_ref_meta
        cdef double[:, :] q_vectors
        cdef double[:, :] q_key_values
        cdef int64_t[:] q_entry_ids
        cdef int64_t[:] q_win_idx
        cdef int64_t[:] q_sid_idx
        cdef int64_t[:] q_sid_rank_idx
        cdef int64_t[:] q_time_idx
        cdef int64_t[:] q_win_size
        # (2026-07-06) Part 1 follow-up: hoisted once per query batch, same
        # rationale as cur_vectors etc below and the memoryview-hoisting fix
        # in monitor_kernels.pyx's update() row loop -- re-deriving a
        # memoryview from a `cdef object` (self._bt_vectors etc) attribute
        # inside the per-(query,block) loop was exactly that anti-pattern,
        # just via array slicing instead of attribute access; it made this
        # rewrite *slower* than the dict-based code until hoisted. See
        # docs/implementation_log.md, "block-level cone pruning: flat
        # typed-array block storage".
        cdef double[:, :] bt_vectors
        cdef double[:, :] bt_keys
        cdef int64_t[:] bt_entry_ids
        cdef int64_t[:] bt_window_idx
        cdef int64_t[:] bt_sid_idx
        cdef int64_t[:] bt_sid_rank
        cdef int64_t[:] bt_time
        cdef int64_t[:] bt_window_size
        cdef double[:] bt_residual_norms
        cdef int64_t[:, :] bt_orders_by_dim
        cdef double[:, :] bt_sorted_keys_by_dim
        cdef int64_t[:] bt_block_offset
        cdef int64_t[:] bt_block_length
        cdef double[:] bt_block_max_residual_norm
        cdef double[:] bt_block_min_cos_to_ref
        cdef double[:, :] bt_block_ref_direction
        cdef int64_t[:, :] bt_block_bound_dims
        cdef double[:, :] bt_block_bound_min
        cdef double[:, :] bt_block_bound_max
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef bint failed = False
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t valid_index_hits = 0
        cdef Py_ssize_t unique_index_hits = 0
        cdef Py_ssize_t duplicate_index_hits = 0
        cdef Py_ssize_t after_coord_filter = 0
        cdef Py_ssize_t partial_bound_checks = 0
        cdef Py_ssize_t after_partial_bound = 0
        cdef Py_ssize_t unique_pre_dot_pairs = 0
        cdef Py_ssize_t duplicate_pre_dot_pairs = 0
        cdef Py_ssize_t dot_checks = 0
        cdef Py_ssize_t after_similarity = 0
        cdef Py_ssize_t pairs_before_dedupe = 0
        cdef Py_ssize_t pairs_after_dedupe = 0
        cdef int64_t a, b, sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w
        cdef int64_t q_rank
        cdef uint8_t *pair_seen_occupied = NULL
        cdef int64_t *pair_seen_a = NULL
        cdef int64_t *pair_seen_b = NULL
        cdef Py_ssize_t pair_seen_capacity = 0
        cdef Py_ssize_t pair_seen_count = 0
        cdef Py_ssize_t blocks_visited = 0
        cdef Py_ssize_t blocks_pruned_by_ub = 0
        cdef Py_ssize_t rows_in_surviving_blocks = 0
        cdef Py_ssize_t dot_checks_saved_by_row_ub = 0

        if buf == NULL:
            raise MemoryError()
        if not _pair_seen_init(&pair_seen_occupied, &pair_seen_a, &pair_seen_b, &pair_seen_capacity, max(1024, recent_entry_ids.shape[0] * 64)):
            free(buf)
            raise MemoryError()
        if recent_entry_ids.shape[0] == 0:
            free(buf)
            _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
            self.last_stats = {
                "num_index_candidates": 0,
                "num_valid_index_candidates": 0,
                "num_unique_index_candidates": 0,
                "num_duplicate_index_candidates": 0,
                "num_unique_pre_dot_pairs": 0,
                "num_duplicate_pre_dot_pairs": 0,
                "num_after_coord_filter": 0,
                "num_partial_bound_checks": 0,
                "num_after_partial_bound": 0,
                "num_dot_checks": 0,
                "num_distance_checks": 0,
                "num_after_similarity": 0,
                "num_after_dot": 0,
                "num_pairs_before_dedupe": 0,
                "num_pairs_after_dedupe": 0,
                "num_rows": 0,
                "num_recent_queries": 0,
                "num_entries": int(self._current_count),
                "num_blocks": int(self._block_count),
                "gamma": float(gamma),
                "tau": float(tau),
                "num_blocks_visited": 0,
                "num_blocks_pruned_by_ub": 0,
                "num_rows_in_surviving_blocks": 0,
                "num_dot_checks_saved_by_row_ub": 0,
            }
            return np.empty((0, 5), dtype=np.int64)

        current_count = self._current_count
        # Hoisted (2026-07-04): these back the "still open, not yet closed
        # into a block" fallback path below and never change during this
        # scan (no insertion happens mid-query) -- fetching them once here
        # instead of on every "not found in closed blocks" iteration avoids
        # the same avoidable memoryview-from-Python-attribute cost fixed in
        # monitor_kernels.pyx's update() row loop. See docs/implementation_log.md.
        cur_entry_ids = self._current_entry_ids_arr
        cur_vectors = self._current_vectors_arr
        cur_key_values = self._current_keys_arr
        cur_win_idx = self._current_window_idx_arr
        cur_sid_idx = self._current_sid_idx_arr
        cur_sid_rank_idx = self._current_sid_rank_arr
        cur_time_idx = self._current_time_arr
        cur_win_size = self._current_window_size_arr
        bt_vectors = self._bt_vectors
        bt_keys = self._bt_keys
        bt_entry_ids = self._bt_entry_ids
        bt_window_idx = self._bt_window_idx
        bt_sid_idx = self._bt_sid_idx
        bt_sid_rank = self._bt_sid_rank
        bt_time = self._bt_time
        bt_window_size = self._bt_window_size
        bt_residual_norms = self._bt_residual_norms
        bt_orders_by_dim = self._bt_orders_by_dim
        bt_sorted_keys_by_dim = self._bt_sorted_keys_by_dim
        bt_block_offset = self._bt_block_offset
        bt_block_length = self._bt_block_length
        bt_block_max_residual_norm = self._bt_block_max_residual_norm
        bt_block_min_cos_to_ref = self._bt_block_min_cos_to_ref
        bt_block_ref_direction = self._bt_block_ref_direction
        bt_block_bound_dims = self._bt_block_bound_dims
        bt_block_bound_min = self._bt_block_bound_min
        bt_block_bound_max = self._bt_block_bound_max
        # (2026-07-04) Build once per query batch, not once per row -- see
        # _compute_current_view_metadata for why. None if the current block
        # is empty (nothing to scan).
        current_meta = self._compute_current_view_metadata()
        have_current_meta = current_meta is not None
        if have_current_meta:
            (cur_orders_by_dim, cur_sorted_keys_by_dim, cur_bound_dims_meta, cur_bound_min_meta,
             cur_bound_max_meta, cur_residual_norms_meta, cur_max_residual_norm_meta,
             cur_ref_direction_meta, cur_min_cos_to_ref_meta) = current_meta
        for i in range(recent_entry_ids.shape[0]):
            q_entry = <int64_t>recent_entry_ids[i]
            found = False
            for b_i in range(self._block_head, self._block_head + self._block_count):
                offset = bt_block_offset[b_i]
                length = bt_block_length[b_i]
                q_entry_ids = bt_entry_ids[offset:offset + length]
                q_row = _bisect_left_i64_n(q_entry_ids, length, q_entry)
                if q_row < length and q_entry_ids[q_row] == q_entry:
                    q_vectors = bt_vectors[offset:offset + length, :]
                    q_key_values = bt_keys[offset:offset + length, :]
                    q_win_idx = bt_window_idx[offset:offset + length]
                    q_sid_idx = bt_sid_idx[offset:offset + length]
                    q_sid_rank_idx = bt_sid_rank[offset:offset + length]
                    q_time_idx = bt_time[offset:offset + length]
                    q_win_size = bt_window_size[offset:offset + length]
                    q_vec = q_vectors[q_row]
                    q_keys = q_key_values[q_row]
                    q_win = q_win_idx[q_row]
                    q_sid = q_sid_idx[q_row]
                    q_rank = q_sid_rank_idx[q_row]
                    q_time = q_time_idx[q_row]
                    q_w = q_win_size[q_row]
                    found = True
                    break
            if not found:
                q_entry_ids = cur_entry_ids
                q_row = _bisect_left_i64_n(q_entry_ids, current_count, q_entry)
                if q_row < current_count and q_entry_ids[q_row] == q_entry:
                    q_vectors = cur_vectors
                    q_key_values = cur_key_values
                    q_win_idx = cur_win_idx
                    q_sid_idx = cur_sid_idx
                    q_sid_rank_idx = cur_sid_rank_idx
                    q_time_idx = cur_time_idx
                    q_win_size = cur_win_size
                    q_vec = q_vectors[q_row]
                    q_keys = q_key_values[q_row]
                    q_win = q_win_idx[q_row]
                    q_sid = q_sid_idx[q_row]
                    q_rank = q_sid_rank_idx[q_row]
                    q_time = q_time_idx[q_row]
                    q_w = q_win_size[q_row]
                    found = True
            if not found:
                continue
            for b_i in range(self._block_head, self._block_head + self._block_count):
                offset = bt_block_offset[b_i]
                length = bt_block_length[b_i]
                if self._bucketed_mode:
                    self._scan_block_bucketed_cosine(
                        q_vec,
                        q_keys,
                        q_win,
                        q_sid,
                        q_rank,
                        q_time,
                        q_w,
                        bt_vectors,
                        bt_keys,
                        bt_window_idx,
                        bt_sid_idx,
                        bt_sid_rank,
                        bt_time,
                        bt_window_size,
                        bt_sorted_keys_by_dim,
                        bt_orders_by_dim,
                        bt_residual_norms,
                        offset,
                        length,
                        bt_block_bound_dims[b_i, :],
                        bt_block_bound_min[b_i, :],
                        bt_block_bound_max[b_i, :],
                        bt_block_max_residual_norm[b_i],
                        bt_block_ref_direction[b_i, :],
                        bt_block_min_cos_to_ref[b_i],
                        gamma,
                        signed_abs,
                        &buf,
                        &count,
                        &cap,
                        &failed,
                        &range_hits,
                        &valid_index_hits,
                        &unique_index_hits,
                        &duplicate_index_hits,
                        &after_coord_filter,
                        &partial_bound_checks,
                        &after_partial_bound,
                        &unique_pre_dot_pairs,
                        &duplicate_pre_dot_pairs,
                        &dot_checks,
                        &after_similarity,
                        &pair_seen_occupied,
                        &pair_seen_a,
                        &pair_seen_b,
                        &pair_seen_capacity,
                        &pair_seen_count,
                        &blocks_visited,
                        &blocks_pruned_by_ub,
                        &rows_in_surviving_blocks,
                        &dot_checks_saved_by_row_ub,
                    )
                else:
                    self._scan_block_cosine(
                        q_vec,
                        q_keys,
                        q_win,
                        q_sid,
                        q_rank,
                        q_time,
                        q_w,
                        bt_vectors,
                        bt_keys,
                        bt_window_idx,
                        bt_sid_idx,
                        bt_sid_rank,
                        bt_time,
                        bt_window_size,
                        bt_sorted_keys_by_dim,
                        bt_orders_by_dim,
                        bt_residual_norms,
                        offset,
                        length,
                        bt_block_bound_dims[b_i, :],
                        bt_block_bound_min[b_i, :],
                        bt_block_bound_max[b_i, :],
                        bt_block_max_residual_norm[b_i],
                        bt_block_ref_direction[b_i, :],
                        bt_block_min_cos_to_ref[b_i],
                        gamma,
                        tau,
                        signed_abs,
                        &buf,
                        &count,
                        &cap,
                        &failed,
                        &range_hits,
                        &valid_index_hits,
                        &unique_index_hits,
                        &duplicate_index_hits,
                        &after_coord_filter,
                        &partial_bound_checks,
                        &after_partial_bound,
                        &unique_pre_dot_pairs,
                        &duplicate_pre_dot_pairs,
                        &dot_checks,
                        &after_similarity,
                        &pair_seen_occupied,
                        &pair_seen_a,
                        &pair_seen_b,
                        &pair_seen_capacity,
                        &pair_seen_count,
                        &blocks_visited,
                        &blocks_pruned_by_ub,
                        &rows_in_surviving_blocks,
                        &dot_checks_saved_by_row_ub,
                    )
                if failed:
                    break
            if failed:
                break
            # (2026-07-04) Route the current (open) block through the
            # same polymorphic per-block scan closed blocks already use,
            # instead of the fixed, backend-agnostic per-current-block scan.
            # See _compute_current_view_metadata.
            if have_current_meta:
                self._scan_block_bucketed_cosine(
                    q_vec, q_keys, q_win, q_sid, q_rank, q_time, q_w,
                    cur_vectors, cur_key_values,
                    cur_win_idx, cur_sid_idx,
                    cur_sid_rank_idx, cur_time_idx,
                    cur_win_size,
                    cur_sorted_keys_by_dim, cur_orders_by_dim,
                    cur_residual_norms_meta,
                    0, current_count,
                    cur_bound_dims_meta, cur_bound_min_meta, cur_bound_max_meta,
                    cur_max_residual_norm_meta,
                    cur_ref_direction_meta, cur_min_cos_to_ref_meta,
                    gamma, signed_abs, &buf, &count, &cap, &failed, &range_hits, &valid_index_hits,
                    &unique_index_hits, &duplicate_index_hits, &after_coord_filter, &partial_bound_checks,
                    &after_partial_bound, &unique_pre_dot_pairs, &duplicate_pre_dot_pairs, &dot_checks,
                    &after_similarity, &pair_seen_occupied, &pair_seen_a, &pair_seen_b, &pair_seen_capacity,
                    &pair_seen_count, &blocks_visited, &blocks_pruned_by_ub, &rows_in_surviving_blocks,
                    &dot_checks_saved_by_row_ub,
                )
                if failed:
                    break

        if failed:
            free(buf)
            _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
            raise MemoryError()

        pairs_before_dedupe = count
        count = _dedupe_pair_buffer(buf, count)
        pairs_after_dedupe = count
        _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
        self.last_stats = {
            "num_index_candidates": int(range_hits),
            "num_valid_index_candidates": int(valid_index_hits),
            "num_unique_index_candidates": int(unique_index_hits),
            "num_duplicate_index_candidates": int(duplicate_index_hits),
            "num_unique_pre_dot_pairs": int(unique_pre_dot_pairs),
            "num_duplicate_pre_dot_pairs": int(duplicate_pre_dot_pairs),
            "num_after_coord_filter": int(after_coord_filter),
            "num_partial_bound_checks": int(partial_bound_checks),
            "num_after_partial_bound": int(after_partial_bound),
            "num_dot_checks": int(dot_checks),
            "num_distance_checks": 0,
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(pairs_before_dedupe),
            "num_pairs_after_dedupe": int(pairs_after_dedupe),
            "num_rows": 0,
            "num_recent_queries": int(recent_entry_ids.shape[0]),
            "num_entries": int(self._current_count + self._total_block_rows()),
            "num_blocks": int(self._block_count),
            "gamma": float(gamma),
            "tau": float(tau),
            "num_blocks_visited": int(blocks_visited),
            "num_blocks_pruned_by_ub": int(blocks_pruned_by_ub),
            "num_rows_in_surviving_blocks": int(rows_in_surviving_blocks),
            "num_dot_checks_saved_by_row_ub": int(dot_checks_saved_by_row_ub),
        }
        if count <= 0:
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr
        out_i = 0
        for i in range(count):
            a = buf[2 * i]
            b = buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            size_a = win_w[a]
            size_b = win_w[b]
            if sid_a < 0 or sid_b < 0 or size_a != size_b:
                continue
            if sid_a == sid_b:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                if time_a >= time_b:
                    row_view[out_i, 2] = time_a
                    row_view[out_i, 3] = time_b
                else:
                    row_view[out_i, 2] = time_b
                    row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            elif time_a == time_b:
                if rank_a <= rank_b:
                    row_view[out_i, 0] = sid_a
                    row_view[out_i, 1] = sid_b
                else:
                    row_view[out_i, 0] = sid_b
                    row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            elif time_a < time_b:
                row_view[out_i, 0] = sid_b
                row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_b
                row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            else:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            out_i += 1

        free(buf)
        self.last_stats["num_rows"] = int(out_i)
        if out_i == count:
            return rows
        return rows[:out_i, :]

    cpdef object find_pair_rows_full_signed(self, long[:] recent_entry_ids, double tau):
        return self._find_pair_rows_full_meta(recent_entry_ids, tau, True)

    cpdef object find_pair_rows_full(self, long[:] recent_entry_ids, double tau):
        return self._find_pair_rows_full_meta(recent_entry_ids, tau, False)

    cpdef object find_pair_rows(self, long[:] recent_entry_ids, double tau):
        return self._find_pair_rows_full_meta(recent_entry_ids, tau, False)

    cdef object _find_pair_rows_full_meta(self, long[:] recent_entry_ids, double tau, bint signed_neg):
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef double[:] q_vec
        cdef double[:] q_keys
        cdef Py_ssize_t i, b_i, count = 0, cap = 1024, out_i
        cdef Py_ssize_t q_row
        cdef Py_ssize_t current_count
        cdef Py_ssize_t offset, length
        cdef int64_t q_entry
        cdef int64_t q_win
        cdef int64_t q_sid
        cdef int64_t q_time
        cdef int64_t q_w
        cdef bint found
        cdef double[:, :] q_vectors
        cdef double[:, :] q_key_values
        cdef int64_t[:] q_entry_ids
        cdef int64_t[:] q_win_idx
        cdef int64_t[:] q_sid_idx
        cdef int64_t[:] q_time_idx
        cdef int64_t[:] q_win_size
        # (2026-07-06) Part 1 follow-up: hoisted once per query batch --
        # see the identical comment in _find_pair_rows_full_cosine_meta.
        cdef double[:, :] bt_vectors
        cdef double[:, :] bt_keys
        cdef int64_t[:] bt_entry_ids
        cdef int64_t[:] bt_window_idx
        cdef int64_t[:] bt_sid_idx
        cdef int64_t[:] bt_time
        cdef int64_t[:] bt_window_size
        cdef int64_t[:, :] bt_orders_by_dim
        cdef double[:, :] bt_sorted_keys_by_dim
        cdef int64_t[:] bt_block_offset
        cdef int64_t[:] bt_block_length
        cdef int64_t[:, :] bt_block_dims
        cdef int64_t[:, :] bt_block_dim_order
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef bint failed = False
        cdef double tau_sq = tau * tau
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t valid_index_hits = 0
        cdef Py_ssize_t after_coord_filter = 0
        cdef Py_ssize_t partial_bound_checks = 0
        cdef Py_ssize_t after_partial_bound = 0
        cdef Py_ssize_t distance_checks = 0
        cdef Py_ssize_t after_similarity = 0
        cdef Py_ssize_t pairs_before_dedupe = 0
        cdef Py_ssize_t pairs_after_dedupe = 0
        cdef int64_t a, b, sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w

        if buf == NULL:
            raise MemoryError()
        if tau < 0.0 or recent_entry_ids.shape[0] == 0:
            free(buf)
            self.last_stats = {
                "num_index_candidates": 0,
                "num_valid_index_candidates": 0,
                "num_unique_index_candidates": 0,
                "num_duplicate_index_candidates": 0,
                "num_after_coord_filter": 0,
                "num_partial_bound_checks": 0,
                "num_after_partial_bound": 0,
                "num_dot_checks": 0,
                "num_distance_checks": 0,
                "num_after_similarity": 0,
                "num_after_dot": 0,
                "num_pairs_before_dedupe": 0,
                "num_pairs_after_dedupe": 0,
                "num_rows": 0,
                "num_recent_queries": 0,
                "num_entries": int(self._current_count),
                "num_blocks": int(self._block_count),
                "gamma": None,
                "tau": float(tau),
            }
            return np.empty((0, 5), dtype=np.int64)

        current_count = self._current_count
        # Hoisted (2026-07-04): see the identical comment in
        # _find_pair_rows_full_cosine_meta / _find_pair_rows_bucketed_cosine_meta.
        cur_entry_ids = self._current_entry_ids_arr
        cur_vectors = self._current_vectors_arr
        cur_key_values = self._current_keys_arr
        cur_win_idx = self._current_window_idx_arr
        cur_sid_idx = self._current_sid_idx_arr
        cur_time_idx = self._current_time_arr
        cur_win_size = self._current_window_size_arr
        bt_vectors = self._bt_vectors
        bt_keys = self._bt_keys
        bt_entry_ids = self._bt_entry_ids
        bt_window_idx = self._bt_window_idx
        bt_sid_idx = self._bt_sid_idx
        bt_time = self._bt_time
        bt_window_size = self._bt_window_size
        bt_orders_by_dim = self._bt_orders_by_dim
        bt_sorted_keys_by_dim = self._bt_sorted_keys_by_dim
        bt_block_offset = self._bt_block_offset
        bt_block_length = self._bt_block_length
        bt_block_dims = self._bt_block_dims
        bt_block_dim_order = self._bt_block_dim_order
        for i in range(recent_entry_ids.shape[0]):
            q_entry = <int64_t>recent_entry_ids[i]
            found = False
            for b_i in range(self._block_head, self._block_head + self._block_count):
                offset = bt_block_offset[b_i]
                length = bt_block_length[b_i]
                q_entry_ids = bt_entry_ids[offset:offset + length]
                q_row = _bisect_left_i64_n(q_entry_ids, length, q_entry)
                if q_row < length and q_entry_ids[q_row] == q_entry:
                    q_vectors = bt_vectors[offset:offset + length, :]
                    q_key_values = bt_keys[offset:offset + length, :]
                    q_win_idx = bt_window_idx[offset:offset + length]
                    q_sid_idx = bt_sid_idx[offset:offset + length]
                    q_time_idx = bt_time[offset:offset + length]
                    q_win_size = bt_window_size[offset:offset + length]
                    q_vec = q_vectors[q_row]
                    q_keys = q_key_values[q_row]
                    q_win = q_win_idx[q_row]
                    q_sid = q_sid_idx[q_row]
                    q_time = q_time_idx[q_row]
                    q_w = q_win_size[q_row]
                    found = True
                    break
            if not found:
                q_entry_ids = cur_entry_ids
                q_row = _bisect_left_i64_n(q_entry_ids, current_count, q_entry)
                if q_row < current_count and q_entry_ids[q_row] == q_entry:
                    q_vectors = cur_vectors
                    q_key_values = cur_key_values
                    q_win_idx = cur_win_idx
                    q_sid_idx = cur_sid_idx
                    q_time_idx = cur_time_idx
                    q_win_size = cur_win_size
                    q_vec = q_vectors[q_row]
                    q_keys = q_key_values[q_row]
                    q_win = q_win_idx[q_row]
                    q_sid = q_sid_idx[q_row]
                    q_time = q_time_idx[q_row]
                    q_w = q_win_size[q_row]
                    found = True
            if not found:
                continue
            for b_i in range(self._block_head, self._block_head + self._block_count):
                offset = bt_block_offset[b_i]
                length = bt_block_length[b_i]
                self._scan_block(
                    q_vec,
                    q_keys,
                    q_win,
                    q_sid,
                    q_time,
                    q_w,
                    bt_vectors[offset:offset + length, :],
                    bt_keys[offset:offset + length, :],
                    bt_window_idx[offset:offset + length],
                    bt_sid_idx[offset:offset + length],
                    bt_time[offset:offset + length],
                    bt_window_size[offset:offset + length],
                    bt_block_dims[b_i, :],
                    bt_block_dim_order[b_i, :],
                    bt_sorted_keys_by_dim[:, offset:offset + length],
                    bt_orders_by_dim[:, offset:offset + length],
                    tau,
                    tau_sq,
                    False,
                    &buf,
                    &count,
                    &cap,
                    &failed,
                    &range_hits,
                    &valid_index_hits,
                    &after_coord_filter,
                    &distance_checks,
                    &after_similarity,
                )
                if failed:
                    break
                if signed_neg:
                    self._scan_block(
                        q_vec,
                        q_keys,
                        q_win,
                        q_sid,
                        q_time,
                        q_w,
                        bt_vectors[offset:offset + length, :],
                        bt_keys[offset:offset + length, :],
                        bt_window_idx[offset:offset + length],
                        bt_sid_idx[offset:offset + length],
                        bt_time[offset:offset + length],
                        bt_window_size[offset:offset + length],
                        bt_block_dims[b_i, :],
                        bt_block_dim_order[b_i, :],
                        bt_sorted_keys_by_dim[:, offset:offset + length],
                        bt_orders_by_dim[:, offset:offset + length],
                        tau,
                        tau_sq,
                        True,
                        &buf,
                        &count,
                        &cap,
                        &failed,
                        &range_hits,
                        &valid_index_hits,
                        &after_coord_filter,
                        &distance_checks,
                        &after_similarity,
                    )
                    if failed:
                        break
            if failed:
                break
            self._scan_current(q_vec, q_keys, q_win, q_sid, q_time, q_w, tau, tau_sq, False, &buf, &count, &cap, &failed, &range_hits, &valid_index_hits, &after_coord_filter, &distance_checks, &after_similarity)
            if failed:
                break
            if signed_neg:
                self._scan_current(q_vec, q_keys, q_win, q_sid, q_time, q_w, tau, tau_sq, True, &buf, &count, &cap, &failed, &range_hits, &valid_index_hits, &after_coord_filter, &distance_checks, &after_similarity)
                if failed:
                    break

        if failed:
            free(buf)
            raise MemoryError()

        pairs_before_dedupe = count
        count = _dedupe_pair_buffer(buf, count)
        pairs_after_dedupe = count
        self.last_stats = {
            "num_index_candidates": int(range_hits),
            "num_valid_index_candidates": int(valid_index_hits),
            "num_unique_index_candidates": int(valid_index_hits),
            "num_duplicate_index_candidates": max(0, int(valid_index_hits) - int(distance_checks)),
            "num_after_coord_filter": int(after_coord_filter),
            "num_partial_bound_checks": int(partial_bound_checks),
            "num_after_partial_bound": int(after_partial_bound),
            "num_dot_checks": 0,
            "num_distance_checks": int(distance_checks),
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(pairs_before_dedupe),
            "num_pairs_after_dedupe": int(pairs_after_dedupe),
            "num_rows": 0,
            "num_recent_queries": int(recent_entry_ids.shape[0]),
            "num_entries": int(self._current_count + self._total_block_rows()),
            "num_blocks": int(self._block_count),
            "gamma": None,
            "tau": float(tau),
        }
        if count <= 0:
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr
        out_i = 0
        for i in range(count):
            a = buf[2 * i]
            b = buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            size_a = win_w[a]
            size_b = win_w[b]
            if sid_a < 0 or sid_b < 0 or size_a != size_b:
                continue
            if sid_a == sid_b:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                if time_a >= time_b:
                    row_view[out_i, 2] = time_a
                    row_view[out_i, 3] = time_b
                else:
                    row_view[out_i, 2] = time_b
                    row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            elif time_a == time_b:
                if rank_a <= rank_b:
                    row_view[out_i, 0] = sid_a
                    row_view[out_i, 1] = sid_b
                else:
                    row_view[out_i, 0] = sid_b
                    row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            elif time_a < time_b:
                row_view[out_i, 0] = sid_b
                row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_b
                row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            else:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            out_i += 1

        free(buf)
        self.last_stats["num_rows"] = int(out_i)
        if out_i == count:
            return rows
        return rows[:out_i, :]


cdef class BucketedMultiIndex(BlockedLazyIndex):
    cdef void _scan_block_bucketed_cosine(self,
                                          double[:] q_vec,
                                          double[:] q_keys,
                                          int64_t q_win,
                                          int64_t q_sid,
                                          int64_t q_rank,
                                          int64_t q_time,
                                          int64_t q_w,
                                          double[:, :] vectors,
                                          double[:, :] keys,
                                          int64_t[:] win_idx,
                                          int64_t[:] sid_idx,
                                          int64_t[:] sid_rank,
                                          int64_t[:] time_idx,
                                          int64_t[:] win_size,
                                          double[:, :] sorted_keys_by_dim,
                                          int64_t[:, :] orders_by_dim,
                                          double[:] residual_norms,
                                          Py_ssize_t row_offset,
                                          Py_ssize_t n_rows,
                                          int64_t[:] bound_dims,
                                          double[:] bound_min,
                                          double[:] bound_max,
                                          double max_residual_norm,
                                          double[:] ref_direction,
                                          double min_cos_to_ref,
                                          double gamma,
                                          bint signed_abs,
                                          int64_t **buf,
                                          Py_ssize_t *count,
                                          Py_ssize_t *cap,
                                          bint *failed,
                                          Py_ssize_t *range_hits,
                                          Py_ssize_t *valid_index_hits,
                                          Py_ssize_t *unique_index_hits,
                                          Py_ssize_t *duplicate_index_hits,
                                          Py_ssize_t *after_coord_filter,
                                          Py_ssize_t *partial_bound_checks,
                                          Py_ssize_t *after_partial_bound,
                                          Py_ssize_t *unique_pre_dot_pairs,
                                          Py_ssize_t *duplicate_pre_dot_pairs,
                                          Py_ssize_t *dot_checks,
                                          Py_ssize_t *after_similarity,
                                          uint8_t **pair_seen_occupied,
                                          int64_t **pair_seen_a,
                                          int64_t **pair_seen_b,
                                          Py_ssize_t *pair_seen_capacity,
                                          Py_ssize_t *pair_seen_count,
                                          Py_ssize_t *blocks_visited,
                                          Py_ssize_t *blocks_pruned_by_ub,
                                          Py_ssize_t *rows_in_surviving_blocks,
                                          Py_ssize_t *dot_checks_saved_by_row_ub):
        # (2026-07-06) Part 1 follow-up: row-level fields (proportional to
        # block size) take the FULL flat table + row_offset/n_rows instead
        # of a pre-sliced per-block sub-array -- see the identical comment
        # in BlockedLazyIndex._scan_block_bucketed_cosine.
        cdef double[:] sorted_keys
        cdef Py_ssize_t key_dims = sorted_keys_by_dim.shape[0]
        cdef Py_ssize_t row, dim, dim_i, pos, left, right, width, pass_i
        cdef Py_ssize_t best_dim_i, best_left, best_right, best_width
        cdef double q_key, lower, upper, dot
        cdef uint8_t *processed = NULL
        cdef int64_t pair_a, pair_b
        cdef bint neg_mode
        # (2026-07-06) Part 1: same opt-in Cauchy-Schwarz block/row
        # upper-bound pruning as BlockedLazyIndex._scan_block_cosine -- see
        # the module-level helper functions and docs/implementation_log.md.
        # This is the function actually used by the "sorted_arrays_bs"
        # backend (BucketedMultiIndex overrides this method independently of
        # its base class).
        cdef bint have_bound_dims = bound_dims.shape[0] > 0
        cdef double q_r_norm = 0.0

        if n_rows == 0 or key_dims <= 0:
            return
        if have_bound_dims and self._enable_row_ub_pruning:
            q_r_norm = _query_residual_norm(q_vec, bound_dims)
        if self._enable_block_ub_pruning:
            if _cone_block_prune_check(q_vec, ref_direction, min_cos_to_ref, gamma, signed_abs):
                blocks_pruned_by_ub[0] += 1
                return
        blocks_visited[0] += 1
        rows_in_surviving_blocks[0] += n_rows
        processed = <uint8_t *>malloc(n_rows * sizeof(uint8_t))
        if processed == NULL:
            failed[0] = True
            return
        for row in range(n_rows):
            processed[row] = <uint8_t>0

        # (2026-07-05) Narrowest-dimension + per-row cascade -- see the
        # sibling rewrite in BlockedLazyIndex._scan_block_bucketed_cosine for
        # the full rationale. Same fused-single-pass role for `processed` as
        # marks_pos/marks_neg had before: a row qualifying under both +q and
        # -q only gets one entry-validity/pair-dedup/dot-check attempt.
        for pass_i in range(2):
            if pass_i == 1 and not signed_abs:
                break
            neg_mode = pass_i == 1
            best_width = n_rows + 1
            best_dim_i = 0
            best_left = row_offset
            best_right = row_offset
            for dim_i in range(key_dims):
                sorted_keys = sorted_keys_by_dim[dim_i]
                q_key = -q_keys[dim_i] if neg_mode else q_keys[dim_i]
                _projection_interval(q_key, gamma, &lower, &upper)
                left = _bisect_left_range(sorted_keys, row_offset, row_offset + n_rows, lower)
                right = _bisect_right_range(sorted_keys, row_offset, row_offset + n_rows, upper)
                width = right - left
                if width < best_width:
                    best_width = width
                    best_left = left
                    best_right = right
                    best_dim_i = dim_i
                    if width == 0:
                        break
            if best_width > n_rows:
                continue
            range_hits[0] += best_width
            for pos in range(best_left, best_right):
                row = orders_by_dim[best_dim_i, pos]
                if processed[row]:
                    duplicate_index_hits[0] += 1
                    continue
                if not self._entry_valid_for_query(q_win, q_sid, q_time, q_w, win_idx[row_offset + row], sid_idx[row_offset + row], time_idx[row_offset + row], win_size[row_offset + row]):
                    continue
                valid_index_hits[0] += 1
                unique_index_hits[0] += 1
                after_coord_filter[0] += 1
                partial_bound_checks[0] += 1
                if not self._key_filter_row_cosine(q_keys, keys, row_offset + row, best_dim_i, gamma, neg_mode):
                    continue
                after_partial_bound[0] += 1
                processed[row] = <uint8_t>1
                _canonical_window_pair(
                    q_win,
                    q_sid,
                    q_rank,
                    q_time,
                    win_idx[row_offset + row],
                    sid_idx[row_offset + row],
                    sid_rank[row_offset + row],
                    time_idx[row_offset + row],
                    &pair_a,
                    &pair_b,
                )
                if not _pair_seen_insert(pair_seen_occupied, pair_seen_a, pair_seen_b, pair_seen_capacity, pair_seen_count, pair_a, pair_b, failed):
                    if failed[0]:
                        free(processed)
                        return
                    duplicate_pre_dot_pairs[0] += 1
                    continue
                unique_pre_dot_pairs[0] += 1
                if have_bound_dims and self._enable_row_ub_pruning:
                    if _row_ub_prune_check(q_vec, vectors, row_offset + row, bound_dims, residual_norms[row_offset + row], q_r_norm, gamma, signed_abs):
                        dot_checks_saved_by_row_ub[0] += 1
                        continue
                dot_checks[0] += 1
                dot = 0.0
                for dim in range(self._n_vectors):
                    dot += q_vec[dim] * vectors[row_offset + row, dim]
                if signed_abs:
                    if fabs(dot) < gamma:
                        continue
                elif dot < gamma:
                    continue
                after_similarity[0] += 1
                if not _append_pair(buf, count, cap, q_win, win_idx[row_offset + row]):
                    failed[0] = True
                    free(processed)
                    return
        free(processed)

    cpdef object find_pair_rows_full_cosine_signed(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_bucketed_cosine_meta(recent_entry_ids, gamma, tau, True)

    cpdef object find_pair_rows_full_cosine(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_bucketed_cosine_meta(recent_entry_ids, gamma, tau, False)

    cdef object _find_pair_rows_bucketed_cosine_meta(self, long[:] recent_entry_ids, double gamma, double tau, bint signed_abs):
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef double[:] q_vec
        cdef double[:] q_keys
        cdef Py_ssize_t i, b_i, count = 0, cap = 1024, out_i
        cdef Py_ssize_t q_row
        cdef Py_ssize_t current_count
        cdef Py_ssize_t offset, length
        cdef int64_t q_entry
        cdef int64_t q_win
        cdef int64_t q_sid
        cdef int64_t q_rank
        cdef int64_t q_time
        cdef int64_t q_w
        cdef bint found
        cdef object current_meta
        cdef bint have_current_meta
        cdef int64_t[:, :] cur_orders_by_dim
        cdef double[:, :] cur_sorted_keys_by_dim
        cdef int64_t[:] cur_bound_dims_meta
        cdef double[:] cur_bound_min_meta
        cdef double[:] cur_bound_max_meta
        cdef double[:] cur_residual_norms_meta
        cdef double cur_max_residual_norm_meta
        cdef double[:] cur_ref_direction_meta
        cdef double cur_min_cos_to_ref_meta
        cdef double[:, :] q_vectors
        cdef double[:, :] q_key_values
        cdef int64_t[:] q_entry_ids
        cdef int64_t[:] q_win_idx
        cdef int64_t[:] q_sid_idx
        cdef int64_t[:] q_sid_rank_idx
        cdef int64_t[:] q_time_idx
        cdef int64_t[:] q_win_size
        # (2026-07-06) Part 1 follow-up: hoisted once per query batch, same
        # rationale as cur_vectors etc below and the memoryview-hoisting fix
        # in monitor_kernels.pyx's update() row loop -- re-deriving a
        # memoryview from a `cdef object` (self._bt_vectors etc) attribute
        # inside the per-(query,block) loop was exactly that anti-pattern,
        # just via array slicing instead of attribute access; it made this
        # rewrite *slower* than the dict-based code until hoisted. See
        # docs/implementation_log.md, "block-level cone pruning: flat
        # typed-array block storage".
        cdef double[:, :] bt_vectors
        cdef double[:, :] bt_keys
        cdef int64_t[:] bt_entry_ids
        cdef int64_t[:] bt_window_idx
        cdef int64_t[:] bt_sid_idx
        cdef int64_t[:] bt_sid_rank
        cdef int64_t[:] bt_time
        cdef int64_t[:] bt_window_size
        cdef double[:] bt_residual_norms
        cdef int64_t[:, :] bt_orders_by_dim
        cdef double[:, :] bt_sorted_keys_by_dim
        cdef int64_t[:] bt_block_offset
        cdef int64_t[:] bt_block_length
        cdef double[:] bt_block_max_residual_norm
        cdef double[:] bt_block_min_cos_to_ref
        cdef double[:, :] bt_block_ref_direction
        cdef int64_t[:, :] bt_block_bound_dims
        cdef double[:, :] bt_block_bound_min
        cdef double[:, :] bt_block_bound_max
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef bint failed = False
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t valid_index_hits = 0
        cdef Py_ssize_t unique_index_hits = 0
        cdef Py_ssize_t duplicate_index_hits = 0
        cdef Py_ssize_t after_coord_filter = 0
        cdef Py_ssize_t partial_bound_checks = 0
        cdef Py_ssize_t after_partial_bound = 0
        cdef Py_ssize_t unique_pre_dot_pairs = 0
        cdef Py_ssize_t duplicate_pre_dot_pairs = 0
        cdef Py_ssize_t dot_checks = 0
        cdef Py_ssize_t after_similarity = 0
        cdef Py_ssize_t pairs_before_dedupe = 0
        cdef Py_ssize_t pairs_after_dedupe = 0
        cdef int64_t a, b, sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w
        cdef uint8_t *pair_seen_occupied = NULL
        cdef int64_t *pair_seen_a = NULL
        cdef int64_t *pair_seen_b = NULL
        cdef Py_ssize_t pair_seen_capacity = 0
        cdef Py_ssize_t pair_seen_count = 0
        cdef Py_ssize_t blocks_visited = 0
        cdef Py_ssize_t blocks_pruned_by_ub = 0
        cdef Py_ssize_t rows_in_surviving_blocks = 0
        cdef Py_ssize_t dot_checks_saved_by_row_ub = 0

        if buf == NULL:
            raise MemoryError()
        if not _pair_seen_init(&pair_seen_occupied, &pair_seen_a, &pair_seen_b, &pair_seen_capacity, max(1024, recent_entry_ids.shape[0] * 64)):
            free(buf)
            raise MemoryError()
        if recent_entry_ids.shape[0] == 0:
            free(buf)
            _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
            self.last_stats = {
                "num_index_candidates": 0,
                "num_valid_index_candidates": 0,
                "num_unique_index_candidates": 0,
                "num_duplicate_index_candidates": 0,
                "num_unique_pre_dot_pairs": 0,
                "num_duplicate_pre_dot_pairs": 0,
                "num_after_coord_filter": 0,
                "num_partial_bound_checks": 0,
                "num_after_partial_bound": 0,
                "num_dot_checks": 0,
                "num_distance_checks": 0,
                "num_after_similarity": 0,
                "num_after_dot": 0,
                "num_pairs_before_dedupe": 0,
                "num_pairs_after_dedupe": 0,
                "num_rows": 0,
                "num_recent_queries": 0,
                "num_entries": int(self._current_count),
                "num_blocks": int(self._block_count),
                "gamma": float(gamma),
                "tau": float(tau),
                "num_blocks_visited": 0,
                "num_blocks_pruned_by_ub": 0,
                "num_rows_in_surviving_blocks": 0,
                "num_dot_checks_saved_by_row_ub": 0,
            }
            return np.empty((0, 5), dtype=np.int64)

        current_count = self._current_count
        # Hoisted (2026-07-04): these back the "still open, not yet closed
        # into a block" fallback path below and never change during this
        # scan (no insertion happens mid-query) -- fetching them once here
        # instead of on every "not found in closed blocks" iteration avoids
        # the same avoidable memoryview-from-Python-attribute cost fixed in
        # monitor_kernels.pyx's update() row loop. See docs/implementation_log.md.
        cur_entry_ids = self._current_entry_ids_arr
        cur_vectors = self._current_vectors_arr
        cur_key_values = self._current_keys_arr
        cur_win_idx = self._current_window_idx_arr
        cur_sid_idx = self._current_sid_idx_arr
        cur_sid_rank_idx = self._current_sid_rank_arr
        cur_time_idx = self._current_time_arr
        cur_win_size = self._current_window_size_arr
        bt_vectors = self._bt_vectors
        bt_keys = self._bt_keys
        bt_entry_ids = self._bt_entry_ids
        bt_window_idx = self._bt_window_idx
        bt_sid_idx = self._bt_sid_idx
        bt_sid_rank = self._bt_sid_rank
        bt_time = self._bt_time
        bt_window_size = self._bt_window_size
        bt_residual_norms = self._bt_residual_norms
        bt_orders_by_dim = self._bt_orders_by_dim
        bt_sorted_keys_by_dim = self._bt_sorted_keys_by_dim
        bt_block_offset = self._bt_block_offset
        bt_block_length = self._bt_block_length
        bt_block_max_residual_norm = self._bt_block_max_residual_norm
        bt_block_min_cos_to_ref = self._bt_block_min_cos_to_ref
        bt_block_ref_direction = self._bt_block_ref_direction
        bt_block_bound_dims = self._bt_block_bound_dims
        bt_block_bound_min = self._bt_block_bound_min
        bt_block_bound_max = self._bt_block_bound_max
        # (2026-07-04) Build once per query batch, not once per row -- see
        # _compute_current_view_metadata for why. None if the current block
        # is empty (nothing to scan).
        current_meta = self._compute_current_view_metadata()
        have_current_meta = current_meta is not None
        if have_current_meta:
            (cur_orders_by_dim, cur_sorted_keys_by_dim, cur_bound_dims_meta, cur_bound_min_meta,
             cur_bound_max_meta, cur_residual_norms_meta, cur_max_residual_norm_meta,
             cur_ref_direction_meta, cur_min_cos_to_ref_meta) = current_meta
        for i in range(recent_entry_ids.shape[0]):
            q_entry = <int64_t>recent_entry_ids[i]
            found = False
            for b_i in range(self._block_head, self._block_head + self._block_count):
                offset = bt_block_offset[b_i]
                length = bt_block_length[b_i]
                q_entry_ids = bt_entry_ids[offset:offset + length]
                q_row = _bisect_left_i64_n(q_entry_ids, length, q_entry)
                if q_row < length and q_entry_ids[q_row] == q_entry:
                    q_vectors = bt_vectors[offset:offset + length, :]
                    q_key_values = bt_keys[offset:offset + length, :]
                    q_win_idx = bt_window_idx[offset:offset + length]
                    q_sid_idx = bt_sid_idx[offset:offset + length]
                    q_sid_rank_idx = bt_sid_rank[offset:offset + length]
                    q_time_idx = bt_time[offset:offset + length]
                    q_win_size = bt_window_size[offset:offset + length]
                    q_vec = q_vectors[q_row]
                    q_keys = q_key_values[q_row]
                    q_win = q_win_idx[q_row]
                    q_sid = q_sid_idx[q_row]
                    q_rank = q_sid_rank_idx[q_row]
                    q_time = q_time_idx[q_row]
                    q_w = q_win_size[q_row]
                    found = True
                    break
            if not found:
                q_entry_ids = cur_entry_ids
                q_row = _bisect_left_i64_n(q_entry_ids, current_count, q_entry)
                if q_row < current_count and q_entry_ids[q_row] == q_entry:
                    q_vectors = cur_vectors
                    q_key_values = cur_key_values
                    q_win_idx = cur_win_idx
                    q_sid_idx = cur_sid_idx
                    q_sid_rank_idx = cur_sid_rank_idx
                    q_time_idx = cur_time_idx
                    q_win_size = cur_win_size
                    q_vec = q_vectors[q_row]
                    q_keys = q_key_values[q_row]
                    q_win = q_win_idx[q_row]
                    q_sid = q_sid_idx[q_row]
                    q_rank = q_sid_rank_idx[q_row]
                    q_time = q_time_idx[q_row]
                    q_w = q_win_size[q_row]
                    found = True
            if not found:
                continue
            for b_i in range(self._block_head, self._block_head + self._block_count):
                offset = bt_block_offset[b_i]
                length = bt_block_length[b_i]
                self._scan_block_bucketed_cosine(
                    q_vec,
                    q_keys,
                    q_win,
                    q_sid,
                    q_rank,
                    q_time,
                    q_w,
                    bt_vectors,
                    bt_keys,
                    bt_window_idx,
                    bt_sid_idx,
                    bt_sid_rank,
                    bt_time,
                    bt_window_size,
                    bt_sorted_keys_by_dim,
                    bt_orders_by_dim,
                    bt_residual_norms,
                    offset,
                    length,
                    bt_block_bound_dims[b_i, :],
                    bt_block_bound_min[b_i, :],
                    bt_block_bound_max[b_i, :],
                    bt_block_max_residual_norm[b_i],
                    bt_block_ref_direction[b_i, :],
                    bt_block_min_cos_to_ref[b_i],
                    gamma,
                    signed_abs,
                    &buf,
                    &count,
                    &cap,
                    &failed,
                    &range_hits,
                    &valid_index_hits,
                    &unique_index_hits,
                    &duplicate_index_hits,
                    &after_coord_filter,
                    &partial_bound_checks,
                    &after_partial_bound,
                    &unique_pre_dot_pairs,
                    &duplicate_pre_dot_pairs,
                    &dot_checks,
                    &after_similarity,
                    &pair_seen_occupied,
                    &pair_seen_a,
                    &pair_seen_b,
                    &pair_seen_capacity,
                    &pair_seen_count,
                    &blocks_visited,
                    &blocks_pruned_by_ub,
                    &rows_in_surviving_blocks,
                    &dot_checks_saved_by_row_ub,
                )
                if failed:
                    break
            if failed:
                break
            # (2026-07-04) Route the current (open) block through the
            # same polymorphic per-block scan closed blocks already use,
            # instead of the fixed, backend-agnostic per-current-block scan.
            # See _compute_current_view_metadata.
            if have_current_meta:
                self._scan_block_bucketed_cosine(
                    q_vec, q_keys, q_win, q_sid, q_rank, q_time, q_w,
                    cur_vectors, cur_key_values,
                    cur_win_idx, cur_sid_idx,
                    cur_sid_rank_idx, cur_time_idx,
                    cur_win_size,
                    cur_sorted_keys_by_dim, cur_orders_by_dim,
                    cur_residual_norms_meta,
                    0, current_count,
                    cur_bound_dims_meta, cur_bound_min_meta, cur_bound_max_meta,
                    cur_max_residual_norm_meta,
                    cur_ref_direction_meta, cur_min_cos_to_ref_meta,
                    gamma, signed_abs, &buf, &count, &cap, &failed, &range_hits, &valid_index_hits,
                    &unique_index_hits, &duplicate_index_hits, &after_coord_filter, &partial_bound_checks,
                    &after_partial_bound, &unique_pre_dot_pairs, &duplicate_pre_dot_pairs, &dot_checks,
                    &after_similarity, &pair_seen_occupied, &pair_seen_a, &pair_seen_b, &pair_seen_capacity,
                    &pair_seen_count, &blocks_visited, &blocks_pruned_by_ub, &rows_in_surviving_blocks,
                    &dot_checks_saved_by_row_ub,
                )
                if failed:
                    break

        if failed:
            free(buf)
            _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
            raise MemoryError()

        pairs_before_dedupe = count
        count = _dedupe_pair_buffer(buf, count)
        pairs_after_dedupe = count
        _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
        self.last_stats = {
            "num_index_candidates": int(range_hits),
            "num_valid_index_candidates": int(valid_index_hits),
            "num_unique_index_candidates": int(unique_index_hits),
            "num_duplicate_index_candidates": int(duplicate_index_hits),
            "num_unique_pre_dot_pairs": int(unique_pre_dot_pairs),
            "num_duplicate_pre_dot_pairs": int(duplicate_pre_dot_pairs),
            "num_after_coord_filter": int(after_coord_filter),
            "num_partial_bound_checks": int(partial_bound_checks),
            "num_after_partial_bound": int(after_partial_bound),
            "num_dot_checks": int(dot_checks),
            "num_distance_checks": 0,
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(pairs_before_dedupe),
            "num_pairs_after_dedupe": int(pairs_after_dedupe),
            "num_rows": 0,
            "num_recent_queries": int(recent_entry_ids.shape[0]),
            "num_entries": int(self._current_count + self._total_block_rows()),
            "num_blocks": int(self._block_count),
            "gamma": float(gamma),
            "tau": float(tau),
            "num_blocks_visited": int(blocks_visited),
            "num_blocks_pruned_by_ub": int(blocks_pruned_by_ub),
            "num_rows_in_surviving_blocks": int(rows_in_surviving_blocks),
            "num_dot_checks_saved_by_row_ub": int(dot_checks_saved_by_row_ub),
        }
        if count <= 0:
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr

        out_i = 0
        for i in range(count):
            a = buf[2 * i]
            b = buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            size_a = win_w[a]
            size_b = win_w[b]
            if size_a != size_b:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            if sid_a < 0 or sid_b < 0:
                continue
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            if sid_a == sid_b:
                if time_a < time_b:
                    a, b = b, a
                    sid_a, sid_b = sid_b, sid_a
                    time_a, time_b = time_b, time_a
            elif time_a == time_b:
                if rank_b < rank_a:
                    a, b = b, a
                    sid_a, sid_b = sid_b, sid_a
                    time_a, time_b = time_b, time_a
            elif time_a < time_b:
                a, b = b, a
                sid_a, sid_b = sid_b, sid_a
                time_a, time_b = time_b, time_a
            row_view[out_i, 0] = sid_a
            row_view[out_i, 1] = sid_b
            row_view[out_i, 2] = time_a
            row_view[out_i, 3] = time_b
            row_view[out_i, 4] = size_a
            out_i += 1
        free(buf)
        self.last_stats["num_rows"] = int(out_i)
        return rows[:out_i, :]


cdef class BalancedIndex:
    cdef public object last_stats
    cdef object _values_arr
    cdef object _left_arr
    cdef object _right_arr
    cdef object _prio_arr
    cdef object _window_idx_arr
    cdef object _sid_idx_arr
    cdef object _time_arr
    cdef object _window_size_arr
    cdef object _win_sid_idx_arr
    cdef object _win_sid_rank_arr
    cdef object _win_time_arr
    cdef object _win_size_arr
    cdef object _active_arr
    cdef object _recent_arr
    cdef object _has_vector_arr
    cdef object _vectors_arr
    cdef object _dim_order_arr
    cdef Py_ssize_t _size
    cdef Py_ssize_t _active_count
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _win_capacity
    cdef Py_ssize_t _n_vectors
    cdef int64_t _root
    cdef uint64_t _rng_state

    def __cinit__(self, Py_ssize_t n_vectors=0, Py_ssize_t initial_capacity=1024, long seed=0):
        if initial_capacity < 16:
            initial_capacity = 16
        if n_vectors < 0:
            n_vectors = 0

        self._size = 0
        self._active_count = 0
        self._capacity = initial_capacity
        self._win_capacity = initial_capacity
        self._n_vectors = n_vectors
        self._root = -1
        self._rng_state = <uint64_t>seed
        if self._rng_state == 0:
            self._rng_state = <uint64_t>0x9E3779B97F4A7C15

        self._values_arr = np.empty(self._capacity, dtype=np.float64)
        self._left_arr = np.empty(self._capacity, dtype=np.int64)
        self._right_arr = np.empty(self._capacity, dtype=np.int64)
        self._prio_arr = np.empty(self._capacity, dtype=np.int64)
        self._window_idx_arr = np.empty(self._capacity, dtype=np.int64)
        self._sid_idx_arr = np.empty(self._capacity, dtype=np.int64)
        self._time_arr = np.empty(self._capacity, dtype=np.int64)
        self._window_size_arr = np.empty(self._capacity, dtype=np.int64)
        self._win_sid_idx_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_sid_rank_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_time_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._win_size_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._active_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._recent_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._has_vector_arr = np.zeros(self._capacity, dtype=np.uint8)
        if self._n_vectors > 0:
            self._vectors_arr = np.zeros((self._capacity, self._n_vectors), dtype=np.float64)
            self._dim_order_arr = np.arange(self._n_vectors, dtype=np.int64)
        else:
            self._vectors_arr = np.empty((0, 0), dtype=np.float64)
            self._dim_order_arr = np.empty(0, dtype=np.int64)
        self.last_stats = {}

    cdef void _refresh_dim_order_from_vectors(self, np.ndarray[np.float64_t, ndim=2] vectors):
        if self._n_vectors <= 0:
            self._dim_order_arr = np.empty(0, dtype=np.int64)
            return
        if vectors.shape[0] <= 1 or vectors.shape[1] != self._n_vectors:
            if np.asarray(self._dim_order_arr).shape[0] != self._n_vectors:
                self._dim_order_arr = np.arange(self._n_vectors, dtype=np.int64)
            return
        variances = np.var(vectors, axis=0)
        self._dim_order_arr = np.ascontiguousarray(np.argsort(variances)[::-1], dtype=np.int64)

    cdef void _ensure_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        cdef np.ndarray[np.float64_t, ndim=1] new_values
        cdef np.ndarray[np.int64_t, ndim=1] new_left
        cdef np.ndarray[np.int64_t, ndim=1] new_right
        cdef np.ndarray[np.int64_t, ndim=1] new_prio
        cdef np.ndarray[np.int64_t, ndim=1] new_window_idx
        cdef np.ndarray[np.int64_t, ndim=1] new_sid_idx
        cdef np.ndarray[np.int64_t, ndim=1] new_time
        cdef np.ndarray[np.int64_t, ndim=1] new_window_size
        cdef np.ndarray[np.uint8_t, ndim=1] new_active
        cdef np.ndarray[np.uint8_t, ndim=1] new_recent
        cdef np.ndarray[np.uint8_t, ndim=1] new_has_vector
        cdef np.ndarray[np.float64_t, ndim=2] new_vectors

        if need <= self._capacity:
            return

        new_cap = self._capacity
        while new_cap < need:
            new_cap *= 2

        new_values = np.empty(new_cap, dtype=np.float64)
        new_left = np.empty(new_cap, dtype=np.int64)
        new_right = np.empty(new_cap, dtype=np.int64)
        new_prio = np.empty(new_cap, dtype=np.int64)
        new_window_idx = np.empty(new_cap, dtype=np.int64)
        new_sid_idx = np.empty(new_cap, dtype=np.int64)
        new_time = np.empty(new_cap, dtype=np.int64)
        new_window_size = np.empty(new_cap, dtype=np.int64)
        new_active = np.zeros(new_cap, dtype=np.uint8)
        new_recent = np.zeros(new_cap, dtype=np.uint8)
        new_has_vector = np.zeros(new_cap, dtype=np.uint8)

        if self._size > 0:
            new_values[:self._size] = self._values_arr[:self._size]
            new_left[:self._size] = self._left_arr[:self._size]
            new_right[:self._size] = self._right_arr[:self._size]
            new_prio[:self._size] = self._prio_arr[:self._size]
            new_window_idx[:self._size] = self._window_idx_arr[:self._size]
            new_sid_idx[:self._size] = self._sid_idx_arr[:self._size]
            new_time[:self._size] = self._time_arr[:self._size]
            new_window_size[:self._size] = self._window_size_arr[:self._size]
            new_active[:self._size] = self._active_arr[:self._size]
            new_recent[:self._size] = self._recent_arr[:self._size]
            new_has_vector[:self._size] = self._has_vector_arr[:self._size]

        self._values_arr = new_values
        self._left_arr = new_left
        self._right_arr = new_right
        self._prio_arr = new_prio
        self._window_idx_arr = new_window_idx
        self._sid_idx_arr = new_sid_idx
        self._time_arr = new_time
        self._window_size_arr = new_window_size
        self._active_arr = new_active
        self._recent_arr = new_recent
        self._has_vector_arr = new_has_vector

        if self._n_vectors > 0:
            new_vectors = np.zeros((new_cap, self._n_vectors), dtype=np.float64)
            if self._size > 0:
                new_vectors[:self._size, :] = self._vectors_arr[:self._size, :]
            self._vectors_arr = new_vectors

        self._capacity = new_cap

    cdef void _ensure_window_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        cdef np.ndarray[np.int64_t, ndim=1] new_sid_idx
        cdef np.ndarray[np.int64_t, ndim=1] new_sid_rank
        cdef np.ndarray[np.int64_t, ndim=1] new_time
        cdef np.ndarray[np.int64_t, ndim=1] new_size

        if need <= self._win_capacity:
            return

        new_cap = self._win_capacity
        while new_cap < need:
            new_cap *= 2

        new_sid_idx = np.full(new_cap, -1, dtype=np.int64)
        new_sid_rank = np.full(new_cap, -1, dtype=np.int64)
        new_time = np.zeros(new_cap, dtype=np.int64)
        new_size = np.zeros(new_cap, dtype=np.int64)
        if self._win_capacity > 0:
            new_sid_idx[:self._win_capacity] = self._win_sid_idx_arr[:self._win_capacity]
            new_sid_rank[:self._win_capacity] = self._win_sid_rank_arr[:self._win_capacity]
            new_time[:self._win_capacity] = self._win_time_arr[:self._win_capacity]
            new_size[:self._win_capacity] = self._win_size_arr[:self._win_capacity]

        self._win_sid_idx_arr = new_sid_idx
        self._win_sid_rank_arr = new_sid_rank
        self._win_time_arr = new_time
        self._win_size_arr = new_size
        self._win_capacity = new_cap

    cdef inline uint64_t _next_rand(self):
        cdef uint64_t x = self._rng_state
        if x == 0:
            x = <uint64_t>0x2545F4914F6CDD1D
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self._rng_state = x
        return x

    cdef inline int64_t _rotate_right(self, int64_t root):
        cdef int64_t[:] left = self._left_arr
        cdef int64_t[:] right = self._right_arr
        cdef int64_t child = left[root]
        left[root] = right[child]
        right[child] = root
        return child

    cdef inline int64_t _rotate_left(self, int64_t root):
        cdef int64_t[:] left = self._left_arr
        cdef int64_t[:] right = self._right_arr
        cdef int64_t child = right[root]
        right[root] = left[child]
        left[child] = root
        return child

    cdef int64_t _insert_rec(self, int64_t root, int64_t node):
        cdef double[:] values = self._values_arr
        cdef int64_t[:] left = self._left_arr
        cdef int64_t[:] right = self._right_arr
        cdef int64_t[:] prio = self._prio_arr
        cdef int64_t child
        if root < 0:
            return node
        if _key_lt(values[node], node, values[root], root):
            left[root] = self._insert_rec(left[root], node)
            child = left[root]
            if child >= 0 and prio[child] > prio[root]:
                root = self._rotate_right(root)
        else:
            right[root] = self._insert_rec(right[root], node)
            child = right[root]
            if child >= 0 and prio[child] > prio[root]:
                root = self._rotate_left(root)
        return root

    cdef int64_t _merge(self, int64_t left_root, int64_t right_root):
        cdef int64_t[:] left = self._left_arr
        cdef int64_t[:] right = self._right_arr
        cdef int64_t[:] prio = self._prio_arr
        if left_root < 0:
            return right_root
        if right_root < 0:
            return left_root
        if prio[left_root] >= prio[right_root]:
            right[left_root] = self._merge(right[left_root], right_root)
            return left_root
        left[right_root] = self._merge(left_root, left[right_root])
        return right_root

    cdef int64_t _erase_rec(self, int64_t root, double value, int64_t node_id):
        cdef double[:] values = self._values_arr
        cdef int64_t[:] left = self._left_arr
        cdef int64_t[:] right = self._right_arr
        cdef double root_value
        if root < 0:
            return -1
        root_value = values[root]
        if _key_lt(value, node_id, root_value, root):
            left[root] = self._erase_rec(left[root], value, node_id)
            return root
        if _key_lt(root_value, root, value, node_id):
            right[root] = self._erase_rec(right[root], value, node_id)
            return root
        return self._merge(left[root], right[root])

    def insert(self, double value, long window_idx, vector=None, long sid_idx=-1, long time=0, long window_size=0, long sid_rank=-1):
        cdef int64_t node
        cdef np.ndarray[np.float64_t, ndim=1] vec
        cdef Py_ssize_t d
        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] prio
        cdef int64_t[:] window_ids
        cdef int64_t[:] sid_ids
        cdef int64_t[:] times
        cdef int64_t[:] window_sizes
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_size
        cdef uint8_t[:] active
        cdef uint8_t[:] recent
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors

        self._ensure_capacity(self._size + 1)
        if window_idx >= 0:
            self._ensure_window_capacity(<Py_ssize_t>window_idx + 1)
        node = <int64_t>self._size
        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        prio = self._prio_arr
        window_ids = self._window_idx_arr
        sid_ids = self._sid_idx_arr
        times = self._time_arr
        window_sizes = self._window_size_arr
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_size = self._win_size_arr
        active = self._active_arr
        recent = self._recent_arr
        has_vector = self._has_vector_arr

        values[node] = value
        window_ids[node] = <int64_t>window_idx
        sid_ids[node] = <int64_t>sid_idx
        times[node] = <int64_t>time
        window_sizes[node] = <int64_t>window_size
        if window_idx >= 0:
            win_sid_idx[window_idx] = <int64_t>sid_idx
            win_sid_rank[window_idx] = <int64_t>sid_rank
            win_time[window_idx] = <int64_t>time
            win_size[window_idx] = <int64_t>window_size
        left[node] = -1
        right[node] = -1
        prio[node] = <int64_t>(self._next_rand() & <uint64_t>0x7FFFFFFFFFFFFFFF)
        active[node] = <uint8_t>1
        recent[node] = <uint8_t>1
        has_vector[node] = <uint8_t>0

        if self._n_vectors > 0:
            vectors = self._vectors_arr
            if vector is None:
                for d in range(self._n_vectors):
                    vectors[node, d] = 0.0
            else:
                vec = np.asarray(vector, dtype=np.float64).ravel()
                if vec.shape[0] != self._n_vectors:
                    raise ValueError("vector size does not match BalancedIndex dimension")
                for d in range(self._n_vectors):
                    vectors[node, d] = vec[d]
                has_vector[node] = <uint8_t>1

        self._root = self._insert_rec(self._root, node)
        self._size += 1
        self._active_count += 1
        return int(node)

    def insert_many(self, values_in, window_idx_in, vectors_in=None, sid_idx_in=None, time_in=None, window_size_in=None, sid_rank_in=None):
        cdef np.ndarray[np.float64_t, ndim=1] values_np = np.asarray(values_in, dtype=np.float64).ravel()
        cdef np.ndarray[np.int64_t, ndim=1] window_idx_np = np.asarray(window_idx_in, dtype=np.int64).ravel()
        cdef np.ndarray[np.float64_t, ndim=2] vectors_np
        cdef np.ndarray[np.int64_t, ndim=1] sid_idx_np
        cdef np.ndarray[np.int64_t, ndim=1] time_np
        cdef np.ndarray[np.int64_t, ndim=1] window_size_np
        cdef np.ndarray[np.int64_t, ndim=1] sid_rank_np
        cdef np.ndarray[np.int64_t, ndim=1] entry_ids
        cdef Py_ssize_t n = values_np.shape[0]
        cdef Py_ssize_t i, d
        cdef int64_t node
        cdef int64_t max_window_idx = -1
        cdef double[:] values_mv
        cdef int64_t[:] window_idx_mv
        cdef int64_t[:] sid_idx_mv
        cdef int64_t[:] time_mv
        cdef int64_t[:] window_size_mv
        cdef int64_t[:] sid_rank_mv
        cdef double[:, :] vectors_mv
        cdef double[:] tree_values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] prio
        cdef int64_t[:] window_ids
        cdef int64_t[:] sid_ids
        cdef int64_t[:] times
        cdef int64_t[:] window_sizes
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_size
        cdef uint8_t[:] active
        cdef uint8_t[:] recent
        cdef uint8_t[:] has_vector
        cdef double[:, :] tree_vectors

        if window_idx_np.shape[0] != n:
            raise ValueError("window_idx size does not match values size")
        if sid_idx_in is None:
            sid_idx_np = np.full(n, -1, dtype=np.int64)
        else:
            sid_idx_np = np.asarray(sid_idx_in, dtype=np.int64).ravel()
        if time_in is None:
            time_np = np.zeros(n, dtype=np.int64)
        else:
            time_np = np.asarray(time_in, dtype=np.int64).ravel()
        if window_size_in is None:
            window_size_np = np.zeros(n, dtype=np.int64)
        else:
            window_size_np = np.asarray(window_size_in, dtype=np.int64).ravel()
        if sid_rank_in is None:
            sid_rank_np = np.full(n, -1, dtype=np.int64)
        else:
            sid_rank_np = np.asarray(sid_rank_in, dtype=np.int64).ravel()
        if sid_idx_np.shape[0] != n or time_np.shape[0] != n or window_size_np.shape[0] != n or sid_rank_np.shape[0] != n:
            raise ValueError("metadata array sizes must match values size")

        if vectors_in is None:
            vectors_np = np.empty((0, 0), dtype=np.float64)
        else:
            vectors_np = np.asarray(vectors_in, dtype=np.float64)
            if vectors_np.ndim != 2 or vectors_np.shape[0] != n:
                raise ValueError("vectors must be a 2D array with one row per value")
            if self._n_vectors > 0 and vectors_np.shape[1] != self._n_vectors:
                raise ValueError("vector size does not match BalancedIndex dimension")
            if self._n_vectors > 0:
                self._refresh_dim_order_from_vectors(np.ascontiguousarray(vectors_np, dtype=np.float64))

        if n == 0:
            return np.empty(0, dtype=np.int64)

        for i in range(n):
            if window_idx_np[i] > max_window_idx:
                max_window_idx = <int64_t>window_idx_np[i]

        self._ensure_capacity(self._size + n)
        if max_window_idx >= 0:
            self._ensure_window_capacity(<Py_ssize_t>max_window_idx + 1)

        entry_ids = np.empty(n, dtype=np.int64)
        values_mv = values_np
        window_idx_mv = window_idx_np
        sid_idx_mv = sid_idx_np
        time_mv = time_np
        window_size_mv = window_size_np
        sid_rank_mv = sid_rank_np
        vectors_mv = vectors_np
        tree_values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        prio = self._prio_arr
        window_ids = self._window_idx_arr
        sid_ids = self._sid_idx_arr
        times = self._time_arr
        window_sizes = self._window_size_arr
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_size = self._win_size_arr
        active = self._active_arr
        recent = self._recent_arr
        has_vector = self._has_vector_arr
        tree_vectors = self._vectors_arr

        for i in range(n):
            node = <int64_t>self._size
            tree_values[node] = values_mv[i]
            window_ids[node] = window_idx_mv[i]
            sid_ids[node] = sid_idx_mv[i]
            times[node] = time_mv[i]
            window_sizes[node] = window_size_mv[i]
            if window_idx_mv[i] >= 0:
                win_sid_idx[window_idx_mv[i]] = sid_idx_mv[i]
                win_sid_rank[window_idx_mv[i]] = sid_rank_mv[i]
                win_time[window_idx_mv[i]] = time_mv[i]
                win_size[window_idx_mv[i]] = window_size_mv[i]
            left[node] = -1
            right[node] = -1
            prio[node] = <int64_t>(self._next_rand() & <uint64_t>0x7FFFFFFFFFFFFFFF)
            active[node] = <uint8_t>1
            recent[node] = <uint8_t>1
            has_vector[node] = <uint8_t>0
            if self._n_vectors > 0:
                if vectors_in is None:
                    for d in range(self._n_vectors):
                        tree_vectors[node, d] = 0.0
                else:
                    for d in range(self._n_vectors):
                        tree_vectors[node, d] = vectors_mv[i, d]
                    has_vector[node] = <uint8_t>1
            self._root = self._insert_rec(self._root, node)
            self._size += 1
            self._active_count += 1
            entry_ids[i] = node
        return entry_ids

    cpdef clear_recent(self):
        cdef uint8_t[:] recent = self._recent_arr
        cdef Py_ssize_t i
        for i in range(self._size):
            recent[i] = <uint8_t>0

    def remove(self, long entry_id):
        cdef int64_t node = <int64_t>entry_id
        cdef double[:] values
        cdef uint8_t[:] active
        cdef uint8_t[:] recent
        if node < 0 or node >= self._size:
            return False
        active = self._active_arr
        if active[node] == 0:
            return False
        values = self._values_arr
        self._root = self._erase_rec(self._root, values[node], node)
        active[node] = <uint8_t>0
        recent = self._recent_arr
        recent[node] = <uint8_t>0
        self._active_count -= 1
        return True

    def remove_many(self, entry_ids_in):
        cdef np.ndarray[np.int64_t, ndim=1] entry_ids = np.asarray(entry_ids_in, dtype=np.int64).ravel()
        cdef Py_ssize_t i
        cdef int64_t node
        cdef int64_t removed = 0
        cdef double[:] values
        cdef uint8_t[:] active
        cdef uint8_t[:] recent
        if entry_ids.shape[0] == 0:
            return 0
        values = self._values_arr
        active = self._active_arr
        recent = self._recent_arr
        for i in range(entry_ids.shape[0]):
            node = <int64_t>entry_ids[i]
            if node < 0 or node >= self._size:
                continue
            if active[node] == 0:
                continue
            self._root = self._erase_rec(self._root, values[node], node)
            active[node] = <uint8_t>0
            recent[node] = <uint8_t>0
            self._active_count -= 1
            removed += 1
        return int(removed)

    cpdef list find_recent_pairs(self, double tau):
        return self._find_recent_pairs_meta(tau, False, False)

    cpdef list find_recent_pairs_with_dist(self, double tau):
        return self._find_recent_pairs_meta(tau, False, True)

    cpdef list find_recent_pairs_full(self, double tau):
        return self._find_recent_pairs_meta(tau, True, False)

    cpdef list find_recent_pairs_full_with_dist(self, double tau):
        return self._find_recent_pairs_meta(tau, True, True)

    cpdef object find_recent_pair_rows(self, double tau):
        return self._find_recent_pair_rows_meta(tau, False, False)

    cpdef object find_recent_pair_rows_full(self, double tau):
        return self._find_recent_pair_rows_meta(tau, True, False)

    cpdef object find_recent_pair_rows_full_signed(self, double tau):
        return self._find_recent_pair_rows_meta(tau, True, True)

    cpdef object find_pair_rows(self, long[:] recent_entry_ids, double tau):
        return self._find_pair_rows_for_entries_meta(recent_entry_ids, tau, False, False)

    cpdef object find_pair_rows_full(self, long[:] recent_entry_ids, double tau):
        return self._find_pair_rows_for_entries_meta(recent_entry_ids, tau, True, False)

    cpdef object find_pair_rows_full_signed(self, long[:] recent_entry_ids, double tau):
        return self._find_pair_rows_for_entries_meta(recent_entry_ids, tau, True, True)

    cpdef object find_pair_rows_full_cosine(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_full_cosine_meta(recent_entry_ids, gamma, tau, False)

    cpdef object find_pair_rows_full_cosine_signed(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_full_cosine_meta(recent_entry_ids, gamma, tau, True)

    cdef object _find_pair_rows_full_cosine_meta(self, long[:] recent_entry_ids, double gamma, double tau, bint signed_abs):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, lower_neg, upper_neg
        cdef int64_t sid_r, time_r, w_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef int64_t[:] sid_idx
        cdef int64_t[:] time_idx
        cdef int64_t[:] win_size
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors
        cdef int64_t[:] dim_order
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t out_i
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t unique_pre_dot_pairs = 0
        cdef Py_ssize_t duplicate_pre_dot_pairs = 0
        cdef Py_ssize_t dot_checks = 0
        cdef Py_ssize_t after_similarity = 0
        cdef Py_ssize_t pairs_before_dedupe = 0
        cdef Py_ssize_t pairs_after_dedupe = 0
        cdef int64_t rank_r
        cdef uint8_t *pair_seen_occupied = NULL
        cdef int64_t *pair_seen_a = NULL
        cdef int64_t *pair_seen_b = NULL
        cdef Py_ssize_t pair_seen_capacity = 0
        cdef Py_ssize_t pair_seen_count = 0

        if buf == NULL:
            raise MemoryError()
        if not _pair_seen_init(&pair_seen_occupied, &pair_seen_a, &pair_seen_b, &pair_seen_capacity, max(1024, n_recent * 64)):
            free(buf)
            raise MemoryError()

        n_dim = self._n_vectors
        if tau < 0.0 or self._root < 0 or size == 0 or n_recent == 0 or n_dim <= 0:
            free(buf)
            _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
            self.last_stats = {
                "num_index_candidates": 0,
                "num_valid_index_candidates": 0,
                "num_unique_index_candidates": 0,
                "num_duplicate_index_candidates": 0,
                "num_unique_pre_dot_pairs": 0,
                "num_duplicate_pre_dot_pairs": 0,
                "num_dot_checks": 0,
                "num_distance_checks": 0,
                "num_after_similarity": 0,
                "num_after_dot": 0,
                "num_pairs_before_dedupe": 0,
                "num_pairs_after_dedupe": 0,
                "num_rows": 0,
                "num_recent_queries": 0,
                "num_entries": int(self._active_count),
                "num_blocks": 1,
                "gamma": float(gamma),
                "tau": float(tau),
            }
            return np.empty((0, 5), dtype=np.int64)

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        sid_idx = self._sid_idx_arr
        time_idx = self._time_arr
        win_size = self._window_size_arr
        win_sid_rank = self._win_sid_rank_arr
        active = self._active_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        if np.asarray(self._dim_order_arr).shape[0] != n_dim:
            self._dim_order_arr = np.arange(n_dim, dtype=np.int64)
        dim_order = self._dim_order_arr
        root_node = self._root

        with nogil:
            for i in range(n_recent):
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0 or has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                sid_r = sid_idx[node]
                rank_r = win_sid_rank[ridx]
                time_r = time_idx[node]
                w_r = win_size[node]
                value = values[node]
                _projection_interval(value, gamma, &lower, &upper)
                # (2026-07-06) Mirror symmetry of _projection_interval under
                # v -> -v (same identity relied on for the sorted_arrays_bs
                # fusion earlier this session): the negative-direction
                # interval is exactly [-upper, -lower], no second
                # _projection_interval call needed.
                if signed_abs:
                    lower_neg = -upper
                    upper_neg = -lower
                else:
                    lower_neg = 0.0
                    upper_neg = -1.0
                _scan_tree_full_meta_cosine(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    sid_idx,
                    win_sid_rank,
                    time_idx,
                    win_size,
                    active,
                    has_vector,
                    vectors,
                    dim_order,
                    lower,
                    upper,
                    lower_neg,
                    upper_neg,
                    node,
                    ridx,
                    sid_r,
                    rank_r,
                    time_r,
                    w_r,
                    n_dim,
                    gamma,
                    signed_abs,
                    &buf,
                    &count,
                    &cap,
                    &failed,
                    &range_hits,
                    &unique_pre_dot_pairs,
                    &duplicate_pre_dot_pairs,
                    &dot_checks,
                    &after_similarity,
                    &pair_seen_occupied,
                    &pair_seen_a,
                    &pair_seen_b,
                    &pair_seen_capacity,
                    &pair_seen_count,
                )
                if failed:
                    break

        if failed:
            free(buf)
            _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
            raise MemoryError()

        pairs_before_dedupe = count
        count = _dedupe_pair_buffer(buf, count)
        pairs_after_dedupe = count
        _pair_seen_free(pair_seen_occupied, pair_seen_a, pair_seen_b)
        self.last_stats = {
            "num_index_candidates": int(range_hits),
            "num_valid_index_candidates": int(range_hits),
            "num_unique_index_candidates": int(dot_checks),
            "num_duplicate_index_candidates": max(0, int(range_hits) - int(dot_checks)),
            "num_unique_pre_dot_pairs": int(unique_pre_dot_pairs),
            "num_duplicate_pre_dot_pairs": int(duplicate_pre_dot_pairs),
            "num_dot_checks": int(dot_checks),
            "num_distance_checks": 0,
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(pairs_before_dedupe),
            "num_pairs_after_dedupe": int(pairs_after_dedupe),
            "num_rows": 0,
            "num_recent_queries": int(n_recent),
            "num_entries": int(self._active_count),
            "num_blocks": 1,
            "gamma": float(gamma),
            "tau": float(tau),
        }
        if count <= 0:
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr

        out_i = 0
        for i in range(count):
            a = buf[2 * i]
            b = buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            size_a = win_w[a]
            size_b = win_w[b]
            if size_a != size_b:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            if sid_a < 0 or sid_b < 0:
                continue
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            if sid_a == sid_b:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                if time_a >= time_b:
                    row_view[out_i, 2] = time_a
                    row_view[out_i, 3] = time_b
                else:
                    row_view[out_i, 2] = time_b
                    row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            elif time_a == time_b:
                if rank_a <= rank_b:
                    row_view[out_i, 0] = sid_a
                    row_view[out_i, 1] = sid_b
                else:
                    row_view[out_i, 0] = sid_b
                    row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            elif time_a < time_b:
                row_view[out_i, 0] = sid_b
                row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_b
                row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            else:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            out_i += 1
        free(buf)
        self.last_stats["num_rows"] = int(out_i)
        if out_i != count:
            return np.asarray(rows[:out_i, :], dtype=np.int64)
        return rows

    cdef object _find_pair_rows_for_entries_meta(self, long[:] recent_entry_ids, double tau, bint full_vector, bint signed_neg):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef int64_t sid_r, time_r, w_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef double *dist_buf = NULL
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef int64_t[:] sid_idx
        cdef int64_t[:] time_idx
        cdef int64_t[:] win_size
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors
        cdef int64_t[:] dim_order
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t out_i
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t distance_checks = 0
        cdef Py_ssize_t after_similarity = 0
        cdef Py_ssize_t pairs_before_dedupe = 0
        cdef Py_ssize_t pairs_after_dedupe = 0

        if buf == NULL:
            raise MemoryError()

        n_dim = self._n_vectors
        if tau < 0.0 or self._root < 0 or size == 0 or n_recent == 0 or (full_vector and n_dim <= 0):
            free(buf)
            self.last_stats = {
                "num_index_candidates": 0,
                "num_dot_checks": 0,
                "num_distance_checks": 0,
                "num_after_similarity": 0,
                "num_after_dot": 0,
                "num_pairs_before_dedupe": 0,
                "num_pairs_after_dedupe": 0,
                "num_rows": 0,
                "num_recent_queries": 0,
                "num_entries": int(self._active_count),
                "num_blocks": 1,
                "gamma": None,
                "tau": float(tau),
            }
            return np.empty((0, 5), dtype=np.int64)

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        sid_idx = self._sid_idx_arr
        time_idx = self._time_arr
        win_size = self._window_size_arr
        active = self._active_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        if full_vector and np.asarray(self._dim_order_arr).shape[0] != n_dim:
            self._dim_order_arr = np.arange(n_dim, dtype=np.int64)
        dim_order = self._dim_order_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in range(n_recent):
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0:
                    continue
                if full_vector and has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                sid_r = sid_idx[node]
                time_r = time_idx[node]
                w_r = win_size[node]
                value = values[node]
                lower = value - tau
                upper = value + tau
                if full_vector:
                    _scan_tree_full_meta(
                        root_node,
                        left,
                        right,
                        values,
                        window_idx,
                        sid_idx,
                        time_idx,
                        win_size,
                        active,
                        has_vector,
                        vectors,
                        dim_order,
                        lower,
                        upper,
                        node,
                        ridx,
                        sid_r,
                        time_r,
                        w_r,
                        n_dim,
                        tau_sq,
                        False,
                        &buf,
                        &dist_buf,
                        &count,
                        &cap,
                        &failed,
                        &range_hits,
                        &distance_checks,
                        &after_similarity,
                    )
                    if signed_neg and not failed:
                        lower = -value - tau
                        upper = -value + tau
                        _scan_tree_full_meta_neg(
                            root_node,
                            left,
                            right,
                            values,
                            window_idx,
                            sid_idx,
                            time_idx,
                            win_size,
                            active,
                            has_vector,
                            vectors,
                            dim_order,
                            lower,
                            upper,
                            node,
                            ridx,
                            sid_r,
                            time_r,
                            w_r,
                            n_dim,
                            tau_sq,
                            &buf,
                            &count,
                            &cap,
                            &failed,
                            &range_hits,
                            &distance_checks,
                            &after_similarity,
                        )
                else:
                    _scan_tree_scalar_meta(
                        root_node,
                        left,
                        right,
                        values,
                        window_idx,
                        sid_idx,
                        time_idx,
                        win_size,
                        active,
                        lower,
                        upper,
                        value,
                        ridx,
                        sid_r,
                        time_r,
                        w_r,
                        False,
                        &buf,
                        &dist_buf,
                        &count,
                        &cap,
                        &failed,
                    )
                if failed:
                    break

        if failed:
            free(buf)
            raise MemoryError()

        pairs_before_dedupe = count
        count = _dedupe_pair_buffer(buf, count)
        pairs_after_dedupe = count
        self.last_stats = {
            "num_index_candidates": int(range_hits),
            "num_dot_checks": 0,
            "num_distance_checks": int(distance_checks),
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(pairs_before_dedupe),
            "num_pairs_after_dedupe": int(pairs_after_dedupe),
            "num_rows": 0,
            "num_recent_queries": int(n_recent),
            "num_entries": int(self._active_count),
            "num_blocks": 1,
            "gamma": None,
            "tau": float(tau),
        }
        if count <= 0:
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr

        out_i = 0
        for i in range(count):
            a = buf[2 * i]
            b = buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            size_a = win_w[a]
            size_b = win_w[b]
            if sid_a < 0 or sid_b < 0 or size_a != size_b:
                continue
            if sid_a == sid_b:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                if time_a >= time_b:
                    row_view[out_i, 2] = time_a
                    row_view[out_i, 3] = time_b
                else:
                    row_view[out_i, 2] = time_b
                    row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            elif time_a == time_b:
                if rank_a <= rank_b:
                    row_view[out_i, 0] = sid_a
                    row_view[out_i, 1] = sid_b
                else:
                    row_view[out_i, 0] = sid_b
                    row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            elif time_a < time_b:
                row_view[out_i, 0] = sid_b
                row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_b
                row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            else:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            out_i += 1

        free(buf)
        self.last_stats["num_rows"] = int(out_i)
        if out_i == count:
            return rows
        return rows[:out_i, :]

    cdef object _find_recent_pair_rows_meta(self, double tau, bint full_vector, bint signed_neg):
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef int64_t sid_r, time_r, w_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef double *dist_buf = NULL
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef int64_t[:] sid_idx
        cdef int64_t[:] time_idx
        cdef int64_t[:] win_size
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w
        cdef uint8_t[:] active
        cdef uint8_t[:] recent
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors
        cdef int64_t[:] dim_order
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t out_i
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t distance_checks = 0
        cdef Py_ssize_t after_similarity = 0

        if buf == NULL:
            raise MemoryError()

        n_dim = self._n_vectors
        if tau < 0.0 or self._root < 0 or size == 0 or (full_vector and n_dim <= 0):
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        sid_idx = self._sid_idx_arr
        time_idx = self._time_arr
        win_size = self._window_size_arr
        active = self._active_arr
        recent = self._recent_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        if full_vector and np.asarray(self._dim_order_arr).shape[0] != n_dim:
            self._dim_order_arr = np.arange(n_dim, dtype=np.int64)
        dim_order = self._dim_order_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in range(size):
                node = <int64_t>i
                if active[node] == 0 or recent[node] == 0:
                    continue
                if full_vector and has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                sid_r = sid_idx[node]
                time_r = time_idx[node]
                w_r = win_size[node]
                value = values[node]
                lower = value - tau
                upper = value + tau
                if full_vector:
                    _scan_tree_full_meta(
                        root_node,
                        left,
                        right,
                        values,
                        window_idx,
                        sid_idx,
                        time_idx,
                        win_size,
                        active,
                        has_vector,
                        vectors,
                        dim_order,
                        lower,
                        upper,
                        node,
                        ridx,
                        sid_r,
                        time_r,
                        w_r,
                        n_dim,
                        tau_sq,
                        False,
                        &buf,
                        &dist_buf,
                        &count,
                        &cap,
                        &failed,
                        &range_hits,
                        &distance_checks,
                        &after_similarity,
                    )
                    if signed_neg and not failed:
                        lower = -value - tau
                        upper = -value + tau
                        _scan_tree_full_meta_neg(
                            root_node,
                            left,
                            right,
                            values,
                            window_idx,
                            sid_idx,
                            time_idx,
                            win_size,
                            active,
                            has_vector,
                            vectors,
                            dim_order,
                            lower,
                            upper,
                            node,
                            ridx,
                            sid_r,
                            time_r,
                            w_r,
                            n_dim,
                            tau_sq,
                            &buf,
                            &count,
                            &cap,
                            &failed,
                            &range_hits,
                            &distance_checks,
                            &after_similarity,
                        )
                else:
                    _scan_tree_scalar_meta(
                        root_node,
                        left,
                        right,
                        values,
                        window_idx,
                        sid_idx,
                        time_idx,
                        win_size,
                        active,
                        lower,
                        upper,
                        value,
                        ridx,
                        sid_r,
                        time_r,
                        w_r,
                        False,
                        &buf,
                        &dist_buf,
                        &count,
                        &cap,
                        &failed,
                    )
                if failed:
                    break

        if failed:
            free(buf)
            raise MemoryError()

        count = _dedupe_pair_buffer(buf, count)
        if count <= 0:
            free(buf)
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr

        out_i = 0
        for i in range(count):
            a = buf[2 * i]
            b = buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            size_a = win_w[a]
            size_b = win_w[b]
            if sid_a < 0 or sid_b < 0 or size_a != size_b:
                continue
            if sid_a == sid_b:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                if time_a >= time_b:
                    row_view[out_i, 2] = time_a
                    row_view[out_i, 3] = time_b
                else:
                    row_view[out_i, 2] = time_b
                    row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            elif time_a == time_b:
                if rank_a <= rank_b:
                    row_view[out_i, 0] = sid_a
                    row_view[out_i, 1] = sid_b
                else:
                    row_view[out_i, 0] = sid_b
                    row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            elif time_a < time_b:
                row_view[out_i, 0] = sid_b
                row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_b
                row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            else:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            out_i += 1

        free(buf)
        if out_i == count:
            return rows
        return rows[:out_i, :]

    cdef list _find_recent_pairs_meta(self, double tau, bint full_vector, bint want_dist):
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef int64_t sid_r, time_r, w_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef double *dist_buf = NULL
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef int64_t[:] sid_idx
        cdef int64_t[:] time_idx
        cdef int64_t[:] win_size
        cdef uint8_t[:] active
        cdef uint8_t[:] recent
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors
        cdef int64_t[:] dim_order
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t distance_checks = 0
        cdef Py_ssize_t after_similarity = 0

        if want_dist:
            dist_buf = <double *>malloc(1024 * sizeof(double))
        if buf == NULL or (want_dist and dist_buf == NULL):
            if buf != NULL:
                free(buf)
            if dist_buf != NULL:
                free(dist_buf)
            raise MemoryError()

        n_dim = self._n_vectors
        if tau < 0.0 or self._root < 0 or size == 0 or (full_vector and n_dim <= 0):
            free(buf)
            if dist_buf != NULL:
                free(dist_buf)
            self.last_stats = {
                "num_index_candidates": 0,
                "num_dot_checks": 0,
                "num_distance_checks": 0,
                "num_after_similarity": 0,
                "num_after_dot": 0,
                "num_pairs_before_dedupe": 0,
                "num_pairs_after_dedupe": 0,
                "num_rows": 0,
                "num_recent_queries": 0,
                "num_entries": int(size),
                "num_blocks": 0,
                "gamma": None,
                "tau": float(tau),
            }
            return []

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        sid_idx = self._sid_idx_arr
        time_idx = self._time_arr
        win_size = self._window_size_arr
        active = self._active_arr
        recent = self._recent_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        if full_vector and np.asarray(self._dim_order_arr).shape[0] != n_dim:
            self._dim_order_arr = np.arange(n_dim, dtype=np.int64)
        dim_order = self._dim_order_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in range(size):
                node = <int64_t>i
                if active[node] == 0 or recent[node] == 0:
                    continue
                if full_vector and has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                sid_r = sid_idx[node]
                time_r = time_idx[node]
                w_r = win_size[node]
                value = values[node]
                lower = value - tau
                upper = value + tau
                if full_vector:
                    _scan_tree_full_meta(
                        root_node,
                        left,
                        right,
                        values,
                        window_idx,
                        sid_idx,
                        time_idx,
                        win_size,
                        active,
                        has_vector,
                        vectors,
                        dim_order,
                        lower,
                        upper,
                        node,
                        ridx,
                        sid_r,
                        time_r,
                        w_r,
                        n_dim,
                        tau_sq,
                        want_dist,
                        &buf,
                        &dist_buf,
                        &count,
                        &cap,
                        &failed,
                        &range_hits,
                        &distance_checks,
                        &after_similarity,
                    )
                else:
                    _scan_tree_scalar_meta(
                        root_node,
                        left,
                        right,
                        values,
                        window_idx,
                        sid_idx,
                        time_idx,
                        win_size,
                        active,
                        lower,
                        upper,
                        value,
                        ridx,
                        sid_r,
                        time_r,
                        w_r,
                        want_dist,
                        &buf,
                        &dist_buf,
                        &count,
                        &cap,
                        &failed,
                    )
                if failed:
                    break

        if failed:
            free(buf)
            if dist_buf != NULL:
                free(dist_buf)
            raise MemoryError()

        if not want_dist:
            count = _dedupe_pair_buffer(buf, count)

        if want_dist:
            pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
        else:
            pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
        self.last_stats = {
            "num_index_candidates": int(range_hits),
            "num_dot_checks": 0,
            "num_distance_checks": int(distance_checks),
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(count),
            "num_pairs_after_dedupe": int(count),
            "num_rows": 0,
            "num_recent_queries": int(size),
            "num_entries": int(size),
            "num_blocks": 0,
            "gamma": None,
            "tau": float(tau),
        }
        free(buf)
        if dist_buf != NULL:
            free(dist_buf)
        return pairs

    cpdef list find_pairs(self,
                          long[:] recent_entry_ids,
                          long[:] win_sid_idx,
                          long[:] win_time,
                          double tau):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t n_win = win_sid_idx.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value
        cdef long sid_r, time_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef uint8_t[:] active

        if buf == NULL:
            raise MemoryError()

        if tau < 0.0 or n_recent == 0 or self._root < 0 or n_win == 0 or win_time.shape[0] != n_win:
            free(buf)
            return []

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        active = self._active_arr
        root_node = self._root

        with nogil:
            for i in range(n_recent):
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0:
                    continue
                ridx = window_idx[node]
                if ridx < 0 or ridx >= n_win:
                    continue
                sid_r = win_sid_idx[ridx]
                time_r = win_time[ridx]
                value = values[node]
                lower = value - tau
                upper = value + tau
                _scan_tree_scalar(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    active,
                    lower,
                    upper,
                    ridx,
                    sid_r,
                    time_r,
                    win_sid_idx,
                    win_time,
                    &buf,
                    &count,
                    &cap,
                    &failed,
                )
                if failed:
                    break

        if failed:
            free(buf)
            raise MemoryError()

        pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
        free(buf)
        return pairs

    cpdef list find_pairs_with_dist(self,
                                    long[:] recent_entry_ids,
                                    long[:] win_sid_idx,
                                    long[:] win_time,
                                    double tau):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t n_win = win_sid_idx.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value
        cdef long sid_r, time_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef double *dist_buf = <double *>malloc(1024 * sizeof(double))
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef uint8_t[:] active

        if buf == NULL or dist_buf == NULL:
            if buf != NULL:
                free(buf)
            if dist_buf != NULL:
                free(dist_buf)
            raise MemoryError()

        if tau < 0.0 or n_recent == 0 or self._root < 0 or n_win == 0 or win_time.shape[0] != n_win:
            free(buf)
            free(dist_buf)
            return []

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        active = self._active_arr
        root_node = self._root

        with nogil:
            for i in range(n_recent):
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0:
                    continue
                ridx = window_idx[node]
                if ridx < 0 or ridx >= n_win:
                    continue
                sid_r = win_sid_idx[ridx]
                time_r = win_time[ridx]
                value = values[node]
                lower = value - tau
                upper = value + tau
                _scan_tree_scalar_dist(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    active,
                    lower,
                    upper,
                    value,
                    ridx,
                    sid_r,
                    time_r,
                    win_sid_idx,
                    win_time,
                    &buf,
                    &dist_buf,
                    &count,
                    &cap,
                    &failed,
                )
                if failed:
                    break

        if failed:
            free(buf)
            free(dist_buf)
            raise MemoryError()

        pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
        free(buf)
        free(dist_buf)
        return pairs

    cpdef list find_pairs_full(self,
                               long[:] recent_entry_ids,
                               long[:] win_sid_idx,
                               long[:] win_time,
                               double tau):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t n_win = win_sid_idx.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef long sid_r, time_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t distance_checks = 0
        cdef Py_ssize_t after_similarity = 0

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors

        if buf == NULL:
            raise MemoryError()

        n_dim = self._n_vectors
        if (
            tau < 0.0
            or n_recent == 0
            or self._root < 0
            or n_dim <= 0
            or n_win == 0
            or win_time.shape[0] != n_win
        ):
            free(buf)
            return []

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        active = self._active_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in range(n_recent):
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0 or has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                if ridx < 0 or ridx >= n_win:
                    continue
                sid_r = win_sid_idx[ridx]
                time_r = win_time[ridx]
                value = values[node]
                lower = value - tau
                upper = value + tau
                _scan_tree_full(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    active,
                    has_vector,
                    vectors,
                    lower,
                    upper,
                    node,
                    ridx,
                    sid_r,
                    time_r,
                    win_sid_idx,
                    win_time,
                    n_dim,
                    tau_sq,
                    &buf,
                    &count,
                    &cap,
                    &failed,
                )
                if failed:
                    break

        if failed:
            free(buf)
            raise MemoryError()

        pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
        free(buf)
        return pairs

    cpdef list find_pairs_full_with_dist(self,
                                         long[:] recent_entry_ids,
                                         long[:] win_sid_idx,
                                         long[:] win_time,
                                         double tau):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t n_win = win_sid_idx.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef long sid_r, time_r
        cdef int64_t *buf = <int64_t *>malloc(1024 * 2 * sizeof(int64_t))
        cdef double *dist_buf = <double *>malloc(1024 * sizeof(double))
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t cap = 1024
        cdef bint failed = False
        cdef Py_ssize_t range_hits = 0
        cdef Py_ssize_t distance_checks = 0
        cdef Py_ssize_t after_similarity = 0

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors

        if buf == NULL or dist_buf == NULL:
            if buf != NULL:
                free(buf)
            if dist_buf != NULL:
                free(dist_buf)
            raise MemoryError()

        n_dim = self._n_vectors
        if (
            tau < 0.0
            or n_recent == 0
            or self._root < 0
            or n_dim <= 0
            or n_win == 0
            or win_time.shape[0] != n_win
        ):
            free(buf)
            free(dist_buf)
            return []

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        active = self._active_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in range(n_recent):
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0 or has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                if ridx < 0 or ridx >= n_win:
                    continue
                sid_r = win_sid_idx[ridx]
                time_r = win_time[ridx]
                value = values[node]
                lower = value - tau
                upper = value + tau
                _scan_tree_full_dist(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    active,
                    has_vector,
                    vectors,
                    lower,
                    upper,
                    node,
                    ridx,
                    sid_r,
                    time_r,
                    win_sid_idx,
                    win_time,
                    n_dim,
                    tau_sq,
                    &buf,
                    &dist_buf,
                    &count,
                    &cap,
                    &failed,
                    &range_hits,
                    &distance_checks,
                    &after_similarity,
                )
                if failed:
                    break

        if failed:
            free(buf)
            free(dist_buf)
            raise MemoryError()

        pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
        free(buf)
        free(dist_buf)
        return pairs

    cpdef list find_pairs_full_parallel(self,
                                        long[:] recent_entry_ids,
                                        long[:] win_sid_idx,
                                        long[:] win_time,
                                        double tau,
                                        int n_workers):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t n_win = win_sid_idx.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int n_threads = n_workers if n_workers > 1 else 1
        if n_threads > n_recent:
            n_threads = <int>n_recent
        cdef int t, tid
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef long sid_r, time_r
        cdef int64_t **bufs
        cdef Py_ssize_t *counts
        cdef Py_ssize_t *caps
        cdef Py_ssize_t *range_hits
        cdef Py_ssize_t *distance_checks
        cdef Py_ssize_t *after_similarity
        cdef bint *failed
        cdef Py_ssize_t total = 0

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors

        n_dim = self._n_vectors
        if (
            tau < 0.0
            or n_recent == 0
            or self._root < 0
            or n_dim <= 0
            or n_win == 0
            or win_time.shape[0] != n_win
        ):
            return []

        bufs = <int64_t **>malloc(n_threads * sizeof(int64_t *))
        counts = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        caps = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        failed = <bint *>malloc(n_threads * sizeof(bint))
        if bufs == NULL or counts == NULL or caps == NULL or failed == NULL:
            if bufs != NULL:
                free(bufs)
            if counts != NULL:
                free(counts)
            if caps != NULL:
                free(caps)
            if failed != NULL:
                free(failed)
            raise MemoryError()

        for t in range(n_threads):
            caps[t] = 1024
            counts[t] = 0
            failed[t] = False
            bufs[t] = <int64_t *>malloc(caps[t] * 2 * sizeof(int64_t))
            if bufs[t] == NULL:
                failed[t] = True

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        active = self._active_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in prange(n_recent, schedule='static', num_threads=n_threads):
                tid = threadid()
                if failed[tid]:
                    continue
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0 or has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                if ridx < 0 or ridx >= n_win:
                    continue
                sid_r = win_sid_idx[ridx]
                time_r = win_time[ridx]
                value = values[node]
                lower = value - tau
                upper = value + tau
                _scan_tree_full(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    active,
                    has_vector,
                    vectors,
                    lower,
                    upper,
                    node,
                    ridx,
                    sid_r,
                    time_r,
                    win_sid_idx,
                    win_time,
                    n_dim,
                    tau_sq,
                    &bufs[tid],
                    &counts[tid],
                    &caps[tid],
                    &failed[tid],
                )

        for t in range(n_threads):
            if failed[t]:
                for tid in range(n_threads):
                    if bufs[tid] != NULL:
                        free(bufs[tid])
                free(bufs)
                free(counts)
                free(caps)
                free(failed)
                raise MemoryError()
            total += counts[t]

        pairs = []
        if total > 0:
            pairs = [None] * total
            total = 0
            for t in range(n_threads):
                for i in range(counts[t]):
                    pairs[total] = (int(bufs[t][2 * i]), int(bufs[t][2 * i + 1]))
                    total += 1
                free(bufs[t])
        else:
            for t in range(n_threads):
                free(bufs[t])
        free(bufs)
        free(counts)
        free(caps)
        free(failed)
        return pairs

    cpdef list find_pairs_full_with_dist_parallel(self,
                                                  long[:] recent_entry_ids,
                                                  long[:] win_sid_idx,
                                                  long[:] win_time,
                                                  double tau,
                                                  int n_workers):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t n_win = win_sid_idx.shape[0]
        cdef Py_ssize_t size = self._size
        cdef Py_ssize_t i, n_dim
        cdef int n_threads = n_workers if n_workers > 1 else 1
        if n_threads > n_recent:
            n_threads = <int>n_recent
        cdef int t, tid
        cdef int64_t node, root_node, ridx
        cdef double lower, upper, value, tau_sq
        cdef long sid_r, time_r
        cdef int64_t **bufs
        cdef double **dist_bufs
        cdef Py_ssize_t *counts
        cdef Py_ssize_t *caps
        cdef bint *failed
        cdef Py_ssize_t total = 0

        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] window_idx
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors

        n_dim = self._n_vectors
        if (
            tau < 0.0
            or n_recent == 0
            or self._root < 0
            or n_dim <= 0
            or n_win == 0
            or win_time.shape[0] != n_win
        ):
            return []

        bufs = <int64_t **>malloc(n_threads * sizeof(int64_t *))
        dist_bufs = <double **>malloc(n_threads * sizeof(double *))
        counts = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        caps = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        range_hits = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        distance_checks = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        after_similarity = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        failed = <bint *>malloc(n_threads * sizeof(bint))
        if bufs == NULL or dist_bufs == NULL or counts == NULL or caps == NULL or range_hits == NULL or distance_checks == NULL or after_similarity == NULL or failed == NULL:
            if bufs != NULL:
                free(bufs)
            if dist_bufs != NULL:
                free(dist_bufs)
            if counts != NULL:
                free(counts)
            if caps != NULL:
                free(caps)
            if range_hits != NULL:
                free(range_hits)
            if distance_checks != NULL:
                free(distance_checks)
            if after_similarity != NULL:
                free(after_similarity)
            if failed != NULL:
                free(failed)
            raise MemoryError()

        for t in range(n_threads):
            caps[t] = 1024
            counts[t] = 0
            range_hits[t] = 0
            distance_checks[t] = 0
            after_similarity[t] = 0
            failed[t] = False
            bufs[t] = <int64_t *>malloc(caps[t] * 2 * sizeof(int64_t))
            dist_bufs[t] = <double *>malloc(caps[t] * sizeof(double))
            if bufs[t] == NULL or dist_bufs[t] == NULL:
                failed[t] = True

        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        window_idx = self._window_idx_arr
        active = self._active_arr
        has_vector = self._has_vector_arr
        vectors = self._vectors_arr
        root_node = self._root
        tau_sq = tau * tau

        with nogil:
            for i in prange(n_recent, schedule='static', num_threads=n_threads):
                tid = threadid()
                if failed[tid]:
                    continue
                node = <int64_t>recent_entry_ids[i]
                if node < 0 or node >= size:
                    continue
                if active[node] == 0 or has_vector[node] == 0:
                    continue
                ridx = window_idx[node]
                if ridx < 0 or ridx >= n_win:
                    continue
                sid_r = win_sid_idx[ridx]
                time_r = win_time[ridx]
                value = values[node]
                lower = value - tau
                upper = value + tau
                _scan_tree_full_dist(
                    root_node,
                    left,
                    right,
                    values,
                    window_idx,
                    active,
                    has_vector,
                    vectors,
                    lower,
                    upper,
                    node,
                    ridx,
                    sid_r,
                    time_r,
                    win_sid_idx,
                    win_time,
                    n_dim,
                    tau_sq,
                    &bufs[tid],
                    &dist_bufs[tid],
                    &counts[tid],
                    &caps[tid],
                    &failed[tid],
                    &range_hits[tid],
                    &distance_checks[tid],
                    &after_similarity[tid],
                )

        for t in range(n_threads):
            if failed[t]:
                for tid in range(n_threads):
                    if bufs[tid] != NULL:
                        free(bufs[tid])
                    if dist_bufs[tid] != NULL:
                        free(dist_bufs[tid])
                free(bufs)
                free(dist_bufs)
                free(counts)
                free(caps)
                free(range_hits)
                free(distance_checks)
                free(after_similarity)
                free(failed)
                raise MemoryError()
            total += counts[t]

        pairs = []
        if total > 0:
            pairs = [None] * total
            total = 0
            for t in range(n_threads):
                for i in range(counts[t]):
                    pairs[total] = (int(bufs[t][2 * i]), int(bufs[t][2 * i + 1]), float(dist_bufs[t][i]))
                    total += 1
                free(bufs[t])
                free(dist_bufs[t])
        else:
            for t in range(n_threads):
                free(bufs[t])
                free(dist_bufs[t])
        free(bufs)
        free(dist_bufs)
        free(counts)
        free(caps)
        free(range_hits)
        free(distance_checks)
        free(after_similarity)
        free(failed)
        return pairs


def find_candidate_pairs(double[:] values,
                         long[:] value_window_idx,
                         double[:] recent_values,
                         long[:] recent_window_idx,
                         long[:] win_sid_idx,
                         long[:] win_time,
                         double tau):
    """Return list of (window_idx, other_idx) pairs within +/- tau range."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t i, j, left, right
    cdef double val, lower, upper
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef bint failed = False
    if buf == NULL:
        raise MemoryError()

    if n_recent == 0 or n_values == 0:
        free(buf)
        return []

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                if count >= cap:
                    cap = cap * 2
                    buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                    if buf == NULL:
                        failed = True
                        break
                buf[2 * count] = <int64_t>ridx
                buf[2 * count + 1] = <int64_t>other_idx
                count += 1
            if failed:
                break

    if failed:
        free(buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
    free(buf)
    return pairs


def find_candidate_pairs_with_dist(double[:] values,
                                   long[:] value_window_idx,
                                   double[:] recent_values,
                                   long[:] recent_window_idx,
                                   long[:] win_sid_idx,
                                   long[:] win_time,
                                   double tau):
    """Return list of (window_idx, other_idx, dist_sq) pairs within +/- tau."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t i, j, left, right
    cdef double val, lower, upper, diff
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef double *dist_buf = <double *>malloc(cap * sizeof(double))
    cdef bint failed = False
    if buf == NULL or dist_buf == NULL:
        if buf != NULL:
            free(buf)
        if dist_buf != NULL:
            free(dist_buf)
        raise MemoryError()

    if n_recent == 0 or n_values == 0:
        free(buf)
        free(dist_buf)
        return []

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                if count >= cap:
                    cap = cap * 2
                    buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                    dist_buf = <double *>realloc(dist_buf, cap * sizeof(double))
                    if buf == NULL or dist_buf == NULL:
                        failed = True
                        break
                diff = val - values[j]
                buf[2 * count] = <int64_t>ridx
                buf[2 * count + 1] = <int64_t>other_idx
                dist_buf[count] = diff * diff
                count += 1
            if failed:
                break

    if failed:
        free(buf)
        free(dist_buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
    free(buf)
    free(dist_buf)
    return pairs


def find_candidate_pairs_full(double[:] values,
                              long[:] value_window_idx,
                              double[:] recent_values,
                              long[:] recent_window_idx,
                              long[:] win_sid_idx,
                              long[:] win_time,
                              double[:, :] entry_vectors,
                              double[:, :] recent_vectors,
                              double tau):
    """Return list of (window_idx, other_idx) pairs within +/- tau and L2 check."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if entry_vectors.shape[0] != n_values:
        raise ValueError("entry_vectors must align with values")
    if recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent_vectors must align with recent_values")
    cdef Py_ssize_t i, j, d, left, right
    cdef double val, lower, upper
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef double tau_sq = tau * tau
    cdef double diff, acc
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef bint failed = False
    if buf == NULL:
        raise MemoryError()

    if n_recent == 0 or n_values == 0 or n_dim == 0 or tau < 0.0:
        free(buf)
        return []

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                acc = 0.0
                for d in range(n_dim):
                    diff = recent_vectors[i, d] - entry_vectors[j, d]
                    acc += diff * diff
                    if acc > tau_sq:
                        break
                if acc > tau_sq:
                    continue
                if count >= cap:
                    cap = cap * 2
                    buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                    if buf == NULL:
                        failed = True
                        break
                buf[2 * count] = <int64_t>ridx
                buf[2 * count + 1] = <int64_t>other_idx
                count += 1
            if failed:
                break

    if failed:
        free(buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
    free(buf)
    return pairs


def find_candidate_pairs_full_with_dist(double[:] values,
                                        long[:] value_window_idx,
                                        double[:] recent_values,
                                        long[:] recent_window_idx,
                                        long[:] win_sid_idx,
                                        long[:] win_time,
                                        double[:, :] entry_vectors,
                                        double[:, :] recent_vectors,
                                        double tau):
    """Return list of (window_idx, other_idx, dist_sq) pairs within +/- tau and L2 check."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if entry_vectors.shape[0] != n_values:
        raise ValueError("entry_vectors must align with values")
    if recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent_vectors must align with recent_values")
    cdef Py_ssize_t i, j, d, left, right
    cdef double val, lower, upper
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef double tau_sq = tau * tau
    cdef double diff, acc
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef double *dist_buf = <double *>malloc(cap * sizeof(double))
    cdef bint failed = False
    if buf == NULL or dist_buf == NULL:
        if buf != NULL:
            free(buf)
        if dist_buf != NULL:
            free(dist_buf)
        raise MemoryError()

    if n_recent == 0 or n_values == 0 or n_dim == 0 or tau < 0.0:
        free(buf)
        free(dist_buf)
        return []

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                acc = 0.0
                for d in range(n_dim):
                    diff = recent_vectors[i, d] - entry_vectors[j, d]
                    acc += diff * diff
                    if acc > tau_sq:
                        break
                if acc > tau_sq:
                    continue
                if count >= cap:
                    cap = cap * 2
                    buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                    dist_buf = <double *>realloc(dist_buf, cap * sizeof(double))
                    if buf == NULL or dist_buf == NULL:
                        failed = True
                        break
                buf[2 * count] = <int64_t>ridx
                buf[2 * count + 1] = <int64_t>other_idx
                dist_buf[count] = acc
                count += 1
            if failed:
                break

    if failed:
        free(buf)
        free(dist_buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
    free(buf)
    free(dist_buf)
    return pairs


def find_candidate_pairs_bucketed(long[:] bucket_ids,
                                  long[:] bucket_starts,
                                  long[:] bucket_ends,
                                  double[:] values,
                                  long[:] value_window_idx,
                                  double[:] recent_values,
                                  long[:] recent_window_idx,
                                  long[:] win_sid_idx,
                                  long[:] win_time,
                                  long[:] win_w,
                                  double bucket_width,
                                  double tau):
    """Return exact scalar candidate pairs using pre-bucketed sorted entries."""
    cdef Py_ssize_t n_buckets = bucket_ids.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    if bucket_starts.shape[0] != n_buckets or bucket_ends.shape[0] != n_buckets:
        raise ValueError("bucket arrays must have matching lengths")
    if value_window_idx.shape[0] != n_values:
        raise ValueError("value_window_idx must align with values")
    if recent_window_idx.shape[0] != n_recent:
        raise ValueError("recent_window_idx must align with recent_values")
    if bucket_width <= 0.0 or tau < 0.0 or n_buckets == 0 or n_values == 0 or n_recent == 0:
        return []

    cdef Py_ssize_t i, bpos, j, left, right
    cdef long first_bucket, last_bucket, bucket_id
    cdef double val, lower, upper, diff
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o, w_r
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef bint failed = False
    if buf == NULL:
        raise MemoryError()

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            first_bucket = <long>floor(lower / bucket_width)
            last_bucket = <long>floor(upper / bucket_width)
            bpos = _bisect_left_long(bucket_ids, first_bucket)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            w_r = win_w[ridx]
            while bpos < n_buckets and bucket_ids[bpos] <= last_bucket:
                bucket_id = bucket_ids[bpos]
                left = _bisect_left_range(values, <Py_ssize_t>bucket_starts[bpos], <Py_ssize_t>bucket_ends[bpos], lower)
                right = _bisect_right_range(values, left, <Py_ssize_t>bucket_ends[bpos], upper)
                for j in range(left, right):
                    other_idx = <Py_ssize_t>value_window_idx[j]
                    if other_idx == ridx:
                        continue
                    if win_w[other_idx] != w_r:
                        continue
                    sid_o = win_sid_idx[other_idx]
                    time_o = win_time[other_idx]
                    if sid_o == sid_r and time_o == time_r:
                        continue
                    if count >= cap:
                        cap = cap * 2
                        buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                        if buf == NULL:
                            failed = True
                            break
                    buf[2 * count] = <int64_t>ridx
                    buf[2 * count + 1] = <int64_t>other_idx
                    count += 1
                if failed:
                    break
                bpos += 1
            if failed:
                break

    if failed:
        free(buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
    free(buf)
    return pairs


def find_candidate_pairs_bucketed_with_dist(long[:] bucket_ids,
                                            long[:] bucket_starts,
                                            long[:] bucket_ends,
                                            double[:] values,
                                            long[:] value_window_idx,
                                            double[:] recent_values,
                                            long[:] recent_window_idx,
                                            long[:] win_sid_idx,
                                            long[:] win_time,
                                            long[:] win_w,
                                            double bucket_width,
                                            double tau):
    """Return exact scalar candidate pairs and squared distances using buckets."""
    cdef Py_ssize_t n_buckets = bucket_ids.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    if bucket_starts.shape[0] != n_buckets or bucket_ends.shape[0] != n_buckets:
        raise ValueError("bucket arrays must have matching lengths")
    if value_window_idx.shape[0] != n_values:
        raise ValueError("value_window_idx must align with values")
    if recent_window_idx.shape[0] != n_recent:
        raise ValueError("recent_window_idx must align with recent_values")
    if bucket_width <= 0.0 or tau < 0.0 or n_buckets == 0 or n_values == 0 or n_recent == 0:
        return []

    cdef Py_ssize_t i, bpos, j, left, right
    cdef long first_bucket, last_bucket
    cdef double val, lower, upper, diff
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o, w_r
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef double *dist_buf = <double *>malloc(cap * sizeof(double))
    cdef bint failed = False
    if buf == NULL or dist_buf == NULL:
        if buf != NULL:
            free(buf)
        if dist_buf != NULL:
            free(dist_buf)
        raise MemoryError()

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            first_bucket = <long>floor(lower / bucket_width)
            last_bucket = <long>floor(upper / bucket_width)
            bpos = _bisect_left_long(bucket_ids, first_bucket)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            w_r = win_w[ridx]
            while bpos < n_buckets and bucket_ids[bpos] <= last_bucket:
                left = _bisect_left_range(values, <Py_ssize_t>bucket_starts[bpos], <Py_ssize_t>bucket_ends[bpos], lower)
                right = _bisect_right_range(values, left, <Py_ssize_t>bucket_ends[bpos], upper)
                for j in range(left, right):
                    other_idx = <Py_ssize_t>value_window_idx[j]
                    if other_idx == ridx:
                        continue
                    if win_w[other_idx] != w_r:
                        continue
                    sid_o = win_sid_idx[other_idx]
                    time_o = win_time[other_idx]
                    if sid_o == sid_r and time_o == time_r:
                        continue
                    if count >= cap:
                        cap = cap * 2
                        buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                        dist_buf = <double *>realloc(dist_buf, cap * sizeof(double))
                        if buf == NULL or dist_buf == NULL:
                            failed = True
                            break
                    diff = val - values[j]
                    buf[2 * count] = <int64_t>ridx
                    buf[2 * count + 1] = <int64_t>other_idx
                    dist_buf[count] = diff * diff
                    count += 1
                if failed:
                    break
                bpos += 1
            if failed:
                break

    if failed:
        free(buf)
        free(dist_buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
    free(buf)
    free(dist_buf)
    return pairs


def find_candidate_pairs_bucketed_full(long[:] bucket_ids,
                                       long[:] bucket_starts,
                                       long[:] bucket_ends,
                                       double[:] values,
                                       long[:] value_window_idx,
                                       double[:] recent_values,
                                       long[:] recent_window_idx,
                                       long[:] win_sid_idx,
                                       long[:] win_time,
                                       long[:] win_w,
                                       double[:, :] entry_vectors,
                                       double[:, :] recent_vectors,
                                       double bucket_width,
                                       double tau):
    """Return exact full-vector candidate pairs using pre-bucketed entries."""
    cdef Py_ssize_t n_buckets = bucket_ids.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if bucket_starts.shape[0] != n_buckets or bucket_ends.shape[0] != n_buckets:
        raise ValueError("bucket arrays must have matching lengths")
    if value_window_idx.shape[0] != n_values or entry_vectors.shape[0] != n_values:
        raise ValueError("entry arrays must align with values")
    if recent_window_idx.shape[0] != n_recent or recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent arrays must align with recent_values")
    if bucket_width <= 0.0 or tau < 0.0 or n_buckets == 0 or n_values == 0 or n_recent == 0 or n_dim == 0:
        return []

    cdef Py_ssize_t i, bpos, j, d, left, right
    cdef long first_bucket, last_bucket
    cdef double val, lower, upper, diff, acc
    cdef double tau_sq = tau * tau
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o, w_r
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef bint failed = False
    if buf == NULL:
        raise MemoryError()

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            first_bucket = <long>floor(lower / bucket_width)
            last_bucket = <long>floor(upper / bucket_width)
            bpos = _bisect_left_long(bucket_ids, first_bucket)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            w_r = win_w[ridx]
            while bpos < n_buckets and bucket_ids[bpos] <= last_bucket:
                left = _bisect_left_range(values, <Py_ssize_t>bucket_starts[bpos], <Py_ssize_t>bucket_ends[bpos], lower)
                right = _bisect_right_range(values, left, <Py_ssize_t>bucket_ends[bpos], upper)
                for j in range(left, right):
                    other_idx = <Py_ssize_t>value_window_idx[j]
                    if other_idx == ridx:
                        continue
                    if win_w[other_idx] != w_r:
                        continue
                    sid_o = win_sid_idx[other_idx]
                    time_o = win_time[other_idx]
                    if sid_o == sid_r and time_o == time_r:
                        continue
                    acc = 0.0
                    for d in range(n_dim):
                        diff = recent_vectors[i, d] - entry_vectors[j, d]
                        acc += diff * diff
                        if acc > tau_sq:
                            break
                    if acc > tau_sq:
                        continue
                    if count >= cap:
                        cap = cap * 2
                        buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                        if buf == NULL:
                            failed = True
                            break
                    buf[2 * count] = <int64_t>ridx
                    buf[2 * count + 1] = <int64_t>other_idx
                    count += 1
                if failed:
                    break
                bpos += 1
            if failed:
                break

    if failed:
        free(buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(count)]
    free(buf)
    return pairs


def find_candidate_pairs_bucketed_full_with_dist(long[:] bucket_ids,
                                                 long[:] bucket_starts,
                                                 long[:] bucket_ends,
                                                 double[:] values,
                                                 long[:] value_window_idx,
                                                 double[:] recent_values,
                                                 long[:] recent_window_idx,
                                                 long[:] win_sid_idx,
                                                 long[:] win_time,
                                                 long[:] win_w,
                                                 double[:, :] entry_vectors,
                                                 double[:, :] recent_vectors,
                                                 double bucket_width,
                                                 double tau):
    """Return exact full-vector candidate pairs and squared distances using buckets."""
    cdef Py_ssize_t n_buckets = bucket_ids.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if bucket_starts.shape[0] != n_buckets or bucket_ends.shape[0] != n_buckets:
        raise ValueError("bucket arrays must have matching lengths")
    if value_window_idx.shape[0] != n_values or entry_vectors.shape[0] != n_values:
        raise ValueError("entry arrays must align with values")
    if recent_window_idx.shape[0] != n_recent or recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent arrays must align with recent_values")
    if bucket_width <= 0.0 or tau < 0.0 or n_buckets == 0 or n_values == 0 or n_recent == 0 or n_dim == 0:
        return []

    cdef Py_ssize_t i, bpos, j, d, left, right
    cdef long first_bucket, last_bucket
    cdef double val, lower, upper, diff, acc
    cdef double tau_sq = tau * tau
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o, w_r
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t cap = 1024
    cdef int64_t *buf = <int64_t *>malloc(cap * 2 * sizeof(int64_t))
    cdef double *dist_buf = <double *>malloc(cap * sizeof(double))
    cdef bint failed = False
    if buf == NULL or dist_buf == NULL:
        if buf != NULL:
            free(buf)
        if dist_buf != NULL:
            free(dist_buf)
        raise MemoryError()

    with nogil:
        for i in range(n_recent):
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            first_bucket = <long>floor(lower / bucket_width)
            last_bucket = <long>floor(upper / bucket_width)
            bpos = _bisect_left_long(bucket_ids, first_bucket)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            w_r = win_w[ridx]
            while bpos < n_buckets and bucket_ids[bpos] <= last_bucket:
                left = _bisect_left_range(values, <Py_ssize_t>bucket_starts[bpos], <Py_ssize_t>bucket_ends[bpos], lower)
                right = _bisect_right_range(values, left, <Py_ssize_t>bucket_ends[bpos], upper)
                for j in range(left, right):
                    other_idx = <Py_ssize_t>value_window_idx[j]
                    if other_idx == ridx:
                        continue
                    if win_w[other_idx] != w_r:
                        continue
                    sid_o = win_sid_idx[other_idx]
                    time_o = win_time[other_idx]
                    if sid_o == sid_r and time_o == time_r:
                        continue
                    acc = 0.0
                    for d in range(n_dim):
                        diff = recent_vectors[i, d] - entry_vectors[j, d]
                        acc += diff * diff
                        if acc > tau_sq:
                            break
                    if acc > tau_sq:
                        continue
                    if count >= cap:
                        cap = cap * 2
                        buf = <int64_t *>realloc(buf, cap * 2 * sizeof(int64_t))
                        dist_buf = <double *>realloc(dist_buf, cap * sizeof(double))
                        if buf == NULL or dist_buf == NULL:
                            failed = True
                            break
                    buf[2 * count] = <int64_t>ridx
                    buf[2 * count + 1] = <int64_t>other_idx
                    dist_buf[count] = acc
                    count += 1
                if failed:
                    break
                bpos += 1
            if failed:
                break

    if failed:
        free(buf)
        free(dist_buf)
        raise MemoryError()

    pairs = [(int(buf[2 * i]), int(buf[2 * i + 1]), float(dist_buf[i])) for i in range(count)]
    free(buf)
    free(dist_buf)
    return pairs


def find_candidate_pairs_full_parallel(double[:] values,
                                       long[:] value_window_idx,
                                       double[:] recent_values,
                                       long[:] recent_window_idx,
                                       long[:] win_sid_idx,
                                       long[:] win_time,
                                       double[:, :] entry_vectors,
                                       double[:, :] recent_vectors,
                                       double tau,
                                       int n_workers):
    """Parallel OpenMP version of find_candidate_pairs_full."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if entry_vectors.shape[0] != n_values:
        raise ValueError("entry_vectors must align with values")
    if recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent_vectors must align with recent_values")
    if n_recent == 0 or n_values == 0 or n_dim == 0 or tau < 0.0:
        return []

    cdef int n_threads = n_workers if n_workers > 1 else 1
    if n_threads > n_recent:
        n_threads = <int>n_recent
    cdef int t, tid
    cdef Py_ssize_t i, j, d, left, right
    cdef double val, lower, upper, diff, acc
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef double tau_sq = tau * tau
    cdef int64_t **bufs = <int64_t **>malloc(n_threads * sizeof(int64_t *))
    cdef Py_ssize_t *counts = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
    cdef Py_ssize_t *caps = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
    cdef bint *failed = <bint *>malloc(n_threads * sizeof(bint))
    cdef Py_ssize_t total = 0

    if bufs == NULL or counts == NULL or caps == NULL or failed == NULL:
        if bufs != NULL:
            free(bufs)
        if counts != NULL:
            free(counts)
        if caps != NULL:
            free(caps)
        if failed != NULL:
            free(failed)
        raise MemoryError()

    for t in range(n_threads):
        caps[t] = 1024
        counts[t] = 0
        failed[t] = False
        bufs[t] = <int64_t *>malloc(caps[t] * 2 * sizeof(int64_t))
        if bufs[t] == NULL:
            failed[t] = True

    with nogil:
        for i in prange(n_recent, schedule='static', num_threads=n_threads):
            tid = threadid()
            if failed[tid]:
                continue
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                acc = _row_l2_sq_until(recent_vectors, entry_vectors, i, j, n_dim, tau_sq)
                if acc > tau_sq:
                    continue
                if not _append_pair(&bufs[tid], &counts[tid], &caps[tid], <int64_t>ridx, <int64_t>other_idx):
                    failed[tid] = True
                    break

    for t in range(n_threads):
        if failed[t]:
            for tid in range(n_threads):
                if bufs[tid] != NULL:
                    free(bufs[tid])
            free(bufs)
            free(counts)
            free(caps)
            free(failed)
            raise MemoryError()
        total += counts[t]

    pairs = []
    if total > 0:
        pairs = [None] * total
        total = 0
        for t in range(n_threads):
            for i in range(counts[t]):
                pairs[total] = (int(bufs[t][2 * i]), int(bufs[t][2 * i + 1]))
                total += 1
            free(bufs[t])
    else:
        for t in range(n_threads):
            free(bufs[t])
    free(bufs)
    free(counts)
    free(caps)
    free(failed)
    return pairs


def find_candidate_pairs_full_with_dist_parallel(double[:] values,
                                                 long[:] value_window_idx,
                                                 double[:] recent_values,
                                                 long[:] recent_window_idx,
                                                 long[:] win_sid_idx,
                                                 long[:] win_time,
                                                 double[:, :] entry_vectors,
                                                 double[:, :] recent_vectors,
                                                 double tau,
                                                 int n_workers):
    """Parallel OpenMP version of find_candidate_pairs_full_with_dist."""
    cdef Py_ssize_t n_recent = recent_values.shape[0]
    cdef Py_ssize_t n_values = values.shape[0]
    cdef Py_ssize_t n_dim = entry_vectors.shape[1]
    if entry_vectors.shape[0] != n_values:
        raise ValueError("entry_vectors must align with values")
    if recent_vectors.shape[0] != n_recent or recent_vectors.shape[1] != n_dim:
        raise ValueError("recent_vectors must align with recent_values")
    if n_recent == 0 or n_values == 0 or n_dim == 0 or tau < 0.0:
        return []

    cdef int n_threads = n_workers if n_workers > 1 else 1
    if n_threads > n_recent:
        n_threads = <int>n_recent
    cdef int t, tid
    cdef Py_ssize_t i, j, d, left, right
    cdef double val, lower, upper, diff, acc
    cdef Py_ssize_t ridx, other_idx
    cdef long sid_r, sid_o, time_r, time_o
    cdef double tau_sq = tau * tau
    cdef int64_t **bufs = <int64_t **>malloc(n_threads * sizeof(int64_t *))
    cdef double **dist_bufs = <double **>malloc(n_threads * sizeof(double *))
    cdef Py_ssize_t *counts = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
    cdef Py_ssize_t *caps = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
    cdef bint *failed = <bint *>malloc(n_threads * sizeof(bint))
    cdef Py_ssize_t total = 0

    if bufs == NULL or dist_bufs == NULL or counts == NULL or caps == NULL or failed == NULL:
        if bufs != NULL:
            free(bufs)
        if dist_bufs != NULL:
            free(dist_bufs)
        if counts != NULL:
            free(counts)
        if caps != NULL:
            free(caps)
        if failed != NULL:
            free(failed)
        raise MemoryError()

    for t in range(n_threads):
        caps[t] = 1024
        counts[t] = 0
        failed[t] = False
        bufs[t] = <int64_t *>malloc(caps[t] * 2 * sizeof(int64_t))
        dist_bufs[t] = <double *>malloc(caps[t] * sizeof(double))
        if bufs[t] == NULL or dist_bufs[t] == NULL:
            failed[t] = True

    with nogil:
        for i in prange(n_recent, schedule='static', num_threads=n_threads):
            tid = threadid()
            if failed[tid]:
                continue
            val = recent_values[i]
            ridx = <Py_ssize_t>recent_window_idx[i]
            lower = val - tau
            upper = val + tau
            left = _bisect_left(values, lower)
            right = _bisect_right(values, upper)
            sid_r = win_sid_idx[ridx]
            time_r = win_time[ridx]
            for j in range(left, right):
                other_idx = <Py_ssize_t>value_window_idx[j]
                if other_idx == ridx:
                    continue
                sid_o = win_sid_idx[other_idx]
                time_o = win_time[other_idx]
                if sid_o == sid_r and time_o == time_r:
                    continue
                acc = _row_l2_sq_until(recent_vectors, entry_vectors, i, j, n_dim, tau_sq)
                if acc > tau_sq:
                    continue
                if not _append_pair_dist(&bufs[tid], &dist_bufs[tid], &counts[tid], &caps[tid], <int64_t>ridx, <int64_t>other_idx, acc):
                    failed[tid] = True
                    break

    for t in range(n_threads):
        if failed[t]:
            for tid in range(n_threads):
                if bufs[tid] != NULL:
                    free(bufs[tid])
                if dist_bufs[tid] != NULL:
                    free(dist_bufs[tid])
            free(bufs)
            free(dist_bufs)
            free(counts)
            free(caps)
            free(failed)
            raise MemoryError()
        total += counts[t]

    pairs = []
    if total > 0:
        pairs = [None] * total
        total = 0
        for t in range(n_threads):
            for i in range(counts[t]):
                pairs[total] = (int(bufs[t][2 * i]), int(bufs[t][2 * i + 1]), float(dist_bufs[t][i]))
                total += 1
            free(bufs[t])
            free(dist_bufs[t])
    else:
        for t in range(n_threads):
            free(bufs[t])
            free(dist_bufs[t])
    free(bufs)
    free(dist_bufs)
    free(counts)
    free(caps)
    free(failed)
    return pairs


def enumerate_candidate_rows(double[:, ::1] data,
                             long[:] window_index,
                             long[:] ref_indices,
                             int window_size,
                             int window_step,
                             double std_thresh=1e-3,
                             long shard_start=-1,
                             long shard_end=-1):
    cdef Py_ssize_t n_series = data.shape[0]
    cdef Py_ssize_t n_cols = data.shape[1]
    cdef Py_ssize_t window_count = n_cols - window_size + 1
    if window_size <= 0 or window_count <= 0:
        return None
    if ref_indices.shape[0] == 0:
        return None

    cdef int step = window_step if window_step > 0 else 1
    cdef Py_ssize_t last_idx = window_count - 1
    if last_idx % step != 0:
        return None

    cdef double threshold = (std_thresh * std_thresh) * window_size
    cdef Py_ssize_t step_count = (window_count + step - 1) // step
    cdef Py_ssize_t cap = n_series * step_count
    if cap <= 0:
        return None

    cdef int64_t *valid_k = <int64_t *>malloc(cap * sizeof(int64_t))
    cdef int64_t *valid_j = <int64_t *>malloc(cap * sizeof(int64_t))
    cdef unsigned char *seeds = <unsigned char *>malloc(n_series * sizeof(unsigned char))
    if valid_k == NULL or valid_j == NULL or seeds == NULL:
        if valid_k != NULL:
            free(valid_k)
        if valid_j != NULL:
            free(valid_j)
        if seeds != NULL:
            free(seeds)
        raise MemoryError()

    cdef Py_ssize_t s, j
    cdef double sum_val, sum_sq, val, outgoing, incoming, var_sum
    cdef Py_ssize_t count = 0
    for s in range(n_series):
        seeds[s] = 0
        sum_val = 0.0
        sum_sq = 0.0
        for j in range(window_size):
            val = data[s, j]
            sum_val += val
            sum_sq += val * val
        for j in range(window_count):
            if j > 0:
                outgoing = data[s, j - 1]
                incoming = data[s, j + window_size - 1]
                sum_val += incoming - outgoing
                sum_sq += incoming * incoming - outgoing * outgoing
            var_sum = sum_sq - (sum_val * sum_val) / window_size
            if var_sum < 0.0:
                var_sum = 0.0
            if (j % step) == 0 and var_sum > threshold:
                if count < cap:
                    valid_k[count] = <int64_t>s
                    valid_j[count] = <int64_t>j
                    count += 1
            if j == last_idx and var_sum > threshold:
                seeds[s] = 1

    if count == 0:
        free(valid_k)
        free(valid_j)
        free(seeds)
        return None

    cdef bint has_seed = False
    cdef long s_idx
    for j in range(ref_indices.shape[0]):
        s_idx = ref_indices[j]
        if s_idx < 0 or s_idx >= n_series:
            continue
        if seeds[s_idx] != 0:
            has_seed = True
            break
    if not has_seed:
        free(valid_k)
        free(valid_j)
        free(seeds)
        return None

    cdef Py_ssize_t out_cap = 1024
    if out_cap < count:
        out_cap = count
    cdef int64_t *out_buf = <int64_t *>malloc(out_cap * 5 * sizeof(int64_t))
    if out_buf == NULL:
        free(valid_k)
        free(valid_j)
        free(seeds)
        raise MemoryError()

    cdef Py_ssize_t out_count = 0
    cdef long curr_start = window_index[last_idx]
    cdef int64_t k_idx, j_idx
    cdef bint apply_shard = shard_start >= 0 and shard_end >= 0
    for j in range(ref_indices.shape[0]):
        s_idx = ref_indices[j]
        if s_idx < 0 or s_idx >= n_series:
            continue
        if seeds[s_idx] == 0:
            continue
        for j_idx in range(count):
            k_idx = valid_k[j_idx]
            if valid_j[j_idx] == last_idx and k_idx <= s_idx:
                continue
            if apply_shard:
                if k_idx < shard_start or k_idx >= shard_end:
                    continue
            if out_count >= out_cap:
                out_cap = out_cap * 2
                out_buf = <int64_t *>realloc(out_buf, out_cap * 5 * sizeof(int64_t))
                if out_buf == NULL:
                    free(valid_k)
                    free(valid_j)
                    free(seeds)
                    raise MemoryError()
            out_buf[5 * out_count] = <int64_t>s_idx
            out_buf[5 * out_count + 1] = k_idx
            out_buf[5 * out_count + 2] = <int64_t>curr_start
            out_buf[5 * out_count + 3] = <int64_t>window_index[valid_j[j_idx]]
            out_buf[5 * out_count + 4] = <int64_t>window_size
            out_count += 1

    free(valid_k)
    free(valid_j)
    free(seeds)

    if out_count == 0:
        free(out_buf)
        return None

    cdef np.ndarray[np.int64_t, ndim=2] out = np.empty((out_count, 5), dtype=np.int64)
    for j in range(out_count):
        out[j, 0] = out_buf[5 * j]
        out[j, 1] = out_buf[5 * j + 1]
        out[j, 2] = out_buf[5 * j + 2]
        out[j, 3] = out_buf[5 * j + 3]
        out[j, 4] = out_buf[5 * j + 4]

    free(out_buf)
    return out



def fast_corr_and_dist(double[:] x, double[:] y):
    """Compute Pearson correlation and Euclidean distance for two vectors."""
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        return float("nan"), float("inf"), (0, 0.0, 0.0, 0.0, 0.0)
    cdef Py_ssize_t i
    cdef double sx = 0.0
    cdef double sy = 0.0
    cdef double sum_xy = 0.0
    cdef double sum_xx = 0.0
    cdef double sum_yy = 0.0
    cdef double xi, yi
    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sy += yi
        sum_xy += xi * yi
        sum_xx += xi * xi
        sum_yy += yi * yi
    cdef double mean_x = sx / n
    cdef double mean_y = sy / n
    cdef double var_x = sum_xx - (sx * sx) / n
    cdef double var_y = sum_yy - (sy * sy) / n
    if var_x < 0.0:
        var_x = 0.0
    if var_y < 0.0:
        var_y = 0.0
    cdef double denom = sqrt(var_x * var_y)
    cdef double corr
    if denom == 0.0:
        corr = float("nan")
    else:
        corr = (sum_xy - (sx * sy) / n) / denom
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
    cdef double dist_sq = sum_xx + sum_yy - 2.0 * sum_xy
    if dist_sq < 0.0:
        dist_sq = 0.0
    cdef double dist = sqrt(dist_sq)
    return corr, dist, (n, mean_x, mean_y, var_x, var_y)


def validate_corr_batch(double[:, :] x,
                        double[:, :] y,
                        double corr_threshold,
                        bint neg_corr,
                        double std_thresh=1e-3,
                        double kurt_thresh=5.0,
                        double[:] current_window_sums=None,
                        double[:] current_window_sums_sq=None,
                        double[:] current_window_sums_cu=None,
                        double[:] current_window_sums_qu=None,
                        long[:] x_series=None,
                        long[:] y_series=None,
                        uint8_t[:] x_is_current=None,
                        uint8_t[:] y_is_current=None):
    """Validate batches of x/y vectors; return list of tuples."""
    cdef Py_ssize_t n_items = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    if y.shape[0] != n_items or y.shape[1] != n:
        raise ValueError("x and y must have the same shape")

    cdef Py_ssize_t i, j
    cdef double sx, sy, sum_xy, sum_xx, sum_yy
    cdef double sum_xxx, sum_yyy, sum_xxxx, sum_yyyy
    cdef double xi, yi
    cdef double mean_x, mean_y, var_x, var_y
    cdef double denom, corr, dist_sq, dist
    cdef double mu4_x, mu4_y, varx, vary, kurt_x, kurt_y
    cdef bint is_const, is_spiked, is_corr
    cdef bint have_current_window_cache = (
        current_window_sums is not None
        and current_window_sums_sq is not None
        and current_window_sums_cu is not None
        and current_window_sums_qu is not None
        and x_series is not None
        and y_series is not None
        and x_is_current is not None
        and y_is_current is not None
    )
    cdef bint use_x_cache, use_y_cache
    cdef long xs, ys

    results = []

    for i in range(n_items):
        if n == 0:
            results.append((False, float("nan"), float("inf"), True, False))
            continue
        # (2026-07-06) "point 1", extended to the default (non-hybrid)
        # validation path: every candidate pair has at least one side
        # anchored at the just-arrived current window, whose sx/sum_xx/
        # sum_xxx/sum_xxxx are already available in an incrementally
        # maintained per-series cache (CorrTrack._cw_raw_sums*). When a
        # side is flagged current by the caller, read its moments from the
        # cache instead of re-accumulating them below; sum_xy always needs
        # a fresh per-row pass regardless. See docs/implementation_log.md.
        use_x_cache = have_current_window_cache and x_is_current[i] != 0
        use_y_cache = have_current_window_cache and y_is_current[i] != 0
        sx = sy = 0.0
        sum_xy = sum_xx = sum_yy = 0.0
        sum_xxx = sum_yyy = 0.0
        sum_xxxx = sum_yyyy = 0.0
        if use_x_cache:
            xs = x_series[i]
            sx = current_window_sums[xs]
            sum_xx = current_window_sums_sq[xs]
            sum_xxx = current_window_sums_cu[xs]
            sum_xxxx = current_window_sums_qu[xs]
        if use_y_cache:
            ys = y_series[i]
            sy = current_window_sums[ys]
            sum_yy = current_window_sums_sq[ys]
            sum_yyy = current_window_sums_cu[ys]
            sum_yyyy = current_window_sums_qu[ys]
        for j in range(n):
            xi = x[i, j]
            yi = y[i, j]
            if not use_x_cache:
                sx += xi
                sum_xx += xi * xi
                sum_xxx += xi * xi * xi
                sum_xxxx += xi * xi * xi * xi
            if not use_y_cache:
                sy += yi
                sum_yy += yi * yi
                sum_yyy += yi * yi * yi
                sum_yyyy += yi * yi * yi * yi
            sum_xy += xi * yi

        mean_x = sx / n
        mean_y = sy / n
        var_x = sum_xx - (sx * sx) / n
        var_y = sum_yy - (sy * sy) / n
        if var_x < 0.0:
            var_x = 0.0
        if var_y < 0.0:
            var_y = 0.0

        is_const = (var_x <= (std_thresh * std_thresh) * n) or (var_y <= (std_thresh * std_thresh) * n)
        if is_const:
            results.append((False, float("nan"), float("inf"), True, False))
            continue

        is_spiked = False
        if n >= 4:
            varx = var_x / n
            if varx > 0.0:
                mu4_x = (
                    sum_xxxx
                    - 4.0 * mean_x * sum_xxx
                    + 6.0 * (mean_x * mean_x) * sum_xx
                    - 4.0 * (mean_x * mean_x * mean_x) * sx
                    + n * (mean_x * mean_x * mean_x * mean_x)
                )
                kurt_x = (mu4_x / n) / (varx * varx) - 3.0
                if kurt_x > kurt_thresh:
                    is_spiked = True
            vary = var_y / n
            if not is_spiked and vary > 0.0:
                mu4_y = (
                    sum_yyyy
                    - 4.0 * mean_y * sum_yyy
                    + 6.0 * (mean_y * mean_y) * sum_yy
                    - 4.0 * (mean_y * mean_y * mean_y) * sy
                    + n * (mean_y * mean_y * mean_y * mean_y)
                )
                kurt_y = (mu4_y / n) / (vary * vary) - 3.0
                if kurt_y > kurt_thresh:
                    is_spiked = True

        if is_spiked:
            results.append((False, float("nan"), float("inf"), False, True))
            continue

        denom = sqrt(var_x * var_y)
        if denom == 0.0:
            corr = float("nan")
        else:
            corr = (sum_xy - (sx * sy) / n) / denom
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0

        dist_sq = sum_xx + sum_yy - 2.0 * sum_xy
        if dist_sq < 0.0:
            dist_sq = 0.0
        dist = sqrt(dist_sq)

        is_corr = False
        if corr == corr:
            if neg_corr:
                is_corr = fabs(corr) >= corr_threshold
            else:
                is_corr = corr >= corr_threshold

        results.append((is_corr, corr, dist, False, False))

    return results


def validate_corr_rows(double[:, :] data,
                       long[:, :] rows,
                       long base_index,
                       double corr_threshold,
                       bint neg_corr,
                       double std_thresh=1e-3,
                       double kurt_thresh=5.0,
                       double[:] current_window_sums=None,
                       double[:] current_window_sums_sq=None,
                       double[:] current_window_sums_cu=None,
                       double[:] current_window_sums_qu=None,
                       long current_window_size=-1):
    """Validate numeric candidate rows [series1, series2, time1, time2, window]."""
    cdef Py_ssize_t n_items = rows.shape[0]
    cdef Py_ssize_t n_series = data.shape[0]
    cdef Py_ssize_t n_cols = data.shape[1]
    if rows.shape[1] < 5:
        raise ValueError("rows must have at least five columns")

    cdef np.ndarray[np.uint8_t, ndim=1] accepted = np.zeros(n_items, dtype=np.uint8)
    cdef np.ndarray[np.float64_t, ndim=1] corrs = np.empty(n_items, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dists = np.empty(n_items, dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=1] constants = np.zeros(n_items, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] spiked = np.zeros(n_items, dtype=np.uint8)

    cdef uint8_t[:] accepted_view = accepted
    cdef double[:] corr_view = corrs
    cdef double[:] dist_view = dists
    cdef uint8_t[:] const_view = constants
    cdef uint8_t[:] spike_view = spiked

    cdef Py_ssize_t i, j
    cdef long s1, s2, t1, t2, w, start1, start2
    cdef double sx, sy, sum_xy, sum_xx, sum_yy
    cdef double sum_xxx, sum_yyy, sum_xxxx, sum_yyyy
    cdef double xi, yi
    cdef double mean_x, mean_y, var_x, var_y
    cdef double denom, corr, dist_sq
    cdef double mu4_x, mu4_y, varx, vary, kurt_x, kurt_y
    cdef double nan_value = np.nan
    cdef double inf_value = np.inf
    cdef bint is_const, is_spiked, is_corr
    # (2026-07-06) "point 1", fully in Cython this time: every candidate row
    # has at least one side anchored at the just-arrived current window.
    # Unlike validate_corr_batch (which receives pre-sliced x/y batches with
    # no series/time metadata), this function already has s1/s2/start1/
    # start2/w/n_cols available per row inside the nogil loop, so "is this
    # side current" is computed directly here at C speed -- no Python-level
    # per-row bookkeeping arrays are needed at all, unlike the earlier
    # (reverted-to-opt-in) validate_corr_batch extension where that
    # bookkeeping's Python-level cost often exceeded the saving. See
    # docs/implementation_log.md.
    cdef bint have_current_window_cache = (
        current_window_sums is not None
        and current_window_sums_sq is not None
        and current_window_sums_cu is not None
        and current_window_sums_qu is not None
        and current_window_size > 0
    )
    cdef bint side1_is_current, side2_is_current

    with nogil:
        for i in range(n_items):
            corr_view[i] = nan_value
            dist_view[i] = inf_value
            s1 = rows[i, 0]
            s2 = rows[i, 1]
            t1 = rows[i, 2]
            t2 = rows[i, 3]
            w = rows[i, 4]
            start1 = t1 - base_index
            start2 = t2 - base_index
            if (
                w <= 0
                or s1 < 0
                or s2 < 0
                or s1 >= n_series
                or s2 >= n_series
                or start1 < 0
                or start2 < 0
                or start1 + w > n_cols
                or start2 + w > n_cols
            ):
                const_view[i] = <uint8_t>1
                continue

            side1_is_current = (
                have_current_window_cache
                and w == current_window_size
                and start1 + w == n_cols
                and s1 < current_window_sums.shape[0]
            )
            side2_is_current = (
                have_current_window_cache
                and w == current_window_size
                and start2 + w == n_cols
                and s2 < current_window_sums.shape[0]
            )

            sx = sy = 0.0
            sum_xy = sum_xx = sum_yy = 0.0
            sum_xxx = sum_yyy = 0.0
            sum_xxxx = sum_yyyy = 0.0
            if side1_is_current:
                sx = current_window_sums[s1]
                sum_xx = current_window_sums_sq[s1]
                sum_xxx = current_window_sums_cu[s1]
                sum_xxxx = current_window_sums_qu[s1]
            if side2_is_current:
                sy = current_window_sums[s2]
                sum_yy = current_window_sums_sq[s2]
                sum_yyy = current_window_sums_cu[s2]
                sum_yyyy = current_window_sums_qu[s2]
            for j in range(w):
                xi = data[s1, start1 + j]
                yi = data[s2, start2 + j]
                sum_xy += xi * yi
                if not side1_is_current:
                    sx += xi
                    sum_xx += xi * xi
                    sum_xxx += xi * xi * xi
                    sum_xxxx += xi * xi * xi * xi
                if not side2_is_current:
                    sy += yi
                    sum_yy += yi * yi
                    sum_yyy += yi * yi * yi
                    sum_yyyy += yi * yi * yi * yi

            mean_x = sx / w
            mean_y = sy / w
            var_x = sum_xx - (sx * sx) / w
            var_y = sum_yy - (sy * sy) / w
            if var_x < 0.0:
                var_x = 0.0
            if var_y < 0.0:
                var_y = 0.0

            is_const = (var_x <= (std_thresh * std_thresh) * w) or (var_y <= (std_thresh * std_thresh) * w)
            if is_const:
                const_view[i] = <uint8_t>1
                continue

            is_spiked = False
            if w >= 4:
                varx = var_x / w
                if varx > 0.0:
                    mu4_x = (
                        sum_xxxx
                        - 4.0 * mean_x * sum_xxx
                        + 6.0 * (mean_x * mean_x) * sum_xx
                        - 4.0 * (mean_x * mean_x * mean_x) * sx
                        + w * (mean_x * mean_x * mean_x * mean_x)
                    )
                    kurt_x = (mu4_x / w) / (varx * varx) - 3.0
                    if kurt_x > kurt_thresh:
                        is_spiked = True
                vary = var_y / w
                if (not is_spiked) and vary > 0.0:
                    mu4_y = (
                        sum_yyyy
                        - 4.0 * mean_y * sum_yyy
                        + 6.0 * (mean_y * mean_y) * sum_yy
                        - 4.0 * (mean_y * mean_y * mean_y) * sy
                        + w * (mean_y * mean_y * mean_y * mean_y)
                    )
                    kurt_y = (mu4_y / w) / (vary * vary) - 3.0
                    if kurt_y > kurt_thresh:
                        is_spiked = True

            if is_spiked:
                spike_view[i] = <uint8_t>1
                continue

            denom = sqrt(var_x * var_y)
            if denom == 0.0:
                corr = nan_value
            else:
                corr = (sum_xy - (sx * sy) / w) / denom
                if corr > 1.0:
                    corr = 1.0
                elif corr < -1.0:
                    corr = -1.0
            corr_view[i] = corr

            dist_sq = sum_xx + sum_yy - 2.0 * sum_xy
            if dist_sq < 0.0:
                dist_sq = 0.0
            dist_view[i] = sqrt(dist_sq)

            is_corr = False
            if corr == corr:
                if neg_corr:
                    is_corr = fabs(corr) >= corr_threshold
                else:
                    is_corr = corr >= corr_threshold
            if is_corr:
                accepted_view[i] = <uint8_t>1

    return accepted, corrs, dists, constants, spiked


cdef inline double _central_mu4_from_sums(double s1,
                                         double s2,
                                         double s3,
                                         double s4,
                                         long n) nogil:
    cdef double mean
    cdef double mu4
    if n <= 0:
        return 0.0
    mean = s1 / n
    mu4 = s4 - 4.0 * mean * s3 + 6.0 * mean * mean * s2 - 4.0 * mean * mean * mean * s1 + n * mean * mean * mean * mean
    if mu4 < 0.0:
        return 0.0
    return mu4


def hybrid_validate_repeated_batch(double[:, :] data,
                                   long[:] series1,
                                   long[:] series2,
                                   long[:] prev_t1,
                                   long[:] prev_t2,
                                   long[:] curr_t1,
                                   long[:] curr_t2,
                                   long[:] window_size,
                                   long base_index,
                                   long window_step,
                                   double[:, :] prev_stats,
                                   double corr_threshold,
                                   bint neg_corr,
                                   double std_thresh=1e-3,
                                   double kurt_thresh=5.0):
    """Exact rolling Pearson validation for repeated CorrTrack candidate relations.

    prev_stats columns: n, sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy.
    Each result tuple is:
      ok, is_corr, corr, dist, is_const, is_spiked, n, sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy
    ok=False means the caller must fall back to standard full-window validation.
    """
    cdef Py_ssize_t n_items = series1.shape[0]
    if (
        series2.shape[0] != n_items
        or prev_t1.shape[0] != n_items
        or prev_t2.shape[0] != n_items
        or curr_t1.shape[0] != n_items
        or curr_t2.shape[0] != n_items
        or window_size.shape[0] != n_items
        or prev_stats.shape[0] != n_items
        or prev_stats.shape[1] < 10
    ):
        raise ValueError("hybrid validation arrays must have matching lengths")

    cdef Py_ssize_t i, j
    cdef long s1, s2, pt1, pt2, ct1, ct2, w, step
    cdef long start1_prev, start2_prev, start1_curr, start2_curr
    cdef double sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy
    cdef double old_x, old_y, new_x, new_y
    cdef double var_x, var_y, denom, corr, dist_sq, dist
    cdef double mu4_x, mu4_y, varx, vary, kurt_x, kurt_y
    cdef bint is_const, is_spiked, is_corr
    cdef long n
    cdef long n_series = data.shape[0]
    cdef long n_cols = data.shape[1]
    results = []

    for i in range(n_items):
        s1 = series1[i]
        s2 = series2[i]
        pt1 = prev_t1[i]
        pt2 = prev_t2[i]
        ct1 = curr_t1[i]
        ct2 = curr_t2[i]
        w = window_size[i]
        step = ct1 - pt1
        if (
            s1 < 0
            or s1 >= n_series
            or s2 < 0
            or s2 >= n_series
            or w <= 0
            or step <= 0
            or step >= w
            or step != window_step
            or (ct2 - pt2) != step
        ):
            results.append((False, False, float("nan"), float("inf"), False, False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        start1_prev = pt1 - base_index
        start2_prev = pt2 - base_index
        start1_curr = ct1 - base_index
        start2_curr = ct2 - base_index
        if (
            start1_prev < 0
            or start2_prev < 0
            or start1_curr < 0
            or start2_curr < 0
            or start1_curr + w > n_cols
            or start2_curr + w > n_cols
        ):
            results.append((False, False, float("nan"), float("inf"), False, False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        n = <long>prev_stats[i, 0]
        if n != w:
            results.append((False, False, float("nan"), float("inf"), False, False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        sx = prev_stats[i, 1]
        sy = prev_stats[i, 2]
        sx2 = prev_stats[i, 3]
        sy2 = prev_stats[i, 4]
        sx3 = prev_stats[i, 5]
        sy3 = prev_stats[i, 6]
        sx4 = prev_stats[i, 7]
        sy4 = prev_stats[i, 8]
        sxy = prev_stats[i, 9]

        for j in range(step):
            old_x = data[s1, start1_prev + j]
            old_y = data[s2, start2_prev + j]
            new_x = data[s1, start1_curr + w - step + j]
            new_y = data[s2, start2_curr + w - step + j]
            sx += new_x - old_x
            sy += new_y - old_y
            sx2 += new_x * new_x - old_x * old_x
            sy2 += new_y * new_y - old_y * old_y
            sx3 += new_x * new_x * new_x - old_x * old_x * old_x
            sy3 += new_y * new_y * new_y - old_y * old_y * old_y
            sx4 += new_x * new_x * new_x * new_x - old_x * old_x * old_x * old_x
            sy4 += new_y * new_y * new_y * new_y - old_y * old_y * old_y * old_y
            sxy += new_x * new_y - old_x * old_y

        var_x = sx2 - (sx * sx) / n
        var_y = sy2 - (sy * sy) / n
        if var_x < 0.0:
            var_x = 0.0
        if var_y < 0.0:
            var_y = 0.0
        dist_sq = sx2 + sy2 - 2.0 * sxy
        if dist_sq < 0.0:
            dist_sq = 0.0
        dist = sqrt(dist_sq)

        is_const = (var_x <= (std_thresh * std_thresh) * n) or (var_y <= (std_thresh * std_thresh) * n)
        if is_const:
            results.append((True, False, float("nan"), float("inf"), True, False, n, sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy))
            continue

        is_spiked = False
        if n >= 4:
            varx = var_x / n
            if varx > 0.0:
                mu4_x = _central_mu4_from_sums(sx, sx2, sx3, sx4, n)
                kurt_x = (mu4_x / n) / (varx * varx) - 3.0
                if kurt_x > kurt_thresh:
                    is_spiked = True
            vary = var_y / n
            if not is_spiked and vary > 0.0:
                mu4_y = _central_mu4_from_sums(sy, sy2, sy3, sy4, n)
                kurt_y = (mu4_y / n) / (vary * vary) - 3.0
                if kurt_y > kurt_thresh:
                    is_spiked = True
        if is_spiked:
            results.append((True, False, float("nan"), float("inf"), False, True, n, sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy))
            continue

        denom = sqrt(var_x * var_y)
        if denom == 0.0:
            corr = float("nan")
        else:
            corr = (sxy - (sx * sy) / n) / denom
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0

        is_corr = False
        if corr == corr:
            if neg_corr:
                is_corr = fabs(corr) >= corr_threshold
            else:
                is_corr = corr >= corr_threshold

        results.append((True, is_corr, corr, dist, False, False, n, sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy))

    return results


# (2026-07-06) Experimental approximate candidate-search backend
# ("InstinctIndex"), per the human's request based on a ChatGPT-authored
# design (see docs/implementation_log.md, "InstinctIndex: experimental
# approximate graph backend"). Unlike every other backend in this file,
# this one is NOT recall-preserving by construction -- it is a best-first
# graph search over a bounded-degree attraction graph with an explicit
# traversal budget (ef_search), not an exhaustive-but-pruned scan. Its
# correctness contract is therefore "measured recall on synthetic ground
# truth stays high," not "brute-force-exact match" like every other
# backend -- see CLAUDE.md's "do not claim recall preservation unless the
# benchmark verifies it." Priorities 1-2 of the human's spec are
# implemented (core graph + insert/lazy-delete/query + threshold/topk/
# hybrid query modes + metrics); Priorities 3-5 (audit mode, fallback
# mode, graph rebuild/repair) are deliberately deferred, not half-built.

cdef inline void _instinct_heap_push(double *heap_score, int64_t *heap_id,
                                      Py_ssize_t *heap_size, double score,
                                      int64_t node_id) noexcept nogil:
    """Binary max-heap push over raw C arrays (same raw-buffer style as
    this file's existing pair_seen_*/buf helpers)."""
    cdef Py_ssize_t i, parent
    cdef double tmp_s
    cdef int64_t tmp_i
    i = heap_size[0]
    heap_score[i] = score
    heap_id[i] = node_id
    heap_size[0] += 1
    while i > 0:
        parent = (i - 1) // 2
        if heap_score[parent] < heap_score[i]:
            tmp_s = heap_score[parent]
            heap_score[parent] = heap_score[i]
            heap_score[i] = tmp_s
            tmp_i = heap_id[parent]
            heap_id[parent] = heap_id[i]
            heap_id[i] = tmp_i
            i = parent
        else:
            break


cdef inline void _instinct_heap_pop_max(double *heap_score, int64_t *heap_id,
                                         Py_ssize_t *heap_size,
                                         double *out_score,
                                         int64_t *out_id) noexcept nogil:
    cdef Py_ssize_t i, left, right, largest, last
    cdef double tmp_s
    cdef int64_t tmp_i
    out_score[0] = heap_score[0]
    out_id[0] = heap_id[0]
    heap_size[0] -= 1
    last = heap_size[0]
    heap_score[0] = heap_score[last]
    heap_id[0] = heap_id[last]
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < heap_size[0] and heap_score[left] > heap_score[largest]:
            largest = left
        if right < heap_size[0] and heap_score[right] > heap_score[largest]:
            largest = right
        if largest == i:
            break
        tmp_s = heap_score[i]
        heap_score[i] = heap_score[largest]
        heap_score[largest] = tmp_s
        tmp_i = heap_id[i]
        heap_id[i] = heap_id[largest]
        heap_id[largest] = tmp_i
        i = largest


cdef Py_ssize_t _instinct_partial_top_k(double *scores, int64_t *ids, Py_ssize_t n,
                                        Py_ssize_t k, double *out_scores,
                                        int64_t *out_ids) noexcept nogil:
    """Selection-sort-based partial top-k over the first n entries of
    (scores, ids): writes the min(k, n) highest-scoring pairs into
    (out_scores, out_ids), descending. O(n*k) -- deliberately simple (n and
    k are both bounded by ef_search/max_degree, at most a few thousand),
    matching this backend's "keep it simple, test the idea" scope."""
    cdef Py_ssize_t limit = k if k < n else n
    cdef Py_ssize_t i, j, best_j
    cdef double best_score
    cdef uint8_t *taken = NULL
    if limit <= 0 or n <= 0:
        return 0
    taken = <uint8_t *>malloc(n * sizeof(uint8_t))
    if taken == NULL:
        return 0
    for i in range(n):
        taken[i] = 0
    for i in range(limit):
        best_j = -1
        best_score = -1e18
        for j in range(n):
            if taken[j]:
                continue
            if scores[j] > best_score:
                best_score = scores[j]
                best_j = j
        if best_j < 0:
            break
        taken[best_j] = 1
        out_scores[i] = scores[best_j]
        out_ids[i] = ids[best_j]
    free(taken)
    return limit


cdef class InstinctIndex:
    cdef public object last_stats
    cdef Py_ssize_t _n_vectors
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _count
    cdef Py_ssize_t _max_degree
    cdef Py_ssize_t _ef_insert
    cdef Py_ssize_t _ef_search
    cdef Py_ssize_t _entry_points_target
    cdef Py_ssize_t _num_entry_points
    cdef Py_ssize_t _query_mode_code  # 0=threshold, 1=topk, 2=hybrid
    cdef Py_ssize_t _top_k
    cdef Py_ssize_t _min_candidates
    cdef int64_t _next_entry_id
    cdef int64_t _min_valid_time
    cdef int64_t _query_stamp
    cdef Py_ssize_t _alive_count
    cdef Py_ssize_t _dead_count
    cdef object _vectors
    cdef object _alive
    cdef object _window_idx
    cdef object _sid_idx
    cdef object _sid_rank
    cdef object _time
    cdef object _window_size
    cdef object _entry_ids
    cdef object _neighbors
    cdef object _degree
    cdef object _entry_point_ids
    cdef object _visited_stamp
    # (2026-07-07) Tier 1 perf fix -- see docs/implementation_log.md,
    # "InstinctIndex Tier 1/2/3 acceleration". These are persistent,
    # allocated once in __cinit__ (fixed budgets: heap/insert/query buffers
    # are sized from ef_search/ef_insert/max_degree, which never change
    # after construction) or grown-in-place (pair output buffer, pair-seen
    # hash set -- both already support realloc-in-place via the existing
    # double-pointer helpers). Previously, _search_raw/_insert_one/
    # _find_pair_rows_meta malloc'd+freed these on every single call (every
    # query, every inserted row) -- pure allocator overhead invisible to
    # every reported metric (dot_checks/edges_scanned only count algorithmic
    # work, not allocation). Freed in __dealloc__.
    # (2026-07-07) Separate query-sized vs. insert-sized heap buffers --
    # NOT one shared max(ef_search, ef_insert)-sized buffer. An earlier
    # version of this shared one buffer and accidentally changed
    # insertion's behavior: the original code sized each call's heap
    # independently ((ef_insert + max_degree)*4+64 for inserts,
    # (ef_search + max_degree)*4+64 for queries), and a smaller heap_cap
    # silently drops candidate pushes once full -- effectively bounding
    # how much of the graph an insert's bootstrap search explores. Sharing
    # one larger buffer let insertion silently explore a much bigger
    # candidate pool than before, which measured slower, not faster, and
    # was a real (if benign-looking) behavior change vs. the original
    # algorithm. Keeping these separate preserves the exact original
    # per-call size.
    cdef double *_q_heap_score_buf
    cdef int64_t *_q_heap_id_buf
    cdef Py_ssize_t _q_heap_buf_cap
    cdef double *_ins_heap_score_buf
    cdef int64_t *_ins_heap_id_buf
    cdef Py_ssize_t _ins_heap_buf_cap
    cdef int64_t *_ins_cand_ids_buf
    cdef double *_ins_cand_scores_buf
    cdef Py_ssize_t _ins_cand_cap
    cdef int64_t *_q_visited_ids_buf
    cdef double *_q_visited_scores_buf
    cdef int64_t *_q_sel_ids_buf
    cdef double *_q_sel_scores_buf
    cdef Py_ssize_t _q_search_cap
    cdef int64_t *_pair_out_buf
    cdef Py_ssize_t _pair_out_cap
    cdef uint8_t *_pair_seen_occupied_buf
    cdef int64_t *_pair_seen_a_buf
    cdef int64_t *_pair_seen_b_buf
    cdef Py_ssize_t _pair_seen_cap_buf
    cdef object _win_sid_idx_arr
    cdef object _win_sid_rank_arr
    cdef object _win_time_arr
    cdef object _win_size_arr
    cdef Py_ssize_t _win_capacity

    def __cinit__(self, Py_ssize_t n_vectors=1, Py_ssize_t initial_capacity=1024,
                  Py_ssize_t max_degree=16, Py_ssize_t ef_insert=128,
                  Py_ssize_t ef_search=256, Py_ssize_t entry_points=8,
                  object query_mode="hybrid", Py_ssize_t top_k=256,
                  Py_ssize_t min_candidates=64, long seed=0):
        if n_vectors <= 0:
            n_vectors = 1
        if initial_capacity < 16:
            initial_capacity = 16
        if max_degree <= 0:
            max_degree = 16
        if ef_insert <= 0:
            ef_insert = max_degree * 4
        if ef_search <= 0:
            ef_search = 256
        if entry_points <= 0:
            entry_points = 8
        if top_k <= 0:
            top_k = 256
        if min_candidates <= 0:
            min_candidates = 64
        self._n_vectors = n_vectors
        self._capacity = initial_capacity
        self._count = 0
        self._max_degree = max_degree
        self._ef_insert = ef_insert
        self._ef_search = ef_search
        self._entry_points_target = entry_points
        self._num_entry_points = 0
        mode_key = str(query_mode or "hybrid").strip().lower()
        if mode_key == "threshold":
            self._query_mode_code = 0
        elif mode_key == "topk":
            self._query_mode_code = 1
        elif mode_key == "hybrid":
            self._query_mode_code = 2
        else:
            raise ValueError(
                f"InstinctIndex query_mode={query_mode!r} not supported -- "
                "'threshold'/'topk'/'hybrid' are implemented; "
                "'fallback'/'audit' are deferred (see docs/implementation_log.md)."
            )
        self._top_k = top_k
        self._min_candidates = min_candidates
        self._next_entry_id = 0
        self._min_valid_time = -9223372036854775807
        self._query_stamp = 0
        self._alive_count = 0
        self._dead_count = 0
        self._vectors = np.empty((self._capacity, self._n_vectors), dtype=np.float64)
        self._alive = np.zeros(self._capacity, dtype=np.uint8)
        self._window_idx = np.empty(self._capacity, dtype=np.int64)
        self._sid_idx = np.empty(self._capacity, dtype=np.int64)
        self._sid_rank = np.empty(self._capacity, dtype=np.int64)
        self._time = np.empty(self._capacity, dtype=np.int64)
        self._window_size = np.empty(self._capacity, dtype=np.int64)
        self._entry_ids = np.empty(self._capacity, dtype=np.int64)
        self._neighbors = np.full((self._capacity, self._max_degree), -1, dtype=np.int64)
        self._degree = np.zeros(self._capacity, dtype=np.int64)
        self._entry_point_ids = np.full(self._entry_points_target, -1, dtype=np.int64)
        self._visited_stamp = np.full(self._capacity, -1, dtype=np.int64)
        self._win_capacity = initial_capacity
        self._win_sid_idx_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_sid_rank_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_time_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._win_size_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self.last_stats = {}

        # (2026-07-07) Tier 1 perf fix -- persistent scratch buffers, see
        # the cdef field comment above. Fixed budgets never change after
        # construction (ef_search/ef_insert/max_degree are immutable
        # hyperparameters), so a single one-time allocation is always
        # large enough -- no growth logic needed for these three.
        #
        # (2026-07-07) Pre-existing bug found and fixed while benchmarking
        # large entry_points settings: _search_raw's entry-point seeding
        # loop silently drops any entry point once the heap fills
        # (`if heap_size < heap_cap: push`) -- this sizing formula did not
        # account for entry_points_target at all, only ef_search/ef_insert
        # and max_degree. With a small ef_search/max_degree config (the
        # fast one found in the prior scale-test entry) and a large
        # entry_points_target, most entry points were being silently
        # discarded before traversal even began -- measured directly:
        # entry_points=1024 gave *worse* recall (0.66) than entry_points=512
        # (1.00) at ef_search=128/max_degree=24 (heap_cap=672 < 1024), the
        # opposite of what more entry points should ever do. This bug
        # pre-dates today's Tier 1 buffer-persistence work (the original
        # per-call malloc used the exact same formula) -- it was just never
        # triggered before because earlier tests paired large entry_points
        # with large ef_search/max_degree (naturally big enough heap_cap).
        # Fixed by ensuring heap_cap is always at least large enough to
        # hold every entry point.
        self._q_heap_buf_cap = max(
            (self._ef_search + self._max_degree) * 4 + 64,
            self._entry_points_target + self._max_degree + 64,
        )
        self._q_heap_score_buf = <double *>malloc(self._q_heap_buf_cap * sizeof(double))
        self._q_heap_id_buf = <int64_t *>malloc(self._q_heap_buf_cap * sizeof(int64_t))
        self._ins_heap_buf_cap = max(
            (self._ef_insert + self._max_degree) * 4 + 64,
            self._entry_points_target + self._max_degree + 64,
        )
        self._ins_heap_score_buf = <double *>malloc(self._ins_heap_buf_cap * sizeof(double))
        self._ins_heap_id_buf = <int64_t *>malloc(self._ins_heap_buf_cap * sizeof(int64_t))
        self._ins_cand_cap = self._ef_insert + self._max_degree + 8
        self._ins_cand_ids_buf = <int64_t *>malloc(self._ins_cand_cap * sizeof(int64_t))
        self._ins_cand_scores_buf = <double *>malloc(self._ins_cand_cap * sizeof(double))
        self._q_search_cap = self._ef_search + self._max_degree + 16
        self._q_visited_ids_buf = <int64_t *>malloc(self._q_search_cap * sizeof(int64_t))
        self._q_visited_scores_buf = <double *>malloc(self._q_search_cap * sizeof(double))
        self._q_sel_ids_buf = <int64_t *>malloc(self._q_search_cap * sizeof(int64_t))
        self._q_sel_scores_buf = <double *>malloc(self._q_search_cap * sizeof(double))
        # These two grow in place across calls (via the existing
        # _append_pair/_pair_seen_grow double-pointer helpers) since their
        # required size depends on batch output size, not a fixed
        # hyperparameter -- start small and let them grow, matching this
        # file's established doubling-capacity convention.
        self._pair_out_cap = 1024
        self._pair_out_buf = <int64_t *>malloc(self._pair_out_cap * 2 * sizeof(int64_t))
        self._pair_seen_cap_buf = 0
        self._pair_seen_occupied_buf = NULL
        self._pair_seen_a_buf = NULL
        self._pair_seen_b_buf = NULL
        if (self._q_heap_score_buf == NULL or self._q_heap_id_buf == NULL
                or self._ins_heap_score_buf == NULL or self._ins_heap_id_buf == NULL
                or self._ins_cand_ids_buf == NULL or self._ins_cand_scores_buf == NULL
                or self._q_visited_ids_buf == NULL or self._q_visited_scores_buf == NULL
                or self._q_sel_ids_buf == NULL or self._q_sel_scores_buf == NULL
                or self._pair_out_buf == NULL):
            raise MemoryError("InstinctIndex: failed to allocate persistent scratch buffers")

    def __dealloc__(self):
        if self._q_heap_score_buf != NULL:
            free(self._q_heap_score_buf)
        if self._q_heap_id_buf != NULL:
            free(self._q_heap_id_buf)
        if self._ins_heap_score_buf != NULL:
            free(self._ins_heap_score_buf)
        if self._ins_heap_id_buf != NULL:
            free(self._ins_heap_id_buf)
        if self._ins_cand_ids_buf != NULL:
            free(self._ins_cand_ids_buf)
        if self._ins_cand_scores_buf != NULL:
            free(self._ins_cand_scores_buf)
        if self._q_visited_ids_buf != NULL:
            free(self._q_visited_ids_buf)
        if self._q_visited_scores_buf != NULL:
            free(self._q_visited_scores_buf)
        if self._q_sel_ids_buf != NULL:
            free(self._q_sel_ids_buf)
        if self._q_sel_scores_buf != NULL:
            free(self._q_sel_scores_buf)
        if self._pair_out_buf != NULL:
            free(self._pair_out_buf)
        if self._pair_seen_occupied_buf != NULL:
            free(self._pair_seen_occupied_buf)
        if self._pair_seen_a_buf != NULL:
            free(self._pair_seen_a_buf)
        if self._pair_seen_b_buf != NULL:
            free(self._pair_seen_b_buf)

    cdef void _ensure_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap, n
        if need <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < need:
            new_cap *= 2
        new_vectors = np.empty((new_cap, self._n_vectors), dtype=np.float64)
        new_alive = np.zeros(new_cap, dtype=np.uint8)
        new_window_idx = np.empty(new_cap, dtype=np.int64)
        new_sid_idx = np.empty(new_cap, dtype=np.int64)
        new_sid_rank = np.empty(new_cap, dtype=np.int64)
        new_time = np.empty(new_cap, dtype=np.int64)
        new_window_size = np.empty(new_cap, dtype=np.int64)
        new_entry_ids = np.empty(new_cap, dtype=np.int64)
        new_neighbors = np.full((new_cap, self._max_degree), -1, dtype=np.int64)
        new_degree = np.zeros(new_cap, dtype=np.int64)
        new_visited_stamp = np.full(new_cap, -1, dtype=np.int64)
        n = self._count
        if n > 0:
            new_vectors[:n, :] = self._vectors[:n, :]
            new_alive[:n] = self._alive[:n]
            new_window_idx[:n] = self._window_idx[:n]
            new_sid_idx[:n] = self._sid_idx[:n]
            new_sid_rank[:n] = self._sid_rank[:n]
            new_time[:n] = self._time[:n]
            new_window_size[:n] = self._window_size[:n]
            new_entry_ids[:n] = self._entry_ids[:n]
            new_neighbors[:n, :] = self._neighbors[:n, :]
            new_degree[:n] = self._degree[:n]
            new_visited_stamp[:n] = self._visited_stamp[:n]
        self._vectors = new_vectors
        self._alive = new_alive
        self._window_idx = new_window_idx
        self._sid_idx = new_sid_idx
        self._sid_rank = new_sid_rank
        self._time = new_time
        self._window_size = new_window_size
        self._entry_ids = new_entry_ids
        self._neighbors = new_neighbors
        self._degree = new_degree
        self._visited_stamp = new_visited_stamp
        self._capacity = new_cap

    cdef void _ensure_window_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        if need <= self._win_capacity:
            return
        new_cap = self._win_capacity
        while new_cap < need:
            new_cap *= 2
        new_sid_idx = np.full(new_cap, -1, dtype=np.int64)
        new_sid_rank = np.full(new_cap, -1, dtype=np.int64)
        new_time = np.zeros(new_cap, dtype=np.int64)
        new_size = np.zeros(new_cap, dtype=np.int64)
        if self._win_capacity > 0:
            new_sid_idx[:self._win_capacity] = self._win_sid_idx_arr[:self._win_capacity]
            new_sid_rank[:self._win_capacity] = self._win_sid_rank_arr[:self._win_capacity]
            new_time[:self._win_capacity] = self._win_time_arr[:self._win_capacity]
            new_size[:self._win_capacity] = self._win_size_arr[:self._win_capacity]
        self._win_sid_idx_arr = new_sid_idx
        self._win_sid_rank_arr = new_sid_rank
        self._win_time_arr = new_time
        self._win_size_arr = new_size
        self._win_capacity = new_cap

    cdef Py_ssize_t _search_raw(self, double[:] q_vec, Py_ssize_t ef_search, bint signed_abs,
                                int64_t *out_ids, double *out_scores, Py_ssize_t out_cap,
                                Py_ssize_t *edges_scanned, Py_ssize_t *dead_skipped,
                                int64_t exclude_id, bint for_insert):
        """Best-first traversal seeded from the current entry points.
        Explores according to an explicit budget (ef_search live nodes
        visited, not "nodes above gamma_emit") -- see the module docstring
        note above and docs/implementation_log.md for why: thresholding
        the traversal itself (not just the returned set) risks getting
        stuck in the wrong basin for loose/diffuse neighborhoods."""
        cdef double[:, :] vectors = self._vectors
        cdef uint8_t[:] alive = self._alive
        cdef int64_t[:, :] neighbors = self._neighbors
        cdef int64_t[:] degree = self._degree
        cdef int64_t[:] visited_stamp = self._visited_stamp
        cdef int64_t[:] entry_point_ids = self._entry_point_ids
        # (2026-07-07) Tier 1 -- reuse persistent, __cinit__-allocated heap
        # buffers instead of malloc/free on every call (this function runs
        # once per query *and* once per inserted row -- see the cdef field
        # comment on the class for why that was a real, previously
        # invisible cost). Deliberately TWO separate buffers (query-sized
        # vs. insert-sized), not one shared max(ef_search, ef_insert)-sized
        # buffer -- see the cdef field comment for why a shared buffer
        # silently changed insertion's behavior (measured slower, not
        # faster) in an earlier version of this fix.
        cdef double *heap_score
        cdef int64_t *heap_id
        cdef Py_ssize_t heap_cap
        if for_insert:
            heap_score = self._ins_heap_score_buf
            heap_id = self._ins_heap_id_buf
            heap_cap = self._ins_heap_buf_cap
        else:
            heap_score = self._q_heap_score_buf
            heap_id = self._q_heap_id_buf
            heap_cap = self._q_heap_buf_cap
        cdef Py_ssize_t heap_size = 0
        cdef Py_ssize_t n_live_visited = 0
        cdef Py_ssize_t i, k, d
        cdef double score, popped_score
        cdef int64_t node_id, popped_id, nb
        cdef int64_t stamp
        cdef bint node_alive

        edges_scanned[0] = 0
        dead_skipped[0] = 0

        self._query_stamp += 1
        stamp = self._query_stamp

        for i in range(self._num_entry_points):
            node_id = entry_point_ids[i]
            if node_id < 0 or node_id == exclude_id:
                continue
            if visited_stamp[node_id] == stamp:
                continue
            visited_stamp[node_id] = stamp
            score = 0.0
            for k in range(self._n_vectors):
                score += q_vec[k] * vectors[node_id, k]
            if signed_abs:
                score = fabs(score)
            if heap_size < heap_cap:
                _instinct_heap_push(heap_score, heap_id, &heap_size, score, node_id)

        while heap_size > 0 and n_live_visited < ef_search:
            _instinct_heap_pop_max(heap_score, heap_id, &heap_size, &popped_score, &popped_id)
            node_id = popped_id
            node_alive = alive[node_id] != 0
            if node_id != exclude_id:
                if node_alive:
                    if n_live_visited < out_cap:
                        out_ids[n_live_visited] = node_id
                        out_scores[n_live_visited] = popped_score
                    n_live_visited += 1
                else:
                    dead_skipped[0] += 1
            # (2026-07-07) A "skip neighbor-expansion for dead nodes" variant
            # was tried here (Tier 2 candidate) and reverted: an A/B test
            # under heavy churn (70% expired, 6-cluster synthetic data)
            # showed it increased the frequency of a pre-existing,
            # deterministic recall failure mode from 1/10 to 6/10 trials --
            # dead nodes evidently do act as real bridges to live neighbors
            # often enough that skipping them is not safe. See
            # docs/implementation_log.md, "InstinctIndex Tier 1/2/3
            # acceleration" for the full A/B result. Kept expanding dead
            # nodes' neighbors unconditionally, matching original behavior.
            for k in range(degree[node_id]):
                nb = neighbors[node_id, k]
                edges_scanned[0] += 1
                if nb < 0 or nb == exclude_id:
                    continue
                if visited_stamp[nb] == stamp:
                    continue
                visited_stamp[nb] = stamp
                score = 0.0
                for d in range(self._n_vectors):
                    score += q_vec[d] * vectors[nb, d]
                if signed_abs:
                    score = fabs(score)
                if heap_size < heap_cap:
                    _instinct_heap_push(heap_score, heap_id, &heap_size, score, nb)

        return n_live_visited

    cdef Py_ssize_t _select_candidates(self, double *scores, int64_t *ids, Py_ssize_t n,
                                        double gamma_emit, double *out_scores, int64_t *out_ids,
                                        Py_ssize_t *out_threshold_count, Py_ssize_t *out_topk_count):
        cdef Py_ssize_t i, count, thresh_count
        if self._query_mode_code == 0:
            count = 0
            for i in range(n):
                if scores[i] >= gamma_emit:
                    out_scores[count] = scores[i]
                    out_ids[count] = ids[i]
                    count += 1
            out_threshold_count[0] = count
            out_topk_count[0] = 0
            return count
        elif self._query_mode_code == 1:
            count = _instinct_partial_top_k(scores, ids, n, self._top_k, out_scores, out_ids)
            out_threshold_count[0] = 0
            out_topk_count[0] = count
            return count
        else:
            thresh_count = 0
            for i in range(n):
                if scores[i] >= gamma_emit:
                    thresh_count += 1
            if thresh_count >= self._min_candidates:
                count = 0
                for i in range(n):
                    if scores[i] >= gamma_emit:
                        out_scores[count] = scores[i]
                        out_ids[count] = ids[i]
                        count += 1
                out_threshold_count[0] = count
                out_topk_count[0] = 0
                return count
            else:
                count = _instinct_partial_top_k(scores, ids, n, self._top_k, out_scores, out_ids)
                out_threshold_count[0] = thresh_count
                out_topk_count[0] = count
                return count

    cdef int64_t _insert_one(self, double[:] vector, int64_t window_idx, int64_t sid_idx,
                              int64_t time_idx, int64_t window_size, int64_t sid_rank):
        cdef Py_ssize_t i, k, d, j, sel_count, weakest_idx
        cdef int64_t new_id, nb, nb2
        cdef double weakest_score, s, new_score
        cdef Py_ssize_t cand_cap
        # (2026-07-07) Tier 1 -- persistent buffers, see cdef field comment
        # on the class. cand_ids/cand_scores reuse self._ins_cand_*_buf
        # (sized ef_insert+max_degree+8 once in __cinit__); sel_ids/
        # sel_scores reuse self._q_sel_*_buf (sized ef_search+max_degree+16,
        # always >= max_degree) rather than allocating a third pair of
        # max_degree-sized buffers -- safe because insertion and query never
        # run concurrently on the same instance (see docs/implementation_log.md).
        cdef int64_t *cand_ids = self._ins_cand_ids_buf
        cdef double *cand_scores = self._ins_cand_scores_buf
        cdef int64_t *sel_ids = self._q_sel_ids_buf
        cdef double *sel_scores = self._q_sel_scores_buf
        cdef Py_ssize_t edges_scanned = 0, dead_skipped = 0
        cdef Py_ssize_t n_visited

        self._ensure_capacity(self._count + 1)
        i = self._count
        cdef double[:, :] vectors_mv = self._vectors
        cdef uint8_t[:] alive_mv = self._alive
        cdef int64_t[:] window_idx_mv = self._window_idx
        cdef int64_t[:] sid_idx_mv = self._sid_idx
        cdef int64_t[:] sid_rank_mv = self._sid_rank
        cdef int64_t[:] time_mv = self._time
        cdef int64_t[:] window_size_mv = self._window_size
        cdef int64_t[:] entry_ids_mv = self._entry_ids
        cdef int64_t[:, :] neighbors = self._neighbors
        cdef int64_t[:] degree = self._degree
        cdef int64_t[:] entry_point_ids = self._entry_point_ids

        for d in range(self._n_vectors):
            vectors_mv[i, d] = vector[d]
        alive_mv[i] = 1
        window_idx_mv[i] = window_idx
        sid_idx_mv[i] = sid_idx
        sid_rank_mv[i] = sid_rank
        time_mv[i] = time_idx
        window_size_mv[i] = window_size
        new_id = self._next_entry_id
        entry_ids_mv[i] = new_id
        self._next_entry_id += 1
        self._count += 1
        self._alive_count += 1

        if self._num_entry_points == 0:
            entry_point_ids[0] = i
            self._num_entry_points = 1
            return new_id

        cand_cap = self._ins_cand_cap
        n_visited = self._search_raw(vector, self._ef_insert, False, cand_ids, cand_scores,
                                      cand_cap, &edges_scanned, &dead_skipped, i, True)
        if n_visited < 0:
            raise MemoryError()

        sel_count = _instinct_partial_top_k(cand_scores, cand_ids, n_visited, self._max_degree,
                                            sel_scores, sel_ids)

        for k in range(sel_count):
            nb = sel_ids[k]
            if degree[i] < self._max_degree:
                neighbors[i, degree[i]] = nb
                degree[i] += 1
            if degree[nb] < self._max_degree:
                neighbors[nb, degree[nb]] = i
                degree[nb] += 1
            else:
                weakest_idx = -1
                weakest_score = 1e18
                for j in range(self._max_degree):
                    nb2 = neighbors[nb, j]
                    if nb2 < 0:
                        weakest_idx = j
                        weakest_score = -1e18
                        break
                    s = 0.0
                    for d in range(self._n_vectors):
                        s += vectors_mv[nb, d] * vectors_mv[nb2, d]
                    if s < weakest_score:
                        weakest_score = s
                        weakest_idx = j
                new_score = 0.0
                for d in range(self._n_vectors):
                    new_score += vectors_mv[nb, d] * vector[d]
                if weakest_idx >= 0 and new_score > weakest_score:
                    neighbors[nb, weakest_idx] = i

        if self._num_entry_points < self._entry_points_target:
            entry_point_ids[self._num_entry_points] = i
            self._num_entry_points += 1
        else:
            entry_point_ids[i % self._entry_points_target] = i

        return new_id

    def insert_many(self, values_in, window_idx_in, vectors_in=None, sid_idx_in=None,
                     time_in=None, window_size_in=None, sid_rank_in=None):
        cdef np.ndarray[np.int64_t, ndim=1] window_idx_np = np.asarray(window_idx_in, dtype=np.int64).ravel()
        cdef Py_ssize_t n = window_idx_np.shape[0]
        cdef np.ndarray[np.float64_t, ndim=2] vectors_np
        cdef np.ndarray[np.int64_t, ndim=1] sid_idx_np
        cdef np.ndarray[np.int64_t, ndim=1] time_np
        cdef np.ndarray[np.int64_t, ndim=1] window_size_np
        cdef np.ndarray[np.int64_t, ndim=1] sid_rank_np
        cdef np.ndarray[np.int64_t, ndim=1] entry_ids
        cdef Py_ssize_t i
        cdef int64_t max_window_idx = -1
        cdef int64_t wi
        cdef double t0 = time.perf_counter()

        if n == 0:
            self.last_stats = dict(self.last_stats or {})
            self.last_stats["instinct_insert_time"] = 0.0
            return np.empty(0, dtype=np.int64)
        if vectors_in is None:
            raise ValueError("InstinctIndex requires full sketch vectors")
        vectors_np = np.ascontiguousarray(np.asarray(vectors_in, dtype=np.float64))
        if vectors_np.ndim != 2 or vectors_np.shape[0] != n:
            raise ValueError("vectors must be a 2D array with one row per value")
        if vectors_np.shape[1] != self._n_vectors:
            raise ValueError("vector size does not match InstinctIndex dimension")
        if sid_idx_in is None:
            sid_idx_np = np.full(n, -1, dtype=np.int64)
        else:
            sid_idx_np = np.asarray(sid_idx_in, dtype=np.int64).ravel()
        if time_in is None:
            time_np = np.zeros(n, dtype=np.int64)
        else:
            time_np = np.asarray(time_in, dtype=np.int64).ravel()
        if window_size_in is None:
            window_size_np = np.zeros(n, dtype=np.int64)
        else:
            window_size_np = np.asarray(window_size_in, dtype=np.int64).ravel()
        if sid_rank_in is None:
            sid_rank_np = np.full(n, -1, dtype=np.int64)
        else:
            sid_rank_np = np.asarray(sid_rank_in, dtype=np.int64).ravel()
        if (window_idx_np.shape[0] != n or sid_idx_np.shape[0] != n or time_np.shape[0] != n
                or window_size_np.shape[0] != n or sid_rank_np.shape[0] != n):
            raise ValueError("metadata array sizes must match values size")

        for i in range(n):
            if window_idx_np[i] > max_window_idx:
                max_window_idx = <int64_t>window_idx_np[i]
        if max_window_idx >= 0:
            self._ensure_window_capacity(<Py_ssize_t>max_window_idx + 1)

        cdef int64_t[:] win_sid_idx = self._win_sid_idx_arr
        cdef int64_t[:] win_sid_rank = self._win_sid_rank_arr
        cdef int64_t[:] win_time = self._win_time_arr
        cdef int64_t[:] win_size = self._win_size_arr

        entry_ids = np.empty(n, dtype=np.int64)
        for i in range(n):
            wi = <int64_t>window_idx_np[i]
            entry_ids[i] = self._insert_one(
                vectors_np[i], wi, <int64_t>sid_idx_np[i], <int64_t>time_np[i],
                <int64_t>window_size_np[i], <int64_t>sid_rank_np[i],
            )
            if wi >= 0:
                win_sid_idx[wi] = <int64_t>sid_idx_np[i]
                win_sid_rank[wi] = <int64_t>sid_rank_np[i]
                win_time[wi] = <int64_t>time_np[i]
                win_size[wi] = <int64_t>window_size_np[i]
        self.last_stats = dict(self.last_stats or {})
        self.last_stats["instinct_insert_time"] = time.perf_counter() - t0
        return entry_ids

    cpdef drop_before_time(self, long min_valid_time):
        cdef Py_ssize_t i
        cdef int64_t[:] time_mv = self._time
        cdef uint8_t[:] alive_mv = self._alive
        self._min_valid_time = <int64_t>min_valid_time
        for i in range(self._count):
            if alive_mv[i] and time_mv[i] < self._min_valid_time:
                alive_mv[i] = 0
                self._alive_count -= 1
                self._dead_count += 1

    cpdef clear_recent(self):
        pass

    cpdef object find_pair_rows_full_cosine(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_meta(recent_entry_ids, gamma, False)

    cpdef object find_pair_rows_full_cosine_signed(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_meta(recent_entry_ids, gamma, True)

    cdef object _find_pair_rows_meta(self, long[:] recent_entry_ids, double gamma, bint signed_abs):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t i, k
        cdef int64_t q_entry, q_win, q_sid, q_rank, q_time, q_w
        # (2026-07-07) Tier 1 -- persistent buffers, see cdef field comment
        # on the class. visited_ids/scores and sel_ids/scores reuse the
        # fixed-budget self._q_*_buf pair (sized once in __cinit__);
        # self._pair_out_buf/self._pair_seen_*_buf grow in place across
        # calls via the existing double-pointer helpers (_append_pair,
        # _pair_seen_grow) -- passing &self._pair_out_buf etc. directly
        # means any realloc during this call is already reflected in the
        # instance fields afterward, no copy-back needed.
        cdef Py_ssize_t search_cap = self._q_search_cap
        cdef int64_t *visited_ids = self._q_visited_ids_buf
        cdef double *visited_scores = self._q_visited_scores_buf
        cdef int64_t *sel_ids = self._q_sel_ids_buf
        cdef double *sel_scores = self._q_sel_scores_buf
        cdef Py_ssize_t n_visited, sel_count, thresh_count, topk_count
        cdef Py_ssize_t edges_scanned, dead_skipped
        cdef Py_ssize_t count = 0, out_i
        cdef bint failed = False
        cdef Py_ssize_t pair_seen_count = 0
        cdef Py_ssize_t wanted_pair_seen_cap
        cdef int64_t pair_a, pair_b, a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t cand_row
        cdef Py_ssize_t total_visited_live = 0, total_edges_scanned = 0, total_dead_skipped = 0
        cdef Py_ssize_t total_threshold = 0, total_topk = 0, total_valid = 0
        cdef Py_ssize_t unique_pre_dot_pairs = 0, duplicate_pre_dot_pairs = 0
        cdef Py_ssize_t after_similarity = 0, pairs_before_dedupe = 0, pairs_after_dedupe = 0
        cdef double best_score_seen = -2.0
        cdef double sum_score_returned = 0.0
        cdef Py_ssize_t queries_with_too_few_live_nodes = 0
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t[:] win_sid_idx
        cdef int64_t[:] win_sid_rank
        cdef int64_t[:] win_time
        cdef int64_t[:] win_w
        cdef double t0 = time.perf_counter()

        if n_recent == 0 or self._count == 0:
            self.last_stats = self._empty_stats(gamma)
            return np.empty((0, 5), dtype=np.int64)

        # (2026-07-07) Tier 1 -- ensure the persistent pair-seen hash set
        # (grown-in-place, never freed between calls) has enough capacity
        # for this batch, then clear it (O(capacity) memset) instead of the
        # previous free-then-malloc-then-zero cycle on every single call.
        wanted_pair_seen_cap = max(1024, n_recent * 64)
        if self._pair_seen_cap_buf == 0:
            if not _pair_seen_init(&self._pair_seen_occupied_buf, &self._pair_seen_a_buf,
                                    &self._pair_seen_b_buf, &self._pair_seen_cap_buf,
                                    wanted_pair_seen_cap):
                raise MemoryError()
        else:
            while self._pair_seen_cap_buf < wanted_pair_seen_cap:
                if not _pair_seen_grow(&self._pair_seen_occupied_buf, &self._pair_seen_a_buf,
                                        &self._pair_seen_b_buf, &self._pair_seen_cap_buf):
                    raise MemoryError()
            _pair_seen_clear(self._pair_seen_occupied_buf, self._pair_seen_cap_buf)

        cdef double[:, :] vectors = self._vectors
        cdef uint8_t[:] alive = self._alive
        cdef int64_t[:] window_idx = self._window_idx
        cdef int64_t[:] sid_idx = self._sid_idx
        cdef int64_t[:] sid_rank = self._sid_rank
        cdef int64_t[:] time_idx = self._time
        cdef int64_t[:] window_size = self._window_size

        for i in range(n_recent):
            q_entry = <int64_t>recent_entry_ids[i]
            if q_entry < 0 or q_entry >= self._count:
                continue
            if not alive[q_entry]:
                continue
            q_win = window_idx[q_entry]
            q_sid = sid_idx[q_entry]
            q_rank = sid_rank[q_entry]
            q_time = time_idx[q_entry]
            q_w = window_size[q_entry]

            n_visited = self._search_raw(vectors[q_entry], self._ef_search, signed_abs,
                                          visited_ids, visited_scores, search_cap,
                                          &edges_scanned, &dead_skipped, q_entry, False)
            if n_visited < 0:
                failed = True
                break
            total_visited_live += n_visited
            total_edges_scanned += edges_scanned
            total_dead_skipped += dead_skipped
            if n_visited < self._min_candidates:
                queries_with_too_few_live_nodes += 1
            for k in range(n_visited):
                if visited_scores[k] > best_score_seen:
                    best_score_seen = visited_scores[k]

            sel_count = self._select_candidates(visited_scores, visited_ids, n_visited, gamma,
                                                 sel_scores, sel_ids, &thresh_count, &topk_count)
            total_threshold += thresh_count
            total_topk += topk_count

            for k in range(sel_count):
                cand_row = sel_ids[k]
                if not (time_idx[cand_row] >= self._min_valid_time
                        and window_idx[cand_row] != q_win
                        and window_size[cand_row] == q_w
                        and not (sid_idx[cand_row] == q_sid and time_idx[cand_row] == q_time)):
                    continue
                total_valid += 1
                _canonical_window_pair(
                    q_win, q_sid, q_rank, q_time,
                    window_idx[cand_row], sid_idx[cand_row], sid_rank[cand_row], time_idx[cand_row],
                    &pair_a, &pair_b,
                )
                if not _pair_seen_insert(&self._pair_seen_occupied_buf, &self._pair_seen_a_buf,
                                         &self._pair_seen_b_buf, &self._pair_seen_cap_buf,
                                         &pair_seen_count, pair_a, pair_b, &failed):
                    if failed:
                        break
                    duplicate_pre_dot_pairs += 1
                    continue
                unique_pre_dot_pairs += 1
                sum_score_returned += sel_scores[k]
                after_similarity += 1
                if not _append_pair(&self._pair_out_buf, &count, &self._pair_out_cap, q_win, window_idx[cand_row]):
                    failed = True
                    break
            if failed:
                break

        if failed:
            raise MemoryError()

        pairs_before_dedupe = count
        count = _dedupe_pair_buffer(self._pair_out_buf, count)
        pairs_after_dedupe = count

        self.last_stats = {
            # Standard fields, same names/semantics every other backend uses,
            # so candidate_search_stats() needs no changes for these.
            "num_index_candidates": int(total_visited_live),
            "num_valid_index_candidates": int(total_valid),
            "num_unique_index_candidates": int(total_valid),
            "num_duplicate_index_candidates": 0,
            "num_unique_pre_dot_pairs": int(unique_pre_dot_pairs),
            "num_duplicate_pre_dot_pairs": int(duplicate_pre_dot_pairs),
            # (2026-07-06) InstinctIndex computes a node's score (the dot
            # product) *during* graph traversal itself -- there is no
            # separate later "dot check" stage like the coordinate-range
            # backends have, so num_dot_checks is deliberately the same
            # count as num_index_candidates here, not a smaller subset.
            "num_dot_checks": int(total_visited_live),
            "num_distance_checks": 0,
            "num_after_similarity": int(after_similarity),
            "num_after_dot": int(after_similarity),
            "num_pairs_before_dedupe": int(pairs_before_dedupe),
            "num_pairs_after_dedupe": int(pairs_after_dedupe),
            "num_rows": 0,
            "num_recent_queries": int(n_recent),
            "num_entries": int(self._alive_count),
            "num_blocks": 1,
            "gamma": float(gamma),
            "tau": 0.0,
            # InstinctIndex-specific fields (per-batch aggregates -- see
            # docs/implementation_log.md: no per-query metrics exist
            # anywhere in this codebase, so these are summed/maxed over the
            # whole recent_entry_ids batch, matching every other backend's
            # granularity, not true per-query instrumentation).
            "instinct_visited_nodes": int(total_visited_live + total_dead_skipped),
            "instinct_visited_live_nodes": int(total_visited_live),
            "instinct_dead_nodes_skipped": int(total_dead_skipped),
            "instinct_edges_scanned": int(total_edges_scanned),
            "instinct_candidates_returned": int(after_similarity),
            "instinct_threshold_candidates": int(total_threshold),
            "instinct_topk_candidates": int(total_topk),
            "instinct_best_score_seen": float(best_score_seen) if n_recent > 0 else 0.0,
            "instinct_mean_score_returned": float(sum_score_returned / after_similarity) if after_similarity > 0 else 0.0,
            "instinct_query_time": float(time.perf_counter() - t0),
            "instinct_queries_with_too_few_live_nodes": int(queries_with_too_few_live_nodes),
            "instinct_num_nodes_total": int(self._count),
            "instinct_num_nodes_alive": int(self._alive_count),
            "instinct_dead_node_ratio": float(self._dead_count) / float(self._count) if self._count > 0 else 0.0,
        }
        if count <= 0:
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((count, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr
        out_i = 0
        for i in range(count):
            a = self._pair_out_buf[2 * i]
            b = self._pair_out_buf[2 * i + 1]
            if a < 0 or b < 0 or a >= self._win_capacity or b >= self._win_capacity:
                continue
            size_a = win_w[a]
            size_b = win_w[b]
            if size_a != size_b:
                continue
            sid_a = win_sid_idx[a]
            sid_b = win_sid_idx[b]
            if sid_a < 0 or sid_b < 0:
                continue
            rank_a = win_sid_rank[a]
            rank_b = win_sid_rank[b]
            time_a = win_time[a]
            time_b = win_time[b]
            if sid_a == sid_b:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                if time_a >= time_b:
                    row_view[out_i, 2] = time_a
                    row_view[out_i, 3] = time_b
                else:
                    row_view[out_i, 2] = time_b
                    row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            elif time_a == time_b:
                if rank_a <= rank_b:
                    row_view[out_i, 0] = sid_a
                    row_view[out_i, 1] = sid_b
                else:
                    row_view[out_i, 0] = sid_b
                    row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            elif time_a < time_b:
                row_view[out_i, 0] = sid_b
                row_view[out_i, 1] = sid_a
                row_view[out_i, 2] = time_b
                row_view[out_i, 3] = time_a
                row_view[out_i, 4] = size_a
            else:
                row_view[out_i, 0] = sid_a
                row_view[out_i, 1] = sid_b
                row_view[out_i, 2] = time_a
                row_view[out_i, 3] = time_b
                row_view[out_i, 4] = size_a
            out_i += 1
        if out_i == count:
            return rows
        return rows[:out_i, :]

    cdef object _empty_stats(self, double gamma):
        return {
            "num_index_candidates": 0, "num_valid_index_candidates": 0,
            "num_unique_index_candidates": 0, "num_duplicate_index_candidates": 0,
            "num_unique_pre_dot_pairs": 0, "num_duplicate_pre_dot_pairs": 0,
            "num_dot_checks": 0, "num_distance_checks": 0,
            "num_after_similarity": 0, "num_after_dot": 0,
            "num_pairs_before_dedupe": 0, "num_pairs_after_dedupe": 0,
            "num_rows": 0, "num_recent_queries": 0,
            "num_entries": int(self._alive_count), "num_blocks": 1,
            "gamma": float(gamma), "tau": 0.0,
            "instinct_visited_nodes": 0, "instinct_visited_live_nodes": 0,
            "instinct_dead_nodes_skipped": 0, "instinct_edges_scanned": 0,
            "instinct_candidates_returned": 0, "instinct_threshold_candidates": 0,
            "instinct_topk_candidates": 0, "instinct_best_score_seen": 0.0,
            "instinct_mean_score_returned": 0.0, "instinct_query_time": 0.0,
            "instinct_queries_with_too_few_live_nodes": 0,
            "instinct_num_nodes_total": int(self._count),
            "instinct_num_nodes_alive": int(self._alive_count),
            "instinct_dead_node_ratio": float(self._dead_count) / float(self._count) if self._count > 0 else 0.0,
        }


cdef class HybridValidationCache:
    cdef object _occupied_arr
    cdef object _key_s1_arr
    cdef object _key_s2_arr
    cdef object _key_lag_arr
    cdef object _key_w_arr
    cdef object _prev_t1_arr
    cdef object _prev_t2_arr
    cdef object _stats_arr
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _count
    cdef double _repeat_rate_ema
    cdef bint _active

    def __cinit__(self, Py_ssize_t initial_capacity=1024):
        self._capacity = 0
        self._count = 0
        self._repeat_rate_ema = 0.0
        self._active = False
        self._allocate(initial_capacity)

    cdef Py_ssize_t _round_capacity(self, Py_ssize_t requested):
        cdef Py_ssize_t cap = 16
        if requested < 16:
            requested = 16
        while cap < requested:
            cap <<= 1
        return cap

    cdef void _allocate(self, Py_ssize_t requested):
        cdef Py_ssize_t cap = self._round_capacity(requested)
        self._occupied_arr = np.zeros(cap, dtype=np.uint8)
        self._key_s1_arr = np.empty(cap, dtype=np.int64)
        self._key_s2_arr = np.empty(cap, dtype=np.int64)
        self._key_lag_arr = np.empty(cap, dtype=np.int64)
        self._key_w_arr = np.empty(cap, dtype=np.int64)
        self._prev_t1_arr = np.empty(cap, dtype=np.int64)
        self._prev_t2_arr = np.empty(cap, dtype=np.int64)
        self._stats_arr = np.empty((cap, 10), dtype=np.float64)
        self._capacity = cap
        self._count = 0

    cpdef clear(self):
        self._allocate(16)
        self._repeat_rate_ema = 0.0
        self._active = False

    property repeat_rate_ema:
        def __get__(self):
            return self._repeat_rate_ema

    property active:
        def __get__(self):
            return bool(self._active)

    cdef void _validate_from_stats(self,
                                   long n,
                                   double sx,
                                   double sy,
                                   double sx2,
                                   double sy2,
                                   double sx3,
                                   double sy3,
                                   double sx4,
                                   double sy4,
                                   double sxy,
                                   double corr_threshold,
                                   bint neg_corr,
                                   double std_thresh,
                                   double kurt_thresh,
                                   bint *is_corr,
                                   double *corr,
                                   double *dist,
                                   bint *is_const,
                                   bint *is_spiked):
        cdef double var_x
        cdef double var_y
        cdef double denom
        cdef double dist_sq
        cdef double varx
        cdef double vary
        cdef double mu4
        cdef double kurt
        is_corr[0] = False
        is_const[0] = False
        is_spiked[0] = False
        corr[0] = float("nan")
        dist[0] = float("inf")
        if n <= 0:
            is_const[0] = True
            return
        var_x = sx2 - (sx * sx) / n
        var_y = sy2 - (sy * sy) / n
        if var_x < 0.0:
            var_x = 0.0
        if var_y < 0.0:
            var_y = 0.0
        dist_sq = sx2 + sy2 - 2.0 * sxy
        if dist_sq < 0.0:
            dist_sq = 0.0
        dist[0] = sqrt(dist_sq)
        is_const[0] = (var_x <= (std_thresh * std_thresh) * n) or (var_y <= (std_thresh * std_thresh) * n)
        if is_const[0]:
            dist[0] = float("inf")
            return
        if n >= 4:
            varx = var_x / n
            if varx > 0.0:
                mu4 = _central_mu4_from_sums(sx, sx2, sx3, sx4, n)
                kurt = (mu4 / n) / (varx * varx) - 3.0
                if kurt > kurt_thresh:
                    is_spiked[0] = True
            vary = var_y / n
            if not is_spiked[0] and vary > 0.0:
                mu4 = _central_mu4_from_sums(sy, sy2, sy3, sy4, n)
                kurt = (mu4 / n) / (vary * vary) - 3.0
                if kurt > kurt_thresh:
                    is_spiked[0] = True
        if is_spiked[0]:
            dist[0] = float("inf")
            return
        denom = sqrt(var_x * var_y)
        if denom != 0.0:
            corr[0] = (sxy - (sx * sy) / n) / denom
            if corr[0] > 1.0:
                corr[0] = 1.0
            elif corr[0] < -1.0:
                corr[0] = -1.0
        if corr[0] == corr[0]:
            if neg_corr:
                is_corr[0] = fabs(corr[0]) >= corr_threshold
            else:
                is_corr[0] = corr[0] >= corr_threshold

    def validate_pairs(self,
                       double[:, :] data,
                       long[:] series1,
                       long[:] series2,
                       long[:] curr_t1,
                       long[:] curr_t2,
                       long[:] window_size,
                       long base_index,
                       long window_step,
                       double corr_threshold,
                       bint neg_corr,
                       double min_repeat_rate=0.25,
                       double disable_rate=0.125,
                       double ema_alpha=0.25,
                       long min_candidates=256,
                       double std_thresh=1e-3,
                       double kurt_thresh=5.0,
                       double[:] current_window_sums=None,
                       double[:] current_window_sums_sq=None,
                       double[:] current_window_sums_cu=None,
                       double[:] current_window_sums_qu=None,
                       long current_window_size=-1):
        cdef Py_ssize_t n_items = series1.shape[0]
        if (
            series2.shape[0] != n_items
            or curr_t1.shape[0] != n_items
            or curr_t2.shape[0] != n_items
            or window_size.shape[0] != n_items
        ):
            raise ValueError("hybrid cache arrays must have matching lengths")

        cdef uint8_t[:] old_occ = self._occupied_arr
        cdef long[:] old_s1 = self._key_s1_arr
        cdef long[:] old_s2 = self._key_s2_arr
        cdef long[:] old_lag = self._key_lag_arr
        cdef long[:] old_w = self._key_w_arr
        cdef long[:] old_t1 = self._prev_t1_arr
        cdef long[:] old_t2 = self._prev_t2_arr
        cdef double[:, :] old_stats = self._stats_arr
        cdef Py_ssize_t old_capacity = self._capacity

        cdef Py_ssize_t next_capacity = self._round_capacity(max(16, n_items * 4))
        cdef np.ndarray[np.uint8_t, ndim=1] next_occ_arr = np.zeros(next_capacity, dtype=np.uint8)
        cdef np.ndarray[np.int64_t, ndim=1] next_s1_arr = np.empty(next_capacity, dtype=np.int64)
        cdef np.ndarray[np.int64_t, ndim=1] next_s2_arr = np.empty(next_capacity, dtype=np.int64)
        cdef np.ndarray[np.int64_t, ndim=1] next_lag_arr = np.empty(next_capacity, dtype=np.int64)
        cdef np.ndarray[np.int64_t, ndim=1] next_w_arr = np.empty(next_capacity, dtype=np.int64)
        cdef np.ndarray[np.int64_t, ndim=1] next_t1_arr = np.empty(next_capacity, dtype=np.int64)
        cdef np.ndarray[np.int64_t, ndim=1] next_t2_arr = np.empty(next_capacity, dtype=np.int64)
        cdef np.ndarray[np.float64_t, ndim=2] next_stats_arr = np.empty((next_capacity, 10), dtype=np.float64)
        cdef uint8_t[:] next_occ = next_occ_arr
        cdef long[:] next_s1 = next_s1_arr
        cdef long[:] next_s2 = next_s2_arr
        cdef long[:] next_lag = next_lag_arr
        cdef long[:] next_w = next_w_arr
        cdef long[:] next_t1 = next_t1_arr
        cdef long[:] next_t2 = next_t2_arr
        cdef double[:, :] next_stats = next_stats_arr

        cdef Py_ssize_t i, j, slot, next_slot
        cdef long s1, s2, t1, t2, lag, w, step
        cdef long prev_t1, prev_t2, start1, start2, n_series, n_cols
        cdef bint found
        cdef Py_ssize_t repeat_count = 0
        cdef double rate
        cdef bint active_now
        cdef double sx, sy, sx2, sy2, sx3, sy3, sx4, sy4, sxy
        cdef double old_x, old_y, new_x, new_y
        cdef bint used_hybrid, valid_pair, is_corr, is_const, is_spiked
        cdef double corr, dist
        cdef Py_ssize_t unique_count = 0
        cdef bint have_current_window_cache = (
            current_window_sums is not None
            and current_window_sums_sq is not None
            and current_window_sums_cu is not None
            and current_window_sums_qu is not None
            and current_window_size > 0
        )
        cdef bint side1_is_current, side2_is_current
        n_series = data.shape[0]
        n_cols = data.shape[1]

        for i in range(n_items):
            s1 = series1[i]
            s2 = series2[i]
            t1 = curr_t1[i]
            t2 = curr_t2[i]
            w = window_size[i]
            lag = t1 - t2
            if old_capacity > 0:
                slot = _hybrid_find_slot(old_occ, old_s1, old_s2, old_lag, old_w, old_capacity, s1, s2, lag, w, &found)
                if found:
                    repeat_count += 1

        rate = (<double>repeat_count / <double>n_items) if n_items > 0 else 0.0
        if ema_alpha < 0.0:
            ema_alpha = 0.0
        elif ema_alpha > 1.0:
            ema_alpha = 1.0
        self._repeat_rate_ema = ema_alpha * rate + (1.0 - ema_alpha) * self._repeat_rate_ema
        if n_items < min_candidates:
            active_now = False
        elif self._active:
            active_now = self._repeat_rate_ema >= disable_rate
        else:
            active_now = self._repeat_rate_ema >= min_repeat_rate

        results = []
        for i in range(n_items):
            s1 = series1[i]
            s2 = series2[i]
            t1 = curr_t1[i]
            t2 = curr_t2[i]
            w = window_size[i]
            lag = t1 - t2
            used_hybrid = False
            valid_pair = True
            sx = sy = 0.0
            sx2 = sy2 = 0.0
            sx3 = sy3 = 0.0
            sx4 = sy4 = 0.0
            sxy = 0.0

            if (
                s1 < 0
                or s1 >= n_series
                or s2 < 0
                or s2 >= n_series
                or w <= 0
            ):
                valid_pair = False

            if valid_pair and active_now and old_capacity > 0:
                slot = _hybrid_find_slot(old_occ, old_s1, old_s2, old_lag, old_w, old_capacity, s1, s2, lag, w, &found)
                if found:
                    prev_t1 = old_t1[slot]
                    prev_t2 = old_t2[slot]
                    step = t1 - prev_t1
                    if step > 0 and step < w and step == window_step and (t2 - prev_t2) == step:
                        start1 = prev_t1 - base_index
                        start2 = prev_t2 - base_index
                        if (
                            start1 >= 0
                            and start2 >= 0
                            and (t1 - base_index) >= 0
                            and (t2 - base_index) >= 0
                            and (t1 - base_index) + w <= n_cols
                            and (t2 - base_index) + w <= n_cols
                        ):
                            sx = old_stats[slot, 1]
                            sy = old_stats[slot, 2]
                            sx2 = old_stats[slot, 3]
                            sy2 = old_stats[slot, 4]
                            sx3 = old_stats[slot, 5]
                            sy3 = old_stats[slot, 6]
                            sx4 = old_stats[slot, 7]
                            sy4 = old_stats[slot, 8]
                            sxy = old_stats[slot, 9]
                            for j in range(step):
                                old_x = data[s1, start1 + j]
                                old_y = data[s2, start2 + j]
                                new_x = data[s1, (t1 - base_index) + w - step + j]
                                new_y = data[s2, (t2 - base_index) + w - step + j]
                                sx += new_x - old_x
                                sy += new_y - old_y
                                sx2 += new_x * new_x - old_x * old_x
                                sy2 += new_y * new_y - old_y * old_y
                                sx3 += new_x * new_x * new_x - old_x * old_x * old_x
                                sy3 += new_y * new_y * new_y - old_y * old_y * old_y
                                sx4 += new_x * new_x * new_x * new_x - old_x * old_x * old_x * old_x
                                sy4 += new_y * new_y * new_y * new_y - old_y * old_y * old_y * old_y
                                sxy += new_x * new_y - old_x * old_y
                            used_hybrid = True

            if valid_pair and not used_hybrid:
                start1 = t1 - base_index
                start2 = t2 - base_index
                if start1 < 0 or start2 < 0 or start1 + w > n_cols or start2 + w > n_cols:
                    valid_pair = False
                else:
                    # (2026-07-06) Every candidate pair reaching validation
                    # necessarily has at least one side anchored at the
                    # just-arrived "current" window (the human's observation).
                    # When a side's window exactly matches what the
                    # per-series incrementally-maintained current-window
                    # moments already cover (same length, same end position
                    # as the live data buffer), its own sx/sx2/sx3/sx4 (or
                    # sy/...) are read from that cache instead of
                    # re-accumulated here -- cutting up to ~4 of 9 per-row
                    # flops for a one-sided match, ~8 of 9 if both sides
                    # happen to be current (e.g. lag 0 between two
                    # just-arrived windows). sxy always needs a fresh pass
                    # since no per-pair cross-moment cache exists for
                    # first-seen pairs (only HybridValidationCache's own
                    # repeat-pair cache has that, handled above). This cache
                    # is only ever passed from the Python side when
                    # preprocessing is disabled, since it is maintained over
                    # raw (pre-differencing) data -- see
                    # docs/implementation_log.md, "point 1: current-window
                    # incremental stats reuse in exact Pearson validation".
                    side1_is_current = (
                        have_current_window_cache
                        and w == current_window_size
                        and start1 + w == n_cols
                        and s1 < current_window_sums.shape[0]
                    )
                    side2_is_current = (
                        have_current_window_cache
                        and w == current_window_size
                        and start2 + w == n_cols
                        and s2 < current_window_sums.shape[0]
                    )
                    if side1_is_current:
                        sx = current_window_sums[s1]
                        sx2 = current_window_sums_sq[s1]
                        sx3 = current_window_sums_cu[s1]
                        sx4 = current_window_sums_qu[s1]
                    if side2_is_current:
                        sy = current_window_sums[s2]
                        sy2 = current_window_sums_sq[s2]
                        sy3 = current_window_sums_cu[s2]
                        sy4 = current_window_sums_qu[s2]
                    for j in range(w):
                        new_x = data[s1, start1 + j]
                        new_y = data[s2, start2 + j]
                        if not side1_is_current:
                            sx += new_x
                            sx2 += new_x * new_x
                            sx3 += new_x * new_x * new_x
                            sx4 += new_x * new_x * new_x * new_x
                        if not side2_is_current:
                            sy += new_y
                            sy2 += new_y * new_y
                            sy3 += new_y * new_y * new_y
                            sy4 += new_y * new_y * new_y * new_y
                        sxy += new_x * new_y

            if not valid_pair:
                results.append((False, False, float("nan"), float("inf"), False, False, False))
                continue

            self._validate_from_stats(
                w,
                sx,
                sy,
                sx2,
                sy2,
                sx3,
                sy3,
                sx4,
                sy4,
                sxy,
                corr_threshold,
                neg_corr,
                std_thresh,
                kurt_thresh,
                &is_corr,
                &corr,
                &dist,
                &is_const,
                &is_spiked,
            )

            next_slot = _hybrid_find_slot(next_occ, next_s1, next_s2, next_lag, next_w, next_capacity, s1, s2, lag, w, &found)
            if not found:
                unique_count += 1
            next_occ[next_slot] = 1
            next_s1[next_slot] = s1
            next_s2[next_slot] = s2
            next_lag[next_slot] = lag
            next_w[next_slot] = w
            next_t1[next_slot] = t1
            next_t2[next_slot] = t2
            next_stats[next_slot, 0] = w
            next_stats[next_slot, 1] = sx
            next_stats[next_slot, 2] = sy
            next_stats[next_slot, 3] = sx2
            next_stats[next_slot, 4] = sy2
            next_stats[next_slot, 5] = sx3
            next_stats[next_slot, 6] = sy3
            next_stats[next_slot, 7] = sx4
            next_stats[next_slot, 8] = sy4
            next_stats[next_slot, 9] = sxy

            results.append((True, bool(is_corr), corr, dist, bool(is_const), bool(is_spiked), bool(used_hybrid)))

        self._occupied_arr = next_occ_arr
        self._key_s1_arr = next_s1_arr
        self._key_s2_arr = next_s2_arr
        self._key_lag_arr = next_lag_arr
        self._key_w_arr = next_w_arr
        self._prev_t1_arr = next_t1_arr
        self._prev_t2_arr = next_t2_arr
        self._stats_arr = next_stats_arr
        self._capacity = next_capacity
        self._count = unique_count
        self._active = active_now

        return {
            "active": bool(active_now),
            "repeat_count": int(repeat_count),
            "repeat_rate": float(rate),
            "repeat_rate_ema": float(self._repeat_rate_ema),
            "results": results,
        }
