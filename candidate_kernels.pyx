# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import time
import itertools
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, floor, ceil, log2, log, exp, lgamma, acos, atan2, cos, sin
from libc.math cimport M_PI, NAN
from libc.stdlib cimport malloc, calloc, realloc, free
from libc.stdint cimport int64_t, uint64_t, uint8_t, int32_t
from cython.parallel cimport prange, threadid

np.import_array()

cdef double _PI = 3.14159265358979323846


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


cdef inline int64_t _popcount64_swar(uint64_t x) nogil:
    # (2026-07-10) Branch-free O(1) SWAR popcount for HammingExactIndex's
    # per-candidate Hamming distance scan -- deliberately
    # NOT the same helper as _popcount64 above (Kernighan's bit-counting
    # trick, O(popcount(x)) loop iterations -- ~32 for a typical balanced
    # 64-bit sign word). Using the slow one here was a real, measured
    # performance bug: the first end-to-end benchmark of HammingExactIndex
    # came out SLOWER than SignLSHBandIndex (0.761x) despite an isolated
    # microbenchmark of the same scan in pure Python/numpy beating it
    # (25.3us vs 65.6us/query) -- root-caused to this exact call, not the
    # scan structure itself. See docs/implementation_log.md.
    x = x - ((x >> 1) & <uint64_t>0x5555555555555555ULL)
    x = (x & <uint64_t>0x3333333333333333ULL) + ((x >> 2) & <uint64_t>0x3333333333333333ULL)
    x = (x + (x >> 4)) & <uint64_t>0x0f0f0f0f0f0f0f0fULL
    return <int64_t>((x * <uint64_t>0x0101010101010101ULL) >> 56)


# (2026-09-13) Option-4 investigation: these three helpers exist ONLY so
# _find_pair_rows_meta_parallel's prange loop body never uses an in-place
# operator (+=/-=) on a variable declared at the enclosing function's top level --
# Cython infers ANY such variable as a cross-iteration OpenMP "reduction" for the
# whole parallel region (summed across all iterations/threads at the end), which
# is wrong here and outright refuses to compile ("Cannot read reduction variable
# in loop body") since these are meant to be fresh per-iteration scratch, not an
# accumulation across queries. Moving each accumulation into its own nogil
# function sidesteps the issue entirely -- the accumulator lives on that
# function's own stack frame, invisible to the caller's prange analysis; the
# caller only ever does a single plain `=` assignment of the return value,
# mirroring _row_l2_sq_until's existing pattern just above (`acc = acc + ...`,
# never `+=`) already used successfully inside a prange loop earlier in this file.
cdef inline Py_ssize_t _lsh_scan_touched_bands(int32_t[:, :] band_keys,
                                               int32_t[:, :] neg_band_keys,
                                               int64_t **post_members,
                                               int32_t *post_count,
                                               int64_t *visited_stamp,
                                               int64_t *touched_buf,
                                               Py_ssize_t n_bands,
                                               Py_ssize_t n_variants,
                                               Py_ssize_t band_mode,
                                               int64_t complement_mask,
                                               Py_ssize_t bucket_count,
                                               int64_t q_entry,
                                               int64_t stamp,
                                               Py_ssize_t max_candidates_per_query) nogil:
    cdef Py_ssize_t band_idx, variant, flat, n_members, mi
    cdef Py_ssize_t touched_count = 0
    cdef int64_t key, cur_key, node
    cdef bint budget_hit = False
    cdef int64_t *members
    for band_idx in range(n_bands):
        if budget_hit:
            break
        key = band_keys[q_entry, band_idx]
        for variant in range(n_variants):
            if budget_hit:
                break
            if band_mode == 0:
                cur_key = key if variant == 0 else (key ^ complement_mask)
            else:
                cur_key = key if variant == 0 else neg_band_keys[q_entry, band_idx]
            flat = band_idx * bucket_count + cur_key
            n_members = post_count[flat]
            members = post_members[flat]
            for mi in range(n_members):
                node = members[mi]
                if node != q_entry and visited_stamp[node] != stamp:
                    visited_stamp[node] = stamp
                    touched_buf[touched_count] = node
                    touched_count = touched_count + 1
                    if max_candidates_per_query > 0 and touched_count >= max_candidates_per_query:
                        budget_hit = True
                        break
    return touched_count


cdef inline Py_ssize_t _hamming_best_dist(int64_t[:, ::1] words,
                                          uint64_t[:] word_masks,
                                          Py_ssize_t n_words,
                                          int64_t q_entry,
                                          int64_t cand,
                                          bint signed_abs) nogil:
    cdef Py_ssize_t hw
    cdef Py_ssize_t hpos = 0, hneg = 0, hbest
    cdef uint64_t qw_word, cw_word, xw_mask
    for hw in range(n_words):
        qw_word = <uint64_t>words[q_entry, hw]
        cw_word = <uint64_t>words[cand, hw]
        xw_mask = word_masks[hw]
        hpos = hpos + _popcount64_swar((qw_word ^ cw_word) & xw_mask)
        if signed_abs:
            hneg = hneg + _popcount64_swar((qw_word ^ (~cw_word)) & xw_mask)
    hbest = hpos
    if signed_abs and hneg < hbest:
        hbest = hneg
    return hbest


cdef inline double _dot_score(double[:, ::1] vectors,
                              int64_t q_entry,
                              int64_t cand,
                              Py_ssize_t n_vectors) nogil:
    cdef Py_ssize_t d
    cdef double score = 0.0
    for d in range(n_vectors):
        score = score + vectors[q_entry, d] * vectors[cand, d]
    return score


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


def fast_corr_and_dist_batch(double[:, :] x, double[:, :] y):
    """(2026-09-08) Batched sibling of fast_corr_and_dist -- IDENTICAL per-row math (plain
    Pearson correlation + Euclidean distance, no constant/spike filtering -- that check
    already happens once per WINDOW, not per pair, via CorrTrack._proxy_window_is_valid before
    a window ever reaches here), just computed for every row of two stacked (n_items,
    window_size) arrays in one nogil Cython loop instead of n_items separate Python-level
    calls. Built to fix a real, measured bottleneck in CorrTrack_optimize.
    _prepare_proxy_anchor_reference (the proxy-anchor hyperopt's own ground-truth-labeling
    step, part of the production hyperopt pipeline itself, not test/benchmark-only code):
    that method calls fast_corr_and_dist once per candidate pair in a plain Python loop --
    ~25us/pair measured, dominated by per-call/per-iteration Python overhead, not the
    correlation math itself (which was already Cython). Deliberately NOT validate_corr_batch
    (also in this file): that function adds is_const/is_spiked filtering at the PAIR level,
    which would silently duplicate (and could disagree with, at different threshold
    granularity) the WINDOW-level filtering _prepare_proxy_anchor_reference already applies
    via is_valid -- using it here would risk quietly changing which pairs count as ground
    truth. This function is a pure, mechanical batching of fast_corr_and_dist's own exact
    per-pair formula, nothing more, so it cannot change ground-truth semantics.

    Returns (corr, dist) as two float64 arrays, one entry per row -- same NaN/clamping
    behavior as fast_corr_and_dist (corr clamped to [-1,1]; NaN when either side's variance is
    zero, which cannot happen for a window that already passed is_valid, but handled
    identically regardless)."""
    cdef Py_ssize_t n_items = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    if y.shape[0] != n_items or y.shape[1] != n:
        raise ValueError("x and y must have the same shape")
    cdef np.ndarray[np.float64_t, ndim=1] corr_out = np.empty(n_items, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dist_out = np.empty(n_items, dtype=np.float64)
    cdef double[:] corr_mv = corr_out
    cdef double[:] dist_mv = dist_out
    cdef Py_ssize_t i, j
    cdef double sx, sy, sum_xy, sum_xx, sum_yy, xi, yi
    cdef double mean_x, mean_y, var_x, var_y, denom, corr, dist_sq, dist
    if n_items == 0:
        return corr_out, dist_out
    if n == 0:
        for i in range(n_items):
            corr_mv[i] = float("nan")
            dist_mv[i] = float("inf")
        return corr_out, dist_out
    with nogil:
        for i in range(n_items):
            sx = 0.0
            sy = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            for j in range(n):
                xi = x[i, j]
                yi = y[i, j]
                sx += xi
                sy += yi
                sum_xy += xi * yi
                sum_xx += xi * xi
                sum_yy += yi * yi
            mean_x = sx / n
            mean_y = sy / n
            var_x = sum_xx - (sx * sx) / n
            var_y = sum_yy - (sy * sy) / n
            if var_x < 0.0:
                var_x = 0.0
            if var_y < 0.0:
                var_y = 0.0
            denom = sqrt(var_x * var_y)
            if denom == 0.0:
                corr = NAN
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
            corr_mv[i] = corr
            dist_mv[i] = dist
    return corr_out, dist_out


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


cdef inline bint _passes_dot_gamma_gate(double score, double gamma, bint signed_abs, bint apply_filter) nogil:
    if not apply_filter:
        return True
    if signed_abs:
        return fabs(score) >= gamma
    return score >= gamma


# (2026-09-0x) Overlap-corrected LSH band sizing -- replaces the independent-bands b_min formula
# below with one that accounts for SignLSHBandIndex's own band construction: each band's
# band_width bits are drawn without replacement WITHIN a band but WITH replacement (deliberate
# overlap) ACROSS bands, from the SAME fixed pool of n_vectors bits (see band_dims construction
# in _finalize_sizing). The old formula treated the n_bands band-successes as independent coin
# flips; they aren't, since they all draw from the same finite bit pool -- a bit that mismatches
# for a given pair drags down every band that happens to include it. Verified against the
# 83-point Sobol sweep's 996-trial calibration grid: the old formula overestimated real recall by
# +7.0 points on average (R^2=-1.08, i.e. worse than just predicting the mean) even on trials with
# enough true positives to trust the recall measurement; this corrected formula's mean prediction
# is within -0.3 points of actual (R^2=0.10 -- removes the systematic bias, doesn't explain all
# the remaining trial-to-trial noise). See docs/implementation_log.md's 2026-09-03 entry for the
# full derivation and validation numbers.
#
# Derivation: let t = how many of the K=n_vectors bits actually match for a given pair
# (t ~ Binomial(K, p_bit)). GIVEN t, each band -- an independently-drawn random w-subset of the K
# bits -- succeeds (all w of its bits match) with probability q(t) = C(t,w)/C(K,w) (a uniformly
# random w-subset landing entirely inside the t matching bits); conditional on t, the n_bands
# bands' successes ARE independent (each band's own subset is drawn independently of every other
# band's), so P(>=1 success | t) = 1-(1-q(t))^n_bands. Averaging over t's own binomial
# distribution gives the exact population recall. (The old formula's p_r = p_bit**band_width is
# recovered exactly as E_t[q(t)] -- for a SINGLE band this is unbiased; the bias only appears once
# multiple overlapping bands are OR-combined, since 1-(1-x)^n is concave in x, so by Jensen's
# inequality E[1-(1-q(t))^n] <= 1-(1-E[q(t)])^n always -- the old formula is a proven upper bound
# on the true recall whenever bands overlap, never a coincidental overestimate.)
cdef inline double _log_binom(Py_ssize_t n, Py_ssize_t k) nogil:
    if k < 0 or k > n:
        return -1e300  # effectively log(0) -- guards a call-site bug rather than raising in a nogil context
    return lgamma(<double>n + 1.0) - lgamma(<double>k + 1.0) - lgamma(<double>(n - k) + 1.0)


cdef double _exact_band_recall(Py_ssize_t K, Py_ssize_t w, Py_ssize_t n_bands, double p_bit) nogil:
    """Exact OR-of-overlapping-bands recall for one pair, per the derivation above."""
    if w > K or w < 0:
        return 0.0
    if p_bit <= 0.0:
        return 0.0
    if p_bit >= 1.0:
        return 1.0
    cdef double log_Ckw = _log_binom(K, w)
    cdef double total = 0.0
    cdef Py_ssize_t t
    cdef double log_pmf, q
    for t in range(w, K + 1):
        log_pmf = _log_binom(K, t) + <double>t * log(p_bit) + <double>(K - t) * log(1.0 - p_bit)
        q = exp(_log_binom(t, w) - log_Ckw)
        total += exp(log_pmf) * (1.0 - (1.0 - q) ** n_bands)
    return total


cdef double _band_recall_ceiling(Py_ssize_t K, Py_ssize_t w, double p_bit) nogil:
    """n_bands -> infinity limit of _exact_band_recall: P(T >= w), the hard ceiling no amount of
    bands can exceed, since a band can only succeed if at least w of the K bits actually match.
    Used to detect a (K, w, p_bit) combination for which the target recall is structurally
    unreachable no matter how many bands are used -- distinct from the old formula's degenerate
    p_r==0 case, which conflated 'no finite b_min under the (wrong) independent-bands formula'
    with 'literally unreachable', these are not the same thing in general."""
    if w > K or w < 0:
        return 0.0
    if p_bit <= 0.0:
        return 0.0
    if p_bit >= 1.0:
        return 1.0
    cdef double total = 0.0
    cdef Py_ssize_t t
    for t in range(w, K + 1):
        total += exp(_log_binom(K, t) + <double>t * log(p_bit) + <double>(K - t) * log(1.0 - p_bit))
    return total


cdef Py_ssize_t _corrected_b_min(Py_ssize_t K, Py_ssize_t w, double p_bit, double target_recall) nogil:
    """Smallest n_bands (floored at 3, matching the old formula's own floor) such that
    _exact_band_recall(K, w, n_bands, p_bit) >= target_recall. _exact_band_recall is monotone
    non-decreasing in n_bands (more bands can only add more chances), so a doubling search
    followed by a binary-search refinement finds it exactly. This runs ONCE per index, inside
    _finalize_sizing when observed_m first becomes known -- never per query or per candidate -- so
    its cost (a few dozen to a few hundred evaluations, each O(K) lgamma calls) is irrelevant to
    steady-state throughput."""
    cdef Py_ssize_t degenerate = 100 * K  # mirrors the old formula's own "max_band_width*100" absurd-sentinel pattern for a structurally-unreachable target -- a clear, boundedly-large flag value, not a silent guess
    cdef Py_ssize_t lo = 3
    cdef Py_ssize_t hi = 3
    cdef Py_ssize_t mid
    if w > K or w < 1:
        return degenerate
    if _band_recall_ceiling(K, w, p_bit) < target_recall:
        return degenerate
    while _exact_band_recall(K, w, hi, p_bit) < target_recall:
        hi *= 2
        if hi > degenerate:
            return degenerate
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _exact_band_recall(K, w, mid, p_bit) >= target_recall:
            hi = mid
        else:
            lo = mid
    return hi if hi >= 3 else 3


def debug_exact_band_recall(Py_ssize_t K, Py_ssize_t w, Py_ssize_t n_bands, double p_bit):
    """Test/debug-only Python-callable wrapper around the cdef helpers above -- lets
    docs/implementation_log.md's validation numbers (and any future test) be reproduced
    independently against this exact compiled implementation, not just the .pyx source."""
    return _exact_band_recall(K, w, n_bands, p_bit)


def debug_band_recall_ceiling(Py_ssize_t K, Py_ssize_t w, double p_bit):
    return _band_recall_ceiling(K, w, p_bit)


def debug_corrected_b_min(Py_ssize_t K, Py_ssize_t w, double p_bit, double target_recall):
    return _corrected_b_min(K, w, p_bit, target_recall)


# (2026-09-04) Data-driven safety margin for _corrected_b_min's own remaining approximation
# error. _corrected_b_min is unbiased ON AVERAGE (mean residual +0.003 across 660 reliable
# Sobol-sweep trials, i.e. actual measured recall minus the formula's own prediction) but still
# has real per-config spread around that average (std 0.061) -- meaning a config sized with
# ZERO margin clears its target_recall only about half the time, not reliably. Padding the
# INTERNAL target used by the sizing search (not the target_recall the caller asked for) trades
# a small, quantified extra n_bands cost for a real, checked reliability improvement, instead of
# selection-time logic gambling on whichever config's noisy proxy measurement happened to read
# highest recall (expensive, and not what you actually want -- see docs/implementation_log.md's
# 2026-09-04 entry for the full derivation and the percentile table this was chosen from).
# 0.028 is the empirical 80th percentile of (actual - predicted) recall in that same reliable-
# trial set: at target_recall=0.95, this internal padding pushes the searched-for target to
# 0.978, which is comfortably below the recall ceiling (measured ~1.0 for typical configs) and
# roughly doubles clearing confidence from the ~50% zero-margin baseline to ~80%. Higher
# confidence levels need disproportionately more padding near the ceiling (90% needs +0.055,
# already pushing past 1.0 and therefore unreachable for target_recall=0.95) -- 80% was chosen
# as a meaningfully-improved, still cheaply-achievable point, not the highest checked. This
# value was calibrated specifically at target_recall=0.95 (the only value used throughout the
# Sobol sweep this analysis is based on) -- applying the same ADDITIVE padding at very different
# target_recall values is a reasonable default, not independently re-validated there.
RECALL_SAFETY_MARGIN = 0.028
# Hard cap on the padded internal target so it never reaches/exceeds 1.0 (an invalid
# probability, which would otherwise send _corrected_b_min searching for an unreachable
# recall). 0.999 leaves searching well-defined even when target_recall + RECALL_SAFETY_MARGIN
# would otherwise land at or above 1.0.
RECALL_SAFETY_MARGIN_CAP = 0.999


def compute_lsh_band_width(Py_ssize_t n_series, Py_ssize_t n_lagged_windows,
                            double target_occupancy, Py_ssize_t n_vectors):
    """Pure replica of SignLSHBandIndex._finalize_sizing's own band_width sizing -- given the
    same (m, L, occupancy, n_vectors) any CorrTrack/SignLSHBandIndex construction would use,
    returns band_width WITHOUT building an index or touching any data. _finalize_sizing calls
    this exact function internally, so the two can never drift apart."""
    cdef Py_ssize_t expected_alive = n_series * n_lagged_windows
    cdef Py_ssize_t max_band_width = 24
    cdef Py_ssize_t band_width
    if expected_alive < 1:
        expected_alive = 1
    if max_band_width > n_vectors:
        max_band_width = n_vectors
    band_width = <Py_ssize_t>ceil(log2(<double>expected_alive / target_occupancy))
    if band_width < 3:
        band_width = 3
    if band_width > max_band_width:
        band_width = max_band_width
    return band_width


def compute_lsh_n_bands(Py_ssize_t band_width, double corr_threshold, Py_ssize_t n_vectors,
                         double target_recall, double n_bands_tolerance,
                         double recall_safety_margin=RECALL_SAFETY_MARGIN):
    """Pure replica of SignLSHBandIndex._finalize_sizing's own n_bands sizing (the
    overlap-corrected _corrected_b_min, padded by recall_safety_margin) -- given the same
    inputs any real construction would use, returns the real n_bands WITHOUT building an index.
    Returns None if n_bands_tolerance<=0 (sizing disabled -- caller's own configured
    candidate_lsh_n_bands applies unchanged, exactly _finalize_sizing's own inert-when-disabled
    behavior). This is what lets a caller (e.g. proxy-anchor hyperopt's ranking logic) know a
    config's real theoretical n_bands ahead of time, at zero cost -- no need to run anything or
    introspect a constructed index's internal state, which the proxy-anchor simulation makes
    awkward anyway (it builds a fresh, throwaway CorrTrack per anchor, not one long-lived index
    whose state a caller could just read back).

    (2026-09-04) recall_safety_margin defaults to the calibrated RECALL_SAFETY_MARGIN but is now
    a real, overridable parameter -- see docs/implementation_log.md's 2026-09-04 (c) entry. The
    margin's cost is NOT flat: it compounds with scale (the same +0.028 pad needs 8-11x more
    bands at m=1000-3000 than the pre-margin, pre-overlap-correction formula did, vs. ~3-4x at
    m=150, because the overlap correction's own penalty already worsens as band_width grows with
    m*L, and the margin pushes further into that same steep region). A caller facing real memory
    pressure at large (m, L) can pass 0.0 here to fall back to the un-padded overlap-corrected
    minimum (still a real, validated fix over the original pre-session formula -- just without
    the extra ~80th-percentile reliability margin on top), or any intermediate value."""
    cdef double p_bit
    cdef double padded_target
    cdef Py_ssize_t b_min, new_n_bands
    if n_bands_tolerance <= 0.0:
        return None
    p_bit = 1.0 - acos(min(max(corr_threshold, -1.0), 1.0)) / M_PI
    padded_target = min(target_recall + recall_safety_margin, RECALL_SAFETY_MARGIN_CAP)
    b_min = _corrected_b_min(n_vectors, band_width, p_bit, padded_target)
    new_n_bands = <Py_ssize_t>ceil(n_bands_tolerance * <double>b_min)
    if new_n_bands < 3:
        new_n_bands = 3
    return new_n_bands


def compute_lsh_sizing(Py_ssize_t n_series, Py_ssize_t n_lagged_windows, double target_occupancy,
                        double corr_threshold, Py_ssize_t n_vectors, double target_recall,
                        double n_bands_tolerance, double recall_safety_margin=RECALL_SAFETY_MARGIN):
    """Convenience wrapper: (band_width, n_bands) together, n_bands possibly None (sizing
    disabled). See compute_lsh_band_width/compute_lsh_n_bands for the pieces."""
    band_width = compute_lsh_band_width(n_series, n_lagged_windows, target_occupancy, n_vectors)
    n_bands = compute_lsh_n_bands(band_width, corr_threshold, n_vectors, target_recall, n_bands_tolerance, recall_safety_margin)
    return band_width, n_bands


cdef class SignLSHBandIndex:
    """LSH sign-bit banding pre-filter (candidate_backend="lsh_sign_dot").

    See docs/implementation_log.md, 2026-07-09 "sign_alignment_frac / LSH
    sign-bit banding" and "SignLSHBandIndex: lsh_sign_dot backend" entries
    for the diagnostic that validated this design. Each window's sign(w_hat)
    pattern (n_vectors bits) is split into n_bands independent random bands
    of band_width bits each (bands may overlap -- there is no requirement
    that n_bands*band_width == n_vectors, and the validated setting
    (band_width=8, n_bands=64) deliberately does not partition the 64
    dimensions cleanly). A candidate is touched if it shares a bucket with
    the query in ANY band (OR-of-bands retrieval, not vote-count threshold
    like TopKInvertedIndex) -- both the band's own key and its bitwise
    complement are checked per band, since a negatively-correlated pair
    flips most sign bits.

    This is a genuine two-stage cascade: stage 1 (band bucket lookups) never
    computes a dot product at all. Stage 2 (the dot+gamma gate) is
    OPTIONAL, controlled by apply_dot_filter -- when disabled, every
    touched, alive candidate flows straight through to validation with zero
    dot products computed at this layer; when enabled (default), the exact
    same _passes_dot_gamma_gate() helper InstinctIndex's threshold mode uses
    gates candidates before they become output rows.

    (2026-07-16) Posting lists are FLAT, swap-remove arrays (_post_members/
    _post_capacity/_post_count, one growable int64 buffer per (band,bucket)
    slot, position tracked per-node via _membership_pos), not the earlier
    doubly-linked-list scheme -- query-time traversal walks a tight
    contiguous array per bucket instead of following _membership_next_node
    pointers scattered across a (capacity, n_bands) table, a real cache-
    locality fix flagged as the concrete next lever in an earlier session's
    enumeration-cost discussion. drop_before_time still removes an expiring
    node from every one of its band buckets immediately (O(n_bands),
    swap-last-into-slot, not the earlier doubly-linked unlink) -- not the
    lazy "mark dead, let it linger until a periodic threshold-triggered
    rebuild" scheme this class started with. That earlier scheme is exactly
    what caused a real, confirmed ~104x wall-clock regression (unbounded
    posting-list growth; see docs/implementation_log.md's 2026-07-09/-10
    entries) before the rebuild was added; eager doubly-linked removal
    achieves the same "never walk a dead node" property without a
    threshold heuristic to tune, and without ever letting any dead nodes
    accumulate in the interim.

    (2026-07-10) band_width is NOT a fixed/tunable constructor parameter
    anymore -- it is computed ONCE, automatically. Root cause this fixes:
    with a FIXED bucket_count=2**band_width, touched-candidates-per-query
    was measured to scale LINEARLY with the number of currently-alive
    windows (confirmed directly: touched/alive stayed at a flat ~0.74
    across a 4x range of alive-window counts on the real dataset) -- i.e.
    Theta(N) per query, Theta(N^2) total, the same asymptotic order as
    brute force, just with a smaller constant factor. The number of
    concurrently-alive windows at steady state is exactly m * L (m = series
    count, L = n_lagged_windows = n_lags//window_step+1) -- a quantity
    fixed by configuration, not something that drifts with stream length
    (confirmed: alive_count plateaus at exactly m*L and stays there for the
    rest of any run, however long). So band_width only needs to be sized
    correctly ONCE, from the (m, L) the deployment will actually run at --
    not continuously re-adapted.

    m is not known at __cinit__ time (this class has no notion of "the
    dataset" -- only CorrTrack, higher up, sees the raw data). Sizing is
    therefore driven from outside via notify_expected_n_series(m), which
    CorrTrack calls automatically -- once, on its own first run() step,
    using len(ids) (the caller's own raw series list, seen before any
    per-step validity filtering can shrink it) -- pushed down through every
    Candidates grid node with zero action required from the human calling
    CorrTrack. insert_many still has its own lazy self-sizing fallback (see
    below) for any lower-level caller that constructs this class directly
    and never calls notify_expected_n_series, but that path is no longer
    the primary one and carries the (small, historical) risk that its own
    first batch is not a perfectly faithful m. NOT re-run on later calls
    either way: m*L does not change after steady state, so no resize
    mechanism exists or is needed for the intended use pattern.
    band_width = ceil(log2(m*L / TARGET_OCCUPANCY)),
    clamped to [3, min(24, n_vectors)]. TARGET_OCCUPANCY=3 is not arbitrary
    -- it is reverse-engineered to reproduce band_width=8 exactly at the
    m*L=550 operating point this class's recall/precision were already
    validated at (see docs/implementation_log.md), so existing validated
    behavior at that scale is unchanged; it only changes behavior at other
    scales, where band_width=8 was never validated (and, per the
    measurement above, was already silently oversubscribed).

    Correctness contract, precisely stated (matches TopKInvertedIndex's,
    not overclaimed): precision is exact by construction when
    apply_dot_filter=True (every returned pair already passed a real full
    dot-product+gamma check, identical to sorted_arrays_bs/bptree) --
    identical to every other backend's contract EXCEPT that setting
    apply_dot_filter=False deliberately widens precision to whatever
    exact validation downstream decides (by design, per the human's
    request -- the dot filter becomes optional). Recall is an empirical
    property of this dataset's sign-bit geometry, not a mathematical
    guarantee, exactly like TopKInvertedIndex's top-m overlap heuristic --
    measured, not assumed, via a dedicated recall-vs-bruteforce test.
    """
    cdef public object last_stats
    cdef Py_ssize_t _n_vectors
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _count
    cdef Py_ssize_t _band_width      # 0 until _finalize_sizing runs on the first insert_many
    cdef Py_ssize_t _n_bands
    cdef Py_ssize_t _bucket_count
    cdef int64_t _complement_mask
    cdef double _target_occupancy    # avg alive entries per bucket band_width is sized for (default 3.0)
    # (2026-08-30) Theory-driven n_bands cap, mirroring the retired cell_size/
    # cell_stretch precedent (library_corrtrack_parallel.py's _resolve_cell_
    # stretch): _n_bands_tolerance<=0 means "disabled" -- n_bands stays the
    # fixed constructor value exactly as before (fully backward compatible,
    # every existing caller unaffected). When >0, _finalize_sizing computes
    # the closed-form LSH-banding minimum n_bands (b_min) for _target_recall
    # at this cell's own (m, n_lagged_windows, target_occupancy, corr_
    # threshold), and OVERRIDES _n_bands with ceil(_n_bands_tolerance *
    # b_min) -- see docs/implementation_log.md's 2026-08-30 entries for the
    # derivation and real-data validation (tolerance=2.0 matched or beat the
    # empirical grid search's own reliability across 12 real cells, at a
    # median of 75% as many bands).
    cdef double _n_bands_tolerance
    cdef double _corr_threshold_for_sizing
    cdef double _target_recall
    # (2026-09-04) Overridable padding on top of _target_recall inside the sizing search --
    # see compute_lsh_n_bands's own docstring for the full derivation and its real, disclosed
    # memory cost at scale. Defaults to the calibrated RECALL_SAFETY_MARGIN; a caller facing
    # memory pressure at large (m, L) can pass 0.0 to fall back to the un-padded (but still
    # overlap-corrected) minimum.
    cdef double _recall_safety_margin
    # (2026-07-22) band_mode="grid" (candidate_backend="lsh_grid_dot") --
    # the human's original grid-bucketing idea (partition sketch dims into
    # bands, quantize raw coordinate values into cells per band) combined
    # with THIS class's AND-within-band/OR-across-band structure, instead
    # of a flat "f percent of groups agree" vote. Reuses every part of this
    # class (posting lists, pair dedup, dot+gamma gate, numeric-rows
    # output) unchanged -- only band-key computation differs (grid-cell
    # quantization + hash-combine instead of sign-bit packing). See
    # docs/implementation_log.md's 2026-07-22 "lsh_grid_dot" entry.
    cdef Py_ssize_t _band_mode        # 0=sign (default), 1=grid
    cdef double _cell_width           # grid mode only: per-dimension quantization cell width
    cdef object _band_offsets         # (n_bands, band_width) float64 -- grid mode only, E2LSH-style per-band-per-dim random offset
    cdef object _neg_band_keys        # (capacity, n_bands) int32 (2026-09-04, was int64 -- see compute_lsh_n_bands's docstring for why n_bands can now be much larger, making this halving worthwhile; safe since band keys are bounded by 2^band_width-1 <= 2^24) -- grid mode only: this node's OWN band key if its vector were negated (computed once at insert, used only when this node later acts as a query)
    cdef bint _apply_dot_filter
    # (2026-07-21) Cheap full-vector sign-Hamming pre-filter -- see
    # docs/implementation_log.md, "lsh_sign_dot Hamming pre-filter
    # diagnosed and landed". Sits between prefiltered_pairs and the real
    # dot product, EXACTLY like HammingExactIndex's own gate (reuses that
    # class's _words/_word_masks/_popcount64_swar representation and
    # computation verbatim, not reinvented) -- diagnosed via a faithful
    # numpy replica before landing: max_frac~0.40-0.45 cuts real dot
    # products 45-75% at ~zero recall cost, consistent across two scales
    # (n=240 and n=2000 synthetic). Opt-in (default off) until validated
    # against the project's own real benchmark data.
    cdef bint _apply_hamming_filter
    cdef Py_ssize_t _n_words
    cdef Py_ssize_t _hamming_max_bits   # derived once from hamming_max_frac * n_vectors, OR from gamma (see below)
    # (2026-07-21j) hamming_max_frac=None -- auto-derive _hamming_max_bits
    # from gamma via the SAME SimHash expected-Hamming-distance relation
    # HammingExactIndex._finalize_threshold already uses and has validated
    # (expected = n_vectors * acos(gamma) / pi, margin = sqrt(n_vectors)
    # std of safety) -- rather than a flat, gamma-blind fraction tuned only
    # for this project's own benchmark gamma. Deferred to the first query
    # batch (gamma isn't known at __cinit__ time), same lazy pattern as
    # HammingExactIndex's own _threshold_sized flag. A concrete float still
    # means "use this fixed fraction" -- unchanged, backward compatible.
    cdef bint _hamming_frac_auto
    cdef bint _hamming_max_bits_sized
    cdef object _words                 # (capacity, n_words) int64 -- packed sign bits, LSB-first per word
    cdef object _word_masks            # (n_words,) uint64 -- valid-bit mask (last word may be partial)
    # (2026-07-21) Per-query candidate examination budget -- a raw safety
    # valve, NOT a prioritized "stop once enough good candidates are
    # found" budget like InstinctIndex's ef_search (LSH's touched set has
    # no natural priority order to cut against -- diagnosed: costs real,
    # somewhat unpredictable recall once triggered, unlike the Hamming
    # filter). 0 = unlimited (default, matches pre-existing behavior
    # exactly).
    cdef Py_ssize_t _max_candidates_per_query
    cdef bint _sized                 # False until band_width/bucket structures are finalized
    cdef Py_ssize_t _n_lagged_windows  # L -- known at construction, used with observed m to size band_width
    cdef long _band_seed
    cdef int64_t _min_valid_time
    cdef int64_t _query_stamp
    cdef Py_ssize_t _alive_count
    cdef Py_ssize_t _dead_count
    # (2026-09-07) Free-list of dead slot indices available for reuse -- see
    # docs/implementation_log.md's memory-leak investigation/fix entry.
    # Before this, a slot vacated by drop_before_time was marked dead but its
    # array index was never reused: insert_many always allocated the NEXT
    # never-before-used index (self._count, monotonically increasing), so
    # _capacity -- and every array/malloc'd buffer sized by it -- grew
    # without bound over a long stream even though the alive population
    # (_alive_count) stayed correctly bounded at m*L. Now allocation pops a
    # freed slot from this stack before ever growing _count/_capacity, so
    # both converge to a real steady-state bound instead of tracking total
    # items ever inserted over the run's lifetime.
    cdef Py_ssize_t *_free_slots
    cdef Py_ssize_t _free_count
    cdef Py_ssize_t _free_capacity
    cdef object _vectors
    cdef object _alive
    cdef object _window_idx
    cdef object _sid_idx
    cdef object _sid_rank
    cdef object _time
    cdef object _window_size
    cdef object _entry_ids
    cdef object _band_dims          # (n_bands, band_width) int64 -- fixed at construction
    cdef object _band_keys          # (capacity, n_bands) int32 (2026-09-04, was int64 -- bounded by 2^band_width-1 <= 2^24, safe) -- computed once per node at insert
    cdef object _membership_pos     # (capacity, n_bands) int32 (2026-09-04, was int64 -- bounded by realistic bucket occupancy, far under int32 range) -- this node's slot index within its bucket's flat member array (for O(1) swap-remove)
    # Flat, swap-remove posting-list storage: one growable int64 buffer per
    # (band, bucket) slot, flat-indexed as band_idx * _bucket_count + key.
    # Replaces the earlier doubly-linked-list scheme -- same O(1) insert/
    # remove, but query-time traversal walks a tight contiguous array
    # instead of following pointers scattered across a (capacity, n_bands)
    # table (real cache-locality fix). Sized once bucket_count is known
    # (_finalize_sizing); freed in __dealloc__.
    cdef int64_t **_post_members
    # (2026-09-04) int32, not int64 -- these are per-slot COUNTS/CAPACITIES (how many items sit
    # in one band's one bucket right now), bounded by realistic alive-population sizes, not by
    # cumulative stream length the way _post_members' own item-id contents are (left at int64,
    # unaudited for that unbounded-growth risk). _post_total_slots = n_bands * bucket_count can
    # itself be tens of millions at large (m, L) now that n_bands is bigger post-2026-09-03, so
    # this scaffolding is a real, comparably-sized memory consumer to _band_keys et al, not a
    # minor one -- verified directly at m=1000: ~383MB at int64 vs ~255MB at int32 for these two
    # arrays alone.
    cdef int32_t *_post_capacity
    cdef int32_t *_post_count
    cdef Py_ssize_t _post_total_slots
    cdef object _visited_stamp
    cdef object _win_sid_idx_arr
    cdef object _win_sid_rank_arr
    cdef object _win_time_arr
    cdef object _win_size_arr
    cdef Py_ssize_t _win_capacity
    # Persistent scratch -- allocated once in __cinit__, grown in place
    # (same Tier-1 lesson applied preemptively as TopKInvertedIndex/
    # InstinctIndex: no malloc/free per call).
    cdef int64_t *_touched_buf
    cdef Py_ssize_t _touched_cap
    cdef int64_t *_pair_out_buf
    cdef Py_ssize_t _pair_out_cap
    cdef uint8_t *_pair_seen_occupied_buf
    cdef int64_t *_pair_seen_a_buf
    cdef int64_t *_pair_seen_b_buf
    cdef Py_ssize_t _pair_seen_cap_buf

    def __cinit__(self, Py_ssize_t n_vectors=1, Py_ssize_t initial_capacity=1024,
                  Py_ssize_t n_lagged_windows=1, Py_ssize_t n_bands=64,
                  bint apply_dot_filter=True, long band_seed=4242,
                  bint apply_hamming_filter=False, object hamming_max_frac=0.40,
                  Py_ssize_t max_candidates_per_query=0,
                  str band_mode="sign", double cell_width=0.0,
                  double target_occupancy=3.0,
                  double n_bands_tolerance=0.0, double corr_threshold_for_sizing=0.7,
                  double target_recall=0.95, double recall_safety_margin=-1.0):
        if n_vectors <= 0:
            n_vectors = 1
        if initial_capacity < 16:
            initial_capacity = 16
        if n_lagged_windows <= 0:
            n_lagged_windows = 1
        if n_bands <= 0:
            n_bands = 64
        self._n_vectors = n_vectors
        self._capacity = initial_capacity
        self._count = 0
        self._band_width = 0
        self._n_bands = n_bands
        self._bucket_count = 0
        self._complement_mask = 0
        # (2026-07-29g) See _finalize_sizing below -- the average alive
        # entries per bucket this class sizes band_width for. Exposed as a
        # constructor param (was a hardcoded local constant) so the actual
        # scale-governing knob can be swept directly, rather than n_bands
        # alone, which does not control band_width's own m*L-dependent sizing.
        self._target_occupancy = target_occupancy if target_occupancy > 0.0 else 3.0
        self._n_bands_tolerance = n_bands_tolerance if n_bands_tolerance > 0.0 else 0.0
        self._corr_threshold_for_sizing = corr_threshold_for_sizing
        self._target_recall = target_recall if 0.0 < target_recall < 1.0 else 0.95
        # (2026-09-04) Negative (the default) means "use the calibrated RECALL_SAFETY_MARGIN" --
        # a real, disclosed value, not a silently-inert sentinel; 0.0 is a legitimate, explicit
        # choice (no margin at all) and must NOT be treated the same as "unset".
        self._recall_safety_margin = recall_safety_margin if recall_safety_margin >= 0.0 else RECALL_SAFETY_MARGIN
        # (2026-07-22) Grid mode has no sign-bit/Hamming notion at all --
        # force the (sign-specific) Hamming pre-filter off regardless of
        # what was requested, rather than silently misbehave.
        self._band_mode = 1 if band_mode == "grid" else 0
        if self._band_mode == 1:
            apply_hamming_filter = False
            if cell_width <= 0.0:
                cell_width = 1.0
        self._cell_width = cell_width
        self._apply_hamming_filter = apply_hamming_filter
        self._n_words = (n_vectors + 63) // 64
        if hamming_max_frac is None:
            self._hamming_frac_auto = True
            self._hamming_max_bits_sized = False
            self._hamming_max_bits = n_vectors  # inert placeholder until the first query supplies gamma
        else:
            self._hamming_frac_auto = False
            self._hamming_max_bits_sized = True
            _frac = <double>hamming_max_frac
            if _frac < 0.0:
                _frac = 0.0
            if _frac > 1.0:
                _frac = 1.0
            self._hamming_max_bits = <Py_ssize_t>(_frac * n_vectors + 0.5)
        self._max_candidates_per_query = max_candidates_per_query if max_candidates_per_query > 0 else 0
        self._words = np.zeros((self._capacity, self._n_words), dtype=np.int64)
        word_masks_np = np.empty(self._n_words, dtype=np.uint64)
        for _w in range(self._n_words):
            _bits_in_word = n_vectors - _w * 64
            if _bits_in_word > 64:
                _bits_in_word = 64
            if _bits_in_word >= 64:
                word_masks_np[_w] = <uint64_t>0xFFFFFFFFFFFFFFFFULL
            else:
                word_masks_np[_w] = (<uint64_t>1 << _bits_in_word) - <uint64_t>1
        self._word_masks = word_masks_np
        self._apply_dot_filter = apply_dot_filter
        self._sized = False
        self._n_lagged_windows = n_lagged_windows
        self._band_seed = band_seed
        self._min_valid_time = -9223372036854775807
        self._query_stamp = 0
        self._alive_count = 0
        self._dead_count = 0
        self._free_slots = NULL
        self._free_count = 0
        self._free_capacity = 0
        self._vectors = np.empty((self._capacity, self._n_vectors), dtype=np.float64)
        self._alive = np.zeros(self._capacity, dtype=np.uint8)
        self._window_idx = np.empty(self._capacity, dtype=np.int64)
        self._sid_idx = np.empty(self._capacity, dtype=np.int64)
        self._sid_rank = np.empty(self._capacity, dtype=np.int64)
        self._time = np.empty(self._capacity, dtype=np.int64)
        self._window_size = np.empty(self._capacity, dtype=np.int64)
        self._entry_ids = np.empty(self._capacity, dtype=np.int64)
        # _band_dims and the flat posting-list buffers depend on band_width/
        # bucket_count (not known yet -- see _finalize_sizing, called from
        # the first insert_many). Everything else that only depends on
        # capacity/n_bands (not band_width) can be allocated now.
        self._band_dims = None
        # (2026-09-04) int32, not int64 -- band keys are bounded by 2^band_width-1 and
        # band_width is capped at min(24, n_vectors), so values never exceed ~16.7M, comfortably
        # inside int32's ~2.1B range. membership_pos (a position within one bucket's posting
        # list) is bounded by realistic capacity values, similarly far under int32's range for
        # this domain. Halves these 3 arrays' memory -- the single biggest consumer of
        # SignLSHBandIndex's own footprint, and now the most cost-sensitive one given n_bands
        # can be in the hundreds to low thousands (see docs/implementation_log.md's 2026-09-04
        # (c) entry for the real, measured memory numbers this was checked against).
        self._band_keys = np.zeros((self._capacity, self._n_bands), dtype=np.int32)
        self._neg_band_keys = np.zeros((self._capacity, self._n_bands), dtype=np.int32)
        self._band_offsets = np.zeros((1, 1), dtype=np.float64)  # grid mode only; real shape set in _finalize_sizing
        self._membership_pos = np.full((self._capacity, self._n_bands), -1, dtype=np.int32)
        self._post_members = NULL
        self._post_capacity = NULL
        self._post_count = NULL
        self._post_total_slots = 0
        self._visited_stamp = np.full(self._capacity, -1, dtype=np.int64)
        self._win_capacity = initial_capacity
        self._win_sid_idx_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_sid_rank_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_time_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._win_size_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self.last_stats = {}

        self._touched_cap = self._capacity
        self._touched_buf = <int64_t *>malloc(self._touched_cap * sizeof(int64_t))
        self._pair_out_cap = 1024
        self._pair_out_buf = <int64_t *>malloc(self._pair_out_cap * 2 * sizeof(int64_t))
        self._pair_seen_cap_buf = 0
        self._pair_seen_occupied_buf = NULL
        self._pair_seen_a_buf = NULL
        self._pair_seen_b_buf = NULL
        if self._touched_buf == NULL or self._pair_out_buf == NULL:
            raise MemoryError("SignLSHBandIndex: failed to allocate persistent scratch buffers")

    def __dealloc__(self):
        cdef Py_ssize_t _i
        if self._touched_buf != NULL:
            free(self._touched_buf)
        if self._free_slots != NULL:
            free(self._free_slots)
        if self._pair_out_buf != NULL:
            free(self._pair_out_buf)
        if self._pair_seen_occupied_buf != NULL:
            free(self._pair_seen_occupied_buf)
        if self._pair_seen_a_buf != NULL:
            free(self._pair_seen_a_buf)
        if self._pair_seen_b_buf != NULL:
            free(self._pair_seen_b_buf)
        if self._post_members != NULL:
            for _i in range(self._post_total_slots):
                if self._post_members[_i] != NULL:
                    free(self._post_members[_i])
            free(self._post_members)
        if self._post_capacity != NULL:
            free(self._post_capacity)
        if self._post_count != NULL:
            free(self._post_count)

    cpdef notify_expected_n_series(self, Py_ssize_t observed_m):
        # (2026-07-10) Public hook so a caller that already knows the true
        # series count -- e.g. CorrTrack, which sees the raw `ids` list on
        # every run() step before any per-step validity filtering can
        # shrink it -- can size band_width immediately, instead of relying
        # on insert_many's lazy inference from its own (possibly filtered,
        # possibly smaller) first batch. No-op once already sized, so it is
        # safe to call unconditionally on every step.
        if not self._sized and observed_m > 0:
            self._finalize_sizing(observed_m)

    cdef void _finalize_sizing(self, Py_ssize_t observed_m):
        # (2026-07-10) Runs exactly once -- either eagerly via
        # notify_expected_n_series (preferred path, see above) or lazily
        # from the first insert_many call as a fallback for any caller that
        # never calls notify_expected_n_series. observed_m is the number of
        # series in play (n_series). NOT re-run on later calls: m*L is
        # fixed by configuration and does not drift after steady state, so
        # there is deliberately no resize path here.
        #
        # (2026-07-10, tried a THIRD time, still REVERTED) Widening
        # band_width from n_bands (to bound the OR-amplification
        # chance-collision rate ~1-(1-2/bucket_count)^n_bands) was tried and
        # reverted twice under sketch_norm="z" (a real methodology bug --
        # the class default at the time -- while the human's real config
        # always uses "z"'s replacement, "mean_l2", via
        # experiment_run_param_grid.py). Re-tested a third time after fixing
        # that default (sketch_norm now defaults to "mean_l2" everywhere,
        # see library_corrtrack_parallel.py): the picture is healthier
        # under the correct config (recall genuinely climbs with n_bands
        # here, unlike the flat ~0.69-0.73 found under "z"), but a direct
        # same-window A/B at n_bands=128 still favors the occupancy-only
        # formula below: recall=0.9907 (occupancy-only) vs 0.8610 (coupled),
        # speedup=0.883x vs 1.192x -- the coupled version trades away
        # recall well below the human's usual 0.95 target for a modest
        # speed gain, and even at n_bands=768 only reached recall=0.9143
        # while speed had degraded to 0.895x (never catches up on either
        # axis). REVERTED for a third and final time, now confirmed under
        # the correct sketch_norm. See docs/implementation_log.md's
        # 2026-07-10 entries for the full three-round history before
        # re-attempting this again.
        cdef double target_occupancy = self._target_occupancy  # (2026-07-29g) now an overridable constructor param; default 3.0 reproduces band_width=8 exactly at the m*L=550 point this class's recall/precision were validated at
        cdef Py_ssize_t band_width
        cdef Py_ssize_t _slot
        cdef Py_ssize_t _new_n_bands
        # (2026-09-04) band_width/n_bands sizing extracted to module-level compute_lsh_band_width/
        # compute_lsh_n_bands so a caller (e.g. proxy-anchor hyperopt's ranking logic) can compute
        # a config's real theoretical n_bands ahead of time, WITHOUT building an index -- this
        # method now just applies whatever those pure functions return, so the two can never
        # drift apart.
        band_width = compute_lsh_band_width(observed_m, self._n_lagged_windows, target_occupancy, self._n_vectors)
        self._band_width = band_width
        self._bucket_count = 1 << band_width
        self._complement_mask = self._bucket_count - 1
        # (2026-08-30) Theory-driven n_bands cap -- see the _n_bands_tolerance
        # field comment above for the full derivation/validation. Disabled
        # (tolerance<=0) is the exact prior behavior: self._n_bands stays
        # whatever the constructor was given, nothing below this block runs.
        #
        # (2026-09-03) The overlap-corrected minimum (_corrected_b_min, see its derivation above
        # _passes_dot_gamma_gate) replaces the old independent-bands formula (1-(1-p_r)^n_bands
        # treating all n_bands as independent, which they aren't -- every band draws from the
        # SAME fixed pool of n_vectors bits). The old formula was a proven upper bound on real
        # recall whenever bands overlap (which they always do here), and measured +7.0 points
        # too optimistic on average against the Sobol sweep's own calibration data; the corrected
        # one is unbiased to within -0.3 points on the same data. n_bands_tolerance keeps working
        # exactly as before (a multiplier on top of the corrected minimum) for anyone who wants
        # extra margin beyond the target, but with this fix it's no longer NECESSARY to tune it
        # just to reach target_recall -- tolerance=1.0 now targets target_recall directly.
        #
        # (2026-09-04) compute_lsh_n_bands applies self._recall_safety_margin internally -- see
        # its own module-level definition for the derivation, and its docstring for the real,
        # disclosed memory cost this margin has at large (m, L).
        if self._n_bands_tolerance > 0.0:
            _new_n_bands = compute_lsh_n_bands(
                band_width, self._corr_threshold_for_sizing, self._n_vectors,
                self._target_recall, self._n_bands_tolerance, self._recall_safety_margin,
            )
            if _new_n_bands != self._n_bands:
                # _band_keys/_neg_band_keys/_membership_pos were allocated in
                # __cinit__ using the OLD (placeholder) n_bands, before
                # observed_m -- and therefore b_min -- was knowable. Nothing
                # has been written into them yet at this point in the
                # lifecycle (this runs before the first insert), so a fresh
                # reallocation at the correct shape is safe, not a resize of
                # live data.
                self._n_bands = _new_n_bands
                self._band_keys = np.zeros((self._capacity, self._n_bands), dtype=np.int32)
                self._neg_band_keys = np.zeros((self._capacity, self._n_bands), dtype=np.int32)
                self._membership_pos = np.full((self._capacity, self._n_bands), -1, dtype=np.int32)
        # Band dimension sets: independent random band_width-subsets of the
        # n_vectors dims, one per band, drawn without replacement WITHIN a
        # band but WITH replacement (deliberate overlap) ACROSS bands --
        # matches diag_sign_lsh_band_index.py's validated design exactly,
        # not a clean non-overlapping partition.
        rng = np.random.default_rng(self._band_seed)
        band_dims_list = [
            rng.choice(self._n_vectors, size=min(band_width, self._n_vectors), replace=False)
            for _ in range(self._n_bands)
        ]
        self._band_dims = np.asarray(band_dims_list, dtype=np.int64)
        if self._band_mode == 1:
            # (2026-07-22) E2LSH-style per-band-per-dimension random cell
            # offset -- avoids every band sharing one hard grid boundary
            # (the exact problem the human's original grid scheme had).
            # Continues the SAME seeded rng stream as band_dims above, so
            # the whole index is reproducible from band_seed alone.
            self._band_offsets = rng.uniform(0.0, self._cell_width, size=(self._n_bands, band_width))
        self._post_total_slots = self._n_bands * self._bucket_count
        self._post_members = <int64_t **>malloc(self._post_total_slots * sizeof(int64_t *))
        self._post_capacity = <int32_t *>malloc(self._post_total_slots * sizeof(int32_t))
        self._post_count = <int32_t *>malloc(self._post_total_slots * sizeof(int32_t))
        if self._post_members == NULL or self._post_capacity == NULL or self._post_count == NULL:
            raise MemoryError("SignLSHBandIndex: failed to allocate posting-list slot tables")
        for _slot in range(self._post_total_slots):
            self._post_members[_slot] = NULL
            self._post_capacity[_slot] = 0
            self._post_count[_slot] = 0
        self._sized = True

    cdef void _finalize_hamming_threshold(self, double gamma):
        # (2026-07-21j) Only called when hamming_max_frac was None (auto)
        # and only once (guarded by _hamming_max_bits_sized) -- deferred
        # here because gamma is not known at __cinit__ time. Same SimHash
        # expected-Hamming-distance relation HammingExactIndex._finalize_
        # threshold already uses and has validated: for two vectors with
        # true cosine similarity gamma, the expected fraction of
        # disagreeing sign bits is acos(gamma)/pi; margin is a generous
        # sqrt(n_vectors)-std safety allowance on the binomial sign-flip
        # count. Verify against the recall-vs-bruteforce test before
        # trusting it at a new (n_vectors, gamma) operating point, exactly
        # as that class's own docstring already cautions.
        cdef double g = gamma
        cdef double expected, margin
        cdef Py_ssize_t bits
        if g < -1.0:
            g = -1.0
        if g > 1.0:
            g = 1.0
        expected = <double>self._n_vectors * acos(g) / _PI
        margin = sqrt(<double>self._n_vectors)
        bits = <Py_ssize_t>ceil(expected + margin)
        if bits < 0:
            bits = 0
        if bits > self._n_vectors:
            bits = self._n_vectors
        self._hamming_max_bits = bits
        self._hamming_max_bits_sized = True

    cdef void _ensure_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap, n
        cdef int64_t *new_touched
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
        new_band_keys = np.zeros((new_cap, self._n_bands), dtype=np.int32)
        new_neg_band_keys = np.zeros((new_cap, self._n_bands), dtype=np.int32)
        new_membership_pos = np.full((new_cap, self._n_bands), -1, dtype=np.int32)
        new_visited_stamp = np.full(new_cap, -1, dtype=np.int64)
        new_words = np.zeros((new_cap, self._n_words), dtype=np.int64)
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
            new_band_keys[:n, :] = self._band_keys[:n, :]
            new_neg_band_keys[:n, :] = self._neg_band_keys[:n, :]
            new_membership_pos[:n, :] = self._membership_pos[:n, :]
            new_visited_stamp[:n] = self._visited_stamp[:n]
            new_words[:n, :] = self._words[:n, :]
        self._vectors = new_vectors
        self._alive = new_alive
        self._window_idx = new_window_idx
        self._sid_idx = new_sid_idx
        self._sid_rank = new_sid_rank
        self._time = new_time
        self._window_size = new_window_size
        self._entry_ids = new_entry_ids
        self._band_keys = new_band_keys
        self._neg_band_keys = new_neg_band_keys
        self._words = new_words
        self._membership_pos = new_membership_pos
        self._visited_stamp = new_visited_stamp
        self._capacity = new_cap
        new_touched = <int64_t *>realloc(self._touched_buf, new_cap * sizeof(int64_t))
        if new_touched == NULL:
            raise MemoryError("SignLSHBandIndex: failed to grow touched-candidate buffer")
        self._touched_buf = new_touched
        self._touched_cap = new_cap

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

    cdef void _push_free_slot(self, Py_ssize_t idx):
        # (2026-09-07) See the _free_slots field comment -- pushes a
        # just-expired slot index so the next insertion reuses it instead of
        # growing _count/_capacity. Same doubling-growth pattern as
        # _post_members' own realloc above.
        cdef Py_ssize_t new_cap
        cdef Py_ssize_t *grown
        if self._free_count >= self._free_capacity:
            new_cap = 4 if self._free_capacity == 0 else self._free_capacity * 2
            grown = <Py_ssize_t *>realloc(self._free_slots, new_cap * sizeof(Py_ssize_t))
            if grown == NULL:
                raise MemoryError("SignLSHBandIndex: failed to grow free-slot list")
            self._free_slots = grown
            self._free_capacity = new_cap
        self._free_slots[self._free_count] = idx
        self._free_count += 1

    cdef int64_t _insert_one(self, double[:] vector, int64_t window_idx, int64_t sid_idx,
                              int64_t time_idx, int64_t window_size, int64_t sid_rank):
        cdef Py_ssize_t i, d, band_idx, w, dim, flat, cap, cnt
        cdef int64_t new_id, key
        cdef int64_t *members
        cdef Py_ssize_t bits_in_word, bit
        cdef uint64_t word_tmp
        cdef int64_t neg_key, cell
        cdef double offv
        # (2026-09-07) Reuse a freed slot (from drop_before_time) before ever
        # growing _count/_capacity -- see the _free_slots field comment.
        # Every field at slot i is unconditionally overwritten below
        # (vector, band keys, membership_pos, words, ...), so a reused slot
        # needs no separate clearing.
        if self._free_count > 0:
            self._free_count -= 1
            i = self._free_slots[self._free_count]
        else:
            self._ensure_capacity(self._count + 1)
            i = self._count
            self._count += 1
        cdef double[:, ::1] vectors_mv = self._vectors
        cdef uint8_t[:] alive_mv = self._alive
        cdef int64_t[:] window_idx_mv = self._window_idx
        cdef int64_t[:] sid_idx_mv = self._sid_idx
        cdef int64_t[:] sid_rank_mv = self._sid_rank
        cdef int64_t[:] time_mv = self._time
        cdef int64_t[:] window_size_mv = self._window_size
        cdef int64_t[:] entry_ids_mv = self._entry_ids
        cdef int64_t[:, :] band_dims = self._band_dims
        cdef int32_t[:, :] band_keys = self._band_keys
        cdef int32_t[:, :] neg_band_keys = self._neg_band_keys
        cdef double[:, :] band_offsets_mv = self._band_offsets
        cdef int32_t[:, :] membership_pos = self._membership_pos
        cdef int64_t[:, ::1] words_mv = self._words

        for d in range(self._n_vectors):
            vectors_mv[i, d] = vector[d]
        if self._band_mode == 0:
            # (2026-07-21) Full-vector sign pattern for the optional Hamming
            # pre-filter -- packed identically to HammingExactIndex's own
            # _insert_one (same bit order, same word layout), so the shared
            # _popcount64_swar helper applies unchanged. Grid mode has no
            # sign/Hamming notion at all, so this is skipped entirely for it
            # (apply_hamming_filter is also forced off in __cinit__).
            for w in range(self._n_words):
                bits_in_word = self._n_vectors - w * 64
                if bits_in_word > 64:
                    bits_in_word = 64
                word_tmp = 0
                for bit in range(bits_in_word):
                    if vector[w * 64 + bit] >= 0:
                        word_tmp |= (<uint64_t>1) << bit
                words_mv[i, w] = <int64_t>word_tmp
        alive_mv[i] = 1
        window_idx_mv[i] = window_idx
        sid_idx_mv[i] = sid_idx
        sid_rank_mv[i] = sid_rank
        time_mv[i] = time_idx
        window_size_mv[i] = window_size
        # (2026-09-07) entry_id MUST equal the physical slot index i: query
        # code (_find_pair_rows_meta) uses a returned entry_id directly as
        # a subscript into every per-slot array (alive/window_idx/band_keys/
        # words/...), not as an independent, separately-tracked id --
        # confirmed by reading that code before making entry_id reusable-
        # slot-safe. A once-monotonic, never-reused _next_entry_id (the
        # pre-fix scheme) only ever coincided with the slot index because,
        # before this fix, slots were NEVER reused either -- the two
        # counters were silently always in lockstep. Reusing slots without
        # also making entry_id track the slot broke that coincidence and
        # caused real, confirmed query misindexing (see
        # docs/implementation_log.md's memory-leak fix entry for the two
        # tests that caught this).
        new_id = i
        entry_ids_mv[i] = new_id
        self._alive_count += 1

        # (2026-07-10) Polarity canonicalization against a single reference
        # dimension was tried and REVERTED -- see docs/implementation_log.md.
        # It is provably correct only for the idealized case where a
        # candidate is the EXACT elementwise negation of the query; real
        # negatively-correlated pairs are statistically, not exactly,
        # anti-correlated, so individual dimension signs (including
        # whichever one is picked as the reference) can disagree even
        # between genuinely correlated pairs, especially near the gamma
        # boundary. A single disagreeing reference bit flips the
        # canonicalization decision for EVERY band simultaneously, causing
        # a global miss rather than the graceful per-band degradation the
        # original two-lookup (raw key + full complement) scheme has.
        # Measured: real recall on the validated dataset dropped from
        # ~0.85 to ~0.69 -- reverted back to raw (non-canonicalized) keys.
        for band_idx in range(self._n_bands):
            if self._band_mode == 0:
                key = 0
                for w in range(self._band_width):
                    dim = band_dims[band_idx, w]
                    key = (key << 1) | (1 if vector[dim] >= 0 else 0)
            else:
                # (2026-07-22) Grid mode: quantize each of this band's
                # band_width raw coordinates into a per-dimension cell
                # index (E2LSH-style, with this band's own random offset),
                # then combine the band_width cell indices into a single
                # bucket key via a simple polynomial hash, reduced into the
                # SAME fixed [0, bucket_count) range the sign-bit packing
                # above produces natively -- reuses the identical posting-
                # list array machinery below unchanged either way. Also
                # computes the key this SAME node's vector would get if it
                # were negated (neg_band_keys) -- needed for neg_corr
                # queries later, since (unlike a sign-bit complement) there
                # is no cheap bit-trick to derive a negated-vector's grid
                # key from the positive one.
                key = 0
                neg_key = 0
                for w in range(self._band_width):
                    dim = band_dims[band_idx, w]
                    offv = band_offsets_mv[band_idx, w]
                    cell = <int64_t>floor((vector[dim] + offv) / self._cell_width)
                    key = key * 1000003 + cell
                    cell = <int64_t>floor((-vector[dim] + offv) / self._cell_width)
                    neg_key = neg_key * 1000003 + cell
                key = key % self._bucket_count
                if key < 0:
                    key += self._bucket_count
                neg_key = neg_key % self._bucket_count
                if neg_key < 0:
                    neg_key += self._bucket_count
                neg_band_keys[i, band_idx] = neg_key
            band_keys[i, band_idx] = key
            flat = band_idx * self._bucket_count + key
            cnt = self._post_count[flat]
            cap = self._post_capacity[flat]
            if cnt >= cap:
                cap = 4 if cap == 0 else cap * 2
                members = <int64_t *>realloc(self._post_members[flat], cap * sizeof(int64_t))
                if members == NULL:
                    raise MemoryError("SignLSHBandIndex: failed to grow posting-list bucket")
                self._post_members[flat] = members
                self._post_capacity[flat] = cap
            self._post_members[flat][cnt] = i
            membership_pos[i, band_idx] = cnt
            self._post_count[flat] = cnt + 1

        return new_id

    @property
    def n_bands(self):
        """(2026-08-30) Read-only introspection of the EFFECTIVE n_bands --
        the constructor value when n_bands_tolerance is disabled, or the
        tolerance*b_min override once _finalize_sizing has run. 0 before
        the first insert/notify_expected_n_series (sizing not yet known)."""
        return self._n_bands

    @property
    def band_width(self):
        return self._band_width

    @property
    def count(self):
        """(2026-09-04, debug/investigation) Read-only introspection of how many entries this
        index currently holds -- added to directly check whether the alive population (and
        therefore the capacity needed to hold it) stays bounded over a long streaming run, or
        grows without bound. See docs/implementation_log.md's 2026-09-04 memory-leak
        investigation entry."""
        return self._count

    @property
    def capacity(self):
        return self._capacity

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

        if n == 0:
            return np.empty(0, dtype=np.int64)
        if not self._sized:
            self._finalize_sizing(n)
        if vectors_in is None:
            raise ValueError("SignLSHBandIndex requires full sketch vectors")
        vectors_np = np.ascontiguousarray(np.asarray(vectors_in, dtype=np.float64))
        if vectors_np.ndim != 2 or vectors_np.shape[0] != n:
            raise ValueError("vectors must be a 2D array with one row per value")
        if vectors_np.shape[1] != self._n_vectors:
            raise ValueError("vector size does not match SignLSHBandIndex dimension")
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
                self._remove_from_postings(i)
                # (2026-09-07) Make the slot reusable -- see _free_slots'
                # field comment and the memory-leak fix entry in
                # docs/implementation_log.md.
                self._push_free_slot(i)

    cdef void _remove_from_postings(self, Py_ssize_t node_idx):
        # (2026-07-16) Eager O(n_bands) removal from every band's flat,
        # swap-remove posting-list array -- replaces the earlier doubly-
        # linked-list unlink (real cache-locality fix, see class
        # docstring), which itself replaced the original threshold-
        # triggered full rebuild (see docs/implementation_log.md's
        # 2026-07-09/-10 entries: lazy-only deletion caused a real,
        # confirmed ~104x wall-clock regression from unbounded posting-list
        # growth). This removes a node from its buckets the instant it
        # expires, so no dead node is ever walked by a future query, and
        # there is no threshold to tune.
        cdef Py_ssize_t band_idx, flat, pos, last, moved
        cdef int64_t key
        cdef int32_t[:, :] band_keys = self._band_keys
        cdef int32_t[:, :] membership_pos = self._membership_pos
        for band_idx in range(self._n_bands):
            key = band_keys[node_idx, band_idx]
            flat = band_idx * self._bucket_count + key
            pos = membership_pos[node_idx, band_idx]
            last = self._post_count[flat] - 1
            if pos != last:
                moved = self._post_members[flat][last]
                self._post_members[flat][pos] = moved
                membership_pos[moved, band_idx] = pos
            self._post_count[flat] = last
            membership_pos[node_idx, band_idx] = -1

    cpdef clear_recent(self):
        pass

    cpdef object find_pair_rows_full_cosine(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_meta(recent_entry_ids, gamma, False)

    cpdef object find_pair_rows_full_cosine_signed(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_meta(recent_entry_ids, gamma, True)

    # (2026-09-13) Option-4 investigation -- see _find_pair_rows_meta_parallel's own
    # comment (defined after _find_pair_rows_meta below) for the design.
    # n_threads<=1 dispatches straight to the unchanged sequential path.
    cpdef object find_pair_rows_full_cosine_parallel(self, long[:] recent_entry_ids, double gamma, double tau, Py_ssize_t n_threads=1):
        if n_threads <= 1:
            return self._find_pair_rows_meta(recent_entry_ids, gamma, False)
        return self._find_pair_rows_meta_parallel(recent_entry_ids, gamma, False, n_threads)

    cpdef object find_pair_rows_full_cosine_signed_parallel(self, long[:] recent_entry_ids, double gamma, double tau, Py_ssize_t n_threads=1):
        if n_threads <= 1:
            return self._find_pair_rows_meta(recent_entry_ids, gamma, True)
        return self._find_pair_rows_meta_parallel(recent_entry_ids, gamma, True, n_threads)

    cdef object _empty_stats(self, double gamma):
        return {
            "num_index_candidates": 0, "num_valid_index_candidates": 0,
            "num_unique_index_candidates": 0, "num_duplicate_index_candidates": 0,
            "num_unique_pre_dot_pairs": 0, "num_duplicate_pre_dot_pairs": 0,
            "num_dot_checks": 0, "num_distance_checks": 0,
            "num_after_similarity": 0, "num_after_dot": 0,
            "num_pairs_before_dedupe": 0, "num_pairs_after_dedupe": 0,
            "num_rows": 0, "num_recent_queries": 0, "num_entries": int(self._alive_count),
            "num_blocks": 1, "gamma": float(gamma), "tau": 0.0,
            "lsh_candidates_touched": 0, "lsh_dot_checks": 0,
            "lsh_candidates_returned": 0, "lsh_query_time": 0.0,
            "lsh_num_nodes_total": int(self._count), "lsh_num_nodes_alive": int(self._alive_count),
            "lsh_dead_node_ratio": float(self._dead_count) / float(self._count) if self._count > 0 else 0.0,
        }

    cdef object _find_pair_rows_meta(self, long[:] recent_entry_ids, double gamma, bint signed_abs):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t i, band_idx, variant, count, out_i
        cdef Py_ssize_t touched_count, t
        cdef int64_t q_entry, q_win, q_sid, q_rank, q_time, q_w
        cdef int64_t key, cur_key, node, cand
        cdef Py_ssize_t d
        cdef int64_t stamp
        cdef double score
        cdef bint passes, failed = False
        cdef Py_ssize_t pair_seen_count = 0
        cdef Py_ssize_t wanted_pair_seen_cap
        cdef int64_t pair_a, pair_b, a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t total_touched = 0, total_dot_checks = 0, total_valid = 0
        cdef Py_ssize_t unique_pre_dot_pairs = 0, duplicate_pre_dot_pairs = 0
        cdef Py_ssize_t after_similarity = 0, pairs_before_dedupe = 0, pairs_after_dedupe = 0
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t[:] win_sid_idx, win_sid_rank, win_time, win_w
        cdef double t0 = time.perf_counter()

        if n_recent == 0 or self._count == 0:
            self.last_stats = self._empty_stats(gamma)
            return np.empty((0, 5), dtype=np.int64)

        if self._apply_hamming_filter and self._hamming_frac_auto and not self._hamming_max_bits_sized:
            self._finalize_hamming_threshold(gamma)

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

        # (2026-07-21) Contiguity annotations (::1) on vectors/words -- pure
        # compile-time hint, zero algorithmic change, mirroring the exact
        # fix already validated for InstinctIndex (2026-07-17b entry, real
        # ~8.4% wall-time win from unblocking auto-vectorization of the
        # per-candidate dot-product reduction loop below). self._vectors/
        # self._words are always freshly np.empty/np.zeros-allocated
        # C-contiguous arrays (see _ensure_capacity), but the generic
        # double[:, :]/int64_t[:, :] memoryview types couldn't tell the C
        # compiler that, so the reduction loops were generated as strided,
        # non-vectorizable code despite -O3 -march=native -ffast-math.
        cdef double[:, ::1] vectors = self._vectors
        cdef uint8_t[:] alive = self._alive
        cdef int64_t[:] window_idx = self._window_idx
        cdef int64_t[:] sid_idx = self._sid_idx
        cdef int64_t[:] sid_rank = self._sid_rank
        cdef int64_t[:] time_idx = self._time
        cdef int64_t[:] window_size = self._window_size
        cdef int32_t[:, :] band_keys = self._band_keys
        cdef int32_t[:, :] neg_band_keys = self._neg_band_keys
        cdef int64_t[:] visited_stamp = self._visited_stamp
        cdef int64_t *touched_buf = self._touched_buf
        cdef Py_ssize_t n_variants = 2 if signed_abs else 1
        cdef Py_ssize_t flat, n_members, mi
        cdef int64_t *members
        cdef int64_t[:, ::1] words = self._words
        cdef uint64_t[:] word_masks = self._word_masks
        cdef Py_ssize_t hw, hpos, hneg, hbest
        cdef uint64_t qw_word, cw_word, xw_mask
        cdef bint budget_hit

        # (2026-09-14) Same fix as HammingExactIndex's identical redundant-enumeration
        # finding (docs/implementation_log.md's 2026-09-14 entry) -- when the queries in
        # one call are all mutually simultaneous (this project's own real usage: every
        # entry in recent_entry_ids was inserted at the same time this same step, before
        # any of them search), a same-time pair (A, B) gets independently discovered
        # twice: once when A's query finds B in a shared bucket, once more when B's query
        # finds A. is_recent + an EXPLICIT time_idx match + a sid_idx tie-break lets each
        # same-time pair be discovered from exactly one direction -- drops work, not
        # coverage (pair_seen already deduped the output before this fix).
        #
        # Two real bugs caught here before trusting this, not assumed safe:
        # 1. The time_idx check is NOT redundant with is_recent in general (only in this
        #    project's own call pattern): a caller may legally pass a MIXED-time
        #    recent_entry_ids batch (e.g. re-querying the whole alive population across
        #    several rounds, as
        #    test_lsh_sign_dot_index_flat_posting_list_survives_repeated_insert_drop_churn
        #    does), where a rank/id check alone would wrongly skip genuinely
        #    different-time comparisons whenever the older side happens to tie-break lower.
        # 2. The tie-break must be sid_idx, NOT sid_rank: sid_rank_in is a general,
        #    caller-supplied insert_many parameter with no uniqueness guarantee -- that
        #    same test passes the ROUND NUMBER for it (shared by every series inserted in
        #    that round), so a `<=` comparison on sid_rank ties for EVERY same-round pair,
        #    making BOTH directions skip and silently dropping the pair entirely (measured:
        #    recall collapsed from 100% to ~1%). sid_idx is the one field guaranteed unique
        #    per logical series by construction -- different series can never share it.
        cdef uint8_t[:] is_recent = np.zeros(self._count, dtype=np.uint8)
        for i in range(n_recent):
            q_entry = <int64_t>recent_entry_ids[i]
            if 0 <= q_entry < self._count:
                is_recent[q_entry] = 1

        count = 0
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

            self._query_stamp += 1
            stamp = self._query_stamp
            touched_count = 0
            budget_hit = False

            # Two lookups per band when signed/neg_corr: the band's own key
            # (catches statistically-positive local agreement) and its
            # bitwise complement (catches statistically-negative local
            # disagreement), each evaluated independently per band using
            # the RAW stored bits -- see _insert_one's comment for why a
            # single-reference-bit canonicalization (tried, reverted) is
            # not safe here: real correlation is not exact bit-negation.
            #
            # (2026-07-21) Optional per-query examination budget
            # (_max_candidates_per_query, 0=unlimited/default) -- a raw
            # safety valve on touched_count, NOT a prioritized cutoff (band
            # iteration order carries no similarity ranking, unlike
            # InstinctIndex's ef_search over a real best-first heap) --
            # diagnosed to cost real, somewhat unpredictable recall once
            # triggered. See docs/implementation_log.md.
            for band_idx in range(self._n_bands):
                if budget_hit:
                    break
                key = band_keys[q_entry, band_idx]
                for variant in range(n_variants):
                    if budget_hit:
                        break
                    if self._band_mode == 0:
                        cur_key = key if variant == 0 else (key ^ self._complement_mask)
                    else:
                        # (2026-07-22) Grid mode: no cheap bit-complement
                        # trick exists for a quantized value, so the
                        # negated-vector key was precomputed once at insert
                        # time (see _insert_one) and just looked up here.
                        cur_key = key if variant == 0 else neg_band_keys[q_entry, band_idx]
                    flat = band_idx * self._bucket_count + cur_key
                    n_members = self._post_count[flat]
                    members = self._post_members[flat]
                    for mi in range(n_members):
                        node = members[mi]
                        if (node != q_entry and visited_stamp[node] != stamp
                                and not (is_recent[node] and time_idx[node] == q_time and sid_idx[node] <= q_sid)):
                            visited_stamp[node] = stamp
                            touched_buf[touched_count] = node
                            touched_count += 1
                            if (self._max_candidates_per_query > 0
                                    and touched_count >= self._max_candidates_per_query):
                                budget_hit = True
                                break

            total_touched += touched_count

            for t in range(touched_count):
                cand = touched_buf[t]
                if not alive[cand]:
                    continue
                # (2026-07-10) Reordered to match sorted_arrays_bs's canonical
                # metric semantics: cheap validity checks (4 integer
                # comparisons) and pair-dedup BEFORE the full dot product,
                # not after. Previously the dot was computed for every
                # alive touched candidate regardless of validity, so
                # num_dot_checks could exceed num_valid_index_candidates/
                # num_unique_pre_dot_pairs -- backwards from what those
                # names mean everywhere else. This ordering also avoids
                # computing a redundant dot product when the same
                # underlying pair is touched from both directions within
                # one batch (A finds B, and separately B finds A) -- the
                # pair-seen dedup now happens before either direction ever
                # reaches the dot, so only the first encounter pays for it.
                if not (time_idx[cand] >= self._min_valid_time
                        and window_idx[cand] != q_win
                        and window_size[cand] == q_w
                        and not (sid_idx[cand] == q_sid and time_idx[cand] == q_time)):
                    continue
                total_valid += 1
                # (2026-07-21g) Hamming pre-filter MOVED to before the
                # per-batch pair-seen dedup (was: dedup first, then Hamming
                # -- see the 2026-07-21 entry this replaces). Rationale:
                # with dedup first, unique_pre_dot_pairs/prefiltered_pairs
                # was IDENTICAL whether the filter was on or off (a
                # Hamming-rejected candidate still paid for dedup/set-
                # insertion bookkeeping before ever reaching the filter) --
                # confirmed directly on real data. Checking Hamming first
                # means a rejected candidate never touches pair_seen at
                # all, so prefiltered_pairs now honestly reflects the
                # filter's real effect, and the (relatively more expensive)
                # hash-set insert is skipped entirely for anything Hamming
                # was always going to reject. Tradeoff, accepted: a pair
                # touched from BOTH directions within one batch now pays
                # the (cheap, O(n_words) SWAR popcount) Hamming check twice
                # instead of once if it's ultimately ACCEPTED (dedup still
                # correctly suppresses all downstream work on the second
                # encounter) -- a small, bounded cost against the larger,
                # more common win of skipping dedup+bookkeeping entirely
                # for REJECTED candidates. Does not change enumerated_pairs
                # (total_touched, computed earlier, untouched) or the final
                # accepted-pair set -- verified via the standard suite plus
                # a direct real-vs-replica candidate-set comparison.
                if self._apply_hamming_filter:
                    hpos = 0
                    hneg = 0
                    for hw in range(self._n_words):
                        qw_word = <uint64_t>words[q_entry, hw]
                        cw_word = <uint64_t>words[cand, hw]
                        xw_mask = word_masks[hw]
                        hpos += _popcount64_swar((qw_word ^ cw_word) & xw_mask)
                        if signed_abs:
                            hneg += _popcount64_swar((qw_word ^ (~cw_word)) & xw_mask)
                    hbest = hpos
                    if signed_abs and hneg < hbest:
                        hbest = hneg
                    if hbest > self._hamming_max_bits:
                        continue
                _canonical_window_pair(
                    q_win, q_sid, q_rank, q_time,
                    window_idx[cand], sid_idx[cand], sid_rank[cand], time_idx[cand],
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
                if self._apply_dot_filter:
                    score = 0.0
                    for d in range(self._n_vectors):
                        score += vectors[q_entry, d] * vectors[cand, d]
                    total_dot_checks += 1
                    passes = _passes_dot_gamma_gate(score, gamma, signed_abs, True)
                    if not passes:
                        continue
                after_similarity += 1
                if not _append_pair(&self._pair_out_buf, &count, &self._pair_out_cap, q_win, window_idx[cand]):
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
            "num_index_candidates": int(total_touched),
            "num_enumerated_candidates": int(total_touched),
            "num_valid_index_candidates": int(total_valid),
            "num_unique_index_candidates": int(total_valid),
            "num_duplicate_index_candidates": 0,
            "num_unique_pre_dot_pairs": int(unique_pre_dot_pairs),
            "num_duplicate_pre_dot_pairs": int(duplicate_pre_dot_pairs),
            "num_dot_checks": int(total_dot_checks),
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
            "lsh_candidates_touched": int(total_touched),
            "lsh_dot_checks": int(total_dot_checks),
            "lsh_candidates_returned": int(after_similarity),
            "lsh_query_time": float(time.perf_counter() - t0),
            "lsh_num_nodes_total": int(self._count),
            "lsh_num_nodes_alive": int(self._alive_count),
            "lsh_dead_node_ratio": float(self._dead_count) / float(self._count) if self._count > 0 else 0.0,
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

    cdef object _find_pair_rows_meta_parallel(self, long[:] recent_entry_ids, double gamma, bint signed_abs, Py_ssize_t n_threads):
        # (2026-09-13) Option-4 investigation (docs/implementation_log.md's 2026-09-13
        # entry, "option 4 -- prange parallelism"): _find_pair_rows_meta's per-query loop
        # (one iteration per newly-arrived series each step) is embarrassingly parallel --
        # each query's bucket scan is independent of every other query's -- but the
        # sequential version's shared, mutating state (_touched_buf, _visited_stamp,
        # _pair_seen_*) is not safe to share across threads without real synchronization.
        # This mirrors the bptree/sorted-arrays kernel's own proven prange pattern (see the
        # cosine-meta prange loop earlier in this file): each thread gets its OWN touched
        # buffer and visited-stamp array (both sized to the full index capacity, so no
        # cross-thread writes are ever possible) and its own growable output-pair buffer
        # (the same _append_pair helper the bptree kernel uses). The one real behavior
        # difference from the sequential path: _pair_seen's cross-query pre-dot dedup is
        # dropped (it would need real cross-thread synchronization to stay safe), so a pair
        # discovered from BOTH directions by two different threads in the same batch pays
        # for the dot product twice instead of once -- a small, bounded amount of duplicate
        # work, not a correctness issue, since the merge step below still produces exactly
        # the same final deduped row set (verified against the sequential path on both a
        # small synthetic case and real data -- see test_candidate_search_parallel_lsh_
        # matches_sequential and verify_parallel_lsh.py).
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t i, out_i, t, tt
        cdef int64_t a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t total_touched = 0, total_valid = 0, total = 0
        cdef double t0 = time.perf_counter()
        # Per-query scratch, thread-private by construction (assigned before read every
        # iteration, never touched with an in-place operator directly in the loop body --
        # see the comment above _lsh_scan_touched_bands for why that distinction matters).
        cdef int tid
        cdef int64_t q_entry, q_win, q_sid, q_time, q_w, stamp, cand
        cdef Py_ssize_t touched_count, k, hbest
        cdef bint passes
        cdef double score

        if n_recent == 0 or self._count == 0:
            self.last_stats = self._empty_stats(gamma)
            return np.empty((0, 5), dtype=np.int64)

        if n_threads < 1:
            n_threads = 1
        if n_threads > n_recent:
            n_threads = n_recent

        if self._apply_hamming_filter and self._hamming_frac_auto and not self._hamming_max_bits_sized:
            self._finalize_hamming_threshold(gamma)

        cdef double[:, ::1] vectors = self._vectors
        cdef uint8_t[:] alive = self._alive
        cdef int64_t[:] window_idx = self._window_idx
        cdef int64_t[:] sid_idx = self._sid_idx
        cdef int64_t[:] time_idx = self._time
        cdef int64_t[:] window_size = self._window_size
        cdef int32_t[:, :] band_keys = self._band_keys
        cdef int32_t[:, :] neg_band_keys = self._neg_band_keys
        cdef Py_ssize_t n_variants = 2 if signed_abs else 1
        cdef int64_t[:, ::1] words = self._words
        cdef uint64_t[:] word_masks = self._word_masks
        cdef Py_ssize_t capacity = self._capacity

        cdef int64_t **touched_bufs = <int64_t **>malloc(n_threads * sizeof(int64_t *))
        cdef int64_t **visited_stamps = <int64_t **>malloc(n_threads * sizeof(int64_t *))
        cdef int64_t **out_bufs = <int64_t **>malloc(n_threads * sizeof(int64_t *))
        cdef Py_ssize_t *out_counts = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        cdef Py_ssize_t *out_caps = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        cdef bint *failed = <bint *>malloc(n_threads * sizeof(bint))
        cdef Py_ssize_t *touched_total = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))
        cdef Py_ssize_t *valid_total = <Py_ssize_t *>malloc(n_threads * sizeof(Py_ssize_t))

        if (touched_bufs == NULL or visited_stamps == NULL or out_bufs == NULL
                or out_counts == NULL or out_caps == NULL or failed == NULL
                or touched_total == NULL or valid_total == NULL):
            if touched_bufs != NULL: free(touched_bufs)
            if visited_stamps != NULL: free(visited_stamps)
            if out_bufs != NULL: free(out_bufs)
            if out_counts != NULL: free(out_counts)
            if out_caps != NULL: free(out_caps)
            if failed != NULL: free(failed)
            if touched_total != NULL: free(touched_total)
            if valid_total != NULL: free(valid_total)
            raise MemoryError()

        for t in range(n_threads):
            failed[t] = False
            out_counts[t] = 0
            out_caps[t] = 1024
            out_bufs[t] = <int64_t *>malloc(out_caps[t] * 2 * sizeof(int64_t))
            touched_bufs[t] = <int64_t *>malloc(capacity * sizeof(int64_t)) if capacity > 0 else NULL
            visited_stamps[t] = <int64_t *>calloc(<size_t>capacity, sizeof(int64_t)) if capacity > 0 else NULL
            touched_total[t] = 0
            valid_total[t] = 0
            if out_bufs[t] == NULL or (capacity > 0 and (touched_bufs[t] == NULL or visited_stamps[t] == NULL)):
                failed[t] = True

        with nogil:
            for i in prange(n_recent, schedule='static', num_threads=n_threads):
                tid = threadid()
                if failed[tid]:
                    continue
                q_entry = <int64_t>recent_entry_ids[i]
                if q_entry < 0 or q_entry >= self._count or not alive[q_entry]:
                    continue
                q_win = window_idx[q_entry]
                q_sid = sid_idx[q_entry]
                q_time = time_idx[q_entry]
                q_w = window_size[q_entry]
                stamp = <int64_t>(i + 1)
                # (2026-09-13) touched_count comes from a single plain `=` call into a
                # standalone nogil helper (not accumulated via += right here) -- see the
                # comment above _lsh_scan_touched_bands for why that distinction matters
                # inside a prange loop body (Cython's reduction-variable inference).
                touched_count = _lsh_scan_touched_bands(
                    band_keys, neg_band_keys, self._post_members, self._post_count,
                    visited_stamps[tid], touched_bufs[tid], self._n_bands, n_variants,
                    self._band_mode, self._complement_mask, self._bucket_count,
                    q_entry, stamp, self._max_candidates_per_query,
                )
                touched_total[tid] += touched_count
                for k in range(touched_count):
                    cand = touched_bufs[tid][k]
                    if not alive[cand]:
                        continue
                    if not (time_idx[cand] >= self._min_valid_time
                            and window_idx[cand] != q_win
                            and window_size[cand] == q_w
                            and not (sid_idx[cand] == q_sid and time_idx[cand] == q_time)):
                        continue
                    valid_total[tid] += 1
                    if self._apply_hamming_filter:
                        hbest = _hamming_best_dist(words, word_masks, self._n_words, q_entry, cand, signed_abs)
                        if hbest > self._hamming_max_bits:
                            continue
                    if self._apply_dot_filter:
                        score = _dot_score(vectors, q_entry, cand, self._n_vectors)
                        passes = _passes_dot_gamma_gate(score, gamma, signed_abs, True)
                        if not passes:
                            continue
                    if not _append_pair(&out_bufs[tid], &out_counts[tid], &out_caps[tid], q_win, window_idx[cand]):
                        failed[tid] = True
                        break

        for t in range(n_threads):
            if failed[t]:
                for tt in range(n_threads):
                    if out_bufs[tt] != NULL: free(out_bufs[tt])
                    if touched_bufs[tt] != NULL: free(touched_bufs[tt])
                    if visited_stamps[tt] != NULL: free(visited_stamps[tt])
                free(out_bufs); free(touched_bufs); free(visited_stamps)
                free(out_counts); free(out_caps); free(failed)
                free(touched_total); free(valid_total)
                raise MemoryError()
            total += out_counts[t]
            total_touched += touched_total[t]
            total_valid += valid_total[t]

        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t[:] win_sid_idx, win_sid_rank, win_time, win_w

        if total <= 0:
            for t in range(n_threads):
                free(out_bufs[t]); free(touched_bufs[t]); free(visited_stamps[t])
            free(out_bufs); free(touched_bufs); free(visited_stamps)
            free(out_counts); free(out_caps); free(failed)
            free(touched_total); free(valid_total)
            self.last_stats = {
                "num_index_candidates": int(total_touched), "num_valid_index_candidates": int(total_valid),
                "lsh_candidates_touched": int(total_touched), "lsh_candidates_returned": 0,
                "num_recent_queries": int(n_recent), "num_entries": int(self._alive_count),
                "gamma": float(gamma), "tau": 0.0, "num_rows": 0,
                "lsh_query_time": float(time.perf_counter() - t0), "n_threads": int(n_threads),
            }
            return np.empty((0, 5), dtype=np.int64)

        rows = np.empty((total, 5), dtype=np.int64)
        row_view = rows
        win_sid_idx = self._win_sid_idx_arr
        win_sid_rank = self._win_sid_rank_arr
        win_time = self._win_time_arr
        win_w = self._win_size_arr
        out_i = 0
        for t in range(n_threads):
            for i in range(out_counts[t]):
                a = out_bufs[t][2 * i]
                b = out_bufs[t][2 * i + 1]
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

        for t in range(n_threads):
            free(out_bufs[t]); free(touched_bufs[t]); free(visited_stamps[t])
        free(out_bufs); free(touched_bufs); free(visited_stamps)
        free(out_counts); free(out_caps); free(failed)
        free(touched_total); free(valid_total)

        cdef object rows_final = rows[:out_i, :] if out_i != total else rows
        if rows_final.shape[0] > 1:
            keys = np.ascontiguousarray(rows_final).view(
                np.dtype((np.void, rows_final.dtype.itemsize * 5))
            ).ravel()
            _, uniq_idx = np.unique(keys, return_index=True)
            rows_final = np.ascontiguousarray(rows_final[np.sort(uniq_idx)])

        self.last_stats = {
            "num_index_candidates": int(total_touched), "num_valid_index_candidates": int(total_valid),
            "lsh_candidates_touched": int(total_touched), "lsh_candidates_returned": int(rows_final.shape[0]),
            "num_recent_queries": int(n_recent), "num_entries": int(self._alive_count),
            "gamma": float(gamma), "tau": 0.0, "num_rows": int(rows_final.shape[0]),
            "lsh_query_time": float(time.perf_counter() - t0), "n_threads": int(n_threads),
        }
        return rows_final

cdef class HammingExactIndex:
    """Exact packed-bit Hamming pre-filter (candidate_backend="lsh_hamming_exact").

    (2026-07-10) "Outside the box" idea 3, diagnosed in
    diag_lsh_outside_box.py / bench_packed_hamming.py before being built for
    real -- see docs/implementation_log.md. SignLSHBandIndex's OR-of-many-
    bands retrieval is a PROBABILISTIC approximation of sign-pattern
    similarity, and its query cost is dominated by scattered per-band
    posting-list lookups (128 separate pointer-chases per query), not by
    anything intrinsic to comparing sign bits. Since n_vectors fits in a
    small, fixed number of machine words (n_words = ceil(n_vectors/64)),
    this class instead computes the EXACT Hamming distance from a query to
    EVERY alive candidate via one tight loop of XOR + popcount -- no bands,
    no buckets, no probabilistic retrieval mechanism at all. Measured
    (bench_packed_hamming.py, real dataset, alive=550): an unoptimized pure
    Python/numpy version of this scan (25.3us/query) already beat the real
    compiled SignLSHBandIndex (65.6us/query) -- this Cython version should
    do better still.

    Two-stage cascade, same contract as SignLSHBandIndex: stage 1 (Hamming
    distance <= _hamming_threshold) is a pre-filter -- EXACT (not
    probabilistic, unlike band matching), but still a HEURISTIC in the
    sense that the threshold itself is derived from the SimHash sign-
    agreement relation (expected_hamming = n_vectors * acos(gamma) / pi)
    plus a margin, not a proof that every true match has Hamming distance
    below it. Recall is an empirical property of this margin, exactly like
    every other backend's own threshold choice -- measured via a dedicated
    recall-vs-bruteforce test, not assumed. Stage 2 (the dot+gamma gate,
    via the shared _passes_dot_gamma_gate helper) is exact by construction
    when apply_dot_filter=True, identical to every other backend.

    _hamming_threshold is sized ONCE, lazily, on the first query call --
    the earliest point gamma is actually known (gamma is a per-query
    argument here, not a constructor argument, unlike SignLSHBandIndex's
    band_width which only needs alive-population size). NOT re-sized on
    later calls, matching every other backend's "size once" philosophy.

    Deletion is O(1) per node via a swap-remove maintained alive-list
    (_alive_list/_alive_pos) -- there are no posting lists to unlink from
    at all in this design (that is the whole point), so this class has
    none of SignLSHBandIndex's original unbounded-tombstone-accumulation
    risk to begin with; nothing to get wrong there.
    """
    cdef public object last_stats
    cdef Py_ssize_t _n_vectors
    cdef Py_ssize_t _n_words
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _count
    cdef Py_ssize_t _hamming_threshold
    cdef bint _threshold_sized
    cdef bint _apply_dot_filter
    cdef int64_t _min_valid_time
    cdef Py_ssize_t _alive_count
    cdef Py_ssize_t _dead_count
    # (2026-09-07) Free-list of dead slot indices available for reuse --
    # same fix and rationale as SignLSHBandIndex's own _free_slots (see its
    # field comment and docs/implementation_log.md's memory-leak fix entry).
    cdef Py_ssize_t *_free_slots
    cdef Py_ssize_t _free_count
    cdef Py_ssize_t _free_capacity

    cdef object _words           # (capacity, n_words) int64 -- packed sign bits
    cdef object _word_masks      # (n_words,) uint64 -- valid-bit mask per word (last word may be partial)
    cdef object _vectors         # (capacity, n_vectors) float64
    cdef object _alive           # (capacity,) uint8
    cdef object _window_idx
    cdef object _sid_idx
    cdef object _sid_rank
    cdef object _time
    cdef object _window_size
    cdef object _entry_ids
    cdef object _alive_list      # (capacity,) int64 -- packed list of currently-alive node indices
    cdef object _alive_pos       # (capacity,) int64 -- node_idx -> position within _alive_list

    cdef Py_ssize_t _win_capacity
    cdef object _win_sid_idx_arr
    cdef object _win_sid_rank_arr
    cdef object _win_time_arr
    cdef object _win_size_arr

    cdef int64_t *_pair_out_buf
    cdef Py_ssize_t _pair_out_cap
    cdef uint8_t *_pair_seen_occupied_buf
    cdef int64_t *_pair_seen_a_buf
    cdef int64_t *_pair_seen_b_buf
    cdef Py_ssize_t _pair_seen_cap_buf

    def __cinit__(self, Py_ssize_t n_vectors=1, Py_ssize_t initial_capacity=1024,
                  Py_ssize_t hamming_threshold=-1, bint apply_dot_filter=True):
        cdef Py_ssize_t w, bits_in_word
        cdef uint64_t mask
        if n_vectors <= 0:
            n_vectors = 1
        if initial_capacity < 16:
            initial_capacity = 16
        self._n_vectors = n_vectors
        self._n_words = (n_vectors + 63) // 64
        self._capacity = initial_capacity
        self._count = 0
        self._hamming_threshold = hamming_threshold
        self._threshold_sized = hamming_threshold >= 0
        self._apply_dot_filter = apply_dot_filter
        self._min_valid_time = -9223372036854775807
        self._alive_count = 0
        self._dead_count = 0
        self._free_slots = NULL
        self._free_count = 0
        self._free_capacity = 0

        self._words = np.zeros((self._capacity, self._n_words), dtype=np.int64)
        word_masks_np = np.empty(self._n_words, dtype=np.uint64)
        for w in range(self._n_words):
            bits_in_word = n_vectors - w * 64
            if bits_in_word > 64:
                bits_in_word = 64
            if bits_in_word >= 64:
                mask = <uint64_t>0xFFFFFFFFFFFFFFFFULL
            else:
                mask = (<uint64_t>1 << bits_in_word) - <uint64_t>1
            word_masks_np[w] = mask
        self._word_masks = word_masks_np
        self._vectors = np.empty((self._capacity, self._n_vectors), dtype=np.float64)
        self._alive = np.zeros(self._capacity, dtype=np.uint8)
        self._window_idx = np.empty(self._capacity, dtype=np.int64)
        self._sid_idx = np.empty(self._capacity, dtype=np.int64)
        self._sid_rank = np.empty(self._capacity, dtype=np.int64)
        self._time = np.empty(self._capacity, dtype=np.int64)
        self._window_size = np.empty(self._capacity, dtype=np.int64)
        self._entry_ids = np.empty(self._capacity, dtype=np.int64)
        self._alive_list = np.full(self._capacity, -1, dtype=np.int64)
        self._alive_pos = np.full(self._capacity, -1, dtype=np.int64)

        self._win_capacity = initial_capacity
        self._win_sid_idx_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_sid_rank_arr = np.full(self._win_capacity, -1, dtype=np.int64)
        self._win_time_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self._win_size_arr = np.zeros(self._win_capacity, dtype=np.int64)
        self.last_stats = {}

        self._pair_out_cap = 1024
        self._pair_out_buf = <int64_t *>malloc(self._pair_out_cap * 2 * sizeof(int64_t))
        self._pair_seen_cap_buf = 0
        self._pair_seen_occupied_buf = NULL
        self._pair_seen_a_buf = NULL
        self._pair_seen_b_buf = NULL
        if self._pair_out_buf == NULL:
            raise MemoryError("HammingExactIndex: failed to allocate persistent scratch buffers")

    def __dealloc__(self):
        if self._pair_out_buf != NULL:
            free(self._pair_out_buf)
        if self._free_slots != NULL:
            free(self._free_slots)
        if self._pair_seen_occupied_buf != NULL:
            free(self._pair_seen_occupied_buf)
        if self._pair_seen_a_buf != NULL:
            free(self._pair_seen_a_buf)
        if self._pair_seen_b_buf != NULL:
            free(self._pair_seen_b_buf)

    cdef void _finalize_threshold(self, double gamma):
        # (2026-07-10) SimHash expected-Hamming-distance relation: for two
        # vectors with true cosine similarity rho, the expected fraction of
        # disagreeing sign bits is acos(rho)/pi. Margin is a fixed number
        # of standard deviations of a binomial(n_vectors, p<=0.5) sign-flip
        # count (std <= sqrt(n_vectors)/2), rounded up generously to
        # sqrt(n_vectors) -- a documented starting point, not a proof;
        # validate via the recall-vs-bruteforce test before trusting it at
        # a new (n_vectors, gamma) operating point.
        cdef double g = gamma
        cdef double expected, margin
        cdef Py_ssize_t threshold
        if g < -1.0:
            g = -1.0
        if g > 1.0:
            g = 1.0
        expected = <double>self._n_vectors * acos(g) / _PI
        margin = sqrt(<double>self._n_vectors)
        threshold = <Py_ssize_t>ceil(expected + margin)
        if threshold < 0:
            threshold = 0
        if threshold > self._n_vectors:
            threshold = self._n_vectors
        self._hamming_threshold = threshold
        self._threshold_sized = True

    cdef void _ensure_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        if need <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < need:
            new_cap *= 2
        new_words = np.zeros((new_cap, self._n_words), dtype=np.int64)
        new_words[:self._capacity, :] = self._words
        self._words = new_words
        new_vectors = np.empty((new_cap, self._n_vectors), dtype=np.float64)
        new_vectors[:self._capacity, :] = self._vectors
        self._vectors = new_vectors
        new_alive = np.zeros(new_cap, dtype=np.uint8)
        new_alive[:self._capacity] = self._alive
        self._alive = new_alive
        new_window_idx = np.empty(new_cap, dtype=np.int64)
        new_window_idx[:self._capacity] = self._window_idx
        self._window_idx = new_window_idx
        new_sid_idx = np.empty(new_cap, dtype=np.int64)
        new_sid_idx[:self._capacity] = self._sid_idx
        self._sid_idx = new_sid_idx
        new_sid_rank = np.empty(new_cap, dtype=np.int64)
        new_sid_rank[:self._capacity] = self._sid_rank
        self._sid_rank = new_sid_rank
        new_time = np.empty(new_cap, dtype=np.int64)
        new_time[:self._capacity] = self._time
        self._time = new_time
        new_window_size = np.empty(new_cap, dtype=np.int64)
        new_window_size[:self._capacity] = self._window_size
        self._window_size = new_window_size
        new_entry_ids = np.empty(new_cap, dtype=np.int64)
        new_entry_ids[:self._capacity] = self._entry_ids
        self._entry_ids = new_entry_ids
        new_alive_list = np.full(new_cap, -1, dtype=np.int64)
        new_alive_list[:self._capacity] = self._alive_list
        self._alive_list = new_alive_list
        new_alive_pos = np.full(new_cap, -1, dtype=np.int64)
        new_alive_pos[:self._capacity] = self._alive_pos
        self._alive_pos = new_alive_pos
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

    cdef void _push_free_slot(self, Py_ssize_t idx):
        # (2026-09-07) See _free_slots' field comment / SignLSHBandIndex's
        # own identical helper.
        cdef Py_ssize_t new_cap
        cdef Py_ssize_t *grown
        if self._free_count >= self._free_capacity:
            new_cap = 4 if self._free_capacity == 0 else self._free_capacity * 2
            grown = <Py_ssize_t *>realloc(self._free_slots, new_cap * sizeof(Py_ssize_t))
            if grown == NULL:
                raise MemoryError("HammingExactIndex: failed to grow free-slot list")
            self._free_slots = grown
            self._free_capacity = new_cap
        self._free_slots[self._free_count] = idx
        self._free_count += 1

    cdef int64_t _insert_one(self, double[:] vector, int64_t window_idx, int64_t sid_idx,
                              int64_t time_val, int64_t window_size, int64_t sid_rank):
        cdef Py_ssize_t node
        cdef Py_ssize_t w, d, bit, bits_in_word
        cdef uint64_t word_tmp
        cdef int64_t entry_id
        # (2026-09-07) Reuse a freed slot before ever growing _count/
        # _capacity -- see _free_slots' field comment. Every field at node
        # is unconditionally overwritten below, so a reused slot needs no
        # separate clearing.
        if self._free_count > 0:
            self._free_count -= 1
            node = self._free_slots[self._free_count]
        else:
            node = self._count
            self._ensure_capacity(node + 1)
            self._count += 1
        # (2026-09-07) entry_id MUST equal the physical slot index: query
        # code (_find_pair_rows_meta) uses a returned entry_id directly as
        # a subscript into every per-slot array (alive/window_idx/words/
        # ...), not as an independent, separately-tracked id -- confirmed by
        # reading that code before making entry_id reusable-slot-safe. A
        # once-monotonic, never-reused _next_entry_id (the pre-fix scheme)
        # only ever coincided with the slot index because, before this fix,
        # slots were NEVER reused either -- the two counters were silently
        # always in lockstep. Reusing slots without also making entry_id
        # track the slot broke that coincidence and caused real, confirmed
        # query misindexing (see docs/implementation_log.md's memory-leak
        # fix entry for the two tests that caught this).
        entry_id = node
        cdef double[:, :] vectors = self._vectors
        cdef int64_t[:, :] words = self._words
        cdef uint8_t[:] alive = self._alive
        cdef int64_t[:] window_idx_arr = self._window_idx
        cdef int64_t[:] sid_idx_arr = self._sid_idx
        cdef int64_t[:] sid_rank_arr = self._sid_rank
        cdef int64_t[:] time_arr = self._time
        cdef int64_t[:] window_size_arr = self._window_size
        cdef int64_t[:] entry_ids_arr = self._entry_ids
        cdef int64_t[:] alive_list = self._alive_list
        cdef int64_t[:] alive_pos = self._alive_pos

        for d in range(self._n_vectors):
            vectors[node, d] = vector[d]
        for w in range(self._n_words):
            bits_in_word = self._n_vectors - w * 64
            if bits_in_word > 64:
                bits_in_word = 64
            word_tmp = 0
            for bit in range(bits_in_word):
                if vector[w * 64 + bit] >= 0:
                    word_tmp |= (<uint64_t>1) << bit
            words[node, w] = <int64_t>word_tmp
        alive[node] = 1
        window_idx_arr[node] = window_idx
        sid_idx_arr[node] = sid_idx
        sid_rank_arr[node] = sid_rank
        time_arr[node] = time_val
        window_size_arr[node] = window_size
        entry_ids_arr[node] = entry_id

        alive_list[self._alive_count] = node
        alive_pos[node] = self._alive_count
        self._alive_count += 1

        return entry_id

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

        if n == 0:
            return np.empty(0, dtype=np.int64)

        if vectors_in is None:
            raise ValueError("HammingExactIndex requires vectors_in")
        vectors_np = np.ascontiguousarray(vectors_in, dtype=np.float64)
        if vectors_np.ndim != 2 or vectors_np.shape[0] != n:
            raise ValueError("vectors must be a 2D array with one row per value")
        if vectors_np.shape[1] != self._n_vectors:
            raise ValueError("vector size does not match HammingExactIndex dimension")
        sid_idx_np = np.asarray(sid_idx_in, dtype=np.int64).ravel() if sid_idx_in is not None else np.full(n, -1, dtype=np.int64)
        time_np = np.asarray(time_in, dtype=np.int64).ravel() if time_in is not None else np.zeros(n, dtype=np.int64)
        window_size_np = np.asarray(window_size_in, dtype=np.int64).ravel() if window_size_in is not None else np.zeros(n, dtype=np.int64)
        sid_rank_np = np.asarray(sid_rank_in, dtype=np.int64).ravel() if sid_rank_in is not None else np.full(n, -1, dtype=np.int64)
        if (sid_idx_np.shape[0] != n or time_np.shape[0] != n
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
        return entry_ids

    cdef void _remove_from_alive_list(self, Py_ssize_t node_idx):
        # O(1) swap-remove: move the LAST alive-list entry into the
        # expiring node's slot, shrink the count. No posting lists exist
        # in this design, so there is nothing else to unlink.
        cdef int64_t[:] alive_list = self._alive_list
        cdef int64_t[:] alive_pos = self._alive_pos
        cdef Py_ssize_t pos = <Py_ssize_t>alive_pos[node_idx]
        cdef Py_ssize_t last_pos = self._alive_count - 1
        cdef int64_t last_node
        if pos != last_pos:
            last_node = alive_list[last_pos]
            alive_list[pos] = last_node
            alive_pos[last_node] = pos
        alive_list[last_pos] = -1
        alive_pos[node_idx] = -1
        self._alive_count -= 1

    cpdef drop_before_time(self, long min_valid_time):
        cdef Py_ssize_t i
        cdef int64_t[:] time_mv = self._time
        cdef uint8_t[:] alive_mv = self._alive
        self._min_valid_time = <int64_t>min_valid_time
        for i in range(self._count):
            if alive_mv[i] and time_mv[i] < self._min_valid_time:
                alive_mv[i] = 0
                self._dead_count += 1
                self._remove_from_alive_list(i)
                # (2026-09-07) Make the slot reusable -- see _free_slots'
                # field comment and the memory-leak fix entry in
                # docs/implementation_log.md.
                self._push_free_slot(i)

    cpdef clear_recent(self):
        pass

    cdef object _empty_stats(self, double gamma):
        return {
            "num_index_candidates": 0, "num_valid_index_candidates": 0,
            "num_enumerated_candidates": 0,
            "num_unique_index_candidates": 0, "num_duplicate_index_candidates": 0,
            "num_unique_pre_dot_pairs": 0, "num_duplicate_pre_dot_pairs": 0,
            "num_dot_checks": 0, "num_distance_checks": 0,
            "num_after_similarity": 0, "num_after_dot": 0,
            "num_pairs_before_dedupe": 0, "num_pairs_after_dedupe": 0,
            "num_rows": 0, "num_recent_queries": 0, "num_entries": int(self._alive_count),
            "num_blocks": 1, "gamma": float(gamma), "tau": 0.0,
            "hexact_candidates_enumerated": 0,
            "hexact_candidates_touched": 0, "hexact_dot_checks": 0,
            "hexact_candidates_returned": 0, "hexact_query_time": 0.0,
            "hexact_hamming_threshold": int(self._hamming_threshold),
            "hexact_num_nodes_total": int(self._count), "hexact_num_nodes_alive": int(self._alive_count),
            "hexact_dead_node_ratio": float(self._dead_count) / float(self._count) if self._count > 0 else 0.0,
        }

    cpdef object find_pair_rows_full_cosine(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_meta(recent_entry_ids, gamma, False)

    cpdef object find_pair_rows_full_cosine_signed(self, long[:] recent_entry_ids, double gamma, double tau):
        return self._find_pair_rows_meta(recent_entry_ids, gamma, True)

    cdef object _find_pair_rows_meta(self, long[:] recent_entry_ids, double gamma, bint signed_abs):
        cdef Py_ssize_t n_recent = recent_entry_ids.shape[0]
        cdef Py_ssize_t i, node_pos, w, count, out_i
        cdef int64_t q_entry, q_win, q_sid, q_rank, q_time, q_w, node
        cdef Py_ssize_t d
        cdef double score
        cdef bint passes, failed = False
        cdef Py_ssize_t pair_seen_count = 0
        cdef Py_ssize_t wanted_pair_seen_cap
        cdef int64_t pair_a, pair_b, a, b
        cdef int64_t sid_a, sid_b, rank_a, rank_b, time_a, time_b, size_a, size_b
        cdef Py_ssize_t total_enumerated = 0
        cdef Py_ssize_t total_touched = 0, total_dot_checks = 0, total_valid = 0
        cdef Py_ssize_t unique_pre_dot_pairs = 0, duplicate_pre_dot_pairs = 0
        cdef Py_ssize_t after_similarity = 0, pairs_before_dedupe = 0, pairs_after_dedupe = 0
        cdef np.ndarray[np.int64_t, ndim=2] rows
        cdef int64_t[:, :] row_view
        cdef int64_t[:] win_sid_idx, win_sid_rank, win_time, win_w
        cdef int64_t hpos, hneg, hbest
        cdef uint64_t qw_word, cw_word, xw_mask
        cdef double t0 = time.perf_counter()

        if n_recent == 0 or self._alive_count == 0:
            self.last_stats = self._empty_stats(gamma)
            return np.empty((0, 5), dtype=np.int64)

        if not self._threshold_sized:
            self._finalize_threshold(gamma)

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

        cdef int64_t[:, :] words = self._words
        cdef uint64_t[:] word_masks = self._word_masks
        cdef double[:, :] vectors = self._vectors
        cdef uint8_t[:] alive = self._alive
        cdef int64_t[:] window_idx = self._window_idx
        cdef int64_t[:] sid_idx = self._sid_idx
        cdef int64_t[:] sid_rank = self._sid_rank
        cdef int64_t[:] time_idx = self._time
        cdef int64_t[:] window_size = self._window_size
        cdef int64_t[:] alive_list = self._alive_list

        # (2026-09-14) Fix for the measured ~1.9x redundant-enumeration finding
        # (docs/implementation_log.md's 2026-09-14 entry, "Hamming-exact enumeration
        # redundancy"): every one of the n_recent queries this step scans the FULL alive
        # set, which includes the OTHER n_recent-1 sibling queries (all inserted at the
        # SAME time this same step, before any of them search) -- so a same-time pair
        # (A, B) gets independently discovered TWICE: once when A's query scans and finds
        # B, once more when B's query scans and finds A. This is the ONLY source of true
        # redundancy here (a "recent vs older" comparison is never repeated -- verified by
        # direct reasoning: the older side's own time was already fixed before the newer
        # side's window existed, so no future or past step ever re-examines that exact
        # (t1, t2) combination). is_recent + an EXPLICIT time_idx match + a sid_idx
        # tie-break below makes each same-time pair discoverable from exactly one
        # direction, chosen by a fixed, arbitrary-but-consistent order (lower sid_idx's
        # query claims it) -- this drops work, not coverage: pair_seen already deduped the
        # OUTPUT before this fix, so the candidate SET returned is unaffected, only the
        # redundant computation is skipped.
        #
        # Two real bugs caught here before trusting this, not assumed safe (both found
        # directly, via SignLSHBandIndex's identical copy of this fix failing a real test
        # first): the time_idx check is required, not just defensive -- a caller may
        # legally pass a MIXED-time recent_entry_ids batch, where a rank/id check alone
        # would wrongly skip genuinely different-time comparisons; and the tie-break must
        # be sid_idx, not sid_rank -- sid_rank_in is a general caller-supplied parameter
        # with no uniqueness guarantee (one test passes the round number for it, shared by
        # every series in that round), so a `<=` on sid_rank ties for same-round pairs and
        # silently drops them from both directions. sid_idx is the one field guaranteed
        # unique per logical series by construction.
        cdef uint8_t[:] is_recent = np.zeros(self._count, dtype=np.uint8)
        for i in range(n_recent):
            q_entry = <int64_t>recent_entry_ids[i]
            if 0 <= q_entry < self._count:
                is_recent[q_entry] = 1

        count = 0
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

            # TRUE enumeration count: this loop always runs alive_count
            # times, unconditionally -- unlike `total_touched` below (which
            # only increments after a candidate survives the cheap Hamming
            # pre-filter), this is the actual O(N) work performed, matching
            # what "enumerated pairs" means for the genuinely-indexed
            # backends (a real visit count), not a post-filter survivor
            # count. See docs/implementation_log.md's 2026-07-15 "three
            # consistent enumeration metrics" entry.
            total_enumerated += self._alive_count
            for node_pos in range(self._alive_count):
                node = alive_list[node_pos]
                if node == q_entry:
                    continue
                if is_recent[node] and time_idx[node] == q_time and sid_idx[node] <= q_sid:
                    continue
                hpos = 0
                hneg = 0
                for w in range(self._n_words):
                    qw_word = <uint64_t>words[q_entry, w]
                    cw_word = <uint64_t>words[node, w]
                    xw_mask = word_masks[w]
                    hpos += _popcount64_swar((qw_word ^ cw_word) & xw_mask)
                    if signed_abs:
                        hneg += _popcount64_swar((qw_word ^ (~cw_word)) & xw_mask)
                hbest = hpos
                if signed_abs and hneg < hbest:
                    hbest = hneg
                if hbest > self._hamming_threshold:
                    continue
                total_touched += 1

                if not (time_idx[node] >= self._min_valid_time
                        and window_idx[node] != q_win
                        and window_size[node] == q_w
                        and not (sid_idx[node] == q_sid and time_idx[node] == q_time)):
                    continue
                total_valid += 1
                _canonical_window_pair(
                    q_win, q_sid, q_rank, q_time,
                    window_idx[node], sid_idx[node], sid_rank[node], time_idx[node],
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
                if self._apply_dot_filter:
                    score = 0.0
                    for d in range(self._n_vectors):
                        score += vectors[q_entry, d] * vectors[node, d]
                    total_dot_checks += 1
                    passes = _passes_dot_gamma_gate(score, gamma, signed_abs, True)
                    if not passes:
                        continue
                after_similarity += 1
                if not _append_pair(&self._pair_out_buf, &count, &self._pair_out_cap, q_win, window_idx[node]):
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
            "num_index_candidates": int(total_touched),
            "num_enumerated_candidates": int(total_enumerated),
            "num_valid_index_candidates": int(total_valid),
            "num_unique_index_candidates": int(total_valid),
            "num_duplicate_index_candidates": 0,
            "num_unique_pre_dot_pairs": int(unique_pre_dot_pairs),
            "num_duplicate_pre_dot_pairs": int(duplicate_pre_dot_pairs),
            "num_dot_checks": int(total_dot_checks),
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
            "hexact_candidates_enumerated": int(total_enumerated),
            "hexact_candidates_touched": int(total_touched),
            "hexact_dot_checks": int(total_dot_checks),
            "hexact_candidates_returned": int(after_similarity),
            "hexact_query_time": float(time.perf_counter() - t0),
            "hexact_hamming_threshold": int(self._hamming_threshold),
            "hexact_num_nodes_total": int(self._count),
            "hexact_num_nodes_alive": int(self._alive_count),
            "hexact_dead_node_ratio": float(self._dead_count) / float(self._count) if self._count > 0 else 0.0,
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

        # (2026-09-13) Preallocated numpy output arrays instead of a Python
        # list-of-tuples built one Python object per candidate -- profiling a
        # real dense run found this loop's OWN Python-object-construction cost
        # (repeated here again on the Python-side consumer, which re-iterated
        # the list) dominated validation_time by ~15x versus hybrid_validation
        # disabled, even at a 92-99% cache hit rate. Same fix shape as the
        # 2026-09-10 numeric-representation work (bulk numpy arrays in, bulk
        # numpy arrays out, no per-row Python object on the hot path).
        cdef np.ndarray[np.uint8_t, ndim=1] ok_arr = np.zeros(n_items, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] is_corr_arr = np.zeros(n_items, dtype=np.uint8)
        cdef np.ndarray[np.float64_t, ndim=1] corr_arr = np.full(n_items, np.nan, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] dist_arr = np.full(n_items, np.inf, dtype=np.float64)
        cdef np.ndarray[np.uint8_t, ndim=1] is_const_arr = np.zeros(n_items, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] is_spiked_arr = np.zeros(n_items, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] used_hybrid_arr = np.zeros(n_items, dtype=np.uint8)
        cdef uint8_t[:] ok_mv = ok_arr
        cdef uint8_t[:] is_corr_mv = is_corr_arr
        cdef double[:] corr_mv = corr_arr
        cdef double[:] dist_mv = dist_arr
        cdef uint8_t[:] is_const_mv = is_const_arr
        cdef uint8_t[:] is_spiked_mv = is_spiked_arr
        cdef uint8_t[:] used_hybrid_mv = used_hybrid_arr
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
                continue    # ok_mv[i] stays 0, other arrays keep their prefilled defaults

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

            ok_mv[i] = 1
            is_corr_mv[i] = 1 if is_corr else 0
            corr_mv[i] = corr
            dist_mv[i] = dist
            is_const_mv[i] = 1 if is_const else 0
            is_spiked_mv[i] = 1 if is_spiked else 0
            used_hybrid_mv[i] = 1 if used_hybrid else 0

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
            "ok": ok_arr,
            "is_corr": is_corr_arr,
            "corr": corr_arr,
            "dist": dist_arr,
            "is_const": is_const_arr,
            "is_spiked": is_spiked_arr,
            "used_hybrid": used_hybrid_arr,
        }


# (2026-07-30) Cython port of distance_corr_sketch.py's
# distance_corr_sketch_proxy (tier-2 refinement gate for
# candidate_backend="distance_corr_sketch"). Explicit user instruction:
# "all gates for all backends should be Cython, I want no Python" -- the
# Python version's own docstring/profiling already found 58% of its cost is
# pure-Python-loop overhead from 256 separate small np.dot() calls at K=8
# (2Kx2K cross-term grid), the exact pattern this file's other kernels
# (e.g. SignLSHBandIndex's inner comparison loops) exist to eliminate via
# one fused compiled loop. Math is identical to the Python version: build
# 2K per-channel (cos/sin at each frequency) mean-centered series for x and
# y, compute the full (2K)x(2K) grid of channel-pair cosine similarities,
# aggregate via sqrt(mean of squares). No representation/state beyond the
# raw x/y window slices and the shared freqs array -- matches the Python
# version's own "computed fresh at validation time, no incremental state
# needed" design exactly.
def distance_corr_sketch_proxy_cy(double[:] x, double[:] y, double[:] freqs):
    cdef Py_ssize_t w = x.shape[0]
    cdef Py_ssize_t K = freqs.shape[0]
    cdef Py_ssize_t n_chan = 2 * K
    cdef Py_ssize_t i, j, t, idx
    cdef double wf, mean_val, sum_val, dot, cos_sim, total
    cdef Py_ssize_t n_terms

    cdef double[:, ::1] chx = np.empty((n_chan, w), dtype=np.float64)
    cdef double[:, ::1] chy = np.empty((n_chan, w), dtype=np.float64)
    cdef double[::1] nx = np.empty(n_chan, dtype=np.float64)
    cdef double[::1] ny = np.empty(n_chan, dtype=np.float64)

    with nogil:
        for i in range(K):
            wf = freqs[i]
            idx = 2 * i
            sum_val = 0.0
            for t in range(w):
                chx[idx, t] = cos(wf * x[t])
                sum_val += chx[idx, t]
            mean_val = sum_val / w
            sum_val = 0.0
            for t in range(w):
                chx[idx, t] -= mean_val
                sum_val += chx[idx, t] * chx[idx, t]
            nx[idx] = sqrt(sum_val)

            idx = 2 * i + 1
            sum_val = 0.0
            for t in range(w):
                chx[idx, t] = sin(wf * x[t])
                sum_val += chx[idx, t]
            mean_val = sum_val / w
            sum_val = 0.0
            for t in range(w):
                chx[idx, t] -= mean_val
                sum_val += chx[idx, t] * chx[idx, t]
            nx[idx] = sqrt(sum_val)

        for i in range(K):
            wf = freqs[i]
            idx = 2 * i
            sum_val = 0.0
            for t in range(w):
                chy[idx, t] = cos(wf * y[t])
                sum_val += chy[idx, t]
            mean_val = sum_val / w
            sum_val = 0.0
            for t in range(w):
                chy[idx, t] -= mean_val
                sum_val += chy[idx, t] * chy[idx, t]
            ny[idx] = sqrt(sum_val)

            idx = 2 * i + 1
            sum_val = 0.0
            for t in range(w):
                chy[idx, t] = sin(wf * y[t])
                sum_val += chy[idx, t]
            mean_val = sum_val / w
            sum_val = 0.0
            for t in range(w):
                chy[idx, t] -= mean_val
                sum_val += chy[idx, t] * chy[idx, t]
            ny[idx] = sqrt(sum_val)

        total = 0.0
        n_terms = 0
        for i in range(n_chan):
            if nx[i] == 0.0:
                continue
            for j in range(n_chan):
                if ny[j] == 0.0:
                    continue
                dot = 0.0
                for t in range(w):
                    dot += chx[i, t] * chy[j, t]
                cos_sim = dot / (nx[i] * ny[j])
                total += cos_sim * cos_sim
                n_terms += 1

    if n_terms == 0:
        return 0.0
    return sqrt(total / n_terms)


# (2026-07-31) Cython ports of the exact Kendall's-tau-b / Spearman's-rho
# validation metrics -- explicit user instruction: "I want no hotpaths in
# python whatsoever." Before this, validation_metric="spearman"/"kendall"
# called scipy.stats.spearmanr/kendalltau directly, once per candidate row,
# from within library_corrtrack_parallel.py's own per-row Python loop
# (_validate_numeric_rows_nonlinear) -- full Python-level function-call
# overhead (argument marshalling, scipy's own general-purpose input
# validation) on every single candidate, every step. Ported the same
# algorithms scipy itself uses (Kendall: Knight 1966 merge-sort inversion
# counting + the standard tau-b tie correction; Spearman: average-rank
# transform, then Pearson on ranks) into tight nogil loops here -- same
# math, not an approximation (unlike the separate, still-Python, opt-in
# Xiao (2017) online validator). Verified bit-identical (~1e-13) against
# scipy.stats.kendalltau/spearmanr across 2000 random trials each,
# spanning heavy ties, monotonic-nonlinear, and independent data (see
# docs/implementation_log.md's 2026-07-31 entry). No Python fallback --
# raises if the compiled extension is unavailable, matching this project's
# existing strict pattern for distance_corr_sketch_proxy/candidate_backend=
# "flat"/"lsh_sign_dot".
cdef void _merge_sort_by_key_stable(
    double* keys, Py_ssize_t* idx,
    double* tmp_keys, Py_ssize_t* tmp_idx,
    Py_ssize_t n,
) noexcept nogil:
    """Stable bottom-up merge sort: sorts `keys` ascending in place while
    permuting the parallel array `idx` the same way (`idx` is caller-owned
    and typically initialized to 0..n-1 before the first call). Stability
    (the `keys[l] <= keys[r]` comparison below always prefers the left/
    earlier element on ties) is what lets two chained calls implement a
    two-key lexsort: sort by the secondary key first, then stable-sort the
    result by the primary key -- exactly matching np.lexsort's semantics
    and np.argsort(kind='mergesort')'s stability guarantee, needed here
    since bulk validation must reproduce the same tie-breaking scipy/numpy
    (and this project's own single-pair Cython kernels) already use."""
    cdef Py_ssize_t width, left, mid, right, l, r, k, i
    width = 1
    while width < n:
        left = 0
        while left < n:
            mid = left + width
            if mid > n:
                mid = n
            right = left + 2 * width
            if right > n:
                right = n
            if mid < right:
                l = left
                r = mid
                k = left
                while l < mid and r < right:
                    if keys[l] <= keys[r]:
                        tmp_keys[k] = keys[l]
                        tmp_idx[k] = idx[l]
                        l += 1
                    else:
                        tmp_keys[k] = keys[r]
                        tmp_idx[k] = idx[r]
                        r += 1
                    k += 1
                while l < mid:
                    tmp_keys[k] = keys[l]
                    tmp_idx[k] = idx[l]
                    l += 1
                    k += 1
                while r < right:
                    tmp_keys[k] = keys[r]
                    tmp_idx[k] = idx[r]
                    r += 1
                    k += 1
                for i in range(left, right):
                    keys[i] = tmp_keys[i]
                    idx[i] = tmp_idx[i]
            left += 2 * width
        width *= 2


cdef Py_ssize_t _merge_sort_count_inversions(double* arr, double* tmp, Py_ssize_t n) nogil:
    """Iterative (bottom-up) merge sort: sorts `arr` in place (ascending,
    using `tmp` as scratch space) and returns the number of inversions
    (pairs i<j with arr[i] > arr[j] in the ORIGINAL order) -- O(n log n),
    the classic Knight (1966) algorithm scipy's own _kendall_dis uses."""
    cdef Py_ssize_t width, left, mid, right, l, r, k, i
    cdef Py_ssize_t total = 0
    width = 1
    while width < n:
        left = 0
        while left < n:
            mid = left + width
            if mid > n:
                mid = n
            right = left + 2 * width
            if right > n:
                right = n
            if mid < right:
                l = left
                r = mid
                k = left
                while l < mid and r < right:
                    if arr[l] <= arr[r]:
                        tmp[k] = arr[l]
                        l += 1
                    else:
                        tmp[k] = arr[r]
                        r += 1
                        total += (mid - l)
                    k += 1
                while l < mid:
                    tmp[k] = arr[l]
                    l += 1
                    k += 1
                while r < right:
                    tmp[k] = arr[r]
                    r += 1
                    k += 1
                for i in range(left, right):
                    arr[i] = tmp[i]
            left += 2 * width
        width *= 2
    return total


def kendall_tau_cy(double[:] x, double[:] y):
    """Exact Kendall's tau-b (scipy.stats.kendalltau's default variant),
    handling ties in x, y, and jointly. x/y are lexsorted by (x, y) via
    numpy (a vectorized C op, not a Python-level loop) so the O(n log n)
    inversion count and all tie-counting run over already-grouped data;
    only the genuinely loop-shaped work (the merge-sort inversion count,
    the tie-run scans) happens inside the nogil block below."""
    cdef Py_ssize_t n = x.shape[0]
    if n < 2:
        return float("nan")

    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    perm = np.lexsort((y_np, x_np))
    # (2026-07-31) xs_arr/ys_arr and ys_work_arr must be genuinely
    # INDEPENDENT arrays, not aliased views over the same buffer --
    # ascontiguousarray() is a no-op (returns the SAME array, not a copy)
    # whenever its input is already contiguous, which fancy-indexed arrays
    # (x_np[perm]) already are. _merge_sort_count_inversions sorts its
    # argument in place; if ys_work_arr aliased ys_arr, that in-place sort
    # would silently corrupt the joint/x-tie scan below, which needs ys_arr
    # in its original (x,y)-lexsorted order, not fully re-sorted. np.array(
    # ..., copy=True) guarantees a real copy regardless of contiguity.
    xs_arr = np.ascontiguousarray(x_np[perm])
    ys_arr = np.ascontiguousarray(y_np[perm])
    ys_work_arr = np.array(ys_arr, copy=True)
    cdef double[::1] xs = xs_arr
    cdef double[::1] ys = ys_arr
    cdef double[::1] ys_work = ys_work_arr
    cdef double[::1] tmp = np.empty(n, dtype=np.float64)
    cdef double[::1] y_sorted_alone = np.ascontiguousarray(np.sort(y_np))

    cdef Py_ssize_t dis, ntie, xtie, ytie, tot
    cdef Py_ssize_t i, j, run_len, sub_len
    cdef double con_minus_dis, denom, tau

    with nogil:
        dis = _merge_sort_count_inversions(&ys_work[0], &tmp[0], n)

        ntie = 0
        xtie = 0
        i = 0
        while i < n:
            run_len = 1
            while i + run_len < n and xs[i + run_len] == xs[i]:
                run_len += 1
            xtie += (run_len * (run_len - 1)) // 2
            j = i
            while j < i + run_len:
                sub_len = 1
                while j + sub_len < i + run_len and ys[j + sub_len] == ys[j]:
                    sub_len += 1
                ntie += (sub_len * (sub_len - 1)) // 2
                j += sub_len
            i += run_len

        ytie = 0
        i = 0
        while i < n:
            run_len = 1
            while i + run_len < n and y_sorted_alone[i + run_len] == y_sorted_alone[i]:
                run_len += 1
            ytie += (run_len * (run_len - 1)) // 2
            i += run_len

    tot = (n * (n - 1)) // 2
    if xtie == tot or ytie == tot:
        return float("nan")
    con_minus_dis = <double>(tot - xtie - ytie + ntie - 2 * dis)
    denom = sqrt(<double>(tot - xtie) * <double>(tot - ytie))
    if denom <= 0.0:
        return float("nan")
    tau = con_minus_dis / denom
    if tau > 1.0:
        tau = 1.0
    elif tau < -1.0:
        tau = -1.0
    return tau


cdef double _kendall_tau_row_nogil(
    double* x, double* y, Py_ssize_t n,
    double* key_buf, double* tmp_key,
    Py_ssize_t* idx, Py_ssize_t* tmp_idx,
    double* xs, double* ys, double* ys_work, double* y_sorted_alone,
    double nan_value,
) noexcept nogil:
    """Row-level Kendall's tau-b, identical algorithm/formula to
    kendall_tau_cy above, but operating entirely on caller-supplied scratch
    buffers with no numpy calls at all -- built for validate_corr_rows_
    nonlinear_cy's single bulk nogil loop over many candidate rows, where
    per-row numpy calls (as kendall_tau_cy itself uses for its lexsort)
    would reintroduce per-row Python-level overhead. The (x,y)-lexsort is
    reproduced via two chained _merge_sort_by_key_stable calls (sort by y,
    then a STABLE sort by x -- see that function's own docstring for why
    this reproduces np.lexsort((y, x))'s exact semantics)."""
    cdef Py_ssize_t i, j, run_len, sub_len, dis, ntie, xtie, ytie, tot
    cdef double con_minus_dis, denom, tau

    if n < 2:
        return nan_value

    for i in range(n):
        y_sorted_alone[i] = y[i]
        idx[i] = i
    _merge_sort_by_key_stable(y_sorted_alone, idx, tmp_key, tmp_idx, n)

    for i in range(n):
        idx[i] = i
        key_buf[i] = y[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    for i in range(n):
        key_buf[i] = x[idx[i]]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    for i in range(n):
        xs[i] = x[idx[i]]
        ys[i] = y[idx[i]]
        ys_work[i] = ys[i]

    dis = _merge_sort_count_inversions(ys_work, tmp_key, n)

    ntie = 0
    xtie = 0
    i = 0
    while i < n:
        run_len = 1
        while i + run_len < n and xs[i + run_len] == xs[i]:
            run_len += 1
        xtie += (run_len * (run_len - 1)) // 2
        j = i
        while j < i + run_len:
            sub_len = 1
            while j + sub_len < i + run_len and ys[j + sub_len] == ys[j]:
                sub_len += 1
            ntie += (sub_len * (sub_len - 1)) // 2
            j += sub_len
        i += run_len

    ytie = 0
    i = 0
    while i < n:
        run_len = 1
        while i + run_len < n and y_sorted_alone[i + run_len] == y_sorted_alone[i]:
            run_len += 1
        ytie += (run_len * (run_len - 1)) // 2
        i += run_len

    tot = (n * (n - 1)) // 2
    if xtie == tot or ytie == tot:
        return nan_value
    con_minus_dis = <double>(tot - xtie - ytie + ntie - 2 * dis)
    denom = sqrt(<double>(tot - xtie) * <double>(tot - ytie))
    if denom <= 0.0:
        return nan_value
    tau = con_minus_dis / denom
    if tau > 1.0:
        tau = 1.0
    elif tau < -1.0:
        tau = -1.0
    return tau


def spearman_rho_cy(double[:] x, double[:] y):
    """Exact Spearman's rho: average-rank transform (ties broken via the
    mean of tied positions, matching scipy.stats.spearmanr), then Pearson
    correlation on the ranks. Sorting (for rank assignment) is a vectorized
    numpy op; the tie-run scans and the correlation reduction run inside
    the nogil block below."""
    cdef Py_ssize_t n = x.shape[0]
    if n < 2:
        return float("nan")

    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    order_x = np.argsort(x_np, kind="mergesort")
    order_y = np.argsort(y_np, kind="mergesort")
    cdef double[::1] xs = np.ascontiguousarray(x_np[order_x])
    cdef double[::1] ys = np.ascontiguousarray(y_np[order_y])
    cdef int64_t[::1] ox = np.ascontiguousarray(order_x.astype(np.int64))
    cdef int64_t[::1] oy = np.ascontiguousarray(order_y.astype(np.int64))
    cdef double[::1] rx = np.empty(n, dtype=np.float64)
    cdef double[::1] ry = np.empty(n, dtype=np.float64)

    cdef Py_ssize_t i, j, run_len
    cdef double avg_rank, mx, my, cx, cy, num, denomx, denomy, denom, rho

    with nogil:
        i = 0
        while i < n:
            run_len = 1
            while i + run_len < n and xs[i + run_len] == xs[i]:
                run_len += 1
            avg_rank = (<double>i + <double>(i + run_len - 1)) / 2.0 + 1.0
            for j in range(i, i + run_len):
                rx[ox[j]] = avg_rank
            i += run_len

        i = 0
        while i < n:
            run_len = 1
            while i + run_len < n and ys[i + run_len] == ys[i]:
                run_len += 1
            avg_rank = (<double>i + <double>(i + run_len - 1)) / 2.0 + 1.0
            for j in range(i, i + run_len):
                ry[oy[j]] = avg_rank
            i += run_len

        mx = 0.0
        my = 0.0
        for i in range(n):
            mx += rx[i]
            my += ry[i]
        mx /= n
        my /= n

        num = 0.0
        denomx = 0.0
        denomy = 0.0
        for i in range(n):
            cx = rx[i] - mx
            cy = ry[i] - my
            num += cx * cy
            denomx += cx * cx
            denomy += cy * cy

    denom = sqrt(denomx * denomy)
    if denom <= 0.0:
        return float("nan")
    rho = num / denom
    if rho > 1.0:
        rho = 1.0
    elif rho < -1.0:
        rho = -1.0
    return rho


cdef double _spearman_rho_row_nogil(
    double* x, double* y, Py_ssize_t n,
    double* key_buf, double* tmp_key,
    Py_ssize_t* idx, Py_ssize_t* tmp_idx,
    double* rx, double* ry,
    double nan_value,
) noexcept nogil:
    """Row-level Spearman's rho, identical algorithm to spearman_rho_cy
    above (average-rank transform, ties broken by mean position, then
    Pearson on ranks), but using caller-supplied scratch buffers and no
    numpy calls -- see _kendall_tau_row_nogil's docstring for why. x and y
    are each argsorted independently (not lexsorted, unlike Kendall) via
    _merge_sort_by_key_stable; key_buf holds the sorted VALUES in place
    after each call (no separate xs/ys buffer needed), with idx holding
    the matching original-index permutation used to scatter average ranks
    back to rx/ry."""
    cdef Py_ssize_t i, j, run_len
    cdef double avg_rank, mx, my, cx, cy, num, denomx, denomy, denom, rho

    if n < 2:
        return nan_value

    for i in range(n):
        idx[i] = i
        key_buf[i] = x[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    i = 0
    while i < n:
        run_len = 1
        while i + run_len < n and key_buf[i + run_len] == key_buf[i]:
            run_len += 1
        avg_rank = (<double>i + <double>(i + run_len - 1)) / 2.0 + 1.0
        for j in range(i, i + run_len):
            rx[idx[j]] = avg_rank
        i += run_len

    for i in range(n):
        idx[i] = i
        key_buf[i] = y[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    i = 0
    while i < n:
        run_len = 1
        while i + run_len < n and key_buf[i + run_len] == key_buf[i]:
            run_len += 1
        avg_rank = (<double>i + <double>(i + run_len - 1)) / 2.0 + 1.0
        for j in range(i, i + run_len):
            ry[idx[j]] = avg_rank
        i += run_len

    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += rx[i]
        my += ry[i]
    mx /= n
    my /= n

    num = 0.0
    denomx = 0.0
    denomy = 0.0
    for i in range(n):
        cx = rx[i] - mx
        cy = ry[i] - my
        num += cx * cy
        denomx += cx * cx
        denomy += cy * cy

    denom = sqrt(denomx * denomy)
    if denom <= 0.0:
        return nan_value
    rho = num / denom
    if rho > 1.0:
        rho = 1.0
    elif rho < -1.0:
        rho = -1.0
    return rho


cdef inline void _fenwick_update(double* tree, Py_ssize_t n, Py_ssize_t index, double delta) noexcept nogil:
    cdef Py_ssize_t i = index + 1
    while i <= n:
        tree[i] += delta
        i += i & (-i)


cdef inline double _fenwick_prefix_sum(double* tree, Py_ssize_t n, Py_ssize_t index) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0.0
    if index < 0:
        return 0.0
    i = index + 1
    if i > n:
        i = n
    while i > 0:
        s += tree[i]
        i -= i & (-i)
    return s


def distance_correlation_1d_fast_cy(double[:] x, double[:] y):
    """Exact O(n log n) univariate distance correlation, the same
    double-centering identity as huo_szekely_distance_correlation.py's
    pure-Python distance_correlation_1d_fast (see that module's docstring
    for the full derivation) -- ported here by explicit instruction ("no
    hotpaths in python whatsoever") since the Python version's per-element
    cross-term sweep (_cross_term_sum_ab) drove 4 pure-Python Fenwick-tree
    objects through an interpreted Python for-loop, an O(n log n) cost paid
    at Python-level overhead on every candidate. Sorting/ranking (for the
    row-sum reconstruction and the y-rank lookup) is done via numpy --
    vectorized C ops, not a Python loop; the genuinely loop-shaped work (the
    O(n) row-sum reductions and the O(n log n) Fenwick-tree cross-term
    sweep) runs inside nogil blocks below using flat C buffers instead of
    the Python _FenwickTree class.
    """
    cdef Py_ssize_t n = x.shape[0]
    if n < 2:
        return float("nan")

    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    # --- row_a[i] = sum_j |x_i-x_j|, row_b[i] = sum_j |y_i-y_j|, for ALL i,
    # in O(n log n): sort once, a running prefix-sum gives each row's sum in
    # O(1) (see huo_szekely_distance_correlation._row_sums_abs_diff).
    order_x = np.argsort(x_np, kind="mergesort")
    xv_sorted = np.ascontiguousarray(x_np[order_x])
    xprefix = np.cumsum(xv_sorted)
    xtotal = float(xprefix[n - 1])
    k = np.arange(n, dtype=np.float64)
    xrow_sorted = xv_sorted * (2.0 * k + 2.0 - n) + xtotal - 2.0 * xprefix
    row_a_np = np.empty(n, dtype=np.float64)
    row_a_np[order_x] = xrow_sorted

    order_y = np.argsort(y_np, kind="mergesort")
    yv_sorted = np.ascontiguousarray(y_np[order_y])
    yprefix = np.cumsum(yv_sorted)
    ytotal = float(yprefix[n - 1])
    yrow_sorted = yv_sorted * (2.0 * k + 2.0 - n) + ytotal - 2.0 * yprefix
    row_b_np = np.empty(n, dtype=np.float64)
    row_b_np[order_y] = yrow_sorted

    # --- Sigma_ab = sum_ij |x_i-x_j|*|y_i-y_j|, via a Fenwick-tree sweep
    # over points in x-sorted order (see huo_szekely_distance_correlation.
    # _cross_term_sum_ab's docstring for the derivation of g1/g2 below).
    order = np.argsort(x_np, kind="mergesort")
    xs_arr = np.ascontiguousarray(x_np[order])
    ys_arr = np.ascontiguousarray(y_np[order])
    _uniq, y_rank_np = np.unique(ys_arr, return_inverse=True)
    y_rank_np = np.ascontiguousarray(y_rank_np.astype(np.int64).ravel())
    cdef Py_ssize_t n_ranks = int(_uniq.shape[0])

    cdef double[::1] xs = xs_arr
    cdef double[::1] ys = ys_arr
    cdef int64_t[::1] y_rank = y_rank_np
    cdef double[::1] cnt = np.zeros(n_ranks + 1, dtype=np.float64)
    cdef double[::1] sumx = np.zeros(n_ranks + 1, dtype=np.float64)
    cdef double[::1] sumy = np.zeros(n_ranks + 1, dtype=np.float64)
    cdef double[::1] sumxy = np.zeros(n_ranks + 1, dtype=np.float64)
    cdef double[::1] row_a = row_a_np
    cdef double[::1] row_b = row_b_np

    cdef Py_ssize_t j, rj, last_rank, i
    cdef double xj, yj, c_le, sx_le, sy_le, sxy_le
    cdef double c_all, sx_all, sy_all, sxy_all
    cdef double c_gt, sx_gt, sy_gt, sxy_gt, g1, g2
    cdef double total_f, sigma_ab
    cdef double sum_row_a, sum_row_b, s_ab, s_aa, s_bb, abar_i, bbar_i
    cdef double sum_x, sum_y, sum_xx, sum_yy

    last_rank = n_ranks - 1
    total_f = 0.0
    with nogil:
        for j in range(n):
            if j > 0:
                xj = xs[j]
                yj = ys[j]
                rj = y_rank[j]

                c_le = _fenwick_prefix_sum(&cnt[0], n_ranks, rj)
                sx_le = _fenwick_prefix_sum(&sumx[0], n_ranks, rj)
                sy_le = _fenwick_prefix_sum(&sumy[0], n_ranks, rj)
                sxy_le = _fenwick_prefix_sum(&sumxy[0], n_ranks, rj)

                c_all = <double>j
                sx_all = _fenwick_prefix_sum(&sumx[0], n_ranks, last_rank)
                sy_all = _fenwick_prefix_sum(&sumy[0], n_ranks, last_rank)
                sxy_all = _fenwick_prefix_sum(&sumxy[0], n_ranks, last_rank)

                c_gt = c_all - c_le
                sx_gt = sx_all - sx_le
                sy_gt = sy_all - sy_le
                sxy_gt = sxy_all - sxy_le

                g1 = yj * c_le - sy_le + sy_gt - yj * c_gt
                g2 = yj * sx_le - sxy_le + sxy_gt - yj * sx_gt
                total_f += xj * g1 - g2

            rj = y_rank[j]
            _fenwick_update(&cnt[0], n_ranks, rj, 1.0)
            _fenwick_update(&sumx[0], n_ranks, rj, xs[j])
            _fenwick_update(&sumy[0], n_ranks, rj, ys[j])
            _fenwick_update(&sumxy[0], n_ranks, rj, xs[j] * ys[j])

        sigma_ab = 2.0 * total_f

        sum_row_a = 0.0
        sum_row_b = 0.0
        for i in range(n):
            sum_row_a += row_a[i]
            sum_row_b += row_b[i]

        s_ab = 0.0
        s_aa = 0.0
        s_bb = 0.0
        for i in range(n):
            abar_i = row_a[i] / n
            bbar_i = row_b[i] / n
            s_ab += abar_i * bbar_i
            s_aa += abar_i * abar_i
            s_bb += bbar_i * bbar_i

    cdef double[::1] x_mv = x_np
    cdef double[::1] y_mv = y_np
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    with nogil:
        for i in range(n):
            sum_x += x_mv[i]
            sum_y += y_mv[i]
            sum_xx += x_mv[i] * x_mv[i]
            sum_yy += y_mv[i] * y_mv[i]

    cdef double abar = sum_row_a / (<double>n * <double>n)
    cdef double bbar = sum_row_b / (<double>n * <double>n)
    cdef double sigma_aa = 2.0 * n * sum_xx - 2.0 * sum_x * sum_x
    cdef double sigma_bb = 2.0 * n * sum_yy - 2.0 * sum_y * sum_y

    cdef double dcov2 = (sigma_ab / (<double>n * <double>n)) - (2.0 * s_ab / n) + abar * bbar
    cdef double dvar_x = (sigma_aa / (<double>n * <double>n)) - (2.0 * s_aa / n) + abar * abar
    cdef double dvar_y = (sigma_bb / (<double>n * <double>n)) - (2.0 * s_bb / n) + bbar * bbar

    cdef double dvar_x_pos = dvar_x if dvar_x > 0.0 else 0.0
    cdef double dvar_y_pos = dvar_y if dvar_y > 0.0 else 0.0
    cdef double denom = sqrt(dvar_x_pos * dvar_y_pos)
    if denom <= 0.0 or denom != denom:
        return float("nan")
    cdef double dcov2_pos = dcov2 if dcov2 > 0.0 else 0.0
    cdef double dcorr2 = dcov2_pos / denom
    if dcorr2 > 1.0:
        dcorr2 = 1.0
    elif dcorr2 < 0.0:
        dcorr2 = 0.0
    return sqrt(dcorr2)


cdef double _dist_corr_naive_row_nogil(
    double* x, double* y, Py_ssize_t n,
    double* row_sum_a, double* row_sum_b,
    double nan_value,
) noexcept nogil:
    """Row-level naive O(w^2) distance correlation, same double-centering
    identity as _distance_correlation_1d (library_corrtrack_parallel.py) --
    direct nested loop, no sorting needed at all (unlike the O(w log w)
    'fast' variant below), matching that function's own documented O(w^2)
    complexity exactly."""
    cdef Py_ssize_t i, j
    cdef double a_ij, b_ij, a_c, b_c
    cdef double dcov2_sum, dvar_x_sum, dvar_y_sum
    cdef double sum_row_a, sum_row_b, abar, bbar, denom, dcorr2

    if n < 2:
        return nan_value

    sum_row_a = 0.0
    sum_row_b = 0.0
    for i in range(n):
        row_sum_a[i] = 0.0
        row_sum_b[i] = 0.0
        for j in range(n):
            row_sum_a[i] += fabs(x[i] - x[j])
            row_sum_b[i] += fabs(y[i] - y[j])
        sum_row_a += row_sum_a[i]
        sum_row_b += row_sum_b[i]

    abar = sum_row_a / (<double>n * <double>n)
    bbar = sum_row_b / (<double>n * <double>n)

    for i in range(n):
        row_sum_a[i] = row_sum_a[i] / n
        row_sum_b[i] = row_sum_b[i] / n

    dcov2_sum = 0.0
    dvar_x_sum = 0.0
    dvar_y_sum = 0.0
    for i in range(n):
        for j in range(n):
            a_ij = fabs(x[i] - x[j])
            b_ij = fabs(y[i] - y[j])
            a_c = a_ij - row_sum_a[i] - row_sum_a[j] + abar
            b_c = b_ij - row_sum_b[i] - row_sum_b[j] + bbar
            dcov2_sum += a_c * b_c
            dvar_x_sum += a_c * a_c
            dvar_y_sum += b_c * b_c

    dcov2_sum /= (<double>n * <double>n)
    dvar_x_sum /= (<double>n * <double>n)
    dvar_y_sum /= (<double>n * <double>n)

    if dvar_x_sum < 0.0:
        dvar_x_sum = 0.0
    if dvar_y_sum < 0.0:
        dvar_y_sum = 0.0
    denom = sqrt(dvar_x_sum * dvar_y_sum)
    if denom <= 0.0:
        return nan_value
    if dcov2_sum < 0.0:
        dcov2_sum = 0.0
    dcorr2 = dcov2_sum / denom
    if dcorr2 > 1.0:
        dcorr2 = 1.0
    elif dcorr2 < 0.0:
        dcorr2 = 0.0
    return sqrt(dcorr2)


cdef double _dist_corr_fast_row_nogil(
    double* x, double* y, Py_ssize_t n,
    double* key_buf, double* tmp_key,
    Py_ssize_t* idx, Py_ssize_t* tmp_idx,
    double* row_a, double* row_b,
    double* xs, double* ys,
    Py_ssize_t* y_rank,
    double* cnt, double* sumx, double* sumy, double* sumxy,
    double nan_value,
) noexcept nogil:
    """Row-level O(w log w) 'fast' distance correlation, identical
    algorithm/formula to distance_correlation_1d_fast_cy above, using
    caller-supplied scratch buffers and no numpy calls -- see
    _kendall_tau_row_nogil's docstring for why. Sorting (for the row-sum
    reconstruction and the y-rank lookup) uses _merge_sort_by_key_stable;
    the Fenwick-tree cross-term sweep reuses _fenwick_update/_fenwick_
    prefix_sum unchanged. cnt/sumx/sumy/sumxy must be allocated with at
    least (max window size + 1) entries -- only the first (n_ranks + 1) are
    read/written per call, since n_ranks is data-dependent per row."""
    cdef Py_ssize_t i, j, rj, last_rank, n_ranks
    cdef double xj, yj, c_le, sx_le, sy_le, sxy_le
    cdef double c_all, sx_all, sy_all, sxy_all
    cdef double c_gt, sx_gt, sy_gt, sxy_gt, g1, g2
    cdef double total_f, sigma_ab
    cdef double sum_row_a, sum_row_b, s_ab, s_aa, s_bb, abar_i, bbar_i
    cdef double sum_x, sum_y, sum_xx, sum_yy
    cdef double xtotal, ytotal, xprefix_val, yprefix_val
    cdef double abar, bbar, sigma_aa, sigma_bb
    cdef double dcov2, dvar_x, dvar_y, denom, dcorr2

    if n < 2:
        return nan_value

    # row_a[i] = sum_j |x_i - x_j|, via sort + closed-form prefix-sum
    # reconstruction (see huo_szekely_distance_correlation._row_sums_
    # abs_diff's derivation).
    for i in range(n):
        idx[i] = i
        key_buf[i] = x[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    xtotal = 0.0
    for i in range(n):
        xtotal += key_buf[i]
    xprefix_val = 0.0
    for i in range(n):
        xprefix_val += key_buf[i]
        row_a[idx[i]] = key_buf[i] * (2.0 * i + 2.0 - n) + xtotal - 2.0 * xprefix_val

    for i in range(n):
        idx[i] = i
        key_buf[i] = y[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    ytotal = 0.0
    for i in range(n):
        ytotal += key_buf[i]
    yprefix_val = 0.0
    for i in range(n):
        yprefix_val += key_buf[i]
        row_b[idx[i]] = key_buf[i] * (2.0 * i + 2.0 - n) + ytotal - 2.0 * yprefix_val

    # xs/ys: x and y in x-sorted order (order = argsort(x)).
    for i in range(n):
        idx[i] = i
        key_buf[i] = x[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    for i in range(n):
        xs[i] = key_buf[i]
        ys[i] = y[idx[i]]

    # y_rank: dense rank of ys (in x-sorted order) by value -- matches
    # np.unique(ys_arr, return_inverse=True)'s inverse array exactly.
    for i in range(n):
        idx[i] = i
        key_buf[i] = ys[i]
    _merge_sort_by_key_stable(key_buf, idx, tmp_key, tmp_idx, n)
    n_ranks = 0
    y_rank[idx[0]] = 0
    for i in range(1, n):
        if key_buf[i] != key_buf[i - 1]:
            n_ranks += 1
        y_rank[idx[i]] = n_ranks
    n_ranks += 1

    for i in range(n_ranks + 1):
        cnt[i] = 0.0
        sumx[i] = 0.0
        sumy[i] = 0.0
        sumxy[i] = 0.0

    last_rank = n_ranks - 1
    total_f = 0.0
    for j in range(n):
        if j > 0:
            xj = xs[j]
            yj = ys[j]
            rj = y_rank[j]

            c_le = _fenwick_prefix_sum(cnt, n_ranks, rj)
            sx_le = _fenwick_prefix_sum(sumx, n_ranks, rj)
            sy_le = _fenwick_prefix_sum(sumy, n_ranks, rj)
            sxy_le = _fenwick_prefix_sum(sumxy, n_ranks, rj)

            c_all = <double>j
            sx_all = _fenwick_prefix_sum(sumx, n_ranks, last_rank)
            sy_all = _fenwick_prefix_sum(sumy, n_ranks, last_rank)
            sxy_all = _fenwick_prefix_sum(sumxy, n_ranks, last_rank)

            c_gt = c_all - c_le
            sx_gt = sx_all - sx_le
            sy_gt = sy_all - sy_le
            sxy_gt = sxy_all - sxy_le

            g1 = yj * c_le - sy_le + sy_gt - yj * c_gt
            g2 = yj * sx_le - sxy_le + sxy_gt - yj * sx_gt
            total_f += xj * g1 - g2

        rj = y_rank[j]
        _fenwick_update(cnt, n_ranks, rj, 1.0)
        _fenwick_update(sumx, n_ranks, rj, xs[j])
        _fenwick_update(sumy, n_ranks, rj, ys[j])
        _fenwick_update(sumxy, n_ranks, rj, xs[j] * ys[j])

    sigma_ab = 2.0 * total_f

    sum_row_a = 0.0
    sum_row_b = 0.0
    for i in range(n):
        sum_row_a += row_a[i]
        sum_row_b += row_b[i]

    s_ab = 0.0
    s_aa = 0.0
    s_bb = 0.0
    for i in range(n):
        abar_i = row_a[i] / n
        bbar_i = row_b[i] / n
        s_ab += abar_i * bbar_i
        s_aa += abar_i * abar_i
        s_bb += bbar_i * bbar_i

    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xx += x[i] * x[i]
        sum_yy += y[i] * y[i]

    abar = sum_row_a / (<double>n * <double>n)
    bbar = sum_row_b / (<double>n * <double>n)
    sigma_aa = 2.0 * n * sum_xx - 2.0 * sum_x * sum_x
    sigma_bb = 2.0 * n * sum_yy - 2.0 * sum_y * sum_y

    dcov2 = (sigma_ab / (<double>n * <double>n)) - (2.0 * s_ab / n) + abar * bbar
    dvar_x = (sigma_aa / (<double>n * <double>n)) - (2.0 * s_aa / n) + abar * abar
    dvar_y = (sigma_bb / (<double>n * <double>n)) - (2.0 * s_bb / n) + bbar * bbar

    if dvar_x < 0.0:
        dvar_x = 0.0
    if dvar_y < 0.0:
        dvar_y = 0.0
    denom = sqrt(dvar_x * dvar_y)
    if denom <= 0.0 or denom != denom:
        return nan_value
    if dcov2 < 0.0:
        dcov2 = 0.0
    dcorr2 = dcov2 / denom
    if dcorr2 > 1.0:
        dcorr2 = 1.0
    elif dcorr2 < 0.0:
        dcorr2 = 0.0
    return sqrt(dcorr2)


def validate_corr_rows_nonlinear(double[:, :] data,
                                  long[:, :] rows,
                                  long base_index,
                                  int metric_id,
                                  double corr_threshold,
                                  bint neg_corr,
                                  double std_thresh=1e-3,
                                  double kurt_thresh=5.0):
    """Bulk validation for non-Pearson metrics -- ONE Cython call handles
    the whole candidate-rows array in a single nogil loop, matching
    validate_corr_rows' (Pearson) single-call design instead of a per-row
    Python loop with a Cython call inside each iteration (the previous
    state of _validate_numeric_rows_nonlinear). metric_id: 0=spearman,
    1=kendall, 2=dist_corr (naive, O(w^2)), 3=dist_corr (fast, O(w log w)).

    The constant/spike pre-checks below are copied verbatim (same raw-
    power-sum formulas) from validate_corr_rows -- confirmed algebraically
    identical to _is_near_constant_stats/_is_structurally_spiked_stats
    (library_corrtrack_parallel.py), the functions the old per-row Python
    path called, before trusting the copy (see docs/implementation_log.md's
    2026-07-31 entry). The "dist" returned uses the metric-distance
    definition sqrt(2*(1-|corr|)) (_metric_distance_from_corr), NOT
    Pearson's raw Euclidean sqrt(sum_xx+sum_yy-2*sum_xy) -- these are
    different quantities on purpose (min_dist tracking for non-Pearson
    metrics has always used the correlation-derived distance, verified by
    reading _validate_numeric_rows_nonlinear's own prior implementation
    directly rather than assumed from Pearson's convention).

    Returns (accepted, corrs, dists, valid_mask, constants, spiked).
    valid_mask[i]==1 iff row i passed the initial bounds check (matching
    the old per-row loop's own "tested += 1" placement exactly -- unlike
    Pearson's own convention, which counts bounds-failed rows as tested
    too via a shared "constant" flag; the two callers had different
    pre-existing counting semantics, confirmed by reading both call sites
    before writing this, not assumed to match)."""
    cdef Py_ssize_t n_items = rows.shape[0]
    cdef Py_ssize_t n_series = data.shape[0]
    cdef Py_ssize_t n_cols = data.shape[1]
    if rows.shape[1] < 5:
        raise ValueError("rows must have at least five columns")

    cdef np.ndarray[np.uint8_t, ndim=1] accepted = np.zeros(n_items, dtype=np.uint8)
    cdef np.ndarray[np.float64_t, ndim=1] corrs = np.empty(n_items, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] dists = np.empty(n_items, dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=1] valid = np.zeros(n_items, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] constants = np.zeros(n_items, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] spiked = np.zeros(n_items, dtype=np.uint8)

    cdef uint8_t[:] accepted_view = accepted
    cdef double[:] corr_view = corrs
    cdef double[:] dist_view = dists
    cdef uint8_t[:] valid_view = valid
    cdef uint8_t[:] const_view = constants
    cdef uint8_t[:] spike_view = spiked

    cdef Py_ssize_t i, j
    cdef long s1, s2, t1, t2, w, start1, start2
    cdef Py_ssize_t max_w
    cdef double sx, sy, sum_xy, sum_xx, sum_yy
    cdef double sum_xxx, sum_yyy, sum_xxxx, sum_yyyy
    cdef double xi, yi
    cdef double mean_x, mean_y, var_x, var_y
    cdef double mu4_x, mu4_y, varx, vary, kurt_x, kurt_y
    cdef double nan_value = np.nan
    cdef double inf_value = np.inf
    cdef bint is_const, is_spiked, is_corr
    cdef double corr, dist_val

    if n_items == 0:
        return accepted, corrs, dists, valid, constants, spiked

    max_w = 1
    for i in range(n_items):
        w = rows[i, 4]
        if w > max_w:
            max_w = w

    x_buf_arr = np.empty(max_w, dtype=np.float64)
    y_buf_arr = np.empty(max_w, dtype=np.float64)
    key_buf_arr = np.empty(max_w, dtype=np.float64)
    tmp_key_arr = np.empty(max_w, dtype=np.float64)
    idx_arr = np.empty(max_w, dtype=np.intp)
    tmp_idx_arr = np.empty(max_w, dtype=np.intp)
    xs_arr = np.empty(max_w, dtype=np.float64)
    ys_arr = np.empty(max_w, dtype=np.float64)
    ys_work_arr = np.empty(max_w, dtype=np.float64)
    y_sorted_alone_arr = np.empty(max_w, dtype=np.float64)
    rx_arr = np.empty(max_w, dtype=np.float64)
    ry_arr = np.empty(max_w, dtype=np.float64)
    row_a_arr = np.empty(max_w, dtype=np.float64)
    row_b_arr = np.empty(max_w, dtype=np.float64)
    y_rank_arr = np.empty(max_w, dtype=np.intp)
    cnt_arr = np.empty(max_w + 1, dtype=np.float64)
    sumx_arr = np.empty(max_w + 1, dtype=np.float64)
    sumy_arr = np.empty(max_w + 1, dtype=np.float64)
    sumxy_arr = np.empty(max_w + 1, dtype=np.float64)

    cdef double[::1] x_buf = x_buf_arr
    cdef double[::1] y_buf = y_buf_arr
    cdef double[::1] key_buf = key_buf_arr
    cdef double[::1] tmp_key = tmp_key_arr
    cdef Py_ssize_t[::1] idx = idx_arr
    cdef Py_ssize_t[::1] tmp_idx = tmp_idx_arr
    cdef double[::1] xs = xs_arr
    cdef double[::1] ys = ys_arr
    cdef double[::1] ys_work = ys_work_arr
    cdef double[::1] y_sorted_alone = y_sorted_alone_arr
    cdef double[::1] rx = rx_arr
    cdef double[::1] ry = ry_arr
    cdef double[::1] row_a = row_a_arr
    cdef double[::1] row_b = row_b_arr
    cdef Py_ssize_t[::1] y_rank = y_rank_arr
    cdef double[::1] cnt = cnt_arr
    cdef double[::1] sumx = sumx_arr
    cdef double[::1] sumy = sumy_arr
    cdef double[::1] sumxy = sumxy_arr

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
                continue

            valid_view[i] = <uint8_t>1

            sx = sy = 0.0
            sum_xy = sum_xx = sum_yy = 0.0
            sum_xxx = sum_yyy = 0.0
            sum_xxxx = sum_yyyy = 0.0
            for j in range(w):
                xi = data[s1, start1 + j]
                yi = data[s2, start2 + j]
                x_buf[j] = xi
                y_buf[j] = yi
                sum_xy += xi * yi
                sx += xi
                sum_xx += xi * xi
                sum_xxx += xi * xi * xi
                sum_xxxx += xi * xi * xi * xi
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

            if metric_id == 0:
                corr = _spearman_rho_row_nogil(
                    &x_buf[0], &y_buf[0], w, &key_buf[0], &tmp_key[0], &idx[0], &tmp_idx[0],
                    &rx[0], &ry[0], nan_value,
                )
            elif metric_id == 1:
                corr = _kendall_tau_row_nogil(
                    &x_buf[0], &y_buf[0], w, &key_buf[0], &tmp_key[0], &idx[0], &tmp_idx[0],
                    &xs[0], &ys[0], &ys_work[0], &y_sorted_alone[0], nan_value,
                )
            elif metric_id == 2:
                corr = _dist_corr_naive_row_nogil(&x_buf[0], &y_buf[0], w, &row_a[0], &row_b[0], nan_value)
            else:
                corr = _dist_corr_fast_row_nogil(
                    &x_buf[0], &y_buf[0], w, &key_buf[0], &tmp_key[0], &idx[0], &tmp_idx[0],
                    &row_a[0], &row_b[0], &xs[0], &ys[0], &y_rank[0],
                    &cnt[0], &sumx[0], &sumy[0], &sumxy[0], nan_value,
                )

            corr_view[i] = corr

            if corr == corr:
                dist_val = 2.0 * (1.0 - fabs(corr))
                if dist_val < 0.0:
                    dist_val = 0.0
                dist_view[i] = sqrt(dist_val)
            else:
                dist_view[i] = inf_value

            is_corr = False
            if corr == corr:
                if neg_corr:
                    is_corr = fabs(corr) >= corr_threshold
                else:
                    is_corr = corr >= corr_threshold
            if is_corr:
                accepted_view[i] = <uint8_t>1

    return accepted, corrs, dists, valid, constants, spiked
