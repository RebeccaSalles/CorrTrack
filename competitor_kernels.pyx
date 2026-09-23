# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython hot loops for the competitor candidate indexes (2026-09-17, implementation plan
section 5 item 3 / comparison plan section 6.3): the competitors' candidate generation was
pure Python while CorrTrack's is Cython, so wall-clock comparisons measured the tier gap. Each
kernel replaces one Python loop behind the existing index interface; the Python paths stay as
the reference and the parity tests assert identical pair sets.

Common conventions
- entries live in flat arrays (vectors (N, D) float64, cells (N, H) int64, alive uint8, sid/time int64);
- a *grid* is the alive entries sorted by an exact mixed-radix cell key (coordinates offset by
  KEY_OFF and packed base KEY_BASE), so a cell's posting list is a contiguous range found by
  binary search: no hash tables, no Python objects inside the loop;
- a query visits its own cell and the 3^H neighbours (StatStream; CorrJoin uses 3^3), or the
  same cell in each of several grids (ParCorr/CSZ); dedupe across probes uses a per-query
  stamp array instead of a set;
- the canonical unordered-pair rule of the Python paths is kept verbatim: skip when
  e_time > q_time, or e_time == q_time and e_sid < q_sid; skip the query's own series/time.
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, int32_t, uint8_t
from libc.stdlib cimport malloc, free

np.import_array()

DEF KEY_BASE = 4096          # cells per axis bound (coordinates are offset by KEY_OFF and must lie in [0, KEY_BASE))
DEF KEY_OFF = 2048


cdef inline int64_t _cell_key(const int64_t* c, int H) nogil:
    cdef int64_t key = 0
    cdef int i
    for i in range(H):
        key = key * KEY_BASE + (c[i] + KEY_OFF)
    return key


cdef inline Py_ssize_t _lower_bound(const int64_t* keys, Py_ssize_t n, int64_t x) nogil:
    cdef Py_ssize_t lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def cell_keys(np.ndarray[np.int64_t, ndim=2, mode="c"] cells not None):
    """Exact packed key per row of an (N, H) integer cell array (H <= 5 keeps the key inside int64)."""
    cdef Py_ssize_t N = cells.shape[0]
    cdef int H = <int>cells.shape[1]
    if H > 5:
        raise ValueError("cell key packing supports at most 5 grid dimensions")
    cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(N, dtype=np.int64)
    cdef Py_ssize_t i
    cdef int64_t* cp = <int64_t*>cells.data
    for i in range(N):
        out[i] = _cell_key(cp + i * H, H)
    return out


def statstream_probe(np.ndarray[np.float64_t, ndim=2, mode="c"] vectors not None,
                     np.ndarray[np.int64_t, ndim=2, mode="c"] cells not None,
                     np.ndarray[np.uint8_t, ndim=1, mode="c"] alive not None,
                     np.ndarray[np.int64_t, ndim=1, mode="c"] sid not None,
                     np.ndarray[np.int64_t, ndim=1, mode="c"] time_ not None,
                     np.ndarray[np.int64_t, ndim=1, mode="c"] order not None,
                     np.ndarray[np.int64_t, ndim=1, mode="c"] sorted_keys not None,
                     np.ndarray[np.int64_t, ndim=1, mode="c"] recent not None,
                     np.ndarray[np.int64_t, ndim=2, mode="c"] offsets not None,
                     double eps, bint signed, bint apply_filter):
    """StatStream grid probe (VLDB 2002 section 3.6, Lemma 2/3, Theorem 2).

    `order` lists the alive entry ids sorted by cell key, `sorted_keys` their keys; `offsets` is
    the (3^H, H) neighbour table. For each recent query: probe the 3^H cells around its own cell
    (sign +1) and, when `signed`, around the cell of -X_hat (sign -1); each (entry, sign) is tested
    once; survivors pass the exact n-approximate filter ||X_hat - sign * Y_hat||^2 <= eps^2 when
    `apply_filter`. Returns (a, b, touched, dist_checks) with a = query ids, b = entry ids."""
    cdef Py_ssize_t N = vectors.shape[0]
    cdef int D = <int>vectors.shape[1]
    cdef int H = <int>cells.shape[1]
    cdef Py_ssize_t n_sorted = order.shape[0]
    cdef Py_ssize_t R = recent.shape[0]
    cdef Py_ssize_t n_off = offsets.shape[0]
    cdef double eps2 = eps * eps
    # first pass: upper bound on the number of candidates touched (sum of probed range sizes)
    cdef Py_ssize_t cap = 0
    cdef Py_ssize_t qi, oi, lo, hi, k, e_idx
    cdef int64_t q, e, key, sgn_i
    cdef int64_t c[5]
    cdef int64_t nc[5]
    cdef int d, s
    cdef double sgn, dist, diff
    cdef int64_t q_sid, q_time
    cdef int64_t* keys_p = <int64_t*>sorted_keys.data
    cdef int64_t* order_p = <int64_t*>order.data
    cdef int64_t* cells_p = <int64_t*>cells.data
    cdef int64_t* off_p = <int64_t*>offsets.data
    cdef double* vec_p = <double*>vectors.data
    cdef uint8_t* alive_p = <uint8_t*>alive.data
    cdef int64_t* sid_p = <int64_t*>sid.data
    cdef int64_t* time_p = <int64_t*>time_.data
    cdef int64_t* recent_p = <int64_t*>recent.data
    cdef int n_signs = 2 if signed else 1
    with nogil:
        for qi in range(R):
            q = recent_p[qi]
            if not alive_p[q]:
                continue
            for s in range(n_signs):
                for d in range(H):
                    if s == 0:
                        c[d] = cells_p[q * H + d]
                    else:
                        c[d] = <int64_t>_floor_div(-vec_p[q * D + d], eps)
                for oi in range(n_off):
                    for d in range(H):
                        nc[d] = c[d] + off_p[oi * H + d]
                    key = _cell_key(nc, H)
                    lo = _lower_bound(keys_p, n_sorted, key)
                    hi = _lower_bound(keys_p, n_sorted, key + 1)
                    cap += hi - lo
    cdef np.ndarray[np.int64_t, ndim=1] out_a = np.empty(cap, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] out_b = np.empty(cap, dtype=np.int64)
    cdef int64_t* stamp = <int64_t*>malloc(N * sizeof(int64_t))
    if stamp == NULL:
        raise MemoryError()
    cdef Py_ssize_t n_out = 0
    cdef Py_ssize_t touched = 0, dist_checks = 0
    cdef int64_t mark
    cdef Py_ssize_t i
    for i in range(N):
        stamp[i] = 0
    with nogil:
        for qi in range(R):
            q = recent_p[qi]
            if not alive_p[q]:
                continue
            q_sid = sid_p[q]; q_time = time_p[q]
            for s in range(n_signs):
                sgn = 1.0 if s == 0 else -1.0
                mark = (qi + 1) * 2 + s
                for d in range(H):
                    if s == 0:
                        c[d] = cells_p[q * H + d]
                    else:
                        c[d] = <int64_t>_floor_div(-vec_p[q * D + d], eps)
                for oi in range(n_off):
                    for d in range(H):
                        nc[d] = c[d] + off_p[oi * H + d]
                    key = _cell_key(nc, H)
                    lo = _lower_bound(keys_p, n_sorted, key)
                    hi = _lower_bound(keys_p, n_sorted, key + 1)
                    for k in range(lo, hi):
                        e = order_p[k]
                        if e == q or not alive_p[e] or stamp[e] == mark:
                            continue
                        stamp[e] = mark
                        touched += 1
                        if sid_p[e] == q_sid and time_p[e] == q_time:
                            continue
                        if time_p[e] > q_time or (time_p[e] == q_time and sid_p[e] < q_sid):
                            continue
                        if apply_filter:
                            dist_checks += 1
                            dist = 0.0
                            for d in range(D):
                                diff = vec_p[q * D + d] - sgn * vec_p[e * D + d]
                                dist += diff * diff
                            if dist > eps2:
                                continue
                        out_a[n_out] = q; out_b[n_out] = e
                        n_out += 1
    free(stamp)
    return out_a[:n_out], out_b[:n_out], touched, dist_checks


cdef inline double _floor_div(double x, double eps) nogil:
    cdef double v = x / eps
    cdef double f = <double>(<int64_t>v)
    if v < 0 and f != v:
        f -= 1.0
    return f


def parcorr_probe(np.ndarray[np.float64_t, ndim=2, mode="c"] vectors not None,
                  np.ndarray[np.uint8_t, ndim=1, mode="c"] alive not None,
                  np.ndarray[np.int64_t, ndim=1, mode="c"] sid not None,
                  np.ndarray[np.int64_t, ndim=1, mode="c"] time_ not None,
                  np.ndarray[np.int64_t, ndim=2, mode="c"] orders not None,
                  np.ndarray[np.int64_t, ndim=2, mode="c"] sorted_keys not None,
                  np.ndarray[np.int64_t, ndim=1, mode="c"] recent not None,
                  np.ndarray[np.int64_t, ndim=3, mode="c"] cells_recent not None,
                  np.ndarray[np.int64_t, ndim=2, mode="c"] offsets not None,
                  int k, double cell_size, bint neighbor_probe, int required_hits):
    """ParCorr / Cole-Shasha-Zhao grid vote (DMKD 2018 section 4.3; KDD 2005 section 4).

    `orders[g]` / `sorted_keys[g]`: alive entries sorted by their packed cell key in grid g
    (n_grids grids of k sketch coordinates each). `cells_recent[qi, g]` is the query's cell in
    grid g. Per query, per grid: the query's cell (ParCorr) or its 3^k neighbours with the exact
    cell_size-ball test on the k-subvector (CSZ, `neighbor_probe`); each entry met gets one vote
    per grid; entries with >= required_hits votes become candidates under the canonical pair rule.
    Returns (a, b, touched, dist_checks)."""
    cdef Py_ssize_t N = vectors.shape[0]
    cdef int D = <int>vectors.shape[1]
    cdef int n_grids = <int>orders.shape[0]
    cdef Py_ssize_t n_sorted = orders.shape[1]
    cdef Py_ssize_t R = recent.shape[0]
    cdef Py_ssize_t n_off = offsets.shape[0]
    cdef double cs2 = cell_size * cell_size
    cdef int64_t* order_p = <int64_t*>orders.data
    cdef int64_t* keys_p = <int64_t*>sorted_keys.data
    cdef int64_t* cr_p = <int64_t*>cells_recent.data
    cdef int64_t* off_p = <int64_t*>offsets.data
    cdef double* vec_p = <double*>vectors.data
    cdef uint8_t* alive_p = <uint8_t*>alive.data
    cdef int64_t* sid_p = <int64_t*>sid.data
    cdef int64_t* time_p = <int64_t*>time_.data
    cdef int64_t* recent_p = <int64_t*>recent.data
    cdef Py_ssize_t qi, oi, lo, hi, kk, cap = 0, n_out = 0, touched = 0, dist_checks = 0, n_cand, ci
    cdef int g, d
    cdef int64_t q, e, key, q_sid, q_time, mark
    cdef int64_t nc[5]
    cdef double dist, diff
    if k > 5:
        raise ValueError("parcorr_probe supports k <= 5")
    with nogil:
        for qi in range(R):
            q = recent_p[qi]
            if not alive_p[q]:
                continue
            for g in range(n_grids):
                for oi in range(n_off):
                    for d in range(k):
                        nc[d] = cr_p[(qi * n_grids + g) * k + d] + off_p[oi * k + d]
                    key = _cell_key(nc, k)
                    lo = _lower_bound(keys_p + g * n_sorted, n_sorted, key)
                    hi = _lower_bound(keys_p + g * n_sorted, n_sorted, key + 1)
                    cap += hi - lo
    cdef np.ndarray[np.int64_t, ndim=1] out_a = np.empty(cap, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] out_b = np.empty(cap, dtype=np.int64)
    cdef int64_t* stamp = <int64_t*>malloc(N * sizeof(int64_t))
    cdef int32_t* votes = <int32_t*>malloc(N * sizeof(int32_t))
    cdef int64_t* gstamp = <int64_t*>malloc(N * sizeof(int64_t))     # (query, grid) visit mark: one vote per grid
    cdef int64_t* cand = <int64_t*>malloc((cap + 1) * sizeof(int64_t))
    if stamp == NULL or votes == NULL or gstamp == NULL or cand == NULL:
        free(stamp); free(votes); free(gstamp); free(cand)
        raise MemoryError()
    cdef Py_ssize_t i
    for i in range(N):
        stamp[i] = 0; gstamp[i] = 0; votes[i] = 0
    with nogil:
        for qi in range(R):
            q = recent_p[qi]
            if not alive_p[q]:
                continue
            q_sid = sid_p[q]; q_time = time_p[q]
            n_cand = 0
            for g in range(n_grids):
                mark = (qi + 1) * n_grids + g + 1
                for oi in range(n_off):
                    for d in range(k):
                        nc[d] = cr_p[(qi * n_grids + g) * k + d] + off_p[oi * k + d]
                    key = _cell_key(nc, k)
                    lo = _lower_bound(keys_p + g * n_sorted, n_sorted, key)
                    hi = _lower_bound(keys_p + g * n_sorted, n_sorted, key + 1)
                    for kk in range(lo, hi):
                        e = order_p[g * n_sorted + kk]
                        if e == q or not alive_p[e] or gstamp[e] == mark:
                            continue
                        gstamp[e] = mark
                        touched += 1
                        if neighbor_probe:
                            dist_checks += 1
                            dist = 0.0
                            for d in range(k):
                                diff = vec_p[q * D + g * k + d] - vec_p[e * D + g * k + d]
                                dist += diff * diff
                            if dist > cs2:
                                continue
                        if stamp[e] != qi + 1:
                            stamp[e] = qi + 1; votes[e] = 0
                            cand[n_cand] = e; n_cand += 1
                        votes[e] += 1
            for ci in range(n_cand):
                e = cand[ci]
                if votes[e] < required_hits:
                    continue
                if sid_p[e] == q_sid and time_p[e] == q_time:
                    continue
                if time_p[e] > q_time or (time_p[e] == q_time and sid_p[e] < q_sid):
                    continue
                out_a[n_out] = q; out_b[n_out] = e
                n_out += 1
    free(stamp); free(votes); free(gstamp); free(cand)
    return out_a[:n_out], out_b[:n_out], touched, dist_checks


def corrjoin_double_filter(np.ndarray[np.float64_t, ndim=2, mode="c"] proj not None,
                           np.ndarray[np.float64_t, ndim=2, mode="c"] ke_block not None,
                           np.ndarray[np.int64_t, ndim=2, mode="c"] cells not None,
                           np.ndarray[np.int64_t, ndim=1, mode="c"] order not None,
                           np.ndarray[np.int64_t, ndim=1, mode="c"] sorted_keys not None,
                           np.ndarray[np.int64_t, ndim=2, mode="c"] offsets not None,
                           double eps1, double eps2,
                           np.ndarray[np.int64_t, ndim=1, mode="c"] queries=None):
    """CorrJoin's two filters (PACMMOD 2023 Alg. 1 / the authors' BucketingFilter): over the m
    alive series of the current window, grid of side eps1 on the kb-dim SVD projections `proj`,
    3^kb neighbourhood, exact eps1-ball on the projection, then the eps2 Euclidean test on the
    PAA_ke block `ke_block`. `order`/`sorted_keys`: rows 0..m-1 sorted by packed cell key.

    `queries` (2026-09-23, lagged extension): row indices to probe FROM. None (the synchronous
    case) probes every row and returns each unordered pair once via the `j <= i` rule, exactly as
    before. When given, only those rows are probed, every met row is returned (`j != i`) and the
    caller applies the canonical later-first rule across times -- the same query semantics the
    ParCorr / StatStream indexes use when the index holds several windows.
    Returns (a, b, touched, n_after_bucketing) with local row indices."""
    cdef Py_ssize_t m = proj.shape[0]
    cdef int kb = <int>proj.shape[1]
    cdef int ke = <int>ke_block.shape[1]
    cdef Py_ssize_t n_off = offsets.shape[0]
    cdef double e1sq = eps1 * eps1, e2sq = eps2 * eps2
    cdef double* pp = <double*>proj.data
    cdef double* kp = <double*>ke_block.data
    cdef int64_t* cells_p = <int64_t*>cells.data
    cdef int64_t* order_p = <int64_t*>order.data
    cdef int64_t* keys_p = <int64_t*>sorted_keys.data
    cdef int64_t* off_p = <int64_t*>offsets.data
    cdef Py_ssize_t i, qi, oi, lo, hi, kk, cap = 0, n_out = 0, touched = 0, n_bucket = 0
    cdef int64_t j, key
    cdef int64_t nc[5]
    cdef int d
    cdef double dist, diff
    if kb > 5:
        raise ValueError("corrjoin_double_filter supports kb <= 5")
    cdef bint all_rows = queries is None
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] q_arr = (
        np.arange(m, dtype=np.int64) if all_rows else queries)
    cdef int64_t* q_p = <int64_t*>q_arr.data
    cdef Py_ssize_t n_q = q_arr.shape[0]
    with nogil:
        for qi in range(n_q):
            i = q_p[qi]
            for oi in range(n_off):
                for d in range(kb):
                    nc[d] = cells_p[i * kb + d] + off_p[oi * kb + d]
                key = _cell_key(nc, kb)
                lo = _lower_bound(keys_p, m, key)
                hi = _lower_bound(keys_p, m, key + 1)
                cap += hi - lo
    cdef np.ndarray[np.int64_t, ndim=1] out_a = np.empty(cap, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] out_b = np.empty(cap, dtype=np.int64)
    with nogil:
        for qi in range(n_q):
            i = q_p[qi]
            for oi in range(n_off):
                for d in range(kb):
                    nc[d] = cells_p[i * kb + d] + off_p[oi * kb + d]
                key = _cell_key(nc, kb)
                lo = _lower_bound(keys_p, m, key)
                hi = _lower_bound(keys_p, m, key + 1)
                for kk in range(lo, hi):
                    j = order_p[kk]
                    if (j <= i) if all_rows else (j == i):
                        continue
                    touched += 1
                    dist = 0.0
                    for d in range(kb):
                        diff = pp[i * kb + d] - pp[j * kb + d]
                        dist += diff * diff
                    if dist > e1sq:
                        continue
                    n_bucket += 1
                    dist = 0.0
                    for d in range(ke):
                        diff = kp[i * ke + d] - kp[j * ke + d]
                        dist += diff * diff
                        if dist > e2sq:
                            break
                    if dist > e2sq:
                        continue
                    out_a[n_out] = i; out_b[n_out] = j
                    n_out += 1
    return out_a[:n_out], out_b[:n_out], touched, n_bucket


def all_pairs_gates(np.ndarray[np.float64_t, ndim=2, mode="c"] vectors not None,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] alive not None,
                    np.ndarray[np.int64_t, ndim=1, mode="c"] sid not None,
                    np.ndarray[np.int64_t, ndim=1, mode="c"] time_ not None,
                    np.ndarray[np.int64_t, ndim=1, mode="c"] recent not None,
                    double gamma, bint signed_abs, bint apply_dot, bint apply_hamming, Py_ssize_t hamming_max_bits):
    """Ablation backend (2026-09-18): NO index. Every recent query is paired with every alive entry
    under the canonical pair rule, then CorrTrack's two candidate gates are applied exactly as
    SignLSHBandIndex applies them: the sign-Hamming gate (bits where sign(vq) != sign(ve), or the
    complement under signed_abs; pass if the best of the two <= hamming_max_bits) and the dot gate
    (sketch dot product >= gamma, |.| under signed_abs). With both gates off it is "sketch + no
    candidate search". Returns (a, b, touched, hamming_checks, dot_checks)."""
    cdef Py_ssize_t N = vectors.shape[0]
    cdef int D = <int>vectors.shape[1]
    cdef Py_ssize_t R = recent.shape[0]
    cdef double* vec_p = <double*>vectors.data
    cdef uint8_t* alive_p = <uint8_t*>alive.data
    cdef int64_t* sid_p = <int64_t*>sid.data
    cdef int64_t* time_p = <int64_t*>time_.data
    cdef int64_t* recent_p = <int64_t*>recent.data
    cdef Py_ssize_t cap = 0, qi, e, n_out = 0, touched = 0, hchecks = 0, dchecks = 0
    cdef int64_t q, q_sid, q_time
    cdef int d, hpos, hneg, hbest
    cdef double score
    cdef bint sq, se
    for qi in range(R):
        if alive_p[recent_p[qi]]:
            cap += N
    cdef np.ndarray[np.int64_t, ndim=1] out_a = np.empty(cap, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] out_b = np.empty(cap, dtype=np.int64)
    with nogil:
        for qi in range(R):
            q = recent_p[qi]
            if not alive_p[q]:
                continue
            q_sid = sid_p[q]; q_time = time_p[q]
            for e in range(N):
                if e == q or not alive_p[e]:
                    continue
                if sid_p[e] == q_sid and time_p[e] == q_time:
                    continue
                if time_p[e] > q_time or (time_p[e] == q_time and sid_p[e] < q_sid):
                    continue
                touched += 1
                if apply_hamming:
                    hchecks += 1
                    hpos = 0; hneg = 0
                    for d in range(D):
                        sq = vec_p[q * D + d] >= 0.0
                        se = vec_p[e * D + d] >= 0.0
                        if sq != se:
                            hpos += 1
                        else:
                            hneg += 1
                    hbest = hpos
                    if signed_abs and hneg < hbest:
                        hbest = hneg
                    if hbest > hamming_max_bits:
                        continue
                if apply_dot:
                    dchecks += 1
                    score = 0.0
                    for d in range(D):
                        score += vec_p[q * D + d] * vec_p[e * D + d]
                    if signed_abs:
                        if score < 0.0:
                            score = -score
                    if score < gamma:
                        continue
                out_a[n_out] = q; out_b[n_out] = e
                n_out += 1
    return out_a[:n_out], out_b[:n_out], touched, hchecks, dchecks
