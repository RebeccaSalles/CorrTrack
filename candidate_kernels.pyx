# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, realloc, free
from libc.stdint cimport int64_t, uint64_t, uint8_t

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


cdef class BalancedIndex:
    cdef object _values_arr
    cdef object _left_arr
    cdef object _right_arr
    cdef object _prio_arr
    cdef object _window_idx_arr
    cdef object _active_arr
    cdef object _has_vector_arr
    cdef object _vectors_arr
    cdef Py_ssize_t _size
    cdef Py_ssize_t _active_count
    cdef Py_ssize_t _capacity
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
        self._active_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._has_vector_arr = np.zeros(self._capacity, dtype=np.uint8)
        if self._n_vectors > 0:
            self._vectors_arr = np.zeros((self._capacity, self._n_vectors), dtype=np.float64)
        else:
            self._vectors_arr = np.empty((0, 0), dtype=np.float64)

    cdef void _ensure_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        cdef np.ndarray[np.float64_t, ndim=1] new_values
        cdef np.ndarray[np.int64_t, ndim=1] new_left
        cdef np.ndarray[np.int64_t, ndim=1] new_right
        cdef np.ndarray[np.int64_t, ndim=1] new_prio
        cdef np.ndarray[np.int64_t, ndim=1] new_window_idx
        cdef np.ndarray[np.uint8_t, ndim=1] new_active
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
        new_active = np.zeros(new_cap, dtype=np.uint8)
        new_has_vector = np.zeros(new_cap, dtype=np.uint8)

        if self._size > 0:
            new_values[:self._size] = self._values_arr[:self._size]
            new_left[:self._size] = self._left_arr[:self._size]
            new_right[:self._size] = self._right_arr[:self._size]
            new_prio[:self._size] = self._prio_arr[:self._size]
            new_window_idx[:self._size] = self._window_idx_arr[:self._size]
            new_active[:self._size] = self._active_arr[:self._size]
            new_has_vector[:self._size] = self._has_vector_arr[:self._size]

        self._values_arr = new_values
        self._left_arr = new_left
        self._right_arr = new_right
        self._prio_arr = new_prio
        self._window_idx_arr = new_window_idx
        self._active_arr = new_active
        self._has_vector_arr = new_has_vector

        if self._n_vectors > 0:
            new_vectors = np.zeros((new_cap, self._n_vectors), dtype=np.float64)
            if self._size > 0:
                new_vectors[:self._size, :] = self._vectors_arr[:self._size, :]
            self._vectors_arr = new_vectors

        self._capacity = new_cap

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

    def insert(self, double value, long window_idx, vector=None):
        cdef int64_t node
        cdef np.ndarray[np.float64_t, ndim=1] vec
        cdef Py_ssize_t d
        cdef double[:] values
        cdef int64_t[:] left
        cdef int64_t[:] right
        cdef int64_t[:] prio
        cdef int64_t[:] window_ids
        cdef uint8_t[:] active
        cdef uint8_t[:] has_vector
        cdef double[:, :] vectors

        self._ensure_capacity(self._size + 1)
        node = <int64_t>self._size
        values = self._values_arr
        left = self._left_arr
        right = self._right_arr
        prio = self._prio_arr
        window_ids = self._window_idx_arr
        active = self._active_arr
        has_vector = self._has_vector_arr

        values[node] = value
        window_ids[node] = <int64_t>window_idx
        left[node] = -1
        right[node] = -1
        prio[node] = <int64_t>(self._next_rand() & <uint64_t>0x7FFFFFFFFFFFFFFF)
        active[node] = <uint8_t>1
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

    def remove(self, long entry_id):
        cdef int64_t node = <int64_t>entry_id
        cdef double[:] values
        cdef uint8_t[:] active
        if node < 0 or node >= self._size:
            return False
        active = self._active_arr
        if active[node] == 0:
            return False
        values = self._values_arr
        self._root = self._erase_rec(self._root, values[node], node)
        active[node] = <uint8_t>0
        self._active_count -= 1
        return True

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
                        double kurt_thresh=5.0):
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

    results = []

    for i in range(n_items):
        if n == 0:
            results.append((False, float("nan"), float("inf"), True, False))
            continue
        sx = sy = 0.0
        sum_xy = sum_xx = sum_yy = 0.0
        sum_xxx = sum_yyy = 0.0
        sum_xxxx = sum_yyyy = 0.0
        for j in range(n):
            xi = x[i, j]
            yi = y[i, j]
            sx += xi
            sy += yi
            sum_xy += xi * yi
            sum_xx += xi * xi
            sum_yy += yi * yi
            sum_xxx += xi * xi * xi
            sum_yyy += yi * yi * yi
            sum_xxxx += xi * xi * xi * xi
            sum_yyyy += yi * yi * yi * yi

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
