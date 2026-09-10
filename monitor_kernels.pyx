# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import time
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t, uint8_t

np.import_array()


cdef inline uint64_t _monitor_hash_key(int64_t s1, int64_t s2, int64_t lag) noexcept nogil:
    cdef uint64_t h = <uint64_t>1469598103934665603
    h ^= <uint64_t>(s1 + 1)
    h *= <uint64_t>1099511628211
    h ^= <uint64_t>(s2 + 1)
    h *= <uint64_t>1099511628211
    h ^= <uint64_t>(lag + 1)
    h *= <uint64_t>1099511628211
    h ^= h >> 33
    h *= <uint64_t>0xff51afd7ed558ccd
    h ^= h >> 33
    return h


cdef inline Py_ssize_t _monitor_next_power2(Py_ssize_t value) noexcept nogil:
    cdef Py_ssize_t out = 16
    while out < value:
        out <<= 1
    return out


cdef class NumericMonitorState:
    cdef object _occupied_arr
    cdef object _active_arr
    cdef object _active_slots_arr
    cdef object _active_pos_arr
    cdef object _frontier_slots_arr
    cdef object _frontier_pos_arr
    cdef object _next_frontier_slots_arr
    cdef object _queued_step_arr
    cdef object _key_s1_arr
    cdef object _key_s2_arr
    cdef object _key_lag_arr
    cdef object _t1_arr
    cdef object _t2_arr
    cdef object _window_arr
    cdef object _length_arr
    cdef object _sign_arr
    cdef object _seen_step_arr
    cdef object _status_rows_arr
    cdef object _anomaly_rows_arr
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _size
    cdef Py_ssize_t _active_count
    cdef Py_ssize_t _frontier_count
    cdef Py_ssize_t _next_frontier_count
    cdef Py_ssize_t _status_capacity
    cdef Py_ssize_t _status_count
    cdef Py_ssize_t _anomaly_capacity
    cdef Py_ssize_t _anomaly_count
    cdef int64_t _step_id
    # Diagnostic-only section timers (2026-07-03): coarse per-call timing to
    # localize the full-Cython monitor regression (5-6s -> 19s -> 22s across
    # three prior fix attempts, see docs/implementation_log.md). Not on the
    # hot per-row path: perf_counter() is called at most 4x per update() call
    # (once per section), not per row, so overhead is negligible.
    cdef double _t_capacity
    cdef double _t_row_loop
    cdef double _t_closeout
    cdef double _t_swap
    cdef Py_ssize_t _profile_calls
    # Row-branch classification counters (2026-07-03, diagnostic only): which
    # path each validated row takes inside the row loop, to find out whether
    # frontier bookkeeping (_remove_previous_frontier_slot/_mark_current_frontier)
    # is being paid unconditionally for rows that turn out to need no update.
    cdef Py_ssize_t _rows_new_activation
    cdef Py_ssize_t _rows_early_unchanged
    cdef Py_ssize_t _rows_extend
    cdef Py_ssize_t _rows_transition

    def __cinit__(self, Py_ssize_t initial_capacity=1024, Py_ssize_t output_capacity=1024):
        if initial_capacity < 16:
            initial_capacity = 16
        if output_capacity < 16:
            output_capacity = 16
        self._capacity = _monitor_next_power2(initial_capacity)
        self._size = 0
        self._active_count = 0
        self._frontier_count = 0
        self._next_frontier_count = 0
        self._status_capacity = output_capacity
        self._status_count = 0
        self._anomaly_capacity = output_capacity
        self._anomaly_count = 0
        self._step_id = 0
        self._t_capacity = 0.0
        self._t_row_loop = 0.0
        self._t_closeout = 0.0
        self._t_swap = 0.0
        self._profile_calls = 0
        self._rows_new_activation = 0
        self._rows_early_unchanged = 0
        self._rows_extend = 0
        self._rows_transition = 0
        self._occupied_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._active_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._active_slots_arr = np.empty(self._capacity, dtype=np.int64)
        self._active_pos_arr = np.full(self._capacity, -1, dtype=np.int64)
        self._frontier_slots_arr = np.empty(self._capacity, dtype=np.int64)
        self._frontier_pos_arr = np.full(self._capacity, -1, dtype=np.int64)
        self._next_frontier_slots_arr = np.empty(self._capacity, dtype=np.int64)
        self._queued_step_arr = np.zeros(self._capacity, dtype=np.int64)
        self._key_s1_arr = np.zeros(self._capacity, dtype=np.int64)
        self._key_s2_arr = np.zeros(self._capacity, dtype=np.int64)
        self._key_lag_arr = np.zeros(self._capacity, dtype=np.int64)
        self._t1_arr = np.zeros(self._capacity, dtype=np.int64)
        self._t2_arr = np.zeros(self._capacity, dtype=np.int64)
        self._window_arr = np.zeros(self._capacity, dtype=np.int64)
        self._length_arr = np.zeros(self._capacity, dtype=np.int64)
        self._sign_arr = np.zeros(self._capacity, dtype=np.int64)
        self._seen_step_arr = np.zeros(self._capacity, dtype=np.int64)
        self._status_rows_arr = np.empty((self._status_capacity, 7), dtype=np.int64)
        self._anomaly_rows_arr = np.empty((self._anomaly_capacity, 5), dtype=np.int64)

    cpdef reset(self):
        self._occupied_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._active_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._active_pos_arr = np.full(self._capacity, -1, dtype=np.int64)
        self._frontier_pos_arr = np.full(self._capacity, -1, dtype=np.int64)
        self._queued_step_arr = np.zeros(self._capacity, dtype=np.int64)
        self._size = 0
        self._active_count = 0
        self._frontier_count = 0
        self._next_frontier_count = 0
        self._status_count = 0
        self._anomaly_count = 0
        self._step_id = 0

    cdef Py_ssize_t _find_slot(self, int64_t s1, int64_t s2, int64_t lag):
        cdef uint8_t[:] occupied = self._occupied_arr
        cdef int64_t[:] key_s1 = self._key_s1_arr
        cdef int64_t[:] key_s2 = self._key_s2_arr
        cdef int64_t[:] key_lag = self._key_lag_arr
        cdef Py_ssize_t mask = self._capacity - 1
        cdef Py_ssize_t idx = <Py_ssize_t>(_monitor_hash_key(s1, s2, lag) & <uint64_t>mask)
        while occupied[idx] != 0:
            if key_s1[idx] == s1 and key_s2[idx] == s2 and key_lag[idx] == lag:
                return idx
            idx = (idx + 1) & mask
        return idx

    cdef void _rehash(self, Py_ssize_t new_capacity, bint keep_inactive):
        cdef object old_occupied_obj = self._occupied_arr
        cdef object old_active_obj = self._active_arr
        cdef object old_frontier_obj = self._frontier_slots_arr
        cdef object old_s1_obj = self._key_s1_arr
        cdef object old_s2_obj = self._key_s2_arr
        cdef object old_lag_obj = self._key_lag_arr
        cdef object old_t1_obj = self._t1_arr
        cdef object old_t2_obj = self._t2_arr
        cdef object old_window_obj = self._window_arr
        cdef object old_length_obj = self._length_arr
        cdef object old_sign_obj = self._sign_arr
        cdef object old_seen_obj = self._seen_step_arr
        cdef uint8_t[:] old_occupied = old_occupied_obj
        cdef uint8_t[:] old_active = old_active_obj
        cdef int64_t[:] old_frontier = old_frontier_obj
        cdef int64_t[:] old_s1 = old_s1_obj
        cdef int64_t[:] old_s2 = old_s2_obj
        cdef int64_t[:] old_lag = old_lag_obj
        cdef int64_t[:] old_t1 = old_t1_obj
        cdef int64_t[:] old_t2 = old_t2_obj
        cdef int64_t[:] old_window = old_window_obj
        cdef int64_t[:] old_length = old_length_obj
        cdef int64_t[:] old_sign = old_sign_obj
        cdef int64_t[:] old_seen = old_seen_obj
        cdef Py_ssize_t old_capacity = self._capacity
        cdef Py_ssize_t old_frontier_count = self._frontier_count
        cdef Py_ssize_t old_i, old_slot, idx, mask, pos
        cdef uint8_t[:] occupied
        cdef uint8_t[:] active
        cdef int64_t[:] active_slots
        cdef int64_t[:] active_pos
        cdef int64_t[:] frontier_slots
        cdef int64_t[:] frontier_pos
        cdef int64_t[:] key_s1
        cdef int64_t[:] key_s2
        cdef int64_t[:] key_lag
        cdef int64_t[:] t1_arr
        cdef int64_t[:] t2_arr
        cdef int64_t[:] window_arr
        cdef int64_t[:] length_arr
        cdef int64_t[:] sign_arr
        cdef int64_t[:] seen_arr

        new_capacity = _monitor_next_power2(new_capacity)
        self._capacity = new_capacity
        self._occupied_arr = np.zeros(new_capacity, dtype=np.uint8)
        self._active_arr = np.zeros(new_capacity, dtype=np.uint8)
        self._active_slots_arr = np.empty(new_capacity, dtype=np.int64)
        self._active_pos_arr = np.full(new_capacity, -1, dtype=np.int64)
        self._frontier_slots_arr = np.empty(new_capacity, dtype=np.int64)
        self._frontier_pos_arr = np.full(new_capacity, -1, dtype=np.int64)
        self._next_frontier_slots_arr = np.empty(new_capacity, dtype=np.int64)
        self._queued_step_arr = np.zeros(new_capacity, dtype=np.int64)
        self._key_s1_arr = np.zeros(new_capacity, dtype=np.int64)
        self._key_s2_arr = np.zeros(new_capacity, dtype=np.int64)
        self._key_lag_arr = np.zeros(new_capacity, dtype=np.int64)
        self._t1_arr = np.zeros(new_capacity, dtype=np.int64)
        self._t2_arr = np.zeros(new_capacity, dtype=np.int64)
        self._window_arr = np.zeros(new_capacity, dtype=np.int64)
        self._length_arr = np.zeros(new_capacity, dtype=np.int64)
        self._sign_arr = np.zeros(new_capacity, dtype=np.int64)
        self._seen_step_arr = np.zeros(new_capacity, dtype=np.int64)

        occupied = self._occupied_arr
        active = self._active_arr
        active_slots = self._active_slots_arr
        active_pos = self._active_pos_arr
        frontier_slots = self._frontier_slots_arr
        frontier_pos = self._frontier_pos_arr
        key_s1 = self._key_s1_arr
        key_s2 = self._key_s2_arr
        key_lag = self._key_lag_arr
        t1_arr = self._t1_arr
        t2_arr = self._t2_arr
        window_arr = self._window_arr
        length_arr = self._length_arr
        sign_arr = self._sign_arr
        seen_arr = self._seen_step_arr
        mask = new_capacity - 1
        self._size = 0
        self._active_count = 0
        self._frontier_count = 0
        self._next_frontier_count = 0
        for old_i in range(old_capacity):
            if old_occupied[old_i] == 0:
                continue
            if not keep_inactive and old_active[old_i] == 0:
                continue
            idx = <Py_ssize_t>(_monitor_hash_key(old_s1[old_i], old_s2[old_i], old_lag[old_i]) & <uint64_t>mask)
            while occupied[idx] != 0:
                idx = (idx + 1) & mask
            occupied[idx] = <uint8_t>1
            key_s1[idx] = old_s1[old_i]
            key_s2[idx] = old_s2[old_i]
            key_lag[idx] = old_lag[old_i]
            t1_arr[idx] = old_t1[old_i]
            t2_arr[idx] = old_t2[old_i]
            window_arr[idx] = old_window[old_i]
            length_arr[idx] = old_length[old_i]
            sign_arr[idx] = old_sign[old_i]
            seen_arr[idx] = old_seen[old_i]
            self._size += 1
            if old_active[old_i] != 0:
                active[idx] = <uint8_t>1
                active_slots[self._active_count] = <int64_t>idx
                active_pos[idx] = <int64_t>self._active_count
                self._active_count += 1

        for pos in range(old_frontier_count):
            old_slot = <Py_ssize_t>old_frontier[pos]
            if old_slot < 0 or old_slot >= old_capacity or old_active[old_slot] == 0:
                continue
            idx = self._find_slot(old_s1[old_slot], old_s2[old_slot], old_lag[old_slot])
            frontier_slots[self._frontier_count] = <int64_t>idx
            frontier_pos[idx] = <int64_t>self._frontier_count
            self._frontier_count += 1

    cdef void _ensure_hash_capacity(self, Py_ssize_t need):
        while need * 2 >= self._capacity:
            self._rehash(self._capacity * 2, True)

    # Periodic compaction (2026-07-03): _occupied_arr never shrinks on its
    # own (_deactivate_slot only clears the _active_arr flag), so on a
    # workload with high churn -- pairs repeatedly flickering active/inactive
    # -- the table grows to hold every distinct (s1,s2,lag) key ever seen and
    # never gives the memory back, even though the live/active working set
    # stays small and roughly constant. Measured on the standard
    # synthetic_50_52560/cosine reproduction: occupied_count() plateaus
    # around 13,500-14,000 while active_count() stays ~1,675 -- an ~8.2x
    # permanent bloat. This method reclaims it by periodically rehashing into
    # a smaller table that keeps only active slots.
    #
    # Trigger (8x) and target (4x, leaving headroom above the fresh active
    # count) are deliberately wide apart. A tighter first attempt
    # (trigger=4x/target=2x) left no room for the same update() call's
    # incoming rows, so _ensure_hash_capacity (called right after this) would
    # immediately re-grow the table just shrunk -- two full O(table size)
    # rehashes per triggering step instead of one occasional one. Keep the
    # gap wide so compaction stays a rare, batched operation, matching how
    # _ensure_hash_capacity's own growth is already batched.
    #
    # Net effect measured (clean, isolated runs, not back-to-back with other
    # heavy benchmarks -- an earlier same-day comparison wrongly concluded
    # this was a large regression, contaminated by memory pressure from many
    # consecutive prior runs on a resource-constrained box; see
    # docs/implementation_log.md, "compaction fix tested and rejected" vs.
    # the corrected follow-up entry, for the full story): a modest but real
    # win -- monit_time 20.68s -> 19.50s (-5.7%), total runtime 69.18s ->
    # 57.06s (-17.5%, i.e. other phases improved too, plausibly from reduced
    # allocation/memory pressure). This does not close the gap to the
    # original ~5-6s Python baseline -- most of that gap remains unexplained;
    # see the implementation_log for what has been ruled out so far.
    cdef void _maybe_compact(self):
        cdef Py_ssize_t new_capacity
        if self._size <= 256:
            return
        if self._size <= self._active_count * 8:
            return
        new_capacity = _monitor_next_power2(max(<Py_ssize_t>16, self._active_count * 4))
        self._rehash(new_capacity, False)
    # The keep_inactive parameter on _rehash() is kept (harmless, always
    # called with True from the one live call site above) so this can be
    # revisited without re-deriving the compaction-loop plumbing from scratch.

    cdef void _ensure_status_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        cdef object new_rows
        if need <= self._status_capacity:
            return
        new_cap = self._status_capacity
        while new_cap < need:
            new_cap *= 2
        new_rows = np.empty((new_cap, 7), dtype=np.int64)
        if self._status_count > 0:
            new_rows[:self._status_count, :] = self._status_rows_arr[:self._status_count, :]
        self._status_rows_arr = new_rows
        self._status_capacity = new_cap

    cdef void _ensure_anomaly_capacity(self, Py_ssize_t need):
        cdef Py_ssize_t new_cap
        cdef object new_rows
        if need <= self._anomaly_capacity:
            return
        new_cap = self._anomaly_capacity
        while new_cap < need:
            new_cap *= 2
        new_rows = np.empty((new_cap, 5), dtype=np.int64)
        if self._anomaly_count > 0:
            new_rows[:self._anomaly_count, :] = self._anomaly_rows_arr[:self._anomaly_count, :]
        self._anomaly_rows_arr = new_rows
        self._anomaly_capacity = new_cap

    # (2026-07-03) These four helpers now take their arrays as memoryview
    # parameters instead of re-deriving them from self._xxx_arr internally.
    # They are called once or twice per row from update()'s hot loop
    # (_remove_previous_frontier_slot: every row; _mark_current_frontier:
    # every row except the early-unchanged ~4% case), so re-fetching a
    # memoryview from a Python attribute inside them, every call, paid the
    # same avoidable cost that hoisting the outer loop's own arrays fixed --
    # see the comment above the hoisted declarations in update(). Callers in
    # finalize() (not hot -- runs once per active pair at the very end of a
    # run) fetch the views locally right before use, same as before.
    cdef void _activate_slot(self, Py_ssize_t slot, uint8_t[:] active,
                              int64_t[:] active_slots, int64_t[:] active_pos):
        if active[slot] != 0:
            return
        active[slot] = <uint8_t>1
        active_slots[self._active_count] = <int64_t>slot
        active_pos[slot] = <int64_t>self._active_count
        self._active_count += 1

    cdef void _deactivate_slot(self, Py_ssize_t slot, uint8_t[:] active,
                                int64_t[:] active_slots, int64_t[:] active_pos):
        cdef Py_ssize_t pos
        cdef Py_ssize_t last_pos
        cdef Py_ssize_t last_slot
        if active[slot] == 0:
            return
        pos = <Py_ssize_t>active_pos[slot]
        last_pos = self._active_count - 1
        last_slot = <Py_ssize_t>active_slots[last_pos]
        if pos != last_pos:
            active_slots[pos] = <int64_t>last_slot
            active_pos[last_slot] = <int64_t>pos
        active_pos[slot] = <int64_t>-1
        active[slot] = <uint8_t>0
        self._active_count -= 1

    cdef void _mark_current_frontier(self, Py_ssize_t slot, int64_t[:] queued_step,
                                      int64_t[:] next_frontier):
        if queued_step[slot] == self._step_id:
            return
        queued_step[slot] = self._step_id
        next_frontier[self._next_frontier_count] = <int64_t>slot
        self._next_frontier_count += 1

    cdef bint _remove_previous_frontier_slot(self, Py_ssize_t slot, int64_t[:] frontier_slots,
                                              int64_t[:] frontier_pos):
        cdef Py_ssize_t pos
        cdef Py_ssize_t last_pos
        cdef Py_ssize_t last_slot
        if frontier_pos[slot] < 0:
            return False
        pos = <Py_ssize_t>frontier_pos[slot]
        last_pos = self._frontier_count - 1
        last_slot = <Py_ssize_t>frontier_slots[last_pos]
        if pos != last_pos:
            frontier_slots[pos] = <int64_t>last_slot
            frontier_pos[last_slot] = <int64_t>pos
        frontier_pos[slot] = <int64_t>-1
        self._frontier_count -= 1
        return True

    cdef void _append_status(self, int64_t slot):
        cdef int64_t[:, :] rows
        cdef uint8_t[:] active = self._active_arr
        cdef int64_t[:] key_s1 = self._key_s1_arr
        cdef int64_t[:] key_s2 = self._key_s2_arr
        cdef int64_t[:] key_lag = self._key_lag_arr
        cdef int64_t[:] t1_arr = self._t1_arr
        cdef int64_t[:] t2_arr = self._t2_arr
        cdef int64_t[:] length_arr = self._length_arr
        cdef int64_t[:] sign_arr = self._sign_arr
        cdef Py_ssize_t pos
        if active[slot] == 0:
            return
        self._ensure_status_capacity(self._status_count + 1)
        rows = self._status_rows_arr
        pos = self._status_count
        rows[pos, 0] = key_s1[slot]
        rows[pos, 1] = key_s2[slot]
        rows[pos, 2] = key_lag[slot]
        rows[pos, 3] = t1_arr[slot]
        rows[pos, 4] = t2_arr[slot]
        rows[pos, 5] = length_arr[slot]
        rows[pos, 6] = sign_arr[slot]
        self._status_count += 1

    cdef void _append_anomaly(self, int64_t s1, int64_t s2, int64_t lag, int64_t time_value, int64_t marker):
        cdef int64_t[:, :] rows
        cdef Py_ssize_t pos
        self._ensure_anomaly_capacity(self._anomaly_count + 1)
        rows = self._anomaly_rows_arr
        pos = self._anomaly_count
        rows[pos, 0] = s1
        rows[pos, 1] = s2
        rows[pos, 2] = lag
        rows[pos, 3] = time_value
        rows[pos, 4] = marker
        self._anomaly_count += 1

    cpdef update(self,
                 long[:, ::1] rows,
                 double[:] corrs,
                 int window_step,
                 bint save_status,
                 bint save_anomalies):
        cdef Py_ssize_t n_rows = rows.shape[0]
        cdef Py_ssize_t n_corrs = corrs.shape[0]
        cdef Py_ssize_t i, slot_i
        cdef int64_t sid1, sid2, key_s1, key_s2
        cdef int64_t t1, t2, key_t1, key_t2, window_size
        cdef int64_t min_time, max_time, lag, corr_sign
        cdef int64_t first_t1, first_t2, last_window_size, last_corr_length, last_corr_sign
        cdef int64_t curr_time, next_corr_time, out_time
        cdef double corr
        cdef uint8_t[:] occupied
        cdef uint8_t[:] active
        cdef int64_t[:] arr_s1
        cdef int64_t[:] arr_s2
        cdef int64_t[:] arr_lag
        cdef int64_t[:] arr_t1
        cdef int64_t[:] arr_t2
        cdef int64_t[:] arr_window
        cdef int64_t[:] arr_length
        cdef int64_t[:] arr_sign
        cdef int64_t[:] arr_seen
        cdef int64_t[:] frontier_slots
        cdef int64_t[:] frontier_pos
        cdef int64_t[:] active_slots
        cdef int64_t[:] active_pos
        cdef int64_t[:] queued_step
        cdef int64_t[:] next_frontier
        cdef object swap_obj
        cdef Py_ssize_t scan_pos
        cdef double _tp0

        _tp0 = time.perf_counter()
        self._maybe_compact()
        self._ensure_hash_capacity(self._size + n_rows)
        self._t_capacity += time.perf_counter() - _tp0
        self._step_id += 1
        self._next_frontier_count = 0
        _tp0 = time.perf_counter()
        # Hoisted out of the loop (2026-07-03): these memoryviews were
        # previously re-derived from self._xxx_arr on every single row, even
        # though nothing touched during row processing (_activate_slot,
        # _deactivate_slot, _mark_current_frontier,
        # _remove_previous_frontier_slot, _append_status, _append_anomaly)
        # ever reallocates the underlying arrays -- only _rehash does, and
        # that only runs once, before this loop, via _maybe_compact()/
        # _ensure_hash_capacity() above. Re-acquiring a memoryview from a
        # Python attribute is not free (attribute lookup + buffer-protocol
        # handshake); doing it ~1750 times/step x 11 arrays x ~2284 steps
        # was a classic avoidable Cython hot-loop cost. See
        # docs/implementation_log.md for the measured effect.
        occupied = self._occupied_arr
        active = self._active_arr
        arr_s1 = self._key_s1_arr
        arr_s2 = self._key_s2_arr
        arr_lag = self._key_lag_arr
        arr_t1 = self._t1_arr
        arr_t2 = self._t2_arr
        arr_window = self._window_arr
        arr_length = self._length_arr
        arr_sign = self._sign_arr
        arr_seen = self._seen_step_arr
        # Same hoisting, extended (2026-07-03) to the arrays used by
        # _activate_slot/_deactivate_slot/_mark_current_frontier/
        # _remove_previous_frontier_slot, now passed in as parameters instead
        # of those helpers re-deriving them from self._xxx_arr on every call.
        active_slots = self._active_slots_arr
        active_pos = self._active_pos_arr
        queued_step = self._queued_step_arr
        next_frontier = self._next_frontier_slots_arr
        frontier_slots = self._frontier_slots_arr
        frontier_pos = self._frontier_pos_arr
        for i in range(n_rows):
            sid1 = <int64_t>rows[i, 0]
            sid2 = <int64_t>rows[i, 1]
            if sid1 < 0 or sid2 < 0:
                continue
            t1 = <int64_t>rows[i, 2]
            t2 = <int64_t>rows[i, 3]
            window_size = <int64_t>rows[i, 4]
            min_time = t1 if t1 <= t2 else t2
            max_time = t1 if t1 >= t2 else t2
            lag = max_time - min_time
            corr = corrs[i] if i < n_corrs else 1.0
            corr_sign = 1 if corr >= 0.0 else -1
            if sid1 <= sid2:
                key_s1 = sid1
                key_s2 = sid2
                key_t1 = t1
                key_t2 = t2
            else:
                key_s1 = sid2
                key_s2 = sid1
                key_t1 = t2
                key_t2 = t1

            slot_i = self._find_slot(key_s1, key_s2, lag)

            if occupied[slot_i] == 0:
                occupied[slot_i] = <uint8_t>1
                arr_s1[slot_i] = key_s1
                arr_s2[slot_i] = key_s2
                arr_lag[slot_i] = lag
                self._size += 1

            self._remove_previous_frontier_slot(slot_i, frontier_slots, frontier_pos)
            if active[slot_i] == 0:
                self._activate_slot(slot_i, active, active_slots, active_pos)
                arr_t1[slot_i] = key_t1
                arr_t2[slot_i] = key_t2
                arr_window[slot_i] = window_size
                arr_length[slot_i] = window_size
                arr_sign[slot_i] = corr_sign
                arr_seen[slot_i] = self._step_id
                self._mark_current_frontier(slot_i, queued_step, next_frontier)
                if save_anomalies:
                    self._append_anomaly(key_s1, key_s2, lag, min_time, 1)
                self._rows_new_activation += 1
                continue

            self._mark_current_frontier(slot_i, queued_step, next_frontier)
            first_t1 = arr_t1[slot_i]
            first_t2 = arr_t2[slot_i]
            last_window_size = arr_window[slot_i]
            last_corr_length = arr_length[slot_i]
            last_corr_sign = arr_sign[slot_i]
            curr_time = max_time + window_size
            if first_t1 >= first_t2:
                next_corr_time = first_t1 + last_corr_length + window_step
            else:
                next_corr_time = first_t2 + last_corr_length + window_step

            if curr_time < next_corr_time:
                arr_seen[slot_i] = self._step_id
                self._rows_early_unchanged += 1
                continue
            if curr_time == next_corr_time and window_size == last_window_size and corr_sign == last_corr_sign:
                arr_length[slot_i] = last_corr_length + window_step
                arr_seen[slot_i] = self._step_id
                self._rows_extend += 1
                continue

            if save_status:
                self._append_status(slot_i)
            arr_t1[slot_i] = key_t1
            arr_t2[slot_i] = key_t2
            arr_window[slot_i] = window_size
            arr_length[slot_i] = window_size
            arr_sign[slot_i] = corr_sign
            arr_seen[slot_i] = self._step_id
            if save_anomalies:
                self._append_anomaly(key_s1, key_s2, lag, min_time, 1 if corr_sign == last_corr_sign else 0)
            self._rows_transition += 1

        self._t_row_loop += time.perf_counter() - _tp0

        # arr_t1/arr_t2/arr_window/arr_length/arr_seen/arr_s1/arr_s2/arr_lag
        # and frontier_slots are still the correct, valid views here (2026-
        # 07-03: removed a redundant re-fetch -- none of these were ever
        # reassigned by the row loop above, only the *values inside* them
        # were written, which a stale-vs-fresh memoryview distinction doesn't
        # affect). frontier_pos is also still valid for the same reason.
        _tp0 = time.perf_counter()
        while self._frontier_count > 0:
            slot_i = <Py_ssize_t>frontier_slots[self._frontier_count - 1]
            min_time = arr_t1[slot_i] if arr_t1[slot_i] <= arr_t2[slot_i] else arr_t2[slot_i]
            out_time = min_time + arr_length[slot_i] - (arr_window[slot_i] - window_step)
            if save_anomalies:
                self._append_anomaly(arr_s1[slot_i], arr_s2[slot_i], arr_lag[slot_i], out_time, -1)
            if save_status:
                self._append_status(slot_i)
            self._remove_previous_frontier_slot(slot_i, frontier_slots, frontier_pos)
            self._deactivate_slot(slot_i, active, active_slots, active_pos)
        self._t_closeout += time.perf_counter() - _tp0
        _tp0 = time.perf_counter()
        swap_obj = self._frontier_slots_arr
        self._frontier_slots_arr = self._next_frontier_slots_arr
        self._next_frontier_slots_arr = swap_obj
        self._frontier_count = self._next_frontier_count
        self._next_frontier_count = 0
        frontier_slots = self._frontier_slots_arr
        frontier_pos = self._frontier_pos_arr
        for scan_pos in range(self._frontier_count):
            frontier_pos[<Py_ssize_t>frontier_slots[scan_pos]] = <int64_t>scan_pos
        self._t_swap += time.perf_counter() - _tp0
        self._profile_calls += 1

    cpdef finalize(self, bint save_status=True):
        cdef uint8_t[:] active = self._active_arr
        cdef int64_t[:] active_slots = self._active_slots_arr
        cdef int64_t[:] active_pos = self._active_pos_arr
        cdef int64_t[:] frontier_slots = self._frontier_slots_arr
        cdef int64_t[:] frontier_pos = self._frontier_pos_arr
        cdef Py_ssize_t slot_i
        cdef Py_ssize_t pos
        for pos in range(self._frontier_count):
            slot_i = <Py_ssize_t>frontier_slots[pos]
            if slot_i >= 0 and slot_i < self._capacity:
                frontier_pos[slot_i] = <int64_t>-1
        while self._active_count > 0:
            slot_i = <Py_ssize_t>active_slots[self._active_count - 1]
            if save_status:
                self._append_status(slot_i)
            self._deactivate_slot(slot_i, active, active_slots, active_pos)
        self._frontier_count = 0
        self._next_frontier_count = 0

    cpdef Py_ssize_t pending_status_count(self):
        return self._status_count

    cpdef Py_ssize_t pending_anomaly_count(self):
        return self._anomaly_count

    cpdef Py_ssize_t active_count(self):
        return self._active_count

    cpdef Py_ssize_t occupied_count(self):
        """Diagnostic-only (2026-07-03): total distinct (s1,s2,lag) slots ever
        occupied, which never shrinks (deactivation only clears _active_arr,
        not _occupied_arr) -- compare against active_count() and capacity()."""
        return self._size

    cpdef Py_ssize_t capacity(self):
        return self._capacity

    cpdef object profile_snapshot(self):
        """Diagnostic-only (2026-07-03): cumulative section timings from update().
        Returns (t_capacity, t_row_loop, t_closeout, t_swap, call_count)."""
        return (
            self._t_capacity,
            self._t_row_loop,
            self._t_closeout,
            self._t_swap,
            self._profile_calls,
        )

    cpdef object row_branch_snapshot(self):
        """Diagnostic-only (2026-07-03): cumulative row-loop branch counts.
        Returns (new_activation, early_unchanged, extend, transition)."""
        return (
            self._rows_new_activation,
            self._rows_early_unchanged,
            self._rows_extend,
            self._rows_transition,
        )

    cpdef object take_status_rows(self):
        cdef object out
        if self._status_count <= 0:
            return np.empty((0, 7), dtype=np.int64)
        out = np.ascontiguousarray(self._status_rows_arr[:self._status_count, :], dtype=np.int64)
        self._status_count = 0
        return out

    cpdef object take_anomaly_rows(self):
        cdef object out
        if self._anomaly_count <= 0:
            return np.empty((0, 5), dtype=np.int64)
        out = np.ascontiguousarray(self._anomaly_rows_arr[:self._anomaly_count, :], dtype=np.int64)
        self._anomaly_count = 0
        return out

    cpdef object copy_status_rows(self):
        if self._status_count <= 0:
            return np.empty((0, 7), dtype=np.int64)
        return np.ascontiguousarray(self._status_rows_arr[:self._status_count, :], dtype=np.int64)

    cpdef object copy_anomaly_rows(self):
        if self._anomaly_count <= 0:
            return np.empty((0, 5), dtype=np.int64)
        return np.ascontiguousarray(self._anomaly_rows_arr[:self._anomaly_count, :], dtype=np.int64)


# (2026-07-27) Cython prototype for candidate_skip_ahead_revalidation's
# per-pair bookkeeping (see docs/implementation_log.md). Profiling found the
# pure-Python version (Python dict, tuple keys) cost a flat ~39% of wall
# time with no algorithmic blowup -- unlike the transitive-bound witness
# search, which had a real capped-search fix instead. This mirrors
# NumericMonitorState's own open-addressing hash table (same
# _monitor_hash_key/_monitor_next_power2 helpers, same array-backed slot
# layout) rather than inventing a new scheme, since that pattern is already
# established and tested in this file for an (s1, s2, lag)-keyed table.
cdef class SkipAheadState:
    cdef object _occupied_arr
    cdef object _key_s1_arr
    cdef object _key_s2_arr
    cdef object _key_lag_arr
    cdef object _corr_arr
    cdef object _steps_arr
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _size

    def __cinit__(self, Py_ssize_t initial_capacity=1024):
        if initial_capacity < 16:
            initial_capacity = 16
        self._capacity = _monitor_next_power2(initial_capacity)
        self._size = 0
        self._occupied_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._key_s1_arr = np.zeros(self._capacity, dtype=np.int64)
        self._key_s2_arr = np.zeros(self._capacity, dtype=np.int64)
        self._key_lag_arr = np.zeros(self._capacity, dtype=np.int64)
        self._corr_arr = np.zeros(self._capacity, dtype=np.float64)
        self._steps_arr = np.zeros(self._capacity, dtype=np.int64)

    cdef Py_ssize_t _find_slot(self, int64_t s1, int64_t s2, int64_t lag):
        cdef uint8_t[:] occupied = self._occupied_arr
        cdef int64_t[:] key_s1 = self._key_s1_arr
        cdef int64_t[:] key_s2 = self._key_s2_arr
        cdef int64_t[:] key_lag = self._key_lag_arr
        cdef Py_ssize_t mask = self._capacity - 1
        cdef Py_ssize_t idx = <Py_ssize_t>(_monitor_hash_key(s1, s2, lag) & <uint64_t>mask)
        while occupied[idx] != 0:
            if key_s1[idx] == s1 and key_s2[idx] == s2 and key_lag[idx] == lag:
                return idx
            idx = (idx + 1) & mask
        return idx

    cdef void _rehash(self, Py_ssize_t new_capacity):
        cdef object old_occupied_obj = self._occupied_arr
        cdef object old_s1_obj = self._key_s1_arr
        cdef object old_s2_obj = self._key_s2_arr
        cdef object old_lag_obj = self._key_lag_arr
        cdef object old_corr_obj = self._corr_arr
        cdef object old_steps_obj = self._steps_arr
        cdef uint8_t[:] old_occupied = old_occupied_obj
        cdef int64_t[:] old_s1 = old_s1_obj
        cdef int64_t[:] old_s2 = old_s2_obj
        cdef int64_t[:] old_lag = old_lag_obj
        cdef double[:] old_corr = old_corr_obj
        cdef int64_t[:] old_steps = old_steps_obj
        cdef Py_ssize_t old_capacity = self._capacity
        cdef Py_ssize_t old_i, idx, mask
        cdef uint8_t[:] occupied
        cdef int64_t[:] key_s1
        cdef int64_t[:] key_s2
        cdef int64_t[:] key_lag
        cdef double[:] corr_arr
        cdef int64_t[:] steps_arr

        new_capacity = _monitor_next_power2(new_capacity)
        self._capacity = new_capacity
        self._occupied_arr = np.zeros(new_capacity, dtype=np.uint8)
        self._key_s1_arr = np.zeros(new_capacity, dtype=np.int64)
        self._key_s2_arr = np.zeros(new_capacity, dtype=np.int64)
        self._key_lag_arr = np.zeros(new_capacity, dtype=np.int64)
        self._corr_arr = np.zeros(new_capacity, dtype=np.float64)
        self._steps_arr = np.zeros(new_capacity, dtype=np.int64)

        occupied = self._occupied_arr
        key_s1 = self._key_s1_arr
        key_s2 = self._key_s2_arr
        key_lag = self._key_lag_arr
        corr_arr = self._corr_arr
        steps_arr = self._steps_arr
        mask = new_capacity - 1
        self._size = 0
        for old_i in range(old_capacity):
            if old_occupied[old_i] == 0:
                continue
            idx = <Py_ssize_t>(_monitor_hash_key(old_s1[old_i], old_s2[old_i], old_lag[old_i]) & <uint64_t>mask)
            while occupied[idx] != 0:
                idx = (idx + 1) & mask
            occupied[idx] = <uint8_t>1
            key_s1[idx] = old_s1[old_i]
            key_s2[idx] = old_s2[old_i]
            key_lag[idx] = old_lag[old_i]
            corr_arr[idx] = old_corr[old_i]
            steps_arr[idx] = old_steps[old_i]
            self._size += 1

    cdef void _ensure_capacity(self, Py_ssize_t need):
        while need * 2 >= self._capacity:
            self._rehash(self._capacity * 2)

    cpdef void update_batch(self, np.ndarray rows, np.ndarray corrs):
        cdef int64_t[:, :] rows_view = rows
        cdef double[:] corrs_view = corrs
        cdef Py_ssize_t n = rows_view.shape[0]
        cdef Py_ssize_t i, idx
        cdef int64_t s1, s2, t1, t2, lag, ns1, ns2, nlag
        cdef uint8_t[:] occupied
        cdef int64_t[:] key_s1
        cdef int64_t[:] key_s2
        cdef int64_t[:] key_lag
        cdef double[:] corr_arr
        cdef int64_t[:] steps_arr

        if n == 0:
            return
        self._ensure_capacity(self._size + n)
        occupied = self._occupied_arr
        key_s1 = self._key_s1_arr
        key_s2 = self._key_s2_arr
        key_lag = self._key_lag_arr
        corr_arr = self._corr_arr
        steps_arr = self._steps_arr

        for i in range(n):
            s1 = rows_view[i, 0]
            s2 = rows_view[i, 1]
            t1 = rows_view[i, 2]
            t2 = rows_view[i, 3]
            lag = t1 - t2
            if s1 == s2:
                ns1 = s1
                ns2 = s2
                nlag = lag if lag >= 0 else -lag
            elif s1 < s2:
                ns1 = s1
                ns2 = s2
                nlag = lag
            else:
                ns1 = s2
                ns2 = s1
                nlag = -lag
            idx = self._find_slot(ns1, ns2, nlag)
            if occupied[idx] == 0:
                occupied[idx] = <uint8_t>1
                key_s1[idx] = ns1
                key_s2[idx] = ns2
                key_lag[idx] = nlag
                self._size += 1
            corr_arr[idx] = corrs_view[i]
            steps_arr[idx] = 0

    cpdef object prune_mask(self, np.ndarray rows, double threshold, double margin, int64_t max_steps, bint neg_corr):
        cdef int64_t[:, :] rows_view = rows
        cdef Py_ssize_t n = rows_view.shape[0]
        cdef Py_ssize_t i, idx
        cdef int64_t s1, s2, t1, t2, lag, ns1, ns2, nlag
        cdef uint8_t[:] occupied
        cdef int64_t[:] key_s1
        cdef int64_t[:] key_s2
        cdef int64_t[:] key_lag
        cdef double[:] corr_arr
        cdef int64_t[:] steps_arr
        cdef double last_corr, floor
        cdef bint comfortably_below
        cdef object keep = np.ones(n, dtype=np.uint8)
        cdef uint8_t[:] keep_view = keep
        cdef Py_ssize_t skipped = 0

        occupied = self._occupied_arr
        key_s1 = self._key_s1_arr
        key_s2 = self._key_s2_arr
        key_lag = self._key_lag_arr
        corr_arr = self._corr_arr
        steps_arr = self._steps_arr
        floor = threshold - margin

        for i in range(n):
            s1 = rows_view[i, 0]
            s2 = rows_view[i, 1]
            t1 = rows_view[i, 2]
            t2 = rows_view[i, 3]
            lag = t1 - t2
            if s1 == s2:
                ns1 = s1
                ns2 = s2
                nlag = lag if lag >= 0 else -lag
            elif s1 < s2:
                ns1 = s1
                ns2 = s2
                nlag = lag
            else:
                ns1 = s2
                ns2 = s1
                nlag = -lag
            idx = self._find_slot(ns1, ns2, nlag)
            if occupied[idx] == 0:
                continue
            last_corr = corr_arr[idx]
            if neg_corr:
                comfortably_below = (last_corr if last_corr >= 0 else -last_corr) < floor
            else:
                comfortably_below = last_corr < floor
            if comfortably_below and steps_arr[idx] < max_steps:
                keep_view[i] = <uint8_t>0
                steps_arr[idx] += 1
                skipped += 1
        return keep, skipped

    cpdef Py_ssize_t size(self):
        return self._size
