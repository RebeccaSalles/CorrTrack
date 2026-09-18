# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import time
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t, uint8_t, int32_t, int8_t, uintptr_t

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


cdef extern from *:
    """
    #define MON_PREFETCH_W(p) __builtin_prefetch((const void *)(p), 1, 3)
    """
    void MON_PREFETCH_W(void* p) nogil


# (2026-09-18, entry i) NumericMonitorState memory layout: array-of-structs. The
# per-slot state used to live in 14 parallel numpy arrays (SoA), so every
# validated row touched ~13 distinct cache lines once the table outgrew L2
# (~29 MB at the 262K-slot capacity a narrow-band FilCorr run needs). It is
# now one 64-byte MonitorSlot per slot (one cache line), the home slot of
# row i+16 is software-prefetched while row i is processed, and the two
# output appenders write through cached raw pointers instead of re-acquiring
# a memoryview per call. Same hash, same linear probing, same insertion
# order, same branch structure, same output order: verified bit-identical
# (status rows, anomaly rows, active counts, branch counts) against both the
# 2026-09-18 SoA kernel and the pre-2026-09-18 one over 400 synthetic steps
# in all three save_status/save_anomalies combinations. Micro-bench on the
# 27K rows/step flood (see docs/implementation_log.md): row loop 0.27 ->
# 0.024 us/row, closeout 0.50s -> 0.02s, ~10x on the whole update() call.
# The int32 fields (series ids, lags, window sizes, episode lengths, step
# ids) are guarded: update() raises ValueError if any input reaches 2**30
# rather than silently truncating.
# One slot = one cache line. Field order keeps natural alignment so sizeof is
# exactly 64 without packing: 2 x int64 (16) + 9 x int32 (36) + 4 x 1 byte
# (4) = 56, then an int64 pad to 64. Series ids, lags, window sizes, episode
# lengths and step ids are int32 here; update() checks every incoming value
# against MON_INT32_LIMIT and raises instead of truncating.
cdef struct MonitorSlot:
    int64_t t1
    int64_t t2
    int32_t s1
    int32_t s2
    int32_t lag
    int32_t window
    int32_t length
    int32_t seen
    int32_t queued_step
    int32_t active_pos
    int32_t frontier_pos
    int8_t sign
    uint8_t occupied
    uint8_t active
    uint8_t _pad0
    int64_t _pad1

cdef int64_t MON_INT32_LIMIT = <int64_t>1 << 30


cdef class NumericMonitorState:
    cdef object _slots_buf          # uint8 backing store, keeps _slots alive
    cdef MonitorSlot* _slots        # 64-byte aligned view into _slots_buf
    cdef object _active_slots_arr
    cdef object _frontier_slots_arr
    cdef object _next_frontier_slots_arr
    cdef object _status_rows_arr
    cdef object _anomaly_rows_arr
    # Raw pointers into the two output buffers, refreshed only when they
    # grow; the appenders write through these instead of re-acquiring a
    # memoryview from the Python attribute on every call.
    cdef int64_t* _status_ptr
    cdef int64_t* _anomaly_ptr
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
    cdef Py_ssize_t _prefetch_dist
    cdef double _t_capacity
    cdef double _t_row_loop
    cdef double _t_closeout
    cdef double _t_swap
    cdef Py_ssize_t _profile_calls
    cdef Py_ssize_t _rows_new_activation
    cdef Py_ssize_t _rows_early_unchanged
    cdef Py_ssize_t _rows_extend
    cdef Py_ssize_t _rows_transition

    def __cinit__(self, Py_ssize_t initial_capacity=1024, Py_ssize_t output_capacity=1024,
                  Py_ssize_t prefetch_dist=16):
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
        self._prefetch_dist = prefetch_dist
        self._t_capacity = 0.0
        self._t_row_loop = 0.0
        self._t_closeout = 0.0
        self._t_swap = 0.0
        self._profile_calls = 0
        self._rows_new_activation = 0
        self._rows_early_unchanged = 0
        self._rows_extend = 0
        self._rows_transition = 0
        self._alloc_slots(self._capacity)
        self._active_slots_arr = np.empty(self._capacity, dtype=np.int64)
        self._frontier_slots_arr = np.empty(self._capacity, dtype=np.int64)
        self._next_frontier_slots_arr = np.empty(self._capacity, dtype=np.int64)
        self._status_rows_arr = np.empty((self._status_capacity, 7), dtype=np.int64)
        self._anomaly_rows_arr = np.empty((self._anomaly_capacity, 5), dtype=np.int64)
        self._status_ptr = <int64_t*>(<np.ndarray>self._status_rows_arr).data
        self._anomaly_ptr = <int64_t*>(<np.ndarray>self._anomaly_rows_arr).data

    cdef void _alloc_slots(self, Py_ssize_t capacity):
        """Zeroed slot table (occupied=0, active=0, queued_step=0, seen=0) with
        active_pos/frontier_pos set to -1, matching the SoA np.full(-1) init."""
        cdef Py_ssize_t i
        cdef uintptr_t base
        cdef np.ndarray[np.uint8_t, ndim=1] buf = np.zeros(capacity * sizeof(MonitorSlot) + 64, dtype=np.uint8)
        self._slots_buf = buf
        base = <uintptr_t>buf.data
        base = (base + 63) & ~<uintptr_t>63
        self._slots = <MonitorSlot*>base
        for i in range(capacity):
            self._slots[i].active_pos = -1
            self._slots[i].frontier_pos = -1

    cpdef reset(self):
        self._alloc_slots(self._capacity)
        self._size = 0
        self._active_count = 0
        self._frontier_count = 0
        self._next_frontier_count = 0
        self._status_count = 0
        self._anomaly_count = 0
        self._step_id = 0

    cdef inline Py_ssize_t _find_slot(self, MonitorSlot* slots, int64_t s1, int64_t s2,
                                      int64_t lag) noexcept nogil:
        cdef Py_ssize_t mask = self._capacity - 1
        cdef Py_ssize_t idx = <Py_ssize_t>(_monitor_hash_key(s1, s2, lag) & <uint64_t>mask)
        while slots[idx].occupied != 0:
            if slots[idx].s1 == s1 and slots[idx].s2 == s2 and slots[idx].lag == lag:
                return idx
            idx = (idx + 1) & mask
        return idx

    cdef void _rehash(self, Py_ssize_t new_capacity, bint keep_inactive):
        cdef object old_buf = self._slots_buf
        cdef MonitorSlot* old = self._slots
        cdef object old_frontier_obj = self._frontier_slots_arr
        cdef int64_t[:] old_frontier = old_frontier_obj
        cdef Py_ssize_t old_capacity = self._capacity
        cdef Py_ssize_t old_frontier_count = self._frontier_count
        cdef Py_ssize_t old_i, old_slot, idx, mask, pos
        cdef MonitorSlot* slots
        cdef int64_t[:] active_slots
        cdef int64_t[:] frontier_slots

        new_capacity = _monitor_next_power2(new_capacity)
        self._capacity = new_capacity
        self._alloc_slots(new_capacity)
        slots = self._slots
        self._active_slots_arr = np.empty(new_capacity, dtype=np.int64)
        self._frontier_slots_arr = np.empty(new_capacity, dtype=np.int64)
        self._next_frontier_slots_arr = np.empty(new_capacity, dtype=np.int64)
        active_slots = self._active_slots_arr
        frontier_slots = self._frontier_slots_arr
        mask = new_capacity - 1
        self._size = 0
        self._active_count = 0
        self._frontier_count = 0
        self._next_frontier_count = 0
        for old_i in range(old_capacity):
            if old[old_i].occupied == 0:
                continue
            if not keep_inactive and old[old_i].active == 0:
                continue
            idx = <Py_ssize_t>(_monitor_hash_key(old[old_i].s1, old[old_i].s2, old[old_i].lag) & <uint64_t>mask)
            while slots[idx].occupied != 0:
                idx = (idx + 1) & mask
            slots[idx].occupied = <uint8_t>1
            slots[idx].s1 = old[old_i].s1
            slots[idx].s2 = old[old_i].s2
            slots[idx].lag = old[old_i].lag
            slots[idx].t1 = old[old_i].t1
            slots[idx].t2 = old[old_i].t2
            slots[idx].window = old[old_i].window
            slots[idx].length = old[old_i].length
            slots[idx].sign = old[old_i].sign
            slots[idx].seen = old[old_i].seen
            self._size += 1
            if old[old_i].active != 0:
                slots[idx].active = <uint8_t>1
                active_slots[self._active_count] = <int64_t>idx
                slots[idx].active_pos = <int32_t>self._active_count
                self._active_count += 1

        for pos in range(old_frontier_count):
            old_slot = <Py_ssize_t>old_frontier[pos]
            if old_slot < 0 or old_slot >= old_capacity or old[old_slot].active == 0:
                continue
            idx = self._find_slot(slots, old[old_slot].s1, old[old_slot].s2, old[old_slot].lag)
            frontier_slots[self._frontier_count] = <int64_t>idx
            slots[idx].frontier_pos = <int32_t>self._frontier_count
            self._frontier_count += 1
        # old_buf is released when this frame returns

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
        self._status_ptr = <int64_t*>(<np.ndarray>new_rows).data
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
        self._anomaly_ptr = <int64_t*>(<np.ndarray>new_rows).data
        self._anomaly_capacity = new_cap

    cdef inline void _activate_slot(self, MonitorSlot* slots, Py_ssize_t slot,
                                    int64_t[:] active_slots) noexcept nogil:
        if slots[slot].active != 0:
            return
        slots[slot].active = <uint8_t>1
        active_slots[self._active_count] = <int64_t>slot
        slots[slot].active_pos = <int32_t>self._active_count
        self._active_count += 1

    cdef inline void _deactivate_slot(self, MonitorSlot* slots, Py_ssize_t slot,
                                      int64_t[:] active_slots) noexcept nogil:
        cdef Py_ssize_t pos
        cdef Py_ssize_t last_pos
        cdef Py_ssize_t last_slot
        if slots[slot].active == 0:
            return
        pos = <Py_ssize_t>slots[slot].active_pos
        last_pos = self._active_count - 1
        last_slot = <Py_ssize_t>active_slots[last_pos]
        if pos != last_pos:
            active_slots[pos] = <int64_t>last_slot
            slots[last_slot].active_pos = <int32_t>pos
        slots[slot].active_pos = -1
        slots[slot].active = <uint8_t>0
        self._active_count -= 1

    cdef inline void _mark_current_frontier(self, MonitorSlot* slots, Py_ssize_t slot,
                                            int64_t[:] next_frontier) noexcept nogil:
        if slots[slot].queued_step == <int32_t>self._step_id:
            return
        slots[slot].queued_step = <int32_t>self._step_id
        next_frontier[self._next_frontier_count] = <int64_t>slot
        self._next_frontier_count += 1

    cdef inline bint _remove_previous_frontier_slot(self, MonitorSlot* slots, Py_ssize_t slot,
                                                    int64_t[:] frontier_slots) noexcept nogil:
        cdef Py_ssize_t pos
        cdef Py_ssize_t last_pos
        cdef Py_ssize_t last_slot
        if slots[slot].frontier_pos < 0:
            return False
        pos = <Py_ssize_t>slots[slot].frontier_pos
        last_pos = self._frontier_count - 1
        last_slot = <Py_ssize_t>frontier_slots[last_pos]
        if pos != last_pos:
            frontier_slots[pos] = <int64_t>last_slot
            slots[last_slot].frontier_pos = <int32_t>pos
        slots[slot].frontier_pos = -1
        self._frontier_count -= 1
        return True

    cdef inline void _append_status(self, MonitorSlot* slots, Py_ssize_t slot):
        cdef int64_t* row
        if slots[slot].active == 0:
            return
        if self._status_count + 1 > self._status_capacity:
            self._ensure_status_capacity(self._status_count + 1)
        row = self._status_ptr + self._status_count * 7
        row[0] = slots[slot].s1
        row[1] = slots[slot].s2
        row[2] = slots[slot].lag
        row[3] = slots[slot].t1
        row[4] = slots[slot].t2
        row[5] = slots[slot].length
        row[6] = slots[slot].sign
        self._status_count += 1

    cdef inline void _append_anomaly(self, int64_t s1, int64_t s2, int64_t lag, int64_t time_value, int64_t marker):
        cdef int64_t* row
        if self._anomaly_count + 1 > self._anomaly_capacity:
            self._ensure_anomaly_capacity(self._anomaly_count + 1)
        row = self._anomaly_ptr + self._anomaly_count * 5
        row[0] = s1
        row[1] = s2
        row[2] = lag
        row[3] = time_value
        row[4] = marker
        self._anomaly_count += 1

    cpdef update(self,
                 long[:, ::1] rows,
                 double[:] corrs,
                 int window_step,
                 bint save_status,
                 bint save_anomalies):
        cdef Py_ssize_t n_rows = rows.shape[0]
        cdef Py_ssize_t n_corrs = corrs.shape[0]
        cdef Py_ssize_t i, j, slot_i, pf, mask
        cdef int64_t sid1, sid2, key_s1, key_s2
        cdef int64_t t1, t2, key_t1, key_t2, window_size
        cdef int64_t min_time, max_time, lag, corr_sign
        cdef int64_t first_t1, first_t2, last_window_size, last_corr_length, last_corr_sign
        cdef int64_t curr_time, next_corr_time, out_time
        cdef int64_t vmax, v
        cdef double corr
        cdef MonitorSlot* slots
        cdef MonitorSlot* sp
        cdef int64_t[:] frontier_slots
        cdef int64_t[:] active_slots
        cdef int64_t[:] next_frontier
        cdef object swap_obj
        cdef Py_ssize_t scan_pos
        cdef double _tp0

        # int32 range guard (sequential scan of the input, ~5 loads/row).
        vmax = 0
        with nogil:
            for i in range(n_rows):
                for j in range(5):
                    v = <int64_t>rows[i, j]
                    if v > vmax:
                        vmax = v
        if vmax >= MON_INT32_LIMIT or self._step_id + 1 >= MON_INT32_LIMIT:
            raise ValueError("NumericMonitorState: series id / time index / window size / step "
                             "count must stay below 2**30 (got %d)" % max(vmax, self._step_id + 1))

        _tp0 = time.perf_counter()
        self._maybe_compact()
        self._ensure_hash_capacity(self._size + n_rows)
        self._t_capacity += time.perf_counter() - _tp0
        self._step_id += 1
        self._next_frontier_count = 0
        _tp0 = time.perf_counter()
        slots = self._slots
        active_slots = self._active_slots_arr
        next_frontier = self._next_frontier_slots_arr
        frontier_slots = self._frontier_slots_arr
        pf = self._prefetch_dist
        mask = self._capacity - 1
        for i in range(n_rows):
            if pf > 0 and i + pf < n_rows:
                sid1 = <int64_t>rows[i + pf, 0]
                sid2 = <int64_t>rows[i + pf, 1]
                if sid1 >= 0 and sid2 >= 0:
                    t1 = <int64_t>rows[i + pf, 2]
                    t2 = <int64_t>rows[i + pf, 3]
                    lag = t1 - t2 if t1 >= t2 else t2 - t1
                    if sid1 <= sid2:
                        key_s1 = sid1
                        key_s2 = sid2
                    else:
                        key_s1 = sid2
                        key_s2 = sid1
                    MON_PREFETCH_W(&slots[<Py_ssize_t>(_monitor_hash_key(key_s1, key_s2, lag) & <uint64_t>mask)])
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

            slot_i = self._find_slot(slots, key_s1, key_s2, lag)
            sp = &slots[slot_i]

            if sp.occupied == 0:
                sp.occupied = <uint8_t>1
                sp.s1 = <int32_t>key_s1
                sp.s2 = <int32_t>key_s2
                sp.lag = <int32_t>lag
                self._size += 1

            self._remove_previous_frontier_slot(slots, slot_i, frontier_slots)
            if sp.active == 0:
                self._activate_slot(slots, slot_i, active_slots)
                sp.t1 = key_t1
                sp.t2 = key_t2
                sp.window = <int32_t>window_size
                sp.length = <int32_t>window_size
                sp.sign = <int8_t>corr_sign
                sp.seen = <int32_t>self._step_id
                self._mark_current_frontier(slots, slot_i, next_frontier)
                if save_anomalies:
                    self._append_anomaly(key_s1, key_s2, lag, min_time, 1)
                self._rows_new_activation += 1
                continue

            self._mark_current_frontier(slots, slot_i, next_frontier)
            first_t1 = sp.t1
            first_t2 = sp.t2
            last_window_size = sp.window
            last_corr_length = sp.length
            last_corr_sign = sp.sign
            curr_time = max_time + window_size
            if first_t1 >= first_t2:
                next_corr_time = first_t1 + last_corr_length + window_step
            else:
                next_corr_time = first_t2 + last_corr_length + window_step

            if curr_time < next_corr_time:
                sp.seen = <int32_t>self._step_id
                self._rows_early_unchanged += 1
                continue
            if curr_time == next_corr_time and window_size == last_window_size and corr_sign == last_corr_sign:
                sp.length = <int32_t>(last_corr_length + window_step)
                sp.seen = <int32_t>self._step_id
                self._rows_extend += 1
                continue

            if save_status:
                self._append_status(slots, slot_i)
            sp.t1 = key_t1
            sp.t2 = key_t2
            sp.window = <int32_t>window_size
            sp.length = <int32_t>window_size
            sp.sign = <int8_t>corr_sign
            sp.seen = <int32_t>self._step_id
            if save_anomalies:
                self._append_anomaly(key_s1, key_s2, lag, min_time, 1 if corr_sign == last_corr_sign else 0)
            self._rows_transition += 1

        self._t_row_loop += time.perf_counter() - _tp0

        _tp0 = time.perf_counter()
        while self._frontier_count > 0:
            slot_i = <Py_ssize_t>frontier_slots[self._frontier_count - 1]
            sp = &slots[slot_i]
            min_time = sp.t1 if sp.t1 <= sp.t2 else sp.t2
            out_time = min_time + sp.length - (sp.window - window_step)
            if save_anomalies:
                self._append_anomaly(sp.s1, sp.s2, sp.lag, out_time, -1)
            if save_status:
                self._append_status(slots, slot_i)
            self._remove_previous_frontier_slot(slots, slot_i, frontier_slots)
            self._deactivate_slot(slots, slot_i, active_slots)
        self._t_closeout += time.perf_counter() - _tp0
        _tp0 = time.perf_counter()
        swap_obj = self._frontier_slots_arr
        self._frontier_slots_arr = self._next_frontier_slots_arr
        self._next_frontier_slots_arr = swap_obj
        self._frontier_count = self._next_frontier_count
        self._next_frontier_count = 0
        frontier_slots = self._frontier_slots_arr
        for scan_pos in range(self._frontier_count):
            slots[<Py_ssize_t>frontier_slots[scan_pos]].frontier_pos = <int32_t>scan_pos
        self._t_swap += time.perf_counter() - _tp0
        self._profile_calls += 1

    cpdef finalize(self, bint save_status=True):
        cdef MonitorSlot* slots = self._slots
        cdef int64_t[:] active_slots = self._active_slots_arr
        cdef int64_t[:] frontier_slots = self._frontier_slots_arr
        cdef Py_ssize_t slot_i
        cdef Py_ssize_t pos
        for pos in range(self._frontier_count):
            slot_i = <Py_ssize_t>frontier_slots[pos]
            if slot_i >= 0 and slot_i < self._capacity:
                slots[slot_i].frontier_pos = -1
        while self._active_count > 0:
            slot_i = <Py_ssize_t>active_slots[self._active_count - 1]
            if save_status:
                self._append_status(slots, slot_i)
            self._deactivate_slot(slots, slot_i, active_slots)
        self._frontier_count = 0
        self._next_frontier_count = 0

    cpdef Py_ssize_t pending_status_count(self):
        return self._status_count

    cpdef Py_ssize_t pending_anomaly_count(self):
        return self._anomaly_count

    cpdef Py_ssize_t active_count(self):
        return self._active_count

    cpdef Py_ssize_t occupied_count(self):
        return self._size

    cpdef Py_ssize_t capacity(self):
        return self._capacity

    cpdef object profile_snapshot(self):
        return (
            self._t_capacity,
            self._t_row_loop,
            self._t_closeout,
            self._t_swap,
            self._profile_calls,
        )

    cpdef object row_branch_snapshot(self):
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
