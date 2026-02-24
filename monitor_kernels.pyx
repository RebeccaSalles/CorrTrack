# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cpdef dict monitor_step(object validated_items,
                        object correlated,
                        object corr_lengths,
                        object corr_anomalies,
                        object previous_correlations,
                        int window_step):
    """
    Update monitoring state in one compiled pass.

    Parameters mirror CorrTrack monitoring dictionaries:
      - validated_items: iterable of (pair, corr)
      - correlated: dict[pair] -> corr
      - corr_lengths: dict[key] -> list[[t1, t2, window_size, length, sign], ...]
      - corr_anomalies: dict[key] -> list[(time, marker), ...]
      - previous_correlations: dict[key] -> last status for previous step
      - window_step: CorrTrack step size

    Returns:
      - new_in dict used to replace previous_correlations
    """
    cdef dict new_in = {}
    cdef object pair, key, key2, corr_hist, last_corr, status, corr_val
    cdef long t1, t2, window_size, corr_lag
    cdef long corr_sign, first_t1, first_t2, last_window_size, last_corr_length, last_corr_sign
    cdef long curr_time, next_corr_time
    cdef long min_time, max_time, out_time
    cdef Py_ssize_t hist_last_idx

    for pair, corr_val in validated_items:
        t1 = <long>pair[2]
        t2 = <long>pair[3]
        window_size = <long>pair[4]
        min_time = t1 if t1 <= t2 else t2
        max_time = t1 if t1 >= t2 else t2
        corr_lag = max_time - min_time
        corr_sign = 1 if float(correlated[pair]) >= 0.0 else -1

        key = (pair[0], pair[1], corr_lag)
        key2 = (pair[1], pair[0], corr_lag)

        if key not in corr_lengths and key2 not in corr_lengths:
            corr_lengths[key] = [[t1, t2, window_size, window_size, corr_sign]]
            corr_anomalies[key] = [(min_time, 1)]
        else:
            if key2 in corr_lengths:
                key = key2
            corr_hist = corr_lengths[key]
            hist_last_idx = len(corr_hist) - 1
            last_corr = corr_hist[hist_last_idx]
            first_t1 = <long>last_corr[0]
            first_t2 = <long>last_corr[1]
            last_window_size = <long>last_corr[2]
            last_corr_length = <long>last_corr[3]
            last_corr_sign = <long>last_corr[4]

            curr_time = max_time + window_size
            if first_t1 >= first_t2:
                next_corr_time = first_t1 + last_corr_length + window_step
            else:
                next_corr_time = first_t2 + last_corr_length + window_step

            if curr_time < next_corr_time:
                continue
            elif curr_time == next_corr_time and window_size == last_window_size and corr_sign == last_corr_sign:
                last_corr[3] = last_corr_length + window_step
            else:
                corr_hist.append([t1, t2, window_size, window_size, corr_sign])
                if corr_sign == last_corr_sign:
                    corr_anomalies[key].append((min_time, 1))
                else:
                    corr_anomalies[key].append((min_time, 0))

        corr_hist = corr_lengths[key]
        hist_last_idx = len(corr_hist) - 1
        new_in[key] = corr_hist[hist_last_idx]
        if key in previous_correlations:
            del previous_correlations[key]

    for key, status in previous_correlations.items():
        t1 = <long>status[0]
        t2 = <long>status[1]
        window_size = <long>status[2]
        last_corr_length = <long>status[3]
        min_time = t1 if t1 <= t2 else t2
        out_time = min_time + last_corr_length - (window_size - window_step)
        if key not in corr_anomalies:
            corr_anomalies[key] = []
        corr_anomalies[key].append((int(out_time), -1))

    return new_in
