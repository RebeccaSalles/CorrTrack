"""Sharded execution with a **static partition** (backend alias `sharded_<base>`).

`workers` threads, each in charge of a data BUNDLE fixed up front doing its
share of the job — NO per-window re-dispatch, NO nested pool. Two distinct
blocks, each partitioned into per-thread bundles:

  Block 1 — sketch + index:
    * sketch : parallel over a WINDOW BUNDLE (1 sketcher/worker, cache not shared).
    * index  : serial merge (store + array-native matrices + O(N) bulk_load).

  Block 2 — select + validate (strategy set through `config.shard_mode`):
    * "split" (default, `_sv_split`): TWO sub-phases, each with its own bundle —
        select   parallel over a WINDOW BUNDLE → gathers every pair (IA/IB),
        validate parallel over a balanced PAIR BUNDLE → array-native cosine+Pearson.
        Balancing by pair count (rather than by window) → better speedup.
    * "fused" (alternative, `_sv_fused`): select+validate FUSED, parallel over a
        window bundle, memory bounded to one window (no global materialization
        of the pairs) — safe on dense data (ASOS, large n_lags).

Optimized data access: each key gets an integer id; sketches and raws live in
contiguous `SK`/`RAW` matrices; pairs are integer arrays `(ia, ib)`; the gather
uses fancy indexing (`SK[ia]`) instead of `store[key]["raw"]` + `np.stack`
(≈7× on the validation).

Correctness: the `causal` filter (the latest window owns the pair) reproduces
EXACTLY the result AND the counters of the streaming path — checked on
5/25/100 series, and identical between "split" and "fused". Trade-off: full
index + matrices in RAM (no eviction) → memory guard on the pipeline side;
otherwise fall back to serial streaming. "split" ALSO materializes every
candidate pair (2×int64/pair); "fused" stays bounded to one window.
"""

import os
import threading
import time

import numpy as np

from ..steps import selection
from ..steps.index import make_index
from ..steps.sketch import make_sketcher

# `split`: chunk size (in pairs) for the validate compute. Bounds the
# `SK[ia]`/`RAW[ia]` gather: without it, a bundle of millions of pairs
# materializes (N_pairs × n_vec) and (N_pairs × window) matrices → OOM (e.g.
# 25M pairs, n_vectors=32, window=168 → tens of GB). 250k pairs ≈ <1 GB
# transient per worker, large enough for efficient numpy batches. Overridable
# through the env.
_VALIDATE_CHUNK = max(1, int(os.environ.get("CORRTRACK_SHARD_VALIDATE_CHUNK", "250000")))


def _add_timing(timings, name, secs, calls, n_in=None, n_out=None):
    """Record a synthetic span into `Timings` (fused phase → the wall time is
    split between select/validate afterwards).

    In sharded mode there is NO per-window call: a SINGLE sample
    `(secs, n_in, n_out)` is pushed (the whole phase) — `phase_metrics()` will
    therefore report min=max=median for that step, which is consistent (a single
    observed span) while keeping the effective input/output throughputs."""
    if name not in timings.totals:
        timings.totals[name] = 0.0
        timings.counts[name] = 0
        timings.samples[name] = []
        timings._order.append(name)
    timings.totals[name] += secs
    timings.counts[name] += calls
    timings.samples[name].append((secs, n_in, n_out))


def _ranges(n_items, n_parts):
    """`n_parts` contiguous (balanced) slices of `range(n_items)`."""
    size = max(1, (n_items + n_parts - 1) // n_parts)
    return [range(i, min(i + size, n_items)) for i in range(0, n_items, size)]


def estimated_bytes(n_series, n_windows, window_size, n_vectors):
    """Approximate RAM of the full matrices (raws + sketches, float64)."""
    return n_series * n_windows * (window_size + n_vectors) * 8


def _run_parallel(n_parts, fn):
    """Launch `n_parts` `fn(pi)` threads (one per slice) and wait for them."""
    if n_parts <= 1:
        fn(0)
        return
    threads = [threading.Thread(target=fn, args=(pi,), name=f"shard{pi}")
               for pi in range(n_parts)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()


def run(config, sketch_be, candidate_be, validate_be, ids, windows,
        n_workers, timings):
    """Run corrtrack with a window-parallel static partition (array-native).

    Returns a dict: correlated (list of (a, b, corr, lag)), per_window (list of
    (start_time, win_corr) in window order, for the monitor), n_candidates,
    n_tested.
    """
    nW = len(windows)
    n_workers = max(1, min(n_workers, nW))
    std_thr = config.std_threshold
    n_lags = config.n_lags
    sk_thr = config.sketch_threshold
    corr_thr = config.corr_threshold
    neg = config.neg_corr

    # --- building the FULL index + array-native matrices ---
    # ID SPACE: store and index are keyed by INTEGER ID (= row in SK/RAW), NOT by
    # the (sid, t) key. `select_candidates` then returns id pairs directly (zero
    # key→id lookup over the millions of candidates); only the survivors are
    # converted back to keys at assembly time. `keys_by_id` keeps the id →
    # (sid, t) mapping for the output/monitor.
    win_parts = _ranges(nW, n_workers)
    n_parts = len(win_parts)
    index = make_index(config, candidate_be)
    store = {}                  # integer id -> {"time", "sketch", "raw"}
    win_ids = [None] * nW       # w -> list of the ids of the window
    keys_by_id = []             # id -> key (sid, t)
    SK_list, RAW_list = [], []

    # sketch parallel over window slices (1 sketcher per worker -> cache not shared)
    sketchers = [make_sketcher(config, sketch_be) for _ in range(n_parts)]
    sk_parts = [None] * n_parts

    def do_sketch(pi):
        sker = sketchers[pi]
        local = []
        for w in win_parts[pi]:
            win = windows[w]
            block, t0, i0 = win.block, win.start_time, win.start_index
            entries = []
            for i, sid in enumerate(ids):
                if std_thr > 0.0 and float(block[i].std()) < std_thr:
                    continue
                entries.append(((sid, t0), sker.sketch(sid, i0, block[i]), block[i]))
            local.append((w, entries))
        sk_parts[pi] = local

    with timings.track("sketch") as rec:
        _run_parallel(n_parts, do_sketch)
        n_sk = sum(len(entries) for pi in range(n_parts)
                   for _w, entries in (sk_parts[pi] or []))
        rec.io(n_in=nW * len(ids), n_out=n_sk)

    # serial merge: store + array-native matrices + index bulk_load (O(N), no
    # insort); the insertion order fixes the ids (consistent SK/RAW/keys_by_id).
    with timings.track("index") as rec:
        items = []
        for pi in range(n_parts):
            for w, entries in sk_parts[pi]:
                cur = []
                for key, vec, raw in entries:
                    i = len(keys_by_id)              # integer id = SK/RAW row
                    store[i] = {"time": key[1], "raw": raw, "sketch": vec}
                    keys_by_id.append(key)
                    SK_list.append(vec)
                    RAW_list.append(raw)
                    items.append((i, vec))           # index keyed BY ID
                    cur.append(i)
                win_ids[w] = cur
        index.bulk_load(items)
        rec.io(n_in=len(keys_by_id), n_out=len(keys_by_id))
    SK = np.asarray(SK_list, dtype=float)
    RAW = np.asarray(RAW_list, dtype=float)
    del sk_parts, SK_list, RAW_list, items

    # --- select + validate BLOCK ---
    # Two ways of splitting into per-thread bundles (see the dedicated functions
    # below):
    #   "split" (default): select parallel over a WINDOW bundle, then validate
    #                      parallel over a balanced PAIR bundle → better speedup.
    #   "fused"          : select+validate fused, parallel over a window bundle
    #                      (memory bounded to one window) — the alternative.
    ctx = dict(timings=timings, validate_be=validate_be, index=index, store=store,
               win_ids=win_ids, keys_by_id=keys_by_id, SK=SK, RAW=RAW,
               win_parts=win_parts, windows=windows, nW=nW, n_workers=n_workers,
               n_lags=n_lags, sk_thr=sk_thr, corr_thr=corr_thr, neg=neg)
    mode = getattr(config, "shard_mode", "split") or "split"
    if mode == "fused":
        correlated, per_window_corr, n_candidates, n_tested = _sv_fused(**ctx)
    else:
        correlated, per_window_corr, n_candidates, n_tested = _sv_split(**ctx)

    per_window = [(windows[w].start_time, per_window_corr[w]) for w in range(nW)]
    return {
        "correlated": correlated,
        "per_window": per_window,
        "n_candidates": n_candidates,
        "n_tested": n_tested,
    }


def _validate_ids(validate_be, SK, RAW, ia, ib, sk_thr, corr_thr, neg):
    """Cosine (sketch filter) then Pearson over aligned id arrays.

    `ia`/`ib` index SK/RAW directly. Returns `(ia_ok, ib_ok, corr_ok,
    n_tested)`: survivors (ids) + correlation, and how many passed the cosine
    filter. 100% array-native (GIL released) → parallelizable per pair bundle.
    """
    cv = validate_be.cosine_rows(SK[ia], SK[ib])
    keep = (np.abs(cv) if neg else cv) >= sk_thr
    ia2, ib2 = ia[keep], ib[keep]
    n_tested = int(ia2.shape[0])
    pr = validate_be.correlate_rows(RAW[ia2], RAW[ib2])
    ok = np.isfinite(pr) & ((np.abs(pr) if neg else pr) >= corr_thr)
    return ia2[ok], ib2[ok], pr[ok], n_tested


def _record(ai, bi, cc, keys_by_id):
    """Build the result tuple in CANONICAL KEY ORDER (sid, t) — the same as the
    serial streaming path (`selection._order` over the keys), for a
    byte-identical result despite the internal work in id space."""
    ka, kb = keys_by_id[ai], keys_by_id[bi]
    if ka <= kb:
        return (ka, kb, cc, int(ka[1] - kb[1]))
    return (kb, ka, cc, int(kb[1] - ka[1]))


def _sv_fused(timings, validate_be, index, store, win_ids, keys_by_id,
              SK, RAW, win_parts, windows, nW, n_workers, n_lags, sk_thr, corr_thr, neg):
    """`shard_mode="fused"` alternative: select + validate FUSED, parallel over
    a window bundle.

    Each worker does, FOR EACH WINDOW of its slice: query+select THEN validate
    (array-native cosine+Pearson), accumulating ONLY the correlations. Memory
    bounded to ONE window — vs materializing every candidate pair globally,
    which explodes (OOM) on highly correlated data (ASOS, large n_lags: ~10⁸
    pairs). Sketches/raws are shared read-only (SK/RAW). Works in id space
    (select returns id pairs) → no key→id lookup. The wall time is split
    synthetically between select/validate in proportion to the thread-cumulated
    time (readable cand_t/val_t columns).
    """
    res = [None] * len(win_parts)
    sel_t = [0.0] * len(win_parts)
    val_t = [0.0] * len(win_parts)

    def worker(pi):
        n_cand = n_tested = 0
        per_w = {}
        ts = tv = 0.0
        for w in win_parts[pi]:
            t0 = time.perf_counter()
            pairs = selection.select_candidates(index, store, win_ids[w],
                                                n_lags, causal=True)
            n_cand += len(pairs)
            ts += time.perf_counter() - t0
            if not pairs:
                per_w[w] = []
                continue
            t0 = time.perf_counter()
            arr = np.asarray(pairs, np.int64)          # (k, 2) ids
            ia_ok, ib_ok, cor_ok, nt = _validate_ids(
                validate_be, SK, RAW, arr[:, 0], arr[:, 1], sk_thr, corr_thr, neg)
            n_tested += nt
            per_w[w] = [_record(ai, bi, cc, keys_by_id)
                        for ai, bi, cc in zip(ia_ok.tolist(), ib_ok.tolist(),
                                              cor_ok.tolist())]
            tv += time.perf_counter() - t0
        res[pi] = (n_cand, n_tested, per_w)
        sel_t[pi], val_t[pi] = ts, tv

    t_phase = time.perf_counter()
    _run_parallel(len(win_parts), worker)
    fused_wall = time.perf_counter() - t_phase

    n_candidates = n_tested = 0
    correlated = []
    per_window_corr = [[] for _ in range(nW)]
    for item in res:
        if item is None:
            continue
        nc, nt, per_w = item
        n_candidates += nc
        n_tested += nt
        for w, wc in per_w.items():
            per_window_corr[w] = wc
            correlated.extend(wc)

    tot_s, tot_v = sum(sel_t), sum(val_t)
    tot = tot_s + tot_v
    n_queries = sum(len(win_ids[w] or ()) for w in range(nW))
    _add_timing(timings, "select", fused_wall * (tot_s / tot if tot else 1.0), nW,
                n_in=n_queries, n_out=n_candidates)
    _add_timing(timings, "validate", fused_wall * (tot_v / tot if tot else 0.0), nW,
                n_in=n_candidates, n_out=len(correlated))
    return correlated, per_window_corr, n_candidates, n_tested


def _sv_split(timings, validate_be, index, store, win_ids, keys_by_id,
              SK, RAW, win_parts, windows, nW, n_workers, n_lags, sk_thr, corr_thr, neg):
    """`shard_mode="split"` strategy (default): select and validate as TWO
    sub-phases, each with its own per-thread bundle.

      select   : parallel over a WINDOW BUNDLE (`win_parts`) → gathers EVERY
                 candidate pair into global ID arrays (IA, IB). In id space (the
                 index returns ids), hence NO key→id lookup over the millions of
                 candidates — that used to be the bottleneck.
      validate : parallel over a balanced PAIR BUNDLE (`n_workers` equal slices
                 of IA/IB) → array-native cosine + Pearson on large batches.
                 Splitting BY PAIR COUNT (rather than by window) removes the
                 straggler threads when a few windows concentrate the candidates
                 → better validate speedup.

    Each sub-phase is a real `timings.track` span (correct time + in/out
    throughput; the validate span measures ONLY the parallel compute, the tuple
    assembly happening afterwards).

    MEMORY (otherwise OOM on dense configs, e.g. 25M pairs):
      * select converts EACH window into an id array right away → never millions
        of live Python tuples at once; only the global `IA/IB` remains (compact,
        2×int64/pair);
      * validate processes each bundle IN CHUNKS of `_VALIDATE_CHUNK` pairs → the
        `SK[ia]`/`RAW[ia]` gather stays bounded (otherwise (N_pairs × window) GB).
    Vs `fused`, bounded to one window: `split` still holds the global `IA/IB` +
    the survivors. On dense configs, validate becomes memory-bandwidth bound
    (little gain beyond ~2-4 workers) and select stays GIL-bound (pure Python).
    """
    n_parts = len(win_parts)
    parts = [None] * n_parts

    def sel_worker(pi):
        # MEMORY: EACH window is converted into an id array right away and its
        # Python tuples are freed — instead of accumulating millions of
        # `(id, id)` tuples (≈80 B/tuple → several GB) before a global conversion.
        chunks = []
        for w in win_parts[pi]:
            pairs = selection.select_candidates(index, store, win_ids[w],
                                                n_lags, causal=True)
            if pairs:
                chunks.append(np.asarray(pairs, np.int64).reshape(-1, 2))
        parts[pi] = (np.concatenate(chunks) if chunks
                     else np.empty((0, 2), np.int64))

    with timings.track("select") as rec:
        _run_parallel(n_parts, sel_worker)
        arrs = [p for p in parts if p is not None and p.shape[0]]
        allp = np.concatenate(arrs) if arrs else np.empty((0, 2), np.int64)
        IA, IB = allp[:, 0], allp[:, 1]
        n_queries = sum(len(win_ids[w] or ()) for w in range(nW))
        rec.io(n_in=n_queries, n_out=int(IA.shape[0]))
    n_candidates = int(IA.shape[0])

    # --- VALIDATE: ONLY the parallel computation (cosine + Pearson) ---
    # The measured span contains ONLY the array-native compute (GIL released)
    # split into balanced pair bundles → scales cleanly with `workers`. Each
    # worker returns ARRAYS of ids+corr; the result-tuple assembly happens
    # AFTERWARDS, off the hot path ("the fastest possible validate").
    out_parts = []
    n_tested = n_corr = 0
    with timings.track("validate") as rec:
        if n_candidates:
            pair_parts = _ranges(n_candidates, n_workers)   # balanced PAIR bundles
            npp = len(pair_parts)
            out_parts = [None] * npp
            tested = [0] * npp

            def val_worker(pi):
                # pair bundle processed IN BOUNDED CHUNKS → bounded SK/RAW gather
                # (otherwise OOM on bundles of millions of pairs). Pair balancing
                # is preserved (each worker keeps its full slice).
                rng = pair_parts[pi]
                a_ok, b_ok, c_ok = [], [], []
                nt = 0
                for s in range(rng.start, rng.stop, _VALIDATE_CHUNK):
                    e = min(s + _VALIDATE_CHUNK, rng.stop)
                    xa, xb, xc, k = _validate_ids(
                        validate_be, SK, RAW, IA[s:e], IB[s:e], sk_thr, corr_thr, neg)
                    nt += k
                    if xa.shape[0]:
                        a_ok.append(xa); b_ok.append(xb); c_ok.append(xc)
                tested[pi] = nt
                if a_ok:
                    out_parts[pi] = (np.concatenate(a_ok), np.concatenate(b_ok),
                                     np.concatenate(c_ok))
                else:
                    z = np.empty(0, np.int64)
                    out_parts[pi] = (z, z, np.empty(0))

            _run_parallel(npp, val_worker)
            n_tested = sum(tested)
            n_corr = sum(int(p[0].shape[0]) for p in out_parts if p is not None)
        rec.io(n_in=n_candidates, n_out=n_corr)

    # --- result assembly (OUTSIDE the validate measurement: output + monitor) ---
    # VECTORIZED owner-window mapping: the latest window owns the pair (causal);
    # its time = max(t_a, t_b) ∈ wtimes (increasing) → `searchsorted` gives the
    # window index directly. The per-key orientation (see `_record`) is only paid
    # on the survivors (≪ candidates).
    correlated = []
    per_window_corr = [[] for _ in range(nW)]
    survivors = [p for p in out_parts if p is not None and p[0].shape[0]]
    if survivors:
        IAok = np.concatenate([p[0] for p in survivors])
        IBok = np.concatenate([p[1] for p in survivors])
        COR = np.concatenate([p[2] for p in survivors])
        TIME = np.fromiter((k[1] for k in keys_by_id), np.int64, len(keys_by_id))
        wtimes = np.fromiter((windows[w].start_time for w in range(nW)), np.int64, nW)
        owner = np.searchsorted(wtimes, np.maximum(TIME[IAok], TIME[IBok]))
        for ai, bi, cc, ow in zip(IAok.tolist(), IBok.tolist(), COR.tolist(),
                                  owner.tolist()):
            wrec = _record(ai, bi, cc, keys_by_id)
            correlated.append(wrec)
            per_window_corr[ow].append(wrec)
    return correlated, per_window_corr, n_candidates, n_tested
