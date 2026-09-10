"""Step 4 — candidate selection + sketch-level validation.

  * select_candidates    : via the index (corrtrack mode).
  * enumerate_candidates : all pairs (brute-force mode).
  * validate_sketches    : cosine filter (delegated to the compute backend).

select_candidates and enumerate_candidates produce the SAME output format
-> feed the same validation. This is what makes corrtrack and bf
interchangeable downstream.
"""


def select_candidates(index, store, current_keys, n_lags, causal=False):
    """Match each current window with its neighbors in the index.

    Uses `index.query_batch` (1 call for ALL the queries of a window → 1 BLAS
    cdist instead of N passes, 1 GPU transfer instead of N for MPS). For the
    indexes without a `query_batch` override (e.g. `grid`), it automatically
    falls back to the sequential loop (default `SketchIndex.query_batch`).

    `causal`: when true, a pair is emitted only when the candidate sits at a
    time <= the query time. In streaming the index only contains the past, so
    each pair is naturally emitted once. In batch mode (FULL index, static
    window-parallel partition), the index contains past AND future; the causal
    filter makes the latest window "own" the pair → each pair emitted exactly
    once (counters == streaming, no double validation). Intra-window pairs
    (equal times) stay deduplicated by the `set` below.
    """
    pairs = set()
    sketches = [store[k]["sketch"] for k in current_keys]
    cands_per_key = index.query_batch(sketches)
    for key, cands in zip(current_keys, cands_per_key):
        kt = store[key]["time"]
        for cand in cands:
            if cand == key:
                continue
            # Skip candidates that have been evicted from `store` but are
            # still returned by the index. This mainly concerns HNSW (see
            # hnsw.py:remove — the "light" deletion leaves self._keys/_coords
            # intact to preserve the graph connectivity → some evicted keys are
            # still returned by a query).
            # Without this guard: KeyError on the next `store[cand]`.
            if cand not in store:
                continue
            ct = store[cand]["time"]
            if abs(kt - ct) > n_lags:
                continue
            if causal and ct > kt:   # the latest window owns the pair
                continue
            pairs.add(_order(key, cand))
    return list(pairs)


def enumerate_candidates(store, current_keys, n_lags):
    """Brute-force: match each current window against the whole buffer (no index)."""
    pairs = set()
    current = set(current_keys)
    for key in current_keys:
        for cand in store:
            if cand == key:
                continue
            if cand in current and cand < key:  # avoid intra-window duplicates
                continue
            if abs(store[key]["time"] - store[cand]["time"]) > n_lags:
                continue
            pairs.add(_order(key, cand))
    return list(pairs)


def validate_sketches(pairs, store, threshold, backend, neg_corr=False):
    """Keep the pairs whose sketches are similar enough (cosine).

    If neg_corr, compare on absolute value (anti-correlated sketches have a
    cosine close to -1).
    """
    if not pairs:
        return []
    uv = [(store[a]["sketch"], store[b]["sketch"]) for a, b in pairs]
    cs = backend.cosine_batch(uv)   # 1 numpy batch instead of one cosine per pair
    kept = []
    for (a, b), c in zip(pairs, cs):
        if (abs(c) if neg_corr else c) >= threshold:
            kept.append((a, b))
    return kept


def _order(a, b):
    """Canonical key of a pair (avoids both (a,b) and (b,a))."""
    return (a, b) if a <= b else (b, a)
