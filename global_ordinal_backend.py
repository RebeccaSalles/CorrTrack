"""Shared cosine/L2-distance-threshold utility, originally introduced
alongside GlobalOrdinalTransformer (ported from corrtrack_release_nonlinear,
2026-07-28).

(2026-07-31) GlobalOrdinalTransformer itself (and its supporting
sample_pair_indices/derive_num_pairs_for_recall helpers) was removed --
incremental_concordance_multichannel (a per-gap union tier-1 over
ConcordanceSketchState, library_corrtrack_parallel.py/concordance_sketch.py)
was found, via direct real-data benchmarking, to strictly dominate
global_ordinal_comparisons' recall at every window_size tested (256/1024/
2048/4096: 97.7%/100%/100%/100% vs. 83.0%/99.1%/98.2%/93.2%), precision
100% throughout -- see docs/implementation_log.md's 2026-07-31 entry.
(2026-07-31, later) Single-vector concordance (its last caller in
library_corrtrack_parallel.py) was also removed, so ordinal_distance_threshold
is now unused there. Left in place as a small, harmless standalone utility
rather than deleted outright.
"""

import math


def ordinal_distance_threshold(candidate_tau=0.5, dist_tol=0.0):
    tau = float(candidate_tau)
    tol = float(dist_tol or 0.0)
    return math.sqrt(max(2.0 * (1.0 - tau), 0.0)) + tol
