"""HNSW — Hierarchical Navigable Small World (Malkov & Yashunin 2018).

State-of-the-art ANN (approximate nearest neighbors) in high dimension: used
by FAISS, hnswlib, qdrant. **Multi-layer graph** where each node gets a
geometrically drawn level; layer 0 contains every point, the upper layers get
sparser and sparser (spatial skip-list). The search navigates **top-down** from
the sparse layers to the dense one → approximately O(log N).

Parameters:
  * `M` (16 by default): max number of connections per node at levels > 0,
    `2·M` at level 0. Larger = better recall, more memory.
  * `ef_construction` (200): size of the candidate queue during insertion;
    larger = better graph quality, slower build.
  * `ef_search` (50): same during the query — directly sets the recall ↔ speed
    trade-off (larger = more recall, slower).
  * `tree_radius`: optional — when provided, every neighbour within the radius
    is returned (radius query on top of HNSW). Otherwise the `n_neighbors`
    nearest are returned (k-NN query).

Compact pure-Python implementation. For large volumes (>10⁵), prefer `hnswlib`
(C++) — this version aims at clarity and v2 portability.
"""

import heapq
import math
import random

import numpy as np

from .base import SketchIndex


class HNSWIndex(SketchIndex):
    def __init__(self, radius=None, backend=None, n_neighbors=32,
                 M=16, ef_construction=200, ef_search=50, seed=0):
        self.radius = float(radius) if radius is not None else None
        self.backend = backend
        self.k = max(1, int(n_neighbors))
        self.M = max(2, int(M))
        self.M0 = self.M * 2                            # level-0 connections
        self.ef_c = max(self.k, int(ef_construction))
        self.ef_s = max(self.k, int(ef_search))
        self._level_mult = 1.0 / math.log(self.M)
        self._rng = random.Random(int(seed))
        # data
        self._keys = []                                 # internal idx -> external key
        self._coords = []                               # internal idx -> sketch np.float32
        self._key2id = {}                               # external key -> internal idx
        self._levels = []                               # level of each node
        self._graph = []                                # graph[layer][id] -> set(neighbours)
        self._entry = None                              # entry point (id) au plus haut layer

    # -------- helpers ----------------------------------------------------
    def _dist(self, a, b):
        d = self._coords[a] - self._coords[b]
        return float(d @ d) ** 0.5

    def _dist_q(self, q, b):
        d = q - self._coords[b]
        return float(d @ d) ** 0.5

    def _rand_level(self):
        # niveau = floor(-log(uniform) * mult)
        return int(-math.log(self._rng.random() + 1e-12) * self._level_mult)

    def _ensure_layers(self, level):
        while len(self._graph) <= level:
            self._graph.append({})

    # -------- search greedy 1-NN au niveau `lc` (descent) ----------------
    def _greedy(self, q, entry, lc):
        cur = entry
        cur_d = self._dist_q(q, cur)
        improved = True
        while improved:
            improved = False
            for nb in self._graph[lc].get(cur, ()):
                d = self._dist_q(q, nb)
                if d < cur_d:
                    cur, cur_d = nb, d
                    improved = True
        return cur, cur_d

    # -------- search ef candidates at layer lc ---------------------------
    def _search_layer(self, q, entry_points, ef, lc):
        """Return a max-heap (by distance) of the `ef` best candidates."""
        visited = set(entry_points)
        # min-heap of the candidates left to explore
        candidates = [(self._dist_q(q, e), e) for e in entry_points]
        heapq.heapify(candidates)
        # max-heap (negated) of the results found
        results = [(-d, e) for d, e in candidates]
        heapq.heapify(results)
        while candidates:
            cur_d, cur = heapq.heappop(candidates)
            worst_d = -results[0][0] if results else math.inf
            if cur_d > worst_d:
                break
            for nb in self._graph[lc].get(cur, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist_q(q, nb)
                worst_d = -results[0][0] if results else math.inf
                if d < worst_d or len(results) < ef:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return results       # max-heap over (-d, id)

    def _select_neighbors(self, results, m):
        """Simple heuristic: keep the m nearest ones from the max-heap."""
        kept = sorted([(-d, nid) for d, nid in results])[:m]
        return [nid for _d, nid in kept]

    # -------- API SketchIndex --------------------------------------------
    def insert(self, key, vec):
        v = np.asarray(vec, dtype=np.float32).ravel()
        if key in self._key2id:
            self.remove(key)
        nid = len(self._keys)
        self._keys.append(key)
        self._coords.append(v)
        self._key2id[key] = nid
        L = self._rand_level()
        self._levels.append(L)
        self._ensure_layers(L)

        if self._entry is None:
            for lc in range(L + 1):
                self._graph[lc][nid] = set()
            self._entry = nid
            return

        # phase 1: greedy descent from the entry point down to layer L+1
        cur = self._entry
        max_layer = self._levels[self._entry]
        for lc in range(max_layer, L, -1):
            cur, _ = self._greedy(v, cur, lc)

        # phase 2: for each layer <= L, search ef_c and link
        eps = [cur]
        for lc in range(min(L, max_layer), -1, -1):
            results = self._search_layer(v, eps, self.ef_c, lc)
            m_max = self.M0 if lc == 0 else self.M
            nbrs = self._select_neighbors(results, m_max)
            self._graph[lc][nid] = set(nbrs)
            # bidirectional insertion + pruning
            for nb in nbrs:
                nb_set = self._graph[lc].setdefault(nb, set())
                nb_set.add(nid)
                if len(nb_set) > m_max:
                    # prune nb: keep only the m_max nearest to nb
                    nb_vec = self._coords[nb]
                    ranked = sorted(nb_set,
                                    key=lambda x, _v=nb_vec: float(((self._coords[x] - _v) ** 2).sum()))
                    self._graph[lc][nb] = set(ranked[:m_max])
            eps = nbrs or [cur]

        # update the entry point if the new node sits at the highest level
        if L > max_layer:
            self._entry = nid

    def remove(self, key):
        # "light" deletion: removed from the graphs, but the coords are kept
        # (the neighbours stay connected to the rest — degraded quality but fine
        # for the v2 sliding window, where the query skips the key through
        # select_candidates).
        nid = self._key2id.pop(key, None)
        if nid is None:
            return
        L = self._levels[nid]
        for lc in range(L + 1):
            for nb in self._graph[lc].pop(nid, ()):
                self._graph[lc].get(nb, set()).discard(nid)
        if self._entry == nid:
            self._entry = next(iter(self._graph[L]), None) if self._graph[L] else None

    # -------- query (k-NN ou rayon) --------------------------------------
    def _query_ids(self, vec):
        if self._entry is None:
            return [], np.empty(0)
        q = np.asarray(vec, dtype=np.float32).ravel()
        cur = self._entry
        # descent greedy
        for lc in range(self._levels[self._entry], 0, -1):
            cur, _ = self._greedy(q, cur, lc)
        ef = self.ef_s if self.radius is None else max(self.ef_s, self.k * 4)
        results = self._search_layer(q, [cur], ef, 0)
        # extraire et trier
        pairs = sorted([(-d, nid) for d, nid in results])
        if self.radius is not None:
            ids = [nid for d, nid in pairs if d <= self.radius]
        else:
            ids = [nid for _, nid in pairs[:self.k]]
        return ids, q

    def query(self, vec):
        ids, _ = self._query_ids(vec)
        return [self._keys[i] for i in ids]

    def query_batch(self, queries):
        # no natural GPU/BLAS batch for HNSW (sequential graph navigation)
        # → on boucle ; chaque query reste O(log N + ef)
        return [self.query(q) for q in queries]
