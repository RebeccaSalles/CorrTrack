"""**Octree** index (3D) — recursive subdivision into 8 octants.

3D variant of the quadtree. Since the sketch is high-dimensional (n_vectors),
it is projected into 3D through a **random projection** ("sketch of sketch"),
far better than truncation: Johnson-Lindenstrauss → distances better preserved.

Subdivision: a node (cube of half-width hw centred at (cx,cy,cz)) splits into
**8 sub-cubes** (octants) indexed by (x>cx, y>cy, z>cz) when its capacity
(`_CAP = 8`) is exceeded. Radius query: recursive descent into the octants that
intersect the sphere of radius `tree_radius` around the query.
"""

import numpy as np

from ._projection import make_random_projection
from .base import SketchIndex

_CAP = 8


class _ONode:
    __slots__ = ("cx", "cy", "cz", "hw", "items", "kids")

    def __init__(self, cx, cy, cz, hw):
        self.cx, self.cy, self.cz, self.hw = cx, cy, cz, hw  # centre + half-width
        self.items = []                                       # [(key, x, y, z)] if leaf
        self.kids = None                                      # 8 child nodes


class OctreeIndex(SketchIndex):
    def __init__(self, radius, backend=None, n_vectors=0, seed=0):
        self.radius = radius
        self.backend = backend
        # 3D random projection: mandatory (an octree is 3D by nature).
        # `make_random_projection` returns a (3, n_vectors) ~ N(0, 1/3).
        self._proj = make_random_projection(3, n_vectors or 1, seed)
        self.points = {}      # key -> projected (x, y, z)
        self._dirty = True
        self._root = None

    def _to3(self, vec):
        v = np.asarray(vec, dtype=float).ravel()
        if v.shape[0] != self._proj.shape[1]:                  # changing n_vectors
            self._proj = make_random_projection(3, v.shape[0], 0)
        p = self._proj @ v
        return float(p[0]), float(p[1]), float(p[2])

    def insert(self, key, vec):
        self.points[key] = self._to3(vec)
        self._dirty = True

    def remove(self, key):
        if self.points.pop(key, None) is not None:
            self._dirty = True

    def _build(self):
        if not self.points:
            self._root = None
            self._dirty = False
            return
        coords = list(self.points.values())
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]; zs = [c[2] for c in coords]
        cx = 0.5 * (min(xs) + max(xs)); cy = 0.5 * (min(ys) + max(ys)); cz = 0.5 * (min(zs) + max(zs))
        hw = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2 + 1e-9
        self._root = _ONode(cx, cy, cz, hw)
        for key, (x, y, z) in self.points.items():
            self._insert_node(self._root, key, x, y, z)
        self._dirty = False

    def _insert_node(self, node, key, x, y, z):
        if node.kids is None:
            node.items.append((key, x, y, z))
            if len(node.items) > _CAP and node.hw > 1e-6:
                items = node.items
                node.items = []
                hw2 = node.hw / 2
                node.kids = [_ONode(node.cx + sx * hw2, node.cy + sy * hw2,
                                    node.cz + sz * hw2, hw2)
                             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                for k, ix, iy, iz in items:
                    self._insert_node(self._pick(node, ix, iy, iz), k, ix, iy, iz)
            return
        self._insert_node(self._pick(node, x, y, z), key, x, y, z)

    def _pick(self, node, x, y, z):
        idx = (1 if x > node.cx else 0) * 4 + (1 if y > node.cy else 0) * 2 + (1 if z > node.cz else 0)
        return node.kids[idx]

    def query(self, vec):
        if self._dirty:
            self._build()
        if self._root is None:
            return []
        qx, qy, qz = self._to3(vec)
        q = np.array([qx, qy, qz])
        # non-python backend -> batched linear scan (identical result).
        if self.backend is not None and getattr(self.backend, "name", "python") != "python":
            keys = list(self.points)
            P = np.asarray([self.points[k] for k in keys], dtype=float)
            d = self.backend.distance_batch(q, P)
            return [k for k, ok in zip(keys, d <= self.radius) if ok]
        # octree descent: pruning cube vs sphere
        r = self.radius
        r2 = r * r
        out = []
        stack = [self._root]
        while stack:
            n = stack.pop()
            if (abs(qx - n.cx) > n.hw + r or abs(qy - n.cy) > n.hw + r
                    or abs(qz - n.cz) > n.hw + r):
                continue
            if n.kids is None:
                for key, x, y, z in n.items:
                    dx, dy, dz = x - qx, y - qy, z - qz
                    if dx * dx + dy * dy + dz * dz <= r2:
                        out.append(key)
            else:
                stack.extend(n.kids)
        return out

    def query_batch(self, queries):
        if not self.points:
            return [[] for _ in queries]
        keys = list(self.points)
        P = np.asarray([self.points[k] for k in keys], dtype=float)  # (N, 3)
        Q3 = np.asarray([self._to3(q) for q in queries], dtype=float)  # projection 3D
        if Q3.ndim == 1:
            Q3 = Q3[None, :]
        D = self.backend.cdist_batch(Q3, P) if self.backend is not None \
            else np.sqrt(((Q3[:, None] - P[None, :]) ** 2).sum(axis=2))
        return [[keys[j] for j, ok in enumerate(D[i] <= self.radius) if ok]
                for i in range(D.shape[0])]
