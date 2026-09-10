"""Indexing backend: grid (buckets per cell)."""

import math
from collections import defaultdict

from .base import SketchIndex


class GridIndex(SketchIndex):
    def __init__(self, cell_size, dims=None, offset=None):
        self.cell_size = cell_size
        self.dims = dims                  # indices of the coords used (None = all)
        self.offset = offset              # shift of the cell boundaries (or None)
        self.buckets = defaultdict(list)  # cell -> [keys]
        self.cell_of = {}                 # key  -> cell

    def _cell(self, vec):
        dims = self.dims if self.dims is not None else range(len(vec))
        if self.offset is None:
            return tuple(int(math.floor(vec[d] / self.cell_size)) for d in dims)
        return tuple(int(math.floor((vec[d] - o) / self.cell_size))
                     for d, o in zip(dims, self.offset))

    def insert(self, key, vec):
        cell = self._cell(vec)
        self.buckets[cell].append(key)
        self.cell_of[key] = cell

    def query(self, vec):
        # Skeleton: exact cell only.
        # TODO: extend to the neighbouring cells (±1 per axis) for recall.
        return list(self.buckets.get(self._cell(vec), []))

    def remove(self, key):
        cell = self.cell_of.pop(key, None)
        if cell is not None and key in self.buckets.get(cell, []):
            self.buckets[cell].remove(key)
