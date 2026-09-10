"""Sketch indexing — interchangeable backends + factory."""

from .base import SketchIndex
from .bptree import BPTreeIndex
from ._keyfn import make_key_fn, make_proj_fn
from .annoy import AnnoyIndex
from .bptree3d import BPTree3DIndex
from .bst import BSTIndex
from .bst3d import BST3DIndex
from .grid import GridIndex
from .hnsw import HNSWIndex
from .kdtree import KDTreeIndex
from .knn import KNNIndex
from .multigrid import MultiGridIndex
from .octree import OctreeIndex
from .quadtree import QuadTreeIndex
from .tree import TreeIndex
from .vptree import VPTreeIndex

INDEXES = ("grid", "tree", "kdtree", "quadtree", "bptree", "bst",
           "octree", "knn", "bptree3d", "bst3d",
           "hnsw", "vptree", "annoy")


def make_index(config, backend):
    """Pick the index structure according to `config.index_backend`.

    Each index uses a subset of Config (see INDEXES.md). In 2026 the names were
    clarified (prefixed by method when index-specific): universal
    `query_radius`; grid → `grid_*`; quadtree → `quadtree_proj_dim`; annoy →
    `annoy_*`; bptree/bst → `key_mode`/`key_proj_dim`; hnsw → `hnsw_*`.
    """
    if config.index_backend == "grid":
        return MultiGridIndex(
            config.grid_n_tables, config.grid_cell, config.grid_vote_min,
            config.grid_n_coords, config.n_vectors, config.seed,
        )
    if config.index_backend == "tree":
        return TreeIndex(config.query_radius, backend)
    if config.index_backend == "kdtree":
        return KDTreeIndex(config.query_radius, backend)
    if config.index_backend == "quadtree":
        return QuadTreeIndex(config.query_radius, backend, dims=config.quadtree_proj_dim,
                             n_vectors=config.n_vectors, seed=config.seed)
    if config.index_backend == "octree":
        return OctreeIndex(config.query_radius, backend,
                           n_vectors=config.n_vectors, seed=config.seed)
    if config.index_backend in ("bptree", "bst"):
        # configurable 1D key (truncate/median/sum/random) — see _keyfn.py
        key_fn = make_key_fn(config.key_mode, config.key_proj_dim,
                             config.n_vectors, config.seed)
        if config.index_backend == "bptree":
            return BPTreeIndex(config.query_radius, backend, key_fn=key_fn)
        return BSTIndex(config.query_radius, backend, key_fn=key_fn)
    if config.index_backend == "knn":
        return KNNIndex(config.n_neighbors, backend)
    if config.index_backend in ("bptree3d", "bst3d"):
        # configurable 3D projection: truncate (=sketch[:3]) | median | sum | random (JL)
        proj_fn = make_proj_fn(config.key_mode, 3, config.n_vectors, config.seed)
        if config.index_backend == "bptree3d":
            return BPTree3DIndex(config.query_radius, backend, proj_fn=proj_fn)
        return BST3DIndex(config.query_radius, backend, proj_fn=proj_fn)
    if config.index_backend == "hnsw":
        return HNSWIndex(radius=config.query_radius, backend=backend,
                         n_neighbors=config.n_neighbors,
                         M=config.hnsw_m, ef_construction=config.hnsw_ef_c,
                         ef_search=config.hnsw_ef_s, seed=config.seed)
    if config.index_backend == "vptree":
        return VPTreeIndex(config.query_radius, backend, seed=config.seed)
    if config.index_backend == "annoy":
        return AnnoyIndex(config.query_radius, backend,
                          n_trees=config.annoy_n_trees,
                          leaf_size=config.annoy_leaf_size,
                          seed=config.seed)
    raise ValueError(f"unknown index {config.index_backend!r} ({'|'.join(INDEXES)})")
