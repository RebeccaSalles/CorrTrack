"""Pipeline parameters.

Resolution by decreasing priority:
    call argument  >  environment variable (CORRTRACK_*)  >  default.
E.g.: ``CORRTRACK_WINDOW_SIZE=48``.
"""

import os
from dataclasses import dataclass, fields

ENV_PREFIX = "CORRTRACK_"


@dataclass
class Config:
    # --- data selection (filtering at read time) ---
    n_series: int = 0              # keep only the first N series (0 = all)
    n_years: int = 0               # keep only N years / rows of obs (0 = everything)
    obs_mode: str = "years"        # "years": last N×365×24 obs; "count": first N rows
    train_ratio: float = 1.0       # fraction (time prefix) used for the sweep (v2.optimize)

    # --- hyperparameter search (v2.optimize optimizer; best-config selection) ---
    target_recall: float = 0.95            # target recall
    target_precision: float = 0.0          # minimum precision (0 = disabled)
    recall_fallback_near_ratio: float = 0.98  # fallback: recall >= ratio × best recall reached
    speedup_near_ratio: float = 0.98          # near-best speedup band before the tie-break
    sweep_validate: bool = True               # False: skip the validate phase (Pearson) on the corrtrack
                                              # side during the sweep — prediction = post-sketch pairs (fast,
                                              # but precision/val_time/speedup no longer reflect the full pipeline)

    # --- windowing (defaults aligned with v1) ---
    window_size: int = 7 * 24      # length of a sub-window (1 week in hours)
    window_step: int = 12          # shift between two consecutive windows
    n_lags: int = 7 * 24 * 100     # max lag horizon to pair two windows

    # --- sketch ---
    n_vectors: int = 16            # sketch dimension (number of projections OR FFT coeffs)
    seed: int = 2468               # seed of the random projections (random_projection)
    basic_window: int = 0          # >0: incremental sketch over sub-windows (divides window_size)
    sketch_method: str = "random_projection"
    # sketch computation method:
    #   random_projection (default, v1 port) — Gaussian N(0,1) matrix × window
    #   fft_lowpass                          — first n_vectors FFT magnitudes
    #   fft_topk                             — n_vectors largest FFT magnitudes
    #   median_blocks                        — median PAA: n_vectors contiguous blocks
    #   median_phase                         — "folded" median: aggregates per phase

    # --- indexing ---
    index_backend: str = "grid"    # see steps/index/__init__.py:INDEXES (13 options)
    # universal: query radius (used by tree/kdtree/quadtree/bptree/bst/
    # octree/bptree3d/bst3d/vptree/annoy/hnsw — not by grid/knn).
    query_radius: float = 0.5
    # grid (LSH multi-grid) — `grid_` prefix throughout.
    grid_cell: float = 0.5         # size of an LSH cell (= former cell_size)
    grid_n_tables: int = 8         # number of shifted grids = number of LSH hash tables
    grid_n_coords: int = 2         # number of sketch coords used per grid (keep it low!)
    grid_vote_min: int = 1         # multi-grid AND vote threshold (>=1 = union)
    # quadtree: dimension of the JL sketch→quadtree projection (≠ grid_n_coords)
    quadtree_proj_dim: int = 2
    # knn / hnsw (kNN mode): number of neighbours returned
    n_neighbors: int = 32
    # bptree / bst (and 3D): key extraction from the sketch
    key_mode: str = "truncate"     # truncate (=sketch[0]) | median | sum | abs_sum (L1 energy) | random (sketch-of-sketch)
    key_proj_dim: int = 1          # projection dimension (random mode only)
    # hnsw: hierarchical graph
    hnsw_m: int = 16               # max connections / level > 0 (2·M at level 0)
    hnsw_ef_c: int = 200           # ef size during construction
    hnsw_ef_s: int = 50            # ef size during the query (recall ↔ speed)
    # annoy: forest of random-projection trees
    annoy_n_trees: int = 8         # number of independent trees (= former n_grids for annoy)
    annoy_leaf_size: int = 32      # leaf size (linear scan below it)

    # --- thresholds ---
    sketch_threshold: float = 0.7  # minimum cosine similarity (sketch level)
    corr_threshold: float = 0.8    # minimum Pearson correlation (exact validation; v1)
    neg_corr: bool = True          # also accept negative correlations (|corr| >= threshold; v1)
    std_threshold: float = 0.0     # skip near-constant windows (std < threshold); v1 ≈ 1e-3; 0 = off

    # --- compute backend (python/vectorized/parallel/cython/mps/cuda swap point) ---
    backend: str = "python"        # default backend, shared by the 3 phases
    sketch_backend: str = ""       # sketch phase (projection); "" -> inherits backend
    candidate_backend: str = ""    # candidate phase (index/cosine); "" -> inherits backend
    validate_backend: str = ""     # validation phase (Pearson); "" -> inherits backend
    workers: int = 0               # number of workers (sharded threads OR parallel processes; 0 = auto)
    cores: str = "auto"            # macOS: "perf" (P-cores) | "eco" (E-cores) | "auto"
    # `sharded_*` backend: strategy of the select+validate block (see core/_static_shard.py)
    shard_mode: str = "split"      # "split": select //windows then validate over a balanced
                                   #          PAIR bundle (best speedup, materializes the pairs).
                                   # "fused": select+validate fused //windows (memory bounded
                                   #          to one window) — the alternative.

    def backend_for(self, phase):
        """Backend resolved for a phase ('sketch'|'candidate'|'validate')."""
        return getattr(self, f"{phase}_backend") or self.backend

    # --- output ---
    output: str = "results"        # output directory (CSV + logs); empty = no writing
    clean: bool = False            # delete the output directory at startup (clean run)
    profile: bool = False          # print the per-phase timing table (bottlenecks)

    # --- logging ---
    log_level: str = "info"        # debug | info | warning | error
    log_file: str = ""             # path of a .log; empty = <output>/run.log (else console only)
    log_every: float = 5.0         # interval (s) between two progress checkpoints

    @classmethod
    def build(cls, **overrides):
        """Build a Config: defaults < ENV < overrides (None ignored)."""
        cfg = cls()
        for f in fields(cls):
            if overrides.get(f.name) is not None:
                value = overrides[f.name]
            else:
                env = os.environ.get(ENV_PREFIX + f.name.upper())
                if env is None:
                    continue
                value = env
            setattr(cfg, f.name, _coerce(value, f.type))
        return cfg


def _coerce(value, type_):
    if type_ in (bool, "bool"):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
    if type_ in (int, "int"):
        return int(value)
    if type_ in (float, "float"):
        return float(value)
    return str(value)
