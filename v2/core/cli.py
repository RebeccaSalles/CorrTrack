"""Argument parsing shared by the corrtrack and bf launchers.

Every Config field is exposed as an option; when not provided -> ENV
(CORRTRACK_*) -> default. Guarantees that both scripts share the same options.
"""

import argparse
import os
import shutil
from dataclasses import fields

from ..backends import available
from .config import Config, _coerce
from .log import get_logger
from .pipeline import steps_for


def build_parser(mode):
    parser = argparse.ArgumentParser(description=f"CorrTrack v2 — mode {mode}.")
    parser.add_argument("csv", help="Path to the CSV (1st column = time).")
    parser.add_argument("--step", default="all", choices=["all"] + steps_for(mode),
                        help="Run the pipeline only up to this step (inclusive).")
    parser.add_argument("--backend", choices=available(), default=None,
                        help="Compute backend shared by the 3 phases (default python). "
                             "sharded_<base> = static window-partition parallelism "
                             "(workers threads, base kernel per worker; scales with n_series).")
    parser.add_argument("--sketch-backend", dest="sketch_backend",
                        choices=available(), default=None,
                        help="Backend for the sketch phase (else: --backend).")
    parser.add_argument("--candidate-backend", dest="candidate_backend",
                        choices=available(), default=None,
                        help="Backend for the candidate/cosine phase (else: --backend).")
    parser.add_argument("--validate-backend", dest="validate_backend",
                        choices=available(), default=None,
                        help="Backend for the validation/Pearson phase (else: --backend).")
    parser.add_argument("--n-series", dest="n_series", type=int, default=None,
                        help="Keep only the first N series (0 = all).")
    parser.add_argument("--n-years", dest="n_years", type=int, default=None,
                        help="Keep only N years of observations (0 = all). See --obs-mode.")
    parser.add_argument("--obs-mode", dest="obs_mode",
                        choices=["years", "count"], default=None,
                        help="'years': last N*365*24 obs; 'count': first N rows.")
    parser.add_argument("--window-size", dest="window_size", type=int, default=None)
    parser.add_argument("--window-step", dest="window_step", type=int, default=None)
    parser.add_argument("--n-lags", dest="n_lags", type=int, default=None)
    parser.add_argument("--n-vectors", dest="n_vectors", type=int, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--basic-window", dest="basic_window", type=int, default=None,
                        help="(>0) enable incremental sketch (must divide window_size).")
    parser.add_argument("--sketch-method", dest="sketch_method",
                        choices=["random_projection", "fft_lowpass", "fft_topk",
                                 "median_blocks", "median_phase"],
                        default=None,
                        help="random_projection (default, Gaussian) | fft_lowpass | "
                             "fft_topk | median_blocks (median PAA) | "
                             "median_phase (folded median). See SKETCHES.md.")
    parser.add_argument("--index-backend", dest="index_backend",
                        choices=["grid", "tree", "kdtree", "quadtree", "bptree", "bst",
                                 "octree", "knn", "bptree3d", "bst3d",
                                 "hnsw", "vptree", "annoy"],
                        default=None)
    parser.add_argument("--n-neighbors", dest="n_neighbors", type=int, default=None,
                        help="k for knn and hnsw (kNN mode when --query-radius is omitted).")
    parser.add_argument("--key-mode", dest="key_mode",
                        choices=["truncate", "median", "sum", "abs_sum", "random"], default=None,
                        help="bst/bptree (+3D): key extraction (truncate=sketch[0], "
                             "median, sum, or random=sketch-of-sketch).")
    parser.add_argument("--key-proj-dim", dest="key_proj_dim", type=int, default=None,
                        help="key projection dimension (`random` mode only).")
    parser.add_argument("--grid-cell", dest="grid_cell", type=float, default=None,
                        help="grid LSH: cell size (= former --cell-size).")
    parser.add_argument("--grid-n-tables", dest="grid_n_tables", type=int, default=None,
                        help="grid LSH: number of tables (shifted grids).")
    parser.add_argument("--grid-n-coords", dest="grid_n_coords", type=int, default=None,
                        help="grid LSH: number of sketch coords per grid.")
    parser.add_argument("--grid-vote-min", dest="grid_vote_min", type=int, default=None,
                        help="grid LSH: multi-grid vote threshold (>=1).")
    parser.add_argument("--quadtree-proj-dim", dest="quadtree_proj_dim", type=int, default=None,
                        help="quadtree: JL projection dimension (default 2).")
    parser.add_argument("--annoy-n-trees", dest="annoy_n_trees", type=int, default=None,
                        help="annoy: number of independent trees.")
    parser.add_argument("--query-radius", dest="query_radius", type=float, default=None,
                        help="query radius (tree/kdtree/quadtree/bptree/bst/octree/"
                             "bptree3d/bst3d/vptree/annoy/hnsw).")
    parser.add_argument("--sketch-threshold", dest="sketch_threshold", type=float, default=None)
    parser.add_argument("--corr-threshold", dest="corr_threshold", type=float, default=None)
    parser.add_argument("--std-threshold", dest="std_threshold", type=float, default=None,
                        help="Skip near-constant windows (std < this); v1 ≈ 0.001, 0 = off.")
    parser.add_argument("--neg-corr", dest="neg_corr",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Also accept negative correlations.")
    parser.add_argument("--workers", dest="workers", type=int, default=None,
                        help="Number of workers (sharded threads OR parallel processes; 0 = auto). "
                             "Aliased as --max-workers for backward compat.")
    parser.add_argument("--max-workers", dest="workers", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cores", dest="cores", choices=["auto", "perf", "eco"],
                        default=None,
                        help="macOS core hint (QoS): perf=P-cores, eco=E-cores, auto=default.")
    parser.add_argument("--output", dest="output", default=None,
                        help="Output directory for CSVs + logs (default 'results'; empty = none).")
    parser.add_argument("--clean", dest="clean",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Delete the output directory at startup (clean run).")
    parser.add_argument("--profile", dest="profile",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Print per-phase timing (bottleneck detection).")
    parser.add_argument("--log-level", dest="log_level",
                        choices=["debug", "info", "warning", "error"], default=None,
                        help="Log verbosity (default info).")
    parser.add_argument("--log-file", dest="log_file", default=None,
                        help="Also write logs to this .log file.")
    parser.add_argument("--log-every", dest="log_every", type=float, default=None,
                        help="Seconds between progress checkpoints (default 5).")

    return parser


_NON_CONFIG = ("csv", "step", "dataset_id", "stats_output")


def parse(mode):
    args = build_parser(mode).parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in _NON_CONFIG}
    return Config.build(**overrides), args.csv, getattr(args, "step", "all")


def parse_bench():
    """Benchmark parser: one flag per Config field, each accepting a value OR a
    comma-separated list (sweep).

    Returns (csv, dataset_id, stats_output, axes) where `axes` is
    {field -> [coerced values]} for the provided fields only.
    """
    p = argparse.ArgumentParser(description="CorrTrack v2 — benchmark / sweep.")
    p.add_argument("csv", help="Path to the CSV (1st column = time).")
    p.add_argument("--dataset-id", dest="dataset_id", default="dataset",
                   help="Dataset identifier (dataset_id column).")
    p.add_argument("--stats-output", dest="stats_output", default=None,
                   help="Stats CSV file (else: stdout).")
    for f in fields(Config):
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None,
                       help=f"{f.name} — value or list 'a,b,c' (sweep).")
    args = p.parse_args()

    axes = {}
    for f in fields(Config):
        raw = getattr(args, f.name)
        if raw is not None:
            axes[f.name] = [_coerce(v.strip(), f.type) for v in raw.split(",")]
    return args.csv, args.dataset_id, args.stats_output, axes


# abbreviations for the per-run folder name (modified parameters)
_SLUG_KEY = {
    "n_series": "ns", "n_years": "ny", "obs_mode": "obs", "train_ratio": "trn",
    "window_size": "ws", "window_step": "step", "n_lags": "lags",
    "n_vectors": "nv", "seed": "seed", "basic_window": "bw",
    "index_backend": "idx",
    # grid
    "grid_cell": "gc", "grid_n_tables": "gnt", "grid_n_coords": "gnc",
    "grid_vote_min": "gvm",
    # quadtree
    "quadtree_proj_dim": "qpd",
    # universal
    "query_radius": "qr", "n_neighbors": "nn",
    # bptree/bst (+3D)
    "key_mode": "km", "key_proj_dim": "kpd",
    # hnsw
    "hnsw_m": "hm", "hnsw_ef_c": "hec", "hnsw_ef_s": "hes",
    # annoy
    "annoy_n_trees": "ant", "annoy_leaf_size": "als",
    "sketch_threshold": "sth", "corr_threshold": "corr", "neg_corr": "neg",
    "backend": "back", "sketch_backend": "sback", "candidate_backend": "cback",
    "validate_backend": "vback", "workers": "w",
    "target_recall": "trec", "recall_fallback_near_ratio": "rfb",
    "speedup_near_ratio": "snr",
}
_SLUG_VAL = {"vectorized": "vect", "parallel": "par", "python": "py",
             "cython": "cy", "mps": "mps", "cuda": "cuda"}


def run_slug(config, mode):
    """Run name = mode + modified parameters (≠ default), abbreviated.

    E.g. bf --n-series 5 --n-years 1 --backend vectorized -> 'bf_ns-5_ny-1_back-vect'.
    """
    base = Config()
    parts = [mode]
    for field, abbr in _SLUG_KEY.items():
        val = getattr(config, field)
        if val != getattr(base, field):
            sval = _SLUG_VAL.get(str(val), str(val)).replace(".", "p")
            parts.append(f"{abbr}-{sval}")
    return "_".join(parts)


def apply_run_dir(config, mode):
    """Nest the output into a per-run sub-folder (results/<slug>/)."""
    if config.output:
        config.output = os.path.join(config.output, run_slug(config, mode))
    return config.output


def clean_output_dir(output, clean):
    """Remove the output directory at startup when --clean is set.

    Call ONCE per process, before any pipeline.run (so multiple runs in the same
    invocation — e.g. bf + corrtrack in bench — don't wipe each other's output).
    """
    if clean and output and os.path.isdir(output):
        shutil.rmtree(output)


def report(result, config):
    """Print a COMPACT summary (counters + runtime) — never the big payload lists.

    Result lists (correlated, episodes, sketches…) are only written to disk via
    `--output`; they are never dumped to the console. With `--profile`, the
    per-phase timing table is printed too.
    """
    timings = result.pop("_timings", None)
    if config.profile and timings is not None:
        print(timings.summary(total=result.get("runtime")))
    compact = {k: v for k, v in result.items()
               if k != "timings" and not isinstance(v, list)}
    print(compact)
    # hint when there are results but nowhere to save them
    if not config.output and result.get("n_correlated"):
        get_logger(config.log_level, config.log_file).info(
            "%d correlated pairs not saved — pass --output <dir> to write "
            "correlated.csv (+ episodes/anomalies).", result["n_correlated"])
