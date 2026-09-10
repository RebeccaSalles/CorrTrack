"""Pipeline JSON generator (cartesian indexes × sketches × backends).

Usage:
    python -m v2.utils.gen_pipeline --output pipe.json \\
        --indexes bptree,kdtree,grid \\
        --backends vectorized,sharded_vectorized \\
        --sketches random_projection,fft_lowpass \\
        --workers 4,8 \\
        --n-series 5 --n-years 1

Generates the cartesian product (sketches × indexes × backends × workers) while
honouring the special rules:
- the `bf_python` baseline is prepended automatically (unless `--no-baseline`)
- workers is ignored for NON sharded backends (= 1 run per combo)
- hnsw/annoy + sharded → backend forced to sharded_vectorized (no-shard fallback)
- knn → n_neighbors=512 added by default
- indexes `bst`/`bptree`/`bptree3d`/`bst3d` → key_mode declined when --key-modes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product

SHARDED_PREFIX = "sharded_"
DEFAULT_INDEXES = ["bptree", "kdtree", "grid", "bst", "tree", "quadtree",
                   "octree", "bptree3d", "bst3d", "knn", "hnsw", "vptree", "annoy"]
DEFAULT_BACKENDS = ["vectorized"]
DEFAULT_SKETCHES = ["random_projection"]
DEFAULT_WORKERS = [4]
_BACKEND_ABBR = {"vectorized": "vect", "cython": "cyth", "python": "py",
                 "mps": "mps", "sharded_vectorized": "shvect",
                 "sharded_cython": "shcyth", "sharded_python": "shpy",
                 "sharded_mps": "shmps", "vectorized_parallel": "vectpar",
                 "cython_parallel": "cythpar"}
_SKETCH_ABBR = {"random_projection": "", "fft_lowpass": "fftlow",
                "fft_topk": "ffttop", "median_blocks": "medbl",
                "median_phase": "medph"}


def _is_sharded(backend):
    return backend.startswith(SHARDED_PREFIX)


def _name_for(index, sketch, backend, workers, key_mode=""):
    parts = ["corrtrack", index]
    if key_mode and key_mode != "truncate":
        parts.append(key_mode)
    if sketch and sketch != "random_projection":
        parts.append(_SKETCH_ABBR.get(sketch, sketch))
    parts.append(_BACKEND_ABBR.get(backend, backend))
    if _is_sharded(backend) and workers:
        parts.append(f"w{workers}")
    return "_".join(p for p in parts if p)


def _params_for(index, sketch, backend, workers, key_mode):
    p = {"index-backend": index, "backend": backend}
    if sketch and sketch != "random_projection":
        p["sketch_method"] = sketch
        # some sketch/index combos need a wider default radius
        if index not in ("grid", "knn"):
            p["query_radius"] = 4
        if sketch.startswith("median_"):
            p["n_vectors"] = 24
    if index == "knn":
        p["n_neighbors"] = 512
    if index in ("bptree", "bst", "bptree3d", "bst3d") and key_mode:
        p["key-mode"] = key_mode
        if key_mode == "random":
            p["key-proj-dim"] = 3
    if _is_sharded(backend):
        p["max_workers"] = workers
    return p


def build_runs(indexes, backends, sketches, workers_list, key_modes,
               extra_params=None, optimize=True):
    runs = []
    extra = extra_params or {}
    for sketch, index, backend in product(sketches, indexes, backends):
        # workers seulement pour sharded
        w_iter = workers_list if _is_sharded(backend) else [None]
        key_iter = key_modes if index in ("bptree", "bst", "bptree3d", "bst3d") else [""]
        for workers, key_mode in product(w_iter, key_iter):
            run = {"name": _name_for(index, sketch, backend, workers, key_mode),
                   "mode": "corrtrack"}
            # `optimize: true` ONLY when a grid is provided — otherwise the run
            # would request a sweep without a grid (warning + static params).
            if optimize:
                run["optimize"] = True
            run["params"] = {**_params_for(index, sketch, backend, workers, key_mode),
                             **extra}
            runs.append(run)
    return runs


def build_pipeline(name, dataset, output, indexes, backends, sketches,
                   workers_list, key_modes, n_series, n_years, neg_corr,
                   std_threshold, baseline, optimize_grid, train_ratio,
                   target_recall, target_precision):
    spec = {
        "name": name,
        # `false` (= --no-baseline): prevents the pipeline from auto-adding a
        # slow bf_python; speedup/recall then become relative to run #0.
        "baseline": 0 if baseline else False,
        "dataset": dataset,
        "clean": False,
        "incremental": True,
        "output": output,
        "params": {
            "n_series": n_series,
            "n_years": n_years,
            "neg_corr": neg_corr,
            "std_threshold": std_threshold,
        },
    }
    optimize = bool(optimize_grid)
    # NB: optimize + --no-baseline is VALID — the sweep computes its own bf on
    # the train_ratio subsample; only the slow pipeline-level bf is omitted
    # (speedup/recall relative to run #0).
    if optimize:
        spec["optimize"] = {
            "grid": optimize_grid,
            "train_ratio": train_ratio,
            "target_recall": target_recall,
            "target_precision": target_precision,
            "auto_derive": True,
            "smart_grid": True,
            "successive_halving": 3,
        }
    runs = []
    if baseline:
        runs.append({"name": "bf_python", "mode": "bf"})
    runs.extend(build_runs(indexes, backends, sketches, workers_list, key_modes,
                           optimize=optimize))
    spec["runs"] = runs
    return spec


def _default_optimize_grid():
    # NB: no `key_mode` / `key_proj_dim` — the key mode is pinned PER RUN
    # (bptree/bst…) hence static, and has no effect for the other indexes.
    # Sweeping it in the grid would be redundant (see the v2/pipeline.py
    # precedence base < grid < run_explicit).
    return {
        "n_vectors": [32, 64],
        "grid_cell": [0.25, 0.5, 0.75, 1.0],
        "grid_n_tables": [8, 16],
        "query_radius": [2, 4, 6, 8, 10, 12],
        "n_neighbors": [32, 64, 128, 256, 512],
    }


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="path of the output JSON")
    p.add_argument("--name", default="generated_pipeline",
                   help="pipeline name (= results/ sub-folder)")
    p.add_argument("--dataset", default="fr-air_temperature.csv")
    p.add_argument("--pipeline-output", default="results",
                   help="results/ folder of the pipeline")
    p.add_argument("--indexes", default=",".join(DEFAULT_INDEXES),
                   help=f"CSV list; default = {','.join(DEFAULT_INDEXES)}")
    p.add_argument("--backends", default=",".join(DEFAULT_BACKENDS),
                   help="CSV list of backends (python/vectorized/cython/mps/"
                        "sharded_*)")
    p.add_argument("--sketches", default=",".join(DEFAULT_SKETCHES),
                   help="CSV list (random_projection/fft_*/median_*)")
    p.add_argument("--workers", default=",".join(map(str, DEFAULT_WORKERS)),
                   help="CSV list of max_workers (only used for the sharded_* "
                        "backends)")
    p.add_argument("--key-modes", default="truncate",
                   help="CSV list for bptree/bst (truncate/median/sum/random)")
    p.add_argument("--n-series", type=int, default=5)
    p.add_argument("--n-years", type=int, default=1)
    p.add_argument("--neg-corr", action="store_true")
    p.add_argument("--std-threshold", type=float, default=0.001)
    p.add_argument("--no-baseline", action="store_true",
                   help="do not prepend the bf_python run")
    p.add_argument("--no-optimize", action="store_true",
                   help="omit the optimize block (sweep disabled)")
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--target-precision", type=float, default=0.9)
    args = p.parse_args(argv)

    workers_list = [int(w) for w in _split(args.workers)]
    grid = None if args.no_optimize else _default_optimize_grid()
    try:
        spec = build_pipeline(
            name=args.name,
            dataset=args.dataset,
            output=args.pipeline_output,
            indexes=_split(args.indexes),
            backends=_split(args.backends),
            sketches=_split(args.sketches),
            workers_list=workers_list,
            key_modes=_split(args.key_modes),
            n_series=args.n_series,
            n_years=args.n_years,
            neg_corr=args.neg_corr,
            std_threshold=args.std_threshold,
            baseline=not args.no_baseline,
            optimize_grid=grid,
            train_ratio=args.train_ratio,
            target_recall=args.target_recall,
            target_precision=args.target_precision,
        )
    except ValueError as e:
        use_color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
        msg = f"[gen_pipeline] ERREUR : {e}"
        sys.exit(f"\x1b[1;31m{msg}\x1b[0m" if use_color else msg)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"[gen_pipeline] {args.output} → {len(spec['runs'])} runs "
          f"({len(_split(args.indexes))} indexes × {len(_split(args.sketches))} "
          f"sketches × {len(_split(args.backends))} backends"
          f"{' × ' + str(len(workers_list)) + ' workers' if any(_is_sharded(b) for b in _split(args.backends)) else ''}"
          f"{' × ' + str(len(_split(args.key_modes))) + ' key_modes' if 'bptree' in _split(args.indexes) or 'bst' in _split(args.indexes) else ''})")


if __name__ == "__main__":
    main()
