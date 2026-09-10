"""Freeze a pipeline's optimization into a new static config.

Reads a pipeline config + its results, and for every run that used
`"optimize": true`, bakes the optimizer's chosen params (from
`<run>/optimize/best_params.json`) into the run as STATIC `params`, drops the
`optimize` flag, and appends a suffix (default `-best`) to the run name. The
pipeline `name` also gets the suffix (so the frozen run writes to its own dir
and doesn't re-optimize). Result: a reproducible, fast config — no sweeping.

    python3 -m v2.utils.freeze pipeline-run.json
    python3 -m v2.utils.freeze pipeline-run.json --out tuned.json --suffix -best
    python3 -m v2.utils.freeze pipeline-run.json --results results/test_bf_corr
"""

import argparse
import json
import os

from ..pipeline import load_jsonc


def freeze(spec, results_dir, suffix="-best"):
    """Return a new config dict with optimized runs baked to static params."""
    out = dict(spec)
    out["name"] = spec.get("name", "pipeline") + suffix
    new_runs, frozen, missing = [], 0, []
    for run in spec.get("runs", []):
        r = dict(run)
        if run.get("optimize") or run.get("use_optimized"):
            rn = run.get("name", "")
            bp = os.path.join(results_dir, rn, "optimize", "best_params.json")
            if os.path.exists(bp):
                with open(bp) as f:
                    best = json.load(f).get("params", {})
                r["params"] = {**run.get("params", {}), **best}  # best params win
                r.pop("optimize", None)
                r.pop("use_optimized", None)
                r["name"] = rn + suffix
                frozen += 1
            else:
                missing.append(rn)  # no optim result -> left as-is
        new_runs.append(r)
    out["runs"] = new_runs
    return out, frozen, missing


def main():
    p = argparse.ArgumentParser(description="Freeze a pipeline's optimization into a static config.")
    p.add_argument("config", help="Pipeline JSON config (the one that was run).")
    p.add_argument("--results", default=None,
                   help="Pipeline results dir (default: <output>/<name>).")
    p.add_argument("--out", default=None,
                   help="Output config path (default: <config>-best.json).")
    p.add_argument("--suffix", default="-best", help="Suffix for name + run names.")
    args = p.parse_args()

    spec = load_jsonc(args.config)
    results_dir = args.results or os.path.join(spec.get("output", "results"),
                                               spec.get("name", "pipeline"))
    out_spec, frozen, missing = freeze(spec, results_dir, args.suffix)

    out_path = args.out or (os.path.splitext(args.config)[0] + args.suffix + ".json")
    with open(out_path, "w") as f:
        json.dump(out_spec, f, indent=2)

    print(f"[freeze] {frozen} run(s) frozen to static best params -> {out_path}")
    if missing:
        print(f"[freeze] WARNING: no best_params.json for {missing} "
              f"(not optimized yet in {results_dir}) — left unchanged.")


if __name__ == "__main__":
    main()
