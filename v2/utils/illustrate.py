"""Illustrate pipeline steps as matplotlib images (JPG/PNG).

For a window, render one image per requested step:
  * `sketch`     (corrtrack) — the series' sketches (first 2 coords) as points;
  * `candidates` — the candidate pairs as edges;
  * `validate`   — the validated (correlated) pairs as edges (green=+, red=−).

Layout: corrtrack uses the 2D sketch scatter (+ grid cells / neighborhood radius
for `candidates`); brute-force has no sketch, so series are placed on a circle
(graph view). Intermediate CSVs are written so images can be regenerated.

    python3 -m v2.utils.illustrate data.csv --step candidates --index-backend grid --image c.jpg
    python3 -m v2.utils.illustrate data.csv --step validate --window 5 --image v.jpg
"""

import argparse
import csv
import math
import os

from ..core.cli import build_parser
from ..core.config import Config
from ..core.pipeline import run

STEPS = ("sketch", "candidates", "validate")


def _gather(config, dataset, mode, steps):
    """Return (sketches_by_window, edges_by_step_and_window, times).

    sketches: {t: {sid: [coords]}} (corrtrack only).
    edges: {step: {t: [(id1, id2, sign)]}} for the requested steps.
    """
    sk_by_t, edges = {}, {s: {} for s in steps}
    times = []

    def add_edges(step, pairs, signed):
        for p in pairs:
            if signed:
                a, b, corr, _ = p
                e = (a[0], b[0], 1 if corr > 0 else -1, a[1])
            else:
                a, b = p
                e = (a[0], b[0], 0, a[1])
            if a[1] == b[1]:  # intra-window pair
                edges[step].setdefault(a[1], []).append(e[:3])

    if mode == "corrtrack" and ("sketch" in steps or "candidates" in steps or "validate" in steps):
        sk = run(config, dataset, mode, step="sketch")["sketches"]
        for (sid, t), vec in sk:
            sk_by_t.setdefault(t, {})[sid] = list(map(float, vec))
        times = sorted(sk_by_t, key=float)
    if "candidates" in steps:
        step = "select" if mode == "corrtrack" else "candidates"
        add_edges("candidates", run(config, dataset, mode, step=step)["candidates"], False)
    if "validate" in steps:
        add_edges("validate", run(config, dataset, mode, step="validate")["correlated"], True)

    if not times:  # bf: derive window times from the edges
        ts = {t for st in edges.values() for t in st}
        times = sorted(ts, key=float)
    return sk_by_t, edges, times


def _write_csv(out_dir, points, edges, step, t):
    os.makedirs(out_dir, exist_ok=True)
    if points:
        with open(os.path.join(out_dir, f"sketches_w{t}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            dim = max((len(v) for v in points.values()), default=0)
            w.writerow(["series"] + [f"sk{i}" for i in range(dim)])
            for sid, vec in points.items():
                w.writerow([sid] + list(vec))
    with open(os.path.join(out_dir, f"{step}_w{t}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id1", "id2", "sign"])
        w.writerows(edges)


def render(points, edges, step, index_backend, grid_cell, query_radius, image, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    if points:                               # corrtrack: 2D sketch scatter
        pos = {sid: (v[0], v[1] if len(v) > 1 else 0.0) for sid, v in points.items()}
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        lo, hi = min(xs + ys) - 1, max(xs + ys) + 1
        if step == "candidates" and index_backend == "grid":
            g = lo
            while g <= hi:
                ax.axvline(g, color="0.9", lw=0.5, zorder=0)
                ax.axhline(g, color="0.9", lw=0.5, zorder=0)
                g += grid_cell
        elif step == "candidates":
            for x, y in zip(xs, ys):
                ax.add_patch(Circle((x, y), query_radius, fill=False, ec="0.85", lw=0.5, zorder=0))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("sketch coord 0")
        ax.set_ylabel("sketch coord 1")
    else:                                    # bf: circular graph of involved series
        sids = sorted({s for e in edges for s in e[:2]} | set())
        pos = {s: (math.cos(2 * math.pi * i / max(len(sids), 1)),
                   math.sin(2 * math.pi * i / max(len(sids), 1))) for i, s in enumerate(sids)}
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

    for e in edges:                          # edges (colored by sign for validate)
        a, b = e[0], e[1]
        if a in pos and b in pos:
            sign = e[2] if len(e) > 2 else 0
            color = "tab:green" if sign > 0 else "tab:red" if sign < 0 else "tab:blue"
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=color, alpha=0.45, lw=1.0, zorder=1)
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
               s=60, color="tab:red", zorder=2)
    for sid, (x, y) in pos.items():
        ax.annotate(sid, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title)
    fig.tight_layout()
    try:
        fig.savefig(image, dpi=130)
    except Exception:
        image = os.path.splitext(image)[0] + ".png"
        fig.savefig(image, dpi=130)
    plt.close(fig)
    return image


def illustrate_run(config, dataset, out_dir, steps=("candidates",), windows=(0,),
                   mode="corrtrack"):
    """Pipeline hook: render the requested steps for the given windows. Quiet."""
    from dataclasses import replace
    steps = [s for s in steps if s in STEPS]
    if mode != "corrtrack":
        steps = [s for s in steps if s != "sketch"]  # bf has no sketch
    if not steps:
        return []
    cfg = replace(config, output="", log_level="error")
    sk_by_t, edges, times = _gather(cfg, dataset, mode, steps)
    if not times:
        return []
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for win in windows:
        t = times[min(int(win), len(times) - 1)]
        points = sk_by_t.get(t, {})
        for step in steps:
            es = [] if step == "sketch" else edges[step].get(t, [])
            _write_csv(out_dir, points, es, step, t)
            img = os.path.join(out_dir, f"{step}_w{int(win)}.jpg")
            title = (f"{step} — {mode} — index={cfg.index_backend} — t={t} — "
                     f"{len(points) or len({s for e in es for s in e[:2]})} series, "
                     f"{len(es)} edges")
            written.append(render(points, es, step, cfg.index_backend,
                                  cfg.grid_cell, cfg.query_radius, img, title))
    return written


def main():
    p = build_parser("corrtrack")
    p.add_argument("--ill-step", dest="ill_step", choices=STEPS, default="candidates",
                   help="Step to illustrate (sketch|candidates|validate).")
    p.add_argument("--mode", choices=["corrtrack", "bf"], default="corrtrack")
    p.add_argument("--window", type=int, default=0, help="Window index to illustrate.")
    p.add_argument("--image", default=None, help="Output image path.")
    args = p.parse_args()

    exclude = ("csv", "step", "ill_step", "mode", "window", "image")
    overrides = {k: v for k, v in vars(args).items() if k not in exclude}
    config = Config.build(**overrides)
    out_dir = os.path.dirname(os.path.abspath(args.image)) if args.image else "."
    imgs = illustrate_run(config, args.csv, out_dir, steps=[args.ill_step],
                          windows=[args.window], mode=args.mode)
    if args.image and imgs:
        os.replace(imgs[0], args.image)
        imgs[0] = args.image
    print(f"[illustrate] step={args.ill_step} -> {imgs}")


if __name__ == "__main__":
    main()
