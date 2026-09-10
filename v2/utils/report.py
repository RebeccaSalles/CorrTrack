"""Generate a PDF + PNG report from `comparison.csv`.

Called automatically at the end of `v2.pipeline` (when `comparison.csv` exists).
It can also be run standalone:

    python -m v2.utils.report results/<pipeline>/

Produces, in `<pipeline>/report/`:
  * `report.pdf` — multi-page report (summary + table + Pareto scatter +
    bar charts per index/backend).
  * `pareto_recall_speedup.png` — main figure exported separately.
  * `barchart_runtime.png`, `barchart_recall.png` — per-run bar charts.
"""

import argparse
import csv
import os
from collections import defaultdict


def _f(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _index_from_name(name):
    """Guess the index used from the run name (best-effort)."""
    for idx in ("bptree3d", "bst3d", "octree", "quadtree", "kdtree", "bptree",
                "vptree", "hnsw", "annoy", "knn", "grid", "tree", "bst"):
        if idx in name:
            return idx
    if name.startswith("bf"):
        return "bf"
    if name.startswith("filcorr"):
        return "filcorr"
    return "?"


def _backend_from_name(name):
    for be in ("vectorized_parallel", "cython_parallel", "vectorized", "cython",
               "parallel", "mps", "coreml", "python"):
        if name.endswith("_" + be) or name == ("bf_" + be):
            return be
    return "python"


def generate(pipeline_dir, dry=False):
    """Generate report.pdf + PNGs in pipeline_dir/report/. Return the PDF path."""
    csv_path = os.path.join(pipeline_dir, "comparison.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows:
        return None
    out_dir = os.path.join(pipeline_dir, "report")
    os.makedirs(out_dir, exist_ok=True)

    if dry:
        return os.path.join(out_dir, "report.pdf")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    name = os.path.basename(os.path.normpath(pipeline_dir)) or "pipeline"
    runs = [r.get("dataset_id", "?") for r in rows]
    recall = [_f(r, "recall") for r in rows]
    prec = [_f(r, "precision") for r in rows]
    speedup = [_f(r, "speedup") for r in rows]
    total = [_f(r, "total_time") for r in rows]
    runtime = [_f(r, "runtime") for r in rows]
    cand_pct = [_f(r, "cand_w_pct") for r in rows]
    corr_pct = [_f(r, "corr_w_pct") for r in rows]
    missed = [_f(r, "missed") for r in rows]
    opt_t = [_f(r, "opt_time") for r in rows]
    indexes = [_index_from_name(n) for n in runs]
    backends = [_backend_from_name(n) for n in runs]

    pdf_path = os.path.join(out_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        # ============ Page 1: textual summary ============
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.suptitle(f"Pipeline Report — {name}", fontsize=15, fontweight="bold")
        ok = [i for i in range(len(rows)) if recall[i] == recall[i]]
        if ok:
            i_rec = max(ok, key=lambda i: recall[i])
            i_spd = max(ok, key=lambda i: speedup[i] if speedup[i] == speedup[i] else -1)
            i_tot = min(ok, key=lambda i: total[i] if total[i] == total[i] else 1e9)
            i_pareto = max(ok, key=lambda i: (recall[i] * speedup[i])
                           if recall[i] == recall[i] and speedup[i] == speedup[i] else -1)
            lines = [
                f"{'='*65}",
                f"{len(rows)} runs",
                f"{'='*65}",
                "",
                f"  Top recall       : {runs[i_rec]:<35} {recall[i_rec]:.3f}",
                f"  Top speedup      : {runs[i_spd]:<35} {speedup[i_spd]:.2f}×",
                f"  Fastest total    : {runs[i_tot]:<35} {total[i_tot]:.2f}s",
                f"  Best Pareto      : {runs[i_pareto]:<35} "
                f"recall={recall[i_pareto]:.3f} speedup={speedup[i_pareto]:.2f}×",
                "",
                f"  Indexes used     : {sorted(set(indexes))}",
                f"  Backends used    : {sorted(set(backends))}",
            ]
            ax.text(0.05, 0.85, "\n".join(lines), fontsize=9, family="monospace",
                    verticalalignment="top")
        pdf.savefig(fig); plt.close(fig)

        # ============ Page 2 : Pareto recall vs speedup ============
        fig, ax = plt.subplots(figsize=(11, 8.5))
        # color by index
        idx_set = sorted(set(indexes))
        cmap = plt.get_cmap("tab20")
        idx_color = {ix: cmap(i % 20) for i, ix in enumerate(idx_set)}
        for ix in idx_set:
            mask = [i for i in range(len(rows)) if indexes[i] == ix
                    and recall[i] == recall[i] and speedup[i] == speedup[i]]
            if not mask: continue
            ax.scatter([speedup[i] for i in mask], [recall[i] for i in mask],
                       s=70, alpha=0.75, c=[idx_color[ix]], label=ix,
                       edgecolors="black", linewidths=0.4)
        ax.set_xlabel("Speedup (× vs baseline)")
        ax.set_ylabel("Recall")
        ax.set_title("Pareto : Recall vs Speedup (par index)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        # annotations on the 3 best Pareto points
        if ok:
            top3 = sorted(ok, key=lambda i: -(recall[i] * speedup[i]
                          if recall[i] == recall[i] and speedup[i] == speedup[i] else -1))[:3]
            for i in top3:
                ax.annotate(runs[i], (speedup[i], recall[i]),
                            fontsize=8, xytext=(5, 5), textcoords="offset points")
        pdf.savefig(fig); plt.close(fig)
        # also exported as a separate PNG
        fig, ax = plt.subplots(figsize=(11, 8.5))
        for ix in idx_set:
            mask = [i for i in range(len(rows)) if indexes[i] == ix
                    and recall[i] == recall[i] and speedup[i] == speedup[i]]
            if not mask: continue
            ax.scatter([speedup[i] for i in mask], [recall[i] for i in mask],
                       s=70, alpha=0.75, c=[idx_color[ix]], label=ix,
                       edgecolors="black", linewidths=0.4)
        ax.set_xlabel("Speedup (× vs baseline)"); ax.set_ylabel("Recall")
        ax.set_title(f"Pareto Recall vs Speedup — {name}")
        ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pareto_recall_speedup.png"), dpi=120)
        plt.close(fig)

        # ============ Page 3: sorted runtime bar chart ============
        order = sorted(range(len(rows)), key=lambda i: runtime[i]
                       if runtime[i] == runtime[i] else 1e9)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        names_o = [runs[i] for i in order]
        rt_o = [runtime[i] for i in order]
        opt_o = [opt_t[i] if opt_t[i] == opt_t[i] else 0 for i in order]
        colors = [idx_color[indexes[i]] for i in order]
        ax.barh(range(len(names_o)), rt_o, color=colors, edgecolor="black", lw=0.3,
                label="runtime")
        ax.barh(range(len(names_o)), opt_o, left=rt_o, color="0.75",
                edgecolor="black", lw=0.3, label="opt_t")
        ax.set_yticks(range(len(names_o)))
        ax.set_yticklabels(names_o, fontsize=6)
        ax.set_xlabel("seconds (runtime + opt_t)")
        ax.set_title(f"Total time per run (sorted) — {name}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)
        fig.savefig(os.path.join(out_dir, "barchart_runtime.png"), dpi=120)
        plt.close(fig)

        # ============ Page 4: sorted recall bar chart ============
        order = sorted(range(len(rows)), key=lambda i: -(recall[i]
                       if recall[i] == recall[i] else -1))
        fig, ax = plt.subplots(figsize=(11, 8.5))
        names_o = [runs[i] for i in order]
        rec_o = [recall[i] for i in order]
        cand_o = [cand_pct[i] if cand_pct[i] == cand_pct[i] else 0 for i in order]
        colors = [idx_color[indexes[i]] for i in order]
        x = list(range(len(names_o)))
        ax.barh(x, rec_o, color=colors, edgecolor="black", lw=0.3, label="recall")
        ax.set_yticks(x); ax.set_yticklabels(names_o, fontsize=6)
        ax.set_xlabel("recall"); ax.set_xlim(0, 1.05)
        ax.set_title(f"Recall per run (sorted) — {name}")
        ax.grid(True, axis="x", alpha=0.3)
        for i, c in zip(x, cand_o):
            ax.text(1.02, i, f"{c:.0f}% cand", fontsize=5,
                    verticalalignment="center", color="gray")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)
        fig.savefig(os.path.join(out_dir, "barchart_recall.png"), dpi=120)
        plt.close(fig)

        # ============ Page 5: summary table (text) ============
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(f"Top 15 runs by total time — {name}", fontsize=12,
                     fontweight="bold", loc="left")
        order = sorted(range(len(rows)), key=lambda i: total[i]
                       if total[i] == total[i] else 1e9)[:15]
        header = f"{'run':<35}{'tot':>8}{'rt':>7}{'opt_t':>7}{'spd':>7}{'rec':>7}{'cand%':>7}"
        body = [header, "-" * len(header)]
        for i in order:
            body.append(
                f"{runs[i]:<35}{total[i]:>7.2f}s{runtime[i]:>6.2f}s"
                f"{opt_t[i]:>6.1f}s{speedup[i]:>6.2f}×{recall[i]:>7.3f}"
                f"{cand_pct[i]:>6.1f}%"
            )
        ax.text(0.02, 0.95, "\n".join(body), fontsize=8, family="monospace",
                verticalalignment="top")
        pdf.savefig(fig); plt.close(fig)

    return pdf_path


def main():
    p = argparse.ArgumentParser(description="Generate a PDF report from comparison.csv.")
    p.add_argument("pipeline_dir", help="Directory of a pipeline run (contains comparison.csv).")
    args = p.parse_args()
    pdf = generate(args.pipeline_dir)
    if pdf is None:
        print(f"[report] no comparison.csv in {args.pipeline_dir}")
    else:
        print(f"[report] generated: {pdf}")


if __name__ == "__main__":
    main()
