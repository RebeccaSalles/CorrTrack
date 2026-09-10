"""Compare a v1 result folder to a v2 pipeline result (metrics side by side).

v1 writes a `*metrics*.csv` (the ~70-column stats, `;`-separated) per dataset; v2
writes `comparison.csv` (same column names + extras). This tool reads both and
prints the key metrics side by side (corr_w_bf, corr_w, cand_w, recall, precision,
speedup, runtime, per-phase times), so you can check whether v2 reproduces v1.

Exact correlated PAIRS are not compared (v1 uses datetime stamps, v2 integer
hours → keys differ); use counts/recall instead. If the bf ground truth count
(`corr_w_bf`) differs, the two are not on the same footing (different data slice,
windowing, or correlation counting) — a warning is printed.

    python3 -m v2.utils.compare_pipeline /path/to/v1_result_dir results/my_pipeline
    python3 -m v2.utils.compare_pipeline v1_dir path/to/comparison.csv
"""

import argparse
import csv
import glob
import os

# (column, label, format)
_METRICS = [("n_ts", "n_ts", "{:.0f}"), ("n_w", "n_w", "{:.0f}"),
            ("total_w", "total_w", "{:.0f}"),
            ("corr_w_bf", "corr_w_bf", "{:.0f}"), ("corr_w", "corr_w", "{:.0f}"),
            ("cand_w", "cand_w", "{:.0f}"), ("recall", "recall", "{:.3f}"),
            ("precision", "prec", "{:.3f}"), ("speedup", "speedup", "{:.2f}"),
            ("runtime", "runtime", "{:.2f}"), ("sk_time", "sk_t", "{:.2f}"),
            ("cand_time", "cand_t", "{:.2f}"), ("val_time", "val_t", "{:.2f}")]


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _find(root, pattern):
    if os.path.isfile(root):
        return root
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return hits[0] if hits else None


def _fmt(row, col, fmt):
    v = row.get(col, "")
    try:
        x = float(v)
        return "nan" if x != x else fmt.format(x)
    except (TypeError, ValueError):
        return str(v) if v else "-"


def main():
    p = argparse.ArgumentParser(description="Compare v1 result vs v2 pipeline (metrics).")
    p.add_argument("v1", help="v1 result folder (contains *metrics*.csv) or that CSV.")
    p.add_argument("v2", help="v2 pipeline dir (contains comparison.csv) or that CSV.")
    p.add_argument("--alg", default="corrtrack", help="v1 alg row to use (default corrtrack).")
    args = p.parse_args()

    v1_csv = _find(args.v1, "*metrics*.csv")
    v2_csv = _find(args.v2, "comparison.csv")
    if not v1_csv:
        raise SystemExit(f"no *metrics*.csv found under {args.v1}")
    if not v2_csv:
        raise SystemExit(f"no comparison.csv found under {args.v2}")

    v1_rows = [r for r in _read(v1_csv) if r.get("alg", "corrtrack") == args.alg] or _read(v1_csv)
    v2_rows = _read(v2_csv)

    labels = [("v1:" + r.get("alg", "corrtrack"), r) for r in v1_rows]
    labels += [("v2:" + r.get("dataset_id", "?"), r) for r in v2_rows]
    w = max(len(lab) for lab, _ in labels)

    print(f"metrics: v1={v1_csv}")
    print(f"         v2={v2_csv}")
    head = f"{'source':<{w}}" + "".join(f"{lab:>10}" for _, lab, _ in _METRICS)
    print(head)
    for lab, r in labels:
        print(f"{lab:<{w}}" + "".join(f"{_fmt(r, c, fm):>10}" for c, _, fm in _METRICS))

    # ground-truth sanity check
    def _g(r, k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return None
    v1_bf = _g(v1_rows[0], "corr_w_bf") if v1_rows else None
    v2_bf = {_g(r, "corr_w_bf") for r in v2_rows if _g(r, "corr_w_bf") is not None}
    if v1_bf is not None and v2_bf and v1_bf not in v2_bf:
        print(f"\n⚠ corr_w_bf differs: v1={v1_bf:.0f} vs v2={sorted(v2_bf)} — the brute-force "
              "ground truth is NOT identical (check n_years / windowing / time units / neg_corr); "
              "recall numbers are not directly comparable.")


if __name__ == "__main__":
    main()
