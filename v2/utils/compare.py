"""Compare two 'correlated pairs' CSV outputs — typically CorrTrack v1 vs v2.

Auto-detects the column layout of each file:
  v1: id1,id2,time1,time2,corr
  v2: id1,t1,id2,t2,lag,corr

Each correlated window-pair is reduced to an ORDER-INDEPENDENT key
{(idA, tA), (idB, tB)} so that (a,b) and (b,a) match. The two sets are then
compared and a report is printed:
  - counts, matched / only-in-A (missed) / only-in-B (extra)
  - precision, recall, f1   (file A treated as ground truth)
  - sign agreement and max |Δcorr| on matched pairs

Both files must come from the SAME dataset and parameters (same time units),
otherwise the keys cannot match.

    python -m v2.compare_v1 v1_correlated.csv v2/out/correlated.csv
    python -m v2.compare_v1 a.csv b.csv --examples 10 --tol 1e-6
"""

import argparse
import csv


def _num(x):
    """Parse a time value to int/float when possible (keeps keys comparable)."""
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return float(x)
        except (TypeError, ValueError):
            return x


def load(path):
    """Return {pair_key: corr} where pair_key = sorted((id,t),(id,t))."""
    pairs = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        t1_col, t2_col = ("time1", "time2") if "time1" in cols else ("t1", "t2")
        for row in reader:
            a = (row["id1"], _num(row[t1_col]))
            b = (row["id2"], _num(row[t2_col]))
            try:
                corr = float(row["corr"])
            except (TypeError, ValueError):
                corr = float("nan")
            pairs[tuple(sorted((a, b), key=lambda e: (str(e[0]), str(e[1]))))] = corr
    return pairs


def compare(a, b, tol=0.0):
    ka, kb = set(a), set(b)
    matched = ka & kb
    only_a, only_b = ka - kb, kb - ka

    tp, na, nb = len(matched), len(ka), len(kb)
    precision = tp / nb if nb else 0.0
    recall = tp / na if na else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    sign_disagree = sum(1 for k in matched if (a[k] > 0) != (b[k] > 0))
    diffs = [abs(a[k] - b[k]) for k in matched
             if a[k] == a[k] and b[k] == b[k]]  # skip NaN
    max_dcorr = max(diffs) if diffs else 0.0
    over_tol = sum(1 for d in diffs if d > tol)
    return {
        "n_a": na, "n_b": nb, "matched": tp,
        "only_a": only_a, "only_b": only_b,
        "precision": precision, "recall": recall, "f1": f1,
        "sign_disagree": sign_disagree, "max_dcorr": max_dcorr, "over_tol": over_tol,
    }


def _fmt_pair(k):
    (i1, t1), (i2, t2) = k
    return f"{i1}@{t1} ~ {i2}@{t2}"


def main():
    p = argparse.ArgumentParser(description="Compare two CorrTrack 'correlated' CSVs (v1 vs v2).")
    p.add_argument("file_a", help="Reference CSV (e.g. v1 output).")
    p.add_argument("file_b", help="CSV to evaluate (e.g. v2 output).")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Tolerance on |corr| difference for matched pairs (default 1e-6).")
    p.add_argument("--examples", type=int, default=0,
                   help="Print up to N example pairs from each difference set.")
    args = p.parse_args()

    a, b = load(args.file_a), load(args.file_b)
    r = compare(a, b, tol=args.tol)

    print(f"A (reference): {args.file_a}  -> {r['n_a']} pairs")
    print(f"B (evaluated): {args.file_b}  -> {r['n_b']} pairs")
    print(f"matched      : {r['matched']}")
    print(f"only in A (B missed) : {len(r['only_a'])}")
    print(f"only in B (B extra)  : {len(r['only_b'])}")
    print(f"precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1']:.4f}")
    print(f"sign disagreements (matched): {r['sign_disagree']}")
    print(f"max |Δcorr| (matched): {r['max_dcorr']:.3e}  | pairs over tol {args.tol:g}: {r['over_tol']}")

    verdict = "IDENTICAL" if (not r['only_a'] and not r['only_b']
                              and r['sign_disagree'] == 0 and r['over_tol'] == 0) else "DIFFERENT"
    print(f"=> {verdict}")

    if args.examples:
        for label, keys in (("only in A (missed)", r["only_a"]), ("only in B (extra)", r["only_b"])):
            if keys:
                print(f"\n{label} (first {args.examples}):")
                for k in list(sorted(keys))[:args.examples]:
                    print(f"  {_fmt_pair(k)}")


if __name__ == "__main__":
    main()
