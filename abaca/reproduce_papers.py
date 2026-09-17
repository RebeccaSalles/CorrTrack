"""Phase R: reproduce each competitor paper's own headline result on the paper's own data (or the
registry's stand-in) with OUR port, before the head-to-head. If a port cannot reproduce what its
authors published in the regime they published it, it is not a fair competitor yet; this is the
faithfulness check comparison plan section 6.5 asks for, run as an experiment rather than a unit
test. Every subcommand prints our number next to the paper's reported value and writes a JSON.

    python abaca/reproduce_papers.py braid       [--datasets motes_humidity,motes_light,sunspots_daily,braid_sines_smoke]
    python abaca/reproduce_papers.py corrjoin    [--datasets corrjoin_stock,corrjoin_chlorine,corrjoin_gas,corrjoin_synthetic] [--m 1000]
    python abaca/reproduce_papers.py statstream  [--m 500] [--T 20000]
    python abaca/reproduce_papers.py parcorr     [--datasets sp500_sub263,corrjoin_stock] [--m 1000]
    python abaca/reproduce_papers.py filcorr     [--ms 25,50,100,200]
    python abaca/reproduce_papers.py tsubasa     [--datasets uscrn2020_temperature]
    python abaca/reproduce_papers.py all

Paper values quoted below come from docs/competitor_comparison_plan.md (sections 2.1, 3.1, 4.2,
4b, 5, 5a.2). Where the plan holds no transcribed number the column says "not transcribed" and
the comparison is qualitative until the value is read off the paper's figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO_DIR", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, REPO)
os.chdir(REPO)

from datasets.competitor_loader import load_dataset, load_raw  # noqa: E402
from library_corrtrack_parallel import (  # noqa: E402
    Candidates_BF_BRAID, CorrTrack, run_and_log_bruteforce, run_and_log_corrtrack,
)

OUT_DIR = Path(os.environ.get("REPRO_OUT", "tmp_artifacts/reproduce_papers"))


def _base(W, step, n_lags, T, neg_corr=False, basic_window=None):
    return dict(window_size=W, window_step=step, basic_window=basic_window, n_lags=n_lags, corr_threshold=T, neg_corr=neg_corr,
                exec="sequential", parallel_sketch=False, parallel_candidates=False, parallel_validation=False, max_workers=0,
                monitor=False, track_min_dist=True, artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
                save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")


def _save(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.json"
    json.dump(payload, open(p, "w"), indent=1, default=str)
    print(f"wrote {p}\n", flush=True)


def _earliest_local_max(absr, gamma):
    if absr.shape[0] == 1:
        return 0 if absr[0] >= gamma else -1
    left = np.r_[-np.inf, absr[:-1]]
    right = np.r_[absr[1:], -np.inf]
    ok = np.nonzero((absr >= left) & (absr > right) & (absr >= gamma))[0]
    return int(ok[0]) if ok.size else -1


# --------------------------------------------------------------------------- BRAID
BRAID_PAPER = {  # relative lag error in percent, TKDD 2010 Table (BRAID / ThinBRAID)
    "braid_sines": (0.000, 1.397), "braid_spiketrains": (0.387, 0.528), "motes_humidity": (0.024, 1.178),
    "motes_light": (0.529, 0.176), "sunspots_daily": (1.038, 0.086),
}


def repro_braid(args):
    """Relative lag error of BRAID's estimate (Definition 1 on the interpolated CCF, gamma=0.4,
    b=16) against the same Definition 1 applied to the *exact* CCF, which is what the paper's
    naive baseline computes; the planted lag (synthetic families, sunspot self-pair) only selects
    which pairs are scored. Percent, mean over (pair, window) with a reference lag > 0."""
    rows = []
    for name in args.datasets.split(","):
        data, ids, meta = load_raw(name)
        m = min(data.shape[0], args.m or data.shape[0])
        X = data[:m]
        X = np.where(np.isnan(X), 0.0, X)
        planted = {}
        if "planted_pairs" in meta:
            idx = {s: i for i, s in enumerate(ids)}
            planted = {(idx[p["a"]], idx[p["b"]]): p["lag"] for p in meta["planted_pairs"] if idx.get(p["a"], m) < m and idx.get(p["b"], m) < m}
        if name.startswith("sunspots"):
            lag = 4017                      # ~11 y in days: the solar cycle, BRAID's Sunspots lag scale
            X = np.vstack([X[0, lag:], X[0, :-lag]])
            planted = {(0, 1): lag}
            ids_use = ["sn", "sn_lagged"]
        else:
            ids_use = list(ids[:m])
        W, step, n_lags = args.W, args.step, args.n_lags
        if name.startswith("sunspots"):
            W, n_lags, step = 16384, 4200, 2048
        T_len = min(X.shape[1], args.T or X.shape[1])
        X = X[:, :T_len]
        res = {}
        for thin in (False, True):
            node = Candidates_BF_BRAID(W, step, n_lags, 0.0, neg_corr=True, b=16, gamma=0.4, thin=thin, thin_d0=400, report_mode="braid")
            errs, n_eval, t0 = [], 0, time.perf_counter()
            for s0 in range(0, T_len - T_len % step, step):
                node.run(np.vstack([np.arange(s0, s0 + step), X[:, s0:s0 + step]]), ids_use, verbose=False, testing=False)
                end = s0 + step
                if end < W + n_lags or node.last_lag_estimates is None:
                    continue
                est = node.last_lag_estimates
                pairs = planted.items() if planted else [((a, b), None) for a in range(X.shape[0]) for b in range(a + 1, X.shape[0])]
                cur = X[:, end - W:end]
                for (a, b), _planted in pairs:
                    # exact CCF reference (Definition 1, the naive baseline of the paper)
                    xa = cur[a] - cur[a].mean(); xa_n = np.linalg.norm(xa)
                    if xa_n == 0:
                        continue
                    ccf = []
                    for l in range(n_lags + 1):
                        yb = X[b, end - W - l:end - l]; yb = yb - yb.mean(); nb = np.linalg.norm(yb)
                        ccf.append(0.0 if nb == 0 else float(xa @ yb) / (xa_n * nb))
                    ref = _earliest_local_max(np.abs(np.asarray(ccf)), 0.4)
                    if ref <= 0:
                        continue
                    e = int(est[a, b]) if est[a, b] >= 0 else int(est[b, a])
                    if e < 0:
                        continue
                    errs.append((abs(e - ref), ref))
                    n_eval += 1
            arr = np.asarray(errs, dtype=float).reshape(-1, 2)
            big = arr[arr[:, 1] >= 16] if arr.size else arr
            res["thinbraid" if thin else "braid"] = dict(
                # |l_est - l_ref| / l_ref, only where the reference lag is >= 16 samples: on flat CCFs
                # (Motes: R(0)=0.9957 vs R(1)=0.9958) Definition 1 lands on 0 or 1 at random and a
                # lag-relative error is meaningless there
                rel_to_lag_pct_mean=float(np.mean(big[:, 0] / big[:, 1]) * 100) if big.size else None, n_ref_ge_16=int(big.shape[0]),
                # |l_est - l_ref| / max_lag over every scored (pair, window)
                rel_to_maxlag_pct_mean=float(np.mean(arr[:, 0]) / n_lags * 100) if arr.size else None,
                exact_hits_pct=float(np.mean(arr[:, 0] == 0) * 100) if arr.size else None, n=n_eval, wall=time.perf_counter() - t0)
        key = next((k for k in BRAID_PAPER if name.startswith(k)), None)
        paper = BRAID_PAPER.get(key, (None, None))
        def fmt(r):
            a = r["rel_to_lag_pct_mean"]; b = r["rel_to_maxlag_pct_mean"]
            return f"rel-to-lag {a:.3f}% (n={r['n_ref_ge_16']}) rel-to-maxlag {b:.3f}% exact-hit {r['exact_hits_pct']:.1f}%" if a is not None and b is not None else str(r)
        print(f"{name:22s} m={X.shape[0]} T={T_len} W={W} lags={n_lags}\n   BRAID     {fmt(res['braid'])}  | paper {paper[0]}%\n"
              f"   ThinBRAID {fmt(res['thinbraid'])}  | paper {paper[1]}%  (paper's normalization to be confirmed from the PDF)", flush=True)
        rows.append(dict(dataset=name, m=int(X.shape[0]), T=T_len, W=W, step=step, n_lags=n_lags, ours=res, paper_braid_pct=paper[0], paper_thinbraid_pct=paper[1],
                         reference="exact CCF earliest local max (Definition 1); pairs selected by planted lags" if planted else "exact CCF earliest local max (Definition 1), all pairs"))
    _save("braid", dict(experiment="BRAID TKDD 2010 relative lag error", rows=rows))


# --------------------------------------------------------------------------- CorrJoin
def repro_corrjoin(args):
    """r1 (fraction of pairs surviving the bucketing filter), candidate ratio and speedup ceiling
    1/r1 on CorrJoin's own files at the paper's W=1020, ks=15, ke=30, kb=3, stride 10, for T in
    {0.7, 0.8, 0.9, 0.95}. The paper's speedup figures are not transcribed in the plan; the
    reproducible claims are: speedup <= 1/r1, gain only for m > 100, and pruning vanishing as
    the correlated fraction approaches ~20% (their Fig. 15)."""
    rows = []
    W, step = 1020, 10
    for name in args.datasets.split(","):
        data, ids = load_dataset(name=name, max_series=args.m)
        test = data.T
        for T in (0.7, 0.8, 0.9, 0.95):
            ct = CorrTrack(window_size=W, basic_window=step, window_step=step, n_vectors=45, n_lags=0, corr_threshold=T, neg_corr=False,
                           exec="sequential", parallel_sketch=False, parallel_candidates=False, parallel_validation=False, numeric_rows=True,
                           data_representation="sketch_paa_svd", candidate_backend="corrjoin_double_filter", corrjoin_ks=15, corrjoin_ke=30, corrjoin_kb=3)
            r1s, t0 = [], time.perf_counter()
            for s0 in range(0, test.shape[1] - test.shape[1] % step, step):
                ct.run(test[:, s0:s0 + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
                ix = ct.grid_nodes[0]._lsh_index if ct.grid_nodes else None
                if ix is not None and ix.last_r1 is not None and s0 + step >= W:
                    r1s.append(ix.last_r1)
            wall = time.perf_counter() - t0
            with tempfile.TemporaryDirectory() as tmp:
                t1 = time.perf_counter()
                bf, _, _ = run_and_log_bruteforce(name, test, ids, dict(_base(W, step, 0, T), baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                                                  metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
                bf_wall = time.perf_counter() - t1
            dens = bf["correlated"] / max(bf["total_candidates"], 1)
            r1 = float(np.mean(r1s)) if r1s else None
            print(f"{name:20s} m={len(ids)} T={T}: r1={r1:.4f} ceiling 1/r1={1 / r1 if r1 else float('nan'):.1f}x  correlated fraction={dens:.4f}  "
                  f"wall corrjoin/bf={wall / bf_wall:.2f} (pure-Python index; counters are the comparison)", flush=True)
            rows.append(dict(dataset=name, m=len(ids), T=T, r1_mean=r1, speedup_ceiling=(1 / r1 if r1 else None), correlated_fraction=dens,
                             wall_corrjoin=wall, wall_bruteforce=bf_wall))
    _save("corrjoin", dict(experiment="CorrJoin PACMMOD 2023 r1 / pruning on the authors' files", W=W, step=step, ks=15, ke=30, kb=3,
                           paper="speedup bounded by 1/r1; gains only for m > 100; pruning indiscernible near 20% correlated (Fig. 15); speedup values not transcribed", rows=rows))


# --------------------------------------------------------------------------- StatStream
def repro_statstream(args):
    """StatStream VLDB 2002 Table 2 / Fig. 5 on its own random walks: grid pruning power
    (candidates / all pairs) from our StatStreamGridIndex, and the precision / recall of the
    paper's *approximate* reporting rule (report a pair when the DFT-approximated correlation is
    >= T - t) with n=16 coefficients, T in {0.85, 0.9}, t in {0.001, 0.0005}. Paper: precision
    0.9765 to 0.9947, recall 0.9987 to 1.0, pruning power 0.01 to 0.09."""
    rng = np.random.default_rng(20260917)
    m, T_len = args.m, args.T
    walks = 100.0 + np.cumsum(rng.uniform(0, 1, size=(m, T_len)) - 0.5, axis=1)
    ids = [f"rw{i}" for i in range(m)]
    # paper (plan section 4.2): sliding window 1,800 to 7,200 points at 1 s, basic window 0.5 to several minutes
    W, step, n = args.W if args.W != 1024 else 1800, args.step if args.step != 256 else 60, 16
    test = np.vstack([np.arange(T_len), walks])
    rows = []
    for T in (0.85, 0.9):
        # grid pruning power from the port (exact filter, recall 1 by Theorem 2)
        with tempfile.TemporaryDirectory() as tmp:
            rec, _, _ = run_and_log_corrtrack("statstream_rw", test, ids, _base(W, step, 0, T, basic_window=step),
                                              dict(n_vectors=2 * n, seed=1, seed_toggle=2, preprocess=False, data_representation="sketch_dft",
                                                   candidate_backend="statstream_grid", statstream_n_coeffs=n, statstream_index_dims=4),
                                              f"{tmp}/ss.csv", recall_by_window=True, verbose=False, testing=False)
        n_windows = (T_len - W) // step + 1
        all_pairs = n_windows * m * (m - 1) // 2
        pruning_power = rec["total_candidates"] / all_pairs
        # the paper's approximate reporting rule, evaluated on every window
        tp = {t: 0 for t in (0.001, 0.0005)}; fp = dict(tp); fn = dict(tp)
        for s0 in range(0, T_len - W + 1, step):
            win = walks[:, s0:s0 + W]
            z = win - win.mean(axis=1, keepdims=True)
            z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            exact = z @ z.T
            F = np.fft.rfft(z, axis=1)[:, 1:n + 1]           # first n coefficients after DC
            # the paper's rule: corr_approx = 1 - d_n^2 / 2 with d_n the distance between the n-coefficient
            # digests (Parseval scaling 2/W for the positive half-spectrum). Truncation can only drop
            # energy, so d_n <= d and corr_approx >= corr: recall ~1, precision < 1, as in their Table 2.
            if args.rule == "truncated":
                # digests as stored: 1 - d_n^2/2. Truncation only drops energy, so corr_approx >= corr:
                # recall exactly 1, precision limited by pairs just under T
                en = (2.0 / W) * np.sum(np.abs(F) ** 2, axis=1)
                d2 = en[:, None] + en[None, :] - 2.0 * (2.0 / W) * np.real(F @ F.conj().T)
                approx = 1.0 - d2 / 2.0
            else:
                # digests renormalized to unit norm (the cosine of the kept coefficients): a two-sided
                # approximation, which is the only reading consistent with the paper's recall < 1
                Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
                approx = np.real(Fn @ Fn.conj().T)
            iu = np.triu_indices(m, 1)
            ex, ap = exact[iu], approx[iu]
            truth = ex >= T
            for t in tp:
                rep = ap >= T - t
                tp[t] += int((rep & truth).sum()); fp[t] += int((rep & ~truth).sum()); fn[t] += int((~rep & truth).sum())
        for t in tp:
            prec = tp[t] / max(tp[t] + fp[t], 1); recall = tp[t] / max(tp[t] + fn[t], 1)
            print(f"m={m} T={T} t={t}: precision={prec:.4f} (paper 0.9765-0.9947) recall={recall:.4f} (paper 0.9987-1.0) | grid pruning power={pruning_power:.4f} (paper 0.01-0.09)", flush=True)
            rows.append(dict(m=m, T=T, tolerance=t, rule=args.rule, precision=prec, recall=recall, grid_pruning_power=pruning_power, grid_recall=1.0 if rec["correlated"] else None))
    _save("statstream", dict(experiment="StatStream VLDB 2002 Table 2 and Fig. 5 on random walks", W=W, basic_window=step, n_coeffs=n, rows=rows))


# --------------------------------------------------------------------------- ParCorr / CSZ
def repro_parcorr(args):
    """ParCorr DMKD 2018 section 5: recall > 90% at T=0.7, > 96% at 0.8, > 95.7% at 0.9 with r=60,
    k=2, f=0.7, w=500, b=20, precision 100% by construction. Their Yahoo Finance set is not
    obtainable; run on the stand-ins (sp500_sub263, corrjoin_stock)."""
    paper = {0.7: 90.0, 0.8: 96.0, 0.9: 95.7}
    rows = []
    for name in args.datasets.split(","):
        data, ids = load_dataset(name=name, max_series=args.m)
        test = data.T
        W, step = 500, 20
        if test.shape[1] < W + 10 * step:
            W, step = 250, 10
        for T in (0.7, 0.8, 0.9):
            with tempfile.TemporaryDirectory() as tmp:
                bf, _, bf_flags = run_and_log_bruteforce(name, test, ids, dict(_base(W, step, 0, T, basic_window=step), baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                                                         metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
                rec, _, flags = run_and_log_corrtrack(name, test, ids, _base(W, step, 0, T, basic_window=step),
                                                      dict(n_vectors=60, seed=1, seed_toggle=2, preprocess=False, data_representation="sketch_proj",
                                                           candidate_backend="parcorr_grid", parcorr_k=2, parcorr_f=0.7, parcorr_c=0.7),
                                                      f"{tmp}/pc.csv", recall_by_window=True, verbose=False, testing=False)
            mtr = CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf["total_candidates"])
            print(f"{name:16s} m={len(ids)} W={W} b={step} T={T}: recall={100 * mtr['recall']:.1f}% (paper > {paper[T]}%) precision={mtr['precision']:.3f} (paper 1.0) "
                  f"candidates/pairs={rec['total_candidates'] / max(bf['total_candidates'], 1):.3f}", flush=True)
            rows.append(dict(dataset=name, m=len(ids), W=W, b=step, T=T, recall=mtr["recall"], precision=mtr["precision"], paper_recall_min_pct=paper[T],
                             candidate_ratio=rec["total_candidates"] / max(bf["total_candidates"], 1), note="cell size c=0.7 (CSZ); the paper does not specify it"))
    _save("parcorr", dict(experiment="ParCorr DMKD 2018 recall at r=60, k=2, f=0.7 (stand-in data)", rows=rows))


# --------------------------------------------------------------------------- FilCorr
def repro_filcorr(args):
    """FilCorr ICDM 2020: data-independent throughput; 'up to 4x more sensors than naive' (a 4x
    sensor multiplier is ~16x less time per step at O(m^2)). White-noise streams at 100 Hz, the
    paper's 3-7 Hz band, W=2000 (20 s), lag 100 here (paper 1000; scaled for the local run), and
    the Yellowstone case study itself (28 stations, band-passed file)."""
    rows = []
    rng = np.random.default_rng(1)
    for m in [int(v) for v in args.ms.split(",")]:
        X = rng.standard_normal((m, args.T))
        test = np.vstack([np.arange(args.T), X]); ids = [f"s{i}" for i in range(m)]
        walls = {}
        with tempfile.TemporaryDirectory() as tmp:
            for mode, extra in (("bruteforce", {}), ("filcorr", dict(filcorr_fs=3.0, filcorr_ft=7.0, filcorr_sampling_rate=100.0))):
                t0 = time.perf_counter()
                rec, _, _ = run_and_log_bruteforce("wn", test, ids, dict(_base(2000, 200, args.n_lags, 0.5, neg_corr=True), baseline_mode=mode, **extra),
                                                   f"{tmp}/{mode}.csv", metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
                walls[mode] = time.perf_counter() - t0
        ratio = walls["bruteforce"] / walls["filcorr"]
        print(f"white noise m={m}: bruteforce {walls['bruteforce']:.1f}s filcorr(3-7 Hz) {walls['filcorr']:.1f}s -> time ratio {ratio:.2f}x, sensor multiplier {np.sqrt(ratio):.2f}x (paper: up to 4x)", flush=True)
        rows.append(dict(m=m, wall_bruteforce=walls["bruteforce"], wall_filcorr=walls["filcorr"], time_ratio=ratio, sensor_multiplier=float(np.sqrt(ratio))))
    # case study
    data, ids = load_dataset(name="yellowstone_bp3_7")
    test = data.T[:, : args.T]
    with tempfile.TemporaryDirectory() as tmp:
        rec, _, _ = run_and_log_bruteforce("yellowstone", test, ids, dict(_base(2000, 200, args.n_lags, 0.5, neg_corr=True), baseline_mode="filcorr",
                                                                        filcorr_fs=3.0, filcorr_ft=7.0, filcorr_sampling_rate=100.0),
                                           f"{tmp}/ys.csv", metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
    print(f"Yellowstone (28 stations, 3-7 Hz, W=2000, lag {args.n_lags}): {rec['correlated']} lagged pairs >= 0.5 over {rec['total_candidates']} pair-windows", flush=True)
    _save("filcorr", dict(experiment="FilCorr ICDM 2020 throughput vs naive and Yellowstone case study", paper="up to 4x more sensors than naive; beats ParCorr below ~700 streams",
                          n_lags=args.n_lags, rows=rows, yellowstone=dict(correlated=rec["correlated"], pair_windows=rec["total_candidates"])))


# --------------------------------------------------------------------------- TSUBASA
def repro_tsubasa(args):
    """TSUBASA SIGMOD 2022: exact Pearson from per-basic-window sketches, 'at least an order of
    magnitude faster than a raw-Pearson baseline' on NOAA / Berkeley Earth. Our bruteforce is an
    incremental Cython all-pairs kernel, not their raw recompute, so the exactness anchor is the
    reproducible part and the speed ratio is reported with that caveat."""
    rows = []
    for name in args.datasets.split(","):
        data, ids = load_dataset(name=name, max_series=args.m)
        test = data.T
        W, step = 168, 12
        for T in (0.7, 0.9):
            walls, counts = {}, {}
            with tempfile.TemporaryDirectory() as tmp:
                for mode in ("bruteforce", "tsubasa"):
                    t0 = time.perf_counter()
                    rec, _, _ = run_and_log_bruteforce(name, test, ids, dict(_base(W, step, 0, T, basic_window=step), baseline_mode=mode), f"{tmp}/{mode}.csv",
                                                       metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
                    walls[mode], counts[mode] = time.perf_counter() - t0, rec["correlated"]
            print(f"{name:24s} m={len(ids)} T={T}: exact match {'yes' if counts['tsubasa'] == counts['bruteforce'] else 'NO'} ({counts['tsubasa']} pairs); "
                  f"wall tsubasa/bruteforce={walls['tsubasa'] / walls['bruteforce']:.2f} (paper: >= 10x faster than *raw* Pearson recompute)", flush=True)
            rows.append(dict(dataset=name, m=len(ids), T=T, exact_match=counts["tsubasa"] == counts["bruteforce"], correlated=counts["tsubasa"], walls=walls))
    _save("tsubasa", dict(experiment="TSUBASA SIGMOD 2022 exactness and speed vs all-pairs", rows=rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=("braid", "corrjoin", "statstream", "parcorr", "filcorr", "tsubasa", "all"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--T", type=int, default=None, help="stream length cap (rows)")
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--n-lags", type=int, default=None)
    ap.add_argument("--ms", default="25,50,100,200")
    ap.add_argument("--rule", choices=("renormalized", "truncated"), default="renormalized", help="statstream: how the approximate correlation is formed from the n-coefficient digests")
    args = ap.parse_args()
    defaults = {
        "braid": dict(datasets="motes_humidity,motes_light,sunspots_daily,braid_sines_smoke,braid_spikes_smoke", n_lags=512, m=None),
        "corrjoin": dict(datasets="corrjoin_stock,corrjoin_chlorine,corrjoin_gas,corrjoin_synthetic", m=1000),
        "statstream": dict(m=500, T=20000),
        "parcorr": dict(datasets="sp500_sub263,corrjoin_stock", m=1000),
        "filcorr": dict(T=20000, n_lags=100),
        "tsubasa": dict(datasets="uscrn2020_temperature", m=None),
    }
    todo = list(defaults) if args.experiment == "all" else [args.experiment]
    for exp in todo:
        a = argparse.Namespace(**vars(args))
        for k, v in defaults[exp].items():
            if getattr(a, k, None) is None:
                setattr(a, k, v)
        print(f"\n=== {exp} ===", flush=True)
        globals()[f"repro_{exp}"](a)


if __name__ == "__main__":
    main()
