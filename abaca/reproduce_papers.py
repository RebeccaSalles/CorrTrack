"""Phase R: reproduce each competitor paper's own headline result on the paper's own data (or the
registry's stand-in) with OUR port, before the head-to-head. If a port cannot reproduce what its
authors published in the regime they published it, it is not a fair competitor yet; this is the
faithfulness check comparison plan section 6.5 asks for, run as an experiment rather than a unit
test. Every subcommand prints our number next to the paper's reported value and writes a JSON.

    python abaca/reproduce_papers.py braid       [--datasets braid_sines_paper,braid_spiketrains_paper,motes_temperature,sunspots_daily] [--T 30000]
    python abaca/reproduce_papers.py corrjoin    [--datasets corrjoin_stock,corrjoin_chlorine,corrjoin_gas,corrjoin_synthetic] [--m 1000]
    python abaca/reproduce_papers.py statstream  [--m 500] [--T 20000]
    python abaca/reproduce_papers.py parcorr     [--datasets sp500_sub263,corrjoin_stock] [--m 1000]   (stated settings AND their calibration)
    python abaca/reproduce_papers.py csz         [--datasets sp500_sub263,csz_steamgen]  (the CSZ protocol at the 0.99 target)
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


def _earliest_local_max(absr, gamma, radius=1):
    """Definition 1: earliest lag whose score is >= gamma and is a local maximum. `radius` widens
    the neighbourhood the maximum must dominate: with radius=1 the exact CCF of noisy data has
    micro-wiggles of order 1/sqrt(n) that create spurious "earliest" maxima on the shoulder of the
    true peak (found on the Sines pilot: naive 634 vs the true peak 668, |R| differing by 0.002).
    The paper's naive baseline is not specified beyond Definition 1; we use radius = b = 16."""
    L = absr.shape[0]
    if L == 1:
        return 0 if absr[0] >= gamma else -1
    ok = absr >= gamma
    for d in range(1, radius + 1):
        left = np.r_[np.full(d, -np.inf), absr[:-d]]
        right = np.r_[absr[d:], np.full(d, -np.inf)]
        ok &= (absr >= left) & (absr > right)
    idx = np.nonzero(ok)[0]
    return int(idx[0]) if idx.size else -1


# --------------------------------------------------------------------------- BRAID
# TKDD 2010 Tables III (BRAID) and IV (ThinBRAID): naive lag, method lag, error E = 100 |l_b - l_n| / l_n (Eq. 32)
BRAID_PAPER = {
    "sines": dict(naive=716, braid=716, thin=706, err_braid=0.000, err_thin=1.397, n=32768),
    "spiketrains": dict(naive=2841, braid=2830, thin=2826, err_braid=0.387, err_thin=0.528, n=100000),
    "humidity": dict(naive=4160, braid=4161, thin=4209, err_braid=0.024, err_thin=1.178, n=50000),
    "light": dict(naive=567, braid=570, thin=566, err_braid=0.529, err_thin=0.176, n=32000),
    "sunspots": dict(naive=1156, braid=1168, thin=1155, err_braid=1.038, err_thin=0.086, n=25900),
    "motes": dict(naive=None, braid=None, thin=None, err_braid=None, err_thin=None, n=30000, note="section 6.5: #1/#10 lag 202 min, #47/#48 lag 224 min (390 and 433 epochs of 31 s)"),
}


def _exact_ccf_def1(x_cur, Y, end, W, max_lag, gamma):
    """Definition 1 on the exact CCF, same windowing as our BRAID port: R(l) = corr(x[end-W:end], y[end-W-l:end-l])."""
    xa = x_cur - x_cur.mean(); xn = np.linalg.norm(xa)
    if xn == 0:
        return -1, None
    ccf = np.empty(max_lag + 1)
    for l in range(max_lag + 1):
        yb = Y[end - W - l:end - l]; yb = yb - yb.mean(); nb = np.linalg.norm(yb)
        ccf[l] = 0.0 if nb == 0 else float(xa @ yb) / (xn * nb)
    return _earliest_local_max(np.abs(ccf), gamma, radius=16), ccf


def repro_braid(args):
    """The paper's regime, not a sliding-window one: one CCF over the whole sequence prefix (their n is
    the full length, max lag m = n/2), Definition 1 (earliest local max of |R| >= gamma = 0.4), b = 16,
    d = 400 / 2^h for ThinBRAID, error E = 100 |l_b - l_n| / l_n against the naive lag (Eq. 32). In our
    port the "whole sequence" is a window of W = n/2 compared with the history shifted by up to n/2, so
    the full prefix is used. Pairs: planted pairs for the synthetic families, Motes #1/#10 and #47/#48
    (section 6.5), contiguous 25,900-day chunks of the sunspot series as separate sequences (their
    construction), all pairs for the humidity/light stand-ins."""
    rows = []
    for name in args.datasets.split(","):
        data, ids, meta = load_raw(name)
        ids = list(ids)
        key = next((k for k in BRAID_PAPER if k in name), None)
        paper = BRAID_PAPER.get(key, {})
        X = np.where(np.isnan(data), 0.0, data)
        if key == "sunspots":
            L = 25900
            chunks = [X[0, i * L:(i + 1) * L] for i in range(X.shape[1] // L)]
            X = np.vstack(chunks); ids = [f"sunspots#{i + 1}" for i in range(len(chunks))]
            pairs = [(i, j) for i in range(len(chunks)) for j in range(i + 1, len(chunks))]
        elif key == "motes":
            want = [("mote01", "mote10"), ("mote47", "mote48")]
            pairs = [(ids.index(a), ids.index(b)) for a, b in want if a in ids and b in ids]
        elif "planted_pairs" in meta:
            idx = {s: i for i, s in enumerate(ids)}
            pairs = [(idx[p["a"]], idx[p["b"]]) for p in meta["planted_pairs"]]
        else:
            pairs = [(i, j) for i in range(X.shape[0]) for j in range(i + 1, X.shape[0])]
        n = min(X.shape[1], args.T or paper.get("n") or X.shape[1])
        # the harness rounds n_lags down to a multiple of the step, so pick step = n/20 and make
        # W and max_lag multiples of it: W = n/2, max_lag = W - step (the paper's m = n/2)
        step = max(1, n // 20)
        W = (n // 2) // step * step
        max_lag = W - step
        n = 2 * W
        X = X[:, :n]
        keep = sorted({i for p in pairs for i in p})
        X = X[keep]; ids_use = [ids[i] for i in keep]; remap = {i: k for k, i in enumerate(keep)}
        pairs = [(remap[a], remap[b]) for a, b in pairs]
        res = {}
        for thin in (False, True):
            node = Candidates_BF_BRAID(W, step, max_lag, 0.0, neg_corr=True, b=16, gamma=0.4, thin=thin, thin_d0=400, report_mode="braid")
            t0 = time.perf_counter()
            for s0 in range(0, n - n % step, step):
                node.run(np.vstack([np.arange(s0, s0 + step), X[:, s0:s0 + step]]), ids_use, verbose=False, testing=False)
            est = node.last_lag_estimates
            per_pair = []
            for a, b in pairs:
                # both orientations, as the paper reports whichever sequence lags; pick the orientation
                # in which the naive Definition 1 fires, preferring the one with the larger lag
                cands = []
                for (p, q) in ((a, b), (b, a)):
                    ref, ccf = _exact_ccf_def1(X[p, n - W:n], X[q], n, W, max_lag, 0.4)
                    if ref > 0 and est is not None and est[p, q] >= 0:
                        cands.append((ref, int(est[p, q]), f"{ids_use[p]} lags {ids_use[q]}", float(abs(ccf[ref]))))
                if not cands:
                    per_pair.append(dict(pair=(ids_use[a], ids_use[b]), status="no lag correlation found by naive Definition 1"))
                    continue
                ref, e, orient, score = max(cands, key=lambda c: c[0])
                per_pair.append(dict(pair=(ids_use[a], ids_use[b]), orientation=orient, naive_lag=ref, method_lag=e, score=score,
                                     error_pct=100.0 * abs(e - ref) / ref))
            errs = [r["error_pct"] for r in per_pair if "error_pct" in r]
            res["thinbraid" if thin else "braid"] = dict(pairs=per_pair, error_pct_mean=float(np.mean(errs)) if errs else None,
                                                        error_pct_max=float(np.max(errs)) if errs else None, wall=time.perf_counter() - t0)
        pb, pt = res["braid"], res["thinbraid"]
        print(f"{name:26s} n={n} W={W} max_lag={max_lag} pairs={len(pairs)}", flush=True)
        for r_b, r_t in zip(pb["pairs"], pt["pairs"]):
            if "error_pct" in r_b:
                print(f"   {r_b['orientation']:28s} naive {r_b['naive_lag']:6d} | BRAID {r_b['method_lag']:6d} E={r_b['error_pct']:.3f}% | ThinBRAID {r_t.get('method_lag', -1):6d} E={r_t.get('error_pct', float('nan')):.3f}%", flush=True)
            else:
                print(f"   {r_b['pair']}: {r_b['status']}", flush=True)
        print(f"   paper: naive {paper.get('naive')} BRAID {paper.get('braid')} E={paper.get('err_braid')}% | ThinBRAID {paper.get('thin')} E={paper.get('err_thin')}%  {paper.get('note', '')}", flush=True)
        rows.append(dict(dataset=name, paper_key=key, n=n, W=W, max_lag=max_lag, ours=res, paper=paper))
    _save("braid", dict(experiment="BRAID TKDD 2010 Tables III and IV: lag error E = 100|l_b - l_n|/l_n over the whole sequence", rows=rows))


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
    """StatStream VLDB 2002 section 5 on its own random walks (s_i = 100 + sum(u - 0.5)): sliding
    window 1 h = 3,600 points, basic windows 30 s to several minutes (b = 60 here), n = 16 sliding-window
    DFT coefficients for the grid (Lemma 2 filter, Theorem 2: no false negatives).
    Reported: Fig. 5 grid pruning power 0.01 to 0.09 and filter precision ~0.55 to 0.9 (16 to 40
    coefficients); Table 2 after post-processing: S0.85, t = 0.0005 -> precision 0.9931, recall 1.0
    (the real R0.85/R0.9 cells: 0.9765 to 0.9947 / 0.9987 to 1.0). The post-processing approximation
    is section 3.4 curve fitting with the first 2 DFT coefficients of EACH BASIC WINDOW (Fig. 4
    caption), summed over the k basic windows of the sliding window, with exact means and standard
    deviations from the running sums; a pair is reported when corr_approx > T - t."""
    rng = np.random.default_rng(20260917)
    m, T_len = args.m, args.T
    walks = 100.0 + np.cumsum(rng.uniform(0, 1, size=(m, T_len)) - 0.5, axis=1)
    ids = [f"rw{i}" for i in range(m)]
    W = args.W if args.W != 1024 else 3600
    b = args.step if args.step != 256 else 60
    n_grid, n_bw = 16, 2
    test = np.vstack([np.arange(T_len), walks])
    rows = []
    for T in (0.85, 0.9):
        with tempfile.TemporaryDirectory() as tmp:
            rec, _, _ = run_and_log_corrtrack("statstream_rw", test, ids, _base(W, b, 0, T, basic_window=b),
                                              dict(n_vectors=2 * n_grid, seed=1, seed_toggle=2, preprocess=False, data_representation="sketch_dft",
                                                   candidate_backend="statstream_grid", statstream_n_coeffs=n_grid, statstream_index_dims=4),
                                              f"{tmp}/ss.csv", recall_by_window=True, verbose=False, testing=False)
        n_windows = (T_len - W) // b + 1
        all_pairs = n_windows * m * (m - 1) // 2
        pruning_power = rec["total_candidates"] / all_pairs
        filter_precision = rec["correlated"] / max(rec["total_candidates"], 1)     # their Fig. 5 "precision" (before post-processing)
        tp = {t: 0 for t in (0.001, 0.0005)}; fp = dict(tp); fn = dict(tp)
        k = W // b
        for s0 in range(0, T_len - W + 1, b):
            win = walks[:, s0:s0 + W]
            mu = win.mean(axis=1); sd = win.std(axis=1)
            exact = np.corrcoef(win)
            # section 3.4: per basic window, DFT (1/sqrt(b) convention) truncated to the first n_bw coefficients;
            # psi(x, y) ~ sum_j sum_m c^x_m conj(c^y_m) (orthogonal family, V(m) absorbed); real signal: DC + 2 Re(...)
            blocks = win.reshape(m, k, b)
            F = np.fft.rfft(blocks, axis=2)[:, :, :n_bw] / np.sqrt(b)
            ip = np.real(np.einsum("ikm,jkm->ij", F[:, :, :1], F[:, :, :1].conj())) + 2.0 * np.real(np.einsum("ikm,jkm->ij", F[:, :, 1:], F[:, :, 1:].conj()))
            approx = (ip / W - np.outer(mu, mu)) / np.maximum(np.outer(sd, sd), 1e-12)
            iu = np.triu_indices(m, 1)
            ex, ap = exact[iu], approx[iu]
            truth = ex >= T
            for t in tp:
                rep = ap > T - t
                tp[t] += int((rep & truth).sum()); fp[t] += int((rep & ~truth).sum()); fn[t] += int((~rep & truth).sum())
        for t in tp:
            prec = tp[t] / max(tp[t] + fp[t], 1); recall = tp[t] / max(tp[t] + fn[t], 1)
            paper = "S0.85 t=0.0005: precision 0.9931 recall 1.0" if (T == 0.85 and t == 0.0005) else "no synthetic cell in Table 2 (real: 0.9765-0.9947 / 0.9987-1.0)"
            print(f"m={m} W={W} b={b} T={T} t={t}: post-processing precision={prec:.4f} recall={recall:.4f} [{paper}] | grid pruning power={pruning_power:.4f} (Fig. 5: 0.01-0.09) "
                  f"filter precision={filter_precision:.3f} (Fig. 5: ~0.55-0.9 at 16 coefficients)", flush=True)
            rows.append(dict(m=m, W=W, b=b, T=T, tolerance=t, n_bw=n_bw, precision=prec, recall=recall, grid_pruning_power=pruning_power, filter_precision=filter_precision))
    _save("statstream", dict(experiment="StatStream VLDB 2002 Fig. 5 and Table 2 on random walks", W=W, basic_window=b, n_grid=n_grid, n_bw=n_bw, rows=rows))


# --------------------------------------------------------------------------- ParCorr / CSZ
def _split_train_test(test, ratio=0.3):
    cut = int(round(ratio * test.shape[1]))
    return test[:, :cut], test[:, cut:]


def _fit_window(n_rows_calib, W, step, min_windows=20):
    """Halve W (and step) until the calibration span holds at least `min_windows` windows; a
    calibration span shorter than a window has no ground truth and would make every setting
    'feasible' with recall 0/0 (seen on sp500 with W = 500 and a 376-row span)."""
    while (n_rows_calib - W) // step + 1 < min_windows and W > 20:
        W //= 2; step = max(1, step // 2)
    return W, step


def _calibrate_grid(label, calib, ids, base, arm, grid_settings, target, floor=0.02):
    """Run every setting on the calibration span against its bruteforce truth; return the settings
    sorted by the CSZ rule (feasible first: recall >= target and precision >= floor; then fewer
    candidates) plus the evaluations. Uses abaca/tune_competitors.py's machinery."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tune_competitors", os.path.join(REPO, "abaca", "tune_competitors.py"))
    tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
    with tempfile.TemporaryDirectory() as tmp:
        gt, _, gt_flags = run_and_log_bruteforce(label, calib, ids, dict(base, baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                                                 metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
        if not gt["correlated"]:
            raise RuntimeError(f"{label}: the calibration span has no correlated pair at T={base['corr_threshold']} (windows={(calib.shape[1] - base['window_size']) // base['window_step'] + 1}); nothing to calibrate against")
        ev = tc.Evaluator(label, calib, ids, base, tmp, gt_flags, gt["total_candidates"])
        evals = [ev.evaluate(arm, st) for st in grid_settings]
    ranked = sorted(evals, key=lambda r: tc.score(r, target))
    return ranked, tc


def _heldout(label, test, ids, base, run_params):
    with tempfile.TemporaryDirectory() as tmp:
        bf, _, bf_flags = run_and_log_bruteforce(label, test, ids, dict(base, baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                                                 metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
        rec, _, flags = run_and_log_corrtrack(label, test, ids, base, run_params, f"{tmp}/run.csv", recall_by_window=True, verbose=False, testing=False)
    m = CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf["total_candidates"])
    return dict(recall=m["recall"], precision=m["precision"], candidate_ratio=rec["total_candidates"] / max(bf["total_candidates"], 1))


def repro_parcorr(args):
    """ParCorr DMKD 2018 section 5: recall > 90% at T=0.7, > 96% at 0.8, > 95.7% at 0.9 with r=60,
    k=2, f=0.7, w=500, b=20, precision 100% by construction. Two rows per (dataset, T):
      (a) the paper's stated settings with the cell size at CSZ's c = 0.7 (the paper gives none);
      (b) the paper's OWN calibration procedure (section 5.3: fix r, k; calibrate on a small sample
          until the desired recall) applied to the two quantities it leaves to the sample, the cell
          size c and the fraction f, on the first 30% of the stream at the 0.95 target, then recall on
          the remaining 70%. (b) is the reproduction of their protocol; (a) of their numbers as printed."""
    paper = {0.7: 90.0, 0.8: 96.0, 0.9: 95.7}
    rows = []
    c_grid = [round(0.1 * i, 1) for i in range(1, 21)]; f_grid = [round(0.1 * i, 1) for i in range(5, 11)]   # c beyond CSZ 1.3: ParCorr states no cell size
    for name in args.datasets.split(","):
        data, ids = load_dataset(name=name, max_series=args.m)
        test = data.T
        calib, hold = _split_train_test(test)
        W, step = _fit_window(calib.shape[1], 500, 20)      # the paper's w = 500, b = 20 when the stream allows
        for T in (0.7, 0.8, 0.9):
            base = _base(W, step, 0, T, basic_window=step)
            stated = dict(n_vectors=60, seed=1, seed_toggle=2, preprocess=False, data_representation="sketch_proj",
                          candidate_backend="parcorr_grid", parcorr_k=2, parcorr_f=0.7, parcorr_c=0.7)
            a = _heldout(name, hold, ids, base, stated)
            settings = [dict(n_vectors=60, parcorr_k=2, parcorr_c=c, parcorr_f=f) for c in c_grid for f in f_grid]
            ranked, tc = _calibrate_grid(name, calib, ids, base, "parcorr", settings, 0.95)
            best = ranked[0]
            b = _heldout(name, hold, ids, base, tc.run_params_for("parcorr", best["setting"]))
            print(f"{name:16s} m={len(ids)} W={W} b={step} T={T}: (a) stated r=60,k=2,f=0.7,c=0.7 -> recall={100 * a['recall']:.1f}% cand/pairs={a['candidate_ratio']:.3f} | "
                  f"(b) their calibration -> c={best['setting']['parcorr_c']} f={best['setting']['parcorr_f']} (calib recall {best['recall']:.3f}) held-out recall={100 * b['recall']:.1f}% "
                  f"cand/pairs={b['candidate_ratio']:.3f} | paper > {paper[T]}%, precision {a['precision']:.3f}/{b['precision']:.3f}", flush=True)
            rows.append(dict(dataset=name, m=len(ids), W=W, b=step, T=T, stated=a, calibrated=dict(setting=best["setting"], calib_recall=best["recall"], heldout=b), paper_recall_min_pct=paper[T]))
    _save("parcorr", dict(experiment="ParCorr DMKD 2018 recall: stated settings vs their own calibration protocol (c, f on a 30% sample)", rows=rows))


def repro_csz(args):
    """Cole, Shasha, Zhao KDD 2005 section 5.4/6: with sw=256, bw=32, the combinatorial-design +
    refinement + bootstrap protocol over N in {30,36,48,60}, g in {1..4}, c in {0.1..1.3}, f in
    {0.1..1.0} reaches recall >= 0.99 at precision >= 0.02 on their sets (CRSP, 10 UCR sets).
    Their data are unavailable; run the protocol (our tuner's 130-row covering array, target 0.99)
    on sp500 and the DaISy stand-ins, calibrate on 30%, report held-out recall / precision."""
    rows = []
    for name in args.datasets.split(","):
        data, ids = load_dataset(name=name, max_series=args.m)
        test = data.T
        calib, hold = _split_train_test(test)
        W, step = _fit_window(calib.shape[1], 256, 32)      # the paper's sw = 256, bw = 32 when the stream allows
        for T in (0.7, 0.9):
            base = _base(W, step, 0, T, basic_window=step)
            import importlib.util
            spec = importlib.util.spec_from_file_location("tune_competitors", os.path.join(REPO, "abaca", "tune_competitors.py"))
            tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
            design = tc.covering_array(tc.parameter_grid("csz", W), "csz")
            ranked, tc = _calibrate_grid(name, calib, ids, base, "csz", design, 0.99)
            best = ranked[0]
            feasible = tc.score(best, 0.99)[0] == 0
            b = _heldout(name, hold, ids, base, tc.run_params_for("csz", best["setting"]))
            print(f"{name:16s} m={len(ids)} W={W} b={step} T={T}: 130-row design, best {best['setting']} calib recall={best['recall']:.4f} prec={best['precision']:.3f} "
                  f"{'FEASIBLE' if feasible else 'no setting reaches 0.99'} | held-out recall={b['recall']:.4f} precision={b['precision']:.3f} cand/pairs={b['candidate_ratio']:.3f} | paper: recall >= 0.99, precision >= 0.02", flush=True)
            rows.append(dict(dataset=name, m=len(ids), W=W, b=step, T=T, best=best["setting"], calib_recall=best["recall"], calib_precision=best["precision"], feasible=feasible, heldout=b))
    _save("csz", dict(experiment="Cole-Shasha-Zhao KDD 2005 tuning protocol at the 0.99 target (stand-in data)", rows=rows))


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
    ys_lags = 1000     # the paper's 10 s at 100 Hz (the synthetic throughput sweep above uses --n-lags for cost)
    with tempfile.TemporaryDirectory() as tmp:
        rec, _, _ = run_and_log_bruteforce("yellowstone", test, ids, dict(_base(2000, 200, ys_lags, 0.5, neg_corr=True), baseline_mode="filcorr",
                                                                        filcorr_fs=3.0, filcorr_ft=7.0, filcorr_sampling_rate=100.0),
                                           f"{tmp}/ys.csv", metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
    print(f"Yellowstone (28 stations, 3-7 Hz, W=2000, lag {ys_lags}): {rec['correlated']} lagged pairs >= 0.5 over {rec['total_candidates']} pair-windows", flush=True)
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
    ap.add_argument("experiment", choices=("braid", "corrjoin", "statstream", "parcorr", "csz", "filcorr", "tsubasa", "all"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--T", type=int, default=None, help="stream length cap (rows)")
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--n-lags", type=int, default=None)
    ap.add_argument("--ms", default="25,50,100,200")
    args = ap.parse_args()
    defaults = {
        "braid": dict(datasets="braid_sines_paper,braid_spiketrains_paper,motes_temperature,motes_humidity,motes_light,sunspots_daily", m=None),
        "corrjoin": dict(datasets="corrjoin_stock,corrjoin_chlorine,corrjoin_gas,corrjoin_synthetic", m=1000),
        "statstream": dict(m=500, T=20000),
        "parcorr": dict(datasets="sp500_sub263,corrjoin_stock", m=1000),
        "csz": dict(datasets="sp500_sub263,csz_steamgen", m=1000),
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
