"""Tune the pruning competitor arms with Cole, Shasha, Zhao's published protocol (KDD 2005,
section 5.4; comparison plan section 4a.1), decided 2026-09-16 for every competitor arm.

Three stages, all on the *calibration span* CorrTrack's own hyperopt uses (the first
TRAIN_RATIO of the stream, ``prepare_training_data``), to the same TARGET_RECALL:

1. **Two-factor combinatorial design**: a strength-2 covering array over the arm's parameter
   grid (every pair of levels of every two factors appears in at least one setting). For
   ParCorr/CSZ the grid is CSZ's own: N in {30, 36, 48, 60}, g in {1, 2, 3, 4},
   c in {0.1 .. 1.3}, f in {0.1 .. 1.0} = 2,080 settings, covered by 130 rows, as in the paper.
   Grids smaller than ``--full-factorial-max`` settings are enumerated in full.
2. **Local neighbourhood refinement**: from the best design row, evaluate every setting that
   moves one factor by one level (coordinate neighbourhood); accept the best improvement;
   repeat until none (max ``--refine-rounds``).
3. **Bootstrap**: recall of the chosen setting on ``--bootstrap-blocks`` contiguous window
   blocks of the calibration span, resampled ``--bootstrap-repeats`` times; the 90% lower
   confidence bound must reach the target (CorrTrack's rule), else the next-best feasible
   setting is tried.

Selection rule (mirrors corrtrack_param_search.py): feasible = bootstrap recall lower bound
>= target and precision >= 0.02 (CSZ's floor); among feasible settings minimize
``total_candidates`` (the implementation-independent counter), ties -> higher precision.
Exact-recall arms (StatStream, CorrJoin; recall is 1.0 by construction) thus get the fastest
filter setting. Each setting is one CorrTrack run on the calibration span, so cap the span
with ``--calib-obs`` on large streams.

Output: ``<out-dir>/best_params_<arm>.json`` (the run_params to merge into the arm) and
``<out-dir>/tuning_<arm>.json`` (every evaluated setting, stage by stage). By default out-dir
is the ``optim`` folder corrtrack_run_corrtrack.py reads, so the tuned arms sit next to
CorrTrack's own best_params.

    python abaca/tune_competitors.py --dataset-config experiment_dataset_motes_temperature.py \
        --arms parcorr,csz,statstream,corrjoin --window-size 96 --window-step 12 \
        --n-lags 0 --corr-threshold 0.9 --calib-obs 3000
"""
from __future__ import annotations

import argparse
import itertools
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

import corrtrack_param_search as psmod  # noqa: E402  (calibration-span helpers)
import corrtrack_run_bruteforce as bfmod  # noqa: E402
from library_corrtrack_parallel import CorrTrack, run_and_log_bruteforce, run_and_log_corrtrack  # noqa: E402

ARMS = ("parcorr", "csz", "statstream", "corrjoin")
PRECISION_FLOOR = 0.02      # CSZ section 5.4


# --------------------------------------------------------------------------- parameter grids
def divisors(n, lo, hi):
    return [d for d in range(lo, hi + 1) if n % d == 0]


def parameter_grid(arm, W, b=None):
    if arm in ("parcorr", "csz"):
        return {
            "n_vectors": [30, 36, 48, 60],
            "parcorr_k": [1, 2, 3, 4],
            "parcorr_c": [round(0.1 * i, 1) for i in range(1, 14)],
            "parcorr_f": [round(0.1 * i, 1) for i in range(1, 11)],
        }
    if arm == "statstream":
        # (2026-09-19, refined 2026-09-22) statstream_bw_coeffs: DFT coefficients kept per basic window for the
        # approximate correlation StatStream REPORTS (the paper fixes 2, tuned on random walks). The reported
        # value is scaled by the share of the basic window's energy the kept coefficients carry, so on data whose
        # spectrum is not concentrated at the bottom the rule misses most true pairs: USCRN hourly temperature,
        # W=168, b=12, T=0.8, same candidates, recall 0.037 (differenced) / 0.254 (raw) at 2 coefficients against
        # 0.993 / 1.000 at the LOSSLESS size b/2+1 (rfft returns b/2+1 coefficients; the old cap of b/2 dropped
        # the Nyquist term and stopped at recall 0.959 / 0.948). Grid up to b/2+1 so the lossless setting is
        # reachable; precision falls as the size grows (0.915 at b/2+1 vs 0.998 at 2) because corr_approx > T - t
        # then also admits pairs just below T, which is the paper's own trade-off.
        full = max(1, int(b or 12) // 2 + 1)
        bw = sorted({min(v, full) for v in (2, 3, 4, 6, full)})
        return {"statstream_n_coeffs": [4, 8, 12, 16, 24, 32], "statstream_index_dims": [1, 2, 3, 4], "statstream_bw_coeffs": bw}
    if arm == "corrjoin":
        ds = divisors(W, 4, W // 2)
        return {"corrjoin_ks": ds, "corrjoin_ke": ds, "corrjoin_kb": [2, 3, 4]}
    raise ValueError(arm)


def valid_setting(arm, s):
    if arm in ("parcorr", "csz"):
        return s["n_vectors"] % s["parcorr_k"] == 0
    if arm == "statstream":
        return s["statstream_index_dims"] <= 2 * s["statstream_n_coeffs"]
    if arm == "corrjoin":
        return s["corrjoin_ks"] < s["corrjoin_ke"] and s["corrjoin_kb"] <= s["corrjoin_ks"]
    return True


def covering_array(grid, arm, seed=0):
    """Greedy strength-2 covering array over ``grid`` (dict factor -> levels), restricted to
    valid settings. Every pair of levels of every pair of factors is covered at least once."""
    factors = list(grid)
    levels = {f: list(grid[f]) for f in factors}
    pairs_needed = set()
    for a, b in itertools.combinations(factors, 2):
        for la in levels[a]:
            for lb in levels[b]:
                pairs_needed.add((a, la, b, lb))
    # seed rows: the full grid of the two largest factors (CSZ: c x f = 130 rows)
    big = sorted(factors, key=lambda f: -len(levels[f]))[:2]
    rest = [f for f in factors if f not in big]
    rows = []
    rng = np.random.default_rng(seed)

    def pairs_of(s):
        return {(a, s[a], b, s[b]) for a, b in itertools.combinations(factors, 2)}

    for la, lb in itertools.product(levels[big[0]], levels[big[1]]):
        base = {big[0]: la, big[1]: lb}
        best, best_gain = None, -1
        for combo in itertools.product(*(levels[f] for f in rest)):
            s = dict(base, **dict(zip(rest, combo)))
            if not valid_setting(arm, s):
                continue
            gain = len(pairs_of(s) & pairs_needed) + rng.uniform(0, 0.5)
            if gain > best_gain:
                best, best_gain = s, gain
        if best is not None:
            rows.append(best)
            pairs_needed -= pairs_of(best)
    # top up until every pair is covered (pairs only reachable through invalid settings are dropped)
    all_settings = [dict(zip(factors, c)) for c in itertools.product(*(levels[f] for f in factors))]
    all_settings = [s for s in all_settings if valid_setting(arm, s)]
    reachable = set().union(*(pairs_of(s) for s in all_settings)) if all_settings else set()
    pairs_needed &= reachable
    while pairs_needed:
        s = max(all_settings, key=lambda s: len(pairs_of(s) & pairs_needed))
        rows.append(s)
        pairs_needed -= pairs_of(s)
    return rows


def neighbours(grid, arm, s):
    out = []
    for f, lv in grid.items():
        i = lv.index(s[f])
        for j in (i - 1, i + 1):
            if 0 <= j < len(lv):
                t = dict(s, **{f: lv[j]})
                if valid_setting(arm, t):
                    out.append(t)
    return out


PREPROCESS = False      # set from --preprocess in main(); every evaluated setting runs in the same space as the truth


def run_params_for(arm, s, seed=2468):
    common = dict(seed=seed, seed_toggle=1357, preprocess=bool(PREPROCESS), n_vectors=int(s.get("n_vectors", 32)))
    if arm in ("parcorr", "csz"):
        return dict(common, data_representation="sketch_proj", candidate_backend="parcorr_grid", parcorr_k=int(s["parcorr_k"]),
                    parcorr_c=float(s["parcorr_c"]), parcorr_f=float(s["parcorr_f"]), parcorr_neighbor_probe=(arm == "csz"))
    if arm == "statstream":
        return dict(common, data_representation="sketch_dft", candidate_backend="statstream_grid",
                    statstream_n_coeffs=int(s["statstream_n_coeffs"]), statstream_index_dims=int(s["statstream_index_dims"]),
                    statstream_bw_coeffs=int(s.get("statstream_bw_coeffs", 2)))
    if arm == "corrjoin":
        return dict(common, data_representation="sketch_paa_svd", candidate_backend="corrjoin_double_filter",
                    corrjoin_ks=int(s["corrjoin_ks"]), corrjoin_ke=int(s["corrjoin_ke"]), corrjoin_kb=int(s["corrjoin_kb"]))
    raise ValueError(arm)


# --------------------------------------------------------------------------- evaluation
class Evaluator:
    def __init__(self, label, calib_data, ids, base, tmp, gt_flags, gt_total):
        self.label, self.data, self.ids, self.base, self.tmp = label, calib_data, ids, base, tmp
        self.gt = CorrTrack._as_row_corr_pair(gt_flags)
        self.gt_total = gt_total
        self.cache = {}
        self.n_runs = 0

    @staticmethod
    def _key(s):
        return tuple(sorted(s.items()))

    def evaluate(self, arm, s):
        k = (arm, self._key(s))
        if k in self.cache:
            return self.cache[k]
        rp = run_params_for(arm, s)
        t0 = time.perf_counter()
        try:
            record, _, flags = run_and_log_corrtrack(self.label, self.data, self.ids, self.base, rp, os.path.join(self.tmp, f"{arm}.csv"),
                                                     metadata={"nodes": 0, "alg": arm}, recall_by_window=True, corr_val=True,
                                                     monitor=False, verbose=False, testing=False)
        except Exception as exc:  # noqa: BLE001
            res = dict(setting=s, status="error", reason=f"{type(exc).__name__}: {exc}", recall=0.0, precision=0.0, total_candidates=None)
            self.cache[k] = res
            return res
        wall = time.perf_counter() - t0
        pred = CorrTrack._as_row_corr_pair(flags)
        m = CorrTrack.compute_metrics_bf(pred, self.gt, windows=True, total_pairs_bf=self.gt_total)
        res = dict(setting=s, status="ok", recall=float(m.get("recall") or 0.0), precision=float(m.get("precision") or 0.0),
                   total_candidates=int(record["total_candidates"]), tested=int(record["tested"]), correlated=int(record["correlated"]),
                   wall=wall, _pred=pred)
        self.cache[k] = res
        self.n_runs += 1
        return res

    def bootstrap_recall(self, pred, blocks, repeats, seed=2468):
        """Recall per contiguous window block (by the later window time), resampled with replacement."""
        gt_rows, gt_corr = self.gt
        if gt_rows.shape[0] == 0:
            return dict(mean=None, lower=None, per_block=[])
        times = gt_rows[:, 2]
        edges = np.quantile(times, np.linspace(0, 1, blocks + 1))
        edges[-1] += 1
        per_block = []
        p_rows, p_corr = pred
        for b in range(blocks):
            g_mask = (times >= edges[b]) & (times < edges[b + 1])
            if not g_mask.any():
                continue
            p_mask = (p_rows[:, 2] >= edges[b]) & (p_rows[:, 2] < edges[b + 1]) if p_rows.shape[0] else np.zeros(0, bool)
            m = CorrTrack.compute_metrics_bf((p_rows[p_mask], p_corr[p_mask]), (gt_rows[g_mask], gt_corr[g_mask]), windows=True)
            per_block.append(float(m.get("recall") or 0.0))
        per_block = np.asarray(per_block)
        rng = np.random.default_rng(seed)
        draws = rng.choice(per_block, size=(repeats, per_block.size), replace=True).mean(axis=1)
        return dict(mean=float(per_block.mean()), lower=float(np.quantile(draws, 0.10)), per_block=per_block.round(4).tolist())


def score(res, target):
    """Sort key: feasible first, then fewer candidates, then higher precision; infeasible by recall."""
    feasible = res["status"] == "ok" and res["recall"] >= target and res["precision"] >= PRECISION_FLOOR
    if feasible:
        return (0, res["total_candidates"], -res["precision"])
    return (1, -res.get("recall", 0.0), res.get("total_candidates") or 1 << 60)


def strip(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--window-size", type=int, default=168)
    ap.add_argument("--window-step", type=int, default=12)
    ap.add_argument("--basic-window", type=int, default=None)
    ap.add_argument("--n-lags", type=int, default=0)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--neg-corr", action="store_true")
    ap.add_argument("--preprocess", action="store_true", help="tune on first differences (the cell's preprocess flag)")
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--n-obs", type=int, default=None)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--calib-obs", type=int, default=None, help="cap the calibration span (rows) for tuning cost")
    ap.add_argument("--calib-windows", type=int, default=None,
                    help="(2026-09-19) cap the calibration span at W + N step rows (N windows); the pilot showed the 130-row "
                         "design at 403 windows x 600 series takes hours per arm, and it scales with m^2 x windows")
    ap.add_argument("--calib-series", type=int, default=None,
                    help="(2026-09-19) tune on a seeded random subset of at most K series (the first row stays the time axis). "
                         "The CSZ filter parameters describe per-pair geometry (cell size, fraction of hits, coefficients), so "
                         "the recall target is per pair and does not depend on m to first order; stated with the results")
    ap.add_argument("--calib-seed", type=int, default=20260919)
    ap.add_argument("--target-recall", type=float, default=None, help="default: exec config TARGET_RECALL (0.95)")
    ap.add_argument("--full-factorial-max", type=int, default=200)
    ap.add_argument("--refine-rounds", type=int, default=3)
    ap.add_argument("--bootstrap-blocks", type=int, default=8)
    ap.add_argument("--bootstrap-repeats", type=int, default=1000)
    ap.add_argument("--set", action="append", default=[],
                    help="accepted for command-line compatibility with nway_compare.py (the campaign passes the paper defaults of "
                         "corrjoin_ks/ke there); the protocol tunes those factors itself, so the values are only logged here")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.set:
        print(f"knob overrides from the command line ({args.set}) are not applied: the CSZ protocol tunes these factors", flush=True)
    target = args.target_recall if args.target_recall is not None else float(psmod.DEFAULT_TARGET_RECALL)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.neg_corr:
        skipped = [a for a in arms if a in ("parcorr", "csz", "corrjoin")]
        arms = [a for a in arms if a not in skipped]
        if skipped:
            print(f"neg_corr=True: {skipped} report N/A (not_available), not tuned", flush=True)
    # (2026-09-23) CorrJoin runs lagged now, as a disclosed extension of ours (its filters are
    # time-agnostic), so it is tuned at the cell's n_lags like any other arm. The old guard here
    # removed it silently, which left every lagged cell running CorrJoin on paper defaults while
    # nway_compare reported it as a tuned arm.

    cfg_dataset = bfmod._load_dataset_config(Path(args.dataset_config))
    bfmod._apply_dataset_config(cfg_dataset)
    country, var, data, ids = next(bfmod.iter_datasets())
    n_obs = args.n_obs if args.n_obs is not None else bfmod.N_YEARS[0]
    n_var = args.n_series if args.n_series is not None else bfmod.N_VARS[0]
    train_ratio = args.train_ratio if args.train_ratio is not None else bfmod.TRAIN_RATIO
    psmod.OBS_MODE = bfmod.OBS_MODE
    calib, ids_n_var = psmod.prepare_training_data(data, ids, n_obs, n_var, train_ratio)   # same span as CorrTrack's hyperopt
    if args.calib_obs is not None and calib.shape[1] > args.calib_obs:
        calib = calib[:, : args.calib_obs]
    if args.calib_windows is not None:
        cap = args.window_size + args.calib_windows * args.window_step
        if calib.shape[1] > cap:
            calib = calib[:, :cap]
    calib_series_note = ""
    if args.calib_series is not None and len(ids_n_var) > args.calib_series:
        rng = np.random.default_rng(args.calib_seed)
        keep = np.sort(rng.choice(len(ids_n_var), size=args.calib_series, replace=False))
        calib = np.vstack([calib[:1], calib[1:][keep]])          # row 0 is the time axis
        ids_n_var = [ids_n_var[i] for i in keep]
        calib_series_note = f" (random subset of {args.calib_series} series, seed {args.calib_seed})"
    slug = bfmod._dataset_slug(country, var)
    dataset_id = f"{slug}_{n_var}_{n_obs}"
    bfmod.WINDOW_SIZE, bfmod.WINDOW_STEP, bfmod.N_LAGS, bfmod.CORR_THRESHOLD, bfmod.EXEC_MODE = args.window_size, args.window_step, args.n_lags, args.corr_threshold, "sequential"
    out_dir = Path(args.out_dir) if args.out_dir else Path("correlation") / bfmod.RESULT_FOLDER / dataset_id / bfmod.config_folder() / "optim"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset {dataset_id}: calibration span {calib.shape[1]} rows x {len(ids_n_var)} series{calib_series_note} (train_ratio={train_ratio}); "
          f"W={args.window_size} step={args.window_step} n_lags={args.n_lags} T={args.corr_threshold} target recall {target}; out {out_dir}", flush=True)

    global PREPROCESS
    PREPROCESS = bool(args.preprocess)
    base = dict(window_size=args.window_size, window_step=args.window_step, basic_window=args.basic_window, n_lags=args.n_lags,
                corr_threshold=args.corr_threshold, neg_corr=args.neg_corr, preprocess=bool(args.preprocess), exec="sequential", parallel_sketch=False,
                parallel_candidates=False, parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True,
                artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
                save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        gt_rec, _, gt_flags = run_and_log_bruteforce(dataset_id, calib, ids_n_var, dict(base, baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                                                     metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
        print(f"ground truth: {gt_rec['correlated']} correlated of {gt_rec['total_candidates']} pairs ({time.perf_counter() - t0:.1f}s)", flush=True)
        ev = Evaluator(dataset_id, calib, ids_n_var, base, tmp, gt_flags, gt_rec["total_candidates"])

        for arm in arms:
            grid = parameter_grid(arm, args.window_size, args.basic_window or args.window_step)
            n_full = int(np.prod([len(v) for v in grid.values()]))
            all_valid = [dict(zip(grid, c)) for c in itertools.product(*grid.values())]
            all_valid = [s for s in all_valid if valid_setting(arm, s)]
            if len(all_valid) <= args.full_factorial_max:
                design, design_kind = all_valid, f"full factorial ({len(all_valid)} valid of {n_full})"
            else:
                design, design_kind = covering_array(grid, arm), f"strength-2 covering array ({n_full} settings)"
            print(f"\n== {arm}: stage 1, {design_kind}, {len(design)} rows", flush=True)
            t1 = time.perf_counter()
            stage1 = [ev.evaluate(arm, s) for s in design]
            ranked = sorted(stage1, key=lambda r: score(r, target))
            best = ranked[0]
            n_feas = sum(1 for r in stage1 if score(r, target)[0] == 0)
            print(f"   {len(stage1)} runs in {time.perf_counter() - t1:.0f}s; {n_feas} feasible; best {best['setting']} "
                  f"recall={best['recall']:.4f} prec={best['precision']:.4f} cand={best['total_candidates']}", flush=True)

            print(f"== {arm}: stage 2, local refinement", flush=True)
            stage2 = []
            for rnd in range(args.refine_rounds):
                cands = [ev.evaluate(arm, t) for t in neighbours(grid, arm, best["setting"])]
                stage2.extend(cands)
                cand_best = min(cands, key=lambda r: score(r, target)) if cands else best
                if score(cand_best, target) < score(best, target):
                    best = cand_best
                    print(f"   round {rnd + 1}: -> {best['setting']} recall={best['recall']:.4f} cand={best['total_candidates']}", flush=True)
                else:
                    print(f"   round {rnd + 1}: no improvement", flush=True)
                    break

            print(f"== {arm}: stage 3, bootstrap ({args.bootstrap_blocks} blocks x {args.bootstrap_repeats})", flush=True)
            pool = sorted({ev._key(r["setting"]): r for r in stage1 + stage2 if r["status"] == "ok"}.values(), key=lambda r: score(r, target))
            chosen, boots, tried = None, None, []
            for r in pool:
                if score(r, target)[0] != 0:
                    break
                b = ev.bootstrap_recall(r["_pred"], args.bootstrap_blocks, args.bootstrap_repeats)
                tried.append(dict(setting=r["setting"], bootstrap=b))
                if b["lower"] is None or b["lower"] >= target:
                    chosen, boots = r, b
                    break
                print(f"   {r['setting']}: bootstrap lower {b['lower']:.4f} < {target}, next", flush=True)
            if chosen is None:
                chosen = pool[0]
                boots = ev.bootstrap_recall(chosen["_pred"], args.bootstrap_blocks, args.bootstrap_repeats)
                status = "NO FEASIBLE SETTING: best-recall setting kept, flagged" if score(chosen, target)[0] != 0 else "no setting passed the bootstrap bound; best point estimate kept, flagged"
            else:
                status = "ok"
            print(f"   chosen {chosen['setting']}: recall={chosen['recall']:.4f} prec={chosen['precision']:.4f} cand={chosen['total_candidates']} "
                  f"bootstrap mean={boots['mean']} lower={boots['lower']} [{status}]", flush=True)

            best_params = run_params_for(arm, chosen["setting"])
            best_params.update(_tuning=dict(protocol="Cole-Shasha-Zhao KDD 2005 section 5.4", target_recall=target, status=status, preprocess=bool(args.preprocess),
                                            calibration_rows=int(calib.shape[1]), calibration_series=len(ids_n_var), recall=chosen["recall"], precision=chosen["precision"],
                                            total_candidates=chosen["total_candidates"], bootstrap=boots))
            json.dump(best_params, open(out_dir / f"best_params_{arm}.json", "w"), indent=1)
            json.dump(dict(arm=arm, dataset=dataset_id, protocol=vars(args), target_recall=target, design=design_kind,
                           ground_truth=dict(correlated=gt_rec["correlated"], total=gt_rec["total_candidates"]),
                           stage1=[strip(r) for r in stage1], stage2=[strip(r) for r in stage2], bootstrap_tried=tried,
                           chosen=strip(chosen), status=status, runs=ev.n_runs),
                      open(out_dir / f"tuning_{arm}.json", "w"), indent=1, default=str)
            print(f"   wrote {out_dir / f'best_params_{arm}.json'}", flush=True)


if __name__ == "__main__":
    main()
