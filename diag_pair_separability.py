"""Sketch Pair Separability Study (a.k.a. Candidate Geometry Diagnostic).

Follows the candidate_selector investigation in docs/implementation_log.md
(2026-07-08 entries: gamma/reliability diagnostics, top-k inverted index
real-data correction). Both of those were single-hypothesis spot checks.
This script generalizes the question: across MANY candidate pairwise
features of normalized sketch vectors (not just the full dot product),
which ones actually separate exact correlated window pairs from
non-correlated pairs, and how cheaply can each be computed relative to a
full sketch dot?

Design (agreed with the human, 2026-07-09): two-tier.

  Tier 1 (exhaustive label discovery): sweep EVERY window-start step across
  the WHOLE dataset (no train-prefix restriction, no anchor sampling) using
  the same causal "current window vs. lagged history window" pair
  enumeration already used by CorrTrack_optimize._prepare_proxy_anchor_reference
  (and replicated in diag_gamma_reliability.py / diag_topk_inverted_index.py),
  extended to cover every step instead of a sampled subset. This gives the
  TRUE population of every true positive / true negative / non-correlated
  pair in the dataset, not a sample-dependent approximation -- exact_corr
  comes from the identical formula CorrTrack's own _fast_corr_and_dist uses
  (mean-centered dot product over sqrt(variance product)).

  Tier 2 (full A-K feature computation): compute the full diagnostic feature
  set on that population directly, or on a stratified sample drawn from it
  (--max-feature-rows safety valve) if the population is too large to be
  tractable -- at the reduced N_SERIES=10/N_OBS=8760 synthetic scale used
  for this run, the exhaustive population (~550k pairs) is small enough
  that Tier 2 runs on literally every pair by default.

This script does NOT modify CorrTrack/Candidates/Sketches. It reuses
SketchCache / _load_module / _resolve_cfg_value from diag_gamma_reliability.py
(real, production-identical incremental sketch construction) and
CorrTrack_optimize._proxy_window_is_valid (the same near-constant /
structurally-spiked window filter the real proxy-anchor hyperopt path uses).

Normalization pipeline preserved as-is (raw window -> incremental sketch ->
post-sketch normalization -> candidate selection -> exact validation); no
raw-window normalization is introduced. Per the human's explicit
constraint, features are computed against three sketch representations:
  raw sketch v            (pre-centering, pre-scaling)
  centered sketch c = v - mean(v)
  normalized sketch w_hat = c / ||c||_2   (the actual production candidate representation)
Families A-F, I, J, K operate on w_hat (the only representation
CorrTrack's real candidate search ever inspects -- gamma is a threshold on
dot(w_hat_q, w_hat_x)). Family G (per-sketch quantile/distribution
features) is computed on all three representations, since normalization
explicitly discards scale/mean information that raw/centered retain, and
that is exactly the kind of "does normalization throw away separating
signal" question this study exists to answer. Family H (rank/order) is
mathematically invariant to the centering+positive-scalar-rescaling that
turns v into w_hat (monotonic in each element), so it is computed once,
not three times -- noted explicitly rather than silently deduplicated.

Two additions beyond the human's original spec, agreed as advisable:
  - `contribution_participation_ratio` (family E extension): a per-window
    effective-dimensionality feature, (sum|v_i|)^2 / sum(v_i^2). Explains
    *why* top-m indexing does or doesn't work -- if a window's mass is
    spread over most of the n_vectors dimensions, no top-m index can ever
    concentrate much of the dot there.
  - `simhash_hamming_r*` (family I extension): actual LSH-bucket-style
    sign-bit Hamming distance of the same fixed random projections, more
    directly actionable for indexability than the raw projected-L2
    distance the spec asked for.
  - A combined cheap-feature logistic-regression classifier (fit only on
    cost_category in {precomputed_per_window, pairwise_scalar} features,
    excluding sketch_dot/l2_dist/l2_dist_neg): answers whether several weak
    cheap signals *together* approximate the full dot even if none does
    alone -- a question pure univariate analysis cannot answer.

Multiple-comparison guard: every feature's threshold-at-95%-recall is fit
on a train split and evaluated on a held-out test split (stratified by
label), not on the same rows used to pick the threshold -- avoids the
gamma=0.58-point-estimate-vs-bootstrap-LCB trap already hit once this
project (docs/implementation_log.md, Phase 1 gamma diagnostics).

Usage:
    PYTHONPATH="$(pwd)" python3 corrtrack_release_dev/diag_pair_separability.py \\
        --dataset-config corrtrack_release_dev/experiment_dataset_synth_demo.py \\
        --exec-param-config corrtrack_release_dev/experiment_run_exec_param.py \\
        --window-size 256 --window-step 16 --n-lags 168 --corr-threshold 0.7 \\
        --neg-corr --n-vectors 64 --gamma 0.55 \\
        --result-folder tmp_artifacts/pair_separability_diag
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import corrtrack_param_search as cps
from library_corrtrack_parallel import CorrTrack, CorrTrack_optimize, _fast_corr_and_dist
from diag_gamma_reliability import _load_module, _resolve_cfg_value, SketchCache

EPS = 1e-12


# =============================================================================
# Per-window precomputation (done once per (series_idx, start), cached).
# =============================================================================

def _quantile_stats(vec):
    if vec.size == 0:
        return dict(min=0.0, max=0.0, median=0.0, q10=0.0, q25=0.0, q75=0.0, q90=0.0, l1=0.0, linf=0.0)
    return dict(
        min=float(np.min(vec)), max=float(np.max(vec)), median=float(np.median(vec)),
        q10=float(np.percentile(vec, 10)), q25=float(np.percentile(vec, 25)),
        q75=float(np.percentile(vec, 75)), q90=float(np.percentile(vec, 90)),
        l1=float(np.sum(np.abs(vec))), linf=float(np.max(np.abs(vec))),
    )


def _participation_ratio(vec):
    denom = float(np.sum(vec ** 2))
    if denom <= EPS:
        return 0.0
    return float(np.sum(np.abs(vec)) ** 2) / denom


class WindowPrecomputeCache:
    """Precomputes and caches, once per (series_idx, start), everything the
    A-K feature families need beyond the raw/normalized vectors SketchCache
    already provides: top-m index sets, sorted arrays, ranks, fixed random
    projections, and per-representation quantile stats."""

    def __init__(self, sketch_cache, proxy_ids, m_values, proj_dirs):
        self.sketch_cache = sketch_cache
        self.proxy_ids = proxy_ids
        self.m_values = m_values
        self.proj_dirs = proj_dirs  # shape (max_r, n_vectors)
        self._cache = {}

    def get(self, sid_idx, start):
        key = (sid_idx, int(start))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        raw_by_sid, norm_by_sid = self.sketch_cache.get(start)
        sid = self.proxy_ids[sid_idx]
        raw_v = raw_by_sid.get(sid)
        norm_w = norm_by_sid.get(sid)
        if raw_v is None or norm_w is None:
            return None
        raw_v = np.asarray(raw_v, dtype=np.float64)
        norm_w = np.asarray(norm_w, dtype=np.float64)
        centered_c = raw_v - float(np.mean(raw_v))
        n = norm_w.size
        order = np.argsort(-np.abs(norm_w))
        topm_idx = {m: order[:min(m, n)] for m in self.m_values}
        result = {
            "raw_v": raw_v, "centered_c": centered_c, "norm_w": norm_w,
            "order": order, "topm_idx": topm_idx, "sorted_w": np.sort(norm_w),
            "rank_w": sp_stats.rankdata(norm_w), "rank_abs_w": sp_stats.rankdata(np.abs(norm_w)),
            "stats_raw": _quantile_stats(raw_v), "stats_centered": _quantile_stats(centered_c),
            "stats_norm": _quantile_stats(norm_w),
            "proj": self.proj_dirs @ norm_w,
            "participation_ratio_raw": _participation_ratio(raw_v),
            "participation_ratio_norm": _participation_ratio(norm_w),
        }
        self._cache[key] = result
        return result


# =============================================================================
# Pairwise feature families (A-K). All operate on normalized w_hat unless noted.
# =============================================================================

def _family_a(q, x):
    dot = float(np.dot(q, x))
    return {
        "sketch_dot": dot, "abs_sketch_dot": abs(dot),
        "l2_dist": float(np.linalg.norm(q - x)), "l2_dist_neg": float(np.linalg.norm(q + x)),
    }


def _family_b(q, x):
    k = q.size
    contrib = q * x
    sign_agree = int(np.sum(np.sign(q) == np.sign(x)))
    frac = sign_agree / k
    same_mass = float(np.sum(contrib[contrib > 0]))
    opp_mass = float(np.sum(np.abs(contrib[contrib < 0])))
    return {
        "sign_agree_count": sign_agree, "sign_agree_frac": frac,
        "sign_opposite_frac": 1.0 - frac, "sign_alignment_frac": max(frac, 1.0 - frac),
        "same_sign_mass": same_mass, "opposite_sign_mass": opp_mass,
        "mass_imbalance": abs(same_mass - opp_mass) / (same_mass + opp_mass + EPS),
    }


def _family_cd(q, x, fq, fx, m_values):
    out = {}
    contrib = q * x
    for m in m_values:
        tq = set(fq["topm_idx"][m].tolist())
        tx = set(fx["topm_idx"][m].tolist())
        inter = tq & tx
        union = tq | tx
        overlap = len(inter)
        out[f"topm_overlap_m{m}"] = overlap
        out[f"topm_jaccard_m{m}"] = overlap / len(union) if union else 0.0
        same_sign = sum(1 for i in inter if contrib[i] > 0)
        opp_sign = sum(1 for i in inter if contrib[i] < 0)
        out[f"topm_same_sign_m{m}"] = same_sign
        out[f"topm_opposite_sign_m{m}"] = opp_sign
        out[f"topm_signed_alignment_m{m}"] = max(same_sign, opp_sign)

        idx_q = np.fromiter(tq, dtype=int) if tq else np.array([], dtype=int)
        idx_x = np.fromiter(tx, dtype=int) if tx else np.array([], dtype=int)
        idx_u = np.fromiter(union, dtype=int) if union else np.array([], dtype=int)
        idx_i = np.fromiter(inter, dtype=int) if inter else np.array([], dtype=int)
        pd_q = float(np.sum(contrib[idx_q])) if idx_q.size else 0.0
        pd_x = float(np.sum(contrib[idx_x])) if idx_x.size else 0.0
        pd_u = float(np.sum(contrib[idx_u])) if idx_u.size else 0.0
        pd_i = float(np.sum(contrib[idx_i])) if idx_i.size else 0.0
        out[f"partial_dot_qtopm_m{m}"] = pd_q
        out[f"partial_dot_xtopm_m{m}"] = pd_x
        out[f"partial_dot_union_m{m}"] = pd_u
        out[f"partial_dot_intersection_m{m}"] = pd_i
        out[f"abs_partial_dot_qtopm_m{m}"] = abs(pd_q)
        out[f"abs_partial_dot_xtopm_m{m}"] = abs(pd_x)
        out[f"abs_partial_dot_union_m{m}"] = abs(pd_u)
        out[f"abs_partial_dot_intersection_m{m}"] = abs(pd_i)
    return out


def _family_e(q, x, fq, fx, r_values):
    contrib = q * x
    abs_contrib = np.abs(contrib)
    order = np.argsort(-abs_contrib)
    sorted_abs = abs_contrib[order]
    sorted_signed = contrib[order]
    total_abs = float(np.sum(abs_contrib)) + EPS
    cum = np.cumsum(sorted_abs)
    out = {}
    for r in r_values:
        rr = min(r, sorted_abs.size)
        mass = float(np.sum(sorted_abs[:rr]))
        out[f"topr_abs_contribution_mass_r{r}"] = mass
        out[f"topr_abs_contribution_ratio_r{r}"] = mass / total_abs
        out[f"topr_signed_contribution_r{r}"] = float(np.sum(sorted_signed[:rr]))
    for pct, name in ((0.5, "50pct"), (0.7, "70pct"), (0.9, "90pct")):
        n_needed = int(np.searchsorted(cum, pct * total_abs) + 1) if total_abs > EPS else 0
        out[f"n_dims_for_{name}_abs_mass"] = n_needed
    pr_q = fq["participation_ratio_norm"]
    pr_x = fx["participation_ratio_norm"]
    out["participation_ratio_min"] = min(pr_q, pr_x)
    out["participation_ratio_max"] = max(pr_q, pr_x)
    out["participation_ratio_product"] = pr_q * pr_x
    return out


def _family_f(q, x, delta_values, theta_values):
    diff = q - x
    diff_neg = q + x
    contrib = q * x
    out = {}
    for d in delta_values:
        close = int(np.sum(np.abs(diff) <= d))
        negclose = int(np.sum(np.abs(diff_neg) <= d))
        tag = str(d).replace(".", "p")
        out[f"coord_close_d{tag}"] = close
        out[f"coord_negclose_d{tag}"] = negclose
        out[f"coord_absmatch_d{tag}"] = max(close, negclose)
    for t in theta_values:
        tag = str(t).replace(".", "p")
        out[f"coord_poscontrib_t{tag}"] = int(np.sum(contrib >= t))
        out[f"coord_negcontrib_t{tag}"] = int(np.sum(contrib <= -t))
        out[f"coord_abscontrib_t{tag}"] = int(np.sum(np.abs(contrib) >= t))
    return out


def _family_g(fq, fx):
    out = {}
    for rep in ("raw", "centered", "norm"):
        sq = fq[f"stats_{rep}"]
        sx = fx[f"stats_{rep}"]
        for key in ("min", "max", "median", "q10", "q25", "q75", "q90", "l1", "linf"):
            out[f"abs_{key}_diff_{rep}"] = abs(sq[key] - sx[key])
    sq = fq["sorted_w"]
    sx = fx["sorted_w"]
    sx_neg_sorted = np.sort(-fx["norm_w"])
    sorted_l1 = float(np.sum(np.abs(sq - sx)))
    sorted_l2 = float(np.linalg.norm(sq - sx))
    sorted_neg_l1 = float(np.sum(np.abs(sq - sx_neg_sorted)))
    sorted_neg_l2 = float(np.linalg.norm(sq - sx_neg_sorted))
    out.update(
        sorted_l1=sorted_l1, sorted_l2=sorted_l2,
        sorted_neg_l1=sorted_neg_l1, sorted_neg_l2=sorted_neg_l2,
        sorted_abs_best_l1=min(sorted_l1, sorted_neg_l1),
        sorted_abs_best_l2=min(sorted_l2, sorted_neg_l2),
    )
    return out


def _family_h(fq, fx, include_kendall):
    rq, rx = fq["rank_w"], fx["rank_w"]
    raq, rax = fq["rank_abs_w"], fx["rank_abs_w"]
    out = {
        "spearman_corr_values": float(np.corrcoef(rq, rx)[0, 1]) if rq.size > 1 else float("nan"),
        "spearman_corr_abs_values": float(np.corrcoef(raq, rax)[0, 1]) if raq.size > 1 else float("nan"),
    }
    if include_kendall:
        tau, _p = sp_stats.kendalltau(fq["norm_w"], fx["norm_w"])
        out["kendall_tau_values"] = float(tau)
    return out


def _family_i(fq, fx, r_values):
    out = {}
    pq, px = fq["proj"], fx["proj"]
    for r in r_values:
        d = pq[:r] - px[:r]
        dneg = pq[:r] + px[:r]
        l2 = float(np.linalg.norm(d))
        negl2 = float(np.linalg.norm(dneg))
        l1 = float(np.sum(np.abs(d)))
        negl1 = float(np.sum(np.abs(dneg)))
        out[f"projection_l2_r{r}"] = l2
        out[f"projection_neg_l2_r{r}"] = negl2
        out[f"projection_abs_best_l2_r{r}"] = min(l2, negl2)
        out[f"projection_l1_r{r}"] = l1
        out[f"projection_abs_best_l1_r{r}"] = min(l1, negl1)
        sign_q = pq[:r] >= 0
        sign_x = px[:r] >= 0
        out[f"simhash_hamming_r{r}"] = int(np.sum(sign_q != sign_x))
    return out


def _family_j(q, x, block_idx_by_b):
    out = {}
    contrib = q * x
    for b, blocks in block_idx_by_b.items():
        dots = np.array([float(np.sum(contrib[idx])) for idx in blocks])
        out[f"block_dot_max_b{b}"] = float(np.max(dots))
        out[f"block_dot_abs_max_b{b}"] = float(np.max(np.abs(dots)))
        out[f"block_dot_mean_b{b}"] = float(np.mean(dots))
        out[f"block_dot_abs_mean_b{b}"] = float(np.mean(np.abs(dots)))
        out[f"block_dot_std_b{b}"] = float(np.std(dots))
        out[f"block_positive_count_b{b}"] = int(np.sum(dots > 0))
        out[f"block_negative_count_b{b}"] = int(np.sum(dots < 0))
        out[f"block_abs_sum_b{b}"] = float(np.sum(np.abs(dots)))
    return out


def _family_k(q, x, fq, fx, m_values, first_block_idx):
    out = {}
    contrib = q * x
    n = q.size
    for m in m_values:
        tq, tx = fq["topm_idx"][m], fx["topm_idx"][m]
        union_idx = np.union1d(tq, tx)
        for s_name, idx in (("qtop", tq), ("xtop", tx), ("union", union_idx)):
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            partial = float(np.sum(contrib[idx])) if idx.size else 0.0
            resid_q = float(np.linalg.norm(q[~mask]))
            resid_x = float(np.linalg.norm(x[~mask]))
            out[f"ub_partial_{s_name}_m{m}"] = partial
            out[f"ub_residual_q_{s_name}_m{m}"] = resid_q
            out[f"ub_residual_x_{s_name}_m{m}"] = resid_x
            out[f"ub_pos_{s_name}_m{m}"] = partial + resid_q * resid_x
            out[f"ub_neg_{s_name}_m{m}"] = -partial + resid_q * resid_x
            out[f"ub_abs_{s_name}_m{m}"] = abs(partial) + resid_q * resid_x
    mask = np.zeros(n, dtype=bool)
    mask[first_block_idx] = True
    partial = float(np.sum(contrib[first_block_idx]))
    resid_q = float(np.linalg.norm(q[~mask]))
    resid_x = float(np.linalg.norm(x[~mask]))
    out["ub_partial_firstblock"] = partial
    out["ub_abs_firstblock"] = abs(partial) + resid_q * resid_x
    return out


def compute_pair_features(q, x, fq, fx, cfg):
    out = {}
    out.update(_family_a(q, x))
    out.update(_family_b(q, x))
    out.update(_family_cd(q, x, fq, fx, cfg["m_values"]))
    out.update(_family_e(q, x, fq, fx, cfg["r_values"]))
    out.update(_family_f(q, x, cfg["delta_values"], cfg["theta_values"]))
    out.update(_family_g(fq, fx))
    out.update(_family_h(fq, fx, cfg["include_kendall"]))
    out.update(_family_i(fq, fx, cfg["proj_r_values"]))
    out.update(_family_j(q, x, cfg["block_idx_by_b"]))
    out.update(_family_k(q, x, fq, fx, cfg["ub_m_values"], cfg["first_block_idx"]))
    return out


# =============================================================================
# Feature metadata (static, not computed from data).
# =============================================================================

def _classify_feature(name):
    """Returns (cost_category, indexability_category, notes)."""
    if name in ("sketch_dot", "abs_sketch_dot", "l2_dist", "l2_dist_neg"):
        return "equivalent_to_dot", "not_useful", "Sanity-check baseline; requires the full dot itself, not a cheaper alternative."
    if name.startswith("topm_") or name.startswith("partial_dot_") or name.startswith("abs_partial_dot_"):
        return "pairwise_partial", "inverted_index", (
            "Top-m dominant-coordinate overlap/partial-dot; see docs/implementation_log.md's 2026-07-08 "
            "TopKInvertedIndex real-data correction -- on this project's real dataset this family degenerated "
            "to a near-full scan at recall-meeting settings, check whether this run reproduces that."
        )
    if name.startswith("topr_") or name.startswith("n_dims_for_") or name.startswith("participation_ratio"):
        return "pairwise_partial", "calibration_only", "Contribution-concentration / effective-dimensionality; informs whether ANY top-m index can work, not directly indexable itself."
    if name.startswith("coord_"):
        return "pairwise_scalar", "multi_bst_voting", "Per-dimension closeness/contribution vote count; candidate for one-BST-per-dimension voting schemes."
    if name.startswith("sign_"):
        return "pairwise_scalar", "inverted_index", (
            "Sign-only comparison is genuinely cheaper per pair (no multiply) but is NOT a single-scalar-per-window "
            "index despite looking like one -- it's a Hamming similarity over the full k-bit sign(w_hat) pattern, "
            "which is a k-dimensional object, not reducible to one sortable number per window. Real indexing needs "
            "LSH bit-sampling/banding over the sign vector (an inverted_index, same family as simhash_hamming_r*), "
            "not a plain BST. See docs/implementation_log.md's 2026-07-09 sign_alignment_frac / LSH-banding entry."
        )
    if name in ("same_sign_mass", "opposite_sign_mass", "mass_imbalance"):
        return "pairwise_partial", "calibration_only", (
            "NOT actually cheaper than the full dot despite the 'mass' name -- computing same/opposite_sign_mass "
            "requires q_i*x_i for every dimension first (identical elementwise work to the dot itself), then just "
            "sums it two different ways. dot = same_sign_mass - opposite_sign_mass exactly, so these are a "
            "rescaling of the dot's own intermediate array, not a cheaper alternative to it."
        )
    if name.startswith("abs_") and name.endswith(("_raw", "_centered", "_norm")):
        return "precomputed_per_window", "scalar_bst", "Per-window quantile/scale statistic difference; precomputable once per window, independent of pairing."
    if name.startswith("sorted_"):
        return "pairwise_partial", "calibration_only", "Requires sorting each vector (O(k log k)) but not the pairing itself; coordinate-position-invariant similarity."
    if name.startswith("spearman_") or name.startswith("kendall_"):
        return "pairwise_full_vector", "calibration_only", "Rank correlation; invariant to the raw->centered->normalized rescaling (monotonic), computed once."
    if name.startswith("projection_") or name.startswith("simhash_"):
        return "pairwise_scalar", "inverted_index", "Fixed random projection / LSH-style sign bits; directly maps to an actual LSH bucket index if separable."
    if name.startswith("block_dot"):
        return "pairwise_partial", "block_bound", (
            "Block-level dot aggregate; see docs/implementation_log.md's 2026-07-06 block-level cone pruning "
            "entries -- this project already found block-level angular structure was arrival-order dependent, "
            "not similarity dependent, on real streaming data by default."
        )
    if name.startswith("block_positive_count") or name.startswith("block_negative_count") or name.startswith("block_abs_sum"):
        return "pairwise_partial", "block_bound", "Block-level sign/mass aggregate."
    if name.startswith("ub_"):
        return "pairwise_partial", "block_bound", (
            "Cauchy-Schwarz-style partial-dot upper bound; see docs/implementation_log.md's 2026-07-06 "
            "row-level UB pruning entry -- previously found ~0 net wall-clock benefit despite exactness."
        )
    return "pairwise_scalar", "not_useful", ""


# =============================================================================
# Separability evaluation.
# =============================================================================

def _evaluate_feature(values, target, cls, test_frac, random_state):
    mask = np.isfinite(values)
    values = values[mask]
    target = target[mask]
    cls = cls[mask]
    n_pos_total = int(target.sum())
    if len(values) < 20 or n_pos_total == 0 or n_pos_total == len(values):
        return None
    try:
        idx_train, idx_test = train_test_split(
            np.arange(len(values)), test_size=test_frac, random_state=random_state, stratify=target,
        )
    except ValueError:
        return None
    v_train, t_train = values[idx_train], target[idx_train]
    v_test, t_test = values[idx_test], target[idx_test]

    mean_pos_train = v_train[t_train].mean() if t_train.any() else np.nan
    mean_rest_train = v_train[~t_train].mean() if (~t_train).any() else np.nan
    direction = "larger_is_more_correlated" if mean_pos_train >= mean_rest_train else "smaller_is_more_correlated"
    sign = 1.0 if direction == "larger_is_more_correlated" else -1.0
    score_train = sign * v_train
    score_test = sign * v_test

    try:
        auc = float(roc_auc_score(t_train, score_train))
        ap = float(average_precision_score(t_train, score_train))
    except ValueError:
        auc = ap = float("nan")

    order = np.argsort(-score_train)
    sorted_labels = t_train[order]
    sorted_scores = score_train[order]
    n_pos_train = int(t_train.sum())
    cum_tp = np.cumsum(sorted_labels)
    recall_curve = cum_tp / n_pos_train if n_pos_train else np.zeros_like(cum_tp, dtype=float)
    idx_95 = int(np.searchsorted(recall_curve, 0.95))
    threshold_score = sorted_scores[min(idx_95, len(sorted_scores) - 1)] if len(sorted_scores) else float("nan")

    retained_test = score_test >= threshold_score
    n_retained_test = int(retained_test.sum())
    n_pos_test = int(t_test.sum())
    n_recovered_test = int((retained_test & t_test).sum())
    recall_test = n_recovered_test / n_pos_test if n_pos_test else float("nan")
    precision_test = n_recovered_test / n_retained_test if n_retained_test else float("nan")
    candidate_rate_test = n_retained_test / len(v_test) if len(v_test) else float("nan")

    raw_threshold = sign * threshold_score
    is_pos_mask = cls == "pos"
    is_neg_mask = cls == "neg"
    is_non_mask = cls == "non"
    return dict(
        direction=direction, threshold_at_95_recall=float(raw_threshold),
        recall_at_threshold=recall_test, precision_at_threshold=precision_test,
        candidate_rate_at_95_recall=candidate_rate_test,
        candidate_reduction_at_95_recall=1.0 - candidate_rate_test if np.isfinite(candidate_rate_test) else float("nan"),
        auc_roc=auc, average_precision=ap,
        median_pos=float(np.median(values[is_pos_mask])) if is_pos_mask.any() else float("nan"),
        median_neg=float(np.median(values[is_neg_mask])) if is_neg_mask.any() else float("nan"),
        median_non=float(np.median(values[is_non_mask])) if is_non_mask.any() else float("nan"),
        mean_pos=float(np.mean(values[is_pos_mask])) if is_pos_mask.any() else float("nan"),
        mean_neg=float(np.mean(values[is_neg_mask])) if is_neg_mask.any() else float("nan"),
        mean_non=float(np.mean(values[is_non_mask])) if is_non_mask.any() else float("nan"),
        separation_score=abs(auc - 0.5) * 2.0 if np.isfinite(auc) else float("nan"),
        n_train_rows=len(v_train), n_test_rows=len(v_test),
        n_pairs_retained_test=n_retained_test, n_positives_recovered_test=n_recovered_test,
    )


def build_separability_summary(df, feature_cols, cls_col, test_frac, random_state):
    cls = df[cls_col].to_numpy()
    label_defs = {
        "is_abs_pos": (cls == "pos") | (cls == "neg"),
        "is_pos": cls == "pos",
        "is_neg": cls == "neg",
    }
    rows = []
    for feature in feature_cols:
        values = df[feature].to_numpy(dtype=np.float64)
        cost_cat, index_cat, notes = _classify_feature(feature)
        for label_type, target in label_defs.items():
            metrics = _evaluate_feature(values, target, cls, test_frac, random_state)
            if metrics is None:
                continue
            row = {"feature_name": feature, "label_type": label_type, "cost_category": cost_cat,
                   "indexability_category": index_cat, "notes": notes}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def fit_combined_cheap_model(df, feature_cols, cls_col, test_frac, random_state):
    cheap_cols = []
    for f in feature_cols:
        cost_cat, _idx_cat, _notes = _classify_feature(f)
        if cost_cat in ("precomputed_per_window", "pairwise_scalar") and f not in ("sketch_dot", "abs_sketch_dot", "l2_dist", "l2_dist_neg"):
            cheap_cols.append(f)
    if not cheap_cols:
        return None, []
    cls = df[cls_col].to_numpy()
    target = (cls == "pos") | (cls == "neg")
    X = df[cheap_cols].to_numpy(dtype=np.float64)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X, target, cls = X[finite_mask], target[finite_mask], cls[finite_mask]
    if len(X) < 50 or target.sum() == 0 or target.sum() == len(target):
        return None, cheap_cols
    idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=test_frac, random_state=random_state, stratify=target)
    scaler = StandardScaler().fit(X[idx_train])
    X_train, X_test = scaler.transform(X[idx_train]), scaler.transform(X[idx_test])
    t_train, t_test = target[idx_train], target[idx_test]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=2000).fit(X_train, t_train)
    score_train = clf.decision_function(X_train)
    score_test = clf.decision_function(X_test)
    auc = float(roc_auc_score(t_train, score_train))
    ap = float(average_precision_score(t_train, score_train))
    order = np.argsort(-score_train)
    sorted_labels = t_train[order]
    n_pos_train = int(t_train.sum())
    cum_tp = np.cumsum(sorted_labels)
    recall_curve = cum_tp / n_pos_train if n_pos_train else np.zeros_like(cum_tp, dtype=float)
    idx_95 = int(np.searchsorted(recall_curve, 0.95))
    threshold_score = np.sort(score_train)[::-1][min(idx_95, len(score_train) - 1)]
    retained_test = score_test >= threshold_score
    n_pos_test = int(t_test.sum())
    n_recovered_test = int((retained_test & t_test).sum())
    result = dict(
        feature_name="COMBINED_CHEAP_MODEL", label_type="is_abs_pos",
        n_features_used=len(cheap_cols), auc_roc=auc, average_precision=ap,
        recall_at_threshold=n_recovered_test / n_pos_test if n_pos_test else float("nan"),
        precision_at_threshold=n_recovered_test / retained_test.sum() if retained_test.sum() else float("nan"),
        candidate_rate_at_95_recall=float(retained_test.mean()),
        candidate_reduction_at_95_recall=1.0 - float(retained_test.mean()),
        n_train_rows=len(X_train), n_test_rows=len(X_test),
        coefficients={col: float(coef) for col, coef in zip(cheap_cols, clf.coef_[0])},
    )
    return result, cheap_cols


# =============================================================================
# Plotting.
# =============================================================================

def plot_feature(df, feature, cls_col, out_path, exact_corr_col):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = df[feature].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    cls = df[cls_col].to_numpy()[finite]
    values = values[finite]
    corr = df[exact_corr_col].to_numpy(dtype=np.float64)[finite]
    pos, neg, non = values[cls == "pos"], values[cls == "neg"], values[cls == "non"]
    target_abs = (cls == "pos") | (cls == "neg")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(feature)

    ax = axes[0, 0]
    bins = 40
    for arr, label in ((non, "non"), (pos, "pos"), (neg, "neg")):
        if arr.size:
            ax.hist(arr, bins=bins, density=True, alpha=0.5, label=f"{label} (n={arr.size})")
    ax.set_title("Distribution by class")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    data_box = [a for a in (non, pos, neg) if a.size]
    labels_box = [l for l, a in (("non", non), ("pos", pos), ("neg", neg)) if a.size]
    if data_box:
        ax.boxplot(data_box, labels=labels_box, showfliers=False)
    ax.set_title("Boxplot by class")

    ax = axes[0, 2]
    sample_n = min(20000, values.size)
    if sample_n:
        rng = np.random.default_rng(0)
        sel = rng.choice(values.size, size=sample_n, replace=False) if values.size > sample_n else np.arange(values.size)
        ax.scatter(corr[sel], values[sel], s=2, alpha=0.3)
    ax.set_xlabel("exact_corr")
    ax.set_ylabel(feature)
    ax.set_title("feature vs exact_corr")

    ax = axes[1, 0]
    if sample_n:
        ax.scatter(np.abs(corr[sel]), values[sel], s=2, alpha=0.3, color="tab:orange")
    ax.set_xlabel("abs(exact_corr)")
    ax.set_ylabel(feature)
    ax.set_title("feature vs abs(exact_corr)")

    ax = axes[1, 1]
    if target_abs.any() and (~target_abs).any():
        fpr, tpr, _ = roc_curve(target_abs, values)
        fpr2, tpr2, _ = roc_curve(target_abs, -values)
        auc1, auc2 = roc_auc_score(target_abs, values), roc_auc_score(target_abs, -values)
        if auc1 >= auc2:
            ax.plot(fpr, tpr, label=f"AUC={auc1:.3f}")
        else:
            ax.plot(fpr2, tpr2, label=f"AUC={auc2:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.legend(fontsize=8)
    ax.set_title("ROC (is_abs_pos)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    ax = axes[1, 2]
    if target_abs.any() and (~target_abs).any():
        score = values if auc1 >= auc2 else -values
        prec, rec, _ = precision_recall_curve(target_abs, score)
        ax.plot(rec, prec)
    ax.set_title("Precision-Recall (is_abs_pos)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# =============================================================================
# Main.
# =============================================================================

def _float_list(s):
    return [float(x) for x in s.split(",")]


def _int_list(s):
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", required=False, default=None, type=Path,
                         help="Required unless --resume-from-pair-features (resume mode reuses an "
                              "already-collected pair_features.csv and never touches the dataset).")
    parser.add_argument("--exec-param-config", required=False, default=None, type=Path)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--n-lags", type=int, default=None)
    parser.add_argument("--corr-threshold", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None, help="Defaults to --corr-threshold.")
    parser.add_argument("--neg-corr", dest="neg_corr", action="store_true", default=None)
    parser.add_argument("--no-neg-corr", dest="neg_corr", action="store_false")
    parser.add_argument("--train-ratio", type=float, default=1.0,
                         help="Fraction of the dataset to use. Default 1.0 (whole dataset -- "
                              "this is a diagnostic study, not hyperparameter selection, so the "
                              "causal train-prefix restriction used by the real hyperopt path "
                              "does not apply here.")
    parser.add_argument("--n-vectors", type=int, default=64)
    parser.add_argument("--sketch-norm", type=str, default="mean_l2")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--seed-toggle", type=int, default=1357)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--m-values", type=_int_list, default=[4, 8, 12, 16, 24, 32])
    parser.add_argument("--r-values", type=_int_list, default=[1, 2, 4, 8, 16, 32], help="Family E top-r contribution.")
    parser.add_argument("--delta-values", type=_float_list, default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--theta-values", type=_float_list, default=[0.001, 0.0025, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--projection-r-values", type=_int_list, default=[2, 4, 8, 16])
    parser.add_argument("--block-b-values", type=_int_list, default=[2, 4, 8])
    parser.add_argument("--ub-m-values", type=_int_list, default=[4, 8, 16, 24, 32])
    parser.add_argument("--include-kendall", action="store_true", default=False)
    parser.add_argument("--hard-band-lo", type=float, default=0.50)
    parser.add_argument("--hard-band-hi", type=float, default=0.60)
    parser.add_argument("--hard-band2-lo", type=float, default=0.55)
    parser.add_argument("--hard-band2-hi", type=float, default=0.65)
    parser.add_argument("--max-feature-rows", type=int, default=1_000_000,
                         help="Safety valve: if the exhaustive label population exceeds this, "
                              "switch to stratified sampling (all positives, all negatives, "
                              "random non-correlated, random near-gamma-band) for the expensive "
                              "Tier-2 feature computation. At the reduced N_SERIES=10/N_OBS=8760 "
                              "scale this run was designed for (~550k exhaustive pairs), the "
                              "default is large enough that Tier 2 runs on every pair.")
    parser.add_argument("--test-frac", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=2468)
    parser.add_argument("--plot-top-n", type=int, default=20)
    parser.add_argument("--plot-all-families", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--result-folder", type=Path, default=Path("tmp_artifacts/pair_separability_diag"))
    parser.add_argument("--resume-from-pair-features", action="store_true", default=False,
                         help="Skip Tier 1/Tier 2 entirely and rerun only the downstream analysis "
                              "(separability summary, combined model, hard band, plots) against an "
                              "existing --result-folder/pair_features.csv. Useful if a prior run "
                              "collected the full feature set but died during the downstream "
                              "analysis (e.g. an OOM on the final CSV read) -- avoids repeating the "
                              "expensive exhaustive sweep just to retry a cheap step.")
    args = parser.parse_args()

    if args.resume_from_pair_features:
        pair_path = args.result_folder / "pair_features.csv"
        int_cols = {"s_current", "cur_start", "s_other", "lag_start"}
        header_cols = pd.read_csv(pair_path, nrows=0).columns
        dtype_map = {
            col: ("category" if col == "cls" else np.int32 if col in int_cols else np.float32)
            for col in header_cols
        }
        print(f"Resuming from {pair_path} (skipping Tier 1/Tier 2)...")
        pair_df = pd.read_csv(pair_path, dtype=dtype_map)
        print(f"Loaded {len(pair_df)} rows, {len(pair_df.columns)} columns")
        _run_downstream_analysis(pair_df, args)
        print("Done.")
        return

    if args.dataset_config is None or args.exec_param_config is None:
        parser.error("--dataset-config and --exec-param-config are required unless --resume-from-pair-features")

    dataset_cfg = _load_module(args.dataset_config, "diag_dataset_config")
    exec_cfg = _load_module(args.exec_param_config, "diag_exec_config")

    window_size = _resolve_cfg_value(args.window_size, exec_cfg, "WINDOW_SIZE", 168)
    window_step = _resolve_cfg_value(args.window_step, exec_cfg, "WINDOW_STEP", 12)
    n_lags = _resolve_cfg_value(args.n_lags, exec_cfg, "N_LAGS", 168)
    corr_threshold = _resolve_cfg_value(args.corr_threshold, exec_cfg, "CORR_THRESHOLD", 0.7)
    tau = args.tau if args.tau is not None else corr_threshold
    neg_corr = _resolve_cfg_value(args.neg_corr, exec_cfg, "NEG_CORR", True)

    print(f"PARAMS: window_size={window_size} window_step={window_step} n_lags={n_lags} "
          f"corr_threshold={corr_threshold} tau={tau} neg_corr={neg_corr} train_ratio={args.train_ratio} "
          f"n_vectors={args.n_vectors} gamma={args.gamma} max_feature_rows={args.max_feature_rows}")

    cps._apply_dataset_config(dataset_cfg)
    country, var, data, ids = next(cps.iter_datasets())
    n_var = cps._get_cfg_attr(dataset_cfg, "N_VARS", "N_SERIES")
    n_year = cps._get_cfg_attr(dataset_cfg, "N_YEARS", "N_OBS")
    n_var = n_var[0] if isinstance(n_var, (list, tuple)) else n_var
    n_year = n_year[0] if isinstance(n_year, (list, tuple)) else n_year
    train_data, proxy_ids = cps.prepare_training_data(data, ids, n_year, n_var, args.train_ratio)
    proxy_ids = [str(x) for x in proxy_ids]
    n_series = train_data.shape[0] - 1
    length_data = train_data.shape[1]
    print(f"Dataset loaded: n_series={n_series} length_data={length_data} (train_ratio={args.train_ratio})")

    basic_window = CorrTrack.get_bst_basic_window(window_size, window_step)
    sketch_cache = SketchCache(
        train_data, proxy_ids, window_size, basic_window, window_step,
        args.seed, args.seed_toggle, args.n_vectors, args.n_vectors,
        preprocess=False, neg_corr=neg_corr, sketch_norm=args.sketch_norm,
    )

    max_start = length_data - window_size
    if max_start < 0:
        raise ValueError("dataset shorter than window_size")
    all_starts = list(range(0, max_start + 1, window_step))
    lag_count = max(1, int(n_lags // window_step) + 1)
    est_pairs_per_step = int(n_series * (n_series - 1) / 2) + int(n_series * n_series * max(lag_count - 1, 0))
    print(f"Exhaustive sweep: {len(all_starts)} window-start steps, ~{est_pairs_per_step} pairs/step, "
          f"~{len(all_starts) * est_pairs_per_step} total pairs (Tier 1, cheap columns only).")

    values = np.asarray(train_data[1:1 + n_series, :], dtype=np.float64)
    window_cache = {}
    valid_cache = {}

    def get_window(series_idx, start_idx):
        key = (series_idx, start_idx)
        cached = window_cache.get(key)
        if cached is None:
            cached = values[series_idx, start_idx:start_idx + window_size]
            window_cache[key] = cached
        return cached

    def is_valid(series_idx, start_idx):
        key = (series_idx, start_idx)
        if key not in valid_cache:
            valid_cache[key] = CorrTrack_optimize._proxy_window_is_valid(get_window(series_idx, start_idx))
        return valid_cache[key]

    # --- Tier 1: exhaustive label discovery (exact_corr + sketch_dot only). ---
    rows = []
    for step_id, cur_start in enumerate(all_starts):
        min_lag_start = max(0, cur_start - n_lags)
        lag_starts = sorted(set(s for s in range(cur_start, min_lag_start - 1, -window_step) if 0 <= s <= max_start))
        if not lag_starts or cur_start not in lag_starts:
            continue
        norm_by_start = {s: sketch_cache.get(s)[1] for s in lag_starts}
        for s_current in range(n_series):
            if not is_valid(s_current, cur_start):
                continue
            x = get_window(s_current, cur_start)
            q_vec = norm_by_start[cur_start].get(proxy_ids[s_current])
            if q_vec is None:
                continue
            for lag_start in lag_starts:
                for s_other in range(n_series):
                    if lag_start == cur_start and s_other <= s_current:
                        continue
                    if not is_valid(s_other, lag_start):
                        continue
                    y = get_window(s_other, lag_start)
                    corr, _dist = _fast_corr_and_dist(x, y)
                    if not np.isfinite(corr):
                        continue
                    x_vec = norm_by_start[lag_start].get(proxy_ids[s_other])
                    if x_vec is None:
                        continue
                    sketch_score = float(np.dot(q_vec, x_vec))
                    rows.append((s_current, cur_start, s_other, lag_start, float(corr), sketch_score))
        if step_id % 50 == 0:
            print(f"  Tier 1 progress: step {step_id}/{len(all_starts)}, {len(rows)} pairs so far", flush=True)

    label_df = pd.DataFrame(rows, columns=["s_current", "cur_start", "s_other", "lag_start", "exact_corr", "sketch_dot"])
    label_df["cls"] = np.where(label_df["exact_corr"] >= tau, "pos",
                       np.where(label_df["exact_corr"] <= -tau, "neg", "non"))
    n_pos = int((label_df["cls"] == "pos").sum())
    n_neg = int((label_df["cls"] == "neg").sum())
    n_non = int((label_df["cls"] == "non").sum())
    print(f"Tier 1 complete: {len(label_df)} pairs. pos={n_pos} neg={n_neg} non={n_non}.")

    # --- Tier 2 selection: exhaustive if under budget, else stratified sample. ---
    if len(label_df) <= args.max_feature_rows:
        selected = label_df
        print(f"Tier 2: computing full feature set on all {len(selected)} pairs (under --max-feature-rows).")
    else:
        rng = np.random.default_rng(args.random_state)
        pos_df = label_df[label_df["cls"] == "pos"]
        neg_df = label_df[label_df["cls"] == "neg"]
        remaining = max(0, args.max_feature_rows - len(pos_df) - len(neg_df))
        band_mask = (label_df["sketch_dot"].abs() >= args.hard_band_lo) & (label_df["sketch_dot"].abs() <= args.hard_band_hi) & (label_df["cls"] == "non")
        band_df = label_df[band_mask]
        non_df = label_df[(label_df["cls"] == "non") & ~band_mask]
        n_band = min(len(band_df), remaining // 2)
        n_non = min(len(non_df), remaining - n_band)
        band_sample = band_df.sample(n=n_band, random_state=args.random_state) if n_band else band_df.iloc[:0]
        non_sample = non_df.sample(n=n_non, random_state=args.random_state) if n_non else non_df.iloc[:0]
        selected = pd.concat([pos_df, neg_df, band_sample, non_sample], ignore_index=True)
        print(f"Tier 2: population {len(label_df)} exceeds --max-feature-rows={args.max_feature_rows}; "
              f"stratified sample of {len(selected)} pairs (pos={len(pos_df)} neg={len(neg_df)} "
              f"near_gamma_band={len(band_sample)} random_non={len(non_sample)}).")

    # --- Feature-family config. ---
    n_vectors = args.n_vectors
    proj_r_max = max(args.projection_r_values)
    proj_rng = np.random.default_rng(4242)
    proj_dirs = proj_rng.normal(size=(proj_r_max, n_vectors))
    proj_dirs /= np.linalg.norm(proj_dirs, axis=1, keepdims=True) + EPS

    def _block_indices(n, b):
        return np.array_split(np.arange(n), b)

    block_idx_by_b = {b: _block_indices(n_vectors, b) for b in args.block_b_values}
    first_block_idx = block_idx_by_b[min(args.block_b_values)][0]

    feat_cfg = dict(
        m_values=args.m_values, r_values=args.r_values, delta_values=args.delta_values,
        theta_values=args.theta_values, include_kendall=args.include_kendall,
        proj_r_values=args.projection_r_values, block_idx_by_b=block_idx_by_b,
        ub_m_values=args.ub_m_values, first_block_idx=first_block_idx,
    )

    precompute = WindowPrecomputeCache(sketch_cache, proxy_ids, args.m_values, proj_dirs)

    # Written in bounded chunks (not accumulated in one Python list of dicts)
    # -- with the full A-K parameter sweep (~300+ columns), a list of ~550k
    # per-pair dicts is far more memory-hungry than the equivalent columnar
    # array and OOM-killed the first attempt at this run (dmesg confirmed:
    # killed at ~4.5GB resident on a 7.6GB machine, before ever reaching
    # pd.DataFrame()). Chunking bounds peak memory to one chunk regardless
    # of total pair count; pair_features.csv is re-read afterward for the
    # (much cheaper, columnar) downstream analysis.
    args.result_folder.mkdir(parents=True, exist_ok=True)
    pair_path = args.result_folder / "pair_features.csv"
    pair_path.unlink(missing_ok=True)  # avoid appending to a stale/partial file from a prior crashed run
    chunk_size = 20000
    feature_rows = []
    n_written = 0
    n_selected = len(selected)
    wrote_header = False
    for i, r in enumerate(selected.itertuples(index=False)):
        fq = precompute.get(r.s_current, r.cur_start)
        fx = precompute.get(r.s_other, r.lag_start)
        if fq is None or fx is None:
            continue
        q, x = fq["norm_w"], fx["norm_w"]
        feats = compute_pair_features(q, x, fq, fx, feat_cfg)
        feats["exact_corr"] = r.exact_corr
        feats["cls"] = r.cls
        feats["s_current"] = r.s_current
        feats["cur_start"] = r.cur_start
        feats["s_other"] = r.s_other
        feats["lag_start"] = r.lag_start
        feature_rows.append(feats)
        if len(feature_rows) >= chunk_size:
            chunk_df = pd.DataFrame(feature_rows)
            chunk_df.to_csv(pair_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            n_written += len(chunk_df)
            feature_rows = []
            print(f"  Tier 2 progress: {i + 1}/{n_selected} pairs processed, {n_written} written to disk", flush=True)
    if feature_rows:
        chunk_df = pd.DataFrame(feature_rows)
        chunk_df.to_csv(pair_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        n_written += len(chunk_df)
        feature_rows = []

    # Read back with dtypes specified UP FRONT, not inferred-then-downcast --
    # the previous version of this script loaded the full CSV as float64
    # (pandas' default) and only downcast to float32 afterward, so the
    # float64 intermediate itself was what triggered a second OOM kill here
    # (confirmed via dmesg: killed at ~5GB resident right after this read
    # started, immediately following the chunked-write fix that solved the
    # first OOM). Passing dtype= directly avoids ever materializing the
    # float64 version and skips pandas' type-inference pass entirely.
    int_cols = {"s_current", "cur_start", "s_other", "lag_start"}
    header_cols = pd.read_csv(pair_path, nrows=0).columns
    dtype_map = {
        col: ("category" if col == "cls" else np.int32 if col in int_cols else np.float32)
        for col in header_cols
    }
    pair_df = pd.read_csv(pair_path, dtype=dtype_map)
    print(f"Wrote {pair_path} ({len(pair_df)} rows, {len(pair_df.columns)} columns)")

    _run_downstream_analysis(pair_df, args)
    print("Done.")


def _run_downstream_analysis(pair_df, args):
    """feature_separability_summary.csv, combined cheap-feature model,
    hard_band_analysis.csv, and plots -- shared between the normal path and
    --resume-from-pair-features (which skips straight to this point against
    an already-collected pair_features.csv)."""
    non_feature_cols = {"exact_corr", "cls", "s_current", "cur_start", "s_other", "lag_start"}
    feature_cols = [c for c in pair_df.columns if c not in non_feature_cols]

    # --- feature_separability_summary.csv ---
    summary_df = build_separability_summary(pair_df, feature_cols, "cls", args.test_frac, args.random_state)
    summary_path = args.result_folder / "feature_separability_summary.csv"
    summary_df.sort_values("candidate_reduction_at_95_recall", ascending=False).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df)} rows)")

    combined_result, cheap_cols = fit_combined_cheap_model(pair_df, feature_cols, "cls", args.test_frac, args.random_state)
    if combined_result is not None:
        coeffs = combined_result.pop("coefficients")
        combined_path = args.result_folder / "combined_cheap_model_summary.csv"
        pd.DataFrame([combined_result]).to_csv(combined_path, index=False)
        coeff_path = args.result_folder / "combined_cheap_model_coefficients.csv"
        pd.DataFrame(sorted(coeffs.items(), key=lambda kv: -abs(kv[1])), columns=["feature", "coefficient"]).to_csv(coeff_path, index=False)
        print(f"Wrote {combined_path} and {coeff_path} ({len(cheap_cols)} cheap features used)")
        print(f"Combined cheap-feature model: AUC={combined_result['auc_roc']:.4f} "
              f"candidate_reduction@95recall={combined_result['candidate_reduction_at_95_recall']:.4f}")
    else:
        print("Combined cheap-feature model: skipped (not enough data or no eligible cheap features).")

    # --- hard_band_analysis.csv ---
    hard_band_rows = []
    for lo, hi, band_name in ((args.hard_band_lo, args.hard_band_hi, f"{args.hard_band_lo}-{args.hard_band_hi}"),
                               (args.hard_band2_lo, args.hard_band2_hi, f"{args.hard_band2_lo}-{args.hard_band2_hi}")):
        band_mask = (pair_df["abs_sketch_dot"] >= lo) & (pair_df["abs_sketch_dot"] <= hi)
        band_df = pair_df[band_mask]
        print(f"Hard band [{lo},{hi}]: {len(band_df)} pairs ({int((band_df['cls'] != 'non').sum())} exact positives/negatives).")
        if len(band_df) < 20:
            continue
        band_summary = build_separability_summary(band_df, feature_cols, "cls", args.test_frac, args.random_state)
        band_summary.insert(0, "band", band_name)
        hard_band_rows.append(band_summary)
    if hard_band_rows:
        hard_band_df = pd.concat(hard_band_rows, ignore_index=True)
        hard_band_path = args.result_folder / "hard_band_analysis.csv"
        hard_band_df.sort_values("candidate_reduction_at_95_recall", ascending=False).to_csv(hard_band_path, index=False)
        print(f"Wrote {hard_band_path} ({len(hard_band_df)} rows)")
    else:
        print("Hard band analysis: skipped (too few pairs in either band).")

    # --- plots ---
    if not args.no_plots:
        plots_dir = args.result_folder / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        abs_pos_rows = summary_df[summary_df["label_type"] == "is_abs_pos"].sort_values(
            "candidate_reduction_at_95_recall", ascending=False)
        top_features = list(abs_pos_rows["feature_name"].head(args.plot_top_n))
        representative = [
            "sketch_dot", "sign_agree_frac", "topm_jaccard_m8", "partial_dot_qtopm_m8",
            "topr_abs_contribution_ratio_r4", "coord_abscontrib_t0p01", "sorted_abs_best_l2",
            "spearman_corr_values", "projection_abs_best_l2_r8", "simhash_hamming_r8",
            "block_dot_abs_max_b4", "ub_abs_union_m8",
        ]
        plot_features = set(top_features) | (set(representative) & set(feature_cols))
        if args.plot_all_families:
            plot_features |= set(feature_cols)
        print(f"Plotting {len(plot_features)} features to {plots_dir}")
        for feat in plot_features:
            try:
                plot_feature(pair_df, feat, "cls", plots_dir / f"{feat}.png", "exact_corr")
            except Exception as exc:
                print(f"  plot failed for {feat}: {exc}")


if __name__ == "__main__":
    main()
