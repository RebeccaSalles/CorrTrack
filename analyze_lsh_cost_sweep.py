"""analyze_lsh_cost_sweep.py -- cost-model fitting + m=10,000 extrapolation
for experiment_lsh_cost_sweep.py's output.

(2026-08-26) Built after the user decided ("Extrapolate from what we have")
to stop chasing m=1000/3000 timing-only cells given repeated WSL OOM
crashes on their machine -- the largest fully-measured point is m=200
(synthetic). Extrapolating to m=10,000 from there is a 50x leap, far
beyond the original plan's assumption of extrapolating ~3.3x from a
largest measured point of m=3000. This script does NOT hide that: every
projection is reported as a wide range with the extrapolation distance
stated explicitly, calibrated against a held-out-SCALE cross-validation
(train on m<=100, predict m=200, compare to the real measurement) rather
than trusted blindly.

Design (per the approved plan, adapted to the data actually collected):
  - sk_time ~ f(m), cand_time ~ f(m) and ~f(L): log-log OLS (power law),
    fit separately along each OFAT line (m swept at fixed L=32, L swept
    at fixed m=200) since the collected grid is factorial+OFAT, not a
    full m x L cross -- a joint multi-variable fit isn't identifiable
    from this data. The m=10,000 projection uses ONLY the m-line fit
    (already measured at L=REFERENCE_L=32, so no separate combination
    with the L-line is needed); the L-line is reported purely as its own
    diagnostic, characterizing L's effect at the single m=200 point where
    it was swept.
  - val_time ~ f(tested_candidates): log-log OLS across ALL rows (this
    one has enough spread to fit densely, unlike the OFAT-only cost
    components).
  - Brute-force cost: CLOSED FORM, not regressed -- m*(m-1)/2 * L *
    n_steps * const. const is CALIBRATED (mean/median across every
    bf_measured=True row), with its own variability reported (if it's
    unstable across rows, that's flagged, not hidden).
  - Held-out-scale CV: fit on m in {50,100} only, predict candidate_time
    at m=200, report relative error against the real measured value.
  - Every fitted line's R^2 is reported -- a low R^2 on 3-4 points is
    real information (the fit's extrapolation is unreliable), not noise
    to ignore.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("tmp_artifacts/lsh_cost_sweep/lsh_cost_sweep.csv")
LONG_N_STEPS_MIN = 1000  # excludes the noisy short-slice diagnostic rows
TARGET_M = 10_000
REFERENCE_L = 32  # the L value both OFAT lines and the L-sweep itself pivot on


def loglog_fit(x, y, label):
    """Power-law fit y = c * x^a via OLS on (log x, log y). Returns a dict
    with the exponent, its intercept-derived constant, R^2, and the raw
    (x, y) pairs actually used -- report R^2 and n honestly; a 2-3 point
    "fit" is a fact about the fit's own fragility, not swept under the rug.
    Also carries what predict_loglog_interval needs: residual std, the
    spread of log(x) around its mean, and n -- the classic OLS ingredients
    for a prediction interval that WIDENS with distance from the fitted
    range, which is exactly the honest behavior a 50x extrapolation needs
    (a fixed +/-X% band, independent of how far out you project, would
    understate risk badly at that distance -- see module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return {"label": label, "n": len(x), "exponent": None, "const": None, "r2": None,
                "note": "fewer than 2 usable points -- cannot fit"}
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    resid = log_y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    n = len(x)
    dof = max(n - 2, 1)
    resid_var = ss_res / dof
    log_x_mean = float(np.mean(log_x))
    ss_x = float(np.sum((log_x - log_x_mean) ** 2))
    return {
        "label": label, "n": n, "exponent": float(slope), "const": float(np.exp(intercept)),
        "r2": float(r2), "x": x.tolist(), "y": y.tolist(),
        "_resid_var": resid_var, "_dof": dof, "_log_x_mean": log_x_mean, "_ss_x": ss_x,
        "_intercept": float(intercept),
    }


def predict_loglog(fit, x_new):
    if fit["exponent"] is None:
        return None
    return fit["const"] * (x_new ** fit["exponent"])


def predict_loglog_interval(fit, x_new, confidence=0.80):
    """OLS prediction interval for a NEW point at x_new, back-transformed
    from log-space -- widens with (log(x_new) - mean(log_x_fitted))^2, so
    extrapolating 50x beyond the fitted range gives a properly wide
    interval instead of a falsely narrow fixed-percentage band. With only
    n=3 points (dof=1 here), the t-critical value is large and the
    interval will be wide -- that width is real information about how
    little this extrapolation is actually supported, not a bug."""
    if fit["exponent"] is None or fit.get("_ss_x") is None:
        return None, None
    from scipy import stats
    log_x_new = np.log(x_new)
    se_pred = np.sqrt(fit["_resid_var"] * (
        1 + 1.0 / fit["n"] + (log_x_new - fit["_log_x_mean"]) ** 2 / fit["_ss_x"]
    ))
    t_crit = stats.t.ppf(0.5 + confidence / 2, fit["_dof"])
    log_point = fit["exponent"] * log_x_new + fit["_intercept"]
    log_low, log_high = log_point - t_crit * se_pred, log_point + t_crit * se_pred
    return float(np.exp(log_low)), float(np.exp(log_high))


def _public_fit_fields(fit):
    """Drops the internal-only OLS bookkeeping (leading underscore) and the
    raw x/y arrays used only for plotting/debugging -- keeps the report's
    JSON focused on what a reader actually needs (exponent, const, R^2, n)."""
    return {k: v for k, v in fit.items() if k not in ("x", "y") and not k.startswith("_")}


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found -- run experiment_lsh_cost_sweep.py first.")
    df = pd.read_csv(CSV_PATH)
    df_long = df[df["n_steps"] >= LONG_N_STEPS_MIN].copy()

    report = {"generated_note": "See docs/implementation_log.md's 2026-08-26 entry for full context."}

    # ---- m-scaling (synthetic, L=REFERENCE_L, corr_prop=0.05 mid, occ=3.0) ----
    m_line = df_long[
        (df_long["dataset_kind"] == "synthetic") & (df_long["L"] == REFERENCE_L)
        & (df_long["corr_prop_requested"].round(3) == 0.05) & (df_long["target_occupancy"] == 3.0)
    ].sort_values("m")
    largest_measured_m = int(m_line["m"].max()) if len(m_line) else None

    m_fits = {
        "sketch_time": loglog_fit(m_line["m"], m_line["sketch_time"], "sketch_time ~ m^a"),
        "candidate_time": loglog_fit(m_line["m"], m_line["candidate_time"], "candidate_time ~ m^a"),
    }

    # ---- L-scaling (synthetic, m=200 the only fully-L-swept point) ----
    l_line = df_long[
        (df_long["dataset_kind"] == "synthetic") & (df_long["m"] == 200)
        & (df_long["corr_prop_requested"].round(3) == 0.05) & (df_long["target_occupancy"] == 3.0)
    ].sort_values("L")
    l_fits = {
        "sketch_time": loglog_fit(l_line["L"], l_line["sketch_time"], "sketch_time ~ L^b"),
        "candidate_time": loglog_fit(l_line["L"], l_line["candidate_time"], "candidate_time ~ L^b"),
    }

    # ---- validation time vs tested_candidates, across ALL long rows ----
    val_fit = loglog_fit(df_long["tested_candidates"], df_long["validation_time"],
                          "validation_time ~ tested_candidates^c")

    # ---- brute-force closed-form constant ----
    bf_rows = df_long[df_long["bf_measured"] == True].copy()
    bf_rows["pair_step_units"] = bf_rows["m"] * (bf_rows["m"] - 1) / 2 * bf_rows["L"] * bf_rows["n_steps"]
    bf_rows["const_per_unit"] = bf_rows["bf_runtime"] / bf_rows["pair_step_units"]
    bf_const_mean = float(bf_rows["const_per_unit"].mean())
    bf_const_median = float(bf_rows["const_per_unit"].median())
    bf_const_std = float(bf_rows["const_per_unit"].std())
    bf_const_cv = bf_const_std / bf_const_mean if bf_const_mean else None

    # ---- held-out-SCALE cross-validation: train on m in {50,100}, predict m=200 ----
    cv_train = m_line[m_line["m"] <= 100]
    cv_holdout = m_line[m_line["m"] == 200]
    cv_result = {}
    if len(cv_train) >= 2 and len(cv_holdout) == 1:
        cv_fit = loglog_fit(cv_train["m"], cv_train["candidate_time"], "cand_time~m^a (trained on m<=100)")
        actual = float(cv_holdout["candidate_time"].iloc[0])
        predicted = predict_loglog(cv_fit, 200.0)
        rel_err = abs(predicted - actual) / actual if predicted is not None else None
        cv_result = {
            "trained_on_m": cv_train["m"].tolist(), "holdout_m": 200,
            "predicted_candidate_time": predicted, "actual_candidate_time": actual,
            "relative_error": rel_err,
        }

    # ---- m=10,000 projection ----
    extrapolation_factor = (TARGET_M / largest_measured_m) if largest_measured_m else None
    proj = {}
    if m_fits["candidate_time"]["exponent"] is not None:
        # m_fits was already fit along the m_line, which is held at
        # L=REFERENCE_L throughout -- no separate combination with l_fits
        # is needed here, the m-exponent already reflects behavior AT that
        # L. l_fits (reported separately above) characterizes how the cost
        # changes with L on its own, at the single m=200 point available --
        # informative on its own, not folded into this m-projection.
        anchor_row = m_line[m_line["m"] == largest_measured_m].iloc[0]

        # Point estimates (single best-fit line each).
        smart_cand_point = predict_loglog(m_fits["candidate_time"], TARGET_M)
        smart_sketch_point = predict_loglog(m_fits["sketch_time"], TARGET_M)
        smart_total_point = (smart_cand_point or 0) + (smart_sketch_point or 0)

        # Proper OLS prediction interval (80% CI), not a fixed +/-X% band --
        # this widens correctly with how far m=10,000 sits from the fitted
        # m in {50,100,200} range (a 50x extrapolation from only 3 points,
        # dof=1, WILL produce a wide interval -- that width is the honest
        # answer, not a defect in the method).
        cand_low, cand_high = predict_loglog_interval(m_fits["candidate_time"], TARGET_M)
        sketch_low, sketch_high = predict_loglog_interval(m_fits["sketch_time"], TARGET_M)
        smart_total_low = (cand_low or 0) + (sketch_low or 0)
        smart_total_high = (cand_high or 0) + (sketch_high or 0)

        bf_time_at_target = bf_const_median * (TARGET_M * (TARGET_M - 1) / 2) * REFERENCE_L * anchor_row["n_steps"]
        speedup_point = bf_time_at_target / smart_total_point if smart_total_point else None
        # Larger smart-side time -> smaller speedup, and vice versa.
        speedup_low = bf_time_at_target / smart_total_high if smart_total_high else None
        speedup_high = bf_time_at_target / smart_total_low if smart_total_low else None

        proj = {
            "target_m": TARGET_M, "largest_measured_m": largest_measured_m,
            "extrapolation_factor_x": extrapolation_factor,
            "reference_L": REFERENCE_L,
            "smart_total_time_point_estimate_s": smart_total_point,
            "smart_total_time_80pct_interval_s": [smart_total_low, smart_total_high],
            "bf_closed_form_time_s": bf_time_at_target,
            "speedup_point_estimate": speedup_point,
            "speedup_80pct_interval": [speedup_low, speedup_high],
            "interval_method": "OLS prediction interval (80% CI) on the log-log fit, back-"
                               "transformed -- NOT a fixed percentage band; widens with "
                               "distance from the fitted m range, per predict_loglog_interval. "
                               "The held_out_scale_cv relative_error above is a separate, "
                               "complementary sanity check (interpolation-only, m<=100 -> "
                               "m=200), not the source of this interval.",
        }

    report.update({
        "data_summary": {
            "n_rows_total": len(df), "n_rows_long_n_steps": len(df_long),
            "largest_measured_synthetic_m": largest_measured_m,
            "real_data_m_values_measured": sorted(df[df["dataset_kind"] == "real"]["m"].unique().tolist()),
        },
        "m_scaling_fits": {k: _public_fit_fields(v) for k, v in m_fits.items()},
        "l_scaling_fits": {k: _public_fit_fields(v) for k, v in l_fits.items()},
        "validation_time_fit": _public_fit_fields(val_fit),
        "bruteforce_closed_form": {
            "formula": "bf_runtime ~= const * m*(m-1)/2 * L * n_steps",
            "const_mean": bf_const_mean, "const_median": bf_const_median,
            "const_std": bf_const_std, "const_coefficient_of_variation": bf_const_cv,
            "n_rows_used": len(bf_rows),
            "note": "coefficient_of_variation close to 0 means the closed-form is a "
                    "good fit across every measured (m,L) combination; a large value "
                    "means the 'constant' isn't very constant in practice.",
        },
        "held_out_scale_cv": cv_result,
        "m_10000_projection": proj,
        "important_caveats": [
            f"Largest fully-measured synthetic m is {largest_measured_m}; the m=10,000 "
            f"projection extrapolates "
            f"{'%.1fx' % extrapolation_factor if extrapolation_factor is not None else 'an undefined amount'} "
            "beyond that -- the original plan "
            "assumed extrapolating ~3.3x from a largest point of m=3000; this is a much "
            "larger, less trustworthy leap, forced by memory constraints on the "
            "development machine (m=1000/3000 timing-only cells consistently failed with "
            "OOM even under a 10GB per-worker cap -- see docs/implementation_log.md's "
            "2026-08-26 entries).",
            "The m=10,000 projection uses only the m-scaling fit (already measured at "
            "L=32); the L-scaling fit is a separate diagnostic (only one m=200 point was "
            "fully swept across L) and is NOT combined into the projection, so it says "
            "nothing about whether L's effect stays this way at m=10,000.",
            "n_vectors/candidate_lsh_n_bands are TUNED per cell (2026-08-25), not held "
            "fixed across the m- or L-sweep -- e.g. the L=64 point's much higher "
            "sketch_time likely reflects a larger calibrated n_vectors at that cell, not "
            "a pure L effect. This confounds the m/L cost model to an unquantified degree.",
            "Real-data anchors only reached m=50 (m=100/121 never completed before this "
            "analysis was requested) -- no real-data cross-check of the synthetic-fitted "
            "model at larger m is available.",
        ],
    })

    out_path = Path("tmp_artifacts/lsh_cost_sweep/cost_model_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
