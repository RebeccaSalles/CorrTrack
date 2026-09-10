"""analyze_lsh_cost_sweep_report_data.py -- builds the full data blob behind
the interactive HTML report (parameter sensitivity, accuracy, runtime,
speedup, phase-by-phase complexity, filtering funnel, extrapolation to
m=500/1000/10000). Extends analyze_lsh_cost_sweep.py's core fitting
machinery (loglog_fit / predict_loglog_interval) rather than duplicating it.

(2026-08-26) Built after the user asked for a broader dashboard than the
original cost-model-only analysis: parameter sensitivity, accuracy/runtime/
speedup together, complexity "in general and by phase", filtering
capabilities, and extrapolation retargeted to the more defensible m=500/
1000 (in addition to keeping 10,000 for reference, heavily caveated) given
the largest fully-measured point is m=200.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_lsh_cost_sweep import loglog_fit, predict_loglog, predict_loglog_interval
from experiment_lsh_cost_sweep import TARGET_RECALL, N_VECTORS_GRID, N_BANDS_GRID

CSV_PATH = Path("tmp_artifacts/lsh_cost_sweep/lsh_cost_sweep.csv")
OUT_PATH = Path("tmp_artifacts/lsh_cost_sweep/report_data.json")
LONG_MIN = 1000
REFERENCE_L = 32
REFERENCE_CORR_PROP = 0.05
REFERENCE_OCC = 3.0
EXTRAPOLATION_TARGETS = [500, 1000, 10_000]
PHASES = ["sketch_time", "candidate_time", "validation_time", "monitor_time"]


def clean(d):
    """Drop internal (_-prefixed) and array-valued fields not meant for the report JSON."""
    return {k: v for k, v in d.items() if not k.startswith("_") and k not in ("x", "y")}


def main():
    df = pd.read_csv(CSV_PATH)
    df_long = df[df["n_steps"] >= LONG_MIN].copy()
    m_line = df_long[
        (df_long["dataset_kind"] == "synthetic") & (df_long["L"] == REFERENCE_L)
        & (df_long["corr_prop_requested"].round(3) == REFERENCE_CORR_PROP)
        & (df_long["target_occupancy"] == REFERENCE_OCC)
    ].sort_values("m")
    l_line = df_long[
        (df_long["dataset_kind"] == "synthetic") & (df_long["m"] == 200)
        & (df_long["corr_prop_requested"].round(3) == REFERENCE_CORR_PROP)
        & (df_long["target_occupancy"] == REFERENCE_OCC)
    ].sort_values("L")
    cp_line = df_long[
        (df_long["dataset_kind"] == "synthetic") & (df_long["m"] == 200) & (df_long["L"] == REFERENCE_L)
        & (df_long["target_occupancy"] == REFERENCE_OCC)
    ].sort_values("corr_prop_requested")
    occ_line = df_long[
        (df_long["dataset_kind"] == "synthetic") & (df_long["m"] == 200) & (df_long["L"] == REFERENCE_L)
        & (df_long["corr_prop_requested"].round(3) == REFERENCE_CORR_PROP)
    ].sort_values("target_occupancy")
    largest_m = int(m_line["m"].max())

    # ---------------------------------------------------------------- rows
    def row_records(sub):
        cols = ["dataset_kind", "m", "L", "corr_prop_requested", "target_occupancy", "n_steps",
                "n_vectors_selected", "n_bands_selected", "gamma_selected",
                "n_vectors_bands_met_target_recall", "n_vectors_bands_trials_tried",
                "gamma_met_target_recall", "gamma_trials_tried",
                "sketch_time", "candidate_time", "validation_time", "monitor_time", "smart_wall_time",
                "bf_runtime", "speedup", "precision", "recall", "f1_score",
                "total_candidates", "tested_candidates", "validated_candidates",
                "bf_tested", "bf_correlated", "peak_rss_gb"]
        out = sub[cols].replace({np.nan: None}).to_dict(orient="records")
        return out

    all_rows = row_records(df)
    long_rows = row_records(df_long)

    # -------------------------------------------------- phase-by-phase fits
    phase_m_fits = {}
    phase_l_fits = {}
    for phase in PHASES:
        phase_m_fits[phase] = clean(loglog_fit(m_line["m"], m_line[phase], f"{phase} ~ m^a"))
        phase_l_fits[phase] = clean(loglog_fit(l_line["L"], l_line[phase], f"{phase} ~ L^b"))

    # phase share of total wall time at each measured m (general + by-phase view)
    phase_share = []
    for _, r in m_line.iterrows():
        total = sum(r[p] for p in PHASES)
        phase_share.append({
            "m": int(r["m"]),
            **{p: (float(r[p]) / total if total else None) for p in PHASES},
            "total_s": float(total),
        })

    # ---------------------------------------------------- validation ~ tested
    val_fit = clean(loglog_fit(df_long["tested_candidates"], df_long["validation_time"],
                                "validation_time ~ tested_candidates^c"))

    # ------------------------------------------------- brute-force closed form
    bf_rows = df_long[df_long["bf_runtime"].notna()].copy()
    bf_rows["pair_step_units"] = bf_rows["m"] * (bf_rows["m"] - 1) / 2 * bf_rows["L"] * bf_rows["n_steps"]
    bf_rows["const_per_unit"] = bf_rows["bf_runtime"] / bf_rows["pair_step_units"]
    bf_const_median = float(bf_rows["const_per_unit"].median())
    bf_const_cv = float(bf_rows["const_per_unit"].std() / bf_rows["const_per_unit"].mean())

    # ------------------------------------------------------- held-out-scale CV
    cv_train = m_line[m_line["m"] <= 100]
    cv_holdout = m_line[m_line["m"] == 200]
    cv = {}
    if len(cv_train) >= 2 and len(cv_holdout) == 1:
        cv_fit = loglog_fit(cv_train["m"], cv_train["candidate_time"], "trained on m<=100")
        pred = predict_loglog(cv_fit, 200.0)
        actual = float(cv_holdout["candidate_time"].iloc[0])
        cv = {"trained_on_m": cv_train["m"].tolist(), "holdout_m": 200,
              "predicted": pred, "actual": actual,
              "relative_error": abs(pred - actual) / actual if pred is not None else None}

    # ------------------------------------------------- extrapolation targets
    extrapolation = {}
    for target_m in EXTRAPOLATION_TARGETS:
        anchor = m_line[m_line["m"] == largest_m].iloc[0]
        phase_proj = {}
        smart_total_point = 0.0
        smart_total_low = 0.0
        smart_total_high = 0.0
        for phase in PHASES:
            fit = phase_m_fits[phase]
            if fit.get("exponent") is None:
                continue
            point = predict_loglog({"const": fit["const"], "exponent": fit["exponent"]}, target_m)
            # rebuild full fit dict (with private fields) for the interval calc
            full_fit = loglog_fit(m_line["m"], m_line[phase], phase)
            low, high = predict_loglog_interval(full_fit, target_m)
            phase_proj[phase] = {"point": point, "low": low, "high": high}
            smart_total_point += point or 0
            smart_total_low += low or 0
            smart_total_high += high or 0

        bf_time = bf_const_median * (target_m * (target_m - 1) / 2) * REFERENCE_L * anchor["n_steps"]
        extrapolation[str(target_m)] = {
            "target_m": target_m, "extrapolation_factor_x": target_m / largest_m,
            "phase_projection_s": phase_proj,
            "smart_total_time_point_s": smart_total_point,
            "smart_total_time_80pct_interval_s": [smart_total_low, smart_total_high],
            "bf_closed_form_time_s": bf_time,
            "speedup_point_estimate": bf_time / smart_total_point if smart_total_point else None,
            "speedup_80pct_interval": [
                bf_time / smart_total_high if smart_total_high else None,
                bf_time / smart_total_low if smart_total_low else None,
            ],
        }

    # ------------------------------------------------------- filtering funnel
    funnel = []
    for _, r in df_long.iterrows():
        bf_tested = r["bf_tested"]
        tested = r["tested_candidates"]
        validated = r["validated_candidates"]
        funnel.append({
            "dataset_kind": r["dataset_kind"], "m": int(r["m"]), "L": int(r["L"]),
            "corr_prop_requested": r["corr_prop_requested"] if pd.notna(r["corr_prop_requested"]) else None,
            "bf_tested": int(bf_tested) if pd.notna(bf_tested) else None,
            "tested_candidates": int(tested) if pd.notna(tested) else None,
            "validated_candidates": int(validated) if pd.notna(validated) else None,
            "bf_correlated": int(r["bf_correlated"]) if pd.notna(r["bf_correlated"]) else None,
            "retrieval_reduction": (tested / bf_tested) if bf_tested else None,
            "validation_yield": (validated / tested) if tested else None,
        })

    # -------------------------------------------------- parameter sensitivity
    def sens_records(sub, x_col):
        base_cols = ["m", "L", "n_vectors_selected", "n_bands_selected", "gamma_selected",
                     "n_vectors_bands_met_target_recall", "n_vectors_bands_trials_tried",
                     "gamma_met_target_recall", "gamma_trials_tried",
                     "speedup", "precision", "recall", "f1_score", "candidate_time", "tested_candidates"]
        cols = [x_col] + [c for c in base_cols if c != x_col]
        return sub[cols].replace({np.nan: None}).to_dict(orient="records")

    parameter_sensitivity = {
        "target_occupancy": sens_records(occ_line, "target_occupancy"),
        "corr_prop": sens_records(cp_line, "corr_prop_requested"),
        "L": sens_records(l_line, "L"),
        "m": sens_records(m_line, "m"),
    }

    # ------------------------------------------- hyperopt-phase sensitivity
    # (2026-08-26) Mined from the winning (n_vectors, n_bands, gamma)
    # selection + calibration diagnostics ALREADY persisted per row by
    # calibrate_hyperparams/calibrate_gamma -- see CSV_COLUMNS in
    # experiment_lsh_cost_sweep.py. NOTE (important, disclosed honestly):
    # this is the winning combo's selection and whether/how fast it was
    # found -- NOT the full (n_vectors x n_bands x gamma) recall/cost
    # surface for every combo tried. That per-combo trial grid is computed
    # in calibrate_hyperparams/calibrate_gamma but discarded once the
    # winner is picked; it was never written to disk. Recovering it needs
    # a separate, explicitly-approved re-run that instruments those
    # functions to persist every trial, not just the winner -- not done
    # here since it costs fresh brute-force + CorrTrack compute, however
    # modest.
    def hp_records(sub, x_col):
        cols = [x_col, "dataset_kind", "m", "L", "corr_prop_requested", "target_occupancy",
                "n_vectors_selected", "n_bands_selected", "gamma_selected",
                "n_vectors_bands_met_target_recall", "n_vectors_bands_trials_tried",
                "gamma_met_target_recall", "gamma_trials_tried", "recall"]
        cols = [c for i, c in enumerate(cols) if c not in cols[:i]]  # de-dup if x_col repeats a base col
        return sub[cols].replace({np.nan: None}).to_dict(orient="records")

    failures = df_long[
        (df_long["n_vectors_bands_met_target_recall"] == False)  # noqa: E712
        | (df_long["gamma_met_target_recall"] == False)  # noqa: E712
    ]
    calibration_failures = hp_records(failures, "dataset_kind") if len(failures) else []

    hyperopt_sensitivity = {
        "note": "Winning-combo selection + calibration diagnostics only (see 'note' fields in "
                "the report UI for the full-trial-grid caveat).",
        "n_vectors_grid": list(N_VECTORS_GRID), "n_bands_grid": list(N_BANDS_GRID),
        "target_recall": TARGET_RECALL,
        "by_m": hp_records(m_line, "m"),
        "by_L": hp_records(l_line, "L"),
        "by_corr_prop": hp_records(cp_line, "corr_prop_requested"),
        "by_occupancy": hp_records(occ_line, "target_occupancy"),
        "calibration_failures": calibration_failures,
    }

    # -------------------------------------------------------------- accuracy
    accuracy_summary = {
        "long_runs": {
            "precision_mean": float(df_long["precision"].mean()),
            "recall_mean": float(df_long["recall"].mean()),
            "recall_min": float(df_long["recall"].min()),
            "f1_mean": float(df_long["f1_score"].mean()),
            "n": len(df_long),
        },
        "short_runs": {
            "precision_mean": float(df[df["n_steps"] < LONG_MIN]["precision"].mean()),
            "recall_mean": float(df[df["n_steps"] < LONG_MIN]["recall"].mean()),
            "n": len(df[df["n_steps"] < LONG_MIN]),
            "note": "short (n_steps=60) runs show much lower/zero recall in several cells -- "
                    "a known statistical-power effect (too few true positives visible in a "
                    "short slice), not an algorithm failure. See docs/implementation_log.md.",
        },
    }

    report = {
        "meta": {
            "largest_measured_synthetic_m": largest_m,
            "real_data_m_values": sorted(df[df["dataset_kind"] == "real"]["m"].unique().tolist()),
            "reference_L": REFERENCE_L, "reference_corr_prop": REFERENCE_CORR_PROP,
            "reference_target_occupancy": REFERENCE_OCC,
            "n_rows_total": len(df), "n_rows_long": len(df_long),
        },
        "all_rows": all_rows,
        "long_rows": long_rows,
        "phase_m_fits": phase_m_fits,
        "phase_l_fits": phase_l_fits,
        "phase_share_by_m": phase_share,
        "validation_time_fit": val_fit,
        "bruteforce_closed_form": {"const_median": bf_const_median, "const_cv": bf_const_cv},
        "held_out_scale_cv": cv,
        "extrapolation": extrapolation,
        "filtering_funnel": funnel,
        "parameter_sensitivity": parameter_sensitivity,
        "hyperopt_sensitivity": hyperopt_sensitivity,
        "accuracy_summary": accuracy_summary,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
