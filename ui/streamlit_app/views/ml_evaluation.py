"""Model evaluation: how accurately each ML model predicts efficiency."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from _shared import AXIS, BLUE, INK_2, cards, empty_state, page_header, session, show_table

from dssc_discovery.evaluation.cross_species import run_cross_species_cv
from dssc_discovery.evaluation.cv_harness import (
    default_feature_specs,
    default_model_specs,
    run_grouped_nested_cv,
)
from dssc_discovery.evaluation.diagnostics import diagnostic_report
from dssc_discovery.evaluation.temporal_split import run_forward_extrapolation_cv
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.registry import FEATURE_BUDGET, feature_names
from dssc_discovery.reports.ui_queries import model_runs_frame

STRATEGIES = {
    "A · Interpolation": "Hold out whole recipes: how well does the model fill gaps inside the explored region?",
    "B · Forward": "Train on earlier stages, test on the latest BO cycle: does it predict new proposals?",
    "C · Cross-species": "Hold out one plant species: does what it learned transfer to a new plant?",
}
MODEL_LABELS = {
    "mean_baseline": "Mean (baseline)",
    "domain_linear_baseline": "Domain linear (baseline)",
    "ridge": "Ridge regression",
    "lasso": "Lasso regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "gpr": "Gaussian process",
}

page_header(
    "Model evaluation",
    "Eight models — including two simple baselines every real model must beat — are scored "
    "with leak-proof cross-validation. Lower error is better; bars show 95% confidence intervals.",
    eyebrow="Machine learning",
)
cards([(name, text) for name, text in STRATEGIES.items()])
st.write("")

data = build_ml_dataset(session())
if data.empty:
    empty_state("No measured devices yet — models need data to evaluate.")
    st.stop()
if data["is_synthetic"].any():
    st.warning(
        "Evaluating on **synthetic** data: this demonstrates the pipeline, not real accuracy.",
        icon="🧪",
    )

with st.form("evaluation"):
    features = st.multiselect(
        f"Input features (tree models use at most {FEATURE_BUDGET['tree']}, "
        f"linear/GP models at most {FEATURE_BUDGET['linear_or_gp']})",
        feature_names(stage="design"),
        default=["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"],
    )
    c1, c2 = st.columns([2, 1])
    strategies = c1.multiselect(
        "Validation strategies", list(STRATEGIES), default=["A · Interpolation"]
    )
    n_repeats = c2.slider("Repeats (more = steadier estimate, slower)", 1, 10, 2)
    run = st.form_submit_button("Run evaluation", type="primary", disabled=False)

if run and features:
    specs = default_feature_specs(features)
    models = default_model_specs(specs, specs)
    results = []
    with st.spinner("Training and cross-validating eight models…"):
        if "A · Interpolation" in strategies:
            results.append(
                run_grouped_nested_cv(data, models, n_outer_splits=5, n_repeats=n_repeats)
            )
        if "B · Forward" in strategies:
            try:
                results.append(run_forward_extrapolation_cv(data, models, n_repeats=n_repeats))
            except ValueError as exc:
                st.info(f"Forward validation skipped: {exc}")
        if "C · Cross-species" in strategies:
            try:
                results.append(run_cross_species_cv(data, models, n_repeats=n_repeats))
            except ValueError as exc:
                st.info(f"Cross-species validation skipped: {exc}")
    st.session_state["ml_results"] = (results, models)


def error_chart(summary):  # type: ignore[no-untyped-def]
    rows = summary.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 0.8))
    y = np.arange(len(rows))
    colours = [AXIS if f == "baseline" else BLUE for f in rows["family"]]
    ax.hlines(y, rows["mae_ci_low"], rows["mae_ci_high"], color=colours, linewidth=2)
    ax.scatter(rows["mae"], y, color=colours, s=60, zorder=3)
    for yi, value in zip(y, rows["mae"], strict=True):
        ax.annotate(
            f"{value:.2f}",
            (value, yi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_2,
        )
    ax.set_yticks(y, [MODEL_LABELS.get(m, m) for m in rows["model"]])
    ax.set_xlabel("Mean absolute error in PCE (percentage points) — lower is better")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


if "ml_results" in st.session_state:
    results, models = st.session_state["ml_results"]
    if not results:
        empty_state("No strategy could run on this dataset.")
        st.stop()
    tabs = st.tabs([r.config["strategy"].replace("_", " ") for r in results] + ["Diagnostics"])
    for tab, result in zip(tabs, results, strict=False):
        with tab:
            summary = result.summary
            best = summary[summary["family"] != "baseline"].sort_values("mae").iloc[0]
            baseline = summary[summary["family"] == "baseline"]["mae"].min()
            m1, m2, m3 = st.columns(3)
            m1.metric("Best model", MODEL_LABELS.get(best["model"], best["model"]))
            m2.metric(
                "Its error (MAE)",
                f"{best['mae']:.2f}",
                f"{best['mae'] - baseline:+.2f} vs best baseline",
                delta_color="inverse",
            )
            m3.metric("Rows evaluated", result.config["n_rows_evaluated"])
            st.pyplot(error_chart(summary))
            st.caption(
                "Grey = baselines, blue = ML models. A model is only useful if its error is clearly below the baselines."
            )
            table = summary.assign(model=summary["model"].map(lambda m: MODEL_LABELS.get(m, m)))
            show_table(
                table, columns=["model", "mae", "rmse", "r2"], key=f"t-{result.config['strategy']}"
            )
    with tabs[-1]:
        names = {MODEL_LABELS.get(m.name, m.name): m for m in models}
        spec = names[st.selectbox("Model to diagnose", list(names), index=len(names) - 1)]
        report = diagnostic_report(results[0], data, spec, include_learning_curve=True)
        d1, d2, d3 = st.columns(3)
        d1.metric("Training error", f"{report.overfitting.train_mae:.2f}")
        d2.metric("Cross-validated error", f"{report.overfitting.cv_mae:.2f}")
        d3.metric(
            "Error grows with prediction?",
            "Yes ⚠️" if report.heteroscedasticity.detected else "No ✓",
        )
        figures = list(report.figures.items())
        for i in range(0, len(figures), 2):
            cols = st.columns(2)
            for col, (_, fig) in zip(cols, figures[i : i + 2], strict=False):
                col.pyplot(fig)
else:
    st.info("Choose features and strategies, then press **Run evaluation**.", icon="👆")

st.markdown("#### Logged model runs")
show_table(
    model_runs_frame(session()),
    columns=[
        "data",
        "model_name",
        "strategy",
        "model_family",
        "metric_mae",
        "metric_rmse",
        "created_at",
    ],
    empty_message="No model runs logged yet.",
)
