"""ML model evaluation (Chunk 11.4): the 4.8 comparison table (baselines
always included) and 4.6 diagnostics."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.evaluation.cross_species import run_cross_species_cv
from dssc_discovery.evaluation.cv_harness import (
    default_feature_specs,
    default_model_specs,
    run_grouped_nested_cv,
    strategy_comparison_table,
)
from dssc_discovery.evaluation.diagnostics import diagnostic_report
from dssc_discovery.evaluation.temporal_split import run_forward_extrapolation_cv
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.registry import FEATURE_BUDGET, feature_names
from dssc_discovery.reports.ui_queries import model_runs_frame

st.title("ML evaluation")
data = build_ml_dataset(session())
if data.empty:
    st.info("No measured devices yet.")
    st.stop()
if data["is_synthetic"].any():
    st.warning("This dataset contains SYNTHETIC rows: results only show the pipeline works.")

features = st.multiselect(
    f"Features (tree budget {FEATURE_BUDGET['tree']}, "
    f"linear/GP budget {FEATURE_BUDGET['linear_or_gp']})",
    feature_names(stage="design"),
    default=["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"],
)
strategies = st.multiselect(
    "Validation strategies",
    ["A (grouped k-fold)", "B (forward)", "C (cross-species)"],
    default=["A (grouped k-fold)"],
)
n_repeats = st.slider("Repeats", 1, 10, 2)

if st.button("Run evaluation", disabled=not features):
    specs = default_feature_specs(features)
    models = default_model_specs(specs, specs)
    results = []
    with st.spinner("Running nested cross-validation..."):
        if "A (grouped k-fold)" in strategies:
            results.append(
                run_grouped_nested_cv(data, models, n_outer_splits=5, n_repeats=n_repeats)
            )
        if "B (forward)" in strategies:
            results.append(run_forward_extrapolation_cv(data, models, n_repeats=n_repeats))
        if "C (cross-species)" in strategies:
            results.append(run_cross_species_cv(data, models, n_repeats=n_repeats))
    st.session_state["ml_results"] = (results, models)

if "ml_results" in st.session_state:
    results, models = st.session_state["ml_results"]
    st.subheader("Model comparison (baselines always shown)")
    st.dataframe(strategy_comparison_table(results), use_container_width=True, hide_index=True)
    model_name = st.selectbox("Diagnostics for model", [m.name for m in models])
    spec = next(m for m in models if m.name == model_name)
    report = diagnostic_report(results[0], data, spec, include_learning_curve=True)
    st.caption(
        f"Heteroscedasticity: rho={report.heteroscedasticity.spearman_rho:.2f}, "
        f"p={report.heteroscedasticity.p_value:.3f} "
        f"({'DETECTED' if report.heteroscedasticity.detected else 'not detected'}); "
        f"train MAE {report.overfitting.train_mae:.2f} vs CV MAE {report.overfitting.cv_mae:.2f}"
    )
    for name, fig in report.figures.items():
        st.pyplot(fig)

st.subheader("Logged model runs")
show_table(model_runs_frame(session()))
