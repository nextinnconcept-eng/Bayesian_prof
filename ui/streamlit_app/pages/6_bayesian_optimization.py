"""Bayesian optimization (Chunk 11.6): recommendations with rationale,
convergence diagnostics, and the PCE-stability Pareto front."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.optimization.convergence import convergence_report
from dssc_discovery.reports.ui_queries import bo_iterations_frame, recommendations_frame
from dssc_discovery.stability.retention import build_multi_objective_dataset, stability_coverage
from dssc_discovery.visualization.pareto_plots import plot_pareto_front

st.title("Bayesian optimization")
data = build_ml_dataset(session())
with_data = set(data["species_code"]) if not data.empty else set()
ordered = sorted(SpeciesCode, key=lambda s: s.value not in with_data)  # species with data first
species = st.selectbox(
    "Species",
    ordered,
    format_func=lambda s: s.value + ("" if s.value in with_data else " (no data yet)"),
)

recommendations = recommendations_frame(session())
if not recommendations.empty:
    recommendations = recommendations[recommendations["species_code"] == species.value]
st.subheader("Recommendations")
show_table(recommendations)

st.subheader("BO iterations")
iterations = bo_iterations_frame(session())
show_table(
    iterations[iterations["species_code"] == species.value] if not iterations.empty else iterations
)

if species.value in with_data:
    st.subheader("Convergence (reported, never automatic)")
    report = convergence_report(session(), data, species)
    st.info(report.summary)
    st.json(
        {
            "best_observed_by_stage": report.best_observed_by_stage,
            "predicted_improvement_by_cycle": report.predicted_improvement_by_cycle,
        }
    )

    st.subheader("PCE vs stability")
    mo_data = build_multi_objective_dataset(session())
    coverage = stability_coverage(mo_data)
    st.caption(
        f"Stability measured for {coverage['measured']} devices; "
        f"{coverage['not_yet_aged']} not yet aged."
    )
    species_rows = mo_data[mo_data["species_code"] == species.value]
    if (species_rows["stability_status"] == "measured").any():
        st.pyplot(plot_pareto_front(species_rows))
    else:
        st.info("No stability measurements for this species yet.")
else:
    st.info(f"No measured devices for {species.value} yet.")
