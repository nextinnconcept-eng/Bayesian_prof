"""Bayesian optimization: what the optimizer recommends next, and whether it is converging."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from _shared import STAGE_LABELS, empty_state, page_header, session, show_table, stage_chart

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.optimization.convergence import convergence_report
from dssc_discovery.reports.ui_queries import recommendations_frame
from dssc_discovery.stability.retention import build_multi_objective_dataset, stability_coverage
from dssc_discovery.visualization.pareto_plots import plot_pareto_front

page_header(
    "Bayesian optimization",
    "A Gaussian-process model of efficiency proposes the recipes most likely to improve on "
    "the best result so far — balancing promising regions (exploitation) against "
    "uncertain ones (exploration). Stopping is always a researcher's decision.",
    eyebrow="Optimization",
)

data = build_ml_dataset(session())
with_data = set(data["species_code"]) if not data.empty else set()
ordered = sorted(SpeciesCode, key=lambda s: s.value not in with_data)
species = st.selectbox(
    "Plant species",
    ordered,
    format_func=lambda s: s.value + ("" if s.value in with_data else "  (no data yet)"),
)
if species.value not in with_data:
    empty_state(f"No measured devices for {species.value} yet.")
    st.stop()

rows = data[data["species_code"] == species.value]
recs = recommendations_frame(session())
recs = recs[recs["species_code"] == species.value] if not recs.empty else recs
report = convergence_report(session(), data, species)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best PCE observed", f"{rows['power_conversion_efficiency_percent'].max():.2f} %")
k2.metric("BO cycles measured", f"{report.cycles_completed} / {report.cycles_planned}")
k3.metric("Recommendations made", len(recs))
k4.metric("Awaiting approval", int((recs["state"] == "proposed").sum()) if not recs.empty else 0)

left, right = st.columns([1.2, 1])
with left:
    st.markdown("#### Progress across optimization stages")
    stages = [s for s in STAGE_LABELS if s in report.best_observed_by_stage]
    stage_chart(
        stages, [report.best_observed_by_stage[s] for s in stages], "Best PCE so far (%)", 260
    )
with right:
    st.markdown("#### Convergence signals")
    signals = {
        "Predicted gains shrinking": report.improvement_decaying,
        "Best result plateaued": report.plateau,
        "Cycle budget used up": report.budget_exhausted,
    }
    for label, flag in signals.items():
        st.markdown(f"{'🟠' if flag else '⚪'} **{label}** — {'yes' if flag else 'not yet'}")
    st.caption(report.summary)

st.markdown("#### Recommended experiments")
show_table(
    recs,
    columns=[
        "cycle_number",
        "rank",
        "condition_code",
        "state",
        "predicted_objective",
        "predictive_uncertainty",
        "predicted_retention",
        "rationale_text",
    ],
    empty_message="No recommendations yet for this species.",
)
if not recs.empty:
    with st.expander("Recipe parameters of each recommendation"):
        params = pd.DataFrame([json.loads(p) for p in recs["parameters_json"]])
        params.insert(0, "condition_code", recs["condition_code"].to_numpy())
        show_table(params, key="rec-params")

st.markdown("#### Efficiency vs stability trade-off")
mo = build_multi_objective_dataset(session())
coverage = stability_coverage(mo)
species_rows = mo[mo["species_code"] == species.value]
if (species_rows["stability_status"] == "measured").any():
    left, right = st.columns([1.4, 1])
    left.pyplot(plot_pareto_front(species_rows))
    right.markdown(
        "Each dot is a device. The **blue frontier** shows the best available compromises: "
        "no other device beats them on both efficiency *and* stability at once. Grey "
        "devices are beaten on both."
    )
    right.caption(
        f"Stability measured for {coverage['measured']} devices; {coverage['not_yet_aged']} still ageing."
    )
else:
    empty_state("No stability measurements for this species yet.")
