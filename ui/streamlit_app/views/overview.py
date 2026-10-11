"""Dashboard: the state of the research programme at a glance."""

from __future__ import annotations

import streamlit as st
from _shared import (
    BLUE,
    STAGE_LABELS,
    STATE_LABELS,
    cards,
    empty_state,
    page_header,
    session,
    show_table,
    stage_chart,
)

from dssc_discovery.closed_loop.demo_data import build_demo_database
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.reports.ui_queries import (
    conditions_frame,
    model_runs_frame,
    state_log_frame,
)

page_header(
    "Research dashboard",
    "Where the pigment-discovery programme stands: experiments run, the best "
    "efficiency found so far, and what is waiting for a researcher's decision.",
    eyebrow="Overview",
)

conditions = conditions_frame(session())
if conditions.empty:
    with st.container(border=True):
        st.markdown("#### 👋 The database is empty")
        st.write(
            "Load a **simulated** demo dataset to explore every page: a 3-cycle closed "
            "loop on *Beta vulgaris*, stability tests, recommendations awaiting approval "
            "and a logged hypothesis. All rows are clearly marked as synthetic."
        )
        if st.button("Load demo data (about 1–2 minutes)", type="primary"):
            with st.spinner(
                "Simulating experiments, training models and running Bayesian optimization…"
            ):
                n_devices = build_demo_database(session())
            st.success(f"Demo data loaded: {n_devices} simulated devices.")
            st.rerun()

data = build_ml_dataset(session())
runs = model_runs_frame(session())

if not conditions.empty and conditions["is_synthetic"].any():
    st.warning(
        "This database contains **synthetic (simulated) data** — numbers demonstrate the "
        "software, not real solar-cell results.",
        icon="🧪",
    )

pending = int((conditions["state"] == "proposed").sum()) if not conditions.empty else 0
best = float(data["power_conversion_efficiency_percent"].max()) if not data.empty else None
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Experimental conditions", len(conditions))
k2.metric("Devices measured", len(data))
k3.metric("Best efficiency (PCE)", f"{best:.2f} %" if best is not None else "—")
k4.metric("Awaiting approval", pending)
k5.metric("Model runs logged", len(runs))

st.markdown("#### How the platform works")
cards(
    [
        (
            "Design",
            "A space-filling initial design proposes the first ~40 recipes per plant species.",
        ),
        ("Approve & make", "A researcher approves each proposal before any cell is fabricated."),
        (
            "Measure & learn",
            "Validated results train ML models; SHAP explains what drives efficiency.",
        ),
        ("Recommend", "Bayesian optimization suggests the next most informative recipes."),
    ],
    numbered=True,
)
st.write("")

left, right = st.columns([1.3, 1])
with left:
    st.markdown("#### Best efficiency found so far")
    if data.empty:
        empty_state("No measured devices yet.")
    else:
        order = [s for s in STAGE_LABELS if s in set(data["design_source"])]
        best_by_stage = (
            data.groupby("design_source")["power_conversion_efficiency_percent"]
            .max()
            .reindex(order)
            .cummax()
        )
        stage_chart(order, best_by_stage.to_numpy(), "Best PCE (%)")
        st.caption("Best device efficiency after each stage of the programme (cumulative).")
with right:
    st.markdown("#### Experiments by status")
    if conditions.empty:
        empty_state("No experiments yet.")
    else:
        counts = conditions["state"].map(lambda s: STATE_LABELS.get(s, s)).value_counts()
        st.bar_chart(
            counts.rename_axis("Status").reset_index(name="Experiments"),
            x="Status",
            y="Experiments",
            color=BLUE,
            horizontal=True,
            height=280,
        )

st.markdown("#### Recent activity")
log = state_log_frame(session())
show_table(
    log.tail(8).iloc[::-1] if not log.empty else log,
    columns=["created_at", "condition_code", "from_state", "to_state", "operator", "reason"],
    empty_message="No activity recorded yet.",
)
