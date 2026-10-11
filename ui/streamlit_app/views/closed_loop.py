"""Approvals & lab loop: the human gate between a recommendation and the lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import (
    COLUMN_LABELS,
    STATE_LABELS,
    cards,
    empty_state,
    page_header,
    session,
    show_table,
)

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.database.repositories import ExperimentalConditionRepository
from dssc_discovery.domain.enums import ClosedLoopState
from dssc_discovery.reports.ui_queries import recommendations_frame

page_header(
    "Approvals & lab loop",
    "Nothing is made in the lab until a researcher approves it. Review each proposed "
    "recipe, approve or reject it with a note, then record fabrication progress. Every "
    "action is stored in the audit history under your name.",
    eyebrow="Optimization",
)
cards(
    [
        ("Proposed", "Suggested by the initial design or the optimizer; waiting for review."),
        ("Approved → Fabricated", "Accepted by a researcher, then built in the lab."),
        ("Measured → Ingested", "Results validated and added to the training data."),
        ("Model updated", "The optimizer has retrained on the new results."),
    ],
    numbered=True,
)
st.write("")

operator = st.session_state.get("user", "admin")
repository = ExperimentalConditionRepository(session())
recs = recommendations_frame(session())
proposed = sorted(repository.filter(state=ClosedLoopState.PROPOSED), key=lambda c: c.condition_code)

st.markdown(f"#### Awaiting your decision ({len(proposed)})")
if not proposed:
    empty_state("Nothing is waiting for approval.")
for condition in proposed:
    rec = recs[recs["condition_code"] == condition.condition_code] if not recs.empty else recs
    label = f"{condition.condition_code} · {condition.species_code.value}"
    if condition.is_synthetic:
        label += " · 🧪 synthetic"
    with st.container(border=True):
        st.markdown(f"**{label}**")
        if not rec.empty:
            r = rec.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted PCE", f"{r['predicted_objective']:.2f} %")
            c2.metric("Uncertainty", f"± {r['predictive_uncertainty']:.2f} %")
            c3.metric("BO cycle", int(r["cycle_number"]))
            st.caption(r["rationale_text"])
        params = condition.target_parameters.model_dump(mode="json")
        with st.expander("Recipe parameters"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Parameter": [COLUMN_LABELS.get(k, k) for k in params],
                        "Value": [str(v) for v in params.values()],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        note = st.text_input(
            "Note (required)", key=f"note-{condition.id}", placeholder="Why approve or reject?"
        )
        b1, b2, _ = st.columns([1, 1, 4])
        try:
            if b1.button("✅ Approve", key=f"approve-{condition.id}", type="primary"):
                sm.approve(session(), condition.id, operator, note)
                session().commit()
                st.toast(f"{condition.condition_code} approved")
                st.rerun()
            if b2.button("✖ Reject", key=f"reject-{condition.id}"):
                sm.reject(session(), condition.id, operator, note)
                session().commit()
                st.rerun()
        except (ValueError, PermissionError) as exc:
            session().rollback()
            st.error(str(exc))

st.markdown("#### In the lab")
in_lab = {
    ClosedLoopState.APPROVED: [
        ("Mark fabricated", sm.mark_fabricated),
        ("Fabrication failed", sm.mark_fabrication_failed),
    ],
    ClosedLoopState.FABRICATED: [("Measurement failed", sm.mark_measurement_failed)],
}
shown = 0
for state, actions in in_lab.items():
    for condition in sorted(repository.filter(state=state), key=lambda c: c.condition_code):
        shown += 1
        cols = st.columns([3, 1.5] + [1.3] * len(actions))
        cols[0].write(f"**{condition.condition_code}**")
        cols[1].write(STATE_LABELS[state.value])
        for col, (label, action) in zip(cols[2:], actions, strict=True):
            if col.button(label, key=f"{label}-{condition.id}"):
                try:
                    action(session(), condition.id, operator, "set from dashboard")
                    session().commit()
                    st.rerun()
                except ValueError as exc:
                    session().rollback()
                    st.error(str(exc))
if not shown:
    empty_state("No approved experiments are in progress.")
st.caption(
    "Measured results are added with `scripts/closed_loop_cli.py ingest-results`, which validates them first."
)

st.markdown("#### All recommendations")
show_table(
    recs,
    columns=[
        "cycle_number",
        "rank",
        "condition_code",
        "species_code",
        "state",
        "predicted_objective",
        "predictive_uncertainty",
        "rationale_text",
    ],
    empty_message="No recommendations yet.",
)
