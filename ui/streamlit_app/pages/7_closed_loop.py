"""Closed-loop review and approval (Chunk 11.6). Every action calls the
7.2 state machine with YOUR name and note; there is no bulk or automatic
approval -- each recommendation needs its own explicit click."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.database.repositories import ExperimentalConditionRepository
from dssc_discovery.domain.enums import ClosedLoopState
from dssc_discovery.reports.ui_queries import recommendations_frame

st.title("Closed loop")
operator = st.text_input("Your name (recorded on every action)")
repository = ExperimentalConditionRepository(session())
recs = recommendations_frame(session())

proposed = repository.filter(state=ClosedLoopState.PROPOSED)
st.subheader(f"Awaiting approval ({len(proposed)})")
for condition in sorted(proposed, key=lambda c: c.condition_code):
    with st.expander(
        f"{condition.condition_code}{' 🧪 SYNTHETIC' if condition.is_synthetic else ''}"
    ):
        rec = recs[recs["condition_code"] == condition.condition_code] if not recs.empty else recs
        if not rec.empty:
            st.write(rec.iloc[0]["rationale_text"])
        st.json(condition.target_parameters.model_dump(mode="json"))
        note = st.text_input("Note / reason", key=f"note-{condition.id}")
        c1, c2 = st.columns(2)
        try:
            if c1.button("Approve", key=f"approve-{condition.id}", disabled=not operator):
                sm.approve(session(), condition.id, operator, note)
                session().commit()
                st.rerun()
            if c2.button("Reject", key=f"reject-{condition.id}", disabled=not operator):
                sm.reject(session(), condition.id, operator, note)
                session().commit()
                st.rerun()
        except (ValueError, PermissionError) as exc:
            session().rollback()
            st.error(str(exc))

st.subheader("Lab progress")
ACTIONS = {
    ClosedLoopState.APPROVED: [
        ("Fabricated", sm.mark_fabricated),
        ("Fabrication failed", sm.mark_fabrication_failed),
    ],
    ClosedLoopState.FABRICATED: [("Measurement failed", sm.mark_measurement_failed)],
}
for state, actions in ACTIONS.items():
    for condition in sorted(repository.filter(state=state), key=lambda c: c.condition_code):
        cols = st.columns([3] + [1] * len(actions))
        cols[0].write(f"{condition.condition_code} — {state.value}")
        for col, (label, action) in zip(cols[1:], actions, strict=True):
            if col.button(label, key=f"{label}-{condition.id}", disabled=not operator):
                try:
                    action(session(), condition.id, operator, "set from dashboard")
                    session().commit()
                    st.rerun()
                except ValueError as exc:
                    session().rollback()
                    st.error(str(exc))
st.caption(
    "Measured results are ingested with scripts/closed_loop_cli.py ingest-results, which runs "
    "validation before anything reaches the training data."
)

st.subheader("All recommendations")
show_table(recs)
