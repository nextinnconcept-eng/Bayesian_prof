"""System history (Chunk 11.7): state transitions, dataset versions and
the low-level audit log."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.reports.ui_queries import (
    audit_log_frame,
    dataset_versions_frame,
    state_log_frame,
)

st.title("History")
st.subheader("Closed-loop state transitions")
show_table(state_log_frame(session()))
st.subheader("Dataset versions")
show_table(dataset_versions_frame(session()))
st.subheader("Database audit log (latest 2000 changes)")
frame = audit_log_frame(session())
if frame.empty:
    st.info("No changes recorded yet.")
else:
    st.dataframe(frame, use_container_width=True, hide_index=True)
