"""Audit history: who changed what, and when."""

from __future__ import annotations

import streamlit as st
from _shared import page_header, session, show_table

from dssc_discovery.reports.ui_queries import (
    audit_log_frame,
    dataset_versions_frame,
    state_log_frame,
)

page_header(
    "Audit history",
    "A permanent, append-only record of every decision and data change — what makes the "
    "results reproducible and the human-approval rule verifiable.",
    eyebrow="Records",
)

log = state_log_frame(session())
versions = dataset_versions_frame(session())
audit = audit_log_frame(session())
k1, k2, k3 = st.columns(3)
k1.metric("Status changes", len(log))
k2.metric("Dataset versions", len(versions))
k3.metric("Database changes (latest)", len(audit))

tab1, tab2, tab3 = st.tabs(["Experiment status changes", "Dataset versions", "Database change log"])
with tab1:
    show_table(
        log.iloc[::-1] if not log.empty else log,
        columns=["created_at", "condition_code", "from_state", "to_state", "operator", "reason"],
        empty_message="No status changes yet.",
    )
with tab2:
    show_table(
        versions,
        columns=["data", "created_at", "row_count", "dataset_hash", "code_version", "operator"],
        empty_message="No dataset versions yet.",
    )
with tab3:
    if audit.empty:
        st.info("No changes recorded yet.")
    else:
        st.dataframe(audit, use_container_width=True, hide_index=True)
