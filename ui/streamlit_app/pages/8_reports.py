"""Reports (Chunk 11.7): manuscript-ready tables as downloads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from _shared import session

from dssc_discovery.reports.closed_loop_audit_export import export_audit_trail
from dssc_discovery.reports.ui_queries import (
    conditions_frame,
    hypotheses_frame,
    model_runs_frame,
    recommendations_frame,
)

st.title("Reports")
for title, frame in (
    ("Experimental conditions", conditions_frame(session())),
    ("Model runs", model_runs_frame(session())),
    ("Recommendations", recommendations_frame(session())),
    ("Hypothesis log", hypotheses_frame(session())),
):
    st.download_button(
        f"Download {title.lower()} (CSV)",
        frame.to_csv(index=False),
        file_name=f"{title.lower().replace(' ', '_')}.csv",
        disabled=frame.empty,
    )

st.subheader("Closed-loop audit trail")
scope = st.radio("Scope", ["real_only", "synthetic_only"], horizontal=True)
if st.button("Build audit export"):
    out_dir = Path(tempfile.mkdtemp(prefix="dssc-audit-"))
    manifest = export_audit_trail(session(), out_dir, scope=scope)  # type: ignore[arg-type]
    st.json(manifest.read_text())
    for path in sorted(out_dir.glob("*.csv")):
        st.download_button(f"Download {path.name}", path.read_text(), file_name=path.name)
