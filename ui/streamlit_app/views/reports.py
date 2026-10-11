"""Reports & exports: tables ready for a thesis, paper or supplementary material."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from _shared import page_header, readable, session

from dssc_discovery.reports.closed_loop_audit_export import export_audit_trail
from dssc_discovery.reports.ui_queries import (
    conditions_frame,
    hypotheses_frame,
    model_runs_frame,
    recommendations_frame,
)

page_header(
    "Reports & exports",
    "Download the project's data as CSV files for analysis, write-ups or supplementary "
    "material. Synthetic and real data are always exported separately.",
    eyebrow="Records",
)

datasets = [
    (
        "🧪 Experimental conditions",
        "Every recipe with its stage and status.",
        conditions_frame(session()),
    ),
    (
        "🤖 Model runs",
        "Every trained model, its metrics and data version.",
        model_runs_frame(session()),
    ),
    (
        "🎯 Recommendations",
        "Optimizer suggestions with predictions and rationale.",
        recommendations_frame(session()),
    ),
    ("🔍 Hypothesis log", "SHAP findings turned into lab hypotheses.", hypotheses_frame(session())),
]
columns = st.columns(2)
for i, (title, description, frame) in enumerate(datasets):
    with columns[i % 2].container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"{description} · {len(frame)} rows")
        st.download_button(
            "Download CSV",
            readable(frame).to_csv(index=False) if not frame.empty else "",
            file_name=f"{title.split(' ', 1)[1].lower().replace(' ', '_')}.csv",
            disabled=frame.empty,
            key=f"dl-{i}",
            use_container_width=True,
        )

st.markdown("#### Closed-loop audit package")
st.write(
    "A complete, reconstructable record of the closed loop — conditions, every status "
    "change, optimizer iterations, recommendations, model runs and dataset versions."
)
c1, c2 = st.columns([1, 2])
scope = c1.radio(
    "Data",
    ["real_only", "synthetic_only"],
    format_func=lambda s: "Real data" if s == "real_only" else "Synthetic data",
)
if c1.button("Build audit package", type="primary"):
    out_dir = Path(tempfile.mkdtemp(prefix="dssc-audit-"))
    manifest = export_audit_trail(session(), out_dir, scope=scope)  # type: ignore[arg-type]
    with c2:
        st.json(manifest.read_text(), expanded=False)
        for path in sorted(out_dir.glob("*.csv")):
            st.download_button(
                f"⬇ {path.name}", path.read_text(), file_name=path.name, key=f"audit-{path.name}"
            )
