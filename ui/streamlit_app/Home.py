"""DSSC discovery research dashboard (Chunk 11.1).

streamlit run ui/streamlit_app/Home.py
"""

from __future__ import annotations

import streamlit as st
from _shared import session

from dssc_discovery.reports.ui_queries import conditions_frame, model_runs_frame

st.set_page_config(page_title="DSSC discovery", layout="wide")
st.title("Natural-pigment DSSC discovery")
st.caption(
    "Closed-loop, ML-assisted discovery platform. Every page reads the project database; "
    "synthetic rows are always badged."
)

conditions = conditions_frame(session())
runs = model_runs_frame(session())
cols = st.columns(4)
cols[0].metric("Conditions", len(conditions))
cols[1].metric(
    "Real conditions", int((~conditions["is_synthetic"]).sum()) if not conditions.empty else 0
)
cols[2].metric(
    "Awaiting approval",
    int((conditions["state"] == "proposed").sum()) if not conditions.empty else 0,
)
cols[3].metric("Model runs", len(runs))

st.markdown("""
**Pages**
- **Experiments / Library / Measurements**: browse records and their provenance.
- **ML evaluation**: model comparison under validation strategies A/B/C, always with baselines.
- **Explainability**: SHAP plots (associational, not causal) and the hypothesis log.
- **Bayesian optimization / Closed loop**: recommendations, convergence, and human approval.
- **Reports / History**: exports for manuscripts and the full audit trail.
""")
