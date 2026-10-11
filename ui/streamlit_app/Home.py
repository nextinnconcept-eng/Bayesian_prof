"""DSSC discovery research dashboard.

    streamlit run ui/streamlit_app/Home.py      (from the repository root)

Everything sits behind a demo login (see _shared.DEMO_USERNAME /
DEMO_PASSWORD). Pages live in views/ and are wired up with st.navigation.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from _shared import apply_style, is_logged_in

st.set_page_config(
    page_title="DSSC Discovery",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_style()

login = st.Page("views/login.py", title="Sign in", icon="🔐")

if not is_logged_in():
    st.navigation([login], position="hidden").run()
    st.stop()

pages = {
    "Overview": [
        st.Page("views/overview.py", title="Dashboard", icon="🏠", default=True),
    ],
    "Lab data": [
        st.Page("views/experiments.py", title="Experiments", icon="🧪"),
        st.Page("views/library.py", title="Plant & pigment library", icon="🌿"),
        st.Page("views/measurements.py", title="Measurements", icon="📈"),
    ],
    "Machine learning": [
        st.Page("views/ml_evaluation.py", title="Model evaluation", icon="🤖"),
        st.Page("views/explainability.py", title="Explainability", icon="🔍"),
    ],
    "Optimization": [
        st.Page("views/bayesian_optimization.py", title="Bayesian optimization", icon="🎯"),
        st.Page("views/closed_loop.py", title="Approvals & lab loop", icon="🔁"),
    ],
    "Records": [
        st.Page("views/reports.py", title="Reports & exports", icon="📄"),
        st.Page("views/history.py", title="Audit history", icon="🕓"),
    ],
}

st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

navigation = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption(f"Signed in as **{st.session_state['user']}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.pop("user", None)
        st.rerun()

navigation.run()
