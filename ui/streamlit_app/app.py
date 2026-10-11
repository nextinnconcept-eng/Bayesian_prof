"""The dashboard application: page setup, demo login gate and navigation.

Both entry points call main():
    streamlit run streamlit_app.py              (repository root; Streamlit Cloud default)
    streamlit run ui/streamlit_app/Home.py

Page paths are absolute, so the app works whichever file Streamlit started from.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from _shared import apply_style, ensure_database, is_logged_in

HERE = Path(__file__).resolve().parent
VIEWS = HERE / "views"


def main() -> None:
    st.set_page_config(
        page_title="DSSC Discovery",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_style()
    ensure_database()

    if not is_logged_in():
        st.navigation(
            [st.Page(VIEWS / "login.py", title="Sign in", icon="🔐")], position="hidden"
        ).run()
        st.stop()

    pages = {
        "Overview": [
            st.Page(VIEWS / "overview.py", title="Dashboard", icon="🏠", default=True),
        ],
        "Lab data": [
            st.Page(VIEWS / "experiments.py", title="Experiments", icon="🧪"),
            st.Page(VIEWS / "library.py", title="Plant & pigment library", icon="🌿"),
            st.Page(VIEWS / "measurements.py", title="Measurements", icon="📈"),
        ],
        "Machine learning": [
            st.Page(VIEWS / "ml_evaluation.py", title="Model evaluation", icon="🤖"),
            st.Page(VIEWS / "explainability.py", title="Explainability", icon="🔍"),
        ],
        "Optimization": [
            st.Page(VIEWS / "bayesian_optimization.py", title="Bayesian optimization", icon="🎯"),
            st.Page(VIEWS / "closed_loop.py", title="Approvals & lab loop", icon="🔁"),
        ],
        "Records": [
            st.Page(VIEWS / "reports.py", title="Reports & exports", icon="📄"),
            st.Page(VIEWS / "history.py", title="Audit history", icon="🕓"),
        ],
    }

    st.logo(str(HERE / "assets" / "logo.svg"), size="large")
    navigation = st.navigation(pages)
    with st.sidebar:
        st.divider()
        st.caption(f"Signed in as **{st.session_state['user']}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()
    navigation.run()
