"""Demo sign-in screen."""

from __future__ import annotations

import streamlit as st
from _shared import BLUE, INK_2, check_credentials

_, middle, _ = st.columns([1, 1.2, 1])
with middle:
    st.markdown("<div style='height: 8vh'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center; font-size:3rem'>☀️</div>"
        f"<h1 style='text-align:center; margin-bottom:0'>DSSC Discovery</h1>"
        f"<p style='text-align:center; color:{INK_2}; margin-top:0.3rem'>"
        "Closed-loop, ML-assisted discovery of natural plant pigments for dye-sensitized "
        "solar cells</p>",
        unsafe_allow_html=True,
    )
    with st.form("login", border=True):
        st.markdown(f"<h4 style='color:{BLUE}; margin-top:0'>Sign in</h4>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        if check_credentials(username, password):
            st.session_state["user"] = username.strip()
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.caption("Demo project — use **admin / admin**.")
