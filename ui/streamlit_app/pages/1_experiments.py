"""Experimental conditions (Chunk 11.2)."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.reports.ui_queries import conditions_frame

st.title("Experiments")
frame = conditions_frame(session())
if not frame.empty:
    c1, c2, c3 = st.columns(3)
    species = c1.multiselect("Species", sorted(frame["species_code"].unique()))
    states = c2.multiselect("State", sorted(frame["state"].unique()))
    data_kind = c3.radio("Data", ["all", "real", "synthetic"], horizontal=True)
    query = c1.text_input("Condition code contains")
    if species:
        frame = frame[frame["species_code"].isin(species)]
    if states:
        frame = frame[frame["state"].isin(states)]
    if data_kind != "all":
        frame = frame[frame["is_synthetic"] == (data_kind == "synthetic")]
    if query:
        frame = frame[frame["condition_code"].str.contains(query, case=False)]
show_table(frame)
