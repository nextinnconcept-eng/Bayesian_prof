"""Plant and pigment library (Chunk 11.2)."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.reports.ui_queries import library_frames

st.title("Plant & pigment library")
frames = library_frames(session())
for title, key in (
    ("Species", "species"),
    ("Plant samples", "plant_samples"),
    ("Pigment extract batches", "pigment_extract_batches"),
):
    st.subheader(title)
    show_table(frames[key])
