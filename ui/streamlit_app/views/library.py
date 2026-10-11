"""Plant & pigment library: species, collected plant samples and extract batches."""

from __future__ import annotations

import streamlit as st
from _shared import card, page_header, session, show_table

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.reports.ui_queries import conditions_frame, library_frames

page_header(
    "Plant & pigment library",
    "The four plant species studied, the samples collected from them, and the pigment "
    "extracts prepared for sensitizing the solar cells.",
    eyebrow="Lab data",
)

frames = library_frames(session())
conditions = conditions_frame(session())
counts = conditions["species_code"].value_counts() if not conditions.empty else {}

columns = st.columns(len(SpeciesCode))
for column, species in zip(columns, SpeciesCode, strict=True):
    n = int(counts.get(species.value, 0)) if len(counts) else 0
    column.markdown(
        card(f"🌿 <i>{species.value}</i>", f"{n} experimental condition{'s' if n != 1 else ''}"),
        unsafe_allow_html=True,
    )
st.write("")

tab1, tab2, tab3 = st.tabs(["Species records", "Plant samples", "Pigment extract batches"])
with tab1:
    show_table(frames["species"], empty_message="No species registered yet.")
with tab2:
    show_table(frames["plant_samples"], empty_message="No plant samples recorded yet.")
with tab3:
    show_table(frames["pigment_extract_batches"], empty_message="No pigment extracts recorded yet.")
