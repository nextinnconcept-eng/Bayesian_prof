"""Experiments: every recipe (condition), where it came from, its status and result."""

from __future__ import annotations

import streamlit as st
from _shared import STAGE_LABELS, STATE_LABELS, page_header, session, show_table

from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.reports.ui_queries import conditions_frame

page_header(
    "Experiments",
    "Each row is one recipe (an <b>experimental condition</b>): how the pigment was "
    "extracted, how the cell was built, which stage proposed it, and the efficiency its "
    "devices reached.",
    eyebrow="Lab data",
)

frame = conditions_frame(session())
data = build_ml_dataset(session())
if not frame.empty and not data.empty:
    pce = data.groupby("condition_code")["power_conversion_efficiency_percent"]
    frame = frame.merge(
        pce.agg(n_devices="size", mean_pce="mean", best_pce="max").reset_index(),
        on="condition_code",
        how="left",
    )

if not frame.empty:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1])
        species = c1.multiselect("Species", sorted(frame["species_code"].unique()))
        stages = c2.multiselect(
            "Stage",
            sorted(frame["design_source"].unique()),
            format_func=lambda s: STAGE_LABELS.get(s, s),
        )
        states = c3.multiselect(
            "Status", sorted(frame["state"].unique()), format_func=lambda s: STATE_LABELS.get(s, s)
        )
        query = c4.text_input("Search code", placeholder="e.g. DOE-012")
    if species:
        frame = frame[frame["species_code"].isin(species)]
    if stages:
        frame = frame[frame["design_source"].isin(stages)]
    if states:
        frame = frame[frame["state"].isin(states)]
    if query:
        frame = frame[frame["condition_code"].str.contains(query, case=False)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Conditions shown", len(frame))
    measured = frame["mean_pce"].notna() if "mean_pce" in frame.columns else frame["state"] == ""
    m2.metric("With measured devices", int(measured.sum()))
    m3.metric(
        "Best PCE in selection",
        (
            f"{frame['best_pce'].max():.2f} %"
            if "best_pce" in frame.columns and measured.any()
            else "—"
        ),
    )

show_table(
    frame,
    columns=[
        "data",
        "condition_code",
        "species_code",
        "design_source",
        "state",
        "n_devices",
        "mean_pce",
        "best_pce",
        "solvent",
        "ph",
        "pigment_concentration_mm",
        "adsorption_time_min",
        "tio2_thickness_um",
        "sintering_temperature_c",
        "surface_treatment",
    ],
    empty_message="No experiments recorded yet. Generate an initial design with "
    "`scripts/generate_initial_doe.py`.",
)
