"""Measurements: one device's full record, from recipe to every measurement."""

from __future__ import annotations

import streamlit as st
from _shared import STAGE_LABELS, STATE_LABELS, empty_state, page_header, session, show_table

from dssc_discovery.reports.ui_queries import conditions_frame, measurement_frames

page_header(
    "Measurements",
    "Pick a solar-cell device to see its complete provenance: the recipe it was built "
    "from, its current–voltage (J–V) performance, impedance, stability, and spectra.",
    eyebrow="Lab data",
)

frames = measurement_frames(session())
devices = frames["devices"]
if devices.empty:
    empty_state(
        "No devices recorded yet.",
        "Results arrive through `scripts/closed_loop_cli.py ingest-results`.",
    )
    st.stop()

conditions = conditions_frame(session()).set_index("id")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Devices", len(devices))
k2.metric("J–V measurements", len(frames["photovoltaic"]))
k3.metric("Stability tests", len(frames["stability"]))
k4.metric("Spectra", len(frames["raw_spectra"]))

device_code = st.selectbox("Device", sorted(devices["device_code"]))
device = devices[devices["device_code"] == device_code].iloc[0]
condition = conditions.loc[device["experimental_condition_id"]]

with st.container(border=True):
    st.markdown(
        f"**Provenance** &nbsp; 🌿 *{condition['species_code']}* &nbsp;→&nbsp; "
        f"recipe **{condition['condition_code']}** ({STAGE_LABELS.get(condition['design_source'])}, "
        f"{STATE_LABELS.get(condition['state'])}) &nbsp;→&nbsp; device **{device_code}** "
        f"(replicate {device['replicate_number']}, {device['active_area_cm2']} cm²)"
    )


def for_device(name: str):  # type: ignore[no-untyped-def]
    frame = frames[name]
    return frame[frame["device_id"] == device["id"]] if not frame.empty else frame


pv = for_device("photovoltaic")
if not pv.empty:
    first = pv.sort_values("measurement_date").iloc[0]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Efficiency (PCE)", f"{first['power_conversion_efficiency_percent']:.2f} %")
    p2.metric("Open-circuit voltage", f"{first['open_circuit_voltage_v']:.3f} V")
    p3.metric(
        "Short-circuit current", f"{first['short_circuit_current_density_ma_cm2']:.2f} mA/cm²"
    )
    p4.metric("Fill factor", f"{first['fill_factor']:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["J–V performance", "Impedance (EIS)", "Stability", "Spectra & descriptors"]
)
with tab1:
    show_table(pv, empty_message="No J–V measurements for this device.")
with tab2:
    show_table(
        for_device("electrochemical"), empty_message="No impedance measurements for this device."
    )
with tab3:
    show_table(for_device("stability"), empty_message="This device has not been aged yet.")
with tab4:
    spectra = frames["raw_spectra"]
    show_table(spectra, empty_message="No spectra recorded yet.")
    if not spectra.empty and not frames["derived_descriptors"].empty:
        spectrum = st.selectbox("Spectrum", spectra["id"])
        descriptors = frames["derived_descriptors"]
        show_table(descriptors[descriptors["source_spectrum_id"] == spectrum])
