"""Raw measurements and derived descriptors with provenance (Chunk 11.3)."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.reports.ui_queries import conditions_frame, measurement_frames

st.title("Measurements & descriptors")
frames = measurement_frames(session())
devices = frames["devices"]
if devices.empty:
    st.info("No devices recorded yet.")
    st.stop()

conditions = conditions_frame(session()).set_index("id")
device_code = st.selectbox("Device", sorted(devices["device_code"]))
device = devices[devices["device_code"] == device_code].iloc[0]
condition = conditions.loc[device["experimental_condition_id"]]
st.markdown(
    f"**Provenance:** condition `{condition['condition_code']}` ({condition['species_code']}, "
    f"{condition['design_source']}, state {condition['state']}) → device `{device_code}` "
    f"(replicate {device['replicate_number']})"
)


def for_device(name: str):  # type: ignore[no-untyped-def]
    frame = frames[name]
    return frame[frame["device_id"] == device["id"]] if not frame.empty else frame


st.subheader("Photovoltaic (J-V)")
show_table(for_device("photovoltaic"))
st.subheader("Electrochemical impedance")
show_table(for_device("electrochemical"))
st.subheader("Stability")
show_table(for_device("stability"))

st.subheader("Raw spectra and derived descriptors")
spectra = frames["raw_spectra"]
show_table(spectra)
if not spectra.empty and not frames["derived_descriptors"].empty:
    spectrum = st.selectbox("Spectrum", spectra["id"])
    descriptors = frames["derived_descriptors"]
    show_table(descriptors[descriptors["source_spectrum_id"] == spectrum])
