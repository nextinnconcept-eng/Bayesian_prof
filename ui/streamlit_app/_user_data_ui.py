"""Small UI helpers shared by the 'Start here' and 'Plan my experiments' pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import DEMO_MODE, data_mode, empty_state

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.user_data import readiness as rd
from dssc_discovery.user_data.field_guide import FIELDS_BY_NAME, plain_value, typical_range

REAL_ONLY_MESSAGE = (
    "You are looking at the **simulated demo** (sidebar → Data). This page only works with "
    "**My real data**, so demo numbers can never be mixed with your lab results. "
    "Switch the sidebar to *My real data* to continue."
)


def require_real_mode() -> None:
    if data_mode() == DEMO_MODE:
        st.info(REAL_ONLY_MESSAGE, icon="🧪")
        st.stop()


def owner_name() -> str:
    """Who is recorded in the audit history for what the owner saves."""
    typed = st.session_state.get("owner_name")
    return (typed or st.session_state.get("user") or "owner").strip() or "owner"


def species_picker(key: str) -> SpeciesCode:
    chosen = st.session_state.get("chosen_species", SpeciesCode.BETA_VULGARIS)
    options = list(SpeciesCode)
    picked = st.selectbox(
        "Which plant is your dye made from?",
        options,
        index=options.index(chosen),
        format_func=lambda s: s.value,
        key=key,
        help="Each plant is modelled separately, because different plants behave differently.",
    )
    st.session_state["chosen_species"] = picked
    return picked


def readiness_card(info: rd.Readiness) -> None:
    with st.container(border=True):
        st.markdown("#### Where you are")
        target = max(info.full_min, 1)
        st.progress(
            min(info.n_experiments / target, 1.0),
            text=f"{info.n_experiments} real experiment(s) saved · "
            f"first estimate at {info.early_min} · trusted results at {info.full_min}",
        )
        if info.has_synthetic_rows:
            st.error(info.headline, icon="🧪")
        elif info.level == rd.FULL:
            st.success(info.headline, icon="✅")
        elif info.level == rd.EARLY:
            st.warning(info.headline, icon="⚠️")
        else:
            st.info(info.headline, icon="ℹ️")
        if info.unchanged_settings:
            st.caption(
                "Tip: you used the same value for "
                + ", ".join(f"**{s}**" for s in info.unchanged_settings)
                + " in every experiment so far, so the model cannot yet learn how it matters. "
                "Try varying it next."
            )


def field_hint(name: str, species: SpeciesCode) -> str:
    spec = FIELDS_BY_NAME[name]
    parts = [spec.help]
    window = typical_range(species, name)
    if window is not None:
        parts.append(f"Range the optimizer explores: {window[0]:g} to {window[1]:g}.")
    if not spec.required and spec.optional_note:
        parts.append(spec.optional_note)
    return " ".join(parts)


def experiments_frame(experiments, with_prediction: bool) -> pd.DataFrame:
    """Planned experiments as a table of plain-language columns."""
    rows = []
    for e in experiments:
        row: dict[str, object] = {"Experiment": e.condition_code}
        if with_prediction:
            row["Expected efficiency (%)"] = e.predicted_pce
            row["Uncertainty (± %)"] = e.uncertainty
        for name, value in e.parameters.items():
            spec = FIELDS_BY_NAME.get(name)
            if spec is None:
                continue
            label = spec.label + (f" ({spec.unit})" if spec.unit and spec.kind == "number" else "")
            row[label] = (
                plain_value(spec, str(value)) if spec.kind == "choice" else round(float(value), 3)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def show_experiments(experiments, with_prediction: bool = False) -> None:
    if not experiments:
        empty_state("Nothing planned yet.")
        return
    st.dataframe(
        experiments_frame(experiments, with_prediction),
        use_container_width=True,
        hide_index=True,
    )
