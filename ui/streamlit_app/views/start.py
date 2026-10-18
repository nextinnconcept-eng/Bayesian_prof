"""Start here: enter your own lab results (typed or uploaded) so the system can learn from them."""

from __future__ import annotations

import streamlit as st
from _shared import page_header, session
from _user_data_ui import (
    field_hint,
    owner_name,
    readiness_card,
    require_real_mode,
    species_picker,
)

from dssc_discovery.user_data import field_guide as fg
from dssc_discovery.user_data import readiness as rd
from dssc_discovery.user_data.past_results import (
    UnreadableFileError,
    check_rows,
    import_past_results,
    missing_required_summary,
    read_uploaded_table,
)

page_header(
    "Start here",
    "Tell the system what you have already tested. It learns from your numbers, then "
    "tells you which experiments to run in the lab next — before you spend any materials. "
    "Nothing here assumes any background: every box is explained.",
    eyebrow="Get started",
)
require_real_mode()

if flash := st.session_state.pop("flash", None):
    st.success(flash, icon="✅")

with st.expander("How this works (read this first)", expanded=False):
    st.markdown("""
1. **You enter results you already have** — cells you built and tested yourself, or numbers
   from published papers. One result = one finished solar cell and its test numbers.
2. **The system checks them** and tells you in plain words if anything is missing or looks wrong.
3. **Once there is enough data**, it builds a model that predicts how well a recipe will work
   and suggests the next recipes worth testing (see *Plan my experiments*).
4. **You run those experiments in the lab**, come back, and enter the new results. Each round
   makes the suggestions better.

**Words you will see**
- **Recipe**: all the settings used to make one cell (solvent, soaking time, film thickness …).
- **Efficiency (PCE)**: the % of light energy the cell turns into electricity. Higher is better.
- **Uncertainty**: how unsure the prediction is. Large uncertainty means "test this to find out".
- **Early estimate**: a rough, low-confidence prediction made from only a few experiments.
        """)

# --- step 1: who and what ---------------------------------------------------------------

st.markdown("### 1 · About you and your dye")
left, right = st.columns(2)
with left:
    st.session_state.setdefault("owner_name", st.session_state.get("user", ""))
    st.text_input(
        "Your name",
        key="owner_name",
        help="Saved next to everything you enter, so there is always a record of who did what.",
    )
with right:
    species = species_picker("start_species")

info = rd.assess_readiness(session(), species)
readiness_card(info)

# --- step 2: what do you have -----------------------------------------------------------

st.markdown("### 2 · What results do you already have?")
source = st.radio(
    "Pick the closest",
    [
        "From my own earlier lab runs",
        "From published papers (literature)",
        "Nothing yet",
    ],
    label_visibility="collapsed",
)
if source == "Nothing yet":
    st.info(
        "That is fine. Open **Plan my experiments** in the left menu: it creates a set of "
        "recipes spread across the whole range, for you to run in the lab first. Come back "
        "here afterwards to enter what you measured.",
        icon="🧭",
    )
    st.stop()
if source.startswith("From published"):
    st.caption(
        "Only enter paper results made with a similar kind of cell and test (same type of "
        "electrolyte and electrode, standard sunlight). Very different set-ups can confuse "
        "the model more than they help. Each row is saved as 'published paper'."
    )
instrument_default = "published paper" if source.startswith("From published") else ""

# --- step 3: enter them ------------------------------------------------------------------

st.markdown("### 3 · Enter your results")
form_tab, upload_tab = st.tabs(["✍️ Type one result", "📄 Upload a file (many results)"])


def _save(rows: list[dict[str, str]]) -> None:
    if instrument_default:
        for row in rows:
            if not (row.get("instrument") or "").strip():
                row["instrument"] = instrument_default
    name = owner_name()
    check = check_rows(rows, name, session())
    if not check.ok:
        st.error("Nothing was saved. Please fix the following and try again:", icon="🛑")
        for problem in check.problems:
            st.markdown(f"- **Row {problem.row_number}:** {problem.message}")
        return
    for warning in check.warnings:
        st.warning(f"Row {warning.row_number}: {warning.message}", icon="⚠️")
    result = import_past_results(session(), rows, name)
    if not result.saved:
        session().rollback()
        st.error("Nothing was saved:", icon="🛑")
        for problem in result.problems:
            st.markdown(f"- **Row {problem.row_number}:** {problem.message}")
        for code, reason in result.blocked.items():
            st.markdown(f"- **{code}:** {reason}")
        return
    st.session_state["flash"] = (
        f"Saved {len(rows)} result(s) across {len(result.conditions_ingested)} experiment(s). "
        "They are stored as real data."
    )
    st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
    st.rerun()


with form_tab:
    version = st.session_state.get("form_version", 0)
    with st.form(f"one_result_{version}"):
        st.caption(
            "Fill in every box. Each one is explained underneath. If you built several "
            "identical cells, save one now and add the others later with the **same "
            "experiment name** (under *Optional details*)."
        )
        values: dict[str, str] = {"species_code": species.value}
        current_group = ""
        for spec in fg.ALL_FIELDS:
            if not spec.required or spec.name == "species_code":
                continue
            if spec.group != current_group:
                current_group = spec.group
                st.markdown(f"##### {current_group}")
            label = spec.label + (f"  ·  {spec.unit}" if spec.unit else "")
            key = f"f_{version}_{spec.name}"
            if spec.kind == "choice":
                picked = st.selectbox(
                    label,
                    [v for v, _ in spec.choices],
                    index=None,
                    format_func=lambda v, s=spec: fg.plain_value(s, str(v)),
                    placeholder="Choose…",
                    key=key,
                )
                values[spec.name] = picked or ""
            elif spec.kind == "date":
                chosen = st.date_input(label, value=None, format="YYYY-MM-DD", key=key)
                values[spec.name] = chosen.isoformat() if chosen else ""
            else:
                values[spec.name] = st.text_input(
                    label, placeholder=f"for example {spec.example}", key=key
                )
            st.caption(field_hint(spec.name, species))
        with st.expander(
            "Optional details (experiment name, repeat number, instrument, efficiency)"
        ):
            for name in (
                "condition_code",
                "device_code",
                "replicate_number",
                "instrument",
                "power_conversion_efficiency_percent",
            ):
                opt = fg.FIELDS_BY_NAME[name]
                values[name] = st.text_input(
                    opt.label + (f"  ·  {opt.unit}" if opt.unit else ""),
                    placeholder=f"for example {opt.example}",
                    key=f"f_{version}_{name}",
                )
                st.caption(field_hint(name, species))
        submitted = st.form_submit_button("Save this result", type="primary")
    if submitted:
        _save([values])

with upload_tab:
    st.markdown(
        "**1.** Download the template and the instructions. **2.** Fill in one row per "
        "finished cell (the template has a worked example row — replace it). "
        "**3.** Upload it here; it is checked before anything is saved."
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Template (CSV, opens in Excel)",
        fg.template_csv(),
        file_name="my_results_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ Instructions for every column",
        fg.instructions_text(),
        file_name="how_to_fill_in_the_template.txt",
        mime="text/plain",
        use_container_width=True,
    )
    uploaded = st.file_uploader("Your filled-in file (.csv or .xlsx)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            rows = read_uploaded_table(uploaded.getvalue(), uploaded.name)
        except UnreadableFileError as exc:
            st.error(str(exc), icon="🛑")
            rows = []
        if uploaded is not None and not rows:
            st.warning("The file has no rows with data in it.")
        elif rows:
            absent = missing_required_summary(rows)
            if absent:
                st.error(
                    "Your file is missing these columns entirely: " + ", ".join(absent) + ". "
                    "Download the template to see every column.",
                    icon="🛑",
                )
            else:
                st.caption(f"Found {len(rows)} row(s). Preview:")
                st.dataframe(rows, use_container_width=True, hide_index=True)
                check = check_rows(rows, owner_name(), session())
                if check.missing_by_field:
                    st.markdown("**Still missing:**")
                    for field_name, row_numbers in check.missing_by_field.items():
                        known = fg.FIELDS_BY_NAME.get(field_name)
                        label = known.label if known else field_name
                        st.markdown(f"- {label}: row(s) {', '.join(map(str, row_numbers))}")
                for problem in check.problems:
                    st.error(f"Row {problem.row_number}: {problem.message}")
                for warning in check.warnings:
                    st.warning(f"Row {warning.row_number}: {warning.message}", icon="⚠️")
                if check.ok:
                    st.success(f"All {len(rows)} row(s) look good.", icon="✅")
                if st.button(
                    f"Save {len(rows)} result(s)",
                    type="primary",
                    disabled=not check.ok,
                    key="save_upload",
                ):
                    _save(rows)

st.markdown("### 4 · Next")
st.write(
    "Once enough results are saved, open **Plan my experiments** in the left menu to get "
    "your predictions and the recipes to try next."
)
