"""Plan my experiments: what to run in the lab next, and what a recipe is expected to give."""

from __future__ import annotations

import streamlit as st
from _shared import page_header, session
from _user_data_ui import (
    field_hint,
    owner_name,
    readiness_card,
    require_real_mode,
    show_experiments,
    species_picker,
)
from pydantic import ValidationError

from dssc_discovery.closed_loop.real_workflow import InsufficientRealDataError
from dssc_discovery.config import get_config
from dssc_discovery.user_data import field_guide as fg
from dssc_discovery.user_data import planner
from dssc_discovery.user_data import readiness as rd

page_header(
    "Plan my experiments",
    "Get the recipes worth running in the lab — and see what the model expects from any "
    "recipe you are thinking of trying — before you use any materials.",
    eyebrow="Get started",
)
require_real_mode()

species = species_picker("plan_species")
info = rd.assess_readiness(session(), species)
readiness_card(info)
name = owner_name()

# --- recipes already waiting ------------------------------------------------------------

waiting = planner.pending_plan(session(), species)
if waiting:
    st.markdown(f"### Recipes waiting to be run ({len(waiting)})")
    st.caption(
        "Run these in the lab, then enter the measured results on **Start here**. "
        "Download the lab sheet to take to the bench: the recipe is filled in and the "
        "result columns are blank, ready to upload back."
    )
    show_experiments(waiting)
    st.download_button(
        "⬇️ Lab sheet (CSV)",
        planner.lab_sheet_csv(species, waiting),
        file_name=f"lab_sheet_{species.name.lower()}.csv",
        mime="text/csv",
    )

# --- get a plan --------------------------------------------------------------------------

st.markdown("### Get my next experiments")
if info.has_synthetic_rows:
    st.error(info.headline)
elif info.level in (rd.NONE, rd.TOO_FEW):
    st.write(
        "There is not enough real data for predictions yet, so the best next step is a "
        "**spread-out starting set**: recipes chosen to differ as much as possible, so every "
        "result teaches the model something new."
    )
    count = st.number_input("How many recipes can you run?", 4, 30, planner.DEFAULT_STARTING_SET)
    if st.button("Create my starting experiments", type="primary"):
        try:
            plan = planner.plan_starting_experiments(session(), species, name, int(count))
        except (ValueError, RuntimeError) as exc:
            session().rollback()
            st.error(f"Could not create the starting set: {exc}")
        else:
            st.session_state["flash"] = (
                f"Created {len(plan.experiments)} starting recipes. {plan.note}"
            )
            st.rerun()
else:
    if info.level == rd.EARLY:
        st.warning(
            "**Early estimate.** You have enough data for a first look, but only "
            f"{info.n_experiments} experiments. Treat the numbers as rough guidance for what to "
            "test next, not as results.",
            icon="⚠️",
        )
    batch_size = st.number_input(
        "How many recipes do you want?", 1, 30, int(get_config().bo_batch_size)
    )
    if st.button("Suggest my next experiments", type="primary"):
        with st.spinner("Learning from your results and choosing the next recipes…"):
            try:
                plan = planner.plan_next_experiments(session(), species, name, int(batch_size))
            except (InsufficientRealDataError, ValueError, RuntimeError) as exc:
                session().rollback()
                st.error(str(exc))
            else:
                st.session_state["last_plan"] = plan
                st.rerun()

last_plan = st.session_state.get("last_plan")
if last_plan is not None and last_plan.kind == "optimizer":
    badge = "🟠" if "low confidence" in last_plan.confidence else "🟢"
    st.markdown(f"#### Suggested experiments — {badge} {last_plan.confidence}")
    st.caption(last_plan.note)
    show_experiments(last_plan.experiments, with_prediction=True)
    for e in last_plan.experiments:
        if e.rationale:
            st.caption(f"**{e.condition_code}** — {e.rationale}")
    st.download_button(
        "⬇️ Lab sheet for these (CSV)",
        planner.lab_sheet_csv(species, last_plan.experiments),
        file_name="lab_sheet_suggested.csv",
        mime="text/csv",
        key="dl_suggested",
    )

# --- try my own recipe -------------------------------------------------------------------

st.markdown("### Try my own recipe")
if not info.can_predict:
    st.info(
        f"Available once you have {info.early_min} real experiments saved "
        f"({info.needed_for_early} to go).",
        icon="🔒",
    )
else:
    st.caption(
        "Type a recipe you are considering. You get the expected efficiency and how unsure it is."
    )
    with st.form("what_if"):
        typed: dict[str, str] = {}
        for field_name in fg.RECIPE_FIELD_NAMES:
            spec = fg.FIELDS_BY_NAME[field_name]
            label = spec.label + (f"  ·  {spec.unit}" if spec.unit else "")
            if spec.kind == "choice":
                typed[field_name] = (
                    st.selectbox(
                        label,
                        [v for v, _ in spec.choices],
                        index=None,
                        format_func=lambda v, s=spec: fg.plain_value(s, str(v)),
                        placeholder="Choose…",
                        key=f"w_{field_name}",
                    )
                    or ""
                )
            else:
                typed[field_name] = st.text_input(
                    label, placeholder=f"for example {spec.example}", key=f"w_{field_name}"
                )
            st.caption(field_hint(field_name, species))
        go = st.form_submit_button("Predict this recipe", type="primary")
    if go:
        problems: list[str] = []
        values: dict[str, object] = {}
        for field_name in fg.RECIPE_FIELD_NAMES:
            spec = fg.FIELDS_BY_NAME[field_name]
            raw = typed[field_name].strip()
            if not raw:
                problems.append(f"'{spec.label}' is missing. {spec.help}")
            elif spec.kind == "number":
                try:
                    values[field_name] = float(raw)
                except ValueError:
                    problems.append(f"'{spec.label}' must be a number, but {raw!r} was entered.")
            else:
                values[field_name] = raw
        if problems:
            st.error("Please fix:", icon="🛑")
            for problem in problems:
                st.markdown(f"- {problem}")
        else:
            try:
                prediction = planner.predict_recipe(session(), species, values, name)
            except (InsufficientRealDataError, ValidationError, ValueError, RuntimeError) as exc:
                session().rollback()
                st.error(f"Could not predict this recipe: {exc}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Expected efficiency", f"{prediction.mean_pce:.2f} %")
                c2.metric("Uncertainty", f"± {prediction.std_pce:.2f} %")
                st.caption(f"Confidence: **{prediction.confidence}**.")
                for problem in prediction.problems:
                    st.warning(
                        f"{problem}. Outside the range the model was built for, so treat "
                        "this prediction with extra caution.",
                        icon="⚠️",
                    )
