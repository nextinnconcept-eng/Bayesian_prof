"""Explainability (Chunk 11.5): SHAP plots with the interpretation caveat
shown prominently, and the hypothesis log."""

from __future__ import annotations

import streamlit as st
from _shared import session, show_table

from dssc_discovery.evaluation.cv_harness import default_feature_specs, model_features
from dssc_discovery.explainability.base import INTERPRETATION_CAVEAT
from dssc_discovery.explainability.hypothesis_log import log_hypothesis, suggest_observations
from dssc_discovery.explainability.plots import (
    plot_beeswarm,
    plot_dependence,
    plot_global_bar,
    plot_waterfall,
)
from dssc_discovery.explainability.shap_kernel import explain
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.registry import feature_names
from dssc_discovery.models.gpr import GPRModel
from dssc_discovery.models.tree_models import RandomForestModel
from dssc_discovery.reports.ui_queries import hypotheses_frame

st.title("Explainability")
st.warning(INTERPRETATION_CAVEAT, icon="⚠️")

data = build_ml_dataset(session())
if data.empty:
    st.info("No measured devices yet.")
    st.stop()

features = st.multiselect(
    "Features",
    feature_names(stage="design") + feature_names(stage="characterization"),
    default=["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"],
)
model_kind = st.radio(
    "Model", ["Random forest (TreeExplainer)", "GP (Kernel SHAP)"], horizontal=True
)
if st.button("Explain", disabled=not features):
    specs = default_feature_specs(features)
    model = (
        RandomForestModel(feature_specs=specs, random_state=0)
        if model_kind.startswith("Random")
        else GPRModel(feature_specs=specs, random_state=0)
    )
    X = model_features(data)
    model.fit(X, data["power_conversion_efficiency_percent"])
    with st.spinner("Computing SHAP values..."):
        st.session_state["explanation"] = explain(model, X)

if "explanation" in st.session_state:
    explanation = st.session_state["explanation"]
    st.pyplot(plot_global_bar(explanation))
    st.pyplot(plot_beeswarm(explanation))
    row = st.number_input("Local explanation for row", 0, len(explanation.values) - 1, 0)
    st.pyplot(plot_waterfall(explanation, int(row)))
    feature = st.selectbox("Dependence plot feature", explanation.feature_names)
    st.pyplot(plot_dependence(explanation, feature))

    st.subheader("Log a hypothesis from this finding")
    with st.form("hypothesis"):
        drafts = suggest_observations(explanation)
        observation = st.text_area("Observation", drafts[0] if drafts else "")
        mechanism = st.text_area("Candidate mechanism (chemistry-grounded)")
        confound = st.text_area("Confound check (what else correlates with this?)")
        experiment = st.text_area("Proposed confirmatory experiment")
        operator = st.text_input("Your name")
        if st.form_submit_button("Log hypothesis"):
            log_hypothesis(
                session(),
                observation=observation,
                candidate_mechanism=mechanism,
                confound_check=confound,
                proposed_confirmatory_experiment=experiment,
                operator=operator,
                explanation=explanation,
                is_synthetic=bool(data["is_synthetic"].any()),
            )
            session().commit()
            st.success("Hypothesis logged as OPEN.")

st.subheader("Hypothesis log")
show_table(hypotheses_frame(session()))
