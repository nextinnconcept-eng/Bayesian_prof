"""Explainability: which recipe parameters drive the model's efficiency predictions."""

from __future__ import annotations

import streamlit as st
from _shared import page_header, session, show_table

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

page_header(
    "Explainability",
    "SHAP values show how much each parameter pushes a model's predicted efficiency up or "
    "down. Promising patterns become hypotheses for the lab to test.",
    eyebrow="Machine learning",
)
st.warning(INTERPRETATION_CAVEAT, icon="⚠️")

data = build_ml_dataset(session())
if data.empty:
    st.info("No measured devices yet.")
    st.stop()

with st.form("explain"):
    c1, c2 = st.columns([2, 1])
    features = c1.multiselect(
        "Features to explain",
        feature_names(stage="design") + feature_names(stage="characterization"),
        default=["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"],
    )
    model_kind = c2.radio("Model", ["Random forest", "Gaussian process"])
    go = st.form_submit_button("Explain", type="primary")

if go and features:
    specs = default_feature_specs(features)
    model = (
        RandomForestModel(feature_specs=specs, random_state=0)
        if model_kind == "Random forest"
        else GPRModel(feature_specs=specs, random_state=0)
    )
    X = model_features(data)
    with st.spinner("Fitting the model and computing SHAP values…"):
        model.fit(X, data["power_conversion_efficiency_percent"])
        st.session_state["explanation"] = explain(model, X)

if "explanation" in st.session_state:
    explanation = st.session_state["explanation"]
    tab1, tab2, tab3 = st.tabs(
        ["Overall importance", "Single prediction", "Effect of one parameter"]
    )
    with tab1:
        left, right = st.columns(2)
        left.pyplot(plot_global_bar(explanation))
        right.pyplot(plot_beeswarm(explanation))
        st.caption(
            "Left: average impact per parameter. Right: each dot is one device — red = high parameter value, blue = low."
        )
    with tab2:
        row = st.number_input("Device row", 0, len(explanation.values) - 1, 0)
        st.pyplot(plot_waterfall(explanation, int(row)))
        st.caption(
            "How each parameter moved this device's prediction away from the average (grey dashed line)."
        )
    with tab3:
        feature = st.selectbox("Parameter", explanation.feature_names)
        st.pyplot(plot_dependence(explanation, feature))

    st.markdown("#### Turn a finding into a lab hypothesis")
    with st.form("hypothesis"):
        drafts = suggest_observations(explanation)
        observation = st.text_area("What did the model show?", drafts[0] if drafts else "")
        c1, c2 = st.columns(2)
        mechanism = c1.text_area("Possible chemical explanation")
        confound = c2.text_area("What else could explain it?")
        experiment = st.text_area("Experiment that would confirm or refute it")
        if st.form_submit_button("Log hypothesis", type="primary"):
            log_hypothesis(
                session(),
                observation=observation,
                candidate_mechanism=mechanism or "to be determined",
                confound_check=confound or "to be determined",
                proposed_confirmatory_experiment=experiment or "to be determined",
                operator=st.session_state.get("user", "admin"),
                explanation=explanation,
                is_synthetic=bool(data["is_synthetic"].any()),
            )
            session().commit()
            st.success("Hypothesis logged with status **Open**.")
else:
    st.info("Pick features and a model, then press **Explain**.", icon="👆")

st.markdown("#### Hypothesis log")
show_table(
    hypotheses_frame(session()),
    columns=[
        "data",
        "status",
        "observation",
        "candidate_mechanism",
        "proposed_confirmatory_experiment",
        "operator",
        "created_at",
    ],
    empty_message="No hypotheses logged yet.",
)
