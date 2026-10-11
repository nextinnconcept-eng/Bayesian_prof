"""M5 gate (Chunk 5.5): synthetic dataset -> model -> SHAP -> hypothesis
log round-trip, plus the hypothesis status rules (5.4)."""

from __future__ import annotations

import pytest

from dssc_discovery.domain.enums import HypothesisStatus
from dssc_discovery.evaluation.cv_harness import default_feature_specs, model_features
from dssc_discovery.explainability.hypothesis_log import (
    IllegalHypothesisTransitionError,
    hypotheses,
    log_hypothesis,
    suggest_observations,
    transition_hypothesis,
)
from dssc_discovery.explainability.shap_tree import explain_tree_model
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.models.tree_models import RandomForestModel

FEATURES = ["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"]


def _log(session, **overrides):
    fields = dict(
        observation="high pH associated with higher PCE",
        candidate_mechanism="anthocyanin speciation shifts with pH",
        confound_check="pH correlates with solvent choice in this dataset",
        proposed_confirmatory_experiment="vary pH at fixed solvent",
        operator="tester",
    )
    fields.update(overrides)
    record = log_hypothesis(session, **fields)
    session.commit()
    return record


def test_shap_finding_becomes_a_queryable_hypothesis(session):
    generate_synthetic_dataset(session, n=25, seed=4)
    session.commit()
    data = build_ml_dataset(session)
    X = model_features(data)
    model = RandomForestModel(
        feature_specs=default_feature_specs(FEATURES), n_estimators=30, random_state=0
    )
    model.fit(X, data["power_conversion_efficiency_percent"])
    explanation = explain_tree_model(model, X)

    drafts = suggest_observations(explanation, top_k=2)
    assert len(drafts) == 2
    record = _log(
        session,
        observation=drafts[0],
        explanation=explanation,
        shap_features=[explanation.global_importance()["feature"].iloc[0]],
        is_synthetic=True,
    )

    stored = hypotheses(session, status=HypothesisStatus.OPEN)
    assert [h.id for h in stored] == [record.id]
    assert stored[0].interpretation_caveat == explanation.interpretation_caveat
    assert stored[0].is_synthetic is True


def test_status_moves_forward_only_with_a_note(session):
    record = _log(session)

    with pytest.raises(ValueError, match="note"):
        transition_hypothesis(session, record.id, HypothesisStatus.TESTING, "  ")
    testing = transition_hypothesis(
        session, record.id, HypothesisStatus.TESTING, "batch 7 scheduled"
    )
    assert testing.status == HypothesisStatus.TESTING
    confirmed = transition_hypothesis(
        session, record.id, HypothesisStatus.CONFIRMED, "batch 7 agreed"
    )
    assert confirmed.status_note == "batch 7 agreed"

    with pytest.raises(IllegalHypothesisTransitionError):
        transition_hypothesis(session, record.id, HypothesisStatus.REFUTED, "changed mind")


def test_open_hypothesis_cannot_jump_to_confirmed(session):
    record = _log(session)
    with pytest.raises(IllegalHypothesisTransitionError):
        transition_hypothesis(session, record.id, HypothesisStatus.CONFIRMED, "skipped testing")
