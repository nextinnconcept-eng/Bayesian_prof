"""M6 gate (Chunk 6.8): search space -> surrogate -> acquisition ->
recommendation -> rejection path, on the synthetic dataset, with DB
traceability (BOIteration -> ModelRun -> MLDatasetVersion)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dssc_discovery.closed_loop.orchestrator import recommend_cycle
from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ClosedLoopStateLogRepository,
    ExperimentalConditionRepository,
    ModelRunRepository,
    RecommendationRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, SpeciesCode
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.models.tracking import log_surrogate_fit
from dssc_discovery.optimization.acquisition import Candidate, optimize_acquisition
from dssc_discovery.optimization.convergence import convergence_report
from dssc_discovery.optimization.recommend import create_recommendations, rationale_text
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.optimization.surrogate import fit_surrogate
from tests.bo_helpers import FAST

SPACE = search_space_for(SpeciesCode.BETA_VULGARIS)


@pytest.fixture
def logged_surrogate(session, tmp_path):
    generate_synthetic_dataset(session, n=25, seed=6)
    session.commit()
    version = create_dataset_version(session, operator="tester")
    data = build_ml_dataset(session)
    surrogate = fit_surrogate(data, SPACE, seed=0, dataset_version_id=version.id)
    log_surrogate_fit(session, surrogate, version, "tester", tracking_uri=str(tmp_path / "mlruns"))
    session.commit()
    return surrogate, data, version


def test_surrogate_uncertainty_is_lower_at_training_points(logged_surrogate):
    surrogate, _, _ = logged_surrogate
    _, std_train = surrogate.predict_encoded(surrogate.train_X.numpy())
    far = SPACE.sample(50, np.random.default_rng(9))
    _, std_far = surrogate.predict_encoded(far)
    assert std_train.mean() < std_far.mean()


def test_optimized_candidates_are_feasible(logged_surrogate):
    surrogate, _, _ = logged_surrogate
    result = optimize_acquisition(surrogate, batch_size=3, seed=1, **FAST)

    assert len(result.candidates) == 3
    for candidate in result.candidates:
        assert SPACE.violations(candidate.vector.model_dump(mode="json")) == []
        assert np.isfinite(candidate.acquisition_value)
        assert candidate.predicted_std > 0


def test_recommendations_are_traceable_and_proposed(session, logged_surrogate):
    surrogate, _, version = logged_surrogate
    batch = recommend_cycle(
        session,
        surrogate,
        cycle_number=1,
        batch_size=2,
        seed=1,
        operator="tester",
        is_synthetic=True,
        acquisition_options=FAST,
    )
    session.commit()

    iteration = BOIterationRepository(session).get(batch.bo_iteration.id)
    model_run = ModelRunRepository(session).get(iteration.model_run_id)
    assert model_run.dataset_version_id == version.id == iteration.dataset_version_id
    assert model_run.strategy == "bo_surrogate_fit"
    recs = RecommendationRepository(session).filter(bo_iteration_id=iteration.id)
    assert len(recs) == 2
    for rec in recs:
        condition = ExperimentalConditionRepository(session).get(rec.experimental_condition_id)
        assert condition.state == ClosedLoopState.PROPOSED
        assert f"{rec.predicted_objective:.2f}%" in rec.rationale_text
        assert f"{rec.predictive_uncertainty:.2f}%" in rec.rationale_text
        log = ClosedLoopStateLogRepository(session).filter(experimental_condition_id=condition.id)
        assert [entry.to_state for entry in log] == [ClosedLoopState.PROPOSED]


def test_new_cycle_supersedes_pending_recommendations(session, logged_surrogate):
    surrogate, _, _ = logged_surrogate
    kwargs = dict(batch_size=1, operator="tester", is_synthetic=True, acquisition_options=FAST)
    first = recommend_cycle(session, surrogate, cycle_number=1, seed=1, **kwargs)
    second = recommend_cycle(session, surrogate, cycle_number=2, seed=2, **kwargs)
    session.commit()

    old = ExperimentalConditionRepository(session).get(first.conditions[0].id)
    assert old.state == ClosedLoopState.SUPERSEDED
    log = ClosedLoopStateLogRepository(session).filter(experimental_condition_id=old.id)
    superseded = [e for e in log if e.to_state == ClosedLoopState.SUPERSEDED]
    assert superseded[0].superseding_bo_iteration_id == second.bo_iteration.id


def test_infeasible_candidate_never_reaches_proposed(session, logged_surrogate):
    surrogate, _, version = logged_surrogate
    good = optimize_acquisition(surrogate, batch_size=1, seed=1, **FAST).candidates[0]
    bad_vector = good.vector.model_copy(update={"ph": 12.0})
    bad = Candidate(
        bad_vector, good.encoded, good.acquisition_value, good.predicted_mean, good.predicted_std
    )

    batch = create_recommendations(
        session,
        space=SPACE,
        candidates=[good, bad],
        rejected=[],
        cycle_number=1,
        model_run_id=surrogate.model_run_id,
        dataset_version_id=version.id,
        best_observed=surrogate.best_observed,
        seed=1,
        operator="tester",
        acquisition_function="test",
        is_synthetic=True,
    )
    session.commit()

    assert len(batch.conditions) == 1
    rejected = json.loads(batch.bo_iteration.rejected_candidates_json)
    assert len(rejected) == 1 and "ph=12" in rejected[0]["reasons"][0]


def test_rationale_classifies_exploitation_and_exploration(logged_surrogate):
    surrogate, _, _ = logged_surrogate
    candidate = optimize_acquisition(surrogate, batch_size=1, seed=1, **FAST).candidates[0]
    exploit = Candidate(candidate.vector, candidate.encoded, 0.0, 8.0, 0.5)
    explore = Candidate(candidate.vector, candidate.encoded, 0.0, 3.0, 0.5)
    assert "EXPLOITATION" in rationale_text(exploit, best_observed=6.0)
    assert "EXPLORATION" in rationale_text(explore, best_observed=6.0)


def test_convergence_report_is_informational_only(session, logged_surrogate):
    surrogate, data, _ = logged_surrogate
    recommend_cycle(
        session,
        surrogate,
        cycle_number=1,
        batch_size=1,
        seed=1,
        operator="tester",
        is_synthetic=True,
        acquisition_options=FAST,
    )
    session.commit()
    report = convergence_report(session, data, SpeciesCode.BETA_VULGARIS)
    assert 1 in report.predicted_improvement_by_cycle
    assert report.cycles_completed == 0
    assert "human decision" in report.summary
