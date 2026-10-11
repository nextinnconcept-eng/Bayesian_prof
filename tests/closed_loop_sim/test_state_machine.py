"""Closed-loop state machine (7.2) and synthetic objective (7.1)."""

from __future__ import annotations

import numpy as np
import pytest

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.closed_loop.orchestrator import propose_conditions
from dssc_discovery.closed_loop.synthetic_objective import MAX_VALUE, SyntheticObjective
from dssc_discovery.domain.enums import ClosedLoopState as S
from dssc_discovery.domain.enums import DesignSource, SpeciesCode
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import search_space_for

SPACE = search_space_for(SpeciesCode.BETA_VULGARIS)


def _propose(session, n=1, synthetic=False):
    conditions = propose_conditions(
        session,
        SpeciesCode.BETA_VULGARIS,
        latin_hypercube_design(SPACE, n, seed=1),
        design_source=DesignSource.INITIAL_DOE,
        code_prefix="SM",
        operator="tester",
        is_synthetic=synthetic,
    )
    session.commit()
    return conditions


def test_full_legal_path_logs_every_transition(session):
    (condition,) = _propose(session)
    sm.approve(session, condition.id, "Dr A", "reviewed")
    sm.mark_fabricated(session, condition.id, "lab")
    sm.mark_measured(session, condition.id, "lab")
    sm.mark_validated(session, condition.id, "lab")
    sm.mark_ingested(session, condition.id, "lab")
    final = sm.mark_model_updated(session, condition.id, "lab", condition.id)
    session.commit()

    assert final.state == S.MODEL_UPDATED
    assert [(e.from_state, e.to_state) for e in sm.history(session, condition.id)] == [
        (None, S.PROPOSED),
        (S.PROPOSED, S.APPROVED),
        (S.APPROVED, S.FABRICATED),
        (S.FABRICATED, S.MEASURED),
        (S.MEASURED, S.VALIDATED),
        (S.VALIDATED, S.INGESTED),
        (S.INGESTED, S.MODEL_UPDATED),
    ]
    assert sm.history(session, condition.id)[1].operator == "Dr A"


def test_skipping_approval_is_illegal(session):
    (condition,) = _propose(session)
    with pytest.raises(sm.IllegalTransitionError):
        sm.mark_measured(session, condition.id, "lab")
    with pytest.raises(sm.IllegalTransitionError):
        sm.mark_fabricated(session, condition.id, "lab")


def test_automation_cannot_approve_a_real_condition(session):
    (real,) = _propose(session, synthetic=False)
    with pytest.raises(sm.HumanApprovalRequiredError):
        sm.approve(session, real.id, "bot", "auto", approved_by_automation=True)
    with pytest.raises(sm.HumanApprovalRequiredError):
        sm.approve(session, real.id, "Dr A", "   ")


def test_automation_may_approve_a_synthetic_condition(session):
    (synthetic,) = _propose(session, synthetic=True)
    approved = sm.approve(session, synthetic.id, "sim", "simulated", approved_by_automation=True)
    assert approved.state == S.APPROVED


def test_failure_and_terminal_states(session):
    first, second = _propose(session, n=2)
    sm.reject(session, first.id, "Dr A", "solvent unavailable")
    with pytest.raises(sm.IllegalTransitionError):
        sm.approve(session, first.id, "Dr A", "changed mind")
    sm.approve(session, second.id, "Dr A", "ok")
    sm.mark_fabrication_failed(session, second.id, "lab", "film cracked")
    assert all(sm.LEGAL_TRANSITIONS[s] == frozenset() for s in sm.TERMINAL_STATES)
    with pytest.raises(ValueError, match="reason"):
        sm.reject(session, second.id, "Dr A", "")


def test_superseding_requires_the_bo_iteration(session):
    (condition,) = _propose(session)
    with pytest.raises(ValueError, match="BOIteration"):
        sm.transition(session, condition.id, S.SUPERSEDED, "system")


def test_synthetic_objective_is_deterministic_with_a_documented_optimum():
    objective = SyntheticObjective(SPACE, noise_std=0.3)
    optimum = objective.optimum_vector()
    assert objective.true_value(optimum) > MAX_VALUE - 0.1
    random_points = latin_hypercube_design(SPACE, 30, seed=2)
    assert max(objective.true_value(v) for v in random_points) < objective.true_value(optimum)

    a = [objective.evaluate(optimum, np.random.default_rng(5)) for _ in range(3)]
    b = [objective.evaluate(optimum, np.random.default_rng(5)) for _ in range(3)]
    assert a == b
