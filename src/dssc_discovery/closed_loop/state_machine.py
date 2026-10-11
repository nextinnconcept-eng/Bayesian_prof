"""Closed-loop state machine (Chunk 7.2): PLAN.txt Section K as code.

    PROPOSED --approve--> APPROVED --fabricate--> FABRICATED --measure--> MEASURED
       |reject              |fail                    |fail                 |validate
       v                    v                        v                     v
    REJECTED         FAILED_FABRICATION       FAILED_MEASUREMENT       VALIDATED
                                                                          |ingest
                                                                       INGESTED
                                                                          |retrain
                                                                     MODEL_UPDATED
    PROPOSED --supersede--> SUPERSEDED (surrogate retrained before it was acted on)

Every transition goes through transition(), which enforces legality and
writes one append-only ClosedLoopStateLog row (who, when, why). The
human-approval gate: approve() requires a written approval note, and an
AUTOMATED approval (approved_by_automation=True, used by the M7
simulator) is refused for any non-synthetic condition -- software never
approves a real lab experiment.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    ClosedLoopStateLogRepository,
    ExperimentalConditionRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState as S
from dssc_discovery.domain.experimental_condition import ExperimentalCondition
from dssc_discovery.optimization.records import ClosedLoopStateLog

LEGAL_TRANSITIONS: dict[S, frozenset[S]] = {
    S.PROPOSED: frozenset({S.APPROVED, S.REJECTED, S.SUPERSEDED}),
    S.APPROVED: frozenset({S.FABRICATED, S.FAILED_FABRICATION}),
    S.FABRICATED: frozenset({S.MEASURED, S.FAILED_MEASUREMENT}),
    S.MEASURED: frozenset({S.VALIDATED}),
    S.VALIDATED: frozenset({S.INGESTED}),
    S.INGESTED: frozenset({S.MODEL_UPDATED}),
    S.MODEL_UPDATED: frozenset(),
    S.REJECTED: frozenset(),
    S.FAILED_FABRICATION: frozenset(),
    S.FAILED_MEASUREMENT: frozenset(),
    S.SUPERSEDED: frozenset(),
}
TERMINAL_STATES = frozenset(s for s, targets in LEGAL_TRANSITIONS.items() if not targets)


class IllegalTransitionError(ValueError):
    pass


class HumanApprovalRequiredError(PermissionError):
    pass


def _condition(session: Session, condition_id: UUID) -> ExperimentalCondition:
    condition = ExperimentalConditionRepository(session).get(condition_id)
    if condition is None:
        raise KeyError(f"no experimental condition {condition_id}")
    return condition


def record_proposal(session: Session, condition: ExperimentalCondition, operator: str) -> None:
    """Log a condition's entry into PROPOSED (creation is a transition too)."""
    if condition.state != S.PROPOSED:
        raise IllegalTransitionError(f"{condition.condition_code} is not PROPOSED")
    ClosedLoopStateLogRepository(session).add(
        ClosedLoopStateLog(
            experimental_condition_id=condition.id,
            from_state=None,
            to_state=S.PROPOSED,
            reason="proposed",
            operator=operator,
            is_synthetic=condition.is_synthetic,
        )
    )


def transition(
    session: Session,
    condition_id: UUID,
    to_state: S,
    operator: str,
    reason: str = "",
    superseding_bo_iteration_id: UUID | None = None,
) -> ExperimentalCondition:
    """The single legal path for a state change. Caller must session.commit()."""
    if not operator.strip():
        raise ValueError("every transition needs an operator")
    condition = _condition(session, condition_id)
    if to_state not in LEGAL_TRANSITIONS[condition.state]:
        raise IllegalTransitionError(
            f"{condition.condition_code}: {condition.state.value} -> {to_state.value} is not "
            "a legal Section K transition"
        )
    if to_state == S.SUPERSEDED and superseding_bo_iteration_id is None:
        raise ValueError("SUPERSEDED must cite the superseding BOIteration")
    updated = ExperimentalConditionRepository(session).set_state(condition_id, to_state)
    ClosedLoopStateLogRepository(session).add(
        ClosedLoopStateLog(
            experimental_condition_id=condition_id,
            from_state=condition.state,
            to_state=to_state,
            reason=reason,
            superseding_bo_iteration_id=superseding_bo_iteration_id,
            operator=operator,
            is_synthetic=condition.is_synthetic,
        )
    )
    return updated


def approve(
    session: Session,
    condition_id: UUID,
    operator: str,
    note: str,
    *,
    approved_by_automation: bool = False,
) -> ExperimentalCondition:
    if not note.strip():
        raise HumanApprovalRequiredError("approval needs a written note from the approver")
    if approved_by_automation and not _condition(session, condition_id).is_synthetic:
        raise HumanApprovalRequiredError(
            "automated approval is only allowed for synthetic (simulated) conditions; "
            "a real experiment must be approved by a person"
        )
    return transition(session, condition_id, S.APPROVED, operator, f"approved: {note}")


def reject(
    session: Session, condition_id: UUID, operator: str, reason: str
) -> ExperimentalCondition:
    if not reason.strip():
        raise ValueError("a rejection needs a reason")
    return transition(session, condition_id, S.REJECTED, operator, reason)


def mark_fabricated(
    session: Session, condition_id: UUID, operator: str, note: str = ""
) -> ExperimentalCondition:
    return transition(session, condition_id, S.FABRICATED, operator, note)


def mark_fabrication_failed(
    session: Session, condition_id: UUID, operator: str, reason: str
) -> ExperimentalCondition:
    return transition(session, condition_id, S.FAILED_FABRICATION, operator, reason)


def mark_measured(
    session: Session, condition_id: UUID, operator: str, note: str = ""
) -> ExperimentalCondition:
    return transition(session, condition_id, S.MEASURED, operator, note)


def mark_measurement_failed(
    session: Session, condition_id: UUID, operator: str, reason: str
) -> ExperimentalCondition:
    return transition(session, condition_id, S.FAILED_MEASUREMENT, operator, reason)


def mark_validated(
    session: Session, condition_id: UUID, operator: str, note: str = ""
) -> ExperimentalCondition:
    return transition(session, condition_id, S.VALIDATED, operator, note)


def mark_ingested(
    session: Session, condition_id: UUID, operator: str, note: str = ""
) -> ExperimentalCondition:
    return transition(session, condition_id, S.INGESTED, operator, note)


def mark_model_updated(
    session: Session, condition_id: UUID, operator: str, model_run_id: UUID
) -> ExperimentalCondition:
    return transition(
        session,
        condition_id,
        S.MODEL_UPDATED,
        operator,
        f"surrogate retrained: ModelRun {model_run_id}",
    )


def supersede_pending(
    session: Session, species_code: object, bo_iteration_id: UUID, operator: str
) -> list[ExperimentalCondition]:
    """Move every still-PROPOSED condition of a species to SUPERSEDED,
    citing the BOIteration whose retrained surrogate invalidated them."""
    pending = ExperimentalConditionRepository(session).filter(
        species_code=species_code, state=S.PROPOSED
    )
    return [
        transition(
            session,
            c.id,
            S.SUPERSEDED,
            operator,
            "surrogate retrained before this recommendation was acted on",
            superseding_bo_iteration_id=bo_iteration_id,
        )
        for c in pending
    ]


def history(session: Session, condition_id: UUID) -> list[ClosedLoopStateLog]:
    rows = ClosedLoopStateLogRepository(session).filter(experimental_condition_id=condition_id)
    return sorted(rows, key=lambda r: r.created_at)
