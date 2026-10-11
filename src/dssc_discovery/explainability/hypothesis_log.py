"""Hypothesis log (Chunk 5.4, PLAN.txt Section I): the mechanism that
connects a SHAP finding to the lab. A notable finding becomes a
structured, queryable record -- observation -> candidate_mechanism ->
confound_check -> proposed_confirmatory_experiment -> status -- so it
survives into Paper 3 with full traceability instead of living in a
notebook.

Status moves only forward through LEGAL_TRANSITIONS, and every change
requires a note explaining it (what experiment confirmed/refuted it).
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import HypothesisLogRepository
from dssc_discovery.domain.enums import HypothesisStatus
from dssc_discovery.explainability.base import SHAPExplanationSet
from dssc_discovery.explainability.records import HypothesisRecord

LEGAL_TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.OPEN: frozenset({HypothesisStatus.TESTING, HypothesisStatus.REFUTED}),
    HypothesisStatus.TESTING: frozenset({HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED}),
    HypothesisStatus.CONFIRMED: frozenset(),
    HypothesisStatus.REFUTED: frozenset(),
}


class IllegalHypothesisTransitionError(ValueError):
    pass


def log_hypothesis(
    session: Session,
    *,
    observation: str,
    candidate_mechanism: str,
    confound_check: str,
    proposed_confirmatory_experiment: str,
    operator: str,
    shap_features: Sequence[str] = (),
    explanation: SHAPExplanationSet | None = None,
    model_run_id: UUID | None = None,
    is_synthetic: bool = False,
) -> HypothesisRecord:
    """Record a new OPEN hypothesis. Passing the SHAPExplanationSet it came
    from carries over its model_run_id and interpretation caveat. Caller
    must session.commit()."""
    record = HypothesisRecord(
        observation=observation,
        candidate_mechanism=candidate_mechanism,
        confound_check=confound_check,
        proposed_confirmatory_experiment=proposed_confirmatory_experiment,
        shap_features=",".join(shap_features),
        model_run_id=model_run_id or (explanation.model_run_id if explanation else None),
        operator=operator,
        is_synthetic=is_synthetic,
        **(
            {"interpretation_caveat": explanation.interpretation_caveat}
            if explanation is not None
            else {}
        ),
    )
    return HypothesisLogRepository(session).add(record)


def transition_hypothesis(
    session: Session, hypothesis_id: UUID, new_status: HypothesisStatus, note: str
) -> HypothesisRecord:
    """Move a hypothesis along LEGAL_TRANSITIONS; note is mandatory."""
    repository = HypothesisLogRepository(session)
    current = repository.get(hypothesis_id)
    if current is None:
        raise KeyError(f"no hypothesis {hypothesis_id}")
    if new_status not in LEGAL_TRANSITIONS[current.status]:
        raise IllegalHypothesisTransitionError(
            f"hypothesis {hypothesis_id}: {current.status.value} -> {new_status.value} "
            "is not allowed"
        )
    if not note.strip():
        raise ValueError("a status change needs a note (which experiment, what result)")
    return repository.set_status(hypothesis_id, new_status, note)


def hypotheses(
    session: Session,
    status: HypothesisStatus | None = None,
    model_run_id: UUID | None = None,
) -> list[HypothesisRecord]:
    filters: dict[str, object] = {}
    if status is not None:
        filters["status"] = status
    if model_run_id is not None:
        filters["model_run_id"] = model_run_id
    repository = HypothesisLogRepository(session)
    rows = repository.filter(**filters) if filters else repository.list()
    return sorted(rows, key=lambda h: h.created_at)


def suggest_observations(explanation: SHAPExplanationSet, top_k: int = 3) -> list[str]:
    """Draft observation sentences for the top-k global features -- a
    starting point for a human to write the actual hypothesis, never
    logged automatically."""
    importance = explanation.global_importance().head(top_k)
    drafts = []
    mean_abs = importance.set_index("feature")["mean_abs_shap"]
    for feature in importance["feature"]:
        j = explanation.feature_names.index(feature)
        values, data = explanation.values[:, j], explanation.data[:, j]
        high = values[data > data.mean()].mean() if (data > data.mean()).any() else 0.0
        direction = "higher" if high > 0 else "lower"
        drafts.append(
            f"Above-average {feature} is associated with {direction} predicted PCE "
            f"in {explanation.model_name} (mean |SHAP| "
            f"{mean_abs[feature]:.3f})."
        )
    return drafts
