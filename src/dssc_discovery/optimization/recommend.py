"""Recommendation generation with rationale (Chunk 6.5): turns acquisition
output into Section G BORecommendations -- one BOIteration row citing the
surrogate's ModelRun and dataset version, one PROPOSED ExperimentalCondition
per candidate, and one Recommendation row with a generated plain-language
rationale (predicted mean, predictive std, and whether the acquisition is
exploiting or exploring). Every candidate is re-validated (Chunk 6.6)
immediately before it can reach PROPOSED.

Recommendations never skip the human gate: they enter PROPOSED and only
closed_loop/state_machine.py can move them on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ExperimentalConditionRepository,
    RecommendationRepository,
    SpeciesRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, DesignSource, SpeciesCode
from dssc_discovery.domain.experimental_condition import ExperimentalCondition
from dssc_discovery.domain.species import Species
from dssc_discovery.optimization.acquisition import Candidate
from dssc_discovery.optimization.records import BOIteration, Recommendation
from dssc_discovery.optimization.search_space import SearchSpace
from dssc_discovery.optimization.validation import RejectedCandidate, validate_candidate
from dssc_discovery.validation.rules import check_duplicate_condition

CYCLE_DESIGN_SOURCE = {
    1: DesignSource.BO_CYCLE_1,
    2: DesignSource.BO_CYCLE_2,
    3: DesignSource.BO_CYCLE_3,
}


@dataclass(frozen=True)
class RecommendationBatch:
    bo_iteration: BOIteration
    recommendations: list[Recommendation]
    conditions: list[ExperimentalCondition]


def rationale_text(
    candidate: Candidate,
    best_observed: float,
    *,
    retention: tuple[float, float] | None = None,
) -> str:
    mean, std = candidate.predicted_mean, candidate.predicted_std
    gap = mean - best_observed
    if std <= 1e-9:
        balance = "with negligible model uncertainty"
    elif gap > std:
        balance = (
            "mainly EXPLOITATION: the predicted mean exceeds the best observed PCE by more "
            "than one predictive standard deviation"
        )
    elif gap > -std:
        balance = (
            "a BALANCE of exploitation and exploration: the predicted mean is within one "
            "standard deviation of the best observed PCE, so both the mean and the "
            "uncertainty contribute to the expected improvement"
        )
    else:
        balance = (
            "mainly EXPLORATION: the predicted mean is below the best observed PCE, and the "
            "expected improvement comes from the model's uncertainty in this region"
        )
    text = (
        f"Predicted PCE {mean:.2f}% (predictive std {std:.2f}%) vs best observed "
        f"{best_observed:.2f}% (difference {gap:+.2f}%). Log acquisition value "
        f"{candidate.acquisition_value:.3f}. This recommendation is {balance}."
    )
    if retention is not None:
        text += f" Predicted stability retention {retention[0]:.1f}% " f"(std {retention[1]:.1f}%)."
    return text


def create_recommendations(
    session: Session,
    *,
    space: SearchSpace,
    candidates: Sequence[Candidate],
    rejected: Sequence[RejectedCandidate],
    cycle_number: int,
    model_run_id: UUID,
    dataset_version_id: UUID,
    best_observed: float,
    seed: int,
    operator: str,
    acquisition_function: str,
    objective: str = "pce",
    is_synthetic: bool = False,
    retention_predictions: Sequence[tuple[float, float]] | None = None,
    condition_code_prefix: str | None = None,
) -> RecommendationBatch:
    """Persist one BOIteration and its PROPOSED conditions + Recommendations.
    Caller must session.commit()."""
    if cycle_number < 1:
        raise ValueError("cycle_number must be >= 1")
    design_source = CYCLE_DESIGN_SOURCE.get(cycle_number)
    if design_source is None:
        if not is_synthetic:
            raise ValueError(
                f"cycle {cycle_number}: DesignSource only tracks 3 real BO cycles "
                "(Section J budget)"
            )
        design_source = DesignSource.BO_CYCLE_3
    _ensure_species(session, space.species, operator, is_synthetic)

    validated = []
    rejected_all = [r.as_dict() for r in rejected]
    for candidate in candidates:
        values = candidate.vector.model_dump(mode="json")
        try:
            validate_candidate(space, values)
        except ValueError as exc:
            rejected_all.append({"values": values, "reasons": [str(exc)]})
            continue
        validated.append(candidate)

    iteration = BOIterationRepository(session).add(
        BOIteration(
            species_code=space.species,
            cycle_number=cycle_number,
            model_run_id=model_run_id,
            dataset_version_id=dataset_version_id,
            acquisition_function=acquisition_function,
            objective=objective,
            search_space_json=space.to_json(),
            seed=seed,
            batch_size=len(validated),
            best_observed=best_observed,
            rejected_candidates_json=json.dumps(rejected_all, default=str),
            operator=operator,
            is_synthetic=is_synthetic,
        )
    )

    condition_repo = ExperimentalConditionRepository(session)
    recommendation_repo = RecommendationRepository(session)
    prefix = (
        condition_code_prefix or f"{space.species.name}-BO{cycle_number}-{str(iteration.id)[:8]}"
    )
    conditions, recommendations = [], []
    for rank, candidate in enumerate(validated, start=1):
        condition = ExperimentalCondition(
            species_code=space.species,
            condition_code=f"{prefix}-{rank:03d}",
            design_source=design_source,
            state=ClosedLoopState.PROPOSED,
            target_parameters=candidate.vector,
            operator=operator,
            is_synthetic=is_synthetic,
        )
        check_duplicate_condition(condition_repo, condition)
        condition = condition_repo.add(condition)
        retention = retention_predictions[rank - 1] if retention_predictions else None
        recommendations.append(
            recommendation_repo.add(
                Recommendation(
                    bo_iteration_id=iteration.id,
                    experimental_condition_id=condition.id,
                    rank=rank,
                    parameters_json=json.dumps(candidate.vector.model_dump(mode="json")),
                    predicted_objective=candidate.predicted_mean,
                    predictive_uncertainty=candidate.predicted_std,
                    acquisition_value=candidate.acquisition_value,
                    predicted_retention=retention[0] if retention else None,
                    predicted_retention_uncertainty=retention[1] if retention else None,
                    rationale_text=rationale_text(candidate, best_observed, retention=retention),
                    operator=operator,
                    is_synthetic=is_synthetic,
                )
            )
        )
        conditions.append(condition)
    return RecommendationBatch(iteration, recommendations, conditions)


def _ensure_species(
    session: Session, species: SpeciesCode, operator: str, is_synthetic: bool
) -> None:
    repository = SpeciesRepository(session)
    if repository.get_one(code=species) is None:
        repository.add(Species(code=species, operator=operator, is_synthetic=is_synthetic))
