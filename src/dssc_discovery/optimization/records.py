"""Domain models for BO and closed-loop provenance (Chunks 6.5, 7.2):
BOIteration -> ModelRun -> MLDatasetVersion makes every Recommendation
reconstructable by query (PLAN.txt Section M). Kept apart from the logic
modules so repositories can import them without cycles.
"""

from __future__ import annotations

from uuid import UUID

from dssc_discovery.domain.base import ProvenancedModel
from dssc_discovery.domain.enums import ClosedLoopState, SpeciesCode


class BOIteration(ProvenancedModel):
    species_code: SpeciesCode
    cycle_number: int
    model_run_id: UUID
    dataset_version_id: UUID
    acquisition_function: str
    objective: str  # "pce" | "pce_retention"
    search_space_json: str
    seed: int
    batch_size: int
    best_observed: float | None = None
    rejected_candidates_json: str = "[]"
    convergence_json: str = "{}"


class Recommendation(ProvenancedModel):
    """Section G's BORecommendation. Its lifecycle state lives on the
    linked ExperimentalCondition (one state, one audit trail)."""

    bo_iteration_id: UUID
    experimental_condition_id: UUID
    rank: int
    parameters_json: str
    predicted_objective: float
    predictive_uncertainty: float
    acquisition_value: float
    predicted_retention: float | None = None
    predicted_retention_uncertainty: float | None = None
    rationale_text: str


class ClosedLoopStateLog(ProvenancedModel):
    """One Section K transition. operator (provenance) is who did it."""

    experimental_condition_id: UUID
    from_state: ClosedLoopState | None
    to_state: ClosedLoopState
    reason: str = ""
    superseding_bo_iteration_id: UUID | None = None
