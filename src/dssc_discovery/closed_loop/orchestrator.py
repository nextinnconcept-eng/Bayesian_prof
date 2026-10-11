"""Closed-loop orchestration (Chunks 7.3, 9.4).

run_simulated_closed_loop() wires the whole loop against the synthetic
objective (7.1):

    initial LHS DoE -> PROPOSED -> (simulated approval) -> FABRICATED ->
    synthetic evaluation -> MEASURED -> VALIDATED -> INGESTED ->
    [dataset version -> surrogate refit -> ModelRun -> MODEL_UPDATED ->
     acquisition -> BOIteration + Recommendations (PROPOSED) -> lab ...] x N

Every row it creates is is_synthetic=True, so it must run against a
database holding no real rows (create_dataset_version refuses mixed
snapshots). Each cycle's retraining produces a new ModelRun and a new
BOIteration; still-PROPOSED recommendations are SUPERSEDED, never
silently dropped. Approval here is simulated -- approve() only permits
that for synthetic conditions, so this code cannot approve a real
experiment.

Multi-objective cycles (9.4) use the same path: recommend_cycle() takes
any acquisition result (single- or multi-objective) and routes it
through the same BOIteration/Recommendation/state-machine code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.closed_loop.synthetic_lab import record_synthetic_devices
from dssc_discovery.closed_loop.synthetic_objective import SyntheticObjective
from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ExperimentalConditionRepository,
    SpeciesRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, DesignSource, SpeciesCode
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.species import Species
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.models.tracking import log_surrogate_fit
from dssc_discovery.optimization.acquisition import (
    ACQUISITION_NAME,
    Candidate,
    optimize_acquisition,
)
from dssc_discovery.optimization.convergence import convergence_report
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.multi_objective import (
    MultiObjectiveSurrogate,
    optimize_multi_objective,
)
from dssc_discovery.optimization.recommend import RecommendationBatch, create_recommendations
from dssc_discovery.optimization.search_space import SearchSpace, search_space_for
from dssc_discovery.optimization.surrogate import Surrogate, fit_surrogate
from dssc_discovery.optimization.validation import RejectedCandidate
from dssc_discovery.validation.rules import check_duplicate_condition

SIMULATION_OPERATOR = "closed-loop-simulator"


@dataclass
class CycleSummary:
    cycle: int
    model_run_id: UUID
    bo_iteration_id: UUID
    n_recommended: int
    n_rejected: int
    best_observed: float
    best_true_value: float
    simple_regret: float
    max_predicted: float


@dataclass
class SimulationReport:
    species: str
    optimum_value: float
    initial_best_true_value: float
    cycles: list[CycleSummary] = field(default_factory=list)
    convergence_summary: str = ""


# --- shared steps ---------------------------------------------------------------


def propose_conditions(
    session: Session,
    species: SpeciesCode,
    vectors: Sequence[ExperimentalConditionVector],
    *,
    design_source: DesignSource,
    code_prefix: str,
    operator: str,
    is_synthetic: bool,
) -> list[ExperimentalCondition]:
    """Persist vectors as PROPOSED conditions (with their proposal log
    rows). Used for initial DoE designs; BO cycles go through
    create_recommendations. Caller must session.commit()."""
    species_repo = SpeciesRepository(session)
    if species_repo.get_one(code=species) is None:
        species_repo.add(Species(code=species, operator=operator, is_synthetic=is_synthetic))
    repository = ExperimentalConditionRepository(session)
    conditions = []
    for i, vector in enumerate(vectors, start=1):
        condition = ExperimentalCondition(
            species_code=species,
            condition_code=f"{code_prefix}-{i:03d}",
            design_source=design_source,
            state=ClosedLoopState.PROPOSED,
            target_parameters=vector,
            operator=operator,
            is_synthetic=is_synthetic,
        )
        check_duplicate_condition(repository, condition)
        condition = repository.add(condition)
        sm.record_proposal(session, condition, operator)
        conditions.append(condition)
    return conditions


def retrain(
    session: Session,
    space: SearchSpace,
    *,
    seed: int,
    operator: str,
    tracking_uri: str | None = None,
) -> Surrogate:
    """Dataset version -> surrogate fit -> ModelRun -> INGESTED conditions
    of this species become MODEL_UPDATED."""
    version = create_dataset_version(session, operator=operator)
    data = build_ml_dataset(session)
    surrogate = fit_surrogate(data, space, seed=seed, dataset_version_id=version.id)
    model_run = log_surrogate_fit(session, surrogate, version, operator, tracking_uri=tracking_uri)
    ingested = ExperimentalConditionRepository(session).filter(
        species_code=space.species, state=ClosedLoopState.INGESTED
    )
    for condition in ingested:
        sm.mark_model_updated(session, condition.id, operator, model_run.id)
    return surrogate


def _route_batch(
    session: Session,
    *,
    space: SearchSpace,
    candidates: Sequence[Candidate],
    rejected: Sequence[RejectedCandidate],
    acquisition_function: str,
    objective: str,
    model_run_id: UUID,
    dataset_version_id: UUID,
    best_observed: float,
    cycle_number: int,
    seed: int,
    operator: str,
    is_synthetic: bool,
    retention_predictions: Sequence[tuple[float, float]] | None = None,
) -> RecommendationBatch:
    """Shared single-/multi-objective path: BOIteration + PROPOSED
    recommendations, superseding whatever the species still had PROPOSED."""
    stale = ExperimentalConditionRepository(session).filter(
        species_code=space.species, state=ClosedLoopState.PROPOSED
    )
    batch = create_recommendations(
        session,
        space=space,
        candidates=candidates,
        rejected=rejected,
        cycle_number=cycle_number,
        model_run_id=model_run_id,
        dataset_version_id=dataset_version_id,
        best_observed=best_observed,
        seed=seed,
        operator=operator,
        acquisition_function=acquisition_function,
        objective=objective,
        is_synthetic=is_synthetic,
        retention_predictions=retention_predictions,
    )
    for condition in stale:
        sm.transition(
            session,
            condition.id,
            ClosedLoopState.SUPERSEDED,
            operator,
            "surrogate retrained before this recommendation was acted on",
            superseding_bo_iteration_id=batch.bo_iteration.id,
        )
    for condition in batch.conditions:
        sm.record_proposal(session, condition, operator)
    return batch


def _require_logged(surrogate: Surrogate) -> tuple[UUID, UUID]:
    if surrogate.model_run_id is None or surrogate.dataset_version_id is None:
        raise ValueError("surrogate must be logged (log_surrogate_fit) before recommending")
    return surrogate.model_run_id, surrogate.dataset_version_id


def recommend_cycle(
    session: Session,
    surrogate: Surrogate,
    *,
    cycle_number: int,
    batch_size: int,
    seed: int,
    operator: str,
    is_synthetic: bool,
) -> RecommendationBatch:
    """Single-objective (PCE) cycle."""
    model_run_id, dataset_version_id = _require_logged(surrogate)
    result = optimize_acquisition(surrogate, batch_size=batch_size, seed=seed)
    return _route_batch(
        session,
        space=surrogate.space,
        candidates=result.candidates,
        rejected=result.rejected,
        acquisition_function=result.acquisition_function or ACQUISITION_NAME,
        objective="pce",
        model_run_id=model_run_id,
        dataset_version_id=dataset_version_id,
        best_observed=surrogate.best_observed,
        cycle_number=cycle_number,
        seed=seed,
        operator=operator,
        is_synthetic=is_synthetic,
    )


def recommend_multi_objective_cycle(
    session: Session,
    surrogate: MultiObjectiveSurrogate,
    *,
    cycle_number: int,
    batch_size: int,
    seed: int,
    operator: str,
    is_synthetic: bool,
) -> RecommendationBatch:
    """Multi-objective (PCE x retention) re-ranking cycle (9.4): same
    BOIteration/Recommendation/approval path; the BOIteration cites the
    PCE surrogate's ModelRun, and each Recommendation carries its
    predicted retention."""
    model_run_id, dataset_version_id = _require_logged(surrogate.pce)
    result = optimize_multi_objective(surrogate, batch_size=batch_size, seed=seed)
    return _route_batch(
        session,
        space=surrogate.space,
        candidates=result.candidates,
        rejected=result.rejected,
        acquisition_function=result.acquisition_function,
        objective="pce_retention",
        model_run_id=model_run_id,
        dataset_version_id=dataset_version_id,
        best_observed=surrogate.pce.best_observed,
        cycle_number=cycle_number,
        seed=seed,
        operator=operator,
        is_synthetic=is_synthetic,
        retention_predictions=result.retention,
    )


def _simulate_lab(
    session: Session,
    conditions: Sequence[ExperimentalCondition],
    objective: SyntheticObjective,
    rng: np.random.Generator,
    n_replicates: int,
    operator: str,
) -> None:
    for condition in conditions:
        sm.approve(
            session, condition.id, operator, "simulated approval", approved_by_automation=True
        )
        sm.mark_fabricated(session, condition.id, operator, "simulated fabrication")
        pce_values = [
            objective.evaluate(condition.target_parameters, rng) for _ in range(n_replicates)
        ]
        record_synthetic_devices(session, condition, pce_values, operator)
        sm.mark_measured(session, condition.id, operator, "synthetic objective evaluated")
        sm.mark_validated(session, condition.id, operator, "synthetic rows pass consistency checks")
        sm.mark_ingested(session, condition.id, operator)


# --- simulation -----------------------------------------------------------------


def run_simulated_closed_loop(
    session: Session,
    species: SpeciesCode = SpeciesCode.BETA_VULGARIS,
    *,
    n_cycles: int = 3,
    initial_doe_size: int = 40,
    batch_size: int = 10,
    n_replicates: int = 2,
    noise_std: float = 0.3,
    seed: int = 0,
    operator: str = SIMULATION_OPERATOR,
    tracking_uri: str | None = None,
) -> SimulationReport:
    """Run the full synthetic loop and return per-cycle diagnostics.
    Commits after every stage so a failure leaves a consistent,
    inspectable database."""
    space = search_space_for(species)
    objective = SyntheticObjective(space, noise_std=noise_std)
    rng = np.random.default_rng(seed)

    initial = propose_conditions(
        session,
        species,
        latin_hypercube_design(space, initial_doe_size, seed),
        design_source=DesignSource.INITIAL_DOE,
        code_prefix=f"{species.name}-SIM-DOE",
        operator=operator,
        is_synthetic=True,
    )
    _simulate_lab(session, initial, objective, rng, n_replicates, operator)
    session.commit()
    best_true = max(objective.true_value(c.target_parameters) for c in initial)
    report = SimulationReport(
        species=species.value,
        optimum_value=objective.optimum_value,
        initial_best_true_value=best_true,
    )

    for cycle in range(1, n_cycles + 1):
        surrogate = retrain(
            session, space, seed=seed + cycle, operator=operator, tracking_uri=tracking_uri
        )
        batch = recommend_cycle(
            session,
            surrogate,
            cycle_number=cycle,
            batch_size=batch_size,
            seed=seed + cycle,
            operator=operator,
            is_synthetic=True,
        )
        session.commit()
        _simulate_lab(session, batch.conditions, objective, rng, n_replicates, operator)
        session.commit()

        best_true = max(
            [best_true] + [objective.true_value(c.target_parameters) for c in batch.conditions]
        )
        data = build_ml_dataset(session)
        convergence = convergence_report(session, data, species, planned_cycles=n_cycles)
        BOIterationRepository(session).set_convergence(batch.bo_iteration.id, convergence.to_json())
        session.commit()
        report.cycles.append(
            CycleSummary(
                cycle=cycle,
                model_run_id=surrogate.model_run_id,  # type: ignore[arg-type]
                bo_iteration_id=batch.bo_iteration.id,
                n_recommended=len(batch.recommendations),
                n_rejected=len(json.loads(batch.bo_iteration.rejected_candidates_json)),
                best_observed=float(
                    data.loc[
                        data["species_code"] == species.value, "power_conversion_efficiency_percent"
                    ].max()
                ),
                best_true_value=best_true,
                simple_regret=objective.optimum_value - best_true,
                max_predicted=max(
                    (r.predicted_objective for r in batch.recommendations), default=float("nan")
                ),
            )
        )
        report.convergence_summary = convergence.summary
    return report
