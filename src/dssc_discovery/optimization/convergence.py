"""Convergence/stopping diagnostics (Chunk 6.7, Section J): the three
signals -- (a) predicted-improvement decay across cycles, (b) plateau in
best-observed PCE, (c) budget exhaustion. REPORTS ONLY: stopping is a
human decision, recorded on the BOIteration (convergence_json), never
automated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd
from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import BOIterationRepository, RecommendationRepository
from dssc_discovery.domain.enums import DesignSource, SpeciesCode
from dssc_discovery.evaluation.cv_harness import TARGET_COLUMN

PLANNED_CYCLES = 3
STAGE_ORDER = ("initial_doe", "bo_cycle_1", "bo_cycle_2", "bo_cycle_3")


@dataclass(frozen=True)
class ConvergenceReport:
    species: str
    cycles_completed: int
    cycles_planned: int
    budget_exhausted: bool
    best_observed_by_stage: dict[str, float]
    predicted_improvement_by_cycle: dict[int, float]
    improvement_decaying: bool
    plateau: bool
    plateau_tolerance: float
    summary: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def best_observed_by_stage(data: pd.DataFrame, species: SpeciesCode) -> dict[str, float]:
    """Cumulative best measured PCE after each programme stage."""
    rows = data[data["species_code"].astype(str) == species.value]
    best: dict[str, float] = {}
    running = float("-inf")
    for stage in STAGE_ORDER:
        stage_rows = rows[rows["design_source"].astype(str) == stage]
        if not stage_rows.empty:
            running = max(running, float(stage_rows[TARGET_COLUMN].max()))
        if running > float("-inf"):
            best[stage] = running
    return best


def predicted_improvement_by_cycle(session: Session, species: SpeciesCode) -> dict[int, float]:
    """max predicted PCE among each cycle's recommendations minus the best
    observed value at recommendation time."""
    iterations = BOIterationRepository(session).filter(species_code=species)
    recommendation_repo = RecommendationRepository(session)
    improvements: dict[int, float] = {}
    for iteration in sorted(iterations, key=lambda i: (i.cycle_number, i.created_at)):
        recs = recommendation_repo.filter(bo_iteration_id=iteration.id)
        if recs and iteration.best_observed is not None:
            improvements[iteration.cycle_number] = max(r.predicted_objective for r in recs) - float(
                iteration.best_observed
            )
    return improvements


def convergence_report(
    session: Session,
    data: pd.DataFrame,
    species: SpeciesCode,
    *,
    planned_cycles: int = PLANNED_CYCLES,
    plateau_tolerance: float = 0.1,
) -> ConvergenceReport:
    best = best_observed_by_stage(data, species)
    improvements = predicted_improvement_by_cycle(session, species)
    completed = sum(
        1
        for stage in (DesignSource.BO_CYCLE_1, DesignSource.BO_CYCLE_2, DesignSource.BO_CYCLE_3)
        if stage.value in best and (data["design_source"].astype(str) == stage.value).any()
    )
    values = list(best.values())
    plateau = len(values) >= 3 and (values[-1] - values[-3]) < plateau_tolerance
    gains = [improvements[c] for c in sorted(improvements)]
    decaying = len(gains) >= 2 and all(b <= a for a, b in zip(gains, gains[1:], strict=False))
    exhausted = completed >= planned_cycles

    parts = [f"{completed}/{planned_cycles} BO cycles measured"]
    if values:
        parts.append(f"best observed PCE {values[-1]:.2f}%")
    if decaying:
        parts.append("predicted improvement is decaying across cycles")
    if plateau:
        parts.append(f"best observed PCE has plateaued (< {plateau_tolerance}% over two stages)")
    if exhausted:
        parts.append("planned cycle budget exhausted")
    parts.append("stopping remains a human decision")

    return ConvergenceReport(
        species=species.value,
        cycles_completed=completed,
        cycles_planned=planned_cycles,
        budget_exhausted=exhausted,
        best_observed_by_stage=best,
        predicted_improvement_by_cycle=improvements,
        improvement_decaying=decaying,
        plateau=plateau,
        plateau_tolerance=plateau_tolerance,
        summary="; ".join(parts) + ".",
    )
