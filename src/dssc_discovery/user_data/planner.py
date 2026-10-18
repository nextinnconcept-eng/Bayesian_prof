"""What to do in the lab next, and what a recipe is expected to give.

Three entry points, all real-data only:

- plan_starting_experiments(): no (or too little) data yet -> a space-filling
  set of recipes to run, saved as PROPOSED conditions of the initial design.
- plan_next_experiments(): enough data -> retrain on real data (early or full
  mode) and ask the optimizer for the next batch. Every result carries its
  confidence level so the UI can never present an early estimate as a
  trusted one.
- predict_recipe(): "what if I try this recipe?" -> predicted efficiency and
  uncertainty from the latest real surrogate.

Nothing here approves anything; recommendations stay PROPOSED until the owner
acts (approvals page, or by recording results).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from sqlalchemy.orm import Session

from dssc_discovery.closed_loop.orchestrator import propose_conditions
from dssc_discovery.closed_loop.real_workflow import (
    InsufficientRealDataError,
    recommend_real_cycle,
    refit_logged_surrogate,
    retrain_on_real_data,
)
from dssc_discovery.config import get_config
from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ExperimentalConditionRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, DesignSource, SpeciesCode
from dssc_discovery.domain.experimental_condition import ExperimentalCondition
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.user_data import readiness as rd
from dssc_discovery.user_data.field_guide import (
    LAB_SHEET_RESULT_COLUMNS,
    RECIPE_FIELD_NAMES,
)

DEFAULT_STARTING_SET = 10


@dataclass
class PlannedExperiment:
    condition_code: str
    parameters: dict[str, object]
    predicted_pce: float | None = None
    uncertainty: float | None = None
    rationale: str = ""


@dataclass
class Plan:
    kind: str  # "starting" | "optimizer"
    confidence: str  # "starting design" | "early estimate (low confidence)" | "full"
    experiments: list[PlannedExperiment]
    note: str


def _abbreviation(species: SpeciesCode) -> str:
    return "".join(w[0] for w in species.value.split()[:2]).upper()


def _planned(condition: ExperimentalCondition) -> PlannedExperiment:
    return PlannedExperiment(
        condition_code=condition.condition_code,
        parameters=condition.target_parameters.model_dump(mode="json"),
    )


def pending_plan(session: Session, species: SpeciesCode) -> list[PlannedExperiment]:
    """Recipes already proposed for this species and still waiting to be run."""
    repository = ExperimentalConditionRepository(session)
    pending = [
        c
        for state in (ClosedLoopState.PROPOSED, ClosedLoopState.APPROVED)
        for c in repository.filter(species_code=species, state=state)
        if not c.is_synthetic
    ]
    return [_planned(c) for c in sorted(pending, key=lambda c: c.condition_code)]


def plan_starting_experiments(
    session: Session, species: SpeciesCode, operator: str, n: int = DEFAULT_STARTING_SET
) -> Plan:
    """A space-filling (Latin hypercube) set across every allowed recipe
    range, so the first few results teach the model as much as possible."""
    space = search_space_for(species)
    repository = ExperimentalConditionRepository(session)
    existing = {c.condition_code for c in repository.filter(species_code=species)}
    batch = 1
    while any(code.startswith(f"{_abbreviation(species)}-START{batch}-") for code in existing):
        batch += 1
    seed = get_config().random_seed + batch
    vectors = latin_hypercube_design(space, n, seed=seed)
    conditions = propose_conditions(
        session,
        species,
        vectors,
        design_source=DesignSource.INITIAL_DOE,
        code_prefix=f"{_abbreviation(species)}-START{batch}",
        operator=operator,
        is_synthetic=False,
    )
    session.commit()
    return Plan(
        kind="starting",
        confidence="starting design",
        experiments=[_planned(c) for c in conditions],
        note=(
            "These recipes are spread across the whole allowed range on purpose, so each "
            "one teaches the model something different. Run them, then enter the results."
        ),
    )


def plan_next_experiments(
    session: Session, species: SpeciesCode, operator: str, batch_size: int | None = None
) -> Plan:
    """Retrain on the owner's real data and propose the next recipes."""
    info = rd.assess_readiness(session, species)
    if info.has_synthetic_rows:
        raise InsufficientRealDataError(info.headline)
    if info.level in (rd.NONE, rd.TOO_FEW):
        raise InsufficientRealDataError(info.headline)
    early = info.level == rd.EARLY
    retrain_on_real_data(session, species, operator, early=early)
    session.commit()
    cycle = len(BOIterationRepository(session).filter(species_code=species)) + 1
    batch = recommend_real_cycle(
        session, species, cycle_number=cycle, operator=operator, batch_size=batch_size
    )
    session.commit()
    experiments = []
    for condition, recommendation in zip(batch.conditions, batch.recommendations, strict=True):
        experiments.append(
            PlannedExperiment(
                condition_code=condition.condition_code,
                parameters=condition.target_parameters.model_dump(mode="json"),
                predicted_pce=float(recommendation.predicted_objective),
                uncertainty=float(recommendation.predictive_uncertainty),
                rationale=recommendation.rationale_text,
            )
        )
    confidence = "early estimate (low confidence)" if early else "full"
    note = (
        "Early estimate: based on few experiments, so predictions are rough and the "
        "uncertainty is large. Use it to choose what to test, not to trust the numbers."
        if early
        else "Based on enough real experiments for the optimizer's normal mode."
    )
    return Plan(kind="optimizer", confidence=confidence, experiments=experiments, note=note)


@dataclass
class Prediction:
    mean_pce: float
    std_pce: float
    confidence: str
    problems: list[str]


def predict_recipe(
    session: Session, species: SpeciesCode, values: dict[str, object], operator: str = "owner"
) -> Prediction:
    """Predicted efficiency (mean and +/- one standard deviation) for a recipe
    typed in by the owner, from the latest real surrogate."""
    info = rd.assess_readiness(session, species)
    if not info.can_predict:
        raise InsufficientRealDataError(info.headline)
    space = search_space_for(species)
    problems = space.violations(values)
    vector = space.to_vector(values)
    try:
        surrogate = refit_logged_surrogate(session, species)
    except InsufficientRealDataError:
        # No model yet, or new results arrived since the last one: retrain first.
        retrain_on_real_data(session, species, operator, early=info.level == rd.EARLY)
        session.commit()
        surrogate = refit_logged_surrogate(session, species)
    mean, std = surrogate.predict([vector])
    return Prediction(
        mean_pce=float(mean[0]),
        std_pce=float(std[0]),
        confidence="early estimate (low confidence)" if info.level == rd.EARLY else "full",
        problems=problems,
    )


def lab_sheet_csv(species: SpeciesCode, experiments: list[PlannedExperiment]) -> str:
    """CSV of planned recipes with the result columns left blank, ready to be
    filled in at the bench and uploaded back unchanged."""
    columns = ["condition_code", "species_code", *RECIPE_FIELD_NAMES, *LAB_SHEET_RESULT_COLUMNS]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for experiment in experiments:
        row = {c: "" for c in columns}
        row["condition_code"] = experiment.condition_code
        row["species_code"] = species.value
        for name in RECIPE_FIELD_NAMES:
            row[name] = str(experiment.parameters[name])
        row["replicate_number"] = "1"
        writer.writerow(row)
    return buffer.getvalue()
