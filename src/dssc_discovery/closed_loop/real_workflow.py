"""Shared real-data steps for the M8 scripts: retraining (8.3) and BO
cycle recommendation (8.4). Both refuse synthetic data and are only ever
triggered manually from a script -- nothing here runs unattended.
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from dssc_discovery.closed_loop.orchestrator import recommend_cycle, retrain
from dssc_discovery.config import get_config, get_git_commit_hash
from dssc_discovery.database.repositories import (
    ExperimentalConditionRepository,
    MLDatasetVersionRepository,
    ModelRunRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, SpeciesCode
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.versioning import QUERY_DEFINITION, compute_dataset_hash
from dssc_discovery.models.model_run import ModelRun
from dssc_discovery.optimization.recommend import RecommendationBatch
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.optimization.surrogate import Surrogate, fit_surrogate

TRAINED_STATES = (ClosedLoopState.INGESTED, ClosedLoopState.MODEL_UPDATED)


class InsufficientRealDataError(RuntimeError):
    pass


def real_trained_condition_count(session: Session, species: SpeciesCode) -> int:
    repository = ExperimentalConditionRepository(session)
    return sum(
        1
        for state in TRAINED_STATES
        for c in repository.filter(species_code=species, state=state)
        if not c.is_synthetic
    )


def retrain_on_real_data(
    session: Session, species: SpeciesCode, operator: str, *, seed: int | None = None
) -> Surrogate:
    minimum = get_config().min_real_conditions_for_retraining
    count = real_trained_condition_count(session, species)
    if count < minimum:
        raise InsufficientRealDataError(
            f"{species.value}: {count} real ingested conditions, need >= {minimum} "
            "(min_real_conditions_for_retraining) before retraining on real data"
        )
    data = build_ml_dataset(session)
    if data["is_synthetic"].any():
        raise InsufficientRealDataError(
            "database contains synthetic rows; real retraining runs on a real-only database"
        )
    return retrain(
        session,
        search_space_for(species),
        seed=get_config().random_seed if seed is None else seed,
        operator=operator,
    )


def latest_surrogate_run(session: Session, species: SpeciesCode) -> ModelRun | None:
    runs = [
        r
        for r in ModelRunRepository(session).filter(
            strategy="bo_surrogate_fit", model_name=f"bo_surrogate:{species.name}"
        )
        if not r.is_synthetic
    ]
    return max(runs, key=lambda r: r.created_at) if runs else None


def refit_logged_surrogate(session: Session, species: SpeciesCode) -> Surrogate:
    """Rebuild the surrogate of the latest real ModelRun exactly: same
    data (dataset hash must still match), search space and seed. Refuses
    if new data arrived since -- retrain first."""
    run = latest_surrogate_run(session, species)
    if run is None:
        raise InsufficientRealDataError(
            f"no real surrogate ModelRun for {species.value}; retrain first"
        )
    version = MLDatasetVersionRepository(session).get(run.dataset_version_id)
    assert version is not None
    data = build_ml_dataset(session)
    current_hash = compute_dataset_hash(
        sorted(str(v) for v in data["device_id"]), QUERY_DEFINITION, version.code_version
    )
    if current_hash != version.dataset_hash:
        raise InsufficientRealDataError(
            "the dataset changed since the latest surrogate ModelRun; "
            "run the retraining script first"
        )
    if run.git_commit_hash != get_git_commit_hash():
        # Different code may fit differently -- allowed, but recorded loudly.
        print(
            f"warning: surrogate was fit at commit {run.git_commit_hash[:10]}, "
            "now at a different commit"
        )
    config = json.loads(run.cv_config_json)
    surrogate = fit_surrogate(
        data, search_space_for(species), seed=int(config["seed"]), dataset_version_id=version.id
    )
    surrogate.model_run_id = run.id
    return surrogate


def recommend_real_cycle(
    session: Session,
    species: SpeciesCode,
    cycle_number: int,
    operator: str,
    batch_size: int | None = None,
) -> RecommendationBatch:
    surrogate = refit_logged_surrogate(session, species)
    return recommend_cycle(
        session,
        surrogate,
        cycle_number=cycle_number,
        batch_size=batch_size or get_config().bo_batch_size,
        seed=surrogate.seed + cycle_number,
        operator=operator,
        is_synthetic=False,
    )
