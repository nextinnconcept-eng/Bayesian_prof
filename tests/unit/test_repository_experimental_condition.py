import pytest
import sqlalchemy as sa

from dssc_discovery.database.repositories import ExperimentalConditionRepository
from dssc_discovery.domain.enums import ClosedLoopState, SpeciesCode
from tests.seed_helpers import seed_experimental_condition, seed_species


def test_add_then_get_returns_identical_domain_object_including_nested_vector(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS)

    fetched = ExperimentalConditionRepository(session).get(condition.id)

    assert fetched == condition
    # explicitly confirm the flatten/unflatten round-trip preserved every
    # vector field, not just object-level equality
    assert fetched.target_parameters == condition.target_parameters
    assert fetched.target_parameters.ph == condition.target_parameters.ph
    assert fetched.target_parameters.solvent == condition.target_parameters.solvent


def test_state_defaults_to_proposed_after_round_trip(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS)
    fetched = ExperimentalConditionRepository(session).get(condition.id)
    assert fetched.state is ClosedLoopState.PROPOSED


def test_rejects_duplicate_condition_code(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-001")
    with pytest.raises(sa.exc.IntegrityError):
        seed_experimental_condition(
            session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-001"
        )


def test_rejects_ph_out_of_range_at_db_level(session):
    # The domain model already rejects this (Chunk 1.2); this confirms the
    # DB-level CheckConstraint is real defense-in-depth, not just decoration.
    from dssc_discovery.database.models import ExperimentalConditionORM

    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS)
    orm_row = session.get(ExperimentalConditionORM, condition.id)
    orm_row.ph = 20.0
    with pytest.raises(sa.exc.IntegrityError):
        session.commit()


def test_exists_by_condition_code(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-042")

    repo = ExperimentalConditionRepository(session)
    assert repo.exists(condition_code="BV-COND-042") is True
    assert repo.exists(condition_code="BV-COND-999") is False
