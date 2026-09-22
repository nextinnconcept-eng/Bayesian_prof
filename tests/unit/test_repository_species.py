import pytest
import sqlalchemy as sa

from dssc_discovery.database.repositories import SpeciesRepository
from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.domain.species import Species
from tests.seed_helpers import seed_species


def test_add_then_get_returns_identical_domain_object(session):
    species = seed_species(session, SpeciesCode.BETA_VULGARIS)
    fetched = SpeciesRepository(session).get(species.id)
    assert fetched == species


def test_get_missing_id_returns_none(session):
    from uuid import uuid4

    assert SpeciesRepository(session).get(uuid4()) is None


def test_list_returns_all_added(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_species(session, SpeciesCode.CHROMOLAENA_ODORATA)
    codes = {s.code for s in SpeciesRepository(session).list()}
    assert codes == {SpeciesCode.BETA_VULGARIS, SpeciesCode.CHROMOLAENA_ODORATA}


def test_exists_detects_present_and_absent(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    repo = SpeciesRepository(session)
    assert repo.exists(code=SpeciesCode.BETA_VULGARIS) is True
    assert repo.exists(code=SpeciesCode.CHROMOLAENA_ODORATA) is False


def test_duplicate_unique_code_is_rejected_at_db_level(session):
    repo = SpeciesRepository(session)
    repo.add(Species(code=SpeciesCode.BETA_VULGARIS, operator="J. Doe"))
    with pytest.raises(sa.exc.IntegrityError):
        repo.add(Species(code=SpeciesCode.BETA_VULGARIS, operator="J. Doe"))
