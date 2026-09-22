import pytest
import sqlalchemy as sa

from dssc_discovery.database.repositories import (
    PigmentExtractBatchRepository,
    PlantSampleRepository,
)
from dssc_discovery.domain.enums import SpeciesCode
from tests.seed_helpers import seed_pigment_extract_batch, seed_plant_sample, seed_species


def test_plant_sample_add_then_get_returns_identical_domain_object(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    fetched = PlantSampleRepository(session).get(sample.id)
    assert fetched == sample


def test_plant_sample_rejects_duplicate_batch_code(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_plant_sample(session, SpeciesCode.BETA_VULGARIS, batch_code="BV-001")
    with pytest.raises(sa.exc.IntegrityError):
        seed_plant_sample(session, SpeciesCode.BETA_VULGARIS, batch_code="BV-001")


def test_plant_sample_rejects_unknown_species_fk(session):
    # No species row exists at all -- FK enforcement must reject this.
    with pytest.raises(sa.exc.IntegrityError):
        seed_plant_sample(session, SpeciesCode.CHROMOLAENA_ODORATA)


def test_pigment_extract_batch_add_then_get_returns_identical_domain_object(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    batch = seed_pigment_extract_batch(session, plant_sample_id=sample.id)
    fetched = PigmentExtractBatchRepository(session).get(batch.id)
    assert fetched == batch


def test_pigment_extract_batch_exists_by_batch_code(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    seed_pigment_extract_batch(session, plant_sample_id=sample.id, batch_code="BV-EXT-001")

    repo = PigmentExtractBatchRepository(session)
    assert repo.exists(batch_code="BV-EXT-001") is True
    assert repo.exists(batch_code="BV-EXT-999") is False
