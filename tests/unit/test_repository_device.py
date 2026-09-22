import pytest
import sqlalchemy as sa

from dssc_discovery.database.repositories import (
    DeviceRepository,
    PhotoanodeFabricationBatchRepository,
    SensitizationRunRepository,
)
from dssc_discovery.domain.enums import SpeciesCode
from tests.seed_helpers import (
    seed_full_device_chain,
    seed_photoanode_fabrication_batch,
    seed_pigment_extract_batch,
    seed_plant_sample,
    seed_sensitization_run,
    seed_species,
)


def test_photoanode_fabrication_batch_round_trips(session):
    batch = seed_photoanode_fabrication_batch(session)
    fetched = PhotoanodeFabricationBatchRepository(session).get(batch.id)
    assert fetched == batch


def test_device_round_trips_through_full_fk_chain(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    fetched = DeviceRepository(session).get(device.id)
    assert fetched == device


def test_device_rejects_unknown_condition_fk(session):
    from uuid import uuid4

    from tests.seed_helpers import seed_device

    fab_batch = seed_photoanode_fabrication_batch(session)
    with pytest.raises(sa.exc.IntegrityError):
        seed_device(
            session,
            experimental_condition_id=uuid4(),  # never inserted
            photoanode_fabrication_batch_id=fab_batch.id,
        )


def test_device_rejects_duplicate_device_code(session):
    seed_full_device_chain(
        session, species_code=SpeciesCode.BETA_VULGARIS, device_code="BV-COND-001-R1"
    )
    with pytest.raises(sa.exc.IntegrityError):
        seed_full_device_chain(
            session, species_code=SpeciesCode.CHROMOLAENA_ODORATA, device_code="BV-COND-001-R1"
        )


def test_sensitization_run_round_trips_and_references_device_not_reverse(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    seed_species(session, SpeciesCode.THUNBERGIA_ERECTA)
    sample = seed_plant_sample(session, SpeciesCode.THUNBERGIA_ERECTA)
    extract_batch = seed_pigment_extract_batch(session, plant_sample_id=sample.id)

    run = seed_sensitization_run(
        session, device_id=device.id, pigment_extract_batch_id=extract_batch.id
    )
    fetched = SensitizationRunRepository(session).get(run.id)

    assert fetched == run
    # Device itself carries no sensitization_run_id (Chunk 1.2 design)
    refetched_device = DeviceRepository(session).get(device.id)
    assert "sensitization_run_id" not in type(refetched_device).model_fields
