from datetime import date, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dssc_discovery.domain.enums import PigmentClass, PlantPart, SolventType, SpeciesCode
from dssc_discovery.domain.extraction import PigmentExtractBatch, PlantSample


def _plant_sample_kwargs(**overrides):
    kwargs = dict(
        species_code=SpeciesCode.TELFAIRIA_OCCIDENTALIS,
        plant_part=PlantPart.LEAF,
        pigment_class=PigmentClass.CHLOROPHYLL,
        batch_code="TO-2026-001",
        collection_date=date(2026, 1, 15),
        collection_location="Test Farm, Lagos",
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_plant_sample_construction():
    sample = PlantSample(**_plant_sample_kwargs())
    assert sample.species_code is SpeciesCode.TELFAIRIA_OCCIDENTALIS


def test_plant_sample_round_trips_through_dict():
    sample = PlantSample(**_plant_sample_kwargs())
    restored = PlantSample.model_validate(sample.model_dump())
    assert restored == sample


def test_plant_sample_rejects_future_collection_date():
    future_date = date.today() + timedelta(days=1)
    with pytest.raises(ValidationError, match="future"):
        PlantSample(**_plant_sample_kwargs(collection_date=future_date))


def test_plant_sample_rejects_empty_batch_code():
    with pytest.raises(ValidationError):
        PlantSample(**_plant_sample_kwargs(batch_code=""))


def _extract_batch_kwargs(**overrides):
    kwargs = dict(
        plant_sample_id=uuid4(),
        batch_code="TO-EXT-001",
        solvent=SolventType.ETHANOL,
        ph=5.5,
        extraction_temperature_c=25.0,
        extraction_time_min=60.0,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_extract_batch_construction():
    batch = PigmentExtractBatch(**_extract_batch_kwargs())
    assert batch.solvent is SolventType.ETHANOL
    # unset concentration/yield stay None, never silently become 0
    assert batch.pigment_concentration_mm is None
    assert batch.extraction_yield_percent is None


def test_extract_batch_round_trips_through_dict():
    batch = PigmentExtractBatch(**_extract_batch_kwargs(pigment_concentration_mm=12.5))
    restored = PigmentExtractBatch.model_validate(batch.model_dump())
    assert restored == batch


@pytest.mark.parametrize("bad_ph", [-0.1, 14.1])
def test_extract_batch_rejects_ph_out_of_range(bad_ph):
    with pytest.raises(ValidationError):
        PigmentExtractBatch(**_extract_batch_kwargs(ph=bad_ph))


@pytest.mark.parametrize("bad_yield", [-5.0, 150.0])
def test_extract_batch_rejects_yield_out_of_range(bad_yield):
    with pytest.raises(ValidationError):
        PigmentExtractBatch(**_extract_batch_kwargs(extraction_yield_percent=bad_yield))


def test_extract_batch_rejects_zero_or_negative_time():
    with pytest.raises(ValidationError):
        PigmentExtractBatch(**_extract_batch_kwargs(extraction_time_min=0))
    with pytest.raises(ValidationError):
        PigmentExtractBatch(**_extract_batch_kwargs(extraction_time_min=-5))


def test_extract_batch_field_names_match_units_registry():
    from dssc_discovery.domain.units_registry import CANONICAL_UNITS

    for field in (
        "extraction_temperature_c",
        "extraction_time_min",
        "pigment_concentration_mm",
        "extraction_yield_percent",
    ):
        assert field in CANONICAL_UNITS
        assert field in PigmentExtractBatch.model_fields
