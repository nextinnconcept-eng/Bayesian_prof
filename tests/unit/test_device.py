from uuid import uuid4

import pytest
from pydantic import ValidationError

from dssc_discovery.domain.device import Device, PhotoanodeFabricationBatch, SensitizationRun
from dssc_discovery.domain.enums import CounterElectrodeType, ElectrolyteType, SurfaceTreatment


def _fab_batch_kwargs(**overrides):
    kwargs = dict(
        batch_code="PHOTOANODE-001",
        tio2_thickness_um=12.0,
        sintering_temperature_c=450.0,
        sintering_time_min=30.0,
        surface_treatment=SurfaceTreatment.NONE,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_fabrication_batch_construction():
    batch = PhotoanodeFabricationBatch(**_fab_batch_kwargs())
    assert batch.tio2_thickness_um == 12.0


def test_fabrication_batch_round_trips_through_dict():
    batch = PhotoanodeFabricationBatch(**_fab_batch_kwargs())
    restored = PhotoanodeFabricationBatch.model_validate(batch.model_dump())
    assert restored == batch


def test_fabrication_batch_rejects_negative_thickness():
    with pytest.raises(ValidationError):
        PhotoanodeFabricationBatch(**_fab_batch_kwargs(tio2_thickness_um=-1))


def _sensitization_kwargs(**overrides):
    kwargs = dict(
        device_id=uuid4(),
        pigment_extract_batch_id=uuid4(),
        adsorption_time_min=120.0,
        sensitization_concentration_mm=0.5,
        sensitization_temperature_c=25.0,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_sensitization_run_construction():
    run = SensitizationRun(**_sensitization_kwargs())
    assert run.adsorption_time_min == 120.0


def test_sensitization_run_round_trips_through_dict():
    run = SensitizationRun(**_sensitization_kwargs())
    restored = SensitizationRun.model_validate(run.model_dump())
    assert restored == run


def test_sensitization_run_rejects_zero_adsorption_time():
    with pytest.raises(ValidationError):
        SensitizationRun(**_sensitization_kwargs(adsorption_time_min=0))


def _device_kwargs(**overrides):
    kwargs = dict(
        experimental_condition_id=uuid4(),
        photoanode_fabrication_batch_id=uuid4(),
        device_code="BV-COND-001-R1",
        replicate_number=1,
        active_area_cm2=0.25,
        electrolyte_type=ElectrolyteType.IODIDE_TRIIODIDE,
        counter_electrode_type=CounterElectrodeType.PLATINUM,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_device_construction():
    device = Device(**_device_kwargs())
    assert device.replicate_number == 1


def test_device_round_trips_through_dict():
    device = Device(**_device_kwargs())
    restored = Device.model_validate(device.model_dump())
    assert restored == device


@pytest.mark.parametrize("bad_replicate", [0, -1])
def test_device_rejects_non_positive_replicate_number(bad_replicate):
    with pytest.raises(ValidationError):
        Device(**_device_kwargs(replicate_number=bad_replicate))


def test_device_rejects_non_positive_active_area():
    with pytest.raises(ValidationError):
        Device(**_device_kwargs(active_area_cm2=0))


def test_device_does_not_reference_sensitization_run():
    # SensitizationRun references device_id, not the other way around --
    # avoids a circular FK (device fabricated before it's sensitized).
    assert "sensitization_run_id" not in Device.model_fields
